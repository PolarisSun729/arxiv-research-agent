from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from services.memory import MemoryService
from services.storage.database_service import DatabaseService
from utils.config import get_memory_runtime_config

try:  # pragma: no cover - optional runtime dependency for LLM parsing
    from dependencies import get_generation_service as _get_generation_service
except Exception:  # pragma: no cover
    _get_generation_service = None

from .graph import build_arxiv_search_graph
from .schemas import AgentStep, AgentStreamEvent, ArxivSearchRequest, ArxivSearchResponse
from .state import AgentState

logger = logging.getLogger(__name__)

# 这个文件是 arXiv Agent 的“运行入口层”。
#
# 如果说：
# - graph.py 负责定义工作流结构，
# - node/ 下的模块负责每个节点做什么，
# 那么 service.py 负责把“外部请求”真正接到这套工作流上。
#
# 它主要承担四类职责：
# 1. 接收并规范化请求对象；
# 2. 注入运行时上下文，例如 user research profile；
# 3. 构造初始 AgentState 并执行 LangGraph；
# 4. 把最终状态转换成同步响应或流式 SSE 事件。
MEMORY_RUNTIME_CONFIG = get_memory_runtime_config()


def _inject_user_memory_context(request_context: Dict[str, Any], user_id: Optional[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """加载统一的 user_memory_summary，并兼容保留 research_profile 字段。"""
    enriched_context = dict(request_context or {})
    normalized_user_id = str(user_id or "").strip()
    debug_flags = {
        "user_memory_loaded": False,
        "profile_applied": False,
        "preference_memory_available": False,
        "interest_vector_available": False,
    }
    if not normalized_user_id:
        return enriched_context, debug_flags

    try:
        memory_service = MemoryService()
        user_memory_summary = memory_service.build_user_memory_summary(normalized_user_id)
        enriched_context["user_memory_summary"] = user_memory_summary
        if not enriched_context.get("research_profile"):
            profile = dict((user_memory_summary or {}).get("profile") or {})
            if profile:
                enriched_context["research_profile"] = profile
        memory_status = dict((user_memory_summary or {}).get("memory_status") or {})
        debug_flags = {
            "user_memory_loaded": True,
            "profile_applied": bool((user_memory_summary or {}).get("profile")),
            "preference_memory_available": bool(memory_status.get("preference_memory_available")),
            "interest_vector_available": bool(memory_status.get("interest_vector_available")),
        }
    except Exception as exc:
        logger.warning("Failed to load user memory summary for agent context: user_id=%s error=%s", normalized_user_id, exc)
        if not enriched_context.get("research_profile") and bool(MEMORY_RUNTIME_CONFIG.get("enable_user_research_profile", False)):
            try:
                profile = DatabaseService().get_user_research_profile(user_id=normalized_user_id)
            except Exception:
                profile = {}
            if isinstance(profile, dict) and profile:
                enriched_context["research_profile"] = profile
                debug_flags["profile_applied"] = True
        return enriched_context, debug_flags

    return enriched_context, debug_flags


def _load_agent_request_context(
    normalized_request: ArxivSearchRequest,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    frontend_context, user_memory_debug = _inject_user_memory_context(
        dict(normalized_request.context or {}),
        normalized_request.user_id,
    )
    try:
        memory_payload = MemoryService().load_agent_memory(
            normalized_request.user_id,
            normalized_request.session_id,
            frontend_context=frontend_context,
        )
    except Exception as exc:
        logger.warning("Failed to load agent session memory: user_id=%s session_id=%s error=%s", normalized_request.user_id, normalized_request.session_id, exc)
        return frontend_context, None, normalized_request.session_id, user_memory_debug

    merged_context = dict(memory_payload.get("merged_context") or frontend_context)
    resolved_session_id = str(memory_payload.get("session_id") or normalized_request.session_id or "").strip() or None
    return merged_context, memory_payload, resolved_session_id, user_memory_debug


def _persist_agent_session_memory(final_state: Any) -> None:
    state = _coerce_state(final_state)
    try:
        MemoryService().save_agent_memory(
            user_id=state.user_id,
            session_id=state.session_id,
            final_state=state,
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist agent session memory: user_id=%s session_id=%s error=%s",
            state.user_id,
            state.session_id,
            exc,
        )


def run_arxiv_search_agent(request: ArxivSearchRequest) -> ArxivSearchResponse:
    """同步执行一次 arXiv Agent，并返回最终聚合响应。

    这是最标准的非流式入口，适合普通 HTTP 调用场景。
    整体流程可以概括为：
    1. 校验并规范化请求；
    2. 注入 research profile 等上下文增强信息；
    3. 构造初始 AgentState；
    4. 构建并执行 LangGraph；
    5. 把最终 state 转换成对外响应模型。

    这里不直接暴露 LangGraph 细节给上层调用方，而是统一收口成 `ArxivSearchResponse`。
    """
    try:
        # 第 1 步：先把入参统一规整成 ArxivSearchRequest，避免上层传 dict 时各处重复判断。
        normalized_request = _coerce_request(request)

        # 第 2 步：把前端 context 与后端 Agent session memory 合并。
        request_context, _agent_memory_payload, resolved_session_id, user_memory_debug = _load_agent_request_context(normalized_request)
        # 入口日志只记录状态摘要，便于排查“前端传了但后端没识别到”的问题，不直接打出完整上下文内容。
        logger.debug(
            "arxiv_agent request received: message=%s context_keys=%s pending_action_status=%s paper_qa_status=%s selected_arxiv_id=%s",
            normalized_request.message,
            sorted(request_context.keys()),
            _safe_status(request_context.get("pending_action")),
            _safe_status(request_context.get("paper_qa_result")),
            _safe_selected_arxiv_id(request_context),
        )
        # 第 3 步：解析生成服务，并构造图执行所需的初始状态。
        generation_service = _resolve_generation_service()
        initial_state = AgentState(
            user_id=normalized_request.user_id,
            session_id=resolved_session_id,
            message=normalized_request.message,
            # 把前端回传的待确认状态提升到顶层，避免后续路由只看 context 时漏掉当前待办。
            context=request_context,
            pending_action=request_context.get("pending_action"),
            paper_qa_result=request_context.get("paper_qa_result"),
            debug=dict(user_memory_debug or {}),
        )

        # 第 4 步：构建图并同步执行，拿到最终状态。
        graph = build_arxiv_search_graph(generation_service=generation_service)
        final_state = _coerce_state(graph.invoke(initial_state.model_dump()))

        # 第 5 步：把跨轮 Agent memory 回写到后端 session。
        _persist_agent_session_memory(final_state)

        # 第 6 步：把内部状态转换成对外响应模型。
        return _state_to_response(final_state)
    except ValidationError as exc:
        return _build_error_response(
            message="请求参数校验失败",
            detail=str(exc),
            code="request_validation_error",
        )
    except Exception as exc:
        return _build_error_response(
            message="arXiv 搜索 Agent 运行失败",
            detail=str(exc),
            code="agent_runtime_error",
        )


def stream_arxiv_search_agent(request: ArxivSearchRequest) -> StreamingResponse:
    """以 SSE 流式方式执行 arXiv Agent。

    与 `run_arxiv_search_agent` 不同，这个入口不会等到整张图执行完再返回，
    而是会把运行过程拆成连续事件推给前端，包括：
    - run_start
    - step_start / step_end
    - tool_call_start / tool_call_end
    - final_response
    - exception / stream_end

    这样前端就能一边展示执行进度，一边更新中间状态，而不是只能等待最终答案。
    """
    normalized_request = _coerce_request(request)

    def event_stream():
        # 阶段 A：初始化流式执行上下文。
        # run_id 和 sequence 一起构成了一次流式执行的事件主线。
        run_id = str(uuid4())
        sequence = 1
        generation_service = _resolve_generation_service()
        current_state: Optional[AgentState] = None

        try:
            # 阶段 B：构造与同步入口一致的初始上下文和状态，保证两条路径行为一致。
            request_context, _agent_memory_payload, resolved_session_id, user_memory_debug = _load_agent_request_context(normalized_request)
            logger.debug(
                "arxiv_agent stream start: run_id=%s message=%s context_keys=%s pending_action_status=%s paper_qa_status=%s selected_arxiv_id=%s",
                run_id,
                normalized_request.message,
                sorted(request_context.keys()),
                _safe_status(request_context.get("pending_action")),
                _safe_status(request_context.get("paper_qa_result")),
                _safe_selected_arxiv_id(request_context),
            )
            current_state = AgentState(
                user_id=normalized_request.user_id,
                session_id=resolved_session_id,
                message=normalized_request.message,
                # 流式路径和同步路径必须使用同一份状态提升规则，确保确认/解析流程一致。
                context=request_context,
                pending_action=request_context.get("pending_action"),
                paper_qa_result=request_context.get("paper_qa_result"),
                debug=dict(user_memory_debug or {}),
            )
            graph = build_arxiv_search_graph(generation_service=generation_service)

            # 阶段 C：先发出 run_start 事件，让前端知道一次新的执行已经开始。
            yield _sse_event(
                _make_stream_event(
                    event_type="run_start",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "status": "started",
                        "request": {
                            "user_id": normalized_request.user_id,
                            "session_id": resolved_session_id,
                            "message": normalized_request.message,
                        },
                    },
                )
            )
            sequence += 1

            # 阶段 D：逐步消费 LangGraph 的 updates 流，并把节点生命周期翻译成 SSE 事件。
            for update in graph.stream(current_state.model_dump(), stream_mode="updates"):
                if not update:
                    continue

                step_name, step_payload = next(iter(update.items()))
                previous_state = current_state

                # 子阶段 D-1：先通知前端“某个节点开始执行”。
                yield _sse_event(
                    _make_stream_event(
                        event_type="step_start",
                        sequence=_next_sequence(sequence),
                        run_id=run_id,
                        data={
                            "step": step_name,
                            "state": _compact_state(previous_state),
                        },
                    )
                )
                sequence += 1

                tool_call_started = False
                if step_name == "invoke_search_tool" and _should_emit_tool_call(previous_state):
                    tool_call_started = True
                    # 子阶段 D-2：如果当前节点会触发工具调用，则补发工具开始事件。
                    yield _sse_event(
                        _make_stream_event(
                            event_type="tool_call_start",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "step": step_name,
                                "tool_call": {
                                    "tool_name": previous_state.tool_name,
                                    "arguments": _compact_tool_args(previous_state.tool_args),
                                },
                                "state": _compact_state(previous_state),
                            },
                        )
                    )
                    sequence += 1

                current_state = _coerce_state(step_payload)
                latest_step = current_state.steps[-1].model_dump() if current_state.steps else None

                # 子阶段 D-3：节点执行结束后，把最新 step 摘要和当前状态回传给前端。
                yield _sse_event(
                    _make_stream_event(
                        event_type="step_end",
                        sequence=_next_sequence(sequence),
                        run_id=run_id,
                        data={
                            "step": step_name,
                            "status": latest_step.get("status") if isinstance(latest_step, dict) else "success",
                            "step_detail": latest_step,
                            "state": _compact_state(current_state),
                        },
                    )
                )
                sequence += 1

                if tool_call_started:
                    # 子阶段 D-4：若本节点触发了工具调用，则在节点结束后补发 tool_call_end。
                    latest_tool_call = current_state.tool_calls[-1].model_dump() if current_state.tool_calls else {
                        "tool_name": current_state.tool_name,
                        "arguments": _compact_tool_args(current_state.tool_args),
                        "status": "failed",
                        "summary": "工具调用未产生结果",
                        "trace": {},
                        "error": None,
                    }
                    yield _sse_event(
                        _make_stream_event(
                            event_type="tool_call_end",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "step": step_name,
                                "tool_call": latest_tool_call,
                                "state": _compact_state(current_state),
                            },
                        )
                    )
                    sequence += 1

            # 阶段 E：整张图执行完成后，输出最终聚合响应和结束事件。
            if current_state is not None:
                _persist_agent_session_memory(current_state)
            final_response = _state_to_response(current_state)
            yield _sse_event(
                _make_stream_event(
                    event_type="final_response",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "response": final_response.model_dump(),
                        "state": _compact_state(current_state),
                    },
                )
            )
            sequence += 1
            yield _sse_event(
                _make_stream_event(
                    event_type="stream_end",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "status": "success",
                        "final_sequence": sequence,
                    },
                )
            )
        except Exception as exc:
            # 阶段 F：流式过程中任何异常都转成结构化事件，而不是让连接直接中断。
            error_response = _build_error_response_from_state(
                current_state,
                message="arXiv 搜索 Agent 运行失败",
                detail=str(exc),
                code="agent_runtime_error",
            )
            yield _sse_event(
                _make_stream_event(
                    event_type="exception",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "message": "Agent 流式执行异常",
                        "detail": str(exc),
                        "response": error_response.model_dump(),
                    },
                )
            )
            sequence += 1
            yield _sse_event(
                _make_stream_event(
                    event_type="final_response",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "response": error_response.model_dump(),
                    },
                )
            )
            sequence += 1
            yield _sse_event(
                _make_stream_event(
                    event_type="stream_end",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "status": "error",
                        "final_sequence": sequence,
                    },
                )
            )

    # 返回真正的 SSE 响应对象，交给 FastAPI 持续推送 event_stream 生成的事件。
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _coerce_request(request: ArxivSearchRequest | Dict[str, Any]) -> ArxivSearchRequest:
    """把请求入参统一规整成 ArxivSearchRequest。"""
    if isinstance(request, ArxivSearchRequest):
        return request
    return ArxivSearchRequest.model_validate(dict(request))


def _safe_status(value: Any) -> str:
    """安全提取状态对象里的 status 字段，仅用于日志摘要。"""
    if not isinstance(value, Mapping):
        return "none"
    return str(value.get("status") or "none").strip() or "none"


def _safe_selected_arxiv_id(context: Mapping[str, Any]) -> str:
    """从上下文中尽量提取当前选中论文的 arXiv ID，用于日志定位。"""
    selected = context.get("selected_paper")
    if isinstance(selected, Mapping):
        selected_id = str(selected.get("arxiv_id") or selected.get("arxivId") or selected.get("id") or "").strip()
        if selected_id:
            return selected_id
    arxiv_id = str(context.get("arxiv_id") or "").strip()
    return arxiv_id or "none"


def _resolve_generation_service() -> Optional[Any]:
    """解析可选的生成服务实例。

    生成服务在这里是“增强能力”而不是“硬依赖”：
    - 有服务时，可以启用 LLM 意图识别等能力；
    - 没有服务时，系统仍可回退到规则链路继续工作。
    """
    if _get_generation_service is None:
        return None
    try:
        return _get_generation_service()
    except Exception:
        return None


def _coerce_state(state: Any) -> AgentState:
    """把 graph 返回的任意 state 形态统一转换为 AgentState。"""
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        return AgentState.model_validate(dict(state))
    return AgentState.model_validate(state)


def _state_to_response(state: Any) -> ArxivSearchResponse:
    """把内部运行态 AgentState 转换成对外响应模型。

    这个函数起到“状态出站适配器”的作用：
    上游图执行过程中会维护很多内部字段，但对外接口只需要暴露用户与调试方关心的那部分。
    这里统一完成字段映射，避免响应构造逻辑分散在多个入口里。
    """
    final_state = state if isinstance(state, AgentState) else AgentState.model_validate(state)
    return ArxivSearchResponse(
        session_id=final_state.session_id,
        intent=final_state.intent or "unsupported",
        intent_source=final_state.intent_source,
        fallback_reason=final_state.fallback_reason,
        llm_confidence=final_state.llm_confidence,
        answer=final_state.answer or "",
        search_spec=final_state.search_spec,
        pending_action=final_state.pending_action,
        paper_qa_result=final_state.paper_qa_result,
        preference_action_result=final_state.preference_action_result,
        plan=list(final_state.plan or []),
        tool_calls=list(final_state.tool_calls or []),
        papers=list(final_state.papers or []),
        warnings=list(final_state.warnings or []),
        next_actions=list(final_state.next_actions or []),
        steps=list(final_state.steps or []),
        debug=dict(final_state.debug or {}),
    )


def _build_error_response(*, message: str, detail: str, code: str) -> ArxivSearchResponse:
    """构造一个不依赖已有 state 的标准错误响应。"""
    return _build_error_response_from_state(None, message=message, detail=detail, code=code)


def _build_error_response_from_state(
    state: Optional[AgentState],
    *,
    message: str,
    detail: str,
    code: str,
) -> ArxivSearchResponse:
    """基于已有 state 构造错误响应，并尽量保留执行现场。

    与 `_build_error_response` 相比，这个版本会尽量复用已有 state 中的：
    - intent
    - warnings
    - next_actions
    - 已记录的 steps
    从而让前端和调试方看到更完整的失败上下文。
    """
    warning = f"{code}: {detail}" if detail else code
    base_state = state.model_copy(deep=True) if isinstance(state, AgentState) else AgentState()
    base_state.intent = base_state.intent or "unsupported"
    base_state.answer = f"{message}: {detail}".strip()
    base_state.warnings = list(base_state.warnings or []) + [warning]
    base_state.next_actions = list(base_state.next_actions or []) or [
        "请修正输入后重试",
        "后续可以接入论文总结或 QA 功能",
    ]
    base_state.steps = list(base_state.steps or []) + [
        AgentStep(
            step="agent_runtime",
            status="failed",
            action="Agent 在执行过程中发生异常并返回错误响应",
            inputs={"message": message, "code": code},
            outputs={},
            error=detail or code,
        )
    ]
    return _state_to_response(base_state)


def _compact_state(state: Optional[AgentState]) -> Dict[str, Any]:
    """把当前运行态压缩成适合流式事件携带的状态快照。"""
    if state is None:
        return {}
    return {
        "intent": state.intent,
        "intent_source": state.intent_source,
        "fallback_reason": state.fallback_reason,
        "llm_confidence": state.llm_confidence,
        "search_spec": state.search_spec.model_dump() if state.search_spec is not None else None,
        "pending_action": state.pending_action,
        "paper_qa_result": state.paper_qa_result,
        "preference_action_result": state.preference_action_result,
        "tool_name": state.tool_name,
        "tool_args": _compact_tool_args(state.tool_args),
        "tool_call_count": len(state.tool_calls or []),
        "paper_count": len(state.papers or []),
        "warning_count": len(state.warnings or []),
        "next_actions": list(state.next_actions or []),
        "personalized_rerank_applied": bool(state.personalized_rerank_applied),
        "debug": dict(state.debug or {}),
    }


def _compact_tool_args(tool_args: Mapping[str, Any]) -> Dict[str, Any]:
    """压缩工具参数，避免在流式事件中携带过多无效字段。"""
    return {
        key: value
        for key, value in dict(tool_args or {}).items()
        if value not in (None, "", [], {})
    }


def _should_emit_tool_call(state: Optional[AgentState]) -> bool:
    """判断当前节点状态是否值得额外发出 tool_call_start/tool_call_end 事件。"""
    return bool(state and state.intent == "arxiv_search" and state.tool_name and state.tool_args)


def _make_stream_event(*, event_type: str, sequence: int, run_id: str, data: Dict[str, Any]) -> AgentStreamEvent:
    """构造一条标准化的流式事件对象。"""
    return AgentStreamEvent(
        event_type=event_type,  # type: ignore[arg-type]
        sequence=sequence,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        data=data,
    )


def _sse_event(event: AgentStreamEvent) -> str:
    """把事件对象编码成符合 SSE 协议的字符串。"""
    return f"event: {event.event_type}\ndata: {json.dumps(event.model_dump(), ensure_ascii=False)}\n\n"


def _next_sequence(sequence: int) -> int:
    """返回当前事件序号。

    这里保留一个单独函数，是为了以后如果需要切换成统一的序号分配策略，
    可以集中修改，而不必到处改事件生成代码。
    """
    return sequence


__all__ = ["run_arxiv_search_agent", "stream_arxiv_search_agent"]
