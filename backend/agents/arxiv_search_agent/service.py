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
from langgraph.types import Command

from core.errors import ErrorCode, make_error_payload
from services.memory import MemoryService
from services.storage.database_service import DatabaseService
from utils.config import get_memory_runtime_config

try:  # pragma: no cover - optional runtime dependency for LLM parsing
    from dependencies import get_generation_service as _get_generation_service
except Exception:  # pragma: no cover
    _get_generation_service = None

from .graph import build_arxiv_search_graph
from .schemas import AgentStep, AgentStreamEvent, ArxivSearchRequest, ArxivSearchResponse, ResumeRequest
from .state import AgentState

logger = logging.getLogger(__name__)
RESUME_CHECKPOINT_NOT_FOUND_CODE = ErrorCode.RESUME_CHECKPOINT_NOT_FOUND
RESUME_CHECKPOINT_NOT_FOUND_MESSAGE = "原执行现场已失效，请重新发起论文解析或问答请求。"

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


class ResumeCheckpointNotFoundError(RuntimeError):
    """表示用户发起 resume 时，后端已没有可恢复的 LangGraph 执行现场。"""

    code = RESUME_CHECKPOINT_NOT_FOUND_CODE

    def __init__(self, *, thread_id: str, reason: str = "checkpoint_missing") -> None:
        self.thread_id = thread_id
        self.reason = reason
        super().__init__(f"未找到可恢复的执行现场，thread_id={thread_id}，reason={reason}")


def _ensure_session_id(session_id: Optional[str]) -> str:
    """统一补全可复用的 session_id。

    LangGraph 的 thread_id 需要稳定，当前阶段直接让业务 session_id 与 thread_id 对齐，
    这样既能继续满足前端对 session_id 的依赖，也能避免“同一次会话每轮都新开执行线程”。
    """
    normalized = str(session_id or "").strip()
    return normalized or str(uuid4())


def _build_langgraph_config(thread_id: str) -> Dict[str, Any]:
    """构建 LangGraph 运行配置。

    这里只放稳定、轻量的恢复标识，避免把大对象直接塞进 checkpoint config。
    """
    return {"configurable": {"thread_id": thread_id}}


def _is_resume_request(request: ArxivSearchRequest) -> bool:
    """统一判断当前请求是否走 interrupt 恢复主路径。"""
    return request.resume is not None


def _build_resume_payload(resume: ResumeRequest) -> Dict[str, Any]:
    """把前端 resume 请求裁剪成传给 Command(resume=...) 的轻量 payload。

    这里只保留执行恢复真正需要的字段，避免把完整请求上下文重复写入 checkpoint 恢复链路。
    """
    payload: Dict[str, Any] = {"decision": resume.decision}
    if resume.note:
        payload["note"] = resume.note
    if resume.step_id:
        payload["step_id"] = resume.step_id
    if resume.interrupt_id:
        payload["interrupt_id"] = resume.interrupt_id
    if resume.edited_arguments:
        payload["edited_arguments"] = dict(resume.edited_arguments)
    return payload


def _ensure_resume_checkpoint(graph: Any, thread_id: str) -> None:
    """在恢复前先校验 checkpoint 是否存在，避免把错误 session 当成新请求执行。

    真实 LangGraph 环境可以通过 get_state 检查线程是否已有执行现场；
    测试桩没有实现时则跳过这层校验，由集成测试里的 fake graph 显式模拟。
    """
    get_state = getattr(graph, "get_state", None)
    if not callable(get_state):
        return

    graph_state = get_state(config=_build_langgraph_config(thread_id))
    if not _has_resume_checkpoint(graph_state):
        # checkpoint 缺失是可预期的恢复失败，不应进入通用 Agent runtime error 分支。
        raise ResumeCheckpointNotFoundError(thread_id=thread_id)


def _has_resume_checkpoint(graph_state: Any) -> bool:
    """判断 LangGraph 返回的线程状态是否代表存在可恢复现场。

    真实 LangGraph 在未命中 checkpoint 时可能返回 None，也可能返回空的 StateSnapshot；
    这里统一收敛判断，避免把空快照误当作可 resume 的执行线程。
    """
    if graph_state is None:
        return False
    if isinstance(graph_state, Mapping):
        return bool(graph_state)

    values = getattr(graph_state, "values", None)
    next_steps = getattr(graph_state, "next", None)
    if values is not None or next_steps is not None:
        return bool(values) or bool(next_steps)

    # 未知状态对象保持向后兼容：只要不是明确的空快照，就交给 LangGraph 自身恢复逻辑处理。
    return True


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


def _build_initial_agent_state(
    normalized_request: ArxivSearchRequest,
    *,
    resolved_session_id: str,
    request_context: Dict[str, Any],
    user_memory_debug: Dict[str, Any],
) -> AgentState:
    """统一构造同步与流式入口共享的初始 AgentState。"""
    return AgentState(
        user_id=normalized_request.user_id,
        session_id=resolved_session_id,
        message=normalized_request.message,
        # 保留业务 memory 的上下文增强职责，但执行现场恢复改由 LangGraph checkpoint 承担。
        context=request_context,
        pending_action=request_context.get("pending_action"),
        paper_qa_result=request_context.get("paper_qa_result"),
        debug=dict(user_memory_debug or {}),
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
        resolved_session_id = _ensure_session_id(resolved_session_id)
        graph_config = _build_langgraph_config(resolved_session_id)
        # 入口日志只记录状态摘要，便于排查“前端传了但后端没识别到”的问题，不直接打出完整上下文内容。
        logger.debug(
            "arxiv_agent request received: session_id=%s thread_id=%s message=%s context_keys=%s pending_action_status=%s paper_qa_status=%s selected_arxiv_id=%s",
            resolved_session_id,
            resolved_session_id,
            normalized_request.message,
            sorted(request_context.keys()),
            _safe_status(request_context.get("pending_action")),
            _safe_status(request_context.get("paper_qa_result")),
            _safe_selected_arxiv_id(request_context),
        )
        # 第 3 步：解析生成服务，并统一构造图对象。
        generation_service = _resolve_generation_service()
        graph = build_arxiv_search_graph(generation_service=generation_service)

        if _is_resume_request(normalized_request):
            # resume 路径必须复用同一个 thread_id，并直接从 interrupt 位置恢复，
            # 不能重新构造一轮完整业务初始状态，否则会把确认恢复退化回“伪恢复”。
            _ensure_resume_checkpoint(graph, resolved_session_id)
            resume_payload = _build_resume_payload(normalized_request.resume)
            final_state = _coerce_state(graph.invoke(Command(resume=resume_payload), config=graph_config))
        else:
            initial_state = _build_initial_agent_state(
                normalized_request,
                resolved_session_id=resolved_session_id,
                request_context=request_context,
                user_memory_debug=user_memory_debug,
            )
            # 普通请求仍从完整初始状态进入主图，保持搜索/推荐/QA 等非确认链路行为不变。
            final_state = _coerce_state(graph.invoke(initial_state.model_dump(), config=graph_config))

        # 第 5 步：把跨轮 Agent memory 回写到后端 session。
        _persist_agent_session_memory(final_state)

        # 第 6 步：把内部状态转换成对外响应模型。
        return _state_to_response(final_state)
    except ValidationError as exc:
        logger.exception("arxiv_agent request validation failed: message=%s", normalized_request.message)
        return _build_error_response(
            message="请求参数校验失败",
            detail=str(exc),
            code=ErrorCode.REQUEST_VALIDATION_ERROR,
        )
    except ResumeCheckpointNotFoundError as exc:
        _log_resume_checkpoint_not_found(
            session_id=resolved_session_id if "resolved_session_id" in locals() else None,
            thread_id=exc.thread_id,
            is_resume=_is_resume_request(normalized_request) if "normalized_request" in locals() else True,
            reason=exc.reason,
        )
        return _build_resume_checkpoint_not_found_response(
            session_id=resolved_session_id if "resolved_session_id" in locals() else None,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("arxiv_agent runtime failed: session_id=%s message=%s", resolved_session_id if 'resolved_session_id' in locals() else None, normalized_request.message)
        return _build_error_response(
            message="arXiv 搜索 Agent 运行失败",
            detail=str(exc),
            code=ErrorCode.AGENT_RUNTIME_ERROR,
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
            resolved_session_id = _ensure_session_id(resolved_session_id)
            graph_config = _build_langgraph_config(resolved_session_id)
            logger.debug(
                "arxiv_agent stream start: run_id=%s session_id=%s thread_id=%s message=%s context_keys=%s pending_action_status=%s paper_qa_status=%s selected_arxiv_id=%s",
                run_id,
                resolved_session_id,
                resolved_session_id,
                normalized_request.message,
                sorted(request_context.keys()),
                _safe_status(request_context.get("pending_action")),
                _safe_status(request_context.get("paper_qa_result")),
                _safe_selected_arxiv_id(request_context),
            )
            graph = build_arxiv_search_graph(generation_service=generation_service)
            graph_input: Any

            if _is_resume_request(normalized_request):
                _ensure_resume_checkpoint(graph, resolved_session_id)
                graph_input = Command(resume=_build_resume_payload(normalized_request.resume))
            else:
                current_state = _build_initial_agent_state(
                    normalized_request,
                    resolved_session_id=resolved_session_id,
                    request_context=request_context,
                    user_memory_debug=user_memory_debug,
                )
                graph_input = current_state.model_dump()

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
            for update in graph.stream(graph_input, config=graph_config, stream_mode="updates"):
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

                # __interrupt__ 是正常的确认暂停信号，不是 AgentState。
                # 流式模式下要把它还原成 waiting_confirmation 状态返回给前端，避免误报成运行时异常。
                if step_name == "__interrupt__":
                    confirmation_payload = _extract_interrupt_payload(step_payload)
                    if not confirmation_payload:
                        raise ValueError("interrupt payload missing confirmation request")
                    current_state = _apply_stream_interrupt_state(previous_state, confirmation_payload)
                else:
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
        except ResumeCheckpointNotFoundError as exc:
            # resume checkpoint 缺失要作为明确业务失败返回，避免前端继续保留失效确认卡片。
            _log_resume_checkpoint_not_found(
                session_id=resolved_session_id if "resolved_session_id" in locals() else None,
                thread_id=exc.thread_id,
                is_resume=_is_resume_request(normalized_request),
                reason=exc.reason,
            )
            error_response = _build_resume_checkpoint_not_found_response(
                session_id=resolved_session_id if "resolved_session_id" in locals() else None,
                detail=str(exc),
            )
            yield _sse_event(
                _make_stream_event(
                    event_type="exception",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "message": RESUME_CHECKPOINT_NOT_FOUND_MESSAGE,
                        "detail": str(exc),
                        "code": RESUME_CHECKPOINT_NOT_FOUND_CODE,
                        "error": make_error_payload(
                            code=RESUME_CHECKPOINT_NOT_FOUND_CODE,
                            message=RESUME_CHECKPOINT_NOT_FOUND_MESSAGE,
                            detail=str(exc),
                            recoverable=False,
                        ),
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
                        "code": RESUME_CHECKPOINT_NOT_FOUND_CODE,
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
                code=ErrorCode.AGENT_RUNTIME_ERROR,
            )
            yield _sse_event(
                _make_stream_event(
                    event_type="exception",
                    sequence=_next_sequence(sequence),
                    run_id=run_id,
                    data={
                        "message": "Agent 流式执行异常",
                        "detail": str(exc),
                        "code": ErrorCode.AGENT_RUNTIME_ERROR,
                        "error": make_error_payload(
                            code=ErrorCode.AGENT_RUNTIME_ERROR,
                            message="Agent 流式执行异常",
                            detail=str(exc),
                            recoverable=True,
                        ),
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


def _extract_interrupt_payload(step_payload: Any) -> Optional[Dict[str, Any]]:
    """从 LangGraph interrupt 更新中提取确认请求载荷。"""
    if isinstance(step_payload, tuple):
        for item in step_payload:
            value = getattr(item, "value", None)
            if isinstance(value, Mapping):
                return dict(value)
    value = getattr(step_payload, "value", None)
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _build_pending_action_from_confirmation(confirmation_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """把确认请求载荷映射成前端沿用的 pending_action 展示结构。

    这个结构只负责让前端继续显示确认卡片；实际恢复现场仍由 LangGraph checkpointer
    和后续 Command(resume=...) 承担，避免把业务摘要误用成执行栈。
    """
    target_paper = dict(confirmation_payload.get("target_paper") or {}) if isinstance(confirmation_payload.get("target_paper"), Mapping) else {}
    arguments_summary = dict(confirmation_payload.get("arguments_summary") or {}) if isinstance(confirmation_payload.get("arguments_summary"), Mapping) else {}
    return {
        "type": "tool_approval",
        "status": "waiting_confirmation",
        "decision": None,
        "step_id": confirmation_payload.get("step_id"),
        "tool_name": confirmation_payload.get("tool_name"),
        "action_type": confirmation_payload.get("action_type"),
        "side_effect_level": confirmation_payload.get("side_effect_level"),
        "reason": confirmation_payload.get("reason"),
        "title": target_paper.get("title") or confirmation_payload.get("title"),
        "title_text": confirmation_payload.get("title"),
        "description": confirmation_payload.get("description"),
        "arxiv_id": target_paper.get("arxiv_id"),
        "original_question": confirmation_payload.get("original_question"),
        "target_paper": target_paper or None,
        "allowed_decisions": [item.get("code") for item in list(confirmation_payload.get("allowed_decisions") or []) if isinstance(item, Mapping) and item.get("code")],
        "allow_argument_edit": bool(confirmation_payload.get("allow_argument_edit")),
        "allow_reject": bool(confirmation_payload.get("allow_reject", True)),
        "allow_note": bool(confirmation_payload.get("allow_note", True)),
        "arguments_summary": arguments_summary,
        "confirmation_request": dict(confirmation_payload),
        "thread_id": confirmation_payload.get("thread_id"),
        "session_id": confirmation_payload.get("session_id"),
        "plan_id": confirmation_payload.get("plan_id"),
        "trace_id": confirmation_payload.get("trace_id"),
        "qa_question": arguments_summary.get("qa_question") or arguments_summary.get("question"),
    }


def _apply_stream_interrupt_state(previous_state: Optional[AgentState], confirmation_payload: Mapping[str, Any]) -> AgentState:
    """把 interrupt 载荷还原成可对外返回的待确认 AgentState。"""
    next_state = previous_state.model_copy(deep=True) if isinstance(previous_state, AgentState) else AgentState()
    pending_action = _build_pending_action_from_confirmation(confirmation_payload)
    next_state.pending_action = pending_action
    next_state.paper_qa_result = {
        "status": "waiting_confirmation",
        "pending_confirmation": dict(confirmation_payload),
        "arxiv_id": pending_action.get("arxiv_id"),
        "title": pending_action.get("title"),
        "original_question": pending_action.get("original_question"),
        "qa_question": pending_action.get("qa_question"),
        "question": pending_action.get("qa_question"),
    }
    debug = dict(next_state.debug or {})
    debug["pending_confirmation"] = dict(confirmation_payload)
    debug["agent_turn"] = {
        **dict(debug.get("agent_turn") or {}),
        "status": "waiting_confirmation",
    }
    next_state.debug = debug
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="run_agent_turn",
            status="success",
            action="等待用户确认是否继续执行论文解析与索引构建",
            inputs={"step_id": confirmation_payload.get("step_id"), "tool_name": confirmation_payload.get("tool_name")},
            outputs={"status": "waiting_confirmation", "pending_action": pending_action},
            error=None,
        )
    ]
    return next_state


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
        goal=final_state.goal,
        execution_plan=final_state.execution_plan,
        plan_runtime=final_state.plan_runtime,
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


def _build_resume_checkpoint_not_found_response(*, session_id: Optional[str], detail: str) -> ArxivSearchResponse:
    """构造 resume 现场失效的专用响应。

    该场景不是 Agent 执行崩溃，而是内存 checkpoint 已无法命中；响应里必须主动清空确认态，
    让前端停止展示旧按钮，并提示用户重新发起论文解析或问答。
    """
    base_state = AgentState(
        session_id=session_id,
        intent="unsupported",
        answer=RESUME_CHECKPOINT_NOT_FOUND_MESSAGE,
        pending_action=None,
        paper_qa_result={
            "status": "failed",
            "error_code": RESUME_CHECKPOINT_NOT_FOUND_CODE,
            "message": RESUME_CHECKPOINT_NOT_FOUND_MESSAGE,
        },
        warnings=[RESUME_CHECKPOINT_NOT_FOUND_MESSAGE],
        next_actions=["请重新发起论文解析或问答请求"],
    )
    base_state.errors = [
        {
            "step": "resume_checkpoint",
            "code": RESUME_CHECKPOINT_NOT_FOUND_CODE,
            "message": RESUME_CHECKPOINT_NOT_FOUND_MESSAGE,
            "detail": detail,
            "recoverable": False,
        }
    ]
    base_state.debug = {
        "runtime_error": {
            "code": RESUME_CHECKPOINT_NOT_FOUND_CODE,
            "message": RESUME_CHECKPOINT_NOT_FOUND_MESSAGE,
            "detail": detail,
            "recoverable": False,
        },
        "confirmation_state": "failed",
    }
    base_state.steps = [
        AgentStep(
            step="resume_checkpoint",
            status="failed",
            action="恢复执行现场失败，已清空待确认状态",
            inputs={"session_id": session_id, "code": RESUME_CHECKPOINT_NOT_FOUND_CODE},
            outputs={"pending_action": None, "paper_qa_status": "failed"},
            error=RESUME_CHECKPOINT_NOT_FOUND_CODE,
        )
    ]
    return _state_to_response(base_state)


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
    base_state = state.model_copy(deep=True) if isinstance(state, AgentState) else AgentState()
    base_state.intent = base_state.intent or "unsupported"
    base_state.answer = message
    base_state.warnings = list(base_state.warnings or []) + [message]
    base_state.errors = list(base_state.errors or []) + [
        {
            "step": "agent_runtime",
            "code": code,
            "message": message,
            "detail": detail,
            "recoverable": False,
        }
    ]
    debug = dict(base_state.debug or {})
    debug["runtime_error"] = {
        "code": code,
        "message": message,
        "detail": detail,
        "recoverable": False,
    }
    base_state.debug = debug
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
            error=code,
        )
    ]
    return _state_to_response(base_state)


def _log_resume_checkpoint_not_found(
    *,
    session_id: Optional[str],
    thread_id: Optional[str],
    is_resume: bool,
    reason: str,
) -> None:
    """记录 resume 失败的关键定位字段，便于区分重启、多 worker 或 checkpoint 清理导致的问题。"""
    logger.warning(
        "arxiv_agent resume failed: session_id=%s thread_id=%s is_resume=%s reason=%s code=%s",
        session_id,
        thread_id,
        is_resume,
        reason,
        RESUME_CHECKPOINT_NOT_FOUND_CODE,
    )


def _compact_state(state: Optional[AgentState]) -> Dict[str, Any]:
    """把当前运行态压缩成适合流式事件携带的状态快照。"""
    if state is None:
        return {}
    execution_plan_runtime = dict((state.debug or {}).get("execution_plan_runtime") or {})
    execution_plan_runtime_steps = {
        str(step.get("step_id") or "").strip(): step
        for step in list(execution_plan_runtime.get("steps", []) or [])
        if isinstance(step, Mapping) and str(step.get("step_id") or "").strip()
    }
    return {
        "intent": state.intent,
        "intent_source": state.intent_source,
        "fallback_reason": state.fallback_reason,
        "llm_confidence": state.llm_confidence,
        "search_spec": state.search_spec.model_dump() if state.search_spec is not None else None,
        "goal": state.goal.model_dump() if state.goal is not None else None,
        "execution_plan": [_compact_execution_plan_step(step, runtime_step=execution_plan_runtime_steps.get(step.step_id)) for step in list((state.execution_plan.steps if state.execution_plan else []) or [])],
        "execution_plan_metadata": state.execution_plan.metadata if state.execution_plan is not None else {},
        "execution_plan_summary": {
            "plan_id": state.execution_plan.plan_id if state.execution_plan is not None else None,
            "entry_step_ids": list(state.execution_plan.entry_step_ids or []) if state.execution_plan is not None else [],
            "final_step_ids": list(state.execution_plan.final_step_ids or []) if state.execution_plan is not None else [],
            "step_count": len((state.execution_plan.steps if state.execution_plan else []) or []),
            "step_ids": [step.step_id for step in list((state.execution_plan.steps if state.execution_plan else []) or [])],
            "action_types": [step.action_type for step in list((state.execution_plan.steps if state.execution_plan else []) or [])],
            "statuses": [step.status for step in list((state.execution_plan.steps if state.execution_plan else []) or [])],
            "current_step_id": execution_plan_runtime.get("current_step_id"),
            "current_action_type": execution_plan_runtime.get("current_action_type"),
            "next_executable_step_id": execution_plan_runtime.get("next_executable_step_id"),
            "next_executable_action_type": execution_plan_runtime.get("next_executable_action_type"),
            "status_counts": dict(execution_plan_runtime.get("status_counts") or {}),
        },
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


def _compact_execution_plan_step(step: Any, runtime_step: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """压缩 execution_plan 单步信息，便于前端展示规划状态。"""
    payload = {
        "step_id": getattr(step, "step_id", None),
        "action_type": getattr(step, "action_type", None),
        "tool_name": getattr(step, "tool_name", None),
        "output_key": getattr(step, "output_key", None),
        "status": getattr(step, "status", None),
        "depends_on": list(getattr(step, "depends_on", []) or []),
    }
    if isinstance(runtime_step, Mapping):
        for key in ("blocked_by", "can_execute", "is_current", "last_tool_observation"):
            value = runtime_step.get(key)
            if value not in (None, "", [], {}):
                payload[key] = value
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


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
