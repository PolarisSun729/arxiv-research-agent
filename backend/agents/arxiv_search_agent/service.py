
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from langgraph.types import Command

from core.errors import ErrorCode, make_error_payload
from services.context_lifecycle import ContextLifecycleService
from services.memory import MemoryService
from services.storage.sqlite import StorageContainer
from services.storage.sqlite.stores import AgentRuntimeCheckpointStore, LangGraphCheckpointStore
from utils.config import get_memory_runtime_config
from utils.logging_utils import RequestTrace, info_event

try:  # pragma: no cover - optional runtime dependency for LLM parsing
    from dependencies import get_generation_service as _get_generation_service
except Exception:  # pragma: no cover
    _get_generation_service = None

try:  # pragma: no cover - 轻量测试可能只装配 Agent stub，不提供完整后台组合根
    from dependencies import get_background_work_coordinator as _get_background_work_coordinator
except Exception:  # pragma: no cover
    _get_background_work_coordinator = None

from .graph import build_arxiv_search_graph
from .execution.interaction_runtime import InteractionRuntimeError, InteractionRuntimeService
from .execution.interactions import AgentInteraction, InteractionResumeRequest
from .runtime_checkpoint import (
    CHECKPOINT_STATUS_CANCELLED,
    CHECKPOINT_STATUS_COMPLETED,
    CHECKPOINT_STATUS_FAILED,
    CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB,
    AgentRuntimeCheckpointManager,
    build_agent_checkpointer,
)
from .schemas import AgentRuntimeState, AgentStep, AgentStreamEvent, ArxivSearchRequest, ArxivSearchResponse
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


def _write_request_trace(trace: RequestTrace, *, reason: str) -> Optional[str]:
    """请求 trace 是日志预览的补充；写入失败不能影响真实业务响应。"""
    try:
        trace_path = trace.write(reason=reason)
    except Exception as exc:  # pragma: no cover - trace 写入失败不应阻断主链路
        logger.warning("request trace write failed: run_id=%s reason=%s error=%s", trace.run_id, reason, exc)
        return None
    if trace_path:
        info_event(logger, "request.trace_written", run_id=trace.run_id, reason=reason, trace_path=trace_path)
    return trace_path


def _build_memory_service(storage: StorageContainer) -> MemoryService:
    """Agent 入口负责把存储容器拆成具体 Store，业务服务不接收万能数据库对象。"""
    return MemoryService(
        paper_catalog_store=storage.paper_catalog,
        user_preference_store=storage.user_preferences,
        interest_vector_store=storage.interest_vectors,
        paper_profile_evidence_store=storage.paper_profile_evidence,
        paper_chat_session_store=storage.paper_chat_sessions,
        paper_chat_message_store=storage.paper_chat_messages,
        paper_note_store=storage.paper_notes,
        profile_event_store=storage.profile_events,
        profile_build_job_store=storage.profile_build_jobs,
        research_profile_store=storage.research_profiles,
        agent_session_store=storage.agent_sessions,
        generation_service=_resolve_generation_service(),
    )


def _build_langgraph_config(thread_id: str) -> Dict[str, Any]:
    """构建 LangGraph 运行配置。

    这里只放稳定、轻量的恢复标识，避免把大对象直接塞进 checkpoint config。
    """
    return {"configurable": {"thread_id": thread_id}}


def _is_resume_request(request: ArxivSearchRequest) -> bool:
    """统一判断当前请求是否走 interrupt 恢复主路径。"""
    return request.resume is not None


def _build_resume_payload(resume: InteractionResumeRequest) -> Dict[str, Any]:
    """只投影业务 interaction 响应，禁止把 LangGraph 或工具内部身份暴露给客户端。"""

    return resume.model_dump(mode="json", exclude_none=True)


def _build_runtime_checkpoint_manager(runtime_checkpoint_store: AgentRuntimeCheckpointStore) -> AgentRuntimeCheckpointManager:
    """创建业务 runtime checkpoint 管理器。

    管理器只负责可恢复现场的持久化和校验，不读取普通会话展示状态，
    避免前端展示镜像反向驱动真实恢复。
    """
    return AgentRuntimeCheckpointManager(runtime_checkpoint_store=runtime_checkpoint_store)


def _build_agent_context_lifecycle_debug(
    *,
    runtime_checkpoint_store: AgentRuntimeCheckpointStore,
    approval_store: Any,
    user_id: Optional[str],
    session_id: str,
    user_memory_debug: Mapping[str, Any],
) -> Dict[str, Any]:
    """构造 Agent 本轮上下文健康度，不读取完整 checkpoint 或消息正文。"""
    context_merge_debug = dict((user_memory_debug or {}).get("context_merge") or {})
    # approval_store 只作为依赖接入健康度信号；这里不读取授权明细，避免 debug 路径反向消费交互状态。
    approval_store_missing = approval_store is None
    return ContextLifecycleService(agent_runtime_checkpoint_store=runtime_checkpoint_store).build_agent_health_debug(
        user_id=str(user_id or "").strip(),
        session_id=session_id,
        context_merge_debug=context_merge_debug,
        degraded={
            "memory_load_failed": not bool((user_memory_debug or {}).get("context_merge")),
            "frontend_rejected_fields": context_merge_debug.get("rejected_frontend_fields"),
            "approval_store_missing": approval_store_missing,
        },
    )


def _build_agent_graph(
    generation_service: Optional[Any] = None,
    *,
    langgraph_checkpoint_store: LangGraphCheckpointStore,
    runtime_checkpoint_store: AgentRuntimeCheckpointStore,
    approval_store: Any,
    background_work_coordinator: Any = None,
) -> Any:
    """构建带持久化 checkpointer 的 Agent 图。"""
    checkpointer = build_agent_checkpointer(langgraph_checkpoint_store=langgraph_checkpoint_store)
    return build_arxiv_search_graph(
        generation_service=generation_service,
        checkpointer=checkpointer,
        runtime_checkpoint_store=runtime_checkpoint_store,
        approval_store=approval_store,
        background_work_coordinator=background_work_coordinator,
    )


def _resolve_background_work_coordinator() -> Any:
    """从组合根取得后台协调器；不可用时返回 None，由批准入口显式拒绝而不是同步回退。"""
    if not callable(_get_background_work_coordinator):
        return None
    try:
        return _get_background_work_coordinator()
    except Exception as exc:
        logger.exception("Failed to resolve Agent background work coordinator: %s", exc)
        return None


def _ensure_resume_checkpoint(
    graph: Any,
    thread_id: str,
    *,
    interaction_runtime: Optional[InteractionRuntimeService] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    resume_request: Optional[InteractionResumeRequest] = None,
) -> Dict[str, Any]:
    """在恢复前同时校验业务 checkpoint 和 LangGraph checkpoint。

    先验证 LangGraph 现场存在，再原子解析业务 interaction；这样图现场缺失时不会误消费授权。
    """
    normalized_session_id = session_id or thread_id
    get_state = getattr(graph, "get_state", None)
    if not callable(get_state):
        graph_state = {"test_stub": True}
    else:
        try:
            graph_state = get_state(config=_build_langgraph_config(thread_id))
        except AttributeError:
            graph_state = {"test_stub": True}

    if not _has_resume_checkpoint(graph_state):
        # checkpoint 缺失是可预期的恢复失败，不应进入通用 Agent runtime error 分支。
        raise ResumeCheckpointNotFoundError(thread_id=thread_id)

    if interaction_runtime is None or resume_request is None:
        raise ResumeCheckpointNotFoundError(thread_id=thread_id, reason="interaction_runtime_missing")
    try:
        return interaction_runtime.resolve(
            user_id=str(user_id or "").strip(),
            session_id=normalized_session_id,
            thread_id=thread_id,
            request=resume_request,
        )
    except InteractionRuntimeError as exc:
        raise ResumeCheckpointNotFoundError(thread_id=thread_id, reason=exc.reason) from exc


def _apply_background_work_started_state(state: Any, resume_payload: Mapping[str, Any]) -> AgentState:
    """把批准事务的后台 ticket 投影成响应状态，不消费 LangGraph interrupt。"""
    current_state = _coerce_state(state).model_copy(deep=True)
    background_work = dict(resume_payload.get("background_work") or {})
    step_id = str(resume_payload.get("step_id") or "").strip()
    current_state.interaction = None
    if current_state.plan_runtime is not None:
        current_state.plan_runtime.interaction = None
        current_state.plan_runtime.turn_status = "waiting_background_job"
        if step_id:
            current_state.plan_runtime.step_status[step_id] = "waiting_background_job"
        current_state.plan_runtime.last_step_output = {
            "step_id": step_id,
            "background_work": background_work,
        }
    if current_state.runtime_state is not None:
        current_state.runtime_state.interaction = None
        current_state.runtime_state.turn_status = "waiting_background_job"
        if step_id:
            current_state.runtime_state.step_status[step_id] = "waiting_background_job"
        current_state.runtime_state.last_step_output = {
            "step_id": step_id,
            "background_work": background_work,
        }
    current_state.paper_qa_result = {
        "status": "waiting_background_job",
        "background_work": background_work,
    }
    current_state.answer = "索引已转入后台构建，完成后会从原执行现场继续回答。"
    return current_state


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


def _state_from_graph_snapshot(graph_state: Any) -> Optional[AgentState]:
    """从 LangGraph 快照中提取待恢复状态，用于在 resume 后预告即将执行的工具。

    SSE 的 updates 模式只有节点结束后才产出事件；确认恢复后如果马上进入耗时索引构建，
    必须先从 checkpoint 快照推断当前 step，才能在真正阻塞前让前端看到“工具执行中”。
    """
    if graph_state is None:
        return None
    values = graph_state.get("values") if isinstance(graph_state, Mapping) else getattr(graph_state, "values", None)
    if values is None and isinstance(graph_state, Mapping):
        values = graph_state
    if not values:
        return None
    try:
        return _coerce_state(values)
    except Exception:
        return None


def _load_graph_snapshot_state(graph: Any, thread_id: str) -> Optional[AgentState]:
    """读取当前 thread 的 checkpoint 状态；读取失败只影响进度提示，不影响 resume 主流程。"""
    get_state = getattr(graph, "get_state", None)
    if not callable(get_state):
        return None
    try:
        return _state_from_graph_snapshot(get_state(config=_build_langgraph_config(thread_id)))
    except Exception as exc:
        logger.debug("arxiv_agent load graph snapshot for progress failed: thread_id=%s error=%s", thread_id, exc)
        return None


def _inject_user_memory_context(
    request_context: Dict[str, Any],
    user_id: Optional[str],
    *,
    memory_service: MemoryService,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
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
                profile = memory_service.load_user_profile(normalized_user_id)
            except Exception:
                profile = {}
            if isinstance(profile, dict) and profile:
                enriched_context["research_profile"] = profile
                debug_flags["profile_applied"] = True
        return enriched_context, debug_flags

    return enriched_context, debug_flags


def _load_agent_request_context(
    normalized_request: ArxivSearchRequest,
    *,
    memory_service: MemoryService,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    frontend_context, user_memory_debug = _inject_user_memory_context(
        dict(normalized_request.context or {}),
        normalized_request.user_id,
        memory_service=memory_service,
    )
    try:
        memory_payload = memory_service.load_agent_memory(
            normalized_request.user_id,
            normalized_request.session_id,
            frontend_context=frontend_context,
        )
    except Exception as exc:
        logger.warning("Failed to load agent session memory: user_id=%s session_id=%s error=%s", normalized_request.user_id, normalized_request.session_id, exc)
        return frontend_context, None, normalized_request.session_id, user_memory_debug

    merged_context = dict(memory_payload.get("merged_context") or frontend_context)
    resolved_session_id = str(memory_payload.get("session_id") or normalized_request.session_id or "").strip() or None
    context_merge_debug = dict(memory_payload.get("context_merge_debug") or {})
    debug_payload = {
        **dict(user_memory_debug or {}),
        "context_merge": context_merge_debug,
    }
    return merged_context, memory_payload, resolved_session_id, debug_payload


def _backend_context_value(memory_payload: Optional[Mapping[str, Any]], key: str) -> Any:
    """只从后端会话记忆读取权威字段，避免前端 context 旧缓存反向驱动 Agent 初始状态。"""
    backend_memory = dict((memory_payload or {}).get("backend_memory") or {})
    return backend_memory.get(key)


def _persist_agent_session_memory(final_state: Any, *, memory_service: MemoryService) -> None:
    state = _coerce_state(final_state)
    try:
        memory_service.save_agent_memory(
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


def _persist_runtime_checkpoint_node(
    checkpoint_manager: AgentRuntimeCheckpointManager,
    state: Any,
    *,
    current_node: str,
) -> None:
    """把图节点执行后的现场落库。

    这里保存的是 resume 真源，不从 context 或 debug 反推出可恢复状态。
    """
    if state is None:
        return
    try:
        checkpoint_manager.persist_state(state, current_node=current_node)
    except Exception as exc:
        logger.warning(
            "Failed to persist agent runtime checkpoint node: node=%s error=%s",
            current_node,
            exc,
        )


def _persist_runtime_checkpoint_after_turn(
    checkpoint_manager: AgentRuntimeCheckpointManager,
    final_state: Any,
    *,
    is_resume: bool,
) -> None:
    """在一次同步/流式执行结束后更新 runtime checkpoint 状态。

    waiting_interaction 会持久化唯一 interaction；终态会清空交互，避免重复解析。
    """
    if final_state is None:
        return
    try:
        state = _coerce_state(final_state)
        interaction = _extract_state_interaction(state)
        if interaction:
            checkpoint_manager.persist_state(state, current_node="execute_step", next_route="waiting_interaction")
            return
        runtime_status = ""
        if state.runtime_state is not None:
            runtime_status = str(state.runtime_state.turn_status or "").strip()
        elif state.plan_runtime is not None:
            runtime_status = str(state.plan_runtime.turn_status or "").strip()
        if runtime_status == CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB:
            checkpoint_manager.persist_state(
                state,
                current_node="execute_step",
                next_route=CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB,
            )
            return
        terminal_status = _terminal_checkpoint_status(state, is_resume=is_resume)
        if terminal_status:
            checkpoint_manager.mark_terminal(
                state,
                status=terminal_status,
                error_summary=_state_error_summary(state),
            )
        else:
            checkpoint_manager.persist_state(state, current_node="finalize")
    except Exception as exc:
        logger.warning("Failed to persist agent runtime checkpoint after turn: error=%s", exc)


def _extract_state_interaction(state: AgentState) -> Optional[Dict[str, Any]]:
    """读取唯一业务交互，不从 context 或 debug 猜测等待状态。"""

    if state.runtime_state is not None and state.runtime_state.interaction is not None:
        return state.runtime_state.interaction.model_dump(mode="json")
    if state.plan_runtime is not None and state.plan_runtime.interaction is not None:
        return state.plan_runtime.interaction.model_dump(mode="json")
    return None


def _terminal_checkpoint_status(state: AgentState, *, is_resume: bool) -> Optional[str]:
    """把 Agent 最终状态映射成 checkpoint 生命周期状态。"""
    if _state_recovery_is_confirmation_rejected(state):
        return CHECKPOINT_STATUS_CANCELLED
    runtime_status = ""
    if state.runtime_state is not None:
        # runtime_state 存在时就是本轮最终业务快照，不能再让旧 plan_runtime 覆盖终态判断。
        runtime_status = str(state.runtime_state.turn_status or "").strip()
    elif state.plan_runtime is not None:
        runtime_status = str(state.plan_runtime.turn_status or "").strip()
    if runtime_status == "failed" or _state_error_summary(state):
        return CHECKPOINT_STATUS_FAILED
    if runtime_status == CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB:
        return None
    if runtime_status:
        return CHECKPOINT_STATUS_COMPLETED
    # resume 后即使 runtime_status 被旧路径漏写，也不能继续保留 waiting_confirmation 真源。
    if is_resume:
        return CHECKPOINT_STATUS_COMPLETED
    return None


def _state_recovery_is_confirmation_rejected(state: AgentState) -> bool:
    """从业务恢复策略识别用户拒绝交互。"""
    strategy: Any = None
    if state.runtime_state is not None:
    # runtime_state 已存在时不回退旧 plan_runtime，避免形成第二份恢复真源。
        strategy = state.runtime_state.recovery_strategy
    elif state.plan_runtime is not None:
        strategy = state.plan_runtime.recovery_strategy
    if not isinstance(strategy, Mapping):
        return False
    return str(strategy.get("type") or "").strip() == "skip_step" and str(strategy.get("reason") or "").strip() == "confirmation_rejected"


def _state_error_summary(state: AgentState) -> str:
    if state.runtime_state is not None:
        # 错误摘要同样以 runtime_state 为边界；旧 plan_runtime.error 不能污染已完成的业务快照。
        if state.runtime_state.failure_reason:
            return str(state.runtime_state.failure_reason)
    elif state.plan_runtime is not None and state.plan_runtime.error:
        return str(state.plan_runtime.error)
    if state.errors:
        latest_error = state.errors[-1] if isinstance(state.errors[-1], Mapping) else {}
        return str(latest_error.get("code") or latest_error.get("message") or "")
    return ""


def _mark_runtime_checkpoint_failed(
    checkpoint_manager: AgentRuntimeCheckpointManager,
    *,
    user_id: Optional[str],
    session_id: Optional[str],
    detail: str,
) -> None:
    """异常终止时尽量把 runtime checkpoint 标记为 failed，避免旧确认现场继续可 resume。"""
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        return
    try:
        checkpoint_manager.runtime_checkpoint_store.mark_agent_runtime_checkpoint_status(
            user_id=str(user_id or "").strip(),
            session_id=normalized_session_id,
            thread_id=normalized_session_id,
            status=CHECKPOINT_STATUS_FAILED,
            error_summary=detail,
            clear_interaction=True,
        )
    except Exception as exc:
        logger.warning("Failed to mark agent runtime checkpoint failed: session_id=%s error=%s", normalized_session_id, exc)


def _build_initial_agent_state(
    normalized_request: ArxivSearchRequest,
    *,
    resolved_session_id: str,
    request_context: Dict[str, Any],
    agent_memory_payload: Optional[Dict[str, Any]],
    user_memory_debug: Dict[str, Any],
    run_id: str,
) -> AgentState:
    """统一构造同步与流式入口共享的初始 AgentState。"""
    # run_id 只作为日志/trace 关联键进入内部上下文，不改变对外请求和响应契约。
    state_context = dict(request_context or {})
    state_context["run_id"] = run_id
    state_debug = dict(user_memory_debug or {})
    state_debug["run_id"] = run_id
    return AgentState(
        user_id=normalized_request.user_id,
        session_id=resolved_session_id,
        message=normalized_request.message,
        # 保留业务 memory 的上下文增强职责，但执行现场恢复改由 LangGraph checkpoint 承担。
        context=state_context,
        paper_qa_result=_backend_context_value(agent_memory_payload, "paper_qa_result"),
        debug=state_debug,
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
    run_id = str(uuid4())
    started = perf_counter()
    trace = RequestTrace(run_id=run_id, route="arxiv_agent.run", input=request)
    normalized_request: Optional[ArxivSearchRequest] = None
    try:
        # 第 1 步：先把入参统一规整成 ArxivSearchRequest，避免上层传 dict 时各处重复判断。
        normalized_request = _coerce_request(request)
        trace.input = normalized_request.message
        trace.user_id = normalized_request.user_id

        storage = StorageContainer()
        memory_service = _build_memory_service(storage)
        # 第 2 步：把前端 context 与后端 Agent session memory 合并。
        request_context, agent_memory_payload, resolved_session_id, user_memory_debug = _load_agent_request_context(
            normalized_request,
            memory_service=memory_service,
        )
        resolved_session_id = _ensure_session_id(resolved_session_id)
        trace.session_id = resolved_session_id
        graph_config = _build_langgraph_config(resolved_session_id)
        info_event(
            logger,
            "arxiv_agent.request_start",
            run_id=run_id,
            session_id=resolved_session_id,
            user_id=normalized_request.user_id,
            input=normalized_request.message,
            context_keys=sorted(request_context.keys()),
            interaction_kind=str(getattr(normalized_request.resume, "interaction_id", "") or "") or None,
            paper_qa_status=_safe_status(request_context.get("paper_qa_result")),
            selected_arxiv_id=_safe_selected_arxiv_id(request_context),
        )
        trace.add_event("arxiv_agent.request_start", session_id=resolved_session_id, user_id=normalized_request.user_id)
        # 第 3 步：解析生成服务，并统一构造图对象。
        checkpoint_manager = _build_runtime_checkpoint_manager(storage.agent_runtime_checkpoints)
        checkpoint_manager.expire_and_cleanup()
        user_memory_debug["context_lifecycle"] = _build_agent_context_lifecycle_debug(
            runtime_checkpoint_store=storage.agent_runtime_checkpoints,
            approval_store=storage.approval_grants,
            user_id=normalized_request.user_id,

            session_id=resolved_session_id,
            user_memory_debug=user_memory_debug,
        )
        generation_service = _resolve_generation_service()
        background_work_coordinator = _resolve_background_work_coordinator()
        graph = _build_agent_graph(
            generation_service=generation_service,
            langgraph_checkpoint_store=storage.langgraph_checkpoints,
            runtime_checkpoint_store=storage.agent_runtime_checkpoints,
            approval_store=storage.approval_grants,
            background_work_coordinator=background_work_coordinator,
        )

        if _is_resume_request(normalized_request):
            # resume 路径必须复用同一个 thread_id，并直接从 interrupt 位置恢复，
            # 不能重新构造一轮完整业务初始状态，否则会把确认恢复退化回“伪恢复”。
            interaction_runtime = InteractionRuntimeService(
                checkpoint_store=storage.agent_runtime_checkpoints,
                approval_store=storage.approval_grants,
                background_work_coordinator=background_work_coordinator,
            )
            request_resume_payload = _build_resume_payload(normalized_request.resume)
            info_event(
                logger,
                "arxiv_agent.resume_received",
                run_id=run_id,
                session_id=resolved_session_id,
                decision=request_resume_payload.get("decision"),
                interaction_id=request_resume_payload.get("interaction_id"),
            )
            trace.add_event(
                "arxiv_agent.resume_received",
                decision=request_resume_payload.get("decision"),
                interaction_id=request_resume_payload.get("interaction_id"),
            )
            resume_payload = _ensure_resume_checkpoint(
                graph,
                resolved_session_id,
                interaction_runtime=interaction_runtime,
                user_id=normalized_request.user_id,
                session_id=resolved_session_id,
                resume_request=normalized_request.resume,
            )
            info_event(
                logger,
                "arxiv_agent.resume_checkpoint_validated",
                run_id=run_id,
                session_id=resolved_session_id,
                interaction_id=resume_payload.get("interaction_id"),
                decision=resume_payload.get("decision"),
            )
            if str(resume_payload.get("decision") or "") == "background_work_started":
                # 批准事务已经清除 interaction 并切换到 waiting_background_job；此请求在这里结束，
                # 不能再消费 LangGraph interrupt，否则会把后台执行重新变成同步工具调用。
                snapshot_state = _load_graph_snapshot_state(graph, resolved_session_id)
                final_state = _apply_background_work_started_state(snapshot_state, resume_payload)
            else:
                final_state = _coerce_state(graph.invoke(Command(resume=resume_payload), config=graph_config))
        else:
            initial_state = _build_initial_agent_state(
                normalized_request,
                resolved_session_id=resolved_session_id,
                request_context=request_context,
                agent_memory_payload=agent_memory_payload,
                user_memory_debug=user_memory_debug,
                run_id=run_id,
            )
            # 普通请求仍从完整初始状态进入主图，保持搜索/推荐/QA 等非确认链路行为不变。
            final_state = _coerce_state(graph.invoke(initial_state.model_dump(), config=graph_config))

        _persist_runtime_checkpoint_after_turn(checkpoint_manager, final_state, is_resume=_is_resume_request(normalized_request))
        # 第 5 步：把跨轮 Agent memory 回写到后端 session。
        _persist_agent_session_memory(final_state, memory_service=memory_service)

        # 第 6 步：把内部状态转换成对外响应模型。
        final_response = _state_to_response(final_state)
        if final_state.fallback_reason:
            trace.mark_fallback()
        trace.set_output(final_response.model_dump())
        trace.add_event(
            "arxiv_agent.request_done",
            status="success",
            intent=final_response.intent,
            paper_count=len(final_response.papers or []),
        )
        trace_path = _write_request_trace(trace, reason="auto")
        info_event(
            logger,
            "arxiv_agent.request_done",
            run_id=run_id,
            session_id=resolved_session_id,
            user_id=normalized_request.user_id,
            status="success",
            intent=final_response.intent,
            paper_count=len(final_response.papers or []),
            output=final_response.answer,
            elapsed_ms=round((perf_counter() - started) * 1000, 1),
            trace_path=trace_path,
        )
        return final_response
    except ValidationError as exc:
        message = normalized_request.message if normalized_request is not None else ""
        logger.exception("arxiv_agent request validation failed: run_id=%s message=%s", run_id, message)
        error_response = _build_error_response(
            message="请求参数校验失败",
            detail=str(exc),
            code=ErrorCode.REQUEST_VALIDATION_ERROR,
        )
        trace.mark_failed()
        trace.set_output(error_response.model_dump())
        trace.add_event("arxiv_agent.request_done", status="error", code=ErrorCode.REQUEST_VALIDATION_ERROR)
        trace_path = _write_request_trace(trace, reason="validation_error")
        info_event(
            logger,
            "arxiv_agent.request_done",
            run_id=run_id,
            status="error",
            code=ErrorCode.REQUEST_VALIDATION_ERROR,
            output=error_response.answer,
            elapsed_ms=round((perf_counter() - started) * 1000, 1),
            trace_path=trace_path,
        )
        return error_response
    except ResumeCheckpointNotFoundError as exc:
        _log_resume_checkpoint_not_found(
            session_id=resolved_session_id if "resolved_session_id" in locals() else None,
            thread_id=exc.thread_id,
            is_resume=_is_resume_request(normalized_request) if "normalized_request" in locals() else True,
            reason=exc.reason,
        )
        error_response = _build_resume_checkpoint_not_found_response(
            session_id=resolved_session_id if "resolved_session_id" in locals() else None,
            detail=str(exc),
        )
        trace.mark_failed()
        trace.set_output(error_response.model_dump())
        trace.add_event("arxiv_agent.request_done", status="error", code=RESUME_CHECKPOINT_NOT_FOUND_CODE)
        trace_path = _write_request_trace(trace, reason="resume_checkpoint_not_found")
        info_event(
            logger,
            "arxiv_agent.request_done",
            run_id=run_id,
            session_id=resolved_session_id if "resolved_session_id" in locals() else None,
            status="error",
            code=RESUME_CHECKPOINT_NOT_FOUND_CODE,
            output=error_response.answer,
            elapsed_ms=round((perf_counter() - started) * 1000, 1),
            trace_path=trace_path,
        )
        return error_response
    except Exception as exc:
        message = normalized_request.message if normalized_request is not None else ""
        logger.exception("arxiv_agent runtime failed: run_id=%s session_id=%s message=%s", run_id, resolved_session_id if 'resolved_session_id' in locals() else None, message)
        if "checkpoint_manager" in locals():
            _mark_runtime_checkpoint_failed(
                checkpoint_manager,
                user_id=normalized_request.user_id if "normalized_request" in locals() else None,
                session_id=resolved_session_id if "resolved_session_id" in locals() else None,
                detail=str(exc),
            )
        error_response = _build_error_response(
            message="arXiv 搜索 Agent 运行失败",
            detail=str(exc),
            code=ErrorCode.AGENT_RUNTIME_ERROR,
        )
        trace.mark_failed()
        trace.set_output(error_response.model_dump())
        trace.add_event("arxiv_agent.request_done", status="error", code=ErrorCode.AGENT_RUNTIME_ERROR)
        trace_path = _write_request_trace(trace, reason="runtime_error")
        info_event(
            logger,
            "arxiv_agent.request_done",
            run_id=run_id,
            session_id=resolved_session_id if "resolved_session_id" in locals() else None,
            user_id=normalized_request.user_id if normalized_request is not None else None,
            status="error",
            code=ErrorCode.AGENT_RUNTIME_ERROR,
            output=error_response.answer,
            elapsed_ms=round((perf_counter() - started) * 1000, 1),
            trace_path=trace_path,
        )
        return error_response


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
        started = perf_counter()
        trace = RequestTrace(run_id=run_id, route="arxiv_agent.stream", input=normalized_request.message)
        event_count = 0
        tool_count = 0
        confirmation_count = 0
        current_state: Optional[AgentState] = None

        try:
            storage = StorageContainer()
            memory_service = _build_memory_service(storage)
            # 阶段 B：构造与同步入口一致的初始上下文和状态，保证两条路径行为一致。
            request_context, agent_memory_payload, resolved_session_id, user_memory_debug = _load_agent_request_context(
                normalized_request,
                memory_service=memory_service,
            )
            resolved_session_id = _ensure_session_id(resolved_session_id)
            trace.session_id = resolved_session_id
            trace.user_id = normalized_request.user_id
            graph_config = _build_langgraph_config(resolved_session_id)
            info_event(
                logger,
                "arxiv_agent.stream_start",
                run_id=run_id,
                session_id=resolved_session_id,
                user_id=normalized_request.user_id,
                input=normalized_request.message,
                context_keys=sorted(request_context.keys()),
                interaction_id=str(getattr(normalized_request.resume, "interaction_id", "") or "") or None,
                paper_qa_status=_safe_status(request_context.get("paper_qa_result")),
                selected_arxiv_id=_safe_selected_arxiv_id(request_context),
            )
            trace.add_event("arxiv_agent.stream_start", session_id=resolved_session_id, user_id=normalized_request.user_id)
            checkpoint_manager = _build_runtime_checkpoint_manager(storage.agent_runtime_checkpoints)
            checkpoint_manager.expire_and_cleanup()
            user_memory_debug["context_lifecycle"] = _build_agent_context_lifecycle_debug(
                runtime_checkpoint_store=storage.agent_runtime_checkpoints,
                approval_store=storage.approval_grants,
                user_id=normalized_request.user_id,
                session_id=resolved_session_id,
                user_memory_debug=user_memory_debug,
            )
            generation_service = _resolve_generation_service()
            background_work_coordinator = _resolve_background_work_coordinator()
            graph = _build_agent_graph(
                generation_service=generation_service,
                langgraph_checkpoint_store=storage.langgraph_checkpoints,
                runtime_checkpoint_store=storage.agent_runtime_checkpoints,
                approval_store=storage.approval_grants,
                background_work_coordinator=background_work_coordinator,
            )
            graph_input: Any

            resume_payload: Optional[Dict[str, Any]] = None
            if _is_resume_request(normalized_request):
                interaction_runtime = InteractionRuntimeService(
                    checkpoint_store=storage.agent_runtime_checkpoints,
                    approval_store=storage.approval_grants,
                    background_work_coordinator=background_work_coordinator,
                )
                request_resume_payload = _build_resume_payload(normalized_request.resume)
                info_event(
                    logger,
                    "arxiv_agent.stream_resume_received",
                    run_id=run_id,
                    session_id=resolved_session_id,
                    decision=request_resume_payload.get("decision"),
                    interaction_id=request_resume_payload.get("interaction_id"),
                )
                trace.add_event(
                    "arxiv_agent.stream_resume_received",
                    decision=request_resume_payload.get("decision"),
                    interaction_id=request_resume_payload.get("interaction_id"),
                )
                resume_payload = _ensure_resume_checkpoint(
                    graph,
                    resolved_session_id,
                    interaction_runtime=interaction_runtime,
                    user_id=normalized_request.user_id,
                    session_id=resolved_session_id,
                    resume_request=normalized_request.resume,
                )
                info_event(
                    logger,
                    "arxiv_agent.stream_resume_checkpoint_validated",
                    run_id=run_id,
                    session_id=resolved_session_id,
                    interaction_id=resume_payload.get("interaction_id"),
                    decision=resume_payload.get("decision"),
                )
                current_state = _load_graph_snapshot_state(graph, resolved_session_id)
                if str(resume_payload.get("decision") or "") == "background_work_started":
                    current_state = _apply_background_work_started_state(current_state, resume_payload)
                    graph_input = None
                else:
                    graph_input = Command(resume=resume_payload)
            else:
                current_state = _build_initial_agent_state(
                    normalized_request,
                    resolved_session_id=resolved_session_id,
                    request_context=request_context,
                    agent_memory_payload=agent_memory_payload,
                    user_memory_debug=user_memory_debug,
                    run_id=run_id,
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
            event_count += 1
            sequence += 1

            if graph_input is None and resume_payload is not None:
                # 后台批准只返回任务 ticket；专用 continuation API 会在 job 成功后建立新的恢复 SSE。
                _persist_runtime_checkpoint_after_turn(checkpoint_manager, current_state, is_resume=True)
                final_response = _state_to_response(current_state)
                yield _sse_event(
                    _make_stream_event(
                        event_type="final_response",
                        sequence=_next_sequence(sequence),
                        run_id=run_id,
                        data={"response": final_response.model_dump(), "state": _compact_state(current_state)},
                    )
                )
                event_count += 1
                sequence += 1
                yield _sse_event(
                    _make_stream_event(
                        event_type="stream_end",
                        sequence=_next_sequence(sequence),
                        run_id=run_id,
                        data={"status": "background_work_started", "final_sequence": sequence},
                    )
                )
                return

            # 批准恢复时，工具身份来自后端已校验的 interaction；只放行这一个 step 的运行中提示，
            # 避免未获批的副作用工具在 UI 上被误显示为已经开始执行。
            approved_step_id = (
                str((resume_payload or {}).get("step_id") or "").strip()
                if str((resume_payload or {}).get("decision") or "").strip() == "approve"
                else None
            )
            preannounced_tool_call: Optional[Dict[str, Any]] = None
            preannounced_tool_identity: Optional[tuple[str, str, str]] = None
            if approved_step_id:
                preannounced_tool_call = _active_runtime_tool_call(current_state, approved_step_id=approved_step_id)
                if preannounced_tool_call is not None:
                    preannounced_tool_identity = _tool_call_identity(preannounced_tool_call)
                    tool_count += 1
                    # LangGraph 的 updates 事件要等节点完成后才返回；批准恢复后如果马上进入 Docling
                    # 这类长耗时工具，必须先从 checkpoint 快照预告工具开始，避免前端在转换期间只显示通用生成中。
                    yield _sse_event(
                        _make_stream_event(
                            event_type="tool_call_start",
                            sequence=_next_sequence(sequence),
                            run_id=run_id,
                            data={
                                "step": "execute_step",
                                "tool_call": preannounced_tool_call,
                                "state": _compact_state(current_state),
                            },
                        )
                    )
                    event_count += 1
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
                event_count += 1
                sequence += 1

                active_tool_call = _active_runtime_tool_call(previous_state, approved_step_id=approved_step_id)
                if active_tool_call is None and step_name == "execute_step" and preannounced_tool_call is not None:
                    # select_next_step 等前置节点可能会改变 previous_state 的投影；已预告的工具仍应在
                    # execute_step 结束时收到 tool_call_end，否则前端的 running 进度卡片无法收尾。
                    active_tool_call = preannounced_tool_call
                tool_call_started = False
                if active_tool_call is not None and (step_name == "execute_step" or _should_emit_tool_call(previous_state)):
                    tool_call_started = True
                    already_preannounced = (
                        preannounced_tool_identity is not None
                        and _tool_call_identity(active_tool_call) == preannounced_tool_identity
                    )
                    if not already_preannounced:
                        tool_count += 1
                        # 子阶段 D-2：如果当前节点会触发工具调用，则补发工具开始事件。
                        yield _sse_event(
                            _make_stream_event(
                                event_type="tool_call_start",
                                sequence=_next_sequence(sequence),
                                run_id=run_id,
                                data={
                                    "step": step_name,
                                    "tool_call": active_tool_call,
                                    "state": _compact_state(previous_state),
                                },
                            )
                        )
                        event_count += 1
                        sequence += 1

                # __interrupt__ 是正常的确认暂停信号，不是 AgentState。
                # 流式模式下要把它还原成 waiting_confirmation 状态返回给前端，避免误报成运行时异常。
                if step_name == "__interrupt__":
                    confirmation_payload = _extract_interrupt_payload(step_payload)
                    if not confirmation_payload:
                        raise ValueError("interrupt payload missing confirmation request")
                    confirmation_count += 1
                    current_state = _apply_stream_interrupt_state(previous_state, confirmation_payload)
                else:
                    current_state = _coerce_state(step_payload)
                _persist_runtime_checkpoint_node(checkpoint_manager, current_state, current_node=step_name)
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
                event_count += 1
                sequence += 1

                if tool_call_started:
                    # 子阶段 D-4：若本节点触发了工具调用，则在节点结束后补发 tool_call_end。
                    latest_tool_call = _merge_tool_call_end(active_tool_call, _latest_runtime_tool_call(current_state))
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
                    event_count += 1
                    sequence += 1
                    if preannounced_tool_identity == _tool_call_identity(active_tool_call):
                        preannounced_tool_call = None
                        preannounced_tool_identity = None

            # 阶段 E：整张图执行完成后，输出最终聚合响应和结束事件。
            if current_state is not None:
                _persist_runtime_checkpoint_after_turn(checkpoint_manager, current_state, is_resume=_is_resume_request(normalized_request))
                _persist_agent_session_memory(current_state, memory_service=memory_service)
            final_response = _state_to_response(current_state)
            trace.set_output(final_response.model_dump())
            trace.add_event(
                "arxiv_agent.stream_done",
                status="success",
                event_count=event_count,
                tool_count=tool_count,
                confirmation_count=confirmation_count,
            )
            trace_path = _write_request_trace(trace, reason="auto")
            info_event(
                logger,
                "arxiv_agent.stream_done",
                run_id=run_id,
                session_id=resolved_session_id,
                user_id=normalized_request.user_id,
                status="success",
                event_count=event_count,
                tool_count=tool_count,
                confirmation_count=confirmation_count,
                output=final_response.answer,
                elapsed_ms=round((perf_counter() - started) * 1000, 1),
                trace_path=trace_path,
            )
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
            event_count += 1
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
            event_count += 1
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
            trace.mark_failed()
            trace.set_output(error_response.model_dump())
            trace.add_event("arxiv_agent.stream_done", status="error", code=RESUME_CHECKPOINT_NOT_FOUND_CODE)
            trace_path = _write_request_trace(trace, reason="resume_checkpoint_not_found")
            info_event(
                logger,
                "arxiv_agent.stream_done",
                run_id=run_id,
                session_id=resolved_session_id if "resolved_session_id" in locals() else None,
                user_id=normalized_request.user_id,
                status="error",
                code=RESUME_CHECKPOINT_NOT_FOUND_CODE,
                event_count=event_count,
                tool_count=tool_count,
                confirmation_count=confirmation_count,
                output=error_response.answer,
                elapsed_ms=round((perf_counter() - started) * 1000, 1),
                trace_path=trace_path,
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
            event_count += 1
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
            event_count += 1
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
            event_count += 1
        except Exception as exc:
            # 阶段 F：流式过程中任何异常都转成结构化事件，而不是让连接直接中断。
            # 同时写入 traceback；前端为了稳定体验会展示泛化错误，后端日志必须保留真实失败点。
            logger.exception(
                "arxiv_agent stream runtime failed: run_id=%s session_id=%s message=%s",
                run_id,
                resolved_session_id if "resolved_session_id" in locals() else None,
                normalized_request.message,
            )
            if "checkpoint_manager" in locals():
                _mark_runtime_checkpoint_failed(
                    checkpoint_manager,
                    user_id=normalized_request.user_id,
                    session_id=resolved_session_id if "resolved_session_id" in locals() else None,
                    detail=str(exc),
                )
            error_response = _build_error_response_from_state(
                current_state,
                message="arXiv 搜索 Agent 运行失败",
                detail=str(exc),
                code=ErrorCode.AGENT_RUNTIME_ERROR,
            )
            trace.mark_failed()
            trace.set_output(error_response.model_dump())
            trace.add_event("arxiv_agent.stream_done", status="error", code=ErrorCode.AGENT_RUNTIME_ERROR)
            trace_path = _write_request_trace(trace, reason="runtime_error")
            info_event(
                logger,
                "arxiv_agent.stream_done",
                run_id=run_id,
                session_id=resolved_session_id if "resolved_session_id" in locals() else None,
                user_id=normalized_request.user_id,
                status="error",
                code=ErrorCode.AGENT_RUNTIME_ERROR,
                event_count=event_count,
                tool_count=tool_count,
                confirmation_count=confirmation_count,
                output=error_response.answer,
                elapsed_ms=round((perf_counter() - started) * 1000, 1),
                trace_path=trace_path,
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
            event_count += 1
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
            event_count += 1
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
            event_count += 1

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


def _apply_stream_interrupt_state(previous_state: Optional[AgentState], payload: Mapping[str, Any]) -> AgentState:
    """把 LangGraph interrupt 投影为唯一 interaction 状态，不生成兼容展示镜像。"""

    next_state = previous_state.model_copy(deep=True) if isinstance(previous_state, AgentState) else AgentState()
    interaction = AgentInteraction.model_validate(dict(payload))
    next_state.interaction = interaction
    next_state.paper_qa_result = {
        "status": "waiting_interaction",
        "interaction": interaction.model_dump(mode="json"),
    }
    runtime_state = next_state.runtime_state.model_copy(deep=True) if next_state.runtime_state is not None else AgentRuntimeState()
    runtime_state.interaction = interaction
    runtime_state.turn_status = "waiting_interaction"
    runtime_state.current_step_id = runtime_state.current_step_id or interaction.step_id
    runtime_state.step_status = dict(runtime_state.step_status or {})
    runtime_state.step_status[interaction.step_id] = "waiting_interaction"
    next_state.runtime_state = runtime_state
    if next_state.plan_runtime is not None:
        next_state.plan_runtime.interaction = interaction
        next_state.plan_runtime.turn_status = "waiting_interaction"
        next_state.plan_runtime.step_status = dict(next_state.plan_runtime.step_status or {})
        next_state.plan_runtime.step_status[interaction.step_id] = "waiting_interaction"
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step="execute_step",
            status="success",
            action="等待用户完成结构化交互",
            inputs={"interaction_id": interaction.interaction_id, "kind": interaction.kind},
            outputs={"status": "waiting_interaction"},
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
    tool_calls = list(final_state.tool_calls or [])
    if not tool_calls:
        # 显式执行环的工具结果主要记录在 runtime trace 中；这里补齐旧响应字段，
        # 让前端调试面板不再误判为“没有发生工具调用”。
        tool_calls = _tool_calls_from_runtime(final_state)

    return ArxivSearchResponse(
        session_id=final_state.session_id,
        intent=final_state.intent or "unsupported",
        intent_source=final_state.intent_source,
        fallback_reason=final_state.fallback_reason,
        llm_confidence=final_state.llm_confidence,
        answer=final_state.answer or "",
        search_spec=final_state.search_spec,
        goal=final_state.goal,
        research_task_profile=final_state.research_task_profile,
        execution_plan=final_state.execution_plan,
        plan_runtime=final_state.plan_runtime,
        runtime_state=final_state.runtime_state,
        interaction=final_state.interaction,
        paper_qa_result=final_state.paper_qa_result,
        preference_action_result=final_state.preference_action_result,
        plan=list(final_state.plan or []),
        tool_calls=tool_calls,
        papers=list(final_state.papers or []),
        warnings=list(final_state.warnings or []),
        next_actions=list(final_state.next_actions or []),
        steps=list(final_state.steps or []),
        debug=dict(final_state.debug or {}),
    )


def _planner_trace_summary_from_state(state: AgentState) -> Dict[str, Any]:
    """为 tool trace 附带本轮 planner 真实路径，避免前端只能从最终 answer 反推。"""
    planner_debug = dict((state.debug or {}).get("planner") or {})
    planner_summary = dict(planner_debug.get("planner_summary") or {})
    if not planner_summary and state.execution_plan is not None:
        planner_summary = dict((state.execution_plan.metadata or {}).get("planner_summary") or {})
    return {
        key: planner_summary.get(key)
        for key in (
            "requested_path",
            "selected_path",
            "final_path",
            "fallback_used",
            "fallback_reason",
        )
        if planner_summary.get(key) not in (None, "", [], {})
    }


def _tool_call_boundary_trace(detail: Mapping[str, Any]) -> Dict[str, Any]:
    """把工具层的能力边界摘要投影到轻量 trace，便于区分规则模块、实验能力和 service 直通。"""
    tool_contract = detail.get("tool_contract")
    tool_execution = detail.get("tool_execution")
    contract = dict(tool_contract or {}) if isinstance(tool_contract, Mapping) else {}
    execution = dict(tool_execution or {}) if isinstance(tool_execution, Mapping) else {}
    metadata = dict(execution.get("metadata") or {}) if isinstance(execution.get("metadata"), Mapping) else {}
    trace_payload = {
        "contract_source": contract.get("contract_source"),
        "adapter": contract.get("adapter"),
        "backend_tool_name": contract.get("backend_tool_name"),
        "clarification_stage": metadata.get("clarification_stage"),
        "analysis_source": metadata.get("analysis_source"),
        "analysis_mode": metadata.get("analysis_mode"),
        "question_source": metadata.get("question_source"),
        "recommendation_stage": metadata.get("recommendation_stage"),
        "core_service": metadata.get("core_service"),
        "core_execution_mode": metadata.get("core_execution_mode"),
        "fallback_mode": metadata.get("fallback_mode"),
        "profile_fallback": metadata.get("profile_fallback"),
        "is_llm_backed": metadata.get("is_llm_backed"),
        "is_algorithm_core_in_agent": metadata.get("is_algorithm_core_in_agent"),
    }
    return {
        key: value
        for key, value in trace_payload.items()
        if value not in (None, "", [], {})
    }


def _tool_calls_from_runtime(state: AgentState) -> list[Dict[str, Any]]:
    """把 PlanRuntime trace 投影成前端沿用的 tool_calls 摘要。

    trace 是当前执行环的真实记录，但包含大量内部细节；前端只需要工具名、状态、
    参数摘要和错误信息来展示进度与排错，所以这里做一次轻量转换。
    """
    runtime = state.plan_runtime
    if runtime is None and state.runtime_state is not None:
        runtime = getattr(state.runtime_state, "runtime", None)
    plan_steps = {

        step.step_id: step
        for step in list((runtime.plan.steps if runtime and runtime.plan else state.execution_plan.steps if state.execution_plan else []) or [])
    }
    calls: list[Dict[str, Any]] = []
    for trace in list((runtime.trace if runtime else []) or []):
        if trace.event not in {"step_succeeded", "step_failed", "confirmation_created", "confirmation_requested", "confirmation_approved", "confirmation_rejected", "confirmation_consumed"}:
            continue
        step = plan_steps.get(trace.step_id)
        detail = dict(trace.detail or {})
        tool_name = str(detail.get("tool_name") or getattr(step, "tool_name", None) or trace.step_id or "unknown")
        status = str(trace.status or "").strip() or "success"
        if trace.event == "confirmation_created":
            summary = "已创建待确认请求"
        elif trace.event == "confirmation_requested":
            summary = "等待用户确认后执行工具"
        elif trace.event == "confirmation_approved":
            summary = "用户已确认，准备执行工具"
        elif trace.event == "confirmation_rejected":
            summary = "用户已拒绝，工具不会执行"
        elif trace.event == "confirmation_consumed":
            summary = "确认请求已消费"
        elif status == "failed":
            summary = str(detail.get("failure_reason") or "工具执行失败")
        else:
            summary = "工具执行完成"
        planner_trace = _planner_trace_summary_from_state(state)
        boundary_trace = _tool_call_boundary_trace(detail)
        calls.append(
            {
                "tool_name": tool_name,
                "arguments": dict(detail.get("resolved_input") or {}),
                "status": status,
                "summary": summary,
                "trace": {
                    "step_id": trace.step_id,
                    "event": trace.event,
                    "started_at": detail.get("started_at"),
                    "finished_at": detail.get("finished_at"),
                    **planner_trace,
                    **boundary_trace,
                },
                "error": {"message": detail.get("error") or detail.get("failure_reason")} if status == "failed" else None,
            }
        )
    return calls


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
            outputs={"interaction": None, "paper_qa_status": "failed"},
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
    planner_debug = dict((state.debug or {}).get("planner") or {})
    return {
        "intent": state.intent,
        "intent_source": state.intent_source,
        "fallback_reason": state.fallback_reason,
        "llm_confidence": state.llm_confidence,
        "search_spec": state.search_spec.model_dump() if state.search_spec is not None else None,
        "goal": state.goal.model_dump() if state.goal is not None else None,
        # research_task_profile 是 intent 与计划之间的科研任务语义层；这里随状态快照一并暴露，
        # 让流式事件和调试面板能看到 Goal → intent → ResearchTaskProfile → ExecutablePlan 的对应关系。
        "research_task_profile": state.research_task_profile.model_dump() if state.research_task_profile is not None else None,
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
        # runtime_state 是新的执行现场真源；这里仅输出摘要，避免 SSE 事件携带完整工具输出和 trace。
        "runtime_state": _compact_runtime_state(state.runtime_state),
        "interaction": state.interaction.model_dump(mode="json") if state.interaction is not None else None,
        "paper_qa_result": state.paper_qa_result,
        "preference_action_result": state.preference_action_result,
        "tool_name": state.tool_name,
        "tool_args": _compact_tool_args(state.tool_args),
        "tool_call_count": len(state.tool_calls or []),
        "paper_count": len(state.papers or []),
        "warning_count": len(state.warnings or []),
        "next_actions": list(state.next_actions or []),
        "personalized_rerank_applied": bool(state.personalized_rerank_applied),
        # 这三个摘要是能力边界的用户态出口，避免前端或排障只能读取内部旧字段名来猜测本轮走了什么 planner。
        "planner_summary": dict(planner_debug.get("planner_summary") or {}),
        "clarification_summary": dict(planner_debug.get("clarification_summary") or {}),
        "recommendation_summary": dict(planner_debug.get("recommendation_summary") or {}),
        "debug": dict(state.debug or {}),
    }


def _compact_runtime_state(runtime_state: Any) -> Optional[Dict[str, Any]]:
    """压缩新的 AgentRuntimeState，供流式事件展示当前执行现场边界。"""
    if runtime_state is None:
        return None
    payload = runtime_state.model_dump() if hasattr(runtime_state, "model_dump") else dict(runtime_state or {})
    return {
        "current_step_id": payload.get("current_step_id"),
        "current_step_index": payload.get("current_step_index"),
        "step_status": dict(payload.get("step_status") or {}),
        "output_keys": sorted((payload.get("outputs") or {}).keys()),
        "last_observation": payload.get("last_observation"),
        "last_step_output_keys": sorted((payload.get("last_step_output") or {}).keys()) if isinstance(payload.get("last_step_output"), Mapping) else [],
        "needs_replan": bool(payload.get("needs_replan")),
        "interaction": payload.get("interaction"),
        "is_finished": bool(payload.get("is_finished")),
        "failure_reason": payload.get("failure_reason"),
        "recovery_strategy": payload.get("recovery_strategy"),
        "turn_status": payload.get("turn_status"),
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
    tool_spec = getattr(step, "tool", None)
    if tool_spec is not None:
        # debug 中只暴露 contract 摘要，避免把 adapter 实例等不可序列化对象塞给前端。
        payload["tool_contract"] = {
            "contract_source": getattr(tool_spec, "contract_source", None),
            "adapter": getattr(tool_spec, "adapter", None),
            "backend_tool_name": getattr(tool_spec, "backend_tool_name", None),
            "side_effect_level": getattr(tool_spec, "side_effect_level", None),
            "requires_confirmation": getattr(tool_spec, "requires_confirmation", None),
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


def _active_runtime_tool_call(state: Optional[AgentState], *, approved_step_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """从显式 runtime_state 里推断 execute_step 即将调用的工具。

        需要用户批准的副作用 step 在批准前只应展示 interaction，不能提前显示成
    running tool；批准恢复后再用 approved_step_id 放行，避免进度 UI 误导用户。
    """
    if state is None:
        return None
    planner_trace = _planner_trace_summary_from_state(state)
    if state.runtime_state is None or state.execution_plan is None:

        if state.tool_name and not state.tool_calls:
            # 兼容旧节点流式状态：没有显式 runtime_state 时，只把当前轻量 tool_name/tool_args 当作进度展示，
            # 不把它写回 checkpoint，也不参与真实恢复判断。
            return {
                "tool_name": state.tool_name,
                "arguments": _compact_tool_args(state.tool_args or {}),
                "status": "running",
                "summary": f"正在执行 {state.tool_name}",
                "trace": {
                    "source": "legacy_stream_state",
                    **planner_trace,
                },
            }
        return None
    current_step_id = str(state.runtime_state.current_step_id or "").strip()
    if not current_step_id:
        return None
    for step in list(state.execution_plan.steps or []):
        if step.step_id == current_step_id:
            requires_confirmation = bool(
                step.confirmation_policy and step.confirmation_policy.requires_confirmation
            )
            if requires_confirmation and current_step_id != str(approved_step_id or "").strip():
                return None
            return {
                "tool_name": step.tool_name,
                "arguments": {},
                "step_id": step.step_id,
                "action_type": step.action_type,
                "status": "running",
                "summary": f"正在执行 {step.tool_name}",
                "trace": {
                    "step_id": step.step_id,
                    "action_type": step.action_type,
                    "plan_status": state.runtime_state.step_status.get(step.step_id),
                    **planner_trace,
                },
            }
    return None


def _latest_runtime_tool_call(state: Optional[AgentState]) -> Optional[Dict[str, Any]]:
    """从显式执行环的单步结果中整理 tool_call_end 事件。"""
    if state is None:
        return None
    if state.tool_calls:
        return state.tool_calls[-1].model_dump()
    last_step_result = dict((state.debug or {}).get("last_step_result") or {})
    if not last_step_result:
        return None
    active_tool = _active_runtime_tool_call(state) or {}
    return {
        "tool_name": active_tool.get("tool_name"),
        "step_id": last_step_result.get("step_id"),
        "status": last_step_result.get("step_status"),
        "summary": last_step_result.get("next_action"),
        "trace": {
            "output_key": last_step_result.get("output_key"),
            "has_observation": bool(last_step_result.get("observation")),
        },
        "error": last_step_result.get("error"),
    }


def _merge_tool_call_end(active_tool_call: Mapping[str, Any], latest_tool_call: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """合并工具开始与结束摘要，确保前端能把 running 卡片正确更新为终态。"""
    latest = dict(latest_tool_call or {})
    merged = {
        **dict(active_tool_call or {}),
        **latest,
    }
    merged["tool_name"] = merged.get("tool_name") or active_tool_call.get("tool_name") or "unknown"
    status = str(merged.get("status") or "").strip()
    if status in {"running", "pending", "waiting_confirmation"}:
        status = "success"
    merged["status"] = status or "success"
    merged["arguments"] = dict(merged.get("arguments") or active_tool_call.get("arguments") or {})
    merged["trace"] = {
        **dict(active_tool_call.get("trace") or {}),
        **dict(merged.get("trace") or {}),
    }
    return merged


def _tool_call_identity(tool_call: Optional[Mapping[str, Any]]) -> tuple[str, str, str]:
    """生成流式工具事件的轻量身份，用于去重已预告的 tool_call_start。

    这里不使用完整 arguments 比较，是因为批准恢复时的预告事件来自 checkpoint 快照，
    execute_step 结束事件可能来自工具 trace；两者字段完整度不同，但 step/tool/action 身份稳定。
    """
    payload = dict(tool_call or {})
    return (
        str(payload.get("step_id") or "").strip(),
        str(payload.get("tool_name") or "").strip(),
        str(payload.get("action_type") or "").strip(),
    )


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
