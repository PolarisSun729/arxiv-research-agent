from __future__ import annotations

import asyncio
import base64
import logging
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from services.storage.database_service import DatabaseService
from utils.config import get_agent_runtime_checkpoint_config

try:  # pragma: no cover - 真实 LangGraph 环境才会走到这些类型
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
except Exception:  # pragma: no cover - 单测桩环境没有真实 langgraph
    BaseCheckpointSaver = object  # type: ignore[assignment]
    CheckpointTuple = None  # type: ignore[assignment]

try:  # pragma: no cover - 仅开发降级路径使用
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:  # pragma: no cover
    InMemorySaver = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CHECKPOINT_STATUS_RUNNING = "running"
CHECKPOINT_STATUS_WAITING = "waiting_confirmation"
CHECKPOINT_STATUS_COMPLETED = "completed"
CHECKPOINT_STATUS_CANCELLED = "cancelled"
CHECKPOINT_STATUS_FAILED = "failed"
CHECKPOINT_STATUS_EXPIRED = "expired"

TERMINAL_CHECKPOINT_STATUSES = {
    CHECKPOINT_STATUS_COMPLETED,
    CHECKPOINT_STATUS_CANCELLED,
    CHECKPOINT_STATUS_FAILED,
    CHECKPOINT_STATUS_EXPIRED,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _json_safe(value: Any) -> Any:
    """把业务 runtime 裁剪成 JSON 友好数据，避免持久化依赖模型实例或临时对象。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in dict(value or {}).items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in list(value or [])]
    return repr(value)


def _pack_pickle(value: Any) -> Dict[str, Any]:
    """保存 LangGraph 原始 checkpoint。

    原始 checkpoint 需要尽量无损恢复，不能像前端 debug 摘要那样随意裁剪；这里把 pickle 结果包进
    JSON 字段中，既能落 SQLite，也能避免业务 runtime 表承担 LangGraph 内部格式。
    """
    return {
        "encoding": "pickle-base64",
        "payload": base64.b64encode(pickle.dumps(value)).decode("ascii"),
    }


def _unpack_pickle(payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        return payload
    if payload.get("encoding") != "pickle-base64":
        return payload
    encoded = str(payload.get("payload") or "")
    if not encoded:
        return None
    return pickle.loads(base64.b64decode(encoded.encode("ascii")))


def _configurable(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict((config or {}).get("configurable") or {})


def _thread_id_from_config(config: Optional[Mapping[str, Any]]) -> str:
    return str(_configurable(config).get("thread_id") or "").strip()


def _checkpoint_ns_from_config(config: Optional[Mapping[str, Any]]) -> str:
    return str(_configurable(config).get("checkpoint_ns") or "").strip()


def _checkpoint_id_from_config(config: Optional[Mapping[str, Any]]) -> Optional[str]:
    checkpoint_id = _configurable(config).get("checkpoint_id")
    text = str(checkpoint_id or "").strip()
    return text or None


def _checkpoint_id_from_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    return str(checkpoint.get("id") or checkpoint.get("checkpoint_id") or "").strip()


def _with_checkpoint_id(config: Mapping[str, Any], checkpoint_ns: str, checkpoint_id: str) -> Dict[str, Any]:
    next_config = dict(config or {})
    configurable = dict(next_config.get("configurable") or {})
    configurable["checkpoint_ns"] = checkpoint_ns
    configurable["checkpoint_id"] = checkpoint_id
    next_config["configurable"] = configurable
    return next_config


def _build_checkpoint_tuple(row: Mapping[str, Any]) -> Any:
    checkpoint_ns = str(row.get("checkpoint_ns") or "")
    checkpoint_id = str(row.get("checkpoint_id") or "")
    thread_id = str(row.get("thread_id") or "")
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}}
    parent_id = str(row.get("parent_checkpoint_id") or "").strip()
    parent_config = (
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": parent_id}}
        if parent_id
        else None
    )
    checkpoint = _unpack_pickle(row.get("checkpoint"))
    metadata = _unpack_pickle(row.get("metadata")) or {}
    pending_writes = [
        (
            str(item.get("task_id") or ""),
            str(item.get("channel") or ""),
            _unpack_pickle(item.get("value")),
        )
        for item in list(row.get("pending_writes") or [])
        if isinstance(item, Mapping)
    ]
    if CheckpointTuple is None:
        return {
            "config": config,
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_config": parent_config,
            "pending_writes": pending_writes,
        }
    try:
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )
    except TypeError:
        # LangGraph 不同版本的 CheckpointTuple 字段略有差异；旧版没有 pending_writes 时，
        # 至少保证 checkpoint 主体可恢复，写入记录仍保存在数据库中供后续版本读取。
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )


class SqliteAgentCheckpointer(BaseCheckpointSaver):  # type: ignore[misc]
    """SQLite 版 LangGraph checkpointer。

    它只处理 LangGraph 原始 checkpoint 的保存/读取；Agent 业务级 ownership、pending confirmation、
    过期和终态语义由 AgentRuntimeCheckpointManager 单独维护，避免把两类状态混在一起。
    """

    def __init__(self, database_service: Optional[DatabaseService] = None) -> None:
        try:
            super().__init__()  # type: ignore[misc]
        except TypeError:
            pass
        self.database_service = database_service or DatabaseService()

    def get_tuple(self, config: Mapping[str, Any]) -> Any:
        thread_id = _thread_id_from_config(config)
        if not thread_id:
            return None
        row = self.database_service.get_langgraph_checkpoint(
            thread_id=thread_id,
            checkpoint_ns=_checkpoint_ns_from_config(config),
            checkpoint_id=_checkpoint_id_from_config(config),
        )
        return _build_checkpoint_tuple(row) if row else None

    def get(self, config: Mapping[str, Any]) -> Any:
        checkpoint_tuple = self.get_tuple(config)
        if checkpoint_tuple is None:
            return None
        if isinstance(checkpoint_tuple, Mapping):
            return checkpoint_tuple.get("checkpoint")
        return getattr(checkpoint_tuple, "checkpoint", None)

    def list(
        self,
        config: Optional[Mapping[str, Any]],
        *,
        filter: Optional[Mapping[str, Any]] = None,
        before: Optional[Mapping[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Any]:
        del filter, before
        thread_id = _thread_id_from_config(config)
        if not thread_id:
            return iter(())
        rows = self.database_service.list_langgraph_checkpoints(
            thread_id=thread_id,
            checkpoint_ns=_checkpoint_ns_from_config(config),
            limit=limit or 10,
        )
        return iter(_build_checkpoint_tuple(row) for row in rows)

    def put(
        self,
        config: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        new_versions: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        del new_versions
        thread_id = _thread_id_from_config(config)
        checkpoint_ns = _checkpoint_ns_from_config(config)
        checkpoint_id = _checkpoint_id_from_checkpoint(checkpoint)
        if not thread_id or not checkpoint_id:
            return dict(config or {})
        parent_id = _checkpoint_id_from_config(config)
        self.database_service.put_langgraph_checkpoint(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            checkpoint=_pack_pickle(dict(checkpoint or {})),
            metadata=_pack_pickle(dict(metadata or {})),
            parent_checkpoint_id=parent_id,
        )
        return _with_checkpoint_id(config, checkpoint_ns, checkpoint_id)

    def put_writes(
        self,
        config: Mapping[str, Any],
        writes: Iterable[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        del task_path
        thread_id = _thread_id_from_config(config)
        checkpoint_id = _checkpoint_id_from_config(config)
        if not thread_id or not checkpoint_id:
            return
        serialized_writes = [
            {"channel": channel, "value": _pack_pickle(value)}
            for channel, value in list(writes or [])
        ]
        self.database_service.put_langgraph_checkpoint_writes(
            thread_id=thread_id,
            checkpoint_ns=_checkpoint_ns_from_config(config),
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            writes=serialized_writes,
        )

    async def aget_tuple(self, config: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self.get_tuple, config)

    async def aget(self, config: Mapping[str, Any]) -> Any:
        return await asyncio.to_thread(self.get, config)

    async def alist(self, config: Optional[Mapping[str, Any]], **kwargs: Any) -> List[Any]:
        return await asyncio.to_thread(lambda: list(self.list(config, **kwargs)))

    async def aput(
        self,
        config: Mapping[str, Any],
        checkpoint: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
        new_versions: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: Mapping[str, Any],
        writes: Iterable[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)


class AgentRuntimeCheckpointManager:
    """维护 Agent 业务级 runtime checkpoint。

    这里负责 session/thread/user 校验、pending confirmation 真源、终态标记和过期清理；
    不读取 agent_sessions.pending_action，避免展示态反向驱动真实恢复。
    """

    def __init__(self, database_service: Optional[DatabaseService] = None) -> None:
        self.database_service = database_service or DatabaseService()
        self.config = get_agent_runtime_checkpoint_config()

    @property
    def ttl_seconds(self) -> int:
        return max(int(self.config.get("ttl_seconds") or 0), 60)

    @property
    def cleanup_retention_days(self) -> int:
        return max(int(self.config.get("cleanup_retention_days") or 0), 1)

    def expire_and_cleanup(self) -> None:
        self.database_service.expire_agent_runtime_checkpoints(now=_iso(_utcnow()))
        self.database_service.cleanup_agent_runtime_checkpoints(retention_days=self.cleanup_retention_days)

    def persist_state(self, state: Any, *, current_node: Optional[str] = None, next_route: Optional[str] = None) -> None:
        payload = _state_payload(state)
        user_id = str(payload.get("user_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return
        runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else None
        plan_runtime = payload.get("plan_runtime") if isinstance(payload.get("plan_runtime"), Mapping) else None
        pending_confirmation = _extract_pending_confirmation(payload)
        status = _status_from_state(payload, pending_confirmation=pending_confirmation)
        expires_at = _iso(_utcnow() + timedelta(seconds=self.ttl_seconds)) if status == CHECKPOINT_STATUS_WAITING else None
        error_summary = _extract_error_summary(payload)
        self.database_service.upsert_agent_runtime_checkpoint(
            user_id=user_id,
            session_id=session_id,
            thread_id=session_id,
            runtime_state=runtime_state or plan_runtime,
            graph_state={
                "session_id": session_id,
                "intent": payload.get("intent"),
                "steps": payload.get("steps") or [],
            },
            pending_confirmation=pending_confirmation,
            current_node=current_node or _extract_current_node(payload),
            next_route=next_route or _extract_next_route(payload),
            status=status,
            error_summary=error_summary,
            expires_at=expires_at,
        )

    def validate_resume(
        self,
        *,
        user_id: Optional[str],
        session_id: str,
        thread_id: str,
        resume_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self.database_service.expire_agent_runtime_checkpoints(now=_iso(_utcnow()))
        checkpoint = self.database_service.get_agent_runtime_checkpoint(
            user_id=str(user_id or "").strip(),
            session_id=session_id,
            thread_id=thread_id,
        )
        if not checkpoint:
            raise AgentRuntimeCheckpointError(thread_id=thread_id, reason="checkpoint_missing")
        status = str(checkpoint.get("status") or "").strip()
        if status != CHECKPOINT_STATUS_WAITING:
            raise AgentRuntimeCheckpointError(thread_id=thread_id, reason=f"checkpoint_not_waiting:{status or 'unknown'}")
        pending_confirmation = checkpoint.get("pending_confirmation")
        if not isinstance(pending_confirmation, Mapping) or not pending_confirmation:
            raise AgentRuntimeCheckpointError(thread_id=thread_id, reason="pending_confirmation_missing")
        expected_step_id = str(pending_confirmation.get("step_id") or "").strip()
        requested_step_id = str(resume_payload.get("step_id") or "").strip()
        if requested_step_id and expected_step_id and requested_step_id != expected_step_id:
            raise AgentRuntimeCheckpointError(thread_id=thread_id, reason="step_id_mismatch")
        return checkpoint

    def mark_terminal(self, state: Any, *, status: str, error_summary: Optional[str] = None) -> None:
        payload = _state_payload(state)
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return
        updated = self.database_service.mark_agent_runtime_checkpoint_status(
            user_id=str(payload.get("user_id") or "").strip(),
            session_id=session_id,
            thread_id=session_id,
            status=status,
            error_summary=error_summary or _extract_error_summary(payload),
            clear_pending_confirmation=True,
        )
        if updated:
            return
        # 普通非确认请求可能此前没有等待现场；这里补一条终态 checkpoint，保证 completed/failed/cancelled
        # 都有可查询记录，同时不引入 pending_confirmation。
        self.database_service.upsert_agent_runtime_checkpoint(
            user_id=str(payload.get("user_id") or "").strip(),
            session_id=session_id,
            thread_id=session_id,
            runtime_state=payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else None,
            graph_state={"session_id": session_id, "intent": payload.get("intent"), "steps": payload.get("steps") or []},
            pending_confirmation=None,
            current_node=_extract_current_node(payload),
            next_route=_extract_next_route(payload),
            status=status,
            error_summary=error_summary or _extract_error_summary(payload),
            expires_at=None,
        )


class AgentRuntimeCheckpointError(RuntimeError):
    """业务级 runtime checkpoint 校验失败。"""

    def __init__(self, *, thread_id: str, reason: str) -> None:
        self.thread_id = thread_id
        self.reason = reason
        super().__init__(f"Agent runtime checkpoint 不可恢复，thread_id={thread_id}，reason={reason}")


def build_agent_checkpointer(database_service: Optional[DatabaseService] = None) -> Any:
    config = get_agent_runtime_checkpoint_config()
    backend = str(config.get("backend") or "sqlite").strip().lower()
    if backend == "memory":
        if InMemorySaver is None:
            raise RuntimeError("AGENT_RUNTIME_CHECKPOINT_BACKEND=memory requires langgraph InMemorySaver")
        logger.warning("arxiv_agent 使用内存 checkpointer，仅适合开发环境；生产环境应使用 sqlite。")
        return InMemorySaver()
    return SqliteAgentCheckpointer(database_service=database_service)


def _state_payload(state: Any) -> Dict[str, Any]:
    if state is None:
        return {}
    model_dump = getattr(state, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
    if isinstance(state, Mapping):
        return dict(state)
    return {}


def _extract_pending_confirmation(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else {}
    plan_runtime = payload.get("plan_runtime") if isinstance(payload.get("plan_runtime"), Mapping) else {}
    debug = payload.get("debug") if isinstance(payload.get("debug"), Mapping) else {}
    pending = (
        runtime_state.get("pending_confirmation")
        or plan_runtime.get("pending_confirmation")
        or debug.get("pending_confirmation")
    )
    return _json_safe(pending) if isinstance(pending, Mapping) else None


def _status_from_state(payload: Mapping[str, Any], *, pending_confirmation: Optional[Mapping[str, Any]]) -> str:
    if pending_confirmation:
        return CHECKPOINT_STATUS_WAITING
    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else {}
    plan_runtime = payload.get("plan_runtime") if isinstance(payload.get("plan_runtime"), Mapping) else {}
    turn_status = str(runtime_state.get("turn_status") or plan_runtime.get("turn_status") or "").strip()
    error = _extract_error_summary(payload)
    pending_action = payload.get("pending_action") if isinstance(payload.get("pending_action"), Mapping) else {}
    if str(pending_action.get("status") or "").strip() == "cancelled":
        return CHECKPOINT_STATUS_CANCELLED
    if turn_status == "failed" or error:
        return CHECKPOINT_STATUS_FAILED
    if turn_status:
        return CHECKPOINT_STATUS_COMPLETED
    return CHECKPOINT_STATUS_RUNNING


def _extract_error_summary(payload: Mapping[str, Any]) -> str:
    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else {}
    plan_runtime = payload.get("plan_runtime") if isinstance(payload.get("plan_runtime"), Mapping) else {}
    errors = payload.get("errors") if isinstance(payload.get("errors"), list) else []
    if runtime_state.get("failure_reason"):
        return str(runtime_state.get("failure_reason"))
    if plan_runtime.get("error"):
        return str(plan_runtime.get("error"))
    if errors:
        return str((errors[-1] or {}).get("code") or (errors[-1] or {}).get("message") or "")
    return ""


def _extract_current_node(payload: Mapping[str, Any]) -> str:
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    if not steps:
        return ""
    latest = steps[-1] if isinstance(steps[-1], Mapping) else {}
    return str(latest.get("step") or "").strip()


def _extract_next_route(payload: Mapping[str, Any]) -> str:
    debug = payload.get("debug") if isinstance(payload.get("debug"), Mapping) else {}
    agent_route = debug.get("agent_route") if isinstance(debug.get("agent_route"), Mapping) else {}
    return str(agent_route.get("decision") or agent_route.get("phase") or "").strip()


__all__ = [
    "AgentRuntimeCheckpointError",
    "AgentRuntimeCheckpointManager",
    "SqliteAgentCheckpointer",
    "build_agent_checkpointer",
    "CHECKPOINT_STATUS_CANCELLED",
    "CHECKPOINT_STATUS_COMPLETED",
    "CHECKPOINT_STATUS_EXPIRED",
    "CHECKPOINT_STATUS_FAILED",
    "CHECKPOINT_STATUS_RUNNING",
    "CHECKPOINT_STATUS_WAITING",
]
