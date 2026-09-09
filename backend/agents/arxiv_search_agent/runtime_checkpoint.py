
from __future__ import annotations

import asyncio
import base64
import logging
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from services.storage.sqlite.stores import AgentRuntimeCheckpointStore, LangGraphCheckpointStore
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
CHECKPOINT_STATUS_WAITING_INTERACTION = "waiting_interaction"
CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB = "waiting_background_job"
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

    def __init__(self, langgraph_checkpoint_store: LangGraphCheckpointStore) -> None:
        try:
            super().__init__()  # type: ignore[misc]
        except TypeError:
            pass
        self.langgraph_checkpoint_store = langgraph_checkpoint_store

    def get_tuple(self, config: Mapping[str, Any]) -> Any:
        thread_id = _thread_id_from_config(config)
        if not thread_id:
            return None
        row = self.langgraph_checkpoint_store.get_langgraph_checkpoint(
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
        rows = self.langgraph_checkpoint_store.list_langgraph_checkpoints(
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
        self.langgraph_checkpoint_store.put_langgraph_checkpoint(
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
        self.langgraph_checkpoint_store.put_langgraph_checkpoint_writes(
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
    不读取会话展示态或 debug，避免排查快照反向驱动真实恢复。
    """

    def __init__(self, runtime_checkpoint_store: AgentRuntimeCheckpointStore) -> None:
        self.runtime_checkpoint_store = runtime_checkpoint_store
        self.config = get_agent_runtime_checkpoint_config()

    @property
    def ttl_seconds(self) -> int:
        return max(int(self.config.get("ttl_seconds") or 0), 60)

    @property
    def cleanup_retention_days(self) -> int:
        return max(int(self.config.get("cleanup_retention_days") or 0), 1)

    def expire_and_cleanup(self) -> None:
        self.runtime_checkpoint_store.expire_agent_runtime_checkpoints(now=_iso(_utcnow()))
        # LangGraph 原始 checkpoint 跟随业务 runtime checkpoint 生命周期清理；
        # 先删 graph 记录，再删 runtime 记录，避免丢失 thread_id 对齐依据。
        self.runtime_checkpoint_store.cleanup_langgraph_checkpoints_for_terminal_runtime(retention_days=self.cleanup_retention_days)
        self.runtime_checkpoint_store.cleanup_agent_runtime_checkpoints(retention_days=self.cleanup_retention_days)

    def persist_state(self, state: Any, *, current_node: Optional[str] = None, next_route: Optional[str] = None) -> None:
        payload = _state_payload(state)
        user_id = str(payload.get("user_id") or "").strip()
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return
        runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else None
        plan_runtime = payload.get("plan_runtime") if isinstance(payload.get("plan_runtime"), Mapping) else None
        interaction = _extract_interaction(payload)
        status = _status_from_state(payload, interaction=interaction)
        expires_at = _iso(_utcnow() + timedelta(seconds=self.ttl_seconds)) if status == CHECKPOINT_STATUS_WAITING_INTERACTION else None
        error_summary = _extract_error_summary(payload)
        self.runtime_checkpoint_store.upsert_agent_runtime_checkpoint(
            user_id=user_id,
            session_id=session_id,
            thread_id=session_id,
            runtime_state=runtime_state or plan_runtime,
            graph_state={
                "session_id": session_id,
                "intent": payload.get("intent"),
                "steps": payload.get("steps") or [],
            },
            interaction=interaction,
            schema_version=2,
            # 节点坐标由图调用方显式提供；缺失时保留未知，不能从展示 debug 伪造恢复位置。
            current_node=current_node,
            next_route=next_route or _next_route_from_status(status),
            status=status,
            error_summary=error_summary,
            expires_at=expires_at,
        )

    def mark_terminal(self, state: Any, *, status: str, error_summary: Optional[str] = None) -> None:
        payload = _state_payload(state)
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            return
        updated = self.runtime_checkpoint_store.mark_agent_runtime_checkpoint_status(
            user_id=str(payload.get("user_id") or "").strip(),
            session_id=session_id,
            thread_id=session_id,
            status=status,
            error_summary=error_summary or _extract_error_summary(payload),
            clear_interaction=True,
        )
        if updated:
            return
        # 普通请求可能此前没有等待现场；这里补一条终态 checkpoint，保证生命周期可查询。
        self.runtime_checkpoint_store.upsert_agent_runtime_checkpoint(
            user_id=str(payload.get("user_id") or "").strip(),
            session_id=session_id,
            thread_id=session_id,
            runtime_state=payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else None,
            graph_state={"session_id": session_id, "intent": payload.get("intent"), "steps": payload.get("steps") or []},
            current_node=None,
            next_route=_next_route_from_status(status),
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


def build_agent_checkpointer(*, langgraph_checkpoint_store: Optional[LangGraphCheckpointStore] = None) -> Any:
    config = get_agent_runtime_checkpoint_config()
    backend = str(config.get("backend") or "sqlite").strip().lower()
    if backend == "memory":
        if InMemorySaver is None:
            raise RuntimeError("AGENT_RUNTIME_CHECKPOINT_BACKEND=memory requires langgraph InMemorySaver")
        logger.warning("arxiv_agent 使用内存 checkpointer，仅适合开发环境；生产环境应使用 sqlite。")
        return InMemorySaver()
    if langgraph_checkpoint_store is None:
        raise RuntimeError("sqlite agent checkpointer requires langgraph_checkpoint_store")
    return SqliteAgentCheckpointer(langgraph_checkpoint_store=langgraph_checkpoint_store)


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


def _extract_interaction(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """读取唯一 interaction 真源，不从 debug、context 或展示字段恢复业务状态。"""

    runtime_state = payload.get("runtime_state") if isinstance(payload.get("runtime_state"), Mapping) else {}
    interaction = runtime_state.get("interaction")
    if isinstance(interaction, Mapping):
        return _json_safe(interaction)
    plan_runtime = payload.get("plan_runtime") if isinstance(payload.get("plan_runtime"), Mapping) else {}
    interaction = plan_runtime.get("interaction")
    return _json_safe(interaction) if isinstance(interaction, Mapping) else None


def _extract_error_summary(payload: Mapping[str, Any]) -> Optional[str]:
    """只从正式运行态提取终态错误，禁止 debug 或展示字段影响 checkpoint 生命周期。"""
    runtime_state, plan_runtime = _business_runtime_payloads(payload)
    for value in (runtime_state.get("error"), plan_runtime.get("error"), payload.get("error")):
        text = str(value or "").strip()
        if text:
            return text
    return None


def _status_from_state(
    payload: Mapping[str, Any],
    *,
    interaction: Optional[Mapping[str, Any]] = None,
) -> str:
    if interaction:
        return CHECKPOINT_STATUS_WAITING_INTERACTION
    runtime_state, plan_runtime = _business_runtime_payloads(payload)
    turn_status = str(runtime_state.get("turn_status") or plan_runtime.get("turn_status") or "").strip()
    error = _extract_error_summary(payload)
    recovery_strategy = runtime_state.get("recovery_strategy") or plan_runtime.get("recovery_strategy")
    if _is_confirmation_rejected_recovery(recovery_strategy):
        return CHECKPOINT_STATUS_CANCELLED
    if turn_status == "failed" or error:
        return CHECKPOINT_STATUS_FAILED
    if turn_status == CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB:
        # 后台等待由 continuation 生命周期管理，不能套用普通 interaction TTL 或误判成 completed。
        return CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB
    if turn_status:
        return CHECKPOINT_STATUS_COMPLETED
    return CHECKPOINT_STATUS_RUNNING


def _business_runtime_payloads(payload: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """返回参与 checkpoint 判定的业务运行态。

    runtime_state 是跨节点/跨请求恢复的序列化快照；存在时不再回退其他镜像。
    """
    if isinstance(payload.get("runtime_state"), Mapping):
        return dict(payload.get("runtime_state") or {}), {}
    plan_runtime = payload.get("plan_runtime") if isinstance(payload.get("plan_runtime"), Mapping) else {}
    return {}, dict(plan_runtime or {})


def _is_confirmation_rejected_recovery(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return str(value.get("type") or "").strip() == "skip_step" and str(value.get("reason") or "").strip() == "confirmation_rejected"


def _next_route_from_status(status: str) -> str:
    """从 checkpoint 业务状态生成可观测 route，避免读取 debug.agent_route 的旧决策。"""
    normalized = str(status or "").strip()
    if normalized == CHECKPOINT_STATUS_WAITING_INTERACTION:
        return CHECKPOINT_STATUS_WAITING_INTERACTION
    if normalized == CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB:
        return CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB
    if normalized in {CHECKPOINT_STATUS_COMPLETED, CHECKPOINT_STATUS_FAILED, CHECKPOINT_STATUS_CANCELLED, CHECKPOINT_STATUS_EXPIRED}:
        return normalized
    return CHECKPOINT_STATUS_RUNNING


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
    "CHECKPOINT_STATUS_WAITING_INTERACTION",
    "CHECKPOINT_STATUS_WAITING_BACKGROUND_JOB",
]
