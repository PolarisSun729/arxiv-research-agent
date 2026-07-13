from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol
from uuid import uuid4

from services.storage.sqlite.stores.agent_work import AgentWorkStore
from services.storage.sqlite.stores.approval_grants import ApprovalGrantStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackgroundJobPreflight:
    """后台 handler 在消费授权前返回的业务事实。"""

    status: str
    arxiv_id: str
    loading_method: str
    job_id: Optional[str] = None
    job: Dict[str, Any] = field(default_factory=dict)
    projected_result: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class BackgroundJobSubmission:
    """表示 continuation 最终绑定到哪个物理 job，以及该 job 是否为复用。"""

    job_id: str
    job: Dict[str, Any]
    attached: bool


@dataclass(frozen=True)
class BackgroundWorkTicket:
    """协调器交还给执行器的安全引用；内部 checkpoint/grant 身份不会进入前端。"""

    status: str
    continuation_id: Optional[str] = None
    job_id: Optional[str] = None
    projected_result: Optional[Dict[str, Any]] = None


class BackgroundWorkCoordinator(Protocol):
    """执行器使用的最小协调接口；事务、handler 与持久化细节全部留在实现层。"""

    def prepare(self, **kwargs: Any) -> BackgroundWorkTicket:
        ...


class BackgroundJobHandler(Protocol):
    """通用后台执行协议；协调器只依赖这三个能力，不依赖 Paper QA 业务细节。"""

    def preflight(self, arguments: Mapping[str, Any]) -> BackgroundJobPreflight:
        ...

    def submit_or_attach(
        self,
        arguments: Mapping[str, Any],
        preflight: BackgroundJobPreflight,
    ) -> BackgroundJobSubmission:
        ...

    def project_result(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        ...


class BackgroundJobHandlerRegistry:
    """按 ToolContract 中的 handler 名称解析后台业务实现。"""

    def __init__(self) -> None:
        self._handlers: Dict[str, BackgroundJobHandler] = {}

    def register(self, name: str, handler: BackgroundJobHandler) -> None:
        normalized = str(name or "").strip()
        if not normalized:
            raise ValueError("background handler name is required")
        self._handlers[normalized] = handler

    def get(self, name: str) -> BackgroundJobHandler:
        normalized = str(name or "").strip()
        handler = self._handlers.get(normalized)
        if handler is None:
            raise KeyError(f"background_job_handler_not_found:{normalized}")
        return handler


class PaperQAIndexBackgroundHandler:
    """把论文索引状态与 durable job 转换成通用后台执行协议。"""

    def __init__(
        self,
        *,
        status_reader: Callable[[str], Mapping[str, Any]],
        active_job_reader: Callable[[str], Optional[Mapping[str, Any]]],
        job_submitter: Callable[[str, str], Mapping[str, Any]],
    ) -> None:
        self.status_reader = status_reader
        self.active_job_reader = active_job_reader
        self.job_submitter = job_submitter

    @staticmethod
    def _arguments(arguments: Mapping[str, Any]) -> tuple[str, str]:
        paper_reference = arguments.get("paper_reference")
        if not isinstance(paper_reference, Mapping):
            paper_reference = arguments.get("paper_ref") if isinstance(arguments.get("paper_ref"), Mapping) else {}
        arxiv_id = str(arguments.get("arxiv_id") or paper_reference.get("arxiv_id") or "").strip()
        loading_method = str(arguments.get("loading_method") or "docling").strip().lower() or "docling"
        return arxiv_id, loading_method

    def preflight(self, arguments: Mapping[str, Any]) -> BackgroundJobPreflight:
        arxiv_id, loading_method = self._arguments(arguments)
        if not arxiv_id:
            return BackgroundJobPreflight(
                status="failed",
                arxiv_id="",
                loading_method=loading_method,
                error_code="missing_arxiv_id",
                error_message="后台索引任务缺少最终论文 arXiv ID。",
            )

        status = dict(self.status_reader(arxiv_id) or {})
        if bool(status.get("has_index")) and str(status.get("status") or "") == "indexed":
            # preflight 已证明副作用不再需要；直接投影等价成功输出，不能创建空 grant/invocation/job。
            projected = {
                "status": "indexed",
                "has_index": True,
                "skipped_rebuild": True,
                "skip_reason": "paper_qa_index_already_available",
                "arxiv_id": arxiv_id,
                "build_id": status.get("active_build_id"),
                "index_version": status.get("active_index_version"),
                "collection_name": status.get("active_collection_name") or status.get("collection_name"),
                "chunk_count": status.get("active_chunk_count") or status.get("chunk_count") or 0,
            }
            return BackgroundJobPreflight(
                status="already_satisfied",
                arxiv_id=arxiv_id,
                loading_method=loading_method,
                projected_result=projected,
            )

        active_job = self.active_job_reader(arxiv_id)
        if active_job:
            job = dict(active_job)
            return BackgroundJobPreflight(
                status="active_job",
                arxiv_id=arxiv_id,
                loading_method=loading_method,
                job_id=str(job.get("job_id") or ""),
                job=job,
            )
        return BackgroundJobPreflight(status="missing", arxiv_id=arxiv_id, loading_method=loading_method)

    def submit_or_attach(
        self,
        arguments: Mapping[str, Any],
        preflight: BackgroundJobPreflight,
    ) -> BackgroundJobSubmission:
        del arguments
        if preflight.status == "active_job" and preflight.job_id:
            return BackgroundJobSubmission(job_id=preflight.job_id, job=dict(preflight.job), attached=True)
        if preflight.status != "missing":
            raise RuntimeError(f"background_submit_not_allowed:{preflight.status}")
        job = dict(self.job_submitter(preflight.arxiv_id, preflight.loading_method) or {})
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            raise RuntimeError("background_job_submission_missing_job_id")
        return BackgroundJobSubmission(job_id=job_id, job=job, attached=not bool(job.get("created", True)))

    def project_result(self, job: Mapping[str, Any]) -> Dict[str, Any]:
        result = job.get("result") if isinstance(job.get("result"), Mapping) else {}
        payload = dict(result or {})
        if str(job.get("status") or "") != "success" or not payload:
            raise RuntimeError("background_job_result_not_ready")
        return {
            **payload,
            "status": "indexed",
            "has_index": True,
            "job_id": job.get("job_id"),
        }


class PersistentBackgroundWorkCoordinator:
    """把交互批准、通用 handler 与持久 continuation 串成短事务执行链。"""

    def __init__(
        self,
        *,
        store: AgentWorkStore,
        approval_store: ApprovalGrantStore,
        handlers: BackgroundJobHandlerRegistry,
    ) -> None:
        self.store = store
        self.approval_store = approval_store
        self.handlers = handlers

    @staticmethod
    def _payload(value: Any) -> Dict[str, Any]:
        model_dump = getattr(value, "model_dump", None)
        payload = model_dump(mode="json") if callable(model_dump) else value
        return dict(payload or {}) if isinstance(payload, Mapping) else {}

    @staticmethod
    def _job_key(preflight: BackgroundJobPreflight) -> str:
        existing = str(preflight.job.get("idempotency_key") or "").strip()
        return existing or f"{preflight.arxiv_id}:{preflight.loading_method}:paper_qa_index_v1"

    def approve_interaction(
        self,
        *,
        checkpoint: Mapping[str, Any],
        interaction: Any,
        approval_payload: Any,
        user_id: str,
        session_id: str,
        thread_id: str,
        handler_name: str,
    ) -> BackgroundWorkTicket:
        """在 interaction 仍待处理时建立 continuation，成功后请求无需恢复原图。"""
        handler = self.handlers.get(handler_name)
        interaction_data = self._payload(interaction)
        payload = self._payload(approval_payload)
        arguments = dict(payload.get("arguments_summary") or {})
        preflight = handler.preflight(arguments)
        if preflight.status == "failed":
            raise RuntimeError(preflight.error_code or "background_preflight_failed")
        if preflight.status == "already_satisfied":
            # 无副作用竞态仍交回普通 grant + 精确图恢复，让 observe_step 消费等价成功输出。
            return BackgroundWorkTicket(status="already_satisfied", projected_result=preflight.projected_result)

        grant_id = str(uuid4())
        invocation_id = str(uuid4())
        continuation_id = str(uuid4())
        paper_reference = arguments.get("paper_reference")
        paper_reference = dict(paper_reference) if isinstance(paper_reference, Mapping) else {}
        continuation = self.store.prepare_background_work(
            checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            interaction_id=str(interaction_data.get("interaction_id") or ""),
            grant_id=grant_id,
            invocation_id=invocation_id,
            continuation_id=continuation_id,
            plan_id=str(interaction_data.get("plan_id") or ""),
            step_id=str(interaction_data.get("step_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            arguments_fingerprint=str(payload.get("arguments_fingerprint") or ""),
            handler_name=handler_name,
            job_id=preflight.job_id,
            job_idempotency_key=self._job_key(preflight),
            handler_state={
                "paper_reference": paper_reference,
                "arxiv_id": preflight.arxiv_id,
                "loading_method": preflight.loading_method,
            },
            display_summary={
                # 失败任务只能用安全字段重新发起授权；不能把完整工具参数或内部 handler state 暴露给前端。
                "arxiv_id": preflight.arxiv_id,
                "paper_title": paper_reference.get("title") or paper_reference.get("paper_title") or preflight.arxiv_id,
                "question_summary": str(arguments.get("message") or arguments.get("question") or "")[:160],
            },
        )

        if preflight.status == "active_job" and preflight.job_id:
            return BackgroundWorkTicket(
                status="waiting_job",
                continuation_id=continuation_id,
                job_id=preflight.job_id,
            )

        try:
            submission = handler.submit_or_attach(arguments, preflight)
            continuation = self.store.attach_job(
                continuation_id,
                job_id=submission.job_id,
                attached=submission.attached,
            )
        except Exception as exc:
            # submitting 是可修复中间态；失败发生在授权事务提交后，禁止同步执行或伪造终态。
            logger.exception(
                "Background job submission deferred to reconciliation: continuation_id=%s error=%s",
                continuation_id,
                exc,
            )
            return BackgroundWorkTicket(status="submitting", continuation_id=continuation_id)
        return BackgroundWorkTicket(
            status=str(continuation.get("status") or "waiting_job"),
            continuation_id=continuation_id,
            job_id=submission.job_id,
        )

    def prepare(self, **kwargs: Any) -> BackgroundWorkTicket:
        """兼容执行器重入；只有无副作用竞态可在此直接投影，真实提交必须走批准短事务。"""
        handler = self.handlers.get(str(kwargs.get("handler_name") or ""))
        arguments = dict(kwargs.get("arguments") or {})
        preflight = handler.preflight(arguments)
        if preflight.status != "already_satisfied" or preflight.projected_result is None:
            raise RuntimeError("background_work_requires_atomic_interaction_approval")
        grant_id = str(kwargs.get("grant_id") or "").strip()
        step = kwargs.get("step")
        runtime = kwargs.get("runtime")
        if grant_id:
            invocation_id = str(uuid4())
            self.approval_store.consume_and_prepare_invocation(
                grant_id=grant_id,
                invocation_id=invocation_id,
                plan_id=str(getattr(getattr(runtime, "plan", None), "plan_id", "") or ""),
                step_id=str(getattr(step, "step_id", "") or ""),
                tool_name=str(getattr(step, "tool_name", "") or ""),
                arguments_fingerprint=str(kwargs.get("arguments_fingerprint") or ""),
            )
            self.approval_store.mark_invocation(
                invocation_id,
                status="succeeded",
                result_summary=preflight.projected_result,
            )
        return BackgroundWorkTicket(status="already_satisfied", projected_result=preflight.projected_result)


__all__ = [
    "BackgroundJobHandler",
    "BackgroundJobHandlerRegistry",
    "BackgroundJobPreflight",
    "BackgroundJobSubmission",
    "BackgroundWorkCoordinator",
    "BackgroundWorkTicket",
    "PaperQAIndexBackgroundHandler",
    "PersistentBackgroundWorkCoordinator",
]
