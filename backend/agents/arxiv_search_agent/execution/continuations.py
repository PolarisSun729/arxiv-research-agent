from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Mapping, Optional

from langgraph.types import Command

from services.storage.sqlite import StorageContainer
from services.storage.sqlite.stores.agent_work import (
    ACTIVE_CONTINUATION_STATUSES,
    AgentWorkConflict,
    AgentWorkStore,
)
from services.storage.sqlite.stores.paper_qa_index import PaperQAIndexStore

from .background_jobs import BackgroundJobHandlerRegistry, PersistentBackgroundWorkCoordinator


logger = logging.getLogger(__name__)

JOB_STAGE_LABELS = {
    "pending": "等待后台 worker",
    "starting": "启动索引构建",
    "validate_loading_method": "校验文档加载方式",
    "create_build_version": "创建索引构建版本",
    "mark_index_processing": "初始化索引构建状态",
    "load_paper_metadata": "读取论文元数据",
    "download_pdf": "下载论文 PDF",
    "load_pdf_document": "解析 PDF 文档",
    "chunk_document": "切分文档内容",
    "structure_table_chunks": "结构化表格片段",
    "save_chunk_file": "保存文档切片",
    "save_sparse_index": "保存稀疏关键词索引",
    "compress_chunks_for_rerank": "压缩重排文本",
    "build_retrieval_indexes": "构建检索索引",
    "save_retrieval_index": "保存检索索引产物",
    "create_retrieval_embeddings": "创建检索 embedding",
    "save_embeddings": "保存 embedding",
    "index_embeddings_to_vector_store": "写入向量数据库",
    "validate_new_collection": "校验新向量集合",
    "validate_sparse_index_artifact": "校验稀疏索引产物",
    "activate_index": "激活新索引版本",
    "worker_recovery": "后台 worker 中断，等待恢复",
}


class AgentWorkContinuationService:
    """组合 continuation 与物理 job 快照，并执行幂等 reconciliation。"""

    def __init__(
        self,
        *,
        store: AgentWorkStore,
        paper_job_store: PaperQAIndexStore,
        handlers: BackgroundJobHandlerRegistry,
    ) -> None:
        self.store = store
        self.paper_job_store = paper_job_store
        self.handlers = handlers

    def reconcile(self, continuation: Mapping[str, Any]) -> Dict[str, Any]:
        """按持久状态修复可证明的中间态；不猜测执行结果，也不触发同步工具 fallback。"""
        current = dict(continuation)
        if current.get("status") == "ready_to_resume":
            # ready 的独立 TTL 只限制恢复窗口；过期时必须同步终结业务 checkpoint。
            current = self.store.expire_ready_continuation_if_needed(str(current["continuation_id"]))
            if current.get("status") == "expired":
                return current
        handler = self.handlers.get(str(current.get("handler_name") or ""))
        if current.get("status") == "submitting":
            # 批准事务已提交但 job 尚未挂接时，依据 handler 幂等 preflight 修复，不重复消费授权。
            handler_state = dict(current.get("handler_state") or {})
            preflight = handler.preflight(handler_state)
            if preflight.status == "already_satisfied" and preflight.projected_result:
                current = self.store.mark_continuation_ready(
                    str(current["continuation_id"]),
                    validated_result=preflight.projected_result,
                )
            elif preflight.status in {"active_job", "missing"}:
                try:
                    submission = handler.submit_or_attach(handler_state, preflight)
                    current = self.store.attach_job(
                        str(current["continuation_id"]),
                        job_id=submission.job_id,
                        attached=submission.attached,
                    )
                except Exception as exc:
                    # submitting 可在后续轮询继续修复，查询接口不能把短暂提交故障伪装成业务失败。
                    logger.warning(
                        "Background continuation submission still pending: continuation_id=%s error=%s",
                        current.get("continuation_id"),
                        exc,
                    )
            elif preflight.status == "failed":
                raise AgentWorkConflict(preflight.error_code or "background_preflight_failed")

        if current.get("status") == "waiting_job" and current.get("job_id"):
            # 物理 job 是完成/失败真源；只有 handler 校验并投影成功后 continuation 才能进入 ready。
            job = self.paper_job_store.get_paper_index_job(str(current["job_id"]))
            if job:
                job_status = str(job.get("status") or "")
                if job_status == "success":
                    try:
                        projected_result = handler.project_result(job)
                        self.store.mark_job_ready(job_id=str(job["job_id"]), validated_result=projected_result)
                    except Exception as exc:
                        self.store.mark_job_failed(
                            job_id=str(job["job_id"]),
                            error_code="completion_validation_failed",
                            error_message=str(exc),
                        )
                elif job_status in {"failed", "cancelled", "stale"}:
                    self.store.mark_job_failed(
                        job_id=str(job["job_id"]),
                        error_code=str(job.get("failure_code") or "index_job_failed"),
                        error_message=str(job.get("error_message") or "索引构建失败"),
                    )
                current = self.store.get_continuation(str(current["continuation_id"])) or current
        return current

    def _safe_view(self, continuation: Mapping[str, Any]) -> Dict[str, Any]:
        job = None
        if continuation.get("job_id"):
            job = self.paper_job_store.get_paper_index_job(str(continuation["job_id"]))
        job_status = str((job or {}).get("status") or "") or None
        current_stage = str((job or {}).get("current_stage") or "") or None
        raw_progress = (job or {}).get("progress")
        progress = int(raw_progress) if raw_progress is not None else None
        status = str(continuation.get("status") or "")
        return {
            "continuation_id": continuation.get("continuation_id"),
            "session_id": continuation.get("session_id"),
            "status": status,
            "display_summary": dict(continuation.get("display_summary") or {}),
            "job_id": continuation.get("job_id"),
            "job": {
                "status": job_status,
                "current_stage": current_stage,
                "stage_label": JOB_STAGE_LABELS.get(current_stage or "", (job or {}).get("stage_message") or current_stage),
                # progress 缺失时保持 None，前端必须显示不定进度，禁止补 12% 等占位值。
                "progress": progress,
                "attempt_no": (job or {}).get("attempt_count"),
                "max_attempts": (job or {}).get("max_attempts"),
                "error_code": (job or {}).get("failure_code"),
                "error_message": (job or {}).get("error_message"),
            } if job is not None else None,
            "error_code": continuation.get("error_code"),
            "error_message": continuation.get("error_message"),
            "can_cancel": status in {"submitting", "waiting_job", "ready_to_resume"},
            "can_resume": status == "ready_to_resume",
            "ready_at": continuation.get("ready_at"),
            "expires_at": continuation.get("expires_at"),
            "resume_run_id": continuation.get("resume_run_id"),
        }

    def get(self, continuation_id: str, *, user_id: str, session_id: str) -> Dict[str, Any]:
        continuation = self.store.get_continuation(continuation_id)
        if continuation is None:
            raise AgentWorkConflict("continuation_missing")
        if (str(continuation.get("user_id")), str(continuation.get("session_id"))) != (user_id, session_id):
            raise AgentWorkConflict("continuation_owner_mismatch")
        return self._safe_view(self.reconcile(continuation))

    def list_active(self, *, user_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        # failed/indeterminate 都是不可恢复终态，只保留审计记录，不能继续混入“活动任务”误导新会话。
        statuses = list(ACTIVE_CONTINUATION_STATUSES)
        continuations = self.store.list_continuations(
            user_id=user_id,
            session_id=session_id,
            statuses=statuses,
            limit=100,
        )
        # resumed 通常是终态，但结果尚未被客户端读取时仍需短暂返回，支持恢复 SSE 断线后的同一答案。
        continuations.extend(
            self.store.list_unretrieved_resume_continuations(
                user_id=user_id,
                session_id=session_id,
                limit=max(1, 100 - len(continuations)),
            )
        )
        return [self._safe_view(self.reconcile(item)) for item in continuations]

    def cancel(self, continuation_id: str, *, user_id: str, session_id: str) -> Dict[str, Any]:
        continuation = self.store.cancel_continuation(
            continuation_id,
            user_id=user_id,
            session_id=session_id,
        )
        return self._safe_view(continuation)




def _invoke_result_has_interrupt(raw_result: Any) -> bool:
    """识别 graph.invoke 在仍挂起 interrupt 时返回的假完成载荷。"""
    if not isinstance(raw_result, Mapping):
        return False
    if raw_result.get("__interrupt__"):
        return True
    try:
        from langgraph.constants import INTERRUPT

        if raw_result.get(INTERRUPT):
            return True
    except Exception:
        pass
    return False


def _background_resume_failure_reason(
    *,
    raw_result: Any,
    final_state: Any,
    continuation: Mapping[str, Any],
) -> Optional[str]:
    """续跑成功判定：不得仍 interrupt，后台 step 必须完成，paper_qa 不得空答。"""
    if _invoke_result_has_interrupt(raw_result):
        return "background_resume_reinterrupted"

    step_id = str(continuation.get("step_id") or "").strip()
    runtime = getattr(final_state, "runtime_state", None)
    if runtime is None:
        runtime = getattr(final_state, "plan_runtime", None)

    step_status = ""
    turn_status = ""
    if runtime is not None:
        step_status_map = getattr(runtime, "step_status", None) or {}
        if isinstance(step_status_map, Mapping):
            step_status = str(step_status_map.get(step_id) or "").strip()
        turn_status = str(getattr(runtime, "turn_status", None) or "").strip()

    if turn_status in {"waiting_interaction", "waiting_background_job"}:
        return f"background_resume_still_waiting:{turn_status}"
    if step_id and step_status and step_status not in {"success", "skipped"}:
        return f"background_step_not_completed:{step_status or 'missing'}"

    intent = str(getattr(final_state, "intent", None) or "").strip()
    if intent == "paper_qa":
        answer = str(getattr(final_state, "answer", None) or "").strip()
        paper_qa_result = getattr(final_state, "paper_qa_result", None)
        if not answer and paper_qa_result is None:
            return "background_resume_empty_answer"
    return None


class AgentResumeRunManager:
    """在独立线程中消费精确 LangGraph checkpoint，SSE 只轮询持久 run 结果。"""

    def __init__(
        self,
        *,
        storage: StorageContainer,
        background_work_coordinator: PersistentBackgroundWorkCoordinator,
    ) -> None:
        self.storage = storage
        self.background_work_coordinator = background_work_coordinator
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}

    def claim_and_start(self, continuation_id: str, *, user_id: str, session_id: str) -> Dict[str, Any]:
        run = self.storage.agent_work.claim_resume_run(
            continuation_id,
            user_id=user_id,
            session_id=session_id,
        )
        if run.get("created") or run.get("status") == "pending":
            self.start(str(run["resume_run_id"]))
        return run

    def start(self, resume_run_id: str) -> None:
        with self._lock:
            existing = self._threads.get(resume_run_id)
            if existing is not None and existing.is_alive():
                return
            worker = threading.Thread(
                target=self._execute,
                args=(resume_run_id,),
                daemon=True,
                name=f"agent-resume-{resume_run_id}",
            )
            self._threads[resume_run_id] = worker
            worker.start()

    def recover_incomplete_runs(self) -> None:
        """重启后只重领 pending；已经 running 的 run 无法证明是否消费 checkpoint，必须 indeterminate。"""
        for run in self.storage.agent_work.list_resume_runs(statuses=["running"], limit=100):
            self.storage.agent_work.fail_resume_run(
                str(run["resume_run_id"]),
                status="indeterminate",
                error_code="resume_process_restarted",
                error_message="恢复进程重启，无法证明 LangGraph checkpoint 是否已经消费。",
            )
        for run in self.storage.agent_work.list_resume_runs(statuses=["pending"], limit=100):
            self.start(str(run["resume_run_id"]))

    def get(
        self,
        resume_run_id: str,
        *,
        user_id: str,
        session_id: Optional[str] = None,
        mark_retrieved: bool = False,
    ) -> Dict[str, Any]:
        run = self.storage.agent_work.get_resume_run(resume_run_id)
        if run is None:
            raise AgentWorkConflict("resume_run_missing")
        if str(run.get("user_id")) != user_id or (session_id and str(run.get("session_id")) != session_id):
            raise AgentWorkConflict("resume_run_owner_mismatch")
        if mark_retrieved and run.get("status") == "completed":
            # 普通 GET 已经决定返回完整结果时记录读取；SSE 轮询必须等 final_response yield 后再标记。
            self.storage.agent_work.mark_resume_result_retrieved(resume_run_id)
            run = self.storage.agent_work.get_resume_run(resume_run_id) or run
        return run

    def mark_result_retrieved(self, resume_run_id: str) -> bool:
        """由传输层在完整 final_response 已开始交付后记录结果读取审计。"""
        return self.storage.agent_work.mark_resume_result_retrieved(resume_run_id)

    def _execute(self, resume_run_id: str) -> None:
        if not self.storage.agent_work.start_resume_run(resume_run_id):
            return
        graph_invocation_started = False
        final_state_for_checkpoint = None
        failure_reason: Optional[str] = None
        try:
            run = self.storage.agent_work.get_resume_run(resume_run_id)
            continuation = self.storage.agent_work.get_continuation(str((run or {}).get("continuation_id") or ""))
            if run is None or continuation is None:
                raise AgentWorkConflict("resume_context_missing")
            validated_result = continuation.get("validated_result")
            if not isinstance(validated_result, Mapping):
                raise AgentWorkConflict("continuation_result_missing")

            # 延迟导入避免 service -> graph -> executor 与本模块形成初始化循环。
            from .. import service as agent_service
            from ..runtime_checkpoint import AgentRuntimeCheckpointManager, CHECKPOINT_STATUS_FAILED

            graph = agent_service._build_agent_graph(
                generation_service=agent_service._resolve_generation_service(),
                langgraph_checkpoint_store=self.storage.langgraph_checkpoints,
                runtime_checkpoint_store=self.storage.agent_runtime_checkpoints,
                approval_store=self.storage.approval_grants,
                background_work_coordinator=self.background_work_coordinator,
            )
            graph_config = agent_service._build_langgraph_config(str(continuation["thread_id"]))
            # 把 continuation_id 注入运行时 config，供 execute 在 interrupt 前短接投影。
            graph_config = dict(graph_config or {})
            configurable = dict(graph_config.get("configurable") or {})
            configurable["background_continuation_id"] = str(continuation["continuation_id"])
            graph_config["configurable"] = configurable

            get_state = getattr(graph, "get_state", None)
            graph_state = get_state(graph_config) if callable(get_state) else None
            if not agent_service._has_resume_checkpoint(graph_state):
                raise AgentWorkConflict("langgraph_checkpoint_missing")

            graph_invocation_started = True
            raw_result = graph.invoke(
                Command(
                    resume={
                        "decision": "background_completed",
                        "continuation_id": continuation["continuation_id"],
                        "validated_result": dict(validated_result),
                    }
                ),
                config=graph_config,
            )
            # 先检查原始返回值中的 interrupt，再 coerce；否则 __interrupt__ 会被状态模型丢弃。
            final_state = agent_service._coerce_state(raw_result)
            if not getattr(final_state, "user_id", None):
                final_state.user_id = str(continuation.get("user_id") or "") or None
            if not getattr(final_state, "session_id", None):
                final_state.session_id = str(continuation.get("session_id") or "") or None
            final_state_for_checkpoint = final_state

            failure_reason = _background_resume_failure_reason(
                raw_result=raw_result,
                final_state=final_state,
                continuation=continuation,
            )
            if failure_reason:
                raise AgentWorkConflict(failure_reason)

            response = agent_service._state_to_response(final_state)
            self.storage.agent_work.complete_resume_run(
                resume_run_id,
                final_response=response.model_dump(mode="json"),
            )
            # 成功后续跑必须收敛业务 checkpoint，避免 waiting_background_job 残影。
            try:
                checkpoint_manager = AgentRuntimeCheckpointManager(
                    runtime_checkpoint_store=self.storage.agent_runtime_checkpoints
                )
                agent_service._persist_runtime_checkpoint_after_turn(
                    checkpoint_manager,
                    final_state,
                    is_resume=True,
                )
            except Exception as checkpoint_exc:
                logger.warning(
                    "Failed to persist runtime checkpoint after successful resume: resume_run_id=%s error=%s",
                    resume_run_id,
                    checkpoint_exc,
                )
        except Exception as exc:

            # graph.invoke 前的失败可安全标记 failed；一旦调用开始，就无法证明 checkpoint 是否已消费，
            # 必须进入 indeterminate 并禁止自动重放原问题。
            # 必须进入 indeterminate 并禁止自动重放原问题。
            # 业务冲突（假完成/仍 interrupt）在 invoke 后也视为可诊断失败，error_code 保留具体原因。
            conflict_code = str(exc) if isinstance(exc, AgentWorkConflict) else ""
            if graph_invocation_started and not conflict_code:
                status = "indeterminate"
                error_code = "resume_execution_indeterminate"
            elif graph_invocation_started and conflict_code:
                status = "failed"
                error_code = conflict_code
            else:
                status = "failed"
                error_code = conflict_code or str(exc)
            logger.exception("Agent continuation resume failed: resume_run_id=%s", resume_run_id)
            self.storage.agent_work.fail_resume_run(
                resume_run_id,
                status=status,
                error_code=error_code[:120],
                error_message=str(exc),
            )
            # 失败时也尽量收敛业务 checkpoint，避免 session 一直停在 waiting_background_job。
            try:
                from .. import service as agent_service
                from ..runtime_checkpoint import AgentRuntimeCheckpointManager, CHECKPOINT_STATUS_FAILED

                checkpoint_manager = AgentRuntimeCheckpointManager(
                    runtime_checkpoint_store=self.storage.agent_runtime_checkpoints
                )
                if final_state_for_checkpoint is not None:
                    if getattr(final_state_for_checkpoint, "runtime_state", None) is not None:
                        final_state_for_checkpoint.runtime_state.failure_reason = error_code[:120]
                        final_state_for_checkpoint.runtime_state.turn_status = "failed"
                    if getattr(final_state_for_checkpoint, "plan_runtime", None) is not None:
                        final_state_for_checkpoint.plan_runtime.error = error_code[:120]
                        final_state_for_checkpoint.plan_runtime.turn_status = "failed"
                    checkpoint_manager.mark_terminal(
                        final_state_for_checkpoint,
                        status=CHECKPOINT_STATUS_FAILED,
                        error_summary=error_code[:120],
                    )
                else:
                    run_meta = self.storage.agent_work.get_resume_run(resume_run_id) or {}
                    agent_service._mark_runtime_checkpoint_failed(
                        checkpoint_manager,
                        user_id=str(run_meta.get("user_id") or ""),
                        session_id=str(run_meta.get("session_id") or ""),
                        detail=error_code[:120],
                    )
            except Exception as checkpoint_exc:
                logger.warning(
                    "Failed to persist runtime checkpoint after resume failure: resume_run_id=%s error=%s",
                    resume_run_id,
                    checkpoint_exc,
                )
        finally:
            with self._lock:
                self._threads.pop(resume_run_id, None)


__all__ = [
    "AgentResumeRunManager",
    "AgentWorkContinuationService",
    "JOB_STAGE_LABELS",
]
