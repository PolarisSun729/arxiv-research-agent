from __future__ import annotations

import logging
import threading
import uuid
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional

from fastapi import HTTPException

from core.errors import AppError, ErrorCode
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.storage.sqlite.stores.paper_qa_index import PaperQAIndexStore
from utils.config import get_qa_index_job_runtime_config

logger = logging.getLogger(__name__)


class IndexJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STALE = "stale"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


ACTIVE_JOB_STATUSES = {IndexJobStatus.PENDING.value, IndexJobStatus.RUNNING.value, IndexJobStatus.RETRYING.value}
RETRYABLE_JOB_STATUSES = {IndexJobStatus.FAILED.value, IndexJobStatus.STALE.value, IndexJobStatus.CANCELLED.value}


class IndexJobManager:
    def __init__(
        self,
        *,
        paper_qa_index_store: PaperQAIndexStore,
        qa_index_builder: PaperQAIndexBuilder,
        qa_status_reader: Optional[Callable[[str], Mapping[str, Any]]] = None,
        timeout_seconds: Optional[int] = None,
        lease_seconds: Optional[int] = None,
        heartbeat_interval_seconds: Optional[float] = None,
        poll_interval_seconds: Optional[float] = None,
        max_attempts: Optional[int] = None,
        recipe_version: Optional[str] = None,
        worker_id: Optional[str] = None,
        on_job_succeeded: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        on_job_failed: Optional[Callable[[str, str, str], None]] = None,
    ):
        """初始化可重启的数据库 lease worker；请求线程只负责持久化提交事实。"""
        self.paper_qa_index_store = paper_qa_index_store
        self.qa_index_builder = qa_index_builder
        self.qa_status_reader = qa_status_reader
        config = get_qa_index_job_runtime_config()
        self.timeout_seconds = max(1, int(timeout_seconds or config.get("timeout_seconds") or 1800))
        self.lease_seconds = max(5, int(lease_seconds or config.get("lease_seconds") or 90))
        self.heartbeat_interval_seconds = max(
            0.2,
            float(heartbeat_interval_seconds or config.get("heartbeat_interval_seconds") or 15),
        )
        self.poll_interval_seconds = max(0.1, float(poll_interval_seconds or config.get("poll_interval_seconds") or 1))
        self.max_attempts = max(1, int(max_attempts or config.get("max_attempts") or 3))
        self.recipe_version = str(recipe_version or config.get("recipe_version") or "paper_qa_index_v1").strip()
        self.worker_id = str(worker_id or f"qa-index-worker-{uuid.uuid4()}")
        self.on_job_succeeded = on_job_succeeded
        self.on_job_failed = on_job_failed
        self._submission_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    @staticmethod
    def _exception_message(exc: Exception) -> str:
        """从异常对象中提取适合写入任务状态表的短错误信息。"""
        detail = getattr(exc, "detail", None)
        if detail is not None:
            text = str(detail).strip()
            if text:
                return text[:500]
        text = str(exc).strip()
        return (text or exc.__class__.__name__)[:500]

    def submit_job(self, arxiv_id: str, loading_method: str) -> Dict[str, Any]:
        """幂等提交持久 job；不在 HTTP 请求线程里启动一次性执行线程。"""
        normalized_loading_method = self.qa_index_builder.validate_loading_method(loading_method)

        with self._submission_lock:
            # 进程内锁只减少本实例的重复写竞争；跨进程互斥由 SQLite 的 BEGIN IMMEDIATE 提供。
            job = self.paper_qa_index_store.acquire_paper_index_job(
                arxiv_id,
                normalized_loading_method,
                timeout_seconds=self.timeout_seconds,
                recipe_version=self.recipe_version,
                max_attempts=self.max_attempts,
            )
            if not job:
                raise AppError(
                    ErrorCode.DATABASE_WRITE_FAILED,
                    detail="acquire_paper_index_job returned empty result",
                    context={"arxiv_id": arxiv_id, "stage": "submit_qa_index_job"},
                )

            # 常驻 worker 可能正处于轮询等待；唤醒信号不承担持久语义，进程重启后仍会从数据库重新发现 job。
            self._wake_event.set()
            logger.info(
                "%s QA index job: job_id=%s arxiv_id=%s status=%s",
                "Submitted" if bool(job.get("created")) else "Reused",
                job.get("job_id"),
                arxiv_id,
                job.get("status"),
            )
            return job

    def start(self) -> None:
        """启动当前进程的常驻 worker；重复调用保持幂等。"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=self.worker_id,
        )
        self._worker_thread.start()
        self._wake_event.set()

    def stop(self, *, join_timeout_seconds: float = 5.0) -> None:
        """停止领取新任务；正在执行的 builder 不强杀，由 lease 在下次启动时安全回收。"""
        self._stop_event.set()
        self._wake_event.set()
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, float(join_timeout_seconds)))

    def _worker_loop(self) -> None:
        """持续从数据库领取工作，保证后端重启后 pending/retrying job 能继续执行。"""
        while not self._stop_event.is_set():
            processed = self.run_next_job()
            if processed:
                continue
            self._wake_event.wait(self.poll_interval_seconds)
            self._wake_event.clear()

    def run_next_job(self) -> bool:
        """领取并执行一个 job；返回值只表示本轮是否实际领取到工作。"""
        job = self.paper_qa_index_store.claim_next_paper_index_job(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if not job:
            return False
        self._execute_claimed_job(job)
        return True

    def run_job(self, job_id: str, arxiv_id: str = "", loading_method: str = "") -> None:
        """兼容管理脚本的定向执行入口；仍必须先通过数据库 lease 领取。"""
        del arxiv_id, loading_method
        job = self.paper_qa_index_store.claim_next_paper_index_job(
            worker_id=self.worker_id,
            lease_seconds=self.lease_seconds,
            job_id=job_id,
        )
        if job:
            self._execute_claimed_job(job)

    def _execute_claimed_job(self, job: Mapping[str, Any]) -> None:
        """执行已经领取的 attempt，并把 builder 输出先验证再写成 job 成功。"""
        job_id = str(job.get("job_id") or "")
        arxiv_id = str(job.get("arxiv_id") or "")
        loading_method = str(job.get("loading_method") or "docling")
        attempt_no = int(job.get("attempt_count") or 0)
        logger.info(
            "Starting QA index job: job_id=%s arxiv_id=%s attempt_no=%s worker_id=%s",
            job_id,
            arxiv_id,
            attempt_no,
            self.worker_id,
        )

        heartbeat_stop = threading.Event()
        heartbeat_worker = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, attempt_no, heartbeat_stop),
            daemon=True,
            name=f"{self.worker_id}-heartbeat-{job_id}",
        )
        heartbeat_worker.start()

        def progress_callback(*, current_stage: str, progress: int, message: str) -> None:
            """阶段回调写入真实里程碑；CAS 同时阻止失去 lease 的旧 attempt 回写。"""
            updated = self.paper_qa_index_store.heartbeat_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                lease_seconds=self.lease_seconds,
                current_stage=current_stage,
                progress=progress,
                stage_message=message,
            )
            if not updated:
                logger.warning(
                    "QA index job progress ignored because lease ownership changed: job_id=%s arxiv_id=%s stage=%s",
                    job_id,
                    arxiv_id,
                    current_stage,
                )
            logger.debug(
                "QA index job progress: job_id=%s arxiv_id=%s stage=%s progress=%s message=%s",
                job_id,
                arxiv_id,
                current_stage,
                progress,
                message,
            )

        try:
            result = self.qa_index_builder.build_qa_index(
                arxiv_id,
                loading_method=loading_method,
                progress_callback=progress_callback,
            )
            validated_result = self._validate_success_result(arxiv_id, result)
            completed = self.paper_qa_index_store.complete_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                result=validated_result,
            )
            if completed:
                logger.info("QA index job completed: job_id=%s arxiv_id=%s attempt_no=%s", job_id, arxiv_id, attempt_no)
                if self.on_job_succeeded is not None:
                    # continuation 投影发生在 job 事务提交后；失败可由查询/reconciler 根据持久 job 结果补做。
                    try:
                        self.on_job_succeeded(job_id, validated_result)
                    except Exception:
                        logger.exception("QA index continuation success projection deferred: job_id=%s", job_id)
            else:
                logger.warning("QA index job completion ignored after lease change: job_id=%s", job_id)
        except AppError as exc:
            error_message = self._exception_message(exc)
            failed_stage = str(exc.context.get("stage") or getattr(exc, "error_stage", "failed") or "failed")
            self.paper_qa_index_store.fail_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                failed_stage=failed_stage,
                error_code=str(exc.code),
                error_message=f"{exc.code}: {error_message}",
            )
            if self.on_job_failed is not None:
                try:
                    self.on_job_failed(job_id, str(exc.code), f"{exc.code}: {error_message}")
                except Exception:
                    logger.exception("QA index continuation failure projection deferred: job_id=%s", job_id)
            logger.warning(
                "QA index job failed: code=%s job_id=%s arxiv_id=%s stage=%s recoverable=%s error=%s",
                exc.code,
                job_id,
                arxiv_id,
                failed_stage,
                exc.recoverable,
                error_message,
            )
        except HTTPException as exc:
            error_message = self._exception_message(exc)
            failed_stage = str(getattr(exc, "error_stage", "failed") or "failed")
            self.paper_qa_index_store.fail_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                failed_stage=failed_stage,
                error_code=f"http_{getattr(exc, 'status_code', 500)}",
                error_message=error_message,
            )
            if self.on_job_failed is not None:
                try:
                    self.on_job_failed(job_id, f"http_{getattr(exc, 'status_code', 500)}", error_message)
                except Exception:
                    logger.exception("QA index continuation failure projection deferred: job_id=%s", job_id)
            logger.warning(
                "QA index job failed: job_id=%s arxiv_id=%s stage=%s error=%s",
                job_id,
                arxiv_id,
                failed_stage,
                error_message,
            )
        except Exception as exc:
            error_message = self._exception_message(exc)
            failed_stage = str(getattr(exc, "error_stage", "failed") or "failed")
            # 完成后置校验失败与 builder 异常语义不同；先持久化明确 failure_code，
            # 再投影 continuation，投影失败也可由 reconciler 根据 job 终态补做。
            self.paper_qa_index_store.fail_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                failed_stage=failed_stage,
                error_code=("completion_validation_failed" if isinstance(exc, CompletionValidationError) else "qa_index_build_failed"),
                error_message=error_message,
            )
            if self.on_job_failed is not None:
                failure_code = "completion_validation_failed" if isinstance(exc, CompletionValidationError) else "qa_index_build_failed"
                try:
                    self.on_job_failed(job_id, failure_code, error_message)
                except Exception:
                    logger.exception("QA index continuation failure projection deferred: job_id=%s", job_id)
            logger.exception(
                "QA index job failed unexpectedly: job_id=%s arxiv_id=%s stage=%s error=%s",
                job_id,
                arxiv_id,
                failed_stage,
                error_message,
            )
        finally:
            heartbeat_stop.set()
            heartbeat_worker.join(timeout=max(0.2, self.heartbeat_interval_seconds * 2))

    def _heartbeat_loop(self, job_id: str, attempt_no: int, stop_event: threading.Event) -> None:
        """即使 builder 长时间没有阶段回调，也按固定周期续租，避免把慢任务误判成失联。"""
        while not stop_event.wait(self.heartbeat_interval_seconds):
            renewed = self.paper_qa_index_store.heartbeat_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                lease_seconds=self.lease_seconds,
            )
            if not renewed:
                return

    def _validate_success_result(self, arxiv_id: str, result: Any) -> Dict[str, Any]:
        """验证 builder 结果与 active 索引指针一致，防止“job success 但 QA 不可用”。"""
        payload = dict(result or {}) if isinstance(result, Mapping) else {}
        required = ("build_id", "index_version", "collection_name", "chunk_count")
        missing = [key for key in required if not payload.get(key)]
        if missing or int(payload.get("chunk_count") or 0) <= 0:
            raise CompletionValidationError(f"builder_result_invalid:{','.join(missing) or 'chunk_count'}")

        if self.qa_status_reader is not None:
            status = dict(self.qa_status_reader(arxiv_id) or {})
        else:
            active = self.paper_qa_index_store.get_paper_qa_index(arxiv_id) or {}
            status = {
                "has_index": active.get("status") == "indexed" and bool(active.get("collection_name")),
                "status": active.get("status"),
                "active_collection_name": active.get("collection_name"),
                "active_index_version": active.get("active_index_version"),
                "active_build_id": active.get("active_build_id"),
                "active_chunk_count": active.get("chunk_count"),
            }

        checks = {
            "has_index": bool(status.get("has_index")),
            "status": str(status.get("status") or "") == "indexed",
            "collection_name": str(status.get("active_collection_name") or status.get("collection_name") or "")
            == str(payload.get("collection_name") or ""),
            "build_id": str(status.get("active_build_id") or "") == str(payload.get("build_id") or ""),
            "index_version": str(status.get("active_index_version") or "") == str(payload.get("index_version") or ""),
            "chunk_count": int(status.get("active_chunk_count") or status.get("chunk_count") or 0) > 0,
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise CompletionValidationError(f"active_index_mismatch:{','.join(failed_checks)}")
        return payload


class CompletionValidationError(RuntimeError):
    """表示 builder 已返回但 active 索引后置条件不成立，禁止恢复上层 Agent。"""
