"""论文 QA 索引构建的持久化任务管理器（基于数据库 lease 的单进程常驻 worker）。

工作模型：
1. HTTP 请求线程只调用 submit_job，把任务幂等写入数据库后立即返回，不在请求线程里建库；
2. 常驻 worker 线程循环 claim 一个 pending/retrying 任务并获得 lease（租约），执行 builder；
3. 执行期间由独立心跳线程定期续租；lease 过期的任务（进程崩溃/重启）会被重新领取；
4. 所有状态写入都带 worker_id + attempt_no 做 CAS，失去 lease 的旧 attempt 无法覆盖新结果。

任务状态的取值与流转规则由 PaperQAIndexStore 定义（见 PAPER_INDEX_ACTIVE_JOB_STATUSES 等常量）。
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable, Dict, Mapping, Optional

from fastapi import HTTPException

from core.errors import AppError, ErrorCode
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.storage.sqlite.stores.paper_qa_index import PaperQAIndexStore
from utils.config import get_qa_index_job_runtime_config

logger = logging.getLogger(__name__)


class CompletionValidationError(RuntimeError):
    """表示 builder 已返回但 active 索引后置条件不成立，禁止恢复上层 Agent。"""


class IndexBuildCancelledError(RuntimeError):
    """当前 attempt 已失去 lease，要求 builder 在下一个阶段边界中止建库。

    abort_build=True 让 PaperQAIndexBuilder._notify_progress 不吞掉该异常（普通回调异常会被吞掉）。
    """

    abort_build = True


class IndexJobManager:
    """QA 索引任务的提交入口与常驻执行 worker。"""

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
        """初始化可重启的数据库 lease worker；请求线程只负责持久化提交事实。

        Args:
            qa_status_reader: 可选的索引状态读取函数，用于成功后的后置校验；缺省时直接读 store。
            timeout_seconds: 单个 job 的超时时间（写入 job 记录，由 store 判定 stale）。
            lease_seconds: 每次领取/续租的租约时长；必须明显大于心跳间隔，否则慢任务会被误判失联。
            heartbeat_interval_seconds: 心跳续租间隔。
            poll_interval_seconds: 没有任务时 worker 的轮询间隔（submit 时会主动唤醒）。
            max_attempts: 单个 job 的最大尝试次数。
            recipe_version: 建库配方版本，参与任务幂等判断，配方变化时会生成新 job。
            on_job_succeeded / on_job_failed: 任务终态写库后的回调（用于恢复挂起的 Agent continuation）。
        其余参数缺省时读取运行时配置，并做下限保护，避免配置成 0 导致忙等或立即过期。
        """
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
            try:
                processed = self.run_next_job()
            except Exception:
                # claim/状态写入等数据库异常（如 SQLite database is locked）不能让常驻线程退出：
                # 线程一旦死掉，本进程内所有后续 job 都不会再被执行，且没有任何自动恢复手段。
                # 这里记录后按轮询间隔退避重试；执行到一半的 job 会因 lease 过期被重新领取。
                logger.exception("QA index worker iteration failed: worker_id=%s", self.worker_id)
                processed = False
            if processed:
                # 刚处理完一个 job 时立即尝试领取下一个，减少排队任务的等待时间。
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

    def _execute_claimed_job(self, job: Mapping[str, Any]) -> None:
        """执行已经领取的 attempt，并把 builder 输出先验证再写成 job 成功。"""
        job_id = str(job.get("job_id") or "")
        arxiv_id = str(job.get("arxiv_id") or "")
        loading_method = str(job.get("loading_method") or "docling")  # 与 builder 的 DEFAULT_LOADING_METHOD 一致
        attempt_no = int(job.get("attempt_count") or 0)
        logger.info(
            "Starting QA index job: job_id=%s arxiv_id=%s attempt_no=%s worker_id=%s",
            job_id,
            arxiv_id,
            attempt_no,
            self.worker_id,
        )

        # builder 可能在单个阶段（如 docling 解析、批量 embedding）停留很久，
        # 因此用独立线程按固定周期续租，而不是依赖阶段回调来续租。
        heartbeat_stop = threading.Event()
        # 一旦确认 lease 已被其他 worker 接管，就置位该事件；builder 在下一个阶段边界经进度回调被中止。
        lease_lost = threading.Event()
        heartbeat_worker = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, attempt_no, heartbeat_stop, lease_lost),
            daemon=True,
            name=f"{self.worker_id}-heartbeat-{job_id}",
        )
        heartbeat_worker.start()

        def progress_callback(*, current_stage: str, progress: int, message: str) -> None:
            """阶段回调写入真实里程碑；CAS 同时阻止失去 lease 的旧 attempt 回写。

            确认 lease 已丢失时抛 IndexBuildCancelledError 中止建库，避免旧 attempt 与新 attempt
            同时构建并先后激活同一篇论文的索引。
            """
            if lease_lost.is_set():
                raise IndexBuildCancelledError(f"lease lost before stage {current_stage}")
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
                if self._lease_lost(job_id, attempt_no):
                    lease_lost.set()
                    logger.warning(
                        "QA index job lease lost, cancelling build: job_id=%s arxiv_id=%s stage=%s",
                        job_id,
                        arxiv_id,
                        current_stage,
                    )
                    raise IndexBuildCancelledError(f"lease lost at stage {current_stage}")
                # 读不到 job（数据库暂时不可用）时无法确认 lease 已丢失，继续执行，由下次心跳再判断。
                logger.warning(
                    "QA index job progress not persisted: job_id=%s arxiv_id=%s stage=%s",
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
            if lease_lost.is_set() or isinstance(exc.__cause__, IndexBuildCancelledError):
                # 本 attempt 因 lease 丢失被主动中止：job 已归新 attempt 所有，不能再写失败终态或回调失败。
                # builder 已把本次 build 标记为 build_failed + cleanup_pending，产物交给清理任务回收。
                logger.warning(
                    "QA index job attempt cancelled after lease loss: job_id=%s arxiv_id=%s attempt_no=%s",
                    job_id,
                    arxiv_id,
                    attempt_no,
                )
                return
            # builder 的标准失败形态：error_code 与 message 都带上业务错误码，便于前端展示。
            error_message = f"{exc.code}: {self._exception_message(exc)}"
            failed_stage = str(exc.context.get("stage") or getattr(exc, "error_stage", "failed") or "failed")
            self._record_job_failure(
                job_id, attempt_no, failed_stage=failed_stage, error_code=str(exc.code), error_message=error_message,
            )
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
            # 当前 builder 会把 HTTPException 统一包装成 AppError；保留此分支是为了兼容
            # 直接抛 HTTP 异常的 builder 替身/旧实现，错误码按 http_<status> 记录。
            error_message = self._exception_message(exc)
            failed_stage = str(getattr(exc, "error_stage", "failed") or "failed")
            self._record_job_failure(
                job_id,
                attempt_no,
                failed_stage=failed_stage,
                error_code=f"http_{getattr(exc, 'status_code', 500)}",
                error_message=error_message,
            )
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
            # 完成后置校验失败与 builder 异常语义不同，需要用不同的 failure_code 区分。
            failure_code = "completion_validation_failed" if isinstance(exc, CompletionValidationError) else "qa_index_build_failed"
            self._record_job_failure(
                job_id, attempt_no, failed_stage=failed_stage, error_code=failure_code, error_message=error_message,
            )
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

    def _record_job_failure(
        self,
        job_id: str,
        attempt_no: int,
        *,
        failed_stage: str,
        error_code: str,
        error_message: str,
    ) -> None:
        """先持久化 job 失败终态，再投影到 continuation 回调；两步各自兜底，绝不向 worker 循环抛异常。

        - 失败状态写库本身也可能失败（如数据库被锁）：此时只记录日志，job 会因 lease 过期被重新领取；
        - 回调失败可由 reconciler 根据 job 终态补做，同样只记录日志。
        """
        try:
            failed = self.paper_qa_index_store.fail_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                failed_stage=failed_stage,
                error_code=error_code,
                error_message=error_message,
            )
        except Exception:
            logger.exception("Persist QA index job failure failed; lease expiry will recover it: job_id=%s", job_id)
            return
        if failed is False:
            # CAS 未命中：job 已被其他 attempt 接管或已是终态，失败不属于当前 job 状态，不能通知上层。
            logger.warning("QA index job failure ignored after lease change: job_id=%s attempt_no=%s", job_id, attempt_no)
            return
        if self.on_job_failed is not None:
            try:
                self.on_job_failed(job_id, error_code, error_message)
            except Exception:
                logger.exception("QA index continuation failure projection deferred: job_id=%s", job_id)

    def _heartbeat_loop(
        self,
        job_id: str,
        attempt_no: int,
        stop_event: threading.Event,
        lease_lost: Optional[threading.Event] = None,
    ) -> None:
        """即使 builder 长时间没有阶段回调，也按固定周期续租，避免把慢任务误判成失联。

        续租失败时先确认 lease 是否真的被接管：确认丢失则置位 lease_lost 并停止心跳，
        builder 会在下一个阶段边界被中止；只是数据库暂时不可用时继续按周期重试续租。
        """
        while not stop_event.wait(self.heartbeat_interval_seconds):
            renewed = self.paper_qa_index_store.heartbeat_paper_index_job(
                job_id,
                worker_id=self.worker_id,
                attempt_no=attempt_no,
                lease_seconds=self.lease_seconds,
            )
            if renewed:
                continue
            if self._lease_lost(job_id, attempt_no):
                logger.warning("QA index job lease lost during heartbeat: job_id=%s attempt_no=%s", job_id, attempt_no)
                if lease_lost is not None:
                    lease_lost.set()
                return
            logger.warning("QA index job heartbeat failed, will retry: job_id=%s attempt_no=%s", job_id, attempt_no)

    def _lease_lost(self, job_id: str, attempt_no: int) -> bool:
        """读取 job 当前归属，确认当前 attempt 是否已失去 lease。

        heartbeat 返回 False 既可能是 CAS 未命中（真的被接管 / 已终态），也可能是数据库异常；
        这里只有在读到 job 且其状态、worker 或 attempt 已变化时才返回 True，读不到时保守返回 False。
        """
        try:
            job = self.paper_qa_index_store.get_paper_index_job(job_id)
        except Exception:
            logger.exception("Read QA index job ownership failed: job_id=%s", job_id)
            return False
        if not job:
            return False
        return (
            str(job.get("status") or "") != "running"
            or str(job.get("worker_id") or "") != self.worker_id
            or int(job.get("attempt_count") or 0) != int(attempt_no)
        )

    def _validate_success_result(self, arxiv_id: str, result: Any) -> Dict[str, Any]:
        """验证 builder 结果与 active 索引指针一致，防止“job success 但 QA 不可用”。"""
        payload = dict(result or {}) if isinstance(result, Mapping) else {}
        # 第一步：builder 返回值本身必须完整。
        required = ("build_id", "index_version", "collection_name", "chunk_count")
        missing = [key for key in required if not payload.get(key)]
        if missing or int(payload.get("chunk_count") or 0) <= 0:
            raise CompletionValidationError(f"builder_result_invalid:{','.join(missing) or 'chunk_count'}")

        # 第二步：数据库中的 active 指针必须已经切到本次 build，问答侧才能真正读到新索引。
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
