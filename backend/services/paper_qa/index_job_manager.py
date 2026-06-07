from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from fastapi import HTTPException

from core.errors import AppError, ErrorCode
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.storage.database_service import DatabaseService

logger = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = {"pending", "running"}


class IndexJobManager:
    def __init__(self, *, db_service: DatabaseService, qa_index_builder: PaperQAIndexBuilder):
        """初始化论文问答索引任务管理器，并持有任务提交所需依赖。"""
        self.db_service = db_service
        self.qa_index_builder = qa_index_builder
        self._submission_lock = threading.Lock()

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

    def _find_active_job(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """查找指定论文当前是否已经存在仍在执行中的建索引任务。"""
        jobs = self.db_service.list_paper_index_jobs(arxiv_id=arxiv_id, limit=20)
        for job in jobs:
            if str(job.get("status") or "").strip().lower() in ACTIVE_JOB_STATUSES:
                return job
        return None

    def submit_job(self, arxiv_id: str, loading_method: str) -> Dict[str, Any]:
        """提交新的论文建索引任务；若已有活动任务，则直接复用已有任务。"""
        normalized_loading_method = self.qa_index_builder.validate_loading_method(loading_method)

        with self._submission_lock:
            # 提交阶段加锁，避免同一篇论文在并发请求下被重复创建多个后台任务。
            existing_job = self._find_active_job(arxiv_id)
            if existing_job:
                logger.info(
                    "Reusing existing QA index job: job_id=%s arxiv_id=%s status=%s",
                    existing_job.get("job_id"),
                    arxiv_id,
                    existing_job.get("status"),
                )
                return existing_job

            job = self.db_service.create_paper_index_job(arxiv_id, normalized_loading_method)
            if not job:
                raise AppError(
                    ErrorCode.DATABASE_WRITE_FAILED,
                    detail="create_paper_index_job returned empty result",
                    context={"arxiv_id": arxiv_id, "stage": "submit_qa_index_job"},
                )

            # 后台线程负责真正的建索引执行，请求线程只负责创建并返回任务信息。
            worker = threading.Thread(
                target=self.run_job,
                args=(job["job_id"], arxiv_id, normalized_loading_method),
                daemon=True,
                name=f"qa-index-{job['job_id']}",
            )
            worker.start()
            logger.info("Submitted QA index job: job_id=%s arxiv_id=%s", job["job_id"], arxiv_id)
            return job

    def run_job(self, job_id: str, arxiv_id: str, loading_method: str) -> None:
        """执行单个建索引任务，并持续把进度与最终结果写回数据库。"""
        logger.info("Starting QA index job: job_id=%s arxiv_id=%s", job_id, arxiv_id)
        self.db_service.update_paper_index_job(
            job_id,
            status="running",
            current_stage="starting",
            progress=0,
            error_message="",
        )

        def progress_callback(*, current_stage: str, progress: int, message: str) -> None:
            """接收索引构建阶段回调，并把阶段进度同步到任务记录中。"""
            self.db_service.update_paper_index_job(
                job_id,
                status="running",
                current_stage=current_stage,
                progress=progress,
                error_message="",
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
            # 真正的建索引逻辑下沉到 builder，这里只负责任务生命周期编排。
            self.qa_index_builder.build_qa_index(
                arxiv_id,
                loading_method=loading_method,
                progress_callback=progress_callback,
            )
            self.db_service.update_paper_index_job(
                job_id,
                status="success",
                current_stage="mark_index_success",
                progress=100,
                error_message="",
            )
            logger.info("QA index job completed: job_id=%s arxiv_id=%s", job_id, arxiv_id)
        except AppError as exc:
            error_message = self._exception_message(exc)
            failed_stage = str(exc.context.get("stage") or getattr(exc, "error_stage", "failed") or "failed")
            # 后台任务失败时同时写入 code 和阶段，前端轮询 job 时能稳定识别失败类型。
            self.db_service.update_paper_index_job(
                job_id,
                status="failed",
                current_stage=failed_stage,
                error_message=f"{exc.code}: {error_message}",
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
            # 业务性失败通常已经带有明确阶段和错误描述，直接写回任务记录即可。
            error_message = self._exception_message(exc)
            failed_stage = str(getattr(exc, "error_stage", "failed") or "failed")
            self.db_service.update_paper_index_job(
                job_id,
                status="failed",
                current_stage=failed_stage,
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
            # 非预期异常也会尽量落库，避免前端只能看到任务卡住而没有失败原因。
            error_message = self._exception_message(exc)
            failed_stage = str(getattr(exc, "error_stage", "failed") or "failed")
            self.db_service.update_paper_index_job(
                job_id,
                status="failed",
                current_stage=failed_stage,
                error_message=error_message,
            )
            logger.exception(
                "QA index job failed unexpectedly: job_id=%s arxiv_id=%s stage=%s error=%s",
                job_id,
                arxiv_id,
                failed_stage,
                error_message,
            )
