from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from fastapi import HTTPException

from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.storage.database_service import DatabaseService

logger = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = {"pending", "running"}


class IndexJobManager:
    def __init__(self, *, db_service: DatabaseService, qa_index_builder: PaperQAIndexBuilder):
        self.db_service = db_service
        self.qa_index_builder = qa_index_builder
        self._submission_lock = threading.Lock()

    @staticmethod
    def _exception_message(exc: Exception) -> str:
        detail = getattr(exc, "detail", None)
        if detail is not None:
            text = str(detail).strip()
            if text:
                return text[:500]
        text = str(exc).strip()
        return (text or exc.__class__.__name__)[:500]

    def _find_active_job(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        jobs = self.db_service.list_paper_index_jobs(arxiv_id=arxiv_id, limit=20)
        for job in jobs:
            if str(job.get("status") or "").strip().lower() in ACTIVE_JOB_STATUSES:
                return job
        return None

    def submit_job(self, arxiv_id: str, loading_method: str) -> Dict[str, Any]:
        normalized_loading_method = self.qa_index_builder.validate_loading_method(loading_method)

        with self._submission_lock:
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
                raise RuntimeError("Failed to create QA index job")

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
        logger.info("Starting QA index job: job_id=%s arxiv_id=%s", job_id, arxiv_id)
        self.db_service.update_paper_index_job(
            job_id,
            status="running",
            current_stage="starting",
            progress=0,
            error_message="",
        )

        def progress_callback(*, current_stage: str, progress: int, message: str) -> None:
            self.db_service.update_paper_index_job(
                job_id,
                status="running",
                current_stage=current_stage,
                progress=progress,
                error_message="",
            )
            logger.info(
                "QA index job progress: job_id=%s arxiv_id=%s stage=%s progress=%s message=%s",
                job_id,
                arxiv_id,
                current_stage,
                progress,
                message,
            )

        try:
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
        except HTTPException as exc:
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
