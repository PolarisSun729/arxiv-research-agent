import uuid
from typing import Any, Dict, List, Optional

from services.storage.database.shared import (
    DEFAULT_USER_ID,
    PROFILE_BUILD_VERSION,
    PROFILE_EXTRACTOR_VERSION,
    PROFILE_NORMALIZER_VERSION,
    logger,
)


class ProfileBuildJobMixin:
    """维护画像构建 job 状态；具体快照生成和激活仍由 profile 主流程负责。"""

    def _ensure_user_profile_build_job_columns(self, conn):
        # build job 是前端轮询的状态源；旧库补齐 metrics_json 后即可承载细粒度进度，不需要破坏现有列结构。
        required_columns = {
            "metrics_json": "TEXT",
        }
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user_profile_build_jobs)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for column_name, column_definition in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE user_profile_build_jobs ADD COLUMN {column_name} {column_definition}"
                )
        conn.commit()

    def list_user_profile_build_jobs(self, user_id: str = DEFAULT_USER_ID, limit: int = 20) -> List[Dict[str, Any]]:
        """返回画像构建任务历史，供前端轮询和排查慢速构建状态。"""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    '''
                    SELECT job_id, user_id, status, snapshot_id, current_stage, progress, error_message,
                           metrics_json, build_config_json, extractor_version, normalizer_version, profile_build_version,
                           created_at, updated_at
                    FROM user_profile_build_jobs
                    WHERE user_id = ?
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    ''',
                    (user_id, max(1, int(limit or 20))),
                ).fetchall()
            return [self._profile_build_job_row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing profile build jobs: {str(e)}")
            return []

    def get_user_profile_build_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """按 job_id 读取单个构建任务，返回结构与列表接口一致。"""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    '''
                    SELECT job_id, user_id, status, snapshot_id, current_stage, progress, error_message,
                           metrics_json, build_config_json, extractor_version, normalizer_version, profile_build_version,
                           created_at, updated_at
                    FROM user_profile_build_jobs
                    WHERE job_id = ?
                    ''',
                    (job_id,),
                ).fetchone()
            if not row:
                return None
            return self._profile_build_job_row_to_dict(row)
        except Exception as e:
            logger.error(f"Error getting profile build job: {str(e)}")
            return None

    def _profile_build_job_row_to_dict(self, row) -> Dict[str, Any]:
        """把 job 行统一展开为前端轮询结构，metrics 同时保留原始对象和常用顶层字段。"""
        metrics = self._deserialize_json_field(row[7]) or {}
        payload = {
            "job_id": row[0],
            "user_id": row[1],
            "status": row[2],
            "snapshot_id": row[3],
            "current_stage": row[4],
            "progress": int(row[5] or 0),
            "error_message": row[6],
            "metrics": metrics,
            "build_config": self._deserialize_json_field(row[8]) or {},
            "extractor_version": row[9],
            "normalizer_version": row[10],
            "profile_build_version": row[11],
            "created_at": row[12],
            "updated_at": row[13],
        }
        for field_name in (
            "total_papers",
            "candidate_papers",
            "cached_papers",
            "uncached_papers",
            "processed_papers",
            "failed_papers",
            "skipped_paper_count",
            "skipped_read_only_papers",
            "skipped_failed_cache_papers",
            "skipped_limit_papers",
            "repair_candidate_papers",
            "successful_papers",
            "cache_hit_count",
            "generated_count",
            "failed_count",
            "skipped_count",
            "average_seconds_per_paper",
            "total_evidence_extraction_seconds",
            "evidence_concurrency",
            "rate_limit_backoff_count",
            "paper_evidence_failure_details",
            "build_mode",
            "paper_limit",
            "evidence_counts",
            "current_arxiv_id",
            "stage_message",
            "recent_logs",
        ):
            if field_name in metrics:
                payload[field_name] = metrics.get(field_name)
        return payload

    def create_user_profile_build_job(
        self,
        user_id: str = DEFAULT_USER_ID,
        *,
        build_config: Optional[Dict[str, Any]] = None,
        status: str = "running",
    ) -> str:
        job_id = str(uuid.uuid4())
        config = dict(build_config or {})
        try:
            with self._get_connection() as conn:
                conn.execute(
                    '''
                    INSERT INTO user_profile_build_jobs (
                        job_id, user_id, status, current_stage, progress, metrics_json, build_config_json,
                        extractor_version, normalizer_version, profile_build_version
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        job_id,
                        user_id,
                        status,
                        "collect_evidence",
                        0,
                        self._serialize_json_field({
                            "current_stage": "collect_evidence",
                            "progress": 0,
                            "build_mode": config.get("build_mode") or "incremental",
                            "paper_limit": config.get("max_papers"),
                            "stage_message": "等待开始收集画像证据",
                        }),
                        self._serialize_json_field(config),
                        PROFILE_EXTRACTOR_VERSION,
                        PROFILE_NORMALIZER_VERSION,
                        PROFILE_BUILD_VERSION,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating profile build job: {str(e)}")
        return job_id

    def update_user_profile_build_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        current_stage: Optional[str] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        updates: List[str] = []
        values: List[Any] = []
        for column, value in {
            "status": status,
            "snapshot_id": snapshot_id,
            "current_stage": current_stage,
            "progress": progress,
            "error_message": error_message,
            "metrics_json": self._serialize_json_field(metrics) if metrics is not None else None,
        }.items():
            if value is None:
                continue
            updates.append(f"{column} = ?")
            values.append(value)
        if not updates:
            return
        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(job_id)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    f"UPDATE user_profile_build_jobs SET {', '.join(updates)} WHERE job_id = ?",
                    values,
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating profile build job: {str(e)}")
