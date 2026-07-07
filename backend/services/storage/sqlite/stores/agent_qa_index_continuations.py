from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


ACTIVE_CONTINUATION_STATUSES = {"waiting_job", "ready_to_resume", "failed"}


class AgentQAIndexContinuationStore(BaseSqliteStore):
    """维护 Agent 确认建索引后的“完成后继续回答”业务状态。"""

    def _row_to_continuation(self, row: Any) -> Dict[str, Any]:
        return {
            "job_id": row[0],
            "user_id": row[1],
            "session_id": row[2],
            "arxiv_id": row[3],
            "status": row[4],
            "pending_action_id": row[5],
            "step_id": row[6],
            "tool_name": row[7],
            "original_question": row[8],
            "resume_payload": self._deserialize_json_field(row[9]) or None,
            "pending_action": self._deserialize_json_field(row[10]) or None,
            "job_snapshot": self._deserialize_json_field(row[11]) or None,
            "error_message": row[12],
            "created_at": row[13],
            "updated_at": row[14],
            "completed_at": row[15],
        }

    def upsert_continuation(
        self,
        *,
        job_id: str,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        arxiv_id: str,
        pending_action_id: Optional[str] = None,
        step_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        original_question: Optional[str] = None,
        resume_payload: Optional[Dict[str, Any]] = None,
        pending_action: Optional[Dict[str, Any]] = None,
        job_snapshot: Optional[Dict[str, Any]] = None,
        status: str = "waiting_job",
    ) -> Optional[Dict[str, Any]]:
        """创建或更新 continuation，重复确认时复用同一个 job 记录。"""
        normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        normalized_status = str(status or "waiting_job").strip() or "waiting_job"
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO agent_qa_index_continuations (
                        job_id, user_id, session_id, arxiv_id, status,
                        pending_action_id, step_id, tool_name, original_question,
                        resume_payload_json, pending_action_json, job_snapshot_json,
                        error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
                    ON CONFLICT(job_id) DO UPDATE SET
                        user_id = excluded.user_id,
                        session_id = excluded.session_id,
                        arxiv_id = excluded.arxiv_id,
                        status = excluded.status,
                        pending_action_id = excluded.pending_action_id,
                        step_id = excluded.step_id,
                        tool_name = excluded.tool_name,
                        original_question = excluded.original_question,
                        resume_payload_json = excluded.resume_payload_json,
                        pending_action_json = excluded.pending_action_json,
                        job_snapshot_json = excluded.job_snapshot_json,
                        error_message = '',
                        completed_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        job_id,
                        normalized_user_id,
                        session_id,
                        arxiv_id,
                        normalized_status,
                        pending_action_id,
                        step_id,
                        tool_name,
                        original_question,
                        self._serialize_json_field(resume_payload),
                        self._serialize_json_field(pending_action),
                        self._serialize_json_field(job_snapshot),
                    ),
                )
                conn.commit()
            return self.get_continuation(job_id, user_id=normalized_user_id)
        except Exception as e:
            logger.error(f"Error upserting agent QA index continuation: {str(e)}")
            return None

    def get_continuation(self, job_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        """按 job_id 读取 continuation，供前端刷新后恢复轮询状态。"""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT job_id, user_id, session_id, arxiv_id, status,
                           pending_action_id, step_id, tool_name, original_question,
                           resume_payload_json, pending_action_json, job_snapshot_json,
                           error_message, created_at, updated_at, completed_at
                    FROM agent_qa_index_continuations
                    WHERE job_id = ? AND user_id = ?
                    """,
                    (job_id, str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID),
                ).fetchone()
            return self._row_to_continuation(row) if row else None
        except Exception as e:
            logger.error(f"Error getting agent QA index continuation: {str(e)}")
            return None

    def list_active_continuations(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """列出仍需前端处理的 continuation，不返回已取消或已恢复的历史记录。"""
        normalized_user_id = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        statuses = tuple(ACTIVE_CONTINUATION_STATUSES)
        placeholders = ",".join("?" for _ in statuses)
        filters = [f"user_id = ?", f"status IN ({placeholders})"]
        values: List[Any] = [normalized_user_id, *statuses]
        if session_id:
            filters.append("session_id = ?")
            values.append(str(session_id).strip())
        values.append(max(1, int(limit or 20)))
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT job_id, user_id, session_id, arxiv_id, status,
                           pending_action_id, step_id, tool_name, original_question,
                           resume_payload_json, pending_action_json, job_snapshot_json,
                           error_message, created_at, updated_at, completed_at
                    FROM agent_qa_index_continuations
                    WHERE {" AND ".join(filters)}
                    ORDER BY updated_at DESC, created_at DESC
                    LIMIT ?
                    """,
                    values,
                ).fetchall()
            return [self._row_to_continuation(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing agent QA index continuations: {str(e)}")
            return []

    def update_status(
        self,
        job_id: str,
        *,
        user_id: str = DEFAULT_USER_ID,
        status: str,
        error_message: Optional[str] = None,
        job_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """更新 continuation 状态；终态会写 completed_at，便于后续清理和排查。"""
        normalized_status = str(status or "").strip()
        if not normalized_status:
            return None
        completed_statuses = {"resumed", "cancelled", "failed"}
        update_fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        values: List[Any] = [normalized_status]
        if error_message is not None:
            update_fields.append("error_message = ?")
            values.append(str(error_message or "").strip())
        if job_snapshot is not None:
            update_fields.append("job_snapshot_json = ?")
            values.append(self._serialize_json_field(job_snapshot))
        if normalized_status in completed_statuses:
            update_fields.append("completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)")
        try:
            with self._get_connection() as conn:
                conn.execute(
                    f"""
                    UPDATE agent_qa_index_continuations
                    SET {", ".join(update_fields)}
                    WHERE job_id = ? AND user_id = ?
                    """,
                    [
                        *values,
                        job_id,
                        str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID,
                    ],
                )
                conn.commit()
            return self.get_continuation(job_id, user_id=user_id)
        except Exception as e:
            logger.error(f"Error updating agent QA index continuation status: {str(e)}")
            return None
