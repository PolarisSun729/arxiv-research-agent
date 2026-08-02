from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import DEFAULT_USER_ID, logger


class AgentRuntimeCheckpointStore(BaseSqliteStore):
    """持久化 Agent 业务执行快照和唯一 interaction，不解释 LangGraph 内部状态。"""

    _SELECT_COLUMNS = """
        checkpoint_id, schema_version, user_id, session_id, thread_id,
        runtime_state_json, graph_state_json, interaction_json,
        current_node, next_route, status, error_summary,
        created_at, updated_at, expires_at
    """

    def _row(self, row: Any) -> Dict[str, Any]:
        return {
            "checkpoint_id": row[0],
            "schema_version": int(row[1] or 1),
            "user_id": row[2],
            "session_id": row[3],
            "thread_id": row[4],
            "runtime_state": self._deserialize_json_field(row[5]) or None,
            "graph_state": self._deserialize_json_field(row[6]) or None,
            "interaction": self._deserialize_json_field(row[7]) or None,
            "current_node": row[8] or "",
            "next_route": row[9] or "",
            "status": row[10] or "running",
            "error_summary": row[11] or "",
            "created_at": row[12],
            "updated_at": row[13],
            "expires_at": row[14],
        }

    def upsert_agent_runtime_checkpoint(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: str,
        runtime_state: Optional[Dict[str, Any]] = None,
        graph_state: Optional[Dict[str, Any]] = None,
        interaction: Optional[Dict[str, Any]] = None,
        schema_version: int = 2,
        current_node: Optional[str] = None,
        next_route: Optional[str] = None,
        status: str = "running",
        error_summary: Optional[str] = None,
        expires_at: Optional[str] = None,
        **_ignored: Any,
    ) -> Optional[Dict[str, Any]]:
        normalized_user = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        session_id = str(session_id or "").strip()
        thread_id = str(thread_id or session_id).strip()
        if not session_id or not thread_id:
            return None
        checkpoint_id = f"{normalized_user}:{session_id}:{thread_id}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO agent_runtime_checkpoints (
                    checkpoint_id, schema_version, user_id, session_id, thread_id,
                    runtime_state_json, graph_state_json, interaction_json,
                    current_node, next_route, status, error_summary, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, session_id, thread_id) DO UPDATE SET
                    checkpoint_id = excluded.checkpoint_id,
                    schema_version = excluded.schema_version,
                    runtime_state_json = excluded.runtime_state_json,
                    graph_state_json = excluded.graph_state_json,
                    interaction_json = excluded.interaction_json,
                    current_node = excluded.current_node,
                    next_route = excluded.next_route,
                    status = excluded.status,
                    error_summary = excluded.error_summary,
                    expires_at = excluded.expires_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    checkpoint_id,
                    int(schema_version),
                    normalized_user,
                    session_id,
                    thread_id,
                    self._serialize_json_field(runtime_state),
                    self._serialize_json_field(graph_state),
                    self._serialize_json_field(interaction),
                    str(current_node or "").strip(),
                    str(next_route or "").strip(),
                    str(status or "running").strip(),
                    str(error_summary or "").strip(),
                    expires_at,
                ),
            )
            conn.commit()
        return self.get_agent_runtime_checkpoint(user_id=normalized_user, session_id=session_id, thread_id=thread_id)

    def get_agent_runtime_checkpoint(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized_user = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        thread_id = str(thread_id or session_id).strip()
        with self._get_connection() as conn:
            row = conn.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM agent_runtime_checkpoints WHERE user_id = ? AND session_id = ? AND thread_id = ?",
                (normalized_user, session_id, thread_id),
            ).fetchone()
        return self._row(row) if row else None

    def list_agent_runtime_checkpoints_by_thread(self, *, session_id: str, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM agent_runtime_checkpoints WHERE session_id = ? AND thread_id = ? ORDER BY updated_at DESC LIMIT ?",
                (session_id, thread_id, max(int(limit), 1)),
            ).fetchall()
        return [self._row(row) for row in rows]

    def resolve_agent_interaction(
        self,
        *,
        user_id: str,
        session_id: str,
        thread_id: str,
        interaction_id: str,
        terminal_status: str = "running",
    ) -> bool:
        normalized_user = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT interaction_json FROM agent_runtime_checkpoints
                   WHERE user_id = ? AND session_id = ? AND thread_id = ?
                     AND schema_version = 2 AND status = 'waiting_interaction'""",
                (normalized_user, session_id, thread_id),
            ).fetchone()
            interaction = self._deserialize_json_field(row[0]) if row else None
            if not isinstance(interaction, dict) or str(interaction.get("interaction_id") or "") != interaction_id:
                return False
            updated = conn.execute(
                """UPDATE agent_runtime_checkpoints
                   SET interaction_json = '', status = ?, next_route = ?, expires_at = NULL,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ? AND session_id = ? AND thread_id = ? AND status = 'waiting_interaction'""",
                (terminal_status, terminal_status, normalized_user, session_id, thread_id),
            ).rowcount
            conn.commit()
            return updated == 1

    def mark_agent_runtime_checkpoint_status(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
        thread_id: Optional[str] = None,
        status: str,
        error_summary: Optional[str] = None,
        clear_interaction: bool = False,
        **_ignored: Any,
    ) -> bool:
        assignments = ["status = ?", "error_summary = ?", "updated_at = CURRENT_TIMESTAMP"]
        values: List[Any] = [status, str(error_summary or "")]
        if clear_interaction:
            assignments.extend(["interaction_json = ''", "expires_at = NULL"])
        normalized_user = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        thread_id = str(thread_id or session_id).strip()
        with self._get_connection() as conn:
            updated = conn.execute(
                f"UPDATE agent_runtime_checkpoints SET {', '.join(assignments)} WHERE user_id = ? AND session_id = ? AND thread_id = ?",
                values + [normalized_user, session_id, thread_id],
            ).rowcount
            conn.commit()
            return updated > 0

    def cancel_agent_runtime_checkpoints_for_session(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        session_id: str,
    ) -> int:
        """清空对话时终止未完成 checkpoint，防止旧 interaction 在新对话中再次恢复。"""
        normalized_user = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        with self._get_connection() as conn:
            updated = conn.execute(
                """
                UPDATE agent_runtime_checkpoints
                SET status = 'cancelled', next_route = 'cancelled', interaction_json = '',
                    expires_at = NULL, error_summary = '', updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND session_id = ?
                  AND status IN ('running', 'waiting_interaction', 'waiting_background_job')
                  AND checkpoint_id NOT IN (
                      SELECT runtime_checkpoint_id
                      FROM agent_work_continuations
                      WHERE user_id = ? AND session_id = ? AND status = 'resuming'
                  )
                """,
                (normalized_user, session_id, normalized_user, session_id),
            ).rowcount
            # 已经进入 resuming 的执行不能安全中断；它保留原历史归属，但不会注入新的会话。
            conn.commit()
            return int(updated or 0)

    def expire_agent_runtime_checkpoints(self, *, now: Optional[str] = None) -> int:
        now = now or datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            updated = conn.execute(
                """UPDATE agent_runtime_checkpoints
                   SET status = 'expired', interaction_json = '', expires_at = NULL,
                       error_summary = COALESCE(NULLIF(error_summary, ''), 'checkpoint_expired'),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE status = 'waiting_interaction' AND expires_at IS NOT NULL AND expires_at <= ?""",
                (now,),
            ).rowcount
            conn.commit()
            return int(updated or 0)

    def cleanup_agent_runtime_checkpoints(self, *, retention_days: int = 7) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(int(retention_days), 1))).isoformat()
        with self._get_connection() as conn:
            deleted = conn.execute(
                "DELETE FROM agent_runtime_checkpoints WHERE status IN ('completed','cancelled','failed','expired') AND updated_at < ?",
                (cutoff,),
            ).rowcount
            conn.commit()
            return int(deleted or 0)

    def cleanup_langgraph_checkpoints_for_terminal_runtime(self, *, retention_days: int = 7) -> Dict[str, int]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(int(retention_days), 1))).isoformat()
        with self._get_connection() as conn:
            thread_rows = conn.execute(
                """SELECT DISTINCT thread_id FROM agent_runtime_checkpoints
                   WHERE status IN ('completed','cancelled','failed','expired') AND updated_at < ?""",
                (cutoff,),
            ).fetchall()
            thread_ids = [str(row[0]) for row in thread_rows if row and row[0]]
            checkpoints = writes = 0
            for thread_id in thread_ids:
                writes += conn.execute("DELETE FROM langgraph_checkpoint_writes WHERE thread_id = ?", (thread_id,)).rowcount
                checkpoints += conn.execute("DELETE FROM langgraph_checkpoints WHERE thread_id = ?", (thread_id,)).rowcount
            conn.commit()
        return {"threads": len(thread_ids), "checkpoints": int(checkpoints), "writes": int(writes)}

    def get_context_lifecycle_stats(
        self,
        *,
        user_id: str = DEFAULT_USER_ID,
        paper_session_id: Optional[str] = None,
        agent_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """汇总上下文存储规模；只读取长度和计数，不加载正文或把 debug 当作恢复状态。"""
        normalized_user = str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID
        stats: Dict[str, Any] = {
            "user_id": normalized_user,
            "paper_chat": {},
            "agent_runtime_checkpoint": {},
            "langgraph_checkpoint": {},
        }
        try:
            with self._get_connection() as conn:
                if paper_session_id:
                    row = conn.execute(
                        """SELECT s.message_count, COUNT(m.message_id), s.summary_updated_at,
                                  s.summary_turn_count, s.summary_last_turn_id,
                                  LENGTH(COALESCE(s.summary_json, ''))
                           FROM paper_chat_sessions s
                           LEFT JOIN paper_chat_messages m ON m.session_id = s.session_id
                           WHERE s.session_id = ? AND s.user_id = ? GROUP BY s.session_id""",
                        (paper_session_id, normalized_user),
                    ).fetchone()
                    stats["paper_chat"] = {
                        "session_id": paper_session_id,
                        "message_count": int(row[0] or 0) if row else 0,
                        "stored_message_count": int(row[1] or 0) if row else 0,
                        "summary_loaded": bool(row and row[2]),
                        "summary_updated_at": row[2] if row else None,
                        "summary_turn_count": int(row[3] or 0) if row else 0,
                        "summary_last_turn_id": row[4] if row else "",
                        "summary_chars": int(row[5] or 0) if row else 0,
                    }
                if agent_session_id:
                    row = conn.execute(
                        """SELECT status, current_node, next_route, expires_at, updated_at,
                                  LENGTH(COALESCE(runtime_state_json, '')),
                                  LENGTH(COALESCE(graph_state_json, '')),
                                  LENGTH(COALESCE(interaction_json, ''))
                           FROM agent_runtime_checkpoints
                           WHERE user_id = ? AND session_id = ? AND thread_id = ?""",
                        (normalized_user, agent_session_id, agent_session_id),
                    ).fetchone()
                    stats["agent_runtime_checkpoint"] = {
                        "session_id": agent_session_id,
                        "exists": bool(row),
                        "status": row[0] if row else None,
                        "current_node": row[1] if row else "",
                        "next_route": row[2] if row else "",
                        "expires_at": row[3] if row else None,
                        "updated_at": row[4] if row else None,
                        "runtime_state_chars": int(row[5] or 0) if row else 0,
                        "graph_state_chars": int(row[6] or 0) if row else 0,
                        "interaction_chars": int(row[7] or 0) if row else 0,
                    }
                    checkpoint_count = conn.execute(
                        "SELECT COUNT(*) FROM langgraph_checkpoints WHERE thread_id = ?", (agent_session_id,)
                    ).fetchone()
                    write_count = conn.execute(
                        "SELECT COUNT(*) FROM langgraph_checkpoint_writes WHERE thread_id = ?", (agent_session_id,)
                    ).fetchone()
                    stats["langgraph_checkpoint"] = {
                        "thread_id": agent_session_id,
                        "checkpoint_count": int((checkpoint_count or [0])[0] or 0),
                        "write_count": int((write_count or [0])[0] or 0),
                    }
            return stats
        except Exception as exc:
            logger.error("Error getting context lifecycle stats: %s", exc)
            stats["error"] = str(exc)
            return stats
