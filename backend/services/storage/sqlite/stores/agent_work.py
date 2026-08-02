from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from services.storage.sqlite.base import BaseSqliteStore


ACTIVE_CONTINUATION_STATUSES = ("submitting", "waiting_job", "ready_to_resume", "resuming")


class AgentWorkConflict(RuntimeError):
    """表示后台工作身份不匹配、状态已被消费或 CAS 推进失败。"""


class AgentWorkStore(BaseSqliteStore):
    """持久化 Agent 后台 continuation、resume run 与只读生命周期事件。"""

    _CONTINUATION_COLUMNS = """
        continuation_id, user_id, session_id, thread_id, runtime_checkpoint_id,
        interaction_id, grant_id, invocation_id, plan_id, step_id, tool_name,
        arguments_fingerprint, handler_name, job_id, job_idempotency_key, status,
        handler_state_json, display_summary_json, validated_result_json, resume_run_id, error_code,
        error_message, created_at, updated_at, ready_at, expires_at, terminal_at
    """

    def _continuation_row(self, row: Any) -> Dict[str, Any]:
        keys = (
            "continuation_id", "user_id", "session_id", "thread_id", "runtime_checkpoint_id",
            "interaction_id", "grant_id", "invocation_id", "plan_id", "step_id", "tool_name",
            "arguments_fingerprint", "handler_name", "job_id", "job_idempotency_key", "status",
            "handler_state_json", "display_summary_json", "validated_result_json", "resume_run_id", "error_code",
            "error_message", "created_at", "updated_at", "ready_at", "expires_at", "terminal_at",
        )
        payload = dict(zip(keys, row))
        payload["handler_state"] = self._deserialize_json_field(payload.pop("handler_state_json", None)) or {}
        payload["display_summary"] = self._deserialize_json_field(payload.pop("display_summary_json", None)) or {}
        payload["validated_result"] = self._deserialize_json_field(payload.pop("validated_result_json", None))
        return payload

    def _append_event(
        self,
        conn: Any,
        *,
        event_type: str,
        continuation_id: Optional[str] = None,
        job_id: Optional[str] = None,
        invocation_id: Optional[str] = None,
        resume_run_id: Optional[str] = None,
        stage: Optional[str] = None,
        progress: Optional[int] = None,
        error_code: Optional[str] = None,
        safe_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        # 事件只解释时间线，业务判断始终读取 continuation/job/resume_run 当前状态表。
        conn.execute(
            """
            INSERT INTO agent_work_events (
                event_id, event_type, occurred_at, continuation_id, job_id, invocation_id,
                resume_run_id, stage, progress, error_code, safe_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                event_type,
                datetime.now(timezone.utc).isoformat(),
                continuation_id,
                job_id,
                invocation_id,
                resume_run_id,
                stage,
                progress,
                error_code,
                self._serialize_json_field(dict(safe_metadata or {})),
            ),
        )

    def _expire_ready_continuation_in_transaction(
        self,
        conn: Any,
        *,
        continuation_id: str,
        runtime_checkpoint_id: str,
        now_text: str,
    ) -> None:
        """在当前事务内同步终结 continuation 与 runtime checkpoint，避免两条记录状态分叉。"""
        conn.execute(
            """
            UPDATE agent_work_continuations
            SET status = 'expired', error_code = 'continuation_ready_expired',
                error_message = '后台结果等待恢复已超时。', terminal_at = ?, updated_at = ?
            WHERE continuation_id = ? AND status = 'ready_to_resume'
            """,
            (now_text, now_text, continuation_id),
        )
        conn.execute(
            """
            UPDATE agent_runtime_checkpoints
            SET status = 'expired', next_route = 'expired',
                error_summary = 'continuation_ready_expired', updated_at = CURRENT_TIMESTAMP
            WHERE checkpoint_id = ? AND status = 'waiting_background_job'
            """,
            (runtime_checkpoint_id,),
        )
        self._append_event(
            conn,
            event_type="continuation_expired",
            continuation_id=continuation_id,
            error_code="continuation_ready_expired",
        )

    def prepare_background_work(
        self,
        *,
        checkpoint_id: str,
        user_id: str,
        session_id: str,
        thread_id: str,
        interaction_id: str,
        grant_id: str,
        invocation_id: str,
        continuation_id: str,
        plan_id: str,
        step_id: str,
        tool_name: str,
        arguments_fingerprint: str,
        handler_name: str,
        job_id: Optional[str],
        job_idempotency_key: str,
        handler_state: Optional[Mapping[str, Any]] = None,
        display_summary: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """原子消费批准并建立 continuation，事务内不提交 job、不调用外部服务。"""
        now = datetime.now(timezone.utc).isoformat()
        continuation_status = "waiting_job" if job_id else "submitting"
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT schema_version, user_id, session_id, thread_id, runtime_state_json,
                       interaction_json, status
                FROM agent_runtime_checkpoints WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("checkpoint_missing")
            if int(row[0] or 1) != 2:
                raise AgentWorkConflict("checkpoint_schema_outdated")
            if tuple(str(item or "") for item in row[1:4]) != (user_id, session_id, thread_id):
                raise AgentWorkConflict("checkpoint_owner_mismatch")
            interaction = self._deserialize_json_field(row[5])
            if not isinstance(interaction, Mapping) or str(interaction.get("interaction_id") or "") != interaction_id:
                raise AgentWorkConflict("interaction_identity_mismatch")
            if str(row[6] or "") != "waiting_interaction":
                raise AgentWorkConflict("checkpoint_not_waiting_interaction")
            payload = interaction.get("payload") if isinstance(interaction.get("payload"), Mapping) else {}
            expected_identity = (
                str(interaction.get("plan_id") or ""),
                str(interaction.get("step_id") or ""),
                str(payload.get("tool_name") or ""),
                str(payload.get("arguments_fingerprint") or ""),
            )
            if expected_identity != (plan_id, step_id, tool_name, arguments_fingerprint):
                raise AgentWorkConflict("background_identity_mismatch")

            conn.execute(
                """
                INSERT INTO approval_grants (
                    grant_id, interaction_id, user_id, session_id, thread_id, plan_id, step_id,
                    tool_name, arguments_fingerprint, status, approved_at, consumed_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'consumed', ?, ?, NULL)
                """,
                (
                    grant_id, interaction_id, user_id, session_id, thread_id, plan_id, step_id,
                    tool_name, arguments_fingerprint, now, now,
                ),
            )
            conn.execute(
                """
                INSERT INTO side_effect_invocations (
                    invocation_id, grant_id, plan_id, step_id, tool_name, arguments_fingerprint,
                    status, prepared_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invocation_id,
                    grant_id,
                    plan_id,
                    step_id,
                    tool_name,
                    arguments_fingerprint,
                    "invoking" if job_id else "prepared",
                    now,
                    now if job_id else None,
                ),
            )
            conn.execute(
                """
                INSERT INTO agent_work_continuations (
                    continuation_id, user_id, session_id, thread_id, runtime_checkpoint_id,
                    interaction_id, grant_id, invocation_id, plan_id, step_id, tool_name,
                    arguments_fingerprint, handler_name, job_id, job_idempotency_key, status,
                    handler_state_json, display_summary_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    continuation_id, user_id, session_id, thread_id, checkpoint_id,
                    interaction_id, grant_id, invocation_id, plan_id, step_id, tool_name,
                    arguments_fingerprint, handler_name, job_id, job_idempotency_key,
                    continuation_status, self._serialize_json_field(dict(handler_state or {})),
                    self._serialize_json_field(dict(display_summary or {})), now, now,
                ),
            )

            runtime_state = self._deserialize_json_field(row[4])
            runtime_state = dict(runtime_state) if isinstance(runtime_state, Mapping) else {}
            runtime_state["background_continuation_id"] = continuation_id
            runtime_state["turn_status"] = "waiting_background_job"
            step_status = dict(runtime_state.get("step_status") or {})
            step_status[step_id] = "waiting_background_job"
            runtime_state["step_status"] = step_status
            updated = conn.execute(
                """
                UPDATE agent_runtime_checkpoints
                SET runtime_state_json = ?, interaction_json = '', status = 'waiting_background_job',
                    next_route = 'waiting_background_job', expires_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = ? AND status = 'waiting_interaction'
                """,
                (self._serialize_json_field(runtime_state), checkpoint_id),
            ).rowcount
            if updated != 1:
                raise AgentWorkConflict("interaction_already_resolved")
            self._append_event(
                conn,
                event_type="background_work_prepared",
                continuation_id=continuation_id,
                job_id=job_id,
                invocation_id=invocation_id,
                safe_metadata={"handler_name": handler_name, "status": continuation_status},
            )
            conn.commit()
        continuation = self.get_continuation(continuation_id)
        if continuation is None:  # pragma: no cover - 事务提交后记录必须可见
            raise RuntimeError("background continuation was not persisted")
        return continuation

    def get_continuation(self, continuation_id: str) -> Optional[Dict[str, Any]]:
        with self.connection_provider.connect() as conn:
            row = conn.execute(
                f"SELECT {self._CONTINUATION_COLUMNS} FROM agent_work_continuations WHERE continuation_id = ?",
                (continuation_id,),
            ).fetchone()
        return self._continuation_row(row) if row else None

    def list_continuations(
        self,
        *,
        user_id: str,
        session_id: Optional[str] = None,
        statuses: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        filters = ["user_id = ?"]
        values: List[Any] = [user_id]
        if session_id:
            filters.append("session_id = ?")
            values.append(session_id)
        normalized_statuses = [str(item) for item in (statuses or []) if str(item).strip()]
        if normalized_statuses:
            filters.append(f"status IN ({','.join('?' for _ in normalized_statuses)})")
            values.extend(normalized_statuses)
        values.append(max(1, int(limit)))
        with self.connection_provider.connect() as conn:
            rows = conn.execute(
                f"SELECT {self._CONTINUATION_COLUMNS} FROM agent_work_continuations "
                f"WHERE {' AND '.join(filters)} ORDER BY updated_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._continuation_row(row) for row in rows]

    def list_unretrieved_resume_continuations(
        self,
        *,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """列出已完成但客户端尚未取回结果的 continuation，供 SSE 断线后恢复气泡。"""
        filters = [
            "c.user_id = ?",
            "c.status = 'resumed'",
            "r.status = 'completed'",
            "r.result_retrieved_at IS NULL",
        ]
        values: List[Any] = [user_id]
        if session_id:
            filters.append("c.session_id = ?")
            values.append(session_id)
        values.append(max(1, int(limit)))
        qualified_columns = ", ".join(
            f"c.{column.strip()}" for column in self._CONTINUATION_COLUMNS.split(",")
        )
        with self.connection_provider.connect() as conn:
            rows = conn.execute(
                f"SELECT {qualified_columns} FROM agent_work_continuations c "
                "JOIN agent_resume_runs r ON r.resume_run_id = c.resume_run_id "
                f"WHERE {' AND '.join(filters)} ORDER BY c.updated_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._continuation_row(row) for row in rows]

    def attach_job(
        self,
        continuation_id: str,
        *,
        job_id: str,
        attached: bool,
    ) -> Dict[str, Any]:
        """把事务外幂等提交得到的 job 绑定回 submitting continuation。"""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT invocation_id, status, job_id FROM agent_work_continuations WHERE continuation_id = ?",
                (continuation_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("continuation_missing")
            if str(row[1]) == "waiting_job" and str(row[2] or "") == job_id:
                conn.commit()
                return self.get_continuation(continuation_id) or {}
            if str(row[1]) != "submitting":
                raise AgentWorkConflict(f"continuation_not_submitting:{row[1]}")
            conn.execute(
                """
                UPDATE agent_work_continuations
                SET job_id = ?, status = 'waiting_job', updated_at = ?
                WHERE continuation_id = ? AND status = 'submitting'
                """,
                (job_id, now, continuation_id),
            )
            conn.execute(
                """
                UPDATE side_effect_invocations
                SET status = 'invoking', started_at = COALESCE(started_at, ?)
                WHERE invocation_id = ? AND status = 'prepared'
                """,
                (now, row[0]),
            )
            self._append_event(
                conn,
                event_type="background_job_attached" if attached else "background_job_submitted",
                continuation_id=continuation_id,
                job_id=job_id,
                invocation_id=row[0],
            )
            conn.commit()
        return self.get_continuation(continuation_id) or {}

    def mark_job_ready(
        self,
        *,
        job_id: str,
        validated_result: Mapping[str, Any],
        ready_ttl_days: int = 7,
    ) -> int:
        """把一个成功物理 job 原子投影到所有仍在等待的独立 continuation。"""
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=max(1, int(ready_ttl_days)))).isoformat()
        result_json = self._serialize_json_field(dict(validated_result))
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT continuation_id, invocation_id, status FROM agent_work_continuations
                WHERE job_id = ? AND status IN ('submitting', 'waiting_job', 'cancelled')
                """,
                (job_id,),
            ).fetchall()
            ready_count = 0
            for continuation_id, invocation_id, continuation_status in rows:
                if continuation_status in {"submitting", "waiting_job"}:
                    conn.execute(
                        """
                        UPDATE agent_work_continuations
                        SET status = 'ready_to_resume', validated_result_json = ?, ready_at = ?,
                            expires_at = ?, updated_at = ?
                        WHERE continuation_id = ? AND status IN ('submitting', 'waiting_job')
                        """,
                        (result_json, now.isoformat(), expires_at, now.isoformat(), continuation_id),
                    )
                    ready_count += 1
                conn.execute(
                    """
                    UPDATE side_effect_invocations
                    SET status = 'succeeded', started_at = COALESCE(started_at, ?), finished_at = ?,
                        result_summary_json = ?, error_code = ''
                    WHERE invocation_id = ? AND status IN ('prepared', 'invoking')
                    """,
                    (now.isoformat(), now.isoformat(), result_json, invocation_id),
                )
                if continuation_status in {"submitting", "waiting_job"}:
                    self._append_event(
                        conn,
                        event_type="continuation_ready",
                        continuation_id=continuation_id,
                        job_id=job_id,
                        invocation_id=invocation_id,
                    )
            conn.commit()
        return ready_count

    def cancel_continuation(
        self,
        continuation_id: str,
        *,
        user_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """只终止当前 continuation 与原问题，不修改可能被共享的物理 job。"""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT user_id, session_id, status, runtime_checkpoint_id, job_id, invocation_id
                FROM agent_work_continuations WHERE continuation_id = ?
                """,
                (continuation_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("continuation_missing")
            if (str(row[0]), str(row[1])) != (user_id, session_id):
                raise AgentWorkConflict("continuation_owner_mismatch")
            status = str(row[2] or "")
            if status == "cancelled":
                conn.commit()
                return self.get_continuation(continuation_id) or {}
            if status not in {"submitting", "waiting_job", "ready_to_resume"}:
                raise AgentWorkConflict(f"continuation_not_cancellable:{status}")
            updated = conn.execute(
                """
                UPDATE agent_work_continuations
                SET status = 'cancelled', terminal_at = ?, updated_at = ?
                WHERE continuation_id = ? AND status IN ('submitting', 'waiting_job', 'ready_to_resume')
                """,
                (now, now, continuation_id),
            ).rowcount
            if updated != 1:
                raise AgentWorkConflict("continuation_cancel_conflict")
            conn.execute(
                """
                UPDATE agent_runtime_checkpoints
                SET status = 'cancelled', next_route = 'cancelled', error_summary = '',
                    updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = ? AND status = 'waiting_background_job'
                """,
                (row[3],),
            )
            self._append_event(
                conn,
                event_type="continuation_cancelled",
                continuation_id=continuation_id,
                job_id=row[4],
                invocation_id=row[5],
            )
            conn.commit()
        return self.get_continuation(continuation_id) or {}

    def cancel_session_continuations(self, *, user_id: str, session_id: str) -> int:
        """终止指定会话尚未开始恢复的 continuation，并保留任务历史用于审计。"""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT continuation_id, runtime_checkpoint_id, job_id, invocation_id
                FROM agent_work_continuations
                WHERE user_id = ? AND session_id = ?
                  AND status IN ('submitting', 'waiting_job', 'ready_to_resume')
                """,
                (user_id, session_id),
            ).fetchall()
            for continuation_id, checkpoint_id, job_id, invocation_id in rows:
                conn.execute(
                    """
                    UPDATE agent_work_continuations
                    SET status = 'cancelled', terminal_at = ?, updated_at = ?
                    WHERE continuation_id = ?
                      AND status IN ('submitting', 'waiting_job', 'ready_to_resume')
                    """,
                    (now, now, continuation_id),
                )
                conn.execute(
                    """
                    UPDATE agent_runtime_checkpoints
                    SET status = 'cancelled', next_route = 'cancelled', interaction_json = '',
                        expires_at = NULL, error_summary = '', updated_at = CURRENT_TIMESTAMP
                    WHERE checkpoint_id = ?
                      AND status IN ('waiting_interaction', 'waiting_background_job')
                    """,
                    (checkpoint_id,),
                )
                self._append_event(
                    conn,
                    event_type="continuation_cancelled",
                    continuation_id=continuation_id,
                    job_id=job_id,
                    invocation_id=invocation_id,
                    safe_metadata={"reason": "agent_session_cleared"},
                )
            conn.commit()
        return len(rows)

    def expire_ready_continuation_if_needed(
        self,
        continuation_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """在查询或恢复前终结过期 ready continuation，并同步终结业务 checkpoint。"""
        current_time = now or datetime.now(timezone.utc)
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status, expires_at, runtime_checkpoint_id
                FROM agent_work_continuations WHERE continuation_id = ?
                """,
                (continuation_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("continuation_missing")
            if str(row[0] or "") != "ready_to_resume" or not str(row[1] or "").strip():
                conn.commit()
                return self.get_continuation(continuation_id) or {}
            try:
                expired = datetime.fromisoformat(str(row[1])) <= current_time
            except ValueError as exc:
                raise AgentWorkConflict("continuation_expiry_invalid") from exc
            if not expired:
                conn.commit()
                return self.get_continuation(continuation_id) or {}

            now_text = current_time.isoformat()
            self._expire_ready_continuation_in_transaction(
                conn,
                continuation_id=continuation_id,
                runtime_checkpoint_id=str(row[2]),
                now_text=now_text,
            )
            conn.commit()
        return self.get_continuation(continuation_id) or {}

    @staticmethod
    def _resume_run_row(row: Any) -> Dict[str, Any]:
        keys = (
            "resume_run_id", "continuation_id", "user_id", "session_id", "thread_id",
            "status", "started_at", "finished_at", "final_response_json", "error_code",
            "error_message", "result_retrieved_at",
        )
        payload = dict(zip(keys, row))
        payload["final_response"] = BaseSqliteStore._deserialize_json_field(payload.pop("final_response_json", None))
        return payload

    def get_resume_run(self, resume_run_id: str) -> Optional[Dict[str, Any]]:
        with self.connection_provider.connect() as conn:
            row = conn.execute(
                """
                SELECT resume_run_id, continuation_id, user_id, session_id, thread_id,
                       status, started_at, finished_at, final_response_json, error_code,
                       error_message, result_retrieved_at
                FROM agent_resume_runs WHERE resume_run_id = ?
                """,
                (resume_run_id,),
            ).fetchone()
        return self._resume_run_row(row) if row else None

    def list_resume_runs(self, *, statuses: Optional[List[str]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        filters: List[str] = []
        values: List[Any] = []
        normalized_statuses = [str(item) for item in (statuses or []) if str(item).strip()]
        if normalized_statuses:
            filters.append(f"status IN ({','.join('?' for _ in normalized_statuses)})")
            values.extend(normalized_statuses)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        values.append(max(1, int(limit)))
        with self.connection_provider.connect() as conn:
            rows = conn.execute(
                """
                SELECT resume_run_id, continuation_id, user_id, session_id, thread_id,
                       status, started_at, finished_at, final_response_json, error_code,
                       error_message, result_retrieved_at
                FROM agent_resume_runs
                """ + where + " ORDER BY COALESCE(started_at, '') ASC LIMIT ?",
                values,
            ).fetchall()
        return [self._resume_run_row(row) for row in rows]

    def claim_resume_run(
        self,
        continuation_id: str,
        *,
        user_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """CAS claim continuation；重复请求只返回同一个 run，绝不再次消费 LangGraph checkpoint。"""
        now = datetime.now(timezone.utc)
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT c.user_id, c.session_id, c.thread_id, c.status, c.expires_at, c.resume_run_id,
                       c.runtime_checkpoint_id, p.user_id, p.session_id, p.thread_id, p.status
                FROM agent_work_continuations c
                LEFT JOIN agent_runtime_checkpoints p ON p.checkpoint_id = c.runtime_checkpoint_id
                WHERE c.continuation_id = ?
                """,
                (continuation_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("continuation_missing")
            if (str(row[0]), str(row[1])) != (user_id, session_id):
                raise AgentWorkConflict("continuation_owner_mismatch")
            existing_run_id = str(row[5] or "").strip()
            if existing_run_id:
                conn.commit()
                existing = self.get_resume_run(existing_run_id)
                if existing is None:
                    raise AgentWorkConflict("resume_run_missing")
                return {**existing, "created": False}
            if row[7] is None:
                raise AgentWorkConflict("continuation_checkpoint_missing")
            if (str(row[7]), str(row[8]), str(row[9])) != (str(row[0]), str(row[1]), str(row[2])):
                raise AgentWorkConflict("continuation_checkpoint_owner_mismatch")
            if str(row[10] or "") != "waiting_background_job":
                raise AgentWorkConflict(f"continuation_checkpoint_not_waiting:{row[10]}")
            if str(row[3] or "") != "ready_to_resume":
                raise AgentWorkConflict(f"continuation_not_ready:{row[3]}")
            expires_at = str(row[4] or "").strip()
            if expires_at:
                try:
                    if datetime.fromisoformat(expires_at) <= now:
                        self._expire_ready_continuation_in_transaction(
                            conn,
                            continuation_id=continuation_id,
                            runtime_checkpoint_id=str(row[6]),
                            now_text=now.isoformat(),
                        )
                        conn.commit()
                        raise AgentWorkConflict("continuation_expired")
                except ValueError as exc:
                    raise AgentWorkConflict("continuation_expiry_invalid") from exc

            resume_run_id = str(uuid4())
            started_at = now.isoformat()
            conn.execute(
                """
                INSERT INTO agent_resume_runs (
                    resume_run_id, continuation_id, user_id, session_id, thread_id, status, started_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', NULL)
                """,
                (resume_run_id, continuation_id, user_id, session_id, row[2]),
            )
            updated = conn.execute(
                """
                UPDATE agent_work_continuations
                SET status = 'resuming', resume_run_id = ?, updated_at = ?
                WHERE continuation_id = ? AND status = 'ready_to_resume' AND resume_run_id IS NULL
                """,
                (resume_run_id, started_at, continuation_id),
            ).rowcount
            if updated != 1:
                raise AgentWorkConflict("continuation_resume_conflict")
            conn.execute(
                """
                UPDATE agent_runtime_checkpoints
                SET status = 'running', next_route = 'observe_step', error_summary = '',
                    updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = ? AND status = 'waiting_background_job'
                """,
                (row[6],),
            )
            self._append_event(
                conn,
                event_type="resume_run_started",
                continuation_id=continuation_id,
                resume_run_id=resume_run_id,
            )
            conn.commit()
        claimed = self.get_resume_run(resume_run_id)
        if claimed is None:  # pragma: no cover
            raise RuntimeError("resume run was not persisted")
        return {**claimed, "created": True}

    def start_resume_run(self, resume_run_id: str) -> bool:
        """只有后台 runner 能把 pending run 推进为 running，CAS 防止多个请求重复执行图恢复。"""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection_provider.connect() as conn:
            updated = conn.execute(
                """
                UPDATE agent_resume_runs SET status = 'running', started_at = ?
                WHERE resume_run_id = ? AND status = 'pending'
                """,
                (now, resume_run_id),
            ).rowcount
            conn.commit()
        return updated == 1

    def complete_resume_run(
        self,
        resume_run_id: str,
        *,
        final_response: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """先持久化完整响应，再把 continuation/checkpoint 标记终态，供断线后安全取回。"""
        now = datetime.now(timezone.utc).isoformat()
        response_json = self._serialize_json_field(dict(final_response))
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT r.continuation_id, r.status, c.runtime_checkpoint_id
                FROM agent_resume_runs r
                JOIN agent_work_continuations c ON c.continuation_id = r.continuation_id
                WHERE r.resume_run_id = ?
                """,
                (resume_run_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("resume_run_missing")
            if str(row[1]) == "completed":
                conn.commit()
                return self.get_resume_run(resume_run_id) or {}
            if str(row[1]) != "running":
                raise AgentWorkConflict(f"resume_run_not_running:{row[1]}")
            conn.execute(
                """
                UPDATE agent_resume_runs
                SET status = 'completed', final_response_json = ?, finished_at = ?,
                    error_code = '', error_message = ''
                WHERE resume_run_id = ? AND status = 'running'
                """,
                (response_json, now, resume_run_id),
            )
            conn.execute(
                """
                UPDATE agent_work_continuations
                SET status = 'resumed', terminal_at = ?, updated_at = ?
                WHERE continuation_id = ? AND status = 'resuming' AND resume_run_id = ?
                """,
                (now, now, row[0], resume_run_id),
            )
            conn.execute(
                """
                UPDATE agent_runtime_checkpoints
                SET status = 'completed', next_route = 'completed', error_summary = '',
                    updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = ?
                """,
                (row[2],),
            )
            self._append_event(
                conn,
                event_type="resume_run_completed",
                continuation_id=row[0],
                resume_run_id=resume_run_id,
            )
            conn.commit()
        return self.get_resume_run(resume_run_id) or {}

    def fail_resume_run(
        self,
        resume_run_id: str,
        *,
        status: str,
        error_code: str,
        error_message: str,
    ) -> Dict[str, Any]:
        """终结失败或结果未知的 run；indeterminate 禁止任何自动重放。"""
        if status not in {"failed", "indeterminate"}:
            raise ValueError("resume run failure status must be failed or indeterminate")
        now = datetime.now(timezone.utc).isoformat()
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT r.continuation_id, r.status, c.runtime_checkpoint_id
                FROM agent_resume_runs r
                JOIN agent_work_continuations c ON c.continuation_id = r.continuation_id
                WHERE r.resume_run_id = ?
                """,
                (resume_run_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("resume_run_missing")
            if str(row[1]) in {"completed", "failed", "indeterminate"}:
                conn.commit()
                return self.get_resume_run(resume_run_id) or {}
            conn.execute(
                """
                UPDATE agent_resume_runs
                SET status = ?, finished_at = ?, error_code = ?, error_message = ?
                WHERE resume_run_id = ? AND status IN ('pending', 'running')
                """,
                (status, now, error_code, str(error_message)[:500], resume_run_id),
            )
            conn.execute(
                """
                UPDATE agent_work_continuations
                SET status = ?, error_code = ?, error_message = ?, terminal_at = ?, updated_at = ?
                WHERE continuation_id = ? AND status = 'resuming'
                """,
                (status, error_code, str(error_message)[:500], now, now, row[0]),
            )
            conn.execute(
                """
                UPDATE agent_runtime_checkpoints
                SET status = 'failed', next_route = ?, error_summary = ?, updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = ?
                """,
                (status, error_code, row[2]),
            )
            self._append_event(
                conn,
                event_type=f"resume_run_{status}",
                continuation_id=row[0],
                resume_run_id=resume_run_id,
                error_code=error_code,
            )
            conn.commit()
        return self.get_resume_run(resume_run_id) or {}

    def mark_resume_result_retrieved(self, resume_run_id: str) -> bool:
        """只记录结果读取审计时间，不改变 run/continuation 业务状态。"""
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT continuation_id FROM agent_resume_runs WHERE resume_run_id = ? AND status = 'completed'",
                (resume_run_id,),
            ).fetchone()
            if row is None:
                conn.commit()
                return False
            updated = conn.execute(
                "UPDATE agent_resume_runs SET result_retrieved_at = CURRENT_TIMESTAMP WHERE resume_run_id = ?",
                (resume_run_id,),
            ).rowcount
            self._append_event(
                conn,
                event_type="resume_result_retrieved",
                continuation_id=row[0],
                resume_run_id=resume_run_id,
            )
            conn.commit()
        return updated == 1

    def mark_continuation_ready(
        self,
        continuation_id: str,
        *,
        validated_result: Mapping[str, Any],
        ready_ttl_days: int = 7,
    ) -> Dict[str, Any]:
        """修复 submitting 竞态：索引已由其他 job 满足时只推进指定 continuation。"""
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=max(1, int(ready_ttl_days)))).isoformat()
        result_json = self._serialize_json_field(dict(validated_result))
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT invocation_id, job_id FROM agent_work_continuations WHERE continuation_id = ?",
                (continuation_id,),
            ).fetchone()
            if row is None:
                raise AgentWorkConflict("continuation_missing")
            updated = conn.execute(
                """
                UPDATE agent_work_continuations
                SET status = 'ready_to_resume', validated_result_json = ?, ready_at = ?,
                    expires_at = ?, updated_at = ?
                WHERE continuation_id = ? AND status IN ('submitting', 'waiting_job')
                """,
                (result_json, now.isoformat(), expires_at, now.isoformat(), continuation_id),
            ).rowcount
            if updated != 1:
                raise AgentWorkConflict("continuation_not_waiting")
            conn.execute(
                """
                UPDATE side_effect_invocations
                SET status = 'succeeded', started_at = COALESCE(started_at, ?), finished_at = ?,
                    result_summary_json = ?, error_code = ''
                WHERE invocation_id = ? AND status IN ('prepared', 'invoking')
                """,
                (now.isoformat(), now.isoformat(), result_json, row[0]),
            )
            self._append_event(
                conn,
                event_type="continuation_ready",
                continuation_id=continuation_id,
                job_id=row[1],
                invocation_id=row[0],
                safe_metadata={"source": "reconciliation"},
            )
            conn.commit()
        return self.get_continuation(continuation_id) or {}

    def mark_job_failed(self, *, job_id: str, error_code: str, error_message: str) -> int:
        """明确业务失败终结所有关联 continuation，不触发自动同步或 Agent 恢复。"""
        now = datetime.now(timezone.utc).isoformat()
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT continuation_id, invocation_id, runtime_checkpoint_id
                FROM agent_work_continuations
                WHERE job_id = ? AND status IN ('submitting', 'waiting_job')
                """,
                (job_id,),
            ).fetchall()
            for continuation_id, invocation_id, checkpoint_id in rows:
                conn.execute(
                    """
                    UPDATE agent_work_continuations
                    SET status = 'failed', error_code = ?, error_message = ?, terminal_at = ?, updated_at = ?
                    WHERE continuation_id = ?
                    """,
                    (error_code, str(error_message)[:500], now, now, continuation_id),
                )
                conn.execute(
                    """
                    UPDATE side_effect_invocations
                    SET status = 'failed', started_at = COALESCE(started_at, ?), finished_at = ?, error_code = ?
                    WHERE invocation_id = ? AND status IN ('prepared', 'invoking')
                    """,
                    (now, now, error_code, invocation_id),
                )
                conn.execute(
                    """
                    UPDATE agent_runtime_checkpoints
                    SET status = 'failed', next_route = 'failed', error_summary = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE checkpoint_id = ? AND status = 'waiting_background_job'
                    """,
                    (error_code, checkpoint_id),
                )
                self._append_event(
                    conn,
                    event_type="background_job_failed",
                    continuation_id=continuation_id,
                    job_id=job_id,
                    invocation_id=invocation_id,
                    error_code=error_code,
                )
            conn.commit()
        return len(rows)


__all__ = ["ACTIVE_CONTINUATION_STATUSES", "AgentWorkConflict", "AgentWorkStore"]
