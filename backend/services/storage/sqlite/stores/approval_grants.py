from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from services.storage.sqlite.base import BaseSqliteStore


class ApprovalGrantConflict(RuntimeError):
    """表示授权已失效、已消费或与即将执行的调用身份不一致。"""


class ApprovalGrantStore(BaseSqliteStore):
    """原子维护副作用授权消费与 invocation 创建，避免一次批准触发多次调用。"""

    def create_grant(self, grant: Any) -> None:
        """保存领域层授权快照；store 只依赖序列化协议，避免反向导入 Agent 包形成循环依赖。"""

        model_dump = getattr(grant, "model_dump", None)
        payload = model_dump(mode="json") if callable(model_dump) else dict(grant)
        if not isinstance(payload, Mapping):
            raise TypeError("approval grant must be a mapping or pydantic model")
        with self.connection_provider.connect() as conn:
            conn.execute(
                """
                INSERT INTO approval_grants (
                    grant_id, interaction_id, user_id, session_id, thread_id,
                    plan_id, step_id, tool_name, arguments_fingerprint, status,
                    approved_at, consumed_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["grant_id"], payload["interaction_id"], payload["user_id"],
                    payload["session_id"], payload["thread_id"], payload["plan_id"],
                    payload["step_id"], payload["tool_name"], payload["arguments_fingerprint"],
                    payload["status"], payload["approved_at"], payload["consumed_at"], payload["revoked_at"],
                ),
            )
            conn.commit()

    def get_grant(self, grant_id: str) -> Optional[Dict[str, Any]]:
        with self.connection_provider.connect() as conn:
            row = conn.execute(
                """SELECT grant_id, interaction_id, user_id, session_id, thread_id, plan_id,
                          step_id, tool_name, arguments_fingerprint, status, approved_at,
                          consumed_at, revoked_at
                   FROM approval_grants WHERE grant_id = ?""",
                (grant_id,),
            ).fetchone()
        if row is None:
            return None
        keys = (
            "grant_id", "interaction_id", "user_id", "session_id", "thread_id", "plan_id",
            "step_id", "tool_name", "arguments_fingerprint", "status", "approved_at",
            "consumed_at", "revoked_at",
        )
        return dict(zip(keys, row))

    def find_approved_grant(
        self,
        *,
        user_id: str,
        session_id: str,
        thread_id: str,
        plan_id: str,
        step_id: str,
        tool_name: str,
        arguments_fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        """只返回与本次最终调用身份完全一致且尚未消费的授权。"""

        with self.connection_provider.connect() as conn:
            row = conn.execute(
                """SELECT grant_id, interaction_id, user_id, session_id, thread_id, plan_id,
                          step_id, tool_name, arguments_fingerprint, status, approved_at,
                          consumed_at, revoked_at
                   FROM approval_grants
                   WHERE user_id = ? AND session_id = ? AND thread_id = ?
                     AND plan_id = ? AND step_id = ? AND tool_name = ?
                     AND arguments_fingerprint = ? AND status = 'approved'
                   ORDER BY approved_at DESC LIMIT 1""",
                (user_id, session_id, thread_id, plan_id, step_id, tool_name, arguments_fingerprint),
            ).fetchone()
        if row is None:
            return None
        keys = (
            "grant_id", "interaction_id", "user_id", "session_id", "thread_id", "plan_id",
            "step_id", "tool_name", "arguments_fingerprint", "status", "approved_at",
            "consumed_at", "revoked_at",
        )
        return dict(zip(keys, row))

    def get_invocation(self, invocation_id: str) -> Optional[Dict[str, Any]]:
        with self.connection_provider.connect() as conn:
            row = conn.execute(
                """SELECT invocation_id, grant_id, plan_id, step_id, tool_name,
                          arguments_fingerprint, status, prepared_at
                   FROM side_effect_invocations WHERE invocation_id = ?""",
                (invocation_id,),
            ).fetchone()
        if row is None:
            return None
        keys = (
            "invocation_id", "grant_id", "plan_id", "step_id", "tool_name",
            "arguments_fingerprint", "status", "prepared_at",
        )
        return dict(zip(keys, row))

    def mark_invocation(self, invocation_id: str, *, status: str, result_summary: Any = None, error_code: str = "") -> bool:
        """推进副作用调用终态；未知结果使用 indeterminate，禁止执行器自动重放。"""

        now = datetime.now(timezone.utc).isoformat()
        with self.connection_provider.connect() as conn:
            updated = conn.execute(
                """UPDATE side_effect_invocations
                   SET status = ?, started_at = COALESCE(started_at, ?),
                       finished_at = CASE WHEN ? IN ('succeeded', 'failed', 'indeterminate') THEN ? ELSE finished_at END,
                       result_summary_json = ?, error_code = ?
                   WHERE invocation_id = ?""",
                (
                    status,
                    now,
                    status,
                    now,
                    self._serialize_json_field(result_summary),
                    error_code,
                    invocation_id,
                ),
            ).rowcount
            conn.commit()
            return updated == 1

    def consume_and_prepare_invocation(
        self,
        *,
        grant_id: str,
        invocation_id: str,
        plan_id: str,
        step_id: str,
        tool_name: str,
        arguments_fingerprint: str,
    ) -> Dict[str, Any]:
        """在同一事务中消费授权并创建 prepared invocation，关闭崩溃前的重复调用窗口。"""

        now = datetime.now(timezone.utc).isoformat()
        expected_identity = (plan_id, step_id, tool_name, arguments_fingerprint)
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT status, plan_id, step_id, tool_name, arguments_fingerprint
                   FROM approval_grants WHERE grant_id = ?""",
                (grant_id,),
            ).fetchone()
            if row is None:
                raise ApprovalGrantConflict("grant_not_found")
            if row[0] != "approved":
                raise ApprovalGrantConflict(f"grant_not_approved:{row[0]}")
            if tuple(row[1:]) != expected_identity:
                raise ApprovalGrantConflict("grant_identity_mismatch")

            updated = conn.execute(
                """UPDATE approval_grants SET status = 'consumed', consumed_at = ?
                   WHERE grant_id = ? AND status = 'approved'""",
                (now, grant_id),
            ).rowcount
            if updated != 1:
                raise ApprovalGrantConflict("grant_not_approved")
            conn.execute(
                """INSERT INTO side_effect_invocations (
                       invocation_id, grant_id, plan_id, step_id, tool_name,
                       arguments_fingerprint, status, prepared_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 'prepared', ?)""",
                (invocation_id, grant_id, plan_id, step_id, tool_name, arguments_fingerprint, now),
            )
            conn.commit()

        invocation = self.get_invocation(invocation_id)
        if invocation is None:  # pragma: no cover - 事务提交成功后记录必须存在
            raise RuntimeError("prepared invocation was not persisted")
        return invocation

    def resolve_side_effect_interaction(self, *, checkpoint_id: str, grant: Any) -> Dict[str, Any]:
        """原子解析副作用交互并创建授权，避免 interaction 已清除但 grant 未落盘。"""

        model_dump = getattr(grant, "model_dump", None)
        payload = model_dump(mode="json") if callable(model_dump) else dict(grant)
        with self.connection_provider.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT schema_version, interaction_json, status
                   FROM agent_runtime_checkpoints WHERE checkpoint_id = ?""",
                (checkpoint_id,),
            ).fetchone()
            if row is None:
                raise ApprovalGrantConflict("checkpoint_missing")
            if int(row[0] or 1) != 2:
                raise ApprovalGrantConflict("checkpoint_schema_outdated")
            interaction = self._deserialize_json_field(row[1])
            if not isinstance(interaction, dict):
                raise ApprovalGrantConflict("interaction_missing")
            if str(interaction.get("interaction_id") or "") != str(payload.get("interaction_id") or ""):
                raise ApprovalGrantConflict("interaction_id_mismatch")
            if str(interaction.get("kind") or "") != "side_effect_approval":
                raise ApprovalGrantConflict("interaction_kind_mismatch")
            if str(row[2] or "") != "waiting_interaction":
                raise ApprovalGrantConflict("checkpoint_not_waiting")

            conn.execute(
                """INSERT INTO approval_grants (
                    grant_id, interaction_id, user_id, session_id, thread_id,
                    plan_id, step_id, tool_name, arguments_fingerprint, status,
                    approved_at, consumed_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["grant_id"], payload["interaction_id"], payload["user_id"],
                    payload["session_id"], payload["thread_id"], payload["plan_id"],
                    payload["step_id"], payload["tool_name"], payload["arguments_fingerprint"],
                    payload["status"], payload["approved_at"], payload.get("consumed_at"), payload.get("revoked_at"),
                ),
            )
            updated = conn.execute(
                """UPDATE agent_runtime_checkpoints
                   SET interaction_json = '', status = 'running', next_route = 'running',
                       expires_at = NULL, updated_at = CURRENT_TIMESTAMP
                   WHERE checkpoint_id = ? AND status = 'waiting_interaction'
                     AND interaction_json IS NOT NULL AND interaction_json <> ''""",
                (checkpoint_id,),
            ).rowcount
            if updated != 1:
                raise ApprovalGrantConflict("interaction_already_resolved")
            conn.commit()
        created = self.get_grant(str(payload["grant_id"]))
        if created is None:  # pragma: no cover - 同一事务提交后授权必须可见
            raise RuntimeError("approval grant was not persisted")
        return created
