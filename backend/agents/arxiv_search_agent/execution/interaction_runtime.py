from datetime import datetime, timezone
from typing import Any, Dict, Mapping
from uuid import uuid4

from services.storage.sqlite.stores.approval_grants import ApprovalGrantConflict, ApprovalGrantStore
from services.storage.sqlite.stores.agent_runtime_checkpoints import AgentRuntimeCheckpointStore

from .approvals import ApprovalGrant
from .interactions import AgentInteraction, InteractionResumeRequest, SideEffectApprovalPayload


RUNTIME_CHECKPOINT_SCHEMA_VERSION = 2


class InteractionRuntimeError(RuntimeError):
    """表示业务交互不存在、过期、版本不兼容或已被其他请求解析。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class InteractionRuntimeService:
    """维护 interaction 恢复事务；LangGraph 只在事务成功后接收轻量 resume 结果。"""

    def __init__(
        self,
        *,
        checkpoint_store: AgentRuntimeCheckpointStore,
        approval_store: ApprovalGrantStore,
    ) -> None:
        self.checkpoint_store = checkpoint_store
        self.approval_store = approval_store

    def resolve(
        self,
        *,
        user_id: str,
        session_id: str,
        thread_id: str,
        request: InteractionResumeRequest,
    ) -> Dict[str, Any]:
        checkpoint = self.checkpoint_store.get_agent_runtime_checkpoint(
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
        )
        if checkpoint is None:
            raise InteractionRuntimeError("checkpoint_missing")
        if int(checkpoint.get("schema_version") or 1) != RUNTIME_CHECKPOINT_SCHEMA_VERSION:
            raise InteractionRuntimeError("checkpoint_schema_outdated")
        raw_interaction = checkpoint.get("interaction")
        if not isinstance(raw_interaction, Mapping):
            raise InteractionRuntimeError("interaction_missing")
        try:
            interaction = AgentInteraction.model_validate(raw_interaction)
        except Exception as exc:
            raise InteractionRuntimeError("interaction_invalid") from exc
        if interaction.interaction_id != request.interaction_id:
            raise InteractionRuntimeError("interaction_id_mismatch")
        if interaction.status != "pending":
            raise InteractionRuntimeError("interaction_not_pending")

        if interaction.kind == "target_selection":
            if request.decision not in {"select", "cancel"}:
                raise InteractionRuntimeError("interaction_decision_invalid")
            selected_candidate = None
            if request.decision == "select":
                candidate_id = str(request.response.get("candidate_id") or "").strip()
                candidates = getattr(interaction.payload, "candidates", [])
                for candidate in list(candidates or []):
                    payload = candidate if isinstance(candidate, Mapping) else candidate.model_dump(mode="json")
                    identities = {
                        str(payload.get(key) or "").strip()
                        for key in ("candidate_id", "arxiv_id", "paper_id", "id")
                        if payload.get(key)
                    }
                    if candidate_id and candidate_id in identities:
                        selected_candidate = dict(payload)
                        break
                if selected_candidate is None:
                    raise InteractionRuntimeError("target_candidate_invalid")
            # 目标选择不创建授权，但必须清除交互，防止刷新或重复点击再次消费同一选择。
            if not self.checkpoint_store.resolve_agent_interaction(
                user_id=user_id,
                session_id=session_id,
                thread_id=thread_id,
                interaction_id=interaction.interaction_id,
                terminal_status="running" if request.decision == "select" else "cancelled",
            ):
                raise InteractionRuntimeError("interaction_already_resolved")
            return {
                "interaction_id": interaction.interaction_id,
                "kind": interaction.kind,
                "decision": request.decision,
                "response": dict(request.response),
                "selected_candidate": selected_candidate,
            }

        if request.decision == "reject":
            if not self.checkpoint_store.resolve_agent_interaction(
                user_id=user_id,
                session_id=session_id,
                thread_id=thread_id,
                interaction_id=interaction.interaction_id,
                terminal_status="cancelled",
            ):
                raise InteractionRuntimeError("interaction_already_resolved")
            return {
                "interaction_id": interaction.interaction_id,
                "kind": interaction.kind,
                "decision": "reject",
            }
        if request.decision != "approve":
            raise InteractionRuntimeError("interaction_decision_invalid")
        payload = SideEffectApprovalPayload.model_validate(interaction.payload)
        grant = ApprovalGrant(
            grant_id=str(uuid4()),
            interaction_id=interaction.interaction_id,
            user_id=user_id,
            session_id=session_id,
            thread_id=thread_id,
            plan_id=interaction.plan_id,
            step_id=interaction.step_id,
            tool_name=payload.tool_name,
            arguments_fingerprint=payload.arguments_fingerprint,
            approved_at=datetime.now(timezone.utc).isoformat(),
        )
        checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        try:
            self.approval_store.resolve_side_effect_interaction(checkpoint_id=checkpoint_id, grant=grant)
        except ApprovalGrantConflict as exc:
            raise InteractionRuntimeError(str(exc)) from exc
        return {
            "interaction_id": interaction.interaction_id,
            "kind": interaction.kind,
            "decision": "approve",
            "grant_id": grant.grant_id,
        }
