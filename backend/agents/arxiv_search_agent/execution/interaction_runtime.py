from datetime import datetime, timezone
from typing import Any, Dict, Mapping
from uuid import uuid4

from services.storage.sqlite.stores.approval_grants import ApprovalGrantConflict, ApprovalGrantStore
from services.storage.sqlite.stores.agent_runtime_checkpoints import AgentRuntimeCheckpointStore

from .approvals import ApprovalGrant
from .interactions import AgentInteraction, InteractionResumeRequest, SideEffectApprovalPayload
from .background_jobs import PersistentBackgroundWorkCoordinator


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
        background_work_coordinator: PersistentBackgroundWorkCoordinator | None = None,
    ) -> None:
        self.checkpoint_store = checkpoint_store
        self.approval_store = approval_store
        self.background_work_coordinator = background_work_coordinator

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
        # ToolContract 是执行模式唯一事实源；批准入口只按 policy 分流，不按 QA 工具名写特判。
        from ..tool_registry import UNIFIED_TOOL_REGISTRY

        contract = UNIFIED_TOOL_REGISTRY.get_contract(payload.tool_name)
        execution_policy = getattr(contract, "execution_policy", None)
        if execution_policy is not None and execution_policy.mode == "background_job":
            if self.background_work_coordinator is None:
                # 后台能力缺失时不能继续创建普通 grant，否则执行器可能退回同步 adapter。
                raise InteractionRuntimeError("background_execution_unavailable")
            ticket = self.background_work_coordinator.approve_interaction(
                checkpoint=checkpoint,
                interaction=interaction,
                approval_payload=payload,
                user_id=user_id,
                session_id=session_id,
                thread_id=thread_id,
                handler_name=str(execution_policy.handler or ""),
            )
            if ticket.status != "already_satisfied":
                return {
                    "interaction_id": interaction.interaction_id,
                    "kind": interaction.kind,
                    "decision": "background_work_started",
                    "step_id": interaction.step_id,
                    "tool_name": payload.tool_name,
                    "action_type": payload.action_type,
                    "background_work": {
                        "continuation_id": ticket.continuation_id,
                        "job_id": ticket.job_id,
                        "status": ticket.status,
                    },
                }

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
            # step/tool 身份只从后端 checkpoint 中投影，前端不能传入；流式层用它放行刚获批的工具进度事件。
            "step_id": interaction.step_id,
            "tool_name": payload.tool_name,
            "action_type": payload.action_type,
        }
