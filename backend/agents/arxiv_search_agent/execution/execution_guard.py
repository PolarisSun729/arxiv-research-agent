from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from services.storage.sqlite.stores.approval_grants import ApprovalGrantStore

from ..schemas import PlanRuntime, PlanStep
from ..state import AgentState
from .approvals import arguments_fingerprint
from .interactions import AgentInteraction, SideEffectApprovalPayload


class GuardDecision(BaseModel):
    """执行守卫的纯决策结果；调用方负责将 interaction 投影到运行态。"""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    action: Literal["execute", "wait_for_interaction"]
    arguments_fingerprint: str
    grant_id: Optional[str] = None
    interaction: Optional[AgentInteraction] = None


class ExecutionGuard:
    """用参数绑定授权守卫副作用工具，不读取旧 step 批准列表或展示镜像。"""

    def __init__(self, approval_store: Optional[ApprovalGrantStore]) -> None:
        self.approval_store = approval_store

    def evaluate(
        self,
        *,
        step: PlanStep,
        runtime: PlanRuntime,
        state: AgentState,
        arguments: Dict[str, Any],
        reason: str,
    ) -> GuardDecision:
        fingerprint = arguments_fingerprint(arguments)
        if not step.confirmation_policy or not step.confirmation_policy.requires_confirmation:
            return GuardDecision(action="execute", arguments_fingerprint=fingerprint)
        plan_id = str((runtime.plan.plan_id if runtime.plan else "") or "").strip()
        user_id = str(state.user_id or "").strip()
        session_id = str(state.session_id or "").strip()
        if self.approval_store is not None and user_id and session_id and plan_id:
            grant = self.approval_store.find_approved_grant(
                user_id=user_id,
                session_id=session_id,
                thread_id=session_id,
                plan_id=plan_id,
                step_id=step.step_id,
                tool_name=step.tool_name,
                arguments_fingerprint=fingerprint,
            )
            if grant is not None:
                return GuardDecision(
                    action="execute",
                    arguments_fingerprint=fingerprint,
                    grant_id=str(grant["grant_id"]),
                )
        now = datetime.now(timezone.utc)
        interaction = AgentInteraction(
            interaction_id=str(uuid4()),
            kind="side_effect_approval",
            plan_id=plan_id,
            step_id=step.step_id,
            payload=SideEffectApprovalPayload(
                tool_name=step.tool_name,
                action_type=step.action_type,
                reason=reason,
                arguments_summary=dict(arguments),
                arguments_fingerprint=fingerprint,
            ),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=10)).isoformat(),
        )
        return GuardDecision(
            action="wait_for_interaction",
            arguments_fingerprint=fingerprint,
            interaction=interaction,
        )
