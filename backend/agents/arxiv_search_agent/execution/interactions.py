from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


InteractionKind = Literal["target_selection", "side_effect_approval"]
InteractionStatus = Literal["pending", "resolved", "cancelled", "expired"]
InteractionDecision = Literal["approve", "reject", "select", "cancel"]


class TargetSelectionPayload(BaseModel):
    """承载目标消歧候选；选择目标只完成输入绑定，不产生副作用授权。"""

    model_config = ConfigDict(extra="forbid")

    candidates: list[Dict[str, Any]] = Field(default_factory=list)
    recommended_candidate_id: str | None = None
    reference_hint: Dict[str, Any] = Field(default_factory=dict)


class SideEffectApprovalPayload(BaseModel):
    """展示即将执行的副作用操作及已完成绑定的参数摘要。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    action_type: str
    reason: str
    arguments_summary: Dict[str, Any] = Field(default_factory=dict)
    arguments_fingerprint: str


class AgentInteraction(BaseModel):
    """Agent 等待用户输入时唯一的结构化业务状态。"""

    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    kind: InteractionKind
    status: InteractionStatus = "pending"
    plan_id: str
    step_id: str
    payload: Union[TargetSelectionPayload, SideEffectApprovalPayload]
    created_at: str
    expires_at: str


class InteractionResumeRequest(BaseModel):
    """恢复请求只使用业务 interaction 身份，禁止客户端指定内部 step 或工具身份。"""

    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    decision: InteractionDecision
    response: Dict[str, Any] = Field(default_factory=dict)
    note: str | None = None
