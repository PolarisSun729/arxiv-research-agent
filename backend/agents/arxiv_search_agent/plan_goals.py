"""规划的目标与目标覆盖检查（设计见 notes/agent-redesign.md D4-2、D4-3 I-3）。

理解层不再只给一个 intent，而是给「要交付什么」：一个或多个 PlanGoal，每个目标对应一个必须产出的
终点类型。规划必须交付全部目标，怎么交付由 LLM 决定。unclear / unsupported 不算目标，不进规划，
由理解层直接追问或回复。

步骤和目标的对应写在 PlanStep.goal_id 上：只标交付步骤，一个目标恰好一个交付步骤，所以两个同类目标
不能共用同一步凑数；中间步骤不标，它们服务哪些目标由数据依赖推出来。

目前只提供目标模型、检查函数和从现有 intent 转换的桥接，还没接入 planner。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schemas import ExecutablePlan, PlanStep
from .semantic_types import SearchSpec
from .state import AgentState
from .tool_registry import UNIFIED_TOOL_REGISTRY, ToolContract, ToolRegistry


GoalKind = Literal["arxiv_search", "paper_qa", "paper_summary", "paper_detail", "recommendation", "preference_action"]
PreferenceAction = Literal["like", "dislike", "remove"]  # 沿用 update_preference_store 现有 action


class GoalSlots(BaseModel):
    """目标的槽位，各 kind 能用哪些见 GOAL_SLOT_FIELDS。"""

    model_config = ConfigDict(extra="forbid")

    # 论文指代（B-3），形如 {"kind": "arxiv_id" | "list_index" | "selected", "value": ...}，
    # 和 project_state_types 的 paper_ref 同一格式；recommendation 用它表示种子论文。
    paper_ref: Optional[Dict[str, Any]] = None
    search_spec: Optional[SearchSpec] = None
    question: Optional[str] = None
    action: Optional[PreferenceAction] = None

    @field_validator("search_spec", mode="before")
    @classmethod
    def _coerce_search_spec(cls, value: Any) -> Any:
        # 测试和轻量运行时可能用不同模块名加载同一份 schema，先转成 dict 避免类身份不一致。
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return value


GOAL_SLOT_FIELDS: Dict[str, FrozenSet[str]] = {
    "arxiv_search": frozenset({"search_spec"}),
    "paper_qa": frozenset({"paper_ref", "question"}),
    "paper_summary": frozenset({"paper_ref", "question"}),
    "paper_detail": frozenset({"paper_ref", "question"}),
    "recommendation": frozenset({"paper_ref"}),
    "preference_action": frozenset({"paper_ref", "action"}),
}


class PlanGoal(BaseModel):
    """理解层给规划的一个目标。和 schemas.Goal（整轮任务的描述）不是一回事。"""

    model_config = ConfigDict(extra="forbid")

    goal_id: str
    kind: GoalKind
    slots: GoalSlots = Field(default_factory=GoalSlots)

    @field_validator("goal_id")
    @classmethod
    def _require_goal_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("goal_id must not be empty")
        return value

    @model_validator(mode="after")
    def _slots_match_kind(self) -> "PlanGoal":
        filled = {name for name in GoalSlots.model_fields if getattr(self.slots, name) is not None}
        extra = sorted(filled - GOAL_SLOT_FIELDS[self.kind])
        if extra:
            raise ValueError(f"{self.kind} goal does not take slots: {extra}")
        return self


@dataclass(frozen=True)
class GoalDeliverable:
    """一个目标必须交付的东西：交付步骤产出的类型，以及交付步骤上游必须有的类型。"""

    semantic_type: str
    answer_kind: Optional[str] = None  # semantic_type 是 AnswerDraft 时，要求的 AnswerDraft.kind
    upstream: Tuple[str, ...] = ()

    def describe(self) -> str:
        return f"{self.semantic_type}（kind={self.answer_kind}）" if self.answer_kind else self.semantic_type


# 目标 kind → 必须交付（D4-2 表）。
GOAL_DELIVERABLES: Dict[str, GoalDeliverable] = {
    "arxiv_search": GoalDeliverable("AnswerDraft", answer_kind="search"),
    "paper_qa": GoalDeliverable("CheckedQAResult"),
    "paper_summary": GoalDeliverable("CheckedQAResult"),
    "paper_detail": GoalDeliverable("CheckedQAResult"),
    "recommendation": GoalDeliverable("AnswerDraft", answer_kind="recommendation"),
    # 只有回复没有写入，就是在告诉用户「已更新」却什么都没写。
    "preference_action": GoalDeliverable("AnswerDraft", answer_kind="preference", upstream=("PreferenceUpdate",)),
}


@dataclass
class GoalCoverageReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    # goal_id -> 交付步骤的 step_id，只含找到唯一交付步骤的目标；供回复合成（D4-6）按目标取段落。
    delivery_step_by_goal: Dict[str, str] = field(default_factory=dict)


def check_goal_coverage(
    plan: Union[ExecutablePlan, Sequence[PlanStep]],
    goals: Sequence[PlanGoal],
    *,
    registry: ToolRegistry = UNIFIED_TOOL_REGISTRY,
) -> GoalCoverageReport:
    """检查每个目标都有自己的交付步骤，且交付的类型对（I-3 目标覆盖）。

    规则：
    1. 每个目标恰好有一步标了它的 goal_id；
    2. 这一步的 produces 包含目标要求的类型，AnswerDraft 还要 answer_kind 一致；
    3. 目标要求上游类型（preference_action 的 PreferenceUpdate）时，交付步骤的依赖链上要有产出它的步骤；
    4. 步骤标的 goal_id 必须是存在的目标。
    数据流本身闭不闭合由 check_plan_closure 负责，这里不重复检查。
    """
    if not goals:
        raise ValueError("check_goal_coverage needs at least one goal; unclear / unsupported should not reach planning")
    goal_ids = [goal.goal_id for goal in goals]
    if len(set(goal_ids)) != len(goal_ids):
        raise ValueError(f"duplicate goal_id in goals: {goal_ids}")

    steps = list(plan.steps or []) if isinstance(plan, ExecutablePlan) else list(plan or [])
    steps_by_id = {step.step_id: step for step in steps}
    errors: List[str] = []
    delivery_step_by_goal: Dict[str, str] = {}

    for step in steps:
        goal_id = _step_goal_id(step)
        if goal_id and goal_id not in goal_ids:
            errors.append(f"{step.step_id}: 标注的目标 {goal_id} 不存在，现有目标：{'、'.join(goal_ids)}")

    for goal in goals:
        deliverable = GOAL_DELIVERABLES[goal.kind]
        label = f"目标 {goal.goal_id}（{goal.kind}）"
        delivery_steps = [step for step in steps if _step_goal_id(step) == goal.goal_id]
        if not delivery_steps:
            errors.append(
                f"{label}: 没有交付步骤，需要一步标注 goal_id={goal.goal_id} 并产出 {deliverable.describe()}，"
                f"每个目标要有自己的交付步骤；{_producer_hint(registry, deliverable.semantic_type, deliverable.answer_kind)}"
            )
            continue
        if len(delivery_steps) > 1:
            step_ids = "、".join(step.step_id for step in delivery_steps)
            errors.append(f"{label}: 有多个交付步骤 {step_ids}，只能保留一个")
            continue
        step = delivery_steps[0]
        problem = _delivery_problem(step, deliverable, registry)
        if problem:
            errors.append(f"{label}: {problem}")
            continue
        for upstream_type in deliverable.upstream:
            if not any(_produces(registry, steps_by_id[ancestor], upstream_type) for ancestor in _ancestors(step, steps_by_id)):
                errors.append(
                    f"{label}: 交付步骤 {step.step_id} 的上游没有产出 {upstream_type} 的步骤；"
                    f"{_producer_hint(registry, upstream_type)}"
                )
        delivery_step_by_goal[goal.goal_id] = step.step_id

    return GoalCoverageReport(ok=not errors, errors=errors, delivery_step_by_goal=delivery_step_by_goal)


def goals_from_state(state: AgentState) -> List[PlanGoal]:
    """临时桥接：把现有理解层的单个 intent + slots 转成目标列表，B-2 改完输出格式后删掉。

    现有理解层只给 intent 和 search_spec，不给论文指代和偏好动作，这两个槽位如实留空：
    论文仍由 state 投影（selected_paper）或 resolve_paper 定位，动作仍由 update_preference_store 从原话判断。
    unclear / unsupported（以及运行时的 confirmation）不算目标，返回空列表。
    """
    intent = str(state.intent or "").strip()
    if intent not in GOAL_DELIVERABLES:
        return []
    if intent == "arxiv_search":
        slots = GoalSlots(search_spec=state.search_spec)
    elif intent in {"paper_qa", "paper_summary", "paper_detail"}:
        slots = GoalSlots(question=str(state.message or "").strip() or None)
    else:
        slots = GoalSlots()
    return [PlanGoal(goal_id="g1", kind=intent, slots=slots)]


def _step_goal_id(step: PlanStep) -> Optional[str]:
    return str(getattr(step, "goal_id", None) or "").strip() or None


def _delivery_problem(step: PlanStep, deliverable: GoalDeliverable, registry: ToolRegistry) -> Optional[str]:
    contract = registry.get_contract(step.tool_name)
    if contract is None:
        return f"交付步骤 {step.step_id} 的工具 {step.tool_name} 未注册"
    if _contract_delivers(contract, deliverable.semantic_type, deliverable.answer_kind):
        return None
    produced = _describe_produces(contract)
    hint = _producer_hint(registry, deliverable.semantic_type, deliverable.answer_kind)
    return f"交付步骤 {step.step_id} {step.tool_name} 产出的是 {produced}，需要 {deliverable.describe()}；{hint}"


def _contract_delivers(contract: ToolContract, semantic_type: str, answer_kind: Optional[str]) -> bool:
    if semantic_type not in contract.produces:
        return False
    return answer_kind is None or contract.answer_kind == answer_kind


def _produces(registry: ToolRegistry, step: PlanStep, semantic_type: str) -> bool:
    contract = registry.get_contract(step.tool_name)
    return contract is not None and semantic_type in contract.produces


def _describe_produces(contract: ToolContract) -> str:
    if not contract.produces:
        return "无语义类型"
    return "、".join(
        f"AnswerDraft（kind={contract.answer_kind}）" if type_name == "AnswerDraft" else type_name
        for type_name in contract.produces
    )


def _producer_hint(registry: ToolRegistry, semantic_type: str, answer_kind: Optional[str] = None) -> str:
    """可产出者提示（D4-4）：列出能产出缺失类型的工具，不替 LLM 选。"""
    target = f"{semantic_type}（kind={answer_kind}）" if answer_kind else semantic_type
    tools = [
        spec.tool_name
        for spec in registry.list_tools()
        if _contract_delivers(registry.get_contract(spec.tool_name), semantic_type, answer_kind)
    ]
    return f"能产出 {target} 的工具：{'、'.join(tools) or '无'}"


def _ancestors(step: PlanStep, steps_by_id: Mapping[str, PlanStep]) -> Set[str]:
    """沿 depends_on 和 step_output 绑定往上找，返回这一步直接或间接依赖的全部步骤。"""
    seen: Set[str] = set()
    pending = list(_direct_dependencies(step))
    while pending:
        step_id = pending.pop()
        if step_id in seen or step_id not in steps_by_id:
            continue
        seen.add(step_id)
        pending.extend(_direct_dependencies(steps_by_id[step_id]))
    return seen


def _direct_dependencies(step: PlanStep) -> List[str]:
    dependencies = [str(step_id or "").strip() for step_id in list(step.depends_on or [])]
    dependencies.extend(
        str(binding.step_id or "").strip()
        for binding in list(step.input_bindings or [])
        if binding.source_type == "step_output"
    )
    return [step_id for step_id in dependencies if step_id]


__all__ = [
    "GOAL_DELIVERABLES",
    "GOAL_SLOT_FIELDS",
    "GoalCoverageReport",
    "GoalDeliverable",
    "GoalKind",
    "GoalSlots",
    "PlanGoal",
    "check_goal_coverage",
    "goals_from_state",
]
