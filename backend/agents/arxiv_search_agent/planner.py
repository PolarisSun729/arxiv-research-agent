"""Planner 入口与各类 Goal/Plan builder。

这里把“目标提取”和“计划生成”拆开，避免继续在单个大函数里堆所有 intent 分支。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from .plan_validator import PlanValidator
from .schemas import ExecutablePlan, Goal, PlanRuntime, PlanStep, StepCondition, StepInputBinding, StepPolicy
from .state import AgentState
from .tool_aware_planner import LLMPlanDraftGenerator, PlanDraftConverter, PlanDraftPlanningError, RuleBasedToolAwarePlanBuilder, ToolCandidateSelector
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry
from .utils.state_utils import _compact_search_spec


SUPPORTED_GOAL_TYPES = {
    "arxiv_search",
    "paper_detail",
    "paper_summary",
    "paper_qa",
    "recommendation",
    "preference_action",
    "unclear",
    "unsupported",
}


def _get_agent_planner_runtime_config() -> Dict[str, Any]:
    try:
        from utils.config import get_agent_planner_runtime_config

        return dict(get_agent_planner_runtime_config())
    except Exception:
        return {
            "enable_tool_aware_planner": True,
            "enable_llm_plan_draft": False,
            "llm_plan_timeout": 8,
            "llm_plan_max_steps": 8,
            "llm_plan_fallback_to_rule": True,
            "llm_plan_fallback_to_template": True,
            "expose_planner_debug": True,
        }


def _resolve_generation_service() -> Any:
    try:
        from dependencies import get_generation_service

        return get_generation_service()
    except Exception:
        return None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_intent(value: Any) -> str:
    normalized = _normalize_text(value) or "unsupported"
    # 未注册目标统一降级为 unsupported，避免历史调用方传入旧 intent 时触发规划异常。
    if normalized not in SUPPORTED_GOAL_TYPES:
        return "unsupported"
    return normalized


def _dedupe_strings(items: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for item in items:
        text = _normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def _get_context_mapping(state: AgentState) -> Mapping[str, Any]:
    return state.context if isinstance(state.context, Mapping) else {}


def _get_user_memory_summary(context: Mapping[str, Any]) -> Optional[str]:
    for key in ("user_memory_summary", "memory_summary"):
        value = context.get(key)
        if isinstance(value, Mapping):
            return key
        text = _normalize_text(value)
        if text:
            return key
    return None


def _get_selected_paper_hint(context: Mapping[str, Any]) -> Optional[str]:
    selected_paper = context.get("selected_paper")
    if isinstance(selected_paper, Mapping):
        for key in ("title", "arxiv_id", "paper_id"):
            text = _normalize_text(selected_paper.get(key))
            if text:
                return text
    for key in ("selected_paper_title", "selected_paper_id"):
        text = _normalize_text(context.get(key))
        if text:
            return text
    return None


def _build_context_refs(state: AgentState) -> List[str]:
    context = _get_context_mapping(state)
    refs: List[str] = []
    if _get_selected_paper_hint(context):
        refs.append("selected_paper")
    if _get_user_memory_summary(context):
        refs.append("user_memory_summary")
    if isinstance(state.pending_action, Mapping):
        refs.append("pending_action")
    if isinstance(state.paper_qa_result, Mapping):
        refs.append("paper_qa_result")
    refs.extend(sorted(str(key) for key in context.keys()))
    return _dedupe_strings(refs)


def _build_search_constraints(state: AgentState) -> List[str]:
    constraints: List[str] = []
    spec = state.search_spec
    if spec is not None:
        if spec.query:
            constraints.append(f"query={spec.query}")
        if spec.title_query:
            constraints.append(f"title_query={spec.title_query}")
        if spec.abstract_query:
            constraints.append(f"abstract_query={spec.abstract_query}")
        if spec.categories:
            constraints.append(f"categories={', '.join(spec.categories)}")
        if spec.submitted_days_ago is not None:
            constraints.append(f"submitted_days_ago={spec.submitted_days_ago}")
        constraints.append(f"max_results={spec.max_results}")
        constraints.append(f"sort_by={spec.sort_by}:{spec.sort_order}")
    return _dedupe_strings(constraints)


def _binding(
    input_key: str,
    *,
    source_type: str,
    source_key: Optional[str] = None,
    step_id: Optional[str] = None,
    value: Any = None,
    required: bool = True,
) -> StepInputBinding:
    return StepInputBinding(
        input_key=input_key,
        source_type=source_type,  # type: ignore[arg-type]
        source_key=source_key,
        step_id=step_id,
        value=value,
        required=required,
    )


def _always_condition() -> StepCondition:
    return StepCondition(condition_type="always")


def _build_plan_step(
    *,
    step_id: str,
    action_type: str,
    tool_name: str,
    output_key: Optional[str],
    tool_registry: ToolRegistry,
    input_bindings: Optional[List[StepInputBinding]] = None,
    depends_on: Optional[List[str]] = None,
    condition: Optional[StepCondition] = None,
    preconditions: Optional[List[StepCondition]] = None,
    postconditions: Optional[List[StepCondition]] = None,
    retry_policy: Optional[StepPolicy] = None,
    failure_policy: Optional[StepPolicy] = None,
    confirmation_policy: Optional[StepPolicy] = None,
    side_effect_level: Optional[str] = None,
    status: str = "pending",
) -> PlanStep:
    normalized_tool_name = tool_registry.validate_tool_name(tool_name)
    tool = tool_registry.get(normalized_tool_name)
    if tool is None:
        raise ValueError(f"Unknown planner tool: {tool_name}")
    resolved_side_effect_level = side_effect_level or tool.side_effect_level
    if tool.requires_confirmation and confirmation_policy is None:
        confirmation_policy = StepPolicy(
            policy_type="confirmation",
            mode="explicit_user_confirmation_required",
            requires_confirmation=True,
            note=f"{normalized_tool_name} requires confirmation by tool policy",
        )
    return PlanStep(
        step_id=step_id,
        action_type=action_type,
        tool_name=normalized_tool_name,
        tool=tool,
        input_bindings=input_bindings or [],
        output_key=output_key,
        depends_on=depends_on or [],
        condition=condition or _always_condition(),
        preconditions=preconditions or [],
        postconditions=postconditions or [],
        retry_policy=retry_policy,
        failure_policy=failure_policy,
        confirmation_policy=confirmation_policy,
        side_effect_level=resolved_side_effect_level,  # type: ignore[arg-type]
        status=status,
    )


def _make_plan(goal: Goal, *, steps: List[PlanStep]) -> ExecutablePlan:
    step_ids = [step.step_id for step in steps]
    depended_ids = {dependency for step in steps for dependency in list(step.depends_on or [])}
    entry_step_ids = [step.step_id for step in steps if not step.depends_on]
    final_step_ids = [step_id for step_id in step_ids if step_id not in depended_ids]
    return ExecutablePlan(
        plan_id=f"{goal.goal_type}:{datetime.utcnow().isoformat(timespec='seconds')}",
        goal=goal,
        steps=steps,
        entry_step_ids=entry_step_ids,
        final_step_ids=final_step_ids,
        metadata={
            "planner_version": "step3-builder-registry-plan",
            "built_at": datetime.utcnow().isoformat(timespec="seconds"),
            "goal_type": goal.goal_type,
            "goal_id": goal.goal_id,
            # 计划只记录 contract 来源摘要；具体 adapter 函数不进入可序列化 plan。
            "tool_contract_source": _tool_contract_source_summary(steps),
        },
    )


def build_plan_runtime(state: AgentState, *, goal: Goal, plan: ExecutablePlan, turn_status: str) -> PlanRuntime:
    return PlanRuntime(
        state={
            "intent": state.intent,
            "message": state.message,
            "context_keys": sorted(_get_context_mapping(state).keys()),
            "search_spec": _compact_search_spec(state.search_spec),
        },
        goal=goal,
        plan=plan,
        outputs={},
        step_status={step.step_id: step.status for step in list(plan.steps or [])},
        trace=[],
        retry_counts={},
        replan_counts={},
        # PlanRuntime.pending_confirmation 只保存标准 ConfirmationRequest。
        # state.pending_action 是前端兼容展示镜像，不能反向当成可恢复的执行现场。
        pending_confirmation=None,
        final_answer=state.answer,
        turn_status=turn_status,  # type: ignore[arg-type]
    )


class GoalBuilder:
    """只负责从 state 提取目标，不生成执行步骤。"""

    @classmethod
    def from_state(cls, state: AgentState) -> Goal:
        intent = _normalize_intent(state.intent)
        context = _get_context_mapping(state)
        message = _normalize_text(state.message) or ""
        risk_level = "low"
        if intent == "preference_action":
            risk_level = "high"
        elif intent in {"paper_summary", "paper_detail", "paper_qa", "recommendation"}:
            risk_level = "medium"

        constraints = _build_search_constraints(state) if intent == "arxiv_search" else []
        if intent in {"paper_summary", "paper_detail", "paper_qa"}:
            constraints = _dedupe_strings([f"target_paper={_get_selected_paper_hint(context)}", "use existing QA index when available"])
        if intent == "unclear":
            constraints = ["do not execute business tools before clarification"]
        if intent == "unsupported":
            constraints = ["do not invoke unsupported business tools"]

        success_criteria_map = {
            "arxiv_search": [
                "Extract a usable search target from the request.",
                "Return relevant arXiv papers or explain why no suitable results were found.",
                "Apply personalization when profile or memory context is available.",
                "Generate a concise response with next actions when helpful.",
            ],
            "paper_summary": [
                "Resolve the requested paper.",
                "Retrieve relevant paper evidence.",
                "Produce an answer grounded in retrieved evidence.",
            ],
            "paper_detail": [
                "Resolve the requested paper.",
                "Retrieve relevant paper evidence.",
                "Produce an answer grounded in retrieved evidence.",
            ],
            "paper_qa": [
                "Resolve the requested paper.",
                "Retrieve relevant paper evidence.",
                "Produce an answer grounded in retrieved evidence.",
            ],
            "recommendation": [
                "Read user profile and candidate paper context.",
                "Generate relevant recommendations.",
                "Explain the recommendation rationale.",
            ],
            "preference_action": [
                "Resolve the preference target.",
                "Persist the preference update.",
                "Return a clear user-facing outcome.",
            ],
            "unclear": [
                "Identify missing information.",
                "Ask a focused clarification question.",
            ],
            "unsupported": [
                "Explain why the request is unsupported.",
                "Return a safe fallback response.",
            ],
        }
        goal_type = "paper_qa" if intent in {"paper_summary", "paper_detail", "paper_qa"} else intent
        return Goal(
            goal_id=f"{goal_type}:{datetime.utcnow().isoformat(timespec='seconds')}",
            goal_type=goal_type,
            user_request=message,
            intent=intent,
            constraints=_dedupe_strings(constraints),
            success_criteria=_dedupe_strings(success_criteria_map.get(intent, success_criteria_map.get(goal_type, []))),
            context_refs=_build_context_refs(state),
            risk_level=risk_level,  # type: ignore[arg-type]
            # 兼容现有响应读取。
            user_goal=message,
            task_scope=goal_type,
        )


class PlanBuilder(Protocol):
    def can_handle(self, goal: Goal) -> bool:
        ...

    def build(self, goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> ExecutablePlan:
        ...


class ArxivSearchPlanBuilder:
    def can_handle(self, goal: Goal) -> bool:
        return goal.goal_type == "arxiv_search"

    def build(self, goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> ExecutablePlan:
        steps = [
            _build_plan_step(
                step_id="normalize_request",
                action_type="write_state",
                tool_name="normalize_request",
                output_key="normalized_request",
                tool_registry=tool_registry,
                input_bindings=[
                    _binding("intent", source_type="state", source_key="intent"),
                    _binding("message", source_type="state", source_key="message"),
                    _binding("search_spec", source_type="search_spec"),
                ],
            ),
            _build_plan_step(
                step_id="build_arxiv_search_spec",
                action_type="search",
                tool_name="build_arxiv_search_spec",
                output_key="search_spec",
                tool_registry=tool_registry,
                input_bindings=[_binding("normalized_request", source_type="step_output", step_id="normalize_request")],
                depends_on=["normalize_request"],
            ),
            _build_plan_step(
                step_id="search_arxiv",
                action_type="search",
                tool_name="search_arxiv",
                output_key="arxiv_results",
                tool_registry=tool_registry,
                input_bindings=[_binding("search_spec", source_type="step_output", step_id="build_arxiv_search_spec")],
                depends_on=["build_arxiv_search_spec"],
                retry_policy=StepPolicy(policy_type="retry", mode="allow_search_relaxation", max_attempts=3),
            ),
            _build_plan_step(
                step_id="validate_arxiv_results",
                action_type="validate",
                tool_name="validate_arxiv_results",
                output_key="arxiv_result_quality",
                tool_registry=tool_registry,
                input_bindings=[_binding("arxiv_results", source_type="step_output", step_id="search_arxiv")],
                depends_on=["search_arxiv"],
            ),
            _build_plan_step(
                step_id="personalize_paper_results",
                action_type="rerank",
                tool_name="personalize_paper_results",
                output_key="ranked_papers",
                tool_registry=tool_registry,
                input_bindings=[
                    _binding("arxiv_results", source_type="step_output", step_id="search_arxiv"),
                    _binding("user_memory_summary", source_type="context", source_key="user_memory_summary", required=False),
                    _binding("research_profile", source_type="context", source_key="research_profile", required=False),
                ],
                depends_on=["search_arxiv"],
            ),
            _build_plan_step(
                step_id="synthesize_arxiv_response",
                action_type="answer",
                tool_name="synthesize_arxiv_response",
                output_key="final_answer",
                tool_registry=tool_registry,
                input_bindings=[
                    _binding("ranked_papers", source_type="step_output", step_id="personalize_paper_results", required=False),
                    _binding("arxiv_result_quality", source_type="step_output", step_id="validate_arxiv_results"),
                ],
                depends_on=["validate_arxiv_results", "personalize_paper_results"],
            ),
        ]
        return _make_plan(goal, steps=steps)


class PaperQAPlanBuilder:
    def can_handle(self, goal: Goal) -> bool:
        return goal.goal_type == "paper_qa"

    def build(self, goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> ExecutablePlan:
        steps = [
            _build_plan_step(
                step_id="resolve_paper",
                action_type="retrieve",
                tool_name="resolve_paper",
                output_key="paper_ref",
                tool_registry=tool_registry,
                # 目标论文解析既要看当前选中论文，也要看 last_papers；
                # 显式绑定完整 context，避免“第二篇”这类序号引用退回到默认 selected_paper。
                input_bindings=[
                    _binding("message", source_type="state", source_key="message"),
                    _binding("selected_paper", source_type="context", source_key="selected_paper", required=False),
                    _binding("context", source_type="state", source_key="context", required=False),
                ],
            ),
            _build_plan_step(step_id="check_paper_index", action_type="validate", tool_name="check_paper_index", output_key="paper_index_status", tool_registry=tool_registry, input_bindings=[_binding("paper_ref", source_type="step_output", step_id="resolve_paper")], depends_on=["resolve_paper"]),
            # Agent 只负责调度真实 PaperQA 工具；检索、重写、rerank 和 grounding 校验均由 PaperQAService 内部完成。
            _build_plan_step(step_id="answer_paper_question", action_type="answer", tool_name="answer_paper_question", output_key="paper_qa_result", tool_registry=tool_registry, input_bindings=[_binding("paper_ref", source_type="step_output", step_id="resolve_paper"), _binding("message", source_type="state", source_key="message")], depends_on=["resolve_paper", "check_paper_index"]),
        ]
        return _make_plan(goal, steps=steps)


class RecommendationPlanBuilder:
    def can_handle(self, goal: Goal) -> bool:
        return goal.goal_type == "recommendation"

    def build(self, goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> ExecutablePlan:
        steps = [
            _build_plan_step(step_id="load_user_profile", action_type="retrieve", tool_name="load_user_profile", output_key="recommendation_profile", tool_registry=tool_registry, input_bindings=[_binding("context", source_type="state", source_key="context", required=False)]),
            _build_plan_step(step_id="load_candidate_papers", action_type="retrieve", tool_name="load_candidate_papers", output_key="candidate_papers", tool_registry=tool_registry, input_bindings=[_binding("recommendation_profile", source_type="step_output", step_id="load_user_profile")], depends_on=["load_user_profile"]),
            _build_plan_step(step_id="generate_recommendations", action_type="search", tool_name="generate_recommendations", output_key="recommendation_result", tool_registry=tool_registry, input_bindings=[_binding("recommendation_profile", source_type="step_output", step_id="load_user_profile"), _binding("candidate_papers", source_type="step_output", step_id="load_candidate_papers", required=False)], depends_on=["load_user_profile", "load_candidate_papers"]),
            _build_plan_step(step_id="validate_recommendations", action_type="validate", tool_name="validate_recommendations", output_key="validated_recommendations", tool_registry=tool_registry, input_bindings=[_binding("recommendation_result", source_type="step_output", step_id="generate_recommendations")], depends_on=["generate_recommendations"]),
            _build_plan_step(step_id="explain_recommendations", action_type="answer", tool_name="explain_recommendations", output_key="final_answer", tool_registry=tool_registry, input_bindings=[_binding("validated_recommendations", source_type="step_output", step_id="validate_recommendations")], depends_on=["validate_recommendations"]),
        ]
        return _make_plan(goal, steps=steps)


class PreferenceActionPlanBuilder:
    def can_handle(self, goal: Goal) -> bool:
        return goal.goal_type == "preference_action"

    def build(self, goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> ExecutablePlan:
        # 偏好计划只声明真实发生的持久化写入；兴趣画像/向量重建应走显式推荐接口，
        # 不能在 Agent 里追加没有实际同步实现的“成功步骤”。
        steps = [
            _build_plan_step(
                step_id="resolve_preference_target",
                action_type="retrieve",
                tool_name="resolve_preference_target",
                output_key="paper_reference",
                tool_registry=tool_registry,
                # 偏好写入同样支持“第一篇/第二篇”，必须让解析器拿到最近论文列表。
                input_bindings=[
                    _binding("message", source_type="state", source_key="message"),
                    _binding("selected_paper", source_type="context", source_key="selected_paper", required=False),
                    _binding("context", source_type="state", source_key="context", required=False),
                ],
            ),
            _build_plan_step(step_id="update_preference_store", action_type="write_state", tool_name="update_preference_store", output_key="preference_action_result", tool_registry=tool_registry, input_bindings=[_binding("paper_reference", source_type="step_output", step_id="resolve_preference_target"), _binding("message", source_type="state", source_key="message")], depends_on=["resolve_preference_target"], side_effect_level="persistent_write"),
            _build_plan_step(step_id="verify_preference_update", action_type="validate", tool_name="verify_preference_update", output_key="verified_preference_update", tool_registry=tool_registry, input_bindings=[_binding("preference_action_result", source_type="step_output", step_id="update_preference_store")], depends_on=["update_preference_store"]),
            _build_plan_step(step_id="synthesize_preference_response", action_type="answer", tool_name="synthesize_preference_response", output_key="final_answer", tool_registry=tool_registry, input_bindings=[_binding("verified_preference_update", source_type="step_output", step_id="verify_preference_update")], depends_on=["verify_preference_update"]),
        ]
        return _make_plan(goal, steps=steps)


class ClarificationPlanBuilder:
    def can_handle(self, goal: Goal) -> bool:
        return goal.goal_type == "unclear"

    def build(self, goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> ExecutablePlan:
        steps = [
            _build_plan_step(step_id="analyze_ambiguity", action_type="clarify", tool_name="analyze_ambiguity", output_key="missing_information", tool_registry=tool_registry, input_bindings=[_binding("message", source_type="state", source_key="message")]),
            _build_plan_step(step_id="generate_clarification", action_type="answer", tool_name="generate_clarification", output_key="final_answer", tool_registry=tool_registry, input_bindings=[_binding("missing_information", source_type="step_output", step_id="analyze_ambiguity")], depends_on=["analyze_ambiguity"]),
        ]
        return _make_plan(goal, steps=steps)


class UnsupportedPlanBuilder:
    def can_handle(self, goal: Goal) -> bool:
        return goal.goal_type == "unsupported"

    def build(self, goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> ExecutablePlan:
        steps = [
            _build_plan_step(step_id="generate_fallback_response", action_type="answer", tool_name="generate_fallback_response", output_key="final_answer", tool_registry=tool_registry, input_bindings=[_binding("message", source_type="state", source_key="message")]),
        ]
        return _make_plan(goal, steps=steps)


class PlanBuilderRegistry:
    """根据 goal_type 分发到对应 PlanBuilder。"""

    def __init__(self) -> None:
        self._builders: List[PlanBuilder] = []

    def register(self, builder: PlanBuilder) -> None:
        self._builders.append(builder)

    def get(self, goal_type: str) -> PlanBuilder:
        normalized_goal_type = _normalize_intent(goal_type)
        for builder in self._builders:
            if builder.can_handle(Goal(goal_type=normalized_goal_type)):
                return builder
        raise ValueError(f"No plan builder registered for goal_type={goal_type}")


PLAN_BUILDER_REGISTRY = PlanBuilderRegistry()
for builder in [
    ArxivSearchPlanBuilder(),
    PaperQAPlanBuilder(),
    RecommendationPlanBuilder(),
    PreferenceActionPlanBuilder(),
    ClarificationPlanBuilder(),
    UnsupportedPlanBuilder(),
]:
    PLAN_BUILDER_REGISTRY.register(builder)


def _build_fixed_template_plan(goal: Goal, state: AgentState, tool_registry: ToolRegistry) -> tuple[ExecutablePlan, PlanBuilder]:
    builder = PLAN_BUILDER_REGISTRY.get(goal.goal_type or "unsupported")
    plan = builder.build(goal, state, tool_registry)
    return plan, builder


def _fixed_template_debug(goal: Goal, plan: ExecutablePlan, builder: PlanBuilder, *, planner_mode: str, tool_registry: ToolRegistry) -> Dict[str, Any]:
    return {
        "planner_mode": planner_mode,
        "goal": goal.model_dump(),
        "candidate_tools": [],
        "excluded_tools": [],
        "draft_steps": [],
        "selected_steps": _debug_steps_from_plan(plan),
        "skipped_steps": [],
        "skipped_tools": [],
        "llm_plan_attempted": False,
        "llm_plan_valid": False,
        "llm_plan_invalid_reasons": [],
        "rule_based_fallback_used": False,
        "template_fallback_used": False,
        "validation_status": "passed",
        "fallback_used": False,
        "fallback_reason": None,
        "final_plan_source": "fixed_template",
        "selected_plan_source": "fixed_template",
        "tool_risk_summary": {},
        "tool_contract_source": _tool_contract_source_summary(plan.steps),
        "tool_contract_matrix": tool_registry.tool_contract_matrix(),
        "confirmation_required_steps": _confirmation_required_step_ids(plan),
        "plan_builder": builder.__class__.__name__,
        "execution_plan": plan.model_dump(),
    }


def _is_tool_aware_planner_enabled(enable_tool_aware_planner: Optional[bool]) -> bool:
    if enable_tool_aware_planner is not None:
        return bool(enable_tool_aware_planner)
    return bool(_get_agent_planner_runtime_config().get("enable_tool_aware_planner", False))


def _debug_steps_from_plan(plan: ExecutablePlan) -> List[Dict[str, Any]]:
    return [
        {
            "step_id": step.step_id,
            "tool_name": step.tool_name,
            "contract_source": step.tool.contract_source,
            "adapter": step.tool.adapter,
            "backend_tool_name": step.tool.backend_tool_name,
            "side_effect_level": step.side_effect_level,
            "requires_confirmation": bool(step.confirmation_policy and step.confirmation_policy.requires_confirmation),
        }
        for step in list(plan.steps or [])
    ]


def _tool_contract_source_summary(steps: Sequence[PlanStep]) -> Dict[str, Any]:
    """在 planner debug 中显式暴露工具来源，确认计划与执行读取同一份 contract。"""
    return {
        step.tool_name: {
            "contract_source": step.tool.contract_source,
            "adapter": step.tool.adapter,
            "backend_tool_name": step.tool.backend_tool_name,
        }
        for step in list(steps or [])
    }


def _confirmation_required_step_ids(plan: ExecutablePlan) -> List[str]:
    return [
        step.step_id
        for step in list(plan.steps or [])
        if bool(step.confirmation_policy and step.confirmation_policy.requires_confirmation)
    ]


def _fallback_to_template_or_unsupported(
    goal: Goal,
    state: AgentState,
    tool_registry: ToolRegistry,
    planning_debug: Dict[str, Any],
    *,
    reason: str,
    allow_template: bool,
) -> tuple[Goal, ExecutablePlan, Dict[str, Any]]:
    if allow_template:
        try:
            plan, builder = _build_fixed_template_plan(goal, state, tool_registry)
            # 固定模板是最后业务兜底，也必须经过 Validator，不能因为 fallback 就绕过安全边界。
            PlanValidator().validate(plan, tool_registry)
            planning_debug.update(
                {
                    "validation_status": "failed",
                    "fallback_used": True,
                    "fallback_reason": reason,
                    "template_fallback_used": True,
                    "final_plan_source": "fixed_template_fallback",
                    "selected_plan_source": "fixed_template_fallback",
                    "plan_builder": builder.__class__.__name__,
                    "tool_contract_source": _tool_contract_source_summary(plan.steps),
                    "tool_contract_matrix": tool_registry.tool_contract_matrix(),
                    "confirmation_required_steps": _confirmation_required_step_ids(plan),
                    "execution_plan": plan.model_dump(),
                }
            )
            return goal, plan, planning_debug
        except Exception as template_exc:
            reason = f"{reason}; template fallback failed: {template_exc}"

    unsupported_goal = goal.model_copy(update={"goal_type": "unsupported", "intent": "unsupported", "risk_level": "low"})
    plan, builder = _build_fixed_template_plan(unsupported_goal, state.model_copy(update={"intent": "unsupported"}), tool_registry)
    PlanValidator().validate(plan, tool_registry)
    planning_debug.update(
        {
            "validation_status": "failed",
            "fallback_used": True,
            "fallback_reason": reason,
            "template_fallback_used": bool(allow_template),
            "final_plan_source": "unsupported_fallback",
            "selected_plan_source": "unsupported_fallback",
            "plan_builder": builder.__class__.__name__,
            "tool_contract_source": _tool_contract_source_summary(plan.steps),
            "tool_contract_matrix": tool_registry.tool_contract_matrix(),
            "confirmation_required_steps": _confirmation_required_step_ids(plan),
            "execution_plan": plan.model_dump(),
        }
    )
    return unsupported_goal, plan, planning_debug


def build_executable_plan(
    state: AgentState,
    tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY,
    *,
    enable_tool_aware_planner: Optional[bool] = None,
    enable_llm_plan_draft: Optional[bool] = None,
    llm_generation_service: Any = None,
) -> tuple[Goal, ExecutablePlan, Dict[str, Any]]:
    """统一 planner 入口：可选启用 Tool-Aware 草稿层，最终始终产出已校验 ExecutablePlan。"""
    goal = GoalBuilder.from_state(state)
    return build_executable_plan_for_goal(
        goal,
        state,
        tool_registry=tool_registry,
        enable_tool_aware_planner=enable_tool_aware_planner,
        enable_llm_plan_draft=enable_llm_plan_draft,
        llm_generation_service=llm_generation_service,
    )


def build_executable_plan_for_goal(
    goal: Goal,
    state: AgentState,
    tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY,
    *,
    enable_tool_aware_planner: Optional[bool] = None,
    enable_llm_plan_draft: Optional[bool] = None,
    llm_generation_service: Any = None,
) -> tuple[Goal, ExecutablePlan, Dict[str, Any]]:
    """基于已构建 Goal 生成计划，供显式 LangGraph planning 节点复用。

    build_goal 节点已经把目标作为一等状态写入 AgentState；planning 节点应消费该目标，
    而不是再次从 state 推断，避免图上 goal 节点变成只做展示的空节点。
    """

    if not _is_tool_aware_planner_enabled(enable_tool_aware_planner):
        plan, builder = _build_fixed_template_plan(goal, state, tool_registry)
        # 固定模板路径保持原行为，但仍通过统一校验器守住 Executor 入口边界。
        PlanValidator().validate(plan, tool_registry)
        return goal, plan, _fixed_template_debug(goal, plan, builder, planner_mode="fixed_template", tool_registry=tool_registry)

    planner_config = _get_agent_planner_runtime_config()
    llm_enabled = bool(planner_config.get("enable_llm_plan_draft", False)) if enable_llm_plan_draft is None else bool(enable_llm_plan_draft)
    llm_fallback_to_rule = bool(planner_config.get("llm_plan_fallback_to_rule", True))
    llm_fallback_to_template = bool(planner_config.get("llm_plan_fallback_to_template", True))
    expose_planner_debug = bool(planner_config.get("expose_planner_debug", True))

    selector = ToolCandidateSelector(tool_registry)
    selection = selector.select(goal, state)
    candidate_tool_names = [tool.tool_name for tool in list(selection.candidate_tools or [])]
    planning_debug: Dict[str, Any] = {
        "planner_mode": "tool_aware_llm" if llm_enabled else "tool_aware_rule_based",
        "goal": goal.model_dump(),
        "candidate_tools": [tool.model_dump() for tool in list(selection.candidate_tools or [])],
        "excluded_tools": [tool.model_dump() for tool in list(selection.excluded_tools or [])],
        "draft_steps": [],
        "selected_steps": [],
        "skipped_steps": [],
        "skipped_tools": [],
        "llm_plan_attempted": False,
        "llm_plan_valid": False,
        "llm_plan_invalid_reasons": [],
        "rule_based_fallback_used": False,
        "template_fallback_used": False,
        "validation_status": "not_started",
        "fallback_used": False,
        "fallback_reason": None,
        "final_plan_source": None,
        "selected_plan_source": None,
        "tool_selection": selection.model_dump(),
        "tool_risk_summary": dict(selection.risk_summary or {}),
        "tool_contract_source": {},
        "tool_contract_matrix": tool_registry.tool_contract_matrix(),
        "confirmation_required_steps": [],
    }

    if llm_enabled:
        planning_debug["llm_plan_attempted"] = True
        try:
            llm_generator = LLMPlanDraftGenerator(
                llm_generation_service if llm_generation_service is not None else _resolve_generation_service(),
                max_steps=int(planner_config.get("llm_plan_max_steps", 8) or 8),
                timeout_seconds=int(planner_config.get("llm_plan_timeout", 8) or 8),
            )
            draft = llm_generator.generate(goal, state, selection.candidate_tools, tool_registry)
            if expose_planner_debug:
                planning_debug["raw_llm_plan"] = llm_generator.last_debug.get("raw_llm_plan")
            planning_debug["draft_steps"] = [step.model_dump() for step in list(draft.steps or [])]
            plan = PlanDraftConverter(tool_registry).convert(draft, goal, allowed_tool_names=candidate_tool_names)
            PlanValidator().validate(plan, tool_registry)
            planning_debug.update(
                {
                    "llm_plan_valid": True,
                    "validation_status": "passed",
                    "final_plan_source": "llm_tool_aware",
                    "selected_plan_source": "llm_tool_aware",
                    "selected_steps": _debug_steps_from_plan(plan),
                    "tool_contract_source": _tool_contract_source_summary(plan.steps),
                    "confirmation_required_steps": _confirmation_required_step_ids(plan),
                    "execution_plan": plan.model_dump(),
                }
            )
            return goal, plan, planning_debug
        except Exception as exc:
            invalid_reason = str(exc)
            planning_debug["llm_plan_valid"] = False
            planning_debug["llm_plan_invalid_reasons"] = [invalid_reason]
            planning_debug["fallback_reason"] = invalid_reason
            if not llm_fallback_to_rule:
                return _fallback_to_template_or_unsupported(
                    goal,
                    state,
                    tool_registry,
                    planning_debug,
                    reason=invalid_reason,
                    allow_template=llm_fallback_to_template,
                )

    try:
        rule_builder = RuleBasedToolAwarePlanBuilder()
        draft = rule_builder.build(goal, state, selection.candidate_tools, tool_registry)
        planning_debug.update(
            {
                "selected_steps": list(rule_builder.last_debug.get("selected_steps") or []),
                "skipped_steps": list(rule_builder.last_debug.get("skipped_steps") or []),
                "skipped_tools": list(rule_builder.last_debug.get("skipped_tools") or []),
                "rule_builder": rule_builder.__class__.__name__,
                "rule_based_fallback_used": bool(llm_enabled),
            }
        )
        if draft.fallback_reason:
            # 规则型 builder 可以显式声明“不要执行这个草稿”，统一交给固定模板兜底。
            raise PlanDraftPlanningError(draft.fallback_reason)
        planning_debug["draft_steps"] = [step.model_dump() for step in list(draft.steps or [])]
        plan = PlanDraftConverter(tool_registry).convert(draft, goal, allowed_tool_names=candidate_tool_names)
        PlanValidator().validate(plan, tool_registry)
        planning_debug.update(
            {
                "validation_status": "passed",
                "final_plan_source": "tool_aware_rule_based",
                "selected_plan_source": "tool_aware_rule_based",
                "selected_steps": _debug_steps_from_plan(plan),
                "tool_contract_source": _tool_contract_source_summary(plan.steps),
                "confirmation_required_steps": _confirmation_required_step_ids(plan),
                "execution_plan": plan.model_dump(),
            }
        )
        return goal, plan, planning_debug
    except Exception as exc:
        fallback_reason = str(exc)
        return _fallback_to_template_or_unsupported(
            goal,
            state,
            tool_registry,
            planning_debug,
            reason=fallback_reason,
            allow_template=llm_fallback_to_template,
        )


__all__ = [
    "GoalBuilder",
    "PlanBuilder",
    "PlanBuilderRegistry",
    "PLAN_BUILDER_REGISTRY",
    "build_executable_plan",
    "build_executable_plan_for_goal",
    "build_plan_runtime",
]
