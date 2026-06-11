"""Planner 入口与各类 Goal/Plan builder。

这里把“目标提取”和“计划生成”拆开，避免继续在单个大函数里堆所有 intent 分支。
"""

from __future__ import annotations

import logging
from datetime import datetime
from inspect import signature
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from .fallbacks import build_fallback_record
from .plan_validator import PlanValidator
from .planner_context import build_planner_context, planner_context_debug
from .schemas import ExecutablePlan, Goal, PlanRuntime, PlanStep, StepCondition, StepInputBinding, StepPolicy
from .state import AgentState
from .tool_aware_planner import LLMPlanDraftGenerator, PlanDraftConverter, PlanDraftPlanningError, RuleBasedToolAwarePlanBuilder, ToolCandidateSelector
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry
from .utils.state_utils import _compact_search_spec

logger = logging.getLogger(__name__)


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

PLANNER_RUNTIME_MODES = {
    "rule_only",
    "llm_preferred",
    "llm_only_strict",
    "demo_rule",
}

LEGACY_TEMPLATE_FALLBACK_SOURCE = "legacy_template_fallback"
LEGACY_TEMPLATE_FALLBACK_PATH = "legacy_template_fallback_planner"
UNSUPPORTED_FALLBACK_SOURCE = "unsupported_fallback"


def _get_agent_planner_runtime_config() -> Dict[str, Any]:
    try:
        from utils.config import get_agent_planner_runtime_config

        return dict(get_agent_planner_runtime_config())
    except Exception:
        return {
            "enable_rule_based_planner": True,
            "enable_tool_aware_planner": True,
            "enable_experimental_llm_planner": False,
            "enable_llm_plan_draft": False,
            "llm_plan_timeout": 8,
            "llm_plan_max_steps": 8,
            "enable_rule_fallback_after_llm_planner": True,
            "llm_plan_fallback_to_rule": True,
            "enable_template_fallback_planner": True,
            "llm_plan_fallback_to_template": True,
            "expose_planner_debug": True,
        }


def _normalized_planner_runtime_flags(planner_config: Mapping[str, Any]) -> Dict[str, Any]:
    planner_mode = str(planner_config.get("planner_runtime_mode") or planner_config.get("planner_mode") or "llm_preferred").strip().lower()
    if planner_mode not in PLANNER_RUNTIME_MODES:
        planner_mode = "llm_preferred"

    if planner_mode == "rule_only":
        rule_planner_enabled = True
        llm_draft_enabled = False
        rule_fallback_enabled = False
        template_fallback_enabled = True
        strict_llm_failure = False
    elif planner_mode == "llm_only_strict":
        rule_planner_enabled = True
        llm_draft_enabled = True
        rule_fallback_enabled = False
        template_fallback_enabled = False
        strict_llm_failure = True
    elif planner_mode == "demo_rule":
        rule_planner_enabled = True
        llm_draft_enabled = False
        rule_fallback_enabled = False
        template_fallback_enabled = True
        strict_llm_failure = False
    else:
        rule_planner_enabled = bool(
            planner_config.get("enable_rule_based_planner", planner_config.get("enable_tool_aware_planner", True))
        )
        llm_draft_enabled = bool(
            planner_config.get("enable_experimental_llm_planner", planner_config.get("enable_llm_plan_draft", True))
        )
        rule_fallback_enabled = bool(
            planner_config.get("enable_rule_fallback_after_llm_planner", planner_config.get("llm_plan_fallback_to_rule", True))
        )
        template_fallback_enabled = bool(
            planner_config.get("enable_template_fallback_planner", planner_config.get("llm_plan_fallback_to_template", True))
        )
        strict_llm_failure = False
    requested_path = "experimental_llm_draft_planner" if llm_draft_enabled else "rule_based_planner"
    configured_fallback_order: List[str] = []
    if llm_draft_enabled and rule_fallback_enabled:
        configured_fallback_order.append("rule_based_planner")
    if template_fallback_enabled:
        configured_fallback_order.append(LEGACY_TEMPLATE_FALLBACK_PATH)
    configured_fallback_order.append("unsupported_fallback_planner")
    return {
        "rule_planner_enabled": rule_planner_enabled,
        "llm_draft_enabled": llm_draft_enabled,
        "rule_fallback_enabled": rule_fallback_enabled,
        "template_fallback_enabled": template_fallback_enabled,
        "strict_llm_failure": strict_llm_failure,
        "planner_runtime_mode": planner_mode,
        "requested_path": requested_path,
        "configured_primary_path": requested_path if rule_planner_enabled else "primary_planner_disabled",
        "configured_fallback_order": configured_fallback_order,
    }


def _planner_path_from_source(source: Optional[str]) -> Optional[str]:
    mapping = {
        "fixed_template": LEGACY_TEMPLATE_FALLBACK_PATH,
        "llm_tool_aware": "experimental_llm_draft_planner",
        "tool_aware_rule_based": "rule_based_planner",
        "fixed_template_fallback": LEGACY_TEMPLATE_FALLBACK_PATH,
        LEGACY_TEMPLATE_FALLBACK_SOURCE: LEGACY_TEMPLATE_FALLBACK_PATH,
        "unsupported_fallback": "unsupported_fallback_planner",
    }
    normalized = str(source or "").strip()
    if not normalized:
        return None
    return mapping.get(normalized, normalized)


def _build_capability_boundary(goal: Goal, plan: ExecutablePlan) -> Dict[str, Any]:
    plan_tools = {step.tool_name for step in list(plan.steps or [])}
    clarification_summary: Optional[Dict[str, Any]] = None
    if {"analyze_ambiguity", "generate_clarification"}.intersection(plan_tools) or goal.goal_type == "unclear":
        clarification_summary = {
            "mode": "missing_information_analysis",
            "analysis_tool": "analyze_ambiguity",
            "question_tool": "generate_clarification",
            "analysis_source": "rule_based_missing_information_analysis",
            "question_source": "template_clarification_response",
            "is_llm_backed": False,
        }

    recommendation_summary: Optional[Dict[str, Any]] = None
    if "generate_recommendations" in plan_tools:
        recommendation_summary = {
            "mode": "agent_orchestration_with_service_core",
            "agent_step": "generate_recommendations",
            "backend_tool_name": "recommend_papers",
            "core_service": "RecommendationService.recommend_papers",
            "is_algorithm_core_in_agent": False,
        }

    return {
        "clarification": clarification_summary,
        "recommendation": recommendation_summary,
    }


def _finalize_plan_debug_and_metadata(
    goal: Goal,
    plan: ExecutablePlan,
    planning_debug: Dict[str, Any],
    runtime_flags: Mapping[str, Any],
) -> tuple[ExecutablePlan, Dict[str, Any]]:
    planner_summary = {
        "configured_primary_path": runtime_flags.get("configured_primary_path"),
        "configured_fallback_order": list(runtime_flags.get("configured_fallback_order") or []),
        "planner_runtime_mode": runtime_flags.get("planner_runtime_mode"),
        "requested_path": runtime_flags.get("requested_path"),
        "selected_path": _planner_path_from_source(planning_debug.get("selected_plan_source")),
        "final_path": _planner_path_from_source(planning_debug.get("final_plan_source")),
        "rule_planner_enabled": bool(runtime_flags.get("rule_planner_enabled")),
        "llm_draft_enabled": bool(runtime_flags.get("llm_draft_enabled")),
        "llm_draft_experimental": bool(runtime_flags.get("llm_draft_enabled")),
        "llm_draft_attempted": bool(planning_debug.get("llm_plan_attempted")),
        "llm_draft_disabled_reason": None if bool(runtime_flags.get("llm_draft_enabled")) else "llm_planner_disabled",
        "llm_draft_valid": bool(planning_debug.get("llm_plan_valid")),
        "rule_fallback_enabled": bool(runtime_flags.get("rule_fallback_enabled")),
        "rule_fallback_used": bool(planning_debug.get("rule_based_fallback_used")),
        "template_fallback_enabled": bool(runtime_flags.get("template_fallback_enabled")),
        "template_fallback_used": bool(planning_debug.get("template_fallback_used")),
        "fallback_used": bool(planning_debug.get("fallback_used")),
        "fallback_reason": planning_debug.get("fallback_reason"),
        "fallback_record": dict(planning_debug.get("fallback_record") or {}),
        "validation_status": planning_debug.get("validation_status"),
        "strict_llm_failure": bool(runtime_flags.get("strict_llm_failure")),
    }
    capability_boundary = _build_capability_boundary(goal, plan)
    planning_debug["planner_runtime_flags"] = dict(runtime_flags)
    planning_debug["planner_summary"] = planner_summary
    planning_debug["planner_requested_path"] = planner_summary["requested_path"]
    planning_debug["planner_selected_path"] = planner_summary["selected_path"]
    planning_debug["planner_final_path"] = planner_summary["final_path"]
    planning_debug["clarification_summary"] = capability_boundary.get("clarification")
    planning_debug["recommendation_summary"] = capability_boundary.get("recommendation")
    metadata = dict(plan.metadata or {})
    metadata["planner_summary"] = planner_summary
    metadata["capability_boundary"] = capability_boundary
    annotated_plan = plan.model_copy(update={"metadata": metadata})
    planning_debug["execution_plan"] = annotated_plan.model_dump()
    if planner_summary["template_fallback_used"]:
        logger.warning(
            "agent planner legacy template fallback used goal_type=%s requested=%s final=%s reason=%s code=%s",
            goal.goal_type,
            planner_summary["requested_path"],
            planner_summary["final_path"],
            planner_summary["fallback_reason"],
            planner_summary["fallback_record"].get("code"),
        )
    else:
        logger.info(
            "agent planner selected path goal_type=%s requested=%s final=%s llm_attempted=%s llm_valid=%s fallback=%s",
            goal.goal_type,
            planner_summary["requested_path"],
            planner_summary["final_path"],
            planner_summary["llm_draft_attempted"],
            planner_summary["llm_draft_valid"],
            planner_summary["fallback_used"],
        )
    return annotated_plan, planning_debug


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
    if (tool.requires_confirmation or resolved_side_effect_level == "persistent_write") and confirmation_policy is None:
        confirmation_policy = StepPolicy(
            policy_type="confirmation",
            mode="explicit_user_confirmation_required",
            requires_confirmation=True,
            note=f"{normalized_tool_name} requires confirmation by tool policy or persistent write risk",
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


class LegacyTemplateFallbackBuilder(Protocol):
    def build(
        self,
        goal: Goal,
        state: AgentState,
        tool_registry: ToolRegistry,
        *,
        fallback_reason: str,
    ) -> ExecutablePlan:
        ...


# 兼容旧测试/外部导入的类型名；新业务规划不要继续扩展这个别名。
PlanBuilder = LegacyTemplateFallbackBuilder


class LegacyTemplateFallbackPlanBuilder:
    """最后兜底模板，只生成安全回复，不承载搜索/QA/推荐等业务规划。

    新 planner 能力应加到 LLM draft 或 RuleBasedToolAwarePlanBuilder；
    这里保留的价值只是当主 planner 关闭、失败或上下文极端异常时，仍能给用户一个可解释回复。
    """

    def build(
        self,
        goal: Goal,
        state: AgentState,
        tool_registry: ToolRegistry,
        *,
        fallback_reason: str,
    ) -> ExecutablePlan:
        reason = _normalize_text(fallback_reason) or "template_fallback_used"
        steps = [
            _build_plan_step(
                step_id="generate_fallback_response",
                action_type="answer",
                tool_name="generate_fallback_response",
                output_key="final_answer",
                tool_registry=tool_registry,
                # legacy fallback 不能再猜业务参数，只把原始请求和结构化原因交给安全回复工具。
                input_bindings=[
                    _binding("message", source_type="state", source_key="message"),
                    _binding("fallback_reason", source_type="literal", value=reason, required=False),
                ],
            )
        ]
        plan = _make_plan(goal, steps=steps)
        metadata = dict(plan.metadata or {})
        metadata.update(
            {
                "planner_version": "legacy-template-fallback-v1",
                "legacy_template_fallback": True,
                "legacy_template_fallback_reason": reason,
                "source": LEGACY_TEMPLATE_FALLBACK_SOURCE,
                # 这里显式写入维护边界，避免后续把新业务分支再塞回模板 builder。
                "maintenance_boundary": "do_not_extend_for_new_business_capabilities",
            }
        )
        return plan.model_copy(update={"metadata": metadata})


class LegacyTemplateFallbackRegistry:
    """保留注册表形态只是为了兼容旧入口；当前只允许一个最小 fallback builder。"""

    def __init__(self) -> None:
        self._builder = LegacyTemplateFallbackPlanBuilder()

    def get(self, _goal_type: str) -> LegacyTemplateFallbackPlanBuilder:
        return self._builder


# 旧名称只作为兼容壳保留；看到它不代表存在多套业务 PlanBuilder。
PlanBuilderRegistry = LegacyTemplateFallbackRegistry
PLAN_BUILDER_REGISTRY = LegacyTemplateFallbackRegistry()


def _build_legacy_template_fallback_plan(
    goal: Goal,
    state: AgentState,
    tool_registry: ToolRegistry,
    *,
    reason: str,
) -> tuple[ExecutablePlan, LegacyTemplateFallbackPlanBuilder]:
    builder = PLAN_BUILDER_REGISTRY.get(goal.goal_type or "unsupported")
    plan = builder.build(goal, state, tool_registry, fallback_reason=reason)
    return plan, builder


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


def _paper_qa_answer_shape(intent: Optional[str]) -> Dict[str, str]:
    normalized_intent = str(intent or "").strip()
    if normalized_intent == "paper_summary":
        return {"step_id": "summarize_paper", "action_type": "summarize", "qa_mode": "summary"}
    if normalized_intent == "paper_detail":
        return {"step_id": "inspect_paper_detail", "action_type": "inspect_detail", "qa_mode": "detail"}
    return {"step_id": "answer_paper_question", "action_type": "answer", "qa_mode": "qa"}


def _select_candidate_tools(selector: ToolCandidateSelector, goal: Goal, state: AgentState, planner_context: Any) -> Any:
    select_signature = signature(selector.select)
    if "planner_context" in select_signature.parameters:
        return selector.select(goal, state, planner_context)
    # 历史测试和少量外部调用会 monkeypatch 旧签名；这里兼容它们，但真实入口仍优先传入 PlannerContext。
    return selector.select(goal, state)


def _fallback_to_template_or_unsupported(
    goal: Goal,
    state: AgentState,
    tool_registry: ToolRegistry,
    planning_debug: Dict[str, Any],
    runtime_flags: Mapping[str, Any],
    *,
    reason: str,
    allow_template: bool,
) -> tuple[Goal, ExecutablePlan, Dict[str, Any]]:
    if allow_template:
        try:
            plan, builder = _build_legacy_template_fallback_plan(goal, state, tool_registry, reason=reason)
            # legacy 模板是最后安全兜底，也必须经过 Validator，不能因为 fallback 就绕过执行边界。
            PlanValidator().validate(plan, tool_registry)
            fallback_record = build_fallback_record(
                reason,
                stage="planner",
                source=LEGACY_TEMPLATE_FALLBACK_SOURCE,
                detail={"goal_type": goal.goal_type, "intent": goal.intent},
            )
            planning_debug.update(
                {
                    "validation_status": "failed",
                    "fallback_validation_status": "passed",
                    "fallback_used": True,
                    "fallback_reason": reason,
                    "fallback_record": fallback_record,
                    "template_fallback_used": True,
                    "final_plan_source": LEGACY_TEMPLATE_FALLBACK_SOURCE,
                    "selected_plan_source": LEGACY_TEMPLATE_FALLBACK_SOURCE,
                    "plan_builder": builder.__class__.__name__,
                    "tool_contract_source": _tool_contract_source_summary(plan.steps),
                    "tool_contract_matrix": tool_registry.tool_contract_matrix(),
                    "confirmation_required_steps": _confirmation_required_step_ids(plan),
                    "execution_plan": plan.model_dump(),
                }
            )
            plan, planning_debug = _finalize_plan_debug_and_metadata(goal, plan, planning_debug, runtime_flags)
            return goal, plan, planning_debug
        except Exception as template_exc:
            reason = f"{reason}; template fallback failed: {template_exc}"

    unsupported_goal = goal.model_copy(update={"goal_type": "unsupported", "intent": "unsupported", "risk_level": "low"})
    plan, builder = _build_legacy_template_fallback_plan(
        unsupported_goal,
        state.model_copy(update={"intent": "unsupported"}),
        tool_registry,
        reason=reason,
    )
    PlanValidator().validate(plan, tool_registry)
    fallback_record = build_fallback_record(
        reason,
        stage="planner",
        source=UNSUPPORTED_FALLBACK_SOURCE,
        detail={"goal_type": goal.goal_type, "intent": goal.intent},
    )
    planning_debug.update(
        {
            "validation_status": "failed",
            "fallback_validation_status": "passed",
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_record": fallback_record,
            "template_fallback_used": bool(allow_template),
            "final_plan_source": UNSUPPORTED_FALLBACK_SOURCE,
            "selected_plan_source": UNSUPPORTED_FALLBACK_SOURCE,
            "plan_builder": builder.__class__.__name__,
            "tool_contract_source": _tool_contract_source_summary(plan.steps),
            "tool_contract_matrix": tool_registry.tool_contract_matrix(),
            "confirmation_required_steps": _confirmation_required_step_ids(plan),
            "execution_plan": plan.model_dump(),
        }
    )
    plan, planning_debug = _finalize_plan_debug_and_metadata(unsupported_goal, plan, planning_debug, runtime_flags)
    return unsupported_goal, plan, planning_debug


def _raise_strict_planner_failure(
    planning_debug: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    llm_summary = dict(planning_debug.get("llm_plan_validation_summary") or {})
    invalid_reasons = list(planning_debug.get("llm_plan_invalid_reasons") or [])
    suffix = f" validation={llm_summary}" if llm_summary else ""
    if invalid_reasons:
        suffix = f" invalid_reasons={invalid_reasons}{suffix}"
    raise PlanDraftPlanningError(f"llm_only_strict planning failed: {reason}{suffix}")


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

    planner_config = _get_agent_planner_runtime_config()
    runtime_flags = _normalized_planner_runtime_flags(planner_config)
    if enable_tool_aware_planner is not None:
        runtime_flags["rule_planner_enabled"] = bool(enable_tool_aware_planner)
    if enable_llm_plan_draft is not None:
        runtime_flags["llm_draft_enabled"] = bool(enable_llm_plan_draft)
        # 显式调用参数优先于运行模式的默认值：
        # 例如测试或灰度入口希望“临时打开 LLM draft 并允许正常 fallback”，
        # 不应继续沿用 rule_only/demo_rule 的禁用语义。
        if bool(enable_llm_plan_draft):
            runtime_flags["rule_fallback_enabled"] = bool(
                planner_config.get("enable_rule_fallback_after_llm_planner", planner_config.get("llm_plan_fallback_to_rule", True))
            )
            runtime_flags["template_fallback_enabled"] = bool(
                planner_config.get("enable_template_fallback_planner", planner_config.get("llm_plan_fallback_to_template", True))
            )
            runtime_flags["strict_llm_failure"] = False
    runtime_flags["requested_path"] = "experimental_llm_draft_planner" if runtime_flags["llm_draft_enabled"] else "rule_based_planner"
    runtime_flags["configured_primary_path"] = runtime_flags["requested_path"] if runtime_flags["rule_planner_enabled"] else "primary_planner_disabled"

    if not bool(runtime_flags.get("rule_planner_enabled")):
        reason = "tool_aware_planner_disabled"
        # 配置关闭主 planner 时也只允许进入 legacy fallback，避免模板重新成为主扩展路径。
        planning_debug = {
            "planner_mode": "primary_planner_disabled",
            "goal": goal.model_dump(),
            "candidate_tools": [],
            "excluded_tools": [],
            "draft_steps": [],
            "selected_steps": [],
            "skipped_steps": [],
            "skipped_tools": [],
            "llm_plan_raw_summary": {},
            "llm_plan_attempted": False,
            "llm_plan_valid": False,
            "llm_plan_invalid_reasons": [],
            "llm_plan_validation_summary": {},
            "rule_based_fallback_used": False,
            "template_fallback_used": False,
            "validation_status": "not_started",
            "fallback_used": False,
            "fallback_reason": reason,
            "fallback_record": build_fallback_record(
                reason,
                stage="planner",
                source="build_executable_plan_for_goal",
                detail={"planner_runtime_mode": runtime_flags.get("planner_runtime_mode")},
            ),
            "final_plan_source": None,
            "selected_plan_source": None,
            "planner_context": {},
            "tool_selection": {},
            "tool_risk_summary": {},
            "tool_contract_source": {},
            "tool_contract_matrix": tool_registry.tool_contract_matrix(),
            "confirmation_required_steps": [],
            "planner_runtime_flags": dict(runtime_flags),
        }
        return _fallback_to_template_or_unsupported(
            goal,
            state,
            tool_registry,
            planning_debug,
            runtime_flags,
            reason=reason,
            allow_template=bool(runtime_flags.get("template_fallback_enabled")),
        )

    planner_config = _get_agent_planner_runtime_config()
    llm_enabled = bool(runtime_flags.get("llm_draft_enabled"))
    llm_fallback_to_rule = bool(runtime_flags.get("rule_fallback_enabled", True))
    llm_fallback_to_template = bool(runtime_flags.get("template_fallback_enabled", True))
    strict_llm_failure = bool(runtime_flags.get("strict_llm_failure"))
    expose_planner_debug = bool(planner_config.get("expose_planner_debug", True))

    selector = ToolCandidateSelector(tool_registry)
    try:
        planner_context = build_planner_context(goal=goal, state=state, tool_registry=tool_registry)
        planner_context_payload = planner_context_debug(planner_context)
    except Exception as exc:
        # Planner 输入层是动态规划的前置增强；构造失败时必须回到旧模板链路，不能影响现有 Agent 可用性。
        planning_debug = {
            "planner_mode": "tool_aware_llm" if llm_enabled else "tool_aware_rule_based",
            "goal": goal.model_dump(),
            "candidate_tools": [],
            "excluded_tools": [],
            "draft_steps": [],
            "selected_steps": [],
            "skipped_steps": [],
            "skipped_tools": [],
            "llm_plan_raw_summary": {},
            "llm_plan_attempted": False,
            "llm_plan_valid": False,
            "llm_plan_invalid_reasons": [],
            "llm_plan_validation_summary": {},
            "rule_based_fallback_used": False,
            "template_fallback_used": False,
            "validation_status": "not_started",
            "fallback_used": False,
            "fallback_reason": None,
            "fallback_record": {},
            "final_plan_source": None,
            "selected_plan_source": None,
            "planner_context": {"build_error": str(exc)},
            "tool_selection": {},
            "tool_risk_summary": {},
            "tool_contract_source": {},
            "tool_contract_matrix": tool_registry.tool_contract_matrix(),
            "confirmation_required_steps": [],
            "planner_runtime_flags": dict(runtime_flags),
        }
        return _fallback_to_template_or_unsupported(
            goal,
            state,
            tool_registry,
            planning_debug,
            runtime_flags,
            reason=f"missing_required_context: planner_context_build_failed: {exc}",
            allow_template=llm_fallback_to_template,
        )

    selection = _select_candidate_tools(selector, goal, state, planner_context)
    candidate_tool_names = [tool.tool_name for tool in list(selection.candidate_tools or [])]
    planning_debug: Dict[str, Any] = {
        "planner_mode": "tool_aware_llm" if llm_enabled else "tool_aware_rule_based",
        "goal": goal.model_dump(),
        "planner_context": planner_context_payload,
        "candidate_tools": [tool.model_dump() for tool in list(selection.candidate_tools or [])],
        "excluded_tools": [tool.model_dump() for tool in list(selection.excluded_tools or [])],
        "draft_steps": [],
        "selected_steps": [],
        "skipped_steps": [],
        "skipped_tools": [],
        "llm_plan_raw_summary": {},
        "llm_plan_attempted": False,
        "llm_plan_valid": False,
        "llm_plan_invalid_reasons": [],
        "llm_plan_validation_summary": {},
        "rule_based_fallback_used": False,
        "template_fallback_used": False,
        "validation_status": "not_started",
        "fallback_used": False,
        "fallback_reason": None,
        "fallback_record": {},
        "final_plan_source": None,
        "selected_plan_source": None,
        "tool_selection": selection.model_dump(),
        "tool_risk_summary": dict(selection.risk_summary or {}),
        "tool_contract_source": {},
        "tool_contract_matrix": tool_registry.tool_contract_matrix(),
        "confirmation_required_steps": [],
        "planner_runtime_flags": dict(runtime_flags),
    }

    if not candidate_tool_names:
        if llm_enabled and strict_llm_failure:
            planning_debug["fallback_reason"] = "planner_context_candidate_tools_empty"
            planning_debug["fallback_record"] = build_fallback_record(
                "planner_context_candidate_tools_empty",
                stage="planner",
                source="build_executable_plan_for_goal",
            )
            _raise_strict_planner_failure(planning_debug, reason="planner_context_candidate_tools_empty")
        # 候选为空说明动态筛选边界过窄或上下文不足；此时保留 debug，再交给 legacy 模板兜底。
        return _fallback_to_template_or_unsupported(
            goal,
            state,
            tool_registry,
            planning_debug,
            runtime_flags,
            reason="missing_required_context: planner_context_candidate_tools_empty",
            allow_template=llm_fallback_to_template,
        )

    if llm_enabled:
        planning_debug["llm_plan_attempted"] = True
        try:
            llm_generator = LLMPlanDraftGenerator(
                llm_generation_service if llm_generation_service is not None else _resolve_generation_service(),
                max_steps=int(planner_config.get("llm_plan_max_steps", 8) or 8),
                timeout_seconds=int(planner_config.get("llm_plan_timeout", 8) or 8),
            )
            draft = llm_generator.generate(goal, state, selection.candidate_tools, tool_registry, planner_context)
            if expose_planner_debug:
                planning_debug["raw_llm_plan"] = llm_generator.last_debug.get("raw_llm_plan")
            planning_debug["llm_plan_raw_summary"] = dict(llm_generator.last_debug.get("raw_llm_plan_summary") or {})
            planning_debug["draft_steps"] = [step.model_dump() for step in list(draft.steps or [])]
            plan = PlanDraftConverter(tool_registry).convert(draft, goal, allowed_tool_names=candidate_tool_names)
            PlanValidator().validate(plan, tool_registry)
            planning_debug.update(
                {
                    "llm_plan_valid": True,
                    "llm_plan_validation_summary": dict(
                        llm_generator.last_debug.get("validation_summary")
                        or {
                            "status": "passed",
                            "validator_stack": ["llm_json_only_parse", "PlanDraft.model_validate", "llm_draft_normalization", "PlanDraftConverter", "PlanValidator"],
                            "reasons": [],
                        }
                    ),
                    "validation_status": "passed",
                    "final_plan_source": "llm_tool_aware",
                    "selected_plan_source": "llm_tool_aware",
                    "selected_steps": _debug_steps_from_plan(plan),
                    "tool_contract_source": _tool_contract_source_summary(plan.steps),
                    "confirmation_required_steps": _confirmation_required_step_ids(plan),
                    "execution_plan": plan.model_dump(),
                }
            )
            plan, planning_debug = _finalize_plan_debug_and_metadata(goal, plan, planning_debug, runtime_flags)
            return goal, plan, planning_debug
        except Exception as exc:
            invalid_reason = f"llm_planner_validation_failed: {exc}"
            planning_debug["llm_plan_valid"] = False
            planning_debug["llm_plan_invalid_reasons"] = [invalid_reason]
            planning_debug["fallback_reason"] = invalid_reason
            planning_debug["fallback_record"] = build_fallback_record(
                invalid_reason,
                stage="planner",
                source="LLMPlanDraftGenerator",
                detail={"planner_runtime_mode": runtime_flags.get("planner_runtime_mode")},
            )
            planning_debug["llm_plan_validation_summary"] = dict(
                getattr(locals().get("llm_generator", None), "last_debug", {}).get("validation_summary")
                or {
                    "status": "failed",
                    "validator_stack": ["llm_json_only_parse", "PlanDraft.model_validate", "llm_draft_normalization", "PlanDraftConverter", "PlanValidator"],
                    "reasons": [invalid_reason],
                }
            )
            if expose_planner_debug:
                planning_debug["raw_llm_plan"] = getattr(locals().get("llm_generator", None), "last_debug", {}).get("raw_llm_plan")
            planning_debug["llm_plan_raw_summary"] = dict(
                getattr(locals().get("llm_generator", None), "last_debug", {}).get("raw_llm_plan_summary") or {}
            )
            if strict_llm_failure:
                _raise_strict_planner_failure(planning_debug, reason=invalid_reason)
            if not llm_fallback_to_rule:
                return _fallback_to_template_or_unsupported(
                    goal,
                    state,
                    tool_registry,
                    planning_debug,
                    runtime_flags,
                    reason=invalid_reason,
                    allow_template=llm_fallback_to_template,
                )

    try:
        rule_builder = RuleBasedToolAwarePlanBuilder()
        draft = rule_builder.build(goal, state, selection.candidate_tools, tool_registry, planner_context)
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
            # 规则型 builder 可以显式声明“不要执行这个草稿”，统一交给 legacy 模板兜底。
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
        if goal.goal_type == "unsupported":
            # unsupported 是规则 planner 主动选择的安全回复，不是 legacy 模板兜底；
            # 仍写入结构化原因，避免 trace 读者从工具名猜测 fallback 语义。
            planning_debug.update(
                {
                    "fallback_used": True,
                    "fallback_reason": "unsupported_goal",
                    "fallback_record": build_fallback_record(
                        "unsupported_goal",
                        stage="planner",
                        source="tool_aware_rule_based",
                        detail={"goal_type": goal.goal_type, "intent": goal.intent},
                    ),
                    "unsupported_fallback_used": True,
                }
            )
        plan, planning_debug = _finalize_plan_debug_and_metadata(goal, plan, planning_debug, runtime_flags)
        return goal, plan, planning_debug
    except Exception as exc:
        fallback_reason = f"tool_aware_planner_failed: {exc}"
        return _fallback_to_template_or_unsupported(
            goal,
            state,
            tool_registry,
            planning_debug,
            runtime_flags,
            reason=fallback_reason,
            allow_template=llm_fallback_to_template,
        )


__all__ = [
    "GoalBuilder",
    "LegacyTemplateFallbackPlanBuilder",
    "LegacyTemplateFallbackRegistry",
    "PlanBuilder",
    "PlanBuilderRegistry",
    "PLAN_BUILDER_REGISTRY",
    "build_executable_plan",
    "build_executable_plan_for_goal",
    "build_plan_runtime",
]
