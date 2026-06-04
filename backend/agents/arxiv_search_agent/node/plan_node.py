"""基于当前状态生成结构化 goal 和 execution_plan。

这个节点位于 parse 之后、具体执行之前，专门负责把：
- intent
- search_spec
- context
- user_memory_summary
整理为可调试、可解释的结构化规划信息。

阶段 1 中它只生成规划，不调用任何真实工具，也不接管后续执行流程。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..schemas import ExecutionPlanStep, Goal
from ..state import AgentState
from ..utils.state_utils import _append_step, _coerce_state, _compact_search_spec


def _normalize_intent(value: Any) -> str:
    text = str(value or "").strip()
    return text or "unsupported"


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        text = _normalize_text(value)
        if text:
            return text
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


def _build_plan_step(
    *,
    step_id: str,
    step_type: str,
    description: str,
    expected_input: Optional[Dict[str, Any]] = None,
    expected_output: Optional[Dict[str, Any]] = None,
    depends_on: Optional[List[str]] = None,
    status: str = "pending",
) -> ExecutionPlanStep:
    return ExecutionPlanStep(
        step_id=step_id,
        step_type=step_type,
        description=description,
        expected_input=expected_input or {},
        expected_output=expected_output or {},
        depends_on=depends_on or [],
        status=status,
    )


def _build_goal_and_plan(state: AgentState) -> tuple[Goal, List[ExecutionPlanStep], Dict[str, Any]]:
    intent = _normalize_intent(state.intent)
    context = _get_context_mapping(state)
    user_memory_summary = _get_user_memory_summary(context)
    selected_paper_hint = _get_selected_paper_hint(context)
    message = _normalize_text(state.message)
    pending_action = state.pending_action if isinstance(state.pending_action, Mapping) else None
    paper_qa_result = state.paper_qa_result if isinstance(state.paper_qa_result, Mapping) else None

    requires_confirmation = False
    if isinstance(pending_action, Mapping) and str(pending_action.get("status") or "").strip() == "waiting_confirmation":
        requires_confirmation = True
    if isinstance(paper_qa_result, Mapping) and str(paper_qa_result.get("status") or "").strip() == "waiting_confirmation":
        requires_confirmation = True

    base_debug: Dict[str, Any] = {
        "intent": intent,
        "user_memory_summary_present": bool(user_memory_summary),
        "selected_paper_hint": selected_paper_hint,
        "search_spec": _compact_search_spec(state.search_spec),
    }

    if intent == "arxiv_search":
        constraints = _build_search_constraints(state)
        goal = Goal(
            goal_type="arxiv_search",
            user_goal=message or "Search arXiv papers relevant to the user's request.",
            task_scope="single-turn literature search and response synthesis",
            constraints=constraints,
            success_criteria=_dedupe_strings(
                [
                    "Extract a usable search target from the request.",
                    "Return relevant arXiv papers or explain why no suitable results were found.",
                    "Apply personalization when profile or memory context is available.",
                    "Generate a concise response with next actions when helpful.",
                ]
            ),
            requires_memory=bool(user_memory_summary or context.get("research_profile")),
            requires_user_confirmation=False,
        )
        plan = [
            _build_plan_step(
                step_id="step_1",
                step_type="goal_interpretation",
                description="Interpret the search request and confirm the target topic, scope, and constraints.",
                expected_input={"intent": intent, "message": message, "search_spec": _compact_search_spec(state.search_spec)},
                expected_output={"goal": "clear search objective", "validated_search_spec": True},
            ),
            _build_plan_step(
                step_id="step_2",
                step_type="search_execution",
                description="Search arXiv papers using the structured search specification.",
                expected_input={"search_spec": _compact_search_spec(state.search_spec)},
                expected_output={"papers": "candidate paper list", "tool_result": "search raw result"},
                depends_on=["step_1"],
            ),
            _build_plan_step(
                step_id="step_3",
                step_type="result_validation",
                description="Check whether search results are sufficient and whether retry or relaxation may be needed.",
                expected_input={"papers": "candidate paper list", "tool_result": "search raw result"},
                expected_output={"result_quality": "enough_or_needs_retry"},
                depends_on=["step_2"],
            ),
            _build_plan_step(
                step_id="step_4",
                step_type="personalization",
                description="Optionally personalize ranking and annotations based on profile or memory context.",
                expected_input={"papers": "candidate paper list", "user_memory_summary": user_memory_summary},
                expected_output={"ranked_papers": "personalized paper list"},
                depends_on=["step_3"],
            ),
            _build_plan_step(
                step_id="step_5",
                step_type="response_synthesis",
                description="Generate a final search response with recommendations, warnings, and next actions.",
                expected_input={"ranked_papers": "personalized paper list", "warnings": list(state.warnings or [])},
                expected_output={"answer": "user-facing response"},
                depends_on=["step_4"],
            ),
        ]
        return goal, plan, base_debug

    if intent in {"paper_summary", "paper_detail", "paper_qa"}:
        goal = Goal(
            goal_type=intent,
            user_goal=message or "Answer a paper reading request based on the selected paper.",
            task_scope="single-paper reading assistance",
            constraints=_dedupe_strings([
                f"target_paper={selected_paper_hint}" if selected_paper_hint else None,
                "use existing QA index when available",
            ]),
            success_criteria=_dedupe_strings(
                [
                    "Resolve which paper the user is asking about.",
                    "Reuse existing paper QA/index data when available.",
                    "If indexing is required, wait for user confirmation before parsing the paper.",
                    "Return a direct answer, summary, or detail explanation for the requested paper.",
                ]
            ),
            requires_memory=False,
            requires_user_confirmation=requires_confirmation,
        )
        plan = [
            _build_plan_step(
                step_id="step_1",
                step_type="paper_resolution",
                description="Confirm the target paper from context or the current request.",
                expected_input={"intent": intent, "message": message, "selected_paper_hint": selected_paper_hint},
                expected_output={"paper_reference": "resolved target paper"},
            ),
            _build_plan_step(
                step_id="step_2",
                step_type="qa_index_check",
                description="Check whether a usable QA index or parsed paper content is already available.",
                expected_input={"paper_reference": "resolved target paper"},
                expected_output={"qa_index_status": "available_or_missing"},
                depends_on=["step_1"],
            ),
            _build_plan_step(
                step_id="step_3",
                step_type="confirmation_gate",
                description="If parsing or indexing is needed, wait for explicit user confirmation before continuing.",
                expected_input={"qa_index_status": "available_or_missing", "pending_action": pending_action or {}, "paper_qa_result": paper_qa_result or {}},
                expected_output={"confirmation_status": "confirmed_or_waiting_or_not_needed"},
                depends_on=["step_2"],
            ),
            _build_plan_step(
                step_id="step_4",
                step_type="paper_response",
                description="Produce the requested summary, detail explanation, or paper QA answer.",
                expected_input={"paper_reference": "resolved target paper", "intent": intent},
                expected_output={"answer": "paper reading response"},
                depends_on=["step_3"],
            ),
        ]
        return goal, plan, base_debug

    if intent == "preference_action":
        goal = Goal(
            goal_type="preference_action",
            user_goal=message or "Update the user's preference on a target paper.",
            task_scope="single preference mutation and acknowledgement",
            constraints=_dedupe_strings([f"target_paper={selected_paper_hint}" if selected_paper_hint else None]),
            success_criteria=_dedupe_strings(
                [
                    "Identify which paper the preference action applies to.",
                    "Apply the requested preference mutation safely.",
                    "Update memory or profile signals when the action succeeds.",
                    "Return a clear confirmation message to the user.",
                ]
            ),
            requires_memory=True,
            requires_user_confirmation=False,
        )
        plan = [
            _build_plan_step(
                step_id="step_1",
                step_type="paper_resolution",
                description="Locate the target paper for the preference action.",
                expected_input={"message": message, "selected_paper_hint": selected_paper_hint},
                expected_output={"paper_reference": "resolved target paper"},
            ),
            _build_plan_step(
                step_id="step_2",
                step_type="preference_update",
                description="Apply the requested preference action to the resolved paper.",
                expected_input={"paper_reference": "resolved target paper", "intent": intent},
                expected_output={"preference_action_result": "mutation result"},
                depends_on=["step_1"],
            ),
            _build_plan_step(
                step_id="step_3",
                step_type="memory_sync",
                description="Update user memory or profile signals that depend on the preference action.",
                expected_input={"preference_action_result": "mutation result", "user_memory_summary": user_memory_summary},
                expected_output={"memory_update": "updated_or_skipped"},
                depends_on=["step_2"],
            ),
            _build_plan_step(
                step_id="step_4",
                step_type="response_synthesis",
                description="Generate a confirmation response that explains the applied preference action.",
                expected_input={"preference_action_result": "mutation result"},
                expected_output={"answer": "user-facing confirmation"},
                depends_on=["step_3"],
            ),
        ]
        return goal, plan, base_debug

    if intent == "recommendation":
        goal = Goal(
            goal_type="recommendation",
            user_goal=message or "Recommend papers aligned with the user's interests.",
            task_scope="single-turn personalized recommendation",
            constraints=_dedupe_strings(["prefer user profile and memory context when available"]),
            success_criteria=_dedupe_strings(
                [
                    "Read available user profile, memory summary, or interest signals.",
                    "Generate relevant recommendation candidates.",
                    "Explain why the recommendations match the user's interests.",
                ]
            ),
            requires_memory=True,
            requires_user_confirmation=False,
        )
        plan = [
            _build_plan_step(
                step_id="step_1",
                step_type="profile_loading",
                description="Read user profile, memory summary, or interest vectors relevant to recommendation.",
                expected_input={"context_keys": sorted(context.keys()), "user_memory_summary": user_memory_summary},
                expected_output={"recommendation_profile": "usable personalization context"},
            ),
            _build_plan_step(
                step_id="step_2",
                step_type="recommendation_generation",
                description="Generate recommendation candidates based on the available profile and request context.",
                expected_input={"recommendation_profile": "usable personalization context", "message": message},
                expected_output={"papers": "recommended paper list"},
                depends_on=["step_1"],
            ),
            _build_plan_step(
                step_id="step_3",
                step_type="recommendation_explanation",
                description="Explain why the recommended papers are relevant for this user.",
                expected_input={"papers": "recommended paper list", "recommendation_profile": "usable personalization context"},
                expected_output={"answer": "recommendation response with rationale"},
                depends_on=["step_2"],
            ),
        ]
        return goal, plan, base_debug

    if intent == "reading_list_action":
        goal = Goal(
            goal_type="reading_list_action",
            user_goal=message or "Handle the user's reading list request.",
            task_scope="reading list action for the current turn",
            constraints=_dedupe_strings([f"target_paper={selected_paper_hint}" if selected_paper_hint else None]),
            success_criteria=_dedupe_strings(
                [
                    "Resolve the reading list action from the request.",
                    "Identify the relevant paper if one is referenced.",
                    "Return a clear outcome or next-step explanation.",
                ]
            ),
            requires_memory=True,
            requires_user_confirmation=False,
        )
        plan = [
            _build_plan_step(
                step_id="step_1",
                step_type="action_resolution",
                description="Interpret the reading list action and target paper from the current request.",
                expected_input={"message": message, "selected_paper_hint": selected_paper_hint},
                expected_output={"reading_list_action": "resolved action"},
            ),
            _build_plan_step(
                step_id="step_2",
                step_type="state_update",
                description="Apply or prepare the reading list state update for the resolved action.",
                expected_input={"reading_list_action": "resolved action"},
                expected_output={"reading_list_result": "updated_or_skipped"},
                depends_on=["step_1"],
            ),
            _build_plan_step(
                step_id="step_3",
                step_type="response_synthesis",
                description="Generate a response describing the reading list outcome or required clarification.",
                expected_input={"reading_list_result": "updated_or_skipped"},
                expected_output={"answer": "user-facing response"},
                depends_on=["step_2"],
            ),
        ]
        return goal, plan, base_debug

    if intent == "unclear":
        goal = Goal(
            goal_type="unclear",
            user_goal=message or "Clarify the user's ambiguous request.",
            task_scope="clarification only, no tool execution",
            constraints=["do not execute tools before clarification"],
            success_criteria=_dedupe_strings(
                [
                    "Identify what key information is missing from the request.",
                    "Ask the user for a focused clarification.",
                ]
            ),
            requires_memory=False,
            requires_user_confirmation=False,
        )
        plan = [
            _build_plan_step(
                step_id="step_1",
                step_type="ambiguity_analysis",
                description="Analyze why the current request is ambiguous or under-specified.",
                expected_input={"message": message, "intent": intent},
                expected_output={"missing_information": "clarification points"},
            ),
            _build_plan_step(
                step_id="step_2",
                step_type="clarification_response",
                description="Generate a clarification question instead of entering any tool execution path.",
                expected_input={"missing_information": "clarification points"},
                expected_output={"answer": "clarification request"},
                depends_on=["step_1"],
            ),
        ]
        return goal, plan, base_debug

    goal = Goal(
        goal_type="unsupported",
        user_goal=message or "Explain why the request cannot be handled by the current agent.",
        task_scope="graceful fallback response only",
        constraints=["do not invoke unsupported tools or services"],
        success_criteria=_dedupe_strings(
            [
                "Identify that the request is outside the supported capability set.",
                "Return a clear limitation explanation and, when possible, suggest a supported alternative.",
            ]
        ),
        requires_memory=False,
        requires_user_confirmation=False,
    )
    plan = [
        _build_plan_step(
            step_id="step_1",
            step_type="capability_check",
            description="Determine why the request is unsupported by the current agent capability set.",
            expected_input={"message": message, "intent": intent},
            expected_output={"unsupported_reason": "capability mismatch"},
        ),
        _build_plan_step(
            step_id="step_2",
            step_type="fallback_response",
            description="Generate a graceful fallback response without entering unsupported execution paths.",
            expected_input={"unsupported_reason": "capability mismatch"},
            expected_output={"answer": "fallback response"},
            depends_on=["step_1"],
        ),
    ]
    return goal, plan, base_debug


def plan_task(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """基于 parse 结果生成结构化 goal 和 execution_plan。

    设计原则：
    - 不调用真实工具；
    - 不抛出中断主流程的异常；
    - 即使失败也回写一个可解释的 fallback goal/plan；
    - 统一追加一条 steps 轨迹，方便调试和前端展示。
    """
    current_state = _coerce_state(state)

    try:
        goal, execution_plan, planning_debug = _build_goal_and_plan(current_state)
        next_state = current_state.model_copy(deep=True)
        next_state.goal = goal
        next_state.execution_plan = execution_plan
        next_state.debug = dict(next_state.debug or {})
        next_state.debug["plan_task"] = {
            **planning_debug,
            "goal": goal.model_dump(),
            "execution_plan": [step.model_dump() for step in execution_plan],
        }
        return _append_step(
            next_state,
            step="plan_task",
            status="success",
            action="generate_goal_and_execution_plan",
            inputs={
                "intent": current_state.intent,
                "message": current_state.message,
                "search_spec": _compact_search_spec(current_state.search_spec),
                "context_keys": sorted(_get_context_mapping(current_state).keys()),
            },
            outputs={
                "goal_type": goal.goal_type,
                "execution_plan_steps": len(execution_plan),
                "requires_user_confirmation": goal.requires_user_confirmation,
                "requires_memory": goal.requires_memory,
            },
        )
    except Exception as exc:
        next_state = current_state.model_copy(deep=True)
        fallback_goal = Goal(
            goal_type="unsupported",
            user_goal=_normalize_text(current_state.message) or "Fallback planning after planning failure.",
            task_scope="graceful fallback response only",
            constraints=["planning failed, keep main workflow running"],
            success_criteria=["Preserve the main workflow and provide a fallback response path."],
            requires_memory=False,
            requires_user_confirmation=False,
        )
        fallback_plan = [
            _build_plan_step(
                step_id="step_1",
                step_type="fallback_response",
                description="Continue the main workflow with a safe fallback planning result.",
                expected_input={"intent": current_state.intent, "message": current_state.message},
                expected_output={"answer": "fallback response"},
            )
        ]
        next_state.goal = fallback_goal
        next_state.execution_plan = fallback_plan
        next_state.debug = dict(next_state.debug or {})
        next_state.debug["plan_task"] = {
            "intent": _normalize_intent(current_state.intent),
            "error": str(exc),
            "goal": fallback_goal.model_dump(),
            "execution_plan": [step.model_dump() for step in fallback_plan],
        }
        return _append_step(
            next_state,
            step="plan_task",
            status="failed",
            action="generate_goal_and_execution_plan",
            inputs={
                "intent": current_state.intent,
                "message": current_state.message,
            },
            outputs={
                "goal_type": fallback_goal.goal_type,
                "execution_plan_steps": len(fallback_plan),
            },
            error=str(exc),
        )
