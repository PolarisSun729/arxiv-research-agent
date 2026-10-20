from __future__ import annotations

import dataclasses
import importlib
from typing import get_args

import pytest
from pydantic import ValidationError

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
plan_goals = importlib.import_module("backend.agents.arxiv_search_agent.plan_goals")
semantic_types = importlib.import_module("backend.agents.arxiv_search_agent.semantic_types")
tool_registry = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
PlanStep = schemas.PlanStep
StepInputBinding = schemas.StepInputBinding
ToolSpec = schemas.ToolSpec
PlanGoal = plan_goals.PlanGoal
check_goal_coverage = plan_goals.check_goal_coverage
goals_from_state = plan_goals.goals_from_state


def _rule_plan(state: AgentState):
    _, plan, _ = planner_module.build_executable_plan(state, enable_tool_aware_planner=True, enable_llm_plan_draft=False)
    return plan


def _tag(plan, step_id: str, goal_id: str):
    """给规则计划的交付步骤标上 goal_id（接入前规则 builder 还不会标）。"""
    return [step.model_copy(update={"goal_id": goal_id}) if step.step_id == step_id else step for step in plan.steps]


def _step(step_id: str, tool_name: str, *, goal_id=None, after=()) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        action_type="answer",
        tool_name=tool_name,
        tool=ToolSpec(tool_name=tool_name),
        depends_on=list(after),
        goal_id=goal_id,
    )


def _qa_chain(prefix: str, goal_id=None):
    return [
        _step(f"{prefix}_check", "check_paper_index"),
        _step(f"{prefix}_answer", "answer_paper_question", after=[f"{prefix}_check"]),
        _step(f"{prefix}_assess", "assess_paper_qa_quality", goal_id=goal_id, after=[f"{prefix}_answer"]),
    ]


# ---- 目标模型与契约声明 ----


def test_every_goal_kind_has_deliverable_and_slots() -> None:
    kinds = set(get_args(plan_goals.GoalKind))
    assert kinds == set(plan_goals.GOAL_DELIVERABLES) == set(plan_goals.GOAL_SLOT_FIELDS)
    assert not kinds & {"unclear", "unsupported"}
    answer_kinds = set(get_args(semantic_types.AnswerDraft.model_fields["kind"].annotation))
    for deliverable in plan_goals.GOAL_DELIVERABLES.values():
        assert deliverable.semantic_type in semantic_types.FINAL_SEMANTIC_TYPES
        assert (deliverable.answer_kind is not None) == (deliverable.semantic_type == "AnswerDraft")
        assert deliverable.answer_kind is None or deliverable.answer_kind in answer_kinds


def test_answer_draft_producers_declare_answer_kind() -> None:
    registry = tool_registry.UNIFIED_TOOL_REGISTRY
    declared = {
        spec.tool_name: registry.get_contract(spec.tool_name).answer_kind
        for spec in registry.list_tools()
        if "AnswerDraft" in registry.get_contract(spec.tool_name).produces
    }
    assert declared == {
        "synthesize_arxiv_response": "search",
        "explain_recommendations": "recommendation",
        "synthesize_preference_response": "preference",
        "generate_clarification": "clarification",
        "generate_fallback_response": "fallback",
    }


def test_registry_rejects_inconsistent_answer_kind() -> None:
    registry = tool_registry.ToolRegistry()
    answer = tool_registry.UNIFIED_TOOL_REGISTRY.get_contract("synthesize_arxiv_response")
    search = tool_registry.UNIFIED_TOOL_REGISTRY.get_contract("search_arxiv")
    with pytest.raises(ValueError, match="answer_kind"):
        registry.register(dataclasses.replace(answer, answer_kind=None))
    with pytest.raises(ValueError, match="answer_kind"):
        registry.register(dataclasses.replace(search, answer_kind="search"))
    with pytest.raises(ValueError, match="summary"):
        registry.register(dataclasses.replace(answer, answer_kind="summary"))


def test_plan_goal_rejects_slots_of_other_kind_and_empty_id() -> None:
    with pytest.raises(ValidationError):
        PlanGoal(goal_id="g1", kind="arxiv_search", slots={"action": "like"})
    with pytest.raises(ValidationError):
        PlanGoal(goal_id="g1", kind="preference_action", slots={"action": "favorite"})
    with pytest.raises(ValidationError):
        PlanGoal(goal_id=" ", kind="recommendation")
    goal = PlanGoal(goal_id="g1", kind="preference_action", slots={"paper_ref": {"kind": "list_index", "value": 2}, "action": "like"})
    assert goal.slots.action == "like"


# ---- 目标覆盖 ----


@pytest.mark.parametrize(
    ("state", "delivery_step"),
    [
        (
            AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
            "synthesize_arxiv_response",
        ),
        (AgentState(intent="recommendation", message="recommend rag papers"), "explain_recommendations"),
        (
            AgentState(
                intent="preference_action",
                message="喜欢这篇论文",
                context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
            ),
            "synthesize_preference_response",
        ),
    ],
    ids=["search", "recommendation", "preference"],
)
def test_single_goal_rule_plan_is_covered_once_delivery_step_is_tagged(state: AgentState, delivery_step: str) -> None:
    goals = goals_from_state(state)
    plan = _rule_plan(state)

    untagged = check_goal_coverage(plan, goals)
    assert not untagged.ok
    assert "没有交付步骤" in untagged.errors[0]

    report = check_goal_coverage(_tag(plan, delivery_step, "g1"), goals)
    assert report.ok, report.errors
    assert report.delivery_step_by_goal == {"g1": delivery_step}


def test_missing_delivery_step_hints_producers() -> None:
    report = check_goal_coverage(_qa_chain("qa"), [PlanGoal(goal_id="g1", kind="paper_qa")])

    assert not report.ok
    assert report.errors == [
        "目标 g1（paper_qa）: 没有交付步骤，需要一步标注 goal_id=g1 并产出 CheckedQAResult，每个目标要有自己的交付步骤；"
        "能产出 CheckedQAResult 的工具：assess_paper_qa_quality"
    ]


def test_answer_kind_mismatch_is_rejected() -> None:
    steps = [
        _step("search_arxiv", "search_arxiv"),
        _step("synthesize_arxiv_response", "synthesize_arxiv_response", goal_id="g1", after=["search_arxiv"]),
    ]

    report = check_goal_coverage(steps, [PlanGoal(goal_id="g1", kind="recommendation")])

    assert not report.ok
    assert len(report.errors) == 1
    error = report.errors[0]
    assert "AnswerDraft（kind=search）" in error and "AnswerDraft（kind=recommendation）" in error
    assert "explain_recommendations" in error


def test_two_same_kind_goals_need_two_delivery_steps() -> None:
    goals = [PlanGoal(goal_id="g1", kind="paper_qa"), PlanGoal(goal_id="g2", kind="paper_summary")]

    shared = check_goal_coverage(_qa_chain("qa", goal_id="g1"), goals)
    assert not shared.ok
    assert len(shared.errors) == 1 and shared.errors[0].startswith("目标 g2（paper_summary）: 没有交付步骤")
    assert shared.delivery_step_by_goal == {"g1": "qa_assess"}

    both = check_goal_coverage(_qa_chain("qa", goal_id="g1") + _qa_chain("summary", goal_id="g2"), goals)
    assert both.ok, both.errors
    assert both.delivery_step_by_goal == {"g1": "qa_assess", "g2": "summary_assess"}


def test_one_goal_cannot_have_two_delivery_steps() -> None:
    steps = _qa_chain("a", goal_id="g1") + _qa_chain("b", goal_id="g1")

    report = check_goal_coverage(steps, [PlanGoal(goal_id="g1", kind="paper_qa")])

    assert not report.ok
    assert report.errors == ["目标 g1（paper_qa）: 有多个交付步骤 a_assess、b_assess，只能保留一个"]


def test_preference_goal_requires_upstream_preference_update() -> None:
    goals = [PlanGoal(goal_id="g1", kind="preference_action")]
    reply_only = [_step("reply", "synthesize_preference_response", goal_id="g1")]

    report = check_goal_coverage(reply_only, goals)
    assert not report.ok
    assert "上游没有产出 PreferenceUpdate" in report.errors[0]
    assert "update_preference_store" in report.errors[0]

    # 计划里有写入，但不在回复步骤的依赖链上，也不算。
    detached = [_step("update", "update_preference_store"), _step("reply", "synthesize_preference_response", goal_id="g1")]
    assert not check_goal_coverage(detached, goals).ok

    linked = [
        _step("update", "update_preference_store"),
        PlanStep(
            step_id="reply",
            action_type="answer",
            tool_name="synthesize_preference_response",
            tool=ToolSpec(tool_name="synthesize_preference_response"),
            input_bindings=[StepInputBinding(input_key="preference_action_result", source_type="step_output", step_id="update")],
            goal_id="g1",
        ),
    ]
    assert check_goal_coverage(linked, goals).ok


def test_step_tagged_with_unknown_goal_is_rejected() -> None:
    steps = _qa_chain("qa", goal_id="g1") + [_step("extra", "explain_recommendations", goal_id="g9")]

    report = check_goal_coverage(steps, [PlanGoal(goal_id="g1", kind="paper_qa")])

    assert not report.ok
    assert report.errors == ["extra: 标注的目标 g9 不存在，现有目标：g1"]


def test_coverage_requires_goals_with_unique_ids() -> None:
    with pytest.raises(ValueError):
        check_goal_coverage(_qa_chain("qa"), [])
    with pytest.raises(ValueError):
        check_goal_coverage(_qa_chain("qa"), [PlanGoal(goal_id="g1", kind="paper_qa"), PlanGoal(goal_id="g1", kind="paper_qa")])


# ---- 桥接 ----


def test_goals_from_state_carries_search_spec() -> None:
    spec = ArxivSearchSpec(intent="arxiv_search", query="RAG")

    goals = goals_from_state(AgentState(intent="arxiv_search", message="找 RAG 论文", search_spec=spec))

    assert [(goal.goal_id, goal.kind) for goal in goals] == [("g1", "arxiv_search")]
    assert goals[0].slots.search_spec.query == "RAG"


@pytest.mark.parametrize("intent", ["paper_qa", "paper_summary", "paper_detail"])
def test_goals_from_state_paper_intents_use_message_as_question(intent: str) -> None:
    goals = goals_from_state(AgentState(intent=intent, message=" 第二篇的方法是什么 "))

    assert [(goal.goal_id, goal.kind) for goal in goals] == [("g1", intent)]
    assert goals[0].slots.question == "第二篇的方法是什么"
    assert goals[0].slots.paper_ref is None  # 现有理解层不给论文指代


@pytest.mark.parametrize("intent", ["recommendation", "preference_action"])
def test_goals_from_state_leaves_unproduced_slots_empty(intent: str) -> None:
    goals = goals_from_state(AgentState(intent=intent, message="喜欢第一篇"))

    assert [(goal.goal_id, goal.kind) for goal in goals] == [("g1", intent)]
    assert goals[0].slots.model_dump(exclude_none=True) == {}


@pytest.mark.parametrize("intent", ["unclear", "unsupported", "confirmation", None, "something_else"])
def test_goals_from_state_returns_no_goal_for_non_plannable_intents(intent) -> None:
    assert goals_from_state(AgentState(intent=intent, message="hi")) == []
