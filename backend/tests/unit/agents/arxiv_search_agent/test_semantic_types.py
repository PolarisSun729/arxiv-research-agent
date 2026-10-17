from __future__ import annotations

import dataclasses
import importlib

import pytest
from pydantic import ValidationError

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
semantic_types = importlib.import_module("backend.agents.arxiv_search_agent.semantic_types")
plan_closure = importlib.import_module("backend.agents.arxiv_search_agent.plan_closure")
tool_registry = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
PlanStep = schemas.PlanStep
StepInputBinding = schemas.StepInputBinding
ToolSpec = schemas.ToolSpec
check_plan_closure = plan_closure.check_plan_closure
project_state_types = semantic_types.project_state_types


def _rule_plan(state: AgentState):
    _, plan, _ = planner_module.build_executable_plan(state, enable_tool_aware_planner=True, enable_llm_plan_draft=False)
    return plan


def _step(step_id: str, tool_name: str, bindings=()) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        action_type="answer",
        tool_name=tool_name,
        tool=ToolSpec(tool_name=tool_name),
        input_bindings=[StepInputBinding(**binding) for binding in bindings],
    )


# ---- 类型不变量 ----


def test_paper_list_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        semantic_types.PaperList(papers=[], origin="search")
    papers = semantic_types.PaperList(papers=[{"arxiv_id": "2401.00001", "title": "RAG"}], origin="search")
    assert papers.papers[0].arxiv_id == "2401.00001"


def test_resolved_paper_requires_arxiv_id() -> None:
    with pytest.raises(ValidationError):
        semantic_types.ResolvedPaper(arxiv_id="  ", resolved_by="selected")
    with pytest.raises(ValidationError):
        semantic_types.PaperRecord(arxiv_id="", title="no id")


def test_user_profile_and_recommendations_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        semantic_types.UserProfile(user_id="u1", research_profile={}, memory_summary=None)
    with pytest.raises(ValidationError):
        semantic_types.Recommendations(items=[])
    assert semantic_types.UserProfile(user_id="u1", memory_summary={"likes": ["rag"]})


def test_every_contract_declares_only_known_semantic_types() -> None:
    known = set(semantic_types.SEMANTIC_TYPES)
    assert len(known) == 11
    assert set(semantic_types.SEMANTIC_TYPE_DESCRIPTIONS) == known
    registry = tool_registry.UNIFIED_TOOL_REGISTRY
    for contract in (registry.get_contract(tool.tool_name) for tool in registry.list_tools()):
        declared = {item.semantic_type for item in contract.consumes} | set(contract.produces)
        assert declared <= known, contract.tool_name


def test_quality_gate_tools_are_no_longer_registered() -> None:
    registry = tool_registry.UNIFIED_TOOL_REGISTRY
    for tool_name in ("validate_arxiv_results", "validate_recommendations", "verify_preference_update"):
        assert registry.get_contract(tool_name) is None
    assert "search_empty" in registry.get_contract("search_arxiv").failure_modes
    assert "recommendation_empty" in registry.get_contract("generate_recommendations").failure_modes


def test_registry_rejects_unknown_semantic_type() -> None:
    registry = tool_registry.ToolRegistry()
    contract = tool_registry.UNIFIED_TOOL_REGISTRY.get_contract("search_arxiv")
    with pytest.raises(ValueError, match="PaperSet"):
        registry.register(dataclasses.replace(contract, produces=("PaperSet",)))


# ---- state 投影 ----


def test_project_state_types_maps_context_to_types() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="第二篇讲了什么",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG"),
        context={
            "last_papers": [
                {"arxiv_id": "2401.00001", "title": "A"},
                {"arxiv_id": "2401.00002", "title": "B"},
            ],
            "research_profile": {"topics": ["rag"]},
        },
    )

    projected = project_state_types(state, paper_ref={"kind": "list_index", "value": 2})

    assert set(projected) == {"SearchSpec", "PaperList", "ResolvedPaper", "UserProfile"}
    assert projected["PaperList"].value.origin == "context"
    assert projected["ResolvedPaper"].value.arxiv_id == "2401.00002"
    assert projected["ResolvedPaper"].source == "paper_ref.list_index"
    assert "IndexedPaper" not in projected


def test_project_state_types_skips_values_that_break_invariants() -> None:
    state = AgentState(
        intent="paper_qa",
        message="这篇讲了什么",
        context={
            "last_papers": [{"title": "no arxiv id"}],
            "selected_paper": {"title": "still no arxiv id"},
            "user_memory_summary": "",
        },
    )

    projected = project_state_types(state, paper_ref={"kind": "list_index", "value": 3})

    assert projected == {}


# ---- 闭合检查 ----


@pytest.mark.parametrize(
    "state",
    [
        AgentState(intent="arxiv_search", message="search rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
        AgentState(
            intent="arxiv_search",
            message="search rag",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG"),
            context={"user_memory_summary": {"likes": ["retrieval"]}},
        ),
        AgentState(intent="recommendation", message="recommend rag papers"),
        AgentState(
            intent="preference_action",
            message="喜欢这篇论文",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
        ),
    ],
    ids=["search", "search_with_profile", "recommendation", "preference"],
)
def test_rule_based_plans_are_closed(state: AgentState) -> None:
    report = check_plan_closure(_rule_plan(state), project_state_types(state))

    assert report.ok, report.errors
    assert report.final_types == ["AnswerDraft"]


def test_current_paper_qa_plan_is_not_closed_until_answer_consumes_indexed_paper() -> None:
    # 已知缺口：answer_paper_question 声明消费 IndexedPaper，但现有计划的 paper_ref 仍绑定 resolve_paper，
    # 要等第 3 步把 QA 链改成从 check_paper_index / parse_and_index_paper 取论文。
    state = AgentState(
        intent="paper_qa",
        message="方法是什么",
        context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
    )

    report = check_plan_closure(_rule_plan(state), project_state_types(state))

    assert not report.ok
    assert all(error.startswith("answer_paper_question") for error in report.errors)
    assert report.final_types == ["CheckedQAResult"]


def test_closure_rejects_wrong_producer_and_missing_state_type() -> None:
    steps = [
        _step("load_user_profile", "load_user_profile"),
        _step(
            "synthesize_arxiv_response",
            "synthesize_arxiv_response",
            [{"input_key": "arxiv_results", "source_type": "step_output", "step_id": "load_user_profile"}],
        ),
    ]

    report = check_plan_closure(steps, available=[])

    assert not report.ok
    assert any("需要 PaperList" in error and "UserProfile" in error for error in report.errors)

    from_state = [
        _step(
            "synthesize_arxiv_response",
            "synthesize_arxiv_response",
            [{"input_key": "arxiv_results", "source_type": "context", "source_key": "last_papers"}],
        )
    ]
    assert not check_plan_closure(from_state, available=[]).ok
    assert check_plan_closure(from_state, available=["PaperList"]).ok


def test_closure_requires_a_final_answer_type() -> None:
    report = check_plan_closure([_step("load_user_profile", "load_user_profile")], available=[])

    assert not report.ok
    assert any("最终回复类型" in error for error in report.errors)
