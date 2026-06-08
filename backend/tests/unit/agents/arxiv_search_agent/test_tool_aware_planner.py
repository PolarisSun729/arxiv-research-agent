from __future__ import annotations

import importlib
import json
import sys

import pytest

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
tool_aware_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_aware_planner")
planner_registry_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")
validator_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_validator")
executor_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_executor")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
Goal = schemas.Goal
PlanDraft = schemas.PlanDraft
PlanDraftStep = schemas.PlanDraftStep
ToolCandidateSelection = schemas.ToolCandidateSelection
PLANNER_TOOL_REGISTRY = planner_registry_module.PLANNER_TOOL_REGISTRY
PlanDraftConverter = tool_aware_module.PlanDraftConverter
PlanDraftConversionError = tool_aware_module.PlanDraftConversionError
PlanValidator = validator_module.PlanValidator
PlanExecutor = executor_module.PlanExecutor
RuleBasedToolAwarePlanBuilder = tool_aware_module.RuleBasedToolAwarePlanBuilder
ToolCandidateSelector = tool_aware_module.ToolCandidateSelector


def _current_modules():
    """测试 helper 会重载 Agent 模块；这里每次取当前模块，避免 monkeypatch 打到旧类实例。"""
    current_planner = sys.modules.get("backend.agents.arxiv_search_agent.planner") or planner_module
    current_tool_aware = sys.modules.get("backend.agents.arxiv_search_agent.tool_aware_planner") or tool_aware_module
    current_registry = sys.modules.get("backend.agents.arxiv_search_agent.tool_registry") or planner_registry_module
    return current_planner, current_tool_aware, current_registry


class _FakeLLMPlanService:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"response": self.payload}


def _candidate_tool_names(goal_type: str) -> set[str]:
    _, current_tool_aware, current_registry = _current_modules()
    selection = current_tool_aware.ToolCandidateSelector(current_registry.PLANNER_TOOL_REGISTRY).select(
        Goal(goal_type=goal_type, intent=goal_type),
        AgentState(intent=goal_type, message="test"),
    )
    return {tool.tool_name for tool in selection.candidate_tools}


def test_arxiv_search_goal_selects_search_related_candidate_tools() -> None:
    tool_names = _candidate_tool_names("arxiv_search")

    assert {"build_arxiv_search_spec", "search_arxiv", "validate_arxiv_results", "personalize_paper_results", "synthesize_arxiv_response"}.issubset(tool_names)


def test_paper_qa_goal_selects_index_check_and_answer_tools() -> None:
    tool_names = _candidate_tool_names("paper_qa")

    assert {"resolve_paper", "check_paper_index", "parse_and_index_paper", "answer_paper_question"}.issubset(tool_names)


def test_unsupported_goal_excludes_search_and_write_business_tools() -> None:
    tool_names = _candidate_tool_names("unsupported")

    assert "generate_fallback_response" in tool_names
    assert not {"search_arxiv", "update_preference_store", "parse_and_index_paper"}.intersection(tool_names)


def test_plan_draft_unknown_tool_conversion_fails() -> None:
    draft = PlanDraft(
        draft_id="draft:test",
        plan_intent="arxiv_search",
        selected_tools=["missing_tool"],
        steps=[PlanDraftStep(step_id="missing", action_type="search", tool_name="missing_tool")],
    )

    with pytest.raises(PlanDraftConversionError, match="Unknown planner tool"):
        PlanDraftConverter(PLANNER_TOOL_REGISTRY).convert(draft, Goal(goal_type="arxiv_search"))


def test_plan_draft_duplicate_step_id_conversion_fails() -> None:
    draft = PlanDraft(
        draft_id="draft:test",
        plan_intent="arxiv_search",
        selected_tools=["search_arxiv"],
        steps=[
            PlanDraftStep(step_id="search", action_type="search", tool_name="search_arxiv", expected_output_key="first"),
            PlanDraftStep(step_id="search", action_type="search", tool_name="search_arxiv", expected_output_key="second"),
        ],
    )

    with pytest.raises(PlanDraftConversionError, match="Duplicate draft step_id"):
        PlanDraftConverter(PLANNER_TOOL_REGISTRY).convert(draft, Goal(goal_type="arxiv_search"))


def test_plan_draft_missing_dependency_conversion_fails() -> None:
    draft = PlanDraft(
        draft_id="draft:test",
        plan_intent="arxiv_search",
        selected_tools=["search_arxiv"],
        steps=[
            PlanDraftStep(
                step_id="search",
                action_type="search",
                tool_name="search_arxiv",
                depends_on=["normalize_request"],
            )
        ],
    )

    with pytest.raises(PlanDraftConversionError, match="depends on missing step"):
        PlanDraftConverter(PLANNER_TOOL_REGISTRY).convert(draft, Goal(goal_type="arxiv_search"))


def test_confirmation_required_tool_gets_confirmation_policy() -> None:
    draft = PlanDraft(
        draft_id="draft:test",
        plan_intent="paper_qa",
        selected_tools=["parse_and_index_paper"],
        steps=[
            PlanDraftStep(
                step_id="parse_and_index_paper",
                action_type="index",
                tool_name="parse_and_index_paper",
                input_bindings=[
                    {
                        "input_key": "paper_reference",
                        "source_type": "literal",
                        "value": {"arxiv_id": "2401.00001"},
                    }
                ],
                expected_output_key="index_build_result",
            )
        ],
    )

    plan = PlanDraftConverter(PLANNER_TOOL_REGISTRY).convert(draft, Goal(goal_type="paper_qa"))

    assert plan.steps[0].confirmation_policy is not None
    assert plan.steps[0].confirmation_policy.requires_confirmation is True
    assert plan.steps[0].side_effect_level == "external_call"


def test_plan_draft_missing_input_bindings_conversion_fails() -> None:
    draft = PlanDraft(
        draft_id="draft:test",
        plan_intent="arxiv_search",
        selected_tools=["search_arxiv"],
        steps=[
            PlanDraftStep(
                step_id="search_arxiv",
                action_type="search",
                tool_name="search_arxiv",
                expected_output_key="arxiv_results",
            )
        ],
    )

    with pytest.raises(PlanDraftConversionError, match="missing required input bindings"):
        PlanDraftConverter(PLANNER_TOOL_REGISTRY).convert(draft, Goal(goal_type="arxiv_search"))


def _build_tool_aware_plan(intent: str, *, state: AgentState | None = None):
    current_planner, _, _ = _current_modules()
    return current_planner.build_executable_plan(
        state or AgentState(intent=intent, message="test"),
        enable_tool_aware_planner=True,
    )


def _tool_names(plan) -> list[str]:
    return [step.tool_name for step in list(plan.steps or [])]


def _step_ids(plan) -> list[str]:
    return [step.step_id for step in list(plan.steps or [])]


def _llm_arxiv_plan_json(**overrides) -> str:
    payload = {
        "draft_id": "llm:arxiv:test",
        "plan_intent": "arxiv_search",
        "selected_tools": [
            "normalize_request",
            "build_arxiv_search_spec",
            "search_arxiv",
            "validate_arxiv_results",
            "synthesize_arxiv_response",
        ],
        "steps": [
            {
                "step_id": "normalize_request",
                "action_type": "write_state",
                "tool_name": "normalize_request",
                "step_reason": "normalize request for downstream search",
                "input_bindings": [
                    {"input_key": "intent", "source_type": "state", "source_key": "intent"},
                    {"input_key": "message", "source_type": "state", "source_key": "message"},
                    {"input_key": "search_spec", "source_type": "search_spec"},
                ],
                "depends_on": [],
                "expected_output_key": "normalized_request",
                "risk_level": "low",
                "requires_confirmation": False,
            },
            {
                "step_id": "build_arxiv_search_spec",
                "action_type": "search",
                "tool_name": "build_arxiv_search_spec",
                "step_reason": "build structured search spec",
                "input_bindings": [
                    {"input_key": "normalized_request", "source_type": "step_output", "step_id": "normalize_request"}
                ],
                "depends_on": ["normalize_request"],
                "expected_output_key": "search_spec",
                "risk_level": "low",
                "requires_confirmation": False,
            },
            {
                "step_id": "search_arxiv",
                "action_type": "search",
                "tool_name": "search_arxiv",
                "step_reason": "search arxiv using candidate search tool",
                "input_bindings": [
                    {"input_key": "search_spec", "source_type": "step_output", "step_id": "build_arxiv_search_spec"}
                ],
                "depends_on": ["build_arxiv_search_spec"],
                "expected_output_key": "arxiv_results",
                "risk_level": "medium",
                "requires_confirmation": False,
            },
            {
                "step_id": "validate_arxiv_results",
                "action_type": "validate",
                "tool_name": "validate_arxiv_results",
                "step_reason": "validate search quality",
                "input_bindings": [
                    {"input_key": "arxiv_results", "source_type": "step_output", "step_id": "search_arxiv"}
                ],
                "depends_on": ["search_arxiv"],
                "expected_output_key": "arxiv_result_quality",
                "risk_level": "low",
                "requires_confirmation": False,
            },
            {
                "step_id": "synthesize_arxiv_response",
                "action_type": "answer",
                "tool_name": "synthesize_arxiv_response",
                "step_reason": "produce final user answer",
                "input_bindings": [
                    {"input_key": "arxiv_result_quality", "source_type": "step_output", "step_id": "validate_arxiv_results"}
                ],
                "depends_on": ["validate_arxiv_results"],
                "expected_output_key": "final_answer",
                "risk_level": "low",
                "requires_confirmation": False,
            },
        ],
        "metadata": {"source": "test"},
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_llm_valid_plan_draft_converts_to_executable_plan() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="search rag",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG"),
    )

    _, plan, debug = _current_modules()[0].build_executable_plan(
        state,
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService(_llm_arxiv_plan_json()),
    )

    assert debug["planner_mode"] == "tool_aware_llm"
    assert debug["llm_plan_attempted"] is True
    assert debug["llm_plan_valid"] is True
    assert debug["selected_plan_source"] == "llm_tool_aware"
    assert _step_ids(plan) == [
        "normalize_request",
        "build_arxiv_search_spec",
        "search_arxiv",
        "validate_arxiv_results",
        "synthesize_arxiv_response",
    ]
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_llm_unknown_tool_falls_back_to_rule_based_planner() -> None:
    invalid = json.loads(_llm_arxiv_plan_json())
    invalid["steps"][2]["tool_name"] = "missing_tool"
    invalid["selected_tools"][2] = "missing_tool"

    _, plan, debug = _current_modules()[0].build_executable_plan(
        AgentState(intent="arxiv_search", message="search rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService(json.dumps(invalid, ensure_ascii=False)),
    )

    assert debug["llm_plan_valid"] is False
    assert debug["rule_based_fallback_used"] is True
    assert debug["selected_plan_source"] == "tool_aware_rule_based"
    assert "not registered" in debug["llm_plan_invalid_reasons"][0] or "outside candidate" in debug["llm_plan_invalid_reasons"][0]
    assert "search_arxiv" in _tool_names(plan)


def test_llm_candidate_outside_tool_falls_back() -> None:
    invalid = json.loads(_llm_arxiv_plan_json())
    invalid["steps"][2]["tool_name"] = "update_preference_store"
    invalid["selected_tools"][2] = "update_preference_store"

    _, _, debug = _current_modules()[0].build_executable_plan(
        AgentState(intent="arxiv_search", message="search rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService(json.dumps(invalid, ensure_ascii=False)),
    )

    assert debug["llm_plan_valid"] is False
    assert debug["rule_based_fallback_used"] is True
    assert "outside candidate tools" in debug["llm_plan_invalid_reasons"][0]


def test_llm_duplicate_step_id_falls_back() -> None:
    invalid = json.loads(_llm_arxiv_plan_json())
    invalid["steps"][1]["step_id"] = "normalize_request"

    _, _, debug = _current_modules()[0].build_executable_plan(
        AgentState(intent="arxiv_search", message="search rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService(json.dumps(invalid, ensure_ascii=False)),
    )

    assert debug["llm_plan_valid"] is False
    assert "duplicate step_id" in debug["llm_plan_invalid_reasons"][0]


def test_llm_invalid_depends_on_falls_back() -> None:
    invalid = json.loads(_llm_arxiv_plan_json())
    invalid["steps"][2]["depends_on"] = ["missing_step"]

    _, _, debug = _current_modules()[0].build_executable_plan(
        AgentState(intent="arxiv_search", message="search rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService(json.dumps(invalid, ensure_ascii=False)),
    )

    assert debug["llm_plan_valid"] is False
    assert "depends on missing" in debug["llm_plan_invalid_reasons"][0]


def test_llm_non_json_falls_back() -> None:
    _, _, debug = _current_modules()[0].build_executable_plan(
        AgentState(intent="arxiv_search", message="search rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService("Here is a plan: use search."),
    )

    assert debug["llm_plan_valid"] is False
    assert debug["selected_plan_source"] == "tool_aware_rule_based"
    assert "JSON-only" in debug["llm_plan_invalid_reasons"][0]


def test_llm_persistent_write_without_target_falls_back_to_clarification() -> None:
    preference_plan = {
        "draft_id": "llm:preference:test",
        "plan_intent": "preference_action",
        "selected_tools": ["resolve_preference_target", "update_preference_store", "verify_preference_update", "synthesize_preference_response"],
        "steps": [
            {
                "step_id": "resolve_preference_target",
                "action_type": "retrieve",
                "tool_name": "resolve_preference_target",
                "step_reason": "resolve target",
                "input_bindings": [{"input_key": "message", "source_type": "state", "source_key": "message"}],
                "depends_on": [],
                "expected_output_key": "paper_reference",
                "risk_level": "low",
                "requires_confirmation": False,
            },
            {
                "step_id": "update_preference_store",
                "action_type": "write_state",
                "tool_name": "update_preference_store",
                "step_reason": "write preference",
                "input_bindings": [
                    {"input_key": "paper_reference", "source_type": "step_output", "step_id": "resolve_preference_target"},
                    {"input_key": "message", "source_type": "state", "source_key": "message"},
                ],
                "depends_on": ["resolve_preference_target"],
                "expected_output_key": "preference_action_result",
                "risk_level": "high",
                "requires_confirmation": False,
            },
            {
                "step_id": "verify_preference_update",
                "action_type": "validate",
                "tool_name": "verify_preference_update",
                "step_reason": "verify preference",
                "input_bindings": [{"input_key": "preference_action_result", "source_type": "step_output", "step_id": "update_preference_store"}],
                "depends_on": ["update_preference_store"],
                "expected_output_key": "verified_preference_update",
                "risk_level": "low",
                "requires_confirmation": False,
            },
            {
                "step_id": "synthesize_preference_response",
                "action_type": "answer",
                "tool_name": "synthesize_preference_response",
                "step_reason": "answer user",
                "input_bindings": [{"input_key": "verified_preference_update", "source_type": "step_output", "step_id": "verify_preference_update"}],
                "depends_on": ["verify_preference_update"],
                "expected_output_key": "final_answer",
                "risk_level": "low",
                "requires_confirmation": False,
            },
        ],
    }

    _, plan, debug = _current_modules()[0].build_executable_plan(
        AgentState(intent="preference_action", message="我喜欢这篇论文"),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService(json.dumps(preference_plan, ensure_ascii=False)),
    )

    assert debug["llm_plan_valid"] is False
    assert "requires a clear target" in debug["llm_plan_invalid_reasons"][0]
    assert _tool_names(plan) == ["analyze_ambiguity", "generate_clarification"]


def test_llm_high_risk_tool_missing_confirmation_is_auto_completed() -> None:
    preference_plan = json.loads(_llm_arxiv_plan_json(plan_intent="preference_action"))
    preference_plan["selected_tools"] = ["resolve_preference_target", "update_preference_store", "verify_preference_update", "synthesize_preference_response"]
    preference_plan["steps"] = [
        {
            "step_id": "resolve_preference_target",
            "action_type": "retrieve",
            "tool_name": "resolve_preference_target",
            "step_reason": "resolve target",
            "input_bindings": [{"input_key": "message", "source_type": "state", "source_key": "message"}],
            "depends_on": [],
            "expected_output_key": "paper_reference",
            "risk_level": "low",
            "requires_confirmation": False,
        },
        {
            "step_id": "update_preference_store",
            "action_type": "write_state",
            "tool_name": "update_preference_store",
            "step_reason": "write preference after target resolution",
            "input_bindings": [
                {"input_key": "paper_reference", "source_type": "step_output", "step_id": "resolve_preference_target"},
                {"input_key": "message", "source_type": "state", "source_key": "message"},
            ],
            "depends_on": ["resolve_preference_target"],
            "expected_output_key": "preference_action_result",
            "risk_level": "high",
            "requires_confirmation": False,
        },
        {
            "step_id": "verify_preference_update",
            "action_type": "validate",
            "tool_name": "verify_preference_update",
            "step_reason": "verify write",
            "input_bindings": [{"input_key": "preference_action_result", "source_type": "step_output", "step_id": "update_preference_store"}],
            "depends_on": ["update_preference_store"],
            "expected_output_key": "verified_preference_update",
            "risk_level": "low",
            "requires_confirmation": False,
        },
        {
            "step_id": "synthesize_preference_response",
            "action_type": "answer",
            "tool_name": "synthesize_preference_response",
            "step_reason": "answer user",
            "input_bindings": [{"input_key": "verified_preference_update", "source_type": "step_output", "step_id": "verify_preference_update"}],
            "depends_on": ["verify_preference_update"],
            "expected_output_key": "final_answer",
            "risk_level": "low",
            "requires_confirmation": False,
        },
    ]

    _, plan, debug = _current_modules()[0].build_executable_plan(
        AgentState(
            intent="preference_action",
            message="喜欢这篇论文",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
        ),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService(json.dumps(preference_plan, ensure_ascii=False)),
    )

    update_step = next(step for step in plan.steps if step.step_id == "update_preference_store")
    assert debug["llm_plan_valid"] is True
    assert update_step.confirmation_policy is not None
    assert "update_preference_store" in debug["confirmation_required_steps"]


def test_rule_based_failure_after_llm_failure_falls_back_to_fixed_template(monkeypatch) -> None:
    current_planner, current_tool_aware, _ = _current_modules()

    def fail_rule_builder(self, *_args, **_kwargs):
        raise RuntimeError("rule builder failed")

    monkeypatch.setattr(current_tool_aware.RuleBasedToolAwarePlanBuilder, "build", fail_rule_builder)

    _, plan, debug = current_planner.build_executable_plan(
        AgentState(intent="arxiv_search", message="search rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG")),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=_FakeLLMPlanService("not json"),
    )

    assert debug["template_fallback_used"] is True
    assert debug["selected_plan_source"] == "fixed_template_fallback"
    assert [step.step_id for step in plan.steps][:2] == ["normalize_request", "build_arxiv_search_spec"]
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_rule_based_arxiv_search_generates_valid_executable_plan() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="search rag",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG evaluation"),
        context={"user_memory_summary": {"likes": ["retrieval"]}},
    )

    _, plan, debug = _build_tool_aware_plan("arxiv_search", state=state)

    assert debug["planner_mode"] == "tool_aware_rule_based"
    assert debug["final_plan_source"] == "tool_aware_rule_based"
    assert _step_ids(plan) == [
        "normalize_request",
        "build_arxiv_search_spec",
        "search_arxiv",
        "validate_arxiv_results",
        "personalize_paper_results",
        "synthesize_arxiv_response",
    ]
    assert plan.steps[2].retry_policy is not None
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_tool_aware_planner_disabled_uses_fixed_template_plan() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="search rag",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG evaluation"),
    )

    _, plan, debug = _current_modules()[0].build_executable_plan(
        state,
        enable_tool_aware_planner=False,
    )

    assert debug["planner_mode"] == "fixed_template"
    assert debug["final_plan_source"] == "fixed_template"
    assert _step_ids(plan) == [
        "normalize_request",
        "build_arxiv_search_spec",
        "search_arxiv",
        "validate_arxiv_results",
        "personalize_paper_results",
        "synthesize_arxiv_response",
    ]


def test_rule_based_paper_qa_generates_resolve_check_answer_plan() -> None:
    _, plan, debug = _build_tool_aware_plan(
        "paper_qa",
        state=AgentState(
            intent="paper_qa",
            message="method?",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
        ),
    )

    assert _step_ids(plan) == ["resolve_paper", "check_paper_index", "answer_paper_question"]
    assert _tool_names(plan) == ["resolve_paper", "check_paper_index", "answer_paper_question"]
    assert any(item["step_id"] == "parse_and_index_paper" for item in debug["skipped_steps"])
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_rule_based_recommendation_generates_full_plan() -> None:
    _, plan, _ = _build_tool_aware_plan("recommendation", state=AgentState(intent="recommendation", message="recommend rag papers"))

    assert _step_ids(plan) == [
        "load_user_profile",
        "load_candidate_papers",
        "generate_recommendations",
        "validate_recommendations",
        "explain_recommendations",
    ]
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_rule_based_preference_action_generates_persistent_write_plan() -> None:
    _, plan, debug = _build_tool_aware_plan(
        "preference_action",
        state=AgentState(
            intent="preference_action",
            message="喜欢这篇论文",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
        ),
    )

    assert _step_ids(plan) == [
        "resolve_preference_target",
        "update_preference_store",
        "verify_preference_update",
        "synthesize_preference_response",
    ]
    assert plan.steps[1].side_effect_level == "persistent_write"
    assert any(item["tool_name"] == "update_preference_store" and item["side_effect_level"] == "persistent_write" for item in debug["selected_steps"])
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_rule_based_preference_action_without_target_clarifies_instead_of_write() -> None:
    _, plan, debug = _build_tool_aware_plan(
        "preference_action",
        state=AgentState(intent="preference_action", message="我喜欢这篇论文"),
    )

    assert _tool_names(plan) == ["analyze_ambiguity", "generate_clarification"]
    assert "update_preference_store" not in _tool_names(plan)
    assert any(item["step_id"] == "update_preference_store" for item in debug["skipped_steps"])
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_rule_based_unclear_only_generates_clarification_plan() -> None:
    _, plan, _ = _build_tool_aware_plan("unclear", state=AgentState(intent="unclear", message="这个东西是什么"))

    assert _tool_names(plan) == ["analyze_ambiguity", "generate_clarification"]
    assert not {"search_arxiv", "answer_paper_question", "update_preference_store"}.intersection(_tool_names(plan))
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_rule_based_unsupported_only_generates_fallback_plan() -> None:
    _, plan, _ = _build_tool_aware_plan("unsupported", state=AgentState(intent="unsupported", message="帮我做一个系统不支持的任务"))

    assert _tool_names(plan) == ["generate_fallback_response"]
    PlanValidator().validate(plan, PLANNER_TOOL_REGISTRY)


def test_rule_based_optional_personalize_missing_does_not_fail(monkeypatch) -> None:
    _, current_tool_aware, _ = _current_modules()
    original_select = current_tool_aware.ToolCandidateSelector.select

    def fake_select(self, goal, state):
        selection = original_select(self, goal, state)
        return selection.model_copy(
            update={
                "candidate_tools": [
                    tool for tool in selection.candidate_tools if tool.tool_name != "personalize_paper_results"
                ]
            }
        )

    monkeypatch.setattr(current_tool_aware.ToolCandidateSelector, "select", fake_select)

    _, plan, debug = _build_tool_aware_plan(
        "arxiv_search",
        state=AgentState(
            intent="arxiv_search",
            message="search rag",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG"),
            context={"user_memory_summary": {"likes": ["retrieval"]}},
        ),
    )

    assert "personalize_paper_results" not in _step_ids(plan)
    assert "synthesize_arxiv_response" in _step_ids(plan)
    assert any(item["step_id"] == "personalize_paper_results" for item in debug["skipped_steps"])


def test_tool_aware_planning_falls_back_to_fixed_template_when_required_tool_missing(monkeypatch) -> None:
    current_planner, current_tool_aware, _ = _current_modules()
    current_schemas = sys.modules.get("backend.agents.arxiv_search_agent.schemas") or schemas
    original_select = current_tool_aware.ToolCandidateSelector.select

    def fake_select(self, goal, state):
        selection = original_select(self, goal, state)
        return current_schemas.ToolCandidateSelection(
            goal_type=selection.goal_type,
            candidate_tools=[tool for tool in selection.candidate_tools if tool.tool_name != "search_arxiv"],
            excluded_tools=selection.excluded_tools,
            selection_reason=selection.selection_reason,
            risk_summary=selection.risk_summary,
        )

    monkeypatch.setattr(current_tool_aware.ToolCandidateSelector, "select", fake_select)

    _, plan, debug = current_planner.build_executable_plan(
        AgentState(intent="arxiv_search", message="search rag"),
        enable_tool_aware_planner=True,
    )

    assert debug["planner_mode"] == "tool_aware_rule_based"
    assert debug["fallback_used"] is True
    assert debug["validation_status"] == "failed"
    assert debug["final_plan_source"] == "fixed_template_fallback"
    assert "missing required tool" in debug["fallback_reason"]
    assert [step.step_id for step in plan.steps][:2] == ["normalize_request", "build_arxiv_search_spec"]


def test_rule_based_plan_can_be_executed_by_existing_plan_executor(monkeypatch) -> None:
    current_executor = sys.modules.get("backend.agents.arxiv_search_agent.plan_executor") or executor_module
    current_schemas = sys.modules.get("backend.agents.arxiv_search_agent.schemas") or schemas
    current_state_module = sys.modules.get("backend.agents.arxiv_search_agent.state") or state_module

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG evaluation"}]},
            "trace": {"tool_name": tool_name, "query": kwargs.get("query")},
            "error": None,
        }

    monkeypatch.setattr(current_executor, "invoke_backend_tool", fake_invoke_tool)
    state = current_state_module.AgentState(
        intent="arxiv_search",
        message="search rag",
        search_spec=current_schemas.ArxivSearchSpec(intent="arxiv_search", query="RAG"),
    )
    _, plan, _ = _build_tool_aware_plan("arxiv_search", state=state)

    result = current_executor.PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert result.final_answer
    assert any(trace.step_id == "search_arxiv" and trace.event == "step_succeeded" for trace in result.trace)
