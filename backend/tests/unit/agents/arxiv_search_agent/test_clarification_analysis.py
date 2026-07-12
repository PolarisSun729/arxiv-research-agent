from __future__ import annotations

import sys

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
clarification_analysis_module = sys.modules["backend.agents.arxiv_search_agent.clarification_analysis"]
clarification_adapter_module = sys.modules["backend.agents.arxiv_search_agent.tool_adapters.clarification"]
planner_module = sys.modules["backend.agents.arxiv_search_agent.planner"]
state_module = _MODULES["state_module"]

build_clarification_diagnostic = clarification_analysis_module.build_clarification_diagnostic
AnalyzeAmbiguityAdapter = clarification_adapter_module.AnalyzeAmbiguityAdapter
AnalyzeAmbiguityInput = clarification_adapter_module.AnalyzeAmbiguityInput
GenerateClarificationAdapter = clarification_adapter_module.GenerateClarificationAdapter
GenerateClarificationInput = clarification_adapter_module.GenerateClarificationInput
AgentState = state_module.AgentState


def test_recommendation_without_identity_or_constraints_reports_structured_gaps() -> None:
    diagnostic = build_clarification_diagnostic(message="帮我推荐几篇论文")

    assert diagnostic["inferred_intent"] == "recommendation"
    assert diagnostic["needs_clarification"] is True
    assert {"recommendation_constraints", "user_identity"}.issubset(set(diagnostic["missing_fields"]))
    assert "scope_constraint" in diagnostic["missing_fields"]
    assert diagnostic["minimum_required_fields"] == ["user_identity_or_profile_or_constraints"]


def test_paper_detail_without_context_reports_missing_target_paper() -> None:
    diagnostic = build_clarification_diagnostic(message="这篇论文的方法是什么？")

    assert diagnostic["inferred_intent"] == "paper_detail"
    assert diagnostic["needs_clarification"] is True
    assert diagnostic["missing_fields"] == ["target_paper"]
    assert "目标论文" in diagnostic["missing_field_details"][0]["reason"]


def test_search_with_explicit_topic_can_continue_without_clarification() -> None:
    diagnostic = build_clarification_diagnostic(message="帮我搜一下 RAG")

    assert diagnostic["inferred_intent"] == "arxiv_search"
    assert diagnostic["needs_clarification"] is False
    assert diagnostic["missing_fields"] == []
    assert diagnostic["resolution_strategy"] == "continue_without_clarification"


def test_preference_action_without_target_reports_preference_target_gap() -> None:
    diagnostic = build_clarification_diagnostic(message="我喜欢这个")

    assert diagnostic["inferred_intent"] == "preference_action"
    assert {"preference_target", "user_identity"}.issubset(set(diagnostic["missing_fields"]))
    assert diagnostic["minimum_required_fields"] == ["preference_action", "preference_target", "user_identity"]


def test_confirmation_message_requires_structured_interaction_resume() -> None:
    diagnostic = build_clarification_diagnostic(message="同意")

    assert diagnostic["inferred_intent"] == "confirmation"
    assert diagnostic["needs_clarification"] is True
    assert diagnostic["missing_fields"] == ["interaction_resume"]
    assert diagnostic["reason"] == "structured_interaction_resume_required"


def test_recommendation_with_topic_can_use_default_identity() -> None:
    diagnostic = build_clarification_diagnostic(message="推荐一些 RAG 论文")

    assert diagnostic["inferred_intent"] == "recommendation"
    assert diagnostic["needs_clarification"] is False
    assert diagnostic["allow_default_continuation"] is True
    assert diagnostic["default_values"]["user_id"]["source"] == "agent_default_user_context"
    assert diagnostic["missing_fields"] == ["user_identity"]


def test_analyze_ambiguity_adapter_returns_structured_diagnostic() -> None:
    adapter = AnalyzeAmbiguityAdapter()
    result = adapter.execute(
        AnalyzeAmbiguityInput(
            message="这篇论文的方法是什么？",
            context={},
            goal={"intent": "paper_detail"},
        )
    )

    assert result.ok is True
    assert result.data is not None
    assert result.data.inferred_intent == "paper_detail"
    assert result.data.missing_fields == ["target_paper"]
    assert result.metadata["analysis_mode"] == "deterministic_request_diagnostic"


def test_generate_clarification_prefers_specific_question_and_default_hint() -> None:
    adapter = GenerateClarificationAdapter()
    result = adapter.execute(
        GenerateClarificationInput(
            missing_information={
                "support_status": "supported",
                "suggested_questions": ["请告诉我你想围绕什么主题推荐论文。"],
                "allow_default_continuation": True,
                "default_values": {"user_id": {"source": "agent_default_user_context"}},
            }
        )
    )

    assert result.ok is True
    assert result.data is not None
    assert "围绕什么主题推荐论文" in result.data.final_answer
    assert "默认值来源" in result.data.final_answer


def test_rule_based_clarification_plan_binds_context_for_diagnostic() -> None:
    goal, plan, _ = planner_module.build_executable_plan(
        AgentState(
            intent="unclear",
            message="这篇论文的方法是什么？",
            context={"selected_paper": {"arxiv_id": "2401.00001"}},
        )
    )

    assert goal.goal_type == "unclear"
    analyze_step = plan.steps[0]
    input_keys = {binding.input_key for binding in analyze_step.input_bindings}
    assert {"message", "context", "user_id", "search_spec", "goal"}.issubset(input_keys)
