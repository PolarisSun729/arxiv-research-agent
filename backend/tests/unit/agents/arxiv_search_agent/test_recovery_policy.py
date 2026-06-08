from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

observer_module = importlib.import_module("backend.agents.arxiv_search_agent.observer")
planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
recovery_policy_module = importlib.import_module("backend.agents.arxiv_search_agent.recovery_policy")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
ObservationResult = schemas.ObservationResult
PlanRuntime = schemas.PlanRuntime

Observer = observer_module.Observer
DEFAULT_RECOVERY_POLICY_REGISTRY = recovery_policy_module.DEFAULT_RECOVERY_POLICY_REGISTRY


def _build_runtime_and_step(state: AgentState, tool_name: str):
    goal, plan, _ = planner_module.build_executable_plan(state)
    step = next(item for item in list(plan.steps or []) if item.tool_name == tool_name)
    return PlanRuntime(goal=goal, plan=plan), step


def _observe(state: AgentState, tool_name: str, *, raw_output, normalized_output=None, resolved_input=None):
    runtime, step = _build_runtime_and_step(state, tool_name)
    observation = Observer().observe(
        step=step,
        resolved_input=resolved_input or {},
        raw_output=raw_output,
        normalized_output=normalized_output if normalized_output is not None else raw_output,
        runtime=runtime,
        state=state,
    )
    return runtime, step, observation


def test_observer_classifies_empty_search_result() -> None:
    _, _, observation = _observe(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
        raw_output={"papers": []},
    )

    assert observation.status == "empty_result"
    assert observation.observation_signal == "success_but_empty_result"
    assert observation.failure_category == "search_empty"
    assert observation.recoverable is True
    assert "patch_plan" in observation.suggested_recovery_types


def test_observer_classifies_low_confidence_validation_result() -> None:
    _, _, observation = _observe(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "validate_arxiv_results",
        raw_output={"result_count": 2, "warnings": ["tool_failed"]},
    )

    assert observation.status == "low_confidence"
    assert observation.failure_category == "search_low_confidence"
    assert observation.retryable is True


def test_observer_classifies_missing_paper_index_as_user_input_recovery() -> None:
    _, _, observation = _observe(
        AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}}),
        "check_paper_index",
        raw_output={"status": "missing", "has_index": False},
    )

    assert observation.status == "need_confirmation"
    assert observation.failure_category == "paper_index_missing"
    assert observation.requires_user_input is True


def test_observer_classifies_empty_user_profile() -> None:
    _, _, observation = _observe(
        AgentState(intent="recommendation", message="recommend rag papers", context={}),
        "load_user_profile",
        raw_output={"research_profile": {}, "user_memory_summary": None},
    )

    assert observation.status == "empty_result"
    assert observation.failure_category in {"empty_user_profile", "insufficient_context"}
    assert observation.recoverable is True


def test_recovery_policies_generate_explainable_candidates() -> None:
    cases = [
        (
            AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
            "search_arxiv",
            ObservationResult(status="empty_result", reason="empty", failure_category="search_empty"),
        ),
        (
            AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
            "validate_arxiv_results",
            ObservationResult(status="low_confidence", reason="low", failure_category="search_low_confidence"),
        ),
        (
            AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}}),
            "check_paper_index",
            ObservationResult(status="need_confirmation", reason="missing", failure_category="paper_index_missing"),
        ),
        (
            AgentState(intent="recommendation", message="recommend rag papers", context={}),
            "load_user_profile",
            ObservationResult(status="empty_result", reason="empty_profile", failure_category="empty_user_profile"),
        ),
    ]

    for state, tool_name, observation in cases:
        runtime, step = _build_runtime_and_step(state, tool_name)
        candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(step=step, observation=observation, runtime=runtime, state=state)

        assert candidates
        assert all(candidate.action_type and candidate.reason and candidate.target_step_id for candidate in candidates)
        assert all(candidate.policy_source for candidate in candidates)
        assert all(isinstance(candidate.tool_recovery_policy, dict) for candidate in candidates)


def test_recovery_policies_ignore_success_and_unsupported_observations() -> None:
    state = AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag"))
    runtime, step = _build_runtime_and_step(state, "search_arxiv")

    success_candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(status="success", reason="ok"),
        runtime=runtime,
        state=state,
    )
    unsupported_candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(status="tool_error", reason="unknown", failure_category="tool_runtime_error"),
        runtime=runtime,
        state=state,
    )

    assert success_candidates == []
    assert unsupported_candidates == []


def test_high_risk_recovery_candidate_requires_confirmation() -> None:
    state = AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    runtime, step = _build_runtime_and_step(state, "check_paper_index")
    candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(status="need_confirmation", reason="missing", failure_category="paper_index_missing"),
        runtime=runtime,
        state=state,
    )

    high_risk_candidates = [candidate for candidate in candidates if candidate.risk_level == "high"]
    assert high_risk_candidates
    assert all(candidate.requires_confirmation for candidate in high_risk_candidates)


def test_qa_no_answer_generates_retry_candidates() -> None:
    state = AgentState(intent="paper_qa", message="unknown?", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    runtime, step = _build_runtime_and_step(state, "answer_paper_question")
    candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(status="low_confidence", reason="empty", failure_category="qa_no_answer"),
        runtime=runtime,
        state=state,
    )

    assert any(candidate.candidate_id.endswith("retry_qa_with_more_top_k") for candidate in candidates)
    assert candidates[0].strategy_payload["strategy_name"] == "retry_qa_with_more_top_k"
    assert "retry_step" in candidates[0].tool_recovery_policy["modes"]


def test_qa_no_sources_generates_source_grounded_candidates() -> None:
    state = AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    runtime, step = _build_runtime_and_step(state, "answer_paper_question")
    candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(status="low_confidence", reason="no_sources", failure_category="qa_no_sources"),
        runtime=runtime,
        state=state,
    )

    assert any(candidate.strategy_payload.get("force_sources") for candidate in candidates)
    assert any(candidate.action_type == "fallback_answer" for candidate in candidates)


def test_index_stale_prefers_last_good_index_when_available() -> None:
    state = AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    runtime, step = _build_runtime_and_step(state, "check_paper_index")
    candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(
            status="low_confidence",
            reason="stale",
            failure_category="paper_index_stale",
            evidence={"has_last_good_index": True},
        ),
        runtime=runtime,
        state=state,
    )

    assert candidates[0].patch_strategy == "use_last_good_index"
    assert any(candidate.requires_confirmation for candidate in candidates if candidate.patch_strategy == "inject_index_confirmation_chain")


def test_index_corrupted_does_not_silent_rebuild() -> None:
    state = AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    runtime, step = _build_runtime_and_step(state, "check_paper_index")
    candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(status="tool_error", reason="corrupted", failure_category="paper_index_corrupted"),
        runtime=runtime,
        state=state,
    )

    rebuild_candidates = [candidate for candidate in candidates if candidate.patch_strategy == "inject_index_confirmation_chain"]
    assert rebuild_candidates
    assert all(candidate.requires_confirmation and candidate.risk_level == "high" for candidate in rebuild_candidates)


def test_preference_target_missing_generates_clarification_candidate() -> None:
    state = AgentState(intent="preference_action", message="我喜欢这篇", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    runtime, step = _build_runtime_and_step(state, "update_preference_store")
    candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=step,
        observation=ObservationResult(status="need_clarification", reason="missing", failure_category="preference_target_missing"),
        runtime=runtime,
        state=state,
    )

    assert candidates[0].action_type == "ask_clarification"
    assert candidates[0].patch_strategy == "clarification_chain"


def test_tool_timeout_policy_differs_by_tool_type() -> None:
    search_state = AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag"))
    search_runtime, search_step = _build_runtime_and_step(search_state, "search_arxiv")
    search_candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=search_step,
        observation=ObservationResult(status="tool_error", reason="timeout", failure_category="tool_timeout"),
        runtime=search_runtime,
        state=search_state,
    )

    pref_state = AgentState(intent="preference_action", message="喜欢这篇", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    pref_runtime, pref_step = _build_runtime_and_step(pref_state, "update_preference_store")
    pref_candidates = DEFAULT_RECOVERY_POLICY_REGISTRY.get_candidates(
        step=pref_step,
        observation=ObservationResult(status="tool_error", reason="timeout", failure_category="tool_timeout"),
        runtime=pref_runtime,
        state=pref_state,
    )

    assert search_candidates[0].action_type == "retry_step"
    assert pref_candidates[0].action_type == "fallback_answer"
