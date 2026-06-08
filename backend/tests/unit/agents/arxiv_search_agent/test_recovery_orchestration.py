from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
plan_validator_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_validator")
replanner_module = importlib.import_module("backend.agents.arxiv_search_agent.replanner")
recovery_chooser_module = importlib.import_module("backend.agents.arxiv_search_agent.recovery_chooser")
recovery_diagnosis_module = importlib.import_module("backend.agents.arxiv_search_agent.recovery_diagnosis")
recovery_safety_module = importlib.import_module("backend.agents.arxiv_search_agent.recovery_safety")
plan_patcher_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_patcher")
planner_registry_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
ObservationResult = schemas.ObservationResult
PlanRuntime = schemas.PlanRuntime
RecoveryAction = schemas.RecoveryAction
RecoveryCandidate = schemas.RecoveryCandidate

PlanValidator = plan_validator_module.PlanValidator
PlanPatcher = plan_patcher_module.PlanPatcher
LLMRecoveryDiagnoser = recovery_diagnosis_module.LLMRecoveryDiagnoser
RecoveryChooser = recovery_chooser_module.RecoveryChooser
RecoverySafetyGuard = recovery_safety_module.RecoverySafetyGuard
Replanner = replanner_module.Replanner
PLANNER_TOOL_REGISTRY = planner_registry_module.PLANNER_TOOL_REGISTRY


def _build_runtime_and_step(state: AgentState, tool_name: str):
    goal, plan, _ = planner_module.build_executable_plan(state)
    step = next(item for item in list(plan.steps or []) if item.tool_name == tool_name)
    runtime = PlanRuntime(goal=goal, plan=plan, step_status={item.step_id: "pending" for item in list(plan.steps or [])})
    return goal, plan, runtime, step


def _candidate(candidate_id: str, *, priority: int, action_type: str = "patch_plan", risk_level: str = "low", max_attempts=None, required_tools=None, requires_confirmation=False):
    return RecoveryCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        failure_category="search_empty",
        priority=priority,
        confidence=0.8,
        reason=f"{candidate_id} reason",
        target_step_id="search_arxiv",
        required_tools=list(required_tools or ["rewrite_arxiv_query"]),
        patch_strategy="rewrite_search_chain" if action_type == "patch_plan" else "fallback_answer",
        risk_level=risk_level,
        requires_confirmation=requires_confirmation,
        max_attempts=max_attempts,
    )


def test_recovery_chooser_selects_highest_priority_candidate() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
    )

    choice = RecoveryChooser().choose(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="empty_result", failure_category="search_empty"),
        recovery_candidates=[_candidate("low", priority=20), _candidate("high", priority=90)],
    )

    assert choice.action.selected_candidate_id == "high"
    assert choice.selection_reason


def test_recovery_chooser_rejects_candidate_over_retry_limit() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
    )
    runtime.step_replan_counts[step.step_id] = 1

    choice = RecoveryChooser().choose(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="empty_result", failure_category="search_empty"),
        recovery_candidates=[
            _candidate("limited", priority=90, max_attempts=1),
            _candidate("fallback", priority=1, action_type="fallback_answer", required_tools=[]),
        ],
    )

    assert choice.action.action_type == "fallback_answer"
    assert choice.rejected_candidates[0]["candidate_id"] == "limited"
    assert "retry_limit_exceeded" in choice.rejected_candidates[0]["reason"]


def test_recovery_chooser_rejects_high_risk_candidate_without_confirmation() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}}),
        "check_paper_index",
    )
    risky = RecoveryCandidate(
        candidate_id="risky",
        action_type="patch_plan",
        failure_category="paper_index_missing",
        priority=90,
        confidence=0.9,
        reason="missing confirmation",
        target_step_id=step.step_id,
        required_tools=["parse_and_index_paper"],
        patch_strategy="inject_index_confirmation_chain",
        risk_level="high",
        requires_confirmation=False,
    )

    choice = RecoveryChooser().choose(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="need_confirmation", failure_category="paper_index_missing"),
        recovery_candidates=[risky],
    )

    assert choice.action.action_type == "fallback_answer"
    assert choice.rejected_candidates[0]["reason"] == "high_risk_candidate_requires_confirmation_capability"


def test_recovery_chooser_selects_fallback_candidate_when_only_fallback_is_available() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
    )

    choice = RecoveryChooser().choose(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="tool_error", failure_category="tool_runtime_error"),
        recovery_candidates=[_candidate("fallback", priority=1, action_type="fallback_answer", required_tools=[])],
    )

    assert choice.action.action_type == "fallback_answer"
    assert choice.action.selected_candidate_id == "fallback"


def test_plan_patcher_inserts_search_rewrite_chain_and_validates_plan() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
    )
    action = RecoveryAction(
        action_type="patch_plan",
        target_step_id=step.step_id,
        selected_candidate_id="search",
        patch_strategy="rewrite_search_chain",
        patch_payload={"failure_category": "search_empty", "risk_level": "low"},
    )

    result = PlanPatcher().apply(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="empty_result", failure_category="search_empty"),
        action=action,
        reason_key="search_arxiv:empty_result",
        current_reason_count=0,
        current_step_count=0,
        recovery_candidates=[],
        rejected_candidates=[],
        selection_reason="test",
        scored_candidates=[],
    )

    assert result.updated_plan is not None
    assert any(item.tool_name == "rewrite_arxiv_query" for item in result.updated_plan.steps)
    assert any(item.tool_name == "validate_arxiv_results" for item in result.updated_plan.steps)
    PlanValidator().validate(result.updated_plan, PLANNER_TOOL_REGISTRY)
    assert result.updated_runtime.step_status[step.step_id] == "pending"
    assert any(trace.detail.get("selected_recovery_action") for trace in result.updated_runtime.trace)


def test_plan_patcher_inserts_index_confirmation_chain_and_validates_plan() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}}),
        "check_paper_index",
    )
    action = RecoveryAction(
        action_type="patch_plan",
        target_step_id=step.step_id,
        selected_candidate_id="index",
        patch_strategy="inject_index_confirmation_chain",
        patch_payload={"failure_category": "paper_index_missing", "risk_level": "high"},
        requires_confirmation=True,
    )

    result = PlanPatcher().apply(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="need_confirmation", failure_category="paper_index_missing"),
        action=action,
        reason_key="check_paper_index:need_confirmation",
        current_reason_count=0,
        current_step_count=0,
        recovery_candidates=[],
        rejected_candidates=[],
        selection_reason="test",
        scored_candidates=[],
    )

    assert result.updated_plan is not None
    assert any(item.tool_name == "request_confirmation" for item in result.updated_plan.steps)
    assert any(item.tool_name == "parse_and_index_paper" for item in result.updated_plan.steps)
    PlanValidator().validate(result.updated_plan, PLANNER_TOOL_REGISTRY)
    assert result.updated_runtime.step_status[step.step_id] == "pending"
    assert result.updated_runtime.trace[-1].detail["patch_result"]["rule_name"] == "rule_missing_paper_index"


def test_replanner_is_orchestrator_without_apply_rule() -> None:
    assert not hasattr(Replanner, "_apply_rule")


def test_replanner_returns_fallback_when_no_candidate_is_available() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
    )

    decision = Replanner().replan(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_or_low_quality_step=step,
        observation_result=ObservationResult(status="tool_error", failure_category="tool_runtime_error", reason="boom"),
    )

    assert decision.fallback is True
    assert decision.updated_runtime is not None
    assert decision.updated_runtime.trace[-1].detail["selected_recovery_action"]["action_type"] == "fallback_answer"


def test_plan_patcher_inserts_qa_retry_step_with_strategy() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="paper_qa", message="unknown?", context={"selected_paper": {"arxiv_id": "2401.00001"}}),
        "answer_paper_question",
    )
    action = RecoveryAction(
        action_type="retry_step",
        target_step_id=step.step_id,
        selected_candidate_id="qa-retry",
        patch_strategy="retry_step_with_adjusted_arguments",
        patch_payload={
            "failure_category": "qa_no_answer",
            "risk_level": "medium",
            "required_tools": ["answer_paper_question"],
            "max_attempts": 1,
            "retry_strategy": {"strategy_name": "retry_qa_with_more_top_k", "retrieval_top_k": 30},
        },
    )

    result = PlanPatcher().apply(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="low_confidence", failure_category="qa_no_answer"),
        action=action,
        reason_key="answer_paper_question:low_confidence",
        current_reason_count=0,
        current_step_count=0,
        recovery_candidates=[],
        rejected_candidates=[],
        selection_reason="test",
        scored_candidates=[],
    )

    retry_steps = [item for item in result.updated_plan.steps if item.step_id != step.step_id and item.tool_name == "answer_paper_question"]
    assert retry_steps
    strategy_bindings = [binding for binding in retry_steps[0].input_bindings if binding.input_key == "qa_recovery_strategy"]
    assert strategy_bindings[0].value["strategy_name"] == "retry_qa_with_more_top_k"
    PlanValidator().validate(result.updated_plan, PLANNER_TOOL_REGISTRY)


def test_plan_patcher_marks_qa_to_use_last_good_index() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}}),
        "check_paper_index",
    )
    action = RecoveryAction(
        action_type="patch_plan",
        target_step_id=step.step_id,
        selected_candidate_id="last-good-index",
        patch_strategy="use_last_good_index",
        patch_payload={"failure_category": "paper_index_stale", "risk_level": "low", "required_tools": ["answer_paper_question"]},
    )

    result = PlanPatcher().apply(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(
            status="low_confidence",
            failure_category="paper_index_stale",
            evidence={"has_last_good_index": True},
        ),
        action=action,
        reason_key="check_paper_index:low_confidence",
        current_reason_count=0,
        current_step_count=0,
        recovery_candidates=[],
        rejected_candidates=[],
        selection_reason="test",
        scored_candidates=[],
    )

    qa_step = next(item for item in result.updated_plan.steps if item.tool_name == "answer_paper_question")
    strategy_bindings = [binding for binding in qa_step.input_bindings if binding.input_key == "index_strategy"]
    assert strategy_bindings[0].value == {"mode": "use_last_good_index"}
    assert result.updated_runtime.outputs[step.output_key]["last_good_index"] is True
    PlanValidator().validate(result.updated_plan, PLANNER_TOOL_REGISTRY)


def test_llm_diagnosis_ignores_illegal_candidate_ids() -> None:
    class FakeGenerationService:
        def complete_with_qwen(self, *args, **kwargs):
            return '{"diagnosis":"x","ranked_candidate_ids":["missing","valid"],"confidence":0.9}'

    state = AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag"))
    goal, plan, runtime, step = _build_runtime_and_step(state, "search_arxiv")
    candidate = _candidate("valid", priority=50)

    diagnosis = LLMRecoveryDiagnoser(FakeGenerationService(), enabled=True).diagnose(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="empty_result", failure_category="search_empty"),
        recovery_candidates=[candidate],
        state=state,
    )

    assert diagnosis.ranked_candidate_ids == ["valid"]
    assert diagnosis.ignored_candidate_ids == ["missing"]


def test_llm_diagnosis_failure_does_not_block_rule_recovery() -> None:
    class BrokenGenerationService:
        def complete_with_qwen(self, *args, **kwargs):
            raise RuntimeError("boom")

    state = AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag"))
    goal, plan, runtime, step = _build_runtime_and_step(state, "search_arxiv")
    candidate = _candidate("valid", priority=50)

    diagnosis = LLMRecoveryDiagnoser(BrokenGenerationService(), enabled=True).diagnose(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="empty_result", failure_category="search_empty"),
        recovery_candidates=[candidate],
        state=state,
    )
    choice = RecoveryChooser().choose(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="empty_result", failure_category="search_empty"),
        recovery_candidates=[candidate],
        llm_diagnosis=diagnosis,
    )

    assert diagnosis.error
    assert choice.action.selected_candidate_id == "valid"


def test_safety_guard_blocks_high_risk_action_without_confirmation() -> None:
    state = AgentState(intent="paper_qa", message="method?", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    _, _, runtime, step = _build_runtime_and_step(state, "check_paper_index")
    unsafe = RecoveryAction(
        action_type="patch_plan",
        target_step_id=step.step_id,
        selected_candidate_id="unsafe",
        patch_strategy="inject_index_confirmation_chain",
        patch_payload={"risk_level": "high", "required_tools": ["parse_and_index_paper"]},
        requires_confirmation=False,
    )

    result = RecoverySafetyGuard().check(action=unsafe, failed_step=step, runtime=runtime)

    assert result.allowed is False
    assert "high_risk_action_without_confirmation" in result.reasons
    assert result.fallback_action.action_type == "fallback_answer"


def test_recovery_trace_records_llm_safety_and_final_status() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
    )
    action = RecoveryAction(
        action_type="patch_plan",
        target_step_id=step.step_id,
        selected_candidate_id="search",
        patch_strategy="rewrite_search_chain",
        patch_payload={"failure_category": "search_empty", "risk_level": "low", "required_tools": []},
    )

    result = PlanPatcher().apply(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_step=step,
        observation=ObservationResult(status="empty_result", failure_category="search_empty", severity="warning", recoverable=True),
        action=action,
        reason_key="search_arxiv:empty_result",
        current_reason_count=0,
        current_step_count=0,
        recovery_candidates=[],
        rejected_candidates=[],
        selection_reason="test",
        scored_candidates=[],
        llm_diagnosis={"diagnosis": "empty"},
        safety_check_result={"allowed": True, "reasons": []},
    )

    detail = result.updated_runtime.trace[-1].detail
    assert detail["llm_diagnosis"]["diagnosis"] == "empty"
    assert detail["safety_check_result"]["allowed"] is True
    assert detail["final_recovery_status"] == "patched"


def test_replanner_records_trace_when_replan_limit_is_exceeded() -> None:
    goal, plan, runtime, step = _build_runtime_and_step(
        AgentState(intent="arxiv_search", message="rag", search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag")),
        "search_arxiv",
    )
    runtime.replan_counts["search_arxiv:empty_result"] = Replanner.MAX_REASON_REPLANS

    decision = Replanner().replan(
        goal=goal,
        current_plan=plan,
        runtime=runtime,
        failed_or_low_quality_step=step,
        observation_result=ObservationResult(
            status="empty_result",
            observation_signal="success_but_empty_result",
            failure_category="search_empty",
            reason="empty",
        ),
    )

    assert decision.fallback is True
    assert decision.updated_runtime is not None
    trace = decision.updated_runtime.trace[-1]
    assert trace.event == "replan_limit_exceeded"
    assert trace.detail["fallback_used"] is True
    assert trace.detail["observation_signal"] == "success_but_empty_result"
    assert trace.detail["reason_replan_count"] == Replanner.MAX_REASON_REPLANS
