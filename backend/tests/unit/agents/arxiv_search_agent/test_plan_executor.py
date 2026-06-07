from __future__ import annotations

import importlib
import json

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
executor_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_executor")
planner_registry_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
ExecutablePlan = schemas.ExecutablePlan
Goal = schemas.Goal
PlanStep = schemas.PlanStep
StepCondition = schemas.StepCondition
StepInputBinding = schemas.StepInputBinding
StepPolicy = schemas.StepPolicy
PlanExecutor = executor_module.PlanExecutor
PLANNER_TOOL_REGISTRY = planner_registry_module.PLANNER_TOOL_REGISTRY


def _tool(tool_name: str):
    tool = PLANNER_TOOL_REGISTRY.get(tool_name)
    assert tool is not None
    return tool


def test_plan_executor_executes_linear_arxiv_plan(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        assert kwargs["query"] == "rag"
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {
                "papers": [
                    {"arxiv_id": "2401.00001", "title": "RAG Foundations"},
                    {"arxiv_id": "2401.00002", "title": "RAG Systems"},
                ]
            },
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(
        intent="arxiv_search",
        message="rag",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
        context={"user_memory_summary": {"likes": ["retrieval"]}},
    )

    goal, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert goal.goal_type == "arxiv_search"
    assert result.status == "success"
    assert "normalized_request" in result.outputs
    assert "search_spec" in result.outputs
    assert "arxiv_results" in result.outputs
    assert result.final_answer
    assert result.plan is not None
    assert [step.status for step in result.plan.steps] == ["success"] * 6
    assert any(trace.event == "step_succeeded" and trace.step_id == "search_arxiv" for trace in result.trace)


def test_plan_executor_skips_false_condition_and_continues_dag() -> None:
    goal = Goal(goal_id="unsupported:test", goal_type="unsupported", user_request="fallback")
    plan = ExecutablePlan(
        plan_id="unsupported:test",
        goal=goal,
        steps=[
            PlanStep(
                step_id="normalize_request",
                action_type="write_state",
                tool_name="normalize_request",
                tool=_tool("normalize_request"),
                output_key="normalized_request",
                input_bindings=[StepInputBinding(input_key="message", source_type="state", source_key="message")],
            ),
            PlanStep(
                step_id="optional_clarify",
                action_type="clarify",
                tool_name="analyze_ambiguity",
                tool=_tool("analyze_ambiguity"),
                output_key="missing_information",
                depends_on=["normalize_request"],
                input_bindings=[StepInputBinding(input_key="message", source_type="state", source_key="message")],
                condition=StepCondition(condition_type="field_equals", field_path="state.intent", expected_value="unclear"),
            ),
            PlanStep(
                step_id="generate_fallback_response",
                action_type="answer",
                tool_name="generate_fallback_response",
                tool=_tool("generate_fallback_response"),
                output_key="final_answer",
                depends_on=["normalize_request", "optional_clarify"],
                input_bindings=[StepInputBinding(input_key="message", source_type="state", source_key="message")],
            ),
        ],
        entry_step_ids=["normalize_request"],
        final_step_ids=["generate_fallback_response"],
    )

    state = AgentState(intent="unsupported", message="do something else")
    result = PlanExecutor().execute(plan, state)

    assert result.status == "fallback"
    assert result.plan is not None
    status_map = {step.step_id: step.status for step in result.plan.steps}
    assert status_map["normalize_request"] == "success"
    assert status_map["optional_clarify"] == "skipped"
    assert status_map["generate_fallback_response"] == "success"


def test_plan_executor_returns_waiting_confirmation_before_side_effect(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("parse_and_index_paper should not run before confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)

    goal = Goal(goal_id="paper_qa:test", goal_type="paper_qa", user_request="index paper")
    plan = ExecutablePlan(
        plan_id="paper_qa:test",
        goal=goal,
        steps=[
            PlanStep(
                step_id="parse_and_index_paper",
                action_type="index",
                tool_name="parse_and_index_paper",
                tool=_tool("parse_and_index_paper"),
                output_key="index_build_result",
                input_bindings=[StepInputBinding(input_key="paper_reference", source_type="literal", value={"arxiv_id": "2401.00001"})],
                confirmation_policy=StepPolicy(
                    policy_type="confirmation",
                    mode="explicit_user_confirmation_required",
                    requires_confirmation=True,
                ),
                side_effect_level="external_call",
            ),
        ],
        entry_step_ids=["parse_and_index_paper"],
        final_step_ids=["parse_and_index_paper"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="paper_qa", message="build index"))

    assert result.status == "waiting_confirmation"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.step_id == "parse_and_index_paper"
    assert result.pending_confirmation.tool_name == "parse_and_index_paper"
    assert result.pending_confirmation.side_effect_level == "external_call"
    assert [item.code for item in result.pending_confirmation.allowed_decisions] == ["approve", "reject"]
    json.dumps(result.pending_confirmation.model_dump(), ensure_ascii=False)
    assert result.pending_confirmation.arguments_summary["paper_reference"]["arxiv_id"] == "2401.00001"
    assert result.plan is not None
    assert result.plan.steps[0].status == "waiting_confirmation"


def test_plan_executor_interrupt_approve_executes_side_effect_tool(monkeypatch) -> None:
    calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"decision": "approve", "note": "go"})

    goal = Goal(goal_id="paper_qa:test", goal_type="paper_qa", user_request="index paper")
    plan = ExecutablePlan(
        plan_id="paper_qa:test",
        goal=goal,
        steps=[
            PlanStep(
                step_id="parse_and_index_paper",
                action_type="index",
                tool_name="parse_and_index_paper",
                tool=_tool("parse_and_index_paper"),
                output_key="index_build_result",
                input_bindings=[StepInputBinding(input_key="paper_reference", source_type="literal", value={"arxiv_id": "2401.00001"})],
                confirmation_policy=StepPolicy(
                    policy_type="confirmation",
                    mode="explicit_user_confirmation_required",
                    requires_confirmation=True,
                ),
                side_effect_level="external_call",
            ),
        ],
        entry_step_ids=["parse_and_index_paper"],
        final_step_ids=["parse_and_index_paper"],
    )

    runtime = planner_module.build_plan_runtime(AgentState(intent="paper_qa", message="build index"), goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
    result = PlanExecutor()._execute_runtime(runtime, AgentState(intent="paper_qa", message="build index"), allow_interrupt=True)

    assert result.status == "success"
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert len(calls) == 1
    assert any(trace.event == "confirmation_requested" for trace in result.trace)
    assert any(trace.event == "confirmation_approved" for trace in result.trace)


def test_plan_executor_interrupt_reject_skips_side_effect_tool(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("side effect tool must not run after reject")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"decision": "reject", "note": "cancel"})

    goal = Goal(goal_id="paper_qa:test", goal_type="paper_qa", user_request="index paper")
    plan = ExecutablePlan(
        plan_id="paper_qa:test",
        goal=goal,
        steps=[
            PlanStep(
                step_id="parse_and_index_paper",
                action_type="index",
                tool_name="parse_and_index_paper",
                tool=_tool("parse_and_index_paper"),
                output_key="index_build_result",
                input_bindings=[StepInputBinding(input_key="paper_reference", source_type="literal", value={"arxiv_id": "2401.00001", "title": "RAG Paper"})],
                confirmation_policy=StepPolicy(
                    policy_type="confirmation",
                    mode="explicit_user_confirmation_required",
                    requires_confirmation=True,
                ),
                side_effect_level="external_call",
            ),
        ],
        entry_step_ids=["parse_and_index_paper"],
        final_step_ids=["parse_and_index_paper"],
    )

    runtime = planner_module.build_plan_runtime(AgentState(intent="paper_qa", message="build index"), goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
    result = PlanExecutor()._execute_runtime(runtime, AgentState(intent="paper_qa", message="build index"), allow_interrupt=True)

    assert result.status in {"success", "waiting_confirmation", "fallback"}
    assert result.final_answer == "已取消解析 RAG Paper，因此无法继续基于全文回答。"
    assert any(trace.event == "confirmation_requested" for trace in result.trace)
    assert any(trace.event == "confirmation_rejected" for trace in result.trace)
    assert result.plan is not None
    assert result.plan.steps[0].status == "skipped"


def test_plan_executor_marks_missing_input_as_failed() -> None:
    goal = Goal(goal_id="unsupported:missing", goal_type="unsupported", user_request="missing input")
    plan = ExecutablePlan(
        plan_id="unsupported:missing",
        goal=goal,
        steps=[
            PlanStep(
                step_id="generate_fallback_response",
                action_type="answer",
                tool_name="generate_fallback_response",
                tool=_tool("generate_fallback_response"),
                output_key="final_answer",
                input_bindings=[StepInputBinding(input_key="message", source_type="state", source_key="missing_field")],
            ),
        ],
        entry_step_ids=["generate_fallback_response"],
        final_step_ids=["generate_fallback_response"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="unsupported", message="ignored"))

    assert result.status == "failed"
    assert result.error == "missing_input:generate_fallback_response:message"
    assert result.plan is not None
    assert result.plan.steps[0].status == "failed"
    assert any(trace.detail.get("failure_reason") == "missing_input" for trace in result.trace)
    assert result.pending_confirmation is None


def test_plan_executor_missing_input_does_not_trigger_confirmation(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("tool must not run when required input is missing")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)

    goal = Goal(goal_id="paper_qa:missing", goal_type="paper_qa", user_request="index paper")
    plan = ExecutablePlan(
        plan_id="paper_qa:missing",
        goal=goal,
        steps=[
            PlanStep(
                step_id="parse_and_index_paper",
                action_type="index",
                tool_name="parse_and_index_paper",
                tool=_tool("parse_and_index_paper"),
                output_key="index_build_result",
                input_bindings=[StepInputBinding(input_key="paper_reference", source_type="state", source_key="missing_paper")],
                confirmation_policy=StepPolicy(
                    policy_type="confirmation",
                    mode="explicit_user_confirmation_required",
                    requires_confirmation=True,
                ),
                side_effect_level="external_call",
            ),
        ],
        entry_step_ids=["parse_and_index_paper"],
        final_step_ids=["parse_and_index_paper"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="paper_qa", message="build index"))

    assert result.status == "failed"
    assert result.pending_confirmation is None
    assert result.plan is not None
    assert result.plan.steps[0].status == "failed"
    assert any(trace.detail.get("failure_reason") == "missing_input" for trace in result.trace)


def test_plan_executor_replans_empty_arxiv_search_before_fallback(monkeypatch) -> None:
    calls = {"search": 0}

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        calls["search"] += 1
        papers = [] if calls["search"] == 1 else [{"arxiv_id": "2401.00001", "title": "RAG retrieval systems"}]
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": papers},
            "trace": {"tool_name": tool_name, "query": kwargs.get("query")},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(
        intent="arxiv_search",
        message="rag",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
    )
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert calls["search"] == 2
    assert result.runtime is not None
    assert result.runtime.replan_counts["search_arxiv:empty_result"] == 1
    replan_traces = [trace for trace in result.trace if trace.event == "plan_replanned" and trace.detail.get("rule_name") == "rule_arxiv_empty_result"]
    assert replan_traces
    assert replan_traces[0].detail.get("failure_category") == "search_empty"
    assert replan_traces[0].detail.get("recovery_candidates")
    assert any(step.tool_name == "rewrite_arxiv_query" for step in result.plan.steps)
    assert any(trace.event == "step_succeeded" and trace.step_id == "rewrite_arxiv_query" for trace in result.trace)


def test_plan_executor_replans_low_confidence_arxiv_validation(monkeypatch) -> None:
    calls = {"search": 0}

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        calls["search"] += 1
        papers = (
            [
                {"arxiv_id": "2401.00001", "title": "RAG duplicate"},
                {"arxiv_id": "2401.00002", "title": "RAG duplicate"},
                {"arxiv_id": "2401.00003", "title": "RAG duplicate"},
                {"arxiv_id": "2401.00004", "title": "RAG duplicate"},
            ]
            if calls["search"] == 1
            else [
                {"arxiv_id": "2401.00005", "title": "RAG retrieval"},
                {"arxiv_id": "2401.00006", "title": "RAG agent"},
            ]
        )
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": papers},
            "trace": {"tool_name": tool_name, "query": kwargs.get("query")},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(
        intent="arxiv_search",
        message="rag",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
    )
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert calls["search"] == 2
    assert result.runtime is not None
    assert result.runtime.replan_counts["validate_arxiv_results:low_confidence"] == 1
    replan_traces = [trace for trace in result.trace if trace.event == "plan_replanned" and trace.detail.get("rule_name") == "rule_arxiv_low_confidence"]
    assert replan_traces
    assert replan_traces[0].detail.get("failure_category") == "search_low_confidence"
    assert replan_traces[0].detail.get("recovery_candidates")


def test_plan_executor_replans_missing_paper_index_to_confirmation(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "check_paper_qa_index":
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "missing",
                "data": {"status": "missing", "has_index": False},
                "trace": {"tool_name": tool_name},
                "error": None,
            }
        raise AssertionError(f"{tool_name} should not run before user confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(intent="paper_qa", message="what is the method?", context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}})
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "waiting_confirmation"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_name == "parse_and_index_paper"
    assert result.pending_confirmation.target_paper is not None
    assert result.pending_confirmation.target_paper["arxiv_id"] == "2401.00001"
    assert [item.code for item in result.pending_confirmation.allowed_decisions] == ["approve", "reject"]
    assert any(step.tool_name == "parse_and_index_paper" for step in result.plan.steps)
    replan_traces = [trace for trace in result.trace if trace.event == "plan_replanned" and trace.detail.get("rule_name") == "rule_missing_paper_index"]
    assert replan_traces
    assert replan_traces[0].detail.get("failure_category") == "paper_index_missing"
    assert replan_traces[0].detail.get("recovery_candidates")[0]["requires_confirmation"] is True


def test_plan_executor_paper_qa_calls_real_answer_tool_once_and_preserves_debug(monkeypatch) -> None:
    answer_calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "check_paper_qa_index":
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "available",
                "data": {"status": "available", "has_index": True},
                "trace": {"tool_name": tool_name},
                "error": None,
            }
        if tool_name == "answer_paper_question":
            answer_calls.append(dict(kwargs))
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "answered",
                "data": {"answer": "grounded answer", "sources": [{"chunk_id": "c1"}], "retrieval_debug": {"stages": {"rerank": {"count": 1}}}},
                "trace": {"tool_name": tool_name},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(intent="paper_qa", message="what is the method?", context={"selected_paper": {"arxiv_id": "2401.00001"}})
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert result.final_answer == "grounded answer"
    assert answer_calls == [{"arxiv_id": "2401.00001", "question": "what is the method?"}]
    assert result.outputs["paper_qa_result"]["sources"] == [{"chunk_id": "c1"}]
    assert result.outputs["paper_qa_result"]["retrieval_debug"] == {"stages": {"rerank": {"count": 1}}}
    pseudo_steps = {"retrieve_paper_chunks", "rewrite_paper_query", "rerank_paper_chunks", "validate_qa_evidence", "generate_paper_answer", "verify_answer_grounding"}
    assert not pseudo_steps.intersection({step.tool_name for step in result.plan.steps})
    assert not pseudo_steps.intersection({trace.step_id for trace in result.trace})


def test_plan_executor_replans_empty_profile_to_message_recommendation(monkeypatch) -> None:
    recommend_calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "recommend_papers"
        recommend_calls.append(dict(kwargs))
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "recommended",
            "data": {"recommendations": [{"arxiv_id": "2401.00001", "title": "RAG recommendations"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(intent="recommendation", message="recommend rag papers", context={})
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert recommend_calls
    assert recommend_calls[0]["message"] == "recommend rag papers"
    assert result.runtime is not None
    assert result.runtime.replan_counts["load_user_profile:empty_result"] == 1
    replan_traces = [trace for trace in result.trace if trace.event == "plan_replanned" and trace.detail.get("rule_name") == "rule_empty_user_profile"]
    assert replan_traces
    assert replan_traces[0].detail.get("failure_category") == "empty_user_profile"
    assert replan_traces[0].detail.get("recovery_candidates")
