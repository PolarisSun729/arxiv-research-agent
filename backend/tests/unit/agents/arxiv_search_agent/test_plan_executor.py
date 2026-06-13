from __future__ import annotations

import importlib
import json
from dataclasses import replace

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
PlanRuntime = schemas.PlanRuntime
StepCondition = schemas.StepCondition
StepInputBinding = schemas.StepInputBinding
StepPolicy = schemas.StepPolicy
PlanExecutor = executor_module.PlanExecutor
PLANNER_TOOL_REGISTRY = planner_registry_module.PLANNER_TOOL_REGISTRY


def _tool(tool_name: str):
    tool = PLANNER_TOOL_REGISTRY.get(tool_name)
    assert tool is not None
    return tool


def _paper_index_confirmation_plan() -> tuple[Goal, ExecutablePlan]:
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
    return goal, plan


def _step(
    *,
    step_id: str,
    action_type: str,
    tool_name: str,
    output_key: str,
    input_bindings: list[StepInputBinding],
    depends_on: list[str] | None = None,
) -> PlanStep:
    tool = _tool(tool_name)
    return PlanStep(
        step_id=step_id,
        action_type=action_type,
        tool_name=tool_name,
        tool=tool,
        output_key=output_key,
        input_bindings=input_bindings,
        depends_on=list(depends_on or []),
        side_effect_level=tool.side_effect_level,
    )


def _llm_prefixed_arxiv_plan() -> ExecutablePlan:
    goal = Goal(goal_id="arxiv_search:prefixed", goal_type="arxiv_search", user_request="帮我找 RAG 论文")
    steps = [
        _step(
            step_id="step_normalize_request",
            action_type="write_state",
            tool_name="normalize_request",
            output_key="normalized_request",
            input_bindings=[
                StepInputBinding(input_key="intent", source_type="state", source_key="intent"),
                StepInputBinding(input_key="message", source_type="state", source_key="message"),
                StepInputBinding(input_key="search_spec", source_type="search_spec", required=False),
            ],
        ),
        _step(
            step_id="step_build_arxiv_search_spec",
            action_type="search",
            tool_name="build_arxiv_search_spec",
            output_key="search_spec",
            input_bindings=[
                StepInputBinding(input_key="normalized_request", source_type="step_output", step_id="step_normalize_request"),
            ],
            depends_on=["step_normalize_request"],
        ),
        _step(
            step_id="step_search_arxiv",
            action_type="search",
            tool_name="search_arxiv",
            output_key="arxiv_results",
            input_bindings=[
                StepInputBinding(input_key="search_spec", source_type="step_output", step_id="step_build_arxiv_search_spec"),
            ],
            depends_on=["step_build_arxiv_search_spec"],
        ),
        _step(
            step_id="step_validate_arxiv_results",
            action_type="validate",
            tool_name="validate_arxiv_results",
            output_key="arxiv_result_quality",
            input_bindings=[
                StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="step_search_arxiv"),
            ],
            depends_on=["step_search_arxiv"],
        ),
        _step(
            step_id="step_synthesize_arxiv_response",
            action_type="answer",
            tool_name="synthesize_arxiv_response",
            output_key="final_answer",
            input_bindings=[
                StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="step_search_arxiv"),
                StepInputBinding(input_key="arxiv_result_quality", source_type="step_output", step_id="step_validate_arxiv_results"),
            ],
            depends_on=["step_validate_arxiv_results"],
        ),
    ]
    return ExecutablePlan(
        plan_id="arxiv_search:prefixed-plan",
        goal=goal,
        steps=steps,
        entry_step_ids=["step_normalize_request"],
        final_step_ids=["step_synthesize_arxiv_response"],
    )


def _resolved_paper_qa_plan(arxiv_id: str = "2401.00001", title: str = "RAG") -> ExecutablePlan:
    """构造已完成最终目标解析的 QA 计划，避免下游执行器测试依赖引用线索提取器。"""
    paper_ref = {"arxiv_id": arxiv_id, "title": title, "final_target_resolved": True}
    goal = Goal(goal_id="paper_qa:resolved", goal_type="paper_qa", user_request="what is the method?")
    return ExecutablePlan(
        plan_id="paper_qa:resolved",
        goal=goal,
        steps=[
            PlanStep(
                step_id="resolve_paper",
                action_type="retrieve",
                tool_name="resolve_paper",
                tool=_tool("resolve_paper"),
                output_key="paper_ref",
                input_bindings=[StepInputBinding(input_key="message", source_type="state", source_key="message")],
                # 下游执行器测试需要一个“最终目标已解析”的输入；这里保留结构占位但不执行引用线索提取器。
                condition=StepCondition(condition_type="field_equals", field_path="state.intent", expected_value="__skip_reference_hint_extractor__"),
            ),
            PlanStep(
                step_id="check_paper_index",
                action_type="validate",
                tool_name="check_paper_index",
                tool=_tool("check_paper_index"),
                output_key="paper_index_status",
                depends_on=["resolve_paper"],
                input_bindings=[StepInputBinding(input_key="paper_ref", source_type="literal", value=paper_ref)],
            ),
            PlanStep(
                step_id="answer_paper_question",
                action_type="answer",
                tool_name="answer_paper_question",
                tool=_tool("answer_paper_question"),
                output_key="paper_qa_result",
                side_effect_level="external_call",
                depends_on=["resolve_paper", "check_paper_index"],
                input_bindings=[
                    StepInputBinding(input_key="paper_ref", source_type="step_output", step_id="resolve_paper", required=False),
                    StepInputBinding(input_key="paper_ref", source_type="literal", value=paper_ref),
                    StepInputBinding(input_key="message", source_type="state", source_key="message"),
                ],
            ),
            PlanStep(
                step_id="assess_paper_qa_quality",
                action_type="validate",
                tool_name="assess_paper_qa_quality",
                tool=_tool("assess_paper_qa_quality"),
                output_key="paper_qa_quality_decision",
                depends_on=["answer_paper_question"],
                input_bindings=[StepInputBinding(input_key="paper_qa_result", source_type="step_output", step_id="answer_paper_question")],
            ),
        ],
        entry_step_ids=["resolve_paper"],
        final_step_ids=["assess_paper_qa_quality"],
    )


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


def test_plan_executor_build_spec_accepts_state_search_spec_object(monkeypatch) -> None:
    """覆盖 LLM planner 直接把 state.search_spec 绑定给 build_spec 的路径。"""

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        assert kwargs["query"] == "rag"
        assert kwargs["max_results"] == 5
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG Foundations"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(
        intent="arxiv_search",
        message="search rag",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
    )
    for input_key in ("search_spec", "normalized_request"):
        goal = Goal(goal_id=f"arxiv_search:llm-direct-spec:{input_key}", goal_type="arxiv_search", user_request="search rag")
        plan = ExecutablePlan(
            plan_id=f"arxiv_search:llm-direct-spec:{input_key}",
            goal=goal,
            steps=[
                PlanStep(
                    step_id="step_1_build_spec",
                    action_type="search",
                    tool_name="build_arxiv_search_spec",
                    tool=_tool("build_arxiv_search_spec"),
                    output_key="search_spec",
                    input_bindings=[StepInputBinding(input_key=input_key, source_type="search_spec")],
                ),
                PlanStep(
                    step_id="step_2_search",
                    action_type="search",
                    tool_name="search_arxiv",
                    tool=_tool("search_arxiv"),
                    output_key="arxiv_results",
                    depends_on=["step_1_build_spec"],
                    input_bindings=[StepInputBinding(input_key="search_spec", source_type="step_output", step_id="step_1_build_spec")],
                ),
                PlanStep(
                    step_id="step_3_validate",
                    action_type="validate",
                    tool_name="validate_arxiv_results",
                    tool=_tool("validate_arxiv_results"),
                    output_key="arxiv_result_quality",
                    depends_on=["step_2_search"],
                    input_bindings=[StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="step_2_search")],
                ),
                PlanStep(
                    step_id="step_4_answer",
                    action_type="answer",
                    tool_name="synthesize_arxiv_response",
                    tool=_tool("synthesize_arxiv_response"),
                    output_key="final_answer",
                    depends_on=["step_3_validate"],
                    input_bindings=[
                        StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="step_2_search"),
                        StepInputBinding(input_key="arxiv_result_quality", source_type="step_output", step_id="step_3_validate"),
                    ],
                ),
            ],
            entry_step_ids=["step_1_build_spec"],
            final_step_ids=["step_4_answer"],
        )

        result = PlanExecutor().execute(plan, state)

        assert result.status == "success"
        assert result.outputs["search_spec"]["query"] == "rag"
        assert result.outputs["search_spec"]["max_results"] == 5
        assert "RAG Foundations" in result.final_answer
        assert not any(
            trace.event == "step_observed"
            and trace.step_id == "step_1_build_spec"
            and trace.detail.get("observation_status") == "tool_error"
            for trace in result.trace
        )


def test_plan_executor_build_spec_normalizes_llm_constraint_list(monkeypatch) -> None:
    """复现 LLM planner 把 goal.constraints 误绑到 normalized_request 的输入形态。"""

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        assert kwargs["query"] == "RAG"
        assert kwargs["categories"] == ["cs.CL", "cs.LG", "cs.IR", "cs.AI"]
        assert kwargs["submitted_days_ago"] == 7
        assert kwargs["max_results"] == 5
        assert kwargs["sort_by"] == "submittedDate"
        assert kwargs["sort_order"] == "descending"
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG Foundations"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    goal = Goal(goal_id="arxiv_search:llm-constraints", goal_type="arxiv_search", user_request="search rag")
    plan = ExecutablePlan(
        plan_id="arxiv_search:llm-constraints",
        goal=goal,
        steps=[
            PlanStep(
                step_id="step_1_build_spec",
                action_type="search",
                tool_name="build_arxiv_search_spec",
                tool=_tool("build_arxiv_search_spec"),
                output_key="search_spec",
                input_bindings=[
                    StepInputBinding(
                        input_key="normalized_request",
                        source_type="literal",
                        value=[
                            "query=RAG",
                            "categories=cs.CL, cs.LG, cs.IR, cs.AI",
                            "submitted_days_ago=7",
                            "max_results=5",
                            "sort_by=submittedDate:descending",
                        ],
                    ),
                    StepInputBinding(
                        input_key="search_spec",
                        source_type="literal",
                        value={
                            "query": "RAG",
                            "categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"],
                            "submitted_days_ago": 7,
                            "max_results": 5,
                            "sort_by": "submittedDate:descending",
                        },
                    ),
                ],
            ),
            PlanStep(
                step_id="step_2_search",
                action_type="search",
                tool_name="search_arxiv",
                tool=_tool("search_arxiv"),
                output_key="arxiv_results",
                depends_on=["step_1_build_spec"],
                input_bindings=[StepInputBinding(input_key="search_spec", source_type="step_output", step_id="step_1_build_spec")],
            ),
            PlanStep(
                step_id="step_3_validate",
                action_type="validate",
                tool_name="validate_arxiv_results",
                tool=_tool("validate_arxiv_results"),
                output_key="arxiv_result_quality",
                depends_on=["step_2_search"],
                input_bindings=[StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="step_2_search")],
            ),
            PlanStep(
                step_id="step_4_answer",
                action_type="answer",
                tool_name="synthesize_arxiv_response",
                tool=_tool("synthesize_arxiv_response"),
                output_key="final_answer",
                depends_on=["step_3_validate"],
                input_bindings=[
                    StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="step_2_search"),
                    StepInputBinding(input_key="arxiv_result_quality", source_type="step_output", step_id="step_3_validate"),
                ],
            ),
        ],
        entry_step_ids=["step_1_build_spec"],
        final_step_ids=["step_4_answer"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="arxiv_search", message="search rag"))

    assert result.status == "success"
    assert result.outputs["search_spec"]["intent"] == "arxiv_search"
    assert result.outputs["search_spec"]["sort_by"] == "submittedDate"
    assert result.outputs["search_spec"]["sort_order"] == "descending"
    assert not any(
        trace.event == "step_observed"
        and trace.step_id == "step_1_build_spec"
        and trace.detail.get("observation_status") == "tool_error"
        for trace in result.trace
    )


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


def test_plan_executor_runtime_approved_step_does_not_request_confirmation_again(monkeypatch) -> None:
    calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    def fail_if_interrupted(*args, **kwargs):
        raise AssertionError("approved runtime step must not request confirmation again")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", fail_if_interrupted)

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
    state = AgentState(intent="paper_qa", message="build index")
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
    # resume approve 后批准状态必须跟随 runtime 进入下一次 execute_step，避免同一副作用工具二次确认。
    runtime.approved_step_ids = ["parse_and_index_paper"]

    result = PlanExecutor()._execute_runtime(runtime, state, allow_interrupt=True)

    assert result.status == "success"
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert not any(trace.event == "confirmation_requested" for trace in result.trace)


def test_plan_executor_checkpoint_approved_step_does_not_request_confirmation_again(monkeypatch) -> None:
    calls = []

    class ApprovedCheckpointDatabase:
        def get_agent_runtime_checkpoint(self, **kwargs):
            assert kwargs["user_id"] == "u1"
            assert kwargs["session_id"] == "s1"
            return {
                "status": "running",
                "pending_confirmation": None,
                "runtime_state": {
                    "approved_step_ids": ["parse_and_index_paper"],
                    "plan": {"plan_id": "paper_qa:test"},
                },
            }

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    def fail_if_interrupted(*args, **kwargs):
        raise AssertionError("approved checkpoint step must not request confirmation again")

    monkeypatch.setattr(executor_module, "DatabaseService", ApprovedCheckpointDatabase)
    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", fail_if_interrupted)

    goal, plan = _paper_index_confirmation_plan()
    state = AgentState(user_id="u1", session_id="s1", intent="paper_qa", message="build index")
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}

    result = PlanExecutor()._execute_runtime(runtime, state, allow_interrupt=True)

    assert result.status == "success"
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert runtime.approved_step_ids == ["parse_and_index_paper"]
    assert state.context["approved_step_ids"] == ["parse_and_index_paper"]
    assert not any(trace.event == "confirmation_requested" for trace in result.trace)


def test_plan_executor_checkpoint_approved_step_uses_default_user_id_when_state_user_missing(monkeypatch) -> None:
    calls = []

    class ApprovedCheckpointDatabase:
        def get_agent_runtime_checkpoint(self, **kwargs):
            # stream resume 可能不显式回填 user_id；executor 需要和数据库层保持同样的默认值归一。
            assert kwargs["user_id"] == executor_module.DEFAULT_USER_ID
            assert kwargs["session_id"] == "s1"
            return {
                "status": "running",
                "pending_confirmation": None,
                "runtime_state": {
                    "approved_step_ids": ["parse_and_index_paper"],
                    "plan": {"plan_id": "paper_qa:test"},
                },
            }

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    def fail_if_interrupted(*args, **kwargs):
        raise AssertionError("approved checkpoint step must not request confirmation again")

    monkeypatch.setattr(executor_module, "DatabaseService", ApprovedCheckpointDatabase)
    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", fail_if_interrupted)

    goal, plan = _paper_index_confirmation_plan()
    state = AgentState(session_id="s1", intent="paper_qa", message="build index")
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}

    result = PlanExecutor()._execute_runtime(runtime, state, allow_interrupt=True)

    assert result.status == "success"
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert runtime.approved_step_ids == ["parse_and_index_paper"]
    assert state.context["approved_step_ids"] == ["parse_and_index_paper"]
    assert not any(trace.event == "confirmation_requested" for trace in result.trace)


def test_plan_executor_resume_reentry_consumes_pending_confirmation(monkeypatch) -> None:
    calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"decision": "approve", "step_id": "parse_and_index_paper"})

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
    state = AgentState(
        intent="paper_qa",
        message="build index",
        debug={"pending_confirmation": {"step_id": "parse_and_index_paper", "tool_name": "parse_and_index_paper"}},
    )
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {"parse_and_index_paper": "waiting_confirmation"}
    runtime.current_step_id = "parse_and_index_paper"
    runtime.pending_confirmation = executor_module.ConfirmationRequest(
        step_id="parse_and_index_paper",
        tool_name="parse_and_index_paper",
        action_type="index",
        side_effect_level="external_call",
        reason="explicit_user_confirmation_required",
    )

    result = PlanExecutor().execute_current_step_tool(runtime, state, allow_interrupt=True)

    assert result.next_action in {"observe", "continue"}
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert runtime.approved_step_ids == ["parse_and_index_paper"]
    assert state.context["approved_step_ids"] == ["parse_and_index_paper"]
    assert runtime.pending_confirmation is None
    assert runtime.recovery_strategy is None
    assert state.pending_action["status"] == "approved"
    assert state.pending_action["confirmation_consumed"] is True
    assert "pending_confirmation" not in state.debug
    assert state.debug["confirmation_consumed"]["decision"] == "approve"
    assert state.runtime_state is not None
    assert state.runtime_state.pending_confirmation is None
    assert state.runtime_state.turn_status is None
    assert any(trace.event == "confirmation_approved" for trace in runtime.trace)
    assert any(trace.event == "confirmation_consumed" for trace in runtime.trace)
    events = [trace.event for trace in runtime.trace]
    assert events.index("confirmation_consumed") < events.index("step_started")


def test_plan_executor_resume_reentry_from_request_confirmation_consumes_target_step_approval(monkeypatch) -> None:
    calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "request_confirmation":
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "waiting",
                "data": {
                    "status": "waiting_confirmation",
                    "pending_action": {
                        "type": "tool_approval",
                        "status": "waiting_confirmation",
                        "step_id": "parse_and_index_paper",
                        "tool_name": "parse_and_index_paper",
                        "action_label": "解析并索引论文",
                    },
                },
                "trace": {},
                "error": None,
            }
        if tool_name == "build_paper_qa_index":
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "indexed",
                "data": {"status": "indexed", "has_index": True},
                "trace": {},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(
        executor_module,
        "interrupt",
        lambda payload: {
            "decision": "approve",
            "step_id": "parse_and_index_paper",
            "tool_name": "parse_and_index_paper",
        },
    )

    goal = Goal(goal_id="paper_qa:test", goal_type="paper_qa", user_request="index paper")
    plan = ExecutablePlan(
        plan_id="paper_qa:test",
        goal=goal,
        steps=[
            PlanStep(
                step_id="request_confirmation",
                action_type="clarify",
                tool_name="request_confirmation",
                tool=_tool("request_confirmation"),
                output_key="confirmation_status",
                input_bindings=[
                    StepInputBinding(
                        input_key="pending_action",
                        source_type="literal",
                        value={
                            "type": "tool_approval",
                            "status": "waiting_confirmation",
                            "step_id": "parse_and_index_paper",
                            "tool_name": "parse_and_index_paper",
                            "action_label": "解析并索引论文",
                        },
                    )
                ],
            ),
            PlanStep(
                step_id="parse_and_index_paper",
                action_type="index",
                tool_name="parse_and_index_paper",
                tool=_tool("parse_and_index_paper"),
                output_key="index_build_result",
                depends_on=["request_confirmation"],
                input_bindings=[StepInputBinding(input_key="paper_reference", source_type="literal", value={"arxiv_id": "2401.00001"})],
                confirmation_policy=StepPolicy(
                    policy_type="confirmation",
                    mode="explicit_user_confirmation_required",
                    requires_confirmation=True,
                ),
                side_effect_level="external_call",
            ),
        ],
        entry_step_ids=["request_confirmation"],
        final_step_ids=["parse_and_index_paper"],
    )
    state = AgentState(intent="paper_qa", message="build index")
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {
        "request_confirmation": "waiting_confirmation",
        "parse_and_index_paper": "pending",
    }
    runtime.current_step_id = "request_confirmation"
    runtime.pending_confirmation = executor_module.ConfirmationRequest(
        step_id="parse_and_index_paper",
        tool_name="parse_and_index_paper",
        action_type="index",
        side_effect_level="external_call",
        reason="explicit_user_confirmation_required",
        request_type="tool_approval",
    )

    step_result = PlanExecutor().execute_current_step_tool(runtime, state, allow_interrupt=True)

    assert step_result.next_action in {"observe", "continue"}
    assert runtime.approved_step_ids == ["parse_and_index_paper"]
    assert state.context["approved_step_ids"] == ["parse_and_index_paper"]
    assert runtime.pending_confirmation is None
    assert state.pending_action["step_id"] == "parse_and_index_paper"
    assert state.pending_action["status"] == "approved"


def test_plan_executor_pending_action_approved_does_not_bypass_confirmation(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("display-only pending_action approval must not execute side effect tool")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)
    _goal, plan = _paper_index_confirmation_plan()
    state = AgentState(
        intent="paper_qa",
        message="build index",
        pending_action={
            "status": "approved",
            "decision": "approve",
            "step_id": "parse_and_index_paper",
        },
    )

    result = PlanExecutor().execute(plan, state)

    assert result.status == "waiting_confirmation"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.step_id == "parse_and_index_paper"
    assert any(trace.event == "confirmation_created" for trace in result.trace)


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

    state = AgentState(
        intent="paper_qa",
        message="build index",
        debug={"pending_confirmation": {"step_id": "parse_and_index_paper", "tool_name": "parse_and_index_paper"}},
    )
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}
    result = PlanExecutor()._execute_runtime(runtime, state, allow_interrupt=True)

    assert result.status in {"success", "waiting_confirmation", "fallback"}
    assert result.final_answer == "已取消解析 RAG Paper，因此无法继续基于全文回答。"
    assert any(trace.event == "confirmation_requested" for trace in result.trace)
    assert any(trace.event == "confirmation_rejected" for trace in result.trace)
    assert any(trace.event == "confirmation_consumed" for trace in result.trace)
    assert runtime.pending_confirmation is None
    assert state.pending_action["status"] == "rejected"
    assert state.pending_action["decision"] == "reject"
    assert state.pending_action["confirmation_consumed"] is True
    assert "pending_confirmation" not in state.debug
    assert state.debug["confirmation_consumed"]["decision"] == "reject"
    assert state.runtime_state is not None
    assert state.runtime_state.pending_confirmation is None
    # 拒绝后中间 waiting_confirmation 已被清掉，但最终整轮会收束成可展示的终态。
    assert state.runtime_state.turn_status == "success"
    assert state.runtime_state.recovery_strategy == {"type": "skip_step", "reason": "confirmation_rejected"}
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
    assert replan_traces[0].detail.get("observation_signal") == "success_but_empty_result"
    assert replan_traces[0].detail.get("selected_recovery_action_semantic") == "append_step_after_current"
    assert replan_traces[0].detail.get("policy_source") == "search_empty_recovery_policy"
    assert replan_traces[0].detail.get("tool_recovery_policy")["modes"] == ["retry_step", "patch_plan"]
    assert replan_traces[0].detail.get("recovery_candidates")
    assert any(step.tool_name == "rewrite_arxiv_query" for step in result.plan.steps)
    assert any(trace.event == "step_succeeded" and trace.step_id == "rewrite_arxiv_query" for trace in result.trace)


def test_plan_executor_replans_prefixed_llm_arxiv_steps_without_losing_search_spec(monkeypatch) -> None:
    calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        calls.append(dict(kwargs))
        papers = [] if len(calls) == 1 else [{"arxiv_id": "2401.00001", "title": "RAG retry result"}]
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
        message="帮我找最近 7 天关于 RAG 的 5 篇论文",
        search_spec=ArxivSearchSpec(
            intent="arxiv_search",
            query="RAG",
            submitted_days_ago=7,
            max_results=5,
            sort_by="submittedDate",
            sort_order="descending",
        ),
    )

    result = PlanExecutor().execute(_llm_prefixed_arxiv_plan(), state)

    assert result.status == "success"
    assert len(calls) == 2
    assert calls[1]["query"] == "RAG"
    assert calls[1]["submitted_days_ago"] == 7
    assert calls[1]["max_results"] == 5
    assert "RAG retry result" in result.final_answer
    assert any(trace.event == "plan_replanned" for trace in result.trace)
    assert any(trace.step_id == "rewrite_arxiv_query" and trace.event == "step_succeeded" for trace in result.trace)


def test_plan_executor_injects_parsed_search_spec_when_llm_plan_omits_optional_binding(monkeypatch) -> None:
    calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        calls.append(dict(kwargs))
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG parsed spec"}]},
            "trace": {"tool_name": tool_name, "query": kwargs.get("query")},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    goal = Goal(goal_id="arxiv_search:omitted-spec-binding", goal_type="arxiv_search", user_request="帮我找最近 7 天关于 RAG 的 5 篇论文")
    plan = ExecutablePlan(
        plan_id="arxiv_search:omitted-spec-binding",
        goal=goal,
        steps=[
            _step(
                step_id="normalize_request",
                action_type="write_state",
                tool_name="normalize_request",
                output_key="normalized_request",
                input_bindings=[StepInputBinding(input_key="message", source_type="state", source_key="message")],
            ),
            _step(
                step_id="build_arxiv_search_spec",
                action_type="search",
                tool_name="build_arxiv_search_spec",
                output_key="search_spec",
                input_bindings=[StepInputBinding(input_key="normalized_request", source_type="step_output", step_id="normalize_request")],
                depends_on=["normalize_request"],
            ),
            _step(
                step_id="search_arxiv",
                action_type="search",
                tool_name="search_arxiv",
                output_key="arxiv_results",
                input_bindings=[StepInputBinding(input_key="search_spec", source_type="step_output", step_id="build_arxiv_search_spec")],
                depends_on=["build_arxiv_search_spec"],
            ),
            _step(
                step_id="validate_arxiv_results",
                action_type="validate",
                tool_name="validate_arxiv_results",
                output_key="arxiv_result_quality",
                input_bindings=[StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="search_arxiv")],
                depends_on=["search_arxiv"],
            ),
            _step(
                step_id="synthesize_arxiv_response",
                action_type="answer",
                tool_name="synthesize_arxiv_response",
                output_key="final_answer",
                input_bindings=[
                    StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="search_arxiv"),
                    StepInputBinding(input_key="arxiv_result_quality", source_type="step_output", step_id="validate_arxiv_results"),
                ],
                depends_on=["validate_arxiv_results"],
            ),
        ],
        entry_step_ids=["normalize_request"],
        final_step_ids=["synthesize_arxiv_response"],
    )
    state = AgentState(
        intent="arxiv_search",
        message="帮我找最近 7 天关于 RAG 的 5 篇论文",
        search_spec=ArxivSearchSpec(
            intent="arxiv_search",
            query="RAG",
            categories=["cs.CL", "cs.LG", "cs.IR", "cs.AI"],
            submitted_days_ago=7,
            max_results=5,
            sort_by="submittedDate",
            sort_order="descending",
        ),
    )

    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert calls == [
        {
            "intent": "arxiv_search",
            "query": "RAG",
            "categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"],
            "submitted_days_ago": 7,
            "max_results": 5,
            "sort_by": "submittedDate",
            "sort_order": "descending",
            "field_operator": "AND",
            "category_operator": "OR",
        }
    ]
    assert "RAG parsed spec" in result.final_answer


def test_plan_executor_accepts_llm_user_request_alias_and_wrapped_ranked_papers(monkeypatch) -> None:
    """复现 19:31 的 LLM 草稿形态，避免 state/user_request 和 ranked_papers 包裹体再次打断执行。"""
    calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        calls.append(dict(kwargs))
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2606.00001", "title": "RAG Agents in Practice"}]},
            "trace": {"tool_name": tool_name, "query": kwargs.get("query")},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    goal = Goal(goal_id="arxiv_search:llm-aliases", goal_type="arxiv_search", user_request="帮我找最近 7 天关于 RAG 的 5 篇论文")
    plan = ExecutablePlan(
        plan_id="arxiv_search:llm-aliases",
        goal=goal,
        steps=[
            _step(
                step_id="normalize_request",
                action_type="tool_call",
                tool_name="normalize_request",
                output_key="normalized_request",
                input_bindings=[
                    StepInputBinding(input_key="message", source_type="state", source_key="user_request"),
                ],
            ),
            _step(
                step_id="build_arxiv_search_spec",
                action_type="tool_call",
                tool_name="build_arxiv_search_spec",
                output_key="arxiv_search_spec",
                input_bindings=[
                    StepInputBinding(input_key="normalized_request", source_type="step_output", step_id="normalize_request"),
                ],
                depends_on=["normalize_request"],
            ),
            _step(
                step_id="search_arxiv",
                action_type="tool_call",
                tool_name="search_arxiv",
                output_key="arxiv_results",
                input_bindings=[
                    StepInputBinding(input_key="search_spec", source_type="step_output", step_id="build_arxiv_search_spec"),
                ],
                depends_on=["build_arxiv_search_spec"],
            ),
            _step(
                step_id="validate_arxiv_results",
                action_type="tool_call",
                tool_name="validate_arxiv_results",
                output_key="validation_result",
                input_bindings=[
                    StepInputBinding(input_key="arxiv_results", source_type="step_output", step_id="search_arxiv"),
                ],
                depends_on=["search_arxiv"],
            ),
            _step(
                step_id="synthesize_arxiv_response",
                action_type="tool_call",
                tool_name="synthesize_arxiv_response",
                output_key="final_answer",
                input_bindings=[
                    StepInputBinding(input_key="ranked_papers", source_type="step_output", step_id="search_arxiv"),
                    StepInputBinding(input_key="warnings", source_type="step_output", step_id="validate_arxiv_results"),
                ],
                depends_on=["search_arxiv", "validate_arxiv_results"],
            ),
        ],
        entry_step_ids=["normalize_request"],
        final_step_ids=["synthesize_arxiv_response"],
    )
    state = AgentState(
        intent="arxiv_search",
        message="帮我找最近 7 天关于 RAG 的 5 篇论文",
        search_spec=ArxivSearchSpec(
            intent="arxiv_search",
            query="RAG",
            categories=["cs.CL", "cs.LG", "cs.IR", "cs.AI"],
            submitted_days_ago=7,
            max_results=5,
            sort_by="submittedDate",
            sort_order="descending",
        ),
    )

    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert calls == [
        {
            "intent": "arxiv_search",
            "query": "RAG",
            "categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"],
            "submitted_days_ago": 7,
            "max_results": 5,
            "sort_by": "submittedDate",
            "sort_order": "descending",
            "field_operator": "AND",
            "category_operator": "OR",
        }
    ]
    assert "RAG Agents in Practice" in result.final_answer
    assert not any(trace.detail.get("failure_reason") == "missing_input" for trace in result.trace)


def test_plan_executor_reuses_persistent_write_output_without_duplicate_call(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("persistent_write tool must not run when output already exists")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)

    goal = Goal(goal_id="preference_action:idempotent", goal_type="preference_action", user_request="喜欢这篇")
    plan = ExecutablePlan(
        plan_id="preference_action:idempotent",
        goal=goal,
        steps=[
            PlanStep(
                step_id="update_preference_store",
                action_type="write_state",
                tool_name="update_preference_store",
                tool=_tool("update_preference_store"),
                output_key="preference_action_result",
                input_bindings=[
                    StepInputBinding(input_key="paper_reference", source_type="literal", value={"arxiv_id": "2401.00001", "title": "RAG"}),
                    StepInputBinding(input_key="message", source_type="state", source_key="message"),
                ],
                side_effect_level="persistent_write",
                confirmation_policy=StepPolicy(policy_type="confirmation", mode="explicit_user_confirmation_required", requires_confirmation=True),
            )
        ],
        entry_step_ids=["update_preference_store"],
        final_step_ids=["update_preference_store"],
    )
    runtime = PlanRuntime(
        goal=goal,
        plan=plan,
        outputs={"preference_action_result": {"ok": True, "arxiv_id": "2401.00001"}},
        step_status={"update_preference_store": "pending"},
    )

    result = PlanExecutor().execute_runtime(runtime, AgentState(intent="preference_action", message="喜欢这篇"))

    assert result.status == "success"
    assert result.outputs["preference_action_result"]["arxiv_id"] == "2401.00001"
    assert any(trace.event == "step_reused_output" and trace.step_id == "update_preference_store" for trace in result.trace)


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

    state = AgentState(intent="paper_qa", message="what is the method?")
    plan = _resolved_paper_qa_plan(title="RAG")
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
                "data": {
                    "answer": "grounded answer",
                    "sources": [{"chunk_id": "c1"}],
                    "retrieval_debug": {"stages": {"rerank": {"count": 1}}},
                    "qa_observation": {
                        "schema_version": "paper_qa_observation_v1",
                        "retrieval_quality": "good",
                        "recommended_repair_actions": [],
                    },
                },
                "trace": {"tool_name": tool_name},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(intent="paper_qa", message="what is the method?")
    plan = _resolved_paper_qa_plan()
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert result.final_answer == "grounded answer"
    assert answer_calls == [{"arxiv_id": "2401.00001", "question": "what is the method?"}]
    assert result.outputs["paper_qa_result"]["sources"] == [{"chunk_id": "c1"}]
    assert result.outputs["paper_qa_result"]["retrieval_debug"] == {"stages": {"rerank": {"count": 1}}}
    assert result.outputs["paper_qa_result"]["qa_observation"]["retrieval_quality"] == "good"
    pseudo_steps = {"retrieve_paper_chunks", "rewrite_paper_query", "rerank_paper_chunks", "validate_qa_evidence", "generate_paper_answer", "verify_answer_grounding"}
    assert not pseudo_steps.intersection({step.tool_name for step in result.plan.steps})
    assert not pseudo_steps.intersection({trace.step_id for trace in result.trace})
    quality_traces = [trace for trace in result.trace if trace.event == "paper_qa_quality_decision"]
    assert quality_traces
    assert quality_traces[-1].detail["paper_qa_quality_trace"]["final_decision"] == "finalize"
    assert quality_traces[-1].detail["paper_qa_quality_trace"]["repair_attempted"] is False


def test_plan_executor_paper_qa_repairs_insufficient_evidence_and_records_quality_diff(monkeypatch) -> None:
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
            if len(answer_calls) == 1:
                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "summary": "answered",
                    "data": {
                        "answer": "当前证据不足。",
                        "sources": [{"chunk_id": "weak-1"}],
                        "retrieval_debug": {"route": "hybrid"},
                        "qa_observation": {
                            "schema_version": "paper_qa_observation_v1",
                            "retrieval_quality": "weak",
                            "answer_quality": "insufficient_evidence",
                            "answer_quality_reason": "verification_insufficient_evidence",
                            "answer_insufficient_evidence": "yes",
                            "recommended_repair_actions": ["retry_with_expanded_context"],
                            "source_count": 1,
                        },
                    },
                    "trace": {"tool_name": tool_name},
                    "error": None,
                }
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "answered",
                "data": {
                    "answer": "grounded repaired answer",
                    "sources": [{"chunk_id": "strong-1"}, {"chunk_id": "strong-2"}],
                    "retrieval_debug": {"route": "hybrid", "repair": True},
                    "qa_observation": {
                        "schema_version": "paper_qa_observation_v1",
                        "retrieval_quality": "good",
                        "answer_quality": "grounded",
                        "answer_insufficient_evidence": "no",
                        "recommended_repair_actions": [],
                        "source_count": 2,
                    },
                },
                "trace": {"tool_name": tool_name},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(intent="paper_qa", message="what is the method?")
    plan = _resolved_paper_qa_plan()
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert result.final_answer == "grounded repaired answer"
    assert len(answer_calls) == 2
    assert answer_calls[1]["qa_recovery_strategy"]["repair_actions"] == ["retry_with_expanded_context"]
    replan_trace = next(trace for trace in result.trace if trace.event == "plan_replanned" and trace.detail.get("paper_qa_repair_trace"))
    assert replan_trace.detail["paper_qa_repair_trace"]["selected_repair_action"] == "retry_with_expanded_context"
    quality_trace = [trace for trace in result.trace if trace.event == "paper_qa_quality_decision"][-1].detail["paper_qa_quality_trace"]
    assert quality_trace["repair_attempted"] is True
    assert quality_trace["retrieval_quality_before"] == "weak"
    assert quality_trace["retrieval_quality_after"] == "good"
    assert quality_trace["answer_insufficient_evidence_before"] == "yes"
    assert quality_trace["answer_insufficient_evidence_after"] == "no"
    assert quality_trace["sources_before"] == ["weak-1"]
    assert quality_trace["sources_after"] == ["strong-1", "strong-2"]
    assert quality_trace["final_decision"] == "finalize"


def test_plan_executor_paper_qa_repair_limit_returns_conservative_fallback(monkeypatch) -> None:
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
                "data": {
                    "answer": "当前证据仍不足。",
                    "sources": [{"chunk_id": f"weak-{len(answer_calls)}"}],
                    "retrieval_debug": {"route": "hybrid"},
                    "qa_observation": {
                        "schema_version": "paper_qa_observation_v1",
                        "retrieval_quality": "weak",
                        "retrieval_quality_reason": "source_chunks_too_few:1",
                        "answer_quality": "insufficient_evidence",
                        "answer_quality_reason": "verification_insufficient_evidence",
                        "answer_insufficient_evidence": "yes",
                        "recommended_repair_actions": ["retry_with_expanded_context"],
                        "source_count": 1,
                    },
                },
                "trace": {"tool_name": tool_name},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(intent="paper_qa", message="what is the method?")
    plan = _resolved_paper_qa_plan()
    result = PlanExecutor().execute(plan, state)

    assert result.status == "fallback"
    assert len(answer_calls) == 3
    assert "无法给出足够可靠的论文回答" in result.final_answer
    assert any(trace.event == "replan_limit_exceeded" for trace in result.trace)
    fallback_trace = next(trace for trace in result.trace if trace.event == "replan_fallback")
    assert fallback_trace.detail["paper_qa_final_decision"]["final_decision"] == "fallback"
    assert fallback_trace.detail["paper_qa_final_decision"]["max_repair_limit_triggered"] is True


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


def test_plan_executor_search_missing_search_spec_returns_structured_input_error() -> None:
    goal = Goal(goal_id="arxiv_search:validation", goal_type="arxiv_search", user_request="search")
    plan = ExecutablePlan(
        plan_id="arxiv_search:validation",
        goal=goal,
        steps=[
            PlanStep(
                step_id="search_arxiv",
                action_type="search",
                tool_name="search_arxiv",
                tool=_tool("search_arxiv"),
                output_key="arxiv_results",
            ),
        ],
        entry_step_ids=["search_arxiv"],
        final_step_ids=["search_arxiv"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="arxiv_search", message="rag"))

    assert result.status == "fallback"
    assert result.error.startswith("fallback:search_arxiv:")
    assert result.runtime is not None
    assert result.runtime.recovery_strategy["type"] == "fallback_answer"
    # 主链路进入最终兜底时，需要把结构化 fallback 信息写入 runtime/output，
    # 这样 trace、前端和后续诊断都能拿到统一字段，而不是各处拼接自由文本。
    assert result.runtime.recovery_strategy["fallback_record"]["code"] == "executor_fallback_answer"
    assert result.runtime.outputs["fallback_record"]["code"] == "executor_fallback_answer"
    assert result.runtime.outputs["fallback_record"]["raw_reason"] == "fallback_answer"
    observation_trace = next(trace for trace in result.trace if trace.event == "step_observed" and trace.step_id == "search_arxiv")
    assert observation_trace.detail["observation_status"] == "tool_error"
    assert observation_trace.detail["failure_category"] == "ambiguous_user_request"
    tool_error = observation_trace.detail["evidence"]["tool_error"]
    assert tool_error["error_code"] == "input_validation_error"
    assert tool_error["failed_stage"] == "input_validation"
    assert tool_error["suggested_recovery"] == "ask_clarification"
    assert any(trace.event == "plan_replanned" and trace.detail.get("failure_category") == "ambiguous_user_request" for trace in result.trace)


def test_plan_executor_paper_qa_missing_arxiv_id_returns_structured_tool_error() -> None:
    goal = Goal(goal_id="paper_qa:missing_arxiv_id", goal_type="paper_qa", user_request="what is the method?")
    plan = ExecutablePlan(
        plan_id="paper_qa:missing_arxiv_id",
        goal=goal,
        steps=[
            PlanStep(
                step_id="answer_paper_question",
                action_type="answer",
                tool_name="answer_paper_question",
                tool=_tool("answer_paper_question"),
                output_key="paper_qa_result",
            ),
        ],
        entry_step_ids=["answer_paper_question"],
        final_step_ids=["answer_paper_question"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="paper_qa", message="what is the method?"))

    assert result.status == "fallback"
    assert result.error.startswith("fallback:answer_paper_question:")
    assert result.runtime is not None
    observation_trace = next(trace for trace in result.trace if trace.event == "step_observed" and trace.step_id == "answer_paper_question")
    assert observation_trace.detail["observation_status"] == "tool_error"
    assert observation_trace.detail["failure_category"] == "ambiguous_user_request"
    tool_error = observation_trace.detail["evidence"]["tool_error"]
    assert tool_error["error_code"] == "missing_arxiv_id"
    assert tool_error["failed_stage"] == "input_validation"
    assert tool_error["suggested_recovery"] == "ask_clarification"


def test_plan_executor_backend_exception_is_structured_tool_error(monkeypatch) -> None:
    def failing_backend_tool(tool_name: str, **kwargs):
        raise RuntimeError(f"{tool_name} unavailable")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", failing_backend_tool)
    goal = Goal(goal_id="arxiv_search:backend_exception", goal_type="arxiv_search", user_request="search")
    plan = ExecutablePlan(
        plan_id="arxiv_search:backend_exception",
        goal=goal,
        steps=[
            PlanStep(
                step_id="search_arxiv",
                action_type="search",
                tool_name="search_arxiv",
                tool=_tool("search_arxiv"),
                output_key="arxiv_results",
                input_bindings=[
                    StepInputBinding(
                        input_key="search_spec",
                        source_type="literal",
                        value={"intent": "arxiv_search", "query": "rag", "max_results": 3},
                    )
                ],
            ),
        ],
        entry_step_ids=["search_arxiv"],
        final_step_ids=["search_arxiv"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="arxiv_search", message="rag"))

    assert result.status == "fallback"
    assert result.error.startswith("fallback:search_arxiv:")
    assert result.runtime is not None
    observation_trace = next(trace for trace in result.trace if trace.event == "step_observed" and trace.step_id == "search_arxiv")
    assert observation_trace.detail["observation_status"] == "tool_error"
    assert observation_trace.detail["failure_category"] == "tool_runtime_error"
    tool_error = observation_trace.detail["evidence"]["tool_error"]
    assert tool_error["error_code"] == "tool_runtime_error"
    assert tool_error["failed_stage"] == "adapter_execution"
    assert tool_error["raw_exception_type"] == "RuntimeError"


def test_plan_executor_rejects_adapter_output_that_violates_contract(monkeypatch) -> None:
    class BadFallbackAdapter:
        def execute(self, tool_input):
            return executor_module.ToolExecutionResult(
                ok=True,
                data={"final_answer": 123},
                adapter_name="BadFallbackAdapter",
                tool_name="generate_fallback_response",
            )

    original_contract = PLANNER_TOOL_REGISTRY.get_contract("generate_fallback_response")
    assert original_contract is not None
    monkeypatch.setitem(
        PLANNER_TOOL_REGISTRY._contracts,
        "generate_fallback_response",
        replace(original_contract, adapter=BadFallbackAdapter(), implementation="tests.BadFallbackAdapter"),
    )

    goal = Goal(goal_id="unsupported:bad_output", goal_type="unsupported", user_request="fallback")
    plan = ExecutablePlan(
        plan_id="unsupported:bad_output",
        goal=goal,
        steps=[
            PlanStep(
                step_id="generate_fallback_response",
                action_type="answer",
                tool_name="generate_fallback_response",
                tool=_tool("generate_fallback_response"),
                output_key="final_answer",
                input_bindings=[StepInputBinding(input_key="message", source_type="literal", value="unsupported")],
            ),
        ],
        entry_step_ids=["generate_fallback_response"],
        final_step_ids=["generate_fallback_response"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="unsupported", message="unsupported"))

    assert result.status == "fallback"
    assert result.error.startswith("fallback:generate_fallback_response:")
    assert result.runtime is not None
    observation_trace = next(trace for trace in result.trace if trace.event == "step_observed" and trace.step_id == "generate_fallback_response")
    assert observation_trace.detail["observation_status"] == "tool_error"
    assert observation_trace.detail["failure_category"] == "tool_invalid_output"
    tool_error = observation_trace.detail["evidence"]["tool_error"]
    assert tool_error["error_code"] == "output_validation_error"
