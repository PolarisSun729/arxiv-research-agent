from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
AgentState = _MODULES["state_module"].AgentState
ExecutionPlanStep = _MODULES["schemas"].ExecutionPlanStep
ToolObservation = _MODULES["schemas"].ToolObservation
ToolCallRequest = _MODULES["schemas"].ToolCallRequest
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
graph_module = _MODULES["graph_module"]
paper_reading_module = sys.modules["backend.agents.arxiv_search_agent.node.paper_reading_node"]
preference_module = sys.modules["backend.agents.arxiv_search_agent.node.preference_node"]
response_module = sys.modules["backend.agents.arxiv_search_agent.node.response_node"]
plan_step_mapping_module = sys.modules["backend.agents.arxiv_search_agent.node.plan_step_mapping"]
search_module = sys.modules["backend.agents.arxiv_search_agent.node.search_node"]
tool_module = sys.modules["backend.agents.arxiv_search_agent.node.tool_node"]


def _coerce_agent_state(state):
    if isinstance(state, AgentState):
        return state
    return AgentState.model_validate(state)


def _visit(state, node_name: str, **updates):
    current = _coerce_agent_state(state).model_copy(deep=True)
    debug = dict(current.debug or {})
    visited = list(debug.get("visited", []))
    visited.append(node_name)
    debug["visited"] = visited
    current.debug = debug
    for key, value in updates.items():
        setattr(current, key, value)
    return current


def _append_mock_observation(state: AgentState, tool_name: str, ok: bool, summary: str) -> AgentState:
    state.tool_observations = list(state.tool_observations or []) + [
        ToolObservation(
            tool_name=tool_name,
            ok=ok,
            status="success" if ok else "failed",
            result_summary=summary,
            result_ref={"tool_name": tool_name},
            error=None if ok else {"message": summary},
            is_sufficient=ok,
            next_action_hint=None,
            raw_trace={"tool_name": tool_name, "source": "test"},
        )
    ]
    return state


class AgentToolProtocolStage3Tests(unittest.TestCase):
    def test_invoke_search_tool_runs_execute_observe_plan_closure(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            search_spec=_MODULES["schemas"].ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="pending", depends_on=["step_1"]),
            ],
        )

        with mock.patch.object(
            tool_module,
            "invoke_tool",
            return_value={
                "ok": True,
                "tool_name": "search_arxiv_structured",
                "summary": "searched",
                "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}]},
                "trace": {"tool_name": "search_arxiv_structured"},
                "error": None,
            },
        ):
            result = search_module.invoke_search_tool(state)

        self.assertEqual(result.steps[-1].step, "search_tool_call")
        self.assertEqual(result.tool_call_request.plan_step_id, "step_2")
        self.assertEqual(result.tool_observations[-1].tool_name, "search_arxiv_structured")
        self.assertTrue(result.tool_observations[-1].ok)
        plan_status = {step.step_id: step.status for step in result.execution_plan}
        self.assertEqual(plan_status["step_2"], "success")

    def test_search_tool_request_is_built_via_plan_step_mapping(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            search_spec=_MODULES["schemas"].ArxivSearchSpec(
                intent="arxiv_search",
                query="rag",
                categories=["cs.CL"],
                max_results=5,
                sort_by="submittedDate",
                sort_order="descending",
            ),
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="pending", depends_on=["step_1"]),
            ],
        )

        mapped_request = ToolCallRequest(
            tool_name="search_arxiv_structured",
            arguments={"query": "mapped-rag", "max_results": 7, "categories": ["cs.CL"], "start": 0},
            reason="mapped by test",
            expected_result="papers",
            plan_step_id="step_2",
            fallback_tools=["search_arxiv_raw"],
        )

        with mock.patch.object(
            search_module,
            "build_tool_call_request_from_plan_step",
            return_value=(
                mapped_request,
                {"status": "mapped", "step_id": "step_2", "step_type": "search_execution", "tool_name": "search_arxiv_structured"},
            ),
        ) as patched:
            result = search_module.build_search_tool_args(state)

        patched.assert_called_once()
        self.assertEqual(result.tool_call_request.plan_step_id, "step_2")
        self.assertEqual(result.tool_call_request.arguments["query"], "mapped-rag")
        self.assertEqual(result.tool_args["max_results"], 7)

    def test_plan_step_mapping_skips_unsupported_step_safely(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            search_spec=_MODULES["schemas"].ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="completed", depends_on=["step_1"]),
                ExecutionPlanStep(step_id="step_3", step_type="result_validation", description="validate", status="pending", depends_on=["step_2"]),
            ],
        )

        request, debug_payload = plan_step_mapping_module.build_tool_call_request_from_plan_step(state)

        self.assertIsNone(request)
        self.assertEqual(debug_payload["status"], "skipped")
        self.assertEqual(debug_payload["reason"], "unsupported_plan_step_mapping")
        self.assertEqual(debug_payload["action_type"], "result_validation")

    def test_invoke_search_tool_skips_unmappable_step_and_routes_to_response(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="pending", depends_on=["step_1"]),
            ],
        )

        result = search_module.invoke_search_tool(state)

        self.assertEqual(result.tool_observations[-1].status, "skipped")
        self.assertEqual(result.execution_plan[1].status, "skipped")
        self.assertIn("工具执行已跳过", result.warnings[-1])

    def test_search_tool_execution_updates_execution_plan_status(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            tool_call_request=ToolCallRequest(
                tool_name="search_arxiv_structured",
                arguments={"query": "rag", "max_results": 5},
                plan_step_id="step_2",
            ),
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="pending"),
            ],
        )

        with mock.patch.object(
            tool_module,
            "invoke_tool",
            return_value={
                "ok": True,
                "tool_name": "search_arxiv_structured",
                "summary": "searched",
                "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}]},
                "trace": {"tool_name": "search_arxiv_structured"},
                "error": None,
            },
        ):
            result = tool_module.execute_tool(state)

        plan_status = {step.step_id: step.status for step in result.execution_plan}
        self.assertEqual(plan_status["step_2"], "success")
        self.assertEqual(result.tool_observations[-1].tool_name, "search_arxiv_structured")
        self.assertTrue(result.tool_observations[-1].ok)

    def test_search_result_validation_updates_plan_and_retry_route(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            search_retry_count=0,
            tool_result={
                "ok": True,
                "tool_name": "search_arxiv_structured",
                "summary": "searched",
                "data": {"papers": []},
                "trace": {"tool_name": "search_arxiv_structured"},
                "error": None,
            },
            papers=[],
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="completed"),
                ExecutionPlanStep(step_id="step_3", step_type="result_validation", description="validate", status="pending"),
            ],
        )

        checked = search_module.check_search_result(state)
        plan_status = {step.step_type: step.status for step in checked.execution_plan}
        self.assertEqual(plan_status["result_validation"], "failed")
        self.assertEqual(checked.steps[-1].step, "search_result_check")

    def test_failed_tool_execution_links_plan_step_to_observation(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            tool_call_request=ToolCallRequest(
                tool_name="search_arxiv_structured",
                arguments={"query": "rag", "max_results": 5},
                plan_step_id="step_2",
            ),
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="pending"),
            ],
        )

        with mock.patch.object(
            tool_module,
            "invoke_tool",
            return_value={
                "ok": False,
                "tool_name": "search_arxiv_structured",
                "summary": "search failed",
                "data": None,
                "trace": {"tool_name": "search_arxiv_structured"},
                "error": {"code": "search_failed", "message": "boom"},
            },
        ):
            result = tool_module.execute_tool(state)

        plan_status = {step.step_id: step.status for step in result.execution_plan}
        self.assertEqual(plan_status["step_2"], "failed")
        runtime = dict((result.debug or {}).get("execution_plan_runtime") or {})
        runtime_steps = {step["step_id"]: step for step in list(runtime.get("steps", []))}
        self.assertEqual(runtime_steps["step_2"]["last_tool_observation"]["tool_name"], "search_arxiv_structured")
        self.assertEqual(runtime_steps["step_2"]["last_tool_observation"]["status"], "failed")

    def test_paper_qa_tool_observation_uses_quality_summary(self) -> None:
        insufficient_details = tool_module._derive_observation_details(
            "answer_paper_question",
            {
                "ok": True,
                "data": {
                    "answer": "guarded answer",
                    "qa_observation": {
                        "answer_quality": "insufficient_evidence",
                        "answer_insufficient_evidence": "yes",
                        "recommended_repair_actions": ["retry_with_expanded_context"],
                    },
                },
            },
        )
        degraded_details = tool_module._derive_observation_details(
            "answer_paper_question",
            {
                "ok": True,
                "data": {
                    "answer": "grounded answer",
                    "qa_observation": {
                        "retrieval_quality": "partial",
                        "degraded_stages": ["rerank"],
                        "recommended_repair_actions": ["retry_with_query_rewrite"],
                    },
                },
            },
        )

        self.assertFalse(insufficient_details["is_sufficient"])
        self.assertEqual(insufficient_details["next_action_hint"], "retry_with_expanded_context")
        self.assertTrue(degraded_details["is_sufficient"])
        self.assertEqual(degraded_details["next_action_hint"], "retry_with_query_rewrite")

    def test_tool_failure_still_allows_final_response_generation(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            search_spec=_MODULES["schemas"].ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="pending", depends_on=["step_1"]),
                ExecutionPlanStep(step_id="step_3", step_type="result_validation", description="validate", status="pending", depends_on=["step_2"]),
                ExecutionPlanStep(step_id="step_4", step_type="personalization", description="personalize", status="pending", depends_on=["step_3"]),
                ExecutionPlanStep(step_id="step_5", step_type="response_synthesis", description="respond", status="pending", depends_on=["step_4"]),
            ],
        )

        with mock.patch.object(tool_module, "invoke_tool", side_effect=RuntimeError("boom")):
            executed = search_module.invoke_search_tool(state)

        adapted = search_module.adapt_search_tool_result(executed)
        checked = search_module.check_search_result(adapted)
        final_state = response_module.synthesize_response(checked)

        self.assertEqual(executed.tool_observations[-1].status, "failed")
        self.assertTrue(final_state.answer)
        self.assertEqual(final_state.execution_plan[-1].step_type, "response_synthesis")

    def test_response_synthesis_marks_plan_completed(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="search rag papers",
            search_spec=_MODULES["schemas"].ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
            papers=[{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
            personalized_rerank_applied=False,
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="goal_interpretation", description="goal", status="completed"),
                ExecutionPlanStep(step_id="step_2", step_type="search_execution", description="search", status="completed"),
                ExecutionPlanStep(step_id="step_3", step_type="result_validation", description="validate", status="completed"),
                ExecutionPlanStep(step_id="step_4", step_type="personalization", description="personalize", status="skipped"),
                ExecutionPlanStep(step_id="step_5", step_type="response_synthesis", description="respond", status="pending"),
            ],
        )

        result = response_module.synthesize_response(state)

        plan_status = {step.step_type: step.status for step in result.execution_plan}
        self.assertEqual(plan_status["response_synthesis"], "success")
        self.assertEqual(result.steps[-1].step, "final_answer_generation")

    def test_paper_reading_uses_check_then_answer_tool_protocol(self) -> None:
        state = AgentState(
            intent="paper_qa",
            message="问一下第一篇论文的方法",
            context={
                "papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
                "last_papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
            },
        )

        def fake_execute_tool(current_state):
            current = _coerce_agent_state(current_state).model_copy(deep=True)
            request = current.tool_call_request
            self.assertIsInstance(request, ToolCallRequest)
            if request.tool_name == "check_paper_qa_index":
                current.tool_result = {
                    "ok": True,
                    "tool_name": request.tool_name,
                    "summary": "checked index",
                    "data": {"status": "indexed", "has_index": True},
                    "trace": {"tool_name": request.tool_name},
                    "error": None,
                }
                return _append_mock_observation(current, request.tool_name, True, "checked index")
            self.assertEqual(request.tool_name, "answer_paper_question")
            current.tool_result = {
                "ok": True,
                "tool_name": request.tool_name,
                "summary": "answered question",
                "data": {"answer": "这是论文答案", "sources": [{"chunk_id": "1"}]},
                "trace": {"tool_name": request.tool_name},
                "error": None,
            }
            return _append_mock_observation(current, request.tool_name, True, "answered question")

        with mock.patch.object(paper_reading_module, "execute_tool", side_effect=fake_execute_tool) as patched:
            result = paper_reading_module.handle_paper_reading_request(state)

        self.assertEqual(patched.call_count, 2)
        self.assertEqual(result.paper_qa_result["status"], "success")
        self.assertEqual(result.paper_qa_result["answer"], "这是论文答案")
        self.assertEqual([obs.tool_name for obs in result.tool_observations], ["check_paper_qa_index", "answer_paper_question"])

    def test_preference_action_uses_tool_protocol(self) -> None:
        state = AgentState(
            intent="preference_action",
            user_id="u1",
            message="喜欢第一篇",
            context={
                "papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
                "last_papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
            },
        )

        def fake_execute_tool(current_state):
            current = _coerce_agent_state(current_state).model_copy(deep=True)
            request = current.tool_call_request
            self.assertEqual(request.tool_name, "record_paper_preference")
            current.tool_result = {
                "ok": True,
                "tool_name": request.tool_name,
                "summary": "recorded preference",
                "data": {"message": "偏好已更新", "paper": {"arxiv_id": "2401.00001", "title": "RAG Paper"}},
                "trace": {"tool_name": request.tool_name},
                "error": None,
            }
            return _append_mock_observation(current, request.tool_name, True, "recorded preference")

        with mock.patch.object(preference_module, "execute_tool", side_effect=fake_execute_tool) as patched:
            result = preference_module.apply_preference_action(state)

        self.assertEqual(patched.call_count, 1)
        self.assertEqual(result.preference_action_result["status"], "success")
        self.assertEqual(result.preference_action_result["label"], "liked")
        self.assertEqual(result.tool_observations[-1].tool_name, "record_paper_preference")

    def test_run_agent_turn_compatibility_node_applies_runtime_result(self) -> None:
        def fake_run_agent_turn(state):
            self.assertEqual(_coerce_agent_state(state).intent, "recommendation")
            return _MODULES["schemas"].AgentTurnResult(
                status="success",
                final_answer="recommended papers",
                outputs={"ranked_papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}]},
                trace=[],
            )

        with mock.patch.object(
            graph_module,
            "run_agent_turn_in_graph",
            side_effect=fake_run_agent_turn,
        ):
            result_state = graph_module.run_agent_turn_node(AgentState(intent="recommendation", message="给我推荐一些论文"))

        self.assertEqual(result_state.answer, "recommended papers")
        self.assertEqual(result_state.debug["agent_turn"]["status"], "success")
        self.assertEqual(len(result_state.papers), 1)


if __name__ == "__main__":
    unittest.main()
