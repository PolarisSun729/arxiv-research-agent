from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
AgentState = _MODULES["state_module"].AgentState
ArxivSearchSpec = _MODULES["schemas"].ArxivSearchSpec
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
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


class AgentGraphFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph_module = _MODULES["graph_module"]
        self.patches: list[mock._patch] = []

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()

    def _patch_graph(self, name: str, value) -> None:
        patcher = mock.patch.object(self.graph_module, name, value)
        self.patches.append(patcher)
        patcher.start()

    def test_search_path_runs_parse_plan_tool_check_and_synthesize(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            return _visit(state, "parse_search_request", intent="arxiv_search")

        def plan(state):
            return _visit(state, "plan_task")

        def build_args(state):
            return _visit(
                state,
                "build_search_tool_args",
                tool_name="search_arxiv_structured",
                tool_args={"query": "rag", "max_results": 5},
            )

        def invoke(state):
            return _visit(state, "invoke_search_tool", tool_result={"ok": True, "papers": [{"arxiv_id": "2401.00001"}]})

        def adapt(state):
            return _visit(state, "adapt_search_tool_result", papers=[{"arxiv_id": "2401.00001", "title": "RAG paper"}])

        def check(state):
            return _visit(state, "check_search_result")

        def rank(state):
            return _visit(state, "personalized_rank_and_annotate_papers", personalized_rerank_applied=True)

        def synthesize(state):
            return _visit(state, "synthesize_response", answer="found papers", next_actions=["read paper"])

        for name, fn in {
            "parse_search_request": parse,
            "plan_task": plan,
            "build_search_tool_args": build_args,
            "invoke_search_tool": invoke,
            "adapt_search_tool_result": adapt,
            "check_search_result": check,
            "personalized_rank_and_annotate_papers": rank,
            "synthesize_response": synthesize,
        }.items():
            self._patch_graph(name, fn)

        graph = build_arxiv_search_graph()
        result = graph.invoke(AgentState(message="search rag papers").model_dump())

        self.assertEqual(result["answer"], "found papers")
        self.assertEqual(
            result["debug"]["visited"],
            [
                "parse_search_request",
                "plan_task",
                "build_search_tool_args",
                "invoke_search_tool",
                "adapt_search_tool_result",
                "check_search_result",
                "personalized_rank_and_annotate_papers",
                "synthesize_response",
            ],
        )
        self.assertEqual(result["papers"][0]["arxiv_id"], "2401.00001")

    def test_empty_search_result_relaxes_and_retries(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            return _visit(state, "parse_search_request", intent="arxiv_search")

        def plan(state):
            return _visit(state, "plan_task")

        def build_args(state):
            current = _coerce_agent_state(state)
            return _visit(
                current,
                "build_search_tool_args",
                tool_name="search_arxiv_structured",
                tool_args={"query": f"rag-{current.search_retry_count}", "max_results": 5},
            )

        def invoke(state):
            current = _coerce_agent_state(state)
            papers = [] if current.search_retry_count == 0 else [{"arxiv_id": "2401.00002", "title": "Retry paper"}]
            return _visit(state, "invoke_search_tool", tool_result={"ok": True, "papers": papers})

        def adapt(state):
            current = _coerce_agent_state(state)
            papers = [] if current.search_retry_count == 0 else [{"arxiv_id": "2401.00002", "title": "Retry paper"}]
            return _visit(state, "adapt_search_tool_result", papers=papers)

        def check(state):
            return _visit(state, "check_search_result")

        def relax(state):
            current = _coerce_agent_state(state)
            return _visit(state, "relax_search_for_retry", search_retry_count=current.search_retry_count + 1)

        def rank(state):
            return _visit(state, "personalized_rank_and_annotate_papers")

        def synthesize(state):
            return _visit(state, "synthesize_response", answer="retry success")

        for name, fn in {
            "parse_search_request": parse,
            "plan_task": plan,
            "build_search_tool_args": build_args,
            "invoke_search_tool": invoke,
            "adapt_search_tool_result": adapt,
            "check_search_result": check,
            "relax_search_for_retry": relax,
            "personalized_rank_and_annotate_papers": rank,
            "synthesize_response": synthesize,
        }.items():
            self._patch_graph(name, fn)

        graph = build_arxiv_search_graph()
        result = graph.invoke(AgentState(message="search retry").model_dump())

        self.assertEqual(result["search_retry_count"], 1)
        self.assertEqual(result["answer"], "retry success")
        self.assertIn("relax_search_for_retry", result["debug"]["visited"])
        self.assertEqual(result["debug"]["visited"].count("build_search_tool_args"), 2)

    def test_pending_confirmation_branch_routes_to_handle_node(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            return _visit(state, "parse_search_request", intent="paper_qa")

        def plan(state):
            return _visit(state, "plan_task")

        def classify(state, generation_service=None):
            del generation_service
            current = _coerce_agent_state(state)
            debug = dict(current.debug or {})
            debug["pending_action_decision"] = "confirm"
            return _visit(current, "classify_pending_action_confirmation", debug=debug)

        def handle(state):
            return _visit(
                state,
                "handle_pending_action_confirmation",
                pending_action=None,
                paper_qa_result={"status": "success", "answer": "qa answer"},
            )

        def synthesize(state):
            return _visit(state, "synthesize_response", answer="qa answer")

        for name, fn in {
            "parse_search_request": parse,
            "plan_task": plan,
            "classify_pending_action_confirmation": classify,
            "handle_pending_action_confirmation": handle,
            "synthesize_response": synthesize,
        }.items():
            self._patch_graph(name, fn)

        graph = build_arxiv_search_graph()
        initial_state = AgentState(
            message="yes",
            pending_action={"type": "parse_then_qa", "status": "waiting_confirmation", "arxiv_id": "2401.00001"},
        )
        result = graph.invoke(initial_state.model_dump())

        self.assertEqual(result["answer"], "qa answer")
        self.assertEqual(
            result["debug"]["visited"],
            [
                "parse_search_request",
                "plan_task",
                "handle_pending_action_confirmation",
                "synthesize_response",
            ],
        )

    def test_preference_action_branch_routes_to_preference_node(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            return _visit(state, "parse_search_request", intent="preference_action")

        def plan(state):
            return _visit(state, "plan_task")

        def preference(state):
            return _visit(
                state,
                "apply_preference_action",
                preference_action_result={"status": "success", "action": "like", "arxiv_id": "2401.00001"},
            )

        def synthesize(state):
            return _visit(state, "synthesize_response", answer="preference updated")

        for name, fn in {
            "parse_search_request": parse,
            "plan_task": plan,
            "apply_preference_action": preference,
            "synthesize_response": synthesize,
        }.items():
            self._patch_graph(name, fn)

        graph = build_arxiv_search_graph()
        result = graph.invoke(AgentState(message="like the first paper").model_dump())

        self.assertEqual(result["answer"], "preference updated")
        self.assertIn("apply_preference_action", result["debug"]["visited"])

    def test_paper_reading_branch_routes_to_paper_qa_node(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            return _visit(state, "parse_search_request", intent="paper_qa")

        def plan(state):
            return _visit(state, "plan_task")

        def handle_reading(state):
            return _visit(
                state,
                "handle_paper_reading_request",
                paper_qa_result={"status": "success", "answer": "paper qa answer"},
            )

        def synthesize(state):
            return _visit(state, "synthesize_response", answer="paper qa answer")

        for name, fn in {
            "parse_search_request": parse,
            "plan_task": plan,
            "handle_paper_reading_request": handle_reading,
            "synthesize_response": synthesize,
        }.items():
            self._patch_graph(name, fn)

        graph = build_arxiv_search_graph()
        result = graph.invoke(AgentState(message="what does this paper say?").model_dump())

        self.assertEqual(result["answer"], "paper qa answer")
        self.assertEqual(
            result["debug"]["visited"],
            ["parse_search_request", "plan_task", "handle_paper_reading_request", "synthesize_response"],
        )

    def test_plan_driven_search_real_nodes_return_papers(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = _coerce_agent_state(state)
            current.intent = "arxiv_search"
            current.search_spec = ArxivSearchSpec(intent="arxiv_search", query="rag", categories=["cs.CL"], max_results=5)
            return current.model_dump()

        self._patch_graph("parse_search_request", parse)

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
            graph = build_arxiv_search_graph()
            result = graph.invoke(AgentState(message="search rag papers").model_dump())

        self.assertEqual(result["papers"][0]["arxiv_id"], "2401.00001")
        self.assertEqual(result["tool_observations"][-1].tool_name, "search_arxiv_structured")
        self.assertTrue(result["tool_observations"][-1].ok)
        self.assertEqual(result["execution_plan"][1].status, "completed")
        self.assertEqual(result["execution_plan"][2].status, "completed")
        self.assertEqual(result["execution_plan"][4].status, "completed")
        self.assertEqual(result["debug"]["plan_step_mapping"]["status"], "mapped")

    def test_plan_driven_search_empty_result_retries_then_succeeds(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = _coerce_agent_state(state)
            current.intent = "arxiv_search"
            current.search_spec = ArxivSearchSpec(intent="arxiv_search", query="rag agents", categories=["cs.CL"], max_results=5)
            return current.model_dump()

        self._patch_graph("parse_search_request", parse)

        def fake_invoke_tool(_tool_name, **kwargs):
            query = kwargs.get("query")
            papers = [] if query == "rag agents" else [{"arxiv_id": "2401.00002", "title": "Retry Paper"}]
            return {
                "ok": True,
                "tool_name": "search_arxiv_structured",
                "summary": f"searched {query}",
                "data": {"papers": papers},
                "trace": {"tool_name": "search_arxiv_structured", "final_search_query": query},
                "error": None,
            }

        with mock.patch.object(tool_module, "invoke_tool", side_effect=fake_invoke_tool):
            graph = build_arxiv_search_graph()
            result = graph.invoke(AgentState(message="search rag agents papers").model_dump())

        self.assertEqual(result["search_retry_count"], 1)
        self.assertEqual(result["papers"][0]["arxiv_id"], "2401.00002")
        self.assertIn("relaxed_query", result["fallback_specs"][0])
        self.assertEqual(result["execution_plan"][1].status, "completed")
        self.assertEqual(result["execution_plan"][2].status, "completed")
        self.assertTrue(any("自动放宽关键词" in warning for warning in result["warnings"]))

    def test_plan_driven_search_tool_failure_degrades_to_response(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = _coerce_agent_state(state)
            current.intent = "arxiv_search"
            current.search_spec = ArxivSearchSpec(intent="arxiv_search", query="rag", categories=["cs.CL"], max_results=5)
            return current.model_dump()

        self._patch_graph("parse_search_request", parse)

        with mock.patch.object(tool_module, "invoke_tool", side_effect=RuntimeError("boom")):
            graph = build_arxiv_search_graph()
            result = graph.invoke(AgentState(message="search rag papers").model_dump())

        self.assertTrue(result["answer"])
        self.assertEqual(result["tool_observations"][-1].status, "failed")
        self.assertEqual(result["execution_plan"][1].status, "failed")
        self.assertEqual(result["execution_plan"][2].status, "failed")
        self.assertTrue(any("工具调用失败" in warning for warning in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
