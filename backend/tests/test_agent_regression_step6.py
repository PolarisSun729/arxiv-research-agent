from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


def _load_agent_runtime_helper():
    helper_path = Path(__file__).resolve().parent / "helpers" / "agent_runtime.py"
    module_name = "backend.tests.helpers.agent_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


load_agent_test_modules = _load_agent_runtime_helper().load_agent_test_modules


_MODULES = load_agent_test_modules()
AgentState = _MODULES["state_module"].AgentState
ArxivSearchSpec = _MODULES["schemas"].ArxivSearchSpec
AgentTurnResult = _MODULES["schemas"].AgentTurnResult
ExecutionTrace = _MODULES["schemas"].ExecutionTrace
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
graph_module = _MODULES["graph_module"]
tool_node_module = sys.modules["backend.agents.arxiv_search_agent.node.tool_node"]
paper_reading_module = sys.modules["backend.agents.arxiv_search_agent.node.paper_reading_node"]
preference_module = sys.modules["backend.agents.arxiv_search_agent.node.preference_node"]


def _coerce_state(state):
    if isinstance(state, AgentState):
        return state
    return AgentState.model_validate(state)


class AgentRegressionStep6Tests(unittest.TestCase):
    def test_arxiv_search_still_works_with_new_turn_runtime(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = _coerce_state(state).model_copy(deep=True)
            current.intent = "arxiv_search"
            current.search_spec = ArxivSearchSpec(intent="arxiv_search", query="RAG agent", max_results=5)
            return current

        def fake_run_agent_turn(state):
            del state
            return AgentTurnResult(
                status="success",
                final_answer="已找到 2 篇与 RAG agent 相关的论文。",
                outputs={
                    "search_spec": {"query": "RAG agent", "max_results": 5},
                    "arxiv_results": {
                        "papers": [
                            {"arxiv_id": "2401.00001", "title": "RAG Agents Survey"},
                            {"arxiv_id": "2401.00002", "title": "Agentic Retrieval for RAG"},
                        ]
                    },
                    "ranked_papers": [
                        {"arxiv_id": "2401.00001", "title": "RAG Agents Survey"},
                        {"arxiv_id": "2401.00002", "title": "Agentic Retrieval for RAG"},
                    ],
                },
                trace=[
                    # 这里保留运行时 trace，验证 graph 节点会把执行摘要正确写回调试状态。
                    ExecutionTrace(step_id="step_search", event="tool_completed", status="success", detail={"tool_name": "search_arxiv"}),
                    ExecutionTrace(step_id="step_rank", event="tool_completed", status="success", detail={"tool_name": "personalize_paper_results"}),
                ],
            )

        with mock.patch.object(graph_module, "parse_search_request", side_effect=parse), mock.patch.object(
            graph_module, "run_agent_turn_in_graph", side_effect=fake_run_agent_turn
        ) as mocked_run_agent_turn:
            graph = build_arxiv_search_graph()
            result = AgentState.model_validate(graph.invoke(AgentState(message="搜索 RAG agent 相关论文").model_dump()))

        mocked_run_agent_turn.assert_called_once()
        self.assertEqual(result.intent, "arxiv_search")
        self.assertEqual(result.steps[-1].step, "run_agent_turn")
        self.assertEqual(result.answer, "已找到 2 篇与 RAG agent 相关的论文。")
        self.assertEqual(len(result.papers), 2)
        self.assertEqual(result.debug["agent_turn"]["status"], "success")
        self.assertIn("tool_completed", result.debug["agent_turn"]["trace_events"])

    def test_paper_qa_with_existing_index_calls_check_then_answer(self) -> None:
        state = AgentState(
            intent="paper_qa",
            message="问当前论文的方法流程",
            context={
                "selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Paper"},
                "last_papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
            },
        )

        def fake_invoke_tool(tool_name, **kwargs):
            if tool_name == "check_paper_qa_index":
                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "summary": "qa index exists",
                    "data": {"status": "indexed", "has_index": True},
                    "trace": {"tool_name": tool_name, "kwargs": kwargs},
                    "error": None,
                }
            if tool_name == "answer_paper_question":
                return {
                    "ok": True,
                    "tool_name": tool_name,
                    "summary": "answered question",
                    "data": {"answer": "这是当前论文的方法流程。", "sources": [{"chunk_id": "1"}]},
                    "trace": {"tool_name": tool_name, "kwargs": kwargs},
                    "error": None,
                }
            raise AssertionError(f"unexpected tool: {tool_name}")

        with mock.patch.object(tool_node_module, "invoke_tool", side_effect=fake_invoke_tool) as mocked_tool:
            result = paper_reading_module.handle_paper_reading_request(state)

        self.assertEqual([call.args[0] for call in mocked_tool.call_args_list], ["check_paper_qa_index", "answer_paper_question"])
        self.assertEqual(result.paper_qa_result["status"], "success")
        self.assertEqual([obs.tool_name for obs in result.tool_observations], ["check_paper_qa_index", "answer_paper_question"])

    def test_paper_qa_without_index_delegates_confirmation_to_plan_executor(self) -> None:
        state = AgentState(
            intent="paper_summary",
            message="总结当前论文",
            context={
                "selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Paper"},
                "last_papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
            },
        )

        with mock.patch.object(
            tool_node_module,
            "invoke_tool",
            return_value={
                "ok": True,
                "tool_name": "check_paper_qa_index",
                "summary": "qa index missing",
                "data": {"status": "missing", "has_index": False},
                "trace": {"tool_name": "check_paper_qa_index"},
                "error": None,
            },
        ) as mocked_tool:
            result = paper_reading_module.handle_paper_reading_request(state)

        self.assertEqual([call.args[0] for call in mocked_tool.call_args_list], ["check_paper_qa_index"])
        self.assertIsNone(result.pending_action)
        self.assertEqual(result.paper_qa_result["status"], "failed")
        self.assertEqual(result.paper_qa_result["error"], "paper_index_missing_requires_plan_executor_confirmation")
        self.assertTrue(any("Agent 主流程" in action for action in result.next_actions))

    def test_preference_action_writes_via_tool_protocol(self) -> None:
        state = AgentState(
            intent="preference_action",
            user_id="u1",
            message="喜欢第一篇",
            context={
                "last_papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
                "papers": [{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
            },
        )

        with mock.patch.object(
            tool_node_module,
            "invoke_tool",
            return_value={
                "ok": True,
                "tool_name": "record_paper_preference",
                "summary": "recorded preference",
                "data": {
                    "message": "偏好已更新",
                    "paper": {"arxiv_id": "2401.00001", "title": "RAG Paper"},
                },
                "trace": {"tool_name": "record_paper_preference"},
                "error": None,
            },
        ) as mocked_tool:
            result = preference_module.apply_preference_action(state)

        self.assertEqual(mocked_tool.call_args_list[0].args[0], "record_paper_preference")
        self.assertEqual(result.preference_action_result["status"], "success")
        self.assertTrue(any(obs.tool_name == "record_paper_preference" for obs in result.tool_observations))

    def test_recommendation_runs_through_new_turn_runtime_and_returns_papers(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = _coerce_state(state).model_copy(deep=True)
            current.intent = "recommendation"
            return current

        def fake_run_agent_turn(state):
            del state
            return AgentTurnResult(
                status="success",
                final_answer="我根据你最近关注的主题整理了 2 篇推荐论文。",
                outputs={
                    "recommended_papers": [
                        {"arxiv_id": "2401.10001", "title": "Personalized RAG Recommender"},
                        {"arxiv_id": "2401.10002", "title": "Interest-aware Agent Retrieval"},
                    ],
                    "ranked_papers": [
                        {"arxiv_id": "2401.10001", "title": "Personalized RAG Recommender"},
                        {"arxiv_id": "2401.10002", "title": "Interest-aware Agent Retrieval"},
                    ],
                },
                trace=[
                    ExecutionTrace(
                        step_id="step_recommend",
                        event="tool_completed",
                        status="success",
                        detail={"tool_name": "generate_recommendations"},
                    )
                ],
            )

        with mock.patch.object(graph_module, "parse_search_request", side_effect=parse), mock.patch.object(
            graph_module, "run_agent_turn_in_graph", side_effect=fake_run_agent_turn
        ) as mocked_run_agent_turn:
            graph = build_arxiv_search_graph()
            initial_state = AgentState(
                user_id="u1",
                message="根据我的兴趣推荐几篇论文",
                context={"user_memory_summary": "对 RAG 和 agent 很感兴趣"},
            )
            result = AgentState.model_validate(graph.invoke(initial_state.model_dump()))

        mocked_run_agent_turn.assert_called_once()
        self.assertEqual(result.steps[-1].step, "run_agent_turn")
        self.assertEqual(result.debug["agent_turn"]["status"], "success")
        self.assertEqual(len(result.papers), 2)
        self.assertTrue(bool(result.papers) or bool(result.answer and result.answer.strip()))


if __name__ == "__main__":
    unittest.main()
