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
graph_module = _MODULES["graph_module"]
tool_node_module = sys.modules["backend.agents.arxiv_search_agent.node.tool_node"]
paper_reading_module = sys.modules["backend.agents.arxiv_search_agent.node.paper_reading_node"]
preference_module = sys.modules["backend.agents.arxiv_search_agent.node.preference_node"]


def _coerce_state(state):
    if isinstance(state, AgentState):
        return state
    return AgentState.model_validate(state)


class AgentRegressionStep6Tests(unittest.TestCase):
    def test_arxiv_search_enters_explicit_goal_and_plan_nodes(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = _coerce_state(state).model_copy(deep=True)
            current.intent = "arxiv_search"
            current.search_spec = ArxivSearchSpec(intent="arxiv_search", query="RAG agent", max_results=5)
            return current

        parsed_state = parse(AgentState(message="搜索 RAG agent 相关论文"))
        goal_state = graph_module.build_goal_node(parsed_state)
        plan_state = graph_module.build_plan_node(goal_state)

        # 测试当前主图的显式 planning 节点，避免旧单节点 shim 继续伪装成正式执行链路。
        self.assertEqual(goal_state.goal.goal_type, "arxiv_search")
        self.assertIsNotNone(plan_state.execution_plan)
        self.assertIsNotNone(plan_state.plan_runtime)
        tool_names = {step.tool_name for step in list(plan_state.execution_plan.steps or [])}
        self.assertTrue({"build_arxiv_search_spec", "search_arxiv", "validate_arxiv_results"}.issubset(tool_names))
        self.assertEqual(plan_state.debug["planner"]["selected_plan_source"], "tool_aware_rule_based")
        self.assertIn("build_plan", [step.step for step in plan_state.steps])

    def test_paper_reading_compat_node_stops_before_tool_execution_without_resolved_target(self) -> None:
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

        # 旧阅读节点只能提取引用线索；最终目标解析和工具执行由主图 PlanExecutor 负责。
        self.assertEqual(mocked_tool.call_args_list, [])
        self.assertEqual(result.paper_qa_result["status"], "failed")
        self.assertIn("论文", result.paper_qa_result["error"])
        self.assertEqual(result.tool_observations, [])

    def test_paper_reading_compat_node_does_not_create_confirmation_without_resolved_target(self) -> None:
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

        self.assertEqual(mocked_tool.call_args_list, [])
        self.assertIsNone(result.pending_action)
        self.assertEqual(result.paper_qa_result["status"], "failed")
        self.assertIn("论文", result.paper_qa_result["error"])
        self.assertTrue(any("arXiv ID" in action for action in result.next_actions))

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

        # 引用线索未解析成最终论文前不能写入持久化偏好，避免旧兼容节点误写错对象。
        self.assertEqual(mocked_tool.call_args_list, [])
        self.assertEqual(result.preference_action_result["status"], "failed")
        self.assertEqual(result.preference_action_result["action"], "like")
        self.assertEqual(result.tool_observations, [])

    def test_recommendation_enters_explicit_goal_and_plan_nodes(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = _coerce_state(state).model_copy(deep=True)
            current.intent = "recommendation"
            return current

        initial_state = parse(
            AgentState(
                user_id="u1",
                message="根据我的兴趣推荐几篇论文",
                context={"user_memory_summary": "对 RAG 和 agent 很感兴趣"},
            )
        )
        goal_state = graph_module.build_goal_node(initial_state)
        plan_state = graph_module.build_plan_node(goal_state)

        # 推荐请求也只验证当前主图的计划节点，避免测试继续依赖旧整轮兼容入口。
        self.assertEqual(goal_state.goal.goal_type, "recommendation")
        self.assertIsNotNone(plan_state.plan_runtime)
        tool_names = {step.tool_name for step in list(plan_state.execution_plan.steps or [])}
        self.assertTrue({"load_candidate_papers", "generate_recommendations", "validate_recommendations"}.issubset(tool_names))
        self.assertEqual(plan_state.debug["planner"]["selected_plan_source"], "tool_aware_rule_based")


if __name__ == "__main__":
    unittest.main()
