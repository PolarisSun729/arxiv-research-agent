from __future__ import annotations

import sys
import unittest
from unittest import mock

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
AgentState = _MODULES["state_module"].AgentState
ToolObservation = _MODULES["schemas"].ToolObservation
ToolCallRequest = _MODULES["schemas"].ToolCallRequest
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
graph_module = _MODULES["graph_module"]
paper_reading_module = sys.modules["backend.agents.arxiv_search_agent.node.paper_reading_node"]
pending_action_module = sys.modules["backend.agents.arxiv_search_agent.node.pending_action_node"]
preference_module = sys.modules["backend.agents.arxiv_search_agent.node.preference_node"]


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

    def test_pending_confirmation_uses_build_then_answer_tool_protocol(self) -> None:
        state = AgentState(
            intent="paper_qa",
            message="解析",
            pending_action={
                "type": "parse_then_qa",
                "status": "waiting_confirmation",
                "arxiv_id": "2401.00001",
                "title": "RAG Paper",
                "original_question": "这篇论文讲了什么？",
                "qa_question": "请总结这篇论文",
                "loading_method": "docling",
            },
        )

        def fake_execute_tool(current_state):
            current = _coerce_agent_state(current_state).model_copy(deep=True)
            request = current.tool_call_request
            self.assertIsInstance(request, ToolCallRequest)
            if request.tool_name == "build_paper_qa_index":
                current.tool_result = {
                    "ok": True,
                    "tool_name": request.tool_name,
                    "summary": "built index",
                    "data": {"status": "indexed", "has_index": True},
                    "trace": {"tool_name": request.tool_name},
                    "error": None,
                }
                return _append_mock_observation(current, request.tool_name, True, "built index")
            self.assertEqual(request.tool_name, "answer_paper_question")
            current.tool_result = {
                "ok": True,
                "tool_name": request.tool_name,
                "summary": "answered question",
                "data": {"answer": "确认后已回答", "sources": []},
                "trace": {"tool_name": request.tool_name},
                "error": None,
            }
            return _append_mock_observation(current, request.tool_name, True, "answered question")

        with mock.patch.object(pending_action_module, "execute_tool", side_effect=fake_execute_tool) as patched:
            result = pending_action_module.handle_pending_action_confirmation(state)

        self.assertEqual(patched.call_count, 2)
        self.assertEqual(result.paper_qa_result["status"], "success")
        self.assertEqual(result.paper_qa_result["answer"], "确认后已回答")
        self.assertIsNone(result.pending_action)
        self.assertEqual([obs.tool_name for obs in result.tool_observations], ["build_paper_qa_index", "answer_paper_question"])

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

    def test_recommendation_branch_routes_through_tool_nodes(self) -> None:
        patches: list[mock._patch] = []

        def patch_graph(name: str, value) -> None:
            patcher = mock.patch.object(graph_module, name, value)
            patches.append(patcher)
            patcher.start()

        def parse(state, generation_service=None):
            del generation_service
            return _visit(state, "parse_search_request", intent="recommendation")

        def plan(state):
            return _visit(state, "plan_task")

        def build_args(state):
            return _visit(state, "build_recommendation_tool_args")

        def invoke(state):
            return _visit(state, "invoke_recommendation_tool", tool_result={"ok": True})

        def adapt(state):
            return _visit(state, "adapt_recommendation_tool_result", papers=[{"arxiv_id": "2401.00001", "title": "RAG Paper"}])

        def synthesize(state):
            return _visit(state, "synthesize_response", answer="recommended papers")

        try:
            for name, fn in {
                "parse_search_request": parse,
                "plan_task": plan,
                "build_recommendation_tool_args": build_args,
                "invoke_recommendation_tool": invoke,
                "adapt_recommendation_tool_result": adapt,
                "synthesize_response": synthesize,
            }.items():
                patch_graph(name, fn)

            graph = build_arxiv_search_graph()
            result = graph.invoke(AgentState(message="给我推荐一些论文").model_dump())
        finally:
            for patcher in reversed(patches):
                patcher.stop()

        self.assertEqual(result["answer"], "recommended papers")
        self.assertEqual(
            result["debug"]["visited"],
            [
                "parse_search_request",
                "plan_task",
                "build_recommendation_tool_args",
                "invoke_recommendation_tool",
                "adapt_recommendation_tool_result",
                "synthesize_response",
            ],
        )


if __name__ == "__main__":
    unittest.main()
