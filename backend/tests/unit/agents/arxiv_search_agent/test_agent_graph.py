from __future__ import annotations

import unittest
from unittest import mock

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
AgentState = _MODULES["state_module"].AgentState
schemas = _MODULES["schemas"]
AgentTurnResult = schemas.AgentTurnResult
ConfirmationDecisionOption = schemas.ConfirmationDecisionOption
ConfirmationRequest = schemas.ConfirmationRequest
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
DEFAULT_GRAPH_CHECKPOINTER = _MODULES["graph_module"].DEFAULT_GRAPH_CHECKPOINTER
graph_module = _MODULES["graph_module"]


class AgentGraphFlowTests(unittest.TestCase):
    def test_build_arxiv_search_graph_uses_default_checkpointer(self) -> None:
        graph = build_arxiv_search_graph()

        self.assertIs(getattr(graph, "_checkpointer", None), DEFAULT_GRAPH_CHECKPOINTER)

    def test_build_arxiv_search_graph_accepts_explicit_checkpointer(self) -> None:
        explicit_checkpointer = object()

        graph = build_arxiv_search_graph(checkpointer=explicit_checkpointer)

        self.assertIs(getattr(graph, "_checkpointer", None), explicit_checkpointer)

    def test_graph_runs_parse_then_unified_turn_runtime(self) -> None:
        def parse(state, generation_service=None):
            del generation_service
            current = AgentState.model_validate(state).model_copy(deep=True)
            current.intent = "arxiv_search"
            return current

        def fake_run_agent_turn(state):
            self.assertEqual(state.intent, "arxiv_search")
            return AgentTurnResult(
                status="success",
                final_answer="found papers",
                outputs={"ranked_papers": [{"arxiv_id": "2401.00001", "title": "RAG paper"}]},
                trace=[],
            )

        with mock.patch.object(graph_module, "parse_search_request", side_effect=parse), mock.patch.object(
            graph_module,
            "run_agent_turn_in_graph",
            side_effect=fake_run_agent_turn,
        ):
            graph = build_arxiv_search_graph()
            result = AgentState.model_validate(graph.invoke(AgentState(message="search rag").model_dump()))

        self.assertEqual(result.answer, "found papers")
        self.assertEqual(result.steps[-1].step, "run_agent_turn")
        self.assertEqual(result.papers[0]["arxiv_id"], "2401.00001")

    def test_confirmation_request_maps_to_tool_approval_pending_action(self) -> None:
        confirmation = ConfirmationRequest(
            step_id="parse_and_index_paper",
            tool_name="parse_and_index_paper",
            action_type="index",
            side_effect_level="external_call",
            reason="paper_index_missing",
            title="确认是否解析论文",
            description="需要先解析 PDF 并创建全文索引。",
            arguments_summary={"paper_reference": {"arxiv_id": "2401.00001", "title": "RAG paper"}},
            original_question="这篇论文的方法是什么？",
            target_paper={"arxiv_id": "2401.00001", "title": "RAG paper"},
            allowed_decisions=[
                ConfirmationDecisionOption(code="approve", label="批准", description="继续执行当前工具操作"),
                ConfirmationDecisionOption(code="reject", label="拒绝", description="取消当前工具操作"),
            ],
            session_id="s1",
            thread_id="s1",
            plan_id="plan-1",
            trace_id="goal-1",
        )
        result = AgentTurnResult(
            status="waiting_confirmation",
            pending_confirmation=confirmation,
            outputs={},
            trace=[],
        )

        state = graph_module._state_from_turn_result(AgentState(session_id="s1", intent="paper_qa"), result)

        self.assertIsNotNone(state.pending_action)
        self.assertEqual(state.pending_action["type"], "tool_approval")
        self.assertEqual(state.pending_action["status"], "waiting_confirmation")
        self.assertEqual(state.pending_action["allowed_decisions"], ["approve", "reject"])
        self.assertEqual(state.paper_qa_result["status"], "waiting_confirmation")


if __name__ == "__main__":
    unittest.main()
