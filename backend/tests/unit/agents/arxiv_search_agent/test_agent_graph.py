from __future__ import annotations

import unittest
from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
AgentState = _MODULES["state_module"].AgentState
schemas = _MODULES["schemas"]
ConfirmationDecisionOption = schemas.ConfirmationDecisionOption
ConfirmationRequest = schemas.ConfirmationRequest
ExecutionTrace = schemas.ExecutionTrace
ExecutablePlan = schemas.ExecutablePlan
Goal = schemas.Goal
PlanRuntime = schemas.PlanRuntime
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

    def test_graph_exposes_explicit_agent_execution_loop(self) -> None:
        graph = build_arxiv_search_graph()

        self.assertIn("parse_search_request", graph._nodes)
        self.assertIn("build_goal", graph._nodes)
        self.assertIn("build_plan", graph._nodes)
        self.assertIn("select_next_step", graph._nodes)
        self.assertIn("execute_step", graph._nodes)
        self.assertIn("observe_step", graph._nodes)
        self.assertIn("route_after_observation", graph._nodes)
        self.assertIn("replan", graph._nodes)
        self.assertIn("finalize", graph._nodes)
        self.assertNotIn("run_agent_turn", graph._nodes)
        self.assertEqual(graph._conditional_edges["route_after_observation"][1]["replan"], "replan")

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
        pending_action = graph_module._build_pending_action_mirror(
            schemas.AgentTurnResult(
                status="waiting_confirmation",
                pending_confirmation=confirmation,
                outputs={},
                trace=[],
            )
        )

        self.assertIsNotNone(pending_action)
        self.assertEqual(pending_action["type"], "tool_approval")
        self.assertEqual(pending_action["status"], "waiting_confirmation")
        self.assertEqual(pending_action["allowed_decisions"], ["approve", "reject"])

    def test_running_plan_status_is_not_written_to_agent_step_status(self) -> None:
        state = AgentState(message="帮我找最近 7 天关于 RAG 的 5 篇论文")
        result = schemas.StepExecutionResult(
            step_id="search_arxiv",
            step_status="running",
            output_key="ranked_papers",
            next_action="continue",
        )

        graph_module._apply_step_result(state, result)

        # AgentStep 是前端/流式事件的节点展示摘要，只允许终态；
        # PlanRuntime 的 running 中间态保留在 outputs/debug 中，避免打断后续 observe/replan。
        self.assertEqual(state.steps[-1].status, "success")
        self.assertEqual(state.steps[-1].outputs["plan_step_status"], "running")

    def test_apply_turn_result_uses_resolved_paper_for_paper_qa_metadata(self) -> None:
        state = AgentState(
            intent="paper_qa",
            message="第二篇论文的方法是什么？",
            context={
                "selected_paper": {"arxiv_id": "2401.00001", "title": "First Paper"},
                "last_papers": [
                    {"arxiv_id": "2401.00001", "title": "First Paper"},
                    {"arxiv_id": "2401.00002", "title": "Second Paper"},
                ],
            },
        )
        runtime = PlanRuntime(
            goal=Goal(goal_type="paper_qa"),
            plan=ExecutablePlan(plan_id="paper_qa:test", goal=Goal(goal_type="paper_qa")),
            outputs={
                "paper_ref": {
                    "status": "success",
                    "arxiv_id": "2401.00002",
                    "title": "Second Paper",
                    "paper": {"arxiv_id": "2401.00002", "title": "Second Paper"},
                },
                "paper_qa_result": {"answer": "grounded answer", "sources": [], "retrieval_debug": {}},
            },
            trace=[ExecutionTrace(step_id="answer_paper_question", event="step_succeeded", status="success")],
        )
        result = schemas.AgentTurnResult(
            status="success",
            final_answer="grounded answer",
            outputs=dict(runtime.outputs),
            trace=list(runtime.trace),
            runtime=runtime,
        )

        graph_module._apply_turn_result(state, result)

        self.assertEqual(state.paper_qa_result["arxiv_id"], "2401.00002")
        self.assertEqual(state.paper_qa_result["title"], "Second Paper")

    def test_apply_turn_result_projects_arxiv_results_to_visible_papers(self) -> None:
        state = AgentState(intent="arxiv_search", message="帮我找最近 7 天关于 RAG 的 5 篇论文")
        result = schemas.AgentTurnResult(
            status="success",
            final_answer="已检索到 1 篇相关论文",
            outputs={
                "arxiv_results": {
                    "papers": [
                        {"arxiv_id": "2606.00001", "title": "RAG Agents in Practice"},
                    ],
                    "tool_result": {"ok": True},
                }
            },
            trace=[],
        )

        graph_module._apply_turn_result(state, result)

        # LLM 计划可能跳过个性化排序步骤；前端论文卡片仍应从 arxiv_results.papers 获得数据。
        self.assertEqual(len(state.papers), 1)
        self.assertEqual(state.papers[0]["arxiv_id"], "2606.00001")

    def test_apply_turn_result_projects_papers_for_llm_named_output_key(self) -> None:
        # 复现间歇性 bug：experimental LLM planner 把 search_arxiv 的 output_key
        # 命名成非 arxiv_results/ranked_papers 的任意名字（这里用 my_search_hits）。
        # 回答可以通过 input_bindings 正常读到论文数，但旧投影只认固定字面量 key，
        # 导致 state.papers 为空 -> 前端“有回答无卡片”。
        PlanStep = schemas.PlanStep
        ToolSpec = schemas.ToolSpec
        plan = ExecutablePlan(
            plan_id="arxiv_search:llm_draft",
            goal=Goal(goal_type="arxiv_search"),
            steps=[
                PlanStep(
                    step_id="search_arxiv",
                    action_type="search",
                    tool_name="search_arxiv",
                    tool=ToolSpec(tool_name="search_arxiv"),
                    output_key="my_search_hits",
                ),
                PlanStep(
                    step_id="synthesize_arxiv_response",
                    action_type="answer",
                    tool_name="synthesize_arxiv_response",
                    tool=ToolSpec(tool_name="synthesize_arxiv_response"),
                    output_key="final_answer",
                ),
            ],
        )
        state = AgentState(intent="arxiv_search", message="帮我找最近 7 天关于 RAG 的 5 篇论文")
        result = schemas.AgentTurnResult(
            status="success",
            final_answer="已检索到 2 篇相关 arXiv 论文",
            plan=plan,
            outputs={
                "my_search_hits": {
                    "papers": [
                        {"arxiv_id": "2606.00001", "title": "RAG Agents in Practice"},
                        {"arxiv_id": "2606.00002", "title": "Agentic Retrieval"},
                    ],
                    "tool_result": {"ok": True},
                },
                "final_answer": "已检索到 2 篇相关 arXiv 论文",
            },
            trace=[],
        )

        graph_module._apply_turn_result(state, result)

        # papers 必须与回答口径一致地稳定回填，无论 LLM 把 output_key 取成什么名字。
        self.assertEqual(len(state.papers), 2)
        self.assertEqual(
            [paper["arxiv_id"] for paper in state.papers],
            ["2606.00001", "2606.00002"],
        )

    def test_apply_turn_result_prefers_ranked_papers_over_raw_search_output(self) -> None:
        # 同时存在原始检索与个性化重排时，papers 应取个性化结果，避免展示未重排的旧顺序。
        PlanStep = schemas.PlanStep
        ToolSpec = schemas.ToolSpec
        plan = ExecutablePlan(
            plan_id="arxiv_search:llm_draft",
            goal=Goal(goal_type="arxiv_search"),
            steps=[
                PlanStep(
                    step_id="search_arxiv",
                    action_type="search",
                    tool_name="search_arxiv",
                    tool=ToolSpec(tool_name="search_arxiv"),
                    output_key="raw_hits",
                ),
                PlanStep(
                    step_id="personalize_paper_results",
                    action_type="rerank",
                    tool_name="personalize_paper_results",
                    tool=ToolSpec(tool_name="personalize_paper_results"),
                    output_key="reranked",
                ),
            ],
        )
        state = AgentState(intent="arxiv_search", message="找 RAG 论文")
        result = schemas.AgentTurnResult(
            status="success",
            final_answer="已检索到 2 篇相关 arXiv 论文",
            plan=plan,
            outputs={
                "raw_hits": {"papers": [{"arxiv_id": "raw-1"}, {"arxiv_id": "raw-2"}]},
                "reranked": {"ranked_papers": [{"arxiv_id": "ranked-1"}, {"arxiv_id": "ranked-2"}]},
            },
            trace=[],
        )

        graph_module._apply_turn_result(state, result)

        self.assertEqual([paper["arxiv_id"] for paper in state.papers], ["ranked-1", "ranked-2"])

    def test_apply_turn_result_keeps_consumed_confirmation_result_but_clears_pending_snapshot(self) -> None:
        state = AgentState(
            intent="paper_qa",
            message="approve",
            pending_action={
                "status": "approved",
                "decision": "approve",
                "step_id": "parse_and_index_paper",
                "tool_name": "parse_and_index_paper",
                "confirmation_consumed": True,
            },
            debug={
                "pending_confirmation": {
                    "step_id": "parse_and_index_paper",
                    "tool_name": "parse_and_index_paper",
                },
                "confirmation_consumed": {
                    "decision": "approve",
                    "step_id": "parse_and_index_paper",
                },
            },
            paper_qa_result={
                "status": "waiting_confirmation",
                "pending_confirmation": {
                    "step_id": "parse_and_index_paper",
                    "tool_name": "parse_and_index_paper",
                },
            },
        )
        runtime = PlanRuntime(
            goal=Goal(goal_type="paper_qa"),
            plan=ExecutablePlan(plan_id="paper_qa:test", goal=Goal(goal_type="paper_qa")),
            trace=[ExecutionTrace(step_id="parse_and_index_paper", event="confirmation_consumed", status="pending")],
        )
        result = schemas.AgentTurnResult(
            status="success",
            final_answer="resume finished",
            outputs={},
            trace=list(runtime.trace),
            runtime=runtime,
        )

        graph_module._apply_turn_result(state, result)

        self.assertIsNotNone(state.pending_action)
        self.assertEqual(state.pending_action["status"], "approved")
        self.assertTrue(state.pending_action["confirmation_consumed"])
        self.assertNotIn("pending_confirmation", state.debug)
        self.assertEqual(state.debug["confirmation_consumed"]["decision"], "approve")
        self.assertEqual(state.paper_qa_result["status"], "ready")
        self.assertIsNone(state.paper_qa_result["pending_confirmation"])
        self.assertTrue(state.paper_qa_result["confirmation_consumed"])

    def test_apply_turn_result_marks_rejected_confirmation_snapshot_as_cancelled(self) -> None:
        state = AgentState(
            intent="paper_qa",
            message="reject",
            pending_action={
                "status": "rejected",
                "decision": "reject",
                "step_id": "parse_and_index_paper",
                "tool_name": "parse_and_index_paper",
                "confirmation_consumed": True,
            },
            paper_qa_result={
                "status": "waiting_confirmation",
                "pending_confirmation": {
                    "step_id": "parse_and_index_paper",
                    "tool_name": "parse_and_index_paper",
                },
            },
            debug={
                "pending_confirmation": {
                    "step_id": "parse_and_index_paper",
                    "tool_name": "parse_and_index_paper",
                },
            },
        )
        runtime = PlanRuntime(
            goal=Goal(goal_type="paper_qa"),
            plan=ExecutablePlan(plan_id="paper_qa:test", goal=Goal(goal_type="paper_qa")),
            trace=[ExecutionTrace(step_id="parse_and_index_paper", event="confirmation_consumed", status="skipped")],
        )
        result = schemas.AgentTurnResult(
            status="success",
            final_answer="cancelled",
            outputs={},
            trace=list(runtime.trace),
            runtime=runtime,
        )

        graph_module._apply_turn_result(state, result)

        self.assertEqual(state.paper_qa_result["status"], "cancelled")
        self.assertIsNone(state.paper_qa_result["pending_confirmation"])
        self.assertTrue(state.paper_qa_result["confirmation_consumed"])
        self.assertEqual(state.paper_qa_result["confirmation_decision"], "reject")
        self.assertNotIn("pending_confirmation", state.debug)


if __name__ == "__main__":
    unittest.main()
