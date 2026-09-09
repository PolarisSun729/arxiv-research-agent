from __future__ import annotations

import sys
import unittest
from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
AgentState = _MODULES["state_module"].AgentState
schemas = _MODULES["schemas"]
ExecutionTrace = schemas.ExecutionTrace
ExecutablePlan = schemas.ExecutablePlan
Goal = schemas.Goal
PlanRuntime = schemas.PlanRuntime
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
DEFAULT_GRAPH_CHECKPOINTER = _MODULES["graph_module"].DEFAULT_GRAPH_CHECKPOINTER
graph_module = _MODULES["graph_module"]
assemble_final_answer = sys.modules["backend.agents.arxiv_search_agent.response_assembler"].assemble_final_answer


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
                    "final_target_resolved": True,
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

        self.assertEqual(
            state.resolved_paper,
            {"arxiv_id": "2401.00002", "title": "Second Paper"},
        )
        self.assertEqual(state.paper_qa_result["arxiv_id"], "2401.00002")
        self.assertEqual(state.paper_qa_result["title"], "Second Paper")

    def test_preference_resume_projects_current_result_instead_of_stale_paper_qa(self) -> None:
        """确认恢复后的偏好结果不会被上一轮 QA 的响应镜像覆盖。"""
        PlanStep = schemas.PlanStep
        ToolSpec = schemas.ToolSpec
        goal = Goal(goal_type="preference_action")
        plan = ExecutablePlan(
            plan_id="preference_action:resume",
            goal=goal,
            steps=[
                PlanStep(
                    step_id="resolve_target",
                    action_type="retrieve",
                    tool_name="resolve_preference_target",
                    tool=ToolSpec(tool_name="resolve_preference_target"),
                    output_key="resolved_target_info",
                ),
                PlanStep(
                    step_id="update_store",
                    action_type="write_state",
                    tool_name="update_preference_store",
                    tool=ToolSpec(tool_name="update_preference_store"),
                    output_key="preference_update_result",
                ),
            ],
        )
        runtime = PlanRuntime(
            goal=goal,
            plan=plan,
            outputs={
                "resolved_target_info": {
                    "status": "resolved",
                    "final_target_resolved": True,
                    "arxiv_id": "2607.28580",
                    "title": "DualG-MRAG",
                    "paper": {"arxiv_id": "2607.28580", "title": "DualG-MRAG"},
                },
                "preference_update_result": {
                    "status": "success",
                    "action": "like",
                    "label": "liked",
                    "arxiv_id": "2607.28580",
                    "liked": True,
                },
                "final_user_response": {"final_answer": "已更新论文偏好：2607.28580。"},
            },
        )
        state = AgentState(
            intent="preference_action",
            answer="上一轮 QA 的旧答案",
            paper_qa_result={"status": "success", "answer": "上一轮 QA 的旧答案"},
        )
        result = schemas.AgentTurnResult(
            status="success",
            final_answer=assemble_final_answer(runtime),
            plan=plan,
            outputs=dict(runtime.outputs),
            runtime=runtime,
        )

        graph_module._apply_turn_result(state, result)

        self.assertEqual(state.answer, "已更新论文偏好：2607.28580。")
        self.assertEqual(state.preference_action_result["action"], "like")
        self.assertEqual(state.resolved_paper, {"arxiv_id": "2607.28580", "title": "DualG-MRAG"})
        self.assertIsNone(state.paper_qa_result)
        response = _MODULES["service_module"]._state_to_response(state)
        self.assertEqual(response.answer, "已更新论文偏好：2607.28580。")
        self.assertEqual(response.preference_action_result["arxiv_id"], "2607.28580")
        self.assertIsNone(response.paper_qa_result)

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


if __name__ == "__main__":
    unittest.main()
