"""重构前的行为基准回归测试。

本文件不追求覆盖所有细节，只锁定 PlanExecutor 在拆分前最容易在重构中悄悄变坏的几类状态链路：

1. postcondition 失败必须判失败、不能被当成成功；
2. checkpoint 批准态在 plan_id 不匹配时不能误放行副作用工具（“不会误用旧 plan”）；
3. output_key 重复必须判失败，避免覆盖已有输出；
4. 副作用工具已有输出时复用、不重复执行；
5. 新增的只读 execution_path 摘要能如实反映 confirmation / resume / retry / replan / reuse / 最终状态。

这些场景与既有 test_plan_executor.py 互补：那里覆盖正常执行、确认门、缺参、replan 与 paper QA 质量闭环，
这里专门补上“状态判定边界”这一类最脆弱的链路，作为后续等价拆分的行为基准。
"""
from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
executor_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_executor")
planner_registry_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

AgentState = state_module.AgentState
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
    """构造单步“解析并索引论文”计划，副作用工具需要显式确认。"""
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
                input_bindings=[
                    StepInputBinding(
                        input_key="paper_reference",
                        source_type="literal",
                        value={"arxiv_id": "2401.00001", "title": "RAG Paper"},
                    )
                ],
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


# --------------------------------------------------------------------------------------
# 1. postcondition 失败：工具成功但结果不满足后置条件，必须判失败，不能误判成功。
# --------------------------------------------------------------------------------------
def test_plan_executor_postcondition_failure_marks_step_failed(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        # 工具本身执行成功并返回论文，但后置条件要求 outputs.arxiv_results.papers 为空，
        # 因此 postcondition 必须把这一步判为失败，验证执行器不会因为“工具 ok”就放行。
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    goal = Goal(goal_id="arxiv_search:postcondition", goal_type="arxiv_search", user_request="rag")
    plan = ExecutablePlan(
        plan_id="arxiv_search:postcondition",
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
                # 故意构造一个不可能满足的后置条件：要求结果不存在。
                postconditions=[
                    StepCondition(
                        condition_type="field_exists",
                        field_path="outputs.arxiv_results",
                        negate=True,
                    )
                ],
            ),
        ],
        entry_step_ids=["search_arxiv"],
        final_step_ids=["search_arxiv"],
    )

    result = PlanExecutor().execute(plan, AgentState(intent="arxiv_search", message="rag"))

    assert result.status == "failed"
    assert result.error == "postcondition_failed:search_arxiv"
    assert result.runtime is not None
    assert result.runtime.recovery_strategy["reason"] == "postcondition_failed"
    assert result.plan is not None
    assert result.plan.steps[0].status == "failed"
    assert any(
        trace.event == "step_failed" and trace.detail.get("failure_reason") == "postcondition_failed"
        for trace in result.trace
    )
    # 后置条件失败不应写入 output，避免下游误读“成功结果”。
    assert "arxiv_results" not in result.outputs


# --------------------------------------------------------------------------------------
# 2. checkpoint 恢复不会误用旧 plan：checkpoint 里的批准态属于另一个 plan_id 时必须忽略，
#    重新经过确认门，而不是直接放行副作用工具。
# --------------------------------------------------------------------------------------
def test_plan_executor_checkpoint_approval_ignored_when_plan_id_mismatch(monkeypatch) -> None:
    class StalePlanCheckpointStore:
        def get_agent_runtime_checkpoint(self, **kwargs):
            # checkpoint 记录的是另一轮（旧 plan）的批准态，plan_id 与当前 plan 不一致。
            return {
                "status": "running",
                "pending_confirmation": None,
                "runtime_state": {
                    "approved_step_ids": ["parse_and_index_paper"],
                    "plan": {"plan_id": "paper_qa:OLD-PLAN"},
                },
            }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("stale-plan checkpoint approval must not execute side effect tool")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)

    goal, plan = _paper_index_confirmation_plan()  # 当前 plan_id = "paper_qa:test"
    state = AgentState(user_id="u1", session_id="s1", intent="paper_qa", message="build index")

    result = PlanExecutor(runtime_checkpoint_store=StalePlanCheckpointStore()).execute(plan, state)

    # 旧 plan 的批准态被忽略，本轮重新进入确认门并暂停，而不是误放行副作用工具。
    assert result.status == "waiting_confirmation"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.step_id == "parse_and_index_paper"
    assert any(trace.event == "confirmation_created" for trace in result.trace)


# --------------------------------------------------------------------------------------
# 3. output_key 重复：runtime 中已存在同名输出时必须判失败，避免覆盖既有结果。
# --------------------------------------------------------------------------------------
def test_plan_executor_duplicate_output_key_marks_step_failed(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    goal = Goal(goal_id="arxiv_search:dup", goal_type="arxiv_search", user_request="rag")
    plan = ExecutablePlan(
        plan_id="arxiv_search:dup",
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
    # 预置一个同名 output，模拟 resume / 局部 replan 后重复落键的情况。
    runtime = PlanRuntime(
        goal=goal,
        plan=plan,
        outputs={"arxiv_results": {"papers": [{"arxiv_id": "PRESET"}]}},
        step_status={"search_arxiv": "pending"},
    )

    result = PlanExecutor().execute_runtime(runtime, AgentState(intent="arxiv_search", message="rag"))

    assert result.status == "failed"
    assert result.error == "duplicate_output_key:arxiv_results"
    assert result.runtime is not None
    assert result.runtime.recovery_strategy["reason"] == "duplicate_output_key"
    # 既有输出不能被覆盖。
    assert result.outputs["arxiv_results"]["papers"][0]["arxiv_id"] == "PRESET"
    assert any(
        trace.event == "step_failed" and trace.detail.get("failure_reason") == "duplicate_output_key"
        for trace in result.trace
    )


# --------------------------------------------------------------------------------------
# 4. execution_path 只读摘要：正常成功链路应如实反映“无确认/无 retry/无 replan/成功终态”。
# --------------------------------------------------------------------------------------
def test_execution_path_summary_reports_clean_success(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    state = AgentState(
        intent="arxiv_search",
        message="rag",
        search_spec=schemas.ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
    )
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    path = state.debug.get("execution_path")
    assert path is not None
    assert path["final_status"] == "success"
    assert path["needs_confirmation"] is False
    assert path["from_checkpoint_resume"] is False
    assert path["retry_occurred"] is False
    assert path["replan_occurred"] is False
    assert path["reused_side_effect_output"] is False
    assert path["pending_confirmation"] is False
    assert path["failure_reason"] is None
    assert any(item["tool_name"] for item in path["steps"])


# --------------------------------------------------------------------------------------
# 5. execution_path 只读摘要：确认暂停链路应反映 needs_confirmation 与 waiting_confirmation 终态。
# --------------------------------------------------------------------------------------
def test_execution_path_summary_reports_waiting_confirmation(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("side effect tool must not run before confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)

    _goal, plan = _paper_index_confirmation_plan()
    state = AgentState(intent="paper_qa", message="build index")
    result = PlanExecutor().execute(plan, state)

    assert result.status == "waiting_confirmation"
    path = state.debug.get("execution_path")
    assert path is not None
    assert path["final_status"] == "waiting_confirmation"
    assert path["needs_confirmation"] is True
    assert path["pending_confirmation"] is True


# --------------------------------------------------------------------------------------
# 6. execution_path 只读摘要：副作用工具复用已有输出时，reused_side_effect_output 必须为真。
# --------------------------------------------------------------------------------------
def test_execution_path_summary_reports_reused_side_effect(monkeypatch) -> None:
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
                    StepInputBinding(
                        input_key="paper_reference",
                        source_type="literal",
                        value={"arxiv_id": "2401.00001", "title": "RAG"},
                    ),
                    StepInputBinding(input_key="message", source_type="state", source_key="message"),
                ],
                side_effect_level="persistent_write",
                confirmation_policy=StepPolicy(
                    policy_type="confirmation",
                    mode="explicit_user_confirmation_required",
                    requires_confirmation=True,
                ),
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

    state = AgentState(intent="preference_action", message="喜欢这篇")
    result = PlanExecutor().execute_runtime(runtime, state)

    assert result.status == "success"
    path = state.debug.get("execution_path")
    assert path is not None
    assert path["reused_side_effect_output"] is True
    assert path["final_status"] == "success"


# --------------------------------------------------------------------------------------
# 7. execution_path 只读摘要：触发 replan 的链路应反映 replan_occurred，且最终成功记为 replanned_success。
# --------------------------------------------------------------------------------------
def test_execution_path_summary_reports_replan_then_success(monkeypatch) -> None:
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
        search_spec=schemas.ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
    )
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert calls["search"] == 2
    path = state.debug.get("execution_path")
    assert path is not None
    assert path["replan_occurred"] is True
    assert path["final_status"] == "replanned_success"
    assert path["replan_counts"]


# --------------------------------------------------------------------------------------
# 8. execution_path 只读摘要：replan 次数耗尽进入 fallback 时，final_status 收敛为 failed。
# --------------------------------------------------------------------------------------
def test_execution_path_summary_reports_fallback_as_failed() -> None:
    # search_arxiv 缺少 search_spec，输入校验失败 -> 进入 replan -> 最终 fallback。
    goal = Goal(goal_id="arxiv_search:fallback", goal_type="arxiv_search", user_request="search")
    plan = ExecutablePlan(
        plan_id="arxiv_search:fallback",
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

    state = AgentState(intent="arxiv_search", message="rag")
    result = PlanExecutor().execute(plan, state)

    assert result.status == "fallback"
    path = state.debug.get("execution_path")
    assert path is not None
    assert path["final_status"] == "failed"
    assert path["failure_reason"]


# --------------------------------------------------------------------------------------
# 9. checkpoint goal 类型不一致：plan_id 偶然相同但 goal 类型不同时，旧批准态绝不能放行
#    副作用工具，必须重新经过确认门。
# --------------------------------------------------------------------------------------
def test_plan_executor_checkpoint_approval_ignored_when_goal_type_mismatch(monkeypatch) -> None:
    class StaleGoalCheckpointStore:
        def get_agent_runtime_checkpoint(self, **kwargs):
            # checkpoint 的 plan_id 与当前 plan 相同，但 goal 类型来自另一种意图（复用 session 切换意图）。
            return {
                "status": "running",
                "pending_confirmation": None,
                "runtime_state": {
                    "approved_step_ids": ["parse_and_index_paper"],
                    "plan": {"plan_id": "paper_qa:test"},
                    "goal": {"goal_type": "preference_action"},
                },
            }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("stale-goal checkpoint approval must not execute side effect tool")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)

    goal, plan = _paper_index_confirmation_plan()  # 当前 goal_type = "paper_qa"
    state = AgentState(user_id="u1", session_id="s1", intent="paper_qa", message="build index")

    result = PlanExecutor(runtime_checkpoint_store=StaleGoalCheckpointStore()).execute(plan, state)

    # goal 类型不一致 -> 旧批准态被忽略 -> 重新进入确认门并暂停，而不是误放行副作用工具。
    assert result.status == "waiting_confirmation"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.step_id == "parse_and_index_paper"
    assert any(trace.event == "confirmation_created" for trace in result.trace)


# --------------------------------------------------------------------------------------
# 10. checkpoint goal 类型一致 + plan_id 一致：批准态正常放行（goal_type 校验不能误伤合法 resume）。
# --------------------------------------------------------------------------------------
def test_plan_executor_checkpoint_approval_honored_when_goal_type_matches(monkeypatch) -> None:
    calls = []

    class MatchingGoalCheckpointStore:
        def get_agent_runtime_checkpoint(self, **kwargs):
            return {
                "status": "running",
                "pending_confirmation": None,
                "runtime_state": {
                    "approved_step_ids": ["parse_and_index_paper"],
                    "plan": {"plan_id": "paper_qa:test"},
                    "goal": {"goal_type": "paper_qa"},
                },
            }

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    def fail_if_interrupted(*args, **kwargs):
        raise AssertionError("matching-goal checkpoint approval must not request confirmation again")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", fail_if_interrupted)

    goal, plan = _paper_index_confirmation_plan()
    state = AgentState(user_id="u1", session_id="s1", intent="paper_qa", message="build index")
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}

    result = PlanExecutor(runtime_checkpoint_store=MatchingGoalCheckpointStore())._execute_runtime(
        runtime,
        state,
        allow_interrupt=True,
    )

    # goal 类型一致 -> 批准态正常放行 -> 副作用工具执行，不再二次确认。
    assert result.status == "success"
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert not any(trace.event == "confirmation_requested" for trace in result.trace)


# --------------------------------------------------------------------------------------
# 11. resume 后状态丢失 user_id：批准态已经写入真实用户的 checkpoint 时，执行器应能按
#     同一 session/thread 安全找回批准态，而不是再次进入确认门。
# --------------------------------------------------------------------------------------
def test_plan_executor_checkpoint_approval_honored_when_resume_state_loses_user_id(monkeypatch) -> None:
    calls = []

    class ResumeCheckpointStore:
        def get_agent_runtime_checkpoint(self, **kwargs):
            # 复现线上日志里的断点：LangGraph 恢复态没有带回真实 user_id，
            # 精确 user 查询会落到默认用户，因此读不到刚刚消费过的批准态。
            assert kwargs.get("user_id") == executor_module.DEFAULT_USER_ID
            return None

        def list_agent_runtime_checkpoints_by_thread(self, **kwargs):
            assert kwargs.get("session_id") == "s1"
            assert kwargs.get("thread_id") == "s1"
            return [
                {
                    "user_id": "local_user",
                    "status": "running",
                    "pending_confirmation": None,
                    "runtime_state": {
                        "approved_step_ids": ["parse_and_index_paper"],
                        "plan": {"plan_id": "paper_qa:test"},
                        "goal": {"goal_type": "paper_qa"},
                    },
                },
            ]

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    def fail_if_interrupted(*args, **kwargs):
        raise AssertionError("consumed checkpoint approval must not request confirmation again")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", fail_if_interrupted)

    goal, plan = _paper_index_confirmation_plan()
    state = AgentState(user_id="", session_id="s1", intent="paper_qa", message="build index")
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}

    result = PlanExecutor(runtime_checkpoint_store=ResumeCheckpointStore())._execute_runtime(
        runtime,
        state,
        allow_interrupt=True,
    )

    assert result.status == "success"
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert not any(trace.event == "confirmation_requested" for trace in result.trace)


# --------------------------------------------------------------------------------------
# 12. resume 后存在多个同 thread checkpoint：只有通过批准态、plan、goal 校验的唯一候选
#     可以放行，避免默认用户旧记录遮住真实用户的已消费批准态。
# --------------------------------------------------------------------------------------
def test_plan_executor_checkpoint_approval_recovers_single_valid_thread_candidate(monkeypatch) -> None:
    calls = []

    class MultiCandidateCheckpointStore:
        def get_agent_runtime_checkpoint(self, **kwargs):
            assert kwargs.get("user_id") == executor_module.DEFAULT_USER_ID
            return {
                "user_id": executor_module.DEFAULT_USER_ID,
                "status": "waiting_confirmation",
                "pending_confirmation": {"step_id": "parse_and_index_paper"},
                "runtime_state": {
                    "approved_step_ids": [],
                    "plan": {"plan_id": "paper_qa:test"},
                    "goal": {"goal_type": "paper_qa"},
                },
            }

        def list_agent_runtime_checkpoints_by_thread(self, **kwargs):
            assert kwargs.get("session_id") == "s1"
            assert kwargs.get("thread_id") == "s1"
            # 默认用户旧记录仍处于 waiting，不应被放行；真实用户记录已经消费确认并写入批准态。
            return [
                self.get_agent_runtime_checkpoint(user_id=executor_module.DEFAULT_USER_ID, session_id="s1", thread_id="s1"),
                {
                    "user_id": "local_user",
                    "status": "running",
                    "pending_confirmation": None,
                    "runtime_state": {
                        "approved_step_ids": ["parse_and_index_paper"],
                        "plan": {"plan_id": "paper_qa:test"},
                        "goal": {"goal_type": "paper_qa"},
                    },
                },
            ]

    def fake_invoke_tool(tool_name: str, **kwargs):
        calls.append((tool_name, dict(kwargs)))
        if tool_name == "build_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    def fail_if_interrupted(*args, **kwargs):
        raise AssertionError("single valid thread fallback candidate must not request confirmation again")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", fail_if_interrupted)

    goal, plan = _paper_index_confirmation_plan()
    state = AgentState(user_id="", session_id="s1", intent="paper_qa", message="build index")
    runtime = planner_module.build_plan_runtime(state, goal=goal, plan=plan, turn_status="success")
    runtime.step_status = {step.step_id: "pending" for step in list(plan.steps or [])}

    result = PlanExecutor(runtime_checkpoint_store=MultiCandidateCheckpointStore())._execute_runtime(
        runtime,
        state,
        allow_interrupt=True,
    )

    assert result.status == "success"
    assert calls and calls[0][0] == "build_paper_qa_index"
    assert not any(trace.event == "confirmation_requested" for trace in result.trace)


# --------------------------------------------------------------------------------------
# 13. replan 后不污染原始成功输出：低质量 search 触发 replan 修复，最终成功；
#     原始已成功步骤的输出在 replan 后仍然可用，未被覆盖或清空。
# --------------------------------------------------------------------------------------
def test_plan_executor_replan_preserves_prior_success_outputs(monkeypatch) -> None:
    calls = {"search": 0}

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        calls["search"] += 1
        # 第一次空结果触发 replan，第二次返回有效结果。
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
        search_spec=schemas.ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
    )
    _, plan, _ = planner_module.build_executable_plan(state)
    result = PlanExecutor().execute(plan, state)

    assert result.status == "success"
    assert calls["search"] == 2
    # replan 发生过：normalize_request / search_spec 这些 replan 前已成功的上游输出未被污染。
    assert "normalized_request" in result.outputs
    assert "search_spec" in result.outputs
    assert result.outputs["search_spec"]["query"] == "rag"
    # 最终答案来自 replan 后的有效结果，原始空结果没有被当成最终输出。
    assert result.final_answer
    assert any(trace.event == "plan_replanned" for trace in result.trace)


# --------------------------------------------------------------------------------------
# 14. 终态可区分性：成功 / 等待确认 / 用户取消（拒绝）三类终态在 turn_status 和
#     execution_path.final_status 上必须可区分，不能都坍缩成同一种状态。
# --------------------------------------------------------------------------------------
def test_terminal_states_are_distinguishable_cancel_vs_waiting_vs_success(monkeypatch) -> None:
    # (a) 等待确认：副作用工具执行前暂停。
    def fail_if_called(*args, **kwargs):
        raise AssertionError("tool must not run before confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)
    _goal, plan = _paper_index_confirmation_plan()
    waiting = PlanExecutor().execute(plan, AgentState(intent="paper_qa", message="build index"))
    assert waiting.status == "waiting_confirmation"
    assert waiting.runtime is not None
    assert waiting.runtime.turn_status == "waiting_confirmation"

    # (b) 用户取消（拒绝确认）：进入安全终止，原副作用步骤被 skip，不执行工具。
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"decision": "reject", "note": "cancel"})
    _goal2, plan2 = _paper_index_confirmation_plan()
    state2 = AgentState(intent="paper_qa", message="build index")
    runtime2 = planner_module.build_plan_runtime(state2, goal=_goal2, plan=plan2, turn_status="success")
    runtime2.step_status = {step.step_id: "pending" for step in list(plan2.steps or [])}
    cancelled = PlanExecutor()._execute_runtime(runtime2, state2, allow_interrupt=True)
    # 拒绝后中间 waiting_confirmation 已清空，本轮收口成可展示终态，但副作用步骤被 skip。
    assert any(trace.event == "confirmation_rejected" for trace in cancelled.trace)
    assert state2.pending_action["status"] == "rejected"
    assert cancelled.plan.steps[0].status == "skipped"
    assert runtime2.pending_confirmation is None

    # (c) 正常成功：与上述两类终态明确不同。
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    state3 = AgentState(
        intent="arxiv_search",
        message="rag",
        search_spec=schemas.ArxivSearchSpec(intent="arxiv_search", query="rag", max_results=5),
    )
    _, plan3, _ = planner_module.build_executable_plan(state3)
    success = PlanExecutor().execute(plan3, state3)
    assert success.status == "success"
    assert success.runtime.turn_status == "success"
    assert state3.debug["execution_path"]["final_status"] in {"success", "replanned_success"}

    # 三类终态两两不同。
    assert waiting.status != cancelled.status
    assert success.status != waiting.status
