from agents.arxiv_search_agent import plan_executor as executor_module
from agents.arxiv_search_agent.execution.approvals import ApprovalGrant
from agents.arxiv_search_agent.execution.background_jobs import BackgroundWorkTicket
from agents.arxiv_search_agent.plan_executor import PlanExecutor
from agents.arxiv_search_agent.schemas import (
    ExecutablePlan,
    Goal,
    PlanStep,
    StepInputBinding,
    StepPolicy,
    ToolSpec,
)
from agents.arxiv_search_agent.state import AgentState
from tests.helpers.sqlite import build_storage_container


def test_background_side_effect_uses_approved_grant_without_inline_tool_fallback(tmp_path, monkeypatch) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "executor.sqlite"))
    calls = []

    def fake_invoke(tool_name: str, **kwargs):
        calls.append((tool_name, kwargs))
        if tool_name == "check_paper_qa_index":
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "missing",
                "data": {"status": "missing", "has_index": False},
                "trace": {},
                "error": None,
            }
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "indexed",
            "data": {"status": "indexed", "has_index": True},
            "trace": {},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke)
    goal = Goal(goal_id="g1", goal_type="paper_qa", user_request="index")
    step = PlanStep(
        step_id="index",
        action_type="index",
        tool_name="parse_and_index_paper",
        tool=ToolSpec(tool_name="parse_and_index_paper", side_effect_level="external_call"),
        input_bindings=[
            StepInputBinding(
                input_key="paper_reference",
                source_type="literal",
                value={"arxiv_id": "2401.00001"},
            )
        ],
        output_key="index_result",
        confirmation_policy=StepPolicy(
            policy_type="confirmation",
            mode="explicit_user_confirmation_required",
            requires_confirmation=True,
        ),
        side_effect_level="external_call",
    )
    plan = ExecutablePlan(
        plan_id="p1",
        goal=goal,
        steps=[step],
        entry_step_ids=["index"],
        final_step_ids=["index"],
    )
    state = AgentState(user_id="u1", session_id="s1", intent="paper_qa", message="index")
    prepared = []

    class _Coordinator:
        def prepare(self, **kwargs):
            prepared.append(kwargs)
            return BackgroundWorkTicket(
                status="waiting_job",
                continuation_id="continuation-1",
                job_id="job-1",
            )

    executor = PlanExecutor(
        approval_store=storage.approval_grants,
        background_work_coordinator=_Coordinator(),
    )

    waiting = executor.execute(plan, state)

    assert waiting.status == "waiting_interaction"
    assert waiting.interaction is not None
    assert calls == []
    payload = waiting.interaction.payload
    storage.approval_grants.create_grant(
        ApprovalGrant(
            grant_id="grant-1",
            interaction_id=waiting.interaction.interaction_id,
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            plan_id="p1",
            step_id="index",
            tool_name="parse_and_index_paper",
            arguments_fingerprint=payload.arguments_fingerprint,
            approved_at="2026-07-13T00:00:00+00:00",
        )
    )

    # LangGraph resume 会恢复到 execute_step 节点；测试显式还原该节点重入前的调度状态。
    waiting.runtime.step_status["index"] = "pending"
    waiting.runtime.interaction = None
    waiting.runtime.turn_status = None
    resumed = executor.execute_runtime(waiting.runtime, state)

    assert resumed.status == "waiting_background_job"
    assert calls == []
    assert prepared[0]["grant_id"] == "grant-1"
    assert prepared[0]["handler_name"] == "paper_qa_index"


def test_persistent_write_asks_confirmation_without_plan_confirmation_policy(tmp_path, monkeypatch) -> None:
    """确认只看契约（I-8）：计划里漏写 confirmation_policy 的持久写入照样弹确认，参数变了要重新确认。"""
    storage = build_storage_container(db_path=str(tmp_path / "executor.sqlite"))
    calls = []
    monkeypatch.setattr(executor_module, "invoke_backend_tool", lambda tool_name, **kwargs: calls.append(tool_name))

    def _plan(arxiv_id: str) -> ExecutablePlan:
        goal = Goal(goal_id="g1", goal_type="preference_action", user_request="收藏")
        step = PlanStep(
            step_id="update",
            action_type="write_state",
            tool_name="update_preference_store",
            tool=ToolSpec(tool_name="update_preference_store", side_effect_level="persistent_write"),
            input_bindings=[
                StepInputBinding(input_key="paper_reference", source_type="literal", value={"arxiv_id": arxiv_id}),
                StepInputBinding(input_key="message", source_type="literal", value="收藏这篇"),
            ],
            output_key="preference_update_result",
            side_effect_level="persistent_write",
        )
        assert step.confirmation_policy is None
        return ExecutablePlan(plan_id="p1", goal=goal, steps=[step], entry_step_ids=["update"], final_step_ids=["update"])

    state = AgentState(user_id="u1", session_id="s1", intent="preference_action", message="收藏这篇")
    executor = PlanExecutor(approval_store=storage.approval_grants)

    waiting = executor.execute(_plan("2401.00001"), state)

    assert waiting.status == "waiting_interaction"
    assert waiting.interaction is not None
    assert calls == []
    storage.approval_grants.create_grant(
        ApprovalGrant(
            grant_id="grant-1",
            interaction_id=waiting.interaction.interaction_id,
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            plan_id="p1",
            step_id="update",
            tool_name="update_preference_store",
            arguments_fingerprint=waiting.interaction.payload.arguments_fingerprint,
            approved_at="2026-07-13T00:00:00+00:00",
        )
    )

    changed = executor.execute(_plan("2401.99999"), state)

    assert changed.status == "waiting_interaction"
    assert calls == []
