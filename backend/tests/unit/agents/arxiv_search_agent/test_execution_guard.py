from agents.arxiv_search_agent.execution.approvals import ApprovalGrant
from agents.arxiv_search_agent.execution.execution_guard import ExecutionGuard
from agents.arxiv_search_agent.schemas import ExecutablePlan, Goal, PlanRuntime, PlanStep, StepPolicy, ToolSpec
from agents.arxiv_search_agent.state import AgentState
from tests.helpers.sqlite import build_storage_container


def _runtime(step: PlanStep) -> PlanRuntime:
    goal = Goal(goal_id="g1", goal_type="paper_qa")
    plan = ExecutablePlan(plan_id="p1", goal=goal, steps=[step], entry_step_ids=[step.step_id], final_step_ids=[step.step_id])
    return PlanRuntime(goal=goal, plan=plan)


def _step() -> PlanStep:
    return PlanStep(
        step_id="index",
        action_type="index",
        tool_name="parse_and_index_paper",
        tool=ToolSpec(tool_name="parse_and_index_paper"),
        confirmation_policy=StepPolicy(
            policy_type="confirmation",
            mode="explicit_user_confirmation_required",
            requires_confirmation=True,
        ),
        side_effect_level="external_call",
    )


def test_guard_creates_parameter_bound_interaction_without_grant() -> None:
    step = _step()
    decision = ExecutionGuard(None).evaluate(
        step=step,
        runtime=_runtime(step),
        state=AgentState(user_id="u1", session_id="s1"),
        arguments={"paper_reference": {"arxiv_id": "2401.00001"}},
        reason="需要建立索引",
    )

    assert decision.action == "wait_for_interaction"
    assert decision.interaction.kind == "side_effect_approval"
    assert decision.interaction.payload.arguments_fingerprint == decision.arguments_fingerprint


def test_guard_only_accepts_grant_for_exact_arguments(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "guard.sqlite"))
    step = _step()
    runtime = _runtime(step)
    arguments = {"paper_reference": {"arxiv_id": "2401.00001"}}
    first = ExecutionGuard(None).evaluate(
        step=step,
        runtime=runtime,
        state=AgentState(user_id="u1", session_id="s1"),
        arguments=arguments,
        reason="需要建立索引",
    )
    storage.approval_grants.create_grant(
        ApprovalGrant(
            grant_id="grant-1",
            interaction_id=first.interaction.interaction_id,
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            plan_id="p1",
            step_id="index",
            tool_name="parse_and_index_paper",
            arguments_fingerprint=first.arguments_fingerprint,
            approved_at="2026-07-13T00:00:00+00:00",
        )
    )

    accepted = ExecutionGuard(storage.approval_grants).evaluate(
        step=step,
        runtime=runtime,
        state=AgentState(user_id="u1", session_id="s1"),
        arguments=arguments,
        reason="需要建立索引",
    )
    changed = ExecutionGuard(storage.approval_grants).evaluate(
        step=step,
        runtime=runtime,
        state=AgentState(user_id="u1", session_id="s1"),
        arguments={"paper_reference": {"arxiv_id": "2401.00002"}},
        reason="需要建立索引",
    )

    assert accepted.action == "execute"
    assert accepted.grant_id == "grant-1"
    assert changed.action == "wait_for_interaction"
