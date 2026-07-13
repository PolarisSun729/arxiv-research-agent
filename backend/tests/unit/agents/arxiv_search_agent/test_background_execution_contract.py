from tests.helpers.agent_runtime import load_agent_test_modules


# 先装配轻量 Agent 运行时，避免契约测试依赖真实 LangGraph、LLM 或向量库。
load_agent_test_modules()

from backend.agents.arxiv_search_agent.tool_registry import UNIFIED_TOOL_REGISTRY
from backend.agents.arxiv_search_agent.execution.background_jobs import (
    BackgroundJobHandlerRegistry,
    BackgroundWorkTicket,
    PaperQAIndexBackgroundHandler,
)
from backend.agents.arxiv_search_agent.execution.approvals import ApprovalGrant
from backend.agents.arxiv_search_agent.plan_executor import PlanExecutor
from backend.agents.arxiv_search_agent.planner import build_plan_runtime
from backend.agents.arxiv_search_agent.schemas import (
    ExecutablePlan,
    Goal,
    PlanStep,
    StepInputBinding,
    StepPolicy,
)
from backend.agents.arxiv_search_agent.state import AgentState
from tests.helpers.sqlite import build_storage_container


def test_parse_and_index_contract_declares_background_handler() -> None:
    contract = UNIFIED_TOOL_REGISTRY.get_contract("parse_and_index_paper")

    assert contract is not None
    assert contract.execution_policy.mode == "background_job"
    assert contract.execution_policy.handler == "paper_qa_index"
    assert contract.debug_summary()["execution_policy"] == {
        "mode": "background_job",
        "handler": "paper_qa_index",
    }


def test_paper_qa_handler_preflights_existing_index_and_missing_job() -> None:
    statuses = {
        "2401.00001": {
            "has_index": True,
            "status": "indexed",
            "active_build_id": "build-1",
            "active_index_version": "version-1",
            "active_collection_name": "qa_collection_1",
            "active_chunk_count": 3,
        },
        "2401.00002": {"has_index": False, "status": "not_indexed"},
    }
    submitted = []
    handler = PaperQAIndexBackgroundHandler(
        status_reader=lambda arxiv_id: statuses[arxiv_id],
        active_job_reader=lambda _arxiv_id: None,
        job_submitter=lambda arxiv_id, loading_method: submitted.append((arxiv_id, loading_method))
        or {"job_id": "job-1", "status": "pending"},
    )
    registry = BackgroundJobHandlerRegistry()
    registry.register("paper_qa_index", handler)

    existing = registry.get("paper_qa_index").preflight(
        {"paper_reference": {"arxiv_id": "2401.00001"}}
    )
    missing = registry.get("paper_qa_index").preflight(
        {"paper_reference": {"arxiv_id": "2401.00002"}, "loading_method": "docling"}
    )
    submission = registry.get("paper_qa_index").submit_or_attach(
        {"paper_reference": {"arxiv_id": "2401.00002"}, "loading_method": "docling"},
        missing,
    )

    assert existing.status == "already_satisfied"
    assert existing.projected_result["build_id"] == "build-1"
    assert missing.status == "missing"
    assert submission.job_id == "job-1"
    assert submission.attached is False
    assert submitted == [("2401.00002", "docling")]


def test_executor_dispatches_background_policy_without_calling_inline_adapter(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "background-executor.sqlite"))
    calls = []

    class _Coordinator:
        def prepare(self, **kwargs):
            calls.append(kwargs)
            return BackgroundWorkTicket(
                status="waiting_job",
                continuation_id="continuation-1",
                job_id="job-1",
            )

    contract = UNIFIED_TOOL_REGISTRY.get_contract("parse_and_index_paper")
    step = PlanStep(
        step_id="index",
        action_type="index",
        tool_name=contract.tool_name,
        tool=contract.to_tool_spec(),
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
        plan_id="plan-1",
        goal=Goal(goal_id="goal-1", goal_type="paper_qa", user_request="index"),
        steps=[step],
        entry_step_ids=["index"],
        final_step_ids=["index"],
    )
    state = AgentState(user_id="user-1", session_id="session-1", intent="paper_qa", message="index")
    executor = PlanExecutor(
        approval_store=storage.approval_grants,
        background_work_coordinator=_Coordinator(),
    )
    waiting = executor.execute(plan, state)
    payload = waiting.interaction.payload
    storage.approval_grants.create_grant(
        ApprovalGrant(
            grant_id="grant-1",
            interaction_id=waiting.interaction.interaction_id,
            user_id="user-1",
            session_id="session-1",
            thread_id="session-1",
            plan_id="plan-1",
            step_id="index",
            tool_name="parse_and_index_paper",
            arguments_fingerprint=payload.arguments_fingerprint,
            approved_at="2026-07-13T00:00:00+00:00",
        )
    )
    waiting.runtime.step_status["index"] = "pending"
    waiting.runtime.interaction = None
    waiting.runtime.turn_status = None

    resumed = executor.execute_runtime(waiting.runtime, state)

    assert resumed.status == "waiting_background_job"
    assert resumed.runtime.step_status["index"] == "waiting_background_job"
    assert resumed.runtime.last_step_output["background_work"]["continuation_id"] == "continuation-1"
    assert calls[0]["handler_name"] == "paper_qa_index"


def test_internal_background_resume_projects_result_without_reinvoking_tool(monkeypatch) -> None:
    projected = {
        "status": "indexed",
        "has_index": True,
        "build_id": "build-1",
        "index_version": "version-1",
        "collection_name": "qa_collection_1",
        "chunk_count": 3,
    }
    monkeypatch.setitem(
        PlanExecutor.execute_current_step_tool.__globals__,
        "interrupt",
        lambda _payload: {
            "decision": "background_completed",
            "continuation_id": "continuation-1",
            "validated_result": projected,
        },
    )
    contract = UNIFIED_TOOL_REGISTRY.get_contract("parse_and_index_paper")
    step = PlanStep(
        step_id="index",
        action_type="index",
        tool_name=contract.tool_name,
        tool=contract.to_tool_spec(),
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
        plan_id="plan-1",
        goal=Goal(goal_id="goal-1", goal_type="paper_qa", user_request="index"),
        steps=[step],
        entry_step_ids=["index"],
        final_step_ids=["index"],
    )
    state = AgentState(user_id="user-1", session_id="session-1", intent="paper_qa", message="index")

    runtime = build_plan_runtime(state, goal=plan.goal, plan=plan, turn_status="success")
    runtime.step_status = {"index": "pending"}
    runtime.current_step_id = "index"
    result = PlanExecutor().execute_current_step_tool(runtime, state, allow_interrupt=True)

    assert result.next_action == "continue"
    assert result.output["build_id"] == "build-1"
    assert runtime.last_step_output["normalized_output"]["build_id"] == "build-1"
    assert any(trace.event == "background_job_result_projected" for trace in runtime.trace)
