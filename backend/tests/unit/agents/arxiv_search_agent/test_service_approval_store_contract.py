from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from tests.helpers.agent_runtime import FakeAgentStorageStore, FakeStorageContainer, load_agent_test_modules


_MODULES = load_agent_test_modules()
service_module = _MODULES["service_module"]
state_module = _MODULES["state_module"]
schemas = _MODULES["schemas"]


def test_context_lifecycle_debug_accepts_approval_store_dependency() -> None:
    store = FakeAgentStorageStore()

    debug = service_module._build_agent_context_lifecycle_debug(
        runtime_checkpoint_store=store,
        approval_store=store,
        user_id="u1",
        session_id="s1",
        user_memory_debug={"context_merge": {"accepted_frontend_fields": ["frontend_visible_paper"]}},
    )

    assert debug["degraded"]["approval_store_missing"] is False
    assert debug["agent_runtime_checkpoint"] == {"exists": False}


def test_run_arxiv_search_agent_passes_approval_store_to_graph(monkeypatch) -> None:
    captured = {}

    class FakeGraph:
        def invoke(self, state, config=None):
            captured["invoke_config"] = config
            payload = dict(state)
            payload["intent"] = payload.get("intent") or "unsupported"
            payload["answer"] = "ok"
            return state_module.AgentState.model_validate(payload)

    def fake_build_agent_graph(
        generation_service=None,
        *,
        langgraph_checkpoint_store,
        runtime_checkpoint_store,
        approval_store,
        background_work_coordinator,
    ):
        # graph 构建是副作用确认链路的入口；这里强制要求 service 把授权 store 透传下来。
        captured["approval_store"] = approval_store
        captured["runtime_checkpoint_store"] = runtime_checkpoint_store
        captured["langgraph_checkpoint_store"] = langgraph_checkpoint_store
        captured["background_work_coordinator"] = background_work_coordinator
        return FakeGraph()

    monkeypatch.setattr(service_module, "_build_agent_graph", fake_build_agent_graph)
    monkeypatch.setattr(service_module, "_write_request_trace", lambda *args, **kwargs: None)

    response = service_module.run_arxiv_search_agent(
        schemas.ArxivSearchRequest(user_id="u1", session_id="s1", message="search rag")
    )

    assert response.answer == "ok"
    assert captured["approval_store"] is not None
    assert captured["approval_store"] is captured["runtime_checkpoint_store"]
    assert captured["invoke_config"]["configurable"]["thread_id"] == "s1"


def _approval_snapshot_state():
    step = schemas.PlanStep(
        step_id="parse_and_index_paper",
        action_type="parse_and_index",
        tool_name="parse_and_index_paper",
        tool=schemas.ToolSpec(
            tool_name="parse_and_index_paper",
            side_effect_level="persistent_write",
            requires_confirmation=True,
        ),
        confirmation_policy=schemas.StepPolicy(
            policy_type="confirmation",
            mode="explicit",
            requires_confirmation=True,
        ),
        side_effect_level="persistent_write",
        status="waiting_interaction",
    )
    plan = schemas.ExecutablePlan(
        plan_id="plan-1",
        goal=schemas.Goal(goal_type="paper_qa", intent="paper_qa"),
        steps=[step],
        entry_step_ids=[step.step_id],
        final_step_ids=[step.step_id],
    )
    return state_module.AgentState(
        user_id="u1",
        session_id="s1",
        message="解析论文",
        intent="paper_qa",
        execution_plan=plan,
        runtime_state=schemas.AgentRuntimeState(
            current_step_id=step.step_id,
            current_step_index=0,
            step_status={step.step_id: "waiting_interaction"},
        ),
    )


async def _collect_stream_events(response, *, limit: int):
    events = []
    async for chunk in response.body_iterator:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        data_line = next((line for line in text.splitlines() if line.startswith("data: ")), "")
        if data_line:
            events.append(json.loads(data_line.removeprefix("data: ")))
        if len(events) >= limit:
            break
    return events


def test_stream_background_approval_ends_after_persisting_continuation_ticket(monkeypatch) -> None:
    snapshot_state = _approval_snapshot_state()

    class ResumeCapableStore(FakeAgentStorageStore):
        def get_agent_runtime_checkpoint(self, **kwargs):
            checkpoint = super().get_agent_runtime_checkpoint(**kwargs)
            if checkpoint is None:
                return None
            return {
                **checkpoint,
                "checkpoint_id": "checkpoint-1",
                "schema_version": 2,
            }

        def resolve_side_effect_interaction(self, *, checkpoint_id, grant):
            # 这个用例只验证流式层的事件时序；授权事务本身由 interaction_runtime 单测覆盖。
            assert checkpoint_id == "checkpoint-1"
            self.approved_grant = grant

    class FakeGraph:
        def get_state(self, config=None):
            del config
            return {"values": snapshot_state.model_dump(mode="json")}

        def stream(self, *_args, **_kwargs):
            raise AssertionError("后台批准请求不能继续消费 LangGraph interrupt")
            yield  # pragma: no cover

    class FakeBackgroundCoordinator:
        def approve_interaction(self, **kwargs):
            assert kwargs["handler_name"] == "paper_qa_index"
            return SimpleNamespace(
                status="waiting_job",
                continuation_id="continuation-1",
                job_id="job-1",
            )

    monkeypatch.setattr(
        service_module,
        "_build_agent_graph",
        lambda **_kwargs: FakeGraph(),
    )
    monkeypatch.setattr(service_module, "StorageContainer", lambda: FakeStorageContainer(ResumeCapableStore()))
    monkeypatch.setattr(service_module, "_resolve_background_work_coordinator", lambda: FakeBackgroundCoordinator())
    monkeypatch.setattr(service_module, "_write_request_trace", lambda *args, **kwargs: None)

    response = service_module.stream_arxiv_search_agent(
        schemas.ArxivSearchRequest(
            user_id="u1",
            session_id="s1",
            message="解析论文",
            resume=schemas.InteractionResumeRequest(
                interaction_id="interaction-1",
                decision="approve",
            ),
        )
    )

    events = asyncio.run(_collect_stream_events(response, limit=3))

    assert [event["event_type"] for event in events] == ["run_start", "final_response", "stream_end"]
    assert events[1]["data"]["response"]["paper_qa_result"]["background_work"] == {
        "continuation_id": "continuation-1",
        "job_id": "job-1",
        "status": "waiting_job",
    }
    assert events[2]["data"]["status"] == "background_work_started"
