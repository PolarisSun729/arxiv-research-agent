from __future__ import annotations

import time

from tests.helpers.agent_runtime import load_agent_test_modules
from tests.helpers.sqlite import build_storage_container


# continuation 依赖 Agent 测试桩提供的轻量运行环境，必须先初始化再导入正式模块。
load_agent_test_modules()

from backend.agents.arxiv_search_agent.execution.continuations import AgentResumeRunManager


def _interaction() -> dict:
    return {
        "interaction_id": "interaction-1",
        "kind": "side_effect_approval",
        "status": "pending",
        "plan_id": "plan-1",
        "step_id": "index-step",
        "payload": {
            "tool_name": "parse_and_index_paper",
            "arguments_fingerprint": "sha256:arguments",
        },
    }


def _prepare_ready_continuation(storage) -> None:
    checkpoint = storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="session-1",
        runtime_state={"current_step_id": "index-step"},
        interaction=_interaction(),
        status="waiting_interaction",
    )
    storage.agent_work.prepare_background_work(
        checkpoint_id=checkpoint["checkpoint_id"],
        user_id="user-1",
        session_id="session-1",
        thread_id="session-1",
        interaction_id="interaction-1",
        grant_id="grant-1",
        invocation_id="invocation-1",
        continuation_id="continuation-1",
        plan_id="plan-1",
        step_id="index-step",
        tool_name="parse_and_index_paper",
        arguments_fingerprint="sha256:arguments",
        handler_name="paper_qa_index",
        job_id="job-1",
        job_idempotency_key="2401.00001:docling:paper_qa_index_v1",
    )
    storage.agent_work.mark_job_ready(
        job_id="job-1",
        validated_result={"build_id": "build-1", "chunk_count": 3},
    )


def test_missing_langgraph_checkpoint_fails_resume_without_replaying_original_question(tmp_path, monkeypatch) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "missing-checkpoint.sqlite"))
    _prepare_ready_continuation(storage)
    invoked = []

    class _GraphWithoutCheckpoint:
        def get_state(self, _config):
            return None

        def invoke(self, *_args, **_kwargs):
            invoked.append(True)
            raise AssertionError("缺失 checkpoint 时禁止把原问题作为新请求重放")

    # pytest 全量收集会多次重载 Agent service；运行时按生产代码相同路径解析，避免 patch 到旧模块对象。
    from backend.agents.arxiv_search_agent import service as runtime_service_module

    monkeypatch.setattr(runtime_service_module, "_build_agent_graph", lambda **_kwargs: _GraphWithoutCheckpoint())
    monkeypatch.setattr(runtime_service_module, "_resolve_generation_service", lambda: object())
    manager = AgentResumeRunManager(storage=storage, background_work_coordinator=object())

    claimed = manager.claim_and_start(
        "continuation-1",
        user_id="user-1",
        session_id="session-1",
    )
    deadline = time.monotonic() + 3
    run = storage.agent_work.get_resume_run(claimed["resume_run_id"])
    while run and run["status"] in {"pending", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
        run = storage.agent_work.get_resume_run(claimed["resume_run_id"])

    assert run is not None
    assert run["status"] == "failed"
    assert "langgraph_checkpoint_missing" in run["error_code"]
    assert invoked == []
    assert storage.agent_work.get_continuation("continuation-1")["status"] == "failed"
