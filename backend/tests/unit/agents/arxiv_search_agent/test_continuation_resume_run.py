from __future__ import annotations

import time

import pytest

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


def _wait_for_terminal_run(storage, resume_run_id: str) -> dict:
    deadline = time.monotonic() + 3
    run = storage.agent_work.get_resume_run(resume_run_id)
    while run and run["status"] in {"pending", "running"} and time.monotonic() < deadline:
        time.sleep(0.01)
        run = storage.agent_work.get_resume_run(resume_run_id)
    assert run is not None
    return run


def test_resume_run_fails_when_graph_returns_interrupt_without_answer(tmp_path, monkeypatch) -> None:
    """假完成防护：续跑后仍 interrupt / 空答时不得 complete_resume_run。"""
    storage = build_storage_container(db_path=str(tmp_path / "false-complete.sqlite"))
    _prepare_ready_continuation(storage)
    invoked = []

    class _GraphReinterrupt:
        def get_state(self, _config):
            return {"values": {"intent": "paper_qa"}, "next": ("execute_step",)}

        def stream(self, command, config=None, stream_mode=None):
            invoked.append(
                {
                    "decision": getattr(command, "resume", None) if not isinstance(command, dict) else command,
                    "continuation_id": ((config or {}).get("configurable") or {}).get("background_continuation_id"),
                }
            )
            # 模拟线上故障：resume 后又停在新的确认 interrupt，且载荷无法投影成交互。
            yield (
                "updates",
                {
                    "__interrupt__": (
                        {
                            "value": {
                                "interaction_id": "new-interaction",
                                "kind": "side_effect_approval",
                                "step_id": "index-step",
                            }
                        },
                    )
                },
            )

    from backend.agents.arxiv_search_agent import service as runtime_service_module

    monkeypatch.setattr(runtime_service_module, "_build_agent_graph", lambda **_kwargs: _GraphReinterrupt())
    monkeypatch.setattr(runtime_service_module, "_resolve_generation_service", lambda: object())
    manager = AgentResumeRunManager(storage=storage, background_work_coordinator=object())

    claimed = manager.claim_and_start(
        "continuation-1",
        user_id="user-1",
        session_id="session-1",
    )
    run = _wait_for_terminal_run(storage, claimed["resume_run_id"])

    assert run["status"] == "failed"
    assert "background_resume_reinterrupted" in str(run.get("error_code") or "")
    assert run.get("final_response_json") in (None, "", {})
    assert invoked and invoked[0]["continuation_id"] == "continuation-1"
    # 不得被标成 completed 空答
    cont = storage.agent_work.get_continuation("continuation-1")
    assert cont["status"] != "resumed"


def test_resume_run_buffers_node_progress_before_completing(tmp_path, monkeypatch) -> None:
    """续跑期间的节点事件要实时进缓存，且全部早于终态落库，SSE 才能先推进度再推最终回答。"""
    storage = build_storage_container(db_path=str(tmp_path / "progress.sqlite"))
    _prepare_ready_continuation(storage)
    final_values = {
        "user_id": "user-1",
        "session_id": "session-1",
        "intent": "paper_qa",
        "answer": "续跑后的回答",
    }
    progress_seen_at_completion = []

    class _GraphWithProgress:
        def get_state(self, _config):
            return {"values": {"intent": "paper_qa"}, "next": ("finalize",)}

        def stream(self, command, config=None, stream_mode=None):
            yield ("tasks", {"name": "finalize", "input": {}})
            yield ("updates", {"finalize": final_values})

    from backend.agents.arxiv_search_agent import service as runtime_service_module

    monkeypatch.setattr(runtime_service_module, "_build_agent_graph", lambda **_kwargs: _GraphWithProgress())
    monkeypatch.setattr(runtime_service_module, "_resolve_generation_service", lambda: object())
    manager = AgentResumeRunManager(storage=storage, background_work_coordinator=object())
    complete_resume_run = storage.agent_work.complete_resume_run

    def _record_progress_then_complete(resume_run_id, **kwargs):
        progress_seen_at_completion.extend(manager.progress_events(resume_run_id))
        return complete_resume_run(resume_run_id, **kwargs)

    monkeypatch.setattr(storage.agent_work, "complete_resume_run", _record_progress_then_complete)

    claimed = manager.claim_and_start(
        "continuation-1",
        user_id="user-1",
        session_id="session-1",
    )
    run = _wait_for_terminal_run(storage, claimed["resume_run_id"])

    assert run["status"] == "completed"
    assert [event["event_type"] for event in progress_seen_at_completion] == ["step_start", "step_end"]
    assert all(event["run_id"] == claimed["resume_run_id"] for event in progress_seen_at_completion)
    assert [event["sequence"] for event in progress_seen_at_completion] == [1, 2]
    # 游标之后只返回新事件，SSE 轮询据此增量推送。
    assert manager.progress_events(claimed["resume_run_id"], after=1) == progress_seen_at_completion[1:]


def test_resume_run_writes_answer_back_to_session_memory(tmp_path, monkeypatch) -> None:
    """续跑完成后会话记忆要换成续跑的问答结果，下一轮追问才能找到上一轮问答的论文。"""
    storage = build_storage_container(db_path=str(tmp_path / "session-memory.sqlite"))
    _prepare_ready_continuation(storage)
    final_values = {
        "user_id": "user-1",
        "session_id": "session-1",
        "intent": "paper_qa",
        "answer": "续跑后的回答",
        "paper_qa_result": {"status": "answered", "arxiv_id": "2401.00001", "answer": "续跑后的回答"},
    }

    class _GraphAnswered:
        def get_state(self, _config):
            return {"values": {"intent": "paper_qa"}, "next": ("finalize",)}

        def stream(self, command, config=None, stream_mode=None):
            yield ("updates", {"finalize": final_values})

    from backend.agents.arxiv_search_agent import service as runtime_service_module

    saved_states = []

    class _RecordingMemoryService:
        def save_agent_memory(self, *, user_id, session_id, final_state):
            saved_states.append({"user_id": user_id, "session_id": session_id, "final_state": final_state})

    monkeypatch.setattr(runtime_service_module, "_build_agent_graph", lambda **_kwargs: _GraphAnswered())
    monkeypatch.setattr(runtime_service_module, "_resolve_generation_service", lambda: object())
    monkeypatch.setattr(runtime_service_module, "_build_memory_service", lambda _storage: _RecordingMemoryService())
    manager = AgentResumeRunManager(storage=storage, background_work_coordinator=object())

    claimed = manager.claim_and_start(
        "continuation-1",
        user_id="user-1",
        session_id="session-1",
    )
    run = _wait_for_terminal_run(storage, claimed["resume_run_id"])

    assert run["status"] == "completed"
    # 挂起时写进会话记忆的是 waiting_interaction；续跑结束必须用续跑后的问答结果覆盖它。
    assert len(saved_states) == 1
    assert saved_states[0]["session_id"] == "session-1"
    assert saved_states[0]["final_state"].paper_qa_result["arxiv_id"] == "2401.00001"


@pytest.mark.parametrize("mode,status", [("jwt", "pending"), ("api_key", "pending"), ("jwt", "running")])
def test_restart_recovery_never_recreates_jwt_authority_from_stored_user_id(tmp_path, monkeypatch, mode, status):
    """真实持久化状态验证：JWT 待运行任务不能在重启后凭旧 user_id 自动恢复执行权限。"""
    monkeypatch.setenv("AUTH_MODE", mode)
    storage = build_storage_container(db_path=str(tmp_path / "restart.sqlite"))
    _prepare_ready_continuation(storage)
    run = storage.agent_work.claim_resume_run("continuation-1", user_id="user-1", session_id="session-1")
    if status == "running":
        storage.agent_work.start_resume_run(run["resume_run_id"])
    manager = AgentResumeRunManager(storage=storage, background_work_coordinator=object())
    started = []
    monkeypatch.setattr(manager, "start", started.append)
    manager.recover_incomplete_runs()
    recovered = storage.agent_work.get_resume_run(run["resume_run_id"])
    if status == "running":
        assert recovered["status"] == "indeterminate" and recovered["error_code"] == "resume_process_restarted"
        assert started == []
    elif mode == "jwt":
        assert recovered["status"] == "failed" and recovered["error_code"] == "authentication_context_lost"
        assert started == []
    else:
        assert started == [run["resume_run_id"]]
