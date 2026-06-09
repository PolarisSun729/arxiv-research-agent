from __future__ import annotations

import pytest

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
runtime_checkpoint = _MODULES["runtime_checkpoint_module"]


class _InMemoryCheckpointDatabase:
    def __init__(self) -> None:
        self.record = None

    def upsert_agent_runtime_checkpoint(self, **kwargs):
        self.record = dict(kwargs)
        self.record.setdefault("status", "running")
        return dict(self.record)

    def get_agent_runtime_checkpoint(self, **kwargs):
        if not self.record:
            return None
        if self.record.get("user_id") != kwargs.get("user_id"):
            return None
        if self.record.get("session_id") != kwargs.get("session_id"):
            return None
        if self.record.get("thread_id") != kwargs.get("thread_id"):
            return None
        return dict(self.record)

    def mark_agent_runtime_checkpoint_status(self, **kwargs):
        if not self.record:
            return False
        self.record["status"] = kwargs.get("status")
        if kwargs.get("clear_pending_confirmation"):
            self.record["pending_confirmation"] = None
        return True

    def expire_agent_runtime_checkpoints(self, **_kwargs):
        return 0

    def cleanup_agent_runtime_checkpoints(self, **_kwargs):
        return 0

    def cleanup_langgraph_checkpoints_for_terminal_runtime(self, **_kwargs):
        self.langgraph_cleanup_called = True
        return {"threads": 0, "checkpoints": 0, "writes": 0}


def test_runtime_checkpoint_persists_pending_confirmation_as_resume_truth() -> None:
    database = _InMemoryCheckpointDatabase()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(database_service=database)

    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {
                "pending_confirmation": {
                    "step_id": "parse_and_index_paper",
                    "tool_name": "parse_and_index_paper",
                }
            },
            # pending_action 只是展示镜像；测试保留它是为了证明真源来自 runtime_state。
            "pending_action": {"status": "waiting_confirmation", "step_id": "display-only"},
        },
        current_node="finalize",
    )

    record = database.record
    assert record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_WAITING
    assert record["pending_confirmation"]["step_id"] == "parse_and_index_paper"
    assert record["expires_at"]
    assert manager.validate_resume(
        user_id="u1",
        session_id="s1",
        thread_id="s1",
        resume_payload={"decision": "approve", "step_id": "parse_and_index_paper"},
    )


def test_runtime_checkpoint_rejects_wrong_step_id() -> None:
    database = _InMemoryCheckpointDatabase()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(database_service=database)
    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"pending_confirmation": {"step_id": "expected_step"}},
        }
    )

    with pytest.raises(runtime_checkpoint.AgentRuntimeCheckpointError) as exc_info:
        manager.validate_resume(
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            resume_payload={"decision": "approve", "step_id": "other_step"},
        )

    assert exc_info.value.reason == "step_id_mismatch"


def test_runtime_checkpoint_terminal_status_cannot_resume_again() -> None:
    database = _InMemoryCheckpointDatabase()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(database_service=database)
    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"pending_confirmation": {"step_id": "parse_and_index_paper"}},
        }
    )
    database.mark_agent_runtime_checkpoint_status(
        user_id="u1",
        session_id="s1",
        thread_id="s1",
        status=runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED,
        clear_pending_confirmation=True,
    )

    with pytest.raises(runtime_checkpoint.AgentRuntimeCheckpointError) as exc_info:
        manager.validate_resume(
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            resume_payload={"decision": "approve", "step_id": "parse_and_index_paper"},
        )

    assert exc_info.value.reason == "checkpoint_not_waiting:completed"


def test_runtime_checkpoint_mark_terminal_creates_record_for_normal_turn() -> None:
    database = _InMemoryCheckpointDatabase()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(database_service=database)

    manager.mark_terminal(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"turn_status": "success"},
        },
        status=runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED,
    )

    assert database.record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED
    assert database.record["pending_confirmation"] is None


def test_runtime_checkpoint_cleanup_aligns_langgraph_lifecycle() -> None:
    database = _InMemoryCheckpointDatabase()
    database.langgraph_cleanup_called = False
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(database_service=database)

    manager.expire_and_cleanup()

    assert database.langgraph_cleanup_called is True
