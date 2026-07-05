from __future__ import annotations

import json
import sys

import pytest

from tests.helpers import build_storage_container
from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
runtime_checkpoint = _MODULES["runtime_checkpoint_module"]
service = _MODULES["service_module"]
AgentState = _MODULES["state_module"].AgentState
schemas = _MODULES["schemas"]
state_utils = sys.modules["backend.agents.arxiv_search_agent.utils.state_utils"]


class _InMemoryRuntimeCheckpointStore:
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

    def consume_agent_runtime_pending_confirmation(self, **kwargs):
        if not self.record:
            return False
        if self.record.get("user_id") != kwargs.get("user_id"):
            return False
        if self.record.get("session_id") != kwargs.get("session_id"):
            return False
        if self.record.get("thread_id") != kwargs.get("thread_id"):
            return False
        if self.record.get("status") != runtime_checkpoint.CHECKPOINT_STATUS_WAITING:
            return False
        if not self.record.get("pending_confirmation"):
            return False
        self.record["status"] = runtime_checkpoint.CHECKPOINT_STATUS_RUNNING
        self.record["pending_confirmation"] = None
        self.record["next_route"] = kwargs.get("next_route")
        self.record["expires_at"] = None
        return True

    def expire_agent_runtime_checkpoints(self, **_kwargs):
        return 0

    def cleanup_agent_runtime_checkpoints(self, **_kwargs):
        return 0

    def cleanup_langgraph_checkpoints_for_terminal_runtime(self, **_kwargs):
        self.langgraph_cleanup_called = True
        return {"threads": 0, "checkpoints": 0, "writes": 0}


def _confirmation_payload(step_id: str = "parse_and_index_paper") -> dict:
    return {
        "pending_action_id": f"pending:{step_id}",
        "step_id": step_id,
        "tool_name": "parse_and_index_paper",
        "action_type": "paper_index",
        "side_effect_level": "external_call",
        "reason": "paper_index_missing",
    }


def test_runtime_checkpoint_persists_pending_confirmation_as_resume_truth() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)

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

    record = store.record
    assert record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_WAITING
    assert record["pending_confirmation"]["step_id"] == "parse_and_index_paper"
    assert record["expires_at"]
    assert manager.validate_resume(
        user_id="u1",
        session_id="s1",
        thread_id="s1",
        resume_payload={"decision": "approve", "step_id": "parse_and_index_paper"},
    )


def test_runtime_checkpoint_ignores_debug_pending_confirmation_after_business_state_cleared() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)

    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"turn_status": "success", "pending_confirmation": None},
            "plan_runtime": {"pending_confirmation": _confirmation_payload("stale-plan-step"), "error": "stale-error"},
            "pending_action": {"status": "waiting_confirmation", "step_id": "display-only"},
            # debug 只保留排查快照；即使残留旧确认和旧 route，也不能重新驱动 checkpoint。
            "debug": {
                "pending_confirmation": _confirmation_payload("stale-debug-step"),
                "agent_route": {"decision": "waiting_confirmation"},
            },
        },
        current_node="finalize",
    )

    record = store.record
    assert record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED
    assert record["pending_confirmation"] is None
    assert record["next_route"] == runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED
    assert record["expires_at"] is None


def test_service_final_persist_uses_runtime_state_over_stale_debug_and_plan_runtime() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)
    stale_confirmation = schemas.ConfirmationRequest(**_confirmation_payload("stale-plan-step"))
    state = AgentState(
        user_id="u1",
        session_id="s1",
        debug={"pending_confirmation": _confirmation_payload("stale-debug-step")},
        pending_action={"status": "waiting_confirmation", "step_id": "display-only"},
        runtime_state=schemas.AgentRuntimeState(turn_status="success", pending_confirmation=None),
        plan_runtime=schemas.PlanRuntime(pending_confirmation=stale_confirmation, error="stale-error"),
    )

    service._persist_runtime_checkpoint_after_turn(manager, state, is_resume=True)

    assert store.record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED
    assert store.record["pending_confirmation"] is None


def test_stream_interrupt_state_promotes_confirmation_to_runtime_truth() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)
    state = AgentState(user_id="u1", session_id="s1", debug={})

    next_state = service._apply_stream_interrupt_state(state, _confirmation_payload())
    manager.persist_state(next_state, current_node="__interrupt__")

    assert next_state.runtime_state is not None
    assert next_state.runtime_state.pending_confirmation is not None
    assert next_state.runtime_state.pending_confirmation.step_id == "parse_and_index_paper"
    assert store.record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_WAITING
    assert store.record["pending_confirmation"]["step_id"] == "parse_and_index_paper"


def test_refresh_execution_plan_runtime_does_not_restore_pending_confirmation_from_pending_action() -> None:
    confirmation = schemas.ConfirmationRequest(**_confirmation_payload("display-only-step"))
    state = AgentState(
        execution_plan=schemas.ExecutablePlan(plan_id="plan-1", goal=schemas.Goal(goal_type="paper_qa")),
        pending_action={
            "status": "waiting_confirmation",
            "step_id": "display-only-step",
            "confirmation_request": confirmation.model_dump(mode="json"),
        },
    )

    next_state = state_utils._refresh_execution_plan_runtime(state)

    assert next_state.plan_runtime is not None
    assert next_state.plan_runtime.pending_confirmation is None


def test_runtime_checkpoint_rejects_wrong_step_id() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)
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


def test_runtime_checkpoint_rejects_wrong_pending_action_identity() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)
    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"pending_confirmation": _confirmation_payload("parse_and_index_paper")},
        }
    )

    with pytest.raises(runtime_checkpoint.AgentRuntimeCheckpointError) as exc_info:
        manager.validate_resume(
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            resume_payload={
                "decision": "approve",
                "step_id": "parse_and_index_paper",
                "tool_name": "other_tool",
                "pending_action_id": "pending:other-step",
            },
        )

    assert exc_info.value.reason in {"tool_name_mismatch", "pending_action_id_mismatch"}


def test_runtime_checkpoint_consumes_pending_confirmation_once() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)
    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"pending_confirmation": {"step_id": "parse_and_index_paper"}},
        }
    )

    manager.validate_resume(
        user_id="u1",
        session_id="s1",
        thread_id="s1",
        resume_payload={"decision": "approve", "step_id": "parse_and_index_paper"},
    )
    manager.consume_pending_confirmation(
        user_id="u1",
        session_id="s1",
        thread_id="s1",
        resume_payload={"decision": "approve", "step_id": "parse_and_index_paper"},
    )

    assert store.record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_RUNNING
    assert store.record["pending_confirmation"] is None
    with pytest.raises(runtime_checkpoint.AgentRuntimeCheckpointError) as consumed_info:
        manager.consume_pending_confirmation(
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            resume_payload={"decision": "approve", "step_id": "parse_and_index_paper"},
        )
    assert consumed_info.value.reason == "confirmation_already_consumed"
    with pytest.raises(runtime_checkpoint.AgentRuntimeCheckpointError) as validate_info:
        manager.validate_resume(
            user_id="u1",
            session_id="s1",
            thread_id="s1",
            resume_payload={"decision": "approve", "step_id": "parse_and_index_paper"},
        )
    assert validate_info.value.reason == "checkpoint_not_waiting:running"


def test_runtime_store_consume_approves_pending_confirmation_target_step() -> None:
    storage = build_storage_container()
    store = storage.agent_runtime_checkpoints
    raw_runtime_state = json.dumps(
        {
            "pending_confirmation": _confirmation_payload("parse_and_index_paper"),
            "approved_step_ids": [],
            "step_status": {
                "parse_and_index_paper": "waiting_confirmation",
                "request_confirmation": "waiting_confirmation",
            },
            "recovery_strategy": {"type": "request_confirmation", "step_id": "request_confirmation"},
        }
    )

    try:
        cleaned_raw = store._runtime_state_without_pending_confirmation(
            raw_runtime_state,
            decision="approve",
            step_id="request_confirmation",
        )
    finally:
        storage._test_temp_db.cleanup()
    cleaned = json.loads(cleaned_raw)

    assert cleaned["pending_confirmation"] is None
    assert cleaned["approved_step_ids"] == ["parse_and_index_paper"]
    assert cleaned["step_status"]["parse_and_index_paper"] == "pending"
    assert cleaned["step_status"]["request_confirmation"] == "waiting_confirmation"
    assert cleaned["recovery_strategy"] is None


def test_runtime_store_lists_runtime_checkpoints_by_thread_candidates(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "runtime-checkpoints.sqlite"), initialize_schema=False)
    store = storage.agent_runtime_checkpoints

    with store._get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE agent_runtime_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                runtime_state_json TEXT,
                graph_state_json TEXT,
                pending_confirmation_json TEXT,
                current_node TEXT,
                next_route TEXT,
                status TEXT,
                error_summary TEXT,
                created_at TEXT,
                updated_at TEXT,
                expires_at TEXT
            )
            """
        )
        rows = [
            (
                "default:s1:s1",
                "default",
                "s1",
                "s1",
                {"approved_step_ids": []},
                {"node": "execute_step"},
                {"step_id": "parse_and_index_paper"},
                "execute_step",
                "running",
                "waiting_confirmation",
                "",
                "2026-07-02T13:07:15",
                "2026-07-02T13:07:15",
                None,
            ),
            (
                "local_user:s1:s1",
                "local_user",
                "s1",
                "s1",
                {"approved_step_ids": ["parse_and_index_paper"]},
                {"node": "execute_step"},
                None,
                "execute_step",
                "running",
                "running",
                "",
                "2026-07-02T13:07:16",
                "2026-07-02T13:07:16",
                None,
            ),
        ]
        for row in rows:
            conn.execute(
                """
                INSERT INTO agent_runtime_checkpoints (
                    checkpoint_id, user_id, session_id, thread_id, runtime_state_json,
                    graph_state_json, pending_confirmation_json, current_node, next_route,
                    status, error_summary, created_at, updated_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    json.dumps(row[4]),
                    json.dumps(row[5]),
                    json.dumps(row[6]) if row[6] is not None else "",
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                    row[11],
                    row[12],
                    row[13],
                ),
            )
        conn.commit()

    candidates = store.list_agent_runtime_checkpoints_by_thread(session_id="s1", thread_id="s1")

    assert [candidate["user_id"] for candidate in candidates] == ["local_user", "default"]
    assert candidates[0]["runtime_state"]["approved_step_ids"] == ["parse_and_index_paper"]
    # 单记录兼容接口只在原始候选唯一时返回，避免 user_id 丢失时直接误用多用户记录。
    assert store.get_agent_runtime_checkpoint_by_thread(session_id="s1", thread_id="s1") is None


def test_runtime_checkpoint_does_not_restore_consumed_confirmation_from_stale_stream_state() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    store.record = {
        "user_id": "u1",
        "session_id": "s1",
        "thread_id": "s1",
        "runtime_state": {
            "pending_confirmation": None,
            "approved_step_ids": ["parse_and_index_paper"],
        },
        "pending_confirmation": None,
        "status": runtime_checkpoint.CHECKPOINT_STATUS_RUNNING,
        "next_route": runtime_checkpoint.CHECKPOINT_STATUS_RUNNING,
        "expires_at": None,
    }
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)

    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {
                "pending_confirmation": _confirmation_payload("parse_and_index_paper"),
                "approved_step_ids": [],
            },
        },
        current_node="execute_step",
    )

    assert store.record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_RUNNING
    assert store.record["pending_confirmation"] is None
    assert store.record["runtime_state"]["approved_step_ids"] == ["parse_and_index_paper"]


def test_runtime_checkpoint_terminal_status_cannot_resume_again() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)
    manager.persist_state(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"pending_confirmation": {"step_id": "parse_and_index_paper"}},
        }
    )
    store.mark_agent_runtime_checkpoint_status(
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
    store = _InMemoryRuntimeCheckpointStore()
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)

    manager.mark_terminal(
        {
            "user_id": "u1",
            "session_id": "s1",
            "runtime_state": {"turn_status": "success"},
        },
        status=runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED,
    )

    assert store.record["status"] == runtime_checkpoint.CHECKPOINT_STATUS_COMPLETED
    assert store.record["pending_confirmation"] is None


def test_runtime_checkpoint_cleanup_aligns_langgraph_lifecycle() -> None:
    store = _InMemoryRuntimeCheckpointStore()
    store.langgraph_cleanup_called = False
    manager = runtime_checkpoint.AgentRuntimeCheckpointManager(runtime_checkpoint_store=store)

    manager.expire_and_cleanup()

    assert store.langgraph_cleanup_called is True
