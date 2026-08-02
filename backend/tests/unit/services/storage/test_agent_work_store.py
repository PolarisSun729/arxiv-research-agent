from tests.helpers.sqlite import build_storage_container
from services.storage.sqlite.stores.agent_work import AgentWorkConflict


def _interaction() -> dict:
    return {
        "interaction_id": "interaction-1",
        "kind": "side_effect_approval",
        "status": "pending",
        "plan_id": "plan-1",
        "step_id": "index-step",
        "payload": {
            "tool_name": "parse_and_index_paper",
            "action_type": "index",
            "reason": "paper_index_missing",
            "arguments_summary": {"paper_reference": {"arxiv_id": "2401.00001"}},
            "arguments_fingerprint": "sha256:arguments",
        },
        "created_at": "2026-07-13T00:00:00+00:00",
        "expires_at": "2026-07-13T00:10:00+00:00",
    }


def test_prepare_background_work_atomically_consumes_grant_and_ends_interaction(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "agent-work.sqlite"))
    checkpoint = storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="session-1",
        runtime_state={"current_step_id": "index-step"},
        interaction=_interaction(),
        status="waiting_interaction",
    )

    continuation = storage.agent_work.prepare_background_work(
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
        job_id=None,
        job_idempotency_key="2401.00001:docling:paper_qa_index_v1",
        display_summary={"paper_title": "A paper", "question_summary": "核心方法是什么？"},
    )

    stored_checkpoint = storage.agent_runtime_checkpoints.get_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="session-1",
    )
    assert continuation["status"] == "submitting"
    assert stored_checkpoint["status"] == "waiting_background_job"
    assert stored_checkpoint["interaction"] is None
    assert stored_checkpoint["runtime_state"]["background_continuation_id"] == "continuation-1"
    assert storage.approval_grants.get_grant("grant-1")["status"] == "consumed"
    assert storage.approval_grants.get_invocation("invocation-1")["status"] == "prepared"


def test_cancel_one_continuation_does_not_block_shared_job_from_readying_another(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "shared-job.sqlite"))
    for suffix in ("1", "2"):
        interaction = _interaction()
        interaction["interaction_id"] = f"interaction-{suffix}"
        checkpoint = storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
            user_id=f"user-{suffix}",
            session_id=f"session-{suffix}",
            thread_id=f"session-{suffix}",
            runtime_state={"current_step_id": "index-step"},
            interaction=interaction,
            status="waiting_interaction",
        )
        storage.agent_work.prepare_background_work(
            checkpoint_id=checkpoint["checkpoint_id"],
            user_id=f"user-{suffix}",
            session_id=f"session-{suffix}",
            thread_id=f"session-{suffix}",
            interaction_id=f"interaction-{suffix}",
            grant_id=f"grant-{suffix}",
            invocation_id=f"invocation-{suffix}",
            continuation_id=f"continuation-{suffix}",
            plan_id="plan-1",
            step_id="index-step",
            tool_name="parse_and_index_paper",
            arguments_fingerprint="sha256:arguments",
            handler_name="paper_qa_index",
            job_id="job-shared",
            job_idempotency_key="2401.00001:docling:paper_qa_index_v1",
        )

    cancelled = storage.agent_work.cancel_continuation(
        "continuation-1",
        user_id="user-1",
        session_id="session-1",
    )
    ready_count = storage.agent_work.mark_job_ready(
        job_id="job-shared",
        validated_result={"build_id": "build-1", "chunk_count": 3},
    )

    assert cancelled["status"] == "cancelled"
    assert ready_count == 1
    assert storage.agent_work.get_continuation("continuation-1")["status"] == "cancelled"
    assert storage.agent_work.get_continuation("continuation-2")["status"] == "ready_to_resume"
    try:
        storage.agent_work.cancel_continuation(
            "continuation-2",
            user_id="user-1",
            session_id="session-1",
        )
    except AgentWorkConflict as exc:
        assert str(exc) == "continuation_owner_mismatch"
    else:  # pragma: no cover - 所有权缺口会直接破坏多用户隔离。
        raise AssertionError("owner mismatch must be rejected")


def test_resume_run_is_claimed_once_and_persists_full_response_before_completion(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "resume-run.sqlite"))
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

    with storage.connection_provider.connect() as conn:
        conn.execute(
            "UPDATE agent_runtime_checkpoints SET thread_id = ? WHERE checkpoint_id = ?",
            ("other-thread", checkpoint["checkpoint_id"]),
        )
        conn.commit()
    try:
        storage.agent_work.claim_resume_run(
            "continuation-1",
            user_id="user-1",
            session_id="session-1",
        )
    except AgentWorkConflict as exc:
        assert str(exc) == "continuation_checkpoint_owner_mismatch"
    else:  # pragma: no cover - checkpoint 串线会破坏精确恢复的身份边界。
        raise AssertionError("checkpoint owner mismatch must be rejected")
    with storage.connection_provider.connect() as conn:
        conn.execute(
            "UPDATE agent_runtime_checkpoints SET thread_id = ? WHERE checkpoint_id = ?",
            ("session-1", checkpoint["checkpoint_id"]),
        )
        conn.commit()

    first = storage.agent_work.claim_resume_run(
        "continuation-1",
        user_id="user-1",
        session_id="session-1",
    )
    duplicate = storage.agent_work.claim_resume_run(
        "continuation-1",
        user_id="user-1",
        session_id="session-1",
    )
    response = {"session_id": "session-1", "answer": "原问题的完整答案", "papers": []}
    assert storage.agent_work.start_resume_run(first["resume_run_id"])
    completed = storage.agent_work.complete_resume_run(
        first["resume_run_id"],
        final_response=response,
    )

    assert first["created"] is True
    assert duplicate["created"] is False
    assert duplicate["resume_run_id"] == first["resume_run_id"]
    assert completed["status"] == "completed"
    assert completed["final_response"] == response
    assert storage.agent_work.get_continuation("continuation-1")["status"] == "resumed"

    pending_delivery = storage.agent_work.list_unretrieved_resume_continuations(
        user_id="user-1",
        session_id="session-1",
    )
    assert [item["continuation_id"] for item in pending_delivery] == ["continuation-1"]

    # 客户端读取持久化结果后，该 resumed continuation 不应继续出现在后台任务列表。
    assert storage.agent_work.mark_resume_result_retrieved(first["resume_run_id"])
    assert storage.agent_work.list_unretrieved_resume_continuations(
        user_id="user-1",
        session_id="session-1",
    ) == []


def test_ready_continuation_expiry_also_terminates_runtime_checkpoint(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "ready-expiry.sqlite"))
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
    storage.agent_work.mark_job_ready(job_id="job-1", validated_result={"build_id": "build-1"})
    with storage.connection_provider.connect() as conn:
        # 只调整测试时钟输入，行为仍通过公开 store 接口观察。
        conn.execute(
            "UPDATE agent_work_continuations SET expires_at = ? WHERE continuation_id = ?",
            ("2000-01-01T00:00:00+00:00", "continuation-1"),
        )
        conn.commit()

    expired = storage.agent_work.expire_ready_continuation_if_needed("continuation-1")
    stored_checkpoint = storage.agent_runtime_checkpoints.get_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="session-1",
    )

    assert expired["status"] == "expired"
    assert expired["error_code"] == "continuation_ready_expired"
    assert stored_checkpoint["status"] == "expired"
    assert stored_checkpoint["next_route"] == "expired"


def test_clear_session_cancels_recovery_state_but_keeps_audit_record(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "clear-agent-session.sqlite"))
    interaction = _interaction()
    storage.agent_sessions.create_or_get_agent_session(user_id="user-1", session_id="session-1")
    storage.agent_sessions.update_agent_session(
        session_id="session-1",
        user_id="user-1",
        memory_patch={
            "selected_paper": {"arxiv_id": "2401.00001"},
            "paper_qa_result": {"status": "waiting_interaction", "interaction": interaction},
            "last_intent": "paper_qa",
        },
    )
    checkpoint = storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="session-1",
        runtime_state={"current_step_id": "index-step"},
        interaction=interaction,
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

    cancelled_continuations = storage.agent_work.cancel_session_continuations(
        user_id="user-1",
        session_id="session-1",
    )
    cancelled_checkpoints = storage.agent_runtime_checkpoints.cancel_agent_runtime_checkpoints_for_session(
        user_id="user-1",
        session_id="session-1",
    )
    assert storage.agent_sessions.clear_agent_session("session-1", user_id="user-1")

    session = storage.agent_sessions.get_agent_session("session-1", user_id="user-1")
    stored_checkpoint = storage.agent_runtime_checkpoints.get_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="session-1",
    )
    continuation = storage.agent_work.get_continuation("continuation-1")

    assert cancelled_continuations == 1
    assert cancelled_checkpoints == 0
    assert session["status"] == "cleared"
    assert session["selected_paper"] is None
    assert session["paper_qa_result"] is None
    assert session["last_intent"] == ""
    assert stored_checkpoint["status"] == "cancelled"
    # 清空会话只关闭恢复入口，不删除 continuation/job 历史，排障信息仍然可审计。
    assert continuation["status"] == "cancelled"
    assert continuation["job_id"] == "job-1"
