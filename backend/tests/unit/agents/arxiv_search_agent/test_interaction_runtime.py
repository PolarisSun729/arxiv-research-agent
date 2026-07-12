from datetime import datetime, timedelta, timezone

import pytest

from agents.arxiv_search_agent.execution.interaction_runtime import InteractionRuntimeError, InteractionRuntimeService
from agents.arxiv_search_agent.execution.interactions import (
    AgentInteraction,
    InteractionResumeRequest,
    SideEffectApprovalPayload,
    TargetSelectionPayload,
)
from tests.helpers.sqlite import build_storage_container


def _interaction() -> AgentInteraction:
    now = datetime.now(timezone.utc)
    return AgentInteraction(
        interaction_id="interaction-1",
        kind="side_effect_approval",
        plan_id="plan-1",
        step_id="step-1",
        payload=SideEffectApprovalPayload(
            tool_name="parse_and_index_paper",
            action_type="parse_and_index",
            reason="需要建立论文问答索引",
            arguments_summary={"arxiv_id": "2401.00001"},
            arguments_fingerprint="sha256:arguments-1",
        ),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=10)).isoformat(),
    )


def test_approve_interaction_atomically_creates_grant_and_clears_checkpoint(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "runtime.sqlite"))
    checkpoint = storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        interaction=_interaction().model_dump(mode="json"),
        schema_version=2,
        status="waiting_interaction",
    )
    service = InteractionRuntimeService(
        checkpoint_store=storage.agent_runtime_checkpoints,
        approval_store=storage.approval_grants,
    )

    result = service.resolve(
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        request=InteractionResumeRequest(interaction_id="interaction-1", decision="approve"),
    )

    assert result["decision"] == "approve"
    assert storage.approval_grants.get_grant(result["grant_id"])["arguments_fingerprint"] == "sha256:arguments-1"
    restored = storage.agent_runtime_checkpoints.get_agent_runtime_checkpoint(
        user_id="user-1", session_id="session-1", thread_id="thread-1"
    )
    assert restored["interaction"] is None
    assert restored["status"] == "running"
    assert checkpoint["checkpoint_id"] == restored["checkpoint_id"]


def test_old_checkpoint_schema_is_rejected_without_consuming_state(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "runtime.sqlite"))
    storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        interaction=_interaction().model_dump(mode="json"),
        schema_version=1,
        status="waiting_interaction",
    )
    service = InteractionRuntimeService(
        checkpoint_store=storage.agent_runtime_checkpoints,
        approval_store=storage.approval_grants,
    )

    with pytest.raises(InteractionRuntimeError, match="checkpoint_schema_outdated"):
        service.resolve(
            user_id="user-1",
            session_id="session-1",
            thread_id="thread-1",
            request=InteractionResumeRequest(interaction_id="interaction-1", decision="approve"),
        )


def test_target_selection_only_accepts_checkpoint_candidate(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "runtime.sqlite"))
    now = datetime.now(timezone.utc)
    interaction = AgentInteraction(
        interaction_id="target-1",
        kind="target_selection",
        plan_id="plan-1",
        step_id="resolve-paper",
        payload=TargetSelectionPayload(
            candidates=[
                {"candidate_id": "paper-1", "arxiv_id": "2401.00001", "title": "First"},
                {"candidate_id": "paper-2", "arxiv_id": "2401.00002", "title": "Second"},
            ]
        ),
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=10)).isoformat(),
    )
    storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        interaction=interaction.model_dump(mode="json"),
        schema_version=2,
        status="waiting_interaction",
    )
    service = InteractionRuntimeService(
        checkpoint_store=storage.agent_runtime_checkpoints,
        approval_store=storage.approval_grants,
    )

    result = service.resolve(
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        request=InteractionResumeRequest(
            interaction_id="target-1",
            decision="select",
            response={"candidate_id": "paper-2"},
        ),
    )

    assert result["selected_candidate"]["arxiv_id"] == "2401.00002"
    assert storage.approval_grants.find_approved_grant(
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        plan_id="plan-1",
        step_id="resolve-paper",
        tool_name="resolve_paper",
        arguments_fingerprint="unused",
    ) is None
