import pytest

from agents.arxiv_search_agent.execution.approvals import ApprovalGrant
from services.storage.sqlite.stores.approval_grants import ApprovalGrantConflict
from tests.helpers.sqlite import build_storage_container


def _grant() -> ApprovalGrant:
    return ApprovalGrant(
        grant_id="grant-1",
        interaction_id="interaction-1",
        user_id="user-1",
        session_id="session-1",
        thread_id="thread-1",
        plan_id="plan-1",
        step_id="step-1",
        tool_name="parse_and_index_paper",
        arguments_fingerprint="sha256:arguments-1",
        status="approved",
        approved_at="2026-07-13T00:00:00+00:00",
    )


def test_consume_grant_and_prepare_invocation_is_atomic(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "approval.sqlite"))
    store = storage.approval_grants
    store.create_grant(_grant())

    invocation = store.consume_and_prepare_invocation(
        grant_id="grant-1",
        invocation_id="invocation-1",
        plan_id="plan-1",
        step_id="step-1",
        tool_name="parse_and_index_paper",
        arguments_fingerprint="sha256:arguments-1",
    )

    assert invocation["status"] == "prepared"
    assert invocation["grant_id"] == "grant-1"
    assert store.get_grant("grant-1")["status"] == "consumed"

    with pytest.raises(ApprovalGrantConflict, match="grant_not_approved"):
        store.consume_and_prepare_invocation(
            grant_id="grant-1",
            invocation_id="invocation-2",
            plan_id="plan-1",
            step_id="step-1",
            tool_name="parse_and_index_paper",
            arguments_fingerprint="sha256:arguments-1",
        )


def test_argument_mismatch_does_not_consume_grant_or_create_invocation(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "approval.sqlite"))
    store = storage.approval_grants
    store.create_grant(_grant())

    with pytest.raises(ApprovalGrantConflict, match="grant_identity_mismatch"):
        store.consume_and_prepare_invocation(
            grant_id="grant-1",
            invocation_id="invocation-1",
            plan_id="plan-1",
            step_id="step-1",
            tool_name="parse_and_index_paper",
            arguments_fingerprint="sha256:changed-arguments",
        )

    assert store.get_grant("grant-1")["status"] == "approved"
    assert store.get_invocation("invocation-1") is None
