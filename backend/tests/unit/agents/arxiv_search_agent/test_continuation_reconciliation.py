from __future__ import annotations

from types import SimpleNamespace

from tests.helpers.agent_runtime import load_agent_test_modules


load_agent_test_modules()

from backend.agents.arxiv_search_agent.execution.continuations import AgentWorkContinuationService


class _Handler:
    def __init__(self, *, preflight_status: str = "missing", submit_error: Exception | None = None) -> None:
        self.preflight_status = preflight_status
        self.submit_error = submit_error

    def preflight(self, _state):
        return SimpleNamespace(
            status=self.preflight_status,
            projected_result={"build_id": "build-existing"} if self.preflight_status == "already_satisfied" else None,
            error_code="preflight_failed" if self.preflight_status == "failed" else None,
        )

    def submit_or_attach(self, _state, _preflight):
        if self.submit_error:
            raise self.submit_error
        return SimpleNamespace(job_id="job-new", attached=False)

    def project_result(self, job):
        return {"build_id": job["result"]["build_id"]}


class _Registry:
    def __init__(self, handler: _Handler) -> None:
        self.handler = handler

    def get(self, _name: str):
        return self.handler


class _Store:
    def __init__(self, continuation: dict) -> None:
        self.continuation = dict(continuation)
        self.ready_calls = []
        self.failed_calls = []
        self.requested_statuses = []

    def expire_ready_continuation_if_needed(self, _continuation_id: str):
        return dict(self.continuation)

    def attach_job(self, continuation_id: str, *, job_id: str, attached: bool):
        assert continuation_id == self.continuation["continuation_id"]
        self.continuation.update(status="waiting_job", job_id=job_id, attached=attached)
        return dict(self.continuation)

    def mark_continuation_ready(self, continuation_id: str, *, validated_result: dict):
        assert continuation_id == self.continuation["continuation_id"]
        self.continuation.update(status="ready_to_resume", validated_result=validated_result)
        return dict(self.continuation)

    def mark_job_ready(self, *, job_id: str, validated_result: dict):
        self.ready_calls.append((job_id, validated_result))
        self.continuation.update(status="ready_to_resume", validated_result=validated_result)

    def mark_job_failed(self, *, job_id: str, error_code: str, error_message: str):
        self.failed_calls.append((job_id, error_code, error_message))
        self.continuation.update(status="failed", error_code=error_code, error_message=error_message)

    def get_continuation(self, _continuation_id: str):
        return dict(self.continuation)

    def list_continuations(self, *, user_id: str, session_id=None, statuses=None, limit: int = 100):
        self.requested_statuses = list(statuses or [])
        return [dict(self.continuation)] if self.continuation.get("status") in self.requested_statuses else []

    def list_unretrieved_resume_continuations(self, **_kwargs):
        return []


class _PaperJobStore:
    def __init__(self, job: dict | None = None) -> None:
        self.job = job

    def get_paper_index_job(self, _job_id: str):
        return dict(self.job) if self.job else None


def _continuation(**overrides) -> dict:
    return {
        "continuation_id": "continuation-1",
        "handler_name": "paper_qa_index",
        "handler_state": {"arxiv_id": "2401.00001"},
        "status": "submitting",
        "job_id": None,
        **overrides,
    }


def test_reconcile_repairs_submitting_continuation_by_attaching_persistent_job() -> None:
    store = _Store(_continuation())
    service = AgentWorkContinuationService(
        store=store,
        paper_job_store=_PaperJobStore(),
        handlers=_Registry(_Handler(preflight_status="missing")),
    )

    reconciled = service.reconcile(store.continuation)

    assert reconciled["status"] == "waiting_job"
    assert reconciled["job_id"] == "job-new"


def test_reconcile_keeps_submitting_state_when_job_submission_is_temporarily_unavailable() -> None:
    store = _Store(_continuation())
    service = AgentWorkContinuationService(
        store=store,
        paper_job_store=_PaperJobStore(),
        handlers=_Registry(_Handler(submit_error=RuntimeError("worker unavailable"))),
    )

    reconciled = service.reconcile(store.continuation)

    assert reconciled["status"] == "submitting"
    assert reconciled["job_id"] is None


def test_reconcile_projects_validated_success_and_explicit_failure_from_persistent_job() -> None:
    success_store = _Store(_continuation(status="waiting_job", job_id="job-1"))
    success_service = AgentWorkContinuationService(
        store=success_store,
        paper_job_store=_PaperJobStore({"job_id": "job-1", "status": "success", "result": {"build_id": "build-1"}}),
        handlers=_Registry(_Handler()),
    )

    success = success_service.reconcile(success_store.continuation)

    assert success["status"] == "ready_to_resume"
    assert success_store.ready_calls == [("job-1", {"build_id": "build-1"})]

    failed_store = _Store(_continuation(status="waiting_job", job_id="job-2"))
    failed_service = AgentWorkContinuationService(
        store=failed_store,
        paper_job_store=_PaperJobStore({
            "job_id": "job-2",
            "status": "failed",
            "failure_code": "pdf_download_failed",
            "error_message": "PDF 下载失败",
        }),
        handlers=_Registry(_Handler()),
    )

    failed = failed_service.reconcile(failed_store.continuation)

    assert failed["status"] == "failed"
    assert failed_store.failed_calls == [("job-2", "pdf_download_failed", "PDF 下载失败")]


def test_list_active_does_not_restore_failed_continuation() -> None:
    store = _Store(_continuation(status="failed", job_id="job-1"))
    service = AgentWorkContinuationService(
        store=store,
        paper_job_store=_PaperJobStore({"job_id": "job-1", "status": "failed"}),
        handlers=_Registry(_Handler()),
    )

    items = service.list_active(user_id="user-1", session_id="session-1")

    # 明确失败是终态，只保留审计记录，不能再作为新页面可恢复的活动任务返回。
    assert items == []
    assert "failed" not in store.requested_statuses
