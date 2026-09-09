"""使用真实会话存储和研究图验证生产 QA 边界，不用宽松 mock 补齐不存在的方法。"""

import json
import asyncio
from importlib import import_module
from contextvars import Context
from types import SimpleNamespace

import pytest

from core.errors import AppError, ErrorCode
from services.evaluation import eval_record
from services.paper_qa.paper_qa_service import PaperQAService
from tests.helpers import FakeEmbeddingService, FakeGenerationService, FakeVectorStoreService, build_storage_container
from tests.integration.test_paper_qa_service import _FakeMemoryService
from tests.unit.services.paper_evidence_research.test_research_service import ScriptedDecisionPolicy, _service


@pytest.fixture
def qa_service(tmp_path, monkeypatch):
    storage = build_storage_container(db_path=str(tmp_path / "qa.sqlite"))
    storage.paper_catalog.add_paper({"arxiv_id": "2310.11511", "title": "Self-RAG", "abstract": "Reflection tokens"})
    storage.paper_qa_index.insert_paper_qa_index(
        "2310.11511", collection_name="self_rag", status="indexed", chunk_count=2,
    )
    service = PaperQAService(
        paper_qa_index_store=storage.paper_qa_index, paper_catalog_store=storage.paper_catalog,
        paper_chat_session_store=storage.paper_chat_sessions, paper_qa_turn_store=storage.paper_qa_turns,
        research_profile_store=storage.research_profiles, agent_runtime_checkpoint_store=storage.agent_runtime_checkpoints,
        memory_service=_FakeMemoryService(), embedding_service=FakeEmbeddingService(),
        vector_store_service=FakeVectorStoreService(), generation_service=FakeGenerationService(),
        enhanced_retrieval_service=SimpleNamespace(), qa_index_builder=object(),
        research_service=_service(ScriptedDecisionPolicy()),
    )
    monkeypatch.setattr(eval_record, "EVAL_RECORDS_DIR", tmp_path / "records")
    monkeypatch.setattr(import_module("services.paper_qa.paper_qa_service"), "_write_qa_trace", lambda *args, **kwargs: None)
    return service, tmp_path


def _payload(**overrides):
    return {"question": "How do reflection tokens work?", "user_id": "evaluation-user", **overrides}


@pytest.mark.parametrize("streaming", [False, True])
def test_research_qa_uses_real_session_and_persists_rescorable_record(qa_service, streaming):
    service, tmp_path = qa_service
    if streaming:
        events = list(service.answer_question_with_research_stream("2310.11511", _payload()))
        assert {"draft_created", "claim_verification_completed"} <= {e["event"] for e in events}
        result = events[-1]["data"]
        assert events[-1]["event"] == "done"
    else:
        result = service.answer_question("2310.11511", _payload())
    assert result["outcome"] == "completed"
    assert result["turn_id"]
    assert result["chat_session"]["message_count"] == 2
    assert result["qa_observation"]["answer_quality"] == "grounded"
    assert set(result["cited_source_ids"]) == {"chunk-method", "chunk-ablation"}
    record = json.loads(next((tmp_path / "records").rglob("*.json")).read_text(encoding="utf-8"))
    assert record["run_status"] == "success"
    assert record["answer"] == result["answer"]
    assert record["termination_reason"] == "core_needs_satisfied"
    assert any(e.get("candidates") for e in record["trace_events"])


@pytest.mark.parametrize("streaming", [False, True])
def test_missing_index_preserves_error_code(qa_service, streaming):
    service, tmp_path = qa_service
    with pytest.raises(AppError) as failure:
        if streaming:
            list(service.answer_question_with_research_stream("missing", _payload()))
        else:
            service.answer_question("missing", _payload())
    assert failure.value.code == ErrorCode.QA_INDEX_NOT_FOUND
    record = json.loads(next((tmp_path / "records").rglob("*.json")).read_text(encoding="utf-8"))
    assert record["run_status"] == "error"
    assert record["outcome"] is None


def test_stateless_research_keeps_internal_run_identity(qa_service, monkeypatch):
    service, _ = qa_service
    monkeypatch.setattr(service, "_resolve_chat_session", lambda *args: {})
    result = service.answer_question("2310.11511", _payload())
    assert result["outcome"] == "completed"
    assert not result["turn_id"]
    # 临时研究标识不能冒充已持久化的会话 ID，否则浏览器下次会恢复不存在的会话。
    assert not result["session_id"]
    assert result["persistence_status"] == "not_saved"


def test_persistence_failure_is_not_an_abstention(qa_service, monkeypatch):
    service, tmp_path = qa_service

    def fail(**kwargs):
        raise AppError(ErrorCode.DATABASE_WRITE_FAILED, detail="private server path")

    monkeypatch.setattr(service, "persist_completed_turn", fail)
    with pytest.raises(AppError) as failure:
        service.answer_question("2310.11511", _payload())
    assert failure.value.code == ErrorCode.DATABASE_WRITE_FAILED
    assert "private server path" not in json.dumps(failure.value.to_payload())
    record = json.loads(next((tmp_path / "records").rglob("*.json")).read_text(encoding="utf-8"))
    assert record["run_status"] == "error"
    assert record["outcome"] is None


@pytest.mark.parametrize("outcome,status,decision", [
    ("completed", "passed", "finalize"),
    ("partial", "degraded", "finalize_with_degradation"),
    ("abstained", "passed", "finalize"),
])
def test_agent_respects_bounded_research_terminal_outcome(outcome, status, decision):
    from agents.arxiv_search_agent.tool_adapters.paper_qa import AssessPaperQAQualityAdapter, AssessPaperQAQualityInput

    result = {"status": "success", "outcome": outcome, "answer": "经过校验的答案或证据拒答说明",
              "sources": [] if outcome == "abstained" else [{"source_id": "source-1"}]}
    quality = AssessPaperQAQualityAdapter()._run(AssessPaperQAQualityInput(paper_qa_result=result))
    assert quality.status == status
    assert quality.decision == decision
    assert not quality.repair_required
    assert not quality.repair_optional


async def _consume_sse(service):
    from routers.qa_router import QaRequest, qa_paper_stream

    response = await qa_paper_stream("2310.11511", QaRequest(**_payload()), paper_qa_service=service, generation_service=object())
    chunks = [chunk async for chunk in response.body_iterator]
    events = []
    for chunk in chunks:
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        lines = text.strip().splitlines()
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


def test_router_sse_delivers_verified_final_answer_and_progress(qa_service):
    service, _ = qa_service
    events = asyncio.run(_consume_sse(service))
    assert [kind for kind, _ in events].count("done") == 1
    assert "error" not in [kind for kind, _ in events]
    assert {"retrieval", "draft", "verification", "completed"} <= {
        data["stage"] for kind, data in events if kind == "progress"
    }
    assert events[-1][1]["outcome"] == "completed"
    assert events[-1][1]["answer"]


def test_router_emits_one_sanitized_error_and_records_failure(qa_service):
    service, tmp_path = qa_service

    class FailingResearch:
        def research_stream(self, request, **kwargs):
            yield {"event": "research_started", "arxiv_id": request.arxiv_id}
            raise RuntimeError("secret-provider-body /private/server/path")

    service.research_service = FailingResearch()
    events = asyncio.run(_consume_sse(service))
    errors = [data for kind, data in events if kind == "error"]
    assert len(errors) == 1
    assert errors[0]["status"] == "failed"
    assert errors[0]["code"] == ErrorCode.UNKNOWN_ERROR
    assert errors[0]["qa_observation"]
    assert not any(kind == "done" for kind, _ in events)
    assert "secret-provider-body" not in json.dumps(events)
    assert "/private/server/path" not in json.dumps(events)
    record = json.loads(next((tmp_path / "records").rglob("*.json")).read_text(encoding="utf-8"))
    assert record["run_status"] == "error"
    assert record["outcome"] is None


def test_stream_can_resume_in_fresh_worker_contexts(qa_service):
    service, _ = qa_service
    stream = service.answer_question_with_research_stream("2310.11511", _payload())
    while True:
        try:
            event = Context().run(next, stream)
        except StopIteration as stop:
            assert stop.value["outcome"] == "completed"
            break
        if event["event"] == "done":
            assert event["data"]["usage"]["llm_calls"] == 0


@pytest.mark.parametrize("outcome", ["partial", "abstained"])
def test_tool_observation_keeps_research_terminal_semantics(outcome):
    from agents.arxiv_search_agent.node.tool_node import _derive_observation_details

    details = _derive_observation_details("answer_paper_question", {"ok": True, "data": {
        "status": "success", "outcome": outcome, "answer": "可靠但有限的答案或证据不足说明", "sources": [],
        "qa_observation": {"answer_quality": "insufficient_evidence", "answer_insufficient_evidence": "yes"},
    }})
    assert details["is_sufficient"]
    assert details["next_action_hint"] == ("finalize_with_degradation" if outcome == "partial" else None)
