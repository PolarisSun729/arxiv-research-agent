"""用真实研究图和结果模型固定模块边界，防止宽松 mock 掩盖评测失真。"""

import asyncio
import json
from unittest.mock import patch

import pytest

from services.evaluation import eval_record, golden_runner, metrics_generation, metrics_retrieval
from tests.unit.services.paper_evidence_research.test_research_service import (
    ScriptedDecisionPolicy,
    _request,
    _service,
)


@pytest.fixture
def completed_research():
    events = []
    service = _service(ScriptedDecisionPolicy())
    service._trace_sink = lambda run_id, trace: events.extend(trace)
    request = _request(run_id="evaluation-contract")
    result = service.research(request)
    return request, result, events


def _case():
    return {
        "case_id": "contract",
        "arxiv_id": "2310.11511",
        "question": "How do reflection tokens work?",
        "expected_answer_points": ["reflection token"],
        "expected_chunk_ids": ["chunk-method", "chunk-ablation"],
        "answerable": True,
        "gold_answer": "Reflection tokens decide whether retrieval is required.",
    }


def test_actual_research_trace_preserves_retrieval_hits(completed_research):
    _, result, events = completed_research
    assert result.outcome == "completed"
    metrics = metrics_retrieval.compute_retrieval_metrics(_case()["expected_chunk_ids"], events, [5])
    assert metrics == {"recall@5": 1.0, "hit@5": 1.0, "mrr": 1.0}


def test_persisted_record_can_be_rescored_with_actual_contract(completed_research, tmp_path):
    request, result, events = completed_research
    with patch.object(eval_record, "EVAL_RECORDS_DIR", tmp_path):
        # 同一份记录同时供线上落盘和离线评分使用，不另造 result 包装层。
        path = eval_record.write_eval_record(result=result, request=request, trace_events=events)
    record = json.loads(__import__("pathlib").Path(path).read_text(encoding="utf-8"))
    assert record["termination_reason"] == "core_needs_satisfied"
    assert record["answer"] == result.answer
    assert record["citations"][0]["source_id"]
    assert record["efficiency"]["llm_calls"] is None
    metrics = metrics_generation.compute_generation_metrics(_case(), record)
    assert metrics["citation_fidelity"] == 1.0
    assert metrics["three_state_accuracy"] == 1.0


def test_runner_preserves_completed_result_and_full_trace():
    record = asyncio.run(golden_runner.run_single_case(_case(), _service(ScriptedDecisionPolicy())))
    assert record["run_status"] == "success"
    assert record["outcome"] == "completed"
    assert record["answer"]
    assert any(e.get("candidates") for e in record["trace_events"])


def test_repair_gain_compares_verified_drafts(completed_research):
    _, _, events = completed_research
    assert metrics_generation.calculate_self_repair_gain(events) == 1.0


def test_chinese_point_and_existing_alias_are_covered():
    point = "\u65b9\u6cd5\u6d41\u7a0b"
    assert metrics_generation.calculate_key_coverage_rate([point], "\u8be5\u8bba\u6587\u7684\u65b9\u6cd5\u6d41\u7a0b\u5305\u62ec\u4e24\u4e2a\u9636\u6bb5") == 1.0
    assert metrics_generation.calculate_key_coverage_rate([point], "The retrieval pipeline has two stages.") == 1.0


def test_stream_delivers_draft_and_verification_progress():
    stream = _service(ScriptedDecisionPolicy()).research_stream(_request(run_id="stream-contract"))
    events = []
    while True:
        try:
            events.append(next(stream)["event"])
        except StopIteration as stop:
            assert stop.value.outcome == "completed"
            break
    assert "draft_created" in events
    assert "claim_verification_completed" in events
