"""生成评分覆盖真实引用字段、可回答性缺失和有害修复，不把缺测当满分。"""

from copy import deepcopy

import pytest

from services.evaluation.metrics_generation import (
    calculate_citation_fidelity, calculate_key_coverage_rate, calculate_self_repair_gain,
    calculate_three_state_accuracy, compute_generation_metrics,
)


@pytest.mark.parametrize("points,answer,expected", [
    (["方法流程"], "该方法流程包含两步", 1.0),
    (["方法流程"], "The retrieval pipeline has two stages.", 1.0),
    (["attention mechanism"], "attention and an unrelated mechanism", 0.0),
    (["cat"], "concatenate", 0.0),
    (["transformer", "pooling"], "transformer architecture", 0.5),
    ([], "任意回答", None),
    (["核心结论"], "", 0.0),
])
def test_key_coverage_uses_phrases_and_shared_aliases(points, answer, expected):
    assert calculate_key_coverage_rate(points, answer) == expected


@pytest.mark.parametrize("verdict,evidence_ids,expected", [
    ("supported", ["chunk-1"], 1.0),
    ("citation_mismatch", ["chunk-1"], 0.0),
    ("unsupported", ["chunk-1"], 0.0),
    ("supported", ["other"], 0.0),
])
def test_citation_requires_supported_claim_binding(verdict, evidence_ids, expected):
    citations = [{"source_id": "chunk-1", "claim_ids": ["claim-1"]}]
    assessments = {"claim-1": {"verdict": verdict, "supporting_evidence_ids": evidence_ids}}
    assert calculate_citation_fidelity(citations, {"chunk-1"}, assessments) == expected


def test_every_claim_on_a_citation_must_be_supported():
    citations = [{"source_id": "chunk-1", "claim_ids": ["supported", "missing"]}]
    assessments = {"supported": {"verdict": "supported", "supporting_evidence_ids": ["chunk-1"]}}
    assert calculate_citation_fidelity(citations, {"chunk-1"}, assessments) == 0.0
    assert calculate_citation_fidelity([], set(), {}) is None
    assert calculate_citation_fidelity(citations, {"chunk-1"}) is None


@pytest.mark.parametrize("answerable,outcome,expected", [
    (True, "completed", 1.0), (True, "partial", 1.0), (True, "abstained", 0.0),
    (False, "completed", 0.0), (False, "partial", 0.0), (False, "abstained", 1.0),
    (False, None, 0.0), (None, "abstained", None),
])
def test_answerability_accuracy(answerable, outcome, expected):
    assert calculate_three_state_accuracy(answerable, outcome) == expected


@pytest.mark.parametrize("counts,expected", [([1, 3], 2.0), ([2, 2], 0.0), ([3, 1], -2.0), ([1], 0.0), ([], None)])
def test_repair_gain_compares_verified_support(counts, expected):
    events = [{"event_type": "evidence_coverage_projected", "draft_version": i + 1,
               "supported_claim_count": count} for i, count in enumerate(counts)]
    assert calculate_self_repair_gain(events) == expected


def test_completed_without_citations_is_not_perfect(golden_case, completed_record):
    completed_record["citations"] = []
    assert compute_generation_metrics(golden_case, completed_record)["citation_fidelity"] == 0.0


def test_correct_abstention_has_no_applicable_citation_score(golden_case, completed_record):
    golden_case.update(answerable=False, expected_chunk_ids=[], expected_answer_points=[], gold_answer="")
    completed_record.update(outcome="abstained", citations=[], answer="证据不足")
    metrics = compute_generation_metrics(golden_case, completed_record)
    assert metrics["three_state_accuracy"] == 1.0
    assert metrics["citation_fidelity"] is None


def test_execution_failure_never_earns_correct_abstention(golden_case, completed_record):
    golden_case["answerable"] = False
    completed_record.update(run_status="error", outcome=None, error={"code": "timeout"})
    assert compute_generation_metrics(golden_case, completed_record)["three_state_accuracy"] == 0.0


def test_fidelity_uses_latest_draft_and_matches_gate(golden_case, completed_record):
    record = deepcopy(completed_record)
    event = [e for e in record["trace_events"] if e["event_type"] == "claim_verification_completed"][-1]
    event["claims"][0]["citation_ids"] = ["another-source"]
    # supported 标签与可见主张的原引用不一致时，离线重算也不能将其记为忠实。
    assert compute_generation_metrics(golden_case, record)["citation_fidelity"] < 1.0


def test_missing_verification_is_explicit(golden_case, completed_record):
    completed_record["trace_events"] = []
    assert compute_generation_metrics(golden_case, completed_record)["citation_fidelity"] is None
def test_failed_persistence_with_citations_is_not_perfect(completed_record, golden_case):
    from services.evaluation.scoring import score_record

    completed_record.update(run_status="error", outcome=None, error={"code": "database_write_failed"})
    golden_case["answerable"] = False
    metrics = score_record(golden_case, completed_record)["metrics"]
    assert metrics["three_state_accuracy"] == 0.0
    assert metrics["citation_fidelity"] == 0.0
