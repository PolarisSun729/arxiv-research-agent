"""落盘契约同时用于在线记录与离线重评分；技术失败与业务终态互斥。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from services.evaluation import eval_record
from services.evaluation.contracts import EvaluationRecord
from services.evaluation.metrics_generation import compute_generation_metrics


@pytest.mark.parametrize("returncode,expected", [(0, "abc123"), (1, "unknown")])
def test_git_version_probe(monkeypatch, returncode, expected):
    monkeypatch.setattr(eval_record.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout="abc123\n"))
    assert eval_record._get_git_commit_hash() == expected


def test_write_and_rescore_actual_result(tmp_path, monkeypatch, research_run, golden_case):
    monkeypatch.setattr(eval_record, "EVAL_RECORDS_DIR", tmp_path)
    request, result, events = research_run
    path = eval_record.write_eval_record(
        request=request, result=result, trace_events=events, raw_question="原始问题", latency_ms=12.34,
    )
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    EvaluationRecord.model_validate(record)
    assert record["query"]["raw"] == "原始问题"
    assert record["query"]["rewritten"] == request.original_question
    assert record["answer"] == result.answer
    assert record["termination_reason"] == result.research_summary.termination_reason
    assert record["efficiency"]["llm_calls"] is None
    assert compute_generation_metrics(golden_case, record)["citation_fidelity"] == 1.0


def test_record_preserves_failure_without_business_outcome(research_run):
    request, result, events = research_run
    record = eval_record.build_eval_record(request=request, result=result, trace_events=events, error={"code": "persist_failed"})
    assert record["run_status"] == "error"
    assert record["outcome"] is None
    with pytest.raises(ValidationError):
        EvaluationRecord.model_validate({**record, "outcome": "abstained"})


def test_bad_record_and_io_failure_are_fail_open(tmp_path, monkeypatch, completed_record):
    assert eval_record.write_eval_record(record={"invalid": True}) is None
    blocked_dir = tmp_path / "a-file"
    blocked_dir.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(eval_record, "EVAL_RECORDS_DIR", blocked_dir)
    assert eval_record.write_eval_record(record=completed_record) is None


def test_run_id_cannot_escape_output_directory(tmp_path, monkeypatch, completed_record):
    monkeypatch.setattr(eval_record, "EVAL_RECORDS_DIR", tmp_path)
    completed_record["trace_ref"] = "../../outside"
    path = Path(eval_record.write_eval_record(record=completed_record))
    assert path.resolve().is_relative_to(tmp_path.resolve())


def test_unknown_intent_is_not_a_classified_other():
    assert eval_record._extract_main_intent(None) == "unknown"
    assert eval_record._extract_main_intent({"intent_profile": {"main_intent": "method_flow"}}) == "method_flow"
