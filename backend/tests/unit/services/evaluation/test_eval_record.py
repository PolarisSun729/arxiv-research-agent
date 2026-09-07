"""评测记录落盘模块单元测试。"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.evaluation.eval_record import (
    EVAL_RECORDS_DIR,
    _build_citations_payload,
    _build_research_summary_payload,
    _extract_main_intent,
    _get_git_commit_hash,
    write_eval_record,
)


def test_get_git_commit_hash_success():
    """测试成功获取 git commit hash。"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="b088ecd\n")
        result = _get_git_commit_hash()
        assert result == "b088ecd"


def test_get_git_commit_hash_failure():
    """测试 git 命令失败时返回 unknown。"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _get_git_commit_hash()
        assert result == "unknown"


def test_extract_main_intent_from_debug():
    """测试从 retrieval_debug 提取 main_intent。"""
    retrieval_debug = {
        "intent_profile": {
            "main_intent": "method_flow",
            "confidence": 0.9,
        }
    }
    assert _extract_main_intent(retrieval_debug) == "method_flow"


def test_extract_main_intent_no_debug():
    """测试无 retrieval_debug 时返回 other。"""
    assert _extract_main_intent(None) == "other"
    assert _extract_main_intent({}) == "other"


def test_build_citations_payload():
    """测试从研究结果提取 citations。"""
    citation1 = MagicMock()
    citation1.citation_id = "1"
    citation1.source_id = "chunk-1"
    citation1.claim_ids = ["claim-a", "claim-b"]

    citation2 = MagicMock()
    citation2.citation_id = "2"
    citation2.source_id = "chunk-2"
    citation2.claim_ids = ["claim-c"]

    result = MagicMock()
    result.citations = [citation1, citation2]

    payload = _build_citations_payload(result)

    assert len(payload) == 2
    assert payload[0]["citation_id"] == "1"
    assert payload[0]["chunk_id"] == "chunk-1"
    assert payload[0]["claim_ids"] == ["claim-a", "claim-b"]
    assert payload[1]["citation_id"] == "2"
    assert payload[1]["chunk_id"] == "chunk-2"


def test_build_research_summary_payload():
    """测试从研究结果提取 research_summary。"""
    summary = MagicMock()
    summary.retrieval_count = 3
    summary.draft_attempt_count = 2
    summary.evidence_pool_size = 15
    summary.supported_claim_count = 8
    summary.citation_repair_count = 1
    summary.unresolved_topics = ["limitation"]

    result = MagicMock()
    result.research_summary = summary

    payload = _build_research_summary_payload(result)

    assert payload["retrieval_count"] == 3
    assert payload["draft_attempt_count"] == 2
    assert payload["evidence_pool_size"] == 15
    assert payload["supported_claim_count"] == 8
    assert payload["citation_repair_count"] == 1
    assert payload["unresolved_topics"] == ["limitation"]


def test_write_eval_record_success(monkeypatch):
    """测试成功写入 eval record。"""
    import tempfile
    # 设置临时目录
    tmp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr("services.evaluation.eval_record.EVAL_RECORDS_DIR", tmp_dir)

    # 构造 mock 对象
    request = MagicMock()
    request.research_run_id = "research-run-test-001"
    request.user_id = "user-123"
    request.session_id = "session-456"
    request.arxiv_id = "2401.00001"
    request.original_question = "论文的训练方法是什么？"

    summary = MagicMock()
    summary.retrieval_count = 2
    summary.draft_attempt_count = 1
    summary.evidence_pool_size = 10
    summary.supported_claim_count = 5
    summary.citation_repair_count = 0
    summary.unresolved_topics = []
    summary.termination_reason = "ALL_NEEDS_RESOLVED"
    summary.model_dump = lambda: {
        "retrieval_count": 2,
        "draft_attempt_count": 1,
        "evidence_pool_size": 10,
        "supported_claim_count": 5,
        "citation_repair_count": 0,
        "unresolved_topics": [],
    }

    citation = MagicMock()
    citation.source_id = "chunk-1"
    citation.claim_ids = ["claim-a"]

    result = MagicMock()
    result.outcome = "completed"
    result.citations = [citation]
    result.research_summary = summary
    result.termination_reason = "ALL_NEEDS_RESOLVED"

    # 写入
    with patch("services.evaluation.eval_record._get_git_commit_hash", return_value="b088ecd"):
        output_path = write_eval_record(
            result=result,
            request=request,
            turn_id="turn-789",
            retrieval_debug=None,
            latency_ms=3500.5,
        )

    assert output_path is not None
    output_file = Path(output_path)
    assert output_file.exists()

    # 验证内容
    content = json.loads(output_file.read_text(encoding="utf-8"))
    assert content["record_id"] == "eval-research-run-test-001"
    assert content["app_version"] == "b088ecd"
    assert content["user_id"] == "user-123"
    assert content["session_id"] == "session-456"
    assert content["turn_id"] == "turn-789"
    assert content["paper_context"]["arxiv_id"] == "2401.00001"
    assert content["query"]["raw"] == "论文的训练方法是什么？"
    assert content["query"]["main_intent"] == "other"
    assert content["outcome"] == "completed"
    assert content["termination_reason"] == "ALL_NEEDS_RESOLVED"
    assert len(content["citations"]) == 1
    assert content["efficiency"]["latency_ms"] == 3500.5
    assert content["efficiency"]["retrieval_count"] == 2
    assert content["trace_ref"] == "research-run-test-001"


def test_write_eval_record_failure_no_crash():
    """测试写入失败时不抛异常（容错）。"""
    request = MagicMock()
    request.research_run_id = "test-run"

    result = MagicMock()
    result.outcome = "completed"
    result.citations = []
    result.research_summary = None  # 故意触发错误

    # 应该捕获异常并返回 None
    output_path = write_eval_record(
        result=result,
        request=request,
        turn_id=None,
        retrieval_debug=None,
        latency_ms=1000.0,
    )

    assert output_path is None  # 写入失败但不崩溃
