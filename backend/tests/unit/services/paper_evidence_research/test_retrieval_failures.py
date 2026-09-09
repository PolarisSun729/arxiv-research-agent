"""故障、可靠有限回答和正常空召回必须在真实研究图中保持不同语义。"""

import asyncio
from types import SimpleNamespace

import pytest

from services.evaluation.golden_runner import run_case_with_repeats
from services.evaluation.metrics_report import generate_report
from services.paper_evidence_research import PaperEvidenceResearchError, PaperEvidenceResearchService, ResearchLimits
from services.paper_evidence_research.dependencies.need_orchestrated_retriever import NeedOrchestratedRetriever
from services.paper_evidence_research.graph import _terminal_node
from services.paper_evidence_research.state import EvidenceNeed, PaperEvidenceResearchState
from tests.unit.services.paper_evidence_research.test_research_service import (
    ScriptedClaimExtractor, ScriptedClaimVerifier, ScriptedDraftGenerator,
    ScriptedPolicyFromActions, ScriptedQuestionAnalyzer, _request,
)


def _search(need_id="need-method"):
    return {"action": "search_paper", "target_need_id": need_id, "objective": "discover",
            "retrieval_mode": "method", "query": "reflection tokens", "reason_code": "OPEN_EVIDENCE_NEED"}


def _engine(responses, actions):
    responses = iter(responses)

    def retrieve(**kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return {"chunks": value}

    return PaperEvidenceResearchService(
        question_analyzer=ScriptedQuestionAnalyzer(), decision_policy=ScriptedPolicyFromActions(actions),
        retriever=NeedOrchestratedRetriever(
            retrieval_pipeline=SimpleNamespace(retrieve=retrieve),
            target_resolver=lambda _: {"collection_name": "offline"},
        ),
        draft_generator=ScriptedDraftGenerator(), claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )


def test_irrelevant_candidate_cannot_hide_later_retrieval_failure():
    engine = _engine([[{"chunk_id": "unrelated", "content": "Background only."}], TimeoutError()],
                     [_search(), _search("need-ablation")])
    case = {"case_id": "failure", "arxiv_id": "offline", "question": "Unsupported conclusion?",
            "answerable": False, "expected_chunk_ids": [], "gold_answer": "证据不足"}
    record = asyncio.run(run_case_with_repeats(case, engine, 1))
    report = generate_report([record])
    assert record["raw_runs"][0]["run_status"] == "error"
    assert record["raw_runs"][0]["outcome"] is None
    assert record["metrics"]["three_state_accuracy"] == 0.0
    assert report["run_summary"]["failed_runs"] == 1


def test_abstention_decision_does_not_depend_on_trace_retention():
    state = PaperEvidenceResearchState(
        request=_request(run_id="without-trace"),
        evidence_needs=[EvidenceNeed(need_id="need-method", description="方法", importance="core", status="open")],
        retrieval_failures={"need-method": "retrieval_failed"}, trace_events=[],
    )
    with pytest.raises(PaperEvidenceResearchError, match="未能取得"):
        _terminal_node(state)


def test_successful_retry_clears_fault_for_the_same_need():
    engine = _engine([TimeoutError(), []], [_search(), _search(), {"action": "abstain", "reason_code": "NO_EVIDENCE"}])
    result = engine.research(_request(run_id="recovered", limits=ResearchLimits(max_no_progress=0)))
    assert result.outcome == "abstained"


def test_verified_partial_survives_unresolved_retrieval_failure():
    engine = _engine(
        [[{"chunk_id": "chunk-method", "content": "The reflection token decides whether retrieval is required."}], TimeoutError()],
        [_search(), {"action": "draft_answer", "addressed_need_ids": ["need-method", "need-ablation"],
                     "reason_code": "DISCOVER_GAPS"}, _search("need-ablation")],
    )
    result = engine.research(_request(run_id="reliable-partial"))
    assert result.outcome == "partial"
    assert [citation.source_id for citation in result.citations] == ["chunk-method"]
