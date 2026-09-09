"""生产依赖完整组装的离线贯通检查；仅替换 SDK/检索 I/O，不伪造研究结果模型。"""

import json
from types import SimpleNamespace

import pytest

from services.paper_evidence_research import PaperEvidenceResearchError
from services.paper_evidence_research.dependencies.production_adapters import ResearchClaimVerifier
from services.paper_evidence_research.dependencies.rule_claim_extractor import RuleClaimExtractor
from services.paper_evidence_research.state import (
    AnswerClaim, ClaimExtractionRequest, ClaimVerificationRequest, EvidenceCandidate, EvidenceContextPack,
)
from tests.unit.services.paper_evidence_research.test_research_service import _request


class ScriptedGeneration:
    def __init__(self):
        self.calls = []
        self.actions = iter([
            {"action": "search_paper", "target_need_id": "need-method", "objective": "discover",
             "retrieval_mode": "method", "query": "retrieval decision", "reason_code": "OPEN_EVIDENCE_NEED"},
            {"action": "draft_answer", "addressed_need_ids": ["need-method"], "reason_code": "DRAFT_NEEDED"},
            {"action": "finalize_answer", "reason_code": "CORE_CLAIMS_SUPPORTED"},
        ])

    def generate(self, *, provider, query, search_results, **kwargs):
        self.calls.append((provider, query, search_results))
        return {"response": "反思 token 决定是否执行检索 [source:chunk-method]。"}

    def complete_with_qwen(self, prompt, *, task_type):
        if task_type == "research_question_analysis":
            return json.dumps({"research_question": "方法如何决定是否检索", "evidence_needs": [
                {"need_id": "need-method", "description": "方法如何决定是否检索", "importance": "core"},
            ]})
        if task_type == "research_decision":
            return json.dumps(next(self.actions))
        if task_type == "research_claim_verification":
            claims = json.loads(prompt.rsplit("\n", 1)[-1])["claims"]
            return json.dumps({"assessments": [
                {"claim_id": c["claim_id"], "verdict": "supported" if c["citation_ids"] else "unsupported",
                 "supporting_evidence_ids": c["citation_ids"]} for c in claims
            ]})
        raise AssertionError(task_type)


class Pipeline:
    def __init__(self):
        self.calls = []

    def retrieve(self, *, user_query, collection_name, paper_context, options):
        self.calls.append((user_query, collection_name, paper_context, options))
        return {
            "chunks": [{"chunk_id": "chunk-method", "content": "The reflection token decides whether retrieval is required.",
                        "chunk_type": "text", "section_path": "Method", "page_number": 2}],
            "debug": {"intent_profile": {"main_intent": "method_flow"}},
        }


def test_production_dependency_factory_completes_real_research(monkeypatch):
    import dependencies
    from services.paper_evidence_research.dependencies import research_trace

    generation, pipeline, events = ScriptedGeneration(), Pipeline(), []
    monkeypatch.setattr(dependencies, "get_generation_service", lambda: generation)
    monkeypatch.setattr(dependencies, "get_enhanced_retrieval_service", lambda: SimpleNamespace(retrieval_pipeline=pipeline))
    monkeypatch.setattr(dependencies, "get_paper_qa_index_store", lambda: SimpleNamespace(
        get_paper_qa_index=lambda arxiv_id: {"status": "indexed", "collection_name": "offline-paper", "active_index_version": 2},
    ))
    monkeypatch.setattr(dependencies, "get_paper_catalog_store", lambda: SimpleNamespace(get_paper=lambda arxiv_id: {"title": "Self-RAG"}))
    monkeypatch.setattr(research_trace, "ResearchTraceRecorder", lambda **kwargs: lambda run_id, trace: events.extend(trace))
    dependencies.get_paper_evidence_research_service.cache_clear()
    try:
        result = dependencies.get_paper_evidence_research_service().research(_request(run_id="production-wiring"))
    finally:
        dependencies.get_paper_evidence_research_service.cache_clear()
    assert result.outcome == "completed"
    assert result.citations[0].source_id == "chunk-method"
    assert pipeline.calls[0][1] == "offline-paper"
    assert pipeline.calls[0][2]["active_index_version"] == 2
    assert generation.calls[0][0] == "qwen"
    assert next(e for e in events if e["event_type"] == "retrieval_completed")["main_intent"] == "method_flow"


def _verification_request():
    return ClaimVerificationRequest(
        research_question="方法是什么", draft_version=1,
        claims=[AnswerClaim(claim_id="claim-1", text="一个可校验的主张", citation_ids=["chunk-1"])],
        context_pack=EvidenceContextPack(candidates=[
            EvidenceCandidate(candidate_id="chunk-1", content="证据一"),
            EvidenceCandidate(candidate_id="chunk-2", content="证据二"),
        ]),
    )


def test_verifier_rejects_supported_label_with_mismatched_citation():
    provider = SimpleNamespace(complete_with_qwen=lambda *args, **kwargs: json.dumps({"assessments": [
        {"claim_id": "claim-1", "verdict": "supported", "supporting_evidence_ids": ["chunk-2"]},
    ]}))
    result = ResearchClaimVerifier(provider).verify(_verification_request())
    assert result["assessments"][0]["verdict"] == "citation_mismatch"


@pytest.mark.parametrize("response", ['invalid JSON', '{"assessments": []}'])
def test_unavailable_verification_is_a_runtime_failure(response):
    provider = SimpleNamespace(complete_with_qwen=lambda *args, **kwargs: response)
    with pytest.raises(PaperEvidenceResearchError) as failure:
        ResearchClaimVerifier(provider).verify(_verification_request())
    assert failure.value.code == "research_verification_failed"


def test_uncited_visible_claims_cannot_disappear_before_verification():
    request = ClaimExtractionRequest(
        research_question="方法和结果", draft_version=1,
        answer="方法先检索相关论文 [source:chunk-1]。准确率提升了 90%。",
    )
    claims = RuleClaimExtractor().extract(request)["claims"]
    assert len(claims) == 2
    assert claims[1]["citation_ids"] == []


def test_sentence_final_citation_stays_with_the_preceding_claim():
    request = ClaimExtractionRequest(
        research_question="方法和结果", draft_version=1,
        answer="检索提升准确率。[source:chunk-1] 未引用的第二个事实。",
    )
    claims = RuleClaimExtractor().extract(request)["claims"]
    assert "检索提升准确率" in claims[0]["text"]
    assert claims[0]["citation_ids"] == ["chunk-1"]
    assert claims[1]["citation_ids"] == []
