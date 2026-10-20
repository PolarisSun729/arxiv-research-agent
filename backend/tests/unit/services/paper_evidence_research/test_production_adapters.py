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


def test_verifier_settles_uncited_and_visual_claims_without_the_model():
    prompts = []

    def complete(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"assessments": [
            {"claim_id": "claim-1", "verdict": "supported", "supporting_evidence_ids": ["chunk-1"]},
        ]})

    request = _verification_request()
    request.claims += [
        AnswerClaim(claim_id="claim-uncited", text="没挂引用的结论"),
        AnswerClaim(claim_id="claim-visual", text="如图所示曲线上升", citation_ids=["chunk-2"],
                    requires_visual_verification=True),
    ]
    result = ResearchClaimVerifier(SimpleNamespace(complete_with_qwen=complete)).verify(request)

    sent_claims = json.loads(prompts[0].rsplit("\n", 1)[-1])["claims"]
    assert [claim["claim_id"] for claim in sent_claims] == ["claim-1"]
    # 结论按原主张顺序返回，规则判定的主张同样带上缺失项，供修订提示使用。
    assert [(item["claim_id"], item["verdict"], item["missing_facets"]) for item in result["assessments"]] == [
        ("claim-1", "supported", []),
        ("claim-uncited", "unsupported", ["citation_ids"]),
        ("claim-visual", "unsupported", ["visual_verification"]),
    ]


def test_verifier_skips_the_model_when_no_claim_has_citations():
    def complete(prompt, **kwargs):
        raise AssertionError("不应调用模型")

    request = ClaimVerificationRequest(
        research_question="方法是什么", draft_version=1,
        claims=[AnswerClaim(claim_id="claim-1", text="没挂引用的结论")],
        context_pack=EvidenceContextPack(candidates=[EvidenceCandidate(candidate_id="chunk-1", content="证据一")]),
    )
    result = ResearchClaimVerifier(SimpleNamespace(complete_with_qwen=complete)).verify(request)

    assert result["assessments"][0]["verdict"] == "unsupported"


@pytest.mark.parametrize("response", ['invalid JSON', '{"assessments": []}'])
def test_unavailable_verification_is_a_runtime_failure(response):
    provider = SimpleNamespace(complete_with_qwen=lambda *args, **kwargs: response)
    with pytest.raises(PaperEvidenceResearchError) as failure:
        ResearchClaimVerifier(provider).verify(_verification_request())
    assert failure.value.code == "research_verification_failed"
    # 重试用完才报错，错误原因要写进 detail，否则轨迹里看不出是哪一步坏的。
    assert failure.value.detail["attempts"] == 3 and failure.value.detail["error"]


def test_invalid_verification_output_is_retried_with_feedback():
    prompts = []
    replies = iter([
        json.dumps({"assessments": []}),
        json.dumps({"assessments": [{"claim_id": "claim-1", "verdict": "supported", "supporting_evidence_ids": ["chunk-1"]}]}),
    ])

    def complete(prompt, **kwargs):
        prompts.append(prompt)
        return next(replies)

    result = ResearchClaimVerifier(SimpleNamespace(complete_with_qwen=complete)).verify(_verification_request())

    assert result["assessments"][0]["verdict"] == "supported"
    assert "claim-1" in prompts[1].split("上一次输出被拒绝：", 1)[1]


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



def test_draft_generator_asks_for_per_sentence_citations_and_passes_revision_notes():
    from services.paper_evidence_research.dependencies.production_adapters import ResearchDraftGenerator
    from services.paper_evidence_research.state import DraftGenerationRequest, DraftRevisionNote, EvidenceContextPack

    captured = {}

    class CapturingAnswerGenerator:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"answer": "改挂后的句子 [source:54]。", "cited_source_ids": ["54"], "citation_debug": {}}

    drafter = ResearchDraftGenerator(SimpleNamespace())
    drafter._generator = CapturingAnswerGenerator()
    base = dict(
        research_question="方法流程是什么",
        addressed_need_ids=["need-a"],
        context_pack=EvidenceContextPack(candidates=[{"candidate_id": "54", "content": "Figure 2"}]),
    )

    drafter.generate(DraftGenerationRequest(version=1, **base))
    assert captured["original_question"] == "方法流程是什么"
    assert "句末单独挂上" in captured["generation_question"]
    assert "修订要求" not in captured["generation_question"]

    drafter.generate(
        DraftGenerationRequest(
            version=2,
            previous_answer="上一版答案正文",
            revision_notes=[
                DraftRevisionNote(
                    claim_text="系统处理原始对话", reason_code="citation_mismatch", reason="挂的引用不对",
                    suggested_citation_ids=["54"], missing_facets=["具体步骤"],
                )
            ],
            **base,
        )
    )
    question = captured["generation_question"]
    assert "修订要求" in question and "上一版答案正文" in question
    assert "「系统处理原始对话」——挂的引用不对" in question
    assert "[source:54]" in question and "具体步骤" in question
