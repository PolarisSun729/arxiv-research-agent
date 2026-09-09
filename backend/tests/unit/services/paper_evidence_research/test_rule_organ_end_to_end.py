from __future__ import annotations

from typing import Any

from services.paper_evidence_research import PaperEvidenceResearchRequest, ResearchLimits
from services.paper_evidence_research.dependencies import (
    DegradationLedger,
    RuleClaimExtractor,
    RuleDecisionPolicy,
    build_paper_evidence_research_service,
)
from services.paper_evidence_research.module import PaperEvidenceResearchService


class StaticRetriever:
    """按需求返回固定候选，模拟两次检索各命中一个需求。"""

    def retrieve(self, action: Any, _state: Any) -> dict[str, Any]:
        if action.target_need_id.endswith("method"):
            candidates = [
                {
                    "content": "The reflection token decides whether retrieval is required.",
                    "chunk_id": "chunk-method",
                    "chunk_type": "text",
                    "section_path": "Method",
                }
            ]
        else:
            candidates = [
                {
                    "content": "Without reflection tokens the score drops from 54.1 to 50.3.",
                    "chunk_id": "chunk-ablation",
                    "chunk_type": "table",
                    "section_path": "Experiments/Ablation",
                }
            ]
        return {"status": "completed", "candidates": candidates}


_CITED_SENTENCES = {
    "chunk-method": "反思 token 决定是否执行检索 [source:chunk-method]；",
    "chunk-ablation": "移除反思模块后分数从 54.1 降至 50.3 [source:chunk-ablation]。",
}


class CitingDraftGenerator:
    """答案只为上下文包里实际存在的候选写 [source:id] 引用，满足完成门禁的引用契约。"""

    def generate(self, request: Any) -> dict[str, Any]:
        available_ids = {candidate.candidate_id for candidate in request.candidates}
        parts = [sentence for chunk_id, sentence in _CITED_SENTENCES.items() if chunk_id in available_ids]
        answer = "".join(parts)
        return {
            "answer": answer,
            "declared_claims": [],
            "used_candidate_ids": sorted(available_ids),
        }


class EchoClaimVerifier:
    """把主张自报的引用原样判为 supported，模拟一个理想校验器。"""

    def verify(self, request: Any) -> dict[str, Any]:
        return {
            "assessments": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supported",
                    "supporting_evidence_ids": list(claim.citation_ids),
                    "missing_facets": [],
                }
                for claim in request.claims
            ]
        }


class FailingGenerationService:
    def complete_with_qwen(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("llm unavailable")


class TwoCoreNeedAnalyzer:
    """脚本分析器：给出两个核心需求，让规则决策策略完整走"两轮检索后起草"。"""

    def analyze(self, _request: Any) -> dict[str, Any]:
        return {
            "research_question": "方法机制与消融结论",
            "evidence_needs": [
                {
                    "need_id": "need-method",
                    "description": "反思 token 如何决定检索",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                },
                {
                    "need_id": "need-ablation",
                    "description": "移除反思模块的消融结果",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                },
            ],
        }


def _request(question: str = "这篇论文的方法机制和消融结论是什么？") -> PaperEvidenceResearchRequest:
    return PaperEvidenceResearchRequest(
        arxiv_id="2401.00001",
        original_question=question,
        user_id="u",
        session_id="s",
        research_run_id="run-e2e",
        limits=ResearchLimits(max_retrievals=3, max_draft_attempts=2),
    )


def _rule_only_service() -> PaperEvidenceResearchService:
    return PaperEvidenceResearchService(
        question_analyzer=TwoCoreNeedAnalyzer(),
        decision_policy=RuleDecisionPolicy(),
        retriever=StaticRetriever(),
        draft_generator=CitingDraftGenerator(),
        claim_extractor=RuleClaimExtractor(),
        claim_verifier=EchoClaimVerifier(),
    )


def test_rule_organs_drive_graph_to_completed_outcome() -> None:
    result = _rule_only_service().research(_request())
    assert result.outcome == "completed"
    assert result.research_summary.retrieval_count == 2
    assert result.research_summary.draft_attempt_count == 1
    assert result.research_summary.satisfied_need_count >= 1
    assert {citation.source_id for citation in result.citations} == {"chunk-method", "chunk-ablation"}


def test_factory_llm_failure_degrades_and_still_completes() -> None:
    ledger = DegradationLedger()
    service = build_paper_evidence_research_service(
        generation_service=FailingGenerationService(),
        retriever=StaticRetriever(),
        draft_generator=CitingDraftGenerator(),
        claim_verifier=EchoClaimVerifier(),
        degradation_listener=ledger,
    )
    result = service.research(_request())
    assert result.outcome == "completed"
    assert ledger.events, "LLM 全程失败必须留下降级事件"
    assert {event.organ for event in ledger.events} <= {"question_analyzer", "decision_policy"}
