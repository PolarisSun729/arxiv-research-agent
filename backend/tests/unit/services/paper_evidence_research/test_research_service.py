from __future__ import annotations

from typing import Any

import pytest

from services.paper_evidence_research import (
    PaperEvidenceResearchError,
    PaperEvidenceResearchRequest,
    PaperEvidenceResearchService,
    ResearchLimits,
)


class ScriptedQuestionAnalyzer:
    def analyze(self, request: Any) -> dict[str, Any]:
        return {
            "research_question": request.original_question,
            "evidence_needs": [
                {
                    "need_id": "need-method",
                    "description": "方法如何决定是否执行检索",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                },
                {
                    "need_id": "need-ablation",
                    "description": "消融实验是否支持反思模块有效",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                },
            ],
        }


class ScriptedDecisionPolicy:
    def __init__(self) -> None:
        self._actions = iter(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "reflection token retrieval decision mechanism",
                    "section_hints": ["Method"],
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {
                    "action": "draft_answer",
                    "addressed_need_ids": ["need-method", "need-ablation"],
                    "reason_code": "DRAFT_NEEDED_TO_DISCOVER_GAPS",
                },
                {
                    "action": "search_paper",
                    "target_need_id": "need-ablation",
                    "target_claim_id": "claim-ablation",
                    "objective": "verify_claim",
                    "retrieval_mode": "experiment",
                    "query": "reflection token versus without reflection token ablation results",
                    "section_hints": ["Ablation", "Experiments"],
                    "reason_code": "UNSUPPORTED_CORE_CLAIM",
                },
                {
                    "action": "draft_answer",
                    "addressed_need_ids": ["need-method", "need-ablation"],
                    "reason_code": "REVISE_AFTER_VERIFICATION",
                },
                {
                    "action": "finalize_answer",
                    "reason_code": "CORE_CLAIMS_SUPPORTED",
                },
            ]
        )

    def decide(self, _context: Any) -> dict[str, Any]:
        return next(self._actions)


class ScriptedPolicyFromActions:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._actions = iter(actions)

    def decide(self, _context: Any) -> dict[str, Any]:
        return next(self._actions)


class ScriptedRetriever:
    def retrieve(self, action: Any, _state: Any) -> dict[str, Any]:
        if action.target_need_id == "need-method":
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
                    "table_id": "table-3",
                }
            ]
        return {"status": "completed", "candidates": candidates}


class DuplicateContentRetriever:
    def retrieve(self, _action: Any, _state: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "candidates": [
                {
                    "content": "The same evidence is returned for both searches.",
                    "chunk_type": "text",
                    "section_path": "Method",
                    "page_number": 3,
                }
            ],
        }


class ContextBudgetRetriever:
    def retrieve(self, _action: Any, _state: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "candidates": [
                {
                    "content": "A" * 700,
                    "chunk_id": "chunk-within-budget",
                    "chunk_type": "text",
                    "section_path": "Method",
                },
                {
                    "content": "B" * 400,
                    "chunk_id": "chunk-over-budget",
                    "chunk_type": "text",
                    "section_path": "Method",
                },
            ],
        }


class ExplodingQuestionAnalyzer:
    def analyze(self, _request: Any) -> dict[str, Any]:
        raise RuntimeError("question analysis unavailable")


class ScriptedDraftGenerator:
    def generate(self, request: Any) -> dict[str, Any]:
        if request.version == 1:
            return {
                "answer": "反思 token 决定是否检索；消融实验证明该模块有效。",
                "declared_claims": [],
                "used_candidate_ids": ["chunk-method"],
            }
        return {
            "answer": "反思 token 决定是否检索；移除该模块后分数从 54.1 降至 50.3。",
            "declared_claims": [],
            "used_candidate_ids": ["chunk-method", "chunk-ablation"],
        }


class RecordingDraftGenerator(ScriptedDraftGenerator):
    def __init__(self) -> None:
        self.received_candidate_ids: list[str] = []
        self.received_context_chars = 0

    def generate(self, request: Any) -> dict[str, Any]:
        self.received_candidate_ids = [candidate.candidate_id for candidate in request.candidates]
        self.received_context_chars = sum(len(candidate.content) for candidate in request.candidates)
        return super().generate(request)


class ScriptedClaimExtractor:
    def extract(self, request: Any) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim_id": "claim-method",
                    "text": "反思 token 决定是否检索",
                    "importance": "core",
                    "addressed_need_ids": ["need-method"],
                    "citation_ids": ["chunk-method"],
                },
                {
                    "claim_id": "claim-ablation",
                    "text": "消融实验支持反思模块有效",
                    "importance": "core",
                    "addressed_need_ids": ["need-ablation"],
                    "citation_ids": ["chunk-ablation"] if request.draft_version == 2 else [],
                },
            ]
        }


class ScriptedClaimVerifier:
    def verify(self, request: Any) -> dict[str, Any]:
        assessments = [
            {
                "claim_id": "claim-method",
                "verdict": "supported",
                "supporting_evidence_ids": ["chunk-method"],
                "missing_facets": [],
            }
        ]
        if request.draft_version == 1:
            assessments.append(
                {
                    "claim_id": "claim-ablation",
                    "verdict": "unsupported",
                    "supporting_evidence_ids": [],
                    "missing_facets": ["缺少移除反思模块后的实验结果"],
                }
            )
        else:
            assessments.append(
                {
                    "claim_id": "claim-ablation",
                    "verdict": "supported",
                    "supporting_evidence_ids": ["chunk-ablation"],
                    "missing_facets": [],
                }
            )
        return {"assessments": assessments}


class NoEvidenceClaimVerifier(ScriptedClaimVerifier):
    def verify(self, request: Any) -> dict[str, Any]:
        assessments = super().verify(request)["assessments"]
        for assessment in assessments:
            assessment["verdict"] = "supported"
            assessment["supporting_evidence_ids"] = []
        return {"assessments": assessments}


def test_research_repairs_an_unsupported_core_claim_before_completion() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=ScriptedQuestionAnalyzer(),
        decision_policy=ScriptedDecisionPolicy(),
        retriever=ScriptedRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )

    result = service.research(
        PaperEvidenceResearchRequest(
            arxiv_id="2310.11511",
            original_question="反思 token 如何工作，消融实验是否支持它有效？",
            user_id="user-1",
            session_id="session-1",
            research_run_id="research-run-1",
        )
    )

    assert result.outcome == "completed"
    assert result.research_summary.retrieval_count == 2
    assert result.research_summary.draft_attempt_count == 2
    assert result.research_summary.supported_claim_count == 2
    assert result.research_summary.satisfied_need_count == 2
    assert result.research_summary.unresolved_topics == []
    assert {citation.source_id for citation in result.citations} == {"chunk-method", "chunk-ablation"}


def _request(*, run_id: str, limits: ResearchLimits | None = None) -> PaperEvidenceResearchRequest:
    return PaperEvidenceResearchRequest(
        arxiv_id="2310.11511",
        original_question="反思 token 如何工作，消融实验是否支持它有效？",
        user_id="user-1",
        session_id="session-1",
        research_run_id=run_id,
        limits=limits or ResearchLimits(),
    )


def _service(policy: Any) -> PaperEvidenceResearchService:
    return PaperEvidenceResearchService(
        question_analyzer=ScriptedQuestionAnalyzer(),
        decision_policy=policy,
        retriever=ScriptedRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )


def test_finalize_is_rejected_when_a_core_claim_is_unsupported() -> None:
    service = _service(
        ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "reflection token retrieval decision mechanism",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "DRAFT_NEEDED_TO_DISCOVER_GAPS"},
                {"action": "finalize_answer", "reason_code": "CORE_CLAIMS_SUPPORTED"},
                {"action": "abstain", "reason_code": "BUDGET_EXHAUSTED_WITHOUT_CORE_SUPPORT"},
            ]
        )
    )

    result = service.research(_request(run_id="research-run-finalize-rejected"))

    assert result.outcome == "abstained"
    assert result.research_summary.termination_reason == "BUDGET_EXHAUSTED_WITHOUT_CORE_SUPPORT"


def test_supported_claim_without_evidence_cannot_finalize() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=ScriptedQuestionAnalyzer(),
        decision_policy=ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "reflection token retrieval decision mechanism",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "DRAFT_NEEDED_TO_DISCOVER_GAPS"},
                {"action": "finalize_answer", "reason_code": "CORE_CLAIMS_SUPPORTED"},
                {"action": "abstain", "reason_code": "MISSING_VERIFIABLE_EVIDENCE"},
            ]
        ),
        retriever=ScriptedRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=NoEvidenceClaimVerifier(),
    )

    result = service.research(_request(run_id="research-run-missing-evidence"))

    assert result.outcome == "abstained"
    assert result.research_summary.termination_reason == "MISSING_VERIFIABLE_EVIDENCE"
    assert result.citations == []


def test_budget_exhaustion_returns_only_verified_claims_as_partial_answer() -> None:
    service = _service(
        ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "reflection token retrieval decision mechanism",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "DRAFT_NEEDED_TO_DISCOVER_GAPS"},
                {
                    "action": "search_paper",
                    "target_need_id": "need-ablation",
                    "objective": "verify_claim",
                    "retrieval_mode": "experiment",
                    "query": "reflection token ablation results",
                    "reason_code": "UNSUPPORTED_CORE_CLAIM",
                },
            ]
        )
    )

    result = service.research(
        _request(
            run_id="research-run-partial",
            limits=ResearchLimits(max_retrievals=1, max_draft_attempts=1),
        )
    )

    assert result.outcome == "partial"
    assert result.research_summary.termination_reason == "RETRIEVAL_BUDGET_EXHAUSTED"
    assert result.answer == "反思 token 决定是否检索。"
    assert {citation.source_id for citation in result.citations} == {"chunk-method"}
    assert result.research_summary.satisfied_need_count == 1
    assert result.research_summary.blocked_need_count == 1


def test_explicit_abstention_is_a_normal_research_result() -> None:
    service = _service(
        ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "reflection token retrieval decision mechanism",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "abstain", "reason_code": "PAPER_DOES_NOT_REPORT_ANSWER"},
            ]
        )
    )

    result = service.research(_request(run_id="research-run-abstained"))

    assert result.outcome == "abstained"
    assert result.answer == "当前论文证据不足以支持问题中的核心结论。"
    assert result.research_summary.termination_reason == "PAPER_DOES_NOT_REPORT_ANSWER"


def test_duplicate_only_search_stops_with_no_progress_reason() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=ScriptedQuestionAnalyzer(),
        decision_policy=ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "reflection token mechanism",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {
                    "action": "search_paper",
                    "target_need_id": "need-ablation",
                    "objective": "verify_claim",
                    "retrieval_mode": "experiment",
                    "query": "reflection token ablation",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
            ]
        ),
        retriever=DuplicateContentRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )

    result = service.research(_request(run_id="research-run-no-progress"))

    assert result.outcome == "abstained"
    assert result.research_summary.retrieval_count == 2
    assert result.research_summary.termination_reason == "NO_PROGRESS_LIMIT_REACHED"


def test_research_does_not_send_candidates_beyond_context_budget_to_drafting() -> None:
    draft_generator = RecordingDraftGenerator()
    service = PaperEvidenceResearchService(
        question_analyzer=ScriptedQuestionAnalyzer(),
        decision_policy=ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "reflection token mechanism",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "DRAFT_NEEDED_TO_DISCOVER_GAPS"},
                {"action": "abstain", "reason_code": "PAPER_DOES_NOT_REPORT_ANSWER"},
            ]
        ),
        retriever=ContextBudgetRetriever(),
        draft_generator=draft_generator,
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )

    service.research(
        _request(
            run_id="research-run-context-budget",
            limits=ResearchLimits(max_total_context_chars=1_000),
        )
    )

    assert draft_generator.received_candidate_ids == ["chunk-within-budget"]
    assert draft_generator.received_context_chars == 700


def test_dependency_failure_is_exposed_as_a_structured_system_error() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=ExplodingQuestionAnalyzer(),
        decision_policy=ScriptedPolicyFromActions([]),
        retriever=ScriptedRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )

    with pytest.raises(PaperEvidenceResearchError) as captured:
        service.research(_request(run_id="research-run-error"))

    assert captured.value.code == "paper_evidence_research_failed"
    assert captured.value.stage == "graph_invoke"
    assert captured.value.detail["research_run_id"] == "research-run-error"
