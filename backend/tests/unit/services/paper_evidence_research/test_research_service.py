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


class SupportingOnlyQuestionAnalyzer:
    def analyze(self, request: Any) -> dict[str, Any]:
        return {
            "research_question": request.original_question,
            "evidence_needs": [
                {
                    "need_id": "need-core",
                    "description": "问题要求的核心结论",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                },
                {
                    "need_id": "need-background",
                    "description": "背景定义",
                    "importance": "supporting",
                    "status": "open",
                },
            ],
        }


class PrematureSatisfiedQuestionAnalyzer:
    def analyze(self, request: Any) -> dict[str, Any]:
        return {
            "research_question": request.original_question,
            "evidence_needs": [
                {
                    "need_id": "need-core",
                    "description": "尚未校验的核心结论",
                    "importance": "core",
                    "status": "satisfied",
                    "directly_required_by_question": True,
                    "supporting_claim_ids": ["fake-claim"],
                    "verified_evidence_ids": ["fake-evidence"],
                }
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
        return {"status": "completed", "candidates": candidates, "index_snapshot": {
            "collection_name": "scripted-paper", "active_build_id": "fixture-v1", "active_index_version": "v1",
            "embedding_model": "offline-embedding-v1", "sparse_index_source_hash": "fixture-source-v1",
        }}


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


class CoverageOrderRetriever:
    def retrieve(self, action: Any, _state: Any) -> dict[str, Any]:
        if action.target_need_id == "need-method":
            candidates = [
                {"content": "Method evidence A", "chunk_id": "method-a"},
                {"content": "Method evidence B", "chunk_id": "method-b"},
            ]
        else:
            candidates = [{"content": "Ablation evidence A", "chunk_id": "ablation-a"}]
        return {"status": "completed", "candidates": candidates}


class SupportingOnlyRetriever:
    def retrieve(self, _action: Any, _state: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "candidates": [
                {
                    "content": "The paper defines the reflection token in its background section.",
                    "chunk_id": "chunk-method",
                    "chunk_type": "text",
                    "section_path": "Background",
                }
            ],
        }


class VisualEvidenceRetriever:
    def retrieve(self, _action: Any, _state: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "candidates": [
                {
                    "content": "Figure 2 contains two curves and a legend.",
                    "chunk_id": "figure-2",
                    "chunk_type": "image",
                    "section_path": "Experiments",
                }
            ],
        }


class VisualDraftGenerator:
    def generate(self, _request: Any) -> dict[str, Any]:
        return {"answer": "图 2 中蓝色曲线始终高于红色曲线。", "used_candidate_ids": ["figure-2"]}


class VisualClaimExtractor:
    def extract(self, _request: Any) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim_id": "claim-visual",
                    "text": "图 2 中蓝色曲线始终高于红色曲线",
                    "importance": "core",
                    "addressed_need_ids": ["need-core"],
                    "citation_ids": ["figure-2"],
                    "requires_visual_verification": True,
                }
            ]
        }


class VisualClaimVerifier:
    def verify(self, _request: Any) -> dict[str, Any]:
        return {
            "assessments": [
                {
                    "claim_id": "claim-visual",
                    "verdict": "supported",
                    "supporting_evidence_ids": ["figure-2"],
                    "missing_facets": [],
                }
            ]
        }


class SupportingOnlyDraftGenerator:
    def generate(self, _request: Any) -> dict[str, Any]:
        return {
            "answer": "论文在背景部分定义了反思 token。",
            "used_candidate_ids": ["chunk-method"],
        }


class SupportingOnlyClaimExtractor:
    def extract(self, _request: Any) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim_id": "claim-background",
                    "text": "论文在背景部分定义了反思 token",
                    "importance": "background",
                    "addressed_need_ids": ["need-background"],
                    "citation_ids": ["chunk-method"],
                }
            ]
        }


class SupportingOnlyClaimVerifier:
    def verify(self, _request: Any) -> dict[str, Any]:
        return {
            "assessments": [
                {
                    "claim_id": "claim-background",
                    "verdict": "supported",
                    "supporting_evidence_ids": ["chunk-method"],
                    "missing_facets": [],
                }
            ]
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


class MismatchedCitationVerifier:
    def verify(self, request: Any) -> dict[str, Any]:
        evidence_ids = ["chunk-ablation", "chunk-method"]
        return {
            "assessments": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supported",
                    "supporting_evidence_ids": [evidence_ids[index]],
                    "missing_facets": [],
                }
                for index, claim in enumerate(request.claims)
            ]
        }


class DraftDropsCoreClaimGenerator(ScriptedDraftGenerator):
    def generate(self, request: Any) -> dict[str, Any]:
        if request.version == 1:
            return super().generate(request)
        return {
            "answer": "反思 token 决定是否检索。",
            "used_candidate_ids": ["chunk-method"],
        }


class DraftAwareClaimExtractor(ScriptedClaimExtractor):
    def extract(self, request: Any) -> dict[str, Any]:
        extracted = super().extract(request)
        if request.draft_version == 2:
            extracted["claims"] = [extracted["claims"][0]]
        else:
            extracted["claims"][1]["citation_ids"] = ["chunk-ablation"]
        return extracted


class AlwaysSupportingClaimVerifier:
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


def _service(policy: Any, *, configuration_provider: Any = None) -> PaperEvidenceResearchService:
    return PaperEvidenceResearchService(
        question_analyzer=ScriptedQuestionAnalyzer(),
        decision_policy=policy,
        retriever=ScriptedRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
        configuration_provider=configuration_provider,
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


def test_citation_mismatch_cannot_finalize_from_another_candidate() -> None:
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
                    "objective": "discover",
                    "retrieval_mode": "experiment",
                    "query": "reflection token ablation",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "DRAFT_NEEDED_TO_DISCOVER_GAPS"},
                {"action": "finalize_answer", "reason_code": "CORE_CLAIMS_SUPPORTED"},
                {"action": "abstain", "reason_code": "CITATION_MISMATCH"},
            ]
        ),
        retriever=ScriptedRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=DraftAwareClaimExtractor(),
        claim_verifier=MismatchedCitationVerifier(),
    )

    result = service.research(_request(run_id="research-run-citation-mismatch"))

    assert result.outcome == "abstained"
    assert result.research_summary.termination_reason == "CITATION_MISMATCH"


def test_new_draft_cannot_reuse_coverage_from_a_removed_core_claim() -> None:
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
                    "objective": "discover",
                    "retrieval_mode": "experiment",
                    "query": "reflection token ablation",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "INITIAL_DRAFT"},
                {"action": "draft_answer", "reason_code": "REVISED_DRAFT"},
                {"action": "finalize_answer", "reason_code": "CORE_CLAIMS_SUPPORTED"},
                {"action": "abstain", "reason_code": "DROPPED_CORE_CLAIM"},
            ]
        ),
        retriever=ScriptedRetriever(),
        draft_generator=DraftDropsCoreClaimGenerator(),
        claim_extractor=DraftAwareClaimExtractor(),
        claim_verifier=AlwaysSupportingClaimVerifier(),
    )

    result = service.research(_request(run_id="research-run-coverage-rollback"))

    assert result.outcome == "abstained"
    assert result.research_summary.termination_reason == "DROPPED_CORE_CLAIM"
    assert result.research_summary.satisfied_need_count == 1


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
    assert result.answer == "反思 token 决定是否检索 [source:chunk-method]。"
    assert {citation.source_id for citation in result.citations} == {"chunk-method"}
    assert result.research_summary.satisfied_need_count == 1
    assert result.research_summary.blocked_need_count == 1


def test_supported_background_only_abstains_when_no_core_need_is_satisfied() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=SupportingOnlyQuestionAnalyzer(),
        decision_policy=ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-background",
                    "objective": "discover",
                    "retrieval_mode": "definition",
                    "query": "reflection token definition",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "INITIAL_DRAFT"},
                {
                    "action": "search_paper",
                    "target_need_id": "need-core",
                    "objective": "verify_claim",
                    "retrieval_mode": "method",
                    "query": "core result",
                    "reason_code": "CORE_EVIDENCE_MISSING",
                },
            ]
        ),
        retriever=SupportingOnlyRetriever(),
        draft_generator=SupportingOnlyDraftGenerator(),
        claim_extractor=SupportingOnlyClaimExtractor(),
        claim_verifier=SupportingOnlyClaimVerifier(),
    )

    result = service.research(
        _request(
            run_id="research-run-background-only",
            limits=ResearchLimits(max_retrievals=1),
        )
    )

    assert result.outcome == "abstained"
    assert result.citations == []
    assert result.research_summary.supported_claim_count == 0


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


def test_question_analyzer_cannot_pre_satisfy_an_evidence_need() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=PrematureSatisfiedQuestionAnalyzer(),
        decision_policy=ScriptedPolicyFromActions(
            [{"action": "abstain", "reason_code": "PAPER_DOES_NOT_REPORT_ANSWER"}]
        ),
        retriever=ScriptedRetriever(),
        draft_generator=ScriptedDraftGenerator(),
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )

    result = service.research(_request(run_id="research-run-ledger-initialization"))

    assert result.outcome == "abstained"
    assert result.research_summary.satisfied_need_count == 0
    assert result.research_summary.blocked_need_count == 1


def test_pure_visual_claim_cannot_finalize_without_visual_verifier() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=PrematureSatisfiedQuestionAnalyzer(),
        decision_policy=ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-core",
                    "objective": "discover",
                    "retrieval_mode": "experiment",
                    "query": "figure 2 curve comparison",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {"action": "draft_answer", "reason_code": "INITIAL_DRAFT"},
                {"action": "finalize_answer", "reason_code": "CORE_CLAIMS_SUPPORTED"},
                {"action": "abstain", "reason_code": "PURE_VISUAL_CLAIM_UNVERIFIABLE"},
            ]
        ),
        retriever=VisualEvidenceRetriever(),
        draft_generator=VisualDraftGenerator(),
        claim_extractor=VisualClaimExtractor(),
        claim_verifier=VisualClaimVerifier(),
    )

    result = service.research(_request(run_id="research-run-pure-visual"))

    assert result.outcome == "abstained"
    assert result.research_summary.termination_reason == "PURE_VISUAL_CLAIM_UNVERIFIABLE"


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


def test_malformed_model_action_consumes_invalid_action_budget() -> None:
    service = _service(
        ScriptedPolicyFromActions(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                }
            ]
        )
    )

    result = service.research(
        _request(
            run_id="research-run-invalid-action",
            limits=ResearchLimits(max_invalid_actions=1),
        )
    )

    assert result.outcome == "abstained"
    assert result.research_summary.termination_reason == "ACTION_SCHEMA_INVALID"


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


def test_draft_context_round_robins_candidates_across_core_needs() -> None:
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
                    "query": "method",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {
                    "action": "search_paper",
                    "target_need_id": "need-ablation",
                    "objective": "discover",
                    "retrieval_mode": "experiment",
                    "query": "ablation",
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {
                    "action": "draft_answer",
                    "addressed_need_ids": ["need-method", "need-ablation"],
                    "reason_code": "INITIAL_DRAFT",
                },
                {"action": "abstain", "reason_code": "TEST_COMPLETE"},
            ]
        ),
        retriever=CoverageOrderRetriever(),
        draft_generator=draft_generator,
        claim_extractor=ScriptedClaimExtractor(),
        claim_verifier=ScriptedClaimVerifier(),
    )

    service.research(_request(run_id="research-run-coverage-order"))

    assert draft_generator.received_candidate_ids == ["method-a", "ablation-a", "method-b"]


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
