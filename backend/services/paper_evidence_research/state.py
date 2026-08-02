from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .actions import ResearchAction
from .contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult


class EvidenceNeed(BaseModel):
    need_id: str
    description: str
    importance: Literal["core", "supporting"] = "supporting"
    status: Literal["provisional", "open", "satisfied", "blocked", "superseded", "abandoned"] = "provisional"
    directly_required_by_question: bool = False
    parent_need_id: str | None = None
    supporting_claim_ids: list[str] = Field(default_factory=list)
    verified_evidence_ids: list[str] = Field(default_factory=list)
    blocking_reason: str | None = None


class EvidenceCandidate(BaseModel):
    candidate_id: str
    content: str
    chunk_type: str = "text"
    chunk_id: str = ""
    parent_chunk_id: str = ""
    original_chunk_id: str = ""
    section_path: str = ""
    page_number: int | None = None
    table_id: str = ""
    matched_need_ids: list[str] = Field(default_factory=list)
    appearance_count: int = 1


class EvidenceContextPack(BaseModel):
    selected_need_ids: list[str] = Field(default_factory=list)
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    total_context_chars: int = 0


class DraftAnswer(BaseModel):
    draft_id: str
    version: int
    answer: str
    declared_claims: list[dict[str, Any]] = Field(default_factory=list)
    used_candidate_ids: list[str] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    claim_id: str
    text: str
    importance: Literal["core", "supporting", "background"] = "supporting"
    addressed_need_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    # 一期没有 FigureVerifier；该标记阻止依赖图像本体读取的主张被文本校验结果误放行。
    requires_visual_verification: bool = False


class ClaimAssessment(BaseModel):
    claim_id: str
    verdict: Literal["supported", "partially_supported", "unsupported", "conflicting", "citation_mismatch"]
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    missing_facets: list[str] = Field(default_factory=list)


class DraftGenerationRequest(BaseModel):
    version: int
    research_question: str
    addressed_need_ids: list[str]
    context_pack: EvidenceContextPack

    @property
    def candidates(self) -> list[EvidenceCandidate]:
        """兼容脚本适配器的候选读取方式，对外仍由上下文包持有选择结果。"""

        return self.context_pack.candidates


class ClaimExtractionRequest(BaseModel):
    research_question: str
    answer: str
    draft_version: int


class ClaimVerificationRequest(BaseModel):
    research_question: str
    draft_version: int
    claims: list[AnswerClaim]
    context_pack: EvidenceContextPack

    @property
    def candidates(self) -> list[EvidenceCandidate]:
        """校验器只读取本轮上下文包，不能绕过选择边界访问完整候选池。"""

        return self.context_pack.candidates


class ResearchDecisionContext(BaseModel):
    research_question: str
    evidence_needs: list[EvidenceNeed]
    candidate_count: int
    current_draft_version: int | None
    claim_assessments: list[ClaimAssessment]
    retrievals_remaining: int
    drafts_remaining: int


class PaperEvidenceResearchState(BaseModel):
    request: PaperEvidenceResearchRequest
    research_question: str = ""
    evidence_needs: list[EvidenceNeed] = Field(default_factory=list)
    evidence_candidates: dict[str, EvidenceCandidate] = Field(default_factory=dict)
    current_draft: DraftAnswer | None = None
    claims: list[AnswerClaim] = Field(default_factory=list)
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    pending_action: ResearchAction | None = None
    action_accepted: bool = False
    action_rejection_code: str | None = None
    retrieval_count: int = 0
    draft_attempt_count: int = 0
    verification_count: int = 0
    no_progress_count: int = 0
    invalid_action_count: int = 0
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    result: PaperEvidenceResearchResult | None = None
