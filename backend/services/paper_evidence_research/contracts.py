from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ResearchLimits(BaseModel):
    max_retrievals: int = Field(default=3, ge=1, le=10)
    max_draft_attempts: int = Field(default=2, ge=1, le=5)
    max_no_progress: int = Field(default=1, ge=0, le=5)
    max_invalid_actions: int = Field(default=2, ge=0, le=5)
    max_evidence_items: int = Field(default=20, ge=1, le=100)
    max_total_context_chars: int = Field(default=30_000, ge=1_000, le=200_000)


class PaperEvidenceResearchRequest(BaseModel):
    arxiv_id: str = Field(min_length=1)
    original_question: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    research_run_id: str = Field(min_length=1)
    conversation_snapshot: dict[str, Any] | None = None
    preferred_answer_style: str | None = None
    limits: ResearchLimits = Field(default_factory=ResearchLimits)


class VerifiedCitation(BaseModel):
    source_id: str
    content: str
    claim_ids: list[str] = Field(default_factory=list)
    chunk_type: str = "text"
    section_path: str = ""
    page_number: int | None = None


class ResearchSummary(BaseModel):
    research_run_id: str
    retrieval_count: int
    draft_attempt_count: int
    verification_count: int
    confirmed_need_count: int
    satisfied_need_count: int
    blocked_need_count: int
    supported_claim_count: int
    removed_claim_count: int = 0
    citation_repair_count: int = 0
    outcome: Literal["completed", "partial", "abstained"]
    termination_reason: str
    unresolved_topics: list[str] = Field(default_factory=list)


class PaperEvidenceResearchResult(BaseModel):
    outcome: Literal["completed", "partial", "abstained"]
    answer: str
    citations: list[VerifiedCitation] = Field(default_factory=list)
    research_summary: ResearchSummary
    research_trace_id: str
