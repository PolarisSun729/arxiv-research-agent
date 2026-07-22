from __future__ import annotations

from .state import ClaimAssessment, PaperEvidenceResearchState


def has_verifiable_support(
    state: PaperEvidenceResearchState,
    assessment: ClaimAssessment | None,
) -> bool:
    if assessment is None or assessment.verdict != "supported" or not assessment.supporting_evidence_ids:
        return False
    # LLM 的 supported 标签不是最终真值；所有支持项都必须能回指当前候选池中的非空原文。
    return all(
        evidence_id in state.evidence_candidates
        and bool(state.evidence_candidates[evidence_id].content.strip())
        for evidence_id in assessment.supporting_evidence_ids
    )


def can_finalize(state: PaperEvidenceResearchState) -> tuple[bool, str]:
    if state.current_draft is None:
        return False, "DRAFT_MISSING"
    if not state.claims or not state.claim_assessments:
        return False, "VERIFICATION_MISSING"
    assessment_by_claim = {item.claim_id: item for item in state.claim_assessments}
    for claim in state.claims:
        assessment = assessment_by_claim.get(claim.claim_id)
        if not has_verifiable_support(state, assessment):
            return False, "VISIBLE_CLAIM_NOT_SUPPORTED"
    for need in state.evidence_needs:
        if need.importance == "core" and need.status != "satisfied":
            return False, "CORE_NEED_NOT_SATISFIED"
    return True, "CORE_CLAIMS_SUPPORTED"
