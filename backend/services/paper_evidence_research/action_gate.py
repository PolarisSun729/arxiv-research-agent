from __future__ import annotations

from dataclasses import dataclass

from .actions import AbstainAction, DraftAnswerAction, FinalizeAnswerAction, ResearchAction, SearchPaperAction
from .completion_gate import can_finalize
from .state import PaperEvidenceResearchState


@dataclass(frozen=True)
class ActionValidationResult:
    accepted: bool
    rejection_code: str | None = None


def validate_action(state: PaperEvidenceResearchState, action: ResearchAction) -> ActionValidationResult:
    limits = state.request.limits
    if isinstance(action, SearchPaperAction):
        if state.retrieval_count >= limits.max_retrievals:
            return ActionValidationResult(False, "RETRIEVAL_BUDGET_EXHAUSTED")
        if not action.query.strip():
            return ActionValidationResult(False, "SEARCH_QUERY_EMPTY")
        if not any(need.need_id == action.target_need_id and need.status in {"provisional", "open"} for need in state.evidence_needs):
            return ActionValidationResult(False, "TARGET_NEED_NOT_OPEN")
        return ActionValidationResult(True)
    if isinstance(action, DraftAnswerAction):
        if state.draft_attempt_count >= limits.max_draft_attempts:
            return ActionValidationResult(False, "DRAFT_BUDGET_EXHAUSTED")
        if not state.evidence_candidates:
            return ActionValidationResult(False, "EVIDENCE_CANDIDATES_EMPTY")
        return ActionValidationResult(True)
    if isinstance(action, FinalizeAnswerAction):
        accepted, reason = can_finalize(state)
        return ActionValidationResult(accepted, None if accepted else reason)
    if isinstance(action, AbstainAction):
        return ActionValidationResult(True)
    return ActionValidationResult(False, "UNKNOWN_ACTION")
