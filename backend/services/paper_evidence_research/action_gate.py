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
        # 检索动作只能面向仍可推进的需求，并在调用昂贵依赖前执行确定性预算和参数检查。
        if state.retrieval_count >= limits.max_retrievals:
            return ActionValidationResult(False, "RETRIEVAL_BUDGET_EXHAUSTED")
        if not action.query.strip():
            return ActionValidationResult(False, "SEARCH_QUERY_EMPTY")
        if not any(need.need_id == action.target_need_id and need.status in {"provisional", "open"} for need in state.evidence_needs):
            return ActionValidationResult(False, "TARGET_NEED_NOT_OPEN")
        return ActionValidationResult(True)
    if isinstance(action, DraftAnswerAction):
        # 没有候选证据时禁止生成草稿，避免模型脱离论文内容直接作答。
        if state.draft_attempt_count >= limits.max_draft_attempts:
            return ActionValidationResult(False, "DRAFT_BUDGET_EXHAUSTED")
        if not state.evidence_candidates:
            return ActionValidationResult(False, "EVIDENCE_CANDIDATES_EMPTY")
        return ActionValidationResult(True)
    if isinstance(action, FinalizeAnswerAction):
        # LLM 只能请求完成，最终放行权由覆盖与引用约束共同决定。
        accepted, reason = can_finalize(state)
        return ActionValidationResult(accepted, None if accepted else reason)
    if isinstance(action, AbstainAction):
        return ActionValidationResult(True)
    # 未知动作保持 fail-closed，确保扩展动作类型时必须显式补充准入语义。
    return ActionValidationResult(False, "UNKNOWN_ACTION")
