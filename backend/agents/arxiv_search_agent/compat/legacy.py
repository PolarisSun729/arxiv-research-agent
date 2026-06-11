from __future__ import annotations

from typing import Any, Dict, Optional

from ..schemas import AgentTurnResult


def build_legacy_pending_action(result: AgentTurnResult) -> Optional[Dict[str, Any]]:
    """把新的确认请求映射成旧前端仍在读取的 pending_action 镜像。

    这层属于显式 compat/legacy 边界，不再作为主状态流扩展点；
    真正可恢复的确认状态仍以 structured pending_confirmation / resume 为准。
    """
    confirmation = result.pending_confirmation
    if confirmation is None:
        return None
    payload = confirmation.model_dump()
    target_paper = dict(confirmation.target_paper or {})
    arguments_summary = dict(confirmation.arguments_summary or {})
    return {
        "type": confirmation.request_type,
        "request_type": confirmation.request_type,
        "status": "waiting_confirmation",
        "decision": None,
        "pending_action_id": confirmation.pending_action_id,
        "step_id": confirmation.step_id,
        "tool_name": confirmation.tool_name,
        "action_type": confirmation.action_type,
        "side_effect_level": confirmation.side_effect_level,
        "reason": confirmation.reason,
        "title": target_paper.get("title") or confirmation.title,
        "title_text": confirmation.title,
        "description": confirmation.description,
        "arxiv_id": target_paper.get("arxiv_id"),
        "original_question": confirmation.original_question,
        "original_message": confirmation.original_message,
        "target_paper": target_paper or None,
        "candidates": list(confirmation.candidates or []),
        "recommended_candidate": dict(confirmation.recommended_candidate or {}) if confirmation.recommended_candidate else None,
        "default_candidate_id": confirmation.default_candidate_id,
        "reference_hint": dict(confirmation.reference_hint or {}),
        "target_resolution": dict(confirmation.target_resolution or {}),
        "confirmation_fields": dict(confirmation.confirmation_fields or {}),
        "created_at": confirmation.created_at,
        "expires_at": confirmation.expires_at,
        "allowed_decisions": [item.code for item in list(confirmation.allowed_decisions or [])],
        "allow_argument_edit": confirmation.allow_argument_edit,
        "allow_reject": confirmation.allow_reject,
        "allow_note": confirmation.allow_note,
        "arguments_summary": arguments_summary,
        "confirmation_request": payload,
        "thread_id": confirmation.thread_id,
        "session_id": confirmation.session_id,
        "plan_id": confirmation.plan_id,
        "trace_id": confirmation.trace_id,
        "qa_question": arguments_summary.get("qa_question") or arguments_summary.get("question"),
    }
