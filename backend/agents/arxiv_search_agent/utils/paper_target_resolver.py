"""论文目标解析器。

Reference Hint Extractor 只说明用户文本里的引用线索；本模块负责把线索和会话/页面状态
合并成候选论文，并决定是否已经能得到唯一目标。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .paper_reference_resolver import _normalize_context_paper, _safe_int
from .text_utils import _normalize_text

logger = logging.getLogger(__name__)

_LOW_RISK_ACTIONS = {
    "paper_qa",
    "paper_summary",
    "paper_detail",
    "paper_download",
    "paper_explain",
    "read",
    "qa",
}

_SIDE_EFFECT_ACTIONS = {
    "preference_action",
    "resolve_preference_target",
    "update_preference_store",
    "favorite",
    "bookmark",
    "like",
    "dislike",
    "mark",
    "feedback",
    "delete",
    "profile_update",
}

_LIST_SOURCE_SPECS: Sequence[Dict[str, Any]] = (
    {"path": "current_papers", "source_type": "current_page", "label": "当前页面展示列表", "priority": 100},
    {"path": "current_page_papers", "source_type": "current_page", "label": "当前页面展示列表", "priority": 100},
    {"path": "displayed_papers", "source_type": "current_page", "label": "当前页面展示列表", "priority": 98},
    {"path": "visible_papers", "source_type": "current_page", "label": "当前页面展示列表", "priority": 98},
    {"path": "page_papers", "source_type": "current_page", "label": "当前页面展示列表", "priority": 96},
    {"path": "papers", "source_type": "current_page", "label": "页面论文列表", "priority": 94},
    {"path": "ranked_papers", "source_type": "search_results", "label": "排序后的搜索结果", "priority": 90},
    {"path": "last_papers", "source_type": "search_results", "label": "上一轮搜索结果", "priority": 88},
    {"path": "search_results", "source_type": "search_results", "label": "搜索结果列表", "priority": 86},
    {"path": "arxiv_results.papers", "source_type": "search_results", "label": "arXiv 搜索结果", "priority": 84},
    {"path": "recommendation_papers", "source_type": "recommendations", "label": "推荐结果列表", "priority": 82},
    {"path": "recommendations", "source_type": "recommendations", "label": "推荐结果列表", "priority": 80},
    {"path": "candidate_papers", "source_type": "recommendations", "label": "推荐候选列表", "priority": 78},
    {"path": "recommendation_result.recommendations", "source_type": "recommendations", "label": "推荐结果列表", "priority": 76},
    {"path": "validated_recommendations.recommendations", "source_type": "recommendations", "label": "已校验推荐结果", "priority": 74},
)

_FOCUS_SOURCE_SPECS: Sequence[Dict[str, Any]] = (
    {"path": "selected_paper", "source_type": "selected_paper", "label": "当前选中论文", "priority": 100},
    {"path": "current_paper", "source_type": "current_paper", "label": "当前打开论文", "priority": 96},
    {"path": "active_paper", "source_type": "current_paper", "label": "当前激活论文", "priority": 94},
    {"path": "current_selected_paper", "source_type": "selected_paper", "label": "当前选中论文", "priority": 92},
    {"path": "recent_opened_paper", "source_type": "recent_focus", "label": "最近打开论文", "priority": 88},
    {"path": "last_opened_paper", "source_type": "recent_focus", "label": "最近打开论文", "priority": 86},
    {"path": "last_clicked_paper", "source_type": "recent_focus", "label": "最近点击论文", "priority": 84},
    {"path": "recent_clicked_paper", "source_type": "recent_focus", "label": "最近点击论文", "priority": 82},
    {"path": "last_assistant_paper", "source_type": "assistant_mention", "label": "最近一次 assistant 提到的论文", "priority": 76},
    {"path": "assistant_last_mentioned_paper", "source_type": "assistant_mention", "label": "最近一次 assistant 提到的论文", "priority": 74},
    {"path": "paper_qa_result", "source_type": "paper_qa_result", "label": "上一轮论文问答目标", "priority": 72},
)

_SCALAR_FOCUS_ID_KEYS = (
    "active_arxiv_id",
    "selected_arxiv_id",
    "current_arxiv_id",
    "last_arxiv_id",
    "arxiv_id",
)


def resolve_paper_target(
    *,
    reference_hint: Mapping[str, Any],
    message: str,
    context: Any,
    action_type: str = "paper_qa",
) -> Dict[str, Any]:
    """把引用线索解析成目标候选或唯一目标。

    这里不执行业务动作，也不调用 LLM；它只做确定性的上下文合并和风险判断，
    让 QA、详情、偏好写入等下游模块只能消费明确的解析结果。
    """
    hint = _normalize_reference_hint(reference_hint)
    paper_context = build_paper_context(context)
    risk_level = _action_risk_level(action_type)
    if _context_expired(context) and bool(hint.get("requires_context")):
        return _log_resolution_decision(
            _build_resolution(
                status="need_clarification",
                reference_hint=hint,
                action_type=action_type,
                risk_level=risk_level,
                candidates=[],
                recommended_candidate=None,
                reason="context_expired",
                confidence=0.0,
                requires_confirmation=False,
                debug={"context_summary": _context_summary(paper_context), "message_source_hint": _infer_requested_list_source(message)},
            ),
            message=message,
            paper_context=paper_context,
        )

    reference_type = str(hint.get("reference_type") or "unknown").strip() or "unknown"
    if reference_type == "arxiv_id":
        return _log_resolution_decision(
            _resolve_arxiv_id_hint(
                hint=hint,
                message=message,
                paper_context=paper_context,
                action_type=action_type,
                risk_level=risk_level,
            ),
            message=message,
            paper_context=paper_context,
        )
    if reference_type == "ordinal":
        return _log_resolution_decision(
            _resolve_ordinal_hint(
                hint=hint,
                message=message,
                paper_context=paper_context,
                action_type=action_type,
                risk_level=risk_level,
                bare_number=False,
            ),
            message=message,
            paper_context=paper_context,
        )
    if reference_type == "last_item":
        return _log_resolution_decision(
            _resolve_last_item_hint(
                hint=hint,
                message=message,
                paper_context=paper_context,
                action_type=action_type,
                risk_level=risk_level,
            ),
            message=message,
            paper_context=paper_context,
        )
    if reference_type == "context_paper":
        return _log_resolution_decision(
            _resolve_context_paper_hint(
                hint=hint,
                message=message,
                paper_context=paper_context,
                action_type=action_type,
                risk_level=risk_level,
            ),
            message=message,
            paper_context=paper_context,
        )
    if reference_type == "bare_number":
        return _log_resolution_decision(
            _resolve_ordinal_hint(
                hint=hint,
                message=message,
                paper_context=paper_context,
                action_type=action_type,
                risk_level=risk_level,
                bare_number=True,
            ),
            message=message,
            paper_context=paper_context,
        )
    return _log_resolution_decision(
        _build_resolution(
            status="need_clarification",
            reference_hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=[],
            recommended_candidate=None,
            reason="reference_unknown",
            confidence=0.0,
            requires_confirmation=False,
            debug={"context_summary": _context_summary(paper_context), "message_source_hint": _infer_requested_list_source(message)},
        ),
        message=message,
        paper_context=paper_context,
    )


def build_paper_context(context: Any) -> Dict[str, Any]:
    """把分散的页面状态、搜索结果、推荐结果和焦点论文统一成候选上下文。

    各上游模块仍可能使用不同字段名；在这里统一收口，避免业务 adapter 各自维护一套
    selected_paper / last_papers / recommendations 的解析规则。
    """
    context_mapping = context if isinstance(context, Mapping) else {}
    list_sources = _extract_list_sources(context_mapping)
    focus_candidates = _extract_focus_candidates(context_mapping)
    all_candidates: List[Dict[str, Any]] = []
    for source in list_sources:
        all_candidates.extend(list(source.get("papers") or []))
    all_candidates.extend(focus_candidates)
    return {
        "list_sources": list_sources,
        "focus_candidates": _dedupe_candidates(focus_candidates),
        "all_candidates": _dedupe_candidates(all_candidates),
    }


def _resolve_arxiv_id_hint(
    *,
    hint: Mapping[str, Any],
    message: str,
    paper_context: Mapping[str, Any],
    action_type: str,
    risk_level: str,
) -> Dict[str, Any]:
    target_id = str(hint.get("value") or "").strip()
    candidates = [
        candidate
        for candidate in list(paper_context.get("all_candidates") or [])
        if _arxiv_ids_match(candidate.get("arxiv_id"), target_id)
    ]
    candidates = _dedupe_candidates(candidates)
    if not candidates and target_id:
        # 显式 arXiv ID 是文本层最高置信线索；即使上下文没有该论文，也可以构造最小目标，
        # 但仍保留来源说明，后续索引/详情工具再负责检查是否存在。
        candidates = [
            _candidate_from_raw(
                {"arxiv_id": target_id},
                source_key="explicit_arxiv_id",
                source_type="explicit_id",
                source_label="用户显式 arXiv ID",
                priority=98,
                rank=1,
            )
        ]

    if not candidates:
        return _build_resolution(
            status="need_clarification",
            reference_hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=[],
            recommended_candidate=None,
            reason="arxiv_id_missing",
            confidence=0.0,
            requires_confirmation=False,
            debug={"context_summary": _context_summary(paper_context), "message_source_hint": _infer_requested_list_source(message)},
        )
    return _single_candidate_resolution(
        hint=hint,
        action_type=action_type,
        risk_level=risk_level,
        candidates=candidates,
        reason="arxiv_id_exact_match" if len(candidates) == 1 else "arxiv_id_multiple_context_matches",
        confidence=max(float(hint.get("confidence") or 0.0), 0.96),
        debug={"context_summary": _context_summary(paper_context), "message_source_hint": _infer_requested_list_source(message)},
    )


def _resolve_ordinal_hint(
    *,
    hint: Mapping[str, Any],
    message: str,
    paper_context: Mapping[str, Any],
    action_type: str,
    risk_level: str,
    bare_number: bool,
) -> Dict[str, Any]:
    ordinal = _safe_int(hint.get("value"), default=0)
    requested_source = _infer_requested_list_source(message)
    candidates = _candidates_at_ordinal(
        list_sources=list(paper_context.get("list_sources") or []),
        ordinal=ordinal,
        requested_source=requested_source,
    )
    candidates = _dedupe_candidates(candidates)
    debug = {
        "context_summary": _context_summary(paper_context),
        "message_source_hint": requested_source,
        "ordinal": ordinal,
        "candidate_source_count": len(candidates),
    }
    if ordinal <= 0:
        return _build_resolution(
            status="need_clarification",
            reference_hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=[],
            recommended_candidate=None,
            reason="ordinal_invalid",
            confidence=0.0,
            requires_confirmation=False,
            debug=debug,
        )
    if not candidates:
        return _build_resolution(
            status="need_clarification",
            reference_hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=[],
            recommended_candidate=None,
            reason="ordinal_context_missing",
            confidence=0.0,
            requires_confirmation=False,
            debug=debug,
        )
    if len(candidates) > 1:
        return _candidate_confirmation_resolution(
            hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=candidates,
            reason="ordinal_multiple_candidate_lists",
            confidence=min(float(hint.get("confidence") or 0.0), 0.72),
            debug=debug,
        )
    if bare_number:
        # 裸数字缺少“第 N 篇/论文”这类锚点；即使上下文能取到候选，也应先确认，避免把数量当下标。
        return _candidate_confirmation_resolution(
            hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=candidates,
            reason="bare_number_requires_confirmation",
            confidence=min(float(hint.get("confidence") or 0.0), 0.58),
            debug=debug,
        )
    return _single_candidate_resolution(
        hint=hint,
        action_type=action_type,
        risk_level=risk_level,
        candidates=candidates,
        reason="ordinal_unique_candidate",
        confidence=min(max(float(hint.get("confidence") or 0.0), 0.8), 0.92),
        debug=debug,
    )


def _resolve_last_item_hint(
    *,
    hint: Mapping[str, Any],
    message: str,
    paper_context: Mapping[str, Any],
    action_type: str,
    risk_level: str,
) -> Dict[str, Any]:
    requested_source = _infer_requested_list_source(message)
    candidates = _last_candidates(
        list_sources=list(paper_context.get("list_sources") or []),
        requested_source=requested_source,
    )
    candidates = _dedupe_candidates(candidates)
    debug = {
        "context_summary": _context_summary(paper_context),
        "message_source_hint": requested_source,
        "candidate_source_count": len(candidates),
    }
    if not candidates:
        return _build_resolution(
            status="need_clarification",
            reference_hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=[],
            recommended_candidate=None,
            reason="last_item_context_missing",
            confidence=0.0,
            requires_confirmation=False,
            debug=debug,
        )
    if len(candidates) > 1:
        return _candidate_confirmation_resolution(
            hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=candidates,
            reason="last_item_multiple_candidate_lists",
            confidence=min(float(hint.get("confidence") or 0.0), 0.72),
            debug=debug,
        )
    return _single_candidate_resolution(
        hint=hint,
        action_type=action_type,
        risk_level=risk_level,
        candidates=candidates,
        reason="last_item_unique_candidate",
        confidence=min(max(float(hint.get("confidence") or 0.0), 0.78), 0.9),
        debug=debug,
    )


def _resolve_context_paper_hint(
    *,
    hint: Mapping[str, Any],
    message: str,
    paper_context: Mapping[str, Any],
    action_type: str,
    risk_level: str,
) -> Dict[str, Any]:
    candidates = _dedupe_candidates(list(paper_context.get("focus_candidates") or []))
    debug = {
        "context_summary": _context_summary(paper_context),
        "message_source_hint": _infer_requested_list_source(message),
        "focus_candidate_count": len(candidates),
    }
    if not candidates:
        return _build_resolution(
            status="need_clarification",
            reference_hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=[],
            recommended_candidate=None,
            reason="context_focus_missing",
            confidence=0.0,
            requires_confirmation=False,
            debug=debug,
        )
    if len(candidates) > 1:
        return _candidate_confirmation_resolution(
            hint=hint,
            action_type=action_type,
            risk_level=risk_level,
            candidates=candidates,
            reason="context_focus_ambiguous",
            confidence=min(float(hint.get("confidence") or 0.0), 0.7),
            debug=debug,
        )
    return _single_candidate_resolution(
        hint=hint,
        action_type=action_type,
        risk_level=risk_level,
        candidates=candidates,
        reason="context_focus_unique_candidate",
        confidence=min(max(float(hint.get("confidence") or 0.0), 0.7), 0.86),
        debug=debug,
    )


def _single_candidate_resolution(
    *,
    hint: Mapping[str, Any],
    action_type: str,
    risk_level: str,
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
    confidence: float,
    debug: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate = dict(candidates[0]) if candidates else None
    requires_confirmation = risk_level == "side_effect"
    return _build_resolution(
        status="resolved",
        reference_hint=hint,
        action_type=action_type,
        risk_level=risk_level,
        candidates=[dict(item) for item in candidates],
        recommended_candidate=candidate,
        reason=reason,
        confidence=confidence,
        requires_confirmation=requires_confirmation,
        debug=dict(debug, side_effect_confirmation_required=requires_confirmation),
    )


def _candidate_confirmation_resolution(
    *,
    hint: Mapping[str, Any],
    action_type: str,
    risk_level: str,
    candidates: Sequence[Mapping[str, Any]],
    reason: str,
    confidence: float,
    debug: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_list = [dict(item) for item in candidates]
    return _build_resolution(
        status="need_confirmation",
        reference_hint=hint,
        action_type=action_type,
        risk_level=risk_level,
        candidates=candidate_list,
        recommended_candidate=candidate_list[0] if candidate_list else None,
        reason=reason,
        confidence=confidence,
        requires_confirmation=True,
        debug=debug,
    )


def _build_resolution(
    *,
    status: str,
    reference_hint: Mapping[str, Any],
    action_type: str,
    risk_level: str,
    candidates: Sequence[Mapping[str, Any]],
    recommended_candidate: Optional[Mapping[str, Any]],
    reason: str,
    confidence: float,
    requires_confirmation: bool,
    debug: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    target = dict(recommended_candidate or {}) if status == "resolved" and recommended_candidate else None
    hint = _normalize_reference_hint(reference_hint)
    candidate_list = [dict(item) for item in candidates]
    target_resolution = {
        "status": status,
        "target": target,
        "candidates": candidate_list,
        "recommended_candidate": dict(recommended_candidate or {}) if recommended_candidate else None,
        "reference_hint": hint,
        "confidence": confidence,
        "resolution_reason": reason,
        "requires_confirmation": requires_confirmation,
        "risk_level": risk_level,
        "action_type": action_type,
        "debug": dict(debug or {}),
    }
    return {
        "status": status,
        "reference_type": hint.get("reference_type") or "unknown",
        "value": hint.get("value"),
        "confidence": confidence,
        "hint_confidence": float(hint.get("confidence") or 0.0),
        "source": hint.get("source"),
        "requires_context": bool(hint.get("requires_context")),
        "reason": reason,
        "resolution_reason": reason,
        "final_target_resolved": status == "resolved",
        "reference_hint": hint,
        "target": target,
        "paper": target,
        "arxiv_id": (target or {}).get("arxiv_id") if target else None,
        "title": (target or {}).get("title") if target else None,
        "matched_by": reason if target else None,
        "target_resolution": target_resolution,
        "candidates": candidate_list,
        "recommended_candidate": dict(recommended_candidate or {}) if recommended_candidate else None,
        "requires_confirmation": requires_confirmation,
        "risk_level": risk_level,
        "action_type": action_type,
        "resolution_debug": dict(debug or {}),
    }


def _log_resolution_decision(
    result: Mapping[str, Any],
    *,
    message: str,
    paper_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """记录目标解析的关键决策，不把完整论文列表写入日志，避免调试信息过大。"""
    payload = dict(result or {})
    hint = payload.get("reference_hint") if isinstance(payload.get("reference_hint"), Mapping) else {}
    candidates = [item for item in list(payload.get("candidates") or []) if isinstance(item, Mapping)]
    recommended = payload.get("recommended_candidate") if isinstance(payload.get("recommended_candidate"), Mapping) else {}
    logger.info(
        "arxiv_agent paper target resolution: message=%r action_type=%s status=%s reason=%s "
        "reference_type=%s reference_value=%s hint_source=%s confidence=%.2f "
        "requires_context=%s requires_confirmation=%s candidate_count=%s candidate_sources=%s "
        "default_candidate=%s context_summary=%s",
        _truncate_text(message, limit=240),
        payload.get("action_type"),
        payload.get("status"),
        payload.get("resolution_reason") or payload.get("reason"),
        hint.get("reference_type") or payload.get("reference_type"),
        hint.get("value") if hint else payload.get("value"),
        hint.get("source") or payload.get("source"),
        float(payload.get("confidence") or 0.0),
        bool(payload.get("requires_context")),
        bool(payload.get("requires_confirmation")),
        len(candidates),
        _candidate_sources_for_log(candidates),
        _candidate_identity_for_log(recommended),
        _context_summary(paper_context),
    )
    return payload


def _candidate_sources_for_log(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """候选来源用于解释为什么需要确认或为什么可直接解析，日志只保留来源和稳定身份。"""
    summaries: List[Dict[str, Any]] = []
    for candidate in list(candidates or [])[:8]:
        if not isinstance(candidate, Mapping):
            continue
        summaries.append(
            {
                "paper_id": candidate.get("paper_id") or candidate.get("candidate_id"),
                "arxiv_id": candidate.get("arxiv_id"),
                "source_type": candidate.get("source_type") or candidate.get("source"),
                "source_key": candidate.get("source_key"),
                "rank": candidate.get("rank"),
            }
        )
    return summaries


def _candidate_identity_for_log(candidate: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, Mapping) or not candidate:
        return None
    return {
        "paper_id": candidate.get("paper_id") or candidate.get("candidate_id"),
        "arxiv_id": candidate.get("arxiv_id"),
        "source_type": candidate.get("source_type") or candidate.get("source"),
        "rank": candidate.get("rank"),
    }


def _truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}..."


def _extract_list_sources(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen_paths = set()
    for spec in _LIST_SOURCE_SPECS:
        path = str(spec["path"])
        if path in seen_paths:
            continue
        seen_paths.add(path)
        raw_list = _value_at_path(context, path)
        if not isinstance(raw_list, list) or not raw_list:
            continue
        papers: List[Dict[str, Any]] = []
        for index, raw_paper in enumerate(raw_list, start=1):
            candidate = _candidate_from_raw(
                raw_paper,
                source_key=path,
                source_type=str(spec["source_type"]),
                source_label=str(spec["label"]),
                priority=int(spec["priority"]),
                rank=index,
            )
            if _candidate_has_identity(candidate):
                papers.append(candidate)
        if papers:
            sources.append(
                {
                    "source_key": path,
                    "source_type": spec["source_type"],
                    "label": spec["label"],
                    "priority": int(spec["priority"]),
                    "papers": papers,
                    "count": len(papers),
                }
            )
    return sources


def _extract_focus_candidates(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for spec in _FOCUS_SOURCE_SPECS:
        raw_paper = _value_at_path(context, str(spec["path"]))
        if not isinstance(raw_paper, Mapping):
            continue
        candidate = _candidate_from_raw(
            raw_paper,
            source_key=str(spec["path"]),
            source_type=str(spec["source_type"]),
            source_label=str(spec["label"]),
            priority=int(spec["priority"]),
            rank=1,
        )
        if _candidate_has_identity(candidate):
            candidates.append(candidate)

    for index, key in enumerate(_SCALAR_FOCUS_ID_KEYS):
        arxiv_id = str(context.get(key) or "").strip()
        if not arxiv_id:
            continue
        # 标量焦点通常来自历史兼容字段，优先级低于结构化 paper，但仍可支撑“这篇论文”定位。
        title = str(context.get(f"{key}_title") or context.get("paper_title") or context.get("title") or "").strip()
        candidate = _candidate_from_raw(
            {"arxiv_id": arxiv_id, "title": title},
            source_key=key,
            source_type="scalar_focus",
            source_label="标量焦点论文",
            priority=68 - index,
            rank=1,
        )
        if _candidate_has_identity(candidate):
            candidates.append(candidate)
    return candidates


def _candidate_from_raw(
    raw: Any,
    *,
    source_key: str,
    source_type: str,
    source_label: str,
    priority: int,
    rank: int,
) -> Dict[str, Any]:
    raw_mapping = raw if isinstance(raw, Mapping) else {}
    paper_payload = raw_mapping.get("paper") if isinstance(raw_mapping.get("paper"), Mapping) else raw_mapping
    if paper_payload is not raw_mapping and isinstance(paper_payload, Mapping):
        merged_payload = dict(paper_payload)
        for key, value in raw_mapping.items():
            if key == "paper" or value in (None, "", [], {}):
                continue
            merged_payload.setdefault(key, value)
        paper_payload = merged_payload

    paper = _normalize_context_paper(paper_payload)
    raw_id = str(raw_mapping.get("id") or "").strip()
    paper_id = str(
        raw_mapping.get("paper_id")
        or raw_mapping.get("paperId")
        or raw_mapping.get("stable_id")
        or (raw_id if raw_id and not _looks_like_arxiv_id(raw_id) else "")
        or paper.get("arxiv_id")
        or paper.get("title")
        or ""
    ).strip()
    raw_rank = raw_mapping.get("rank") or raw_mapping.get("position") or raw_mapping.get("index") or rank
    normalized_rank = _safe_int(raw_rank, default=rank)
    if normalized_rank <= 0:
        normalized_rank = rank
    candidate = {
        **paper,
        "paper_id": paper_id,
        "source": source_type,
        "source_type": source_type,
        "source_key": source_key,
        "source_label": source_label,
        "source_list": [source_key],
        "source_types": [source_type],
        "rank": normalized_rank,
        "list_name": source_label,
        "source_priority": priority,
    }
    return candidate


def _candidates_at_ordinal(
    *,
    list_sources: Sequence[Mapping[str, Any]],
    ordinal: int,
    requested_source: Optional[str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for source in _filter_list_sources(list_sources, requested_source):
        papers = list(source.get("papers") or [])
        if ordinal <= 0 or ordinal > len(papers):
            continue
        candidate = dict(papers[ordinal - 1])
        candidate["rank"] = ordinal
        candidates.append(candidate)
    return candidates


def _last_candidates(
    *,
    list_sources: Sequence[Mapping[str, Any]],
    requested_source: Optional[str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for source in _filter_list_sources(list_sources, requested_source):
        papers = list(source.get("papers") or [])
        if not papers:
            continue
        candidate = dict(papers[-1])
        candidate["rank"] = len(papers)
        candidates.append(candidate)
    return candidates


def _filter_list_sources(
    list_sources: Sequence[Mapping[str, Any]],
    requested_source: Optional[str],
) -> List[Mapping[str, Any]]:
    if requested_source:
        filtered = [
            source
            for source in list_sources
            if str(source.get("source_type") or "") == requested_source
        ]
        if filtered:
            return filtered
    return list(list_sources)


def _dedupe_candidates(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    sorted_candidates = sorted(
        [dict(item) for item in candidates if isinstance(item, Mapping) and _candidate_has_identity(item)],
        key=lambda item: (-int(item.get("source_priority") or 0), int(item.get("rank") or 9999)),
    )
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for candidate in sorted_candidates:
        identity = _candidate_identity(candidate)
        if not identity:
            continue
        if identity not in merged:
            merged[identity] = dict(candidate)
            order.append(identity)
            continue
        existing = merged[identity]
        existing["source_priority"] = max(int(existing.get("source_priority") or 0), int(candidate.get("source_priority") or 0))
        existing["rank"] = min(int(existing.get("rank") or 9999), int(candidate.get("rank") or 9999))
        existing["source_list"] = _dedupe_texts(list(existing.get("source_list") or []) + list(candidate.get("source_list") or []))
        existing["source_types"] = _dedupe_texts(list(existing.get("source_types") or []) + list(candidate.get("source_types") or []))
        if not existing.get("title") and candidate.get("title"):
            existing["title"] = candidate.get("title")
        if not existing.get("arxiv_id") and candidate.get("arxiv_id"):
            existing["arxiv_id"] = candidate.get("arxiv_id")
        if not existing.get("paper_id") and candidate.get("paper_id"):
            existing["paper_id"] = candidate.get("paper_id")
    return [merged[key] for key in order]


def _normalize_reference_hint(reference_hint: Mapping[str, Any]) -> Dict[str, Any]:
    hint_source = reference_hint.get("reference_hint") if isinstance(reference_hint.get("reference_hint"), Mapping) else reference_hint
    hint = dict(hint_source or {})
    hint.setdefault("reference_type", "unknown")
    hint.setdefault("value", None)
    hint.setdefault("confidence", 0.0)
    hint.setdefault("source", "unknown")
    hint.setdefault("requires_context", False)
    hint.setdefault("final_target_resolved", False)
    return hint


def _infer_requested_list_source(message: str) -> Optional[str]:
    text = _normalize_text(message)
    if not text:
        return None
    if re.search(r"(推荐|recommendation|recommend)", text, flags=re.IGNORECASE):
        return "recommendations"
    if re.search(r"(搜索|检索|查询|结果|search|result)", text, flags=re.IGNORECASE):
        return "search_results"
    if re.search(r"(当前|页面|展示|列表|current|displayed|visible)", text, flags=re.IGNORECASE):
        return "current_page"
    return None


def _context_expired(context: Any) -> bool:
    if not isinstance(context, Mapping):
        return False
    return bool(
        context.get("paper_context_expired")
        or context.get("target_context_expired")
        or context.get("context_expired")
    )


def _action_risk_level(action_type: str) -> str:
    normalized = str(action_type or "").strip()
    if normalized in _SIDE_EFFECT_ACTIONS:
        return "side_effect"
    if normalized in _LOW_RISK_ACTIONS:
        return "low"
    return "low"


def _value_at_path(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _candidate_has_identity(candidate: Mapping[str, Any]) -> bool:
    return bool(str(candidate.get("arxiv_id") or "").strip() or str(candidate.get("paper_id") or "").strip() or str(candidate.get("title") or "").strip())


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    arxiv_key = _arxiv_base(candidate.get("arxiv_id"))
    if arxiv_key:
        return f"arxiv:{arxiv_key}"
    paper_id = str(candidate.get("paper_id") or "").strip().lower()
    if paper_id:
        return f"paper_id:{paper_id}"
    title = _normalize_text(str(candidate.get("title") or "")).lower()
    return f"title:{title}" if title else ""


def _looks_like_arxiv_id(value: Any) -> bool:
    return bool(_arxiv_base(value))


def _arxiv_ids_match(left: Any, right: Any) -> bool:
    return bool(_arxiv_base(left) and _arxiv_base(left) == _arxiv_base(right))


def _arxiv_base(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?", text)
    return match.group(1) if match else ""


def _dedupe_texts(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _context_summary(paper_context: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "list_sources": [
            {
                "source_key": source.get("source_key"),
                "source_type": source.get("source_type"),
                "count": source.get("count"),
            }
            for source in list(paper_context.get("list_sources") or [])
        ],
        "focus_candidate_count": len(list(paper_context.get("focus_candidates") or [])),
        "all_candidate_count": len(list(paper_context.get("all_candidates") or [])),
    }


__all__ = ["build_paper_context", "resolve_paper_target"]
