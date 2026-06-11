from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    from .utils.paper_reference_resolver import _resolve_paper_reference
except Exception:  # pragma: no cover - 轻量导入场景允许缺失解析器
    _resolve_paper_reference = None


SUPPORTED_INTENTS = {
    "arxiv_search",
    "paper_summary",
    "paper_detail",
    "paper_qa",
    "recommendation",
    "preference_action",
    "confirmation",
    "unclear",
    "unsupported",
}

# 每类任务的最小可执行条件独立保留，方便澄清 trace 解释“为什么现在不能继续”。
_MINIMUM_EXECUTION_REQUIREMENTS: Dict[str, List[str]] = {
    "arxiv_search": ["search_topic"],
    "paper_summary": ["target_paper"],
    "paper_detail": ["target_paper"],
    "paper_qa": ["target_paper", "user_question"],
    "recommendation": ["user_identity_or_profile_or_constraints"],
    "preference_action": ["preference_action", "preference_target", "user_identity"],
    "confirmation": ["confirmation_request", "confirmation_decision"],
    "unclear": ["task_intent"],
    "unsupported": [],
}

_CONFIRM_APPROVE_TOKENS = {
    "approve",
    "approved",
    "accept",
    "accepted",
    "yes",
    "ok",
    "sure",
    "同意",
    "确认",
    "可以",
    "继续",
    "批准",
}

_CONFIRM_REJECT_TOKENS = {
    "reject",
    "rejected",
    "deny",
    "denied",
    "no",
    "cancel",
    "stop",
    "拒绝",
    "取消",
    "不要",
    "停止",
}

_PREFERENCE_ACTION_TOKENS = {
    "like",
    "liked",
    "dislike",
    "favorite",
    "favourite",
    "remove",
    "unlike",
    "喜欢",
    "不喜欢",
    "收藏",
    "取消收藏",
    "移除",
    "删除",
}

_RECOMMENDATION_TOKENS = {
    "recommend",
    "recommendation",
    "推荐",
}

_SEARCH_TOKENS = {
    "search",
    "find",
    "lookup",
    "搜",
    "搜索",
    "查",
    "检索",
}

_SUMMARY_TOKENS = {
    "summary",
    "summarize",
    "overview",
    "总结",
    "概述",
}

_DETAIL_TOKENS = {
    "method",
    "methods",
    "detail",
    "details",
    "experiment",
    "result",
    "implementation",
    "methodology",
    "方法",
    "细节",
    "实验",
    "结果",
    "实现",
    "原理",
}

_PAPER_REFERENCE_TOKENS = {
    "this paper",
    "that paper",
    "this one",
    "that one",
    "the paper",
    "这篇论文",
    "这篇",
    "那篇论文",
    "那篇",
    "该论文",
    "这个",
    "那个",
}

_GENERIC_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "arxiv",
    "article",
    "for",
    "help",
    "me",
    "of",
    "on",
    "paper",
    "papers",
    "please",
    "recommend",
    "recommendation",
    "related",
    "research",
    "search",
    "show",
    "some",
    "the",
    "topic",
    "with",
    "一个",
    "一些",
    "主题",
    "几篇",
    "几篇论文",
    "可以",
    "围绕",
    "帮我",
    "推荐",
    "推荐几篇",
    "搜索",
    "搜一下",
    "搜一搜",
    "找",
    "找一下",
    "查",
    "查一下",
    "看看",
    "研究",
    "相关",
    "论文",
    "这个",
    "这篇",
    "这篇论文",
}

_QUESTION_HINTS = {
    "?",
    "？",
    "how",
    "what",
    "why",
    "when",
    "which",
    "who",
    "method",
    "methods",
    "detail",
    "details",
    "result",
    "results",
    "compare",
    "comparison",
    "difference",
    "流程",
    "为什么",
    "什么",
    "怎么",
    "如何",
    "哪些",
    "哪种",
    "方法",
    "细节",
    "结果",
    "对比",
    "区别",
}

_UNSUPPORTED_HINTS = {
    "poster",
    "save for later",
    "reading list",
    "saved papers",
    "bookmark list",
    "海报",
    "稍后阅读",
    "阅读列表",
    "已保存论文",
    "收藏夹",
}

_DEFAULT_USER_ID_SOURCE = "agent_default_user_context"


def build_clarification_diagnostic(
    *,
    message: Any,
    context: Any = None,
    goal: Any = None,
    user_id: Any = None,
    pending_action: Any = None,
    search_spec: Any = None,
) -> Dict[str, Any]:
    """构造稳定的澄清诊断结果。

    这里不负责“替用户把请求补全”，而是回答三件事：
    1. 当前更像哪类任务；
    2. 这类任务距离“最小可执行条件”还缺什么；
    3. 缺口能否安全默认继续，还是必须先向用户追问。
    """

    normalized_message = _normalize_text(message) or ""
    context_mapping = dict(context) if isinstance(context, Mapping) else {}
    goal_mapping = _coerce_mapping(goal)
    pending_mapping = dict(pending_action) if isinstance(pending_action, Mapping) else {}
    if not pending_mapping and isinstance(context_mapping.get("pending_action"), Mapping):
        pending_mapping = dict(context_mapping.get("pending_action") or {})
    search_spec_mapping = _coerce_mapping(search_spec)

    selected_paper = _coerce_mapping(context_mapping.get("selected_paper"))
    last_papers = _coerce_paper_list(
        context_mapping.get("last_papers")
        or context_mapping.get("papers")
        or context_mapping.get("candidate_papers")
    )
    user_memory_summary = context_mapping.get("user_memory_summary", context_mapping.get("memory_summary"))
    research_profile = context_mapping.get("research_profile")
    normalized_user_id = _normalize_text(user_id) or _normalize_text(context_mapping.get("user_id"))
    has_identity_context = bool(normalized_user_id)

    preferred_intent = _normalize_intent(
        goal_mapping.get("intent")
        or goal_mapping.get("goal_type")
        or context_mapping.get("intent")
    )
    inferred_intent, confidence = _infer_intent(
        normalized_message,
        preferred_intent=preferred_intent,
        pending_action=pending_mapping,
    )
    if inferred_intent not in SUPPORTED_INTENTS:
        inferred_intent = "unclear"

    topic_terms = _extract_topic_terms(normalized_message)
    has_search_constraints = _has_search_constraints(search_spec_mapping, topic_terms)
    resolved_target = _resolve_target_paper(normalized_message, context_mapping)
    has_target_paper = resolved_target.get("status") == "success"
    has_contextual_paper_reference = _message_has_contextual_paper_reference(normalized_message)
    has_question = _has_question_content(normalized_message)
    has_profile_context = user_memory_summary not in (None, "", [], {}) or research_profile not in (None, "", [], {})
    has_pending_confirmation = _has_pending_confirmation(pending_mapping)
    confirmation_decision = _extract_confirmation_decision(normalized_message)
    preference_action = _extract_preference_action_kind(normalized_message)
    scope_too_broad = _is_scope_too_broad(
        inferred_intent=inferred_intent,
        message=normalized_message,
        topic_terms=topic_terms,
        has_profile_context=has_profile_context,
    )

    details: List[Dict[str, Any]] = []
    default_values: Dict[str, Any] = {}
    resolution_strategy = "ask_clarification"
    support_status = "supported"
    reason = "request_is_ambiguous"
    minimum_required_fields = list(_MINIMUM_EXECUTION_REQUIREMENTS.get(inferred_intent, []))

    if inferred_intent == "unsupported":
        support_status = "unsupported"
        resolution_strategy = "fallback_unsupported_request"
        reason = "unsupported_request"
        details.append(
            _missing_detail(
                "unsupported_request",
                "当前请求不在这个 agent 的能力范围内，继续追问也无法安全落到已支持工具。",
                "你可以改成 arXiv 搜索、单篇论文问答、论文推荐、偏好更新，或处理待确认操作。",
                required=False,
            )
        )
    elif inferred_intent == "confirmation":
        if not has_pending_confirmation:
            details.append(
                _missing_detail(
                    "confirmation_request",
                    "当前没有待确认的任务，系统无法判断你这句“同意/拒绝”对应哪一步。",
                    "请先告诉我你要确认哪一步，或者重新发起需要确认的请求。",
                )
            )
            reason = "missing_confirmation_request"
        elif not confirmation_decision:
            details.append(
                _missing_detail(
                    "confirmation_decision",
                    "当前确实有待确认任务，但这条消息没有明确表达是批准还是拒绝。",
                    "请直接回复“同意”或“拒绝”。",
                )
            )
            reason = "missing_confirmation_decision"
        else:
            reason = "confirmation_ready"
            resolution_strategy = "continue_without_clarification"
    elif inferred_intent == "arxiv_search":
        if not has_search_constraints:
            details.append(
                _missing_detail(
                    "search_topic",
                    "搜索请求还没有提供可执行的主题、关键词或分类约束。",
                    "请补充你想搜的主题，例如“RAG 评测”“Agent memory”，或指定 arXiv 分类。",
                )
            )
            reason = "missing_search_topic"
        if scope_too_broad:
            details.append(
                _missing_detail(
                    "scope_constraint",
                    "当前搜索范围过宽，直接执行会返回过于发散的结果。",
                    "请补充研究方向、方法关键词、时间范围或分类范围，帮助我收窄搜索。",
                )
            )
            reason = "request_too_broad"
    elif inferred_intent in {"paper_summary", "paper_detail", "paper_qa"}:
        if not has_target_paper:
            details.append(
                _missing_detail(
                    "target_paper",
                    _target_paper_reason(resolved_target, has_contextual_paper_reference),
                    "请提供论文标题、arXiv ID，或者先选中/搜索到目标论文。",
                )
            )
            reason = "missing_target_paper"
        elif inferred_intent == "paper_qa" and not has_question:
            details.append(
                _missing_detail(
                    "user_question",
                    "我已经知道你想基于论文继续，但还没拿到具体问题，无法进入问答。",
                    "请补充你想问的内容，例如“这篇论文的方法是什么？”或“实验结果如何？”。",
                )
            )
            reason = "missing_user_question"
    elif inferred_intent == "recommendation":
        if not has_search_constraints and not has_profile_context:
            details.append(
                _missing_detail(
                    "recommendation_constraints",
                    "当前既没有可复用的用户画像，也没有给出推荐主题或筛选条件。",
                    "请补充研究方向、关键词、时间范围，或者先提供可用的用户画像/偏好。",
                )
            )
            reason = "missing_recommendation_constraints"
        if not has_identity_context and not has_profile_context:
            if has_search_constraints:
                default_values["user_id"] = {
                    "value": "default",
                    "source": _DEFAULT_USER_ID_SOURCE,
                    "effect": "use_message_only_recommendation",
                }
                details.append(
                    _missing_detail(
                        "user_identity",
                        "当前没有明确 user_id，系统只能按这轮消息做非个性化推荐。",
                        "如果你希望基于个人画像推荐，请补充 user_id；否则我也可以先按当前主题继续。",
                        required=False,
                        can_use_default=True,
                        default_value_source=_DEFAULT_USER_ID_SOURCE,
                    )
                )
                resolution_strategy = "continue_with_defaults"
                reason = "missing_user_identity_but_default_allowed"
            else:
                details.append(
                    _missing_detail(
                        "user_identity",
                        "当前没有明确 user_id，也没有可复用画像，无法做个性化推荐。",
                        "请补充 user_id，或者至少告诉我你希望围绕什么主题推荐。",
                    )
                )
                reason = "missing_user_identity"
        if scope_too_broad and not has_profile_context:
            details.append(
                _missing_detail(
                    "scope_constraint",
                    "推荐请求过于宽泛，缺少约束会导致结果过散。",
                    "请补充主题、关键词、时间范围，或者先标记几篇喜欢/不喜欢的论文。",
                )
            )
            reason = "request_too_broad"
    elif inferred_intent == "preference_action":
        if not preference_action:
            details.append(
                _missing_detail(
                    "preference_action",
                    "当前只看到了偏好相关表达，但还不明确是喜欢、不喜欢还是取消偏好。",
                    "请明确告诉我是“喜欢”“不喜欢”还是“取消偏好”。",
                )
            )
            reason = "missing_preference_action"
        if not has_target_paper:
            details.append(
                _missing_detail(
                    "preference_target",
                    _target_paper_reason(resolved_target, has_contextual_paper_reference),
                    "请提供论文标题、arXiv ID，或者基于当前结果说“喜欢第一篇 / 不喜欢这篇”。",
                )
            )
            reason = "missing_preference_target"
        if not has_identity_context:
            details.append(
                _missing_detail(
                    "user_identity",
                    "偏好写入属于持久化操作，缺少 user_id 时不能安全写入默认用户。",
                    "请先提供 user_id，确认这次偏好要写入哪个用户画像。",
                )
            )
            reason = "missing_user_identity"
    else:
        if has_contextual_paper_reference and not has_target_paper:
            details.append(
                _missing_detail(
                    "target_paper",
                    _target_paper_reason(resolved_target, has_contextual_paper_reference),
                    "请补充目标论文标题、arXiv ID，或者先让我搜索相关论文。",
                )
            )
            reason = "missing_target_paper"
        elif not has_search_constraints:
            details.append(
                _missing_detail(
                    "search_topic",
                    "当前请求还没有提供足够明确的主题或任务目标。",
                    "请补充你想搜索、问答、推荐还是更新偏好，并说明目标论文或主题。",
                )
            )
            reason = "request_is_ambiguous"

    blocking_details = [
        item
        for item in details
        if item.get("required", True) and not item.get("can_use_default", False)
    ]
    needs_clarification = support_status == "supported" and bool(blocking_details)
    if support_status == "supported" and not details:
        resolution_strategy = "continue_without_clarification"
        reason = "enough_information"
    elif support_status == "supported" and not needs_clarification and default_values:
        resolution_strategy = "continue_with_defaults"

    suggested_questions = _dedupe_strings(item.get("suggested_question") for item in details)
    missing_fields = _dedupe_strings(item.get("field") for item in details)
    allow_default_continuation = support_status == "supported" and bool(default_values) and not needs_clarification

    return {
        "message": normalized_message,
        "needs_clarification": needs_clarification,
        "missing_fields": missing_fields,
        "missing_field_details": details,
        "reason": reason,
        "inferred_intent": inferred_intent,
        "confidence": confidence,
        "suggested_questions": suggested_questions,
        "allow_default_continuation": allow_default_continuation,
        "default_values": default_values,
        "minimum_required_fields": minimum_required_fields,
        "used_context_fields": _used_context_fields(
            selected_paper=selected_paper,
            last_papers=last_papers,
            pending_action=pending_mapping,
            user_memory_summary=user_memory_summary,
            research_profile=research_profile,
            normalized_user_id=normalized_user_id,
            search_spec=search_spec_mapping,
        ),
        "support_status": support_status,
        "resolution_strategy": resolution_strategy,
        "analysis_source": "rule_based_missing_information_analysis",
        "analysis_mode": "deterministic_request_diagnostic",
    }


def request_requires_clarification(diagnostic: Mapping[str, Any]) -> bool:
    return bool(diagnostic.get("needs_clarification"))


def request_allows_default_continuation(diagnostic: Mapping[str, Any]) -> bool:
    return bool(diagnostic.get("allow_default_continuation"))


def _infer_intent(
    message: str,
    *,
    preferred_intent: Optional[str],
    pending_action: Mapping[str, Any],
) -> tuple[str, float]:
    normalized_preferred = _normalize_intent(preferred_intent)
    if normalized_preferred == "unsupported":
        return "unsupported", 0.92
    if normalized_preferred in SUPPORTED_INTENTS and normalized_preferred not in {"unclear", "unsupported"}:
        return normalized_preferred, 0.92
    if _extract_confirmation_decision(message) and _has_pending_confirmation(pending_action):
        return "confirmation", 0.9
    if _extract_confirmation_decision(message):
        return "confirmation", 0.82

    lowered = message.lower()
    if _looks_like_unsupported_request(lowered, message):
        return "unsupported", 0.88
    if _extract_preference_action_kind(message):
        return "preference_action", 0.86
    if _contains_any(lowered, _RECOMMENDATION_TOKENS):
        return "recommendation", 0.84
    if _contains_any(lowered, _SUMMARY_TOKENS):
        return "paper_summary", 0.8
    if _contains_any(lowered, _DETAIL_TOKENS) and _message_mentions_paper(lowered):
        return "paper_detail", 0.78
    if _contains_any(lowered, _SEARCH_TOKENS):
        return "arxiv_search", 0.8
    if _message_mentions_paper(lowered):
        return "paper_qa", 0.72
    return "unclear", 0.45


def _missing_detail(
    field: str,
    reason: str,
    suggested_question: str,
    *,
    required: bool = True,
    can_use_default: bool = False,
    default_value_source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "field": field,
        "reason": reason,
        "suggested_question": suggested_question,
        "required": required,
        "can_use_default": can_use_default,
        "default_value_source": default_value_source,
    }


def _normalize_intent(value: Any) -> Optional[str]:
    text = _normalize_text(value)
    if text is None:
        return None
    if text in SUPPORTED_INTENTS:
        return text
    if text in {"paper", "qa"}:
        return "paper_qa"
    return None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _coerce_paper_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _extract_topic_terms(message: str) -> List[str]:
    if not message:
        return []
    normalized = message
    for token in sorted(_GENERIC_TOPIC_STOPWORDS, key=len, reverse=True):
        # 英文 stopword 只按完整词移除，避免把 RAG 之类主题词拆碎成单字母。
        if re.fullmatch(r"[A-Za-z0-9+_.-]+", token):
            pattern = rf"(?<![A-Za-z0-9+_.-]){re.escape(token)}(?![A-Za-z0-9+_.-])"
        else:
            pattern = re.escape(token)
        normalized = re.sub(pattern, " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"https?://\S+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d{4}\.\d{4,5}(v\d+)?\b", " ", normalized)
    candidates = re.findall(r"[A-Za-z0-9][A-Za-z0-9+_.-]{1,}|[\u4e00-\u9fff]{2,}", normalized)
    terms: List[str] = []
    for item in candidates:
        lowered = item.lower()
        if lowered in _GENERIC_TOPIC_STOPWORDS:
            continue
        if item in {"第一篇", "第二篇", "这篇论文", "那篇论文"}:
            continue
        terms.append(item)
    return _dedupe_strings(terms)


def _has_search_constraints(search_spec: Mapping[str, Any], topic_terms: Sequence[str]) -> bool:
    if topic_terms:
        return True
    if not isinstance(search_spec, Mapping):
        return False
    return any(
        search_spec.get(key) not in (None, "", [], {})
        for key in ("query", "title_query", "abstract_query", "categories")
    )


def _has_question_content(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    if any(token in lowered for token in _QUESTION_HINTS):
        return True
    cleaned = lowered
    for token in sorted(_PAPER_REFERENCE_TOKENS, key=len, reverse=True):
        cleaned = cleaned.replace(token, " ")
    for token in ("paper", "papers", "论文", "这篇", "那篇", "总结", "概述"):
        cleaned = cleaned.replace(token, " ")
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", cleaned)
    filtered = [item for item in candidates if item.lower() not in _GENERIC_TOPIC_STOPWORDS]
    return bool(filtered)


def _message_mentions_paper(message: str) -> bool:
    return (
        _contains_any(message, _PAPER_REFERENCE_TOKENS)
        or "paper" in message
        or "论文" in message
        or "arxiv" in message
        or bool(re.search(r"\b\d{4}\.\d{4,5}(v\d+)?\b", message))
    )


def _message_has_contextual_paper_reference(message: str) -> bool:
    return _contains_any(message.lower(), _PAPER_REFERENCE_TOKENS)


def _resolve_target_paper(message: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    if callable(_resolve_paper_reference):
        resolution = _resolve_paper_reference(message, context)
        if isinstance(resolution, Mapping):
            return dict(resolution)

    arxiv_match = re.search(r"\b\d{4}\.\d{4,5}(v\d+)?\b", message)
    if arxiv_match:
        # 主解析器不可用时也只能保留显式 ID 线索，不能回退 selected_paper 或单条最近论文。
        return {
            "status": "hint_extracted",
            "reference_type": "arxiv_id",
            "value": arxiv_match.group(0),
            "confidence": 0.9,
            "source": "explicit_arxiv_id_fallback",
            "requires_context": False,
            "final_target_resolved": False,
            "reason": None,
        }
    return {
        "status": "unknown",
        "reference_type": "unknown",
        "value": None,
        "confidence": 0.0,
        "source": "fallback_no_reference",
        "requires_context": False,
        "final_target_resolved": False,
        "reason": "当前消息里没有可解析的论文引用线索。",
    }


def _target_paper_reason(resolved_target: Mapping[str, Any], has_contextual_paper_reference: bool) -> str:
    resolution_reason = _normalize_text(resolved_target.get("reason")) if isinstance(resolved_target, Mapping) else None
    if resolution_reason:
        return resolution_reason
    if str(resolved_target.get("status") or "") == "hint_extracted":
        reference_type = str(resolved_target.get("reference_type") or "unknown")
        # 诊断原因要保留“目标论文”这个稳定语义，方便测试和前端不用解析 reference_type。
        return f"已识别到目标论文引用线索（{reference_type}），但还没有结合会话上下文解析成最终论文。"
    if has_contextual_paper_reference:
        return "你在引用“这篇论文/这个”，但当前上下文里没有可直接对应的论文对象。"
    return "当前没有可解析的目标论文，无法继续摘要、细节解释或论文问答。"


def _has_pending_confirmation(pending_action: Mapping[str, Any]) -> bool:
    if not isinstance(pending_action, Mapping) or not pending_action:
        return False
    status = str(pending_action.get("status") or "").strip().lower()
    if status == "cancelled":
        return False
    return bool(
        pending_action.get("step_id")
        or pending_action.get("confirmation_request")
        or str(pending_action.get("type") or "").strip()
    )


def _extract_confirmation_decision(message: str) -> Optional[str]:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return None
    if normalized in _CONFIRM_APPROVE_TOKENS:
        return "approve"
    if normalized in _CONFIRM_REJECT_TOKENS:
        return "reject"
    return None


def _extract_preference_action_kind(message: str) -> Optional[str]:
    lowered = str(message or "").lower()
    if not lowered:
        return None
    if any(token in lowered for token in ("取消喜欢", "取消不喜欢", "撤销喜欢", "撤销不喜欢", "unlike", "remove like", "remove dislike", "取消收藏", "移除")):
        return "remove"
    if any(token in lowered for token in ("不喜欢", "dislike", "thumbs down")):
        return "dislike"
    if any(token in lowered for token in ("喜欢", "like", "favorite", "favourite", "收藏")):
        return "like"
    return None


def _looks_like_unsupported_request(lowered_message: str, raw_message: str) -> bool:
    return any(token in lowered_message or token in raw_message for token in _UNSUPPORTED_HINTS)


def _is_scope_too_broad(
    *,
    inferred_intent: str,
    message: str,
    topic_terms: Sequence[str],
    has_profile_context: bool,
) -> bool:
    if inferred_intent not in {"arxiv_search", "recommendation", "unclear"}:
        return False
    if topic_terms or has_profile_context:
        return False
    lowered = message.lower()
    broad_markers = {
        "推荐几篇论文",
        "recommend papers",
        "related papers",
        "搜一下论文",
        "search papers",
        "最新论文",
        "papers",
        "论文",
    }
    return any(marker in lowered or marker in message for marker in broad_markers)


def _used_context_fields(
    *,
    selected_paper: Mapping[str, Any],
    last_papers: Sequence[Mapping[str, Any]],
    pending_action: Mapping[str, Any],
    user_memory_summary: Any,
    research_profile: Any,
    normalized_user_id: Optional[str],
    search_spec: Mapping[str, Any],
) -> List[str]:
    fields: List[str] = []
    if selected_paper:
        fields.append("selected_paper")
    if last_papers:
        fields.append("last_papers")
    if pending_action:
        fields.append("pending_action")
    if user_memory_summary not in (None, "", [], {}):
        fields.append("user_memory_summary")
    if research_profile not in (None, "", [], {}):
        fields.append("research_profile")
    if normalized_user_id:
        fields.append("user_id")
    if search_spec:
        fields.append("search_spec")
    return fields


def _contains_any(message: str, tokens: Sequence[str]) -> bool:
    return any(token in message for token in tokens)


def _dedupe_strings(items: Sequence[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = _normalize_text(item)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result
