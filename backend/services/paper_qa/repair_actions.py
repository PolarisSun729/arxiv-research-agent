from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

RETRY_WITH_QUERY_REWRITE = "retry_with_query_rewrite"
RETRY_WITH_SECTION_FOCUS = "retry_with_section_focus"
RETRY_WITH_EXPANDED_CONTEXT = "retry_with_expanded_context"
RETRY_WITHOUT_HYDE = "retry_without_hyde"
RETRY_WITH_KEYWORD_EMPHASIS = "retry_with_keyword_emphasis"
ASK_USER_TO_REBUILD_INDEX = "ask_user_to_rebuild_index"
ASK_CLARIFICATION = "ask_clarification"

PAPER_QA_REPAIR_ACTION_ORDER = (
    ASK_USER_TO_REBUILD_INDEX,
    RETRY_WITH_SECTION_FOCUS,
    RETRY_WITH_EXPANDED_CONTEXT,
    RETRY_WITH_KEYWORD_EMPHASIS,
    RETRY_WITH_QUERY_REWRITE,
    RETRY_WITHOUT_HYDE,
    ASK_CLARIFICATION,
)

PAPER_QA_REPAIR_ACTIONS: Dict[str, Dict[str, Any]] = {
    RETRY_WITH_QUERY_REWRITE: {
        "applicable_conditions": ["retrieval_quality:failed", "retrieval_quality:weak", "rewrite_query_count:0"],
        "input_params": {"enable_query_rewrite": True, "top_k_min": 20, "stricter_grounding": True, "debug": True},
        "max_attempts": 1,
        "fallback": "fallback_answer_with_observation_reason",
    },
    RETRY_WITH_SECTION_FOCUS: {
        "applicable_conditions": ["weak_source_reason:section_mismatch", "missing_evidence_type:method_flow|experiment_setup|result_table|limitation"],
        "input_params": {"enable_query_rewrite": True, "enable_context_expansion": True, "section_focus": True, "top_k_min": 20},
        "max_attempts": 1,
        "fallback": "fallback_answer_with_section_mismatch_reason",
    },
    RETRY_WITH_EXPANDED_CONTEXT: {
        "applicable_conditions": ["missing_evidence_type:cross_paragraph_evidence", "weak_source_reason:evidence_too_short", "answer_insufficient_evidence:yes"],
        "input_params": {"enable_context_expansion": True, "top_k_min": 24, "stricter_grounding": True},
        "max_attempts": 1,
        "fallback": "fallback_answer_with_insufficient_evidence_reason",
    },
    RETRY_WITHOUT_HYDE: {
        "applicable_conditions": ["hyde:success_with_weak_retrieval", "hyde:fallback", "weak_source_reason:context_noise_contamination"],
        "input_params": {"enable_hyde": False, "enable_query_rewrite": True, "debug": True},
        "max_attempts": 1,
        "fallback": "fallback_answer_with_noise_reason",
    },
    RETRY_WITH_KEYWORD_EMPHASIS: {
        "applicable_conditions": ["question_mentions:formula|table|figure|metric", "missing_evidence_type:formula_derivation|result_table"],
        "input_params": {"enable_keyword_search": True, "keyword_emphasis": True, "top_k_min": 20},
        "max_attempts": 1,
        "fallback": "fallback_answer_with_keyword_retrieval_reason",
    },
    ASK_USER_TO_REBUILD_INDEX: {
        "applicable_conditions": ["error_code:qa_index_not_found", "chunk_quality:empty_or_abnormal"],
        "input_params": {"requires_confirmation": True},
        "max_attempts": 1,
        "fallback": "fallback_answer_without_rebuild",
    },
    ASK_CLARIFICATION: {
        "applicable_conditions": ["missing_evidence_type:unknown", "question_context:ambiguous_or_followup_without_context"],
        "input_params": {"requires_user_input": True},
        "max_attempts": 1,
        "fallback": "fallback_answer_with_clarification_needed",
    },
}

_LEGACY_REPAIR_ACTION_ALIASES = {
    "retry_with_rewrite": RETRY_WITH_QUERY_REWRITE,
    "expand_context": RETRY_WITH_EXPANDED_CONTEXT,
    "prefer_method_sections": RETRY_WITH_SECTION_FOCUS,
    "disable_hyde": RETRY_WITHOUT_HYDE,
    "rebuild_index": ASK_USER_TO_REBUILD_INDEX,
    "ask_user_clarification": ASK_CLARIFICATION,
}


def normalize_repair_action(value: Any) -> str:
    """把历史 action 名和外部输入归一到受控协议名，避免修复策略在各层散落。"""
    action = str(value or "").strip()
    if not action:
        return ""
    return _LEGACY_REPAIR_ACTION_ALIASES.get(action, action if action in PAPER_QA_REPAIR_ACTIONS else "")


def normalize_repair_actions(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values or []:
        action = normalize_repair_action(value)
        if action and action not in seen:
            seen.add(action)
            result.append(action)
    return sorted(result, key=lambda item: PAPER_QA_REPAIR_ACTION_ORDER.index(item) if item in PAPER_QA_REPAIR_ACTION_ORDER else 999)


def describe_repair_actions(actions: Iterable[Any]) -> List[Dict[str, Any]]:
    """返回 action 的适用条件、输入参数和 fallback，供 Agent trace 直接解释修复决策。"""
    details: List[Dict[str, Any]] = []
    for action in normalize_repair_actions(actions):
        spec = dict(PAPER_QA_REPAIR_ACTIONS.get(action) or {})
        details.append(
            {
                "action": action,
                "applicable_conditions": list(spec.get("applicable_conditions") or []),
                "input_params": dict(spec.get("input_params") or {}),
                "max_attempts": int(spec.get("max_attempts") or 1),
                "fallback": str(spec.get("fallback") or "fallback_answer"),
            }
        )
    return details


def repair_action_input_params(actions: Iterable[Any]) -> Dict[str, Any]:
    """合并多个 repair action 的执行参数；后续工具层只读取这个受控摘要。"""
    params: Dict[str, Any] = {}
    for detail in describe_repair_actions(actions):
        action_params = dict(detail.get("input_params") or {})
        top_k_min = action_params.pop("top_k_min", None)
        if top_k_min is not None:
            params["top_k_min"] = max(int(params.get("top_k_min") or 0), int(top_k_min))
        params.update(action_params)
    return params


def build_repair_strategy_payload(
    *,
    actions: Iterable[Any],
    reason: str = "",
    observation: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """把质量门选择的动作封装成 retry step 的稳定输入，保留原始 observation 便于 trace 回放。"""
    normalized_actions = normalize_repair_actions(actions)
    action_details = describe_repair_actions(normalized_actions)
    return {
        "strategy_name": "paper_qa_controlled_repair",
        "repair_actions": normalized_actions,
        "repair_action_details": action_details,
        "input_params": repair_action_input_params(normalized_actions),
        "reason": reason or "paper_qa_repair",
        "max_attempts": max((int(item.get("max_attempts") or 1) for item in action_details), default=1),
        "fallback": next((str(item.get("fallback") or "") for item in action_details if item.get("fallback")), "fallback_answer"),
        "observation": dict(observation or {}),
    }
