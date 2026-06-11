from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


_FALLBACK_CODE_CATALOG: Dict[str, Dict[str, str]] = {
    "tool_aware_planner_disabled": {
        "stage": "planner",
        "category": "planner_disabled",
        "message": "Tool-Aware 主 planner 已被配置关闭，已进入 legacy 模板兜底。",
    },
    "llm_planner_disabled": {
        "stage": "planner",
        "category": "llm_planner_disabled",
        "message": "LLM Planner 已关闭，本轮不会尝试 LLM 草稿规划。",
    },
    "llm_planner_validation_failed": {
        "stage": "planner",
        "category": "llm_planner_validation_failed",
        "message": "LLM Planner 输出未通过结构化校验，已进入安全回退。",
    },
    "tool_aware_planner_failed": {
        "stage": "planner",
        "category": "tool_aware_planner_failed",
        "message": "Tool-Aware 规则 planner 生成计划失败，已进入兜底路径。",
    },
    "missing_required_context": {
        "stage": "planner",
        "category": "missing_required_context",
        "message": "planner 缺少必要上下文，无法继续主规划路径。",
    },
    "planner_candidate_tools_empty": {
        "stage": "planner",
        "category": "planner_input_missing",
        "message": "planner 缺少可用工具候选，无法继续主规划路径。",
    },
    "llm_plan_invalid": {
        "stage": "planner",
        "category": "llm_planner_validation_failed",
        "message": "LLM Planner 输出未通过结构化校验，已进入安全回退。",
    },
    "rule_planner_failed": {
        "stage": "planner",
        "category": "rule_planner_failed",
        "message": "规则型 planner 生成计划失败，已进入兜底路径。",
    },
    "template_fallback_used": {
        "stage": "planner",
        "category": "template_fallback",
        "message": "正式 planner 不可用，使用最小模板兜底计划。",
    },
    "unsupported_request": {
        "stage": "planner",
        "category": "unsupported_request",
        "message": "请求不在当前 Agent 支持范围内，返回安全兜底回复。",
    },
    "replan_limit_exceeded": {
        "stage": "replan",
        "category": "replan_limit",
        "message": "重规划次数已达上限，停止继续自动修复。",
    },
    "recovery_patch_failed": {
        "stage": "replan",
        "category": "recovery_patch_failed",
        "message": "恢复动作未能成功修改计划，已进入兜底回复。",
    },
    "unsupported_patch_strategy": {
        "stage": "replan",
        "category": "unsupported_patch_strategy",
        "message": "恢复策略未实现对应 patch 逻辑，已降级到 fallback。",
    },
    "clarification_required": {
        "stage": "clarification",
        "category": "clarification_required",
        "message": "当前信息不足以安全继续执行，需要先向用户澄清。",
    },
    "recommendation_cold_start": {
        "stage": "recommendation",
        "category": "context_cold_start",
        "message": "缺少稳定长期画像，推荐已降级为当前请求驱动的冷启动路径。",
    },
    "recommendation_context_fallback": {
        "stage": "recommendation",
        "category": "context_fallback",
        "message": "推荐链路因上下文或兴趣向量不可用而退化为保守模式。",
    },
    "executor_fallback_answer": {
        "stage": "executor",
        "category": "fallback_answer",
        "message": "执行阶段无法继续自动恢复，已输出最终兜底回答。",
    },
}


def _normalize_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def classify_fallback_code(reason: Any) -> str:
    """把分散的历史 fallback 字符串收敛到稳定 code，避免每个模块各写一套含义接近的 reason。"""
    text = _normalize_text(reason) or "unknown_fallback"
    if text.startswith("tool_aware_planner_disabled"):
        return "tool_aware_planner_disabled"
    if text.startswith("llm_planner_disabled"):
        return "llm_planner_disabled"
    if text.startswith("llm_planner_validation_failed"):
        return "llm_planner_validation_failed"
    if text.startswith("tool_aware_planner_failed"):
        return "tool_aware_planner_failed"
    if text.startswith("missing_required_context"):
        return "missing_required_context"
    if text == "unsupported_goal":
        return "unsupported_request"
    if text == "planner_context_candidate_tools_empty":
        return "planner_candidate_tools_empty"
    if text.startswith("replan_limit_exceeded:"):
        return "replan_limit_exceeded"
    if text.startswith("recovery_patch_failed:"):
        return "recovery_patch_failed"
    if text.startswith("unsupported_patch_strategy:"):
        return "unsupported_patch_strategy"
    if text in {"clarification_required", "missing_required_input"}:
        return "clarification_required"
    if "llm" in text and ("invalid" in text or "failed" in text or "validation" in text):
        return "llm_planner_validation_failed"
    if "template" in text and "fallback" in text:
        return "template_fallback_used"
    if text in {"message_driven_recommendation", "context_driven_recommendation"}:
        return "recommendation_context_fallback"
    return text


def build_fallback_record(
    reason: Any,
    *,
    stage: Optional[str] = None,
    source: Optional[str] = None,
    detail: Optional[Mapping[str, Any]] = None,
    resolution: Optional[str] = None,
) -> Dict[str, Any]:
    raw_reason = _normalize_text(reason)
    normalized_source = _normalize_text(source)
    code = classify_fallback_code(reason)
    catalog_entry = dict(_FALLBACK_CODE_CATALOG.get(code) or {})
    fallback_stage = stage or catalog_entry.get("stage") or "unknown"

    # 某些历史路径仍会把底层校验错误直接透传为自由文本 reason。
    # 这里按层级和来源再次归一化，保证 trace/debug 能稳定分辨主路径与兜底路径。
    if normalized_source == "LLMPlanDraftGenerator":
        code = "llm_planner_validation_failed"
        catalog_entry = dict(_FALLBACK_CODE_CATALOG.get(code) or {})
        fallback_stage = stage or catalog_entry.get("stage") or fallback_stage
    elif normalized_source in {"fixed_template_fallback", "legacy_template_fallback"}:
        # source 只说明最终由 legacy 模板兜底；code 仍优先保留“为什么 fallback”的触发原因。
        # 只有无法归类的自由文本才退回 template_fallback_used，避免 trace 丢失 planner 失败类别。
        if code not in _FALLBACK_CODE_CATALOG:
            code = "template_fallback_used"
        catalog_entry = dict(_FALLBACK_CODE_CATALOG.get(code) or {})
        fallback_stage = stage or catalog_entry.get("stage") or fallback_stage
    elif fallback_stage == "executor" and raw_reason == "fallback_answer":
        code = "executor_fallback_answer"
        catalog_entry = dict(_FALLBACK_CODE_CATALOG.get(code) or {})

    return {
        "code": code,
        "stage": fallback_stage,
        "category": catalog_entry.get("category") or "generic_fallback",
        "message": catalog_entry.get("message") or (raw_reason or "发生了未分类 fallback"),
        "raw_reason": raw_reason,
        "source": normalized_source or fallback_stage,
        "resolution": _normalize_text(resolution) or "fallback",
        "detail": dict(detail or {}),
    }


def fallback_reason_text(record: Optional[Mapping[str, Any]], default: Any = None) -> Optional[str]:
    if isinstance(record, Mapping):
        return _normalize_text(record.get("raw_reason")) or _normalize_text(record.get("code")) or _normalize_text(default)
    return _normalize_text(default)
