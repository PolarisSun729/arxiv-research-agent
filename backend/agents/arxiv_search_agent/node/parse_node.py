"""解析用户输入，决定本轮请求应该进入哪条 arXiv Agent 流程。

这个模块是整个工作流的入口节点，负责把自然语言请求收敛成可执行状态：
1. 预处理 message、context 和 research_profile；
2. 并行组织 LLM 识别与规则推断；
3. 做 fallback、降级保护和 search spec 清洗；
4. 把最终 intent、plan、warnings、debug 等信息写回 AgentState。

输入/输出说明：
- 输入：原始 AgentState 或兼容 Mapping；
- 输出：携带结构化 intent 与调试信息的 AgentState，供后续节点直接消费。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

from .intent_support import (
    HARD_RULE_PATTERNS,
    LLM_CONFIDENCE_THRESHOLD,
    SEARCH_TRIGGER_PATTERNS,
    SUPPORTED_INTENTS,
    _build_intent_guidance,
    _build_llm_prompt_with_profile,
    _dedupe_preserve_order,
    _looks_search_like,
    _looks_like_paper_detail_request,
    _looks_like_paper_qa_request,
    _looks_like_paper_summary_request,
    _looks_like_preference_action_request,
    _looks_like_reading_list_action_request,
    _looks_like_recommendation_request,
    _validation_error_summary,
)
from ..schemas import ArxivSearchSpec
from ..state import AgentState
from ..utils.search_spec_builder import (
    _apply_rule_enrichment,
    _build_spec_from_rules,
    _normalize_and_validate_spec,
    _post_process_cleaned_spec,
)
from ..utils.state_utils import _append_step, _coerce_state, _compact_search_spec
from ..utils.text_utils import _extract_json_block, _matches_any, _normalize_optional_str, _normalize_text


def _prepare_parse_search_request_input(state: Union[AgentState, Mapping[str, Any]]) -> Dict[str, Any]:
    """把输入 state 规整成 parse 阶段需要的最小字段集合。
    
    主要步骤：
    1. 统一把外部传入对象转成 AgentState；
    2. 归一化 message 文本，避免后续规则和 LLM 处理空白噪声；
    3. 只提取 parse 阶段真正需要的 research_profile，减少后续函数耦合。
    
    输出：返回包含 current_state、message、research_profile 的轻量字典。
    """
    current_state = _coerce_state(state)
    message = _normalize_text(current_state.message or "")
    context = current_state.context if isinstance(current_state.context, Mapping) else {}
    research_profile = context.get("research_profile") if isinstance(context.get("research_profile"), Mapping) else None
    return {
        "current_state": current_state,
        "message": message,
        "research_profile": research_profile,
    }


def _parse_llm_intent(
    message: str,
    generation_service: Optional[Any],
    research_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """调用 LLM 执行意图识别，并要求返回 JSON 载荷。
    
    主要步骤：
    1. 先检查 generation_service 是否可用，缺失时直接返回失败原因；
    2. 使用带研究画像的 prompt 调用 complete_with_qwen；
    3. 对模型输出再次抽取 JSON 块并反序列化，防止夹带解释文本；
    4. 若输出不是 dict，统一视为不可用结果。
    
    输出结构：
    - ok=True 时返回 payload；
    - ok=False 时返回 reason，供上层决定是否 fallback 到规则层。
    """
    if generation_service is None or not hasattr(generation_service, "complete_with_qwen"):
        return {
            "ok": False,
            "reason": "llm service is unavailable",
            "payload": None,
        }

    try:
        # 提示词里会显式约束 schema；这里仍要再做一次 JSON 提取，防止模型夹带解释文本。
        response = generation_service.complete_with_qwen(
            _build_llm_prompt_with_profile(message, research_profile=research_profile),
            task_type="intent_recognition",
        )
        payload = json.loads(_extract_json_block(str(response)))
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"llm parse failed: {exc}",
            "payload": None,
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "reason": "llm output is not a JSON object",
            "payload": None,
        }
    return {"ok": True, "reason": None, "payload": payload}


def _normalize_llm_intent_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """把 LLM 返回的原始 JSON 归一化成后续决策可直接消费的结构。
    
    主要步骤：
    1. 归一化 intent，并把未知值降级为 unsupported；
    2. 尝试解析 confidence，失败时保留为 None；
    3. 仅在 arxiv_search 分支下校验 search_spec，其余 intent 只保留元信息；
    4. 清洗 missing_info、warnings、next_actions 等列表字段。
    
    输出：返回统一的 dict，便于后续决策层同时读取 intent、confidence 和结构化搜索条件。
    """
    intent = str(payload.get("intent", "unsupported") or "unsupported").strip().lower()
    if intent not in SUPPORTED_INTENTS:
        intent = "unsupported"

    confidence_raw = payload.get("confidence")
    confidence: Optional[float]
    try:
        confidence = float(confidence_raw) if confidence_raw is not None and str(confidence_raw).strip() != "" else None
    except Exception:
        confidence = None

    search_spec: Optional[ArxivSearchSpec] = None
    search_spec_payload: Optional[Dict[str, Any]] = None
    # 只有搜索意图才尝试解析结构化 search spec；其他意图只保留基础元信息。
    if intent == "arxiv_search":
        search_spec = _normalize_and_validate_spec(payload)
        search_spec_payload = _compact_search_spec(search_spec)

    missing_info = [str(item).strip() for item in (payload.get("missing_info") or []) if str(item).strip()]
    warnings = [str(item).strip() for item in (payload.get("warnings") or []) if str(item).strip()]
    next_actions = [str(item).strip() for item in (payload.get("next_actions") or []) if str(item).strip()]

    return {
        "intent": intent,
        "confidence": confidence,
        "reasoning_summary": _normalize_optional_str(payload.get("reasoning_summary")),
        "missing_info": missing_info,
        "warnings": warnings,
        "next_actions": next_actions,
        "search_spec": search_spec,
        "search_spec_payload": search_spec_payload,
        "raw": dict(payload),
    }


def _detect_non_search_rule_intent(message: str) -> Optional[str]:
    """用规则快速识别非搜索类意图，避免被误送进 arXiv 搜索链路。
    
    判断顺序按照“论文总结 -> 论文详情 -> 论文 QA -> 推荐 -> 偏好 -> 阅读列表”依次进行，
    只要命中某一类规则或启发式函数就立即返回对应 intent；若都不命中则返回 None。
    """
    if _matches_any(message, HARD_RULE_PATTERNS.get("paper_summary", [])):
        return "paper_summary"
    if _looks_like_paper_summary_request(message):
        return "paper_summary"
    if _matches_any(message, HARD_RULE_PATTERNS.get("paper_detail", [])):
        return "paper_detail"
    if _looks_like_paper_detail_request(message):
        return "paper_detail"
    if _matches_any(message, HARD_RULE_PATTERNS.get("paper_qa", [])):
        return "paper_qa"
    if _looks_like_paper_qa_request(message):
        return "paper_qa"
    if _looks_like_recommendation_request(message):
        return "recommendation"
    if _matches_any(message, HARD_RULE_PATTERNS.get("preference_action", [])):
        return "preference_action"
    if _looks_like_preference_action_request(message):
        return "preference_action"
    if _matches_any(message, HARD_RULE_PATTERNS.get("reading_list_action", [])):
        return "reading_list_action"
    if _looks_like_reading_list_action_request(message):
        return "reading_list_action"
    return None


def _build_rule_decision(message: str) -> Dict[str, Any]:
    """构造纯规则侧的意图判断结果，供 LLM 结果冲突时兜底使用。
    
    主要分支：
    1. 若命中非搜索 intent，直接返回高置信度规则结论；
    2. 若无法抽取搜索条件，则在 unclear 和 unsupported 之间做区分；
    3. 若能抽取搜索条件，则继续做 enrichment 和校验，生成可执行 search_spec。
    
    输出：返回统一的 rule_result 字典，供 fallback 和调试层复用。
    """
    non_search_intent = _detect_non_search_rule_intent(message)
    if non_search_intent is not None:
        plan, next_actions, warnings = _build_intent_guidance(non_search_intent)
        return {
            "intent": non_search_intent,
            "confidence": 0.92,
            "reason": "rule matched non-search intent",
            "source": "rule",
            "search_spec_before_enrichment": None,
            "search_spec_after_enrichment": None,
            "warnings": warnings,
            "next_actions": next_actions,
            "plan": plan,
        }

    # 若不是明显的非搜索请求，则进一步尝试从规则中直接抽取搜索条件。
    search_spec_before = _build_spec_from_rules(message)
    if search_spec_before is None:
        plan, next_actions, warnings = _build_intent_guidance("unclear" if _looks_search_like(message) else "unsupported")
        intent = "unclear" if _looks_search_like(message) else "unsupported"
        return {
            "intent": intent,
            "confidence": 0.48 if intent == "unclear" else 0.35,
            "reason": "rule could not build a concrete search spec",
            "source": "rule",
            "search_spec_before_enrichment": None,
            "search_spec_after_enrichment": None,
            "warnings": warnings,
            "next_actions": next_actions,
            "plan": plan,
        }

    # 规则抽取成功后还会做一次 enrichment，用于补足类别、时间等隐含条件。
    search_spec_after = _apply_rule_enrichment(message, search_spec_before)
    if search_spec_after is None:
        plan, next_actions, warnings = _build_intent_guidance("unclear")
        return {
            "intent": "unclear",
            "confidence": 0.5,
            "reason": "rule search spec failed validation",
            "source": "rule",
            "search_spec_before_enrichment": _compact_search_spec(search_spec_before),
            "search_spec_after_enrichment": None,
            "warnings": warnings + ["rule search spec validation failed"],
            "next_actions": next_actions,
            "plan": plan,
        }

    plan, next_actions, warnings = _build_intent_guidance("arxiv_search")
    return {
        "intent": "arxiv_search",
        "confidence": 0.68,
        "reason": "rule built a search spec",
        "source": "rule",
        "search_spec_before_enrichment": _compact_search_spec(search_spec_before),
        "search_spec_after_enrichment": _compact_search_spec(search_spec_after),
        "warnings": warnings,
        "next_actions": next_actions,
        "plan": plan,
        "search_spec": search_spec_after,
    }


def _build_debug_payload(
    *,
    message: str,
    final_intent: str,
    intent_source: str,
    llm_result: Optional[Dict[str, Any]],
    rule_result: Optional[Dict[str, Any]],
    hard_rule_result: Optional[Dict[str, Any]],
    fallback_reason: Optional[str],
    final_search_spec: Optional[ArxivSearchSpec],
    search_spec_before_enrichment: Optional[Dict[str, Any]],
    search_spec_after_enrichment: Optional[Dict[str, Any]],
    warnings: Sequence[str],
    next_actions: Sequence[str],
    cleaning_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """汇总 parse 阶段的重要中间信息，方便调试和前端解释执行过程。
    
    这个函数会把 LLM、规则、fallback、search_spec 清洗前后结果统一打包，
    并在有 cleaning_debug 时补充 final_query / cleaned_topic_* 等字段，帮助排查意图识别分歧。
    """
    payload: Dict[str, Any] = {
        "original_message": message,
        "final_intent": final_intent,
        "intent_source": intent_source,
        "llm_result": llm_result,
        "rule_result": rule_result,
        "hard_rule_result": hard_rule_result,
        "llm_confidence": llm_result.get("confidence") if isinstance(llm_result, dict) else None,
        "fallback_reason": fallback_reason,
        "final_search_spec": _compact_search_spec(final_search_spec),
        "search_spec_before_enrichment": search_spec_before_enrichment,
        "search_spec_after_enrichment": search_spec_after_enrichment,
        "warnings": list(warnings),
        "next_actions": list(next_actions),
    }
    if cleaning_debug:
        payload["cleaned_topic_cn"] = cleaning_debug.get("cleaned_topic_cn")
        payload["cleaned_topic_en"] = cleaning_debug.get("cleaned_topic_en")
        payload["final_query"] = cleaning_debug.get("final_query")
        payload["final_title_query"] = cleaning_debug.get("final_title_query")
        payload["final_abstract_query"] = cleaning_debug.get("final_abstract_query")
    return payload


def _run_parse_search_request_detectors(
    message: str,
    *,
    generation_service: Optional[Any] = None,
    research_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """组织 LLM 检测和规则检测，并统一收集 warning / fallback 信息。
    
    主要步骤：
    1. 调用 LLM 解析并在成功时归一化 payload；
    2. 若 LLM 输出结构不合法，则记录 schema 失败原因并清空 llm_result；
    3. 无论 LLM 是否可用，始终执行规则检测，保证后续具备 fallback 基线；
    4. 额外提取 cleaned_topic_cn/en 等调试字段，供后续搜索词清洗使用。
    
    输出：返回 llm_result、rule_result、warnings、fallback_reason 和 cleaning_debug。
    """
    warnings: List[str] = []
    fallback_reason: Optional[str] = None
    llm_result: Optional[Dict[str, Any]] = None
    cleaning_debug: Dict[str, Any] = {}

    llm_payload_result = _parse_llm_intent(
        message,
        generation_service=generation_service,
        research_profile=research_profile,
    )
    if llm_payload_result.get("ok"):
        try:
            # 先把模型原始 JSON 规范化；只有规范化成功的结果才允许进入最终决策阶段。
            llm_result = _normalize_llm_intent_payload(llm_payload_result["payload"])
            warnings.extend(str(item) for item in (llm_result.get("warnings") or []) if str(item).strip())
            raw_payload = llm_payload_result.get("payload")
            if isinstance(raw_payload, dict):
                cleaning_debug["cleaned_topic_cn"] = _normalize_optional_str(raw_payload.get("cleaned_topic_cn"))
                cleaning_debug["cleaned_topic_en"] = _normalize_optional_str(raw_payload.get("cleaned_topic_en"))
        except ValidationError as exc:
            # 模型给出 intent 但结构不合法时，不能信任该结果，必须退回规则层。
            llm_result = None
            fallback_reason = f"llm output failed schema validation: {_validation_error_summary(exc)}"
            warnings.append(fallback_reason)
    else:
        fallback_reason = str(llm_payload_result.get("reason") or "llm unavailable")
        warnings.append(fallback_reason)

    return {
        "llm_result": llm_result,
        "rule_result": _build_rule_decision(message),
        "fallback_reason": fallback_reason,
        "warnings": warnings,
        "cleaning_debug": cleaning_debug,
    }


def _build_parse_search_request_rule_fallback(
    *,
    rule_result: Optional[Dict[str, Any]],
    warnings: List[str],
    default_intent: str = "unsupported",
) -> Dict[str, Any]:
    """把规则判断结果包装成统一的 fallback 决策结构。
    
    这个函数不重新做规则推断，只负责：
    1. 合并 rule_result 与已有 warnings；
    2. 补齐 intent_source、plan、next_actions、search_spec 等公共字段；
    3. 让不同 fallback 场景都返回同一套结构，降低后续分支复杂度。
    """
    resolved = rule_result or {}
    warnings.extend(str(item) for item in (resolved.get("warnings") or []) if str(item).strip())
    return {
        "intent": str(resolved.get("intent") or default_intent),
        "intent_source": "fallback",
        "search_spec": resolved.get("search_spec"),
        "search_spec_before_enrichment": resolved.get("search_spec_before_enrichment"),
        "search_spec_after_enrichment": resolved.get("search_spec_after_enrichment"),
        "plan": list(resolved.get("plan") or []),
        "next_actions": list(resolved.get("next_actions") or []),
        "warnings": warnings,
    }


def _decide_parse_search_request_intent(
    *,
    message: str,
    llm_result: Optional[Dict[str, Any]],
    rule_result: Optional[Dict[str, Any]],
    fallback_reason: Optional[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    """在 LLM 结果和规则结果之间做最终意图决策。
    
    核心原则：
    1. 高置信度、结构合法的 LLM 结果优先；
    2. confidence 缺失、search_spec 非法、规则冲突严重或置信度过低时回退到规则层；
    3. recommendation 等特殊 intent 还会做额外启发式校验，防止模型泛化误判。
    
    输出：返回最终 intent、intent_source、search_spec、warnings、fallback_reason 等决策字段。
    """
    decision_warnings = [str(item) for item in warnings if str(item).strip()]
    intent = "unsupported"
    intent_source = "fallback"
    search_spec: Optional[ArxivSearchSpec] = None
    plan: List[str] = []
    next_actions: List[str] = []
    search_spec_before_enrichment: Optional[Dict[str, Any]] = None
    search_spec_after_enrichment: Optional[Dict[str, Any]] = None

    # 完全拿不到可用的 LLM 结果时，直接回落到规则层。
    # 没有 LLM 结果时，整条链路必须完全依赖规则层产出的 fallback 结构。
    if llm_result is None:
        fallback = _build_parse_search_request_rule_fallback(
            rule_result=rule_result,
            warnings=decision_warnings,
            default_intent="unsupported",
        )
        if fallback_reason is None:
            fallback_reason = str((rule_result or {}).get("reason") or "llm unavailable, rule fallback used")
        fallback["fallback_reason"] = fallback_reason
        return fallback

    llm_intent = str(llm_result.get("intent") or "unsupported")
    llm_confidence = llm_result.get("confidence")
    confidence_value = float(llm_confidence) if isinstance(llm_confidence, (int, float)) else None

    # 没有 confidence 的模型输出不可直接信任，因为后续无法判断是否该触发兜底。
    if confidence_value is None:
        fallback_reason = "llm confidence missing"
        decision_warnings.append(fallback_reason)
        fallback = _build_parse_search_request_rule_fallback(
            rule_result=rule_result,
            warnings=decision_warnings,
            default_intent=llm_intent,
        )
        fallback["fallback_reason"] = fallback_reason
        return fallback

    # 搜索 intent 需要比非搜索 intent 更严格的 search_spec 校验与 enrichment。
    if llm_intent == "arxiv_search":
        search_spec = llm_result.get("search_spec")
        search_spec_before_enrichment = llm_result.get("search_spec_payload")
        if search_spec is None:
            fallback_reason = "llm search intent is missing a valid search spec"
            decision_warnings.append(fallback_reason)
            fallback = _build_parse_search_request_rule_fallback(
                rule_result=rule_result,
                warnings=decision_warnings,
                default_intent="unclear",
            )
            fallback["fallback_reason"] = fallback_reason
            return fallback

        # 先利用 cleaned_topic_* 做一次语义清洗，再叠加规则 enrichment，双重提高可执行性。
        search_spec, post_warnings = _post_process_cleaned_spec(
            search_spec,
            message,
            cleaned_topic_cn=_normalize_optional_str((llm_result.get("raw") or {}).get("cleaned_topic_cn")),
            cleaned_topic_en=_normalize_optional_str((llm_result.get("raw") or {}).get("cleaned_topic_en")),
        )
        decision_warnings.extend(post_warnings)
        search_spec_after_enrichment = _compact_search_spec(search_spec)
        # LLM 给出的 search spec 往往缺少隐含类别、时间或语义修正，因此再叠加一次规则 enrichment。
        enriched_spec = _apply_rule_enrichment(message, search_spec)
        if enriched_spec is None:
            fallback_reason = "rule enrichment failed after llm search parse"
            decision_warnings.append(fallback_reason)
            fallback = _build_parse_search_request_rule_fallback(
                rule_result=rule_result,
                warnings=decision_warnings,
                default_intent="unclear",
            )
            fallback["fallback_reason"] = fallback_reason
            return fallback

        if confidence_value < LLM_CONFIDENCE_THRESHOLD:
            fallback_reason = f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
            decision_warnings.append(fallback_reason)
            fallback = _build_parse_search_request_rule_fallback(
                rule_result=rule_result,
                warnings=decision_warnings,
                default_intent="arxiv_search",
            )
            fallback["fallback_reason"] = fallback_reason
            return fallback

        intent = "arxiv_search"
        intent_source = "llm"
        search_spec = enriched_spec
        search_spec_after_enrichment = _compact_search_spec(enriched_spec)
        plan, next_actions, intent_warnings = _build_intent_guidance(intent)
        decision_warnings.extend(intent_warnings)
        # LLM 和规则结论不一致时不立即推翻 LLM，但会把冲突写进 warnings 供后续排查。
        if rule_result and str(rule_result.get("intent") or "") != "arxiv_search":
            decision_warnings.append(f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}")
    else:
        # recommendation 等非搜索意图会再过一层启发式检查，防止泛化误判。
        if llm_intent == "recommendation" and not _looks_like_recommendation_request(message):
            fallback_reason = "recommendation intent downgraded because request does not look personalized"
            decision_warnings.append(fallback_reason)
            fallback = _build_parse_search_request_rule_fallback(
                rule_result=rule_result,
                warnings=decision_warnings,
                default_intent="arxiv_search",
            )
            fallback["fallback_reason"] = fallback_reason
            return fallback

        if confidence_value < LLM_CONFIDENCE_THRESHOLD:
            fallback_reason = f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
            decision_warnings.append(fallback_reason)
            fallback = _build_parse_search_request_rule_fallback(
                rule_result=rule_result,
                warnings=decision_warnings,
                default_intent=llm_intent,
            )
            fallback["fallback_reason"] = fallback_reason
            return fallback

        intent = llm_intent
        intent_source = "llm"
        plan, next_actions, intent_warnings = _build_intent_guidance(llm_intent)
        decision_warnings.extend(intent_warnings)
        if rule_result and str(rule_result.get("intent") or "") != llm_intent:
            decision_warnings.append(f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}")

    return {
        "intent": intent,
        "intent_source": intent_source,
        "search_spec": search_spec,
        "fallback_reason": fallback_reason,
        "warnings": decision_warnings,
        "plan": plan,
        "next_actions": next_actions,
        "search_spec_before_enrichment": search_spec_before_enrichment,
        "search_spec_after_enrichment": search_spec_after_enrichment,
    }


def _finalize_parse_search_request_decision(
    *,
    intent: str,
    search_spec: Optional[ArxivSearchSpec],
    plan: Sequence[str],
    next_actions: Sequence[str],
    warnings: Sequence[str],
    fallback_reason: Optional[str],
    cleaning_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """补齐 plan / next_actions，并对最终决策做最后的降级保护。
    
    主要职责：
    1. 给缺失的展示字段补默认 guidance；
    2. 对“search intent 但没有合法 search_spec”的异常情况做最终降级；
    3. 把最终 query / title_query / abstract_query 写入调试信息，便于后续回放。
    """
    finalized_plan = list(plan or [])
    finalized_next_actions = list(next_actions or [])
    finalized_warnings = [str(item) for item in warnings if str(item).strip()]
    finalized_debug = dict(cleaning_debug or {})

    # 某些 fallback 分支只返回 intent，此处统一补齐展示层需要的解释信息。
    if not finalized_plan or not finalized_next_actions:
        default_plan, default_next_actions, intent_warnings = _build_intent_guidance(intent)
        if not finalized_plan:
            finalized_plan = default_plan
        if not finalized_next_actions:
            finalized_next_actions = default_next_actions
        finalized_warnings.extend(intent_warnings)

    # 搜索 intent 不能脱离 search_spec 独立存在，否则后续节点无法构造工具参数。
    if intent == "arxiv_search" and search_spec is None:
        intent = "unclear"
        downgrade_plan, downgrade_next_actions, intent_warnings = _build_intent_guidance("unclear")
        finalized_plan = downgrade_plan
        finalized_next_actions = downgrade_next_actions
        finalized_warnings.extend(intent_warnings)
        if fallback_reason is None:
            fallback_reason = "search intent was downgraded because no valid search spec was produced"

    if search_spec is not None:
        finalized_debug["final_query"] = search_spec.query
        finalized_debug["final_title_query"] = search_spec.title_query
        finalized_debug["final_abstract_query"] = search_spec.abstract_query

    return {
        "intent": intent,
        "search_spec": search_spec,
        "plan": finalized_plan,
        "next_actions": finalized_next_actions,
        "warnings": _dedupe_preserve_order(finalized_warnings),
        "fallback_reason": fallback_reason,
        "cleaning_debug": finalized_debug,
    }


def _write_parse_search_request_state(
    *,
    current_state: AgentState,
    message: str,
    intent: str,
    intent_source: str,
    fallback_reason: Optional[str],
    llm_result: Optional[Dict[str, Any]],
    rule_result: Optional[Dict[str, Any]],
    hard_rule_result: Optional[Dict[str, Any]],
    search_spec: Optional[ArxivSearchSpec],
    plan: Sequence[str],
    warnings: Sequence[str],
    next_actions: Sequence[str],
    search_spec_before_enrichment: Optional[Dict[str, Any]],
    search_spec_after_enrichment: Optional[Dict[str, Any]],
    cleaning_debug: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """把 parse 决策结果落回 AgentState，并追加统一 step trace。
    
    写回内容包括 intent、search_spec、warnings、plan、next_actions、debug 等字段。
    如果存在 fallback_reason 或清洗后的搜索条件，也会一并写入，保证后续节点和调试层看到的是同一份最终状态。
    """
    normalized_state = current_state.model_copy(deep=True)
    normalized_state.intent = intent
    normalized_state.intent_source = intent_source
    normalized_state.fallback_reason = fallback_reason
    normalized_state.llm_confidence = (
        float(llm_result.get("confidence"))
        if llm_result and isinstance(llm_result.get("confidence"), (int, float))
        else None
    )
    normalized_state.search_spec = search_spec
    normalized_state.plan = list(plan)
    normalized_state.warnings = _dedupe_preserve_order([str(item) for item in warnings if str(item).strip()])
    normalized_state.next_actions = list(next_actions)
    # parse 成功后要清空上一轮搜索执行遗留，确保后续节点只基于本轮决策工作。
    normalized_state.search_retry_count = 0
    normalized_state.fallback_specs = []
    normalized_state.tool_name = None
    normalized_state.tool_args = {}
    normalized_state.tool_result = None
    normalized_state.tool_calls = []
    normalized_state.papers = []
    normalized_state.answer = None
    normalized_state.errors = []
    normalized_state.preference_action_result = None
    normalized_state.debug = _build_debug_payload(
        message=message,
        final_intent=intent,
        intent_source=intent_source,
        llm_result=llm_result,
        rule_result=rule_result,
        hard_rule_result=hard_rule_result,
        fallback_reason=fallback_reason,
        final_search_spec=search_spec,
        search_spec_before_enrichment=search_spec_before_enrichment,
        search_spec_after_enrichment=search_spec_after_enrichment,
        warnings=normalized_state.warnings,
        next_actions=normalized_state.next_actions,
        cleaning_debug=cleaning_debug,
    )
    return _append_step(
        normalized_state,
        step="intent_recognition",
        status="success",
        action="识别用户意图并决定要进入什么流程",
        inputs={"message": message},
        outputs={
            "intent": intent,
            "intent_source": intent_source,
            "llm_confidence": normalized_state.llm_confidence,
            "fallback_reason": fallback_reason,
            "search_spec": _compact_search_spec(search_spec),
            "plan": list(plan),
            "warnings": list(normalized_state.warnings),
            "next_actions": list(next_actions),
            "debug": normalized_state.debug,
        },
    )


def parse_search_request(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    """执行 parse 节点主流程，把自然语言请求转成结构化 AgentState。
    
    主流程分为四步：
    1. 预处理输入并抽取 message / research_profile；
    2. 运行 LLM 与规则检测，收集候选结果；
    3. 做最终 intent 决策与降级保护；
    4. 把结果写回 state，并记录调试信息和 step trace。
    
    输出：返回可直接驱动后续 search / reading / preference 节点的 AgentState。
    """
    prepared = _prepare_parse_search_request_input(state)
    current_state = prepared["current_state"]
    message = prepared["message"]
    research_profile = prepared["research_profile"]

    hard_rule_result: Optional[Dict[str, Any]] = None
    detector_result = _run_parse_search_request_detectors(
        message,
        generation_service=generation_service,
        research_profile=research_profile,
    )
    llm_result = detector_result["llm_result"]
    rule_result = detector_result["rule_result"]

    # 先做“选哪条链路”，再做“如何把结果补齐成可执行状态”的最终收口。
    decision = _decide_parse_search_request_intent(
        message=message,
        llm_result=llm_result,
        rule_result=rule_result,
        fallback_reason=detector_result["fallback_reason"],
        warnings=detector_result["warnings"],
    )
    # 统一做最终降级和展示字段补齐，避免后续节点拿到半成品决策。
    finalized = _finalize_parse_search_request_decision(
        intent=decision["intent"],
        search_spec=decision["search_spec"],
        plan=decision["plan"],
        next_actions=decision["next_actions"],
        warnings=decision["warnings"],
        fallback_reason=decision["fallback_reason"],
        cleaning_debug=detector_result["cleaning_debug"],
    )

    return _write_parse_search_request_state(
        current_state=current_state,
        message=message,
        intent=finalized["intent"],
        intent_source=decision["intent_source"],
        fallback_reason=finalized["fallback_reason"],
        llm_result=llm_result,
        rule_result=rule_result,
        hard_rule_result=hard_rule_result,
        search_spec=finalized["search_spec"],
        plan=finalized["plan"],
        warnings=finalized["warnings"],
        next_actions=finalized["next_actions"],
        search_spec_before_enrichment=decision["search_spec_before_enrichment"],
        search_spec_after_enrichment=decision["search_spec_after_enrichment"],
        cleaning_debug=finalized["cleaning_debug"],
    )


__all__ = ["parse_search_request"]
