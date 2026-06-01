from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from pydantic import ValidationError

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:  # pragma: no cover - import path differs between backend cwd and package import
    from tools.tool_registry import invoke_tool
except ModuleNotFoundError:  # pragma: no cover
    from backend.tools.tool_registry import invoke_tool

try:  # pragma: no cover - import path differs between backend cwd and package import
    from dependencies import get_recommendation_service
except ModuleNotFoundError:  # pragma: no cover
    from backend.dependencies import get_recommendation_service

from .schemas import AgentStep, AgentToolCall, ArxivSearchSpec, get_default_agent_arxiv_categories, get_valid_arxiv_categories
from .state import AgentState

SEARCH_TOOL_NAME = "search_arxiv_structured"
SUPPORTED_INTENTS = {
    "arxiv_search",
    "paper_detail",
    "paper_summary",
    "paper_qa",
    "recommendation",
    "preference_action",
    "reading_list_action",
    "unclear",
    "unsupported",
}
SEARCH_INTENTS = {"arxiv_search"}
NON_SEARCH_INTENTS = {
    "paper_detail",
    "paper_summary",
    "paper_qa",
    "recommendation",
    "preference_action",
    "reading_list_action",
}
LLM_CONFIDENCE_THRESHOLD = 0.55

HARD_RULE_PATTERNS: Dict[str, Sequence[str]] = {
    "paper_summary": (
        r"总结这篇论文",
        r"概述这篇论文",
        r"帮我总结.*这篇",
        r"讲讲这篇论文",
        r"这篇论文讲了什么",
        r"(第一篇|这篇|这几篇).*?(讲了什么|讲什么|概述|总结)",
    ),
    "paper_detail": (
        r"解释这篇论文的方法",
        r"这篇论文的方法",
        r"第一篇讲了什么",
        r"这篇论文做了什么",
        r"介绍这篇论文",
        r"(第一篇|这篇|这几篇).*?(方法|细节|原理|流程|怎么做)",
    ),
    "paper_qa": (
        r"问一下这篇论文",
        r"这篇论文.*是否",
        r"这篇论文.*为什么",
        r"关于这篇论文",
        r"(第一篇|这篇|这几篇).*?(问|提问|QA|question)",
    ),
    "preference_action": (
        r"加入收藏",
        r"加入待读",
        r"标记我喜欢",
        r"标记.*不喜欢",
        r"收藏这篇",
        r"喜欢第一篇",
        r"不喜欢第一篇",
        r"取消.*标记",
        r"取消.*收藏",
        r"取消.*喜欢",
        r"撤销.*标记",
        r"撤销.*收藏",
        r"撤销.*喜欢",
    ),
    "reading_list_action": (
        r"查看我的收藏",
        r"我的收藏",
        r"阅读列表",
        r"reading list",
        r"收藏夹",
        r"我的收藏有哪些",
        r"阅读列表有哪些",
    ),
}

LLM_INTENT_HINTS: Dict[str, Sequence[str]] = {
    "paper_summary": ("summary", "summarize", "概述", "总结", "讲什么"),
    "paper_detail": ("detail", "method", "explain", "方法", "细节", "流程"),
    "paper_qa": ("qa", "question", "ask", "问", "提问"),
    "recommendation": ("recommend", "推荐", "suggest"),
    "preference_action": ("like", "dislike", "收藏", "喜欢", "不喜欢", "取消", "撤销"),
    "reading_list_action": ("reading list", "阅读列表", "收藏夹", "我的收藏"),
}

SEARCH_TRIGGER_PATTERNS: Sequence[str] = (
    r"\barxiv\b",
    r"\bpaper(s)?\b",
    r"论文",
    r"文献",
    r"找.+论文",
    r"搜.+论文",
    r"搜索",
    r"检索",
    r"查找",
    r"推荐.+论文",
    r"recent paper",
    r"recent papers",
    r"search",
    r"find",
    r"look for",
)


def _compact_search_spec(spec: Optional[ArxivSearchSpec]) -> Dict[str, Any]:
    if spec is None:
        return {}
    payload = {
        "intent": spec.intent,
        "query": spec.query,
        "title_query": spec.title_query,
        "abstract_query": spec.abstract_query,
        "categories": list(spec.categories or []),
        "submitted_days_ago": spec.submitted_days_ago,
        "max_results": spec.max_results,
        "sort_by": spec.sort_by,
        "sort_order": spec.sort_order,
        "field_operator": spec.field_operator,
        "category_operator": spec.category_operator,
        "reasoning_summary": spec.reasoning_summary,
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _compact_paper_summaries(papers: Sequence[Mapping[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for paper in list(papers or [])[: max(0, limit)]:
        if not isinstance(paper, Mapping):
            continue
        summary: Dict[str, Any] = {}
        for key in ("arxiv_id", "title", "published", "primary_category", "score", "rank"):
            value = paper.get(key)
            if value not in (None, ""):
                summary[key] = value
        if summary:
            summaries.append(summary)
    return summaries


def _append_step(
    state: AgentState,
    *,
    step: str,
    status: str,
    action: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> AgentState:
    next_state = state.model_copy(deep=True)
    # 轨迹只记录摘要级信息，避免把完整论文列表复制到每个阶段。
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step=step,
            status=status,
            action=action,
            inputs=inputs or {},
            outputs=outputs or {},
            error=error,
        )
    ]
    return next_state


def _compact_tool_args(tool_args: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in (
        "query",
        "title_query",
        "abstract_query",
        "author_query",
        "categories",
        "comment_query",
        "journal_ref_query",
        "report_number_query",
        "id_list",
        "field_operator",
        "category_operator",
        "submitted_days_ago",
        "max_results",
        "start",
        "sort_by",
        "sort_order",
    ):
        value = tool_args.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    return payload


def _build_tool_call_trace(
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    result: Optional[Mapping[str, Any]] = None,
    paper_count: Optional[int] = None,
    source: Optional[str] = None,
    normalized_inputs: Optional[Dict[str, Any]] = None,
    final_search_query: Optional[str] = None,
) -> Dict[str, Any]:
    trace: Dict[str, Any] = {}
    if isinstance(result, Mapping):
        trace.update(_result_mapping(result, "trace") or {})

    compact_inputs = _compact_tool_args(tool_args)
    trace.setdefault("tool_name", tool_name)
    trace.setdefault("inputs", compact_inputs)
    trace.setdefault("raw_inputs", compact_inputs)
    trace.setdefault("normalized_inputs", normalized_inputs or compact_inputs)
    trace.setdefault("final_search_query", final_search_query or compact_inputs.get("query"))
    trace.setdefault("source", source or trace.get("source") or "agent")
    trace.setdefault("sort_by", compact_inputs.get("sort_by"))
    trace.setdefault("sort_order", compact_inputs.get("sort_order"))
    trace.setdefault("start", compact_inputs.get("start"))
    trace.setdefault("max_results", compact_inputs.get("max_results"))
    trace.setdefault("returned_count", paper_count if paper_count is not None else 0)
    trace.setdefault("id_list", list(compact_inputs.get("id_list") or []))
    if isinstance(result, Mapping):
        if "returned_count" not in trace:
            trace["returned_count"] = len(_extract_papers_from_tool_result(_to_plain_dict(result)))
        trace.setdefault("result_ok", _result_ok(result))
        error = _result_mapping(result, "error")
        if error is not None:
            trace.setdefault("error", error)
    return trace

UNSUPPORTED_PATTERNS: Sequence[str] = (
    r"总结.*(这篇|本文|这份).*论文",
    r"概述.*(这篇|本文|这份).*论文",
    r"解释.*(这篇|本文|这份).*论文",
    r"这篇论文.*(方法|method|approach|做了什么)",
    r"我喜欢(第一篇|这篇|这几个)",
    r"加入待读",
    r"加入收藏",
    r"标记喜欢",
    r"标记不喜欢",
    r"reading list",
    r"favorite",
)

UNCLEAR_PATTERNS: Sequence[str] = (
    r"找一些论文",
    r"推荐几篇",
    r"最近的$",
    r"最新的$",
    r"相关的$",
    r"最近\s*的\s*论文",
    r"最近\s*的\s*paper",
)

TIME_PATTERNS: Sequence[Tuple[str, int]] = (
    (r"最近\s*(\d+)\s*天", -1),
    (r"近\s*(\d+)\s*天", -1),
    (r"最近\s*一周", 7),
    (r"近\s*一周", 7),
    (r"最近\s*两周", 14),
    (r"近\s*两周", 14),
    (r"最近\s*一个月", 30),
    (r"近\s*一个月", 30),
)

COUNT_PATTERNS: Sequence[str] = (
    r"(\d+)\s*(?:篇|paper(?:s)?|论文)",
    r"([一二三四五六七八九十两]+)\s*(?:篇|paper(?:s)?|论文)",
)

QUERY_HINT_PATTERNS: Sequence[str] = (
    r"(?:关于|围绕|面向|针对)\s*([^\n，。；;:]+)",
    r"(?:about|on|for)\s+([^\n,.;:]+)",
)

TITLE_HINT_PATTERNS: Sequence[str] = (
    r"(?:标题|题目|title)(?:是|为|：|:)?\s*([^\n，。；;:]+)",
)

ABSTRACT_HINT_PATTERNS: Sequence[str] = (
    r"(?:摘要|abstract)(?:是|为|：|:)?\s*([^\n，。；;:]+)",
)

CATEGORY_RULES: Sequence[Tuple[Sequence[str], Sequence[str]]] = (
    (("rag", "retrieval", "retrieve", "retriever", "embedding", "vector", "rerank", "re-rank"), ("cs.IR",)),
    (("llm", "large language model", "language model", "gpt", "prompt", "prompting"), ("cs.CL", "cs.AI")),
    (("agent", "agents"), ("cs.AI",)),
    (("nlp", "natural language", "dialogue", "chatbot", "translation", "question answering", "qa"), ("cs.CL",)),
    (("machine learning", "deep learning", "neural network", "neural", "optimization", "fine-tune", "finetune"), ("cs.LG",)),
    (("ai", "artificial intelligence", "reasoning", "planning"), ("cs.AI",)),
    (("recommendation", "recommender", "recommender system", "recommend"), ("cs.IR", "cs.LG")),
)

GENERIC_STOPWORDS = {
    "帮我",
    "给我",
    "请",
    "一些",
    "几篇",
    "几条",
    "篇",
    "paper",
    "papers",
    "论文",
    "文献",
    "arxiv",
    "最近",
    "最新",
    "相关",
    "最相关",
    "相关度高",
    "搜索",
    "查找",
    "检索",
    "推荐",
    "找",
    "看",
    "查询",
    "for",
    "on",
    "about",
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "with",
    "in",
}

CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _detect_hard_rule_intent(message: str) -> Optional[Dict[str, Any]]:
    text = _normalize_text(message)
    if not text:
        return None

    for intent, patterns in HARD_RULE_PATTERNS.items():
        match_text = _first_matching_pattern(text, patterns)
        if match_text is not None:
            return {
                "intent": intent,
                "intent_source": "hard_rule",
                "reason": f"matched obvious pattern: {match_text}",
                "confidence": 1.0,
                "plan": [
                    "识别到明确的非搜索请求",
                    "当前入口先返回意图识别结果，不进入 arXiv 搜索工具链",
                ],
                "next_actions": [
                    "如果你要的是 arXiv 搜索，请改成明确的论文检索需求",
                    "如果你想要论文总结、解释或问答，请提供目标论文标题或 arXiv ID",
                ],
                "warnings": ["hard_rule intercepted obvious non-search request"],
            }
    return None


def _build_intent_guidance(intent: str) -> Tuple[List[str], List[str], List[str]]:
    if intent == "arxiv_search":
        return (
            [
                "LLM 已识别为 arXiv 搜索请求",
                "规则会补全和校验检索参数，然后调用搜索工具",
                "整理结果并返回给用户",
            ],
            [
                "继续细化检索范围",
                "选择一篇论文查看详情",
                "如果需要，可以继续接论文总结或 QA 能力",
            ],
            [],
        )

    if intent in NON_SEARCH_INTENTS:
        if intent == "preference_action":
            return (
                [
                    "解析用户偏好动作",
                    "解析被操作的目标论文",
                    "保留 intent，等待后续偏好更新节点执行",
                ],
                [
                    "继续对其他论文执行喜欢、不喜欢或收藏动作",
                    "后续也可以继续搜索、查看推荐或打开论文详情",
                ],
                [f"identified non-search intent: {intent}"],
            )
        if intent == "reading_list_action":
            return (
                [
                    "识别到用户在查询阅读列表或收藏列表",
                    "当前入口先保留 intent，等待后续列表查询节点接入",
                ],
                [
                    "如果你想找论文，请改成明确的 arXiv 搜索需求",
                    "如果你想看收藏内容，可以后续直接进入阅读列表页面",
                ],
                [f"identified non-search intent: {intent}"],
            )
        if intent in {"paper_summary", "paper_detail", "paper_qa"}:
            return (
                [
                    "识别到论文详情相关请求",
                    "当前入口先保留 intent，等待论文详情、摘要或 QA 节点接入",
                ],
                [
                    "如果你是在找论文，请改成明确的 arXiv 搜索需求",
                    "如果你已经有目标论文标题或 arXiv ID，可以直接提供给后续详情能力",
                ],
                [f"identified non-search intent: {intent}"],
            )
        if intent == "recommendation":
            return (
                [
                    "识别到论文推荐请求",
                    "当前入口先保留 intent，等待个性化推荐服务接入",
                ],
                [
                    "如果你要的是普通搜索，请直接描述论文主题或关键词",
                    "如果你想看推荐结果，可以继续补充偏好方向或兴趣主题",
                ],
                [f"identified non-search intent: {intent}"],
            )
        return (
            [
                "LLM 已识别出论文系统内的非搜索请求类型",
                "当前 agent 暂未把该能力完整接入执行链路",
            ],
            [
                "如果你是在找论文，请改成明确的 arXiv 搜索需求",
                "如果你要总结、解释或问答某篇论文，请提供标题或 arXiv ID",
            ],
            [f"identified non-search intent: {intent}"],
        )

    if intent == "unclear":
        return (
            [
                "LLM 判断用户想要论文相关能力，但主题还不够明确",
                "需要补充研究方向、关键词或时间范围后再继续",
            ],
            [
                "补充主题、关键词或类别",
                "例如：RAG、LLM、Agent、NLP、推荐系统",
            ],
            ["search topic is unclear"],
        )

    return (
        [
            "当前请求超出 arXiv 搜索 Agent 的处理范围",
            "先返回可解释的边界说明，再等待用户改写请求",
        ],
        [
            "改写成 arXiv 论文搜索需求",
            "后续可以接论文总结或 QA 功能",
        ],
        ["request is outside the supported search workflow"],
    )


def _looks_like_preference_action_request(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("like", "dislike", "favorite", "bookmark", "cancel", "unlike", "unbookmark")
    ) or any(token in message for token in ("喜欢", "不喜欢", "收藏", "标记", "取消", "撤销"))


def _looks_like_reading_list_action_request(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("reading list", "favorites", "saved papers")) or any(
        token in message for token in ("我的收藏", "收藏有哪些", "收藏夹", "阅读列表", "待读列表")
    )


def _looks_like_paper_summary_request(message: str) -> bool:
    return any(token in message for token in ("讲什么", "总结", "概述"))


def _looks_like_paper_detail_request(message: str) -> bool:
    return any(token in message for token in ("方法", "流程", "细节", "原理", "怎么做"))


def _looks_like_paper_qa_request(message: str) -> bool:
    return any(token in message for token in ("问", "提问", "QA", "question"))


def _looks_like_recommendation_request(message: str) -> bool:
    lowered = message.lower()
    return "推荐" in message or "recommen" in lowered or "suggest" in lowered


def _build_llm_prompt(message: str) -> str:
    return (
        "You are an intent parser for a natural-language arXiv paper agent.\n"
        "Return JSON only.\n"
        "Classify the message into one of: arxiv_search, paper_detail, paper_summary, paper_qa, recommendation, preference_action, reading_list_action, unclear, unsupported.\n"
        "If it is a search request, extract a structured search spec.\n"
        "Schema:\n"
        "{"
        "\"intent\":\"arxiv_search|paper_detail|paper_summary|paper_qa|recommendation|preference_action|reading_list_action|unclear|unsupported\","
        "\"confidence\":0.0,"
        "\"query\":null|string,"
        "\"title_query\":null|string,"
        "\"abstract_query\":null|string,"
        "\"submitted_days_ago\":null|int,"
        "\"max_results\":10,"
        "\"sort_by\":\"submittedDate|relevance|lastUpdatedDate\","
        "\"sort_order\":\"ascending|descending\","
        "\"field_operator\":\"AND|OR|ANDNOT\","
        "\"category_operator\":\"AND|OR\","
        "\"reasoning_summary\":null|string,"
        "\"missing_info\":[],"
        "\"warnings\":[],"
        "\"next_actions\":[]"
        "}\n"
        "Rules:\n"
        "- Use arxiv_search when the user wants to search papers on arXiv.\n"
        "- Use paper_summary when the user asks to summarize a paper.\n"
        "- Use paper_detail when the user asks to explain a paper's method, contribution, or first section/content.\n"
        "- Use paper_qa when the user asks questions about a specific paper.\n"
        "- Use recommendation when the user wants paper recommendations.\n"
        "- Use preference_action when the user wants to like, dislike, favorite, or bookmark a paper.\n"
        "- Use reading_list_action when the user wants to inspect a reading list or favorites list.\n"
        "- Use unclear if the topic is missing or too vague.\n"
        "- Use unsupported only if the request is clearly outside the paper system.\n"
        "- Use max_results between 1 and 20. Default to 10 if not specified.\n"
        "- Use submitted_days_ago for recent-time expressions.\n"
        "- Do not output categories; the system applies a fixed configured category scope.\n"
        f"User message: {message}"
    )


def _parse_llm_intent(message: str, generation_service: Optional[Any]) -> Dict[str, Any]:
    if generation_service is None or not hasattr(generation_service, "complete_with_qwen"):
        return {
            "ok": False,
            "reason": "llm service is unavailable",
            "payload": None,
        }

    try:
        # 意图识别只负责分流，不需要大模型推理，优先使用小模型降低时延和成本。
        response = generation_service.complete_with_qwen(
            _build_llm_prompt(message),
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
    if intent == "arxiv_search":
        search_spec = ArxivSearchSpec(
            intent="arxiv_search",
            query=_normalize_optional_str(payload.get("query")),
            title_query=_normalize_optional_str(payload.get("title_query")),
            abstract_query=_normalize_optional_str(payload.get("abstract_query")),
            categories=get_default_agent_arxiv_categories(),
            submitted_days_ago=_safe_optional_int(payload.get("submitted_days_ago")),
            max_results=_clamp(_safe_int(payload.get("max_results"), default=10), 1, 20),
            sort_by=_normalize_sort_by(payload.get("sort_by")),
            sort_order=_normalize_sort_order(payload.get("sort_order")),
            field_operator=_normalize_field_operator(payload.get("field_operator")),
            category_operator=_normalize_category_operator(payload.get("category_operator")),
            reasoning_summary=_normalize_optional_str(payload.get("reasoning_summary")),
        )
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


def _build_rule_decision(message: str) -> Dict[str, Any]:
    hard_rule = _detect_hard_rule_intent(message)
    if hard_rule is not None:
        return {
            "intent": str(hard_rule.get("intent") or "unsupported"),
            "confidence": 1.0,
            "reason": hard_rule.get("reason"),
            "source": "hard_rule",
            "search_spec_before_enrichment": None,
            "search_spec_after_enrichment": None,
            "warnings": list(hard_rule.get("warnings") or []),
            "next_actions": list(hard_rule.get("next_actions") or []),
            "plan": list(hard_rule.get("plan") or []),
        }

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

    # 只有 arxiv_search 才允许继续构造 search_spec，其余 intent 必须在这里直接保留并退出搜索链路。
    search_spec_before = _build_rule_search_spec(message)
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


def _detect_non_search_rule_intent(message: str) -> Optional[str]:
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
    if _matches_any(message, (
        r"推荐.*论文",
        r"给我推荐",
        r"papers? recommendation",
        r"recommend.*paper",
        r"我可能感兴趣",
    )) or _looks_like_recommendation_request(message):
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


def _looks_search_like(message: str) -> bool:
    lowered = message.lower()
    explicit_cn_patterns = (
        r"找.*论文",
        r"搜.*论文",
        r"检索.*论文",
        r"查找.*论文",
        r"最近.*论文",
        r"新论文",
        r"新工作",
        r"最新进展",
        r"值得读",
        r"研究.*论文",
    )
    return _matches_any(message, SEARCH_TRIGGER_PATTERNS) or _matches_any(message, explicit_cn_patterns) or any(
        hint in lowered for hint in ("search", "find", "look for", "recent paper", "recent papers", "latest", "newest", "recent")
    )


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
) -> Dict[str, Any]:
    return {
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


def _build_rule_search_spec(message: str) -> Optional[ArxivSearchSpec]:
    query = _extract_query_from_message(message)
    title_query = _extract_marked_query(message, TITLE_HINT_PATTERNS)
    abstract_query = _extract_marked_query(message, ABSTRACT_HINT_PATTERNS)
    categories = get_default_agent_arxiv_categories()
    submitted_days_ago = _extract_submitted_days_ago(message)
    max_results = _extract_max_results(message)
    sort_by, sort_order = _extract_sorting(message)

    if not any([query, title_query, abstract_query]):
        return None

    spec = ArxivSearchSpec(
        intent="arxiv_search",
        query=query,
        title_query=title_query,
        abstract_query=abstract_query,
        categories=categories,
        submitted_days_ago=submitted_days_ago,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
        field_operator="AND",
        category_operator="OR",
        reasoning_summary=_build_reasoning_summary(query, categories, submitted_days_ago, max_results, sort_by),
    )
    return _apply_rule_enrichment(message, spec)


def parse_search_request(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    current_state = _coerce_state(state)
    message = _normalize_text(current_state.message or "")

    intent = _classify_intent(message)
    warnings: List[str] = []
    plan: List[str]
    next_actions: List[str]
    search_spec: Optional[ArxivSearchSpec] = None

    if intent == "unsupported":
        plan = [
            "识别到当前请求不属于 arXiv 论文搜索",
            "输出支持范围说明并结束",
        ]
        next_actions = [
            "改写为自然语言 arXiv 论文搜索需求",
            "后续可以接入论文总结或 QA 功能",
        ]
        warnings.append("当前请求不属于自然语言 arXiv 论文搜索")
    elif intent == "unclear":
        plan = [
            "识别到用户想搜索论文，但主题不够明确",
            "提示用户补充研究方向后再试",
        ]
        next_actions = [
            "补充主题、关键词或类别",
            "例如：RAG、LLM、Agent、NLP、推荐系统",
        ]
        warnings.append("搜索主题不够明确")
    else:
        search_spec, spec_warnings = _build_search_spec(message, generation_service=generation_service)
        warnings.extend(spec_warnings)
        if search_spec is None:
            intent = "unclear"
            plan = [
                "尝试从输入中提取搜索条件",
                "若条件不足则提示用户补充信息",
            ]
            next_actions = [
                "补充主题、关键词或类别",
                "例如：RAG、LLM、Agent、NLP、推荐系统",
            ]
            warnings.append("搜索条件不足，无法生成有效的 arXiv 查询")
        else:
            intent = "arxiv_search"
            plan = [
                "解析自然语言搜索条件",
                "调用 search_arxiv_structured 执行检索",
                "整理并返回论文结果",
            ]
            next_actions = [
                "继续缩小到某个子方向搜索",
                "选择一篇论文查看详情",
                "后续可以接入论文总结或 QA 功能",
            ]

    normalized_state = current_state.model_copy(deep=True)
    normalized_state.intent = intent
    normalized_state.search_spec = search_spec
    normalized_state.plan = plan
    normalized_state.warnings = _dedupe_preserve_order(warnings)
    normalized_state.next_actions = next_actions
    normalized_state.tool_name = None
    normalized_state.tool_args = {}
    normalized_state.tool_result = None
    normalized_state.tool_calls = []
    normalized_state.papers = []
    normalized_state.answer = None
    normalized_state.errors = []
    return _append_step(
        normalized_state,
        step="intent_recognition",
        status="success",
        action="识别用户意图并判断是否进入 arXiv 搜索流程",
        inputs={"message": message},
        outputs={
            "intent": intent,
            "search_spec": _compact_search_spec(search_spec),
            "plan": list(plan),
            "warnings": list(normalized_state.warnings),
            "next_actions": list(next_actions),
        },
    )


def build_search_tool_args(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search" or next_state.search_spec is None:
        next_state.tool_name = None
        next_state.tool_args = {}
        return _append_step(
            next_state,
            step="tool_argument_construction",
            status="skipped",
            action="基于搜索条件构造 arXiv 工具参数",
            inputs={"intent": next_state.intent, "search_spec": _compact_search_spec(next_state.search_spec)},
            outputs={"reason": "非 arXiv 搜索或搜索条件缺失"},
        )

    spec = next_state.search_spec
    next_state.tool_name = SEARCH_TOOL_NAME
    next_state.tool_args = {
        "query": spec.query,
        "title_query": spec.title_query,
        "abstract_query": spec.abstract_query,
        "categories": list(spec.categories or []),
        "submitted_days_ago": spec.submitted_days_ago,
        "max_results": spec.max_results or 10,
        "start": 0,
        "sort_by": spec.sort_by or "submittedDate",
        "sort_order": spec.sort_order or "descending",
        "field_operator": spec.field_operator or "AND",
        "category_operator": spec.category_operator or "OR",
    }
    return _append_step(
        next_state,
        step="tool_argument_construction",
        status="success",
        action="基于搜索条件构造 arXiv 工具参数",
        inputs={"search_spec": _compact_search_spec(spec)},
        outputs={"tool_name": SEARCH_TOOL_NAME, "tool_args": {key: value for key, value in next_state.tool_args.items() if key != "query" or value}},
    )


def invoke_search_tool(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search":
        return _append_step(
            next_state,
            step="search_tool_call",
            status="skipped",
            action="调用 arXiv 搜索工具",
            inputs={"intent": next_state.intent},
            outputs={"reason": "当前意图不是 arXiv 搜索"},
        )

    if next_state.tool_name != SEARCH_TOOL_NAME or not next_state.tool_args:
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + ["搜索工具参数未准备好，跳过工具调用"],
        )
        return _append_step(
            next_state,
            step="search_tool_call",
            status="failed",
            action="调用 arXiv 搜索工具",
            inputs={"tool_name": next_state.tool_name, "tool_args": dict(next_state.tool_args or {})},
            outputs={"paper_count": 0},
            error="搜索工具参数未准备好",
        )

    try:
        raw_result = invoke_tool(SEARCH_TOOL_NAME, **dict(next_state.tool_args))
    except Exception as exc:
        next_state.tool_result = {"ok": False, "error": {"message": str(exc)}}
        next_state.tool_calls = list(next_state.tool_calls) + [
            AgentToolCall(
                tool_name=SEARCH_TOOL_NAME,
                arguments=dict(next_state.tool_args),
                status="failed",
                summary="工具调用异常",
                trace=_build_tool_call_trace(
                    tool_name=SEARCH_TOOL_NAME,
                    tool_args=next_state.tool_args,
                    paper_count=0,
                    source="agent",
                ),
                error={"message": str(exc)},
            )
        ]
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + ["工具调用失败，请检查搜索参数或 arXiv 服务状态"],
        )
        next_state.papers = []
        return _append_step(
            next_state,
            step="search_tool_call",
            status="failed",
            action="调用 arXiv 搜索工具",
            inputs={"tool_name": SEARCH_TOOL_NAME, "tool_args": dict(next_state.tool_args)},
            outputs={"paper_count": 0},
            error=str(exc),
        )

    result = _to_plain_dict(raw_result)
    next_state.tool_result = result

    tool_call = AgentToolCall(
        tool_name=SEARCH_TOOL_NAME,
        arguments=dict(next_state.tool_args),
        status="success" if _result_ok(result) else "failed",
        summary=_result_text(result, "summary"),
        trace=_build_tool_call_trace(
            tool_name=SEARCH_TOOL_NAME,
            tool_args=next_state.tool_args,
            result=result,
            paper_count=len(next_state.papers or []),
            source=_result_mapping(result, "trace").get("source") if _result_mapping(result, "trace") else None,
            normalized_inputs=_result_mapping(result, "trace").get("normalized_inputs") if _result_mapping(result, "trace") else None,
            final_search_query=_result_mapping(result, "trace").get("final_search_query") if _result_mapping(result, "trace") else None,
        ),
        error=_result_mapping(result, "error"),
    )
    next_state.tool_calls = list(next_state.tool_calls) + [tool_call]

    if _result_ok(result):
        next_state.papers = _extract_papers_from_tool_result(result)
    else:
        next_state.papers = []
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + [_format_tool_failure_warning(result)],
        )

    return _append_step(
        next_state,
        step="search_tool_call",
        status="success" if _result_ok(result) else "failed",
        action="调用 arXiv 搜索工具",
        inputs={"tool_name": SEARCH_TOOL_NAME, "tool_args": dict(next_state.tool_args)},
        outputs={
            "paper_count": len(next_state.papers or []),
            "tool_call_status": tool_call.status,
            "tool_summary": tool_call.summary,
        },
        error=_extract_error_message(result) if not _result_ok(result) else None,
    )


def check_search_result(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search":
        return _append_step(
            next_state,
            step="search_result_check",
            status="skipped",
            action="检查搜索结果质量并补充提示",
            inputs={"intent": next_state.intent},
            outputs={"reason": "当前意图不是 arXiv 搜索"},
        )

    warnings = list(next_state.warnings)
    tool_result = _to_plain_dict(next_state.tool_result)
    papers = list(next_state.papers or [])
    max_results = _determine_requested_max_results(next_state)

    if not tool_result:
        warnings.append("搜索工具结果不存在，请先执行工具调用")
    else:
        if not _result_ok(tool_result):
            warnings.append("工具调用失败，请检查搜索参数或 arXiv 服务状态")
        error_message = _extract_error_message(tool_result)
        if error_message:
            warnings.append(error_message)

    if _result_ok(tool_result):
        if not papers:
            warnings.append("搜索结果为空，建议扩大时间范围或减少关键词")
        elif _papers_are_significantly_fewer_than_requested(len(papers), max_results):
            warnings.append("结果数量较少，可能是查询条件过窄")

    next_state.warnings = _dedupe_preserve_order(warnings)
    return _append_step(
        next_state,
        step="search_result_check",
        status="success" if _result_ok(tool_result) else "failed",
        action="检查搜索结果质量并补充提示",
        inputs={
            "paper_count": len(papers),
            "tool_result_ok": _result_ok(tool_result),
            "requested_max_results": max_results,
        },
        outputs={
            "warning_count": len(next_state.warnings),
            "paper_count": len(papers),
        },
        error=_extract_error_message(tool_result) if tool_result and not _result_ok(tool_result) else None,
    )


def personalized_rank_and_annotate_papers(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search" or not next_state.user_id or not next_state.papers:
        next_state.personalized_rerank_applied = False
        return _append_step(
            next_state,
            step="personalized_rerank",
            status="skipped",
            action="基于用户偏好对搜索结果做个性化重排",
            inputs={
                "intent": next_state.intent,
                "user_id_present": bool(next_state.user_id),
                "paper_count": len(next_state.papers or []),
            },
            outputs={"personalized_rerank_applied": False},
        )

    try:
        recommendation_service = get_recommendation_service()
    except Exception as exc:
        next_state.personalized_rerank_applied = False
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + [f"无法初始化推荐服务，已保留普通搜索排序: {exc}"],
        )
        return _append_step(
            next_state,
            step="personalized_rerank",
            status="failed",
            action="基于用户偏好对搜索结果做个性化重排",
            inputs={
                "user_id": str(next_state.user_id),
                "paper_count": len(next_state.papers or []),
                "query": query_text,
                "search_spec": _compact_search_spec(next_state.search_spec),
            },
            outputs={"personalized_rerank_applied": False},
            error=str(exc),
        )

    search_spec_payload = next_state.search_spec.model_dump() if next_state.search_spec is not None else None
    query_text = next_state.search_spec.query if next_state.search_spec is not None else None

    try:
        rerank_result = recommendation_service.rerank_search_results_for_user(
            user_id=str(next_state.user_id),
            papers=list(next_state.papers or []),
            query=query_text,
            top_n=_determine_requested_max_results(next_state),
            search_spec=search_spec_payload,
        )
    except Exception as exc:
        next_state.personalized_rerank_applied = False
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + [f"个性化重排失败，已保留普通搜索排序: {exc}"],
        )
        return _append_step(
            next_state,
            step="personalized_rerank",
            status="failed",
            action="基于用户偏好对搜索结果做个性化重排",
            inputs={
                "user_id": str(next_state.user_id),
                "paper_count": len(next_state.papers or []),
                "query": query_text,
                "search_spec": _compact_search_spec(next_state.search_spec),
            },
            outputs={"personalized_rerank_applied": False},
            error=str(exc),
        )

    reranked_papers = rerank_result.get("papers") if isinstance(rerank_result, Mapping) else None
    if isinstance(reranked_papers, list) and reranked_papers:
        next_state.papers = [paper for paper in reranked_papers if isinstance(paper, dict)]

    next_state.personalized_rerank_applied = bool(rerank_result.get("personalized_applied")) if isinstance(rerank_result, Mapping) else False
    rerank_warnings = rerank_result.get("warnings", []) if isinstance(rerank_result, Mapping) else []
    if isinstance(rerank_warnings, list):
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [str(item) for item in rerank_warnings if str(item).strip()])

    if not next_state.personalized_rerank_applied:
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + ["用户兴趣向量不可用或个性化重排未生效，已退化为普通搜索结果"],
        )

    return _append_step(
        next_state,
        step="personalized_rerank",
        status="success",
        action="基于用户偏好对搜索结果做个性化重排",
        inputs={
            "user_id": str(next_state.user_id),
            "paper_count": len(current_state.papers or []),
            "query": query_text,
            "search_spec": _compact_search_spec(next_state.search_spec),
        },
        outputs={
            "personalized_rerank_applied": bool(next_state.personalized_rerank_applied),
            "paper_count": len(next_state.papers or []),
            "top_papers": _compact_paper_summaries(next_state.papers, limit=3),
        },
    )


def synthesize_response(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent == "unsupported":
        next_state.answer = "当前功能只支持自然语言 arXiv 论文搜索；如果你想查论文，请改成明确的搜索需求。"
        next_state.next_actions = [
            "改写为 arXiv 论文搜索问题后重试",
            "后续可以接入论文总结或 QA 功能",
        ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={
                "intent": next_state.intent,
                "paper_count": len(next_state.papers or []),
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
            },
        )

    if next_state.intent == "unclear":
        next_state.answer = "你想搜索论文，但主题还不够明确。请补充研究方向、关键词或类别后重试。"
        next_state.next_actions = [
            "补充研究方向或关键词",
            "例如：RAG、LLM、Agent、NLP、推荐系统",
        ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={
                "intent": next_state.intent,
                "paper_count": len(next_state.papers or []),
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
            },
        )

    if next_state.intent != "arxiv_search":
        next_state.answer = "当前请求暂时无法处理，请改写为 arXiv 论文搜索需求。"
        next_state.next_actions = ["改写为 arXiv 搜索问题后重试"]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={
                "intent": next_state.intent,
                "paper_count": len(next_state.papers or []),
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
            },
        )

    spec = next_state.search_spec
    papers = list(next_state.papers or [])
    paper_count = len(papers)
    max_results = spec.max_results if spec is not None else 10
    summary = _summarize_search_spec(spec)
    priority_titles = _collect_priority_titles(papers, limit=3)
    personalized_applied = bool(next_state.personalized_rerank_applied)

    if paper_count > 0:
        next_state.answer = f"已按“{summary}”搜索 arXiv，当前返回 {paper_count} 篇论文。"
        if personalized_applied:
            if priority_titles:
                next_state.answer += f" 本次结果已根据用户兴趣进行个性化重排，建议优先阅读：{', '.join(priority_titles)}。"
            else:
                next_state.answer += " 本次结果已根据用户兴趣进行个性化重排，建议优先阅读排序靠前的论文。"
        else:
            next_state.answer += " 本次结果未使用用户兴趣向量，保持普通搜索排序。"
        next_state.next_actions = [
            "继续缩小到某个子方向搜索",
            "选择一篇论文查看详情",
            "后续可以接入论文总结或 QA 功能",
        ]
    else:
        next_state.answer = f"已按“{summary}”搜索 arXiv，但当前没有找到结果。"
        next_state.next_actions = [
            "放宽关键词或扩大时间范围后重试",
            "只保留核心主题词后再搜索",
            "后续可以接入论文总结或 QA 功能",
        ]

    if max_results and paper_count < max_results:
        next_state.answer += f" 本次最多期望返回 {max_results} 篇。"

    return _append_step(
        next_state,
        step="final_answer_generation",
        status="success",
        action="生成最终答复并给出后续动作",
        inputs={
            "intent": next_state.intent,
            "paper_count": paper_count,
            "personalized_rerank_applied": personalized_applied,
        },
        outputs={
            "answer": next_state.answer,
            "next_actions": list(next_state.next_actions),
            "top_papers": _collect_priority_titles(papers, limit=3),
        },
    )


def route_after_parse(state: Any) -> str:
    current_state = _coerce_state(state)
    intent = str(current_state.intent or "").strip()
    return intent if intent in SUPPORTED_INTENTS else "unsupported"


def _build_search_spec(
    message: str,
    generation_service: Optional[Any] = None,
) -> Tuple[Optional[ArxivSearchSpec], List[str]]:
    warnings: List[str] = []

    llm_payload = _parse_with_llm(message, generation_service=generation_service)
    if llm_payload is not None:
        try:
            llm_spec = _normalize_and_validate_spec(llm_payload)
        except ValidationError as exc:
            llm_spec = None
            warnings.append(f"LLM 解析结果未通过结构校验: {_validation_error_summary(exc)}")
        if llm_spec is not None:
            enriched = _apply_rule_enrichment(message, llm_spec)
            if enriched is not None:
                return enriched, _dedupe_preserve_order(warnings)
        warnings.append("LLM 输出未通过校验，已回退到规则解析")

    try:
        spec = _build_spec_from_rules(message)
    except ValidationError as exc:
        warnings.append(f"规则解析结果未通过结构校验: {_validation_error_summary(exc)}")
        return None, _dedupe_preserve_order(warnings)

    return spec, _dedupe_preserve_order(warnings)


def _parse_with_llm(message: str, generation_service: Optional[Any]) -> Optional[Dict[str, Any]]:
    if generation_service is None or not hasattr(generation_service, "complete_with_qwen"):
        return None

    prompt = (
        "You are an intent parser for a natural-language arXiv search agent.\n"
        "Return JSON only.\n"
        "Classify the message into one of: arxiv_search, unclear, unsupported.\n"
        "If it is a valid search request, extract a structured search spec.\n"
        "Schema:\n"
        "{"
        "\"intent\":\"arxiv_search|unclear|unsupported\","
        "\"query\":null|string,"
        "\"title_query\":null|string,"
        "\"abstract_query\":null|string,"
        "\"submitted_days_ago\":null|int,"
        "\"max_results\":10,"
        "\"sort_by\":\"submittedDate|relevance|lastUpdatedDate\","
        "\"sort_order\":\"ascending|descending\","
        "\"field_operator\":\"AND|OR|ANDNOT\","
        "\"category_operator\":\"AND|OR\","
        "\"reasoning_summary\":null|string"
        "}\n"
        "Rules:\n"
        "- If the user wants to search papers on arXiv, use arxiv_search.\n"
        "- If the user wants paper summary, method explanation, preference marking, or reading-list actions, use unsupported.\n"
        "- If the user is searching but topic is missing or too vague, use unclear.\n"
        "- Use max_results between 1 and 20. Default to 10 if not specified.\n"
        "- Use submitted_days_ago for recent-time expressions.\n"
        "- Do not output categories; the system applies a fixed configured category scope.\n"
        f"User message: {message}"
    )

    try:
        # 搜索规格抽取同样属于轻量解析任务，统一走小模型路由。
        response = generation_service.complete_with_qwen(
            prompt,
            task_type="search_spec_parse",
        )
        payload = json.loads(_extract_json_block(str(response)))
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def _normalize_and_validate_spec(payload: Mapping[str, Any]) -> Optional[ArxivSearchSpec]:
    intent = str(payload.get("intent", "arxiv_search") or "arxiv_search").strip().lower()
    if intent not in SUPPORTED_INTENTS or intent != "arxiv_search":
        return None

    return ArxivSearchSpec(
        intent="arxiv_search",
        query=_normalize_optional_str(payload.get("query")),
        title_query=_normalize_optional_str(payload.get("title_query")),
        abstract_query=_normalize_optional_str(payload.get("abstract_query")),
        categories=get_default_agent_arxiv_categories(),
        submitted_days_ago=_safe_optional_int(payload.get("submitted_days_ago")),
        max_results=_clamp(_safe_int(payload.get("max_results"), default=10), 1, 20),
        sort_by=_normalize_sort_by(payload.get("sort_by")),
        sort_order=_normalize_sort_order(payload.get("sort_order")),
        field_operator=_normalize_field_operator(payload.get("field_operator")),
        category_operator=_normalize_category_operator(payload.get("category_operator")),
        reasoning_summary=_normalize_optional_str(payload.get("reasoning_summary")),
    )


def _build_spec_from_rules(message: str) -> Optional[ArxivSearchSpec]:
    query = _extract_query_from_message(message)
    title_query = _extract_marked_query(message, TITLE_HINT_PATTERNS)
    abstract_query = _extract_marked_query(message, ABSTRACT_HINT_PATTERNS)
    categories = get_default_agent_arxiv_categories()
    submitted_days_ago = _extract_submitted_days_ago(message)
    max_results = _extract_max_results(message)
    sort_by, sort_order = _extract_sorting(message)

    if not any([query, title_query, abstract_query, categories]):
        return None

    return ArxivSearchSpec(
        intent="arxiv_search",
        query=query,
        title_query=title_query,
        abstract_query=abstract_query,
        categories=categories,
        submitted_days_ago=submitted_days_ago,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
        field_operator="AND",
        category_operator="OR",
        reasoning_summary=_build_reasoning_summary(query, categories, submitted_days_ago, max_results, sort_by),
    )


def _apply_rule_enrichment(message: str, spec: ArxivSearchSpec) -> Optional[ArxivSearchSpec]:
    query = spec.query or _extract_query_from_message(message)
    title_query = spec.title_query or _extract_marked_query(message, TITLE_HINT_PATTERNS)
    abstract_query = spec.abstract_query or _extract_marked_query(message, ABSTRACT_HINT_PATTERNS)
    # Always use the configured category scope for this agent instead of model-generated categories.
    categories = get_default_agent_arxiv_categories()
    submitted_days_ago = spec.submitted_days_ago if spec.submitted_days_ago is not None else _extract_submitted_days_ago(message)
    max_results = _clamp(spec.max_results or 10, 1, 20)
    sort_by, sort_order = _extract_sorting(message)
    if spec.sort_by in {"relevance", "submittedDate", "lastUpdatedDate"}:
        sort_by = spec.sort_by
        sort_order = spec.sort_order or sort_order

    return ArxivSearchSpec(
        intent="arxiv_search",
        query=query,
        title_query=title_query,
        abstract_query=abstract_query,
        categories=categories,
        submitted_days_ago=submitted_days_ago,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
        field_operator=_normalize_field_operator(spec.field_operator),
        category_operator=_normalize_category_operator(spec.category_operator),
        reasoning_summary=spec.reasoning_summary
        or _build_reasoning_summary(query, categories, submitted_days_ago, max_results, sort_by),
    )


def _classify_intent(message: str) -> str:
    lowered = message.lower()
    if not message:
        return "unclear"

    if _matches_any(message, UNSUPPORTED_PATTERNS):
        return "unsupported"

    search_like = _matches_any(message, SEARCH_TRIGGER_PATTERNS) or any(
        hint in lowered for hint in ("search", "find", "look for", "recent paper", "recent papers")
    )
    vague_search = _matches_any(message, UNCLEAR_PATTERNS) or any(
        hint in lowered for hint in ("latest", "newest", "recent", "relevant")
    )

    if search_like:
        if _extract_query_from_message(message) or _extract_marked_query(message, TITLE_HINT_PATTERNS) or _extract_marked_query(message, ABSTRACT_HINT_PATTERNS):
            return "arxiv_search"
        return "unclear"

    if vague_search or "arxiv" in lowered:
        return "unclear"

    return "unsupported"


def _extract_query_from_message(message: str) -> Optional[str]:
    for pattern in QUERY_HINT_PATTERNS:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            candidate = _normalize_topic_phrase(match.group(1))
            if candidate:
                return candidate

    cleaned = _remove_noise(message)
    tokens = [token for token in _tokenize_mixed(cleaned) if _is_topic_token(token)]
    candidate = _normalize_topic_phrase(" ".join(_dedupe_preserve_order(tokens)))
    return candidate


def _extract_marked_query(message: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            candidate = _normalize_topic_phrase(match.group(1))
            if candidate:
                return candidate
    return None


def _extract_submitted_days_ago(message: str) -> Optional[int]:
    lowered = message.lower()
    for pattern, fixed_value in TIME_PATTERNS:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            if fixed_value > 0:
                return fixed_value
            return _clamp(_safe_int(match.group(1), default=7), 1, 365)

    if "最近一周" in message or "近一周" in message or "last week" in lowered:
        return 7
    if "最近两周" in message or "近两周" in message:
        return 14
    if "最近一个月" in message or "近一个月" in message or "last month" in lowered:
        return 30
    return None


def _extract_max_results(message: str) -> int:
    for pattern in COUNT_PATTERNS:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            raw_value = match.group(1)
            value = _parse_small_chinese_number(raw_value)
            if value is None:
                value = _safe_int(raw_value, default=10)
            return _clamp(value, 1, 20)
    return 10


def _extract_sorting(message: str) -> Tuple[str, str]:
    lowered = message.lower()
    if any(keyword in lowered for keyword in ("最相关", "相关度高", "most relevant", "relevant", "relevance")):
        return "relevance", "descending"
    if any(keyword in lowered for keyword in ("最新", "最近", "latest", "newest", "recent")):
        return "submittedDate", "descending"
    return "submittedDate", "descending"


def _infer_categories(message: str, *texts: Optional[str]) -> List[str]:
    valid_categories = get_valid_arxiv_categories()
    allowed_map = {category.lower(): category for category in valid_categories}
    combined = " ".join([message, *[text or "" for text in texts]]).lower()

    categories: List[str] = []
    for keywords, mapped_categories in CATEGORY_RULES:
        if any(keyword in combined for keyword in keywords):
            for category in mapped_categories:
                canonical = allowed_map.get(category.lower())
                if canonical and canonical not in categories:
                    categories.append(canonical)
    return categories


def _build_reasoning_summary(
    query: Optional[str],
    categories: List[str],
    submitted_days_ago: Optional[int],
    max_results: int,
    sort_by: str,
) -> str:
    parts: List[str] = []
    if query:
        parts.append(f"主题={query}")
    if categories:
        parts.append(f"类别={', '.join(categories)}")
    if submitted_days_ago is not None:
        parts.append(f"最近{submitted_days_ago}天")
    parts.append(f"数量={max_results}")
    parts.append(f"排序={sort_by}")
    return "；".join(parts)


def _normalize_topic_phrase(value: Optional[str]) -> Optional[str]:
    text = _normalize_text(value or "")
    if not text:
        return None
    text = _remove_noise(text)
    if not text:
        return None
    text = re.sub(r"[，。；;:：]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in _tokenize_mixed(text) if _is_topic_token(token)]
    if not tokens:
        return None
    return " ".join(_dedupe_preserve_order(tokens))


def _remove_noise(text: str) -> str:
    result = text
    result = _remove_patterns(result, TIME_PATTERNS)
    result = _strip_count_phrases(result)
    result = re.sub(r"(?i)\bmost relevant\b", " ", result)
    result = re.sub(r"(?i)\b(arxiv|paper|papers|search|find|look for|recent|latest|newest|relevant)\b", " ", result)
    result = re.sub(r"(帮我|请帮|给我|推荐|找|搜索|查找|检索|看看|看一下|论文|文献|最近|最新|相关|最相关|相关度高)", " ", result)
    result = re.sub(r"(\d+\s*(?:篇|paper(?:s)?|论文))", " ", result, flags=re.IGNORECASE)
    result = re.sub(r"([一二三四五六七八九十两]+\s*(?:篇|paper(?:s)?|论文))", " ", result, flags=re.IGNORECASE)
    result = re.sub(r"(篇论文|篇\s*paper(?:s)?|paper(?:s)?\s*篇)", " ", result, flags=re.IGNORECASE)
    result = re.sub(r"[，。；;:：、/\\|()\[\]{}！？?!]", " ", result)
    return re.sub(r"\s+", " ", result).strip()


def _remove_patterns(text: str, compiled_patterns: Iterable[Tuple[str, int]]) -> str:
    result = text
    for pattern, _ in compiled_patterns:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return result


def _strip_count_phrases(text: str) -> str:
    result = text
    for pattern in COUNT_PATTERNS:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return result


def _tokenize_mixed(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-_/\.]*|[\u4e00-\u9fff]{2,}", text)
    return [token.strip() for token in tokens if token and token.strip()]


def _is_topic_token(token: str) -> bool:
    normalized = token.strip().lower()
    if not normalized:
        return False
    if normalized in GENERIC_STOPWORDS:
        return False
    if normalized in {"arxiv", "paper", "papers"}:
        return False
    if len(normalized) == 1 and not re.search(r"[\u4e00-\u9fff]", normalized):
        return False
    return True


def _normalize_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_optional_str(value: Any) -> Optional[str]:
    text = _normalize_text(str(value or ""))
    return text or None


def _normalize_categories(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        raise ValueError("categories must be a list of category codes")

    valid_categories = get_valid_arxiv_categories()
    allowed_map = {category.lower(): category for category in valid_categories}
    normalized: List[str] = []
    invalid: List[str] = []
    seen = set()
    for item in candidates:
        text = _normalize_text(str(item or ""))
        if not text:
            continue
        canonical = allowed_map.get(text.lower())
        if canonical is None:
            invalid.append(text)
            continue
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)

    if invalid:
        raise ValueError(f"invalid arxiv categories: {', '.join(invalid)}")
    return normalized


def _normalize_sort_by(value: Any) -> str:
    text = _normalize_text(str(value or ""))
    lowered = text.lower()
    if lowered in {"relevance"}:
        return "relevance"
    if lowered in {"lastupdateddate", "lastupdated"}:
        return "lastUpdatedDate"
    return "submittedDate"


def _normalize_sort_order(value: Any) -> str:
    text = _normalize_text(str(value or "")).lower()
    return "ascending" if text == "ascending" else "descending"


def _normalize_field_operator(value: Any) -> str:
    text = _normalize_text(str(value or "")).upper()
    return text if text in {"AND", "OR", "ANDNOT"} else "AND"


def _normalize_category_operator(value: Any) -> str:
    text = _normalize_text(str(value or "")).upper()
    return text if text in {"AND", "OR"} else "OR"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _parse_small_chinese_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in CHINESE_NUMBER_MAP:
        return CHINESE_NUMBER_MAP[text]
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = CHINESE_NUMBER_MAP.get(left, 1 if left == "" else 0)
        ones = CHINESE_NUMBER_MAP.get(right, 0) if right else 0
        if tens == 0 and left:
            return None
        return tens * 10 + ones
    return None


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = _normalize_text(item).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(_normalize_text(item))
    return result


def _validation_error_summary(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", []) if part is not None)
    message = str(first.get("msg", "")).strip() or str(exc)
    return f"{location}: {message}" if location else message


def _extract_papers_from_tool_result(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data = result.get("data")
    candidates: Any = data
    if isinstance(data, dict):
        if isinstance(data.get("papers"), list):
            candidates = data["papers"]
        elif isinstance(data.get("result"), dict) and isinstance(data["result"].get("papers"), list):
            candidates = data["result"]["papers"]
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def _extract_error_message(result: Mapping[str, Any]) -> Optional[str]:
    error = result.get("error")
    if not isinstance(error, dict):
        return None
    message = str(error.get("message", "") or "").strip()
    detail = error.get("detail")
    if message and detail:
        return f"{message}: {detail}"
    return message or None


def _format_tool_failure_warning(result: Mapping[str, Any]) -> str:
    error_message = _extract_error_message(result)
    if error_message:
        return f"工具调用失败，请检查搜索参数或 arXiv 服务状态: {error_message}"
    return "工具调用失败，请检查搜索参数或 arXiv 服务状态"


def _papers_are_significantly_fewer_than_requested(actual_count: int, max_results: int) -> bool:
    if actual_count <= 0 or max_results <= 0:
        return False
    if max_results <= 3:
        return actual_count < max_results
    return actual_count <= max(1, max_results // 2)


def _summarize_search_spec(spec: Optional[ArxivSearchSpec]) -> str:
    if spec is None:
        return "当前搜索条件"

    parts: List[str] = []
    if spec.query:
        parts.append(f"主题 {spec.query}")
    if spec.title_query:
        parts.append(f"标题 {spec.title_query}")
    if spec.abstract_query:
        parts.append(f"摘要 {spec.abstract_query}")
    if spec.categories:
        parts.append(f"类别 {', '.join(spec.categories)}")
    if spec.submitted_days_ago is not None:
        parts.append(f"最近 {spec.submitted_days_ago} 天")
    parts.append(f"排序 {spec.sort_by} / {spec.sort_order}")
    parts.append(f"最多 {spec.max_results} 篇")
    return "，".join(parts)


def _extract_json_block(text: str) -> str:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)
    raw_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if raw_match:
        return raw_match.group(1)
    return text


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _result_ok(result: Mapping[str, Any]) -> bool:
    ok = result.get("ok")
    if isinstance(ok, bool):
        return ok
    if ok is None:
        return False
    return bool(ok)


def _result_text(result: Mapping[str, Any], key: str) -> Optional[str]:
    value = result.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _result_mapping(result: Mapping[str, Any], key: str) -> Optional[Dict[str, Any]]:
    value = result.get(key)
    return value if isinstance(value, dict) else None


def _determine_requested_max_results(state: AgentState) -> int:
    if state.search_spec is not None:
        return max(1, int(state.search_spec.max_results or 10))
    if isinstance(state.tool_args, dict) and state.tool_args.get("max_results") is not None:
        return max(1, int(state.tool_args.get("max_results") or 10))
    return 10


def _collect_priority_titles(papers: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    prioritized = sorted(
        [paper for paper in papers if isinstance(paper, dict)],
        key=lambda paper: (
            float(paper.get("priority", 0) or 0) if float(paper.get("priority", 0) or 0) > 0 else 10_000.0,
            -float(paper.get("final_score", 0.0) or 0.0),
            -float(paper.get("query_match_score", 0.0) or 0.0),
            str(paper.get("arxiv_id", "") or paper.get("id", "") or ""),
        ),
    )
    titles: List[str] = []
    for paper in prioritized[: max(1, int(limit or 3))]:
        title = str(paper.get("title", "") or "").strip()
        if not title:
            continue
        titles.append(title)
    return titles


def _coerce_state(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    return AgentState.model_validate(dict(state))


def _first_matching_pattern(text: str, patterns: Sequence[str]) -> Optional[str]:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def parse_search_request(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    current_state = _coerce_state(state)
    message = _normalize_text(current_state.message or "")

    # 先收集所有中间结果，再统一落到 normalized_state，避免分支里直接改原状态。
    warnings: List[str] = []
    plan: List[str] = []
    next_actions: List[str] = []
    search_spec: Optional[ArxivSearchSpec] = None
    intent = "unsupported"
    intent_source = "fallback"
    fallback_reason: Optional[str] = None
    llm_result: Optional[Dict[str, Any]] = None
    rule_result: Optional[Dict[str, Any]] = None
    hard_rule_result: Optional[Dict[str, Any]] = None
    search_spec_before_enrichment: Optional[Dict[str, Any]] = None
    search_spec_after_enrichment: Optional[Dict[str, Any]] = None

    # 硬规则优先：一些明显的非搜索请求直接拦截，不再进入 LLM 解析。
    hard_rule_result = _detect_hard_rule_intent(message)
    if hard_rule_result is not None:
        intent = str(hard_rule_result.get("intent") or "unsupported")
        intent_source = str(hard_rule_result.get("intent_source") or "hard_rule")
        plan = list(hard_rule_result.get("plan") or [])
        next_actions = list(hard_rule_result.get("next_actions") or [])
        warnings.extend(str(item) for item in hard_rule_result.get("warnings", []) if str(item).strip())
        rule_result = hard_rule_result
    else:
        # 先尝试 LLM，再用规则结果做兜底和校验。
        llm_payload_result = _parse_llm_intent(message, generation_service=generation_service)
        if llm_payload_result.get("ok"):
            try:
                llm_result = _normalize_llm_intent_payload(llm_payload_result["payload"])
            except ValidationError as exc:
                llm_result = None
                fallback_reason = f"llm output failed schema validation: {_validation_error_summary(exc)}"
                warnings.append(fallback_reason)
        else:
            fallback_reason = str(llm_payload_result.get("reason") or "llm unavailable")
            warnings.append(fallback_reason)

        rule_result = _build_rule_decision(message)

        if llm_result is not None:
            # LLM 只有在置信度和结构都满足时，才允许直接主导最终意图。
            llm_intent = str(llm_result.get("intent") or "unsupported")
            llm_confidence = llm_result.get("confidence")
            confidence_value = float(llm_confidence) if isinstance(llm_confidence, (int, float)) else None

            if confidence_value is None:
                # 没有置信度时，不信任 LLM 输出，退回到规则判断。
                fallback_reason = "llm confidence missing"
                warnings.append(fallback_reason)
                intent = str(rule_result.get("intent") or llm_intent)
                intent_source = "fallback"
                search_spec = rule_result.get("search_spec")
                search_spec_before_enrichment = rule_result.get("search_spec_before_enrichment")
                search_spec_after_enrichment = rule_result.get("search_spec_after_enrichment")
            elif llm_intent == "arxiv_search":
                # 搜索请求需要先拿到结构化 spec，再做规则补全和约束修正。
                search_spec = llm_result.get("search_spec")
                search_spec_before_enrichment = llm_result.get("search_spec_payload")
                if search_spec is None:
                    fallback_reason = "llm search intent is missing a valid search spec"
                    warnings.append(fallback_reason)
                    intent = str(rule_result.get("intent") or "unsupported")
                    intent_source = "fallback"
                    search_spec = rule_result.get("search_spec")
                    search_spec_before_enrichment = rule_result.get("search_spec_before_enrichment")
                    search_spec_after_enrichment = rule_result.get("search_spec_after_enrichment")
                else:
                    search_spec_after_enrichment = _compact_search_spec(search_spec)
                    enriched_spec = _apply_rule_enrichment(message, search_spec)
                    if enriched_spec is None:
                        fallback_reason = "rule enrichment failed after llm search parse"
                        warnings.append(fallback_reason)
                        intent = str(rule_result.get("intent") or "unclear")
                        intent_source = "fallback"
                        search_spec = rule_result.get("search_spec")
                        search_spec_before_enrichment = rule_result.get("search_spec_before_enrichment")
                        search_spec_after_enrichment = rule_result.get("search_spec_after_enrichment")
                    elif confidence_value is not None and confidence_value < LLM_CONFIDENCE_THRESHOLD:
                        # 低置信度搜索请求按规则结果回退，避免错误检索参数直接生效。
                        fallback_reason = f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
                        warnings.append(fallback_reason)
                        intent = str(rule_result.get("intent") or "arxiv_search")
                        intent_source = "fallback"
                        search_spec = rule_result.get("search_spec")
                        search_spec_before_enrichment = rule_result.get("search_spec_before_enrichment")
                        search_spec_after_enrichment = rule_result.get("search_spec_after_enrichment")
                    else:
                        intent = "arxiv_search"
                        intent_source = "llm"
                        search_spec = enriched_spec
                        search_spec_after_enrichment = _compact_search_spec(enriched_spec)
                        if rule_result and str(rule_result.get("intent") or "") != "arxiv_search":
                            # LLM 和规则对意图判断不一致时，只保留提示，不阻断流程。
                            warnings.append(
                                f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}"
                            )
            else:
                if confidence_value is not None and confidence_value < LLM_CONFIDENCE_THRESHOLD:
                    # 非搜索意图如果置信度不足，也先退回规则判断。
                    fallback_reason = f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
                    warnings.append(fallback_reason)
                    intent = str(rule_result.get("intent") or llm_intent)
                    intent_source = "fallback"
                    search_spec = rule_result.get("search_spec")
                    search_spec_before_enrichment = rule_result.get("search_spec_before_enrichment")
                    search_spec_after_enrichment = rule_result.get("search_spec_after_enrichment")
                else:
                    intent = llm_intent
                    intent_source = "llm"
                    if llm_intent in NON_SEARCH_INTENTS:
                        plan, next_actions, intent_warnings = _build_intent_guidance(llm_intent)
                        warnings.extend(intent_warnings)
                    if rule_result and str(rule_result.get("intent") or "") != llm_intent:
                        # 记录冲突，方便后续排查 LLM 与规则的分歧。
                        warnings.append(
                            f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}"
                        )
        else:
            # LLM 不可用时，完全走规则路径。
            intent = str(rule_result.get("intent") or "unsupported")
            intent_source = "fallback"
            search_spec = rule_result.get("search_spec")
            search_spec_before_enrichment = rule_result.get("search_spec_before_enrichment")
            search_spec_after_enrichment = rule_result.get("search_spec_after_enrichment")
            plan = list(rule_result.get("plan") or [])
            next_actions = list(rule_result.get("next_actions") or [])
            warnings.extend(str(item) for item in rule_result.get("warnings", []) if str(item).strip())
            if fallback_reason is None:
                fallback_reason = str(rule_result.get("reason") or "llm unavailable, rule fallback used")

    # 如果最后既没有 plan 也没有 next_actions，就补一组统一的引导文案。
    if not plan and not next_actions:
        plan, next_actions, intent_warnings = _build_intent_guidance(intent)
        warnings.extend(intent_warnings)

    # 只有搜索 intent 才允许因为缺少 search_spec 而回退为 unclear，避免把动作型请求误压成搜索问题。
    if intent == "arxiv_search" and search_spec is None:
        plan, next_actions, intent_warnings = _build_intent_guidance("unclear")
        warnings.extend(intent_warnings)
        intent = "unclear"
        if fallback_reason is None:
            fallback_reason = "search intent was downgraded because no valid search spec was produced"

    # 统一重置派生字段，保证后续节点从干净状态继续执行。
    normalized_state = current_state.model_copy(deep=True)
    normalized_state.intent = intent
    normalized_state.intent_source = intent_source
    normalized_state.fallback_reason = fallback_reason
    normalized_state.llm_confidence = float(llm_result.get("confidence")) if llm_result and isinstance(llm_result.get("confidence"), (int, float)) else None
    normalized_state.search_spec = search_spec
    normalized_state.plan = plan
    normalized_state.warnings = _dedupe_preserve_order(warnings)
    normalized_state.next_actions = next_actions
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


def _build_non_search_answer(intent: str) -> Tuple[str, List[str]]:
    if intent == "paper_summary":
        return (
            "我已经识别到你想总结某篇论文，但这个入口目前还没有接入论文总结能力。你可以先给我论文标题或 arXiv ID，后续再切到论文总结功能。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你要总结某篇论文，请提供标题或 arXiv ID",
            ],
        )

    if intent == "paper_detail":
        return (
            "我已经识别到你想解释某篇论文的方法或细节，但这个入口目前还没有接入论文详情解读能力。你可以先给我论文标题或 arXiv ID。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你已经有论文标题或 arXiv ID，请把它发给我",
            ],
        )

    if intent == "paper_qa":
        return (
            "我已经识别到你想围绕某篇论文提问，但这个入口目前还没有接入论文问答能力。你可以先给我目标论文标题或 arXiv ID。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你已经有论文标题或 arXiv ID，请把它发给我",
            ],
        )

    if intent == "recommendation":
        return (
            "我已经识别到你想做论文推荐，但这个入口目前还没有接入独立的推荐对话能力。你可以直接给出研究方向，我先帮你做 arXiv 检索。",
            [
                "如果你要的是搜索，请直接描述研究方向",
                "如果你想看推荐结果，请说明偏好方向或关键词",
            ],
        )

    if intent == "preference_action":
        return (
            "我已经识别到你想做偏好操作，但这个入口目前还没有直接接通偏好写入链路。你可以在论文卡片上继续点击喜欢、不喜欢或收藏。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你要标记某篇论文，请在论文卡片上操作",
            ],
        )

    if intent == "reading_list_action":
        return (
            "我已经识别到你想查看阅读列表或收藏列表，但这个入口目前还没有接通列表查询能力。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你要查看收藏，请切换到收藏页或列表页",
            ],
        )

    if intent == "unclear":
        return (
            "我能确定你是在找论文，但主题还不够明确。",
            [
                "补充研究方向、关键词或时间范围",
                "例如：RAG、LLM、Agent、NLP、推荐系统",
            ],
        )

    return (
        "当前请求超出 arXiv 搜索 Agent 的处理范围，请改写成明确的论文检索需求。",
        [
            "改写成 arXiv 论文搜索问题",
            "如果需要论文总结或问答，请先提供目标论文标题或 arXiv ID",
        ],
    )


_PREFERENCE_ORDINAL_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
    "十三": 13,
    "十四": 14,
    "十五": 15,
    "十六": 16,
    "十七": 17,
    "十八": 18,
    "十九": 19,
    "二十": 20,
}


def _normalize_context_paper(raw: Any) -> Dict[str, Any]:
    paper = raw if isinstance(raw, Mapping) else {}
    arxiv_id = str(paper.get("arxiv_id") or paper.get("arxivId") or paper.get("id") or "").strip()
    if arxiv_id.startswith("http"):
        arxiv_id = arxiv_id.rsplit("/", 1)[-1]

    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors_value = [item.strip() for item in authors.split(",") if item.strip()]
    elif isinstance(authors, (list, tuple, set)):
        authors_value = [str(item).strip() for item in authors if str(item).strip()]
    else:
        authors_value = []

    categories = paper.get("categories", [])
    if isinstance(categories, str):
        categories_value = [item.strip() for item in categories.split(",") if item.strip()]
    elif isinstance(categories, (list, tuple, set)):
        categories_value = [str(item).strip() for item in categories if str(item).strip()]
    else:
        categories_value = []

    abstract = str(paper.get("abstract") or paper.get("summary") or "").strip()
    title = str(paper.get("title") or "").strip()
    published_date = str(
        paper.get("published_date")
        or paper.get("published")
        or paper.get("publishedAt")
        or paper.get("updated")
        or paper.get("updatedAt")
        or ""
    ).strip()
    url = str(paper.get("url") or paper.get("abs_url") or paper.get("absUrl") or paper.get("pdf_url") or paper.get("pdfUrl") or "").strip()

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "summary": abstract,
        "authors": authors_value,
        "categories": categories_value,
        "published_date": published_date,
        "published": published_date,
        "url": url,
        "abs_url": str(paper.get("abs_url") or paper.get("absUrl") or paper.get("url") or "").strip(),
        "pdf_url": str(paper.get("pdf_url") or paper.get("pdfUrl") or "").strip(),
    }


def _extract_last_papers(context: Any) -> List[Dict[str, Any]]:
    if not isinstance(context, Mapping):
        return []
    last_papers = context.get("last_papers") or []
    if not isinstance(last_papers, list):
        return []
    return [_normalize_context_paper(paper) for paper in last_papers if isinstance(paper, Mapping)]


def _parse_preference_action(message: str) -> Optional[Dict[str, str]]:
    text = _normalize_text(message)
    lowered = text.lower()
    if not text:
        return None

    remove_patterns = (
        r"取消.*喜欢",
        r"取消.*不喜欢",
        r"取消.*标记",
        r"移除.*标记",
        r"撤销.*喜欢",
        r"撤销.*不喜欢",
        r"撤销.*标记",
    )
    dislike_patterns = (
        r"不喜欢",
        r"不感兴趣",
        r"标记.*不喜欢",
        r"标记.*不感兴趣",
        r"对.*不感兴趣",
    )
    like_patterns = (
        r"喜欢",
        r"感兴趣",
        r"收藏",
        r"标记.*喜欢",
        r"标记.*感兴趣",
        r"对.*感兴趣",
    )

    remove_scope = "both"
    if _matches_any(text, (r"取消.*喜欢", r"撤销.*喜欢")):
        remove_scope = "liked"
    elif _matches_any(text, (r"取消.*不喜欢", r"撤销.*不喜欢")):
        remove_scope = "disliked"

    if _matches_any(text, remove_patterns):
        return {"action": "remove", "remove_scope": remove_scope}
    if _matches_any(text, dislike_patterns) or "dislike" in lowered:
        return {"action": "dislike", "remove_scope": "none"}
    if _matches_any(text, like_patterns) or "like" in lowered:
        return {"action": "like", "remove_scope": "none"}
    return None


def _parse_target_reference(message: str) -> Optional[Dict[str, Any]]:
    text = _normalize_text(message)
    if not text:
        return None

    arxiv_match = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text, flags=re.IGNORECASE)
    if arxiv_match:
        return {
            "target_type": "arxiv_id",
            "target_value": arxiv_match.group(0),
            "arxiv_id": arxiv_match.group(0),
        }

    ordinal_match = re.search(r"(?:第\s*)?([一二三四五六七八九十]{1,3}|[1-9]|1[0-9]|20)\s*篇", text)
    if ordinal_match:
        raw_value = ordinal_match.group(1)
        ordinal = _PREFERENCE_ORDINAL_MAP.get(raw_value)
        if ordinal is None:
            ordinal = _safe_int(raw_value, default=0)
        if 1 <= ordinal <= 20:
            return {
                "target_type": "ordinal",
                "target_value": ordinal,
                "ordinal": ordinal,
            }

    bare_match = re.search(r"(?<!\d)([1-9]|1[0-9]|20)(?!\d)", text)
    if bare_match and (text.strip() in {bare_match.group(1), f"第{bare_match.group(1)}", f"第{bare_match.group(1)}篇"} or any(token in text for token in ("喜欢", "不喜欢", "收藏", "标记", "取消", "撤销"))):
        ordinal = _safe_int(bare_match.group(1), default=0)
        if 1 <= ordinal <= 20:
            return {
                "target_type": "ordinal",
                "target_value": ordinal,
                "ordinal": ordinal,
            }

    return None


def _resolve_paper_reference(message: str, context: Any) -> Dict[str, Any]:
    reference = _parse_target_reference(message)
    last_papers = _extract_last_papers(context)

    if reference is None:
        return {
            "status": "failed",
            "reason": "无法解析目标论文，请使用“第一篇 / 第二篇”或直接提供 arXiv ID",
            "target": None,
            "paper": None,
            "arxiv_id": None,
            "title": None,
        }

    if reference["target_type"] == "ordinal":
        ordinal = int(reference["target_value"])
        if not last_papers:
            return {
                "status": "failed",
                "reason": "没有可用的上一轮搜索结果，请先搜索论文，或者直接提供 arXiv ID",
                "target": reference,
                "paper": None,
                "arxiv_id": None,
                "title": None,
            }
        if ordinal > len(last_papers):
            return {
                "status": "failed",
                "reason": f"上一轮搜索结果只有 {len(last_papers)} 篇，无法选择第 {ordinal} 篇",
                "target": reference,
                "paper": None,
                "arxiv_id": None,
                "title": None,
            }
        paper = dict(last_papers[ordinal - 1])
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if not arxiv_id:
            return {
                "status": "failed",
                "reason": "上一轮结果中目标论文缺少 arXiv ID，无法执行偏好动作",
                "target": reference,
                "paper": paper,
                "arxiv_id": None,
                "title": paper.get("title"),
            }
        return {
            "status": "success",
            "reason": None,
            "target": reference,
            "paper": paper,
            "arxiv_id": arxiv_id,
            "title": paper.get("title"),
        }

    arxiv_id = str(reference.get("arxiv_id") or reference.get("target_value") or "").strip()
    if not arxiv_id:
        return {
            "status": "failed",
            "reason": "无法解析 arXiv ID",
            "target": reference,
            "paper": None,
            "arxiv_id": None,
            "title": None,
        }

    paper = next((paper for paper in last_papers if str(paper.get("arxiv_id") or "").strip() == arxiv_id), None)
    return {
        "status": "success",
        "reason": None,
        "target": reference,
        "paper": dict(paper) if paper is not None else None,
        "arxiv_id": arxiv_id,
        "title": paper.get("title") if isinstance(paper, Mapping) else None,
    }


def apply_preference_action(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "preference_action":
        return _append_step(
            next_state,
            step="preference_action_execution",
            status="skipped",
            action="执行论文偏好动作",
            inputs={"intent": next_state.intent},
            outputs={"reason": "当前请求不是 preference_action"},
        )

    message = _normalize_text(next_state.message or "")
    parsed_action = _parse_preference_action(message)
    resolution = _resolve_paper_reference(message, next_state.context or {})
    user_id = str(next_state.user_id or "default").strip() or "default"
    preference_service = get_recommendation_service()
    tool_args: Dict[str, Any] = {"user_id": user_id, "message": message}
    tool_name = "apply_preference_action"
    preference_result: Dict[str, Any]
    paper_payload = resolution.get("paper")
    arxiv_id = str(resolution.get("arxiv_id") or "").strip()
    paper_title = str(resolution.get("title") or (paper_payload or {}).get("title") or "").strip()

    if not next_state.plan:
        next_state.plan = [
            "解析偏好动作",
            "解析目标论文",
            "更新用户偏好",
        ]

    # 先把动作和目标解析清楚，再决定是否调用偏好服务，避免把无法定位论文的请求写进数据库。
    if parsed_action is None:
        preference_result = {
            "status": "failed",
            "action": "remove",
            "label": "none",
            "arxiv_id": None,
            "title": None,
            "message": "我没有识别到明确的偏好动作，请使用“喜欢 / 不喜欢 / 取消标记”",
            "paper": None,
            "error": "unsupported preference action",
        }
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [preference_result["message"]])
    elif resolution.get("status") != "success" or not arxiv_id:
        preference_result = {
            "status": "failed",
            "action": parsed_action["action"],
            "label": "none",
            "arxiv_id": resolution.get("arxiv_id"),
            "title": resolution.get("title"),
            "message": str(resolution.get("reason") or "无法解析目标论文，请先搜索论文或直接提供 arXiv ID"),
            "paper": paper_payload,
            "error": str(resolution.get("reason") or "paper reference resolution failed"),
        }
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [preference_result["message"]])
    else:
        try:
            if parsed_action["action"] in {"like", "dislike"}:
                tool_name = "record_user_paper_preference"
                liked = parsed_action["action"] == "like"
                service_result = preference_service.record_user_paper_preference(
                    user_id=user_id,
                    arxiv_id=arxiv_id,
                    liked=liked,
                    paper_payload=paper_payload,
                )
                preference_result = {
                    "status": "success",
                    "action": parsed_action["action"],
                    "label": "liked" if liked else "disliked",
                    "arxiv_id": arxiv_id,
                    "title": paper_title,
                    "message": str(service_result.get("message") or "偏好已更新"),
                    "paper": service_result.get("paper") or paper_payload,
                    "error": None,
                }
            else:
                tool_name = "remove_user_paper_preference"
                remove_scope = str(parsed_action.get("remove_scope") or "both")
                removed_liked = False
                removed_disliked = False
                remove_errors: List[str] = []
                if remove_scope in {"both", "liked"}:
                    removed_liked = bool(preference_service.db_service.remove_liked_paper(user_id=user_id, arxiv_id=arxiv_id))
                if remove_scope in {"both", "disliked"}:
                    removed_disliked = bool(preference_service.db_service.remove_disliked_paper(user_id=user_id, arxiv_id=arxiv_id))
                if not removed_liked and not removed_disliked:
                    remove_errors.append("未找到可移除的喜欢/不喜欢标记")
                preference_result = {
                    "status": "success" if (removed_liked or removed_disliked) else "failed",
                    "action": "remove",
                    "label": "none",
                    "arxiv_id": arxiv_id,
                    "title": paper_title,
                    "message": "已取消偏好标记" if (removed_liked or removed_disliked) else "未找到可取消的偏好标记",
                    "paper": paper_payload,
                    "error": None if (removed_liked or removed_disliked) else "; ".join(remove_errors),
                }
                if not (removed_liked or removed_disliked):
                    next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + remove_errors)
        except Exception as exc:
            preference_result = {
                "status": "failed",
                "action": parsed_action["action"],
                "label": "none",
                "arxiv_id": arxiv_id,
                "title": paper_title,
                "message": f"偏好动作执行失败：{exc}",
                "paper": paper_payload,
                "error": str(exc),
            }
            next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [str(exc)])

    next_state.preference_action_result = preference_result
    if not next_state.next_actions:
        next_state.next_actions = [
            "继续对其他论文执行喜欢、不喜欢或收藏动作",
            "也可以继续搜索、查看推荐或打开论文详情",
        ]

    next_state.tool_name = tool_name
    next_state.tool_args = tool_args
    next_state.tool_result = dict(preference_result)
    next_state.tool_calls = list(next_state.tool_calls or []) + [
        AgentToolCall(
            tool_name=tool_name,
            arguments=tool_args,
            status=preference_result["status"],
            summary=str(preference_result["message"]),
            trace={
                "action": parsed_action["action"] if parsed_action else None,
                "target": resolution.get("target"),
                "arxiv_id": preference_result.get("arxiv_id"),
            },
            error={"message": preference_result["error"]} if preference_result.get("error") else None,
        )
    ]

    return _append_step(
        next_state,
        step="preference_action_execution",
        status="success" if preference_result["status"] == "success" else "failed",
        action="执行论文偏好动作",
        inputs={"intent": next_state.intent, "message": next_state.message, "context": next_state.context},
        outputs={
            "preference_action_result": dict(preference_result),
            "plan": list(next_state.plan),
            "next_actions": list(next_state.next_actions),
        },
        error=None if preference_result["status"] == "success" else str(preference_result.get("error") or preference_result.get("message") or "preference action failed"),
    )


def synthesize_response(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent == "preference_action":
        result = next_state.preference_action_result or {}
        title = str(result.get("title") or "").strip()
        arxiv_id = str(result.get("arxiv_id") or "").strip()
        action = str(result.get("action") or "remove")
        label = str(result.get("label") or "none")
        message = str(result.get("message") or "").strip()
        error = str(result.get("error") or "").strip()

        if result.get("status") == "success":
            if action == "like":
                next_state.answer = "已将该论文标记为感兴趣。"
            elif action == "dislike":
                next_state.answer = "已将该论文标记为不感兴趣。"
            else:
                next_state.answer = "已取消该论文的偏好标记。"
            if title:
                next_state.answer += f"\n\n《{title}》"
            if arxiv_id:
                next_state.answer += f"\narXiv ID: {arxiv_id}"
            if action == "like":
                next_state.answer += "\n\n后续推荐会参考这个偏好。"
            elif action == "dislike":
                next_state.answer += "\n\n后续推荐会尽量降低类似论文的权重。"
            else:
                next_state.answer += "\n\n后续推荐会恢复对这篇论文的中性处理。"
            next_state.next_actions = [
                "继续对其他论文执行喜欢、不喜欢或收藏动作",
                "也可以继续搜索、查看推荐或打开论文详情",
            ]
        else:
            next_state.answer = message or error or "偏好动作执行失败。"
            next_state.next_actions = [
                "先搜索论文，再使用“第一篇 / 第二篇”来标记",
                "也可以直接提供 arXiv ID 后重试",
            ]

        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={
                "intent": next_state.intent,
                "preference_action_result": dict(result),
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
                "label": label,
            },
        )

    if next_state.intent == "arxiv_search":
        spec = next_state.search_spec
        papers = list(next_state.papers or [])
        paper_count = len(papers)
        max_results = spec.max_results if spec is not None else 10
        summary = _summarize_search_spec(spec)
        priority_titles = _collect_priority_titles(papers, limit=3)
        personalized_applied = bool(next_state.personalized_rerank_applied)

        if paper_count > 0:
            next_state.answer = f"已按“{summary}”搜索 arXiv，当前返回 {paper_count} 篇论文。"
            if personalized_applied:
                if priority_titles:
                    next_state.answer += f" 本次结果已根据用户偏好重新排序，建议优先阅读：{', '.join(priority_titles)}。"
                else:
                    next_state.answer += " 本次结果已根据用户偏好重新排序，建议优先阅读排序靠前的论文。"
            else:
                next_state.answer += " 本次结果未使用用户偏好向量，保持普通搜索排序。"
            next_state.next_actions = [
                "继续缩小到某个子方向搜索",
                "选择一篇论文查看详情",
                "后续可以接论文总结或 QA 功能",
            ]
        else:
            next_state.answer = f"已按“{summary}”搜索 arXiv，但当前没有找到结果。"
            next_state.next_actions = [
                "放宽关键词或扩大时间范围后重试",
                "只保留核心主题词再搜索",
                "后续可以接论文总结或 QA 功能",
            ]

        if max_results and paper_count < max_results:
            next_state.answer += f" 本次最多期望返回 {max_results} 篇。"

        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={
                "intent": next_state.intent,
                "paper_count": paper_count,
                "personalized_rerank_applied": personalized_applied,
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
                "top_papers": _collect_priority_titles(papers, limit=3),
            },
        )

    answer, next_actions = _build_non_search_answer(next_state.intent or "unsupported")
    next_state.answer = answer
    next_state.next_actions = next_actions
    return _append_step(
        next_state,
        step="final_answer_generation",
        status="success",
        action="生成最终答复并给出后续动作",
        inputs={
            "intent": next_state.intent,
            "paper_count": len(next_state.papers or []),
        },
        outputs={
            "answer": next_state.answer,
            "next_actions": list(next_state.next_actions),
        },
    )


__all__ = [
    "SEARCH_TOOL_NAME",
    "apply_preference_action",
    "build_search_tool_args",
    "check_search_result",
    "invoke_search_tool",
    "parse_search_request",
    "personalized_rank_and_annotate_papers",
    "route_after_parse",
    "synthesize_response",
]
