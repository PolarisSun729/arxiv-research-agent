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

from .schemas import AgentToolCall, ArxivSearchSpec, get_valid_arxiv_categories
from .state import AgentState

SEARCH_TOOL_NAME = "search_arxiv_structured"
SUPPORTED_INTENTS = {"arxiv_search", "unclear", "unsupported"}

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
    return normalized_state


def build_search_tool_args(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search" or next_state.search_spec is None:
        next_state.tool_name = None
        next_state.tool_args = {}
        return next_state

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
    return next_state


def invoke_search_tool(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search":
        return next_state

    if next_state.tool_name != SEARCH_TOOL_NAME or not next_state.tool_args:
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + ["搜索工具参数未准备好，跳过工具调用"],
        )
        return next_state

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
                error={"message": str(exc)},
            )
        ]
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + ["工具调用失败，请检查搜索参数或 arXiv 服务状态"],
        )
        next_state.papers = []
        return next_state

    result = _to_plain_dict(raw_result)
    next_state.tool_result = result

    tool_call = AgentToolCall(
        tool_name=SEARCH_TOOL_NAME,
        arguments=dict(next_state.tool_args),
        status="success" if _result_ok(result) else "failed",
        summary=_result_text(result, "summary"),
        trace=_result_mapping(result, "trace"),
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

    return next_state


def check_search_result(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search":
        return next_state

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
    return next_state


def synthesize_response(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent == "unsupported":
        next_state.answer = "当前功能只支持自然语言 arXiv 论文搜索；如果你想查论文，请改成明确的搜索需求。"
        next_state.next_actions = [
            "改写为 arXiv 论文搜索问题后重试",
            "后续可以接入论文总结或 QA 功能",
        ]
        return next_state

    if next_state.intent == "unclear":
        next_state.answer = "你想搜索论文，但主题还不够明确。请补充研究方向、关键词或类别后重试。"
        next_state.next_actions = [
            "补充研究方向或关键词",
            "例如：RAG、LLM、Agent、NLP、推荐系统",
        ]
        return next_state

    if next_state.intent != "arxiv_search":
        next_state.answer = "当前请求暂时无法处理，请改写为 arXiv 论文搜索需求。"
        next_state.next_actions = ["改写为 arXiv 搜索问题后重试"]
        return next_state

    spec = next_state.search_spec
    papers = list(next_state.papers or [])
    paper_count = len(papers)
    max_results = spec.max_results if spec is not None else 10
    summary = _summarize_search_spec(spec)

    if paper_count > 0:
        next_state.answer = f"已按“{summary}”搜索 arXiv，当前返回 {paper_count} 篇论文。"
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

    return next_state


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
                if not enriched.categories:
                    warnings.append("未能自动补全 categories")
                return enriched, _dedupe_preserve_order(warnings)
        warnings.append("LLM 输出未通过校验，已回退到规则解析")

    try:
        spec = _build_spec_from_rules(message)
    except ValidationError as exc:
        warnings.append(f"规则解析结果未通过结构校验: {_validation_error_summary(exc)}")
        return None, _dedupe_preserve_order(warnings)

    if spec is not None and not spec.categories:
        warnings.append("未能自动补全 categories")
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
        "\"categories\":[\"cs.AI\"],"
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
        f"User message: {message}"
    )

    try:
        response = generation_service.complete_with_qwen(prompt)
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
        categories=_normalize_categories(payload.get("categories")),
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
    categories = _infer_categories(message, query, title_query, abstract_query)
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
    categories = list(spec.categories or []) or _infer_categories(message, query, title_query, abstract_query)
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


def _coerce_state(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    return AgentState.model_validate(dict(state))


__all__ = [
    "SEARCH_TOOL_NAME",
    "build_search_tool_args",
    "check_search_result",
    "invoke_search_tool",
    "parse_search_request",
    "route_after_parse",
    "synthesize_response",
]
