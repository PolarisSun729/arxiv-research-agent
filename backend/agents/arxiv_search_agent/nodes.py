from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
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

try:  # pragma: no cover - import path differs between backend cwd and package import
    from dependencies import get_paper_qa_service
except ModuleNotFoundError:  # pragma: no cover
    from backend.dependencies import get_paper_qa_service

try:  # pragma: no cover - import path differs between backend cwd and package import
    from dependencies import get_generation_service
except ModuleNotFoundError:  # pragma: no cover
    from backend.dependencies import get_generation_service

from fastapi import HTTPException

from .schemas import AgentStep, AgentToolCall, ArxivSearchSpec, get_default_agent_arxiv_categories, get_valid_arxiv_categories
from .state import AgentState

logger = logging.getLogger(__name__)

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

HARD_RULE_PATTERNS: Dict[str, Sequence[str]] = {}

LLM_INTENT_HINTS: Dict[str, Sequence[str]] = {
    "paper_summary": ("summary", "summarize", "概述", "总结", "讲什么"),
    "paper_detail": ("detail", "method", "explain", "方法", "细节", "流程"),
    "paper_qa": ("qa", "question", "ask", "问", "提问"),
    "recommendation": ("recommend", "推荐", "suggest"),
    "preference_action": ("like", "dislike", "收藏", "喜欢", "不喜欢", "取消", "撤销"),
    "reading_list_action": ("reading list", "阅读列表", "收藏夹", "我的收藏"),
}

SEARCH_TRIGGER_PATTERNS: Sequence[str] = ()


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


def _contains_any_term(message: str, terms: Sequence[str]) -> bool:
    # 中英文混合匹配：英文统一按 lower() 做子串判断，中文保留原文匹配。
    lowered = message.lower()
    for term in terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        if normalized.isascii():
            if normalized.lower() in lowered:
                return True
        elif normalized in message:
            return True
    return False


def _references_specific_paper(message: str) -> bool:
    # 详情/总结/问答/偏好动作通常都需要一个明确论文目标，这里先做轻量识别。
    lowered = message.lower()
    if bool(re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", lowered)):
        return True
    return _contains_any_term(
        message,
        (
            "这篇论文",
            "本文",
            "该论文",
            "这篇 paper",
            "this paper",
            "the paper",
            "paper id",
            "arxiv id",
            "标题",
            "title",
        ),
    )


def _looks_like_preference_action_request(message: str) -> bool:
    # 先排除“查看列表”类请求，避免把“打开收藏夹”误判成“收藏某篇论文”。
    if _contains_any_term(message, ("阅读列表", "收藏夹", "reading list", "favorites", "bookmarks")):
        return False
    action_hit = _contains_any_term(
        message,
        (
            "喜欢",
            "不喜欢",
            "收藏",
            "取消收藏",
            "加入收藏",
            "加入待读",
            "取消喜欢",
            "bookmark",
            "favorite",
            "favourite",
            "dislike",
            "thumbs up",
            "thumbs down",
        ),
    )
    if not action_hit:
        return False
    return _references_specific_paper(message) or _contains_any_term(message, ("第一篇", "第二篇", "第1篇", "第2篇", "这篇"))


def _looks_like_reading_list_action_request(message: str) -> bool:
    return _contains_any_term(
        message,
        (
            "阅读列表",
            "待读",
            "待读列表",
            "收藏夹",
            "我的收藏",
            "favorites",
            "bookmarks",
            "reading list",
            "saved papers",
        ),
    )


def _looks_like_paper_summary_request(message: str) -> bool:
    # 用户说“摘要包含 xxx”时更像搜索约束，不应该被 summary intent 抢走。
    if _matches_any(message, ABSTRACT_HINT_PATTERNS):
        return False
    return _contains_any_term(
        message,
        (
            "总结",
            "概述",
            "概括",
            "摘要一下",
            "summarize",
            "summary",
            "tl;dr",
        ),
    ) and not _looks_search_like(message)


def _looks_like_paper_detail_request(message: str) -> bool:
    # 先排除普通搜索请求，避免“找讲某个方法的论文”被误判成论文详情。
    if _looks_search_like(message):
        return False
    detail_hit = _contains_any_term(
        message,
        (
            "解释",
            "讲讲",
            "介绍一下",
            "方法",
            "贡献",
            "细节",
            "流程",
            "第一节",
            "第二节",
            "method",
            "approach",
            "contribution",
            "details",
            "section",
        ),
    )
    return detail_hit and (_references_specific_paper(message) or _contains_any_term(message, ("这篇", "本文", "paper")))


def _looks_like_paper_qa_request(message: str) -> bool:
    # QA 需要同时满足“像个问题”以及“指向具体论文”两个条件。
    question_hit = (
        "?" in message
        or "？" in message
        or _contains_any_term(message, ("为什么", "怎么", "是否", "能否", "区别", "question", "ask"))
    )
    if not question_hit or _looks_search_like(message):
        return False
    return _references_specific_paper(message) or _contains_any_term(message, ("这篇", "本文", "paper"))


def _looks_like_recommendation_request(message: str) -> bool:
    lowered = message.lower()
    # 这里只识别“个性化推荐”语义，避免“推荐几篇 xxx 论文”误入 recommendation 分支。
    if "推荐系统" in message or "recommender system" in lowered or "recommendation system" in lowered:
        return False
    return bool(
        _contains_any_term(
            message,
            (
                "根据我的兴趣",
                "基于我的兴趣",
                "根据我的偏好",
                "基于我的偏好",
                "个性化推荐",
                "为我推荐",
                "适合我",
                "我可能感兴趣",
                "我感兴趣",
            ),
        )
        or re.search(r"recommend(?: me)?(?: papers?| articles?)? based on my (?:interests?|preferences?)", lowered)
        or re.search(r"personalized (?:paper )?recommendations?", lowered)
    )


def _build_llm_prompt(message: str) -> str:
    return _build_llm_prompt_with_profile(message, research_profile=None)


def _compact_research_profile_for_prompt(research_profile: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(research_profile, Mapping):
        return "N/A"

    parts: List[str] = []
    for key, label in (
        ("positive_topics", "positive_topics"),
        ("negative_topics", "negative_topics"),
        ("recent_topics", "recent_topics"),
        ("preferred_categories", "preferred_categories"),
        ("common_question_types", "common_question_types"),
        ("representative_papers", "representative_papers"),
    ):
        raw_value = research_profile.get(key)
        if isinstance(raw_value, list):
            values = [str(item).strip() for item in raw_value if str(item).strip()]
            if values:
                parts.append(f"{label}: {', '.join(values[:5])}")

    preferred_answer_style = str(research_profile.get("preferred_answer_style") or "").strip()
    if preferred_answer_style:
        parts.append(f"preferred_answer_style: {preferred_answer_style}")

    return " | ".join(parts) if parts else "N/A"


def _build_llm_prompt_with_profile(message: str, research_profile: Optional[Mapping[str, Any]] = None) -> str:
    profile_hint = _compact_research_profile_for_prompt(research_profile)
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
        "\"cleaned_topic_cn\":null|string,"
        "\"cleaned_topic_en\":null|string,"
        "\"missing_info\":[],"
        "\"warnings\":[],"
        "\"next_actions\":[]"
        "}\n"
        "Intent classification rules:\n"
        "- Use arxiv_search when the user wants to search papers on arXiv.\n"
        "- Use paper_summary when the user asks to summarize a paper.\n"
        "- Use paper_detail when the user asks to explain a paper's method, contribution, or first section/content.\n"
        "- Use paper_qa when the user asks questions about a specific paper.\n"
        "- Use recommendation only when the user wants personalized paper recommendations based on their interests or preferences.\n"
        "- If the user asks for topic-based recommendations such as \"推荐几篇 xxx 论文\", classify it as arxiv_search instead of recommendation.\n"
        "- Use preference_action when the user wants to like, dislike, favorite, or bookmark a paper.\n"
        "- Use reading_list_action when the user wants to inspect a reading list or favorites list.\n"
        "- Use unclear if the topic is missing or too vague.\n"
        "- Use unsupported only if the request is clearly outside the paper system.\n"
        "Search spec cleaning rules (for arxiv_search intent):\n"
        "- Remove ALL polite words: 请, 请你, 帮我, 给我, 麻烦, 谢谢, etc.\n"
        "- Remove ALL action words: 找, 搜索, 检索, 查找, 推荐, 看看, 寻找, etc.\n"
        "- Remove ALL quantifiers: 几篇, 一些, 几个, N篇, N个, 一篇, 两篇, etc.\n"
        "- Remove ALL filler words: 和...相关的, 关于, 的, 论文, 文献, paper, papers, arxiv.\n"
        "- KEEP only the research topic keywords.\n"
        "- Convert Chinese research topics to English keywords suitable for arXiv search.\n"
        "  Examples: 知识图谱构建->knowledge graph construction, 实体链接->entity linking, 大语言模型->large language model.\n"
        "- If unsure about exact translation, use broader but searchable English terms.\n"
        "- The \"query\" field MUST be the English keywords, NOT the original Chinese words.\n"
        "- The \"cleaned_topic_cn\" field should contain the cleaned Chinese topic keywords.\n"
        "- The \"cleaned_topic_en\" field should contain the English search keywords.\n"
        "- For normal natural language search, ONLY set the \"query\" field.\n"
        "- Do NOT set both \"query\" and \"abstract_query\" unless the user explicitly says \"摘要包含...\" or \"abstract\".\n"
        "- Do NOT set \"title_query\" unless the user explicitly says \"标题包含...\" or \"title\".\n"
        "- Use max_results between 1 and 20. Default to 10 if not specified.\n"
        "- Use submitted_days_ago for recent-time expressions.\n"
        "- Do not output categories; the system applies a fixed configured category scope.\n"
        "- The research profile is only a lightweight personalization hint. Use it to disambiguate vague requests or recommendation intent, but never override the user's explicit request.\n"
        f"User research profile: {profile_hint}\n"
        f"User message: {message}"
    )


def _parse_llm_intent(
    message: str,
    generation_service: Optional[Any],
    research_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if generation_service is None or not hasattr(generation_service, "complete_with_qwen"):
        return {
            "ok": False,
            "reason": "llm service is unavailable",
            "payload": None,
        }

    try:
        # 意图识别只负责分流，不需要大模型推理，优先使用小模型降低时延和成本。
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
    cleaning_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
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


def _legacy_parse_search_request(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    current_state = _coerce_state(state)
    message = _normalize_text(current_state.message or "")

    # 这一段只做“意图分流 + 搜索参数准备”，不直接执行工具。
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
    cleaning_debug: Dict[str, Any] = {}

    def _apply_rule_fallback(default_intent: str = "unsupported") -> None:
        # 把规则系统的产物一次性映射到当前上下文，避免多个分支重复赋值。
        nonlocal intent, intent_source, search_spec, search_spec_before_enrichment, search_spec_after_enrichment
        nonlocal plan, next_actions
        resolved = rule_result or {}
        intent = str(resolved.get("intent") or default_intent)
        intent_source = "fallback"
        search_spec = resolved.get("search_spec")
        search_spec_before_enrichment = resolved.get("search_spec_before_enrichment")
        search_spec_after_enrichment = resolved.get("search_spec_after_enrichment")
        plan = list(resolved.get("plan") or [])
        next_actions = list(resolved.get("next_actions") or [])
        warnings.extend(str(item) for item in (resolved.get("warnings") or []) if str(item).strip())

    # 先尝试 LLM：它更擅长区分 search / summary / detail / QA / recommendation 等多意图。
    llm_payload_result = _parse_llm_intent(
        message,
        generation_service=generation_service,
        research_profile=state.context.get("research_profile") if isinstance(state.context, dict) else None,
    )
    if llm_payload_result.get("ok"):
        try:
            llm_result = _normalize_llm_intent_payload(llm_payload_result["payload"])
            warnings.extend(str(item) for item in (llm_result.get("warnings") or []) if str(item).strip())
            raw_payload = llm_payload_result.get("payload")
            if isinstance(raw_payload, dict):
                cleaning_debug["cleaned_topic_cn"] = _normalize_optional_str(raw_payload.get("cleaned_topic_cn"))
                cleaning_debug["cleaned_topic_en"] = _normalize_optional_str(raw_payload.get("cleaned_topic_en"))
        except ValidationError as exc:
            llm_result = None
            fallback_reason = f"llm output failed schema validation: {_validation_error_summary(exc)}"
            warnings.append(fallback_reason)
    else:
        fallback_reason = str(llm_payload_result.get("reason") or "llm unavailable")
        warnings.append(fallback_reason)

    # 无论 LLM 成不成功，都先准备一份规则结果，后面用来兜底或做冲突比对。
    rule_result = _build_rule_decision(message)

    if llm_result is not None:
        llm_intent = str(llm_result.get("intent") or "unsupported")
        llm_confidence = llm_result.get("confidence")
        confidence_value = float(llm_confidence) if isinstance(llm_confidence, (int, float)) else None

        if confidence_value is None:
            # 没有置信度的 LLM 结果不直接采用，退回规则分流。
            fallback_reason = "llm confidence missing"
            warnings.append(fallback_reason)
            _apply_rule_fallback(default_intent=llm_intent)
        elif llm_intent == "arxiv_search":
            # 搜索意图除了分类正确，还必须产出合法 search spec 才能进入工具调用链路。
            search_spec = llm_result.get("search_spec")
            search_spec_before_enrichment = llm_result.get("search_spec_payload")
            if search_spec is None:
                fallback_reason = "llm search intent is missing a valid search spec"
                warnings.append(fallback_reason)
                _apply_rule_fallback(default_intent="unclear")
            else:
                # 先做清洗与约束修正，再用规则补全时间/排序/类别等默认值。
                search_spec, post_warnings = _post_process_cleaned_spec(
                    search_spec,
                    message,
                    cleaned_topic_cn=cleaning_debug.get("cleaned_topic_cn"),
                    cleaned_topic_en=cleaning_debug.get("cleaned_topic_en"),
                )
                warnings.extend(post_warnings)
                search_spec_after_enrichment = _compact_search_spec(search_spec)
                enriched_spec = _apply_rule_enrichment(message, search_spec)
                if enriched_spec is None:
                    fallback_reason = "rule enrichment failed after llm search parse"
                    warnings.append(fallback_reason)
                    _apply_rule_fallback(default_intent="unclear")
                elif confidence_value < LLM_CONFIDENCE_THRESHOLD:
                    fallback_reason = (
                        f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
                    )
                    warnings.append(fallback_reason)
                    _apply_rule_fallback(default_intent="arxiv_search")
                else:
                    # 高置信度且 spec 合法时，才真正采用 LLM 主导的搜索结果。
                    intent = "arxiv_search"
                    intent_source = "llm"
                    search_spec = enriched_spec
                    search_spec_after_enrichment = _compact_search_spec(enriched_spec)
                    plan, next_actions, intent_warnings = _build_intent_guidance(intent)
                    warnings.extend(intent_warnings)
                    if rule_result and str(rule_result.get("intent") or "") != "arxiv_search":
                        warnings.append(
                            f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}"
                        )
        else:
            # 非搜索意图不要求 search spec，只看 intent + confidence 是否可信。
            if confidence_value < LLM_CONFIDENCE_THRESHOLD:
                fallback_reason = (
                    f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
                )
                warnings.append(fallback_reason)
                _apply_rule_fallback(default_intent=llm_intent)
            else:
                intent = llm_intent
                intent_source = "llm"
                plan, next_actions, intent_warnings = _build_intent_guidance(llm_intent)
                warnings.extend(intent_warnings)
                if rule_result and str(rule_result.get("intent") or "") != llm_intent:
                    warnings.append(
                        f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}"
                    )
    else:
        # LLM 不可用时，整个流程完全退回规则系统。
        _apply_rule_fallback(default_intent="unsupported")
        if fallback_reason is None:
            fallback_reason = str((rule_result or {}).get("reason") or "llm unavailable, rule fallback used")

    # 某些 fallback 分支只给了 intent，统一在这里补齐用户可见的 plan/next_actions。
    if not plan and not next_actions:
        plan, next_actions, intent_warnings = _build_intent_guidance(intent)
        warnings.extend(intent_warnings)

    # 只有搜索意图才依赖 search_spec；动作类/详情类请求不能因为没有 spec 被错误打回。
    if intent == "arxiv_search" and search_spec is None:
        plan, next_actions, intent_warnings = _build_intent_guidance("unclear")
        warnings.extend(intent_warnings)
        intent = "unclear"
        if fallback_reason is None:
            fallback_reason = "search intent was downgraded because no valid search spec was produced"

    if search_spec is not None:
        cleaning_debug["final_query"] = search_spec.query
        cleaning_debug["final_title_query"] = search_spec.title_query
        cleaning_debug["final_abstract_query"] = search_spec.abstract_query

    # 统一重置派生字段，确保后续节点从一个干净、可追踪的状态继续运行。
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
    normalized_state.plan = plan
    normalized_state.warnings = _dedupe_preserve_order(warnings)
    normalized_state.next_actions = next_actions
    # 新一轮检索必须从干净状态开始，避免上一次的 fallback 轮次污染当前搜索。
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
            retry_count = int(next_state.search_retry_count or 0)
            current_query = next_state.search_spec.query if next_state.search_spec else None
            if retry_count < MAX_SEARCH_RETRIES:
                warnings.append(
                    f"搜索结果为空（第 {retry_count + 1} 轮），"
                    f"将自动放宽关键词后重试"
                )
            else:
                warnings.append("搜索结果为空，已尝试多轮放宽关键词，建议扩大时间范围或减少关键词")
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


def relax_search_for_retry(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """当搜索结果为空时，放宽关键词后重试。"""
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    retry_count = int(next_state.search_retry_count or 0)
    next_state.search_retry_count = retry_count + 1

    if next_state.search_spec is not None:
        original_query = next_state.search_spec.query
        relaxed_query = _relax_query_for_fallback(original_query, next_state.search_retry_count)

        # 记录 fallback 历史
        fallback_record = {
            "round": next_state.search_retry_count,
            "original_query": original_query,
            "relaxed_query": relaxed_query,
            "categories": list(next_state.search_spec.categories or []),
        }
        next_state.fallback_specs = list(next_state.fallback_specs) + [fallback_record]

        # 更新 debug 信息
        debug = dict(next_state.debug or {})
        debug["fallback_round"] = next_state.search_retry_count
        fallback_queries = list(debug.get("fallback_queries", []))
        fallback_queries.append({
            "round": next_state.search_retry_count,
            "query": relaxed_query,
        })
        debug["fallback_queries"] = fallback_queries
        next_state.debug = debug

        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + [
                f"自动放宽关键词: \"{original_query}\" -> \"{relaxed_query or '(仅按类别搜索)'}\""
            ]
        )

        # 更新 search_spec 为放宽后的查询
        next_state.search_spec.query = relaxed_query
        # 重新构建 reasoning_summary
        next_state.search_spec.reasoning_summary = _build_reasoning_summary(
            relaxed_query,
            list(next_state.search_spec.categories or []),
            None,
            next_state.search_spec.max_results or 10,
            next_state.search_spec.sort_by or "submittedDate",
        ) + f" (fallback round {next_state.search_retry_count})"

    # 重置搜索工具调用状态，准备重新执行
    next_state.tool_name = None
    next_state.tool_args = {}
    next_state.tool_result = None
    next_state.papers = []

    return _append_step(
        next_state,
        step="search_fallback_retry",
        status="success",
        action=f"搜索结果为空，自动放宽关键词后重试（第 {next_state.search_retry_count} 轮）",
        inputs={
            "retry_count": next_state.search_retry_count,
            "relaxed_query": next_state.search_spec.query if next_state.search_spec else None,
        },
        outputs={
            "new_query": next_state.search_spec.query if next_state.search_spec else None,
            "fallback_specs": list(next_state.fallback_specs),
        },
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
    pending_action = dict(next_state.pending_action or (next_state.context or {}).get("pending_action") or {})
    pending_decision = str((next_state.debug or {}).get("pending_action_decision") or "").strip().lower()

    if pending_action and pending_decision == "reject":
        title = str(pending_action.get("title") or "").strip()
        arxiv_id = str(pending_action.get("arxiv_id") or "").strip()
        next_state.pending_action = None
        next_state.context = dict(next_state.context or {})
        next_state.context.pop("pending_action", None)
        next_state.answer = (
            f"已取消这次论文解析任务。{f'《{title}》' if title else ''}{f' arXiv ID: {arxiv_id}' if arxiv_id else ''}"
        ).strip()
        next_state.next_actions = [
            "如果还想继续问这篇论文，可以稍后再次发起解析",
            "也可以重新搜索或指定另一篇论文",
        ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={"intent": next_state.intent, "pending_action_decision": pending_decision},
            outputs={"answer": next_state.answer, "next_actions": list(next_state.next_actions)},
        )

    if pending_action and pending_decision in {"unclear", "unrelated"}:
        next_state.answer = (
            "我这边还有一个待确认的论文解析任务。"
            "如果你要继续执行，请回复“确认 / 继续 / 解析”；如果不想执行，请回复“取消”。"
        )
        next_state.next_actions = [
            "回复“解析”继续原来的论文问答任务",
            "回复“取消”放弃当前待解析任务",
            "如果你想发起新问题，请在下一条消息里直接给出完整请求",
        ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={"intent": next_state.intent, "pending_action_decision": pending_decision},
            outputs={"answer": next_state.answer, "next_actions": list(next_state.next_actions)},
        )

    if isinstance(next_state.paper_qa_result, Mapping) and next_state.paper_qa_result:
        if not next_state.answer:
            next_state.answer = str(next_state.paper_qa_result.get("answer") or next_state.paper_qa_result.get("error") or "").strip()
        if not next_state.next_actions:
            next_state.next_actions = [
                "继续追问这篇论文的其他细节",
                "或者切换到另一篇论文继续阅读",
            ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={
                "intent": next_state.intent,
                "paper_qa_status": next_state.paper_qa_result.get("status"),
                "paper_count": len(next_state.papers or []),
            },
            outputs={"answer": next_state.answer, "next_actions": list(next_state.next_actions)},
        )

    if next_state.intent == "preference_action" and isinstance(next_state.preference_action_result, Mapping):
        if not next_state.answer:
            next_state.answer = str(next_state.preference_action_result.get("message") or "偏好动作已处理").strip()
        if not next_state.next_actions:
            next_state.next_actions = [
                "继续对其他论文执行喜欢、不喜欢或取消标记",
                "也可以继续搜索或查看论文详情",
            ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={"intent": next_state.intent, "paper_count": len(next_state.papers or [])},
            outputs={"answer": next_state.answer, "next_actions": list(next_state.next_actions)},
        )

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
        next_state.answer, next_state.next_actions = _build_non_search_answer(next_state.intent or "unsupported")
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
        retry_count = int(next_state.search_retry_count or 0)
        next_state.answer = f"已按“{summary}”搜索 arXiv，当前返回 {paper_count} 篇论文。"
        if retry_count > 0:
            next_state.answer += f" 严格关键词搜索没有结果，已自动放宽关键词后返回以下论文。"
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


def _legacy_build_search_spec(
    message: str,
    generation_service: Optional[Any] = None,
) -> Tuple[Optional[ArxivSearchSpec], List[str], Dict[str, Any]]:
    warnings: List[str] = []
    debug_info: Dict[str, Any] = {}

    llm_payload = _parse_with_llm(message, generation_service=generation_service)
    if llm_payload is not None:
        # 提取清洗过程的 debug 信息
        debug_info["cleaned_topic_cn"] = _normalize_optional_str(llm_payload.get("cleaned_topic_cn"))
        debug_info["cleaned_topic_en"] = _normalize_optional_str(llm_payload.get("cleaned_topic_en"))
        try:
            llm_spec = _normalize_and_validate_spec(llm_payload)
        except ValidationError as exc:
            llm_spec = None
            warnings.append(f"LLM 解析结果未通过结构校验: {_validation_error_summary(exc)}")
        if llm_spec is not None:
            # 后处理：强制清洗规则
            llm_spec, post_warnings = _post_process_cleaned_spec(llm_spec, message)
            warnings.extend(post_warnings)
            enriched = _apply_rule_enrichment(message, llm_spec)
            if enriched is not None:
                debug_info["final_query"] = enriched.query
                debug_info["final_title_query"] = enriched.title_query
                debug_info["final_abstract_query"] = enriched.abstract_query
                return enriched, _dedupe_preserve_order(warnings), debug_info
        warnings.append("LLM 输出未通过校验，已回退到规则解析")

    try:
        spec = _build_spec_from_rules(message)
    except ValidationError as exc:
        warnings.append(f"规则解析结果未通过结构校验: {_validation_error_summary(exc)}")
        return None, _dedupe_preserve_order(warnings), debug_info

    debug_info["final_query"] = spec.query if spec else None
    debug_info["final_title_query"] = spec.title_query if spec else None
    debug_info["final_abstract_query"] = spec.abstract_query if spec else None
    return spec, _dedupe_preserve_order(warnings), debug_info


def _legacy_parse_with_llm(message: str, generation_service: Optional[Any]) -> Optional[Dict[str, Any]]:
    if generation_service is None or not hasattr(generation_service, "complete_with_qwen"):
        return None

    prompt = (
        "You are a search intent cleaner and parser for a natural-language arXiv search agent.\n"
        "Your job is to:\n"
        "1. Clean the user's natural language input by removing polite words, action words, quantifiers and filler words.\n"
        "2. Convert Chinese research topics to English keywords suitable for arXiv search.\n"
        "3. Extract a structured search spec from the cleaned intent.\n"
        "Return JSON only.\n"
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
        "\"reasoning_summary\":null|string,"
        "\"cleaned_topic_cn\":null|string,"
        "\"cleaned_topic_en\":null|string"
        "}\n"
        "Cleaning rules:\n"
        "- Remove ALL polite words: 请, 请你, 帮我, 给我, 麻烦, 谢谢, etc.\n"
        "- Remove ALL action words: 找, 搜索, 检索, 查找, 推荐, 看看, 寻找, etc.\n"
        "- Remove ALL quantifiers: 几篇, 一些, 几个, N篇, N个, 一篇, 两篇, etc.\n"
        "- Remove ALL filler words: 和...相关的, 关于, 的, 论文, 文献, paper, papers, arxiv.\n"
        "- KEEP only the research topic keywords.\n"
        "- The cleaned_topic_cn field should contain the cleaned Chinese topic keywords.\n"
        "- The cleaned_topic_en field should contain English search keywords.\n"
        "Chinese to English translation rules:\n"
        "- Convert Chinese research topics to English keywords suitable for arXiv search.\n"
        "- Examples:\n"
        "  知识图谱构建 -> knowledge graph construction\n"
        "  知识图谱补全 -> knowledge graph completion\n"
        "  实体链接 -> entity linking\n"
        "  关系抽取 -> relation extraction\n"
        "  多跳问答 -> multi-hop question answering\n"
        "  检索增强生成 -> retrieval augmented generation\n"
        "  论文推荐 -> paper recommendation\n"
        "  大语言模型 -> large language model\n"
        "- If unsure about exact translation, use broader but searchable English terms.\n"
        "Search spec field rules:\n"
        "- For normal natural language search, ONLY set the \"query\" field. Use the English keywords.\n"
        "- Do NOT set both \"query\" and \"abstract_query\" at the same time unless the user explicitly says \"摘要包含...\" or \"abstract contains...\" or \"abstract\".\n"
        "- Do NOT set \"title_query\" unless the user explicitly says \"标题包含...\", \"标题里有...\", \"title contains...\", or \"title\".\n"
        "- The \"query\" field MUST be the English keywords, NOT the original Chinese words.\n"
        "- Use max_results between 1 and 20. Default to 10 if not specified.\n"
        "- Use submitted_days_ago for recent-time expressions.\n"
        "- Default sort_by to \"submittedDate\" with sort_order \"descending\" unless user asks for relevance.\n"
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


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def _user_mentioned_abstract(message: str) -> bool:
    return bool(re.search(r"(?:摘要|abstract)\s*(?:包含|是|有|为|里|中|搜索|contains|contain)", message, re.IGNORECASE))


def _user_mentioned_title(message: str) -> bool:
    return bool(re.search(r"(?:标题|题目|title)\s*(?:包含|是|有|为|里|中|搜索|contains|contain)", message, re.IGNORECASE))


def _post_process_cleaned_spec(
    spec: ArxivSearchSpec,
    message: str,
    *,
    cleaned_topic_cn: Optional[str] = None,
    cleaned_topic_en: Optional[str] = None,
) -> Tuple[ArxivSearchSpec, List[str]]:
    """对 LLM 输出的搜索规格做后处理，确保清洗规则得到强制执行。

    1. 先把 query 再做一次主题抽取，去掉自然语言噪声
    2. 如果同时设置了 query 和 abstract_query 但用户没有明确要求摘要搜索，清空 abstract_query
    3. 如果同时设置了 query 和 title_query 但用户没有明确要求标题搜索，清空 title_query
    """
    warnings: List[str] = []

    # 优先使用 LLM 已经清洗过的 topic，其次再回退到本地规则抽取，尽量避免把整句口语带进 query。
    normalized_query = _normalize_topic_phrase(cleaned_topic_en or "") if cleaned_topic_en else None
    if not normalized_query:
        normalized_query = _normalize_topic_phrase(cleaned_topic_cn or "") if cleaned_topic_cn else None
    if not normalized_query:
        normalized_query = _normalize_topic_phrase(spec.query) if spec.query else None
    if not normalized_query:
        normalized_query = _extract_query_from_message(message)
    if normalized_query:
        # 优先保留更干净的主题短语，避免 LLM 把自然语言整句塞进 query。
        if spec.query and normalized_query != spec.query:
            warnings.append(
                f"LLM 输出的 query 已重新清洗: \"{spec.query}\" -> \"{normalized_query}\""
            )
        spec.query = normalized_query
    elif spec.query:
        warnings.append(
            f"LLM 输出的 query 未能抽取出稳定主题: \"{spec.query}\"，将继续依赖后续规则兜底"
        )

    if spec.query and spec.abstract_query and not _user_mentioned_abstract(message):
        warnings.append(
            f"用户未明确要求摘要搜索，已自动清空 abstract_query"
            f" (原值: \"{spec.abstract_query}\")"
        )
        spec.abstract_query = None

    if spec.query and spec.title_query and not _user_mentioned_title(message):
        warnings.append(
            f"用户未明确要求标题搜索，已自动清空 title_query"
            f" (原值: \"{spec.title_query}\")"
        )
        spec.title_query = None

    return spec, _dedupe_preserve_order(warnings)


def _relax_query_for_fallback(query: Optional[str], retry_count: int) -> Optional[str]:
    """根据重试次数逐步放宽搜索关键词。

    第 1 次重试 (retry_count=1): 取前 N-1 个词（如 "knowledge graph construction" -> "knowledge graph"）
    第 2 次重试 (retry_count=2): 取第一个词（如 "knowledge graph" -> "knowledge"）
    第 3 次重试 (retry_count=3): 返回 None，表示不再使用 query 字段，只靠类别搜索
    """
    if not query or not str(query).strip():
        return None

    query = str(query).strip()

    if retry_count >= 3:
        return None

    # 英文：按空格分词后取前缀
    tokens = query.split()
    if len(tokens) > 1:
        keep = max(1, len(tokens) - retry_count)
        return " ".join(tokens[:keep])

    # 中文：按字符缩短
    if _contains_chinese(query) and len(query) > 2:
        keep = max(2, len(query) - retry_count * 2)
        return query[:keep]

    return query


MAX_SEARCH_RETRIES = 3


def _build_spec_from_rules(message: str) -> Optional[ArxivSearchSpec]:
    query = _extract_query_from_message(message)
    title_query = _extract_marked_query(message, TITLE_HINT_PATTERNS)
    abstract_query = _extract_marked_query(message, ABSTRACT_HINT_PATTERNS)
    categories = get_default_agent_arxiv_categories()
    submitted_days_ago = _extract_submitted_days_ago(message)
    max_results = _extract_max_results(message)
    sort_by, sort_order = _extract_sorting(message)

    if not any([query, title_query, abstract_query]):
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
    # 只有当 LLM 没有提供 query 时，才用规则提取兜底。
    # title_query 和 abstract_query 不由规则自动补全：用户没有明确要求时不应被设置。
    query = spec.query or _extract_query_from_message(message)
    title_query = spec.title_query
    abstract_query = spec.abstract_query
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


def _legacy_classify_intent(message: str) -> str:
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


def _extract_exception_detail(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("detail") or detail.get("error") or "").strip()
        if message:
            return message
        return json.dumps(detail, ensure_ascii=False)
    if detail is not None:
        text = str(detail).strip()
        if text:
            return text
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _extract_exception_stage(exc: Exception, default_stage: str) -> str:
    stage = str(getattr(exc, "error_stage", "") or getattr(exc, "failed_stage", "") or "").strip()
    if stage:
        return stage
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        stage = str(detail.get("stage") or detail.get("failed_stage") or "").strip()
        if stage:
            return stage
    return default_stage


def _build_pending_action_failure_result(
    *,
    arxiv_id: Optional[str],
    title: Optional[str],
    question: Optional[str],
    qa_index_status: Optional[Dict[str, Any]],
    error: str,
    error_stage: str,
    error_type: str,
    index_created: bool = False,
) -> Dict[str, Any]:
    # 失败结果需要同时保留阶段、错误类型和原始错误信息，方便前端和日志一起定位问题。
    return {
        "status": "failed",
        "arxiv_id": arxiv_id,
        "title": title,
        "question": question,
        "answer": "",
        "sources": [],
        "retrieval_debug": None,
        "qa_index_status": qa_index_status,
        "index_created": index_created,
        "error": error,
        "error_stage": error_stage,
        "failed_stage": error_stage,
        "error_type": error_type,
    }


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
    cleaning_debug: Dict[str, Any] = {}

    def apply_rule_result(default_intent: str, override_intent: Optional[str] = None) -> None:
        nonlocal intent, intent_source, search_spec
        nonlocal plan, next_actions
        nonlocal search_spec_before_enrichment, search_spec_after_enrichment

        resolved = rule_result or {}
        intent = str(override_intent or resolved.get("intent") or default_intent)
        intent_source = "fallback"
        search_spec = resolved.get("search_spec")
        search_spec_before_enrichment = resolved.get("search_spec_before_enrichment")
        search_spec_after_enrichment = resolved.get("search_spec_after_enrichment")
        plan = list(resolved.get("plan") or [])
        next_actions = list(resolved.get("next_actions") or [])
        warnings.extend(str(item) for item in (resolved.get("warnings") or []) if str(item).strip())

    llm_payload_result = _parse_llm_intent(
        message,
        generation_service=generation_service,
        research_profile=state.context.get("research_profile") if isinstance(state.context, dict) else None,
    )
    if llm_payload_result.get("ok"):
        try:
            llm_result = _normalize_llm_intent_payload(llm_payload_result["payload"])
            warnings.extend(str(item) for item in (llm_result.get("warnings") or []) if str(item).strip())
            raw_payload = llm_payload_result.get("payload")
            if isinstance(raw_payload, dict):
                cleaning_debug["cleaned_topic_cn"] = _normalize_optional_str(raw_payload.get("cleaned_topic_cn"))
                cleaning_debug["cleaned_topic_en"] = _normalize_optional_str(raw_payload.get("cleaned_topic_en"))
        except ValidationError as exc:
            llm_result = None
            fallback_reason = f"llm output failed schema validation: {_validation_error_summary(exc)}"
            warnings.append(fallback_reason)
    else:
        fallback_reason = str(llm_payload_result.get("reason") or "llm unavailable")
        warnings.append(fallback_reason)

    rule_result = _build_rule_decision(message)

    if llm_result is None:
        apply_rule_result("unsupported")
        if fallback_reason is None:
            fallback_reason = str((rule_result or {}).get("reason") or "llm unavailable, rule fallback used")
    else:
        llm_intent = str(llm_result.get("intent") or "unsupported")
        llm_confidence = llm_result.get("confidence")
        confidence_value = float(llm_confidence) if isinstance(llm_confidence, (int, float)) else None

        if confidence_value is None:
            fallback_reason = "llm confidence missing"
            warnings.append(fallback_reason)
            apply_rule_result(llm_intent)
        elif llm_intent == "arxiv_search":
            search_spec = llm_result.get("search_spec")
            search_spec_before_enrichment = llm_result.get("search_spec_payload")
            if search_spec is None:
                fallback_reason = "llm search intent is missing a valid search spec"
                warnings.append(fallback_reason)
                apply_rule_result("unclear")
            else:
                search_spec, post_warnings = _post_process_cleaned_spec(search_spec, message)
                warnings.extend(post_warnings)
                enriched_spec = _apply_rule_enrichment(message, search_spec)
                if enriched_spec is None:
                    fallback_reason = "rule enrichment failed after llm search parse"
                    warnings.append(fallback_reason)
                    apply_rule_result("unclear")
                elif confidence_value < LLM_CONFIDENCE_THRESHOLD:
                    fallback_reason = (
                        f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
                    )
                    warnings.append(fallback_reason)
                    apply_rule_result("arxiv_search")
                else:
                    intent = "arxiv_search"
                    intent_source = "llm"
                    search_spec = enriched_spec
                    search_spec_after_enrichment = _compact_search_spec(enriched_spec)
                    plan, next_actions, intent_warnings = _build_intent_guidance(intent)
                    warnings.extend(intent_warnings)
                    if rule_result and str(rule_result.get("intent") or "") != "arxiv_search":
                        warnings.append(
                            f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}"
                        )
        else:
            if llm_intent == "recommendation" and not _looks_like_recommendation_request(message):
                fallback_reason = "llm recommendation intent downgraded because the request is topic-based, not personalized"
                warnings.append(fallback_reason)
                if str((rule_result or {}).get("intent") or "") == "arxiv_search":
                    apply_rule_result("arxiv_search", override_intent="arxiv_search")
                else:
                    apply_rule_result("unclear", override_intent="unclear")
            elif confidence_value < LLM_CONFIDENCE_THRESHOLD:
                fallback_reason = (
                    f"llm confidence {confidence_value:.2f} below threshold {LLM_CONFIDENCE_THRESHOLD:.2f}"
                )
                warnings.append(fallback_reason)
                apply_rule_result(llm_intent)
            else:
                intent = llm_intent
                intent_source = "llm"
                plan, next_actions, intent_warnings = _build_intent_guidance(llm_intent)
                warnings.extend(intent_warnings)
                if rule_result and str(rule_result.get("intent") or "") != llm_intent:
                    warnings.append(
                        f"llm/rule intent conflict: llm={llm_intent}, rule={rule_result.get('intent')}"
                    )

    if not plan and not next_actions:
        plan, next_actions, intent_warnings = _build_intent_guidance(intent)
        warnings.extend(intent_warnings)

    if intent == "arxiv_search" and search_spec is None:
        plan, next_actions, intent_warnings = _build_intent_guidance("unclear")
        warnings.extend(intent_warnings)
        intent = "unclear"
        if fallback_reason is None:
            fallback_reason = "search intent was downgraded because no valid search spec was produced"

    if search_spec is not None:
        cleaning_debug["final_query"] = search_spec.query
        cleaning_debug["final_title_query"] = search_spec.title_query
        cleaning_debug["final_abstract_query"] = search_spec.abstract_query

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


def _merge_context_paper_lists(*paper_lists: Any) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen_keys = set()

    for paper_list in paper_lists:
        if not isinstance(paper_list, list):
            continue
        for raw_paper in paper_list:
            if not isinstance(raw_paper, Mapping):
                continue
            paper = _normalize_context_paper(raw_paper)
            identity = str(paper.get("arxiv_id") or "").strip() or str(paper.get("title") or "").strip().lower()
            if not identity or identity in seen_keys:
                continue
            seen_keys.add(identity)
            merged.append(paper)

    return merged


def _extract_selected_paper(context: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(context, Mapping):
        return None

    for key in (
        "selected_paper",
        "current_paper",
        "active_paper",
        "target_paper",
        "last_target_paper",
        "current_selected_paper",
    ):
        raw_paper = context.get(key)
        if isinstance(raw_paper, Mapping):
            paper = _normalize_context_paper(raw_paper)
            if paper.get("arxiv_id") or paper.get("title"):
                return paper

    paper_qa_result = context.get("paper_qa_result")
    if isinstance(paper_qa_result, Mapping):
        paper = _normalize_context_paper(
            {
                "arxiv_id": paper_qa_result.get("arxiv_id"),
                "title": paper_qa_result.get("title"),
            }
        )
        if paper.get("arxiv_id") or paper.get("title"):
            return paper

    arxiv_id = str(context.get("arxiv_id") or "").strip()
    title = str(context.get("paper_title") or context.get("title") or "").strip()
    if arxiv_id or title:
        paper = _normalize_context_paper({"arxiv_id": arxiv_id, "title": title})
        if paper.get("arxiv_id") or paper.get("title"):
            return paper

    return None


def _extract_last_papers(context: Any) -> List[Dict[str, Any]]:
    if not isinstance(context, Mapping):
        return []
    return _merge_context_paper_lists(
        context.get("last_papers") or [],
        context.get("papers") or [],
        context.get("search_results") or [],
        context.get("recent_papers") or [],
    )


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

    arxiv_match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)", text, flags=re.IGNORECASE)
    if arxiv_match:
        return {
            "target_type": "arxiv_id",
            "target_value": arxiv_match.group(1),
            "arxiv_id": arxiv_match.group(1),
        }

    if _matches_any(
        text,
        (
            r"这篇",
            r"该论文",
            r"这篇论文",
            r"本文",
            r"当前选中",
            r"当前论文",
            r"当前这篇",
        ),
    ):
        return {
            "target_type": "context_paper",
            "target_value": "selected_or_recent",
        }

    ordinal_match = re.search(r"(?:第\s*)?([一二三四五六七八九十两]{1,3}|[1-9]|1[0-9]|20)\s*(?:篇|个)?(?:论文|paper)?", text)
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
    selected_paper = _extract_selected_paper(context)

    def build_success(paper_payload: Optional[Mapping[str, Any]], *, target: Optional[Dict[str, Any]], matched_from: str) -> Dict[str, Any]:
        normalized_paper = _normalize_context_paper(paper_payload or {})
        resolved_arxiv_id = str(normalized_paper.get("arxiv_id") or "").strip() or None
        resolved_title = str(normalized_paper.get("title") or "").strip() or None
        return {
            "status": "success",
            "reason": None,
            "target": target,
            "paper": normalized_paper,
            "arxiv_id": resolved_arxiv_id,
            "title": resolved_title,
            "matched_from": matched_from,
        }

    if reference is None:
        if selected_paper is not None:
            return build_success(selected_paper, target=None, matched_from="selected_paper")
        if len(last_papers) == 1:
            return build_success(last_papers[0], target=None, matched_from="single_recent_paper")
        return {
            "status": "failed",
            "reason": "没有解析到目标论文。请先搜索论文，或直接提供 arXiv ID，或使用“第一篇 / 第二篇”指定搜索结果中的论文。",
            "target": None,
            "paper": None,
            "arxiv_id": None,
            "title": None,
        }

    if reference["target_type"] == "context_paper":
        if selected_paper is not None:
            return build_success(selected_paper, target=reference, matched_from="selected_paper")
        if len(last_papers) == 1:
            return build_success(last_papers[0], target=reference, matched_from="single_recent_paper")
        return {
            "status": "failed",
            "reason": "当前没有可直接指代的目标论文。请先搜索论文、传入 selected_paper，或用“第一篇 / 第二篇”明确指定。",
            "target": reference,
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
        return build_success(paper, target=reference, matched_from="last_papers")

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
    if paper is None and selected_paper is not None and str(selected_paper.get("arxiv_id") or "").strip() == arxiv_id:
        paper = selected_paper
    if paper is not None:
        return build_success(paper, target=reference, matched_from="explicit_arxiv_id")

    return {
        "status": "success",
        "reason": None,
        "target": reference,
        "paper": None,
        "arxiv_id": arxiv_id,
        "title": selected_paper.get("title") if isinstance(selected_paper, Mapping) and str(selected_paper.get("arxiv_id") or "").strip() == arxiv_id else None,
        "matched_from": "explicit_arxiv_id",
    }


def _resolve_generation_service_instance(generation_service: Optional[Any] = None) -> Optional[Any]:
    if generation_service is not None:
        return generation_service
    try:
        return get_generation_service()
    except Exception:
        return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = _normalize_text(text)
    if not candidate:
        return None

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s*```$", "", candidate).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_confirmation_decision(value: Any) -> str:
    decision = _normalize_text(str(value or "")).lower()
    if decision in {"confirm", "confirmed", "accept", "yes", "ok", "continue"}:
        return "confirm"
    if decision in {"reject", "rejected", "cancel", "cancelled", "no", "rejecting"}:
        return "reject"
    if decision in {"unrelated", "other", "new_request"}:
        return "unrelated"
    return "unclear"


def _fast_path_pending_action_decision(message: str) -> Optional[str]:
    text = _normalize_text(message)
    if not text:
        return None

    confirm_tokens = {
        "确认",
        "是",
        "好",
        "好的",
        "可以",
        "继续",
        "解析",
        "开始解析",
        "需要解析",
        "帮我解析",
        "ok",
        "yes",
        "continue",
        "go ahead",
        "继续解析",
    }
    reject_tokens = {
        "不用",
        "取消",
        "不解析",
        "先不",
        "暂时不用",
        "算了",
        "不要",
        "先不看",
        "no",
        "cancel",
        "别解析",
    }

    if text in confirm_tokens:
        return "confirm"
    if text in reject_tokens:
        return "reject"

    lowered = text.lower()
    if lowered in confirm_tokens:
        return "confirm"
    if lowered in reject_tokens:
        return "reject"

    reject_patterns = (
        r"先不解析",
        r"不要解析",
        r"不需要解析",
        r"别解析",
        r"别建索引",
        r"不要建索引",
        r"不用建索引",
        r"取消.*(?:任务|解析|索引)",
    )
    confirm_patterns = (
        r"解析\s*pdf",
        r"开始解析",
        r"继续解析",
        r"继续处理",
        r"开始处理",
        r"解析并回答",
        r"创建.*全文检索索引",
        r"创建.*索引",
        r"建立.*索引",
        r"构建.*索引",
        r"可以解析",
        r"确认解析",
    )

    if _matches_any(text, reject_patterns) or _matches_any(lowered, reject_patterns):
        return "reject"
    if _matches_any(text, confirm_patterns) or _matches_any(lowered, confirm_patterns):
        return "confirm"
    return None


def _build_qa_question_for_paper(intent: str, message: str, paper: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    original_question = _normalize_text(message)
    title = _normalize_text(str(paper.get("title") or reference.get("title") or ""))

    cleaned_question = original_question
    cleaned_question = re.sub(r"^(问一下|请问一下|请问|问|帮我问一下|帮我问|想问一下)\s*", "", cleaned_question).strip()
    cleaned_question = re.sub(
        r"^(?:第\s*[一二三四五六七八九十两0-9]+\s*篇(?:论文|paper)?|[1-9]|1[0-9]|20)\s*[:：,，]?\s*",
        "",
        cleaned_question,
    ).strip()
    cleaned_question = re.sub(r"^(?:这篇论文|这篇|该论文|当前选中论文|当前论文|本文)\s*", "", cleaned_question).strip()
    cleaned_question = re.sub(r"^arxiv\s*id\s*[:：]?\s*\d{4}\.\d{4,5}(?:v\d+)?\s*", "", cleaned_question, flags=re.IGNORECASE).strip()
    cleaned_question = re.sub(r"^\d{4}\.\d{4,5}(?:v\d+)?\s*", "", cleaned_question).strip()
    cleaned_question = cleaned_question.lstrip("，,:：.。;； ")

    if intent == "paper_summary":
        return (
            f"请基于论文全文总结这篇论文，包含研究问题、核心贡献、方法流程、实验设置、主要结果和局限性。"
            f"{f' 论文标题：{title}' if title else ''}"
        ).strip()

    if intent == "paper_detail":
        detail_question = cleaned_question or original_question
        if _matches_any(detail_question, (r"^解释$", r"^讲讲$", r"^介绍一下$", r"^方法$", r"^讲讲方法$", r"^解释方法$")):
            detail_question = "请详细解释这篇论文的方法设计、关键模块、输入输出流程，以及这样设计的原因。"
        if not detail_question:
            detail_question = "请详细解释这篇论文的主要方法、实验设计和贡献。"
        return detail_question

    if not cleaned_question:
        cleaned_question = "请基于论文全文回答这个问题。"
    return cleaned_question


def _make_pending_action_payload(
    *,
    arxiv_id: str,
    title: str,
    original_question: str,
    qa_question: str,
    loading_method: str = "docling",
) -> Dict[str, Any]:
    return {
        "type": "parse_then_qa",
        "arxiv_id": arxiv_id,
        "title": title,
        "original_question": original_question,
        "qa_question": qa_question,
        "loading_method": loading_method,
        "status": "waiting_confirmation",
        # 记录待确认任务的创建时间，便于后续做状态追踪、排查超时卡住的解析流程。
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def classify_pending_action_confirmation(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    pending_action = dict(next_state.pending_action or (next_state.context or {}).get("pending_action") or {})
    # 如果前端只回传了 paper_qa_result，也要能把等待确认的解析任务恢复出来，避免确认链路断开。
    if not pending_action and isinstance(next_state.paper_qa_result, Mapping):
        paper_qa_result = dict(next_state.paper_qa_result or {})
        if str(paper_qa_result.get("status") or "").strip() == "waiting_confirmation":
            pending_action = {
                "type": "parse_then_qa",
                "status": "waiting_confirmation",
                "arxiv_id": paper_qa_result.get("arxiv_id"),
                "title": paper_qa_result.get("title"),
                # 优先恢复原始问题，其次才回退到 QA 改写问题或 question 字段，避免前端只回传结果时丢失用户原话。
                "original_question": paper_qa_result.get("original_question") or paper_qa_result.get("question"),
                "qa_question": paper_qa_result.get("qa_question") or paper_qa_result.get("question"),
                "loading_method": (next_state.context or {}).get("loading_method") if isinstance(next_state.context, dict) else None,
            }
            logger.info(
                "arxiv_agent recovered pending_action from paper_qa_result: arxiv_id=%s title=%s",
                pending_action.get("arxiv_id") or "none",
                pending_action.get("title") or "none",
            )
    message = _normalize_text(next_state.message or "")
    debug = dict(next_state.debug or {})

    debug["pending_action"] = pending_action
    debug["pending_action_confirmation_model"] = "qwen3.6-flash"

    if not pending_action or str(pending_action.get("type") or "").strip() != "parse_then_qa":
        debug["pending_action_decision"] = "unclear"
        debug["confirmation_confidence"] = 0.0
        debug["confirmation_reason"] = "当前没有待确认的论文解析任务"
        next_state.debug = debug
        return _append_step(
            next_state,
            step="classify_pending_action_confirmation",
            status="skipped",
            action="判断用户是否确认解析任务",
            inputs={"message": message, "pending_action": pending_action},
            outputs={"decision": "unclear", "reason": debug["confirmation_reason"]},
        )

    decision = _fast_path_pending_action_decision(message)
    confidence = 0.99 if decision in {"confirm", "reject"} else 0.0
    reason = ""
    # 再加一层明确的中文关键词兜底：用户只回复“解析”“索引”时，也应视作继续执行，而不是继续卡在确认问题上。
    if decision is None and any(keyword in message for keyword in ("解析", "索引", "全文检索", "问答索引")):
        decision = "confirm"
        confidence = 0.95
        reason = "关键词兜底命中明确的继续执行意图"
    reason = reason or ("fast path 命中明确确认/取消表达" if decision in {"confirm", "reject"} else "")

    if decision is None:
        service = _resolve_generation_service_instance(generation_service)
        if service is not None:
            prompt = (
                "你正在判断用户是否要继续执行一个待确认的论文解析任务。\n"
                "请只输出严格 JSON，不要输出解释性文本。\n\n"
                "JSON schema:\n"
                '{\n'
                '  "decision": "confirm | reject | unrelated | unclear",\n'
                '  "confidence": 0.0,\n'
                '  "reason": "简短原因"\n'
                '}\n\n'
                "待确认任务信息：\n"
                f"- 论文标题: {pending_action.get('title') or ''}\n"
                f"- arXiv ID: {pending_action.get('arxiv_id') or ''}\n"
                f"- 原始问题: {pending_action.get('original_question') or ''}\n"
                f"- qa_question: {pending_action.get('qa_question') or ''}\n"
                f"- 当前用户输入: {message}\n\n"
                "判定规则：\n"
                "- confirm: 用户明确表示继续解析或同意执行\n"
                "- reject: 用户明确取消或拒绝执行\n"
                "- unrelated: 用户提出了新的独立请求，而不是确认/取消\n"
                "- unclear: 无法判断用户是否确认\n"
            )
            try:
                raw_output = service.complete_with_qwen(
                    prompt,
                    task_type="pending_action_confirmation",
                    enable_thinking=False,
                )
                parsed = _extract_json_object(raw_output) or {}
                decision = _normalize_confirmation_decision(parsed.get("decision"))
                confidence = float(parsed.get("confidence") or 0.0)
                reason = _normalize_text(str(parsed.get("reason") or ""))
                if not reason:
                    reason = "LLM 分类结果未返回原因"
            except Exception as exc:
                decision = "unclear"
                confidence = 0.0
                reason = f"确认判断失败: {exc}"
        else:
            decision = "unclear"
            confidence = 0.0
            reason = "缺少生成服务，无法进行确认判断"

    debug["pending_action_decision"] = decision
    debug["confirmation_decision"] = decision
    debug["pending_action_confidence"] = confidence
    debug["confirmation_confidence"] = confidence
    debug["pending_action_reason"] = reason
    debug["confirmation_reason"] = reason
    logger.info(
        "arxiv_agent confirmation classified: decision=%s confidence=%.2f reason=%s message=%s arxiv_id=%s title=%s",
        decision or "unclear",
        confidence,
        reason or "none",
        message,
        pending_action.get("arxiv_id") or "none",
        pending_action.get("title") or "none",
    )
    next_state.debug = debug
    if not next_state.plan:
        next_state.plan = [
            "判断用户是否确认解析任务",
            "必要时创建论文 QA 索引",
            "继续回答原始问题",
        ]

    return _append_step(
        next_state,
        step="classify_pending_action_confirmation",
        status="success",
        action="判断用户是否确认解析任务",
        inputs={"message": message, "pending_action": pending_action},
        outputs={
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "pending_action": pending_action,
        },
    )


def handle_pending_action_confirmation(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    pending_action = dict(next_state.pending_action or (next_state.context or {}).get("pending_action") or {})
    # 兼容只回传 paper_qa_result 的流式/非流式请求，保证“确认解析”能够真正触发创建索引。
    if not pending_action and isinstance(next_state.paper_qa_result, Mapping):
        paper_qa_result = dict(next_state.paper_qa_result or {})
        if str(paper_qa_result.get("status") or "").strip() == "waiting_confirmation":
            pending_action = {
                "type": "parse_then_qa",
                "status": "waiting_confirmation",
                "arxiv_id": paper_qa_result.get("arxiv_id"),
                "title": paper_qa_result.get("title"),
                # 同步保留原始提问和改写后的 QA 问题，保证确认回调后还能继续执行同一条用户意图。
                "original_question": paper_qa_result.get("original_question") or paper_qa_result.get("question"),
                "qa_question": paper_qa_result.get("qa_question") or paper_qa_result.get("question"),
                "loading_method": (next_state.context or {}).get("loading_method") if isinstance(next_state.context, dict) else None,
            }
            logger.info(
                "arxiv_agent handle confirmation recovered pending_action from paper_qa_result: arxiv_id=%s title=%s",
                pending_action.get("arxiv_id") or "none",
                pending_action.get("title") or "none",
            )
    qa_service = get_paper_qa_service()
    debug = dict(next_state.debug or {})

    arxiv_id = str(pending_action.get("arxiv_id") or "").strip()
    title = str(pending_action.get("title") or "").strip()
    qa_question = _normalize_text(str(pending_action.get("qa_question") or ""))
    loading_method = str(pending_action.get("loading_method") or "docling").strip() or "docling"

    logger.info(
        "arxiv_agent handle confirmation start: arxiv_id=%s title=%s loading_method=%s pending_status=%s paper_qa_status=%s",
        arxiv_id or "none",
        title or "none",
        loading_method,
        str(pending_action.get("status") or "none"),
        _normalize_text(str((next_state.paper_qa_result or {}).get("status") if isinstance(next_state.paper_qa_result, Mapping) else "none")),
    )

    if not arxiv_id:
        qa_index_status = None
        result = {
            "status": "failed",
            "arxiv_id": None,
            "title": title or None,
            "original_question": message,
            "qa_question": None,
            "question": qa_question or None,
            "answer": "",
            "sources": [],
            "retrieval_debug": None,
            "qa_index_status": None,
            "index_created": False,
            "error": "待确认任务缺少 arXiv ID",
            "error_stage": "pending_action_validation",
            "failed_stage": "pending_action_validation",
            "error_type": "ValueError",
        }
        next_state.paper_qa_result = result
        next_state.answer = result["error"]
        debug["qa_index_status"] = None
        debug["index_created"] = False
        debug["error_stage"] = result["error_stage"]
        debug["error_type"] = result["error_type"]
        debug["error_detail"] = result["error"]
        next_state.debug = debug
        return _append_step(
            next_state,
            step="handle_pending_action_confirmation",
            status="failed",
            action="创建并执行论文问答任务",
            inputs={"pending_action": pending_action},
            outputs={"paper_qa_result": result, "qa_index_status": qa_index_status, "index_created": False},
            error=result["error"],
        )

    try:
        index_result = qa_service.build_qa_index(arxiv_id, loading_method=loading_method)
        qa_status = qa_service.get_qa_status(arxiv_id)
        index_created = str(index_result.get("status") or "").lower() in {"success", "indexed"}
        answer_result = qa_service.answer_question(
            arxiv_id,
            {
                "question": qa_question or pending_action.get("original_question") or "请基于论文全文回答问题。",
            },
        )
        result = {
            "status": "success",
            "arxiv_id": arxiv_id,
            "title": title or answer_result.get("title") or title,
            "original_question": message,
            "qa_question": qa_question,
            "question": qa_question or pending_action.get("original_question") or "",
            "answer": answer_result.get("answer", ""),
            "sources": answer_result.get("sources", []),
            "retrieval_debug": answer_result.get("retrieval_debug"),
            "qa_index_status": qa_status,
            "index_created": index_created,
            "error": None,
        }
        next_state.paper_qa_result = result
        next_state.answer = result["answer"]
        next_state.papers = list(answer_result.get("papers", [])) if isinstance(answer_result, Mapping) and answer_result.get("papers") else list(next_state.papers or [])
        next_state.pending_action = None
        next_state.context = dict(next_state.context or {})
        next_state.context.pop("pending_action", None)
        debug["qa_index_status"] = qa_status
        debug["index_created"] = index_created
        debug["qa_question"] = result["question"]
        debug["paper_qa_arxiv_id"] = arxiv_id
        next_state.debug = debug
        next_state.tool_name = "paper_qa"
        next_state.tool_args = {"arxiv_id": arxiv_id, "question": result["question"], "loading_method": loading_method}
        next_state.tool_result = dict(result)
        next_state.tool_calls = list(next_state.tool_calls or []) + [
            AgentToolCall(
                tool_name="paper_qa",
                arguments=next_state.tool_args,
                status="success",
                summary=f"已创建 QA 索引并回答论文问题: {title or arxiv_id}",
                trace={
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "qa_index_status": qa_status,
                    "index_created": index_created,
                    "sources_count": len(result.get("sources") or []),
                    "retrieval_debug_present": bool(result.get("retrieval_debug")),
                },
                error=None,
            )
        ]
        if not next_state.next_actions:
            next_state.next_actions = [
                "继续追问这篇论文的细节",
                "切换到其他论文继续解析",
            ]
        return _append_step(
            next_state,
            step="handle_pending_action_confirmation",
            status="success",
            action="创建并执行论文问答任务",
            inputs={"pending_action": pending_action},
            outputs={
                "paper_qa_result": result,
                "qa_index_status": qa_status,
                "index_created": index_created,
            },
        )
    except Exception as exc:
        error_stage = _extract_exception_stage(exc, "build_qa_index")
        error_message = _extract_exception_detail(exc)
        qa_index_status = qa_service.get_qa_status(arxiv_id) if hasattr(qa_service, "get_qa_status") else None
        result = {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "title": title,
            "original_question": pending_action.get("original_question") or qa_question,
            "qa_question": qa_question,
            "question": qa_question or pending_action.get("original_question") or "",
            "answer": "",
            "sources": [],
            "retrieval_debug": None,
            "qa_index_status": qa_index_status,
            "index_created": False,
            "error": error_message,
            "error_stage": error_stage,
            "failed_stage": error_stage,
            "error_type": type(exc).__name__,
        }
        next_state.paper_qa_result = result
        next_state.answer = f"论文解析或问答执行失败：{error_message}"
        debug["qa_index_status"] = result["qa_index_status"]
        debug["index_created"] = False
        debug["error_stage"] = error_stage
        debug["error_type"] = type(exc).__name__
        debug["error_detail"] = error_message
        next_state.debug = debug
        next_state.tool_name = "paper_qa"
        next_state.tool_args = {"arxiv_id": arxiv_id, "question": result["question"], "loading_method": loading_method}
        next_state.tool_result = dict(result)
        next_state.tool_calls = list(next_state.tool_calls or []) + [
            AgentToolCall(
                tool_name="paper_qa",
                arguments=next_state.tool_args,
                status="failed",
                summary=f"论文问答执行失败: {title or arxiv_id}",
                trace={
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "qa_index_status": result["qa_index_status"],
                    "index_created": False,
                    "error_stage": error_stage,
                    "error_type": type(exc).__name__,
                },
                error={"message": result["error"], "stage": error_stage, "type": type(exc).__name__},
            )
        ]
        return _append_step(
            next_state,
            step="handle_pending_action_confirmation",
            status="failed",
            action="创建并执行论文问答任务",
            inputs={"pending_action": pending_action},
            outputs={"paper_qa_result": result},
            error=result["error"],
        )


def handle_paper_reading_request(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    intent = str(next_state.intent or "").strip()
    if intent not in {"paper_summary", "paper_detail", "paper_qa"}:
        return _append_step(
            next_state,
            step="handle_paper_reading_request",
            status="skipped",
            action="处理论文阅读请求",
            inputs={"intent": intent},
            outputs={"reason": "当前 intent 不是论文阅读类请求"},
        )

    message = _normalize_text(next_state.message or "")
    context = dict(next_state.context or {})
    resolution = _resolve_paper_reference(message, context)
    if resolution.get("status") != "success":
        result = {
            "status": "failed",
            "arxiv_id": None,
            "title": None,
            "question": message,
            "answer": str(resolution.get("reason") or "无法解析目标论文"),
            "sources": [],
            "retrieval_debug": None,
            "qa_index_status": None,
            "index_created": False,
            "error": str(resolution.get("reason") or "paper reference resolution failed"),
        }
        next_state.paper_qa_result = result
        next_state.answer = result["answer"]
        next_state.next_actions = [
            "请先搜索论文，或直接提供 arXiv ID",
            "也可以用“第一篇 / 第二篇”再次指定目标论文",
        ]
        next_state.debug = dict(next_state.debug or {})
        next_state.debug["qa_question"] = None
        next_state.debug["paper_resolution"] = resolution
        return _append_step(
            next_state,
            step="handle_paper_reading_request",
            status="failed",
            action="处理论文阅读请求",
            inputs={"message": message, "intent": intent, "context": context},
            outputs={"paper_qa_result": result, "resolution": resolution},
            error=result["error"],
        )

    paper = resolution.get("paper") or {}
    arxiv_id = str(resolution.get("arxiv_id") or "").strip()
    title = str(resolution.get("title") or paper.get("title") or "").strip()
    qa_question = _build_qa_question_for_paper(intent, message, paper, resolution)
    qa_service = get_paper_qa_service()
    qa_status = qa_service.get_qa_status(arxiv_id) if arxiv_id else None
    loading_method = str(context.get("loading_method") or "docling").strip() or "docling"
    debug = dict(next_state.debug or {})
    debug["qa_question"] = qa_question
    debug["qa_index_status"] = qa_status
    debug["paper_resolution"] = resolution
    debug["paper_reading_intent"] = intent
    next_state.debug = debug
    next_state.context = dict(next_state.context or {})
    next_state.context["selected_paper"] = _normalize_context_paper(
        {
            **({} if not isinstance(paper, Mapping) else dict(paper)),
            "arxiv_id": arxiv_id,
            "title": title,
        }
    )
    next_state.context["arxiv_id"] = arxiv_id
    next_state.tool_name = "paper_qa_status"
    next_state.tool_args = {
        "arxiv_id": arxiv_id,
        "intent": intent,
        "question": qa_question,
        "loading_method": loading_method,
    }

    qa_status = qa_status or {}

    if bool(qa_status.get("has_index")) or str(qa_status.get("status") or "").strip().lower() == "indexed":
        try:
            answer_result = qa_service.answer_question(arxiv_id, {"question": qa_question})
            result = {
                "status": "success",
                "arxiv_id": arxiv_id,
                "title": title,
                "question": qa_question,
                "answer": answer_result.get("answer", ""),
                "sources": answer_result.get("sources", []),
                "retrieval_debug": answer_result.get("retrieval_debug"),
                "qa_index_status": qa_status,
                "index_created": False,
                "error": None,
            }
            next_state.paper_qa_result = result
            next_state.answer = result["answer"]
            next_state.papers = list(answer_result.get("papers", [])) if isinstance(answer_result, Mapping) and answer_result.get("papers") else list(next_state.papers or [])
            next_state.next_actions = [
                "继续追问这篇论文的其他细节",
                "切换到其他论文继续阅读",
            ]
            next_state.pending_action = None
            next_state.context.pop("pending_action", None)
            next_state.tool_result = dict(result)
            next_state.tool_calls = list(next_state.tool_calls or []) + [
                AgentToolCall(
                    tool_name="paper_qa",
                    arguments=next_state.tool_args,
                    status="success",
                    summary=f"直接回答已建索引论文: {title or arxiv_id}",
                    trace={
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "qa_index_status": qa_status,
                        "sources_count": len(result.get("sources") or []),
                        "retrieval_debug_present": bool(result.get("retrieval_debug")),
                    },
                    error=None,
                )
            ]
            return _append_step(
                next_state,
                step="handle_paper_reading_request",
                status="success",
                action="处理论文阅读请求",
                inputs={"message": message, "intent": intent, "context": context},
                outputs={
                    "paper_qa_result": result,
                    "qa_index_status": qa_status,
                    "qa_question": qa_question,
                },
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "arxiv_id": arxiv_id,
                "title": title,
                "question": qa_question,
                "answer": "",
                "sources": [],
                "retrieval_debug": None,
                "qa_index_status": qa_status,
                "index_created": False,
                "error": str(exc),
            }
            next_state.paper_qa_result = result
            next_state.answer = f"论文问答执行失败: {exc}"
            next_state.next_actions = [
                "稍后重试该论文问答",
                "或者先确认是否要重新解析 PDF",
            ]
            return _append_step(
                next_state,
                step="handle_paper_reading_request",
                status="failed",
                action="处理论文阅读请求",
                inputs={"message": message, "intent": intent, "context": context},
                outputs={"paper_qa_result": result, "qa_index_status": qa_status},
                error=str(exc),
            )

    pending_action = _make_pending_action_payload(
        arxiv_id=arxiv_id,
        title=title,
        original_question=message,
        qa_question=qa_question,
        loading_method=loading_method,
    )
    next_state.pending_action = pending_action
    next_state.context = dict(next_state.context or {})
    next_state.context["pending_action"] = pending_action
    result = {
        "status": "waiting_confirmation",
        "arxiv_id": arxiv_id,
        "title": title,
        # 待确认态必须持久化原始问题与 QA 问题；后续“解析”按钮回调依赖这两个字段恢复上下文。
        "original_question": message,
        "qa_question": qa_question,
        "question": qa_question,
        "answer": "",
        "sources": [],
        "retrieval_debug": None,
        "qa_index_status": qa_status,
        "index_created": False,
        "error": None,
    }
    next_state.paper_qa_result = result
    next_state.answer = (
        f"这篇论文还没有建立问答索引。"
        f"{f'《{title}》' if title else ''}"
        f"{f' arXiv ID: {arxiv_id}' if arxiv_id else ''}"
        " 是否现在解析 PDF 并创建全文检索索引？"
    )
    next_state.next_actions = ["解析", "取消"]
    next_state.tool_calls = list(next_state.tool_calls or []) + [
        AgentToolCall(
            tool_name="paper_qa_status",
            arguments={"arxiv_id": arxiv_id, "intent": intent},
            status="success",
            summary=f"论文尚未建立 QA 索引，等待用户确认: {title or arxiv_id}",
            trace={
                "arxiv_id": arxiv_id,
                "title": title,
                "qa_index_status": qa_status,
                "qa_question": qa_question,
            },
            error=None,
        )
    ]
    return _append_step(
        next_state,
        step="handle_paper_reading_request",
        status="success",
        action="处理论文阅读请求",
        inputs={"message": message, "intent": intent, "context": context},
        outputs={
            "paper_qa_result": result,
            "pending_action": pending_action,
            "qa_index_status": qa_status,
            "qa_question": qa_question,
        },
    )


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

    pending_action = dict(next_state.pending_action or (next_state.context or {}).get("pending_action") or {})
    paper_qa_result = dict(next_state.paper_qa_result or {})
    confirmation_decision = str((next_state.debug or {}).get("pending_action_decision") or "").strip().lower()

    # 论文阅读流优先级最高：要么已经拿到全文问答结果，要么仍然停留在“是否解析 PDF”的确认阶段。
    if next_state.intent in {"paper_summary", "paper_detail", "paper_qa"} and paper_qa_result.get("status") == "success":
        next_state.answer = str(paper_qa_result.get("answer") or next_state.answer or "").strip()
        if not next_state.next_actions:
            next_state.next_actions = [
                "继续追问这篇论文的其他细节",
                "切换到其他论文继续阅读",
            ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成论文阅读回复",
            inputs={
                "intent": next_state.intent,
                "paper_qa_result": dict(paper_qa_result),
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
                "qa_index_status": paper_qa_result.get("qa_index_status"),
            },
        )

    if next_state.intent in {"paper_summary", "paper_detail", "paper_qa"} and paper_qa_result.get("status") == "failed" and confirmation_decision == "confirm":
        error_message = str(paper_qa_result.get("error") or paper_qa_result.get("answer") or "论文解析或问答执行失败").strip()
        next_state.answer = error_message
        next_state.next_actions = [
            "重新确认是否需要解析 PDF",
            "或稍后重试这篇论文",
        ]
        return _append_step(
            next_state,
            step="final_answer_generation",
            status="failed",
            action="生成论文阅读回复",
            inputs={
                "intent": next_state.intent,
                "paper_qa_result": dict(paper_qa_result),
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
            },
            error=error_message,
        )

    if isinstance(pending_action, dict) and str(pending_action.get("type") or "").strip() == "parse_then_qa":
        title = str(pending_action.get("title") or "").strip()
        arxiv_id = str(pending_action.get("arxiv_id") or "").strip()
        qa_question = str(pending_action.get("qa_question") or "").strip()
        base_prompt = (
            f"《{title}》" if title else "这篇论文"
        )
        if confirmation_decision == "reject":
            next_state.answer = f"已取消解析{base_prompt}。暂时无法基于全文回答。"
            next_state.paper_qa_result = {
                **paper_qa_result,
                "status": "failed",
                "arxiv_id": arxiv_id or paper_qa_result.get("arxiv_id"),
                "title": title or paper_qa_result.get("title"),
                "question": qa_question or paper_qa_result.get("question"),
                "answer": "",
                "error": "user cancelled pending action",
            }
            next_state.pending_action = None
            next_state.context = dict(next_state.context or {})
            next_state.context.pop("pending_action", None)
            next_state.next_actions = [
                "继续搜索其他论文",
                "重新发起论文阅读请求",
            ]
        else:
            pending_message = str(paper_qa_result.get("answer") or "").strip()
            if not pending_message:
                pending_message = (
                    f"{base_prompt} 还没有建立问答索引。是否现在解析 PDF 并创建全文检索索引？"
                    if confirmation_decision != "unrelated"
                    else f"当前还有一个待确认的论文解析任务：{base_prompt}。请先回复“解析”或“取消”。"
                )
            if confirmation_decision == "unrelated":
                next_state.answer = f"当前还有一个待确认的论文解析任务：{base_prompt}。请先回复“解析”或“取消”。"
                next_state.next_actions = ["解析", "取消"]
            elif confirmation_decision == "unclear" or not confirmation_decision:
                next_state.answer = f"{base_prompt} 还没有建立问答索引。请回复“解析”继续，或回复“取消”放弃。"
                next_state.next_actions = ["解析", "取消"]
            else:
                next_state.answer = pending_message
                next_state.next_actions = ["解析", "取消"]

        return _append_step(
            next_state,
            step="final_answer_generation",
            status="success",
            action="生成论文阅读回复",
            inputs={
                "intent": next_state.intent,
                "pending_action": pending_action,
                "paper_qa_result": dict(paper_qa_result),
                "confirmation_decision": confirmation_decision,
            },
            outputs={
                "answer": next_state.answer,
                "next_actions": list(next_state.next_actions),
            },
        )

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
    "classify_pending_action_confirmation",
    "handle_paper_reading_request",
    "handle_pending_action_confirmation",
    "invoke_search_tool",
    "parse_search_request",
    "personalized_rank_and_annotate_papers",
    "route_after_parse",
    "synthesize_response",
]
