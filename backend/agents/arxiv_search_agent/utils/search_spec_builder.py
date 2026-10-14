"""搜索规格构建、清洗和规则补全过程工具。

这个模块负责把自然语言里的主题、时间范围、数量和排序偏好，
逐步整理成可执行的 ArxivSearchSpec。

它是 parse_node 背后的“搜索参数工程层”：
1. 负责把零散文本线索提取成结构化字段；
2. 负责对 LLM 输出的 spec 做规则化清洗和兜底；
3. 负责生成适合调试和解释的 reasoning_summary。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..schemas import (
    ArxivSearchSpec,
    SearchConcept,
    get_agent_max_search_results,
    get_default_agent_arxiv_categories,
    get_valid_arxiv_categories,
)
from .search_ladder import text_tokens
from .text_utils import _normalize_optional_str, _normalize_text, _parse_small_chinese_number
from ..intent_definitions import PARSE_SUPPORTED_INTENTS

# intent 名单不再本地维护：单一事实来源在 ..intent_definitions。
SUPPORTED_INTENTS = set(PARSE_SUPPORTED_INTENTS)

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

# 必须同时命中的核心概念上限；概念之间是 AND，拆得越细越容易 0 命中，多出来的降为补充词。
MAX_CORE_CONCEPTS = 3

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


def _safe_int(value: Any, default: int = 0) -> int:
    """安全地把值转换为 int；失败时返回默认值。

    这类函数主要服务于自然语言解析后的弱类型数据，避免单个字段解析失败中断整轮 spec 构建。
    """
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_optional_int(value: Any) -> Optional[int]:
    """把输入安全解析成非负整数；不合法时返回 None。

    与 _safe_int 的区别在于，这里显式保留“缺失值”语义，适合 submitted_days_ago 这类可选字段。
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def _clamp(value: int, minimum: int, maximum: int) -> int:
    """把整数裁剪到指定闭区间内。

    用于限制 max_results、submitted_days_ago 等字段，防止自然语言或模型输出给出异常范围。
    """
    return max(minimum, min(maximum, int(value)))


def _normalize_sort_by(value: Any) -> str:
    """把排序字段标准化为 arXiv 工具可接受的枚举值。"""
    text = _normalize_text(str(value or ""))
    lowered = text.lower()
    if lowered in {"relevance"}:
        return "relevance"
    if lowered in {"lastupdateddate", "lastupdated"}:
        return "lastUpdatedDate"
    return "submittedDate"


def _normalize_sort_order(value: Any) -> str:
    """把排序方向标准化为 ascending / descending。"""
    text = _normalize_text(str(value or "")).lower()
    return "ascending" if text == "ascending" else "descending"


def _normalize_field_operator(value: Any) -> str:
    """标准化字段组合运算符，限制在允许集合内。"""
    text = _normalize_text(str(value or "")).upper()
    return text if text in {"AND", "OR", "ANDNOT"} else "AND"


def _normalize_category_operator(value: Any) -> str:
    """标准化类别组合运算符，限制在允许集合内。"""
    text = _normalize_text(str(value or "")).upper()
    return text if text in {"AND", "OR"} else "OR"


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """按顺序去重字符串列表。

    这里保留原顺序是因为 query token、warnings 等序列本身就带有用户表达顺序信息。
    """
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = _normalize_text(item).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(_normalize_text(item))
    return result


def _normalize_and_validate_spec(payload: Mapping[str, Any]) -> Optional[ArxivSearchSpec]:
    """把原始 payload 规范化并校验为 ArxivSearchSpec。

    该函数主要用于消费 LLM 输出，目标是把宽松 JSON 约束成稳定、可执行的业务对象。
    """
    intent = str(payload.get("intent", "arxiv_search") or "arxiv_search").strip().lower()
    if intent not in SUPPORTED_INTENTS or intent != "arxiv_search":
        return None

    concepts = _concepts_from_payload(payload.get("concepts"))
    legacy_query = _normalize_optional_str(payload.get("query"))
    return ArxivSearchSpec(
        intent="arxiv_search",
        concepts=concepts,
        optional_terms=_string_list(payload.get("optional_terms")),
        query=" ".join(concept.term for concept in concepts) if concepts else legacy_query,
        title_query=_normalize_optional_str(payload.get("title_query")),
        abstract_query=_normalize_optional_str(payload.get("abstract_query")),
        categories=get_default_agent_arxiv_categories(),
        submitted_days_ago=_safe_optional_int(payload.get("submitted_days_ago")),
        max_results=_clamp(_safe_int(payload.get("max_results"), default=10), 1, get_agent_max_search_results()),
        sort_by=_normalize_sort_by(payload.get("sort_by")),
        sort_order=_normalize_sort_order(payload.get("sort_order")),
        field_operator=_normalize_field_operator(payload.get("field_operator")),
        category_operator=_normalize_category_operator(payload.get("category_operator")),
        reasoning_summary=_normalize_optional_str(payload.get("reasoning_summary")),
    )


def _user_mentioned_abstract(message: str) -> bool:
    """判断用户是否明确提到了摘要字段搜索。"""
    return bool(re.search(r"(?:摘要|abstract)\s*(?:包含|是|有|为|里|中|搜索|contains|contain)", message, re.IGNORECASE))


def _user_mentioned_title(message: str) -> bool:
    """判断用户是否明确提到了标题字段搜索。"""
    return bool(re.search(r"(?:标题|题目|title)\s*(?:包含|是|有|为|里|中|搜索|contains|contain)", message, re.IGNORECASE))


def _post_process_cleaned_spec(
    spec: ArxivSearchSpec,
    message: str,
    *,
    cleaned_topic_cn: Optional[str] = None,
    cleaned_topic_en: Optional[str] = None,
) -> Tuple[ArxivSearchSpec, List[str]]:
    """对已生成的 search spec 做二次清洗和字段纠偏。

    这一步主要服务于 LLM 输出：优先保留模型识别的主题，但会结合 cleaned_topic_*、
    原始消息和显式字段提示，把 query/title_query/abstract_query 修正到更可执行的状态。
    """
    warnings: List[str] = []

    # 结构化概念已经是主题的唯一来源，query 只是其摘要，不再用 cleaned_topic_* 改写。
    if not spec.concepts:
        normalized_query = _normalize_topic_phrase(cleaned_topic_en or "") if cleaned_topic_en else None
        if not normalized_query:
            normalized_query = _normalize_topic_phrase(cleaned_topic_cn or "") if cleaned_topic_cn else None
        if not normalized_query:
            normalized_query = _normalize_topic_phrase(spec.query) if spec.query else None
        if not normalized_query:
            normalized_query = _extract_query_from_message(message)
        if normalized_query:
            if spec.query and normalized_query != spec.query:
                warnings.append(f'LLM 输出的 query 已重新清洗: "{spec.query}" -> "{normalized_query}"')
            spec.query = normalized_query
        elif spec.query:
            warnings.append(f'LLM 输出的 query 未能抽取出稳定主题: "{spec.query}"，将继续依赖后续规则兜底')

    if spec.query and spec.abstract_query and not _user_mentioned_abstract(message):
        warnings.append(f'用户未明确要求摘要搜索，已自动清空 abstract_query (原值: "{spec.abstract_query}")')
        spec.abstract_query = None

    if spec.query and spec.title_query and not _user_mentioned_title(message):
        warnings.append(f'用户未明确要求标题搜索，已自动清空 title_query (原值: "{spec.title_query}")')
        spec.title_query = None

    return spec, _dedupe_preserve_order(warnings)


def _build_spec_from_rules(message: str) -> Optional[ArxivSearchSpec]:
    """仅基于规则从消息中直接构造 search spec。

    当 LLM 不可用、输出不可信或置信度不足时，这个函数提供纯规则兜底能力。
    """
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
    """在已有 spec 基础上叠加规则补全，生成更可执行的最终版本。

    它不会推翻已有 spec 的核心语义，而是补齐时间范围、排序、默认类别、运算符等隐含字段。
    """
    query = spec.query or _extract_query_from_message(message)
    title_query = spec.title_query
    abstract_query = spec.abstract_query
    categories = get_default_agent_arxiv_categories()
    submitted_days_ago = spec.submitted_days_ago if spec.submitted_days_ago is not None else _extract_submitted_days_ago(message)
    max_results = _clamp(spec.max_results or 10, 1, get_agent_max_search_results())
    sort_by, sort_order = _extract_sorting(message)
    if spec.sort_by in {"relevance", "submittedDate", "lastUpdatedDate"}:
        sort_by = spec.sort_by
        sort_order = spec.sort_order or sort_order

    return ArxivSearchSpec(
        intent="arxiv_search",
        concepts=list(spec.concepts or []),
        optional_terms=list(spec.optional_terms or []),
        constraint_sources=dict(spec.constraint_sources or {}),
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
        reasoning_summary=spec.reasoning_summary or _build_reasoning_summary(query, categories, submitted_days_ago, max_results, sort_by),
    )


def _extract_query_from_message(message: str) -> Optional[str]:
    """从用户消息中提取最可能的主题 query。

    提取顺序为：显式提示词模式 -> 噪声清理 -> 中英混合 token 过滤，
    尽量在可解释性和鲁棒性之间取得平衡。
    """
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
    """根据给定提示模式提取标题/摘要等字段查询词。"""
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            candidate = _normalize_topic_phrase(match.group(1))
            if candidate:
                return candidate
    return None


def _extract_submitted_days_ago(message: str) -> Optional[int]:
    """从消息中解析“最近多少天/周/月”这类时间范围。"""
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


def _extract_requested_count(message: str) -> Optional[int]:
    """从消息中读出用户要的论文数量（未裁剪）；没说数量时返回 None。"""
    for pattern in COUNT_PATTERNS:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            raw_value = match.group(1)
            value = _parse_small_chinese_number(raw_value)
            return value if value is not None else _safe_int(raw_value, default=10)
    return None


def _extract_max_results(message: str) -> int:
    """从消息中提取期望返回的论文数量，并裁剪到允许范围。"""
    requested = _extract_requested_count(message)
    return 10 if requested is None else _clamp(requested, 1, get_agent_max_search_results())


def _extract_sorting(message: str) -> Tuple[str, str]:
    """从消息中推断排序字段与方向。

    “最近/近 N 天”通常是时间过滤条件，不应自动压过主题相关性；只有用户明确
    表达“按最新/最新论文”时才切到 submittedDate。
    """
    lowered = message.lower()
    if any(keyword in lowered for keyword in ("最相关", "相关度高", "most relevant", "relevant", "relevance")):
        return "relevance", "descending"
    if any(keyword in lowered for keyword in ("最新", "latest", "newest")):
        return "submittedDate", "descending"
    return "relevance", "descending"


def _build_reasoning_summary(
    query: Optional[str],
    categories: List[str],
    submitted_days_ago: Optional[int],
    max_results: int,
    sort_by: str,
) -> str:
    """把关键搜索参数拼成简洁的中文 reasoning 摘要。

    该摘要主要用于调试和前端解释，让调用方快速看到 spec 的核心组成部分。
    """
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
    """把候选主题短语清洗成稳定、紧凑的 query 表达。"""
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
    """移除时间、数量、泛化动词和标点等搜索噪声。"""
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
    """按给定正则模式批量删除文本片段。"""
    result = text
    for pattern, _ in compiled_patterns:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return result


def _strip_count_phrases(text: str) -> str:
    """移除“3 篇论文”“十篇 paper”这类数量短语。"""
    result = text
    for pattern in COUNT_PATTERNS:
        result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    return result


def _tokenize_mixed(text: str) -> List[str]:
    """对中英文混合文本做轻量 token 切分。"""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-_/\.]*|[\u4e00-\u9fff]{2,}", text)
    return [token.strip() for token in tokens if token and token.strip()]


def _is_topic_token(token: str) -> bool:
    """判断一个 token 是否值得保留为主题词。"""
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


_CJK_PATTERN = re.compile(r"[一-鿿]")
# 防画像词混入检索条件时三类主题都要识别；但用户没给主题时只能用正向主题补全，排除主题绝不能变成检索词。
_PROFILE_TOPIC_KEYS = ("positive_topics", "recent_topics", "negative_topics")
_PROFILE_FILL_TOPIC_KEYS = ("positive_topics", "recent_topics")


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple, set)) else []
    return [text for text in (_normalize_text(str(item or "")) for item in items) if text]


def _concepts_from_payload(value: Any) -> List[SearchConcept]:
    """把 LLM 的 concepts 载荷转成 SearchConcept；单个字符串也按一个概念处理。"""
    if value is None:
        return []
    items = [value] if isinstance(value, (str, Mapping)) else list(value) if isinstance(value, (list, tuple)) else []
    concepts: List[SearchConcept] = []
    for item in items:
        if isinstance(item, str):
            term, aliases = item, []
        elif isinstance(item, Mapping):
            term, aliases = item.get("term"), item.get("aliases")
        else:
            continue
        term_text = _normalize_text(str(term or ""))
        if term_text:
            concepts.append(SearchConcept(term=term_text, aliases=aliases or [], source="user"))
    return concepts


def _profile_topics(research_profile: Optional[Mapping[str, Any]], keys: Sequence[str] = _PROFILE_TOPIC_KEYS) -> List[str]:
    if not isinstance(research_profile, Mapping):
        return []
    topics: List[str] = []
    for key in keys:
        topics.extend(_string_list(research_profile.get(key)))
    return _dedupe_preserve_order(topics)


def _evidence_in_message(evidence: str, message: str) -> bool:
    compact = lambda text: re.sub(r"\s+", "", text).lower()  # noqa: E731
    return bool(evidence) and compact(evidence) in compact(message)


def _strip_profile_tokens(text: str, allowed_tokens: set[str], profile_tokens: set[str]) -> str:
    """去掉“用户没说过、但出现在画像主题里”的英文 token，其余写法（复数、连字符等）保留。"""
    kept = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-_/\.]*", text)
        if not (set(text_tokens(token)) - allowed_tokens) or not (set(text_tokens(token)) & profile_tokens)
    ]
    return " ".join(kept)


def _looks_like_profile_topic(text: str, profile_topics: Sequence[str]) -> bool:
    tokens = set(text_tokens(text))
    if len(tokens) < 2:
        return False
    for topic in profile_topics:
        topic_tokens = set(text_tokens(topic))
        if len(topic_tokens) >= 2 and (tokens <= topic_tokens or topic_tokens <= tokens):
            return True
    return False


def _ground_concepts(
    raw_concepts: Any,
    fallback_concepts: Sequence[SearchConcept],
    message: str,
    profile_topics: Sequence[str],
) -> Tuple[List[SearchConcept], List[str]]:
    """只保留能在用户原话里找到依据的概念，防止画像主题混进检索条件。

    规则：
    - evidence（LLM 标注的原话片段）含中文时，信任其英文翻译；
    - evidence 是英文时，term 里“原话没有、画像里有”的 token 会被剥掉；
    - 没有可信 evidence 且 term 带画像独有词时，整个概念丢弃；
    - 同义写法若本身就是一条画像主题，也丢弃，避免借 OR 把画像带回来。
    """
    warnings: List[str] = []
    message_tokens = set(text_tokens(message))
    profile_tokens = {token for topic in profile_topics for token in text_tokens(topic)}
    raw_items = list(raw_concepts) if isinstance(raw_concepts, (list, tuple)) else []
    evidence_by_term = {
        _normalize_text(str(item.get("term") or "")).lower(): _normalize_text(str(item.get("evidence") or ""))
        for item in raw_items
        if isinstance(item, Mapping)
    }

    grounded: List[SearchConcept] = []
    for concept in fallback_concepts:
        term = concept.term
        evidence = evidence_by_term.get(term.lower(), "")
        aliases = [alias for alias in concept.aliases if not _looks_like_profile_topic(alias, profile_topics) or set(text_tokens(alias)) <= message_tokens]
        term_tokens = set(text_tokens(term))
        foreign_profile_tokens = (term_tokens - message_tokens) & profile_tokens
        if foreign_profile_tokens:
            evidence_ok = _evidence_in_message(evidence, message)
            if evidence_ok and _CJK_PATTERN.search(evidence):
                pass
            elif evidence_ok:
                allowed = message_tokens | set(text_tokens(evidence))
                stripped = _strip_profile_tokens(term, allowed, profile_tokens)
                if stripped and text_tokens(stripped):
                    warnings.append(f'检索概念 "{term}" 中有用户未提及的画像词，已收敛为 "{stripped}"')
                    term = stripped
                else:
                    warnings.append(f'检索概念 "{term}" 无法在用户原话中找到依据，已改用原话 "{evidence}"')
                    term = evidence
            else:
                warnings.append(f'检索概念 "{term}" 来自画像而非用户原话，已丢弃')
                continue
        grounded.append(SearchConcept(term=term, aliases=[alias for alias in aliases if alias.lower() != term.lower()], source="user"))
    return grounded, warnings


def _ground_optional_terms(terms: Sequence[str], message: str, profile_topics: Sequence[str]) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    message_tokens = set(text_tokens(message))
    profile_tokens = {token for topic in profile_topics for token in text_tokens(topic)}
    has_cjk = bool(_CJK_PATTERN.search(message))
    kept: List[str] = []
    for term in terms:
        term_tokens = set(text_tokens(term))
        foreign = (term_tokens - message_tokens) & profile_tokens
        # 中文原话的翻译天然对不上英文 token，只在补充词整体就是画像主题时丢弃。
        dropped = _looks_like_profile_topic(term, profile_topics) if has_cjk else bool(foreign)
        if dropped and not term_tokens <= message_tokens:
            warnings.append(f'补充检索词 "{term}" 来自画像而非用户原话，已丢弃')
            continue
        kept.append(term)
    return kept, warnings


def _extract_explicit_categories(message: str) -> List[str]:
    """识别用户直接写出的 arXiv 分类代码，例如 cs.CL、stat.ML。"""
    valid = {category.lower(): category for category in get_valid_arxiv_categories()}
    found: List[str] = []
    for candidate in re.findall(r"\b[a-zA-Z\-]+\.[a-zA-Z\-]+\b", message):
        canonical = valid.get(candidate.lower())
        if canonical and canonical not in found:
            found.append(canonical)
    return found


def _ground_search_spec(
    spec: ArxivSearchSpec,
    message: str,
    *,
    raw_payload: Optional[Mapping[str, Any]] = None,
    research_profile: Optional[Mapping[str, Any]] = None,
) -> Tuple[ArxivSearchSpec, List[str]]:
    """parse 阶段的最后一道收口：主题落地到用户原话，并标注每个约束的来源。

    - concepts 只保留有原话依据的部分；全被剔除时回退到规则抽取的主题；
    - 用户完全没给主题时才用画像补一个 source=profile 的概念，回答里会说明；
    - 时间、数量、排序、分类按“原话里能否直接读出来”标注 user / inferred / default。
    """
    warnings: List[str] = []
    profile_topics = _profile_topics(research_profile)
    payload = raw_payload if isinstance(raw_payload, Mapping) else {}

    # 旧格式只给 query 时同样要过一遍原话校验，否则画像词会从 query 这条路混进来。
    candidate_concepts = list(spec.concepts or []) or (
        [SearchConcept(term=spec.query, source="user")] if spec.query and text_tokens(spec.query) else []
    )
    concepts, concept_warnings = _ground_concepts(payload.get("concepts"), candidate_concepts, message, profile_topics)
    warnings.extend(concept_warnings)
    optional_terms, optional_warnings = _ground_optional_terms(list(spec.optional_terms or []), message, profile_topics)
    warnings.extend(optional_warnings)
    if len(concepts) > MAX_CORE_CONCEPTS:
        # LLM 偶尔仍会把研究角度也拆成概念；排在后面的降为补充词，阶梯检索才有放宽余地。
        demoted = concepts[MAX_CORE_CONCEPTS:]
        concepts = concepts[:MAX_CORE_CONCEPTS]
        optional_terms = [concept.term for concept in demoted] + [term for term in optional_terms if term.lower() not in {c.term.lower() for c in demoted}]
        warnings.append(f"检索概念过多，已把 {'、'.join(concept.term for concept in demoted)} 降为补充词")

    if not concepts:
        rule_query = _extract_query_from_message(message)
        if rule_query and text_tokens(rule_query):
            concepts = [SearchConcept(term=rule_query, source="user")]
        elif not rule_query and not spec.title_query and not spec.abstract_query:
            fill_topics = _profile_topics(research_profile, _PROFILE_FILL_TOPIC_KEYS)
            if fill_topics:
                concepts = [SearchConcept(term=fill_topics[0], source="profile")]
                warnings.append(f'用户未给出检索主题，已按研究画像补全为 "{fill_topics[0]}"')

    sources: dict[str, str] = {}
    explicit_categories = _extract_explicit_categories(message)
    categories = explicit_categories or list(spec.categories or [])
    sources["categories"] = "user" if explicit_categories else "default"

    rule_days = _extract_submitted_days_ago(message)
    submitted_days_ago = rule_days if rule_days is not None else spec.submitted_days_ago
    if submitted_days_ago is not None:
        sources["submitted_days_ago"] = "user" if rule_days is not None else "inferred"

    requested_count = _extract_requested_count(message)
    max_results = _extract_max_results(message) if requested_count is not None else spec.max_results
    if requested_count is None:
        sources["max_results"] = "default"
    else:
        # 超过单次上限时不能静默截断，标成 user_capped 让回答说明实际按多少篇检索。
        sources["max_results"] = "user_capped" if requested_count > max_results else "user"

    # 排序以原话为准：只有用户说“最新/最相关”才算明确要求，否则默认按相关性。
    sort_by, sort_order = _extract_sorting(message)
    lowered = message.lower()
    sort_stated = any(keyword in lowered for keyword in ("最相关", "相关度高", "most relevant", "relevance", "最新", "latest", "newest"))
    sources["sort_by"] = "user" if sort_stated else "default"

    grounded = ArxivSearchSpec(
        intent="arxiv_search",
        concepts=concepts,
        optional_terms=optional_terms,
        constraint_sources=sources,
        query=" ".join(concept.term for concept in concepts) if concepts else spec.query,
        title_query=spec.title_query,
        abstract_query=spec.abstract_query,
        categories=categories,
        submitted_days_ago=submitted_days_ago,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
        field_operator=spec.field_operator,
        category_operator=spec.category_operator,
        reasoning_summary=spec.reasoning_summary,
    )
    return grounded, _dedupe_preserve_order(warnings)


__all__ = [
    "_ground_search_spec",
    "ABSTRACT_HINT_PATTERNS",
    "MAX_CORE_CONCEPTS",
    "QUERY_HINT_PATTERNS",
    "TITLE_HINT_PATTERNS",
    "TIME_PATTERNS",
    "COUNT_PATTERNS",
    "_apply_rule_enrichment",
    "_build_reasoning_summary",
    "_build_spec_from_rules",
    "_extract_marked_query",
    "_extract_max_results",
    "_extract_query_from_message",
    "_extract_sorting",
    "_extract_submitted_days_ago",
    "_normalize_and_validate_spec",
    "_normalize_category_operator",
    "_normalize_field_operator",
    "_normalize_sort_by",
    "_normalize_sort_order",
    "_post_process_cleaned_spec",
    "_safe_int",
    "_safe_optional_int",
]
