from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schemas import ArxivSearchSpec
from ..utils.search_ladder import build_search_ladder, compose_search_query, paper_key, text_tokens
from .base import BaseToolAdapter, backend_tool_error
from .models import ToolExecutionResult


class NormalizeRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    intent: Optional[str] = None
    message: str = ""
    search_spec: Any = None


class NormalizedRequestOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    message: str
    search_spec: Optional[Dict[str, Any]] = None


_SEARCH_SPEC_FIELDS = {
    "intent",
    "concepts",
    "optional_terms",
    "constraint_sources",
    "query",
    "title_query",
    "abstract_query",
    "categories",
    "submitted_days_ago",
    "max_results",
    "sort_by",
    "sort_order",
    "field_operator",
    "category_operator",
    "reasoning_summary",
}


def _coerce_model_payload(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return value


def _payload_from_constraint_list(value: Any) -> Dict[str, Any]:
    """把 LLM planner 偶尔传来的 goal.constraints 列表恢复成搜索字段。"""
    if not isinstance(value, (list, tuple, set)):
        return {}
    payload: Dict[str, Any] = {}
    for item in value:
        text = str(item or "").strip()
        if not text or "=" not in text:
            continue
        key, raw_value = text.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            continue
        if key == "categories":
            payload[key] = [part.strip() for part in raw_value.split(",") if part.strip()]
        elif key in {"submitted_days_ago", "max_results"}:
            try:
                payload[key] = int(raw_value)
            except ValueError:
                payload[key] = raw_value
        elif key in _SEARCH_SPEC_FIELDS:
            payload[key] = raw_value
    return payload


def _split_sort_by_order(payload: Dict[str, Any]) -> None:
    sort_by = str(payload.get("sort_by") or "").strip()
    if ":" not in sort_by:
        return
    field, order = [part.strip() for part in sort_by.split(":", 1)]
    if field:
        payload["sort_by"] = field
    if order and not payload.get("sort_order"):
        payload["sort_order"] = order


def _normalize_search_spec_payload(value: Any) -> Any:
    """归一化 planner 草稿中的搜索规格，修复缺 intent 和 sort_by:sort_order 混写。"""
    value = _coerce_model_payload(value)
    if value in (None, "", [], {}):
        return value
    if isinstance(value, (list, tuple, set)):
        value = _payload_from_constraint_list(value)
    if not isinstance(value, Mapping):
        return value

    payload = dict(value)
    constraints_payload = _payload_from_constraint_list(payload.get("constraints"))
    if constraints_payload:
        # constraints 是 planner 的解释性产物，只作为缺省补齐；显式字段优先级更高。
        payload = {**constraints_payload, **payload}
    _split_sort_by_order(payload)
    if _looks_like_search_spec(payload):
        payload.setdefault("intent", "arxiv_search")
    return payload


def _search_spec_payload_for_validation(value: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _normalize_search_spec_payload(value)
    if not isinstance(payload, Mapping):
        return {}
    cleaned = {key: payload.get(key) for key in _SEARCH_SPEC_FIELDS if payload.get(key) not in (None, "", [], {})}
    if cleaned:
        cleaned.setdefault("intent", "arxiv_search")
    return cleaned


def _rewrite_search_spec_payload(value: Any) -> Dict[str, Any]:
    """从恢复链输入中取出可执行搜索规格，兼容直接规格和搜索结果包裹两种形态。"""
    payload = _normalize_search_spec_payload(value)
    if isinstance(payload, Mapping) and _looks_like_search_spec(payload):
        return _search_spec_payload_for_validation(payload)
    if isinstance(payload, Mapping):
        for key in ("search_spec", "arxiv_search_spec", "rewritten_search_spec"):
            nested_payload = _normalize_search_spec_payload(payload.get(key))
            if isinstance(nested_payload, Mapping) and _looks_like_search_spec(nested_payload):
                return _search_spec_payload_for_validation(nested_payload)
    return {}


class BuildArxivSearchSpecInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    normalized_request: Optional[Dict[str, Any]] = None
    search_spec: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    @field_validator("normalized_request", "search_spec", mode="before")
    @classmethod
    def _coerce_search_spec_model(cls, value: Any) -> Any:
        # LLM planner 可能把 AgentState.search_spec 或 goal.constraints 绑定到任一输入名；
        # 在工具边界统一降级和修正，避免执行器因草稿字段形态差异提前兜底。
        return _normalize_search_spec_payload(value)


class ArxivSearchSpecOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    optional_terms: List[str] = Field(default_factory=list)
    constraint_sources: Dict[str, str] = Field(default_factory=dict)
    query: Optional[str] = None
    title_query: Optional[str] = None
    abstract_query: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    submitted_days_ago: Optional[int] = None
    max_results: int = 10
    sort_by: str = "submittedDate"
    sort_order: str = "descending"
    field_operator: str = "AND"
    category_operator: str = "OR"
    reasoning_summary: Optional[str] = None


class SearchArxivInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_spec: ArxivSearchSpec


class SearchArxivOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    papers: List[Dict[str, Any]] = Field(default_factory=list)
    tool_result: Dict[str, Any] = Field(default_factory=dict)
    search_spec: Dict[str, Any] = Field(default_factory=dict)
    # 阶梯检索的执行记录：每级查询与命中数、是否用尽、被锁定不放宽的用户约束。
    relaxation: Dict[str, Any] = Field(default_factory=dict)


def _relaxation_summary(
    spec: ArxivSearchSpec,
    *,
    attempts: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    note: Optional[str] = None,
) -> Dict[str, Any]:
    sources = dict(spec.constraint_sources or {})
    locked: List[str] = []
    if spec.submitted_days_ago is not None and sources.get("submitted_days_ago") == "user":
        locked.append("submitted_days_ago")
    if spec.categories and sources.get("categories") == "user":
        locked.append("categories")
    requested = int(spec.max_results or 10)
    applied = next((item["level"] for item in reversed(attempts) if item.get("ok") and item.get("new_papers")), None)
    summary: Dict[str, Any] = {
        "attempts": attempts,
        "requested": requested,
        "returned": len(papers),
        "applied_level": applied,
        # 只要最终仍未凑够数量，就说明可自动放宽的条件已经用完。
        "exhausted": len(papers) < requested,
        "locked_constraints": locked,
        "profile_concepts": [concept.term for concept in spec.concepts if concept.source == "profile"],
        "count_capped": sources.get("max_results") == "user_capped",
    }
    if note:
        summary["note"] = note
    return summary


class ValidateArxivResultsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_results: Dict[str, Any] = Field(default_factory=dict)


class ValidateArxivResultsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    result_count: int = 0
    warnings: List[str] = Field(default_factory=list)


class RewriteArxivQueryInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    search_spec: Optional[Dict[str, Any]] = None


class RewriteArxivQueryOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    intent: str = "arxiv_search"
    concepts: List[Dict[str, Any]] = Field(default_factory=list)
    optional_terms: List[str] = Field(default_factory=list)
    constraint_sources: Dict[str, str] = Field(default_factory=dict)
    query: Optional[str] = None
    title_query: Optional[str] = None
    abstract_query: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    submitted_days_ago: Optional[int] = None
    max_results: int = 10
    sort_by: str = "submittedDate"
    sort_order: str = "descending"
    field_operator: str = "AND"
    category_operator: str = "OR"
    reasoning_summary: Optional[str] = None


class PersonalizePaperResultsInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    arxiv_results: Any = None
    user_memory_summary: Any = None
    research_profile: Any = None


class RankedPapersOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranked_papers: List[Dict[str, Any]] = Field(default_factory=list)


class SynthesizeArxivResponseInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    ranked_papers: Optional[List[Dict[str, Any]]] = None
    arxiv_result_quality: Optional[Dict[str, Any]] = None
    arxiv_results: Any = None
    personalized_rerank_applied: bool = False

    @field_validator("ranked_papers", mode="before")
    @classmethod
    def _coerce_ranked_papers(cls, value: Any) -> Any:
        # LLM 草稿可能把 search_arxiv 的完整输出包裹体绑定给 ranked_papers；
        # 回答工具只需要论文列表，因此在 adapter 边界展开 papers，避免最终回复阶段因类型不匹配失败。
        if isinstance(value, Mapping):
            return extract_papers(value)
        return value


class FinalAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str


def extract_papers(search_result: Any) -> List[Dict[str, Any]]:
    if isinstance(search_result, Mapping):
        papers = search_result.get("papers")
        if isinstance(papers, list):
            return [dict(item) for item in papers if isinstance(item, Mapping)]
    if isinstance(search_result, list):
        return [dict(item) for item in search_result if isinstance(item, Mapping)]
    return []


def _looks_like_search_spec(value: Mapping[str, Any]) -> bool:
    # normalized_request 正常应包含 message/search_spec；若它本身已经带检索字段，
    # 说明 planner 直接绑定了结构化规格，adapter 应保留这些约束而不是退回 message 兜底。
    return any(value.get(key) not in (None, "", [], {}) for key in ("concepts", "query", "title_query", "abstract_query", "categories"))


class NormalizeRequestAdapter(BaseToolAdapter[NormalizeRequestInput, NormalizedRequestOutput]):
    tool_name = "normalize_request"
    input_model = NormalizeRequestInput
    output_model = NormalizedRequestOutput

    def _run(self, tool_input: NormalizeRequestInput) -> NormalizedRequestOutput:
        search_spec = tool_input.search_spec
        if isinstance(search_spec, ArxivSearchSpec):
            search_spec = search_spec.model_dump()
        elif hasattr(search_spec, "model_dump"):
            search_spec = search_spec.model_dump()
        return NormalizedRequestOutput(
            intent=str(tool_input.intent or "").strip() or "unsupported",
            message=str(tool_input.message or "").strip(),
            search_spec=search_spec if isinstance(search_spec, Mapping) else None,
        )


class BuildArxivSearchSpecAdapter(BaseToolAdapter[BuildArxivSearchSpecInput, ArxivSearchSpecOutput]):
    tool_name = "build_arxiv_search_spec"
    input_model = BuildArxivSearchSpecInput
    output_model = ArxivSearchSpecOutput

    def _run(self, tool_input: BuildArxivSearchSpecInput) -> ArxivSearchSpecOutput:
        normalized_request = tool_input.normalized_request or {}
        existing_spec = _normalize_search_spec_payload(normalized_request.get("search_spec")) if isinstance(normalized_request, Mapping) else None
        if isinstance(existing_spec, Mapping) and existing_spec:
            spec = ArxivSearchSpec.model_validate(_search_spec_payload_for_validation(existing_spec))
        elif isinstance(normalized_request, Mapping) and _looks_like_search_spec(normalized_request):
            # LLM 草稿有时会把结构化 search_spec 直接塞进 normalized_request；
            # 这里按同一规格解析，避免因为字段名选择不同而丢失已解析好的检索约束。
            spec = ArxivSearchSpec.model_validate(_search_spec_payload_for_validation(normalized_request))
        elif isinstance(tool_input.search_spec, Mapping) and tool_input.search_spec:
            spec = ArxivSearchSpec.model_validate(_search_spec_payload_for_validation(tool_input.search_spec))
        else:
            message = str((normalized_request or {}).get("message") or tool_input.message or "").strip()
            spec = ArxivSearchSpec(intent="arxiv_search", query=message, max_results=10)
        return ArxivSearchSpecOutput.model_validate(spec.model_dump())


class SearchArxivAdapter(BaseToolAdapter[SearchArxivInput, SearchArxivOutput]):
    tool_name = "search_arxiv"
    input_model = SearchArxivInput
    output_model = SearchArxivOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def execute(self, tool_input: SearchArxivInput) -> ToolExecutionResult:
        result = super().execute(tool_input)
        if result.ok and isinstance(result.data, SearchArxivOutput):
            backend_result = result.data.tool_result
            if backend_result and not bool(backend_result.get("ok", False)):
                return result.model_copy(
                    update={
                        "ok": False,
                        "error": backend_tool_error(
                            error_code=((backend_result.get("error") or {}).get("code") if isinstance(backend_result.get("error"), Mapping) else None) or "backend_tool_failed",
                            message=str((backend_result.get("error") or {}).get("message") if isinstance(backend_result.get("error"), Mapping) else backend_result.get("summary") or "arXiv 搜索工具失败"),
                            detail={"backend_error": backend_result.get("error")},
                            retryable=True,
                            suggested_recovery="rewrite_arxiv_query",
                            safe_debug={"backend_tool_name": "search_arxiv_structured"},
                        ),
                    }
                )
        return result

    def _run(self, tool_input: SearchArxivInput) -> SearchArxivOutput:
        spec = tool_input.search_spec
        tool_args = spec.model_dump(exclude_none=True)
        levels = build_search_ladder(spec)
        if not levels:
            return SearchArxivOutput(
                papers=[],
                tool_result={"ok": True, "tool_name": "search_arxiv_raw", "summary": "没有可检索的英文主题词", "data": {"papers": []}},
                search_spec=tool_args,
                relaxation=_relaxation_summary(spec, attempts=[], papers=[], note="no_searchable_terms"),
            )

        requested = int(spec.max_results or 10)
        collected: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        attempts: List[Dict[str, Any]] = []
        last_result: Dict[str, Any] = {}
        for level in levels:
            search_query = compose_search_query(level)
            tool_result = dict(
                self.invoke_backend_tool(
                    "search_arxiv_raw",
                    search_query=search_query,
                    max_results=requested,
                    submitted_days_ago=spec.submitted_days_ago,
                    sort_by=spec.sort_by,
                    sort_order=spec.sort_order,
                )
                or {}
            )
            attempt = {
                "level": level.level,
                "label": level.label,
                "query": level.query,
                "categories": list(level.categories),
                "submitted_days_ago": spec.submitted_days_ago,
                "ok": bool(tool_result.get("ok", False)),
            }
            if not attempt["ok"]:
                attempts.append(attempt)
                if collected:
                    # 已有更严格级别的结果时，后续放宽失败不应吞掉已拿到的论文。
                    break
                # 首次检索就失败属于工具故障，交给 execute() 转成可恢复错误。
                return SearchArxivOutput(papers=[], tool_result=tool_result, search_spec=tool_args, relaxation=_relaxation_summary(spec, attempts=attempts, papers=[]))
            last_result = tool_result
            data = tool_result.get("data") if isinstance(tool_result.get("data"), Mapping) else {}
            level_papers = extract_papers(data)
            added = 0
            for paper in level_papers:
                key = paper_key(paper)
                if key and key in seen_keys:
                    continue
                if key:
                    seen_keys.add(key)
                # 记录论文来自哪一级，越严格的级别排得越靠前，也方便回答里解释“这些是放宽后找到的”。
                collected.append({**paper, "match_level": level.level})
                added += 1
            attempt["total_results"] = int(data.get("total_results") or len(level_papers))
            attempt["new_papers"] = added
            attempts.append(attempt)
            if len(collected) >= requested:
                break

        papers = collected[:requested]
        merged_result = dict(last_result)
        merged_data = dict(last_result.get("data") or {}) if isinstance(last_result.get("data"), Mapping) else {}
        merged_data["papers"] = papers
        merged_result["data"] = merged_data
        merged_result["summary"] = f"找到 {len(papers)} 篇 arXiv 论文"
        merged_trace = dict(last_result.get("trace") or {}) if isinstance(last_result.get("trace"), Mapping) else {}
        merged_trace["ladder_attempts"] = attempts
        merged_result["trace"] = merged_trace
        return SearchArxivOutput(
            papers=papers,
            tool_result=merged_result,
            search_spec=tool_args,
            relaxation=_relaxation_summary(spec, attempts=attempts, papers=papers),
        )


class ValidateArxivResultsAdapter(BaseToolAdapter[ValidateArxivResultsInput, ValidateArxivResultsOutput]):
    tool_name = "validate_arxiv_results"
    input_model = ValidateArxivResultsInput
    output_model = ValidateArxivResultsOutput

    def _run(self, tool_input: ValidateArxivResultsInput) -> ValidateArxivResultsOutput:
        papers = extract_papers(tool_input.arxiv_results)
        tool_result = tool_input.arxiv_results.get("tool_result") if isinstance(tool_input.arxiv_results, Mapping) else None
        warnings: List[str] = []
        if not papers:
            warnings.append("no_results")
        if isinstance(tool_result, Mapping) and not bool(tool_result.get("ok", False)):
            warnings.append("tool_failed")
        return ValidateArxivResultsOutput(ok=bool(papers) and "tool_failed" not in warnings, result_count=len(papers), warnings=warnings)


class RewriteArxivQueryAdapter(BaseToolAdapter[RewriteArxivQueryInput, RewriteArxivQueryOutput]):
    tool_name = "rewrite_arxiv_query"
    input_model = RewriteArxivQueryInput
    output_model = RewriteArxivQueryOutput

    def _run(self, tool_input: RewriteArxivQueryInput) -> RewriteArxivQueryOutput:
        payload = _rewrite_search_spec_payload(tool_input.search_spec)
        if not payload:
            payload = {"intent": "arxiv_search", "max_results": 10}
        # 重写兜底优先扩大普通 query，清空精确字段以避免再次被 title/abstract 约束卡住。
        payload["title_query"] = None
        payload["abstract_query"] = None
        payload.setdefault("intent", "arxiv_search")
        payload.setdefault("max_results", 10)
        return RewriteArxivQueryOutput.model_validate(payload)


class PersonalizePaperResultsAdapter(BaseToolAdapter[PersonalizePaperResultsInput, RankedPapersOutput]):
    tool_name = "personalize_paper_results"
    input_model = PersonalizePaperResultsInput
    output_model = RankedPapersOutput

    def _run(self, tool_input: PersonalizePaperResultsInput) -> RankedPapersOutput:
        papers = extract_papers(tool_input.arxiv_results)
        positive_topics, negative_topics = _profile_topic_lists(tool_input.research_profile, tool_input.user_memory_summary)
        scored = []
        for index, paper in enumerate(papers):
            text = f"{paper.get('title') or ''} {paper.get('summary') or paper.get('abstract') or ''}"
            matched = [topic for topic in positive_topics if _topic_matches(topic, text)]
            avoided = [topic for topic in negative_topics if _topic_matches(topic, text)]
            # 匹配级别（阶梯越严格越靠前）始终是第一排序键：画像只在同一检索级别内部调整先后，
            # 不能把只是“符合画像”的放宽结果排到严格命中的论文前面。
            level_rank = _LEVEL_ORDER.get(str(paper.get("match_level") or ""), 0)
            scored.append(((level_rank, len(avoided) - len(matched), index), paper, matched))
        ranked: List[Dict[str, Any]] = []
        for rank, (_, paper, matched) in enumerate(sorted(scored, key=lambda item: item[0]), start=1):
            next_paper = dict(paper)
            next_paper["rank"] = rank
            if matched:
                next_paper.setdefault("reason", f"matched_profile_topics:{', '.join(matched[:3])}")
            ranked.append(next_paper)
        return RankedPapersOutput(ranked_papers=ranked)


_LEVEL_ORDER = {"strict": 0, "drop_optional_terms": 1, "loosen_phrases": 2, "drop_inferred_categories": 3}


def _profile_topic_lists(*sources: Any) -> tuple[List[str], List[str]]:
    positive: List[str] = []
    negative: List[str] = []
    for source in sources:
        source = _coerce_model_payload(source)
        if not isinstance(source, Mapping):
            continue
        for key in ("positive_topics", "recent_topics"):
            positive.extend(str(item).strip() for item in source.get(key) or [] if str(item).strip())
        negative.extend(str(item).strip() for item in source.get("negative_topics") or [] if str(item).strip())
    return list(dict.fromkeys(positive)), list(dict.fromkeys(negative))


def _topic_matches(topic: str, text: str) -> bool:
    """画像主题的英文词全部出现在标题/摘要中才算命中，避免单个泛词把排序带偏。"""
    topic_tokens = set(text_tokens(topic))
    return bool(topic_tokens) and topic_tokens <= set(text_tokens(text))


class SynthesizeArxivResponseAdapter(BaseToolAdapter[SynthesizeArxivResponseInput, FinalAnswerOutput]):
    tool_name = "synthesize_arxiv_response"
    input_model = SynthesizeArxivResponseInput
    output_model = FinalAnswerOutput

    def _run(self, tool_input: SynthesizeArxivResponseInput) -> FinalAnswerOutput:
        ranked_papers = tool_input.ranked_papers or extract_papers(tool_input.arxiv_results)
        quality = tool_input.arxiv_result_quality or {}
        result_count = int(quality.get("result_count") or len(ranked_papers or []))
        relaxation = _relaxation_from(tool_input.arxiv_results)
        if result_count <= 0:
            return FinalAnswerOutput(final_answer=_empty_result_answer(relaxation))
        titles = [str((paper or {}).get("title") or "").strip() for paper in list(ranked_papers or [])[:3] if str((paper or {}).get("title") or "").strip()]
        title_summary = "；".join(titles) if titles else "已返回相关论文"
        if tool_input.personalized_rerank_applied:
            answer = f"已检索到 {result_count} 篇相关 arXiv 论文，并结合你的研究画像调整了排序。优先关注：{title_summary}。"
        else:
            answer = f"已检索到 {result_count} 篇相关 arXiv 论文。优先关注：{title_summary}。"
        notes = _relaxation_notes(relaxation, result_count)
        return FinalAnswerOutput(final_answer="\n\n".join([answer, *notes]) if notes else answer)


def _relaxation_from(value: Any) -> Dict[str, Any]:
    value = _coerce_model_payload(value)
    relaxation = value.get("relaxation") if isinstance(value, Mapping) else None
    return dict(relaxation) if isinstance(relaxation, Mapping) else {}


def _describe_attempt(attempt: Mapping[str, Any]) -> str:
    parts = [f"`{attempt.get('query')}`"]
    categories = [str(item) for item in attempt.get("categories") or []]
    parts.append(f"分类 {', '.join(categories)}" if categories else "不限分类")
    if attempt.get("submitted_days_ago") is not None:
        parts.append(f"最近 {attempt.get('submitted_days_ago')} 天")
    result = f"{attempt.get('total_results', 0)} 篇" if attempt.get("ok") else "检索失败"
    return f"{attempt.get('label') or attempt.get('level')}：{' · '.join(parts)} → {result}"


def _empty_result_answer(relaxation: Mapping[str, Any]) -> str:
    attempts = [item for item in relaxation.get("attempts") or [] if isinstance(item, Mapping)]
    if relaxation.get("note") == "no_searchable_terms":
        return "没能从你的描述里提取出可用于 arXiv 检索的英文关键词。可以直接给出英文术语（例如 retrieval-augmented generation），我再帮你检索。"
    if not attempts:
        return "没有检索到符合条件的 arXiv 论文。可以尝试放宽时间范围或换一个更通用的关键词。"
    lines = ["没有找到符合条件的论文。已依次尝试以下检索："]
    lines.extend(f"{index}. {_describe_attempt(attempt)}" for index, attempt in enumerate(attempts, start=1))
    suggestions: List[str] = []
    days = next((item.get("submitted_days_ago") for item in attempts if item.get("submitted_days_ago") is not None), None)
    locked = set(relaxation.get("locked_constraints") or [])
    if days is not None:
        reason = "这是你指定的时间范围，我没有自动放宽" if "submitted_days_ago" in locked else "当前限定了时间范围"
        suggestions.append(f"放宽时间范围（{reason}），例如改为最近 {max(int(days) * 4, 30)} 天")
    if "categories" in locked:
        suggestions.append("去掉你指定的分类限制")
    suggestions.append("换一个更通用的关键词，或补充常见的英文写法")
    lines.append("建议：" + "；".join(suggestions) + "。")
    return "\n".join(lines)


def _relaxation_notes(relaxation: Mapping[str, Any], result_count: int) -> List[str]:
    notes: List[str] = []
    profile_concepts = [str(item) for item in relaxation.get("profile_concepts") or [] if str(item).strip()]
    if profile_concepts:
        notes.append(f"你没有指定检索主题，我按研究画像补全为「{'、'.join(profile_concepts)}」。")
    attempts = [item for item in relaxation.get("attempts") or [] if isinstance(item, Mapping) and item.get("ok")]
    relaxed = [item for item in attempts[1:] if item.get("new_papers")]
    if relaxed:
        labels = "、".join(str(item.get("label") or item.get("level")) for item in relaxed)
        strict_count = int(attempts[0].get("new_papers") or 0) if attempts else 0
        notes.append(f"严格条件下只找到 {strict_count} 篇，其余结果来自放宽后的检索（{labels}）。")
    requested = int(relaxation.get("requested") or 0)
    if relaxation.get("count_capped") and requested:
        notes.append(f"你要的数量超过了单次检索上限，本次最多返回 {requested} 篇。")
    if requested and result_count < requested and relaxation.get("attempts"):
        notes.append(f"可自动放宽的条件已经用完，实际只找到 {result_count} 篇（你要的是 {requested} 篇）。")
    return notes
