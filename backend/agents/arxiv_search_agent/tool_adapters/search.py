from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import ArxivSearchSpec
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


class BuildArxivSearchSpecInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    normalized_request: Optional[Dict[str, Any]] = None
    search_spec: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ArxivSearchSpecOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
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
    query: Optional[str] = None
    title_query: Optional[str] = None
    abstract_query: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    max_results: int = 10


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
        existing_spec = normalized_request.get("search_spec") if isinstance(normalized_request, Mapping) else None
        if isinstance(existing_spec, Mapping) and existing_spec:
            spec = ArxivSearchSpec.model_validate(existing_spec)
        elif isinstance(tool_input.search_spec, Mapping) and tool_input.search_spec:
            spec = ArxivSearchSpec.model_validate(tool_input.search_spec)
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
        tool_args = tool_input.search_spec.model_dump(exclude_none=True)
        tool_result = self.invoke_backend_tool("search_arxiv_structured", **tool_args)
        return SearchArxivOutput(
            papers=extract_papers((tool_result or {}).get("data") or {}),
            tool_result=dict(tool_result or {}),
            search_spec=tool_args,
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
        payload = dict(tool_input.search_spec or {})
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
        personalized = bool(tool_input.user_memory_summary or tool_input.research_profile)
        ranked: List[Dict[str, Any]] = []
        for index, paper in enumerate(papers, start=1):
            next_paper = dict(paper)
            next_paper["rank"] = index
            if personalized:
                next_paper.setdefault("reason", "matched_profile_context")
            ranked.append(next_paper)
        return RankedPapersOutput(ranked_papers=ranked)


class SynthesizeArxivResponseAdapter(BaseToolAdapter[SynthesizeArxivResponseInput, FinalAnswerOutput]):
    tool_name = "synthesize_arxiv_response"
    input_model = SynthesizeArxivResponseInput
    output_model = FinalAnswerOutput

    def _run(self, tool_input: SynthesizeArxivResponseInput) -> FinalAnswerOutput:
        ranked_papers = tool_input.ranked_papers or extract_papers(tool_input.arxiv_results)
        quality = tool_input.arxiv_result_quality or {}
        result_count = int(quality.get("result_count") or len(ranked_papers or []))
        if result_count <= 0:
            return FinalAnswerOutput(final_answer="当前没有检索到合适的 arXiv 结果，建议收窄或改写查询后重试。")
        titles = [str((paper or {}).get("title") or "").strip() for paper in list(ranked_papers or [])[:3] if str((paper or {}).get("title") or "").strip()]
        title_summary = "；".join(titles) if titles else "已返回相关论文"
        if tool_input.personalized_rerank_applied:
            return FinalAnswerOutput(final_answer=f"已检索到 {result_count} 篇相关 arXiv 论文，并结合用户上下文完成排序。优先关注：{title_summary}。")
        return FinalAnswerOutput(final_answer=f"已检索到 {result_count} 篇相关 arXiv 论文。优先关注：{title_summary}。")
