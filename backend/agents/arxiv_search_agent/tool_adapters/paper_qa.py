from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import BaseToolAdapter, backend_tool_error
from .models import ToolExecutionResult

try:
    from ..utils.paper_reference_resolver import _resolve_paper_reference
except Exception:  # pragma: no cover - 测试轻量导入场景下允许缺失。
    _resolve_paper_reference = None


logger = logging.getLogger(__name__)


class ResolvePaperInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""
    selected_paper: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class PaperReferenceOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    query: Optional[str] = None
    matched_by: Optional[str] = None
    source: Optional[str] = None


class CheckPaperIndexInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_ref: Optional[Dict[str, Any]] = None
    paper_reference: Optional[Dict[str, Any]] = None

    @property
    def arxiv_id(self) -> str:
        paper_ref = self.paper_ref or self.paper_reference or {}
        return str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""


class PaperIndexStatusOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    has_index: bool = False
    tool_result: Optional[Dict[str, Any]] = None


class ParseAndIndexPaperInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_reference: Optional[Dict[str, Any]] = None
    paper_ref: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _require_arxiv_id(self) -> "ParseAndIndexPaperInput":
        if not self.arxiv_id:
            raise ValueError("missing_arxiv_id")
        return self

    @property
    def arxiv_id(self) -> str:
        paper_ref = self.paper_reference or self.paper_ref or {}
        return str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""


class IndexBuildOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    has_index: Optional[bool] = None
    tool_result: Optional[Dict[str, Any]] = None


class AnswerPaperQuestionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_ref: Optional[Dict[str, Any]] = None
    paper_reference: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    question: Optional[str] = None
    qa_recovery_strategy: Optional[Dict[str, Any]] = None
    index_strategy: Optional[Dict[str, Any]] = None

    @property
    def arxiv_id(self) -> str:
        paper_ref = self.paper_ref or self.paper_reference or {}
        return str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""

    @property
    def resolved_question(self) -> str:
        return str(self.question or self.message or "").strip()


class PaperQAAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "failed"
    answer: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_debug: Any = None
    error: Any = None
    arxiv_id: Optional[str] = None
    question: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


def _resolve_paper_reference_fallback(message: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    selected_paper = context.get("selected_paper")
    if isinstance(selected_paper, Mapping):
        return dict(selected_paper)
    for key in ("papers", "last_papers"):
        papers = context.get(key)
        if isinstance(papers, list) and papers and isinstance(papers[0], Mapping):
            return dict(papers[0])
    arxiv_id_match = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", message or "")
    if arxiv_id_match:
        return {"arxiv_id": arxiv_id_match.group(0), "query": message}
    return {"query": message}


class ResolvePaperAdapter(BaseToolAdapter[ResolvePaperInput, PaperReferenceOutput]):
    tool_name = "resolve_paper"
    input_model = ResolvePaperInput
    output_model = PaperReferenceOutput

    def _run(self, tool_input: ResolvePaperInput) -> PaperReferenceOutput:
        message = str(tool_input.message or "").strip()
        context = tool_input.context if isinstance(tool_input.context, Mapping) else {}
        selected_paper = tool_input.selected_paper
        if callable(_resolve_paper_reference):
            resolution = _resolve_paper_reference(message, context)
            if isinstance(resolution, Mapping):
                return PaperReferenceOutput.model_validate(dict(resolution))
        if isinstance(selected_paper, Mapping):
            return PaperReferenceOutput.model_validate(dict(selected_paper))
        return PaperReferenceOutput.model_validate(_resolve_paper_reference_fallback(message, context))


class CheckPaperIndexAdapter(BaseToolAdapter[CheckPaperIndexInput, PaperIndexStatusOutput]):
    tool_name = "check_paper_index"
    input_model = CheckPaperIndexInput
    output_model = PaperIndexStatusOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def _run(self, tool_input: CheckPaperIndexInput) -> PaperIndexStatusOutput:
        if not tool_input.arxiv_id:
            return PaperIndexStatusOutput(status="missing", has_index=False)
        started = perf_counter()
        logger.info("arxiv_agent paper qa index check started: arxiv_id=%s", tool_input.arxiv_id)
        tool_result = self.invoke_backend_tool("check_paper_qa_index", arxiv_id=tool_input.arxiv_id)
        logger.info(
            "arxiv_agent paper qa index check finished: arxiv_id=%s ok=%s elapsed_ms=%.1f",
            tool_input.arxiv_id,
            (tool_result or {}).get("ok") if isinstance(tool_result, Mapping) else None,
            (perf_counter() - started) * 1000,
        )
        data = (tool_result or {}).get("data") if isinstance(tool_result, Mapping) else {}
        return PaperIndexStatusOutput.model_validate(dict(data or {"status": "unknown", "has_index": False, "tool_result": tool_result}))


class ParseAndIndexPaperAdapter(BaseToolAdapter[ParseAndIndexPaperInput, IndexBuildOutput]):
    tool_name = "parse_and_index_paper"
    input_model = ParseAndIndexPaperInput
    output_model = IndexBuildOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def execute(self, tool_input: ParseAndIndexPaperInput) -> ToolExecutionResult:
        result = super().execute(tool_input)
        if result.ok and isinstance(result.data, IndexBuildOutput) and result.data.tool_result:
            backend_result = result.data.tool_result
            if not bool(backend_result.get("ok", False)):
                return result.model_copy(
                    update={
                        "ok": False,
                        "error": backend_tool_error(
                            error_code=((backend_result.get("error") or {}).get("code") if isinstance(backend_result.get("error"), Mapping) else None) or "index_build_failed",
                            message=str((backend_result.get("error") or {}).get("message") if isinstance(backend_result.get("error"), Mapping) else backend_result.get("summary") or "论文索引构建失败"),
                            detail={"backend_error": backend_result.get("error")},
                            retryable=False,
                            suggested_recovery="request_confirmation",
                            safe_debug={"backend_tool_name": "build_paper_qa_index"},
                        ),
                    }
                )
        return result

    def _run(self, tool_input: ParseAndIndexPaperInput) -> IndexBuildOutput:
        # 解析和索引构建通常耗时最长，adapter 层日志能直接证明后端工具是否已经真正开始执行。
        started = perf_counter()
        logger.info("arxiv_agent paper qa index build started: arxiv_id=%s", tool_input.arxiv_id)
        tool_result = self.invoke_backend_tool("build_paper_qa_index", arxiv_id=tool_input.arxiv_id)
        logger.info(
            "arxiv_agent paper qa index build finished: arxiv_id=%s ok=%s elapsed_ms=%.1f",
            tool_input.arxiv_id,
            (tool_result or {}).get("ok") if isinstance(tool_result, Mapping) else None,
            (perf_counter() - started) * 1000,
        )
        data = dict((tool_result or {}).get("data") or {})
        data.setdefault("tool_result", tool_result)
        return IndexBuildOutput.model_validate(data or {"tool_result": tool_result})


class AnswerPaperQuestionAdapter(BaseToolAdapter[AnswerPaperQuestionInput, PaperQAAnswerOutput]):
    tool_name = "answer_paper_question"
    input_model = AnswerPaperQuestionInput
    output_model = PaperQAAnswerOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def execute(self, tool_input: AnswerPaperQuestionInput) -> ToolExecutionResult:
        if not tool_input.arxiv_id:
            return self._error_result(
                started=__import__("time").perf_counter(),
                error_code="missing_arxiv_id",
                message="论文 QA 缺少 arxiv_id",
                detail={"paper_ref": tool_input.paper_ref or tool_input.paper_reference},
                failed_stage="input_validation",
                recoverable=True,
                retryable=False,
                suggested_recovery="ask_clarification",
            )
        return super().execute(tool_input)

    def _run(self, tool_input: AnswerPaperQuestionInput) -> PaperQAAnswerOutput:
        tool_kwargs: Dict[str, Any] = {"arxiv_id": tool_input.arxiv_id, "question": tool_input.resolved_question}
        # recovery 策略只以结构化 payload 传递给 PaperQAService，避免在 adapter 里散落临时参数。
        if tool_input.qa_recovery_strategy:
            tool_kwargs["qa_recovery_strategy"] = dict(tool_input.qa_recovery_strategy)
        if tool_input.index_strategy:
            tool_kwargs["index_strategy"] = dict(tool_input.index_strategy)
        tool_result = self.invoke_backend_tool("answer_paper_question", **tool_kwargs)
        tool_data = dict((tool_result or {}).get("data") or {})
        tool_data.setdefault("status", "success" if bool((tool_result or {}).get("ok", False)) and str(tool_data.get("answer") or "").strip() else "failed")
        tool_data.setdefault("sources", [])
        tool_data.setdefault("retrieval_debug", None)
        if not bool((tool_result or {}).get("ok", False)):
            error_payload = (tool_result or {}).get("error") if isinstance((tool_result or {}).get("error"), Mapping) else {}
            tool_data.setdefault("error", error_payload.get("message") or (tool_result or {}).get("summary") or "answer_paper_question failed")
        else:
            tool_data.setdefault("error", None)
        tool_data.setdefault("arxiv_id", tool_input.arxiv_id)
        tool_data.setdefault("question", tool_input.resolved_question)
        tool_data["tool_result"] = tool_result
        return PaperQAAnswerOutput.model_validate(tool_data)
