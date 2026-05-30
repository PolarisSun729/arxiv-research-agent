from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Type

from pydantic import BaseModel, ValidationError

from . import arxiv_tools, paper_qa_tools, recommendation_tools
from .schemas import (
    AnswerPaperQuestionInput,
    BuildPaperQAIndexInput,
    CheckPaperQAIndexInput,
    GetPaperMetadataInput,
    RecommendPapersInput,
    RecordPaperPreferenceInput,
    SearchArxivRawInput,
    SearchArxivStructuredInput,
)
from .tool_result import make_tool_error, make_tool_result


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Type[BaseModel]
    func: Callable[..., Dict[str, Any]]


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "search_arxiv_raw": ToolSpec(
        name="search_arxiv_raw",
        description="Search arXiv using a raw query string.",
        input_schema=SearchArxivRawInput,
        func=arxiv_tools.search_arxiv_raw,
    ),
    "search_arxiv_structured": ToolSpec(
        name="search_arxiv_structured",
        description="Search arXiv using structured fields.",
        input_schema=SearchArxivStructuredInput,
        func=arxiv_tools.search_arxiv_structured,
    ),
    "get_paper_metadata": ToolSpec(
        name="get_paper_metadata",
        description="Get paper metadata by arXiv ID.",
        input_schema=GetPaperMetadataInput,
        func=arxiv_tools.get_paper_metadata,
    ),
    "recommend_papers": ToolSpec(
        name="recommend_papers",
        description="Recommend papers for a user.",
        input_schema=RecommendPapersInput,
        func=recommendation_tools.recommend_papers,
    ),
    "record_paper_preference": ToolSpec(
        name="record_paper_preference",
        description="Record a user preference for a paper.",
        input_schema=RecordPaperPreferenceInput,
        func=recommendation_tools.record_paper_preference,
    ),
    "check_paper_qa_index": ToolSpec(
        name="check_paper_qa_index",
        description="Check whether a paper has a QA index.",
        input_schema=CheckPaperQAIndexInput,
        func=paper_qa_tools.check_paper_qa_index,
    ),
    "build_paper_qa_index": ToolSpec(
        name="build_paper_qa_index",
        description="Build a QA index for a paper.",
        input_schema=BuildPaperQAIndexInput,
        func=paper_qa_tools.build_paper_qa_index,
    ),
    "answer_paper_question": ToolSpec(
        name="answer_paper_question",
        description="Answer a question about a paper.",
        input_schema=AnswerPaperQuestionInput,
        func=paper_qa_tools.answer_paper_question,
    ),
}


def _validate_arguments(schema: Type[BaseModel], arguments: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        if hasattr(schema, "model_validate"):
            model = schema.model_validate(dict(arguments))
            return model.model_dump()
        model = schema.parse_obj(dict(arguments))  # type: ignore[attr-defined]
        return model.dict()
    except ValidationError as exc:
        raise exc


def get_tool_registry() -> Dict[str, ToolSpec]:
    return dict(TOOL_REGISTRY)


def get_tool_names() -> list[str]:
    return list(TOOL_REGISTRY.keys())


def invoke_tool(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="Unknown tool",
            data=None,
            trace={},
            error=make_tool_error("tool_not_found", f"Unknown tool: {tool_name}"),
        )

    try:
        validated_arguments = _validate_arguments(spec.input_schema, kwargs)
    except ValidationError as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="Tool argument validation failed",
            data=None,
            trace={"tool_name": tool_name, "validated": False},
            error=make_tool_error("tool_argument_validation_failed", "Tool arguments failed schema validation", exc.errors()),
        )

    try:
        return spec.func(**validated_arguments)
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="Tool execution failed",
            data=None,
            trace={"tool_name": tool_name, "validated": True},
            error=make_tool_error("tool_execution_failed", str(exc)),
        )
