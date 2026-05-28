from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict


def _lazy_tool(module_path: str, function_name: str) -> Callable[..., Dict[str, Any]]:
    def _invoke(**kwargs: Any) -> Dict[str, Any]:
        module = import_module(module_path)
        tool = getattr(module, function_name)
        return tool(**kwargs)

    return _invoke


TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "search_arxiv_raw": _lazy_tool("tools.arxiv_tools", "search_arxiv_raw"),
    "search_arxiv_structured": _lazy_tool("tools.arxiv_tools", "search_arxiv_structured"),
    "get_paper_metadata": _lazy_tool("tools.arxiv_tools", "get_paper_metadata"),
    "recommend_papers": _lazy_tool("tools.recommendation_tools", "recommend_papers"),
    "record_paper_preference": _lazy_tool("tools.recommendation_tools", "record_paper_preference"),
    "check_paper_qa_index": _lazy_tool("tools.paper_qa_tools", "check_paper_qa_index"),
    "build_paper_qa_index": _lazy_tool("tools.paper_qa_tools", "build_paper_qa_index"),
    "answer_paper_question": _lazy_tool("tools.paper_qa_tools", "answer_paper_question"),
}


def get_tool_registry() -> Dict[str, Callable[..., Dict[str, Any]]]:
    return dict(TOOL_REGISTRY)


def get_tool_names() -> list[str]:
    return list(TOOL_REGISTRY.keys())


def invoke_tool(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        raise KeyError(f"Unknown tool: {tool_name}")
    return tool(**kwargs)
