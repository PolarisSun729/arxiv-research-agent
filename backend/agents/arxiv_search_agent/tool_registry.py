"""Planner 层工具注册表。

这层注册表只负责回答两个问题：
1. 当前 agent planner 被允许选择哪些工具；
2. 每个工具的能力标签、输入输出、确认需求和副作用约束是什么。

它不直接执行工具，真实执行仍由底层 tool registry 或节点逻辑负责。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .schemas import ToolSpec


class ToolRegistry:
    """集中维护 planner 可见的工具集合。"""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool_spec: ToolSpec) -> None:
        self._tools[tool_spec.tool_name] = tool_spec

    def get(self, tool_name: str) -> Optional[ToolSpec]:
        return self._tools.get(str(tool_name or "").strip())

    def exists(self, tool_name: str) -> bool:
        return self.get(tool_name) is not None

    def validate_tool_name(self, tool_name: str) -> str:
        normalized_tool_name = str(tool_name or "").strip()
        if not normalized_tool_name or normalized_tool_name not in self._tools:
            raise ValueError(f"Unknown planner tool: {tool_name}")
        return normalized_tool_name

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def list_by_capability(self, tag: str) -> List[ToolSpec]:
        normalized_tag = str(tag or "").strip()
        if not normalized_tag:
            return []
        return [tool for tool in self._tools.values() if normalized_tag in list(tool.capability_tags or [])]


def _tool(
    tool_name: str,
    *,
    capability_tags: List[str],
    input_schema: Optional[Dict[str, object]] = None,
    output_schema: Optional[Dict[str, object]] = None,
    side_effect_level: str = "none",
    requires_confirmation: bool = False,
    can_retry: bool = False,
    failure_modes: Optional[List[str]] = None,
    implementation: Optional[str] = None,
) -> ToolSpec:
    return ToolSpec(
        tool_name=tool_name,
        capability_tags=list(capability_tags or []),
        input_schema=dict(input_schema or {}),
        output_schema=dict(output_schema or {}),
        side_effect_level=side_effect_level,  # type: ignore[arg-type]
        requires_confirmation=requires_confirmation,
        can_retry=can_retry,
        failure_modes=list(failure_modes or []),
        implementation=implementation,
    )


PLANNER_TOOL_REGISTRY = ToolRegistry()


for tool_spec in [
    _tool("normalize_request", capability_tags=["search", "clarify"], input_schema={"message": "str"}, output_schema={"normalized_request": "dict"}, side_effect_level="session_write", implementation="plan_node.normalize_request"),
    _tool("build_arxiv_search_spec", capability_tags=["search"], input_schema={"normalized_request": "dict", "search_spec": "ArxivSearchSpec"}, output_schema={"arxiv_search_spec": "dict"}, side_effect_level="session_write", implementation="plan_node.build_arxiv_search_spec"),
    _tool("search_arxiv", capability_tags=["search"], input_schema={"search_spec": "ArxivSearchSpec"}, output_schema={"papers": "list", "tool_result": "dict"}, side_effect_level="external_call", can_retry=True, failure_modes=["tool_argument_validation_failed", "tool_execution_failed", "empty_results"], implementation="tools.search_arxiv_structured"),
    _tool("validate_arxiv_results", capability_tags=["search", "validate"], input_schema={"search_result": "dict"}, output_schema={"validated_search_result": "dict"}, implementation="search_node.check_search_result"),
    _tool("rewrite_arxiv_query", capability_tags=["search", "rewrite"], input_schema={"search_spec": "ArxivSearchSpec"}, output_schema={"rewritten_search_spec": "dict"}, side_effect_level="session_write", can_retry=True, implementation="search_node.relax_search_for_retry"),
    _tool("personalize_paper_results", capability_tags=["rerank", "personalize"], input_schema={"papers": "list", "user_memory_summary": "dict"}, output_schema={"ranked_papers": "list"}, implementation="search_node.personalized_rank_and_annotate_papers"),
    _tool("synthesize_arxiv_response", capability_tags=["answer"], input_schema={"ranked_papers": "list", "warnings": "list"}, output_schema={"final_answer": "str"}, implementation="response_node.synthesize_response"),
    _tool("resolve_paper", capability_tags=["retrieve"], input_schema={"message": "str", "context": "dict"}, output_schema={"paper_reference": "dict"}, implementation="paper_reading_node.resolve_paper_reference"),
    _tool("check_paper_index", capability_tags=["retrieve", "validate"], input_schema={"paper_reference": "dict"}, output_schema={"qa_index_status": "dict"}, implementation="tools.check_paper_qa_index"),
    _tool("request_confirmation", capability_tags=["clarify", "confirm"], input_schema={"pending_action": "dict"}, output_schema={"confirmation_status": "dict"}, side_effect_level="session_write", implementation="planner.confirmation_request"),
    _tool("parse_and_index_paper", capability_tags=["retrieve", "index"], input_schema={"paper_reference": "dict"}, output_schema={"index_build_result": "dict"}, side_effect_level="external_call", requires_confirmation=True, failure_modes=["paper_not_found", "index_build_failed"], implementation="tools.build_paper_qa_index"),
    _tool("answer_paper_question", capability_tags=["answer"], input_schema={"paper_reference": "dict", "question": "str"}, output_schema={"paper_qa_result": "dict"}, side_effect_level="external_call", implementation="tools.answer_paper_question"),
    _tool("load_user_profile", capability_tags=["retrieve", "profile"], input_schema={"context": "dict"}, output_schema={"recommendation_profile": "dict"}, implementation="recommendation_node.load_user_profile"),
    _tool("load_candidate_papers", capability_tags=["retrieve", "recommendation"], input_schema={"recommendation_profile": "dict"}, output_schema={"candidate_papers": "list"}, implementation="recommendation_node.load_candidate_papers"),
    _tool("generate_recommendations", capability_tags=["recommendation"], input_schema={"recommendation_profile": "dict", "candidate_papers": "list"}, output_schema={"recommendation_result": "dict"}, side_effect_level="external_call", implementation="tools.recommend_papers"),
    _tool("validate_recommendations", capability_tags=["validate", "recommendation"], input_schema={"recommendation_result": "dict"}, output_schema={"validated_recommendations": "dict"}, implementation="recommendation_node.validate_recommendations"),
    _tool("explain_recommendations", capability_tags=["answer", "recommendation"], input_schema={"validated_recommendations": "dict"}, output_schema={"final_answer": "str"}, implementation="recommendation_node.explain_recommendations"),
    _tool("resolve_preference_target", capability_tags=["retrieve", "preference"], input_schema={"message": "str"}, output_schema={"paper_reference": "dict"}, implementation="preference_node.resolve_target"),
    _tool("update_preference_store", capability_tags=["memory_write", "preference"], input_schema={"paper_reference": "dict", "message": "str"}, output_schema={"preference_action_result": "dict"}, side_effect_level="persistent_write", implementation="tools.record_paper_preference"),
    _tool("verify_preference_update", capability_tags=["validate", "preference"], input_schema={"preference_action_result": "dict"}, output_schema={"verified_preference_update": "dict"}, implementation="preference_node.verify_preference_update"),
    _tool("synthesize_preference_response", capability_tags=["answer", "preference"], input_schema={"verified_preference_update": "dict"}, output_schema={"final_answer": "str"}, implementation="response_node.synthesize_preference_response"),
    _tool("analyze_ambiguity", capability_tags=["clarify"], input_schema={"message": "str"}, output_schema={"missing_information": "dict"}, implementation="plan_node.analyze_ambiguity"),
    _tool("generate_clarification", capability_tags=["clarify", "answer"], input_schema={"missing_information": "dict"}, output_schema={"final_answer": "str"}, implementation="response_node.generate_clarification"),
    _tool("generate_fallback_response", capability_tags=["fallback", "answer"], input_schema={"message": "str"}, output_schema={"final_answer": "str"}, implementation="response_node.generate_fallback_response"),
]:
    PLANNER_TOOL_REGISTRY.register(tool_spec)


__all__ = ["PLANNER_TOOL_REGISTRY", "ToolRegistry"]
