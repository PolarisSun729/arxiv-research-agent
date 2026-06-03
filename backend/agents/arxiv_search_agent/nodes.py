from __future__ import annotations

# Compatibility facade for graph.py and other callers.
#
# This module intentionally stays thin: concrete business logic now lives in
# node_modules/*, while this file keeps the historical import surface stable.

from typing import Any, Mapping, Optional, Union

from .node_modules.intent_support import (
    HARD_RULE_PATTERNS,
    LLM_CONFIDENCE_THRESHOLD,
    SEARCH_TRIGGER_PATTERNS,
    SUPPORTED_INTENTS,
    _build_intent_guidance,
    _build_llm_prompt_with_profile,
    _contains_any_term,
    _dedupe_preserve_order,
    _extract_json_block,
    _looks_like_paper_detail_request,
    _looks_like_paper_qa_request,
    _looks_like_paper_summary_request,
    _looks_like_preference_action_request,
    _looks_like_reading_list_action_request,
    _looks_like_recommendation_request,
    _looks_search_like,
    _matches_any,
    _references_specific_paper,
    _validation_error_summary,
)
from .node_modules.search_node import SEARCH_TOOL_NAME
from .state import AgentState


def parse_search_request(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    from .node_modules.parse_node import parse_search_request as _impl

    return _impl(state, generation_service=generation_service)


def build_search_tool_args(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.search_node import build_search_tool_args as _impl

    return _impl(state)


def invoke_search_tool(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.search_node import invoke_search_tool as _impl

    return _impl(state)


def check_search_result(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.search_node import check_search_result as _impl

    return _impl(state)


def relax_search_for_retry(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.search_node import relax_search_for_retry as _impl

    return _impl(state)


def personalized_rank_and_annotate_papers(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.search_node import personalized_rank_and_annotate_papers as _impl

    return _impl(state)


def apply_preference_action(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.preference_node import apply_preference_action as _impl

    return _impl(state)


def classify_pending_action_confirmation(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    from .node_modules.pending_action_node import classify_pending_action_confirmation as _impl

    return _impl(state, generation_service=generation_service)


def handle_pending_action_confirmation(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.pending_action_node import handle_pending_action_confirmation as _impl

    return _impl(state)


def handle_paper_reading_request(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.paper_reading_node import handle_paper_reading_request as _impl

    return _impl(state)


def synthesize_response(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    from .node_modules.response_node import synthesize_response as _impl

    return _impl(state)


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
    "relax_search_for_retry",
    "synthesize_response",
]
