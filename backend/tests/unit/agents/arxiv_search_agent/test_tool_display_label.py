from __future__ import annotations

import dataclasses
import importlib

import pytest

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
service_module = _MODULES["service_module"]
state_module = _MODULES["state_module"]
schemas = _MODULES["schemas"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
tool_registry = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")


def _search_state():
    state = state_module.AgentState(
        intent="arxiv_search",
        message="rag",
        search_spec=schemas.ArxivSearchSpec(intent="arxiv_search", query="RAG"),
    )
    _, plan, _ = planner_module.build_executable_plan(state, enable_tool_aware_planner=True, enable_llm_plan_draft=False)
    state.execution_plan = plan
    state.runtime_state = schemas.AgentRuntimeState(plan=plan, current_step_id="search_arxiv")
    return state


def test_every_registered_tool_has_display_label() -> None:
    registry = tool_registry.UNIFIED_TOOL_REGISTRY
    labels = {spec.tool_name: registry.get_contract(spec.tool_name).display_label for spec in registry.list_tools()}

    assert labels and all(str(label).strip() for label in labels.values()), labels
    assert labels["search_arxiv"] == "检索 arXiv 论文"


def test_registry_rejects_empty_display_label() -> None:
    search = tool_registry.UNIFIED_TOOL_REGISTRY.get_contract("search_arxiv")

    with pytest.raises(ValueError, match="display_label"):
        tool_registry.ToolRegistry().register(dataclasses.replace(search, display_label=" "))


def test_tool_call_events_and_plan_projection_carry_display_label() -> None:
    state = _search_state()

    start = service_module._active_runtime_tool_call(state)
    assert start["tool_name"] == "search_arxiv"
    assert start["display_label"] == "检索 arXiv 论文"

    end = service_module._merge_tool_call_end(start, {"tool_name": "search_arxiv", "status": "success"})
    assert end["display_label"] == "检索 arXiv 论文"

    steps = service_module._compact_state(state)["execution_plan"]
    assert steps and all(step["display_label"] for step in steps)
    by_tool = {step["tool_name"]: step["display_label"] for step in steps}
    assert by_tool["synthesize_arxiv_response"] == "撰写检索结果摘要"


def test_unregistered_tool_has_no_display_label() -> None:
    assert service_module._tool_display_label("not_a_tool") is None
    assert "display_label" not in service_module._merge_tool_call_end({"tool_name": "not_a_tool"}, None)
