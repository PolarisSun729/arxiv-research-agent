from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

executor_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_executor")
planner_registry_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
run_agent_turn = executor_module.run_agent_turn
PLANNER_TOOL_REGISTRY = planner_registry_module.PLANNER_TOOL_REGISTRY


def _step_ids(result):
    return [trace.step_id for trace in result.trace if trace.event == "step_succeeded"]


def test_run_agent_turn_arxiv_search_success(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        return {
            "ok": True,
            "tool_name": tool_name,
            "summary": "searched",
            "data": {"papers": [{"arxiv_id": "2401.00001", "title": "RAG Agents"}]},
            "trace": {"tool_name": tool_name},
            "error": None,
        }

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="arxiv_search",
            message="搜索 RAG agent 论文",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG agent", max_results=5),
        )
    )

    assert result.status == "success"
    assert result.final_answer
    assert _step_ids(result) == [
        "normalize_request",
        "build_arxiv_search_spec",
        "search_arxiv",
        "validate_arxiv_results",
        "personalize_paper_results",
        "synthesize_arxiv_response",
    ]
    assert {"search_spec", "arxiv_results", "ranked_papers"}.issubset(result.outputs.keys())


def test_run_agent_turn_arxiv_empty_result_replans(monkeypatch) -> None:
    calls = {"search": 0}

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_structured"
        calls["search"] += 1
        papers = [] if calls["search"] == 1 else [{"arxiv_id": "2401.00001", "title": "RAG retry"}]
        return {"ok": True, "tool_name": tool_name, "summary": "searched", "data": {"papers": papers}, "trace": {}, "error": None}

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="arxiv_search",
            message="搜索 RAG",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", max_results=5),
        )
    )

    assert result.status == "success"
    assert calls["search"] == 2
    assert any(trace.event == "plan_replanned" for trace in result.trace)
    assert any(trace.step_id == "rewrite_arxiv_query" and trace.event == "step_succeeded" for trace in result.trace)
    assert result.runtime is not None
    assert sum(result.runtime.replan_counts.values()) <= 5


def test_run_agent_turn_paper_qa_available_index(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "available", "data": {"status": "available", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            return {"ok": True, "tool_name": tool_name, "summary": "answered", "data": {"answer": "grounded answer", "sources": [{"chunk_id": "c1"}]}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="这篇论文的方法是什么？",
            context={
                "selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Method"},
                "paper_chunks": [{"chunk_id": "c1", "text": "method evidence", "score": 1.0}],
            },
        )
    )

    assert result.status == "success"
    assert result.final_answer == "grounded answer"
    assert {"paper_ref", "retrieved_chunks", "reranked_chunks", "draft_answer"}.issubset(result.outputs.keys())
    assert not any(trace.step_id == "request_confirmation" for trace in result.trace)


def test_run_agent_turn_paper_qa_missing_index_waits_for_confirmation(monkeypatch) -> None:
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append(tool_name)
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "missing", "data": {"status": "missing", "has_index": False}, "trace": {}, "error": None}
        raise AssertionError(f"{tool_name} should not run before confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="这篇论文的方法是什么？",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Method"}},
        )
    )

    assert result.status == "waiting_confirmation"
    assert result.pending_confirmation
    assert any(trace.step_id == "request_confirmation" for trace in result.trace)
    assert "build_paper_qa_index" not in called_tools


def test_run_agent_turn_paper_qa_low_evidence_replans(monkeypatch) -> None:
    retrieve_calls = {"count": 0}

    def fake_retrieve(self, resolved_input, state, runtime, step):
        retrieve_calls["count"] += 1
        if retrieve_calls["count"] == 1:
            return [{"chunk_id": "c1", "text": "irrelevant", "score": 0.05}]
        return [{"chunk_id": "c2", "text": "method question evidence", "score": 1.0}]

    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "available", "data": {"status": "available", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            return {"ok": True, "tool_name": tool_name, "summary": "answered", "data": {"answer": "grounded answer", "sources": [{"chunk_id": "c2"}]}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module.PlanExecutor, "_retrieve_paper_chunks", fake_retrieve)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="method question",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "method question"}},
        )
    )

    assert result.status == "success"
    assert any(trace.step_id == "rewrite_paper_query" for trace in result.trace)
    assert retrieve_calls["count"] >= 2
    assert result.runtime is not None
    assert sum(result.runtime.replan_counts.values()) <= 5


def test_run_agent_turn_preference_action_persistent_write(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "record_paper_preference"
        return {"ok": True, "tool_name": tool_name, "summary": "recorded", "data": {"ok": True}, "trace": {}, "error": None}

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="preference_action",
            message="喜欢这篇论文",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
        )
    )

    side_effects = {step.tool_name: step.side_effect_level for step in result.plan.steps}
    assert side_effects["update_preference_store"] == "persistent_write"
    assert side_effects["update_interest_profile"] == "persistent_write"
    assert result.status == "success"
    assert result.final_answer
    assert any(trace.step_id == "update_preference_store" and trace.event == "step_succeeded" for trace in result.trace)


def test_run_agent_turn_unclear_only_clarifies(monkeypatch) -> None:
    def fail_business_tool(*args, **kwargs):
        raise AssertionError("unclear plan must not call business tools")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_business_tool)

    result = run_agent_turn(AgentState(intent="unclear", message="帮我找那个"))

    assert result.status == "need_clarification"
    assert result.final_answer
    assert _step_ids(result) == ["analyze_ambiguity", "generate_clarification"]
    forbidden = {"search_arxiv", "retrieve_paper_chunks", "generate_recommendations", "update_preference_store"}
    assert not forbidden.intersection({step.tool_name for step in result.plan.steps})


def test_run_agent_turn_unsupported_only_fallback(monkeypatch) -> None:
    def fail_business_tool(*args, **kwargs):
        raise AssertionError("unsupported plan must not call business tools")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_business_tool)

    result = run_agent_turn(AgentState(intent="unsupported", message="帮我画海报"))

    assert result.status == "fallback"
    assert result.final_answer
    assert [step.tool_name for step in result.plan.steps] == ["generate_fallback_response"]
