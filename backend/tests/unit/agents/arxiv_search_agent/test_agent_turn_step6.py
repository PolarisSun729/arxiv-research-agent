from __future__ import annotations

import importlib
import json

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

executor_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_executor")
planner_registry_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
run_agent_turn = executor_module.run_agent_turn
run_agent_turn_in_graph = executor_module.run_agent_turn_in_graph
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
    step_ids = _step_ids(result)
    assert step_ids[:4] == ["normalize_request", "build_arxiv_search_spec", "search_arxiv", "validate_arxiv_results"]
    assert step_ids[-1] == "synthesize_arxiv_response"
    assert "personalize_paper_results" not in step_ids or step_ids.index("personalize_paper_results") < step_ids.index("synthesize_arxiv_response")
    assert {"search_spec", "arxiv_results", "final_answer"}.issubset(result.outputs.keys())


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
            return {"ok": True, "tool_name": tool_name, "summary": "answered", "data": {"answer": "grounded answer", "sources": [{"chunk_id": "c1"}], "retrieval_debug": {"route": "hybrid"}} , "trace": {}, "error": None}
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
    assert {"paper_ref", "paper_qa_result"}.issubset(result.outputs.keys())
    assert result.outputs["paper_qa_result"]["retrieval_debug"] == {"route": "hybrid"}
    assert not {"retrieved_chunks", "reranked_chunks", "draft_answer"}.intersection(result.outputs.keys())
    assert not any(trace.step_id == "request_confirmation" for trace in result.trace)


def test_run_agent_turn_paper_qa_ordinal_uses_last_papers_over_selected(monkeypatch) -> None:
    answer_calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "check_paper_qa_index":
            assert kwargs["arxiv_id"] == "2401.00002"
            return {"ok": True, "tool_name": tool_name, "summary": "available", "data": {"status": "available", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            answer_calls.append(dict(kwargs))
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "answered",
                "data": {"answer": "second paper answer", "sources": [{"chunk_id": "c2"}], "retrieval_debug": {}},
                "trace": {},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="这第2篇论文的方法是什么？",
            context={
                # 前端会默认把第一篇作为 selected_paper；序号引用必须以 last_papers 为准。
                "selected_paper": {"arxiv_id": "2401.00001", "title": "First Paper"},
                "last_papers": [
                    {"arxiv_id": "2401.00001", "title": "First Paper"},
                    {"arxiv_id": "2401.00002", "title": "Second Paper"},
                ],
            },
        )
    )

    assert result.status == "success"
    assert result.outputs["paper_ref"]["arxiv_id"] == "2401.00002"
    assert result.outputs["paper_ref"]["title"] == "Second Paper"
    assert answer_calls == [{"arxiv_id": "2401.00002", "question": "这第2篇论文的方法是什么？"}]


def test_run_agent_turn_preference_action_ordinal_uses_last_papers_over_selected(monkeypatch) -> None:
    preference_calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "record_paper_preference":
            preference_calls.append(dict(kwargs))
            return {"ok": True, "tool_name": tool_name, "summary": "recorded", "data": {"ok": True}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="preference_action",
            message="喜欢这第2篇论文",
            context={
                # 偏好写入是持久化动作，序号解析必须使用最近列表而不是默认选中第一篇。
                "selected_paper": {"arxiv_id": "2401.00001", "title": "First Paper"},
                "last_papers": [
                    {"arxiv_id": "2401.00001", "title": "First Paper"},
                    {"arxiv_id": "2401.00002", "title": "Second Paper"},
                ],
            },
        )
    )

    assert result.status == "waiting_confirmation"
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_name == "update_preference_store"
    assert result.pending_confirmation.target_paper is not None
    assert result.pending_confirmation.target_paper["arxiv_id"] == "2401.00002"
    assert result.outputs["paper_reference"]["arxiv_id"] == "2401.00002"
    assert preference_calls == []


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
    assert result.pending_confirmation.step_id == "parse_and_index_paper"
    assert [item.code for item in result.pending_confirmation.allowed_decisions] == ["approve", "reject"]
    json.dumps(result.pending_confirmation.model_dump(), ensure_ascii=False)
    assert any(trace.step_id == "parse_and_index_paper" and trace.event == "confirmation_requested" for trace in result.trace)
    assert "build_paper_qa_index" not in called_tools


def test_run_agent_turn_in_graph_reject_skips_index_build(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "missing", "data": {"status": "missing", "has_index": False}, "trace": {}, "error": None}
        raise AssertionError(f"{tool_name} should not run before confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"decision": "reject"})

    result = run_agent_turn_in_graph(
        AgentState(
            intent="paper_qa",
            message="杩欑瘒璁烘枃鐨勬柟娉曟槸浠€涔堬紵",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Method"}},
        )
    )

    assert result.final_answer == "已取消解析 RAG Method，因此无法继续基于全文回答。"
    assert any(trace.event == "confirmation_requested" for trace in result.trace)
    assert any(trace.event == "confirmation_rejected" for trace in result.trace)


def test_run_agent_turn_paper_qa_trace_only_real_answer_tool(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "available", "data": {"status": "available", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            return {"ok": True, "tool_name": tool_name, "summary": "answered", "data": {"answer": "grounded answer", "sources": [{"chunk_id": "c2"}], "retrieval_debug": {"stages": ["real_rag"]}}, "trace": {}, "error": None}
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="method question",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "method question"}},
        )
    )

    assert result.status == "success"
    assert result.outputs["paper_qa_result"]["retrieval_debug"] == {"stages": ["real_rag"]}
    pseudo_steps = {"retrieve_paper_chunks", "rewrite_paper_query", "rerank_paper_chunks", "validate_qa_evidence", "generate_paper_answer", "verify_answer_grounding"}
    assert not pseudo_steps.intersection({trace.step_id for trace in result.trace})
    assert [step.tool_name for step in result.plan.steps] == [
        "resolve_paper",
        "check_paper_index",
        "answer_paper_question",
        "assess_paper_qa_quality",
    ]


def test_run_agent_turn_preference_action_persistent_write(monkeypatch) -> None:
    preference_calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "record_paper_preference"
        preference_calls.append(dict(kwargs))
        return {"ok": True, "tool_name": tool_name, "summary": "recorded", "data": {"ok": True}, "trace": {}, "error": None}

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"decision": "approve"})

    result = run_agent_turn_in_graph(
        AgentState(
            intent="preference_action",
            message="喜欢这篇论文",
            context={"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG"}},
        )
    )

    side_effects = {step.tool_name: step.side_effect_level for step in result.plan.steps}
    assert side_effects["update_preference_store"] == "persistent_write"
    assert not any("interest" in tool_name or "profile" in tool_name for tool_name in side_effects)
    assert result.status == "success"
    assert result.final_answer
    assert preference_calls == [
        {
            "user_id": "",
            "arxiv_id": "2401.00001",
            "liked": True,
            "paper": {"arxiv_id": "2401.00001", "title": "RAG", "query": None, "matched_by": None, "source": None},
        }
    ]
    assert any(trace.event == "confirmation_approved" and trace.step_id == "update_preference_store" for trace in result.trace)
    assert any(trace.step_id == "update_preference_store" and trace.event == "step_succeeded" for trace in result.trace)
    assert not any("interest" in output_key or "profile" in output_key for output_key in result.outputs)


def test_run_agent_turn_unclear_only_clarifies(monkeypatch) -> None:
    def fail_business_tool(*args, **kwargs):
        raise AssertionError("unclear plan must not call business tools")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_business_tool)

    result = run_agent_turn(AgentState(intent="unclear", message="帮我找那个"))

    assert result.status == "need_clarification"
    assert result.final_answer
    assert _step_ids(result) == ["analyze_ambiguity", "generate_clarification"]
    forbidden = {"search_arxiv", "answer_paper_question", "generate_recommendations", "update_preference_store"}
    assert not forbidden.intersection({step.tool_name for step in result.plan.steps})


def test_run_agent_turn_unsupported_only_fallback(monkeypatch) -> None:
    def fail_business_tool(*args, **kwargs):
        raise AssertionError("unsupported plan must not call business tools")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_business_tool)

    result = run_agent_turn(AgentState(intent="unsupported", message="帮我画海报"))

    assert result.status == "fallback"
    assert result.final_answer
    assert [step.tool_name for step in result.plan.steps] == ["generate_fallback_response"]
