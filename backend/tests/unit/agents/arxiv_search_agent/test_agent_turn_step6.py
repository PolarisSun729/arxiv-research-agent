from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

executor_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_executor")
planner_registry_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")
paper_reference_module = importlib.import_module("backend.agents.arxiv_search_agent.utils.paper_reference")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
run_agent_turn = executor_module.run_agent_turn
run_agent_turn_in_graph = executor_module.run_agent_turn_in_graph
PLANNER_TOOL_REGISTRY = planner_registry_module.PLANNER_TOOL_REGISTRY


def _step_ids(result):
    return [trace.step_id for trace in result.trace if trace.event == "step_succeeded"]


def _understood(context: dict, kind: str, *arxiv_ids: str) -> dict:
    """模拟理解阶段 LLM 的指代判断：按 arXiv ID 选出候选编号，再走正式的落地校验。"""
    candidates = paper_reference_module.build_reference_candidates(context)
    keys = [next(item["key"] for item in candidates if item["record"].get("arxiv_id") == arxiv_id) for arxiv_id in arxiv_ids]
    return paper_reference_module.ground_paper_reference({"kind": kind, "refs": keys}, candidates)


def _reference_hint_from_trace(result, step_id: str) -> dict:
    for trace in result.trace:
        if trace.step_id == step_id and trace.event == "step_observed":
            evidence = trace.detail.get("evidence") if isinstance(trace.detail, dict) else {}
            hint = (evidence or {}).get("reference_hint") if isinstance(evidence, dict) else None
            return dict(hint or {})
    for trace in result.trace:
        if trace.step_id == step_id and trace.event == "step_succeeded":
            output = trace.detail.get("normalized_output") if isinstance(trace.detail, dict) else {}
            hint = (output or {}).get("reference_hint") if isinstance(output, dict) else None
            return dict(hint or {})
    return {}


def test_run_agent_turn_arxiv_search_success(monkeypatch) -> None:
    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_raw"
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

    # 目标选择已经完成；后续 QA 质量策略可以独立决定降级，不影响交互协议本身。
    assert result.status in {"success", "fallback"}
    assert result.final_answer
    step_ids = _step_ids(result)
    assert step_ids[:3] == ["normalize_request", "build_arxiv_search_spec", "search_arxiv"]
    assert step_ids[-1] == "synthesize_arxiv_response"
    assert "personalize_paper_results" not in step_ids or step_ids.index("personalize_paper_results") < step_ids.index("synthesize_arxiv_response")
    assert {"search_spec", "arxiv_results", "final_answer"}.issubset(result.outputs.keys())


def test_run_agent_turn_arxiv_empty_after_ladder_returns_suggestion_without_replan(monkeypatch) -> None:
    queries = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_raw"
        queries.append(kwargs["search_query"])
        return {"ok": True, "tool_name": tool_name, "summary": "searched", "data": {"papers": [], "total_results": 0}, "trace": {}, "error": None}

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="arxiv_search",
            message="搜索最近 7 天的 RAG 论文",
            search_spec=ArxivSearchSpec(
                intent="arxiv_search",
                concepts=[{"term": "RAG", "aliases": ["retrieval-augmented generation"]}],
                categories=["cs.CL"],
                submitted_days_ago=7,
                constraint_sources={"categories": "default", "submitted_days_ago": "user"},
                max_results=5,
            ),
        )
    )

    # 阶梯在一次工具调用里依次放宽：严格 → 短语改分词 → 放开默认分类；时间窗始终保留。
    assert len(queries) == 3
    assert "cat:cs.CL" in queries[0] and "cat:cs.CL" not in queries[-1]
    # 空结果即 search_arxiv 失败（search_empty）；阶梯已用尽，恢复层直接兜底，不再插入 rewrite→search。
    assert result.status == "fallback"
    replans = [trace for trace in result.trace if trace.event == "plan_replanned"]
    assert [trace.detail["selected_recovery_action"]["action_type"] for trace in replans] == ["fallback_answer"]
    assert not any(trace.step_id == "rewrite_arxiv_query" for trace in result.trace)
    assert result.final_answer
    assert "没有找到符合条件的论文" in result.final_answer
    assert "我没有自动放宽" in result.final_answer
    assert "replan_limit_exceeded" not in result.final_answer


def test_run_agent_turn_arxiv_ladder_relaxes_inside_one_search_step(monkeypatch) -> None:
    queries = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        assert tool_name == "search_arxiv_raw"
        queries.append(kwargs["search_query"])
        papers = [] if len(queries) == 1 else [{"arxiv_id": "2401.00001", "title": "RAG retry"}]
        return {"ok": True, "tool_name": tool_name, "summary": "searched", "data": {"papers": papers, "total_results": len(papers)}, "trace": {}, "error": None}

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="arxiv_search",
            message="搜索 RAG",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
        )
    )

    assert result.status == "success"
    assert len(queries) == 2
    assert not any(trace.event == "plan_replanned" for trace in result.trace)
    papers = result.outputs["arxiv_results"]["papers"]
    assert papers[0]["match_level"] == "drop_inferred_categories"
    assert "放宽" in result.final_answer


def test_run_agent_turn_paper_qa_context_reference_resolves_selected_and_answers(monkeypatch) -> None:
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        if tool_name == "check_paper_qa_index":
            assert kwargs["arxiv_id"] == "2401.00001"
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            assert kwargs["arxiv_id"] == "2401.00001"
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "answered",
                "data": {"answer": "method answer", "sources": [{"chunk_id": "c1"}], "retrieval_debug": {}},
                "trace": {},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="这篇论文的方法是什么？",
            context=(
                context := {
                    "selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Method"},
                    "paper_chunks": [{"chunk_id": "c1", "text": "method evidence", "score": 1.0}],
                }
            ),
            paper_reference=_understood(context, "single", "2401.00001"),
        )
    )

    assert result.status == "success"
    assert result.interaction is None
    assert result.outputs["paper_ref"]["arxiv_id"] == "2401.00001"
    assert result.outputs["paper_ref"]["final_target_resolved"] is True
    assert result.outputs["paper_ref"]["reference_hint"]["reference_type"] == "llm_reference"
    assert result.outputs["paper_ref"]["reference_hint"]["requires_context"] is True
    assert result.outputs["paper_qa_result"]["arxiv_id"] == "2401.00001"
    assert [tool_name for tool_name, _ in called_tools] == ["check_paper_qa_index", "answer_paper_question"]
    assert not {"retrieved_chunks", "reranked_chunks", "draft_answer"}.intersection(result.outputs.keys())
    assert not any(trace.step_id == "request_confirmation" for trace in result.trace)


def test_run_agent_turn_paper_qa_ordinal_resolves_second_paper(monkeypatch) -> None:
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        if tool_name == "check_paper_qa_index":
            assert kwargs["arxiv_id"] == "2401.00002"
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            assert kwargs["arxiv_id"] == "2401.00002"
            assert kwargs["question"] == "这篇论文的方法是什么？"
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
            context=(
                context := {
                    # 前端会默认把第一篇作为 selected_paper；序号指的是列表第二篇，由理解阶段判断。
                    "selected_paper": {"arxiv_id": "2401.00001", "title": "First Paper"},
                    "last_papers": [
                        {"arxiv_id": "2401.00001", "title": "First Paper"},
                        {"arxiv_id": "2401.00002", "title": "Second Paper"},
                    ],
                }
            ),
            paper_reference=_understood(context, "single", "2401.00002"),
        )
    )

    assert result.status == "success"
    assert result.outputs["paper_ref"]["arxiv_id"] == "2401.00002"
    assert result.outputs["paper_ref"]["final_target_resolved"] is True
    assert result.outputs["paper_ref"]["reference_hint"]["reference_type"] == "llm_reference"
    assert [kwargs["arxiv_id"] for _, kwargs in called_tools] == ["2401.00002", "2401.00002"]


def test_run_agent_turn_paper_qa_ordinal_normalizes_question_before_answer(monkeypatch) -> None:
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        if tool_name == "check_paper_qa_index":
            assert kwargs["arxiv_id"] == "2401.00002"
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            assert kwargs["arxiv_id"] == "2401.00002"
            assert kwargs["question"] == "给我讲一下这篇论文的核心内容"
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "answered",
                "data": {"answer": "core content", "sources": [{"chunk_id": "c2"}], "retrieval_debug": {}},
                "trace": {},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="给我讲一下第二篇论文的核心内容",
            context=(
                context := {
                    "last_papers": [
                        {"arxiv_id": "2401.00001", "title": "First Paper"},
                        {"arxiv_id": "2401.00002", "title": "Second Paper"},
                    ],
                }
            ),
            paper_reference=_understood(context, "single", "2401.00002"),
        )
    )

    assert result.status == "success"
    assert result.outputs["paper_ref"]["arxiv_id"] == "2401.00002"
    assert result.outputs["paper_qa_result"]["question"] == "给我讲一下这篇论文的核心内容"


def _two_lists_context() -> dict:
    # 搜索结果和推荐列表各有一个「第二篇」，单看「讲一下第二篇论文」分不清指哪个列表。
    return {
        "last_papers": [
            {"arxiv_id": "2401.00001", "title": "Search First"},
            {"arxiv_id": "2401.00002", "title": "Search Second"},
        ],
        "recommendations": [
            {"arxiv_id": "2501.00001", "title": "Rec First"},
            {"arxiv_id": "2501.00002", "title": "Rec Second"},
        ],
    }


def test_run_agent_turn_paper_qa_ambiguous_reference_answers_with_stated_assumption(monkeypatch) -> None:
    """问答是低成本动作：指代有歧义时按最可能的一篇回答，并在开头说明按哪篇理解。"""
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "indexed", "data": {"status": "indexed", "has_index": True}, "trace": {}, "error": None}
        if tool_name == "answer_paper_question":
            return {
                "ok": True,
                "tool_name": tool_name,
                "summary": "answered",
                "data": {"answer": "search second answer", "sources": [{"chunk_id": "c2"}], "retrieval_debug": {}},
                "trace": {},
                "error": None,
            }
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    context = _two_lists_context()

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="讲一下第二篇论文",
            context=context,
            paper_reference=_understood(context, "ambiguous", "2401.00002", "2501.00002"),
        )
    )

    assert result.status == "success"
    assert result.interaction is None
    assert result.outputs["paper_ref"]["arxiv_id"] == "2401.00002"
    assert result.outputs["paper_ref"]["resolution_reason"] == "llm_reference_ambiguous_assumed"
    assert [kwargs["arxiv_id"] for _, kwargs in called_tools] == ["2401.00002", "2401.00002"]
    assert result.final_answer.startswith("我先按《Search Second》来理解你说的论文")
    assert result.final_answer.endswith("search second answer")


def test_run_agent_turn_in_graph_target_confirmation_uses_user_selected_candidate(monkeypatch) -> None:
    called_tools = []
    interaction_kinds = []

    def fake_interrupt(payload):
        interaction_kinds.append(payload.get("kind"))
        if payload.get("kind") == "target_selection":
            selected = next(
                candidate
                for candidate in payload["payload"]["candidates"]
                if candidate.get("arxiv_id") == "2501.00002"
            )
            return {
                "interaction_id": payload["interaction_id"],
                "decision": "select",
                "selected_candidate": selected,
            }
        # 选完目标后还有偏好写入确认；这里拒绝，只验证目标选择本身。
        return {"interaction_id": payload["interaction_id"], "decision": "reject"}

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "interrupt", fake_interrupt)
    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    context = _two_lists_context()

    result = run_agent_turn_in_graph(
        AgentState(
            intent="preference_action",
            message="喜欢第二篇论文",
            context=context,
            paper_reference=_understood(context, "ambiguous", "2401.00002", "2501.00002"),
        )
    )

    assert result.status in {"success", "fallback"}
    assert result.interaction is None
    assert interaction_kinds[0] == "target_selection"
    assert result.outputs["paper_reference"]["arxiv_id"] == "2501.00002"
    assert result.outputs["paper_reference"]["confirmed_by_user"] is True
    assert result.outputs["paper_reference"]["target_resolution"]["resolution_reason"] == "user_confirmed_target"
    assert called_tools == []


def test_run_agent_turn_in_graph_target_confirmation_reject_cancels_original_action(monkeypatch) -> None:
    called_tools = []

    def fake_interrupt(payload):
        assert payload.get("kind") == "target_selection"
        assert payload.get("interaction_id")
        return {"interaction_id": payload["interaction_id"], "decision": "cancel"}

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        raise AssertionError(f"business tool must not run after target rejection: {tool_name}")

    monkeypatch.setattr(executor_module, "interrupt", fake_interrupt)
    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    context = _two_lists_context()

    result = run_agent_turn_in_graph(
        AgentState(
            intent="preference_action",
            message="喜欢第二篇论文",
            context=context,
            paper_reference=_understood(context, "ambiguous", "2401.00002", "2501.00002"),
        )
    )

    assert result.status == "success"
    assert result.interaction is None
    assert result.outputs["paper_reference"]["status"] == "need_confirmation"
    assert called_tools == []
    assert any(trace.event == "interaction_requested" and trace.step_id == "resolve_preference_target" for trace in result.trace)
    assert not any(trace.step_id == "update_preference_store" and trace.event == "step_succeeded" for trace in result.trace)


def test_run_agent_turn_preference_action_ordinal_resolves_then_waits_for_write_confirmation(monkeypatch) -> None:
    preference_calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        preference_calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="preference_action",
            message="喜欢这第2篇论文",
            context=(
                context := {
                    # 偏好写入是持久化动作，序号指的是列表第二篇而不是默认选中的第一篇。
                    "selected_paper": {"arxiv_id": "2401.00001", "title": "First Paper"},
                    "last_papers": [
                        {"arxiv_id": "2401.00001", "title": "First Paper"},
                        {"arxiv_id": "2401.00002", "title": "Second Paper"},
                    ],
                }
            ),
            paper_reference=_understood(context, "single", "2401.00002"),
        )
    )

    assert result.status == "waiting_interaction"
    assert result.interaction is not None
    assert result.interaction.kind == "side_effect_approval"
    assert result.interaction.payload.tool_name == "update_preference_store"
    assert result.outputs["paper_reference"]["arxiv_id"] == "2401.00002"
    assert result.outputs["paper_reference"]["final_target_resolved"] is True
    assert result.outputs["paper_reference"]["requires_confirmation"] is True
    assert result.outputs["paper_reference"]["reference_hint"]["reference_type"] == "llm_reference"
    assert preference_calls == []


def test_run_agent_turn_paper_qa_context_target_checks_index_then_requests_build_confirmation(monkeypatch) -> None:
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append(tool_name)
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "missing", "data": {"status": "missing", "has_index": False}, "trace": {}, "error": None}
        raise AssertionError(f"{tool_name} should not run before index build confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="这篇论文的方法是什么？",
            context=(context := {"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Method"}}),
            paper_reference=_understood(context, "single", "2401.00001"),
        )
    )

    assert result.status == "waiting_interaction"
    assert result.interaction is not None
    assert result.interaction.kind == "side_effect_approval"
    assert result.interaction.payload.tool_name == "parse_and_index_paper"
    assert result.outputs["paper_ref"]["reference_hint"]["reference_type"] == "llm_reference"
    # 缺索引链路会在 check step 和确认前预检查各查一次，保证确认卡弹出前能感知刚完成的异步索引。
    assert called_tools == ["check_paper_qa_index"]
    assert any(trace.event == "interaction_requested" for trace in result.trace)


def test_run_agent_turn_in_graph_reject_skips_index_build(monkeypatch) -> None:
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        if tool_name == "check_paper_qa_index":
            return {"ok": True, "tool_name": tool_name, "summary": "missing", "data": {"status": "missing", "has_index": False}, "trace": {}, "error": None}
        raise AssertionError(f"{tool_name} should not run after rejected confirmation")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)
    monkeypatch.setattr(executor_module, "interrupt", lambda payload: {"decision": "reject"})

    result = run_agent_turn_in_graph(
        AgentState(
            intent="paper_qa",
            message="这篇论文的方法是什么？",
            context=(context := {"selected_paper": {"arxiv_id": "2401.00001", "title": "RAG Method"}}),
            paper_reference=_understood(context, "single", "2401.00001"),
        )
    )

    assert result.status == "success"
    assert result.interaction is None
    # 即使用户拒绝构建，确认前预检查仍会先保持幂等状态判断，真正的建索引工具不能被调用。
    assert [tool_name for tool_name, _ in called_tools] == ["check_paper_qa_index"]
    assert any(trace.event == "interaction_requested" for trace in result.trace)
    assert not any(tool_name == "parse_and_index_paper" for tool_name, _ in called_tools)


def test_run_agent_turn_paper_qa_without_reference_does_not_call_real_answer_tool(monkeypatch) -> None:
    called_tools = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        called_tools.append((tool_name, dict(kwargs)))
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="paper_qa",
            message="method question",
            context=(context := {"selected_paper": {"arxiv_id": "2401.00001", "title": "method question"}}),
            # 理解阶段判断这句话没指向具体论文：不能默认拿选中论文去问答，要追问。
            paper_reference=_understood(context, "none"),
        )
    )

    assert result.status == "fallback"
    assert _reference_hint_from_trace(result, "resolve_paper")["reference_type"] == "llm_reference"
    assert result.error == "fallback:resolve_paper:clarification_required:reference_not_mentioned"
    assert "我不确定你指的是哪篇论文" in result.final_answer
    assert "method question（arXiv 2401.00001）" in result.final_answer
    assert "paper_qa_result" not in result.outputs
    assert called_tools == []
    pseudo_steps = {"retrieve_paper_chunks", "rewrite_paper_query", "rerank_paper_chunks", "validate_qa_evidence", "generate_paper_answer", "verify_answer_grounding"}
    assert not pseudo_steps.intersection({trace.step_id for trace in result.trace})
    assert [step.tool_name for step in result.plan.steps] == [
        "resolve_paper",
        "check_paper_index",
        "answer_paper_question",
        "assess_paper_qa_quality",
    ]


def test_run_agent_turn_preference_action_ambiguous_target_does_not_persistent_write(monkeypatch) -> None:
    preference_calls = []

    def fake_invoke_tool(tool_name: str, **kwargs):
        preference_calls.append((tool_name, dict(kwargs)))
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fake_invoke_tool)

    result = run_agent_turn(
        AgentState(
            intent="preference_action",
            message="喜欢第二篇论文",
            context=(
                context := {
                    "last_papers": [
                        {"arxiv_id": "2401.00001", "title": "Search First"},
                        {"arxiv_id": "2401.00002", "title": "Search Second"},
                    ],
                    "recommendations": [
                        {"arxiv_id": "2501.00001", "title": "Rec First"},
                        {"arxiv_id": "2501.00002", "title": "Rec Second"},
                    ],
                }
            ),
            # 搜索结果和推荐列表各有第二篇：理解阶段判成有歧义，偏好写入必须让用户选。
            paper_reference=_understood(context, "ambiguous", "2401.00002", "2501.00002"),
        )
    )

    side_effects = {step.tool_name: step.side_effect_level for step in result.plan.steps}
    assert side_effects["update_preference_store"] == "persistent_write"
    assert not any("interest" in tool_name or "profile" in tool_name for tool_name in side_effects)
    assert result.status == "waiting_interaction"
    assert result.interaction is not None
    assert result.interaction.kind == "target_selection"
    assert result.interaction.step_id == "resolve_preference_target"
    assert len(result.interaction.payload.candidates) == 2
    hint = _reference_hint_from_trace(result, "resolve_preference_target")
    assert hint["reference_type"] == "llm_reference"
    assert result.outputs["paper_reference"]["status"] == "need_confirmation"
    assert len(result.outputs["paper_reference"]["candidates"]) == 2
    assert preference_calls == []
    assert not any(trace.event == "confirmation_approved" and trace.step_id == "update_preference_store" for trace in result.trace)
    assert not any(trace.step_id == "update_preference_store" and trace.event == "step_succeeded" for trace in result.trace)
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
