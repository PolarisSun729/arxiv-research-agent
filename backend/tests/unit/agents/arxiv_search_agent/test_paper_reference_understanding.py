"""论文指代改由理解阶段 LLM 判断：候选清单、落地校验、目标解析和追问话术。"""

from __future__ import annotations

import importlib
import json
import sys

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
state_module = _MODULES["state_module"]
parse_module = sys.modules["backend.agents.arxiv_search_agent.node.parse_node"]
contextualizer = importlib.import_module("backend.agents.arxiv_search_agent.request_contextualizer")
paper_reference = importlib.import_module("backend.agents.arxiv_search_agent.utils.paper_reference")
paper_qa_adapters = importlib.import_module("backend.agents.arxiv_search_agent.tool_adapters.paper_qa")
executor_module = importlib.import_module("backend.agents.arxiv_search_agent.plan_executor")

AgentState = state_module.AgentState

MATRAG = {"arxiv_id": "2610.01767", "title": "MatRAG: Multi-hop Retrieval for Materials Science"}
SEARCH_RESULTS = [MATRAG] + [{"arxiv_id": f"2610.0{index}000", "title": f"Other RAG Paper {index}"} for index in range(2, 8)]
# 线上故障现场：前端选中 MatRAG，上一轮搜索结果 7 篇且 MatRAG 排第一，上一轮刚讲完它的方法。
INCIDENT_CONTEXT = {"frontend_visible_paper": dict(MATRAG), "last_papers": [dict(paper) for paper in SEARCH_RESULTS]}
INCIDENT_TURNS = [
    {"role": "user", "content": "给我讲一下第一篇论文提出的方法吧"},
    {"role": "assistant", "content": "MatRAG 提出了一个面向材料科学的多跳检索框架……"},
]
INCIDENT_MESSAGE = "那在进行多跳问答的时候，他是怎么做的呢"


class _ScriptedLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_with_qwen(self, prompt: str, task_type: str | None = None, **_: object) -> str:
        self.calls.append({"task_type": task_type, "prompt": prompt})
        return self.responses.pop(0)


def _llm_reply(standalone: str, kind: str, refs: list[str]) -> str:
    return json.dumps({"standalone_request": standalone, "paper_reference": {"kind": kind, "refs": refs}}, ensure_ascii=False)


def _resolve(message: str, context: dict, reference, action_type: str = "paper_qa") -> dict:
    result = paper_qa_adapters.ResolvePaperAdapter().execute(
        paper_qa_adapters.ResolvePaperInput(message=message, context=context, action_type=action_type, paper_reference=reference)
    )
    assert result.ok, result.error
    return result.data.model_dump()


def test_candidates_put_focus_first_and_merge_sources_of_the_same_paper() -> None:
    candidates = paper_reference.build_reference_candidates(INCIDENT_CONTEXT)

    assert [item["key"] for item in candidates] == [f"P{index}" for index in range(1, 8)]
    assert candidates[0]["record"]["arxiv_id"] == "2610.01767"
    # 同一篇论文既是前端选中又是搜索第一篇，只占一个编号，来源都写清楚，模型才能对上「第一篇」和「他」。
    assert candidates[0]["sources"] == ["前端当前选中论文", "上一轮搜索结果第1篇"]
    assert candidates[1]["sources"] == ["上一轮搜索结果第2篇"]


def test_incident_follow_up_pronoun_resolves_to_paper_in_context() -> None:
    llm = _ScriptedLLM(_llm_reply("那在进行多跳问答的时候，MatRAG是怎么做的呢", "single", ["P1"]))

    contextualization = contextualizer.contextualize_request(INCIDENT_MESSAGE, INCIDENT_TURNS, llm, context=INCIDENT_CONTEXT)

    prompt = llm.calls[0]["prompt"]
    assert '"key": "P1"' in prompt and MATRAG["title"] in prompt and "上一轮搜索结果第1篇" in prompt
    reference = contextualization["paper_reference"]
    assert reference["status"] == "success" and reference["kind"] == "single" and reference["keys"] == ["P1"]

    resolution = _resolve(INCIDENT_MESSAGE, INCIDENT_CONTEXT, reference)
    assert resolution["status"] == "resolved"
    assert resolution["arxiv_id"] == "2610.01767"
    assert resolution["resolution_reason"] == "llm_reference_single"


def test_first_turn_with_selected_paper_still_asks_llm_about_reference() -> None:
    llm = _ScriptedLLM(_llm_reply("这个方法的创新点是什么", "single", ["P1"]))

    contextualization = contextualizer.contextualize_request("这个方法的创新点是什么", [], llm, context={"frontend_visible_paper": dict(MATRAG)})

    assert len(llm.calls) == 1
    assert contextualization["paper_reference"]["papers"][0]["arxiv_id"] == "2610.01767"


def test_no_history_and_no_candidates_skips_the_call() -> None:
    llm = _ScriptedLLM()

    contextualization = contextualizer.contextualize_request("找 RAG 的论文", [], llm, context={})

    assert llm.calls == []
    assert contextualization["status"] == "skipped"
    assert contextualization["paper_reference"] is None


def test_invented_key_is_rejected_and_resolution_asks_with_candidates() -> None:
    llm = _ScriptedLLM(_llm_reply(INCIDENT_MESSAGE, "single", ["P42"]))

    contextualization = contextualizer.contextualize_request(INCIDENT_MESSAGE, INCIDENT_TURNS, llm, context=INCIDENT_CONTEXT)

    reference = contextualization["paper_reference"]
    assert reference["status"] == "invalid_output" and reference["papers"] == []
    # 改写句本身没问题，照常交给意图识别；只是指代判断不可信。
    assert contextualization["status"] == "success"
    resolution = _resolve(INCIDENT_MESSAGE, INCIDENT_CONTEXT, reference)
    assert resolution["status"] == "need_clarification"
    assert resolution["resolution_reason"] == "reference_understanding_unavailable"
    assert resolution["candidates"][0]["arxiv_id"] == "2610.01767"


def test_llm_unavailable_or_failed_never_falls_back_to_regex() -> None:
    unavailable = contextualizer.contextualize_request("讲一下第二篇", INCIDENT_TURNS, None, context=INCIDENT_CONTEXT)
    failed = contextualizer.contextualize_request("讲一下第二篇", INCIDENT_TURNS, _ScriptedLLM("not json"), context=INCIDENT_CONTEXT)

    assert unavailable["paper_reference"]["status"] == "llm_unavailable"
    assert failed["paper_reference"]["status"] == "failed"
    for reference in (unavailable["paper_reference"], failed["paper_reference"], None):
        # 「第二篇」正则本来认得出；这里必须追问，不能绕过 LLM 自己猜。
        resolution = _resolve("讲一下第二篇", INCIDENT_CONTEXT, reference)
        assert resolution["status"] == "need_clarification"
        assert resolution["resolution_reason"] == "reference_understanding_unavailable"


def test_explicit_arxiv_id_resolves_without_understanding_result() -> None:
    resolution = _resolve("2404.03868 这篇讲了什么", {}, None)

    assert resolution["status"] == "resolved"
    assert resolution["arxiv_id"] == "2404.03868"


def test_ordinal_comparison_outside_and_topic_switch_kinds() -> None:
    candidates = paper_reference.build_reference_candidates(INCIDENT_CONTEXT)

    ordinal = paper_reference.ground_paper_reference({"kind": "single", "refs": ["p2"]}, candidates)
    assert _resolve("第二篇讲了什么", INCIDENT_CONTEXT, ordinal)["arxiv_id"] == "2610.02000"

    comparison = paper_reference.ground_paper_reference({"kind": "multiple", "refs": ["P1", "P2"]}, candidates)
    resolution = _resolve("比较一下第一篇和第二篇", INCIDENT_CONTEXT, comparison)
    assert resolution["resolution_reason"] == "reference_multiple_targets"
    assert [item["arxiv_id"] for item in resolution["candidates"]] == ["2610.01767", "2610.02000"]

    outside = paper_reference.ground_paper_reference({"kind": "outside", "refs": []}, candidates)
    assert _resolve("讲讲 GraphRAG 那篇", INCIDENT_CONTEXT, outside)["resolution_reason"] == "reference_outside_context"

    switch = paper_reference.ground_paper_reference({"kind": "none", "refs": []}, candidates)
    assert _resolve("多跳问答一般怎么做", INCIDENT_CONTEXT, switch)["resolution_reason"] == "reference_not_mentioned"


def test_ambiguous_reference_for_side_effect_asks_user_to_pick() -> None:
    candidates = paper_reference.build_reference_candidates(INCIDENT_CONTEXT)
    ambiguous = paper_reference.ground_paper_reference({"kind": "ambiguous", "refs": ["P1", "P2"]}, candidates)

    qa = _resolve("讲讲那篇", INCIDENT_CONTEXT, ambiguous)
    favorite = _resolve("收藏那篇", INCIDENT_CONTEXT, ambiguous, action_type="preference_action")

    assert qa["status"] == "resolved" and qa["assumption_note"].startswith("我先按《MatRAG")
    assert favorite["status"] == "need_confirmation"
    assert len(favorite["candidates"]) == 2


def test_parse_node_stores_grounded_reference_on_state() -> None:
    llm = _ScriptedLLM(
        _llm_reply("那在进行多跳问答的时候，MatRAG是怎么做的呢", "single", ["P1"]),
        json.dumps({"intent": "paper_qa", "confidence": 0.9}),
    )

    state = parse_module.parse_search_request(
        {"message": INCIDENT_MESSAGE, "context": {**INCIDENT_CONTEXT, "recent_turns": INCIDENT_TURNS}},
        generation_service=llm,
    )

    assert state.paper_reference["kind"] == "single"
    assert state.paper_reference["papers"][0]["arxiv_id"] == "2610.01767"
    assert state.debug["paper_reference"] == {"status": "success", "kind": "single", "keys": ["P1"], "error": None}


def test_unresolved_reference_ends_turn_with_clarification_listing_candidates(monkeypatch) -> None:
    def fail_if_called(tool_name: str, **kwargs):
        raise AssertionError(f"目标论文没定下来不能调用业务工具: {tool_name}")

    monkeypatch.setattr(executor_module, "invoke_backend_tool", fail_if_called)

    result = executor_module.run_agent_turn(
        AgentState(
            intent="paper_qa",
            message=INCIDENT_MESSAGE,
            context=INCIDENT_CONTEXT,
            paper_reference=paper_reference.unavailable_paper_reference("failed", "timeout"),
        )
    )

    assert result.status == "fallback"
    # 原因码保留真实原因，而不是恢复候选的 fallback_if_failed（旧代码这里是 fallback_answer）。
    assert result.error == "fallback:resolve_paper:clarification_required:reference_understanding_unavailable"
    assert result.final_answer.startswith("我不确定你指的是哪篇论文")
    assert f"1. {MATRAG['title']}（arXiv 2610.01767）" in result.final_answer
    assert "这次请求没能顺利完成" not in result.final_answer
