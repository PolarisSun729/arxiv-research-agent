from __future__ import annotations

import importlib
import json
import logging

import httpx
import openai
import pytest

from scripts.planning_baseline_stats import format_report, parse_records, percentile, summarize
from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")
planner_context_module = importlib.import_module("backend.agents.arxiv_search_agent.planner_context")
planning_record = importlib.import_module("backend.agents.arxiv_search_agent.planning_record")
tool_aware_module = importlib.import_module("backend.agents.arxiv_search_agent.tool_aware_planner")
tool_registry = importlib.import_module("backend.agents.arxiv_search_agent.tool_registry")

USER_MESSAGE = "帮我找最近的 RAG 论文"
RULE_TOOLS = ["normalize_request", "build_arxiv_search_spec", "search_arxiv", "synthesize_arxiv_response"]


class _FakeGenerationService:
    """按给定行为模拟 generation service：返回文本，或抛出指定异常。"""

    def __init__(self, *, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def generate(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return {"response": self.response}


def _search_state():
    return state_module.AgentState(
        intent="arxiv_search",
        message=USER_MESSAGE,
        session_id="s1",
        context={"run_id": "run-1"},
        search_spec=schemas.ArxivSearchSpec(intent="arxiv_search", query="RAG"),
    )


def _valid_llm_plan_json(state) -> str:
    """规则 builder 的草稿本身就能通过 LLM 草稿校验，用它当「LLM 返回了合格计划」。"""
    registry = tool_registry.PLANNER_TOOL_REGISTRY
    goal = planner_module.GoalBuilder.from_state(state)
    context = planner_context_module.build_planner_context(goal=goal, state=state, tool_registry=registry)
    selection = tool_aware_module.ToolCandidateSelector(registry).select(goal, state, context)
    draft = tool_aware_module.RuleBasedToolAwarePlanBuilder().build(goal, state, selection.candidate_tools, registry, context)
    return draft.model_dump_json()


def _plan_with(service, caplog):
    caplog.set_level(logging.INFO)
    _, _, debug = planner_module.build_executable_plan(
        _search_state(),
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=True,
        llm_generation_service=service,
    )
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("agent_planning_record ")]
    assert len(lines) == 1
    logged = json.loads(lines[0][len("agent_planning_record "):])
    assert logged == debug["planning_record"]
    # 日志只有元数据：不含用户原话，也不含 LLM 原始输出。
    assert USER_MESSAGE not in lines[0] and "raw_llm_plan" not in lines[0]
    return debug["planning_record"]


def _assert_fell_back_to_rule(record) -> None:
    assert record["plan_source"] == "rule"
    assert record["plan_source_detail"] == "tool_aware_rule_based"
    rule_attempt = record["plan_attempts"][1]
    assert rule_attempt["source"] == "rule" and rule_attempt["ok"] is True
    assert rule_attempt["tools"] == RULE_TOOLS


def test_record_for_llm_plan_that_passes(caplog) -> None:
    record = _plan_with(_FakeGenerationService(response=_valid_llm_plan_json(_search_state())), caplog)

    assert record["intent"] == "arxiv_search"
    assert record["run_id"] == "run-1" and record["session_id"] == "s1"
    assert record["plan_source"] == "llm"
    assert record["plan_source_detail"] == "llm_tool_aware"
    assert len(record["plan_attempts"]) == 1
    attempt = record["plan_attempts"][0]
    assert attempt == {
        "source": "llm",
        "ok": True,
        "failure_kind": "none",
        "reasons": [],
        "elapsed_ms": attempt["elapsed_ms"],
        "step_count": 4,
        "tools": RULE_TOOLS,
    }
    assert attempt["elapsed_ms"] >= 0
    assert record["total_planning_ms"] >= attempt["elapsed_ms"]


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=httpx.Request("POST", "http://llm.invalid/v1/responses")),
        TimeoutError("timed out"),
    ],
    ids=["openai_api_timeout", "builtin_timeout"],
)
def test_record_marks_llm_timeout(caplog, error) -> None:
    record = _plan_with(_FakeGenerationService(error=error), caplog)

    llm_attempt = record["plan_attempts"][0]
    assert llm_attempt["source"] == "llm" and llm_attempt["ok"] is False
    assert llm_attempt["failure_kind"] == "timeout"
    assert llm_attempt["reasons"] == [type(error).__name__]
    assert llm_attempt["step_count"] == 0 and llm_attempt["tools"] == []
    _assert_fell_back_to_rule(record)


def test_record_marks_validation_failure_with_short_reasons(caplog) -> None:
    empty_plan = json.dumps({"draft_id": "llm:empty", "plan_intent": "arxiv_search", "selected_tools": [], "steps": []})

    record = _plan_with(_FakeGenerationService(response=empty_plan), caplog)

    llm_attempt = record["plan_attempts"][0]
    assert llm_attempt["ok"] is False
    assert llm_attempt["failure_kind"] == "validation_failed"
    assert "LLM draft has no steps" in llm_attempt["reasons"]
    assert all(len(reason) <= 160 for reason in llm_attempt["reasons"])
    _assert_fell_back_to_rule(record)


@pytest.mark.parametrize(
    ("service", "failure_kind"),
    [
        (_FakeGenerationService(response="这是计划：先检索再总结"), "parse_error"),
        (_FakeGenerationService(error=RuntimeError("upstream 500")), "llm_error"),
    ],
    ids=["not_json", "llm_call_error"],
)
def test_record_separates_parse_and_call_errors(caplog, service, failure_kind) -> None:
    record = _plan_with(service, caplog)

    assert record["plan_attempts"][0]["failure_kind"] == failure_kind
    _assert_fell_back_to_rule(record)


def test_rule_only_round_has_no_llm_attempt(caplog) -> None:
    caplog.set_level(logging.INFO)
    _, _, debug = planner_module.build_executable_plan(_search_state(), enable_tool_aware_planner=True, enable_llm_plan_draft=False)

    record = debug["planning_record"]
    assert [attempt["source"] for attempt in record["plan_attempts"]] == ["rule"]
    assert record["plan_source"] == "rule"


# ---- 统计脚本 ----


def _attempt(source, ok=True, failure_kind="none", elapsed_ms=10):
    return {"source": source, "ok": ok, "failure_kind": failure_kind, "reasons": [], "elapsed_ms": elapsed_ms, "step_count": 3, "tools": []}


SAMPLE_RECORDS = [
    # 4 轮 LLM 直接通过
    *[{"plan_attempts": [_attempt("llm", elapsed_ms=ms)], "plan_source": "llm", "total_planning_ms": ms} for ms in (100, 200, 300, 400)],
    # 2 轮超时、1 轮校验失败、1 轮不是 JSON，都回退到规则
    {"plan_attempts": [_attempt("llm", False, "timeout", 8000), _attempt("rule", elapsed_ms=5)], "plan_source": "rule", "total_planning_ms": 8010},
    {"plan_attempts": [_attempt("llm", False, "timeout", 9000), _attempt("rule", elapsed_ms=5)], "plan_source": "rule", "total_planning_ms": 9010},
    {"plan_attempts": [_attempt("llm", False, "validation_failed", 500), _attempt("rule", elapsed_ms=5)], "plan_source": "rule", "total_planning_ms": 510},
    {"plan_attempts": [_attempt("llm", False, "parse_error", 600), _attempt("rule", elapsed_ms=5)], "plan_source": "rule", "total_planning_ms": 610},
    # 不走 LLM 的两轮：一轮规则、一轮兜到模板
    {"plan_attempts": [_attempt("rule", elapsed_ms=5)], "plan_source": "rule", "total_planning_ms": 6},
    {"plan_attempts": [_attempt("rule", False, "validation_failed", 5), _attempt("legacy_template", elapsed_ms=2)], "plan_source": "legacy_template", "total_planning_ms": 8},
]


def test_stats_script_parses_log_lines_and_skips_noise() -> None:
    lines = ["2026-10-17 10:00:00 INFO uvicorn started"]
    for record in SAMPLE_RECORDS:
        lines.append(f"Oct 17 10:00:01 host arxiv-agent[1]: 2026-10-17 10:00:01 INFO agents.planning_record agent_planning_record {json.dumps(record)}")
    lines.append("2026-10-17 10:00:02 INFO x agent_planning_record {broken json")

    assert parse_records(lines) == SAMPLE_RECORDS


def test_stats_script_summary_on_sample_data() -> None:
    summary = summarize(SAMPLE_RECORDS)

    assert summary["rounds"] == 10
    assert summary["llm_rounds"] == 8 and summary["llm_attempts"] == 8
    assert summary["llm_first_pass_rate"] == 0.5
    assert summary["llm_timeout_rate"] == 0.25
    assert {kind: item["count"] for kind, item in summary["llm_failure_kinds"].items()} == {
        "none": 4,
        "timeout": 2,
        "llm_error": 0,
        "parse_error": 1,
        "validation_failed": 1,
    }
    assert summary["llm_failure_kinds"]["parse_error"]["ratio"] == 0.125
    assert {source: item["ratio"] for source, item in summary["plan_sources"].items()} == {
        "llm": 0.4,
        "rule": 0.5,
        "legacy_template": 0.1,
        "unsupported": 0.0,
    }
    # 整轮耗时排序：6 8 100 200 300 400 510 610 8010 9010 → p50 取第 5 个，p95 取第 10 个。
    assert summary["total_planning_ms"] == {"p50": 300.0, "p95": 9010.0, "samples": 10}
    assert summary["llm_attempt_ms"] == {"p50": 400.0, "p95": 9000.0, "samples": 8}
    report = format_report(summary)
    assert "LLM 计划首次通过率：50.0%" in report and "LLM 超时率：25.0%" in report


def test_stats_script_handles_empty_input() -> None:
    summary = summarize([])

    assert summary["llm_first_pass_rate"] is None and summary["llm_timeout_rate"] is None
    assert summary["total_planning_ms"]["p50"] is None
    assert "-" in format_report(summary)
    assert percentile([5], 0.95) == 5
