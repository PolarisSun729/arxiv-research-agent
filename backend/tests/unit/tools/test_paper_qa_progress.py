"""论文问答工具把研究引擎事件翻译成时间线文案：要说清做了什么，而不是只报计数。"""

from __future__ import annotations

from tools.paper_qa_tools import _report_research_progress
from tools.tool_progress import use_tool_progress_reporter


def _collect(events: list[dict]) -> list[dict]:
    reported: list[dict] = []
    with use_tool_progress_reporter(reported.append):
        for event in events:
            _report_research_progress(event)
    return reported


def test_needs_are_listed_and_each_decision_says_what_it_chose() -> None:
    reported = _collect([
        {"event": "question_analyzed", "need_count": 2, "needs": [
            {"number": 1, "description": "XoG 整体框架", "importance": "core"},
            {"number": 2, "description": "实验数据集", "importance": "supporting"},
        ]},
        {"event": "action_decided", "action": "search_paper", "need_number": 1, "need_description": "XoG 整体框架", "query": "XoG"},
        {"event": "stage_started", "stage": "search", "retrieval_round": 1, "need_number": 1, "need_description": "XoG 整体框架"},
        {"event": "retrieval_completed", "query": "XoG framework overview", "new_candidate_count": 34, "duplicate_candidate_count": 2},
        {"event": "action_decided", "action": "draft_answer", "revision": False, "need_numbers": [1, 2]},
        {"event": "action_rejected", "action": "draft_answer", "rejection_code": "DRAFT_BUDGET_EXHAUSTED"},
    ])

    assert reported[0]["items"] == ["需求 1（核心）：XoG 整体框架", "需求 2（辅助）：实验数据集"]
    assert reported[0]["detail"] == "拆出 2 个证据需求"
    assert [item.get("detail") for item in reported[1:]] == [
        "→ 检索需求 1",
        "需求 1：XoG 整体框架",
        "检索词：XoG framework overview",
        "新增 34 段证据，2 段已有",
        "→ 撰写回答，覆盖需求 1、需求 2",
        "→ 想撰写回答，但修订次数已用完",
    ]


def test_rule_fallback_and_uncovered_needs_are_visible() -> None:
    reported = _collect([
        {"event": "claims_extracted", "claim_count": 39, "extractor_source": "rule"},
        {"event": "coverage_projected", "need_count": 4, "satisfied_need_numbers": [2, 3], "unsatisfied_need_numbers": [1, 4]},
    ])

    assert reported[0]["detail"] == "39 条要点（模型拆分失败，已按标点切句兜底）"
    assert reported[1] == {"stage": "verify_claims", "label": "", "status": "done", "detail": "需求覆盖 2/4（未覆盖：需求 1、需求 4）"}
