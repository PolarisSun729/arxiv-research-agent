"""研究轨迹 → 问答面板执行过程的投影测试。"""

from __future__ import annotations

from services.paper_qa.research_process_debug import build_research_process_debug


def test_projects_rounds_with_decision_reason_and_ordered_steps() -> None:
    events = [
        {"event_type": "research_started"},
        {"event_type": "action_proposed", "action": "search_paper", "reason_code": "OPEN_EVIDENCE_NEED"},
        {"event_type": "action_accepted", "action": "search_paper"},
        {
            "event_type": "retrieval_completed", "target_need_id": "need-1", "query": "q1", "status": "completed",
            "retrieval_count": 1, "new_candidate_count": 20, "duplicate_candidate_count": 0,
            "stage_summary": {"route_counts": {"vector_original": 30}, "final_context": ["chunk 54 · figure"]},
        },
        {"event_type": "action_proposed", "action": "search_paper", "reason_code": "VERIFY_STAGES"},
        {
            "event_type": "retrieval_completed", "target_need_id": "need-1", "query": "q2", "status": "completed",
            "retrieval_count": 2, "new_candidate_count": 0, "duplicate_candidate_count": 0, "stage_summary": None,
        },
        {"event_type": "fallback_draft_scheduled", "reason_code": "NO_PROGRESS_FALLBACK_DRAFT"},
        {"event_type": "draft_created", "version": 1},
        {"event_type": "claim_verification_completed", "draft_version": 1, "assessment_count": 3, "supported_claim_count": 2},
        {"event_type": "action_proposed", "action": "search_paper", "reason_code": "GAP"},
        {"event_type": "action_rejected", "action": "search_paper", "rejection_code": "NO_PROGRESS_LIMIT_REACHED"},
        {"event_type": "research_completed", "outcome": "partial"},
    ]

    debug = build_research_process_debug(events)

    first, second = debug["research_rounds"]
    assert first["round"] == 1 and first["decision_reason"] == "OPEN_EVIDENCE_NEED"
    assert first["route_counts"] == {"vector_original": 30}
    assert first["final_context"] == ["chunk 54 · figure"]
    assert second["decision_reason"] == "VERIFY_STAGES" and second["new_candidate_count"] == 0
    assert [step["kind"] for step in debug["research_steps"]] == [
        "search", "search", "fallback", "draft", "verify", "rejected", "done",
    ]
    # 兜底起草不是决策器提议的，不能把上一轮检索的理由挂到草稿上。
    assert debug["research_steps"][3]["detail"] == ""
    assert debug["research_steps"][4]["detail"] == "2/3 条主张有证据支持"
    assert debug["research_steps"][5]["detail"] == "NO_PROGRESS_LIMIT_REACHED"


def test_restored_best_draft_is_shown_as_a_step():
    debug = build_research_process_debug([
        {"event_type": "best_verified_draft_restored", "draft_version": 1, "supported_claim_count": 10},
        {"event_type": "research_completed", "outcome": "partial"},
    ])

    assert debug["research_steps"][0] == {
        "kind": "restore", "title": "退回校验结果最好的第 1 版草稿", "detail": "10 条主张有证据支持",
    }


def test_claim_extraction_step_shows_extractor_source():
    debug = build_research_process_debug([
        {"event_type": "claims_extracted", "draft_version": 1, "claim_count": 7, "extractor_source": "llm"},
        {"event_type": "claims_extracted", "draft_version": 2, "claim_count": 9, "extractor_source": "rule"},
        {"event_type": "claims_extracted", "draft_version": 3, "claim_count": 2},
    ])

    assert [step["detail"] for step in debug["research_steps"]] == [
        "7 条主张 · LLM 拆解", "9 条主张 · 规则兜底", "2 条主张",
    ]
    assert debug["research_steps"][0]["title"] == "拆解第 1 版草稿的主张"


def test_removed_claims_are_exposed_for_partial_answers():
    removed = {
        "claim_id": "claim-v2-1", "text": "根据证据，方法包含两个层面。", "importance": "core",
        "citation_ids": [], "verdict": "unsupported", "supporting_evidence_ids": [], "missing_facets": [],
        "reason_code": "uncited", "reason": "没有挂引用，无法对照论文核对", "unexpected": "dropped",
    }
    debug = build_research_process_debug([
        {"event_type": "research_completed", "outcome": "partial", "removed_claims": [removed], "answer_pruning": "in_place"},
    ])

    assert debug["answer_pruning"] == "in_place"
    assert debug["removed_claims"] == [{key: value for key, value in removed.items() if key != "unexpected"}]
    assert debug["research_steps"][-1]["detail"] == "partial · 移除 1 条无证据支持的句子"


def test_completed_research_has_no_removed_claims():
    debug = build_research_process_debug([{"event_type": "research_completed", "outcome": "completed"}])

    assert debug["removed_claims"] == [] and debug["answer_pruning"] is None


def test_discarded_evidence_and_decision_reading_are_shown():
    debug = build_research_process_debug([
        {"event_type": "action_proposed", "action": "search_paper", "reason_code": "OPEN_NEED",
         "decision_evidence_count": 0, "decision_image_count": 0},
        {"event_type": "retrieval_completed", "retrieval_count": 1, "target_need_id": "need-1",
         "new_candidate_count": 20, "duplicate_candidate_count": 0},
        {"event_type": "action_proposed", "action": "draft_answer", "reason_code": "ON_TOPIC",
         "decision_evidence_count": 20, "decision_image_count": 2},
        {"event_type": "evidence_discarded", "discarded_candidate_ids": ["7", "9"], "remaining_candidate_count": 18},
        {"event_type": "evidence_discarded", "discarded_candidate_ids": [], "remaining_candidate_count": 18},
        {"event_type": "draft_created", "version": 1},
    ])

    steps = debug["research_steps"]
    assert steps[0]["detail"] == "新增 20 条候选，重复 0 条 · 决策理由 OPEN_NEED · 决策时阅读 0 条证据全文"
    # 只有受保护或未知 ID 时没有真正丢弃，不单独列一步。
    assert steps[1] == {"kind": "filter", "title": "筛掉 2 条无关证据", "detail": "剩余 18 条"}
    assert steps[2]["detail"] == "决策理由 ON_TOPIC · 决策时阅读 20 条证据全文、2 张图"
    assert len(steps) == 3


def test_old_traces_without_decision_reading_keep_previous_detail():
    debug = build_research_process_debug([
        {"event_type": "action_proposed", "action": "draft_answer", "reason_code": "INITIAL_DRAFT"},
        {"event_type": "draft_created", "version": 1},
    ])

    assert debug["research_steps"][0]["detail"] == "决策理由 INITIAL_DRAFT"
