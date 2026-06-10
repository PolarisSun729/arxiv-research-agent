from __future__ import annotations

from services.paper_qa.qa_observation import build_error_qa_observation, build_qa_observation


def _source(section: str, text: str = "Detailed evidence " * 20):
    return {"source_id": section, "section_title": section, "content": text}


def test_section_mismatch_recommends_section_focused_repair() -> None:
    observation = build_qa_observation(
        retrieval_debug={
            "query_profile": {"question_type": "method", "section_preferences": ["method"]},
            "query_rewrite": {"enabled": True, "selected_queries": ["method pipeline"]},
        },
        sources=[_source("related work"), _source("background")],
        generation_result={"answer": "answer"},
        verification_result={"status": "warning", "source_count": 2},
    )

    assert "retry_with_section_focus" in observation["recommended_repair_actions"]
    assert "retry_with_expanded_context" in observation["recommended_repair_actions"]
    assert any(item["action"] == "retry_with_section_focus" for item in observation["repair_action_details"])


def test_metric_question_recommends_keyword_emphasis() -> None:
    observation = build_qa_observation(
        retrieval_debug={"query_profile": {"question_type": "metric", "section_preferences": ["results"]}},
        sources=[_source("results"), _source("experiments")],
        generation_result={"answer": "answer"},
        verification_result={"status": "passed", "source_count": 2},
    )

    assert "retry_with_keyword_emphasis" in observation["recommended_repair_actions"]


def test_weak_hyde_retrieval_recommends_retry_without_hyde() -> None:
    observation = build_qa_observation(
        retrieval_debug={
            "hyde": {"enabled": True, "text": "hypothetical noisy answer"},
            "query_profile": {"question_type": "definition"},
        },
        sources=[_source("unknown", text="short")],
        generation_result={"answer": "answer"},
        verification_result={"status": "warning", "source_count": 1},
    )

    assert observation["retrieval_quality"] == "weak"
    assert "retry_without_hyde" in observation["recommended_repair_actions"]


def test_missing_index_recommends_confirmation_rebuild_action_only() -> None:
    observation = build_error_qa_observation(
        error_code="qa_index_not_found",
        error_stage="build_qa_context",
        error_reason="index_status:missing",
    )

    assert observation["retrieval_quality"] == "failed"
    assert "ask_user_to_rebuild_index" in observation["recommended_repair_actions"]
    detail = next(item for item in observation["repair_action_details"] if item["action"] == "ask_user_to_rebuild_index")
    assert detail["input_params"]["requires_confirmation"] is True
