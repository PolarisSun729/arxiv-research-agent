from __future__ import annotations

from services.paper_qa.qa_observation import (
    RETRIEVAL_STAGE_NAMES,
    ROUTE_STAGE_NAMES,
    build_error_qa_observation,
    build_qa_observation,
    build_research_qa_observation,
)


def _source(section: str, text: str = "Detailed evidence " * 20):
    return {"source_id": section, "section_title": section, "content": text}


def test_observation_shape_keeps_all_stage_and_route_keys() -> None:
    observation = build_qa_observation(sources=[_source("method"), _source("results")])

    assert set(RETRIEVAL_STAGE_NAMES) <= set(observation["retrieval_stage_status"])
    assert set(observation["route_status"]) == set(ROUTE_STAGE_NAMES)
    assert observation["retrieval_stage_status"]["rerank"]["status"] == "unknown"
    assert observation["rerank_failed_reason"] == "not_available"


def test_good_sources_with_passed_verification_need_no_repair() -> None:
    observation = build_qa_observation(
        sources=[_source("method"), _source("results")],
        generation_result={"answer": "answer"},
        verification_result={"status": "passed", "source_count": 2},
    )

    assert observation["retrieval_quality"] == "good"
    assert observation["answer_quality"] == "grounded"
    assert observation["recommended_repair_actions"] == []


def test_single_short_source_recommends_expanded_context() -> None:
    observation = build_qa_observation(
        sources=[_source("unknown", text="short")],
        generation_result={"answer": "answer"},
        verification_result={"status": "warning", "source_count": 1},
    )

    assert observation["retrieval_quality"] == "weak"
    assert observation["weak_source_reason"] == "citation_source_insufficient"
    assert "retry_with_query_rewrite" in observation["recommended_repair_actions"]
    assert "retry_with_expanded_context" in observation["recommended_repair_actions"]
    assert any(item["action"] == "retry_with_expanded_context" for item in observation["repair_action_details"])


def test_noisy_sources_are_partial_quality() -> None:
    observation = build_qa_observation(
        sources=[_source("references"), _source("appendix")],
        generation_result={"answer": "answer"},
        verification_result={"status": "passed", "source_count": 2},
    )

    assert observation["retrieval_quality"] == "partial"
    assert observation["retrieval_quality_reason"] == "context_noise_contamination"


def test_missing_index_recommends_confirmation_rebuild_action_only() -> None:
    observation = build_error_qa_observation(
        error_code="qa_index_not_found",
        error_stage="vector_store",
        error_reason="index_status:missing",
    )

    assert observation["retrieval_quality"] == "failed"
    assert observation["retrieval_stage_status"]["vector"]["status"] == "failed"
    assert "ask_user_to_rebuild_index" in observation["recommended_repair_actions"]
    detail = next(item for item in observation["repair_action_details"] if item["action"] == "ask_user_to_rebuild_index")
    assert detail["input_params"]["requires_confirmation"] is True


def test_research_graph_error_stage_marks_generation_failed() -> None:
    observation = build_error_qa_observation(
        error_code="research_generation_empty",
        error_stage="draft_generation",
        error_reason="research_generation_empty",
    )

    assert observation["retrieval_stage_status"]["generation"]["status"] == "failed"
    assert "generation" in observation["degraded_stages"]
    assert observation["answer_quality"] == "generation_failed"


def test_unmapped_error_stage_only_reported_at_top_level() -> None:
    observation = build_error_qa_observation(
        error_code="unknown_error",
        error_stage="persist_completed_turn",
        error_reason="unknown_error",
    )

    assert observation["error_stage"] == "persist_completed_turn"
    assert observation["degraded_stages"] == []


def test_research_observation_uses_success_for_passed_verification() -> None:
    observation = build_research_qa_observation(
        outcome="completed",
        research_summary={"termination_reason": "core_needs_resolved"},
        sources=[_source("method"), _source("results")],
    )

    assert observation["retrieval_stage_status"]["verification"]["status"] == "success"
    assert observation["answer_quality"] == "grounded"
    assert observation["recommended_repair_actions"] == []
