from __future__ import annotations

from tools.multi_index_retrieval_benchmark import (
    SCENARIOS,
    evaluate_case,
    load_cases,
    run_scenario,
)


def test_evaluate_case_reports_recall_and_mrr() -> None:
    metrics = evaluate_case(
        [{"chunk_id": "chunk-a"}, {"chunk_id": "chunk-b"}],
        ["chunk-b"],
    )

    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["gold_rank"] == 2


def test_multi_index_benchmark_outputs_core_metrics_for_single_case() -> None:
    cases = load_cases("en_method_flow")

    result = run_scenario(
        "vector_multi_index",
        SCENARIOS["vector_multi_index"],
        cases,
        enable_rerank=False,
    )

    assert result["case_count"] == 1
    assert "recall@5" in result
    assert "recall@10" in result
    assert "MRR" in result
    assert result["route_hit_distribution"]
    assert result["index_type_contribution"]


def test_benchmark_reports_actual_keyword_backend_for_bm25_scenario() -> None:
    cases = load_cases("en_method_flow")

    result = run_scenario(
        "vector_bm25",
        SCENARIOS["vector_bm25"],
        cases,
        enable_rerank=False,
    )

    assert sum(result["keyword_backend_distribution"].values()) == 1
    assert sum(result["keyword_backend_config_distribution"].values()) == 1
    assert result["cases"][0]["keyword_backend"] in result["keyword_backend_distribution"]
    assert "keyword_backend_init_fallback" in result["cases"][0]
