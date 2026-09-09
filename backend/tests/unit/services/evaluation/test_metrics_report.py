"""报告同时暴露样本分母、运行失败和重复波动，并限制基线比较条件。"""

from copy import deepcopy

import pytest

from services.evaluation.metrics_report import bucket_by_field, compute_diff, compute_repeat_stats, generate_report


def _case(case_id, metrics, *, failed=False):
    return {
        "golden_case": {"case_id": case_id, "difficulty": "easy", "answerable": True},
        "metrics": metrics, "run_metrics": [metrics],
        "raw_runs": [{"run_status": "error" if failed else "success", "outcome": None if failed else "completed"}],
    }


def test_failure_stays_in_accuracy_denominator():
    records = [_case("ok", {"three_state_accuracy": 1.0}), _case("bad", {"three_state_accuracy": 0.0}, failed=True)]
    report = generate_report(records)
    assert report["overall_metrics"]["three_state_accuracy"] == 0.5
    assert report["run_summary"]["run_failure_rate"] == 0.5
    assert report["run_summary"]["successful_runs"] == 1
    assert report["metric_counts"]["three_state_accuracy"]["failed_scored_runs"] == 1
    assert report["metric_counts"]["three_state_accuracy"]["scored_cases"] == 2
    assert report["outcome_matrix"]["answerable"]["error"] == 1
    assert not report["baseline_ready"]


def test_legacy_empty_metrics_do_not_disappear():
    report = generate_report([_case("ok", {"three_state_accuracy": 1.0}), _case("bad", {}, failed=True)])
    assert report["overall_metrics"]["three_state_accuracy"] == 0.5
    assert "run_failures" in report["red_flags"]


def test_missing_annotations_are_not_zero_quality():
    report = generate_report([_case("unlabeled", {"three_state_accuracy": None})])
    assert report["overall_metrics"]["three_state_accuracy"] is None
    assert report["metric_counts"]["three_state_accuracy"]["unscored_cases"] == 1


def test_repeat_stats_include_missing_count_and_negative_gain():
    stats = compute_repeat_stats([{"gain": 1.0}, {"gain": None}, {"gain": -1.0}])["gain"]
    assert stats["median"] == 0.0
    assert stats["std"] > 0
    assert stats["valid_count"] == 2
    assert stats["missing_count"] == 1


def test_bucket_means_and_counts():
    grouped = bucket_by_field([_case("a", {"mrr": 1.0}), _case("b", {"mrr": 0.0})], "difficulty")
    assert grouped["easy"]["count"] == 2
    assert grouped["easy"]["metrics"]["mrr"] == 0.5


def test_diff_threshold_and_comparability():
    diff, flags = compute_diff({"recall@5": 0.7}, {"recall@5": 0.8})
    assert diff["recall@5"] == pytest.approx(-0.1)
    assert flags == ["recall@5"]
    assert compute_diff({"recall@5": 0.75}, {"recall@5": 0.8})[1] == []
    previous = generate_report([_configured_case(0.8)], metadata={"dataset_hash": "same", "repeat_count": 3})
    records = [_configured_case(0.7)]
    assert generate_report(records, previous, metadata={"dataset_hash": "changed"})["diff"] == {}
    report = generate_report(records, previous, metadata={"dataset_hash": "same", "repeat_count": 3})
    assert report["comparison_status"] == "comparable"
    assert "recall@5" in report["red_flags"]


def _configured_case(score=1.0):
    record = _case("a", {"recall@5": score})
    run = record["raw_runs"][0]
    run.update({
        "configuration": {"research_limits": {"max_retrievals": 3}, "engine": {
            "generation": {"provider": "offline", "models": {"default": "fixed-v1"}},
            "retrieval": {"top_k": 8, "rerank_model": "fixed-rerank-v1"},
        }},
        "paper_context": {"arxiv_id": "offline", "indexes": [{
            "collection_name": "offline", "active_build_id": "build-v1", "active_index_version": "v1",
            "embedding_model": "fixed-embedding-v1", "sparse_index_source_hash": "source-v1",
        }]},
    })
    record["raw_runs"] = [deepcopy(run) for _ in range(3)]
    record["run_metrics"] *= 3
    return record


@pytest.mark.parametrize("change", ["limits", "model", "index", "missing"])
def test_baseline_diff_rejects_changed_or_unknown_run_configuration(change):
    metadata = {"dataset_hash": "same", "repeat_count": 3}
    previous = generate_report([_configured_case()], metadata=metadata)
    current = _configured_case(0.5)
    for run in current["raw_runs"]:
        if change == "limits":
            run["configuration"]["research_limits"]["max_retrievals"] = 2
        elif change == "model":
            run["configuration"]["engine"]["generation"]["models"]["default"] = "fixed-v2"
        elif change == "index":
            run["paper_context"]["indexes"][0]["active_build_id"] = "build-v2"
        else:
            run.pop("configuration")
    report = generate_report([current], previous, metadata=metadata)
    assert report["comparison_status"] != "comparable"
    assert report["diff"] == {}
    if change == "missing":
        assert not report["baseline_ready"]


def test_configuration_changes_between_repeats_disable_baseline():
    record = _configured_case()
    record["raw_runs"][1]["paper_context"]["indexes"][0]["active_build_id"] = "new-index"
    assert not generate_report([record])["baseline_ready"]
