"""检索指标使用原始轮内 rank；去重、多轮和未标注样本有明确口径。"""

import pytest

from services.evaluation.metrics_retrieval import calculate_mrr, calculate_recall_at_k, compute_retrieval_metrics


def test_actual_trace_event_type_and_multiround_union():
    events = [
        {"event_type": "retrieval_completed", "candidates": [
            {"candidate_id": "a", "rank": 1}, {"candidate_id": "b", "rank": 8},
        ]},
        {"event_type": "retrieval_completed", "candidates": [
            {"candidate_id": "b", "rank": 2}, {"candidate_id": "c", "rank": 4},
        ]},
    ]
    metrics = compute_retrieval_metrics(["a", "b", "c"], events, [3])
    assert metrics == {"recall@3": pytest.approx(2 / 3), "hit@3": 1.0, "mrr": 1.0}


def test_mrr_does_not_renumber_after_filtering():
    candidates = [{"candidate_id": "hit", "rank": 7}]
    assert calculate_mrr(["hit"], candidates) == pytest.approx(1 / 7)
    assert calculate_recall_at_k(["hit"], candidates, 5) == 0.0


@pytest.mark.parametrize("expected", [None, []])
def test_unlabeled_or_no_relevant_evidence_has_no_retrieval_score(expected):
    assert all(value is None for value in compute_retrieval_metrics(expected, []).values())


def test_annotated_no_hit_scores_zero():
    assert all(value == 0.0 for value in compute_retrieval_metrics(["expected"], []).values())


@pytest.mark.parametrize("rank", [0, -1, True, None])
def test_invalid_rank_cannot_create_a_hit(rank):
    assert calculate_mrr(["expected"], [{"candidate_id": "expected", "rank": rank}]) == 0.0


def test_nonpositive_k_is_rejected():
    with pytest.raises(ValueError):
        calculate_recall_at_k(["expected"], [], 0)
