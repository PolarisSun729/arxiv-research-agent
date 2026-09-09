"""检索指标：按每轮真实 rank 的 top-k 并集计算运行级召回；无标注时不可评分。"""

from typing import Any


def _valid_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for c in candidates if c.get("candidate_id")
            and isinstance(c.get("rank"), int) and not isinstance(c["rank"], bool) and c["rank"] > 0]


def calculate_recall_at_k(expected_chunk_ids, retrieved_candidates, k: int) -> float | None:
    if k <= 0:
        raise ValueError("k 必须为正整数")
    expected = set(expected_chunk_ids or [])
    if not expected:
        return None
    # 多轮 rank 不构成全局排名；用各轮 top-k 的并集，不能合并后再任意截断 k 个。
    found = {c["candidate_id"] for c in _valid_candidates(retrieved_candidates) if c["rank"] <= k}
    return len(expected & found) / len(expected)


def calculate_hit_at_k(expected_chunk_ids, retrieved_candidates, k: int) -> float | None:
    recall = calculate_recall_at_k(expected_chunk_ids, retrieved_candidates, k)
    return float(recall > 0) if recall is not None else None


def calculate_mrr(expected_chunk_ids, retrieved_candidates) -> float | None:
    expected = set(expected_chunk_ids or [])
    if not expected:
        return None
    ranks = [c["rank"] for c in _valid_candidates(retrieved_candidates) if c["candidate_id"] in expected]
    # 轨迹可能因去重而缺少中间项，倒排名必须取原始 rank，不能用列表下标重新编号。
    return 1.0 / min(ranks) if ranks else 0.0


def extract_retrieval_candidates_from_trace(trace_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for event in trace_events:
        if event.get("event_type") != "retrieval_completed":
            continue
        for candidate in _valid_candidates(event.get("candidates") or []):
            candidate_id = candidate["candidate_id"]
            if candidate_id not in candidates or candidate["rank"] < candidates[candidate_id]["rank"]:
                candidates[candidate_id] = dict(candidate)
    return list(candidates.values())


def compute_retrieval_metrics(expected_chunk_ids, trace_events, k_values=None) -> dict[str, float | None]:
    candidates = extract_retrieval_candidates_from_trace(trace_events)
    metrics = {}
    for k in [3, 5, 10] if k_values is None else k_values:
        metrics[f"recall@{k}"] = calculate_recall_at_k(expected_chunk_ids, candidates, k)
        metrics[f"hit@{k}"] = calculate_hit_at_k(expected_chunk_ids, candidates, k)
    metrics["mrr"] = calculate_mrr(expected_chunk_ids, candidates)
    return metrics
