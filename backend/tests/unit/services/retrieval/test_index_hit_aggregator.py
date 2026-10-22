import unittest

from services.retrieval.index_hit_aggregator import IndexHitAggregator


def _hit(chunk_id: str, index_type: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "matched_index_id": f"{chunk_id}:{index_type}",
        "matched_index_type": index_type,
        "matched_index_score": score,
    }


class IndexHitAggregatorSortTests(unittest.TestCase):
    def test_sorts_by_weighted_score_not_raw_hit_rank(self) -> None:
        # question 原始分更高、排在第 1，但加权后（×0.82）低于排第 2 的 table（×1.12）。
        hits = [
            _hit("a", "question", 0.60),
            _hit("b", "table_or_figure", 0.58),
        ]

        candidates = IndexHitAggregator().aggregate(hits, route_name="vector_original")

        self.assertEqual([item["chunk_id"] for item in candidates], ["b", "a"])
        self.assertEqual([item["route_rank"] for item in candidates], [1, 2])
        self.assertEqual(candidates[0]["index_aggregation_best_rank"], 2)

    def test_equal_route_score_falls_back_to_hit_rank(self) -> None:
        hits = [_hit("a", "body", 0.5), _hit("b", "body", 0.5)]

        candidates = IndexHitAggregator().aggregate(hits, route_name="vector_original")

        self.assertEqual([item["chunk_id"] for item in candidates], ["a", "b"])

    def test_top_k_applies_after_weighted_sort(self) -> None:
        hits = [_hit("a", "question", 0.60), _hit("b", "table_or_figure", 0.58), _hit("c", "body", 0.40)]

        candidates = IndexHitAggregator().aggregate(hits, route_name="vector_original", top_k=1)

        self.assertEqual([item["chunk_id"] for item in candidates], ["b"])


if __name__ == "__main__":
    unittest.main()
