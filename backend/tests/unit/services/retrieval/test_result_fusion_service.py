import unittest

from services.retrieval.contracts import QueryProfile
from services.retrieval.result_fusion_service import ResultFusionService
from services.retrieval.trace_builder import RetrievalTraceBuilder


def _chunk(route: str, **extra):
    return {
        "chunk_id": 2,
        "original_chunk_id": 2,
        "chunk_type": "text",
        "source": "paper.pdf",
        "page_range": "1-1",
        "order_index": 0,
        "content": "chunk body",
        "retrieval_route": route,
        "route_score": 0.5,
        "route_confidence": 1.0,
        **extra,
    }


def _query_profile():
    return QueryProfile(
        original_query="q", normalized_query="q", language="en", intent_profile=None, tokens=[], keywords=[],
        intent_tags=[], ambiguity_score=0.0, section_preferences=[],
    )


class ChunkUniqueKeyTests(unittest.TestCase):
    def test_empty_part_label_falls_back_to_subchunk_label(self) -> None:
        # 关键词索引从 chunk 文件构建，只有 subchunk_label；向量库入库时已把它写进 content_part_label。
        keyword_hit = _chunk("keyword", content_part_label="", subchunk_label="chunk 2 part 1/1")
        vector_hit = _chunk("vector_original", content_part_label="chunk 2 part 1/1", subchunk_label="chunk 2 part 1/1")

        self.assertEqual(ResultFusionService.chunk_unique_key(keyword_hit), ResultFusionService.chunk_unique_key(vector_hit))

    def test_parts_of_same_chunk_keep_distinct_keys(self) -> None:
        first = _chunk("keyword", content_part_label="", subchunk_label="chunk 2 part 1/2")
        second = _chunk("vector_original", content_part_label="chunk 2 part 2/2")

        self.assertNotEqual(ResultFusionService.chunk_unique_key(first), ResultFusionService.chunk_unique_key(second))

    def test_trace_builder_uses_same_key(self) -> None:
        item = _chunk("keyword", content_part_label="", subchunk_label="chunk 2 part 1/1")

        self.assertEqual(RetrievalTraceBuilder.chunk_unique_key(item), ResultFusionService.chunk_unique_key(item))

    def test_keyword_and_vector_hits_of_same_chunk_fuse_into_one_row(self) -> None:
        fusion = ResultFusionService(rrf_k=60, route_weights={"keyword": 1.0, "vector_original": 1.0})
        routes = {
            "vector_original": [_chunk("vector_original", content_part_label="chunk 2 part 1/1", subchunk_label="chunk 2 part 1/1")],
            "keyword": [_chunk("keyword", content_part_label="", subchunk_label="chunk 2 part 1/1")],
        }

        result = fusion.fuse(routes, fused_limit=10, raw_limit=10, fused_stage_limit=10, query_profile=_query_profile())

        self.assertEqual(len(result.fused_results), 1)
        self.assertEqual(set(result.fused_results[0]["rrf_votes"]), {"vector_original", "keyword"})


if __name__ == "__main__":
    unittest.main()
