import unittest
import time

from tests.helpers import build_retrieval_service, load_retrieval_modules


class RouteRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.modules = load_retrieval_modules()
        self.service, self.collection_name, *_ = build_retrieval_service()
        self.options = self.modules["enhanced"].RetrievalOptions(debug=True)

    def _build_route_bundle(
        self,
        question: str,
        *,
        enable_query_rewrite: bool = True,
        enable_hyde: bool = True,
        enable_keyword_search: bool = True,
    ):
        query_bundle = self.service.query_planner.build_query_bundle(
            user_query=question,
            collection_name=self.collection_name,
            enable_query_rewrite=enable_query_rewrite,
        )
        return self.service.route_retriever.build_route_bundle(
            collection_name=self.collection_name,
            user_query=question,
            query_profile=query_bundle["query_profile"],
            query_views=query_bundle["query_views"],
            options=self.options,
            enable_hyde=enable_hyde,
            enable_keyword_search=enable_keyword_search,
            recall_candidate_limit=6,
        )

    def test_semantic_route_returns_fixed_method_chunk(self) -> None:
        bundle = self._build_route_bundle("What is the method framework of the paper?", enable_hyde=False, enable_keyword_search=False)
        vector_original = bundle["routes"]["vector_original"]

        self.assertTrue(vector_original)
        self.assertEqual(vector_original[0]["chunk_id"], "chunk-method")
        self.assertEqual(vector_original[0]["retrieval_route"], "vector_original")
        self.assertFalse(bundle["collection_profile"]["profile_cache_hit"])
        self.assertEqual(bundle["collection_profile"]["embedding_provider"], "fake")
        self.assertEqual(bundle["collection_profile"]["embedding_model"], "fake-embedding-model")
        self.assertEqual(bundle["collection_profile"]["vector_dimension"], 3)

    def test_query_rewrite_toggle_controls_vector_rewrite_route(self) -> None:
        enabled = self._build_route_bundle("What is the method framework of the paper?", enable_query_rewrite=True)
        disabled = self._build_route_bundle("What is the method framework of the paper?", enable_query_rewrite=False)

        self.assertTrue(enabled["routes"]["vector_rewrite"])
        self.assertEqual(disabled["routes"]["vector_rewrite"], [])

    def test_hyde_toggle_controls_hyde_route(self) -> None:
        enabled = self._build_route_bundle("Explain the paper method.", enable_hyde=True)
        disabled = self._build_route_bundle("Explain the paper method.", enable_hyde=False)

        self.assertTrue(enabled["hyde_debug"]["enabled"])
        self.assertTrue(enabled["routes"]["vector_hyde"])
        self.assertFalse(disabled["hyde_debug"]["enabled"])
        self.assertEqual(disabled["routes"]["vector_hyde"], [])

    def test_keyword_toggle_controls_keyword_route(self) -> None:
        enabled = self._build_route_bundle("Which dataset benchmark is used?", enable_keyword_search=True)
        disabled = self._build_route_bundle("Which dataset benchmark is used?", enable_keyword_search=False)

        self.assertTrue(enabled["routes"]["keyword"])
        self.assertIn("chunk-dataset", [item["chunk_id"] for item in enabled["routes"]["keyword"]])
        self.assertEqual(disabled["routes"]["keyword"], [])

    def test_keyword_route_returns_fixed_chunks_for_figure_table_query(self) -> None:
        bundle = self._build_route_bundle("What does Figure 2 and Table 3 show?", enable_hyde=False)
        keyword_hits = bundle["routes"]["keyword"]

        self.assertTrue(keyword_hits)
        self.assertIn("chunk-figure-table", [item["chunk_id"] for item in keyword_hits[:3]])
        figure_hit = next(item for item in keyword_hits if item["chunk_id"] == "chunk-figure-table")
        self.assertEqual(figure_hit["chunk_type"], "figure")

    def test_multi_route_rrf_fusion_dedupes_and_preserves_scores(self) -> None:
        query_bundle = self.service.query_planner.build_query_bundle(
            user_query="What does the method framework do?",
            collection_name=self.collection_name,
            enable_query_rewrite=True,
        )
        profile = query_bundle["query_profile"]
        base_chunk = {
            "chunk_id": "chunk-method",
            "parent_chunk_id": "parent-method",
            "content": "Method chunk",
            "source": "paper.pdf",
            "chunk_type": "text",
            "route_score": 0.9,
            "route_confidence": 0.8,
            "source_query": "method",
        }
        routes = {
            "vector_original": [dict(base_chunk, route_score=0.9)],
            "keyword": [dict(base_chunk, route_score=0.7)],
        }

        fused = self.service._fuse_routes(routes, top_k=5, query_profile=profile)

        self.assertEqual(len(fused), 1)
        self.assertEqual(set(fused[0]["matched_routes"]), {"vector_original", "keyword"})
        self.assertIn("vector_original", fused[0]["route_scores"])
        self.assertIn("keyword", fused[0]["route_scores"])
        self.assertGreater(fused[0]["score"], 0.0)

    def test_vector_route_reuses_collection_profile_cache(self) -> None:
        self._build_route_bundle("What is the method framework of the paper?", enable_hyde=False, enable_keyword_search=False)
        self._build_route_bundle("What is the method framework of the paper?", enable_hyde=False, enable_keyword_search=False)

        collection_info_calls = [
            call for call in self.service.vector_store_service.calls
            if call["method"] == "get_collection_info"
        ]
        profile_chunk_reads = [
            call for call in self.service.vector_store_service.calls
            if call["method"] == "get_all_chunks" and call.get("limit") is None
        ]

        self.assertEqual(len(collection_info_calls), 1)
        self.assertEqual(len(profile_chunk_reads), 2)

    def test_keyword_route_uses_cached_keyword_index_without_reloading_chunks(self) -> None:
        first = self._build_route_bundle("Which dataset benchmark is used?", enable_hyde=False, enable_keyword_search=True)
        second = self._build_route_bundle("Which dataset benchmark is used?", enable_hyde=False, enable_keyword_search=True)

        full_chunk_reads = [
            call for call in self.service.vector_store_service.calls
            if call["method"] == "get_all_chunks" and call.get("limit") is None
        ]

        self.assertTrue(first["routes"]["keyword"])
        self.assertTrue(second["keyword_debug"]["keyword_index_hit"])
        self.assertFalse(second["keyword_debug"]["keyword_full_scan_used"])
        self.assertEqual(len(full_chunk_reads), 2)

    def test_memory_route_exact_lookup_by_source_id_without_full_scan(self) -> None:
        self.service.memory_runtime_config["enable_memory_aware_retrieval"] = True
        self.options.memory_context = {
            "enabled": True,
            "reason": "follow-up referenced prior source",
            "query_keywords": ["method"],
            "referenced_source_ids": ["chunk-method"],
            "candidates": [
                {
                    "source_id": "chunk-method",
                    "content_preview": "Method source",
                    "memory_reason": "User referred to this source.",
                    "reference_strength": 1.0,
                }
            ],
        }

        bundle = self._build_route_bundle("Explain this method again.", enable_hyde=False, enable_keyword_search=False)
        memory_hits = bundle["routes"]["memory_context"]

        self.assertTrue(memory_hits)
        self.assertEqual(memory_hits[0]["chunk_id"], "chunk-method")
        self.assertEqual(bundle["memory_debug"]["memory_lookup_mode"], "source_id")
        self.assertGreaterEqual(bundle["memory_debug"]["memory_exact_hit_count"], 1)
        self.assertFalse(bundle["memory_debug"]["memory_fallback_used"])
        self.assertFalse(bundle["memory_debug"]["memory_full_scan_used"])

    def test_collection_profile_rebuilds_when_index_record_changes(self) -> None:
        provider = self.service.collection_profile_provider
        first_profile = provider.get_profile(
            self.collection_name,
            index_record={"chunk_count": 8, "embedding_model": "fake-embedding-model"},
        )
        refreshed_profile = provider.get_profile(
            self.collection_name,
            index_record={"chunk_count": 9, "embedding_model": "fake-embedding-model-v2"},
        )

        self.assertFalse(first_profile.cache_hit)
        self.assertFalse(refreshed_profile.cache_hit)
        self.assertEqual(refreshed_profile.chunk_count, 9)
        self.assertEqual(refreshed_profile.embedding_model, "fake-embedding-model-v2")
        self.assertIn("rebuilt_after_stale", refreshed_profile.fallback_reason)

    def test_vector_routes_share_one_batch_embedding_call(self) -> None:
        question = "What is the method framework and dataset setup?"
        query_bundle = self.service.query_planner.build_query_bundle(
            user_query=question,
            collection_name=self.collection_name,
            enable_query_rewrite=True,
        )
        bundle = self._build_route_bundle(
            question,
            enable_query_rewrite=True,
            enable_hyde=True,
            enable_keyword_search=False,
        )
        embedding_calls = [
            call for call in self.service.embedding_service.calls
            if call["method"] == "create_text_embeddings_with_usage"
        ]

        self.assertEqual(len(embedding_calls), 1)
        self.assertGreaterEqual(bundle["embedding_batch"]["batch_size"], 2)
        self.assertEqual(bundle["embedding_batch"]["status"], "ok")
        self.assertIn(query_bundle["query_profile"].semantic_query, embedding_calls[0]["texts"])
        self.assertIn(query_bundle["query_profile"].evidence_query, embedding_calls[0]["texts"])
        self.assertIn("vector_original", bundle["routes"])

    def test_enhancement_route_timeout_is_recorded_without_blocking_pipeline(self) -> None:
        self.service.route_retriever.route_executor.timeouts["keyword"] = 0.01
        original_keyword_retrieve = self.service.route_retriever.keyword_retrieve

        def slow_keyword_retrieve(*args, **kwargs):
            time.sleep(0.2)
            return original_keyword_retrieve(*args, **kwargs)

        self.service.route_retriever.keyword_retrieve = slow_keyword_retrieve
        started = time.perf_counter()
        bundle = self._build_route_bundle(
            "Which dataset benchmark is used?",
            enable_hyde=False,
            enable_keyword_search=True,
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.18)
        self.assertEqual(bundle["routes"]["keyword"], [])
        self.assertEqual(bundle["route_metrics"]["keyword"]["status"], "timeout")
        self.assertTrue(bundle["route_metrics"]["keyword"]["timeout"])
        self.assertIn("timed out", bundle["route_metrics"]["keyword"]["fallback_reason"])
        self.service.route_retriever.route_executor.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
