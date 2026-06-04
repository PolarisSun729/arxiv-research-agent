import unittest

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


if __name__ == "__main__":
    unittest.main()
