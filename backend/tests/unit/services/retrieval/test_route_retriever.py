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
        enable_table_structured_route: bool = True,
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
            enable_table_structured_route=enable_table_structured_route,
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

    def test_vector_route_aggregates_multiple_index_hits_to_one_chunk_candidate(self) -> None:
        base_metadata = {
            "chunk_id": "chunk-method",
            "parent_chunk_id": "parent-method",
            "original_chunk_id": "parent-method",
            "page_number": 2,
            "page_range": "2",
            "section_title": "Method",
            "section_path": "2 Method",
            "source": "paper.pdf",
            "chunk_type": "text",
            "order_index": 1,
        }
        collection_rows = self.service.vector_store_service.collections[self.collection_name]
        for suffix, index_type, index_text, index_weight in (
            ("question", "question", "What method framework pipeline is used?", 0.82),
            ("summary", "summary", "The method framework uses a retrieval pipeline.", 0.78),
        ):
            metadata = dict(
                base_metadata,
                retrieval_index_id=f"chunk-method:{suffix}:1",
                retrieval_index_type=index_type,
                retrieval_index_text=index_text,
                retrieval_index_weight=index_weight,
                retrieval_index_enabled_routes=["vector_original"],
            )
            collection_rows.append(
                {
                    "id": len(collection_rows) + 1,
                    "content": "Method section: the framework uses a retrieval pipeline with two encoder stages.",
                    "embedding": self.service.embedding_service.create_single_embedding(index_text),
                    "metadata": metadata,
                }
            )

        bundle = self._build_route_bundle(
            "What is the method framework of the paper?",
            enable_hyde=False,
            enable_keyword_search=False,
        )
        method_hits = [
            item for item in bundle["routes"]["vector_original"]
            if item["chunk_id"] == "chunk-method"
        ]

        self.assertEqual(len(method_hits), 1)
        hit = method_hits[0]
        self.assertTrue(hit["index_aggregation_applied"])
        self.assertLessEqual(len(hit["matched_indexes"]), 3)
        self.assertIn("best_matched_index", hit)
        self.assertIn("body", hit["matched_index_types"])
        self.assertTrue({"question", "summary"} & set(hit["matched_index_types"]))
        self.assertIn("What is the method framework of the paper?", hit["source_queries"])

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
        self.assertTrue(figure_hit["keyword_hit_asset_field"])

    def test_keyword_route_expands_chinese_method_query_and_reports_fields(self) -> None:
        bundle = self._build_route_bundle("这篇论文的方法流程是怎样的？", enable_hyde=False)
        keyword_debug = bundle["keyword_debug"]
        keyword_hits = bundle["routes"]["keyword"]

        self.assertTrue(keyword_hits)
        self.assertEqual(keyword_hits[0]["chunk_id"], "chunk-method")
        self.assertIn("method", keyword_debug["expanded_tokens"])
        self.assertIn("pipeline", keyword_debug["expanded_tokens"])
        self.assertIn("body", keyword_hits[0]["keyword_match_fields"])
        self.assertFalse(keyword_hits[0]["keyword_hit_asset_field"])
        self.assertEqual(keyword_debug["matched_chunks"][0]["chunk_id"], "chunk-method")
        self.assertIn("body", keyword_debug["matched_chunks"][0]["matched_fields"])

    def test_keyword_route_expands_english_plural_dataset_query(self) -> None:
        bundle = self._build_route_bundle("What datasets are used?", enable_hyde=False)
        keyword_debug = bundle["keyword_debug"]
        keyword_hits = bundle["routes"]["keyword"]

        self.assertTrue(keyword_hits)
        self.assertEqual(keyword_hits[0]["chunk_id"], "chunk-dataset")
        self.assertIn("dataset", keyword_debug["expanded_tokens"])
        self.assertIn("corpus", keyword_debug["expanded_tokens"])

    def test_keyword_route_downweights_asset_chunks_for_non_figure_query(self) -> None:
        bundle = self._build_route_bundle("What is the method pipeline?", enable_hyde=False)
        keyword_hits = bundle["routes"]["keyword"]

        self.assertTrue(keyword_hits)
        self.assertEqual(keyword_hits[0]["chunk_id"], "chunk-method")
        top_asset_hits = [item for item in keyword_hits[:2] if item.get("chunk_type") in {"figure", "table"}]
        self.assertEqual(top_asset_hits, [])

    def test_keyword_route_debug_contains_query_view_contributions(self) -> None:
        bundle = self._build_route_bundle("What datasets are used?", enable_hyde=False)
        keyword_debug = bundle["keyword_debug"]
        top_hit = bundle["routes"]["keyword"][0]

        self.assertEqual(top_hit["chunk_id"], "chunk-dataset")
        self.assertTrue(keyword_debug["query_views"])
        self.assertEqual(keyword_debug["keyword_fusion_strategy"], "weighted_rrf")
        self.assertTrue(keyword_debug["query_contributions"])
        self.assertIn("view_contributions", keyword_debug["matched_chunks"][0])
        self.assertIn("rank", keyword_debug["matched_chunks"][0]["view_contributions"][0])
        self.assertIn("rrf_vote", keyword_debug["matched_chunks"][0]["view_contributions"][0])
        self.assertIn("keyword_query_contributions", top_hit)
        self.assertGreater(top_hit["route_score"], 0.0)
        self.assertTrue(top_hit["index_aggregation_applied"])
        self.assertIn("best_matched_index", top_hit)
        self.assertTrue(top_hit["matched_indexes"])

    def test_keyword_backend_aggregates_index_hits_to_chunk_candidates(self) -> None:
        from services.retrieval.keyword_backend import InternalBM25Backend
        from services.retrieval.retrieval_index import CollectionRetrievalIndexProvider

        source_chunk = {
            "content": "The final evidence chunk discusses training details.",
            "chunk_id": "chunk-indexed-method",
            "parent_chunk_id": "parent-indexed-method",
            "chunk_type": "text",
            "section_title": "Training",
            "section_path": "3 Training",
            "source": "paper.pdf",
        }
        rows = [
            {
                **source_chunk,
                "retrieval_index_id": "chunk-indexed-method:question:1",
                "retrieval_index_type": "question",
                "retrieval_index_text": "Which contrastive alignment objective is used?",
                "retrieval_index_weight": 0.82,
                "retrieval_index_enabled_routes": ["keyword"],
            },
            {
                **source_chunk,
                "retrieval_index_id": "chunk-indexed-method:summary:1",
                "retrieval_index_type": "summary",
                "retrieval_index_text": "The section explains the contrastive alignment objective for training.",
                "retrieval_index_weight": 0.78,
                "retrieval_index_enabled_routes": ["keyword"],
            },
            {
                "content": "A different chunk mentions evaluation only.",
                "chunk_id": "chunk-other",
                "parent_chunk_id": "parent-other",
                "chunk_type": "text",
                "section_title": "Evaluation",
                "section_path": "4 Evaluation",
                "source": "paper.pdf",
                "retrieval_index_id": "chunk-other:body:1",
                "retrieval_index_type": "body",
                "retrieval_index_text": "Evaluation baseline metrics.",
                "retrieval_index_weight": 1.0,
                "retrieval_index_enabled_routes": ["keyword"],
            },
        ]

        class Store:
            def get_all_chunks(self, _collection_name: str):
                return list(rows)

        provider = CollectionRetrievalIndexProvider(
            vector_store_service=Store(),
            chunk_normalizer=self.service._normalize_chunk,
            tokenizer=self.service._tokenize_for_keyword_search,
        )
        retrieval_index = provider.get_index("paper_collection")
        query_bundle = self.service.query_planner.build_query_bundle(
            user_query="Which contrastive alignment objective is used?",
            collection_name=self.collection_name,
            enable_query_rewrite=False,
        )
        backend = InternalBM25Backend(
            query_tools=self.service.query_planner,
            route_confidence_builder=self.service._route_confidence,
            structural_bonus_builder=self.service._compute_structural_bonus,
            fusion_service=self.service.fusion_service,
        )

        payload = backend._internal_keyword_retrieve(
            query_views=[
                {
                    "view_id": "original:0",
                    "query": "Which contrastive alignment objective is used?",
                    "source": "original",
                    "source_index": 0,
                    "selected": True,
                    "reason": "kept",
                    "weight": 1.0,
                }
            ],
            top_k=5,
            query_profile=query_bundle["query_profile"],
            retrieval_index=retrieval_index,
        )

        hits = payload["results"]
        indexed_hits = [hit for hit in hits if hit["chunk_id"] == "chunk-indexed-method"]
        self.assertEqual(len(indexed_hits), 1)
        hit = indexed_hits[0]
        self.assertEqual(hit["content"], source_chunk["content"])
        self.assertEqual(hit["keyword_matched_index_count"], 2)
        self.assertEqual(
            {item["matched_index_id"] for item in hit["matched_indexes"]},
            {"chunk-indexed-method:question:1", "chunk-indexed-method:summary:1"},
        )
        deduped = self.service.fusion_service.dedupe_route_results([hit])
        self.assertEqual(
            {item["matched_index_id"] for item in deduped[0]["matched_indexes"]},
            {"chunk-indexed-method:question:1", "chunk-indexed-method:summary:1"},
        )
        matched_debug = payload["debug"]["matched_chunks"][0]
        self.assertEqual(matched_debug["chunk_id"], "chunk-indexed-method")
        self.assertEqual(matched_debug["matched_index_count"], 2)
        self.assertTrue(matched_debug["top_matched_indexes"])
        self.assertIn("query_sources", matched_debug["top_matched_indexes"][0])

    def test_keyword_route_dedupes_high_similarity_query_views(self) -> None:
        query_bundle = self.service.query_planner.build_query_bundle(
            user_query="What is the method pipeline?",
            collection_name=self.collection_name,
            enable_query_rewrite=True,
        )
        query_bundle["query_views"]["selected_queries"] = [
            "method pipeline framework",
            "framework method pipeline",
            "method pipeline framework component",
        ]
        bundle = self.service.route_retriever.build_route_bundle(
            collection_name=self.collection_name,
            user_query="What is the method pipeline?",
            query_profile=query_bundle["query_profile"],
            query_views=query_bundle["query_views"],
            options=self.options,
            enable_hyde=False,
            enable_keyword_search=True,
            enable_table_structured_route=False,
            recall_candidate_limit=6,
        )
        query_views = bundle["keyword_debug"]["query_views"]
        rewrite_views = [item for item in query_views if item["source"] == "rewrite"]

        self.assertLessEqual(len(rewrite_views), 2)
        merged_views = [
            merged
            for item in query_views
            for merged in item.get("merged_views", [])
        ]
        self.assertTrue(merged_views)
        self.assertTrue(any(item["reason"].startswith("high_similarity") for item in merged_views))

    def test_keyword_route_rewrite_count_does_not_linearly_boost_keyword_score(self) -> None:
        query_bundle = self.service.query_planner.build_query_bundle(
            user_query="What is the method pipeline?",
            collection_name=self.collection_name,
            enable_query_rewrite=True,
        )
        sparse_views = dict(query_bundle["query_views"])
        sparse_views["selected_queries"] = ["method pipeline framework"]
        dense_views = dict(query_bundle["query_views"])
        dense_views["selected_queries"] = [
            "method pipeline framework",
            "method pipeline architecture",
            "framework method pipeline",
            "method approach pipeline",
        ]

        sparse_bundle = self.service.route_retriever.build_route_bundle(
            collection_name=self.collection_name,
            user_query="What is the method pipeline?",
            query_profile=query_bundle["query_profile"],
            query_views=sparse_views,
            options=self.options,
            enable_hyde=False,
            enable_keyword_search=True,
            enable_table_structured_route=False,
            recall_candidate_limit=6,
        )
        dense_bundle = self.service.route_retriever.build_route_bundle(
            collection_name=self.collection_name,
            user_query="What is the method pipeline?",
            query_profile=query_bundle["query_profile"],
            query_views=dense_views,
            options=self.options,
            enable_hyde=False,
            enable_keyword_search=True,
            enable_table_structured_route=False,
            recall_candidate_limit=6,
        )

        sparse_score = sparse_bundle["routes"]["keyword"][0]["route_score"]
        dense_score = dense_bundle["routes"]["keyword"][0]["route_score"]
        self.assertEqual(sparse_bundle["routes"]["keyword"][0]["chunk_id"], "chunk-method")
        self.assertEqual(dense_bundle["routes"]["keyword"][0]["chunk_id"], "chunk-method")
        self.assertLess(dense_score, sparse_score * 1.35)

    def test_keyword_result_exposes_bm25_matched_terms_with_idf_and_fields(self) -> None:
        bundle = self._build_route_bundle("Which datasets and benchmark corpus are used?", enable_hyde=False)
        top_hit = bundle["routes"]["keyword"][0]

        self.assertEqual(top_hit["chunk_id"], "chunk-dataset")
        self.assertGreater(top_hit["bm25_raw_score"], 0.0)
        self.assertGreater(top_hit["bm25_fused_score"], 0.0)
        matched_terms = top_hit["keyword_matched_terms"]
        self.assertTrue(matched_terms)
        tokens = {term["token"] for term in matched_terms}
        self.assertIn("dataset", tokens)
        for term in matched_terms:
            self.assertIn("idf", term)
            self.assertIn("fields", term)
            self.assertGreaterEqual(term["idf"], 0.0)
        dataset_term = next(term for term in matched_terms if term["token"] == "dataset")
        self.assertIn("body", dataset_term["fields"])
        self.assertTrue(top_hit["keyword_query_sources"])
        self.assertEqual(top_hit["keyword_noise_flags"], [])

    def test_keyword_debug_matched_chunks_carry_terms_and_noise_flags(self) -> None:
        bundle = self._build_route_bundle("Which datasets and benchmark corpus are used?", enable_hyde=False)
        matched = bundle["keyword_debug"]["matched_chunks"][0]

        self.assertEqual(matched["chunk_id"], "chunk-dataset")
        self.assertIn("matched_terms", matched)
        self.assertIn("query_sources", matched)
        self.assertIn("noise_flags", matched)
        self.assertIn("base_route_confidence", bundle["keyword_debug"])
        term = matched["matched_terms"][0]
        self.assertIn("token", term)
        self.assertIn("idf", term)

    def test_keyword_confidence_downweighted_for_overview_versus_method(self) -> None:
        overview = self._build_route_bundle(
            "Give an overview and summary of the paper's main contributions.",
            enable_hyde=False,
        )
        method = self._build_route_bundle("What is the method pipeline and framework?", enable_hyde=False)

        overview_conf = overview["routes"]["keyword"][0]["route_confidence"]
        method_conf = method["routes"]["keyword"][0]["route_confidence"]
        overview_base = overview["routes"]["keyword"][0]["keyword_base_route_confidence"]
        method_base = method["routes"]["keyword"][0]["keyword_base_route_confidence"]

        # overview 问题降权后 keyword confidence 应低于其自身基线，并低于 method 问题。
        self.assertLess(overview_conf, overview_base)
        self.assertGreaterEqual(method_conf, method_base)
        self.assertLess(overview_conf, method_conf)

    def test_keyword_confidence_drops_when_only_low_idf_token_matches(self) -> None:
        bundle = self._build_route_bundle(
            "Give an overview and summary of the paper's main contributions.",
            enable_hyde=False,
        )
        keyword_hits = bundle["routes"]["keyword"]
        section_only = [
            hit
            for hit in keyword_hits
            if len(hit["keyword_matched_terms"]) == 1 and hit["keyword_matched_terms"][0]["idf"] < 1.2
        ]
        self.assertTrue(section_only, "expected at least one chunk matching only a low-idf generic token")
        weak_hit = section_only[0]
        full_hit = next(hit for hit in keyword_hits if len(hit["keyword_matched_terms"]) > 1)
        self.assertLess(weak_hit["route_confidence"], full_hit["route_confidence"])

    def test_keyword_bm25_does_not_dominate_final_context_for_overview(self) -> None:
        result = self.service.enhanced_retrieve(
            user_query="Give an overview and summary of the paper's main contributions.",
            collection_name=self.collection_name,
            options=self.modules["enhanced"].RetrievalOptions(debug=True, enable_llm_rerank=False),
        )
        stages = result["debug"]["stages"]
        final_routes = [item.get("retrieval_route") for item in stages["final_context_top15"]]
        self.assertTrue(final_routes)
        # overview 问题里 BM25-only 命中不应主导最终上下文，向量召回必须保留主导地位。
        self.assertGreater(
            sum(1 for route in final_routes if route and route.startswith("vector")),
            sum(1 for route in final_routes if route == "keyword"),
        )

    def test_keyword_method_question_supplements_vector_with_exact_term_chunk(self) -> None:
        bundle = self._build_route_bundle("What is the method pipeline and framework?", enable_hyde=False)
        keyword_ids = [hit["chunk_id"] for hit in bundle["routes"]["keyword"]]

        self.assertIn("chunk-method", keyword_ids)
        method_hit = next(hit for hit in bundle["routes"]["keyword"] if hit["chunk_id"] == "chunk-method")
        method_tokens = {term["token"] for term in method_hit["keyword_matched_terms"]}
        self.assertTrue({"method", "pipeline", "framework"} & method_tokens)

    def test_table_structured_route_hits_specific_max_cell_for_metric_question(self) -> None:
        bundle = self._build_route_bundle(
            "表 2 中最高的 accuracy 是多少？",
            enable_hyde=False,
            enable_keyword_search=False,
            enable_table_structured_route=True,
        )

        table_hits = bundle["routes"]["table_structured"]
        self.assertTrue(table_hits)
        top_hit = table_hits[0]
        self.assertEqual(top_hit["chunk_id"], "chunk-table-results")
        self.assertEqual(top_hit["retrieval_route"], "table_structured")
        self.assertEqual(top_hit["table_structured_evidence"]["numeric_operation"], "max")
        self.assertEqual(top_hit["table_structured_evidence"]["matched_rows"], ["Ours"])
        self.assertEqual(top_hit["table_structured_evidence"]["matched_columns"], ["Accuracy"])
        self.assertEqual(top_hit["table_structured_evidence"]["matched_cells"][0]["raw_value"], "89.2%")
        self.assertTrue(bundle["table_structured_debug"]["enabled"])
        self.assertEqual(bundle["table_structured_debug"]["matched_tables"][0]["table_id"], "paper-table-2")

    def test_table_structured_route_computes_difference_for_ablation_question(self) -> None:
        bundle = self._build_route_bundle(
            "ablation 里去掉 memory 后下降多少？",
            enable_hyde=False,
            enable_keyword_search=False,
            enable_table_structured_route=True,
        )

        table_hits = bundle["routes"]["table_structured"]
        self.assertTrue(table_hits)
        evidence = table_hits[0]["table_structured_evidence"]
        self.assertEqual(evidence["numeric_operation"], "difference")
        self.assertEqual(evidence["matched_rows"], ["w/o memory", "Ours"])
        self.assertEqual(evidence["matched_columns"], ["Accuracy"])
        self.assertAlmostEqual(float(evidence["computed_value"]), 0.031, places=6)
        self.assertEqual(len(evidence["matched_cells"]), 2)

    def test_table_structured_route_prefers_structured_evidence_in_fusion(self) -> None:
        bundle = self._build_route_bundle(
            "表 2 中最高的 accuracy 是多少？",
            enable_hyde=False,
            enable_keyword_search=False,
            enable_table_structured_route=True,
        )

        fused = self.service._fuse_routes(
            bundle["routes"],
            top_k=6,
            query_profile=self.service.query_planner.build_query_bundle(
                user_query="表 2 中最高的 accuracy 是多少？",
                collection_name=self.collection_name,
                enable_query_rewrite=True,
            )["query_profile"],
        )

        self.assertEqual(fused[0]["chunk_id"], "chunk-table-results")
        self.assertIn("table_structured", fused[0]["matched_routes"])

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
            "vector_original": [
                dict(
                    base_chunk,
                    route_score=0.9,
                    matched_index_id="chunk-method:body:1",
                    matched_index_type="body",
                    matched_index_text="Method chunk",
                )
            ],
            "keyword": [
                dict(
                    base_chunk,
                    route_score=0.7,
                    matched_index_id="chunk-method:question:1",
                    matched_index_type="question",
                    matched_index_text="What does the method do?",
                )
            ],
        }

        fused = self.service._fuse_routes(routes, top_k=5, query_profile=profile)

        self.assertEqual(len(fused), 1)
        self.assertEqual(set(fused[0]["matched_routes"]), {"vector_original", "keyword"})
        self.assertIn("vector_original", fused[0]["route_scores"])
        self.assertIn("keyword", fused[0]["route_scores"])
        self.assertEqual(
            {item["matched_index_id"] for item in fused[0]["matched_indexes"]},
            {"chunk-method:body:1", "chunk-method:question:1"},
        )
        self.assertEqual(fused[0]["matched_index_id"], "chunk-method:body:1")
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
