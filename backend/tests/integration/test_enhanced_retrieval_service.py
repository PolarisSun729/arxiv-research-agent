import unittest
from unittest import mock
from pathlib import Path

from tests.helpers import build_retrieval_service, build_sample_chunks
from services.paper_qa.context_pack_builder import ContextPackBuilder
from services.retrieval.contracts import RetrievalOptions


def build_context_expansion_chunks():
    return [
        {
            "content": "Method opening: the approach has three stages and this paragraph introduces the pipeline.",
            "metadata": {
                "chunk_id": "method-parent-1",
                "parent_chunk_id": "method-parent",
                "subchunk_index": 1,
                "subchunk_count": 3,
                "page_number": 2,
                "page_range": "2",
                "section_title": "Method",
                "section_path": "2 Method",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 1,
            },
        },
        {
            "content": "Method anchor: the retrieval pipeline first builds query views and then runs route fusion.",
            "metadata": {
                "chunk_id": "method-parent-2",
                "parent_chunk_id": "method-parent",
                "subchunk_index": 2,
                "subchunk_count": 3,
                "page_number": 2,
                "page_range": "2",
                "section_title": "Method",
                "section_path": "2 Method",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 2,
            },
        },
        {
            "content": "Method continuation: the final stage reranks candidates and prepares answer context.",
            "metadata": {
                "chunk_id": "method-parent-3",
                "parent_chunk_id": "method-parent",
                "subchunk_index": 3,
                "subchunk_count": 3,
                "page_number": 3,
                "page_range": "3",
                "section_title": "Method",
                "section_path": "2 Method",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 3,
            },
        },
        {
            "content": "Figure 1 describes the method pipeline with query rewrite, fusion, rerank, and context expansion.",
            "metadata": {
                "chunk_id": "method-figure",
                "parent_chunk_id": "method-figure",
                "subchunk_index": 1,
                "subchunk_count": 1,
                "page_number": 3,
                "page_range": "3",
                "section_title": "Method",
                "section_path": "2 Method",
                "source": "paper.pdf",
                "chunk_type": "figure",
                "asset_kind": "image",
                "asset_path": "figure-method.png",
                "asset_summary": "Pipeline figure showing query rewrite, route fusion, rerank, and context expansion.",
                "asset_preview_text": "rewrite -> fusion -> rerank -> expansion",
                "order_index": 4,
            },
        },
        {
            "content": "Experiment setup: evaluation uses a held-out benchmark and ablation metrics.",
            "metadata": {
                "chunk_id": "experiment-setup",
                "parent_chunk_id": "experiment-setup",
                "subchunk_index": 1,
                "subchunk_count": 1,
                "page_number": 5,
                "page_range": "5",
                "section_title": "Experiments",
                "section_path": "5 Experiments",
                "source": "paper.pdf",
                "chunk_type": "text",
                "order_index": 5,
            },
        },
    ]


class EnhancedRetrievalServiceFacadeSmokeTests(unittest.TestCase):
    def test_enhanced_retrieval_service_does_not_reintroduce_private_forwarding_wrappers(self) -> None:
        source_path = Path(__file__).resolve().parents[2] / "services" / "retrieval" / "enhanced_retrieval_service.py"
        source = source_path.read_text(encoding="utf-8")
        forbidden_wrappers = [
            "_build_paper_context",
            "_build_query_plan",
            "_build_query_profile",
            "_build_query_views",
            "_normalize_route_results",
            "_fuse_routes",
            "_load_llm_reranker",
            "_normalize_chunk",
            "_compute_structural_bonus",
            "_tokenize_for_keyword_search",
            "_route_confidence",
            "_normalize_query_text",
            "_debug_query_profile",
            "_debug_intent_profile",
        ]

        for wrapper_name in forbidden_wrappers:
            with self.subTest(wrapper_name=wrapper_name):
                self.assertNotIn(f"def {wrapper_name}", source)

    def test_enhanced_retrieve_delegates_to_retrieval_pipeline(self) -> None:
        service, collection_name, *_ = build_retrieval_service()
        options = RetrievalOptions(top_k=2, debug=True)
        expected = {"chunks": [{"chunk_id": "smoke"}]}

        with mock.patch.object(service.retrieval_pipeline, "retrieve", return_value=expected) as retrieve:
            result = service.enhanced_retrieve(
                "What is the method?",
                collection_name,
                paper_context={"arxiv_id": "2401.00001"},
                options=options,
            )

        self.assertEqual(result, expected)
        retrieve.assert_called_once_with(
            user_query="What is the method?",
            collection_name=collection_name,
            paper_context={"arxiv_id": "2401.00001"},
            options=options,
        )


class RetrievalPipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service, self.collection_name, *_ = build_retrieval_service()
        self.pipeline = self.service.retrieval_pipeline
        self.sample_chunks = build_sample_chunks()
        self.options_cls = RetrievalOptions

    def test_retrieve_top_k_boundaries_and_debug_snapshot(self) -> None:
        default_result = self.pipeline.retrieve(
            "What is the method of the paper?",
            self.collection_name,
            options=self.options_cls(top_k=None, debug=True, enable_llm_rerank=False),
        )
        one_result = self.pipeline.retrieve(
            "What is the method of the paper?",
            self.collection_name,
            options=self.options_cls(top_k=1, debug=True, enable_llm_rerank=False),
        )
        capped_result = self.pipeline.retrieve(
            "What is the method of the paper?",
            self.collection_name,
            options=self.options_cls(top_k=99, debug=True, enable_llm_rerank=False),
        )

        self.assertEqual(len(default_result["chunks"]), 3)
        self.assertEqual(len(one_result["chunks"]), 1)
        self.assertEqual(len(capped_result["chunks"]), 4)
        debug = default_result["debug"]
        self.assertIn("raw_retrieval_top30", debug["stages"])
        self.assertIn("fused_top30", debug["stages"])
        self.assertIn("final_context_top15", debug["stages"])
        self.assertIn("query_profile", debug)
        self.assertIn("collection_profile", debug)
        self.assertIn("profile_cache_hit", debug["collection_profile"])
        self.assertEqual(debug["collection_profile"]["vector_dimension"], 3)
        self.assertIn("embedding_batch", debug)
        self.assertEqual(debug["embedding_batch"]["status"], "ok")
        self.assertIn("route_metrics", debug)
        self.assertEqual(debug["route_metrics"]["vector_original"]["status"], "ok")
        self.assertIn("latency_ms", debug["route_metrics"]["keyword"])

    def test_retrieve_fuses_routes_dedupes_and_preserves_source_fields(self) -> None:
        result = self.pipeline.retrieve(
            "What does Figure 2 and Table 3 show?",
            self.collection_name,
            options=self.options_cls(debug=True, enable_llm_rerank=False),
        )

        first_chunk = result["chunks"][0]
        debug = result["debug"]
        raw_ids = [item["chunk_id"] for item in debug["stages"]["raw_retrieval_top30"]]
        fused_ids = [item["chunk_id"] for item in debug["stages"]["fused_top30"]]

        self.assertIn("page_number", first_chunk)
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("parent_chunk_id", first_chunk)
        self.assertIn("section_path", first_chunk.get("metadata", {}))
        self.assertEqual(len(fused_ids), len(set(fused_ids)))
        self.assertTrue(any(len(item.get("matched_routes", [])) > 1 for item in result["chunks"]))
        self.assertIn("mode", debug["llm_rerank"])

    def test_retrieve_rerank_failure_falls_back_to_fused_order(self) -> None:
        with mock.patch.object(self.service.rerank_service, "load_llm_reranker", return_value=None):
            self.service.llm_rerank_provider = "local"
            self.service._llm_reranker_error = "unavailable"
            result = self.pipeline.retrieve(
                "What is the method of the paper?",
                self.collection_name,
                options=self.options_cls(debug=True, enable_llm_rerank=True, top_k=2),
            )

        fused_ids = [item["chunk_id"] for item in result["debug"]["stages"]["fused_top30"][:2]]
        final_ids = [item["chunk_id"] for item in result["chunks"]]

        self.assertFalse(result["debug"]["llm_rerank"]["applied"])
        self.assertEqual(final_ids, fused_ids)

    def test_debug_snapshot_contains_required_trace_sections(self) -> None:
        result = self.pipeline.retrieve(
            "Which dataset and results are most important?",
            self.collection_name,
            options=self.options_cls(debug=True, enable_llm_rerank=False),
        )
        debug = result["debug"]

        self.assertIn("query_profile", debug)
        self.assertIn("llm_rerank", debug)
        self.assertIn("raw_retrieval_top30", debug["stages"])
        self.assertIn("fused_top30", debug["stages"])
        self.assertIn("final_context_top15", debug["stages"])
        self.assertIn("routes", debug)
        self.assertIn("collection_profile", debug)
        self.assertIn("multi_index", debug)
        self.assertIn("raw_index_hits", debug["multi_index"])
        self.assertIn("chunk_aggregation", debug["multi_index"])
        self.assertIn("route_hit_distribution", debug["multi_index"])

    def test_debug_trace_records_multi_index_hit_aggregation(self) -> None:
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

        result = self.pipeline.retrieve(
            "What is the method framework of the paper?",
            self.collection_name,
            options=self.options_cls(debug=True, enable_llm_rerank=False, enable_keyword_search=False),
        )
        multi_index = result["debug"]["multi_index"]
        vector_hits = multi_index["raw_index_hits"]["vector_original"]
        method_aggregation = [
            row for row in multi_index["chunk_aggregation"]["vector_original"]
            if row["chunk_id"] == "chunk-method"
        ]
        fused_ids = [item["chunk_id"] for item in result["debug"]["stages"]["fused_top30"]]

        self.assertTrue({"question", "summary"} & {hit["matched_index_type"] for hit in vector_hits})
        self.assertEqual(len(method_aggregation), 1)
        self.assertTrue({"question", "summary"} & set(method_aggregation[0]["matched_index_types"]))
        self.assertEqual(len(fused_ids), len(set(fused_ids)))

    def test_non_debug_response_keeps_debug_payload_hidden(self) -> None:
        result = self.pipeline.retrieve(
            "What is the method of the paper?",
            self.collection_name,
            options=self.options_cls(debug=False, enable_llm_rerank=False),
        )

        self.assertIn("chunks", result)
        self.assertNotIn("debug", result)
        self.assertNotIn("trace_export", result)

    def test_context_expansion_debug_links_subchunks_sections_and_assets(self) -> None:
        service, collection_name, *_ = build_retrieval_service(chunks=build_context_expansion_chunks())
        options_cls = RetrievalOptions

        result = service.retrieval_pipeline.retrieve(
            "How does the method pipeline work step by step?",
            collection_name,
            options=options_cls(top_k=1, debug=True, enable_llm_rerank=False),
        )

        debug = result["debug"]
        expansion = debug["context_expansion"]
        relations = expansion["relations"]
        anchor_ids = [relation["anchor"]["chunk_id"] for relation in relations]
        candidate_pairs = [
            (relation["anchor"]["chunk_id"], candidate["candidate_chunk_id"], candidate["expansion_type"])
            for relation in relations
            for candidate in relation["candidates"]
        ]

        self.assertEqual(len(result["chunks"]), 1)
        self.assertGreater(expansion["anchor_count"], len(result["chunks"]))
        self.assertIn("method-parent-2", anchor_ids)
        self.assertEqual(expansion["policy"]["name"], "method_flow")
        self.assertIn(("method-parent-2", "method-parent-1", "sibling"), candidate_pairs)
        self.assertIn(("method-parent-2", "method-parent-3", "sibling"), candidate_pairs)
        self.assertTrue(
            any(
                anchor == "method-parent-2"
                and candidate == "method-figure"
                and expansion_type in {"asset_related", "section_neighbors"}
                for anchor, candidate, expansion_type in candidate_pairs
            )
        )
        self.assertTrue(
            any(
                relation["anchor"]["chunk_id"] == "method-parent-2"
                and any(candidate["expansion_type"] == "section_header" for candidate in relation["candidates"])
                for relation in relations
            )
        )
        merged = next(item for item in expansion["candidate_pool"] if item["candidate_chunk_id"] == "method-parent-1")
        self.assertIn("method-parent-2", merged["expansion_source_anchor_ids"])
        self.assertIn("sibling", merged["relationship_types"])
        self.assertIn("section_header", merged["relationship_types"])
        self.assertGreater(merged["expansion_score"], 0)

    def test_context_expansion_uses_question_type_specific_policies(self) -> None:
        service, collection_name, *_ = build_retrieval_service(chunks=build_context_expansion_chunks())
        options_cls = RetrievalOptions

        figure_result = service.retrieval_pipeline.retrieve(
            "What does Figure 1 show in the method pipeline?",
            collection_name,
            options=options_cls(top_k=1, debug=True, enable_llm_rerank=False),
        )
        figure_expansion = figure_result["debug"]["context_expansion"]
        self.assertEqual(figure_expansion["policy"]["name"], "figure_table")
        figure_candidate = next(item for item in figure_expansion["candidate_pool"] if item["candidate_chunk_id"] == "method-figure")
        self.assertTrue({"asset_related", "cited_asset_context"} & set(figure_candidate["relationship_types"]))

        experiment_result = service.retrieval_pipeline.retrieve(
            "Which dataset baseline metric and implementation details are used in the experiments?",
            collection_name,
            options=options_cls(top_k=1, debug=True, enable_llm_rerank=False),
        )
        experiment_expansion = experiment_result["debug"]["context_expansion"]
        self.assertEqual(experiment_expansion["policy"]["name"], "experiment_setup")
        self.assertIn("page_neighbors", experiment_expansion["policy"]["actions"])
        experiment_candidate = next(item for item in experiment_expansion["candidate_pool"] if item["candidate_chunk_id"] == "experiment-setup")
        self.assertTrue({"self", "section_neighbors", "page_neighbors"} & set(experiment_candidate["relationship_types"]))

        result_result = service.retrieval_pipeline.retrieve(
            "What results ablation and performance comparison are reported?",
            collection_name,
            options=options_cls(top_k=1, debug=True, enable_llm_rerank=False),
        )
        result_expansion = result_result["debug"]["context_expansion"]
        self.assertEqual(result_expansion["policy"]["name"], "result_analysis")
        result_asset = next(item for item in result_expansion["candidate_pool"] if item["candidate_chunk_id"] == "method-figure")
        self.assertEqual(result_asset["chunk_type"], "figure")
        self.assertIn("asset_related", result_asset["relationship_types"])
        self.assertGreaterEqual(result_asset["expansion_score"], result_expansion["candidate_pool"][0]["expansion_score"])

    def test_context_expansion_degrades_when_structure_fields_are_missing(self) -> None:
        service, collection_name, *_ = build_retrieval_service(
            chunks=[
                {
                    "content": "A relevant method chunk without parent or section metadata.",
                    "metadata": {"chunk_id": "loose-method", "chunk_type": "text", "source": "paper.pdf"},
                }
            ]
        )
        options_cls = RetrievalOptions

        result = service.retrieval_pipeline.retrieve(
            "What is the method?",
            collection_name,
            options=options_cls(top_k=1, debug=True, enable_llm_rerank=False),
        )

        relation = result["debug"]["context_expansion"]["relations"][0]
        self.assertEqual(relation["anchor"]["chunk_id"], "loose-method")
        self.assertTrue(any(candidate["expansion_type"] == "self" for candidate in relation["candidates"]))

    def test_context_budget_selects_expanded_final_context_and_can_be_disabled(self) -> None:
        service, collection_name, *_ = build_retrieval_service(chunks=build_context_expansion_chunks())
        options_cls = RetrievalOptions

        expanded = service.retrieval_pipeline.retrieve(
            "How does the method pipeline work step by step?",
            collection_name,
            options=options_cls(top_k=4, debug=True, enable_llm_rerank=False),
        )
        disabled = service.retrieval_pipeline.retrieve(
            "How does the method pipeline work step by step?",
            collection_name,
            options=options_cls(top_k=4, debug=True, enable_llm_rerank=False, enable_context_expansion=False),
        )

        expanded_ids = [chunk["chunk_id"] for chunk in expanded["chunks"]]
        disabled_ids = [chunk["chunk_id"] for chunk in disabled["chunks"]]
        reranked_ids = [chunk["chunk_id"] for chunk in expanded["debug"]["stages"]["reranked_top30"][:4]]

        self.assertEqual(disabled_ids, reranked_ids)
        self.assertTrue(expanded["debug"]["context_budget"]["applied"])
        self.assertFalse(disabled["debug"]["context_budget"]["applied"])
        self.assertEqual(set(expanded_ids), set(expanded["debug"]["context_budget"]["included_chunk_ids"]))
        self.assertIn("sibling_context", expanded["debug"]["context_budget"]["role_counts"])
        self.assertIn("figure_evidence", expanded["debug"]["context_budget"]["role_counts"])
        self.assertIn("method-parent-3", expanded["debug"]["context_budget"]["included_chunk_ids"])
        self.assertTrue(any(chunk.get("context_role") for chunk in expanded["chunks"]))
        self.assertFalse(any(chunk.get("context_role") for chunk in disabled["chunks"]))

    def test_context_pack_sources_preserve_context_budget_metadata(self) -> None:
        service, collection_name, *_ = build_retrieval_service(chunks=build_context_expansion_chunks())
        options_cls = RetrievalOptions

        result = service.retrieval_pipeline.retrieve(
            "How does the method pipeline work step by step?",
            collection_name,
            options=options_cls(top_k=4, debug=True, enable_llm_rerank=False),
        )
        context_pack = ContextPackBuilder().build(result["chunks"])
        sources = context_pack["source_payload"]

        sibling_source = next(source for source in sources if source["chunk_id"] == "method-parent-3")
        figure_source = next(source for source in sources if source["chunk_id"] == "method-figure")
        self.assertEqual(sibling_source["context_role"], "sibling_context")
        self.assertIn("sibling", sibling_source["relationship_types"])
        self.assertEqual(figure_source["context_role"], "figure_evidence")
        self.assertIn("context_budget_score", figure_source)
        self.assertIn("role: sibling_context", context_pack["text_context"])

    def test_table_structured_route_promotes_cell_level_evidence_for_table_question(self) -> None:
        result = self.pipeline.retrieve(
            "表 2 中最高的 accuracy 是多少？",
            self.collection_name,
            options=self.options_cls(debug=True, enable_llm_rerank=False, enable_hyde=False, enable_keyword_search=False),
        )

        self.assertTrue(result["chunks"])
        top_chunk = result["chunks"][0]
        self.assertEqual(top_chunk["chunk_id"], "chunk-table-results")
        self.assertEqual(top_chunk["retrieval_route"], "table_structured")
        evidence = top_chunk["table_evidence"]
        self.assertEqual(evidence["schema_version"], "table_evidence_v2")
        self.assertEqual(evidence["decision"], "compute")
        self.assertEqual(evidence["operation_hint"], "max")
        self.assertEqual(evidence["final_evidence"]["rows"], ["Ours"])
        self.assertEqual(evidence["final_evidence"]["columns"], ["Accuracy"])
        self.assertEqual(evidence["final_evidence"]["cells"][0]["raw_value"], "89.2%")
        self.assertIn("table_structured", result["debug"]["routes"])
        self.assertEqual(result["debug"]["route_metrics"]["table_structured"]["status"], "ok")
        self.assertEqual(result["debug"]["table_structured"]["matched_tables"][0]["table_id"], "paper-table-2")
        self.assertEqual(result["debug"]["stages"]["fused_top30"][0]["chunk_id"], "chunk-table-results")

    def test_table_structured_route_falls_back_cleanly_for_summary_question(self) -> None:
        result = self.pipeline.retrieve(
            "What is the main contribution of the paper?",
            self.collection_name,
            options=self.options_cls(debug=True, enable_llm_rerank=False),
        )

        self.assertTrue(result["chunks"])
        self.assertEqual(result["debug"]["route_metrics"]["table_structured"]["candidate_count"], 0)
        self.assertEqual(result["debug"]["table_structured"]["reason"], "query_not_table_like")


if __name__ == "__main__":
    unittest.main()
