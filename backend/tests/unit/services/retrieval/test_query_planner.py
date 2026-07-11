import unittest

from tests.helpers import build_retrieval_service


class QueryPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service, self.collection_name, *_ = build_retrieval_service()
        self.planner = self.service.query_planner

    def _build_bundle(self, question: str, *, enable_query_rewrite: bool = True):
        return self.planner.build_query_bundle(
            user_query=question,
            collection_name=self.collection_name,
            enable_query_rewrite=enable_query_rewrite,
        )

    def test_method_question_builds_method_flow_profile(self) -> None:
        bundle = self._build_bundle("What is the method flow and architecture of the paper?")
        profile = bundle["query_profile"]

        self.assertEqual(profile.intent_profile.main_intent, "method_flow")
        self.assertIn("method", profile.semantic_query)
        self.assertTrue(bundle["query_views"]["selected_queries"])

    def test_experiment_question_builds_experiment_profile(self) -> None:
        bundle = self._build_bundle("What is the experimental setup, dataset, and baseline configuration?")
        profile = bundle["query_profile"]

        self.assertEqual(profile.intent_profile.main_intent, "experiment_setup")
        self.assertIn("experiment", bundle["rerank_query"].lower())

    def test_result_question_builds_results_profile(self) -> None:
        bundle = self._build_bundle("What results and ablation findings does the paper report?")
        profile = bundle["query_profile"]

        self.assertEqual(profile.intent_profile.main_intent, "result_analysis")
        self.assertTrue(bundle["query_views"]["selected_queries"])

    def test_limitation_question_builds_limitation_profile(self) -> None:
        bundle = self._build_bundle("What are the limitations or future work discussed in the paper?")
        profile = bundle["query_profile"]

        self.assertEqual(profile.intent_profile.main_intent, "limitation")
        self.assertTrue(any("limit" in item.lower() for item in profile.section_preferences))

    def test_dataset_question_builds_dataset_profile(self) -> None:
        bundle = self._build_bundle("Which dataset or benchmark corpus is used?")
        profile = bundle["query_profile"]

        self.assertIn(profile.intent_profile.main_intent, {"dataset", "experiment_setup"})
        self.assertTrue(any(token in profile.tokens for token in ["dataset", "benchmark", "corpus"]))

    def test_figure_table_question_builds_visual_profile(self) -> None:
        bundle = self._build_bundle("Please explain Figure 2 and Table 3.")
        profile = bundle["query_profile"]

        self.assertEqual(profile.intent_profile.main_intent, "figure_table")
        self.assertTrue(any(token in bundle["rerank_query"].lower() for token in ["figure", "table"]))

    def test_mixed_language_question_keeps_bilingual_tokens(self) -> None:
        bundle = self._build_bundle("这篇 paper 的 method 和 results 分别说明了什么？")
        profile = bundle["query_profile"]

        self.assertIn("paper", profile.tokens)
        self.assertTrue(any(token in profile.tokens for token in ["这篇", "分别说明", "什么"]))
        self.assertTrue(bundle["query_views"]["selected_queries"])

    def test_query_rewrite_toggle_changes_selected_query_sources(self) -> None:
        enabled = self._build_bundle("What is the method of the paper?", enable_query_rewrite=True)
        disabled = self._build_bundle("What is the method of the paper?", enable_query_rewrite=False)

        enabled_sources = {row["source"] for row in enabled["query_views"]["candidates"]}
        disabled_sources = {row["source"] for row in disabled["query_views"]["candidates"]}

        self.assertIn("heuristic", enabled_sources)
        self.assertNotIn("heuristic", disabled_sources)
        self.assertTrue(enabled["query_views"]["enabled"])
        self.assertFalse(disabled["query_views"]["enabled"])

    def test_query_bundle_keeps_main_intent_only_in_intent_profile(self) -> None:
        self.planner.generation_service.plan_queries_for_retrieval = lambda **_kwargs: {
            "question_type": "method",
            "main_intent": "method",
            "sub_intents": ["deep_method"],
            "paper_terms": ["retrieval"],
            "preferred_sections": ["method"],
            "rewrite_queries": [{"query": "retrieval method pipeline"}],
        }
        bundle = self._build_bundle("How does the method pipeline work?")
        profile = bundle["query_profile"]

        self.assertEqual(profile.intent_profile.main_intent, "method_flow")
        self.assertFalse(hasattr(profile, "question_type"))
        self.assertNotIn("question_type", profile.query_plan)
        self.assertNotIn("main_intent", profile.query_plan)
        self.assertNotIn("sub_intents", profile.query_plan)


if __name__ == "__main__":
    unittest.main()
