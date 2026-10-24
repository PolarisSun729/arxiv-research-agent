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

    def test_original_question_translation_is_attached_to_query_views(self) -> None:
        self.planner.generation_service.translate_query_for_retrieval = lambda query, **_kwargs: "  What is the   method? "

        translation = self._build_bundle("这篇论文的方法是什么？")["query_views"]["original_translation"]

        self.assertEqual(translation["query"], "What is the method?")
        self.assertEqual(translation["status"], "ok")

    def test_translation_failure_falls_back_to_original_question(self) -> None:
        def fail(_query, **_kwargs):
            raise TimeoutError("llm timeout")

        self.planner.generation_service.translate_query_for_retrieval = fail
        question = "这篇论文的方法是什么？"

        translation = self._build_bundle(question)["query_views"]["original_translation"]

        self.assertEqual(translation["query"], question)
        self.assertEqual(translation["status"], "error")
        self.assertIn("llm timeout", translation["error"])

    def test_query_profile_keywords_come_from_english_translation(self) -> None:
        self.planner.generation_service.translate_query_for_retrieval = (
            lambda query, **_kwargs: "Which datasets are evaluated?"
        )

        profile = self._build_bundle("这篇论文用了哪些数据集？")["query_profile"]

        self.assertEqual(profile.retrieval_query, "Which datasets are evaluated?")
        self.assertEqual(profile.keywords, ["dataset", "evalu"])
        self.assertEqual(profile.language, "zh")

    def test_method_question_builds_method_flow_profile(self) -> None:
        bundle = self._build_bundle("What is the method flow and architecture of the paper?")
        profile = bundle["query_profile"]

        self.assertEqual(profile.intent_profile.main_intent, "method_flow")
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

    def test_rewrite_disabled_skips_llm_call(self) -> None:
        views = self._build_bundle("What is the method of the paper?", enable_query_rewrite=False)["query_views"]

        self.assertFalse(views["enabled"])
        self.assertEqual(views["rewrite_status"], "disabled")
        self.assertEqual(views["selected_queries"], [])
        rewrite_calls = [
            call for call in self.planner.generation_service.calls if call["method"] == "rewrite_query_for_retrieval"
        ]
        self.assertEqual(rewrite_calls, [])

    def test_rewrite_gets_paper_context_and_keeps_llm_queries(self) -> None:
        captured = {}

        def rewrite(query, **kwargs):
            captured.update(kwargs)
            return ["How does the sparse projection method work?", "Which datasets are evaluated?"]

        self.planner.generation_service.rewrite_query_for_retrieval = rewrite
        views = self._build_bundle("这个方法怎么做的，用了哪些数据集？")["query_views"]

        self.assertEqual(views["rewrite_status"], "ok")
        self.assertEqual(
            views["selected_queries"],
            ["How does the sparse projection method work?", "Which datasets are evaluated?"],
        )
        self.assertEqual(set(captured["paper_context"]), {"title", "abstract", "section_titles"})

    def test_rewrite_failure_leaves_route_without_queries(self) -> None:
        def fail(_query, **_kwargs):
            raise ValueError("bad json")

        self.planner.generation_service.rewrite_query_for_retrieval = fail
        views = self._build_bundle("这个方法怎么做的？")["query_views"]

        self.assertEqual(views["rewrite_status"], "error")
        self.assertIn("bad json", views["llm_error"])
        self.assertEqual(views["selected_queries"], [])

    def test_chinese_rewrite_is_dropped_as_invalid_output(self) -> None:
        self.planner.generation_service.rewrite_query_for_retrieval = lambda _query, **_kwargs: ["这个方法怎么做的？"]
        views = self._build_bundle("这个方法怎么做的？")["query_views"]

        self.assertEqual(views["selected_queries"], [])
        self.assertEqual(views["candidates"][0]["reason"], "not_english")
        self.assertEqual(views["rewrite_status"], "invalid_output")

    def test_rewrite_same_as_translation_or_duplicate_is_dropped(self) -> None:
        self.planner.generation_service.translate_query_for_retrieval = lambda _query, **_kwargs: "What is the method?"
        self.planner.generation_service.rewrite_query_for_retrieval = lambda _query, **_kwargs: [
            "what is the  method?",
            "Which datasets are used?",
            "which datasets are used?",
        ]
        views = self._build_bundle("方法是什么，用了哪些数据集？")["query_views"]

        self.assertEqual(views["selected_queries"], ["Which datasets are used?"])
        self.assertEqual(
            [row["reason"] for row in views["candidates"]],
            ["same_as_original", "kept", "duplicate"],
        )

    def test_rewrite_fully_covered_by_translation_reports_status(self) -> None:
        self.planner.generation_service.translate_query_for_retrieval = lambda _query, **_kwargs: "What is the method?"
        self.planner.generation_service.rewrite_query_for_retrieval = lambda _query, **_kwargs: ["What is the method?"]
        views = self._build_bundle("方法是什么？")["query_views"]

        self.assertEqual(views["selected_queries"], [])
        self.assertEqual(views["rewrite_status"], "covered_by_original")

    def test_section_preferences_come_from_intent_only(self) -> None:
        bundle = self._build_bundle("How does the method pipeline work?")
        profile = bundle["query_profile"]

        self.assertEqual(profile.section_preferences, list(profile.intent_profile.preferred_sections))


if __name__ == "__main__":
    unittest.main()
