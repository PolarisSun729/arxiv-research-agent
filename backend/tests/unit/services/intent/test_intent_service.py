import json
import unittest

from services.intent.intent_service import IntentService


class _FakeIntentGenerationService:
    def __init__(self, *, payload=None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls = []

    def complete_with_qwen(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        if self.error is not None:
            raise self.error
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload or {}, ensure_ascii=False)


class IntentServiceTests(unittest.TestCase):
    def test_heuristic_intent_classification_for_method_question(self) -> None:
        service = IntentService()

        profile = service.build_intent_profile("What is the method pipeline and architecture?")

        self.assertEqual(profile.main_intent, "method_flow")
        self.assertEqual(profile.source, "heuristic")
        self.assertIn("method", profile.preferred_sections)

    def test_chinese_input_uses_heuristic_intent_classification(self) -> None:
        service = IntentService()

        profile = service.build_intent_profile("这篇论文的实验结果和数据集是什么？")

        self.assertEqual(profile.language, "zh")
        self.assertIn(profile.main_intent, {"experiment_setup", "result_analysis", "dataset"})

    def test_empty_input_falls_back_to_other_intent(self) -> None:
        service = IntentService()

        profile = service.build_intent_profile("")

        self.assertEqual(profile.main_intent, "other")
        self.assertEqual(profile.language, "unknown")
        self.assertEqual(profile.source, "heuristic")

    def test_llm_fallback_success_uses_llm_profile(self) -> None:
        generation_service = _FakeIntentGenerationService(
            payload={
                "main_intent": "figure_table",
                "sub_intents": ["table_lookup"],
                "confidence": 0.91,
                "intent_summary": "find the target figure",
                "preferred_sections": ["figure", "results"],
                "avoid_sections": ["references"],
                "rewrite_focus": ["figure", "table", "caption"],
                "rerank_focus": ["figure", "table"],
                "use_keyword_search": True,
                "use_hyde": False,
                "final_context_policy": {"allow_tables": True, "allow_figures": True, "allow_appendix": False, "max_table_chunks": 1, "max_appendix_chunks": 0},
            }
        )
        service = IntentService(generation_service=generation_service)

        profile = service.build_intent_profile("Explain Figure 2 in the paper")

        self.assertEqual(profile.main_intent, "figure_table")
        self.assertEqual(profile.source, "llm")
        self.assertEqual(profile.fallback_reason, "")
        self.assertTrue(generation_service.calls)

    def test_llm_failure_falls_back_to_heuristic_profile(self) -> None:
        generation_service = _FakeIntentGenerationService(error=RuntimeError("intent llm unavailable"))
        service = IntentService(generation_service=generation_service)

        profile = service.build_intent_profile("What are the limitations of the paper?")

        self.assertEqual(profile.main_intent, "limitation")
        self.assertEqual(profile.source, "heuristic")
        self.assertIn("intent llm unavailable", profile.fallback_reason)


if __name__ == "__main__":
    unittest.main()
