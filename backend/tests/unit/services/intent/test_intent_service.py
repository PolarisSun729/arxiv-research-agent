import json
import unittest

from services.intent.intent_service import MAIN_INTENT_NAMES, IntentService
from utils.config import INTENT_ROUTING_CONFIG


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
    def test_route_weight_config_exactly_matches_canonical_main_intents(self) -> None:
        self.assertEqual(set(INTENT_ROUTING_CONFIG["intent_route_weights"]), set(MAIN_INTENT_NAMES))

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

    def test_legacy_llm_intent_is_rejected_and_falls_back_to_heuristic_profile(self) -> None:
        generation_service = _FakeIntentGenerationService(
            payload={
                "main_intent": "method",
                "sub_intents": ["deep_method"],
                "confidence": 0.92,
                "preferred_sections": ["method"],
            }
        )
        service = IntentService(generation_service=generation_service)

        profile = service.build_intent_profile("How does the method pipeline work?")

        self.assertEqual(profile.main_intent, "method_flow")
        self.assertEqual(profile.source, "heuristic")
        self.assertEqual(profile.fallback_reason, "invalid_main_intent")

    def test_unknown_llm_intent_is_rejected_and_falls_back_to_heuristic_profile(self) -> None:
        generation_service = _FakeIntentGenerationService(
            payload={"main_intent": "future_unknown_intent", "confidence": 0.99}
        )
        service = IntentService(generation_service=generation_service)

        profile = service.build_intent_profile("What limitations does the paper discuss?")

        self.assertEqual(profile.main_intent, "limitation")
        self.assertEqual(profile.source, "heuristic")
        self.assertEqual(profile.fallback_reason, "invalid_main_intent")


if __name__ == "__main__":
    unittest.main()
