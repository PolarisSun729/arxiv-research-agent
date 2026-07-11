from __future__ import annotations

import json

import pytest

from services.llm.generation_service import GenerationService


class _QueryPlanGenerationService(GenerationService):
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete_with_qwen(self, prompt: str, **kwargs) -> str:
        return json.dumps(self.payload)


@pytest.mark.parametrize(
    ("main_intent", "expected_focuses"),
    [
        ("method_flow", ["seed", "method", "implementation"]),
        ("implementation_detail", ["seed", "method", "implementation"]),
        ("definition", ["seed", "method", "implementation"]),
        ("experiment_setup", ["seed", "setup", "evaluation"]),
        ("result_analysis", ["seed", "results", "analysis"]),
    ],
)
def test_query_plan_uses_canonical_intent_for_specialized_fallback_rewrites(
    main_intent: str,
    expected_focuses: list[str],
) -> None:
    service = _QueryPlanGenerationService(
        {
            "paper_terms": ["RAG"],
            "preferred_sections": ["method"],
            "rewrite_queries": [{"query": "RAG evidence", "focus": "seed"}],
        }
    )

    plan = service.plan_queries_for_retrieval(
        "How does the method work?",
        max_queries=3,
        intent_profile={"main_intent": main_intent},
    )

    assert "question_type" not in plan
    assert "main_intent" not in plan
    assert [item["focus"] for item in plan["rewrite_queries"]] == expected_focuses
