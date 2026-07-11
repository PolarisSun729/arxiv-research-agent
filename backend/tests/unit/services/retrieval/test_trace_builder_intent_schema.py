from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from services.retrieval.contracts import QueryProfile
from services.retrieval.trace_builder import RetrievalTraceBuilder


def test_debug_profiles_only_expose_canonical_intent_schema(tmp_path: Path) -> None:
    intent_profile = SimpleNamespace(
        original_query="How does the method work?",
        normalized_query="how does the method work?",
        language="en",
        main_intent="method_flow",
        sub_intents=["deep_method"],
        confidence=0.9,
        ambiguity_score=0.1,
        intent_summary="explain method flow",
        preferred_sections=["method"],
        route_weights={"keyword": 1.0},
        rewrite_count=3,
        use_keyword_search=True,
        use_hyde=False,
        rewrite_focus=["method"],
        rerank_focus=["implementation"],
        fallback_reason="",
        source="heuristic",
    )
    query_profile = QueryProfile(
        original_query=intent_profile.original_query,
        normalized_query=intent_profile.normalized_query,
        language="en",
        intent_profile=intent_profile,
        tokens=["method"],
        keywords=["method"],
        intent_tags=["deep_method"],
        paper_terms=[],
        ambiguity_score=0.1,
        semantic_query="method",
        evidence_query="method evidence",
        keyword_query="method implementation",
        section_preferences=["method"],
        query_plan={"rewrite_queries": []},
    )
    builder = RetrievalTraceBuilder(
        trace_export_enabled=False,
        trace_export_dir=tmp_path,
        rrf_k=60,
        route_weights={"keyword": 1.0},
        route_confidence_builder=lambda *args, **kwargs: 1.0,
        route_weights_builder=lambda profile: profile.route_weights,
    )

    debug_payload = {
        "intent_profile": builder.debug_intent_profile(intent_profile),
        "query_profile": builder.debug_query_profile(query_profile),
    }
    serialized = json.dumps(debug_payload)

    assert debug_payload["intent_profile"]["main_intent"] == "method_flow"
    assert "question_type" not in serialized
    assert "intent_bucket" not in serialized
    assert "legacy_intent" not in serialized
