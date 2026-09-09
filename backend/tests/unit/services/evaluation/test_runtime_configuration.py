"""配置指纹必须跟随实际管线的参数解析，而非另一份相似的默认配置。"""

import pytest

from tests.helpers import build_retrieval_service


@pytest.mark.parametrize("key,value", [("context_budget_max_chars", 100), ("enable_context_expansion", False)])
def test_snapshot_changes_with_actual_enhanced_runtime_option(monkeypatch, key, value):
    owner, _, *_ = build_retrieval_service()
    before = owner.evaluation_configuration()
    monkeypatch.setitem(owner.retrieval_pipeline.enhanced_config, key, value)
    after = owner.evaluation_configuration()
    assert before != after


def test_research_snapshot_uses_effective_normal_and_final_round_options(monkeypatch):
    from services.paper_evidence_research.dependencies.need_orchestrated_retriever import NeedOrchestratedRetriever

    owner, _, *_ = build_retrieval_service()
    monkeypatch.setitem(owner.retrieval_pipeline.enhanced_config, "max_final_context_top_k", 5)
    retriever = NeedOrchestratedRetriever(retrieval_pipeline=owner.retrieval_pipeline, target_resolver=lambda _: None, top_k=8)
    snapshot = retriever.evaluation_configuration()
    assert snapshot["normal_round"]["effective_top_k"] == 5
    assert snapshot["final_round"]["effective_top_k"] == 5
    assert snapshot["final_round"]["enable_query_rewrite"] is False
    assert snapshot["final_round"]["enable_llm_rerank"] is False
