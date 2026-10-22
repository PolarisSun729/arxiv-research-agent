from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import admin_retrieval_debug_router
from services.retrieval.contracts import QueryProfile
from services.retrieval.result_fusion_service import ResultFusionService


def _chunk(chunk_id, route, *, score=0.5, **extra):
    return {
        "chunk_id": chunk_id,
        "original_chunk_id": chunk_id,
        "content": f"chunk {chunk_id} body",
        "retrieval_route": route,
        "route_score": score,
        "section_path": f"{chunk_id} Section",
        **extra,
    }


def _query_profile():
    return QueryProfile(
        original_query="q", normalized_query="q", retrieval_query="q", language="en", intent_profile=None, tokens=[], keywords=["lora"],
        intent_tags=[], ambiguity_score=0.0, section_preferences=[],
    )


class _FakePipeline:
    def __init__(self, fusion):
        self.fusion = fusion
        self.calls = []

    def retrieve(self, *, user_query, collection_name, paper_context, options):
        self.calls.append({"collection_name": collection_name, "options": options})
        routes = {
            "vector_original": [_chunk(1, "vector_original"), _chunk(2, "vector_original")],
            "keyword": [_chunk(2, "keyword", keyword_matched_terms=["lora"]), _chunk(3, "keyword")],
        }
        fused = self.fusion.fuse_routes(routes, 10, _query_profile())
        reranked = [{**fused[1], "llm_rerank_score": 0.9}, {**fused[0], "llm_rerank_score": 0.4}, fused[2]]
        expansion = {**_chunk(9, ""), "context_role": "neighbor"}
        return {
            "chunks": reranked[:1],
            "debug": {
                "original_query": user_query,
                "intent_profile": {"main_intent": "method", "sub_intents": []},
                "query_profile": {"keywords": ["lora"]},
                "query_views": {"enabled": True, "selected_queries": ["q", "q2"], "candidates": [{"query": "q2", "source": "rewrite", "selected": True, "reason": "kept"}], "original_translation": {"query": "q-en", "status": "ok", "error": ""}},
                "route_metrics": {
                    "vector_original": {"status": "ok", "enabled": True, "latency_ms": 12.0},
                    "keyword": {"status": "ok", "enabled": True, "latency_ms": 3.0},
                    "vector_hyde": {"status": "disabled", "enabled": False},
                    "rerank": {"status": "ok", "latency_ms": 200.0},
                },
                "fusion": {"rrf_k": 60, "route_weights": {"vector_original": 1.0, "keyword": 0.5}},
                "llm_rerank": {"enabled": True, "applied": True, "mode": "remote"},
                "config": {"effective_top_k": 1, "rrf_candidate_limit": 2},
            },
            "stage_chunks": {"routes": routes, "fused": fused, "reranked": reranked, "final": [reranked[0], expansion]},
        }


class _FakeIndexStore:
    def get_paper_qa_index(self, arxiv_id):
        if arxiv_id != "p1":
            return None
        return {"status": "indexed", "collection_name": "paper_p1", "active_build_id": "b1"}


class _FakeCatalogStore:
    def get_paper(self, arxiv_id):
        return {"title": "Paper One"}


@pytest.fixture
def env():
    fusion = ResultFusionService(rrf_k=60, route_weights={"vector_original": 1.0, "keyword": 0.5})
    pipeline = _FakePipeline(fusion)
    service = SimpleNamespace(retrieval_pipeline=pipeline, fusion_service=fusion)
    app = FastAPI()
    app.include_router(admin_retrieval_debug_router.router, prefix="/api")
    app.dependency_overrides[dependencies.get_enhanced_retrieval_service] = lambda: service
    app.dependency_overrides[dependencies.get_paper_qa_index_store] = lambda: _FakeIndexStore()
    app.dependency_overrides[dependencies.get_paper_catalog_store] = lambda: _FakeCatalogStore()
    return TestClient(app), pipeline


def test_run_aligns_route_and_stage_ranks_per_chunk(env):
    client, pipeline = env
    response = client.post("/api/admin/retrieval-debug/runs", json={"arxiv_id": "p1", "question": " what is lora ", "enable_hyde": False})
    assert response.status_code == 200
    body = response.json()

    options = pipeline.calls[0]["options"]
    assert pipeline.calls[0]["collection_name"] == "paper_p1"
    assert options.debug is True and options.export_trace is False and options.include_stage_chunks is True
    assert options.enable_hyde is False and options.enable_llm_rerank is None
    assert body["question"] == "what is lora" and body["title"] == "Paper One" and body["active_build_id"] == "b1"

    chunks = {row["chunk_id"]: row for row in body["chunks"]}
    # chunk 2 被两路同时召回，融合排第一；每路投票都能拆开看。
    assert chunks[2]["route_ranks"] == {"vector_original": 2, "keyword": 1}
    assert chunks[2]["fused_rank"] == 1
    assert set(chunks[2]["rrf_votes"]) == {"vector_original", "keyword"}
    assert chunks[2]["rrf_votes"]["keyword"]["rank"] == 1
    assert chunks[2]["rrf_votes"]["keyword"]["vote"] == pytest.approx(0.5 / 61)
    assert chunks[2]["keyword_terms"] == ["lora"]
    # rerank 把融合第二的 chunk 1 提到第一，只有它进了最终上下文。
    assert chunks[2]["rerank_rank"] == 2 and chunks[2]["rerank_score"] == 0.4 and chunks[2]["final_rank"] is None
    assert chunks[1]["fused_rank"] == 2 and chunks[1]["rerank_rank"] == 1 and chunks[1]["final_rank"] == 1
    # 只出现在最终上下文里的扩展块没有召回和融合名次，排在最后。
    assert chunks[9]["route_ranks"] == {} and chunks[9]["fused_rank"] is None
    assert chunks[9]["final_rank"] == 2 and chunks[9]["context_role"] == "neighbor"
    assert body["chunks"][-1]["chunk_id"] == 9

    routes = {route["name"]: route for route in body["routes"]}
    assert set(routes) == {"vector_original", "keyword", "vector_hyde"}
    assert routes["keyword"]["candidate_count"] == 2 and routes["keyword"]["weight"] == 0.5
    assert routes["vector_hyde"]["enabled"] is False
    assert body["rerank"]["applied"] is True and body["rerank"]["latency_ms"] == 200.0
    assert body["query"]["selected_queries"] == ["q", "q2"]
    assert body["query"]["original_translation"]["query"] == "q-en"


def test_run_rejects_paper_without_active_index(env):
    client, pipeline = env
    response = client.post("/api/admin/retrieval-debug/runs", json={"arxiv_id": "missing", "question": "q"})
    assert response.status_code == 404
    assert pipeline.calls == []
