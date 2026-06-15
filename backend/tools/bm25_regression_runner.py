#!/usr/bin/env python
"""BM25 召回回归诊断 runner。

一键跑固定回归问题集，对每个问题打印：
  - BM25 keyword route top-k（含 matched_terms / fields / query_sources / noise_flags）
  - 融合前 raw 召回 top-k
  - 融合后 fused top-k
  - rerank 后 top-k（若启用）

默认使用离线 fixture collection（确定性、无需 API key），便于对比 BM25 修改前后的变化；
也可用 --collection 指向真实 collection，配合真实 EnhancedRetrievalService 运行。

用法：
  python -m tools.bm25_regression_runner                 # 离线 fixture，全部 case
  python -m tools.bm25_regression_runner --case en_dataset
  python -m tools.bm25_regression_runner --json out.json # 结果落盘，方便 diff 前后两次
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CASES_PATH = BACKEND_DIR / "tests" / "fixtures" / "bm25_regression_cases.json"


def load_cases(case_id: Optional[str] = None) -> List[Dict[str, Any]]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if case_id:
        cases = [case for case in cases if case.get("id") == case_id]
    return cases


def build_offline_service():
    """构建离线 fixture 检索服务，避免诊断脚本依赖真实向量库与外部模型。"""
    from tests.helpers.retrieval import build_retrieval_service

    service, collection_name, *_ = build_retrieval_service()
    return service, collection_name


def build_live_service(collection_name: str):
    from services.embedding.embedding_service import EmbeddingService
    from services.storage.vector_store_service import VectorStoreService

    try:
        from services.llm.generation_service import GenerationService

        generation_service = GenerationService()
    except Exception:
        generation_service = None
    from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService

    service = EnhancedRetrievalService(
        embedding_service=EmbeddingService(),
        vector_store_service=VectorStoreService(),
        generation_service=generation_service,
    )
    return service, collection_name


def _chunk_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": item.get("chunk_id"),
        "chunk_type": item.get("chunk_type", "text"),
        "retrieval_route": item.get("retrieval_route"),
        "score": round(float(item.get("score", 0.0) or 0.0), 4),
        "route_score": round(float(item.get("route_score", 0.0) or 0.0), 4),
        "route_confidence": round(float(item.get("route_confidence", 0.0) or 0.0), 4),
    }


def _keyword_brief(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": item.get("chunk_id"),
        "chunk_type": item.get("chunk_type", "text"),
        "bm25_raw_score": round(float(item.get("bm25_raw_score", 0.0) or 0.0), 4),
        "bm25_fused_score": round(float(item.get("bm25_fused_score", 0.0) or 0.0), 4),
        "route_confidence": round(float(item.get("route_confidence", 0.0) or 0.0), 4),
        "matched_terms": [
            {"token": term.get("token"), "idf": round(float(term.get("idf", 0.0) or 0.0), 3), "fields": term.get("fields", [])}
            for term in (item.get("keyword_matched_terms", []) or [])[:6]
        ],
        "matched_fields": item.get("keyword_match_fields", []),
        "query_sources": item.get("keyword_query_sources", []),
        "noise_flags": item.get("keyword_noise_flags", []),
    }


def run_case(service: Any, collection_name: str, case: Dict[str, Any], top_k: int, enable_rerank: bool) -> Dict[str, Any]:
    from services.retrieval.contracts import RetrievalOptions

    options = RetrievalOptions(debug=True, enable_llm_rerank=enable_rerank)
    response = service.enhanced_retrieve(
        user_query=case["question"],
        collection_name=collection_name,
        options=options,
    )
    debug = response.get("debug") or {}
    routes = debug.get("routes", {}) or {}
    stages = debug.get("stages", {}) or {}
    keyword_route = routes.get("keyword", []) or []
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": case.get("question"),
        "expected_top_chunk": case.get("expected_top_chunk"),
        "bm25_top_k": [_keyword_brief(item) for item in keyword_route[:top_k]],
        "raw_top_k": [_chunk_brief(item) for item in stages.get("raw_retrieval_top30", [])[:top_k]],
        "fused_top_k": [_chunk_brief(item) for item in stages.get("fused_top30", [])[:top_k]],
        "reranked_top_k": [_chunk_brief(item) for item in stages.get("reranked_top30", [])[:top_k]],
        "final_context": [_chunk_brief(item) for item in stages.get("final_context_top15", [])[:top_k]],
        "keyword_base_route_confidence": (debug.get("keyword_search", {}) or {}).get("base_route_confidence"),
    }


def print_case(result: Dict[str, Any]) -> None:
    print("=" * 88)
    print(f"[{result['id']}] ({result['category']}) {result['question']}")
    expected = result.get("expected_top_chunk")
    if expected:
        print(f"  expected_top_chunk: {expected}")
    print(f"  keyword base_route_confidence: {result.get('keyword_base_route_confidence')}")
    print("\n  -- BM25 keyword route top-k --")
    if not result["bm25_top_k"]:
        print("     (no keyword hits)")
    for rank, hit in enumerate(result["bm25_top_k"], start=1):
        terms = ", ".join(f"{t['token']}({t['idf']}|{'/'.join(t['fields'])})" for t in hit["matched_terms"])
        print(
            f"     {rank}. {hit['chunk_id']} [{hit['chunk_type']}] raw={hit['bm25_raw_score']} "
            f"fused={hit['bm25_fused_score']} conf={hit['route_confidence']}"
        )
        print(f"        terms: {terms or '(none)'}")
        print(f"        sources: {hit['query_sources']}  fields: {hit['matched_fields']}  noise: {hit['noise_flags'] or 'clean'}")
    for label, key in (("raw recall top-k", "raw_top_k"), ("fused top-k", "fused_top_k"), ("reranked top-k", "reranked_top_k"), ("final context", "final_context")):
        rows = result.get(key, [])
        if not rows:
            continue
        chain = " > ".join(f"{r['chunk_id']}({r['retrieval_route']})" for r in rows)
        print(f"\n  -- {label} --\n     {chain}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="BM25 召回回归诊断 runner")
    parser.add_argument("--case", help="只跑指定 case id", default=None)
    parser.add_argument("--collection", help="真实 collection 名；省略时用离线 fixture", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="启用 LLM rerank（真实 collection 才有意义）")
    parser.add_argument("--json", dest="json_out", help="把结果写入 JSON 文件，便于前后对比 diff", default=None)
    args = parser.parse_args()

    cases = load_cases(args.case)
    if not cases:
        print(f"no cases matched (case={args.case!r})", file=sys.stderr)
        return 1

    if args.collection:
        service, collection_name = build_live_service(args.collection)
    else:
        service, collection_name = build_offline_service()

    results = [run_case(service, collection_name, case, args.top_k, args.rerank) for case in cases]
    for result in results:
        print_case(result)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"results written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
