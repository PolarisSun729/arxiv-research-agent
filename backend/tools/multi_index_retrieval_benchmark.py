#!/usr/bin/env python
"""Multi-index RAG 最小检索评估 runner。

默认使用离线 fixture collection，分别跑：
  - vector_only
  - vector_multi_index
  - vector_bm25
  - vector_bm25_multi_index
  - full_pipeline

输出 recall@5、recall@10、MRR、route hit distribution 和 index type contribution。
该 runner 不依赖真实 Milvus/LLM，后续接真实论文集时只需要替换 case 和 service 构造。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CASES_PATH = BACKEND_DIR / "tests" / "fixtures" / "bm25_regression_cases.json"


SCENARIOS = {
    "vector_only": {
        "description": "旧 chunk-level vector baseline",
        "multi_index": False,
        "enable_keyword_search": False,
        "enable_query_rewrite": False,
        "enable_hyde": False,
        "enable_context_expansion": False,
    },
    "vector_multi_index": {
        "description": "vector route 使用 retrieval index 入口",
        "multi_index": True,
        "enable_keyword_search": False,
        "enable_query_rewrite": False,
        "enable_hyde": False,
        "enable_context_expansion": False,
    },
    "vector_bm25": {
        "description": "旧 chunk-level vector + BM25 baseline",
        "multi_index": False,
        "enable_keyword_search": True,
        "enable_query_rewrite": False,
        "enable_hyde": False,
        "enable_context_expansion": False,
    },
    "vector_bm25_multi_index": {
        "description": "vector + index-level BM25",
        "multi_index": True,
        "enable_keyword_search": True,
        "enable_query_rewrite": False,
        "enable_hyde": False,
        "enable_context_expansion": False,
    },
    "full_pipeline": {
        "description": "query views + vector + BM25 + context selection；rerank 由 --rerank 控制",
        "multi_index": True,
        "enable_keyword_search": True,
        "enable_query_rewrite": True,
        "enable_hyde": True,
        "enable_context_expansion": True,
    },
}


MULTI_INDEX_ROWS = {
    "chunk-method": [
        ("question", "What method framework pipeline is used?", 0.82),
        ("summary", "The method framework uses a retrieval pipeline with two encoder stages.", 0.78),
    ],
    "chunk-experiment": [
        ("question", "What experimental setup and evaluation metrics are used?", 0.82),
        ("summary", "Evaluation uses LongBench, strong baselines, and exact match metrics.", 0.78),
    ],
    "chunk-results": [
        ("question", "What does the ablation study show against baselines?", 0.82),
        ("summary", "Ablation results show stronger performance and comparison against baselines.", 0.78),
    ],
    "chunk-dataset": [
        ("question", "Which datasets and benchmark corpus are used?", 0.82),
        ("summary", "The benchmark corpus includes training, development, and test splits.", 0.78),
    ],
    "chunk-limitation": [
        ("question", "What limitations and weaknesses are discussed?", 0.82),
        ("summary", "The limitation section says the method struggles on noisy prompts.", 0.78),
    ],
    "chunk-figure-table": [
        ("question", "What do Figure 2 and Table 3 show about the results?", 0.82),
        ("summary", "Figure 2 and Table 3 summarize the main experimental trend and best score.", 0.78),
    ],
}


def load_cases(case_id: Optional[str] = None) -> List[Dict[str, Any]]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = [
        case for case in payload.get("cases", [])
        if case.get("expected_top_chunk")
    ]
    if case_id:
        cases = [case for case in cases if case.get("id") == case_id]
    return cases


def build_offline_service(*, multi_index: bool) -> tuple[Any, str]:
    """构建离线检索服务；multi-index 场景追加 retrieval index 级向量行。"""
    from tests.helpers.retrieval import build_retrieval_service

    service, collection_name, *_ = build_retrieval_service()
    # benchmark 需要观察 recall@10，测试 helper 默认 top_k 较小，这里只调整本 runner 的运行时副本。
    service.retrieval_pipeline.enhanced_config.update(
        {
            "recall_candidate_limit": 30,
            "rrf_candidate_limit": 30,
            "rerank_candidate_limit": 30,
            "final_context_top_k": 10,
            "max_final_context_top_k": 10,
        }
    )
    if multi_index:
        add_fixture_multi_index_rows(service, collection_name)
    return service, collection_name


def add_fixture_multi_index_rows(service: Any, collection_name: str) -> None:
    rows = service.vector_store_service.collections[collection_name]
    by_chunk_id = {
        str((row.get("metadata") or {}).get("chunk_id") or row.get("chunk_id") or ""): row
        for row in rows
    }
    for chunk_id, index_specs in MULTI_INDEX_ROWS.items():
        source_row = by_chunk_id.get(chunk_id)
        if not source_row:
            continue
        base_metadata = dict(source_row.get("metadata") or {})
        content = str(source_row.get("content") or base_metadata.get("content") or "")
        for ordinal, (index_type, index_text, index_weight) in enumerate(index_specs, start=1):
            metadata = dict(
                base_metadata,
                retrieval_index_id=f"{chunk_id}:{index_type}:{ordinal}",
                retrieval_index_type=index_type,
                retrieval_index_text=index_text,
                retrieval_index_weight=index_weight,
                retrieval_index_enabled_routes=["vector_original", "vector_rewrite", "vector_hyde", "keyword"],
            )
            rows.append(
                {
                    "id": len(rows) + 1,
                    "content": content,
                    "embedding": service.embedding_service.create_single_embedding(index_text),
                    "metadata": metadata,
                }
            )


def run_scenario(
    scenario_name: str,
    scenario: Dict[str, Any],
    cases: List[Dict[str, Any]],
    *,
    enable_rerank: bool,
) -> Dict[str, Any]:
    from services.retrieval.contracts import RetrievalOptions

    service, collection_name = build_offline_service(multi_index=bool(scenario["multi_index"]))
    per_case: List[Dict[str, Any]] = []
    route_distribution: Counter[str] = Counter()
    index_type_contribution: Counter[str] = Counter()
    route_gold_hits: Counter[str] = Counter()
    keyword_backend_distribution: Counter[str] = Counter()
    keyword_backend_config_distribution: Counter[str] = Counter()
    keyword_backend_init_fallbacks: Counter[str] = Counter()
    keyword_backend_execution_fallbacks: Counter[str] = Counter()

    for case in cases:
        options = RetrievalOptions(
            top_k=10,
            debug=True,
            enable_query_rewrite=bool(scenario["enable_query_rewrite"]),
            enable_hyde=bool(scenario["enable_hyde"]),
            enable_keyword_search=bool(scenario["enable_keyword_search"]),
            enable_llm_rerank=bool(enable_rerank if scenario_name == "full_pipeline" else False),
            enable_context_expansion=bool(scenario["enable_context_expansion"]),
        )
        response = service.enhanced_retrieve(
            user_query=str(case["question"]),
            collection_name=collection_name,
            options=options,
        )
        debug = response.get("debug") or {}
        chunks = list(response.get("chunks") or [])
        gold_ids = [str(case["expected_top_chunk"])]
        case_result = evaluate_case(chunks, gold_ids)
        case_result.update(
            {
                "id": case.get("id"),
                "question": case.get("question"),
                "gold_chunk_ids": gold_ids,
                "top_chunks": [chunk.get("chunk_id") for chunk in chunks[:10]],
            }
        )
        per_case.append(case_result)

        multi_index = debug.get("multi_index") or {}
        route_distribution.update(multi_index.get("route_hit_distribution") or {})
        index_type_contribution.update(multi_index.get("index_type_contribution") or {})
        route_gold_hits.update(routes_that_hit_gold(debug.get("routes") or {}, gold_ids))
        if scenario["enable_keyword_search"]:
            keyword_debug = debug.get("keyword_search") or {}
            backend_name = str(keyword_debug.get("keyword_backend") or "unknown")
            backend_config = str(keyword_debug.get("keyword_backend_config") or "unknown")
            # benchmark 对比的是 keyword route 的真实后端；显式记录配置值、实际值和降级原因，
            # 避免 bm25s 缺失时把 internal_bm25 的指标误读成 bm25s 指标。
            keyword_backend_distribution[backend_name] += 1
            keyword_backend_config_distribution[backend_config] += 1
            case_result["keyword_backend"] = backend_name
            case_result["keyword_backend_config"] = backend_config
            case_result["keyword_backend_init_fallback"] = bool(
                keyword_debug.get("keyword_backend_init_fallback")
            )
            init_fallback_reason = str(
                keyword_debug.get("keyword_backend_init_fallback_reason") or ""
            )
            if case_result["keyword_backend_init_fallback"] or init_fallback_reason:
                keyword_backend_init_fallbacks[init_fallback_reason or "unknown"] += 1
            case_result["keyword_backend_execution_fallback"] = bool(
                keyword_debug.get("keyword_backend_fallback")
            )
            execution_fallback_reason = str(
                keyword_debug.get("keyword_backend_fallback_reason") or ""
            )
            if case_result["keyword_backend_execution_fallback"] or execution_fallback_reason:
                keyword_backend_execution_fallbacks[execution_fallback_reason or "unknown"] += 1

    return {
        "scenario": scenario_name,
        "description": scenario["description"],
        "case_count": len(per_case),
        "recall@5": mean(row["recall@5"] for row in per_case),
        "recall@10": mean(row["recall@10"] for row in per_case),
        "MRR": mean(row["mrr"] for row in per_case),
        "route_hit_distribution": dict(route_distribution),
        "route_gold_hit_distribution": dict(route_gold_hits),
        "index_type_contribution": dict(index_type_contribution),
        "keyword_backend_distribution": dict(keyword_backend_distribution),
        "keyword_backend_config_distribution": dict(keyword_backend_config_distribution),
        "keyword_backend_init_fallback_distribution": dict(keyword_backend_init_fallbacks),
        "keyword_backend_execution_fallback_distribution": dict(keyword_backend_execution_fallbacks),
        "cases": per_case,
    }


def evaluate_case(chunks: List[Dict[str, Any]], gold_ids: List[str]) -> Dict[str, Any]:
    ranked_ids = [str(chunk.get("chunk_id") or "") for chunk in chunks]
    gold = {str(item) for item in gold_ids if str(item)}
    rank = next((idx for idx, chunk_id in enumerate(ranked_ids, start=1) if chunk_id in gold), None)
    return {
        "recall@5": 1.0 if any(chunk_id in gold for chunk_id in ranked_ids[:5]) else 0.0,
        "recall@10": 1.0 if any(chunk_id in gold for chunk_id in ranked_ids[:10]) else 0.0,
        "mrr": (1.0 / rank) if rank else 0.0,
        "gold_rank": rank,
    }


def routes_that_hit_gold(routes: Dict[str, List[Dict[str, Any]]], gold_ids: Iterable[str]) -> Counter[str]:
    gold = {str(item) for item in gold_ids}
    hits: Counter[str] = Counter()
    for route_name, route_results in routes.items():
        if any(str(item.get("chunk_id") or "") in gold for item in route_results):
            hits[route_name] += 1
    return hits


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("# Multi-index Retrieval Benchmark")
    for result in results:
        print("")
        print(f"## {result['scenario']}")
        print(f"- description: {result['description']}")
        print(f"- cases: {result['case_count']}")
        print(f"- recall@5: {result['recall@5']}")
        print(f"- recall@10: {result['recall@10']}")
        print(f"- MRR: {result['MRR']}")
        print(f"- route_hit_distribution: {result['route_hit_distribution']}")
        print(f"- route_gold_hit_distribution: {result['route_gold_hit_distribution']}")
        print(f"- index_type_contribution: {result['index_type_contribution']}")
        if result.get("keyword_backend_distribution"):
            print(f"- keyword_backend_distribution: {result['keyword_backend_distribution']}")
            print(f"- keyword_backend_config_distribution: {result['keyword_backend_config_distribution']}")
        if result.get("keyword_backend_init_fallback_distribution"):
            print(
                "- keyword_backend_init_fallback_distribution: "
                f"{result['keyword_backend_init_fallback_distribution']}"
            )
        if result.get("keyword_backend_execution_fallback_distribution"):
            print(
                "- keyword_backend_execution_fallback_distribution: "
                f"{result['keyword_backend_execution_fallback_distribution']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-index RAG 最小检索评估 runner")
    parser.add_argument("--case", help="只跑指定 case id", default=None)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        help="只跑指定 scenario；省略时跑全部",
        default=None,
    )
    parser.add_argument("--rerank", action="store_true", help="full_pipeline 场景启用 LLM rerank")
    parser.add_argument("--json", dest="json_out", help="把结果写入 JSON 文件，便于前后对比", default=None)
    args = parser.parse_args()

    cases = load_cases(args.case)
    if not cases:
        print(f"no cases matched (case={args.case!r})", file=sys.stderr)
        return 1

    scenario_items = (
        [(args.scenario, SCENARIOS[args.scenario])]
        if args.scenario
        else list(SCENARIOS.items())
    )
    results = [
        run_scenario(name, scenario, cases, enable_rerank=args.rerank)
        for name, scenario in scenario_items
    ]
    print_summary(results)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nresults written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
