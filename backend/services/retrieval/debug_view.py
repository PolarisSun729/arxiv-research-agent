"""检索调试页的视图投影：把一次检索各阶段的原始 chunk 对齐成「chunk × 路由 / 阶段」的名次表。

这里只做整理，不检索也不打分；名次、融合票数和 rerank 分都来自 RetrievalPipeline 的真实输出
（检索时需带 include_stage_chunks=True 与 debug=True）。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

PREVIEW_CHARS = 160

# 运行配置里调试页需要展示的字段；其余（预算、token 估算等）与召回融合无关，不往前端带。
_CONFIG_KEYS = (
    "effective_top_k",
    "recall_candidate_limit",
    "rrf_candidate_limit",
    "rerank_candidate_limit",
    "enable_query_rewrite",
    "enable_hyde",
    "enable_keyword_search",
    "enable_table_structured_route",
    "enable_llm_rerank",
    "enable_context_expansion",
)
_INTENT_KEYS = ("main_intent", "sub_intents", "confidence", "intent_summary", "source", "fallback_reason")


def build_retrieval_debug_view(result: Dict[str, Any], *, chunk_key: Callable[[Dict[str, Any]], str]) -> Dict[str, Any]:
    """返回查询规划、各路概览和按融合名次排序的 chunk 行；chunk_key 必须与融合去重用的键一致。"""
    debug = result.get("debug") or {}
    stages = result.get("stage_chunks") or {}
    routes: Dict[str, List[Dict[str, Any]]] = stages.get("routes") or {}
    rerank_debug = debug.get("llm_rerank") or {}
    rows: Dict[str, Dict[str, Any]] = {}

    def row_for(item: Dict[str, Any]) -> Dict[str, Any]:
        key = chunk_key(item)
        if key not in rows:
            rows[key] = _new_row(key, item)
        return rows[key]

    for route_name, items in routes.items():
        for rank, item in enumerate(items, start=1):
            row = row_for(item)
            row["route_ranks"][route_name] = rank
            row["route_scores"][route_name] = _number(item.get("route_score"))
            if route_name == "keyword":
                row["keyword_terms"] = [str(term) for term in item.get("keyword_matched_terms") or []]
    for rank, item in enumerate(stages.get("fused") or [], start=1):
        row = row_for(item)
        row["fused_rank"] = rank
        row["fused_score"] = _number(item.get("score"))
        row["rrf_votes"] = {
            route_name: {name: _number(value) for name, value in vote.items()}
            for route_name, vote in (item.get("rrf_votes") or {}).items()
        }
        row["matched_indexes"] = [_matched_index(entry) for entry in item.get("matched_indexes") or []]
        row["source_queries"] = [str(query) for query in item.get("source_queries") or []]
    # rerank 关闭或失败时这一阶段只是沿用融合顺序，不算 rerank 名次，免得看起来像 rerank 给出的结果。
    if rerank_debug.get("applied"):
        for rank, item in enumerate(stages.get("reranked") or [], start=1):
            row = row_for(item)
            row["rerank_rank"] = rank
            row["rerank_score"] = _number(item.get("llm_rerank_score"))
    for rank, item in enumerate(stages.get("final") or [], start=1):
        row = row_for(item)
        row["final_rank"] = rank
        row["context_role"] = str(item.get("context_role") or "")
        row["final_reason"] = str(item.get("final_context_reason") or item.get("context_budget_reason") or "")

    fusion = debug.get("fusion") or {}
    route_weights = fusion.get("route_weights") or {}
    route_metrics = debug.get("route_metrics") or {}
    return {
        "query": _query_section(debug),
        "config": {key: (debug.get("config") or {}).get(key) for key in _CONFIG_KEYS},
        "fusion": {"rrf_k": fusion.get("rrf_k"), "route_weights": route_weights},
        "routes": [
            _route_summary(name, metric, len(routes.get(name) or []), route_weights.get(name))
            for name, metric in route_metrics.items()
            # rerank 的指标也挂在 route_metrics 里，但它不是召回路由。
            if name != "rerank"
        ],
        "rerank": {
            **{key: rerank_debug.get(key) for key in ("enabled", "applied", "mode", "reason", "input_chunks", "output_chunks")},
            "latency_ms": _number((route_metrics.get("rerank") or {}).get("latency_ms")),
        },
        "chunks": sorted(rows.values(), key=lambda row: (_rank_or_inf(row["fused_rank"]), _rank_or_inf(row["final_rank"]))),
    }


def _new_row(key: str, item: Dict[str, Any]) -> Dict[str, Any]:
    content = str(item.get("content") or item.get("text") or "")
    return {
        "key": key,
        "chunk_id": item.get("chunk_id"),
        "chunk_type": str(item.get("chunk_type") or "text"),
        "section": str(item.get("section_path") or item.get("section_title") or ""),
        "page": item.get("page_range") or item.get("page_number") or None,
        "preview": " ".join(content[:PREVIEW_CHARS].split()),
        "content": content,
        "route_ranks": {},
        "route_scores": {},
        "keyword_terms": [],
        "fused_rank": None,
        "fused_score": None,
        "rrf_votes": {},
        "matched_indexes": [],
        "source_queries": [],
        "rerank_rank": None,
        "rerank_score": None,
        "final_rank": None,
        "context_role": "",
        "final_reason": "",
    }


def _query_section(debug: Dict[str, Any]) -> Dict[str, Any]:
    intent = debug.get("intent_profile") or {}
    views = debug.get("query_views") or {}
    query_profile = debug.get("query_profile") or {}
    return {
        "original": debug.get("original_query") or "",
        # 原问向量路由实际检索用的英文译文；status 为 error 时退回原问。
        "original_translation": views.get("original_translation") or {},
        "intent": {key: intent.get(key) for key in _INTENT_KEYS},
        "rewrite_enabled": bool(views.get("enabled")),
        "rewrite_error": views.get("llm_error") or "",
        "view_queries": views.get("view_queries") or {},
        "selected_queries": list(views.get("selected_queries") or []),
        "candidates": [
            {key: row.get(key) for key in ("query", "source", "selected", "reason")}
            for row in views.get("candidates") or []
        ],
        "keywords": list(query_profile.get("keywords") or []),
        "hyde_text": debug.get("hyde_text") or "",
        "rerank_query": debug.get("rerank_query") or "",
    }


def _route_summary(name: str, metric: Dict[str, Any], candidate_count: int, weight: Any) -> Dict[str, Any]:
    return {
        "name": name,
        "status": str(metric.get("status") or ""),
        "enabled": bool(metric.get("enabled", True)),
        "latency_ms": _number(metric.get("latency_ms")),
        # 用去重后的条数，和名次表里这一路的行数对得上；metric 里的是去重前的命中数。
        "candidate_count": candidate_count,
        "weight": _number(weight),
        "error": str(metric.get("error") or ""),
        "fallback_reason": str(metric.get("fallback_reason") or ""),
    }


def _matched_index(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": str(entry.get("matched_index_type") or ""),
        "text": " ".join(str(entry.get("matched_index_text") or "")[:PREVIEW_CHARS].split()),
        "score": _number(entry.get("matched_index_score")),
        "routes": [str(route) for route in entry.get("retrieval_routes") or []],
    }


def _number(value: Any) -> Optional[float]:
    """转成可 JSON 序列化的 float；缺失、非数字和 NaN/inf 都返回 None。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rank_or_inf(rank: Optional[int]) -> float:
    return float(rank) if rank is not None else math.inf
