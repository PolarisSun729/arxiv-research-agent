from __future__ import annotations

from typing import Any, Dict, List

from services.retrieval.contracts import FusionResult, QueryProfile


class ResultFusionService:
    """负责 route 去重和 RRF 融合，不承担具体召回与 rerank。"""

    def __init__(
        self,
        *,
        rrf_k: int,
        route_weights: Dict[str, float],
    ) -> None:
        self.rrf_k = rrf_k
        self.route_weights = dict(route_weights or {})

    def fuse(
        self,
        routes: Dict[str, List[Dict[str, Any]]],
        *,
        fused_limit: int,
        raw_limit: int,
        fused_stage_limit: int,
        query_profile: QueryProfile,
    ) -> FusionResult:
        """先对各 route 去重，再做 RRF 融合，输出 raw/fused 两级结果。"""
        deduped_routes = {
            route_name: self.dedupe_route_results(route_results)
            for route_name, route_results in routes.items()
        }
        raw_retrieval_top_n = self.build_raw_retrieval_top_n(deduped_routes, limit=raw_limit)
        fused_results = self.fuse_routes(deduped_routes, fused_limit, query_profile)
        return FusionResult(
            deduped_routes=deduped_routes,
            raw_retrieval_top_n=raw_retrieval_top_n,
            fused_results=fused_results,
            fused_top_n=fused_results[:fused_stage_limit],
        )

    def fuse_routes(
        self,
        routes: Dict[str, List[Dict[str, Any]]],
        top_k: int,
        query_profile: QueryProfile,
    ) -> List[Dict[str, Any]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        route_weights = self.route_weights_for_intent(query_profile.intent_profile)
        for route_name, route_results in routes.items():
            weight = route_weights.get(route_name, self.route_weights.get(route_name, 1.0))
            for rank, item in enumerate(route_results):
                chunk_key = self.chunk_unique_key(item)
                route_confidence = float(item.get("route_confidence", 1.0) or 1.0)
                entry = aggregated.setdefault(
                    chunk_key,
                    {
                        **item,
                        "score": 0.0,
                        "matched_routes": [],
                        "route_scores": {},
                        "route_confidences": {},
                        "source_queries": [],
                    },
                )
                # RRF 投票只依赖 route 排名、route 权重和 route confidence，避免提前混入 rerank 语义。
                vote = weight * route_confidence * (1.0 / (self.rrf_k + rank + 1))
                vote *= self._table_structured_vote_multiplier(route_name, item)
                entry["score"] += vote
                entry["matched_routes"].append(route_name)
                entry["route_scores"][route_name] = item.get("route_score")
                entry["route_confidences"][route_name] = route_confidence
                # 融合结果如果命中了结构化表格证据，主路由应指向 table_structured，方便下游调试与断言看到真实证据来源。
                if route_name == "table_structured" and item.get("table_structured_evidence"):
                    entry["retrieval_route"] = "table_structured"
                    entry["table_structured_text"] = item.get("table_structured_text", entry.get("table_structured_text", ""))
                    entry["table_structured_evidence"] = item.get("table_structured_evidence", entry.get("table_structured_evidence", {}))
                    entry["table_structured_reason"] = item.get("table_structured_reason", entry.get("table_structured_reason", ""))
                    entry["table_structured_confidence"] = item.get(
                        "table_structured_confidence",
                        entry.get("table_structured_confidence", 0.0),
                    )
                if item.get("source_query") and item["source_query"] not in entry["source_queries"]:
                    entry["source_queries"].append(item["source_query"])

        fused = sorted(
            aggregated.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                max(item.get("route_scores", {}).values() or [0.0]),
                max(item.get("route_confidences", {}).values() or [0.0]),
            ),
            reverse=True,
        )
        return fused[:top_k]

    def route_weights_for_intent(self, intent_profile: Any) -> Dict[str, float]:
        if intent_profile is None:
            weights = dict(self.route_weights)
        else:
            weights = dict(getattr(intent_profile, "route_weights", None) or self.route_weights)
        weights.setdefault("memory_context", 0.42)
        return weights

    def dedupe_preserve_order(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for item in items:
            key = self.chunk_unique_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def dedupe_route_results(self, route_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in route_results:
            key = self.chunk_unique_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(item))
        return deduped

    def build_raw_retrieval_top_n(
        self,
        routes: Dict[str, List[Dict[str, Any]]],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        raw_results: List[Dict[str, Any]] = []
        for route_results in routes.values():
            for item in route_results:
                raw_results.append(dict(item))
                if len(raw_results) >= limit:
                    return raw_results[:limit]
        return raw_results[:limit]

    @staticmethod
    def _table_structured_vote_multiplier(route_name: str, item: Dict[str, Any]) -> float:
        """结构化单元格证据更接近最终答案，融合时给一个有限提权。"""
        if route_name != "table_structured":
            return 1.0
        evidence = item.get("table_structured_evidence") or {}
        if not isinstance(evidence, dict) or not evidence.get("matched_cells"):
            return 1.0
        numeric_operation = str(evidence.get("numeric_operation", "") or "").strip().lower()
        confidence = float(evidence.get("confidence", 0.0) or 0.0)
        multiplier = 1.0 + min(0.3, confidence * 0.22)
        if numeric_operation in {"max", "min", "difference"}:
            multiplier += 0.08
        return multiplier

    @staticmethod
    def chunk_unique_key(item: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(item.get("chunk_type", "text")),
                str(item.get("asset_kind", "")),
                str(item.get("asset_path", "")),
                str(item.get("source", "")),
                str(item.get("original_chunk_id", item.get("parent_chunk_id", item.get("chunk_id", 0)))),
                str(item.get("content_part_label", "")),
                str(item.get("page_range", "")),
                str(item.get("order_index", "")),
            ]
        )
