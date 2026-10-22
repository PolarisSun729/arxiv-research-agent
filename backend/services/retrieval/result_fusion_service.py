from __future__ import annotations

from typing import Any, Dict, List

from services.retrieval.contracts import FusionResult, QueryProfile, legacy_body_index_id


class ResultFusionService:
    """负责 route 去重和 RRF 融合，不承担具体召回与 rerank。

    RRF（Reciprocal Rank Fusion）：每条路由按名次给候选投票 1/(k + rank)，
    再乘以路由权重和路由置信度后累加；只看名次不看原始分，避免不同路由分数量纲不一致。
    """

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
        """对已去重的各路由结果做加权 RRF 融合，返回融合分最高的 top_k 个 chunk。

        同分时依次比较各路由原始分的最大值、路由置信度的最大值。
        """
        aggregated: Dict[str, Dict[str, Any]] = {}
        route_weights = self.route_weights_for_intent(query_profile.intent_profile)
        for route_name, route_results in routes.items():
            weight = route_weights.get(route_name, self.route_weights.get(route_name, 1.0))
            for rank, item in enumerate(route_results):
                item = self.with_matched_index_fields(item, route_name=route_name)
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
                        "matched_indexes": [],
                        "rrf_votes": {},
                    },
                )
                # RRF 投票只依赖 route 排名、route 权重和 route confidence，避免提前混入 rerank 语义。
                multiplier = self._table_structured_vote_multiplier(route_name, item)
                vote = weight * route_confidence * (1.0 / (self.rrf_k + rank + 1)) * multiplier
                entry["score"] += vote
                # 逐路记下投票的各个因子，排查时能看出融合名次具体由哪一路、哪个因子决定。
                entry["rrf_votes"][route_name] = {
                    "rank": rank + 1,
                    "weight": weight,
                    "confidence": route_confidence,
                    "multiplier": multiplier,
                    "vote": vote,
                }
                entry["matched_routes"].append(route_name)
                entry["route_scores"][route_name] = item.get("route_score")
                entry["route_confidences"][route_name] = route_confidence
                entry["matched_indexes"] = self.merge_matched_indexes(
                    entry.get("matched_indexes", []),
                    item,
                    route_name=route_name,
                )
                # 融合结果如果命中 v2 表格证据，主路由指向 table_structured，便于下游优先展开结构化证据。
                if route_name == "table_structured" and item.get("table_evidence"):
                    entry["retrieval_route"] = "table_structured"
                    entry["table_evidence"] = item.get("table_evidence", entry.get("table_evidence", {}))
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
        """取当前意图下的路由权重（意图未配置时用服务默认值），并保证 memory_context 有默认权重。"""
        if intent_profile is None:
            weights = dict(self.route_weights)
        else:
            weights = dict(getattr(intent_profile, "route_weights", None) or self.route_weights)
        weights.setdefault("memory_context", 0.42)
        return weights

    def dedupe_preserve_order(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """与 dedupe_route_results 完全相同（历史上是两份复制的实现），保留该名字兼容已有调用方。"""
        return self.dedupe_route_results(items)

    def dedupe_route_results(self, route_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 chunk 去重并保持首次出现的顺序；重复命中的 index 明细合并到首条记录的 matched_indexes。"""
        deduped: List[Dict[str, Any]] = []
        seen: Dict[str, Dict[str, Any]] = {}
        for item in route_results:
            item = self.with_matched_index_fields(item)
            key = self.chunk_unique_key(item)
            if key in seen:
                seen[key]["matched_indexes"] = self.merge_matched_indexes(
                    seen[key].get("matched_indexes", []),
                    item,
                    route_name=item.get("retrieval_route"),
                )
                continue
            normalized = dict(item)
            normalized["matched_indexes"] = self.merge_matched_indexes([], normalized, route_name=normalized.get("retrieval_route"))
            seen[key] = normalized
            deduped.append(normalized)
        return deduped

    def build_raw_retrieval_top_n(
        self,
        routes: Dict[str, List[Dict[str, Any]]],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """按路由顺序依次拼接各路由结果，取前 limit 条作为“融合前”的原始召回快照，供 trace 对比。"""
        raw_results: List[Dict[str, Any]] = []
        for route_results in routes.values():
            for item in route_results:
                raw_results.append(dict(item))
                if len(raw_results) >= limit:
                    return raw_results[:limit]
        return raw_results[:limit]

    @staticmethod
    def with_matched_index_fields(item: Dict[str, Any], route_name: Any = None) -> Dict[str, Any]:
        """返回补齐了 index_* / retrieval_index_* / matched_index_* 三套字段的副本（不修改入参）。

        matched_index_* 取值优先级：matched → retrieval → index；都缺失时合成 legacy body index。
        """
        normalized = dict(item)
        matched_index_id = (
            normalized.get("matched_index_id")
            or normalized.get("retrieval_index_id")
            or normalized.get("index_id")
            or ""
        )
        matched_index_type = (
            normalized.get("matched_index_type")
            or normalized.get("retrieval_index_type")
            or normalized.get("index_type")
            or ""
        )
        matched_index_text = (
            normalized.get("matched_index_text")
            or normalized.get("retrieval_index_text")
            or normalized.get("index_text")
            or ""
        )
        if not matched_index_id:
            chunk_ref = normalized.get("parent_chunk_id") or normalized.get("original_chunk_id") or normalized.get("chunk_id") or 0
            # 旧 chunk-level 结果没有 index 字段，融合层合成 body index，保证外部结果始终有命中入口。
            matched_index_id = legacy_body_index_id(chunk_ref)
            matched_index_type = "body"
            matched_index_text = str(normalized.get("content") or normalized.get("text") or "")
        matched_index_score = normalized.get("matched_index_score", normalized.get("route_score", normalized.get("score")))
        normalized["index_id"] = normalized.get("index_id") or matched_index_id
        normalized["index_type"] = normalized.get("index_type") or matched_index_type
        normalized["index_text"] = normalized.get("index_text") or matched_index_text
        normalized["index_weight"] = normalized.get("index_weight") or normalized.get("retrieval_index_weight", 1.0)
        normalized["retrieval_index_id"] = normalized.get("retrieval_index_id") or matched_index_id
        normalized["retrieval_index_type"] = normalized.get("retrieval_index_type") or matched_index_type
        normalized["retrieval_index_text"] = normalized.get("retrieval_index_text") or matched_index_text
        normalized["retrieval_index_weight"] = normalized.get("retrieval_index_weight") or normalized.get("index_weight", 1.0)
        normalized["matched_index_id"] = matched_index_id
        normalized["matched_index_type"] = matched_index_type
        normalized["matched_index_text"] = matched_index_text
        normalized["matched_index_score"] = matched_index_score
        if route_name and not normalized.get("retrieval_route"):
            normalized["retrieval_route"] = route_name
        return normalized

    @classmethod
    def merge_matched_indexes(
        cls,
        current: List[Dict[str, Any]],
        item: Dict[str, Any],
        *,
        route_name: Any = None,
    ) -> List[Dict[str, Any]]:
        """把 item 的命中 index（或其自带的 matched_indexes 明细）合并进 current，返回新列表。"""
        merged = [dict(row) for row in (current or []) if isinstance(row, dict)]
        normalized = cls.with_matched_index_fields(item, route_name=route_name)
        source_indexes = [
            row for row in normalized.get("matched_indexes", []) or []
            if isinstance(row, dict)
        ]
        if source_indexes:
            # keyword route 可能已经聚合了同一 chunk 的多个 index 命中；融合层需要保留这些明细。
            for source_index in source_indexes:
                merged = cls._merge_single_matched_index(
                    merged,
                    {**normalized, **source_index},
                    route_name=route_name,
                )
            return merged
        merged = cls._merge_single_matched_index(merged, normalized, route_name=route_name)
        return merged

    @classmethod
    def _merge_single_matched_index(
        cls,
        merged: List[Dict[str, Any]],
        item: Dict[str, Any],
        *,
        route_name: Any = None,
    ) -> List[Dict[str, Any]]:
        """合并单条 index 命中，保持 route/source 去重逻辑集中在一个位置。"""
        normalized = cls.with_matched_index_fields(item, route_name=route_name)
        index_id = str(normalized.get("matched_index_id") or "")
        if not index_id:
            return merged
        entry = {
            "matched_index_id": index_id,
            "matched_index_type": str(normalized.get("matched_index_type") or ""),
            "matched_index_text": str(normalized.get("matched_index_text") or ""),
            "matched_index_score": normalized.get("matched_index_score"),
            "retrieval_route": str(route_name or normalized.get("retrieval_route") or ""),
            "source_query": str(normalized.get("source_query") or ""),
        }
        for existing in merged:
            if existing.get("matched_index_id") != index_id:
                continue
            route = entry["retrieval_route"]
            if route:
                routes = list(existing.get("retrieval_routes") or [])
                if route not in routes:
                    routes.append(route)
                existing["retrieval_routes"] = routes
            source_query = entry["source_query"]
            if source_query:
                source_queries = list(existing.get("source_queries") or [])
                if source_query not in source_queries:
                    source_queries.append(source_query)
                existing["source_queries"] = source_queries
            if existing.get("matched_index_score") is None:
                existing["matched_index_score"] = entry["matched_index_score"]
            return merged
        entry["retrieval_routes"] = [entry["retrieval_route"]] if entry["retrieval_route"] else []
        entry["source_queries"] = [entry["source_query"]] if entry["source_query"] else []
        merged.append(entry)
        return merged

    @staticmethod
    def _table_structured_vote_multiplier(route_name: str, item: Dict[str, Any]) -> float:
        """结构化单元格证据更接近最终答案，融合时给一个有限提权。"""
        if route_name != "table_structured":
            return 1.0
        evidence = item.get("table_evidence") or {}
        if not isinstance(evidence, dict):
            return 1.0
        final_evidence = evidence.get("final_evidence") if isinstance(evidence.get("final_evidence"), dict) else {}
        candidate_evidence = evidence.get("candidate_evidence") if isinstance(evidence.get("candidate_evidence"), dict) else {}
        has_cells = bool(final_evidence.get("cells") or candidate_evidence.get("candidate_cells"))
        if not has_cells:
            return 1.0
        numeric_operation = str(evidence.get("operation_hint", "") or "").strip().lower()
        decision = str(evidence.get("decision", "") or "").strip().lower()
        confidence = float(evidence.get("confidence", 0.0) or 0.0)
        multiplier = 1.0 + min(0.3, confidence * 0.22)
        if decision == "compute" and numeric_operation in {"max", "min", "difference"}:
            multiplier += 0.08
        elif decision == "defer_to_llm":
            # 候选证据能帮助 rerank，但还没有规则层最终答案，因此只给有限提权。
            multiplier += 0.03
        return multiplier

    @staticmethod
    def chunk_unique_key(item: Dict[str, Any]) -> str:
        """生成跨路由的 chunk 唯一键：同一资产/来源/chunk/分片/页码/顺序视为同一个证据。

        分片标签为空时取 subchunk_label，与向量库入库时的规则一致：关键词索引从 chunk 文件构建，
        只有 subchunk_label；不补齐的话，同一 chunk 的关键词命中和向量命中键不同，融合时合并不上。
        超长 chunk 拆出的各段共用 original_chunk_id，只能靠分片标签区分，所以标签不能从键里去掉。
        """
        return "|".join(
            [
                str(item.get("chunk_type", "text")),
                str(item.get("asset_kind", "")),
                str(item.get("asset_path", "")),
                str(item.get("source", "")),
                str(item.get("original_chunk_id", item.get("parent_chunk_id", item.get("chunk_id", 0)))),
                str(item.get("content_part_label") or item.get("subchunk_label") or ""),
                str(item.get("page_range", "")),
                str(item.get("order_index", "")),
            ]
        )
