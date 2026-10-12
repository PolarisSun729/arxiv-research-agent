from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.retrieval.contracts import legacy_body_index_id
from services.retrieval.numeric_utils import first_float, min_max_normalize
from services.retrieval.retrieval_index import RETRIEVAL_INDEX_WEIGHTS


class IndexHitAggregator:
    """将 index-level 命中聚合为 chunk-level candidate，保持下游 RRF/rerank/context 仍按 chunk 工作。

    一个 chunk 可能通过多个 index（正文、标题、表格摘要等）被同一路由命中多次；
    聚合时取加权分最高的 index 作为代表，并保留前 max_matched_indexes 个命中供 trace 解释。
    """

    def __init__(
        self,
        *,
        max_matched_indexes: int = 3,
        diversity_bonus_per_type: float = 0.005,
        max_diversity_bonus: float = 0.01,
    ) -> None:
        self.max_matched_indexes = max(1, int(max_matched_indexes or 3))
        self.diversity_bonus_per_type = max(0.0, float(diversity_bonus_per_type or 0.0))
        self.max_diversity_bonus = max(0.0, float(max_diversity_bonus or 0.0))

    def aggregate(
        self,
        hits: List[Dict[str, Any]],
        *,
        route_name: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """把一条路由的 index 级命中聚合成 chunk 级候选，并按首次命中名次排序、截断、归一化分数。

        排序以 best_rank（该 chunk 最早出现的命中名次）为主，保持路由原有的相对顺序；
        同名次时再按路由分和原始 index 分倒序。
        """
        if not hits:
            return []

        groups: Dict[str, Dict[str, Any]] = {}
        for hit_rank, raw_hit in enumerate(hits, start=1):
            if not isinstance(raw_hit, dict):
                continue
            hit = dict(raw_hit)
            chunk_key = self._chunk_key(hit, fallback_index=hit_rank)
            index_entries = self._matched_indexes_for_hit(hit, route_name=route_name)
            if not index_entries:
                continue

            group = groups.setdefault(
                chunk_key,
                {
                    "best_hit": hit,
                    "best_weighted_score": float("-inf"),
                    "best_raw_score": float("-inf"),
                    "best_rank": hit_rank,
                    "matched_indexes": [],
                    "matched_index_ids": set(),
                    "matched_index_types": [],
                    "source_queries": [],
                },
            )
            for entry in index_entries:
                index_id = str(entry.get("matched_index_id") or entry.get("index_id") or "")
                # 同一 index 被多条子查询重复命中时只记一次，避免命中数虚高。
                if index_id and index_id in group["matched_index_ids"]:
                    continue
                if index_id:
                    group["matched_index_ids"].add(index_id)
                weighted_score = float(entry.get("weighted_index_score", 0.0) or 0.0)
                raw_score = float(entry.get("matched_index_score", 0.0) or 0.0)
                entry["hit_rank"] = hit_rank
                group["matched_indexes"].append(entry)
                index_type = str(entry.get("matched_index_type") or entry.get("index_type") or "body")
                if index_type not in group["matched_index_types"]:
                    group["matched_index_types"].append(index_type)
                source_query = str(entry.get("source_query") or hit.get("source_query") or "").strip()
                if source_query and source_query not in group["source_queries"]:
                    group["source_queries"].append(source_query)
                # 加权分更高者成为代表命中；分数相同时保留名次更靠前的那条。
                if (
                    weighted_score > float(group["best_weighted_score"])
                    or (
                        weighted_score == float(group["best_weighted_score"])
                        and hit_rank < int(group.get("best_rank", hit_rank))
                    )
                ):
                    group["best_weighted_score"] = weighted_score
                    group["best_raw_score"] = raw_score
                    group["best_hit"] = hit
                    group["best_rank"] = hit_rank

        candidates: List[Dict[str, Any]] = []
        for group in groups.values():
            matched_indexes = sorted(
                group["matched_indexes"],
                key=lambda item: (
                    float(item.get("weighted_index_score", 0.0) or 0.0),
                    float(item.get("matched_index_score", 0.0) or 0.0),
                    -int(item.get("hit_rank", 0) or 0),
                ),
                reverse=True,
            )
            if not matched_indexes:
                continue
            best_index = dict(matched_indexes[0])
            matched_index_types = list(group.get("matched_index_types", []) or [])
            # 多种 index_type 同时命中说明证据更稳定，但 bonus 很小，避免长 chunk 因 index 数量多而占优。
            diversity_bonus = min(
                self.max_diversity_bonus,
                max(0, len(set(matched_index_types)) - 1) * self.diversity_bonus_per_type,
            )
            route_score = float(group["best_weighted_score"]) + diversity_bonus
            candidate = dict(group["best_hit"])
            candidate["retrieval_route"] = route_name
            candidate["route_score"] = float(route_score)
            candidate["matched_indexes"] = matched_indexes[: self.max_matched_indexes]
            candidate["best_matched_index"] = best_index
            candidate["matched_index_types"] = matched_index_types
            candidate["source_queries"] = list(group.get("source_queries", []) or [])
            candidate["index_aggregation_applied"] = True
            candidate["index_aggregation_strategy"] = "max_weighted_index_score"
            candidate["index_aggregation_bonus"] = float(diversity_bonus)
            candidate["index_aggregation_hit_count"] = len(matched_indexes)
            candidate["index_aggregation_distinct_type_count"] = len(set(matched_index_types))
            candidate["index_aggregation_best_rank"] = int(group.get("best_rank", 0) or 0)
            self._apply_best_index_aliases(candidate, best_index)
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                int(item.get("index_aggregation_best_rank", 0) or 0),
                -float(item.get("route_score", 0.0) or 0.0),
                -float((item.get("best_matched_index") or {}).get("matched_index_score", 0.0) or 0.0),
            )
        )
        limited = candidates[:top_k] if top_k is not None and top_k > 0 else candidates
        normalized_scores = min_max_normalize([float(item.get("route_score", 0.0) or 0.0) for item in limited])
        for rank, (item, normalized_score) in enumerate(zip(limited, normalized_scores), start=1):
            item["route_rank"] = rank
            item["normalized_route_score"] = float(normalized_score)
        return limited

    def _matched_indexes_for_hit(self, hit: Dict[str, Any], *, route_name: str) -> List[Dict[str, Any]]:
        """取出命中里携带的 matched_indexes；没有时把命中本身视为一个 index 条目。"""
        raw_entries = [row for row in (hit.get("matched_indexes") or []) if isinstance(row, dict)]
        if raw_entries:
            return [self._normalize_index_entry(hit, row, route_name=route_name) for row in raw_entries]
        return [self._normalize_index_entry(hit, {}, route_name=route_name)]

    def _normalize_index_entry(
        self,
        hit: Dict[str, Any],
        entry: Dict[str, Any],
        *,
        route_name: str,
    ) -> Dict[str, Any]:
        """把 index 条目补齐为统一结构：entry 字段优先、hit 字段兜底，并计算 index 类型加权后的分数。"""
        index_id = str(
            entry.get("matched_index_id")
            or entry.get("index_id")
            or hit.get("matched_index_id")
            or hit.get("retrieval_index_id")
            or hit.get("index_id")
            or ""
        )
        index_type = str(
            entry.get("matched_index_type")
            or entry.get("index_type")
            or hit.get("matched_index_type")
            or hit.get("retrieval_index_type")
            or hit.get("index_type")
            or "body"
        )
        index_text = str(
            entry.get("matched_index_text")
            or entry.get("index_text")
            or hit.get("matched_index_text")
            or hit.get("retrieval_index_text")
            or hit.get("index_text")
            or ""
        )
        if not index_id:
            chunk_ref = hit.get("chunk_id") or hit.get("parent_chunk_id") or hit.get("original_chunk_id") or "unknown"
            # 旧 chunk-level 命中没有 index 字段时合成 body index，保持聚合输出字段完整。
            index_id = legacy_body_index_id(chunk_ref)
            index_type = "body"
            index_text = str(hit.get("content") or hit.get("text") or "")

        # 不同路由把原始分写在不同字段里（向量分、BM25 融合分等），按优先级取第一个有效值。
        raw_score = first_float(
            entry.get("matched_index_score"),
            entry.get("bm25_fusion_score"),
            entry.get("best_raw_bm25_score"),
            hit.get("matched_index_score"),
            hit.get("route_score"),
            hit.get("score"),
        )
        try:
            index_type_weight = float(
                entry.get("index_weight")
                or entry.get("retrieval_index_weight")
                or hit.get("index_weight")
                or hit.get("retrieval_index_weight")
                or RETRIEVAL_INDEX_WEIGHTS.get(index_type, 1.0)
                or 1.0
            )
        except (TypeError, ValueError):
            index_type_weight = float(RETRIEVAL_INDEX_WEIGHTS.get(index_type, 1.0) or 1.0)

        normalized = dict(entry)
        normalized.update(
            {
                "index_id": index_id,
                "index_type": index_type,
                "index_text": index_text,
                "index_weight": index_type_weight,
                "index_type_weight": index_type_weight,
                "matched_index_id": index_id,
                "matched_index_type": index_type,
                "matched_index_text": index_text,
                "matched_index_score": float(raw_score),
                "weighted_index_score": float(raw_score) * index_type_weight,
                "retrieval_route": str(entry.get("retrieval_route") or hit.get("retrieval_route") or route_name),
                "source_query": str(entry.get("source_query") or hit.get("source_query") or ""),
            }
        )
        return normalized

    @staticmethod
    def _apply_best_index_aliases(candidate: Dict[str, Any], best_index: Dict[str, Any]) -> None:
        """把代表 index 的字段同步写到候选顶层（index_* / retrieval_index_* / matched_index_* 三套别名）。"""
        candidate["index_id"] = best_index.get("index_id", "")
        candidate["index_type"] = best_index.get("index_type", "")
        candidate["index_text"] = best_index.get("index_text", "")
        candidate["index_weight"] = best_index.get("index_weight", 1.0)
        candidate["retrieval_index_id"] = best_index.get("index_id", "")
        candidate["retrieval_index_type"] = best_index.get("index_type", "")
        candidate["retrieval_index_text"] = best_index.get("index_text", "")
        candidate["retrieval_index_weight"] = best_index.get("index_weight", 1.0)
        candidate["matched_index_id"] = best_index.get("matched_index_id", best_index.get("index_id", ""))
        candidate["matched_index_type"] = best_index.get("matched_index_type", best_index.get("index_type", ""))
        candidate["matched_index_text"] = best_index.get("matched_index_text", best_index.get("index_text", ""))
        candidate["matched_index_score"] = best_index.get("matched_index_score")

    @staticmethod
    def _chunk_key(hit: Dict[str, Any], *, fallback_index: int) -> str:
        """生成 chunk 聚合键：优先用非 0 的 chunk 编号；都缺失时用来源+页码+顺序+命中名次拼出唯一键。"""
        for key in ("chunk_id", "parent_chunk_id", "original_chunk_id"):
            value = str(hit.get(key, "") or "").strip()
            if value and value != "0":
                return f"{key}:{value}"
        return "|".join(
            [
                "fallback",
                str(hit.get("source", "")),
                str(hit.get("page_range", "")),
                str(hit.get("order_index", "")),
                str(fallback_index),
            ]
        )
