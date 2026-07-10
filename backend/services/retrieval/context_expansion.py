from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from services.retrieval.retrieval_index import CollectionRetrievalIndex
from services.retrieval.trace_builder import RetrievalTraceBuilder


@dataclass
class ExpansionAnchor:
    """rerank 命中 chunk 的结构化锚点，后续预算层会基于它决定是否真正扩上下文。"""

    chunk_id: Any
    parent_chunk_id: Any
    subchunk_index: Optional[int]
    subchunk_count: Optional[int]
    section_title: str
    section_path: str
    page_number: Any
    page_range: str
    chunk_type: str
    source_route: str
    fusion_score: Optional[float]
    rerank_score: Optional[float]
    is_final_context_chunk: bool
    original_rank: int
    content_preview: str = ""
    matched_routes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class ExpansionCandidate:
    """anchor 可扩展到的候选 chunk；这里只记录关系，不改变最终回答上下文。"""

    anchor_chunk_id: Any
    candidate_chunk_id: Any
    expansion_type: str
    relationship_type: str
    distance: int
    is_anchor: bool
    is_final_context_chunk: bool
    is_original_retrieval_hit: bool
    chunk_type: str
    section_path: str
    section_title: str
    page_number: Any
    page_range: str
    parent_chunk_id: Any
    subchunk_index: Optional[int]
    source_route: str
    fusion_score: Optional[float]
    rerank_score: Optional[float]
    expansion_score: float
    reason: str
    expansion_reason: str
    matched_routes: List[str] = field(default_factory=list)


@dataclass
class ExpansionPolicy:
    """按问题类型组合扩展动作和范围，避免策略散落成不可维护的 if-else。"""

    name: str
    actions: Tuple[str, ...]
    sibling_window: int = 1
    section_window: int = 2
    section_limit: int = 4
    page_window: int = 1
    page_limit: int = 4
    asset_limit: int = 3
    per_anchor_limit: int = 8
    summary_section_only: bool = False
    preferred_section_terms: Tuple[str, ...] = ()


class ContextBudgetSelector:
    """在 retrieval 阶段预筛 anchor 和扩展候选；最终 token 预算由 PromptBudgetPlanner 负责。"""

    def __init__(self, *, trace_builder: RetrievalTraceBuilder) -> None:
        self.trace_builder = trace_builder

    def select(
        self,
        *,
        reranked_chunks: List[Dict[str, Any]],
        context_expansion: Dict[str, Any],
        retrieval_index: Optional[CollectionRetrievalIndex],
        final_context_top_k: int,
        max_context_chars: int,
        enabled: bool,
        candidate_max_blocks: Optional[int] = None,
        candidate_max_tokens_soft: Optional[int] = None,
        token_chars_per_token: float = 3.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        candidate_limit = max(1, int(candidate_max_blocks or final_context_top_k))
        effective_max_context_chars = self._resolve_effective_char_budget(
            max_context_chars=max_context_chars,
            candidate_max_tokens_soft=candidate_max_tokens_soft,
            token_chars_per_token=token_chars_per_token,
        )
        fallback_chunks = self.trace_builder.mark_final_context_chunks(reranked_chunks[:candidate_limit])
        if not enabled:
            return fallback_chunks, self._fallback_debug(
                reason="context_expansion_disabled",
                enabled=False,
                fallback_chunks=fallback_chunks,
                max_context_chars=effective_max_context_chars,
                candidate_max_blocks=candidate_limit,
                candidate_max_tokens_soft=candidate_max_tokens_soft,
            )
        if not context_expansion.get("candidate_pool"):
            return fallback_chunks, self._fallback_debug(
                reason="candidate_pool_empty",
                enabled=True,
                fallback_chunks=fallback_chunks,
                max_context_chars=effective_max_context_chars,
                candidate_max_blocks=candidate_limit,
                candidate_max_tokens_soft=candidate_max_tokens_soft,
            )

        try:
            chunk_lookup = self._build_chunk_lookup(reranked_chunks, retrieval_index)
            anchor_order = [
                str(anchor.get("chunk_id", "") or "").strip()
                for anchor in (context_expansion.get("anchors") or [])
                if str(anchor.get("chunk_id", "") or "").strip()
            ]
            anchor_rank = {chunk_id: index for index, chunk_id in enumerate(anchor_order)}
            policy = context_expansion.get("policy") or {}
            policy_name = str(policy.get("name", "default") or "default")
            candidates = self._score_budget_candidates(
                context_expansion=context_expansion,
                chunk_lookup=chunk_lookup,
                anchor_rank=anchor_rank,
                policy=policy,
            )
            if not candidates:
                return fallback_chunks, self._fallback_debug(
                    reason="budget_candidates_empty",
                    enabled=True,
                    fallback_chunks=fallback_chunks,
                    max_context_chars=effective_max_context_chars,
                    candidate_max_blocks=candidate_limit,
                    candidate_max_tokens_soft=candidate_max_tokens_soft,
                )

            selected, decisions = self._select_with_budget(
                candidates,
                anchor_rank=anchor_rank,
                final_context_top_k=candidate_limit,
                max_context_chars=max(1, effective_max_context_chars),
                policy_name=str(policy.get("name", "default") or "default"),
            )
            if not selected:
                return fallback_chunks, self._fallback_debug(
                    reason="budget_selected_empty",
                    enabled=True,
                    fallback_chunks=fallback_chunks,
                    max_context_chars=effective_max_context_chars,
                    candidate_max_blocks=candidate_limit,
                    candidate_max_tokens_soft=candidate_max_tokens_soft,
                )

            ordered = self._order_final_context(selected, anchor_rank=anchor_rank, policy_name=policy_name)
            final_chunks = [self._annotate_final_chunk(item["chunk"], item) for item in ordered]
            debug = self._budget_debug(
                original_top_chunks=reranked_chunks[:candidate_limit],
                context_expansion=context_expansion,
                candidates=candidates,
                decisions=decisions,
                final_chunks=final_chunks,
                max_context_chars=effective_max_context_chars,
                policy_name=policy_name,
                candidate_max_blocks=candidate_limit,
                candidate_max_tokens_soft=candidate_max_tokens_soft,
            )
            return final_chunks, debug
        except Exception as exc:  # pragma: no cover - 预算层不能影响基础 QA 可用性
            return fallback_chunks, self._fallback_debug(
                reason=f"context_budget_error: {exc}",
                enabled=True,
                fallback_chunks=fallback_chunks,
                max_context_chars=effective_max_context_chars,
                candidate_max_blocks=candidate_limit,
                candidate_max_tokens_soft=candidate_max_tokens_soft,
            )

    def _score_budget_candidates(
        self,
        *,
        context_expansion: Dict[str, Any],
        chunk_lookup: Dict[str, Dict[str, Any]],
        anchor_rank: Dict[str, int],
        policy: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        scored: List[Dict[str, Any]] = []
        policy_name = str(policy.get("name", "default") or "default")
        for candidate in context_expansion.get("candidate_pool") or []:
            candidate_id = str(candidate.get("candidate_chunk_id", "") or "").strip()
            chunk = chunk_lookup.get(candidate_id)
            if not candidate_id or chunk is None:
                continue
            relationship_types = [str(item) for item in (candidate.get("relationship_types") or []) if str(item).strip()]
            role = self._context_role(candidate, chunk, relationship_types)
            anchor_ids = [str(item) for item in (candidate.get("expansion_source_anchor_ids") or []) if str(item).strip()]
            base_score = float(candidate.get("expansion_score", 0.0) or 0.0)
            role_score = self._role_score(role, policy_name)
            multi_anchor_bonus = min(max(len(anchor_ids) - 1, 0) * 0.08, 0.24)
            original_hit_bonus = 0.08 if candidate.get("is_original_retrieval_hit") else 0.0
            distance_penalty = self._relationship_distance_penalty(candidate)
            budget_score = round(base_score + role_score + multi_anchor_bonus + original_hit_bonus - distance_penalty, 6)
            scored.append(
                {
                    "candidate": candidate,
                    "chunk": chunk,
                    "candidate_chunk_id": candidate_id,
                    "context_role": role,
                    "relationship_types": relationship_types,
                    "anchor_ids": anchor_ids,
                    "anchor_group_index": min([anchor_rank.get(anchor_id, 9999) for anchor_id in anchor_ids] or [9999]),
                    "budget_score": budget_score,
                    "estimated_chars": self._estimate_chars(chunk),
                    "include_priority": self._include_priority(role, policy_name, candidate),
                }
            )
        return sorted(
            scored,
            key=lambda item: (item["include_priority"], item["budget_score"], -item["estimated_chars"]),
            reverse=True,
        )

    def _select_with_budget(
        self,
        candidates: List[Dict[str, Any]],
        *,
        anchor_rank: Dict[str, int],
        final_context_top_k: int,
        max_context_chars: int,
        policy_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        selected: List[Dict[str, Any]] = []
        selected_ids: Set[str] = set()
        decisions: List[Dict[str, Any]] = []
        used_chars = 0

        anchor_slots = 1 if final_context_top_k <= 1 else max(1, min(len(anchor_rank), int(final_context_top_k * 0.65)))
        # figure/table 问题常同时需要正文解释和命中的图表证据，至少保留两个 anchor 槽位避免主证据被预算规则挤掉。
        if policy_name == "figure_table" and final_context_top_k > 1:
            anchor_slots = max(anchor_slots, min(len(anchor_rank), 2))
        anchor_candidates = [
            item for item in candidates
            if item["context_role"] in {"anchor_evidence", "memory_context"}
        ]
        selected_anchor_count = 0
        for item in anchor_candidates[:anchor_slots]:
            include_reason = "include_anchor_evidence"
            selected.append(item)
            selected_ids.add(item["candidate_chunk_id"])
            used_chars += item["estimated_chars"]
            selected_anchor_count += 1
            decisions.append(self._decision(item, include=True, reason=include_reason, used_chars=used_chars, max_context_chars=max_context_chars))
            if len(selected) >= final_context_top_k:
                return selected, self._drop_unselected(candidates, selected_ids, decisions, reason="top_k_budget_exhausted")

        for item in candidates:
            if item["candidate_chunk_id"] in selected_ids:
                continue
            if (
                policy_name == "method_flow"
                and item["context_role"] == "anchor_evidence"
                and "sibling" in item.get("relationship_types", [])
                and final_context_top_k > 1
            ):
                item = {**item, "context_role": "sibling_context"}
            if (
                item["context_role"] in {"anchor_evidence", "memory_context"}
                and selected_anchor_count >= anchor_slots
                and final_context_top_k > 1
            ):
                decisions.append(self._decision(item, include=False, reason="anchor_slot_reserved_for_expansion", used_chars=used_chars, max_context_chars=max_context_chars))
                continue
            if len(selected) >= final_context_top_k:
                decisions.append(self._decision(item, include=False, reason="top_k_budget_exhausted", used_chars=used_chars, max_context_chars=max_context_chars))
                continue
            if used_chars + item["estimated_chars"] > max_context_chars and not self._is_protected_role(item["context_role"], policy_name):
                decisions.append(self._decision(item, include=False, reason="char_budget_exceeded", used_chars=used_chars, max_context_chars=max_context_chars))
                continue
            selected.append(item)
            selected_ids.add(item["candidate_chunk_id"])
            used_chars += item["estimated_chars"]
            if item["context_role"] in {"anchor_evidence", "memory_context"}:
                selected_anchor_count += 1
            # debug 需要能逐块复盘 include/drop，扩展上下文不能只在 final_context_after_budget 里隐式出现。
            decisions.append(self._decision(item, include=True, reason=self._include_reason(item), used_chars=used_chars, max_context_chars=max_context_chars))
        if len(selected) < final_context_top_k:
            for item in anchor_candidates:
                if item["candidate_chunk_id"] in selected_ids:
                    continue
                if len(selected) >= final_context_top_k:
                    break
                if used_chars + item["estimated_chars"] > max_context_chars and not self._is_protected_role(item["context_role"], policy_name):
                    continue
                # 扩展候选不足时用剩余高分 anchor 补齐，保证旧索引或稀疏结构不会因为预算层丢失基础证据。
                selected.append(item)
                selected_ids.add(item["candidate_chunk_id"])
                used_chars += item["estimated_chars"]
                decisions.append(self._decision(item, include=True, reason="include_anchor_to_fill_budget", used_chars=used_chars, max_context_chars=max_context_chars))
        return selected, decisions

    def _order_final_context(self, selected: List[Dict[str, Any]], *, anchor_rank: Dict[str, int], policy_name: str) -> List[Dict[str, Any]]:
        return sorted(
            selected,
            key=lambda item: (
                self._final_context_priority(item, policy_name),
                item["anchor_group_index"],
                self._role_order(item["context_role"]),
                self._optional_int(item["chunk"].get("page_number")) or 9999,
                self._optional_int(item["chunk"].get("order_index")) or 9999,
                self._optional_int(item["chunk"].get("subchunk_index")) or 9999,
                str(item["candidate_chunk_id"]),
            ),
        )

    def _final_context_priority(self, item: Dict[str, Any], policy_name: str) -> int:
        # figure_table 场景里，结构化表格单元格证据要优先于旁边的正文解释段落，避免答案被“更像上下文”的文本挤下去。
        if policy_name == "figure_table":
            chunk = item["chunk"]
            matched_routes = set(item["candidate"].get("matched_routes") or [])
            if str(chunk.get("chunk_type", "") or "").lower() == "table" and "table_structured" in matched_routes:
                return 0
            if str(chunk.get("chunk_type", "") or "").lower() == "figure":
                return 1
            if item["context_role"] in {"anchor_evidence", "table_evidence", "figure_evidence"}:
                return 2
            return 3
        return 0

    def _annotate_final_chunk(self, chunk: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
        annotated = dict(chunk)
        candidate = item["candidate"]
        annotated["final_context_uses_original_chunk"] = True
        annotated["context_role"] = item["context_role"]
        annotated["context_budget_score"] = item["budget_score"]
        annotated["context_budget_reason"] = self._include_reason(item)
        annotated["expansion_source_anchor_ids"] = list(candidate.get("expansion_source_anchor_ids") or [])
        annotated["relationship_types"] = list(candidate.get("relationship_types") or [])
        annotated["expansion_reasons"] = list(candidate.get("expansion_reasons") or [])
        annotated["expansion_score"] = candidate.get("expansion_score")
        annotated["final_context_reason"] = candidate.get("final_context_reason") or annotated["context_budget_reason"]
        return annotated

    def _build_chunk_lookup(
        self,
        reranked_chunks: List[Dict[str, Any]],
        retrieval_index: Optional[CollectionRetrievalIndex],
    ) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        for chunk in reranked_chunks:
            key = str(chunk.get("chunk_id", "") or "").strip()
            if key:
                lookup[key] = dict(chunk)
        if retrieval_index is not None:
            for document in retrieval_index.documents:
                chunk = document.chunk
                key = str(chunk.get("chunk_id", "") or "").strip()
                if key and key not in lookup:
                    lookup[key] = dict(chunk)
        return lookup

    @staticmethod
    def _resolve_effective_char_budget(
        *,
        max_context_chars: int,
        candidate_max_tokens_soft: Optional[int],
        token_chars_per_token: float,
    ) -> int:
        legacy_chars = max(1, int(max_context_chars or 1))
        if candidate_max_tokens_soft in (None, ""):
            return legacy_chars
        try:
            token_budget = max(1, int(candidate_max_tokens_soft))
            chars_per_token = max(1.0, float(token_chars_per_token or 3.0))
        except (TypeError, ValueError):
            return legacy_chars
        # retrieval 阶段只是候选预筛，soft token 预算换算成字符上限供旧选择器复用，最终精确预算在 prompt planner 完成。
        return max(legacy_chars, int(token_budget * chars_per_token))

    def _budget_debug(
        self,
        *,
        original_top_chunks: List[Dict[str, Any]],
        context_expansion: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        final_chunks: List[Dict[str, Any]],
        max_context_chars: int,
        policy_name: str,
        candidate_max_blocks: Optional[int] = None,
        candidate_max_tokens_soft: Optional[int] = None,
    ) -> Dict[str, Any]:
        role_counts: Dict[str, int] = {}
        chunk_type_counts: Dict[str, int] = {}
        for chunk in final_chunks:
            role = str(chunk.get("context_role", "fallback_context") or "fallback_context")
            chunk_type = str(chunk.get("chunk_type", "text") or "text")
            role_counts[role] = role_counts.get(role, 0) + 1
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
        included_ids = {str(chunk.get("chunk_id", "") or "") for chunk in final_chunks}
        anchor_extension_map: Dict[str, List[str]] = {}
        for chunk in final_chunks:
            for anchor_id in chunk.get("expansion_source_anchor_ids", []) or []:
                anchor_extension_map.setdefault(str(anchor_id), []).append(str(chunk.get("chunk_id", "")))
        return {
            "enabled": True,
            "applied": True,
            "mode": "candidate_preselector",
            "policy_name": policy_name,
            "candidate_max_blocks": candidate_max_blocks,
            "candidate_max_tokens_soft": candidate_max_tokens_soft,
            "max_context_chars": max_context_chars,
            "used_context_chars": sum(self._estimate_chars(chunk) for chunk in final_chunks),
            "original_top_chunks": [self.trace_builder.debug_chunk_item(chunk) for chunk in original_top_chunks],
            "anchor_chunks": context_expansion.get("anchors", []),
            "expansion_candidates": context_expansion.get("candidate_pool", []),
            "final_context_before_budget": [
                {
                    "chunk_id": item["candidate_chunk_id"],
                    "context_role": item["context_role"],
                    "budget_score": item["budget_score"],
                    "estimated_chars": item["estimated_chars"],
                    "relationship_types": item["relationship_types"],
                }
                for item in candidates[:50]
            ],
            "final_context_after_budget": [self.trace_builder.debug_chunk_item(chunk) for chunk in final_chunks],
            "decisions": decisions[:100],
            "role_counts": role_counts,
            "chunk_type_counts": chunk_type_counts,
            "anchor_extension_map": anchor_extension_map,
            "included_chunk_ids": sorted(included_ids),
        }

    def _fallback_debug(
        self,
        *,
        reason: str,
        enabled: bool,
        fallback_chunks: List[Dict[str, Any]],
        max_context_chars: int,
        candidate_max_blocks: Optional[int] = None,
        candidate_max_tokens_soft: Optional[int] = None,
    ) -> Dict[str, Any]:
        decisions = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "include": True,
                "reason": "fallback_to_rerank_candidates",
                "context_role": chunk.get("context_role", "fallback_context"),
            }
            for chunk in fallback_chunks
        ]
        return {
            # enabled 表示运行期开关状态；applied 表示是否真正走了扩展预算，二者分开便于定位降级原因。
            "enabled": enabled,
            "applied": False,
            "mode": "candidate_preselector",
            "fallback_reason": reason,
            "candidate_max_blocks": candidate_max_blocks,
            "candidate_max_tokens_soft": candidate_max_tokens_soft,
            "max_context_chars": max_context_chars,
            "used_context_chars": sum(self._estimate_chars(chunk) for chunk in fallback_chunks),
            "original_top_chunks": [self.trace_builder.debug_chunk_item(chunk) for chunk in fallback_chunks],
            "anchor_chunks": [],
            "expansion_candidates": [],
            "final_context_before_budget": [self.trace_builder.debug_chunk_item(chunk) for chunk in fallback_chunks],
            "final_context_after_budget": [self.trace_builder.debug_chunk_item(chunk) for chunk in fallback_chunks],
            "decisions": decisions,
            "role_counts": {"fallback_context": len(fallback_chunks)},
            "chunk_type_counts": self.trace_builder.count_chunk_types(fallback_chunks),
            "anchor_extension_map": {},
            "included_chunk_ids": sorted(str(chunk.get("chunk_id", "") or "") for chunk in fallback_chunks),
        }

    def _drop_unselected(
        self,
        candidates: List[Dict[str, Any]],
        selected_ids: Set[str],
        decisions: List[Dict[str, Any]],
        *,
        reason: str,
    ) -> List[Dict[str, Any]]:
        for item in candidates:
            if item["candidate_chunk_id"] not in selected_ids:
                decisions.append(self._decision(item, include=False, reason=reason, used_chars=0, max_context_chars=0))
        return decisions

    def _decision(self, item: Dict[str, Any], *, include: bool, reason: str, used_chars: int, max_context_chars: int) -> Dict[str, Any]:
        return {
            "chunk_id": item["candidate_chunk_id"],
            "include": include,
            "reason": reason,
            "context_role": item["context_role"],
            "budget_score": item["budget_score"],
            "estimated_chars": item["estimated_chars"],
            "used_chars": used_chars,
            "max_context_chars": max_context_chars,
            "relationship_types": item["relationship_types"],
            "anchor_ids": item["anchor_ids"],
        }

    @staticmethod
    def _context_role(candidate: Dict[str, Any], chunk: Dict[str, Any], relationship_types: List[str]) -> str:
        chunk_type = str(chunk.get("chunk_type", candidate.get("chunk_type", "text")) or "text").lower()
        matched_routes = set(candidate.get("matched_routes", []) or [])
        if "memory_context" in matched_routes:
            return "memory_context"
        # 进入预算层后 candidate 只保留 route 痕迹，不再携带完整 evidence；因此用命中的 route 标记识别结构化表格主证据。
        if chunk_type == "table" and "table_structured" in matched_routes:
            return "anchor_evidence"
        if chunk_type == "figure":
            return "figure_evidence"
        if chunk_type == "table":
            return "table_evidence"
        if "self" in relationship_types and candidate.get("is_original_retrieval_hit"):
            return "anchor_evidence"
        if "sibling" in relationship_types:
            return "sibling_context"
        if "section_header" in relationship_types or "section_neighbors" in relationship_types:
            return "section_context"
        if "parent_section" in relationship_types:
            return "parent_context"
        return "fallback_context"

    @staticmethod
    def _role_score(role: str, policy_name: str) -> float:
        base = {
            "anchor_evidence": 0.42,
            "memory_context": 0.36,
            "figure_evidence": 0.25,
            "table_evidence": 0.25,
            "sibling_context": 0.20,
            "section_context": 0.17,
            "parent_context": 0.12,
            "fallback_context": 0.05,
        }.get(role, 0.05)
        if policy_name == "figure_table" and role in {"figure_evidence", "table_evidence"}:
            base += 0.20
        if policy_name == "result_analysis" and role in {"figure_evidence", "table_evidence"}:
            base += 0.14
        if policy_name == "method_flow" and role == "sibling_context":
            base += 0.12
        if policy_name == "summary" and role == "sibling_context":
            base -= 0.12
        return base

    @staticmethod
    def _include_priority(role: str, policy_name: str, candidate: Dict[str, Any]) -> float:
        priority = 0.0
        if role == "anchor_evidence":
            priority += 5.0
        if role in {"figure_evidence", "table_evidence"} and policy_name in {"figure_table", "result_analysis"}:
            priority += 4.0
        if role == "sibling_context" and policy_name == "method_flow":
            priority += 3.0
        if len(candidate.get("expansion_source_anchor_ids") or []) > 1:
            priority += 2.0
        if candidate.get("is_original_retrieval_hit"):
            priority += 1.0
        return priority

    @staticmethod
    def _relationship_distance_penalty(candidate: Dict[str, Any]) -> float:
        relationships = candidate.get("relationship_types") or []
        if "self" in relationships:
            return 0.0
        return 0.02 * max(len(relationships), 1)

    @staticmethod
    def _is_protected_role(role: str, policy_name: str) -> bool:
        if role == "anchor_evidence":
            return True
        if policy_name == "figure_table" and role in {"figure_evidence", "table_evidence"}:
            return True
        if policy_name == "method_flow" and role == "sibling_context":
            return True
        return False

    @staticmethod
    def _include_reason(item: Dict[str, Any]) -> str:
        role = item["context_role"]
        if role == "anchor_evidence":
            return "include_high_score_anchor"
        if role in {"figure_evidence", "table_evidence"}:
            return "include_question_relevant_asset"
        if role == "sibling_context":
            return "include_nearby_parent_sibling"
        if role == "section_context":
            return "include_section_context"
        if role == "parent_context":
            return "include_parent_section_context"
        if role == "memory_context":
            return "include_memory_context"
        return "include_budget_candidate"

    @staticmethod
    def _role_order(role: str) -> int:
        return {
            "section_context": 0,
            "parent_context": 1,
            "anchor_evidence": 2,
            "sibling_context": 3,
            "figure_evidence": 4,
            "table_evidence": 4,
            "memory_context": 5,
            "fallback_context": 6,
        }.get(role, 9)

    @staticmethod
    def _estimate_chars(chunk: Dict[str, Any]) -> int:
        return len(
            "\n".join(
                [
                    str(chunk.get("content", "") or ""),
                    str(chunk.get("asset_summary", "") or ""),
                    str(chunk.get("asset_preview_text", "") or ""),
                ]
            )
        )

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None


class ContextExpansionPreparer:
    """把 rerank 后的 top_n 命中转换成可扩展 anchor/candidate 关系。"""

    def __init__(self, *, trace_builder: RetrievalTraceBuilder) -> None:
        self.trace_builder = trace_builder

    def prepare(
        self,
        *,
        reranked_chunks: List[Dict[str, Any]],
        final_context_chunks: List[Dict[str, Any]],
        retrieval_index: Optional[CollectionRetrievalIndex],
        query_profile: Any,
        anchor_limit: int,
    ) -> Dict[str, Any]:
        final_keys = {self._chunk_key(chunk) for chunk in final_context_chunks}
        original_hit_map = {self._candidate_key(chunk): chunk for chunk in reranked_chunks}
        policy = self._select_policy(query_profile)
        anchors = [
            self._build_anchor(
                chunk,
                original_rank=rank,
                final_keys=final_keys,
                query_profile=query_profile,
            )
            for rank, chunk in enumerate(reranked_chunks[: max(1, anchor_limit)], start=1)
        ]

        if retrieval_index is None or not retrieval_index.documents:
            # 索引不可用时只保留 anchor 自身，保证缺失 parent/section 等字段不会中断 QA。
            relations = [
                self._relation_payload(anchor, [self._self_candidate(anchor, final_keys=final_keys, policy=policy)], policy=policy)
                for anchor in anchors
            ]
            candidate_pool = self._merge_candidate_pool(relations, original_hit_map=original_hit_map)
            return self._debug_payload(
                anchors,
                relations,
                candidate_pool=candidate_pool,
                policy=policy,
                fallback_reason="retrieval_index_unavailable",
            )

        relations = []
        for anchor in anchors:
            candidate_map: Dict[str, ExpansionCandidate] = {}
            for action in policy.actions:
                # 每个 action 只处理一种结构关系，策略选择由 policy 决定，后续新增动作不需要改主流程。
                if action == "self":
                    self._add_candidate(
                        candidate_map,
                        anchor=anchor,
                        chunk=self._lookup_anchor_chunk(anchor, retrieval_index) or {},
                        expansion_type="self",
                        distance=0,
                        final_keys=final_keys,
                        original_hit_map=original_hit_map,
                        policy=policy,
                        reason="原始 rerank 命中 chunk 必须作为候选基线保留。",
                    )
                elif action == "sibling":
                    self._add_parent_siblings(candidate_map, anchor=anchor, retrieval_index=retrieval_index, final_keys=final_keys, original_hit_map=original_hit_map, policy=policy)
                elif action == "section_neighbors":
                    self._add_section_neighbors(candidate_map, anchor=anchor, retrieval_index=retrieval_index, final_keys=final_keys, original_hit_map=original_hit_map, policy=policy)
                elif action == "section_header":
                    self._add_section_header(candidate_map, anchor=anchor, retrieval_index=retrieval_index, final_keys=final_keys, original_hit_map=original_hit_map, policy=policy)
                elif action == "parent_section":
                    self._add_parent_section(candidate_map, anchor=anchor, retrieval_index=retrieval_index, final_keys=final_keys, original_hit_map=original_hit_map, policy=policy)
                elif action == "asset_related":
                    self._add_asset_related(candidate_map, anchor=anchor, retrieval_index=retrieval_index, final_keys=final_keys, original_hit_map=original_hit_map, policy=policy)
                elif action == "page_neighbors":
                    self._add_page_neighbors(candidate_map, anchor=anchor, retrieval_index=retrieval_index, final_keys=final_keys, original_hit_map=original_hit_map, policy=policy)
                elif action == "cited_asset_context":
                    self._add_cited_asset_context(candidate_map, anchor=anchor, retrieval_index=retrieval_index, final_keys=final_keys, original_hit_map=original_hit_map, policy=policy, query_profile=query_profile)
            relations.append(self._relation_payload(anchor, list(candidate_map.values()), policy=policy))

        candidate_pool = self._merge_candidate_pool(relations, original_hit_map=original_hit_map)
        return self._debug_payload(anchors, relations, candidate_pool=candidate_pool, policy=policy, fallback_reason="")

    def _build_anchor(
        self,
        chunk: Dict[str, Any],
        *,
        original_rank: int,
        final_keys: Set[str],
        query_profile: Any,
    ) -> ExpansionAnchor:
        chunk_type = self._chunk_type(chunk)
        matched_routes = [str(item) for item in (chunk.get("matched_routes") or []) if str(item).strip()]
        reasons = [f"rerank_top_n_rank_{original_rank}"]
        if len(matched_routes) > 1:
            reasons.append("matched_by_multiple_routes")
        subchunk_index = self._optional_int(chunk.get("subchunk_index"))
        subchunk_count = self._optional_int(chunk.get("subchunk_count"))
        if subchunk_index and subchunk_count and subchunk_count > 1:
            reasons.append("subchunk_needs_parent_siblings")
        if chunk_type in {"figure", "table"}:
            reasons.append("asset_chunk_needs_visual_context")
        if self._query_needs_structural_context(query_profile):
            reasons.append("question_type_needs_section_context")

        return ExpansionAnchor(
            chunk_id=chunk.get("chunk_id"),
            parent_chunk_id=chunk.get("parent_chunk_id"),
            subchunk_index=subchunk_index,
            subchunk_count=subchunk_count,
            section_title=str(chunk.get("section_title", "") or ""),
            section_path=str(chunk.get("section_path", "") or ""),
            page_number=chunk.get("page_number"),
            page_range=str(chunk.get("page_range", "") or ""),
            chunk_type=chunk_type,
            source_route=str(chunk.get("retrieval_route") or (matched_routes[0] if matched_routes else "")),
            fusion_score=self._optional_float(chunk.get("fusion_score", chunk.get("score"))),
            rerank_score=self._optional_float(chunk.get("llm_rerank_score")),
            is_final_context_chunk=self._chunk_key(chunk) in final_keys,
            original_rank=original_rank,
            content_preview=str(chunk.get("content", "") or chunk.get("text", "") or "")[:400],
            matched_routes=matched_routes,
            reasons=reasons,
        )

    def _add_parent_siblings(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
    ) -> None:
        parent_id = str(anchor.parent_chunk_id or "").strip()
        if not parent_id:
            return
        doc_ids = retrieval_index.by_parent_chunk_id.get(parent_id, [])
        added = 0
        for doc_id in self._sort_doc_ids(doc_ids, retrieval_index):
            chunk = retrieval_index.documents[doc_id].chunk
            distance = self._subchunk_distance(anchor, chunk, fallback_doc_id=doc_id)
            if abs(distance) > policy.sibling_window:
                continue
            self._add_candidate(
                candidate_map,
                anchor=anchor,
                chunk=chunk,
                expansion_type="sibling" if distance else "self",
                distance=distance,
                final_keys=final_keys,
                original_hit_map=original_hit_map,
                policy=policy,
                reason="同 parent 的相邻 subchunk 可补足定义、步骤前后文或被切断的句段。",
            )
            added += 1
            if added >= policy.sibling_window * 2 + 1:
                break

    def _add_section_neighbors(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
    ) -> None:
        section_path = str(anchor.section_path or "").strip()
        if not section_path:
            return
        doc_ids = self._sort_doc_ids(retrieval_index.by_section_path.get(section_path, []), retrieval_index)
        anchor_doc_ids = self._doc_ids_for_anchor(anchor, retrieval_index)
        anchor_doc_id = anchor_doc_ids[0] if anchor_doc_ids else None
        added = 0
        for doc_id in doc_ids:
            distance = 0 if anchor_doc_id is None else doc_id - anchor_doc_id
            if abs(distance) > policy.section_window:
                continue
            chunk = retrieval_index.documents[doc_id].chunk
            self._add_candidate(
                candidate_map,
                anchor=anchor,
                chunk=chunk,
                expansion_type="section_neighbors" if distance else "self",
                distance=distance,
                final_keys=final_keys,
                original_hit_map=original_hit_map,
                policy=policy,
                reason="同 section_path 的近邻 chunk 可补足方法流程、实验设置或结果解释的局部上下文。",
            )
            added += 1
            if added >= policy.section_limit:
                break

    def _add_section_header(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
    ) -> None:
        section_path = str(anchor.section_path or "").strip()
        if not section_path:
            return
        doc_ids = self._sort_doc_ids(retrieval_index.by_section_path.get(section_path, []), retrieval_index)
        for doc_id in doc_ids[:2]:
            chunk = retrieval_index.documents[doc_id].chunk
            self._add_candidate(
                candidate_map,
                anchor=anchor,
                chunk=chunk,
                expansion_type="section_header",
                distance=abs(self._doc_distance(anchor, chunk, retrieval_index, fallback_doc_id=doc_id)),
                final_keys=final_keys,
                original_hit_map=original_hit_map,
                policy=policy,
                reason="section opening/header 往往承载章节主题，后续预算层可优先用它补标题语义。",
            )

    def _add_parent_section(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
    ) -> None:
        parent_path = self._parent_section_path(anchor.section_path)
        if not parent_path:
            return
        doc_ids = self._sort_doc_ids(retrieval_index.by_section_path.get(parent_path, []), retrieval_index)
        for doc_id in doc_ids[: policy.section_limit]:
            chunk = retrieval_index.documents[doc_id].chunk
            self._add_candidate(
                candidate_map,
                anchor=anchor,
                chunk=chunk,
                expansion_type="parent_section",
                distance=abs(self._doc_distance(anchor, chunk, retrieval_index, fallback_doc_id=doc_id)),
                final_keys=final_keys,
                original_hit_map=original_hit_map,
                policy=policy,
                reason="父级章节可提供当前子章节的任务边界和上位概念，适合方法/实验类问题补结构背景。",
            )

    def _add_asset_related(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
    ) -> None:
        added = 0
        section_path = str(anchor.section_path or "").strip()
        section_doc_ids = retrieval_index.by_section_path.get(section_path, []) if section_path else []
        page_doc_ids: List[int] = []
        for page in self._nearby_pages(anchor.page_number, window=policy.page_window):
            page_doc_ids.extend(retrieval_index.by_page_number.get(page, []))
        for doc_id in self._sort_doc_ids([*section_doc_ids, *page_doc_ids], retrieval_index):
            chunk = retrieval_index.documents[doc_id].chunk
            if self._chunk_type(chunk) not in {"figure", "table"}:
                continue
            self._add_candidate(
                candidate_map,
                anchor=anchor,
                chunk=chunk,
                expansion_type="asset_related",
                distance=min(
                    abs(self._page_distance(anchor.page_number, chunk.get("page_number"))),
                    abs(self._doc_distance(anchor, chunk, retrieval_index, fallback_doc_id=doc_id)),
                ),
                final_keys=final_keys,
                original_hit_map=original_hit_map,
                policy=policy,
                reason="同章节、同页或相邻页的 figure/table 常补充正文没有展开的视觉证据。",
            )
            added += 1
            if added >= policy.asset_limit:
                break

    def _add_page_neighbors(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
    ) -> None:
        added = 0
        for page in self._nearby_pages(anchor.page_number, window=policy.page_window):
            for doc_id in self._sort_doc_ids(retrieval_index.by_page_number.get(page, []), retrieval_index):
                chunk = retrieval_index.documents[doc_id].chunk
                self._add_candidate(
                    candidate_map,
                    anchor=anchor,
                    chunk=chunk,
                    expansion_type="page_neighbors",
                    distance=self._page_distance(anchor.page_number, chunk.get("page_number")),
                    final_keys=final_keys,
                    original_hit_map=original_hit_map,
                    policy=policy,
                    reason="同页或相邻页通常包含实验设置、图表引用或结果解释的连续段落。",
                )
                added += 1
                if added >= policy.page_limit:
                    return

    def _add_cited_asset_context(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
        query_profile: Any,
    ) -> None:
        cited_labels = self._extract_cited_asset_labels(anchor, query_profile)
        if not cited_labels:
            return
        added = 0
        for doc_id, document in enumerate(retrieval_index.documents):
            chunk = document.chunk
            if self._chunk_type(chunk) not in {"figure", "table"}:
                continue
            if not self._asset_matches_labels(chunk, cited_labels):
                continue
            self._add_candidate(
                candidate_map,
                anchor=anchor,
                chunk=chunk,
                expansion_type="cited_asset_context",
                distance=self._page_distance(anchor.page_number, chunk.get("page_number")),
                final_keys=final_keys,
                original_hit_map=original_hit_map,
                policy=policy,
                reason="正文或问题显式提到 Figure/Table 编号时，优先补对应资产 chunk。",
            )
            added += 1
            if added >= policy.asset_limit:
                break

    def _add_candidate(
        self,
        candidate_map: Dict[str, ExpansionCandidate],
        *,
        anchor: ExpansionAnchor,
        chunk: Dict[str, Any],
        expansion_type: str,
        distance: int,
        final_keys: Set[str],
        original_hit_map: Dict[str, Dict[str, Any]],
        policy: ExpansionPolicy,
        reason: str,
    ) -> None:
        if chunk and policy.summary_section_only and expansion_type not in {"self", "section_header"} and not self._matches_preferred_section(chunk, policy):
            # 摘要/局限类问题只需要跨章节信号，避免被某个局部段落的邻近 chunk 过度占满候选池。
            return
        if not chunk:
            candidate = self._self_candidate(anchor, final_keys=final_keys, policy=policy)
        else:
            candidate_key = self._candidate_key(chunk)
            original_hit = original_hit_map.get(candidate_key)
            is_original_hit = original_hit is not None
            expansion_score = self._score_candidate(
                anchor=anchor,
                chunk=chunk,
                expansion_type=expansion_type,
                distance=distance,
                final_keys=final_keys,
                is_original_retrieval_hit=is_original_hit,
                policy=policy,
            )
            candidate = ExpansionCandidate(
                anchor_chunk_id=anchor.chunk_id,
                candidate_chunk_id=chunk.get("chunk_id"),
                expansion_type=expansion_type,
                relationship_type=expansion_type,
                distance=int(distance or 0),
                is_anchor=self._string_key(chunk.get("chunk_id")) == self._string_key(anchor.chunk_id),
                is_final_context_chunk=self._chunk_key(chunk) in final_keys,
                is_original_retrieval_hit=is_original_hit,
                chunk_type=self._chunk_type(chunk),
                section_path=str(chunk.get("section_path", "") or ""),
                section_title=str(chunk.get("section_title", "") or ""),
                page_number=chunk.get("page_number"),
                page_range=str(chunk.get("page_range", "") or ""),
                parent_chunk_id=chunk.get("parent_chunk_id"),
                subchunk_index=self._optional_int(chunk.get("subchunk_index")),
                source_route=str(chunk.get("retrieval_route") or (original_hit or {}).get("retrieval_route") or ""),
                fusion_score=self._optional_float(chunk.get("fusion_score", chunk.get("score"))),
                rerank_score=self._optional_float(chunk.get("llm_rerank_score")),
                expansion_score=expansion_score,
                reason=reason,
                expansion_reason=reason,
                matched_routes=[str(item) for item in ((original_hit or chunk).get("matched_routes") or []) if str(item).strip()],
            )
        key = f"{candidate.relationship_type}|{self._string_key(candidate.candidate_chunk_id)}"
        current = candidate_map.get(key)
        if current is None or candidate.expansion_score > current.expansion_score:
            candidate_map[key] = candidate

    def _self_candidate(self, anchor: ExpansionAnchor, *, final_keys: Set[str], policy: ExpansionPolicy) -> ExpansionCandidate:
        score = self._score_candidate(
            anchor=anchor,
            chunk={},
            expansion_type="self",
            distance=0,
            final_keys=final_keys,
            is_original_retrieval_hit=True,
            policy=policy,
        )
        return ExpansionCandidate(
            anchor_chunk_id=anchor.chunk_id,
            candidate_chunk_id=anchor.chunk_id,
            expansion_type="self",
            relationship_type="self",
            distance=0,
            is_anchor=True,
            is_final_context_chunk=anchor.is_final_context_chunk or self._string_key(anchor.chunk_id) in final_keys,
            is_original_retrieval_hit=True,
            chunk_type=anchor.chunk_type,
            section_path=anchor.section_path,
            section_title=anchor.section_title,
            page_number=anchor.page_number,
            page_range=anchor.page_range,
            parent_chunk_id=anchor.parent_chunk_id,
            subchunk_index=anchor.subchunk_index,
            source_route=anchor.source_route,
            fusion_score=anchor.fusion_score,
            rerank_score=anchor.rerank_score,
            expansion_score=score,
            reason="原始 rerank 命中 chunk 必须作为候选基线保留。",
            expansion_reason="原始 rerank 命中 chunk 必须作为候选基线保留。",
            matched_routes=list(anchor.matched_routes),
        )

    def _relation_payload(self, anchor: ExpansionAnchor, candidates: List[ExpansionCandidate], *, policy: ExpansionPolicy) -> Dict[str, Any]:
        ordered = sorted(candidates, key=lambda item: (item.expansion_score, -abs(item.distance), item.expansion_type), reverse=True)
        ordered = ordered[: policy.per_anchor_limit]
        return {
            "anchor": asdict(anchor),
            "candidates": [asdict(candidate) for candidate in ordered],
        }

    def _debug_payload(
        self,
        anchors: List[ExpansionAnchor],
        relations: List[Dict[str, Any]],
        *,
        candidate_pool: List[Dict[str, Any]],
        policy: ExpansionPolicy,
        fallback_reason: str,
    ) -> Dict[str, Any]:
        candidate_count = sum(len(item.get("candidates", [])) for item in relations)
        return {
            "enabled": True,
            "mode": "prepare_only",
            "policy": asdict(policy),
            "anchor_count": len(anchors),
            "candidate_count": candidate_count,
            "deduped_candidate_count": len(candidate_pool),
            "fallback_reason": fallback_reason,
            "anchors": [asdict(anchor) for anchor in anchors],
            "relations": relations,
            "candidate_pool": candidate_pool,
        }

    def _select_policy(self, query_profile: Any) -> ExpansionPolicy:
        question_type = str(getattr(query_profile, "question_type", "") or "").strip().lower()
        intent_profile = getattr(query_profile, "intent_profile", None)
        main_intent = str(getattr(intent_profile, "main_intent", "") or "").strip().lower()
        query_text = " ".join(
            [
                str(getattr(query_profile, "original_query", "") or ""),
                str(getattr(query_profile, "normalized_query", "") or ""),
                " ".join(str(tag) for tag in (getattr(query_profile, "intent_tags", []) or [])),
            ]
        )
        aliases = {
            "method": "method_flow",
            "implementation_detail": "method_flow",
            "experiment": "experiment_setup",
            "dataset": "experiment_setup",
            "metric": "experiment_setup",
            "comparison": "result_analysis",
            "results_analysis": "result_analysis",
            "contribution": "summary",
            "paper_overview": "summary",
            "figure": "figure_table",
            "table": "figure_table",
        }
        policy_name = aliases.get(question_type, question_type) or aliases.get(main_intent, main_intent) or "default"
        if self._looks_like_figure_table_question(query_text):
            # 显式 Figure/Table 编号或图表词是强结构信号，优先使用图表策略，避免被 method/result 等主题词吞掉。
            policy_name = "figure_table"
        if policy_name not in self._policy_templates():
            policy_name = aliases.get(main_intent, main_intent) if aliases.get(main_intent, main_intent) in self._policy_templates() else "default"

        base_policy = self._policy_templates()[policy_name]
        preferred_terms = [
            *base_policy.preferred_section_terms,
            *[str(item).lower() for item in (getattr(query_profile, "section_preferences", []) or [])],
            *[str(item).lower() for item in (getattr(intent_profile, "preferred_sections", []) or [])],
        ]
        return ExpansionPolicy(
            **{
                **asdict(base_policy),
                "preferred_section_terms": tuple(self._dedupe_strings(preferred_terms)),
            }
        )

    @staticmethod
    def _policy_templates() -> Dict[str, ExpansionPolicy]:
        return {
            "method_flow": ExpansionPolicy(
                name="method_flow",
                actions=("self", "section_header", "sibling", "section_neighbors", "parent_section"),
                sibling_window=2,
                section_window=2,
                section_limit=5,
                per_anchor_limit=8,
                preferred_section_terms=("method", "approach", "model", "architecture", "algorithm", "implementation"),
            ),
            "experiment_setup": ExpansionPolicy(
                name="experiment_setup",
                actions=("self", "section_neighbors", "page_neighbors", "asset_related", "sibling"),
                sibling_window=1,
                section_window=2,
                section_limit=5,
                page_limit=5,
                asset_limit=2,
                per_anchor_limit=8,
                preferred_section_terms=("experiment", "evaluation", "dataset", "baseline", "metric", "implementation", "setup"),
            ),
            "result_analysis": ExpansionPolicy(
                name="result_analysis",
                actions=("self", "asset_related", "cited_asset_context", "section_neighbors", "sibling", "page_neighbors"),
                sibling_window=1,
                section_window=2,
                section_limit=5,
                page_limit=4,
                asset_limit=4,
                per_anchor_limit=9,
                preferred_section_terms=("result", "analysis", "ablation", "comparison", "performance", "experiment", "table", "figure"),
            ),
            "figure_table": ExpansionPolicy(
                name="figure_table",
                actions=("self", "asset_related", "cited_asset_context", "page_neighbors", "section_neighbors"),
                sibling_window=1,
                section_window=1,
                section_limit=3,
                page_limit=5,
                asset_limit=5,
                per_anchor_limit=9,
                preferred_section_terms=("figure", "table", "result", "experiment", "appendix", "caption"),
            ),
            "summary": ExpansionPolicy(
                name="summary",
                actions=("self", "section_header", "section_neighbors"),
                sibling_window=0,
                section_window=1,
                section_limit=3,
                per_anchor_limit=5,
                summary_section_only=True,
                preferred_section_terms=("abstract", "introduction", "conclusion", "summary", "contribution"),
            ),
            "limitation": ExpansionPolicy(
                name="limitation",
                actions=("self", "section_header", "section_neighbors", "page_neighbors"),
                sibling_window=0,
                section_window=2,
                section_limit=4,
                page_limit=3,
                per_anchor_limit=6,
                summary_section_only=True,
                preferred_section_terms=("limitation", "discussion", "conclusion", "future", "appendix"),
            ),
            "default": ExpansionPolicy(
                name="default",
                actions=("self", "sibling", "section_neighbors", "asset_related"),
                sibling_window=1,
                section_window=1,
                section_limit=3,
                asset_limit=2,
                per_anchor_limit=6,
                preferred_section_terms=(),
            ),
        }

    def _score_candidate(
        self,
        *,
        anchor: ExpansionAnchor,
        chunk: Dict[str, Any],
        expansion_type: str,
        distance: int,
        final_keys: Set[str],
        is_original_retrieval_hit: bool,
        policy: ExpansionPolicy,
    ) -> float:
        anchor_score = anchor.rerank_score if anchor.rerank_score is not None else anchor.fusion_score
        base = max(0.0, min(float(anchor_score if anchor_score is not None else 0.5), 1.0))
        relationship_weight = {
            "self": 0.35,
            "cited_asset_context": 0.30,
            "asset_related": 0.26,
            "section_header": 0.22,
            "sibling": 0.21,
            "section_neighbors": 0.18,
            "parent_section": 0.15,
            "page_neighbors": 0.13,
        }.get(expansion_type, 0.10)
        distance_penalty = min(abs(int(distance or 0)) * 0.05, 0.25)
        score = base * 0.55 + relationship_weight - distance_penalty

        if chunk:
            if self._string_key(chunk.get("parent_chunk_id")) and self._string_key(chunk.get("parent_chunk_id")) == self._string_key(anchor.parent_chunk_id):
                score += 0.08
            if self._section_key(chunk) and self._section_key(chunk) == self._string_key(anchor.section_path).lower():
                score += 0.07
            if self._matches_preferred_section(chunk, policy):
                score += 0.07
            if self._chunk_type(chunk) in {"figure", "table"} and policy.name in {"figure_table", "result_analysis"}:
                score += 0.12
            if self._chunk_key(chunk) in final_keys:
                score += 0.05

        if is_original_retrieval_hit:
            score += 0.10
        if expansion_type == "self":
            score += 0.12
        return round(max(0.0, score), 6)

    def _merge_candidate_pool(
        self,
        relations: List[Dict[str, Any]],
        *,
        original_hit_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for relation in relations:
            anchor = relation.get("anchor", {})
            anchor_id = anchor.get("chunk_id")
            for candidate in relation.get("candidates", []):
                key = self._string_key(candidate.get("candidate_chunk_id"))
                if not key:
                    continue
                original_hit = original_hit_map.get(key, {})
                entry = merged.setdefault(
                    key,
                    {
                        "candidate_chunk_id": candidate.get("candidate_chunk_id"),
                        "chunk_type": candidate.get("chunk_type"),
                        "section_path": candidate.get("section_path"),
                        "section_title": candidate.get("section_title"),
                        "page_number": candidate.get("page_number"),
                        "page_range": candidate.get("page_range"),
                        "parent_chunk_id": candidate.get("parent_chunk_id"),
                        "expansion_source_anchor_ids": [],
                        "relationship_types": [],
                        "expansion_reasons": [],
                        "original_retrieval_routes": [],
                        "matched_routes": [],
                        "fusion_score": self._optional_float(original_hit.get("fusion_score", original_hit.get("score"))),
                        "rerank_score": self._optional_float(original_hit.get("llm_rerank_score")),
                        "expansion_score": 0.0,
                        "max_expansion_score": 0.0,
                        "is_original_retrieval_hit": bool(original_hit),
                        "is_final_context_chunk": bool(candidate.get("is_final_context_chunk")),
                        "final_context_reason": "original_final_context" if candidate.get("is_final_context_chunk") else "",
                    },
                )
                self._append_unique(entry["expansion_source_anchor_ids"], anchor_id)
                self._append_unique(entry["relationship_types"], candidate.get("relationship_type") or candidate.get("expansion_type"))
                self._append_unique(entry["expansion_reasons"], candidate.get("expansion_reason") or candidate.get("reason"))
                self._append_unique(entry["original_retrieval_routes"], candidate.get("source_route"))
                for route in candidate.get("matched_routes", []) or []:
                    self._append_unique(entry["matched_routes"], route)
                entry["is_final_context_chunk"] = bool(entry["is_final_context_chunk"] or candidate.get("is_final_context_chunk"))
                if candidate.get("is_final_context_chunk"):
                    entry["final_context_reason"] = "original_final_context"
                score = float(candidate.get("expansion_score", 0.0) or 0.0)
                entry["expansion_score"] += score
                entry["max_expansion_score"] = max(float(entry["max_expansion_score"]), score)
                entry["fusion_score"] = self._max_optional(entry.get("fusion_score"), candidate.get("fusion_score"))
                entry["rerank_score"] = self._max_optional(entry.get("rerank_score"), candidate.get("rerank_score"))

        for entry in merged.values():
            # 多个 anchor 指向同一 chunk 时提升合并分，保留“为什么被多次扩展命中”的证据。
            anchor_bonus = min(0.08 * max(len(entry["expansion_source_anchor_ids"]) - 1, 0), 0.24)
            entry["expansion_score"] = round(float(entry["max_expansion_score"]) + anchor_bonus, 6)
        return sorted(
            merged.values(),
            key=lambda item: (float(item.get("expansion_score", 0.0)), len(item.get("relationship_types", []))),
            reverse=True,
        )

    def _lookup_anchor_chunk(
        self,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
    ) -> Optional[Dict[str, Any]]:
        doc_ids = self._doc_ids_for_anchor(anchor, retrieval_index)
        if doc_ids:
            return retrieval_index.documents[doc_ids[0]].chunk
        return None

    def _doc_ids_for_anchor(
        self,
        anchor: ExpansionAnchor,
        retrieval_index: CollectionRetrievalIndex,
    ) -> List[int]:
        for key, mapping in (
            (anchor.chunk_id, retrieval_index.by_chunk_id),
            (anchor.parent_chunk_id, retrieval_index.by_parent_chunk_id),
        ):
            normalized = str(key or "").strip()
            if normalized and mapping.get(normalized):
                return list(mapping[normalized])
        return []

    def _sort_doc_ids(self, doc_ids: Iterable[int], retrieval_index: CollectionRetrievalIndex) -> List[int]:
        return sorted(
            {int(doc_id) for doc_id in doc_ids if 0 <= int(doc_id) < len(retrieval_index.documents)},
            key=lambda doc_id: (
                self._optional_int(retrieval_index.documents[doc_id].chunk.get("order_index")) or doc_id,
                self._optional_int(retrieval_index.documents[doc_id].chunk.get("subchunk_index")) or 0,
                doc_id,
            ),
        )

    def _subchunk_distance(self, anchor: ExpansionAnchor, chunk: Dict[str, Any], *, fallback_doc_id: int) -> int:
        candidate_index = self._optional_int(chunk.get("subchunk_index"))
        if anchor.subchunk_index is not None and candidate_index is not None:
            return candidate_index - anchor.subchunk_index
        return fallback_doc_id

    def _doc_distance(
        self,
        anchor: ExpansionAnchor,
        chunk: Dict[str, Any],
        retrieval_index: CollectionRetrievalIndex,
        *,
        fallback_doc_id: int,
    ) -> int:
        anchor_doc_ids = self._doc_ids_for_anchor(anchor, retrieval_index)
        return fallback_doc_id - anchor_doc_ids[0] if anchor_doc_ids else 0

    @staticmethod
    def _page_distance(anchor_page: Any, candidate_page: Any) -> int:
        anchor_int = ContextExpansionPreparer._optional_int(anchor_page)
        candidate_int = ContextExpansionPreparer._optional_int(candidate_page)
        if anchor_int is None or candidate_int is None:
            return 0
        return candidate_int - anchor_int

    @staticmethod
    def _nearby_pages(page_number: Any, *, window: int = 1) -> List[str]:
        page = ContextExpansionPreparer._optional_int(page_number)
        if page is None:
            normalized = str(page_number or "").strip()
            return [normalized] if normalized else []
        window = max(0, int(window or 0))
        return [str(item) for item in range(page - window, page + window + 1) if item > 0]

    @staticmethod
    def _query_needs_structural_context(query_profile: Any) -> bool:
        question_type = str(getattr(query_profile, "question_type", "") or "").lower()
        intent_profile = getattr(query_profile, "intent_profile", None)
        main_intent = str(getattr(intent_profile, "main_intent", "") or "").lower()
        tags = {str(tag).lower() for tag in (getattr(query_profile, "intent_tags", []) or [])}
        structural_types = {"method_flow", "experiment_setup", "result_analysis", "figure_table"}
        structural_intents = {"method", "experiment", "comparison", "figure_table"}
        return question_type in structural_types or main_intent in structural_intents or bool(tags & structural_types)

    @staticmethod
    def _chunk_type(chunk: Dict[str, Any]) -> str:
        return str(chunk.get("chunk_type", "text") or "text").strip().lower() or "text"

    @staticmethod
    def _chunk_key(chunk: Dict[str, Any]) -> str:
        return RetrievalTraceBuilder.chunk_unique_key(chunk)

    @staticmethod
    def _candidate_key(chunk: Dict[str, Any]) -> str:
        chunk_id = str(chunk.get("chunk_id", "") or "").strip()
        if chunk_id:
            return chunk_id
        return RetrievalTraceBuilder.chunk_unique_key(chunk)

    @staticmethod
    def _section_key(chunk: Dict[str, Any]) -> str:
        return str(chunk.get("section_path", "") or "").strip().lower()

    @staticmethod
    def _matches_preferred_section(chunk: Dict[str, Any], policy: ExpansionPolicy) -> bool:
        if not policy.preferred_section_terms:
            return False
        section_text = " ".join(
            [
                str(chunk.get("section_path", "") or ""),
                str(chunk.get("section_title", "") or ""),
                str(chunk.get("content", "") or "")[:240],
                str(chunk.get("asset_summary", "") or "")[:240],
            ]
        ).lower()
        return any(term and term in section_text for term in policy.preferred_section_terms)

    @staticmethod
    def _parent_section_path(section_path: str) -> str:
        normalized = str(section_path or "").strip()
        if not normalized:
            return ""
        for separator in ("/", ">", "\\"):
            if separator in normalized:
                return normalized.rsplit(separator, 1)[0].strip()
        return ""

    def _extract_cited_asset_labels(self, anchor: ExpansionAnchor, query_profile: Any) -> List[str]:
        text = " ".join(
            [
                str(getattr(query_profile, "original_query", "") or ""),
                str(getattr(query_profile, "normalized_query", "") or ""),
                anchor.section_title,
                anchor.section_path,
                anchor.content_preview,
            ]
        )
        labels: List[str] = []
        for kind, number in re.findall(r"\b(fig(?:ure)?|table|tab)\.?\s*([0-9]+[a-zA-Z]?)", text, flags=re.IGNORECASE):
            normalized_kind = "figure" if kind.lower().startswith("fig") else "table"
            self._append_unique(labels, f"{normalized_kind} {number.lower()}")
        return labels

    @staticmethod
    def _looks_like_figure_table_question(text: str) -> bool:
        return bool(
            re.search(r"\b(fig(?:ure)?|table|tab)\.?\s*[0-9]+[a-zA-Z]?\b", text, flags=re.IGNORECASE)
            or re.search(r"\b(figure|table|diagram|caption|图|表格|表)\b", text, flags=re.IGNORECASE)
        )

    def _asset_matches_labels(self, chunk: Dict[str, Any], cited_labels: List[str]) -> bool:
        if not cited_labels:
            return False
        text = " ".join(
            [
                str(chunk.get("content", "") or ""),
                str(chunk.get("asset_summary", "") or ""),
                str(chunk.get("asset_caption", "") or ""),
                str(chunk.get("asset_path", "") or ""),
                str(chunk.get("section_title", "") or ""),
                str(chunk.get("section_path", "") or ""),
            ]
        ).lower()
        normalized_text = re.sub(r"[_\-]+", " ", text)
        return any(label in normalized_text for label in cited_labels)

    @staticmethod
    def _append_unique(values: List[Any], value: Any) -> None:
        if value in (None, ""):
            return
        if value not in values:
            values.append(value)

    @staticmethod
    def _dedupe_strings(values: Iterable[str]) -> List[str]:
        deduped: List[str] = []
        for value in values:
            normalized = str(value or "").strip().lower()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    @classmethod
    def _max_optional(cls, left: Any, right: Any) -> Optional[float]:
        left_value = cls._optional_float(left)
        right_value = cls._optional_float(right)
        if left_value is None:
            return right_value
        if right_value is None:
            return left_value
        return max(left_value, right_value)

    @staticmethod
    def _string_key(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
