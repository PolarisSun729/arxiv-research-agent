from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.token_counter import TokenCounter


@dataclass
class RuleCompactionConfig:
    enabled: bool = True
    rag_max_block_tokens: int = 4000
    rag_sibling_context_target_tokens: int = 350
    rag_section_context_target_tokens: int = 250
    rag_fallback_preview_tokens: int = 160
    table_candidate_cell_limit: int = 8
    figure_preview_tokens: int = 220
    recent_turns_recent_full_count: int = 2
    recent_turns_source_id_limit: int = 6


class RuleCompactor:
    """按 block 类型做结构保真压缩，不在规则层重新做语义相关性排序。"""

    def __init__(self, *, token_counter: TokenCounter, config: RuleCompactionConfig | None = None) -> None:
        self.token_counter = token_counter
        self.config = config or RuleCompactionConfig()

    def compact(self, block: PromptBlock) -> PromptBlock:
        if not self.config.enabled or not block.compactable:
            return self._with_token_counts(block)
        if block.block_type in {"table_final_evidence", "table_candidate_evidence", "plain_table_context"}:
            return self._compact_table_block(block)
        if block.block_type == "figure_evidence":
            return self._compact_figure_block(block)
        if block.section == "recent_turns":
            return self._compact_recent_turn(block)
        if block.section in {"session_summary", "user_memory"}:
            return self._compact_mapping_like_block(block)
        if block.section == "rag_evidence":
            return self._compact_rag_text_block(block)
        return self._with_token_counts(block)

    def _compact_rag_text_block(self, block: PromptBlock) -> PromptBlock:
        counted = self._with_token_counts(block)
        if counted.token_count <= self.config.rag_max_block_tokens and counted.context_role in {"anchor_evidence", "memory_context"}:
            return counted

        role = counted.context_role or counted.block_type
        if role in {"anchor_evidence", "memory_context"}:
            target = self.config.rag_max_block_tokens
            text = self._head_tail_by_tokens(counted.text, target)
            strategy = "anchor_head_tail"
            loss = "low"
        elif role == "sibling_context":
            target = self.config.rag_sibling_context_target_tokens
            text = self._head_by_tokens(counted.text, target)
            strategy = "sibling_head"
            loss = "medium"
        elif role in {"section_context", "parent_context"}:
            target = self.config.rag_section_context_target_tokens
            text = self._head_by_tokens(counted.text, target)
            strategy = "section_head"
            loss = "medium"
        else:
            target = self.config.rag_fallback_preview_tokens
            text = self._head_by_tokens(counted.text, target)
            strategy = "fallback_preview"
            loss = "high"

        return self._mark_rule_compacted(
            counted,
            text=text,
            strategy=strategy,
            content_loss_level=loss,
        )

    def _compact_table_block(self, block: PromptBlock) -> PromptBlock:
        evidence = block.metadata.get("table_evidence") if isinstance(block.metadata.get("table_evidence"), dict) else {}
        if block.block_type == "table_final_evidence" and evidence:
            text = self._render_table_final_evidence(block, evidence)
            return self._mark_rule_compacted(
                block,
                text=text,
                strategy="preserve_final_cells_drop_preview",
                content_loss_level="low",
                can_support_numeric_claim=True,
            )
        if block.block_type == "table_candidate_evidence" and evidence:
            text = self._render_table_candidate_evidence(block, evidence)
            return self._mark_rule_compacted(
                block,
                text=text,
                strategy="preserve_candidate_cells_limit",
                content_loss_level="medium",
                can_support_numeric_claim=False,
            )
        text = self._head_by_tokens(block.text, self.config.figure_preview_tokens)
        return self._mark_rule_compacted(
            block,
            text=text,
            strategy="plain_table_preview",
            content_loss_level="medium",
        )

    def _compact_figure_block(self, block: PromptBlock) -> PromptBlock:
        metadata = block.metadata or {}
        preview = self._head_by_tokens(str(metadata.get("asset_preview_text") or block.text), self.config.figure_preview_tokens)
        parts = [
            f"source_id: {block.source_id or metadata.get('source_id', '')}",
            f"page: {metadata.get('page_number', '')}" if metadata.get("page_number") else "",
            f"section: {metadata.get('section_path', '')}" if metadata.get("section_path") else "",
            f"caption: {metadata.get('caption', '')}" if metadata.get("caption") else "",
            f"summary: {metadata.get('asset_summary', '')}" if metadata.get("asset_summary") else "",
            f"preview: {preview}" if preview else "",
        ]
        return self._mark_rule_compacted(
            block,
            text="\n".join(part for part in parts if part).strip(),
            strategy="caption_summary_preview",
            content_loss_level="medium",
        ).with_updates(
            metadata={
                **dict(block.metadata or {}),
                "requires_image_input_for_visual_details": True,
            }
        )

    def _compact_recent_turn(self, block: PromptBlock) -> PromptBlock:
        metadata = block.metadata or {}
        source_ids = [
            str(item).strip()
            for item in list(metadata.get("cited_source_ids") or metadata.get("source_ids") or [])
            if str(item).strip()
        ][: self.config.recent_turns_source_id_limit]
        parts = [
            f"turn_id: {metadata.get('turn_id', block.block_id)}",
            f"question: {metadata.get('question', '')}" if metadata.get("question") else block.text,
            f"answer_summary: {metadata.get('answer_summary', '')}" if metadata.get("answer_summary") else "",
            f"resolved_reference: {metadata.get('resolved_reference', '')}" if metadata.get("resolved_reference") else "",
            f"cited_source_ids: {', '.join(source_ids)}" if source_ids else "",
        ]
        return self._mark_rule_compacted(
            block,
            text="\n".join(part for part in parts if part).strip(),
            strategy="recent_turn_tiered",
            content_loss_level="medium",
        )

    def _compact_mapping_like_block(self, block: PromptBlock) -> PromptBlock:
        payload = block.metadata.get("payload") if isinstance(block.metadata.get("payload"), dict) else None
        if payload is None:
            return self._with_token_counts(block)
        if block.section == "session_summary":
            keep_keys = {
                "topic",
                "current_focus",
                "resolved_constraints",
                "user_preferences",
                "open_questions",
                "referenced_entities",
                "referenced_sources",
            }
        else:
            keep_keys = {
                "preferred_answer_style",
                "positive_topics",
                "negative_topics",
                "recent_topics",
                "preferred_categories",
                "common_question_types",
            }
        compacted_payload = {key: value for key, value in payload.items() if key in keep_keys and value not in (None, "", [], {})}
        text = json.dumps(compacted_payload, ensure_ascii=False, default=str)
        return self._mark_rule_compacted(
            block,
            text=text,
            strategy=f"{block.section}_field_whitelist",
            content_loss_level="medium",
        )

    def _render_table_final_evidence(self, block: PromptBlock, evidence: Dict[str, Any]) -> str:
        table = evidence.get("table") if isinstance(evidence.get("table"), dict) else {}
        final = evidence.get("final_evidence") if isinstance(evidence.get("final_evidence"), dict) else {}
        cells = [cell for cell in list(final.get("cells") or []) if isinstance(cell, dict)]
        parts = [
            "[Table Final Evidence]",
            f"source_id: {block.source_id or block.metadata.get('source_id', '')}",
            f"table_id: {table.get('table_id', block.metadata.get('table_id', ''))}",
            f"page: {block.metadata.get('page_number', '')}" if block.metadata.get("page_number") else "",
            f"section: {block.metadata.get('section_path', '')}" if block.metadata.get("section_path") else "",
            f"operation_hint: {evidence.get('operation_hint', '')}" if evidence.get("operation_hint") else "",
            f"decision: {evidence.get('decision', '')}" if evidence.get("decision") else "",
            "cells:",
        ]
        for cell in cells:
            parts.append(
                "- row_index={row_index}; row_label={row_label}; col_name={col_name}; raw_value={raw_value}; "
                "normalized_value={normalized_value}; unit={unit}; confidence={confidence}".format(
                    row_index=cell.get("row_index", ""),
                    row_label=cell.get("row_label", ""),
                    col_name=cell.get("col_name", ""),
                    raw_value=cell.get("raw_value", ""),
                    normalized_value=cell.get("normalized_value", ""),
                    unit=cell.get("unit", ""),
                    confidence=cell.get("confidence", ""),
                )
            )
        return "\n".join(part for part in parts if part).strip()

    def _render_table_candidate_evidence(self, block: PromptBlock, evidence: Dict[str, Any]) -> str:
        table = evidence.get("table") if isinstance(evidence.get("table"), dict) else {}
        candidate = evidence.get("candidate_evidence") if isinstance(evidence.get("candidate_evidence"), dict) else {}
        cells = [cell for cell in list(candidate.get("candidate_cells") or []) if isinstance(cell, dict)]
        cells = cells[: self.config.table_candidate_cell_limit]
        parts = [
            "[Table Candidate Evidence]",
            f"source_id: {block.source_id or block.metadata.get('source_id', '')}",
            f"table_id: {table.get('table_id', block.metadata.get('table_id', ''))}",
            f"operation_hint: {evidence.get('operation_hint', '')}" if evidence.get("operation_hint") else "",
            "candidate_cells:",
        ]
        for cell in cells:
            parts.append(
                "- row_label={row_label}; col_name={col_name}; raw_value={raw_value}; unit={unit}; confidence={confidence}".format(
                    row_label=cell.get("row_label", ""),
                    col_name=cell.get("col_name", ""),
                    raw_value=cell.get("raw_value", ""),
                    unit=cell.get("unit", ""),
                    confidence=cell.get("confidence", ""),
                )
            )
        return "\n".join(part for part in parts if part).strip()

    def _mark_rule_compacted(
        self,
        block: PromptBlock,
        *,
        text: str,
        strategy: str,
        content_loss_level: str,
        can_support_numeric_claim: bool | None = None,
    ) -> PromptBlock:
        original_tokens = block.original_token_count or self.token_counter.count(block.text)
        compacted_tokens = self.token_counter.count(text)
        metadata = {
            **dict(block.metadata or {}),
            "compacted_by_rule": True,
            "compaction_strategy": strategy,
            "content_loss_level": content_loss_level,
            "original_tokens": original_tokens,
            "compacted_tokens": compacted_tokens,
        }
        return block.with_updates(
            text=text,
            token_count=compacted_tokens,
            original_token_count=original_tokens,
            evidence_origin="rule_compacted",
            can_support_numeric_claim=block.can_support_numeric_claim if can_support_numeric_claim is None else can_support_numeric_claim,
            metadata=metadata,
        )

    def _with_token_counts(self, block: PromptBlock) -> PromptBlock:
        token_count = block.token_count or self.token_counter.count(block.text)
        original_token_count = block.original_token_count or token_count
        return block.with_updates(token_count=token_count, original_token_count=original_token_count)

    def _head_by_tokens(self, text: str, token_limit: int) -> str:
        return self._trim_to_token_limit(self._normalize_text(text), token_limit)

    def _head_tail_by_tokens(self, text: str, token_limit: int) -> str:
        normalized = self._normalize_text(text)
        if self.token_counter.count(normalized) <= token_limit:
            return normalized
        head_budget = max(1, int(token_limit * 0.65))
        tail_budget = max(1, token_limit - head_budget - 8)
        head = self._trim_to_token_limit(normalized, head_budget)
        tail = self._trim_tail_to_token_limit(normalized, tail_budget)
        return f"{head}\n...[middle omitted by rule compaction]...\n{tail}".strip()

    def _trim_to_token_limit(self, text: str, token_limit: int) -> str:
        if token_limit <= 0:
            return ""
        normalized = self._normalize_text(text)
        if self.token_counter.count(normalized) <= token_limit:
            return normalized
        lo, hi = 0, len(normalized)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.token_counter.count(normalized[:mid]) <= token_limit:
                lo = mid
            else:
                hi = mid - 1
        return normalized[:lo].rstrip()

    def _trim_tail_to_token_limit(self, text: str, token_limit: int) -> str:
        if token_limit <= 0:
            return ""
        normalized = self._normalize_text(text)
        if self.token_counter.count(normalized) <= token_limit:
            return normalized
        lo, hi = 0, len(normalized)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.token_counter.count(normalized[len(normalized) - mid :]) <= token_limit:
                lo = mid
            else:
                hi = mid - 1
        return normalized[len(normalized) - lo :].lstrip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+\n", "\n", str(text or "")).strip()
