from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.token_counter import TokenCounter


# 三类表格 block 都走结构化表格压缩；LLM 压缩器也用同一集合识别并跳过表格证据。
TABLE_BLOCK_TYPES = {"table_final_evidence", "table_candidate_evidence", "plain_table_context"}

# 会话摘要规则压缩时保留的字段。
# 前半部分对应 paper_qa.session_service.compact_session_summary 的实际输出结构，
# 后半部分是早期设计中的字段名，保留以兼容其他调用方传入的摘要。
SESSION_SUMMARY_KEEP_KEYS = {
    "topic",
    "summary",
    "confirmed_facts",
    "user_preferences",
    "task_progress",
    "source_clues",
    "current_focus",
    "resolved_constraints",
    "open_questions",
    "referenced_entities",
    "referenced_sources",
}

# 用户画像规则压缩时保留的字段，与 PromptContextBuilder._format_user_memory 输出的字段一致。
USER_MEMORY_KEEP_KEYS = {
    "preferred_answer_style",
    "positive_topics",
    "negative_topics",
    "recent_topics",
    "preferred_categories",
    "common_question_types",
}


@dataclass
class RuleCompactionConfig:
    """规则压缩参数，单位均为 token（table_candidate_cell_limit 与 recent_turns_* 除外）。

    Attributes:
        enabled: 关闭后所有 block 只补 token 数，不做任何改写。
        rag_max_block_tokens: anchor / memory 证据的上限，超过时保留头尾、省略中间。
        rag_sibling_context_target_tokens: 兄弟 chunk 上下文只保留开头的长度。
        rag_section_context_target_tokens: 章节 / 父级上下文只保留开头的长度。
        rag_fallback_preview_tokens: 未知角色的 RAG 文本只保留的预览长度。
        table_candidate_cell_limit: 候选表格证据最多保留的单元格数。
        figure_preview_tokens: 图片证据预览长度；无结构化证据的普通表格也复用该长度。
        recent_turns_recent_full_count: 最近 N 轮历史对话保留全文（不做规则压缩）。
            轮次新旧由 block.metadata["turn_recency"] 标识（0 表示最新一轮），由 builder 写入。
        recent_turns_older_answer_tokens: 更早轮次的回答摘要最多保留的 token 数。
        recent_turns_source_id_limit: 更早轮次最多保留的引用 source_id 数。
    """

    enabled: bool = True
    rag_max_block_tokens: int = 4000
    rag_sibling_context_target_tokens: int = 350
    rag_section_context_target_tokens: int = 250
    rag_fallback_preview_tokens: int = 160
    table_candidate_cell_limit: int = 8
    figure_preview_tokens: int = 220
    recent_turns_recent_full_count: int = 2
    recent_turns_older_answer_tokens: int = 120
    recent_turns_source_id_limit: int = 6


class RuleCompactor:
    """按 block 类型做结构保真压缩，不在规则层重新做语义相关性排序。

    “结构保真”指压缩只做确定性的裁剪或字段筛选（截断、白名单、保留表格单元格等），
    不改写语义；需要语义级压缩时由下游可选的 LLMCompactor 处理。
    """

    def __init__(self, *, token_counter: TokenCounter, config: RuleCompactionConfig | None = None) -> None:
        self.token_counter = token_counter
        self.config = config or RuleCompactionConfig()

    def compact(self, block: PromptBlock) -> PromptBlock:
        """按 block 类型分派到对应压缩策略；不可压缩或未匹配的 block 只补 token 数。

        分派优先级：表格 → 图片 → 历史对话 → 摘要/画像 → 其他 RAG 文本。
        """
        if not self.config.enabled or not block.compactable:
            return self._with_token_counts(block)
        if block.block_type in TABLE_BLOCK_TYPES:
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
        """按 context_role 决定普通 RAG 文本的压缩力度。

        - anchor_evidence / memory_context：直接命中的核心证据，超长时保留头尾（信息损失低）；
        - sibling_context：相邻 chunk，只保留开头；
        - section_context / parent_context：所在章节上下文，只保留开头；
        - 其他：只保留一小段预览（信息损失高）。
        文本本身未超过目标长度时原样返回，不打“已压缩”标记，避免诊断中出现虚假的压缩记录。
        """
        counted = self._with_token_counts(block)
        role = counted.context_role or counted.block_type
        if role in {"anchor_evidence", "memory_context"}:
            target = self.config.rag_max_block_tokens
            strategy = "anchor_head_tail"
            loss = "low"
        elif role == "sibling_context":
            target = self.config.rag_sibling_context_target_tokens
            strategy = "sibling_head"
            loss = "medium"
        elif role in {"section_context", "parent_context"}:
            target = self.config.rag_section_context_target_tokens
            strategy = "section_head"
            loss = "medium"
        else:
            target = self.config.rag_fallback_preview_tokens
            strategy = "fallback_preview"
            loss = "high"

        if counted.token_count <= target:
            return counted

        if strategy == "anchor_head_tail":
            text = self._head_tail_by_tokens(counted.text, target)
        else:
            text = self._head_by_tokens(counted.text, target)
        return self._mark_rule_compacted(
            counted,
            text=text,
            strategy=strategy,
            content_loss_level=loss,
        )

    def _compact_table_block(self, block: PromptBlock) -> PromptBlock:
        """表格证据压缩：结构化证据只保留单元格，丢掉表格预览；没有结构化证据时退化为文本预览。"""
        evidence = block.metadata.get("table_evidence") if isinstance(block.metadata.get("table_evidence"), dict) else {}
        if block.block_type == "table_final_evidence" and evidence:
            # 已确定的最终单元格可以支撑数值结论，因此显式置 can_support_numeric_claim=True。
            text = self._render_table_final_evidence(block, evidence)
            return self._mark_rule_compacted(
                block,
                text=text,
                strategy="preserve_final_cells_drop_preview",
                content_loss_level="low",
                can_support_numeric_claim=True,
            )
        if block.block_type == "table_candidate_evidence" and evidence:
            # 候选单元格存在歧义，不能直接支撑数值结论。
            text = self._render_table_candidate_evidence(block, evidence)
            return self._mark_rule_compacted(
                block,
                text=text,
                strategy="preserve_candidate_cells_limit",
                content_loss_level="medium",
                can_support_numeric_claim=False,
            )
        # 普通表格上下文没有单独的长度配置，复用 figure_preview_tokens。
        text = self._head_by_tokens(block.text, self.config.figure_preview_tokens)
        return self._mark_rule_compacted(
            block,
            text=text,
            strategy="plain_table_preview",
            content_loss_level="medium",
        )

    def _compact_figure_block(self, block: PromptBlock) -> PromptBlock:
        """图片证据压缩：只保留 source_id、页码、章节、标题、摘要和一段文字预览。

        额外打上 requires_image_input_for_visual_details 标记，提示下游：
        涉及图中视觉细节的问题，仅靠文本 prompt 无法可靠回答。
        """
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
        """历史对话分层压缩。

        - 最近 recent_turns_recent_full_count 轮（metadata.turn_recency 小于该值）保留全文，不做改写；
        - 更早的轮次（或没有 turn_recency 的 block）按固定字段重建文本：
          回答摘要截断到 recent_turns_older_answer_tokens，source_id 只保留前 recent_turns_source_id_limit 个。
        - metadata 中既没有问题也没有回答摘要时，无法安全地重建文本，原样保留。
        重建结果与原文完全相同时不打“已压缩”标记，避免诊断中出现虚假的压缩记录。
        """
        metadata = block.metadata or {}
        recency = metadata.get("turn_recency")
        if isinstance(recency, int) and recency < max(0, int(self.config.recent_turns_recent_full_count)):
            return self._with_token_counts(block)
        if not metadata.get("question") and not metadata.get("answer_summary"):
            return self._with_token_counts(block)
        source_ids = [
            str(item).strip()
            for item in list(metadata.get("cited_source_ids") or metadata.get("source_ids") or [])
            if str(item).strip()
        ][: self.config.recent_turns_source_id_limit]
        answer_summary = str(metadata.get("answer_summary") or "").strip()
        if answer_summary:
            answer_summary = self._head_by_tokens(answer_summary, max(1, int(self.config.recent_turns_older_answer_tokens)))
        parts = [
            f"turn_id: {metadata.get('turn_id') or block.block_id}",
            f"question: {metadata.get('question', '')}" if metadata.get("question") else "",
            f"answer_summary: {answer_summary}" if answer_summary else "",
            f"resolved_reference: {metadata.get('resolved_reference', '')}" if metadata.get("resolved_reference") else "",
            f"cited_source_ids: {', '.join(source_ids)}" if source_ids else "",
        ]
        text = "\n".join(part for part in parts if part).strip()
        if text == self._normalize_text(block.text):
            return self._with_token_counts(block)
        return self._mark_rule_compacted(
            block,
            text=text,
            strategy="recent_turn_older_summary",
            content_loss_level="medium",
        )

    def _compact_mapping_like_block(self, block: PromptBlock) -> PromptBlock:
        """会话摘要 / 用户画像压缩：从 metadata.payload 中按白名单挑字段，序列化为紧凑 JSON。

        没有 payload，或白名单过滤后为空时原样保留：宁可多占一些 token，也不要把摘要整个清空。
        """
        payload = block.metadata.get("payload") if isinstance(block.metadata.get("payload"), dict) else None
        if payload is None:
            return self._with_token_counts(block)
        keep_keys = SESSION_SUMMARY_KEEP_KEYS if block.section == "session_summary" else USER_MEMORY_KEEP_KEYS
        compacted_payload = {key: value for key, value in payload.items() if key in keep_keys and value not in (None, "", [], {})}
        if not compacted_payload:
            return self._with_token_counts(block)
        text = json.dumps(compacted_payload, ensure_ascii=False, default=str)
        return self._mark_rule_compacted(
            block,
            text=text,
            strategy=f"{block.section}_field_whitelist",
            content_loss_level="medium",
        )

    def _render_table_final_evidence(self, block: PromptBlock, evidence: Dict[str, Any]) -> str:
        """把最终表格证据渲染为逐单元格的文本，保留行列定位、原值、归一化值和单位。"""
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
        """把候选表格证据渲染为文本，只保留前 table_candidate_cell_limit 个候选单元格。"""
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
        """用压缩后的文本替换 block，并在 metadata 中记录压缩策略、损失等级与前后 token 数。

        can_support_numeric_claim 为 None 时沿用原值，只有表格策略会显式设置。
        """
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
        """不改写文本，只补齐 token_count / original_token_count。"""
        token_count = block.token_count or self.token_counter.count(block.text)
        original_token_count = block.original_token_count or token_count
        return block.with_updates(token_count=token_count, original_token_count=original_token_count)

    def _head_by_tokens(self, text: str, token_limit: int) -> str:
        """保留开头不超过 token_limit 的部分。"""
        return self._trim_by_tokens(text, token_limit, from_tail=False)

    def _head_tail_by_tokens(self, text: str, token_limit: int) -> str:
        """头部约 65%、尾部约 35% 的预算各保留一段，中间用省略标记连接。

        论文段落的结论常出现在开头或结尾，头尾保留比单纯截断开头丢失的关键信息更少。
        尾部预算额外扣掉约 8 个 token，留给省略标记本身。
        """
        normalized = self._normalize_text(text)
        if self.token_counter.count(normalized) <= token_limit:
            return normalized
        head_budget = max(1, int(token_limit * 0.65))
        tail_budget = max(1, token_limit - head_budget - 8)
        head = self._trim_by_tokens(normalized, head_budget, from_tail=False)
        tail = self._trim_by_tokens(normalized, tail_budget, from_tail=True)
        return f"{head}\n...[middle omitted by rule compaction]...\n{tail}".strip()

    def _trim_by_tokens(self, text: str, token_limit: int, *, from_tail: bool) -> str:
        """二分查找能放进 token_limit 的最长前缀（from_tail=True 时为最长后缀）。

        tokenizer 对子串的计数不一定单调，但二分结果总能保证返回值不超过上限，足以满足预算需求。
        """
        if token_limit <= 0:
            return ""
        normalized = self._normalize_text(text)
        if self.token_counter.count(normalized) <= token_limit:
            return normalized
        total = len(normalized)

        def piece(length: int) -> str:
            return normalized[total - length :] if from_tail else normalized[:length]

        lo, hi = 0, total
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.token_counter.count(piece(mid)) <= token_limit:
                lo = mid
            else:
                hi = mid - 1
        return piece(lo).lstrip() if from_tail else piece(lo).rstrip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """去掉行尾空白并 strip，避免空白字符白白占用 token。"""
        return re.sub(r"\s+\n", "\n", str(text or "")).strip()
