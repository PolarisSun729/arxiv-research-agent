from __future__ import annotations

from typing import Any, Dict, List, Tuple

from services.prompt_context.blocks import PromptBlock
from services.retrieval.table_evidence_formatter import render_table_evidence_prompt_block
from services.retrieval.table_evidence_schema import validate_table_evidence_payload


class ContextPackBuilder:
    """把检索结果统一打包成生成模型和前端都能复用的上下文载荷。"""

    def build(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_parts: List[str] = []
        image_inputs: List[Dict[str, Any]] = []
        asset_metadata: List[Dict[str, Any]] = []
        source_payload: List[Dict[str, Any]] = []
        generation_search_results: List[Dict[str, Any]] = []
        prompt_blocks: List[Dict[str, Any]] = []
        chunk_type_counts: Dict[str, int] = {}
        skipped_empty_content_count = 0
        table_evidence_count = 0
        table_cell_evidence_count = 0
        table_numeric_operations: List[str] = []

        for index, result in enumerate(search_results or [], start=1):
            source_id = self.build_source_id(result, index)
            chunk_type = self.normalize_chunk_type(result.get("chunk_type", "text"))
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            table_evidence = self.normalize_table_evidence(result)
            if table_evidence:
                table_evidence_count += 1
                table_cell_evidence_count += self.count_table_evidence_cells(table_evidence)
                operation = str(table_evidence.get("operation_hint", "") or "").strip()
                if operation and operation not in table_numeric_operations:
                    table_numeric_operations.append(operation)

            source = self.build_source_item(result, source_id=source_id, chunk_type=chunk_type)
            source_payload.append(source)

            asset_info = self.build_asset_metadata(result, source_id=source_id, chunk_type=chunk_type)
            text_block = self.build_text_block(result, index=index, source_id=source_id, chunk_type=chunk_type)
            if text_block:
                text_parts.append(text_block)
            else:
                skipped_empty_content_count += 1

            # 图片资产既要进入多模态输入，也要保留元数据，方便回答后回放证据链。
            if chunk_type == "figure" and asset_info.get("asset_abs_path"):
                image_inputs.append(
                    {
                        "source_id": source_id,
                        "image_path": asset_info["asset_abs_path"],
                        "page_number": asset_info["page_number"],
                        "asset_summary": asset_info["asset_summary"],
                        "section_path": asset_info["section_path"],
                        "asset_section_match_type": asset_info["asset_section_match_type"],
                        "asset_section_match_confidence": asset_info["asset_section_match_confidence"],
                    }
                )
                asset_metadata.append(asset_info)
            elif chunk_type == "table":
                asset_metadata.append(asset_info)

            generation_search_results.append(
                {
                    "source_id": source_id,
                    "text": text_block or str(result.get("content", "") or "").strip(),
                    "page_number": result.get("page_number", ""),
                    "source": result.get("source", ""),
                    "subchunk_label": result.get("subchunk_label", ""),
                    "chunk_label": result.get("chunk_label", result.get("subchunk_label", "")),
                    "section_path": result.get("section_path", ""),
                    "chunk_id": result.get("chunk_id", ""),
                    "parent_chunk_id": result.get("parent_chunk_id", result.get("chunk_id", "")),
                    "original_chunk_id": result.get("original_chunk_id", ""),
                    "chunk_type": chunk_type,
                    "context_role": result.get("context_role", ""),
                    "context_budget_score": result.get("context_budget_score"),
                    "context_budget_reason": result.get("context_budget_reason", ""),
                    "expansion_source_anchor_ids": result.get("expansion_source_anchor_ids", []),
                    "relationship_types": result.get("relationship_types", []),
                    "expansion_reasons": result.get("expansion_reasons", []),
                    "final_context_reason": result.get("final_context_reason", ""),
                    "asset_kind": result.get("asset_kind", ""),
                    "asset_summary": result.get("asset_summary", ""),
                    "asset_preview_text": result.get("asset_preview_text", ""),
                    "asset_section_match_type": result.get("asset_section_match_type", ""),
                    "asset_section_match_confidence": result.get("asset_section_match_confidence", 0.0),
                    "asset_section_match_reason": result.get("asset_section_match_reason", ""),
                    "asset_section_match_is_heuristic": result.get("asset_section_match_is_heuristic", False),
                    "asset_section_match_allow_embedding": result.get("asset_section_match_allow_embedding", False),
                    "table_id": result.get("table_id", ""),
                    "table_evidence_source_id": source_id if table_evidence else "",
                    "table_evidence": table_evidence,
                    "table_cell_citations": self.build_table_cell_citations(result, source_id=source_id),
                }
            )
            prompt_blocks.append(
                self.build_prompt_block(
                    result,
                    index=index,
                    source_id=source_id,
                    chunk_type=chunk_type,
                    text_block=text_block,
                    table_evidence=table_evidence,
                ).to_dict()
            )

        text_context = "\n\n".join(text_parts)
        context_budget_debug = {
            "source_count": len(source_payload),
            "chunk_type_counts": chunk_type_counts,
            "text_context_chars": len(text_context),
            "image_input_count": len(image_inputs),
            "asset_metadata_count": len(asset_metadata),
            "skipped_empty_content_count": skipped_empty_content_count,
            "table_evidence_count": table_evidence_count,
            "table_cell_evidence_count": table_cell_evidence_count,
            "table_numeric_operations": table_numeric_operations,
            "table_numeric_calculation_used": any(item in {"max", "min", "difference"} for item in table_numeric_operations),
        }
        return {
            "text_context": text_context,
            "image_inputs": image_inputs,
            "asset_metadata": asset_metadata,
            "source_payload": source_payload,
            "generation_search_results": generation_search_results,
            "prompt_blocks": prompt_blocks,
            "context_budget_debug": context_budget_debug,
        }

    def build_generation_context(
        self,
        search_results: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        context_pack = self.build(search_results)
        return context_pack["text_context"], context_pack["image_inputs"], context_pack["asset_metadata"]

    def build_source_payload(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.build(search_results)["source_payload"]

    @staticmethod
    def normalize_chunk_type(value: Any) -> str:
        chunk_type = str(value or "text").strip().lower()
        return chunk_type or "text"

    @staticmethod
    def build_source_id(result: Dict[str, Any], index: int) -> str:
        table_evidence = ContextPackBuilder.normalize_table_evidence(result)
        if table_evidence:
            table = table_evidence.get("table") if isinstance(table_evidence.get("table"), dict) else {}
            table_id = table.get("table_id") or result.get("table_id") or "table"
            chunk_id = result.get("chunk_id") or result.get("parent_chunk_id") or result.get("original_chunk_id") or index
            # v2 表格证据使用独立 source_id，避免和同一个 table chunk 的摘要/预览证据混淆。
            return "table-evidence-%s-%s" % (
                ContextPackBuilder.safe_source_token(table_id),
                ContextPackBuilder.safe_source_token(chunk_id),
            )
        # source_id 优先沿用检索结果中的稳定标识；没有时退回到 chunk 维度，保证历史记忆能重新关联来源。
        for key in ("source_id", "chunk_id", "parent_chunk_id", "original_chunk_id", "id"):
            value = result.get(key)
            if value not in (None, ""):
                normalized = str(value).strip()
                if normalized:
                    return normalized
        chunk_type = ContextPackBuilder.normalize_chunk_type(result.get("chunk_type", "text"))
        page_number = str(result.get("page_number", "") or result.get("page_range", "") or "unknown").strip()
        return f"source-{index}-{chunk_type}-p{page_number or 'unknown'}"

    @staticmethod
    def build_asset_metadata(result: Dict[str, Any], *, source_id: str, chunk_type: str) -> Dict[str, Any]:
        return {
            "source_id": source_id,
            "chunk_id": result.get("chunk_id", ""),
            "parent_chunk_id": result.get("parent_chunk_id", result.get("chunk_id", "")),
            "original_chunk_id": result.get("original_chunk_id", ""),
            "chunk_type": chunk_type,
            "asset_kind": result.get("asset_kind", ""),
            "asset_path": result.get("asset_path", ""),
            "asset_abs_path": result.get("asset_abs_path", ""),
            "asset_summary": result.get("asset_summary", ""),
            "asset_preview_text": result.get("asset_preview_text", ""),
            "table_id": result.get("table_id", ""),
            "table_evidence_source_id": source_id if ContextPackBuilder.normalize_table_evidence(result) else "",
            "table_evidence": ContextPackBuilder.normalize_table_evidence(result),
            "table_cell_citations": ContextPackBuilder.build_table_cell_citations(result, source_id=source_id),
            "page_number": result.get("page_number", ""),
            "page_range": result.get("page_range", ""),
            "section_path": result.get("section_path", ""),
            "section_title": result.get("section_title", ""),
            "asset_section_match_type": result.get("asset_section_match_type", ""),
            "asset_section_match_confidence": result.get("asset_section_match_confidence", 0.0),
            "asset_section_match_reason": result.get("asset_section_match_reason", ""),
            "asset_section_match_is_heuristic": result.get("asset_section_match_is_heuristic", False),
            "asset_section_match_allow_embedding": result.get("asset_section_match_allow_embedding", False),
            "source": result.get("source", ""),
        }

    @staticmethod
    def build_source_item(result: Dict[str, Any], *, source_id: str, chunk_type: str) -> Dict[str, Any]:
        return {
            "source_id": source_id,
            "content": result.get("content", ""),
            "page_number": result.get("page_number", ""),
            "source": result.get("source", ""),
            "subchunk_label": result.get("subchunk_label", ""),
            "chunk_label": result.get("chunk_label", result.get("subchunk_label", "")),
            "section_path": result.get("section_path", ""),
            "section_title": result.get("section_title", ""),
            "asset_section_match_type": result.get("asset_section_match_type", ""),
            "asset_section_match_confidence": result.get("asset_section_match_confidence", 0.0),
            "asset_section_match_reason": result.get("asset_section_match_reason", ""),
            "asset_section_match_is_heuristic": result.get("asset_section_match_is_heuristic", False),
            "asset_section_match_allow_embedding": result.get("asset_section_match_allow_embedding", False),
            "chunk_id": result.get("chunk_id", ""),
            "parent_chunk_id": result.get("parent_chunk_id", result.get("chunk_id", 0)),
            "original_chunk_id": result.get("original_chunk_id", ""),
            "chunk_type": chunk_type,
            "context_role": result.get("context_role", ""),
            "context_budget_score": result.get("context_budget_score"),
            "context_budget_reason": result.get("context_budget_reason", ""),
            "expansion_source_anchor_ids": result.get("expansion_source_anchor_ids", []),
            "relationship_types": result.get("relationship_types", []),
            "expansion_reasons": result.get("expansion_reasons", []),
            "final_context_reason": result.get("final_context_reason", ""),
            "asset_kind": result.get("asset_kind", ""),
            "asset_path": result.get("asset_path", ""),
            "asset_summary": result.get("asset_summary", ""),
            "asset_preview_text": result.get("asset_preview_text", ""),
            "asset_section_match_type": result.get("asset_section_match_type", ""),
            "asset_section_match_confidence": result.get("asset_section_match_confidence", 0.0),
            "asset_section_match_reason": result.get("asset_section_match_reason", ""),
            "asset_section_match_is_heuristic": result.get("asset_section_match_is_heuristic", False),
            "asset_section_match_allow_embedding": result.get("asset_section_match_allow_embedding", False),
            "table_id": result.get("table_id", ""),
            "table_evidence_source_id": source_id if ContextPackBuilder.normalize_table_evidence(result) else "",
            "table_evidence": ContextPackBuilder.normalize_table_evidence(result),
            "table_cell_citations": ContextPackBuilder.build_table_cell_citations(result, source_id=source_id),
        }

    @staticmethod
    def build_text_block(result: Dict[str, Any], *, index: int, source_id: str, chunk_type: str) -> str:
        content = str(result.get("content", "") or "").strip()
        asset_summary = str(result.get("asset_summary", "") or "").strip()
        asset_preview_text = str(result.get("asset_preview_text", "") or "").strip()
        section_path = str(result.get("section_path", "") or "").strip()
        page_number = str(result.get("page_number", "") or result.get("page_range", "") or "").strip()
        allow_section_anchor = ContextPackBuilder.asset_section_anchor_allowed(result, chunk_type)

        if chunk_type == "table":
            table_evidence_block = ContextPackBuilder.build_table_evidence_block(
                result,
                index=index,
                source_id=source_id,
                page_number=page_number,
                section_path=section_path if allow_section_anchor else "",
            )
            # 表格 chunk 的正文常为空；低置信度章节锚点只留在 metadata，避免误导生成上下文。
            parts = [
                f"[Table {index}]",
                f"source_id: {source_id}",
                f"page: {page_number}" if page_number else "",
                f"section: {section_path}" if section_path and allow_section_anchor else "",
                table_evidence_block,
                "Table Supplemental Context:" if table_evidence_block and (asset_summary or asset_preview_text) else "",
                asset_summary,
                asset_preview_text,
            ]
            return "\n".join(part for part in parts if part).strip()

        if chunk_type == "figure":
            # 图片 chunk 由 image_inputs 承载原图；弱章节锚点只在可信时进入文本上下文。
            parts = [
                f"[Figure {index}]",
                f"source_id: {source_id}",
                f"page: {page_number}" if page_number else "",
                f"section: {section_path}" if section_path and allow_section_anchor else "",
                asset_summary,
                asset_preview_text,
                content,
            ]
            return "\n".join(part for part in parts if part).strip()

        if not content:
            return ""
        # 文本证据也使用稳定 ID 描述，避免把旧的序号标签暴露给模型后被误复制成不可定位引用。
        header_parts = ["Evidence", f"source_id: {source_id}"]
        if page_number:
            header_parts.append(f"page: {page_number}")
        if section_path:
            header_parts.append(f"section: {section_path}")
        context_role = str(result.get("context_role", "") or "").strip()
        if context_role:
            header_parts.append(f"role: {context_role}")
        return "\n".join(header_parts + [content]).strip()

    @staticmethod
    def build_prompt_block(
        result: Dict[str, Any],
        *,
        index: int,
        source_id: str,
        chunk_type: str,
        text_block: str,
        table_evidence: Dict[str, Any],
    ) -> PromptBlock:
        context_role = str(result.get("context_role", "") or "").strip() or "fallback_context"
        block_type = ContextPackBuilder.prompt_block_type(
            chunk_type=chunk_type,
            context_role=context_role,
            table_evidence=table_evidence,
        )
        # priority 只表达粗粒度证据层级，相关性细排沿用检索/rerank 分数，避免在 prompt 层再做一套语义打分。
        priority = ContextPackBuilder.prompt_block_priority(block_type=block_type, context_role=context_role)
        score = ContextPackBuilder.optional_float(
            result.get("context_budget_score", result.get("llm_rerank_score", result.get("fusion_score", result.get("score", 0.0))))
        )
        metadata = {
            "page_number": result.get("page_number", ""),
            "page_range": result.get("page_range", ""),
            "section_path": result.get("section_path", ""),
            "section_title": result.get("section_title", ""),
            "chunk_id": result.get("chunk_id", ""),
            "parent_chunk_id": result.get("parent_chunk_id", ""),
            "original_chunk_id": result.get("original_chunk_id", ""),
            "chunk_type": chunk_type,
            "context_role": context_role,
            "relationship_types": result.get("relationship_types", []),
            "matched_routes": result.get("matched_routes", []),
            "table_id": result.get("table_id", ""),
            "table_evidence": table_evidence,
            "asset_summary": result.get("asset_summary", ""),
            "asset_preview_text": result.get("asset_preview_text", ""),
            "asset_abs_path": result.get("asset_abs_path", ""),
        }
        return PromptBlock(
            block_id=f"rag:{source_id}",
            section="rag_evidence",
            block_type=block_type,
            source="context_pack_builder",
            text=text_block or str(result.get("content", "") or "").strip(),
            priority=priority,
            score=score,
            original_rank=index,
            context_role=context_role,
            budget_group="rag_core" if priority <= 1 else "rag_expansion",
            protected=False,
            droppable=priority > 0,
            compactable=True,
            evidence_origin="raw",
            source_id=source_id,
            original_source_id=source_id,
            can_support_numeric_claim=block_type == "table_final_evidence",
            can_support_direct_quote=chunk_type == "text",
            metadata=metadata,
        )

    @staticmethod
    def prompt_block_type(*, chunk_type: str, context_role: str, table_evidence: Dict[str, Any]) -> str:
        if chunk_type == "table":
            final = table_evidence.get("final_evidence") if isinstance(table_evidence, dict) else {}
            candidate = table_evidence.get("candidate_evidence") if isinstance(table_evidence, dict) else {}
            if isinstance(final, dict) and final.get("cells"):
                return "table_final_evidence"
            if isinstance(candidate, dict) and candidate.get("candidate_cells"):
                return "table_candidate_evidence"
            return "plain_table_context"
        if chunk_type == "figure":
            return "figure_evidence"
        return context_role or "fallback_context"

    @staticmethod
    def prompt_block_priority(*, block_type: str, context_role: str) -> int:
        if block_type in {"anchor_evidence", "table_final_evidence"}:
            return 0
        if block_type in {"table_candidate_evidence", "figure_evidence", "memory_context"}:
            return 1
        if context_role in {"sibling_context", "section_context", "parent_context"}:
            return 2
        return 4

    @staticmethod
    def optional_float(value: Any) -> float:
        try:
            if value in (None, ""):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def asset_section_anchor_allowed(result: Dict[str, Any], chunk_type: str) -> bool:
        """旧数据缺少 match 字段时沿用历史行为；新 asset 按显式门控决定是否文本化 section。"""
        normalized_type = ContextPackBuilder.normalize_chunk_type(chunk_type)
        if normalized_type not in {"figure", "table"}:
            return True
        match_type = str(result.get("asset_section_match_type", "") or "").strip()
        if not match_type:
            return True
        return bool(result.get("asset_section_match_allow_embedding", False))

    @staticmethod
    def normalize_table_evidence(result: Dict[str, Any]) -> Dict[str, Any]:
        evidence = result.get("table_evidence")
        if not isinstance(evidence, dict):
            return {}
        if not evidence:
            return {}
        # 这里只接受 v2 schema；旧表格证据字段不做翻译，避免继续扩散兼容层。
        validate_table_evidence_payload(evidence)
        return dict(evidence)

    @staticmethod
    def build_table_evidence_block(
        result: Dict[str, Any],
        *,
        index: int,
        source_id: str,
        page_number: str,
        section_path: str,
    ) -> str:
        evidence = ContextPackBuilder.normalize_table_evidence(result)
        if not evidence:
            return ""
        return render_table_evidence_prompt_block(evidence, source_id=source_id).strip()

    @staticmethod
    def table_evidence_cells(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从 v2 final/candidate evidence 中抽取单元格，供引用和统计共用同一口径。"""
        cells: List[Dict[str, Any]] = []
        final = evidence.get("final_evidence") if isinstance(evidence.get("final_evidence"), dict) else {}
        candidate = evidence.get("candidate_evidence") if isinstance(evidence.get("candidate_evidence"), dict) else {}
        for cell in list(final.get("cells") or []) + list(candidate.get("candidate_cells") or []):
            if not isinstance(cell, dict):
                continue
            dedupe_key = (
                cell.get("row_index"),
                cell.get("row_label"),
                cell.get("col_name"),
                cell.get("raw_value"),
            )
            if any(
                (
                    existing.get("row_index"),
                    existing.get("row_label"),
                    existing.get("col_name"),
                    existing.get("raw_value"),
                ) == dedupe_key
                for existing in cells
            ):
                continue
            cells.append(cell)
        return cells

    @staticmethod
    def count_table_evidence_cells(evidence: Dict[str, Any]) -> int:
        return len(ContextPackBuilder.table_evidence_cells(evidence))

    @staticmethod
    def build_table_cell_citations(result: Dict[str, Any], *, source_id: str) -> List[Dict[str, Any]]:
        evidence = ContextPackBuilder.normalize_table_evidence(result)
        if not evidence:
            return []
        table = evidence.get("table") if isinstance(evidence.get("table"), dict) else {}
        table_id = str(table.get("table_id") or result.get("table_id") or "").strip()
        decision = str(evidence.get("decision") or "").strip()
        operation = str(evidence.get("operation_hint") or "").strip()
        citations: List[Dict[str, Any]] = []
        for offset, cell in enumerate(ContextPackBuilder.table_evidence_cells(evidence), start=1):
            if not isinstance(cell, dict):
                continue
            # cell_source_id 绑定到 table/source/row/column，前端后续可直接定位到具体单元格。
            citations.append(
                {
                    "cell_source_id": "%s-cell-%s-%s-%s" % (
                        source_id,
                        offset,
                        ContextPackBuilder.safe_source_token(cell.get("row_label", "row")),
                        ContextPackBuilder.safe_source_token(cell.get("col_name", "col")),
                    ),
                    "source_id": source_id,
                    "table_id": table_id,
                    "decision": decision,
                    "operation": operation,
                    "page_number": result.get("page_number", ""),
                    "section_path": result.get("section_path", ""),
                    "source_chunk_id": result.get("chunk_id", ""),
                    "original_chunk_id": result.get("original_chunk_id", ""),
                    "row_index": cell.get("row_index"),
                    "row_label": cell.get("row_label"),
                    "col_name": cell.get("col_name"),
                    "raw_value": cell.get("raw_value"),
                    "normalized_value": cell.get("normalized_value"),
                    "unit": cell.get("unit"),
                    "confidence": cell.get("confidence"),
                }
            )
        return citations

    @staticmethod
    def safe_source_token(value: Any) -> str:
        normalized = str(value or "").strip()
        normalized = "".join(ch if ch.isalnum() else "-" for ch in normalized)
        normalized = "-".join(part for part in normalized.split("-") if part)
        return normalized.lower() or "unknown"
