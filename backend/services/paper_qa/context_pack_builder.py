from __future__ import annotations

from typing import Any, Dict, List, Tuple


class ContextPackBuilder:
    """把检索结果统一打包成生成模型和前端都能复用的上下文载荷。"""

    def build(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_parts: List[str] = []
        image_inputs: List[Dict[str, Any]] = []
        asset_metadata: List[Dict[str, Any]] = []
        source_payload: List[Dict[str, Any]] = []
        generation_search_results: List[Dict[str, Any]] = []
        chunk_type_counts: Dict[str, int] = {}
        skipped_empty_content_count = 0

        for index, result in enumerate(search_results or [], start=1):
            source_id = self.build_source_id(result, index)
            chunk_type = self.normalize_chunk_type(result.get("chunk_type", "text"))
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1

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
                }
            )

        text_context = "\n\n".join(text_parts)
        context_budget_debug = {
            "source_count": len(source_payload),
            "chunk_type_counts": chunk_type_counts,
            "text_context_chars": len(text_context),
            "image_input_count": len(image_inputs),
            "asset_metadata_count": len(asset_metadata),
            "skipped_empty_content_count": skipped_empty_content_count,
        }
        return {
            "text_context": text_context,
            "image_inputs": image_inputs,
            "asset_metadata": asset_metadata,
            "source_payload": source_payload,
            "generation_search_results": generation_search_results,
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
            "table_structured_text": result.get("table_structured_text", ""),
            "table_structured_evidence": result.get("table_structured_evidence", {}),
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
            "table_structured_text": result.get("table_structured_text", ""),
            "table_structured_evidence": result.get("table_structured_evidence", {}),
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
            # 表格 chunk 的正文常为空；低置信度章节锚点只留在 metadata，避免误导生成上下文。
            parts = [
                f"[Table {index}]",
                f"source_id: {source_id}",
                f"page: {page_number}" if page_number else "",
                f"section: {section_path}" if section_path and allow_section_anchor else "",
                asset_summary,
                str(result.get("table_structured_text", "") or "").strip(),
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
        header_parts = [f"[Source {index}]", f"source_id: {source_id}"]
        if page_number:
            header_parts.append(f"page: {page_number}")
        if section_path:
            header_parts.append(f"section: {section_path}")
        context_role = str(result.get("context_role", "") or "").strip()
        if context_role:
            header_parts.append(f"role: {context_role}")
        return "\n".join(header_parts + [content]).strip()

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
