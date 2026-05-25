from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

MAX_CHUNK_CONTENT_LENGTH = 8000
CHUNK_OVERLAP_LENGTH = 200


class ChunkingService:
    """
    Chunk PDF pages using page and heading structure only.
    """

    def chunk_pymupdf(
        self,
        text: Union[str, dict],
        metadata: dict,
        page_map: list = None,
        method: str = "by_titles",
        chunk_size: int = 500,
    ) -> dict:
        pymupdf_metadata = dict(metadata or {})
        pymupdf_metadata["loading_method"] = "pymupdf"
        return self.chunk_text(
            text=text,
            method=method,
            metadata=pymupdf_metadata,
            page_map=page_map,
            chunk_size=chunk_size,
        )

    def chunk_docling(
        self,
        text: Union[str, dict],
        metadata: dict,
        page_map: list = None,
        chunk_size: int = 500,
    ) -> dict:
        docling_metadata = dict(metadata or {})
        docling_metadata["loading_method"] = "docling"
        return self.chunk_text(
            text=text,
            method="docling_sections",
            metadata=docling_metadata,
            page_map=page_map,
            chunk_size=chunk_size,
        )

    def chunk_text(
        self,
        text: Union[str, dict],
        method: str,
        metadata: dict,
        page_map: list = None,
        chunk_size: int = 500,
    ) -> dict:
        try:
            normalized_page_map = self._normalize_page_map(text, page_map)
            if not normalized_page_map:
                raise ValueError("Page map is required for chunking.")

            filename = metadata.get("filename", "")
            source_name = metadata.get("filename", "") or metadata.get("source", "")
            loading_method = str(metadata.get("loading_method", "") or "").strip().lower()
            if loading_method == "docling":
                chunks = self._chunk_docling_sections(text, normalized_page_map, source_name, chunk_size)
            elif method == "by_pages":
                chunks = self._chunk_by_pages(normalized_page_map, source_name)
            elif method == "fixed_size":
                chunks = self._chunk_fixed_size(normalized_page_map, source_name, chunk_size)
            elif method == "by_titles":
                chunks = self._chunk_by_titles(normalized_page_map, source_name, chunk_size)
            elif method == "by_paragraphs":
                chunks = self._chunk_by_paragraphs(normalized_page_map, source_name, chunk_size)
            elif method == "by_sentences":
                chunks = self._chunk_by_sentences(normalized_page_map, source_name, chunk_size)
            else:
                raise ValueError(f"Unsupported chunking method: {method}")

            return self._finalize_chunk_output(
                filename=filename,
                chunks=chunks,
                total_pages=len(normalized_page_map),
                loading_method=metadata.get("loading_method", ""),
                chunking_method=method,
                pages=normalized_page_map,
            )
        except Exception as e:
            logger.error(f"Error in chunk_text: {str(e)}")
            raise

    def _finalize_chunk_output(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        total_pages: int,
        loading_method: str,
        chunking_method: str,
        pages: List[Dict[str, Any]],
    ) -> dict:
        for index, chunk in enumerate(chunks, start=1):
            chunk.setdefault("content", str(chunk.get("content", "") or ""))
            metadata = chunk.setdefault("metadata", {})
            metadata.setdefault("chunk_type", "text")
            chunk["metadata"]["chunk_index"] = index
            chunk["metadata"]["chunk_id"] = index
            chunk["metadata"]["total_chunks"] = len(chunks)

        chunks = self._expand_overlong_chunks(
            chunks,
            max_length=MAX_CHUNK_CONTENT_LENGTH,
            overlap=CHUNK_OVERLAP_LENGTH,
        )

        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk["metadata"]
            metadata["chunk_index"] = index
            metadata["chunk_id"] = index
            metadata["total_chunks"] = len(chunks)
            metadata["parent_chunk_id"] = int(metadata.get("parent_chunk_id", index))
            metadata["subchunk_index"] = int(metadata.get("subchunk_index", 1))
            metadata["subchunk_count"] = int(metadata.get("subchunk_count", 1))
            metadata["subchunk_label"] = str(
                metadata.get(
                    "subchunk_label",
                    f"chunk {metadata['parent_chunk_id']} part {metadata['subchunk_index']}/{metadata['subchunk_count']}",
                )
            )

        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "total_pages": total_pages,
            "loading_method": loading_method,
            "chunking_method": chunking_method,
            "timestamp": datetime.now().isoformat(),
            "pages": pages,
            "chunks": chunks,
        }

    def _expand_overlong_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_length: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        for chunk in chunks:
            content = str(chunk.get("content", "") or "")
            metadata = dict(chunk.get("metadata", {}))
            parent_chunk_id = int(metadata.get("chunk_id", metadata.get("chunk_index", len(expanded) + 1)))

            if len(content) <= max_length:
                metadata["parent_chunk_id"] = parent_chunk_id
                metadata["subchunk_index"] = 1
                metadata["subchunk_count"] = 1
                metadata["subchunk_label"] = f"chunk {parent_chunk_id} part 1/1"
                metadata["word_count"] = len(content.split())
                expanded.append({"content": content, "metadata": metadata})
                continue

            parts = self._split_overlong_content(content, max_length=max_length, overlap=overlap)
            part_count = len(parts)
            for part_index, part in enumerate(parts, start=1):
                part_metadata = {
                    **metadata,
                    "parent_chunk_id": parent_chunk_id,
                    "subchunk_index": part_index,
                    "subchunk_count": part_count,
                    "subchunk_label": f"chunk {parent_chunk_id} part {part_index}/{part_count}",
                    "word_count": len(part.split()),
                }
                expanded.append({"content": part, "metadata": part_metadata})

        return expanded

    def _split_overlong_content(self, text: str, max_length: int, overlap: int) -> List[str]:
        normalized = (text or "").replace("\r\n", "\n").strip()
        if not normalized:
            return [""]
        if len(normalized) <= max_length:
            return [normalized]

        parts: List[str] = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            end = min(start + max_length, text_length)
            if end < text_length:
                boundary = self._find_split_boundary(normalized, start, end)
                if boundary > start:
                    end = boundary

            part = normalized[start:end].strip()
            if part:
                parts.append(part)

            if end >= text_length:
                break

            next_start = max(end - overlap, start + 1)
            start = next_start

        return parts or [normalized[:max_length]]

    def _find_split_boundary(self, text: str, start: int, end: int, search_window: int = 400) -> int:
        lower = max(start + 1, end - search_window)
        candidates = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " "]
        best = start

        for token in candidates:
            idx = text.rfind(token, lower, end)
            if idx != -1:
                boundary = idx + len(token)
                if boundary > best:
                    best = boundary

        return best

    def _normalize_page_map(self, text: Union[str, dict], page_map: Optional[list]) -> List[Dict[str, Any]]:
        if page_map:
            normalized = []
            for page in page_map:
                if not isinstance(page, dict):
                    continue
                normalized.append(
                    {
                        **page,
                        "page": int(page.get("page", page.get("page_number", len(normalized) + 1))),
                        "text": str(page.get("text", "") or page.get("content", "")),
                    }
                )
            return normalized

        if isinstance(text, dict):
            pages = text.get("pages") or text.get("content") or []
            if isinstance(pages, list):
                return self._normalize_page_map("", pages)

        if isinstance(text, str) and text.strip():
            return [{"page": 1, "text": text.strip()}]

        return []

    def _build_chunk_metadata(
        self,
        source: str,
        page_start: int,
        page_end: int,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        chunking_method: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = {
            "source": source,
            "document_name": source,
            "chunk_index": int(chunk_index),
            "chunk_id": int(chunk_index),
            "page_start": int(page_start),
            "page_end": int(page_end),
            "page_number": str(page_start),
            "page_range": f"{page_start}-{page_end}",
            "word_count": len(chunk_text.split()),
            "total_chunks": int(total_chunks),
            "chunking_method": chunking_method,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata

    def _chunk_by_pages(self, normalized_page_map: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for page in normalized_page_map:
            text = self._page_text(page)
            if not text:
                continue

            page_num = int(page.get("page", page.get("page_number", 1)))
            chunks.append(
                {
                    "content": text,
                    "metadata": self._build_chunk_metadata(
                        source=source_name,
                        page_start=page_num,
                        page_end=page_num,
                        chunk_text=text,
                        chunk_index=0,
                        total_chunks=0,
                        chunking_method="by_pages",
                    ),
                }
            )
        return chunks

    def _chunk_fixed_size(
        self,
        normalized_page_map: List[Dict[str, Any]],
        source_name: str,
        chunk_size: int,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for page in normalized_page_map:
            text = self._page_text(page)
            if not text:
                continue

            page_num = int(page.get("page", page.get("page_number", 1)))
            for part in self._split_text_by_words(text, chunk_size):
                part = part.strip()
                if not part:
                    continue
                chunks.append(
                    {
                        "content": part,
                        "metadata": self._build_chunk_metadata(
                            source=source_name,
                            page_start=page_num,
                            page_end=page_num,
                            chunk_text=part,
                            chunk_index=0,
                            total_chunks=0,
                            chunking_method="fixed_size",
                        ),
                    }
                )
        return chunks

    def _chunk_by_titles(
        self,
        normalized_page_map: List[Dict[str, Any]],
        source_name: str,
        chunk_size: int,
    ) -> List[Dict[str, Any]]:
        lines = self._merge_heading_fragments(self._flatten_document_lines(normalized_page_map))
        if not lines:
            return []

        chunks: List[Dict[str, Any]] = []
        current_title: Optional[str] = None
        current_level: int = 0
        current_lines: List[Dict[str, Any]] = []

        def flush_section() -> None:
            nonlocal current_title, current_level, current_lines
            if not current_lines:
                return

            section_lines = current_lines[1:] if current_title else current_lines
            section_body = self._compose_text_from_lines(section_lines).strip()
            section_title = current_title or "Preamble"
            page_start = int(current_lines[0]["page"])
            page_end = int(current_lines[-1]["page"])

            section_text = f"{current_title}\n{section_body}".strip() if current_title else section_body
            if not section_text:
                current_title = None
                current_level = 0
                current_lines = []
                return

            section_chunks = self._split_text_by_sentence_chunks(section_body, chunk_size)
            if not section_chunks:
                section_chunks = [section_body or section_text]

            for part_index, part in enumerate(section_chunks, start=1):
                part_text = part.strip()
                if current_title:
                    part_text = f"{current_title}\n{part_text}".strip()

                chunks.append(
                    {
                        "content": part_text,
                        "metadata": self._build_chunk_metadata(
                            source=source_name,
                            page_start=page_start,
                            page_end=page_end,
                            chunk_text=part_text,
                            chunk_index=0,
                            total_chunks=0,
                            chunking_method="by_titles",
                            extra_metadata={
                                "chunk_type": "text",
                                "section_title": section_title,
                                "section_level": current_level,
                                "section_part_index": part_index,
                                "section_part_count": len(section_chunks),
                            },
                        ),
                    }
                )

            current_title = None
            current_level = 0
            current_lines = []

        for line in lines:
            text = str(line.get("text", "")).strip()
            if not text:
                continue

            heading_level = self._heading_level(line)
            # Keep the old level-1 split logic for later reuse.
            # if heading_level == 1:
            #     flush_section()
            #     current_title = text
            #     current_level = int(heading_level)
            #     current_lines = [line]
            #     continue

            if heading_level == 2:
                flush_section()
                current_title = text
                current_level = int(heading_level)
                current_lines = [line]
                continue

            if not current_lines:
                current_lines = [line]
            else:
                current_lines.append(line)

        flush_section()
        return chunks

    def _chunk_docling_sections(
        self,
        text: Union[str, dict],
        normalized_page_map: List[Dict[str, Any]],
        source_name: str,
        chunk_size: int,
    ) -> List[Dict[str, Any]]:
        docling_items = self._collect_docling_items(text, normalized_page_map)
        if not docling_items:
            return self._chunk_by_titles(normalized_page_map, source_name, chunk_size)

        sections = self._build_docling_sections(docling_items)
        if not sections:
            return self._chunk_by_titles(normalized_page_map, source_name, chunk_size)

        chunks: List[Dict[str, Any]] = []
        for section in sections:
            section_chunks = self._split_docling_section(section, chunk_size)
            section_part_count = len(section_chunks)
            if not section_part_count:
                continue

            for part_index, part in enumerate(section_chunks, start=1):
                part_text = part.strip()
                if not part_text:
                    continue

                chunks.append(
                    {
                        "content": part_text,
                        "metadata": self._build_chunk_metadata(
                            source=source_name,
                            page_start=int(section["page_start"]),
                            page_end=int(section["page_end"]),
                            chunk_text=part_text,
                            chunk_index=0,
                            total_chunks=0,
                            chunking_method="docling_sections",
                            extra_metadata={
                                "chunk_type": "text",
                                "section_title": str(section["title"]),
                                "section_level": int(section["level"]),
                                "section_path": str(section["path"]),
                                "section_part_index": part_index,
                                "section_part_count": section_part_count,
                            },
                        ),
                    }
                )

        asset_chunks = self._build_docling_asset_chunks(
            text=text,
            normalized_page_map=normalized_page_map,
            source_name=source_name,
            sections=sections,
        )
        return chunks + asset_chunks

    def _build_docling_asset_chunks(
        self,
        text: Union[str, dict],
        normalized_page_map: List[Dict[str, Any]],
        source_name: str,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        asset_chunks: List[Dict[str, Any]] = []
        assets = self._collect_docling_assets(text=text, normalized_page_map=normalized_page_map)
        for asset in assets:
            asset_kind = str(asset.get("asset_kind", "") or "").strip().lower()
            chunk_type = "figure" if asset_kind == "picture" else ("table" if asset_kind == "table" else "text")
            if chunk_type == "text":
                continue

            page_start = int(asset.get("page_start") or asset.get("page") or 1)
            page_end = int(asset.get("page_end") or page_start)
            section = self._match_docling_asset_section(asset, sections)
            section_title = str(section.get("title", "") or "").strip()
            section_level = int(section.get("level") or 0) if section else 0
            section_path = str(section.get("path", "") or "").strip()
            asset_summary = self._normalize_asset_summary(asset)
            asset_preview_text = self._build_asset_preview_text(asset)
            content = self._build_docling_asset_content(
                asset=asset,
                chunk_type=chunk_type,
                asset_summary=asset_summary,
                asset_preview_text=asset_preview_text,
                section_title=section_title,
                section_path=section_path,
            )
            if not content:
                continue

            extra_metadata = {
                "chunk_type": chunk_type,
                "asset_kind": asset_kind,
                "asset_path": str(asset.get("asset_path", "") or ""),
                "asset_abs_path": str(asset.get("asset_abs_path", "") or ""),
                "asset_summary": asset_summary,
                "asset_preview_text": asset_preview_text,
                "asset_rows": int(asset.get("asset_rows") or 0),
                "asset_columns": int(asset.get("asset_columns") or 0),
                "asset_caption": str(asset.get("caption", "") or asset.get("text", "") or ""),
                "section_title": section_title,
                "section_level": section_level,
                "section_path": section_path,
                "order_index": int(asset.get("order_index") or 0),
            }

            asset_chunks.append(
                {
                    "content": content,
                    "metadata": self._build_chunk_metadata(
                        source=source_name,
                        page_start=page_start,
                        page_end=page_end,
                        chunk_text=content,
                        chunk_index=0,
                        total_chunks=0,
                        chunking_method="docling_assets",
                        extra_metadata=extra_metadata,
                    ),
                }
            )

        return asset_chunks

    def _collect_docling_assets(
        self,
        text: Union[str, dict],
        normalized_page_map: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        seen: set[tuple] = set()

        def append_asset(item: Dict[str, Any]) -> None:
            if not isinstance(item, dict):
                return
            asset_kind = str(item.get("asset_kind", "") or "").strip().lower()
            asset_path = str(item.get("asset_path", "") or "")
            page_start = self._safe_int(item.get("page_start", item.get("page")))
            page_end = self._safe_int(item.get("page_end", page_start))
            order_index = self._safe_int(item.get("order_index", item.get("index")))
            key = (asset_kind, asset_path, page_start, page_end, order_index)
            if key in seen:
                return
            seen.add(key)
            collected.append(dict(item))

        if isinstance(text, dict):
            for key in ("docling_picture_items", "docling_table_items"):
                raw_items = text.get(key)
                if isinstance(raw_items, list):
                    for item in raw_items:
                        append_asset(item)

        if not collected:
            for page in normalized_page_map:
                for key in ("docling_picture_items", "docling_table_items"):
                    raw_items = page.get(key)
                    if isinstance(raw_items, list):
                        for item in raw_items:
                            append_asset(item)

        collected.sort(
            key=lambda item: (
                int(item.get("page_start") or item.get("page") or 0),
                int(item.get("order_index") or 0),
            )
        )
        return collected

    def _match_docling_asset_section(
        self,
        asset: Dict[str, Any],
        sections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not sections:
            return {}

        asset_page = int(asset.get("page_start") or asset.get("page") or 0)
        asset_order = int(asset.get("order_index") or 0)
        best_section: Dict[str, Any] = {}
        best_score: Optional[tuple] = None

        for section in sections:
            section_start = int(section.get("page_start") or 0)
            section_end = int(section.get("page_end") or section_start)
            if section_start <= asset_page <= section_end:
                page_gap = asset_page - section_start
                heading_order = int((section.get("heading_item") or {}).get("order_index") or 0)
                order_gap = abs(asset_order - heading_order)
                score = (0, page_gap, order_gap)
            elif section_end < asset_page:
                score = (1, asset_page - section_end, abs(asset_order - int((section.get("heading_item") or {}).get("order_index") or 0)))
            else:
                score = (2, section_start - asset_page, abs(asset_order - int((section.get("heading_item") or {}).get("order_index") or 0)))

            if best_score is None or score < best_score:
                best_score = score
                best_section = section

        return best_section

    def _normalize_asset_summary(self, asset: Dict[str, Any]) -> str:
        summary = str(asset.get("asset_summary", "") or "").strip()
        if summary:
            return re.sub(r"\s+", " ", summary).strip()
        caption = str(asset.get("caption", "") or asset.get("text", "") or "").strip()
        return re.sub(r"\s+", " ", caption).strip()

    def _build_asset_preview_text(self, asset: Dict[str, Any], max_rows: int = 5) -> str:
        asset_kind = str(asset.get("asset_kind", "") or "").strip().lower()
        if asset_kind != "table":
            return self._normalize_asset_summary(asset)

        preview = asset.get("asset_preview")
        if not isinstance(preview, list) or not preview:
            return self._normalize_asset_summary(asset)

        preview_lines: List[str] = []
        for row in preview[:max_rows]:
            if not isinstance(row, dict):
                continue
            parts = []
            for key, value in row.items():
                key_text = re.sub(r"\s+", " ", str(key or "")).strip()
                value_text = re.sub(r"\s+", " ", str(value or "")).strip()
                if key_text or value_text:
                    parts.append(f"{key_text}: {value_text}".strip(": "))
            if parts:
                preview_lines.append("; ".join(parts))

        return " | ".join(preview_lines).strip()

    def _build_docling_asset_content(
        self,
        asset: Dict[str, Any],
        chunk_type: str,
        asset_summary: str,
        asset_preview_text: str,
        section_title: str,
        section_path: str,
    ) -> str:
        page_text = f"page {int(asset.get('page_start') or asset.get('page') or 1)}"
        anchor_parts = [part for part in [section_title, section_path, page_text] if part]
        anchor_text = " | ".join(self._dedupe_text_units(anchor_parts))

        if chunk_type == "figure":
            parts = [
                "Figure evidence",
                asset_summary,
                asset_preview_text if asset_preview_text and asset_preview_text != asset_summary else "",
                anchor_text,
            ]
            return "\n".join(part for part in parts if part).strip()

        if chunk_type == "table":
            parts = [
                "Table evidence",
                asset_summary,
                asset_preview_text,
                anchor_text,
            ]
            return "\n".join(part for part in parts if part).strip()

        return ""

    def _dedupe_text_units(self, units: List[str]) -> List[str]:
        deduped: List[str] = []
        seen: set[str] = set()
        for unit in units:
            normalized = re.sub(r"\s+", " ", str(unit or "")).strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            deduped.append(normalized)
        return deduped

    def _collect_docling_items(
        self,
        text: Union[str, dict],
        normalized_page_map: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        seen: set[tuple] = set()

        def append_item(item: Dict[str, Any]) -> None:
            normalized = self._normalize_docling_item(item)
            if not normalized:
                return
            parent = str(normalized.get("parent", "") or "").strip().lower()
            if self._docling_parent_is_visual_context(parent):
                return
            if str(normalized.get("node_kind", "") or "").strip().lower() in {"visual_text", "noise"}:
                return
            key = (
                normalized.get("label", ""),
                normalized.get("text", ""),
                normalized.get("page_start"),
                normalized.get("page_end"),
                normalized.get("order_index"),
            )
            if key in seen:
                return
            seen.add(key)
            collected.append(normalized)

        if isinstance(text, dict):
            raw_items = text.get("docling_text_items")
            if not isinstance(raw_items, list):
                raw_items = text.get("docling_items")
            if isinstance(raw_items, list):
                for item in raw_items:
                    append_item(item)

            structure = text.get("docling_structure")
            if isinstance(structure, dict):
                structured_items = structure.get("items")
                if isinstance(structured_items, list) and not collected:
                    for item in structured_items:
                        append_item(item)

        if not collected:
            for page in normalized_page_map:
                page_items = page.get("docling_text_items")
                if not isinstance(page_items, list):
                    page_items = page.get("docling_items")
                if isinstance(page_items, list):
                    for item in page_items:
                        append_item(item)

        collected.sort(
            key=lambda item: (
                int(item.get("page_start") or item.get("page") or 0),
                int(item.get("order_index") or 0),
            )
        )
        return collected

    def _normalize_docling_item(self, item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        text = str(item.get("text", "") or item.get("content", "") or item.get("orig", "") or "").strip()
        if not text:
            return None

        label = str(item.get("label", "") or "").strip()
        role = str(item.get("role", "") or "").strip().lower()
        node_kind = str(item.get("node_kind", "") or "").strip().lower()
        heading_level = self._safe_int(item.get("heading_level", item.get("level")))
        content_layer = str(item.get("content_layer", "") or "").strip()
        parent = str(item.get("parent", "") or "").strip().lower()
        page_start = self._safe_int(item.get("page_start", item.get("page", item.get("page_number"))))
        page_end = self._safe_int(item.get("page_end", page_start))
        order_index = self._safe_int(item.get("order_index", item.get("line_no", item.get("index"))))
        heading_candidate = self._is_docling_heading_candidate(text, heading_level)
        caption_labels = {"caption", "table_caption", "figure_caption"}

        if label in caption_labels:
            node_kind = "caption"
        elif node_kind == "title":
            node_kind = "title"
        elif node_kind == "section_header":
            node_kind = "section_header" if heading_candidate else "paragraph"
        elif role == "title":
            node_kind = "title"
        elif role == "heading":
            node_kind = "section_header" if heading_candidate else "paragraph"
        elif heading_level is not None:
            node_kind = "section_header" if heading_candidate else "paragraph"
        elif heading_candidate:
            node_kind = "section_header"
        else:
            node_kind = "paragraph"

        if self._docling_parent_is_visual_context(parent) and node_kind == "section_header":
            if label not in caption_labels:
                node_kind = "paragraph"
                heading_level = None

        if not heading_level and node_kind == "section_header":
            heading_level = self._heading_level({"text": text, "role": "heading"})
            if heading_level is None:
                heading_level = 1

        normalized = {
            "text": text,
            "label": label,
            "role": "heading" if node_kind == "section_header" else ("title" if node_kind == "title" else "text"),
            "heading_level": heading_level,
            "content_layer": content_layer,
            "parent": self._serialize_docling_value(item.get("parent")),
            "formatting": self._serialize_docling_value(item.get("formatting")),
            "hyperlink": self._serialize_docling_value(item.get("hyperlink")),
            "prov": self._serialize_docling_value(item.get("prov")),
            "page_start": page_start,
            "page_end": page_end or page_start,
            "page": page_start,
            "page_number": page_start,
            "order_index": order_index if order_index is not None else 0,
            "node_kind": node_kind,
            "is_heading": node_kind == "section_header",
        }
        return normalized

    def _docling_item_is_noise(self, item: Dict[str, Any]) -> bool:
        content_layer = str(item.get("content_layer", "") or "").strip().lower()
        label = str(item.get("label", "") or "").strip().lower()
        node_kind = str(item.get("node_kind", "") or "").strip().lower()

        if node_kind in {"noise", "visual_text"}:
            return True
        if content_layer in {"header", "footer"}:
            return True
        if label in {"page_number", "pageheader", "pagefooter"}:
            return True
        return False

    def _docling_item_is_heading(self, item: Dict[str, Any]) -> bool:
        if self._docling_item_is_noise(item):
            return False

        node_kind = str(item.get("node_kind", "") or "").strip().lower()
        if node_kind == "caption":
            return False
        if node_kind == "section_header":
            return True
        if node_kind == "title":
            return True

        label = str(item.get("label", "") or "").strip().lower()
        if label in {"caption", "table_caption", "figure_caption"}:
            return False
        if label in {"section_header", "sectionheaderitem"}:
            return self._is_docling_heading_candidate(str(item.get("text", "") or ""), self._safe_int(item.get("heading_level", item.get("level"))))
        if label in {"title", "titleitem"}:
            return True

        parent = str(item.get("parent", "") or "").strip().lower()
        if self._docling_parent_is_visual_context(parent):
            return False

        heading_level = item.get("heading_level")
        text = str(item.get("text", "") or "").strip()
        if heading_level is not None and text:
            return self._is_docling_heading_candidate(text, self._safe_int(heading_level))
        return self._is_docling_heading_candidate(text)

    def _docling_heading_level(self, item: Dict[str, Any]) -> int:
        heading_level = self._safe_int(item.get("heading_level"))
        if heading_level is not None and heading_level > 0:
            return heading_level

        text = str(item.get("text", "") or "").strip()
        inferred = self._heading_level({"text": text, "role": "heading"}) if text else None
        if inferred is not None:
            return int(inferred)
        return 1

    def _build_docling_sections(self, docling_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        stack: List[Dict[str, Any]] = []
        current_section: Optional[Dict[str, Any]] = None

        def flush_current_section() -> None:
            nonlocal current_section
            if not current_section:
                return
            if current_section.get("heading_text") or current_section.get("body_items"):
                sections.append(current_section)
            current_section = None

        for item in docling_items:
            if self._docling_item_is_noise(item):
                continue

            if self._docling_item_is_heading(item):
                heading_text = str(item.get("text", "") or "").strip()
                heading_level = self._docling_heading_level(item)

                flush_current_section()

                while stack and int(stack[-1]["level"]) >= heading_level:
                    stack.pop()

                path_titles = [str(entry["title"]) for entry in stack] + [heading_text]
                current_section = {
                    "title": heading_text,
                    "level": heading_level,
                    "path": " > ".join(path_titles),
                    "heading_text": heading_text,
                    "heading_item": item,
                    "body_items": [],
                    "section_items": [item],
                    "page_start": item.get("page_start") or item.get("page") or 1,
                    "page_end": item.get("page_end") or item.get("page_start") or item.get("page") or 1,
                }
                stack.append({"title": heading_text, "level": heading_level})
                continue

            if current_section is None:
                current_section = {
                    "title": "Preamble",
                    "level": 0,
                    "path": "Preamble",
                    "heading_text": "",
                    "heading_item": None,
                    "body_items": [],
                    "section_items": [],
                    "page_start": item.get("page_start") or item.get("page") or 1,
                    "page_end": item.get("page_end") or item.get("page_start") or item.get("page") or 1,
                }

            current_section["body_items"].append(item)
            current_section["section_items"].append(item)
            page_start = item.get("page_start") or item.get("page") or current_section["page_start"]
            page_end = item.get("page_end") or item.get("page_start") or item.get("page") or current_section["page_end"]
            current_section["page_start"] = min(int(current_section["page_start"]), int(page_start))
            current_section["page_end"] = max(int(current_section["page_end"]), int(page_end))

        flush_current_section()
        return [section for section in sections if section.get("heading_text") or section.get("body_items")]

    def _split_docling_section(self, section: Dict[str, Any], chunk_size: int) -> List[str]:
        heading_text = str(section.get("heading_text", "") or "").strip()
        body_items = section.get("body_items", []) or []
        body_units: List[str] = []

        for item in body_items:
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            if item.get("node_kind") in {"list_item", "caption", "code", "formula"}:
                body_units.append(text)
                continue
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
            body_units.extend(paragraphs or [text])

        body_text = self._compose_docling_body_text(body_units)
        if not body_text and heading_text:
            return [heading_text]

        if heading_text:
            combined = f"{heading_text}\n{body_text}".strip() if body_text else heading_text
        else:
            combined = body_text

        if not combined:
            return []

        if len(combined.split()) <= chunk_size:
            return [combined]

        body_chunks = self._pack_text_units(
            body_units or [body_text],
            source_name="",
            page_start=int(section.get("page_start") or 1),
            page_end=int(section.get("page_end") or section.get("page_start") or 1),
            chunk_size=chunk_size,
            chunking_method="docling_sections",
        )
        if not body_chunks:
            return [combined]

        packed = []
        for chunk in body_chunks:
            text = str(chunk.get("content", "") or "").strip()
            if not text:
                continue
            if heading_text and not text.startswith(heading_text):
                text = f"{heading_text}\n{text}".strip()
            packed.append(text)
        return packed or [combined]

    def _compose_docling_body_text(self, units: List[str]) -> str:
        cleaned = [re.sub(r"\s+", " ", str(unit or "")).strip() for unit in units if str(unit or "").strip()]
        if not cleaned:
            return ""
        return "\n\n".join(cleaned).strip()

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _serialize_docling_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._serialize_docling_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_docling_value(item) for item in value]
        if hasattr(value, "__dict__") and value.__dict__:
            return {
                str(key): self._serialize_docling_value(val)
                for key, val in value.__dict__.items()
                if not str(key).startswith("_")
            }
        if hasattr(value, "value"):
            return self._serialize_docling_value(getattr(value, "value"))
        return str(value)

    def _stringify_docling_value(self, value: Any) -> str:
        serialized = self._serialize_docling_value(value)
        if serialized is None:
            return ""
        return str(serialized).strip()

    def _merge_heading_fragments(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not lines:
            return []

        merged: List[Dict[str, Any]] = []
        index = 0
        while index < len(lines):
            current = dict(lines[index])
            next_line = lines[index + 1] if index + 1 < len(lines) else None

            if next_line and current.get("page") == next_line.get("page"):
                current_text = str(current.get("text", "")).strip()
                next_text = str(next_line.get("text", "")).strip()
                if self._is_section_number_only(current_text) and self._looks_like_heading_text(next_text):
                    try:
                        section_number = int(current_text.split(".")[0])
                    except ValueError:
                        section_number = None
                    if section_number is None or section_number > 20:
                        merged.append(current)
                        index += 1
                        continue
                    current["text"] = f"{current_text} {next_text}".strip()
                    current["heading_level"] = 1 if "." not in current_text else None
                    current["font_size"] = max(
                        [value for value in [current.get("font_size"), next_line.get("font_size")] if isinstance(value, (int, float))],
                        default=current.get("font_size"),
                    )
                    merged.append(current)
                    index += 2
                    continue

            merged.append(current)
            index += 1

        return merged

    def _chunk_by_paragraphs(
        self,
        normalized_page_map: List[Dict[str, Any]],
        source_name: str,
        chunk_size: int,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for page in normalized_page_map:
            text = self._page_text(page)
            if not text:
                continue

            page_num = int(page.get("page", page.get("page_number", 1)))
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not paragraphs:
                paragraphs = [text.strip()]

            chunks.extend(
                self._pack_text_units(
                    paragraphs,
                    source_name=source_name,
                    page_start=page_num,
                    page_end=page_num,
                    chunk_size=chunk_size,
                    chunking_method="by_paragraphs",
                )
            )
        return chunks

    def _chunk_by_sentences(
        self,
        normalized_page_map: List[Dict[str, Any]],
        source_name: str,
        chunk_size: int,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for page in normalized_page_map:
            text = self._page_text(page)
            if not text:
                continue

            page_num = int(page.get("page", page.get("page_number", 1)))
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
            if not sentences:
                sentences = [text.strip()]

            chunks.extend(
                self._pack_text_units(
                    sentences,
                    source_name=source_name,
                    page_start=page_num,
                    page_end=page_num,
                    chunk_size=chunk_size,
                    chunking_method="by_sentences",
                )
            )
        return chunks

    def _pack_text_units(
        self,
        units: List[str],
        source_name: str,
        page_start: int,
        page_end: int,
        chunk_size: int,
        chunking_method: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        current_units: List[str] = []

        def flush() -> None:
            nonlocal current_units
            if not current_units:
                return
            content = " ".join(current_units).strip()
            if content:
                chunks.append(
                    {
                        "content": content,
                        "metadata": self._build_chunk_metadata(
                            source=source_name,
                            page_start=page_start,
                            page_end=page_end,
                            chunk_text=content,
                            chunk_index=0,
                            total_chunks=0,
                            chunking_method=chunking_method,
                        ),
                    }
                )
            current_units = []

        for unit in units:
            text = str(unit).strip()
            if not text:
                continue

            if len(text.split()) > chunk_size:
                flush()
                for part in self._split_text_by_words(text, chunk_size):
                    part = part.strip()
                    if not part:
                        continue
                    chunks.append(
                        {
                            "content": part,
                            "metadata": self._build_chunk_metadata(
                                source=source_name,
                                page_start=page_start,
                                page_end=page_end,
                                chunk_text=part,
                                chunk_index=0,
                                total_chunks=0,
                                chunking_method=chunking_method,
                            ),
                        }
                    )
                continue

            candidate = " ".join(current_units + [text]).strip()
            if current_units and len(candidate.split()) > chunk_size:
                flush()
            current_units.append(text)

        flush()
        return chunks

    def _flatten_document_lines(self, normalized_page_map: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        for page in normalized_page_map:
            page_num = int(page.get("page", page.get("page_number", len(lines) + 1)))
            page_lines = page.get("lines")

            if isinstance(page_lines, list) and page_lines:
                ordered_page_lines = list(page_lines)
                for index, item in enumerate(ordered_page_lines):
                    text = str(item.get("text", "")).strip()
                    if not text:
                        continue

                    current = {**item, "page": page_num, "text": text}
                    bbox = current.get("bbox") if isinstance(current.get("bbox"), (list, tuple)) and len(current.get("bbox")) >= 4 else None
                    prev_item = ordered_page_lines[index - 1] if index > 0 else None
                    next_item = ordered_page_lines[index + 1] if index + 1 < len(ordered_page_lines) else None

                    if bbox and prev_item and isinstance(prev_item.get("bbox"), (list, tuple)) and len(prev_item.get("bbox")) >= 4:
                        current["gap_before"] = float(bbox[1]) - float(prev_item["bbox"][3])
                    else:
                        current["gap_before"] = None

                    if bbox and next_item and isinstance(next_item.get("bbox"), (list, tuple)) and len(next_item.get("bbox")) >= 4:
                        current["gap_after"] = float(next_item["bbox"][1]) - float(bbox[3])
                    else:
                        current["gap_after"] = None

                    lines.append(current)
                continue

            raw_text = self._page_text(page)
            for line_no, text in enumerate(raw_text.split("\n"), start=1):
                text = text.strip()
                if not text:
                    continue
                lines.append(
                    {
                        "page": page_num,
                        "text": text,
                        "font_size": None,
                        "role": None,
                        "heading_level": None,
                        "line_no": line_no,
                    }
                )

        return lines

    def _compose_text_from_lines(self, lines: List[Dict[str, Any]]) -> str:
        if not lines:
            return ""

        parts: List[str] = []
        for line in lines:
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            if not parts:
                parts.append(text)
                continue
            if parts[-1].endswith("-") and text and text[0].isalnum():
                parts[-1] = parts[-1][:-1] + text
            else:
                parts.append(text)
        return " ".join(parts).strip()

    def _heading_level(self, line: Dict[str, Any]) -> Optional[int]:
        text = str(line.get("text", "")).strip()
        if not text:
            return None

        if line.get("role") == "heading":
            return int(line.get("heading_level") or 1)

        if line.get("heading_level") is not None:
            try:
                return int(line["heading_level"])
            except (TypeError, ValueError):
                return 1

        if text in {"Abstract", "References"}:
            return 1

        numbered_match = re.match(r"^([1-9]\d*(?:\.\d+)*)\s+([A-Z].{0,140})$", text)
        if numbered_match and len(text) <= 160:
            prefix = numbered_match.group(1)
            try:
                leading_number = int(prefix.split(".")[0])
            except ValueError:
                return None
            if leading_number > 20:
                return None
            # Numbered top-level sections such as "1 Introduction" remain level 1.
            # Numbered subsections such as "2.1 ..." are treated as level 2.
            if "." in prefix:
                return 2
            return 1

        words = text.split()
        if len(words) > 6 or len(text) > 80:
            return None
        if text.endswith((".", ",", ";", ":")):
            return None
        if any(ch.isdigit() for ch in text):
            return None
        if "," in text or "/" in text:
            return None
        if not (text[0].isupper() or text.isupper()):
            return None

        bbox = line.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            line_height = float(bbox[3]) - float(bbox[1])
        else:
            line_height = None

        gap_before = line.get("gap_before")
        gap_after = line.get("gap_after")
        if not (
            isinstance(line_height, (int, float))
            and line_height > 0
            and isinstance(gap_before, (int, float))
            and isinstance(gap_after, (int, float))
            and gap_before >= line_height * 0.75
            and gap_after >= line_height * 0.6
        ):
            return None

        font_size = line.get("font_size")
        if not (isinstance(font_size, (int, float)) and float(font_size) >= 10.0):
            return None

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            if float(bbox[0]) > 85.0:
                return None

        if text.isupper() and len(words) <= 6:
            return 1

        if len(words) <= 4:
            return 1

        return None

    def _looks_like_heading_text(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        if len(normalized) > 80:
            return False
        if ":" in normalized:
            return False
        if normalized.endswith((".", ",", ";")):
            return False
        if any(ch.isdigit() for ch in normalized):
            return False
        if "," in normalized or "/" in normalized:
            return False
        words = normalized.split()
        if len(words) > 5:
            return False
        if len(words) == 1 and normalized.isupper() and len(normalized) <= 4:
            return False
        return normalized[0].isupper() or normalized.isupper()

    def _is_docling_numbered_heading_text(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        return bool(
            re.fullmatch(
                r"(?:\d+(?:\.\d+)*|[IVXLCM]+\.?|[A-Z]\.?)\s+[A-Z].{0,160}",
                normalized,
            )
        )

    def _is_docling_heading_candidate(self, text: str, heading_level: Optional[int] = None) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False

        if self._is_docling_numbered_heading_text(normalized):
            return True

        if normalized.lower() in {"abstract", "references", "acknowledgements", "acknowledgments"}:
            return True

        if self._looks_like_heading_text(normalized):
            return True

        if heading_level is not None and heading_level > 0 and len(normalized) <= 80:
            return self._looks_like_heading_text(normalized)

        return False

    def _docling_parent_is_visual_context(self, parent: str) -> bool:
        normalized = str(parent or "").strip().lower()
        if not normalized:
            return False
        return any(token in normalized for token in ("#/pictures/", "#/figures/", "#/tables/"))

    def _is_section_number_only(self, text: str) -> bool:
        normalized = str(text or "").strip()
        return bool(re.match(r"^[1-9]\d*(?:\.\d+)*$", normalized))

    def _page_text(self, page: Dict[str, Any]) -> str:
        text = page.get("text") or page.get("content") or ""
        return str(text).strip()

    def _split_text_by_words(self, text: str, chunk_size: int) -> List[str]:
        words = str(text or "").split()
        if not words:
            return []
        if len(words) <= chunk_size:
            return [" ".join(words)]

        chunks: List[str] = []
        current: List[str] = []
        for word in words:
            if current and len(current) + 1 > chunk_size:
                chunks.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            chunks.append(" ".join(current))
        return chunks

    def _split_text_by_sentence_chunks(self, text: str, max_words: int) -> List[str]:
        normalized = re.sub(r"\s+", " ", str(text or "").replace("\r\n", "\n")).strip()
        if not normalized:
            return []

        sentences = self._split_into_sentences(normalized)
        if not sentences:
            return [normalized]

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_words = 0

        def flush() -> None:
            nonlocal current_sentences, current_words
            if not current_sentences:
                return
            chunk = " ".join(current_sentences).strip()
            if chunk:
                chunks.append(chunk)
            current_sentences = []
            current_words = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_words = self._word_count(sentence)
            if sentence_words > max_words:
                flush()
                chunks.extend(self._split_text_by_words(sentence, max_words))
                continue

            if current_sentences and current_words + sentence_words > max_words:
                flush()

            current_sentences.append(sentence)
            current_words += sentence_words

        flush()
        return chunks or [normalized]

    def _split_into_sentences(self, text: str) -> List[str]:
        normalized = str(text or "").strip()
        if not normalized:
            return []
        sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？；;])\s+", normalized) if part.strip()]
        return sentences or [normalized]

    def _word_count(self, text: str) -> int:
        return len(str(text or "").split())
