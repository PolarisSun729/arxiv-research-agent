import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class LoadingService:
    """
    Load PDF documents with either PyMuPDF or Docling and keep the page structure intact.
    """

    def __init__(self):
        self.total_pages = 0
        self.current_page_map: List[Dict[str, Any]] = []

    def load_pdf(
        self,
        file_path: str,
        method: str = "pymupdf",
        strategy: str = None,
        chunking_strategy: str = None,
        chunking_options: dict = None,
    ) -> dict:
        """
        Load a PDF document and return page-level structured data.
        """
        normalized_method = str(method or "pymupdf").strip().lower()
        if normalized_method == "pymupdf":
            return self._load_with_pymupdf(file_path)
        if normalized_method == "docling":
            return self._load_with_docling(file_path)
        raise ValueError(f"Unsupported PDF loading method: {method}")

    def get_total_pages(self) -> int:
        return self.total_pages or (max(page_data["page"] for page_data in self.current_page_map) if self.current_page_map else 0)

    def get_page_map(self) -> list:
        return self.current_page_map

    def to_full_text(self, document_data: Optional[dict] = None) -> str:
        if document_data and isinstance(document_data, dict):
            pages = document_data.get("pages") or document_data.get("content") or []
        else:
            pages = self.current_page_map

        return "\n\n".join(
            page.get("text", "").strip()
            for page in pages
            if isinstance(page, dict) and page.get("text", "").strip()
        )

    def save_document(
        self,
        filename: str,
        chunks: list,
        metadata: dict,
        loading_method: str,
        strategy: str = None,
        chunking_strategy: str = None,
        page_map: list = None,
        document_data: Optional[dict] = None,
    ) -> str:
        """
        Persist the loaded document, chunk metadata, and page map.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            base_name = filename.replace(".pdf", "").split("_")[0]

            if loading_method == "unstructured" and strategy:
                doc_name = f"{base_name}_{loading_method}_{strategy}_{chunking_strategy}_{timestamp}"
            else:
                doc_name = f"{base_name}_{loading_method}_{timestamp}"

            payload = {
                "filename": str(filename),
                "total_chunks": int(len(chunks)),
                "total_pages": int(metadata.get("total_pages", self.total_pages or 1)),
                "loading_method": str(loading_method),
                "loading_strategy": str(strategy) if strategy else None,
                "chunking_strategy": str(chunking_strategy) if chunking_strategy else None,
                "chunking_method": str(chunking_strategy or strategy or "loaded"),
                "timestamp": datetime.now().isoformat(),
                "pages": page_map or self.current_page_map,
                "chunks": chunks,
            }

            if isinstance(document_data, dict):
                for key, value in document_data.items():
                    if key in {
                        "filename",
                        "total_chunks",
                        "total_pages",
                        "loading_method",
                        "loading_strategy",
                        "chunking_strategy",
                        "chunking_method",
                        "timestamp",
                        "pages",
                        "chunks",
                    }:
                        continue
                    payload[key] = value
                payload["pages"] = page_map or self.current_page_map
                payload["chunks"] = chunks
                payload["document_metadata"] = dict(document_data.get("metadata", {}) or {})
                payload["document_metadata"]["saved_at"] = datetime.now().isoformat()
                payload["document_metadata"]["source_filename"] = str(filename)

            os.makedirs("01-loaded-docs", exist_ok=True)
            filepath = os.path.join("01-loaded-docs", f"{doc_name}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            raise

    def _load_with_pymupdf(self, file_path: str) -> dict:
        pages: List[Dict[str, Any]] = []
        try:
            with fitz.open(file_path) as doc:
                self.total_pages = len(doc)
                for page_num, page in enumerate(doc, start=1):
                    pages.append(self._extract_pymupdf_page(page, page_num, os.path.basename(file_path)))

            self.current_page_map = pages
            return self._build_document_payload(
                filename=os.path.basename(file_path),
                pages=pages,
                loading_method="pymupdf",
                source_path=file_path,
            )
        except Exception as e:
            logger.error(f"PyMuPDF error: {str(e)}")
            raise

    def _load_with_docling(self, file_path: str) -> dict:
        pages: List[Dict[str, Any]] = []
        try:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError as exc:
                raise ImportError(
                    "Docling is not installed. Install the `docling` package to enable this loading method."
                ) from exc

            converter = DocumentConverter()
            result = converter.convert(file_path)
            document = result.document
            docling_assets = self._extract_docling_assets(document, file_path)
            docling_text_items = docling_assets["text_items"]
            docling_picture_items = docling_assets["picture_items"]
            docling_table_items = docling_assets["table_items"]

            docling_text_items_by_page = self._group_docling_items_by_page(docling_text_items)
            docling_picture_items_by_page = self._group_docling_items_by_page(docling_picture_items)
            docling_table_items_by_page = self._group_docling_items_by_page(docling_table_items)

            page_items_obj = getattr(document, "pages", None)
            if isinstance(page_items_obj, dict):
                page_items = list(page_items_obj.values())
            elif isinstance(page_items_obj, (list, tuple)):
                page_items = list(page_items_obj)
            else:
                page_items = list(page_items_obj or [])

            if not page_items:
                fallback_text_items = docling_text_items_by_page.get(1, [])
                fallback_text = self._compose_docling_page_text(fallback_text_items)
                fallback_markdown = self._compose_docling_page_markdown(fallback_text_items)
                page_record = self._finalize_docling_page_record(
                    page_num=1,
                    filename=os.path.basename(file_path),
                    raw_text=fallback_text,
                    markdown_text=fallback_markdown,
                    lines=self._split_docling_lines(fallback_text),
                    docling_text_items=fallback_text_items,
                    docling_picture_items=docling_picture_items_by_page.get(1, []),
                    docling_table_items=docling_table_items_by_page.get(1, []),
                    page_width=None,
                    page_height=None,
                )
                pages = [page_record]
            else:
                self.total_pages = len(page_items)
                for page_num, page_item in enumerate(page_items, start=1):
                    page_number = int(getattr(page_item, "page_no", page_num) or page_num)
                    page_width, page_height = self._extract_docling_page_size(page_item)
                    page_text_items = docling_text_items_by_page.get(page_number, [])
                    raw_text = self._compose_docling_page_text(page_text_items)
                    markdown_text = self._compose_docling_page_markdown(page_text_items)
                    pages.append(
                        self._finalize_docling_page_record(
                            page_num=page_number,
                            filename=os.path.basename(file_path),
                            raw_text=raw_text,
                            markdown_text=markdown_text,
                            lines=self._split_docling_lines(raw_text),
                            docling_text_items=page_text_items,
                            docling_picture_items=docling_picture_items_by_page.get(page_number, []),
                            docling_table_items=docling_table_items_by_page.get(page_number, []),
                            page_width=page_width,
                            page_height=page_height,
                        )
                    )

            self.current_page_map = pages
            full_markdown = self._compose_docling_page_markdown(docling_text_items)
            return self._build_document_payload(
                filename=os.path.basename(file_path),
                pages=pages,
                loading_method="docling",
                source_path=file_path,
                strategy="docling",
                chunking_strategy="docling_sections",
            ) | {
                "document_markdown": full_markdown,
                "document_markdown_length": len(full_markdown or ""),
                "docling_items": docling_text_items,
                "docling_item_count": len(docling_text_items),
                "docling_text_items": docling_text_items,
                "docling_text_item_count": len(docling_text_items),
                "docling_picture_items": docling_picture_items,
                "docling_picture_item_count": len(docling_picture_items),
                "docling_table_items": docling_table_items,
                "docling_table_item_count": len(docling_table_items),
                "docling_asset_root": docling_assets.get("asset_root"),
                "docling_asset_manifest": docling_assets.get("asset_manifest", {}),
            }
        except Exception as e:
            logger.error(f"Docling error: {str(e)}")
            raise

    def _extract_pymupdf_page(self, page, page_num: int, filename: str) -> Dict[str, Any]:
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        raw_dict = page.get_text("dict")
        raw_text = page.get_text("text").strip()
        blocks = self._extract_pymupdf_blocks(raw_dict, page_num, filename, page_width, page_height)
        layout = self._detect_page_layout(blocks, page_width, page_height)
        ordered_blocks = self._order_blocks_by_layout(blocks, layout, page_width, page_height)
        lines = self._flatten_blocks_to_lines(ordered_blocks)

        return self._finalize_page_record(
            page_num=page_num,
            filename=filename,
            raw_text=raw_text,
            lines=lines,
            page_width=page_width,
            page_height=page_height,
            layout_mode=layout["mode"],
            layout_info=layout,
        )

    def _extract_pymupdf_blocks(
        self,
        raw_dict: Dict[str, Any],
        page_num: int,
        filename: str,
        page_width: float,
        page_height: float,
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for block_no, block in enumerate(raw_dict.get("blocks", [])):
            if block.get("type") != 0:
                continue

            block_bbox = block.get("bbox") or [0, 0, 0, 0]
            block_lines: List[Dict[str, Any]] = []
            for line_no, line in enumerate(block.get("lines", []), start=1):
                spans = line.get("spans", [])
                text = "".join(str(span.get("text", "")) for span in spans)
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue

                bbox = line.get("bbox") or block_bbox or [0, 0, 0, 0]
                font_sizes = [float(span.get("size", 0) or 0) for span in spans if span.get("size")]
                font_names = [str(span.get("font", "") or "").strip() for span in spans if span.get("font")]

                block_lines.append(
                    {
                        "text": text,
                        "bbox": list(bbox) if isinstance(bbox, (list, tuple)) else [0, 0, 0, 0],
                        "font_size": max(font_sizes) if font_sizes else None,
                        "font_name": self._pick_dominant_font_name(font_names),
                        "is_bold": any(self._font_name_indicates_bold(font_name) for font_name in font_names),
                        "is_italic": any(self._font_name_indicates_italic(font_name) for font_name in font_names),
                        "block_no": block_no,
                        "line_no": line_no,
                        "span_count": len(spans),
                        "source": filename,
                        "page": page_num,
                        "page_number": page_num,
                    }
                )

            merged_block_lines = self._merge_block_line_fragments(block_lines)
            if not merged_block_lines:
                continue

            block_lines_bbox = self._merge_bboxes([line.get("bbox") for line in merged_block_lines])
            if block_lines_bbox:
                block_bbox_list = list(block_lines_bbox)
            elif isinstance(block_bbox, (list, tuple)):
                block_bbox_list = list(block_bbox)
            else:
                block_bbox_list = [0, 0, 0, 0]
            text = "\n".join(line["text"] for line in merged_block_lines).strip()
            if not text:
                continue

            blocks.append(
                {
                    "block_no": block_no,
                    "bbox": block_bbox_list,
                    "text": text,
                    "lines": merged_block_lines,
                    "line_count": len(merged_block_lines),
                    "page": page_num,
                    "page_number": page_num,
                    "source": filename,
                    "x0": float(block_bbox_list[0]) if len(block_bbox_list) >= 4 else None,
                    "y0": float(block_bbox_list[1]) if len(block_bbox_list) >= 4 else None,
                    "x1": float(block_bbox_list[2]) if len(block_bbox_list) >= 4 else None,
                    "y1": float(block_bbox_list[3]) if len(block_bbox_list) >= 4 else None,
                    "width": float(block_bbox_list[2] - block_bbox_list[0]) if len(block_bbox_list) >= 4 else None,
                    "height": float(block_bbox_list[3] - block_bbox_list[1]) if len(block_bbox_list) >= 4 else None,
                    "center_x": float((block_bbox_list[0] + block_bbox_list[2]) / 2) if len(block_bbox_list) >= 4 else None,
                    "center_y": float((block_bbox_list[1] + block_bbox_list[3]) / 2) if len(block_bbox_list) >= 4 else None,
                }
            )

        return blocks

    def _merge_block_line_fragments(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not lines:
            return []

        ordered_lines = sorted(
            (dict(line) for line in lines),
            key=lambda item: (
                float(item.get("y0", item.get("bbox", [0, 0, 0, 0])[1])),
                float(item.get("x0", item.get("bbox", [0, 0, 0, 0])[0])),
                int(item.get("line_no", 0)),
            ),
        )

        merged_lines: List[Dict[str, Any]] = []
        current_group: List[Dict[str, Any]] = []

        for line in ordered_lines:
            if not current_group:
                current_group = [line]
                continue

            if self._is_same_visual_row(current_group[-1], line):
                current_group.append(line)
                continue

            merged_lines.append(self._merge_line_group(current_group))
            current_group = [line]

        if current_group:
            merged_lines.append(self._merge_line_group(current_group))

        return merged_lines

    def _merge_line_group(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {}

        ordered_group = sorted(
            group,
            key=lambda item: (
                float(item.get("x0", item.get("bbox", [0, 0, 0, 0])[0])),
                int(item.get("line_no", 0)),
            ),
        )
        merged = dict(ordered_group[0])
        merged_text_parts = [str(item.get("text", "")).strip() for item in ordered_group if str(item.get("text", "")).strip()]
        merged["text"] = self._join_text_fragments(merged_text_parts)

        bbox_list = [item.get("bbox") for item in ordered_group]
        merged_bbox = self._merge_bboxes(bbox_list)
        if merged_bbox:
            merged["bbox"] = list(merged_bbox)
            merged["x0"], merged["y0"], merged["x1"], merged["y1"] = map(float, merged["bbox"])

        merged["span_count"] = sum(int(item.get("span_count", 0) or 0) for item in ordered_group)
        merged["line_no"] = int(ordered_group[0].get("line_no", 0) or 0)
        return merged

    def _is_same_visual_row(self, first: Dict[str, Any], second: Dict[str, Any]) -> bool:
        first_bbox = first.get("bbox") if isinstance(first.get("bbox"), (list, tuple)) and len(first.get("bbox")) >= 4 else None
        second_bbox = second.get("bbox") if isinstance(second.get("bbox"), (list, tuple)) and len(second.get("bbox")) >= 4 else None
        if not first_bbox or not second_bbox:
            return False

        first_text = str(first.get("text", "")).strip()
        second_text = str(second.get("text", "")).strip()
        if not first_text or not second_text:
            return False

        first_y0, first_y1 = float(first_bbox[1]), float(first_bbox[3])
        second_y0, second_y1 = float(second_bbox[1]), float(second_bbox[3])
        row_height = max(first_y1 - first_y0, second_y1 - second_y0, 1.0)
        vertical_overlap = min(first_y1, second_y1) - max(first_y0, second_y0)
        if vertical_overlap < 0:
            return False
        if vertical_overlap / row_height < 0.5:
            return False

        first_font = first.get("font_size")
        second_font = second.get("font_size")
        if first_font is not None and second_font is not None and abs(float(first_font) - float(second_font)) > 1.5:
            return False

        horizontal_gap = float(second_bbox[0]) - float(first_bbox[2])
        reverse_gap = float(first_bbox[0]) - float(second_bbox[2])
        if horizontal_gap < -2.0 or reverse_gap < -2.0:
            return False
        if horizontal_gap > 60.0:
            return False

        return True

    def _detect_page_layout(self, blocks: List[Dict[str, Any]], page_width: float, page_height: float) -> Dict[str, Any]:
        text_blocks = [block for block in blocks if str(block.get("text", "")).strip()]
        if not text_blocks:
            return {"mode": "single", "split_x": page_width / 2, "column_count": 1, "confidence": 0.0}

        full_width_threshold = page_width * 0.72
        narrow_blocks = [block for block in text_blocks if float(block.get("width") or 0) < full_width_threshold]
        wide_blocks = [block for block in text_blocks if float(block.get("width") or 0) >= full_width_threshold]

        if len(narrow_blocks) < 4:
            return {
                "mode": "single",
                "split_x": page_width / 2,
                "column_count": 1,
                "confidence": 0.4,
                "full_width_threshold": full_width_threshold,
            }

        centers = sorted(float(block.get("center_x") or 0) for block in narrow_blocks)
        gaps = [(centers[idx + 1] - centers[idx], idx) for idx in range(len(centers) - 1)]
        if not gaps:
            return {
                "mode": "single",
                "split_x": page_width / 2,
                "column_count": 1,
                "confidence": 0.4,
                "full_width_threshold": full_width_threshold,
            }

        largest_gap, gap_index = max(gaps, key=lambda item: item[0])
        split_x = (centers[gap_index] + centers[gap_index + 1]) / 2
        left_count = sum(1 for block in narrow_blocks if float(block.get("center_x") or 0) < split_x)
        right_count = sum(1 for block in narrow_blocks if float(block.get("center_x") or 0) >= split_x)
        min_width_gap = max(page_width * 0.12, 60.0)

        if largest_gap >= min_width_gap and left_count >= 2 and right_count >= 2:
            confidence = min(1.0, (largest_gap / page_width) * 4.0)
            mode = "double" if not wide_blocks else "mixed"
            return {
                "mode": mode,
                "split_x": float(split_x),
                "column_count": 2,
                "confidence": confidence,
                "full_width_threshold": full_width_threshold,
            }

        if wide_blocks and left_count >= 1 and right_count >= 1:
            return {
                "mode": "mixed",
                "split_x": float(split_x),
                "column_count": 2,
                "confidence": 0.55,
                "full_width_threshold": full_width_threshold,
            }

        return {
            "mode": "single",
            "split_x": float(split_x),
            "column_count": 1,
            "confidence": 0.5,
            "full_width_threshold": full_width_threshold,
        }

    def _order_blocks_by_layout(
        self,
        blocks: List[Dict[str, Any]],
        layout: Dict[str, Any],
        page_width: float,
        page_height: float,
    ) -> List[Dict[str, Any]]:
        if not blocks:
            return []

        mode = layout.get("mode", "single")
        if mode == "single":
            return sorted(
                blocks,
                key=lambda block: (
                    float(block.get("y0") if block.get("y0") is not None else block.get("bbox", [0, 0, 0, 0])[1]),
                    float(block.get("x0") if block.get("x0") is not None else block.get("bbox", [0, 0, 0, 0])[0]),
                    int(block.get("block_no", 0)),
                ),
            )

        split_x = float(layout.get("split_x", page_width / 2))
        full_width_threshold = float(layout.get("full_width_threshold", page_width * 0.72))
        body_blocks = [block for block in blocks if float(block.get("width") or 0) < full_width_threshold]
        body_top = min((float(block.get("y0") or 0) for block in body_blocks), default=0.0)
        body_bottom = max((float(block.get("y1") or 0) for block in body_blocks), default=page_height)
        top_band_cutoff = body_top + page_height * 0.12
        bottom_band_cutoff = body_bottom - page_height * 0.12

        def lane_order(block: Dict[str, Any]) -> int:
            width = float(block.get("width") or 0)
            y0 = float(block.get("y0") or 0)
            y1 = float(block.get("y1") or 0)
            center_x = float(block.get("center_x") or split_x)

            if width >= full_width_threshold:
                if y1 <= top_band_cutoff:
                    return 0
                if y0 >= bottom_band_cutoff:
                    return 3
                return 1
            return 1 if center_x < split_x else 2

        return sorted(
            blocks,
            key=lambda block: (
                lane_order(block),
                float(block.get("y0") if block.get("y0") is not None else block.get("bbox", [0, 0, 0, 0])[1]),
                float(block.get("x0") if block.get("x0") is not None else block.get("bbox", [0, 0, 0, 0])[0]),
                int(block.get("block_no", 0)),
            ),
        )

    def _flatten_blocks_to_lines(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        for block in blocks:
            block_lines = block.get("lines")
            if not isinstance(block_lines, list) or not block_lines:
                continue
            for index, line in enumerate(block_lines, start=1):
                normalized_line = dict(line)
                normalized_line["block_no"] = int(block.get("block_no", normalized_line.get("block_no", 0)) or 0)
                normalized_line["block_bbox"] = list(block.get("bbox", normalized_line.get("bbox", [0, 0, 0, 0])))
                normalized_line["block_text"] = block.get("text", "")
                normalized_line["block_line_no"] = index
                lines.append(normalized_line)
        return lines

    def _merge_bboxes(self, bboxes: List[Any]) -> Optional[List[float]]:
        valid = [bbox for bbox in bboxes if isinstance(bbox, (list, tuple)) and len(bbox) >= 4]
        if not valid:
            return None
        return [
            float(min(bbox[0] for bbox in valid)),
            float(min(bbox[1] for bbox in valid)),
            float(max(bbox[2] for bbox in valid)),
            float(max(bbox[3] for bbox in valid)),
        ]

    def _join_text_fragments(self, parts: List[str]) -> str:
        if not parts:
            return ""

        merged = parts[0]
        for part in parts[1:]:
            if not merged:
                merged = part
                continue
            if merged.endswith("-") and part and part[0].isalnum():
                merged = merged[:-1] + part
                continue
            if merged.endswith(("(", "[", "{", '"', "'")):
                merged += part
                continue
            if part[:1] in {".", ",", ";", ":", ")", "]", "}", '"', "'"}:
                merged += part
                continue
            merged += f" {part}"
        return re.sub(r"\s+", " ", merged).strip()

    def _finalize_page_record(
        self,
        page_num: int,
        filename: str,
        raw_text: str,
        lines: List[Dict[str, Any]],
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
        layout_mode: Optional[str] = None,
        layout_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_lines: List[Dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            text = str(line.get("text", "")).strip()
            if not text:
                continue

            normalized_line = dict(line)
            normalized_line["text"] = text
            normalized_line["page"] = int(normalized_line.get("page", page_num))
            normalized_line["page_number"] = int(normalized_line.get("page_number", page_num))
            normalized_line["source"] = filename

            bbox = normalized_line.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                normalized_line["x0"] = float(bbox[0])
                normalized_line["y0"] = float(bbox[1])
                normalized_line["x1"] = float(bbox[2])
                normalized_line["y1"] = float(bbox[3])
            else:
                normalized_line.setdefault("x0", None)
                normalized_line.setdefault("y0", None)
                normalized_line.setdefault("x1", None)
                normalized_line.setdefault("y1", None)

            normalized_line.setdefault("line_no", index)
            normalized_lines.append(normalized_line)

        normalized_lines = self._merge_heading_number_lines(normalized_lines)
        final_text = "\n".join(line["text"] for line in normalized_lines).strip() or raw_text
        page_record: Dict[str, Any] = {
            "page": page_num,
            "page_number": page_num,
            "text": final_text,
            "raw_text": raw_text,
            "source": filename,
            "line_count": len(normalized_lines),
            "lines": normalized_lines,
            "layout_mode": layout_mode or "single",
        }

        if layout_info:
            page_record["layout_info"] = layout_info

        if page_width is not None:
            page_record["page_width"] = page_width
        if page_height is not None:
            page_record["page_height"] = page_height

        return page_record

    def _finalize_docling_page_record(
        self,
        page_num: int,
        filename: str,
        raw_text: str,
        markdown_text: str,
        lines: List[Dict[str, Any]],
        docling_text_items: Optional[List[Dict[str, Any]]] = None,
        docling_picture_items: Optional[List[Dict[str, Any]]] = None,
        docling_table_items: Optional[List[Dict[str, Any]]] = None,
        page_width: Optional[float] = None,
        page_height: Optional[float] = None,
    ) -> Dict[str, Any]:
        normalized_lines: List[Dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            text = str(line.get("text", "")).strip()
            if not text:
                continue

            normalized_line = dict(line)
            normalized_line["text"] = text
            normalized_line["page"] = page_num
            normalized_line["page_number"] = page_num
            normalized_line["source"] = filename
            normalized_line.setdefault("line_no", index)
            normalized_lines.append(normalized_line)

        final_text = "\n".join(line["text"] for line in normalized_lines).strip() or raw_text
        page_record: Dict[str, Any] = {
            "page": page_num,
            "page_number": page_num,
            "text": final_text,
            "raw_text": raw_text,
            "markdown": markdown_text,
            "source": filename,
            "line_count": len(normalized_lines),
            "lines": normalized_lines,
            "layout_mode": "docling",
            "docling_items": docling_text_items or [],
            "docling_text_items": docling_text_items or [],
            "docling_picture_items": docling_picture_items or [],
            "docling_table_items": docling_table_items or [],
        }

        if page_width is not None:
            page_record["page_width"] = page_width
        if page_height is not None:
            page_record["page_height"] = page_height

        return page_record

    def _safe_docling_export(self, document: Any, export_type: str, page_no: Optional[int] = None) -> str:
        if document is None:
            return ""

        export_kwargs = {"traverse_pictures": True}
        if page_no is not None:
            export_kwargs["page_no"] = page_no

        try:
            if export_type == "text":
                return str(document.export_to_text(**export_kwargs)).strip()
            if export_type == "markdown":
                return str(document.export_to_markdown(**export_kwargs)).strip()
        except Exception as exc:
            logger.warning("Docling %s export failed for page %s: %s", export_type, page_no, exc)
        return ""

    def _extract_docling_assets(self, document: Any, source_path: str) -> Dict[str, Any]:
        asset_root = self._build_docling_asset_root(source_path)
        text_items = self._extract_docling_text_items(document)
        picture_items = self._extract_docling_picture_items(document, asset_root, source_path)
        table_items = self._extract_docling_table_items(document, asset_root, source_path)
        return {
            "text_items": text_items,
            "picture_items": picture_items,
            "table_items": table_items,
            "asset_root": asset_root,
            "asset_manifest": {
                "picture_count": len(picture_items),
                "table_count": len(table_items),
            },
        }

    def _extract_docling_text_items(self, document: Any) -> List[Dict[str, Any]]:
        raw_items = list(getattr(document, "texts", []) or [])
        normalized: List[Dict[str, Any]] = []

        for index, item in enumerate(raw_items, start=1):
            normalized_item = self._normalize_docling_text_item(item, index)
            if normalized_item and normalized_item.get("text") and normalized_item.get("node_kind") not in {"visual_text", "noise"}:
                normalized.append(normalized_item)

        return normalized

    def _extract_docling_picture_items(self, document: Any, asset_root: str, source_path: str) -> List[Dict[str, Any]]:
        raw_items = list(getattr(document, "pictures", []) or [])
        normalized: List[Dict[str, Any]] = []

        for index, item in enumerate(raw_items, start=1):
            normalized_item = self._normalize_docling_asset_item(item, index, asset_kind="picture")
            if normalized_item:
                export_info = self._export_docling_picture_asset(document, item, asset_root, source_path, index)
                if export_info:
                    normalized_item.update(export_info)
                normalized.append(normalized_item)

        return normalized

    def _extract_docling_table_items(self, document: Any, asset_root: str, source_path: str) -> List[Dict[str, Any]]:
        raw_items = list(getattr(document, "tables", []) or [])
        normalized: List[Dict[str, Any]] = []

        for index, item in enumerate(raw_items, start=1):
            normalized_item = self._normalize_docling_asset_item(item, index, asset_kind="table")
            if normalized_item:
                export_info = self._export_docling_table_asset(document, item, asset_root, source_path, index)
                if export_info:
                    normalized_item.update(export_info)
                normalized.append(normalized_item)

        return normalized

    def _normalize_docling_text_item(self, item: Any, order_index: int) -> Optional[Dict[str, Any]]:
        if item is None:
            return None

        label = self._stringify_docling_value(getattr(item, "label", None))
        text = self._stringify_docling_value(getattr(item, "text", None) or getattr(item, "orig", None) or "")
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if not text:
            return None

        level_value = getattr(item, "level", None)
        level = self._safe_int(level_value)
        content_layer = self._stringify_docling_value(getattr(item, "content_layer", None))
        formatting = self._serialize_docling_value(getattr(item, "formatting", None))
        hyperlink = self._serialize_docling_value(getattr(item, "hyperlink", None))
        provenance = self._normalize_docling_provenance(getattr(item, "prov", None))
        page_numbers = [entry.get("page_no") for entry in provenance if isinstance(entry.get("page_no"), int)]
        if not page_numbers:
            page_no = self._safe_int(getattr(item, "page_no", None) or getattr(item, "page_number", None))
            if page_no is not None:
                page_numbers = [page_no]

        page_start = min(page_numbers) if page_numbers else None
        page_end = max(page_numbers) if page_numbers else page_start

        parent = self._stringify_docling_value(getattr(item, "parent", None))
        node_kind = self._classify_docling_item_kind(label, text, level, content_layer, parent)

        normalized: Dict[str, Any] = {
            "text": text,
            "label": label,
            "role": "heading" if node_kind == "section_header" else ("title" if node_kind == "title" else "text"),
            "heading_level": level if node_kind == "section_header" and level is not None else (1 if node_kind == "section_header" else None),
            "content_layer": content_layer,
            "formatting": formatting,
            "hyperlink": hyperlink,
            "prov": provenance,
            "page_start": page_start,
            "page_end": page_end,
            "page_number": page_start,
            "page": page_start,
            "order_index": int(order_index),
            "node_kind": node_kind,
            "is_heading": node_kind in {"section_header", "title"},
        }

        if hasattr(item, "id"):
            normalized["id"] = self._stringify_docling_value(getattr(item, "id", None))
        if hasattr(item, "parent"):
            normalized["parent"] = self._stringify_docling_value(getattr(item, "parent", None))
        if hasattr(item, "children"):
            normalized["children"] = self._serialize_docling_value(getattr(item, "children", None))
        if hasattr(item, "orig"):
            normalized["orig"] = self._stringify_docling_value(getattr(item, "orig", None))
        if hasattr(item, "url"):
            normalized["url"] = self._stringify_docling_value(getattr(item, "url", None))

        return normalized

    def _normalize_docling_asset_item(self, item: Any, order_index: int, asset_kind: str) -> Optional[Dict[str, Any]]:
        if item is None:
            return None

        label = self._stringify_docling_value(getattr(item, "label", None))
        text = self._stringify_docling_value(getattr(item, "text", None) or getattr(item, "orig", None) or getattr(item, "caption", None) or "")
        content_layer = self._stringify_docling_value(getattr(item, "content_layer", None))
        provenance = self._normalize_docling_provenance(getattr(item, "prov", None))
        page_numbers = [entry.get("page_no") for entry in provenance if isinstance(entry.get("page_no"), int)]
        if not page_numbers:
            page_no = self._safe_int(getattr(item, "page_no", None) or getattr(item, "page_number", None))
            if page_no is not None:
                page_numbers = [page_no]

        page_start = min(page_numbers) if page_numbers else None
        page_end = max(page_numbers) if page_numbers else page_start

        normalized: Dict[str, Any] = {
            "asset_kind": asset_kind,
            "text": text,
            "label": label,
            "content_layer": content_layer,
            "prov": provenance,
            "page_start": page_start,
            "page_end": page_end,
            "page_number": page_start,
            "page": page_start,
            "order_index": int(order_index),
            "parent": self._stringify_docling_value(getattr(item, "parent", None)),
            "orig": self._stringify_docling_value(getattr(item, "orig", None)),
            "is_heading": False,
        }

        for attr in ("id", "name", "url", "caption", "bbox", "image", "rows", "cols", "num_rows", "num_cols", "row_count", "column_count"):
            if hasattr(item, attr):
                value = getattr(item, attr, None)
                if attr == "image":
                    continue
                normalized[attr] = self._serialize_docling_value(value)

        return normalized

    def _build_docling_asset_root(self, source_path: str) -> str:
        source_name = os.path.basename(source_path or "") or "document"
        paper_id = os.path.splitext(source_name)[0].split("_")[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        asset_root = os.path.join("03-docling-assets", paper_id, timestamp)
        os.makedirs(asset_root, exist_ok=True)
        return asset_root

    def _export_docling_picture_asset(
        self,
        document: Any,
        picture_item: Any,
        asset_root: str,
        source_path: str,
        order_index: int,
    ) -> Dict[str, Any]:
        picture_dir = os.path.join(asset_root, "pictures")
        os.makedirs(picture_dir, exist_ok=True)

        output_name = f"picture-{order_index:03d}.png"
        output_path = os.path.join(picture_dir, output_name)

        image_obj = None
        get_image = getattr(picture_item, "get_image", None)
        if callable(get_image):
            for args in ((document,), tuple()):
                try:
                    image_obj = get_image(*args)
                    if image_obj is not None:
                        break
                except TypeError:
                    continue
                except Exception as exc:
                    logger.warning("Docling picture export failed for %s: %s", source_path, exc)
                    return {}

        if image_obj is None:
            return {}

        try:
            if hasattr(image_obj, "save"):
                image_obj.save(output_path)
            elif isinstance(image_obj, (bytes, bytearray)):
                with open(output_path, "wb") as f:
                    f.write(image_obj)
            else:
                with open(output_path, "wb") as f:
                    f.write(bytes(image_obj))
        except Exception as exc:
            logger.warning("Failed to write docling picture asset for %s: %s", source_path, exc)
            return {}

        size = getattr(image_obj, "size", None)
        width = height = None
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            width, height = size[0], size[1]

        rel_path = os.path.relpath(output_path, start=os.getcwd())
        caption = self._stringify_docling_value(getattr(picture_item, "caption", None) or getattr(picture_item, "text", None) or getattr(picture_item, "orig", None))
        summary = caption or f"picture {order_index}"
        if width and height:
            summary = f"{summary} ({width}x{height})"

        return {
            "asset_path": rel_path,
            "asset_abs_path": os.path.abspath(output_path),
            "asset_summary": summary,
            "asset_file_name": output_name,
            "asset_size": [width, height] if width and height else None,
        }

    def _export_docling_table_asset(
        self,
        document: Any,
        table_item: Any,
        asset_root: str,
        source_path: str,
        order_index: int,
    ) -> Dict[str, Any]:
        table_dir = os.path.join(asset_root, "tables")
        os.makedirs(table_dir, exist_ok=True)

        output_base = f"table-{order_index:03d}"
        csv_path = os.path.join(table_dir, f"{output_base}.csv")
        json_path = os.path.join(table_dir, f"{output_base}.json")

        dataframe = None
        export_df = getattr(table_item, "export_to_dataframe", None)
        if callable(export_df):
            for args in ((document,), tuple()):
                try:
                    dataframe = export_df(*args)
                    if dataframe is not None:
                        break
                except TypeError:
                    continue
                except Exception as exc:
                    logger.warning("Docling table dataframe export failed for %s: %s", source_path, exc)
                    return {}

        if dataframe is None:
            return {}

        try:
            if hasattr(dataframe, "to_csv"):
                dataframe.to_csv(csv_path, index=False)
            if hasattr(dataframe, "to_json"):
                dataframe.to_json(json_path, orient="records", force_ascii=False, indent=2)
            else:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self._serialize_docling_value(dataframe), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to write docling table asset for %s: %s", source_path, exc)
            return {}

        rel_csv_path = os.path.relpath(csv_path, start=os.getcwd())
        rel_json_path = os.path.relpath(json_path, start=os.getcwd())

        row_count = None
        column_count = None
        try:
            row_count = int(getattr(dataframe, "shape", [None, None])[0]) if getattr(dataframe, "shape", None) else None
            column_count = int(getattr(dataframe, "shape", [None, None])[1]) if getattr(dataframe, "shape", None) else None
        except Exception:
            row_count = None
            column_count = None

        caption = self._stringify_docling_value(getattr(table_item, "caption", None) or getattr(table_item, "text", None) or getattr(table_item, "orig", None))
        summary = caption or f"table {order_index}"
        if row_count is not None and column_count is not None:
            summary = f"{summary} ({row_count}x{column_count})"

        preview = None
        if hasattr(dataframe, "head"):
            try:
                preview = self._serialize_docling_value(dataframe.head(5).to_dict(orient="records"))
            except Exception:
                preview = None

        return {
            "asset_path": rel_csv_path,
            "asset_json_path": rel_json_path,
            "asset_abs_path": os.path.abspath(csv_path),
            "asset_summary": summary,
            "asset_file_name": os.path.basename(csv_path),
            "asset_rows": row_count,
            "asset_columns": column_count,
            "asset_preview": preview,
        }

    def _classify_docling_item_kind(
        self,
        label: Any,
        text: str,
        level: Optional[int],
        content_layer: Optional[str],
        parent: Optional[str] = None,
    ) -> str:
        normalized_label = str(label or "").strip().lower()
        normalized_text = str(text or "").strip()
        normalized_layer = str(content_layer or "").strip().lower()
        normalized_parent = str(parent or "").strip().lower()

        if normalized_label in {"section_header", "sectionheaderitem"}:
            return "section_header"
        if normalized_label in {"title", "titleitem"}:
            return "title"
        if normalized_layer in {"header", "footer"}:
            return "noise"

        if self._docling_parent_is_visual_context(normalized_parent):
            return "visual_text"

        if normalized_label in {"caption", "table_caption", "figure_caption"}:
            return "caption"
        if normalized_label in {"list_item", "listitem"}:
            return "list_item"
        if normalized_label in {"code", "codeitem"}:
            return "code"
        if normalized_label in {"formula", "formulaitem"}:
            return "formula"

        if level is not None and normalized_text:
            return "section_header"

        if normalized_text and self._looks_like_heading_text(normalized_text):
            return "section_header"

        return "paragraph"

    def _normalize_docling_provenance(self, provenance: Any) -> List[Dict[str, Any]]:
        if provenance is None:
            return []

        entries = provenance if isinstance(provenance, (list, tuple)) else [provenance]
        normalized: List[Dict[str, Any]] = []
        for entry in entries:
            if entry is None:
                continue
            bbox = getattr(entry, "bbox", None)
            if bbox is None and isinstance(entry, dict):
                bbox = entry.get("bbox")

            page_no = getattr(entry, "page_no", None)
            if page_no is None and isinstance(entry, dict):
                page_no = entry.get("page_no")

            normalized.append(
                {
                    "page_no": self._safe_int(page_no),
                    "bbox": self._normalize_bbox(bbox),
                    "char_span": self._serialize_docling_value(getattr(entry, "charspan", None) or getattr(entry, "char_span", None) or (entry.get("char_span") if isinstance(entry, dict) else None)),
                    "source": self._stringify_docling_value(getattr(entry, "source", None) or (entry.get("source") if isinstance(entry, dict) else None)),
                }
            )
        return normalized

    def _normalize_bbox(self, bbox: Any) -> Optional[List[float]]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
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

    def _looks_like_heading_text(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return False
        if len(normalized) > 80:
            return False
        if normalized.endswith((".", ",", ";", ":")):
            return False
        if any(ch.isdigit() for ch in normalized):
            return False
        if "," in normalized or "/" in normalized:
            return False
        words = normalized.split()
        if len(words) > 5:
            return False
        return normalized[0].isupper() or normalized.isupper()

    def _docling_parent_is_visual_context(self, parent: str) -> bool:
        normalized = str(parent or "").strip().lower()
        if not normalized:
            return False
        return any(token in normalized for token in ("#/pictures/", "#/figures/", "#/tables/"))

    def _compose_docling_page_text(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            return ""
        ordered = sorted(
            items,
            key=lambda item: (
                int(item.get("page_start") or item.get("page") or 0),
                int(item.get("order_index") or 0),
            ),
        )
        parts = [str(item.get("text", "") or "").strip() for item in ordered if str(item.get("text", "") or "").strip()]
        return "\n".join(parts).strip()

    def _compose_docling_page_markdown(self, items: List[Dict[str, Any]]) -> str:
        return self._compose_docling_page_text(items)

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _group_docling_items_by_page(self, items: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for item in items:
            page_start = item.get("page_start")
            page_end = item.get("page_end")
            if isinstance(page_start, int) and isinstance(page_end, int) and page_end >= page_start:
                for page_no in range(page_start, page_end + 1):
                    grouped.setdefault(page_no, []).append(item)
                continue
            if isinstance(page_start, int):
                grouped.setdefault(page_start, []).append(item)
        return grouped

    def _split_docling_lines(self, text: str) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        for index, raw_line in enumerate(str(text or "").splitlines(), start=1):
            line = re.sub(r"\s+", " ", str(raw_line)).strip()
            if not line:
                continue
            lines.append({"text": line, "line_no": index})
        return lines

    def _extract_docling_page_size(self, page_item: Any) -> tuple[Optional[float], Optional[float]]:
        size = getattr(page_item, "size", None)
        if size is None:
            return None, None

        width = getattr(size, "width", None)
        height = getattr(size, "height", None)

        if width is None and isinstance(size, (list, tuple)) and len(size) >= 2:
            width = size[0]
            height = size[1]

        try:
            return (float(width) if width is not None else None, float(height) if height is not None else None)
        except (TypeError, ValueError):
            return None, None

    def _merge_heading_number_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(lines) < 2:
            return lines

        merged_lines: List[Dict[str, Any]] = []
        index = 0
        while index < len(lines):
            current = dict(lines[index])
            if index + 1 >= len(lines):
                merged_lines.append(current)
                break

            nxt = dict(lines[index + 1])
            if self._should_merge_heading_number_pair(current, nxt):
                merged = dict(current)
                merged["text"] = f"{str(current.get('text', '')).strip()} {str(nxt.get('text', '')).strip()}".strip()
                merged_bbox = self._merge_bboxes([current.get("bbox"), nxt.get("bbox")])
                if merged_bbox:
                    merged["bbox"] = list(merged_bbox)
                    merged["x0"], merged["y0"], merged["x1"], merged["y1"] = map(float, merged["bbox"])
                merged["span_count"] = int(current.get("span_count", 0) or 0) + int(nxt.get("span_count", 0) or 0)
                merged_lines.append(merged)
                index += 2
                continue

            merged_lines.append(current)
            index += 1

        for line_no, line in enumerate(merged_lines, start=1):
            line["line_no"] = line_no
        return merged_lines

    def _should_merge_heading_number_pair(self, current: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
        current_text = str(current.get("text", "")).strip()
        next_text = str(nxt.get("text", "")).strip()
        if not current_text or not next_text:
            return False

        is_section_number = bool(
            re.fullmatch(r"\d+(?:\.\d+)*", current_text)
            or re.fullmatch(r"(?:[IVXLCM]+)\.?", current_text, flags=re.IGNORECASE)
            or re.fullmatch(r"[A-Z]\.?", current_text)
        )
        if not is_section_number:
            return False
        if not re.match(r"^[A-Za-z][A-Za-z0-9\s\-:,()]{0,120}$", next_text):
            return False

        current_bbox = current.get("bbox")
        next_bbox = nxt.get("bbox")
        if not (
            isinstance(current_bbox, (list, tuple))
            and len(current_bbox) >= 4
            and isinstance(next_bbox, (list, tuple))
            and len(next_bbox) >= 4
        ):
            return False

        current_x0, current_y0, current_x1, current_y1 = map(float, current_bbox[:4])
        next_x0, next_y0, next_x1, next_y1 = map(float, next_bbox[:4])
        current_h = max(current_y1 - current_y0, 1.0)
        next_h = max(next_y1 - next_y0, 1.0)

        x_aligned = abs(current_x0 - next_x0) <= 24.0
        tight_vertical_gap = 0 <= (next_y0 - current_y1) <= max(current_h, next_h) * 1.6
        same_column = abs((current_x0 + current_x1) / 2 - (next_x0 + next_x1) / 2) <= 80.0
        is_not_too_long = len(next_text.split()) <= 8

        return x_aligned and tight_vertical_gap and same_column and is_not_too_long

    def _build_document_payload(
        self,
        filename: str,
        pages: List[Dict[str, Any]],
        loading_method: str,
        source_path: Optional[str] = None,
        strategy: str = None,
        chunking_strategy: str = None,
    ) -> dict:
        normalized_pages = []
        for page in pages:
            page_number = int(page.get("page", page.get("page_number", len(normalized_pages) + 1)))
            normalized_pages.append(
                {
                    **page,
                    "page": page_number,
                    "page_number": page_number,
                    "source": filename,
                }
            )

        self.current_page_map = normalized_pages
        self.total_pages = len(normalized_pages)

        return {
            "filename": filename,
            "metadata": {
                "filename": filename,
                "source_path": source_path,
                "loading_method": loading_method,
                "loading_strategy": strategy,
                "chunking_strategy": chunking_strategy,
                "total_pages": self.total_pages,
                "timestamp": datetime.now().isoformat(),
            },
            "source_path": source_path,
            "loading_method": loading_method,
            "loading_strategy": strategy,
            "chunking_strategy": chunking_strategy,
            "total_pages": self.total_pages,
            "pages": normalized_pages,
            "text": self.to_full_text({"pages": normalized_pages}),
            "timestamp": datetime.now().isoformat(),
        }

    def _pick_dominant_font_name(self, font_names: List[str]) -> Optional[str]:
        for font_name in font_names:
            if font_name:
                return font_name
        return None

    def _font_name_indicates_bold(self, font_name: Optional[str]) -> bool:
        if not font_name:
            return False
        return bool(re.search(r"(bold|black|heavy|semibold|demi|medium)", font_name, re.IGNORECASE))

    def _font_name_indicates_italic(self, font_name: Optional[str]) -> bool:
        if not font_name:
            return False
        return bool(re.search(r"(italic|oblique|ital)", font_name, re.IGNORECASE))
