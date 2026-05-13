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
    Load PDF documents with PyMuPDF and keep the page structure intact.
    """

    def __init__(self):
        self.total_pages = 0
        self.current_page_map: List[Dict[str, Any]] = []

    def load_pdf(
        self,
        file_path: str,
        method: str,
        strategy: str = None,
        chunking_strategy: str = None,
        chunking_options: dict = None,
    ) -> dict:
        """
        Load a PDF document and return page-level structured data.
        """
        if method != "pymupdf":
            raise ValueError("Only pymupdf loading is supported in this loader.")
        return self._load_with_pymupdf(file_path)

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

            document_data = {
                "filename": str(filename),
                "total_chunks": int(len(chunks)),
                "total_pages": int(metadata.get("total_pages", self.total_pages or 1)),
                "loading_method": str(loading_method),
                "loading_strategy": str(strategy) if loading_method == "unstructured" and strategy else None,
                "chunking_strategy": str(chunking_strategy) if loading_method == "unstructured" and chunking_strategy else None,
                "chunking_method": str(chunking_strategy or strategy or "loaded"),
                "timestamp": datetime.now().isoformat(),
                "pages": page_map or self.current_page_map,
                "chunks": chunks,
            }

            os.makedirs("01-loaded-docs", exist_ok=True)
            filepath = os.path.join("01-loaded-docs", f"{doc_name}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
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

    def _extract_pymupdf_page(self, page, page_num: int, filename: str) -> Dict[str, Any]:
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        raw_dict = page.get_text("dict")
        raw_text = page.get_text("text").strip()
        lines = self._extract_pymupdf_lines(raw_dict, page_num, filename, page_width, page_height)

        return self._finalize_page_record(
            page_num=page_num,
            filename=filename,
            raw_text=raw_text,
            lines=lines,
            page_width=page_width,
            page_height=page_height,
        )

    def _extract_pymupdf_lines(
        self,
        raw_dict: Dict[str, Any],
        page_num: int,
        filename: str,
        page_width: float,
        page_height: float,
    ) -> List[Dict[str, Any]]:
        lines: List[Dict[str, Any]] = []
        for block_no, block in enumerate(raw_dict.get("blocks", [])):
            if block.get("type") != 0:
                continue

            block_bbox = block.get("bbox") or [0, 0, 0, 0]
            for line_no, line in enumerate(block.get("lines", []), start=1):
                spans = line.get("spans", [])
                text = "".join(str(span.get("text", "")) for span in spans)
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue

                bbox = line.get("bbox") or block_bbox or [0, 0, 0, 0]
                font_sizes = [float(span.get("size", 0) or 0) for span in spans if span.get("size")]
                font_names = [str(span.get("font", "") or "").strip() for span in spans if span.get("font")]

                lines.append(
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

        return lines

    def _finalize_page_record(
        self,
        page_num: int,
        filename: str,
        raw_text: str,
        lines: List[Dict[str, Any]],
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

        final_text = "\n".join(line["text"] for line in normalized_lines).strip() or raw_text
        page_record: Dict[str, Any] = {
            "page": page_num,
            "page_number": page_num,
            "text": final_text,
            "raw_text": raw_text,
            "source": filename,
            "line_count": len(normalized_lines),
            "lines": normalized_lines,
        }

        if page_width is not None:
            page_record["page_width"] = page_width
        if page_height is not None:
            page_record["page_height"] = page_height

        return page_record

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
