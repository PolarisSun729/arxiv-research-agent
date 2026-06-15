"""表格结构化服务模块。

该模块负责在 PDF 解析与 chunk 构建完成之后，把每个表格 chunk 从“文本片段”
升级为“结构化证据对象”。它不参与召回链路，只在建索引阶段额外生成一份带
列、行、单元格、数值、单位与来源信息的结构化表格数据，方便后续精确表格问答
（例如“哪个方法最高”“某个指标是多少”“去掉某个模块后下降多少”）复用。

设计要点：
1. 优先从 Docling 已经导出的结构化表格（asset_json_path 指向的完整行记录）抽取，
   其次回退到 chunk/资产里保留的预览记录，最后才尝试保守的文本解析。
2. 解析失败时只跳过单个表格并计数，绝不影响整篇论文建索引。
3. 每个结构化表格对象都保留稳定 table_id，并能追溯到 chunk、页码和章节锚点。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 列名里出现这些关键词时，可以较有把握地推断单元格的指标单位。
_METRIC_UNIT_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("accuracy", "accuracy"),
    ("acc.", "accuracy"),
    ("f1", "f1"),
    ("bleu", "bleu"),
    ("rouge", "rouge"),
    ("precision", "precision"),
    ("prec.", "precision"),
    ("recall", "recall"),
    ("em", "exact_match"),
)

# 单元格里可识别的显式单位符号。
_EXPLICIT_UNIT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("%", "percent"),
)

# 匹配单元格中第一个数值（支持负号、小数、科学计数法、千分位）。
_NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class TableStructureService:
    """把表格 chunk 升级为结构化表格证据对象。

    该服务是无状态的：每次调用 ``build_structured_tables`` 都根据传入的
    document 与 chunks 重新计算，便于在建索引链路里安全复用。
    """

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """初始化服务。

        参数:
            workspace_root (Optional[str]): 用于解析相对资产路径的工作目录根。
                为空时使用当前工作目录，与建索引链路保持一致。
        """
        self._workspace_root = workspace_root

    def build_structured_tables(
        self,
        arxiv_id: str,
        document: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """为所有表格 chunk 生成结构化表格对象，并回填 table_id 关联。

        参数:
            arxiv_id (str): 论文标识，用于构造稳定 table_id。
            document (Dict[str, Any]): 加载阶段产出的文档对象，含 docling_table_items。
            chunks (List[Dict[str, Any]]): 切分阶段产出的 chunk 列表（会被原地补充 table_id）。

        返回:
            Dict[str, Any]: 包含 ``structured_tables`` 列表与 ``debug`` 统计的结果。
                debug 至少包含 table_count、structured_table_count、failed_table_parse_count。

        说明:
            该方法不会抛出影响主流程的异常：单个表格解析失败只计入
            failed_table_parse_count，保证整篇论文仍能完成建索引。
        """
        table_chunks = [
            chunk
            for chunk in (chunks or [])
            if str((chunk.get("metadata", {}) or {}).get("chunk_type", "")).strip().lower() == "table"
        ]
        docling_items = self._collect_docling_table_items(document)

        structured_tables: List[Dict[str, Any]] = []
        failed_count = 0
        used_table_ids: set[str] = set()

        for chunk in table_chunks:
            metadata = chunk.setdefault("metadata", {})
            try:
                matched_item = self._match_docling_item(metadata, docling_items)
                table_object = self._build_table_object(
                    arxiv_id=arxiv_id,
                    chunk=chunk,
                    metadata=metadata,
                    docling_item=matched_item,
                    used_table_ids=used_table_ids,
                )
            except Exception as exc:  # 单表失败不能影响整篇论文建索引。
                failed_count += 1
                logger.warning(
                    "Failed to structure table chunk for %s (chunk_id=%s): %s",
                    arxiv_id,
                    metadata.get("chunk_id"),
                    exc,
                )
                continue

            if table_object is None:
                failed_count += 1
                continue

            # 回填双向关联：chunk 元数据记录 table_id，便于召回结果反查结构化表格。
            metadata["table_id"] = table_object["table_id"]
            structured_tables.append(table_object)

        debug = {
            "table_count": len(table_chunks),
            "structured_table_count": len(structured_tables),
            "failed_table_parse_count": failed_count,
            "docling_table_item_count": len(docling_items),
        }
        logger.info(
            "Structured tables for %s: table_count=%d structured=%d failed=%d",
            arxiv_id,
            debug["table_count"],
            debug["structured_table_count"],
            debug["failed_table_parse_count"],
        )
        return {"structured_tables": structured_tables, "debug": debug}

    # ------------------------------------------------------------------
    # 表格对象构建
    # ------------------------------------------------------------------
    def _build_table_object(
        self,
        arxiv_id: str,
        chunk: Dict[str, Any],
        metadata: Dict[str, Any],
        docling_item: Optional[Dict[str, Any]],
        used_table_ids: set[str],
    ) -> Optional[Dict[str, Any]]:
        """根据 chunk 与匹配到的 docling item 构造单个结构化表格对象。"""
        records = self._resolve_table_records(metadata, docling_item)
        columns, rows = self._records_to_columns_and_rows(records)
        parse_source = "structured" if records else "none"

        if not columns or not rows:
            # 没有结构化记录时，退回到 chunk 预览文本做保守解析。
            fallback_records = self._fallback_parse_records(chunk, metadata)
            if fallback_records:
                columns, rows = self._records_to_columns_and_rows(fallback_records)
                parse_source = "fallback_text"

        if not columns or not rows:
            return None

        page_number = self._safe_int(metadata.get("page_start") or metadata.get("page_number"))
        order_index = self._safe_int(metadata.get("order_index")) or 0
        table_id = self._build_table_id(arxiv_id, page_number, order_index, used_table_ids)

        cells = self._build_cells(columns, rows)
        confidence_values = [cell["confidence"] for cell in cells if cell["confidence"] is not None]
        table_confidence = round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0

        caption = str(
            metadata.get("asset_caption")
            or (docling_item or {}).get("asset_caption")
            or metadata.get("asset_summary")
            or ""
        ).strip()

        return {
            "table_id": table_id,
            "arxiv_id": arxiv_id,
            "caption": caption,
            "section_path": str(metadata.get("section_path", "") or "").strip(),
            "section_title": str(metadata.get("section_title", "") or "").strip(),
            "asset_section_match_type": str(metadata.get("asset_section_match_type", "") or "").strip(),
            "asset_section_match_confidence": float(metadata.get("asset_section_match_confidence", 0.0) or 0.0),
            "asset_section_match_reason": str(metadata.get("asset_section_match_reason", "") or "").strip(),
            "asset_section_match_is_heuristic": bool(metadata.get("asset_section_match_is_heuristic", False)),
            "asset_section_match_allow_embedding": bool(metadata.get("asset_section_match_allow_embedding", False)),
            "page_number": page_number,
            "columns": columns,
            "rows": rows,
            "cells": cells,
            "row_count": len(rows),
            "column_count": len(columns),
            "source_chunk_id": self._stable_identifier(metadata.get("chunk_id")),
            "original_chunk_id": self._stable_identifier(metadata.get("original_chunk_id") or metadata.get("parent_chunk_id") or metadata.get("chunk_id")),
            "asset_path": str(metadata.get("asset_path", "") or "").strip(),
            "order_index": order_index,
            "parse_source": parse_source,
            "confidence": table_confidence,
        }

    def _build_cells(self, columns: List[str], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把行记录展开为带数值、单位与置信度的单元格列表。"""
        cells: List[Dict[str, Any]] = []
        row_label_col = columns[0] if columns else None
        for row_index, row in enumerate(rows):
            row_label = str(row.get(row_label_col, "") or "").strip() if row_label_col else ""
            for col_name in columns:
                raw_value = row.get(col_name)
                raw_text = "" if raw_value is None else str(raw_value).strip()
                normalized_value, unit, confidence = self._normalize_cell_value(raw_text, col_name)
                cells.append(
                    {
                        "row_index": row_index,
                        "col_name": col_name,
                        "raw_value": raw_text,
                        "normalized_value": normalized_value,
                        "unit": unit,
                        "row_label": row_label,
                        "confidence": confidence,
                    }
                )
        return cells

    def _normalize_cell_value(
        self,
        raw_text: str,
        col_name: str,
    ) -> Tuple[Optional[float], Optional[str], float]:
        """从单元格文本中抽取归一化数值、单位与置信度。

        返回:
            Tuple[Optional[float], Optional[str], float]:
                (normalized_value, unit, confidence)。无法解析为数值时
                normalized_value 为 None，confidence 为 0。
        """
        text = str(raw_text or "").strip()
        if not text:
            return None, None, 0.0

        matches = _NUMBER_RE.findall(text)
        if not matches:
            return None, self._detect_unit(text, col_name), 0.0

        try:
            normalized_value = float(matches[0].replace(",", ""))
        except (TypeError, ValueError):
            return None, self._detect_unit(text, col_name), 0.0

        unit = self._detect_unit(text, col_name)
        if unit == "percent":
            # 百分号数值统一折算到 0-1 区间，便于跨表比较。
            normalized_value = normalized_value / 100.0

        # 置信度：单值且去掉数字/单位后无残留文本最可信；多值或夹杂文本则降级。
        residual = _NUMBER_RE.sub("", text).replace("%", "").strip(" ()[]↑↓±/,;:")
        if len(matches) == 1 and not residual:
            confidence = 0.95
        elif len(matches) == 1:
            confidence = 0.7
        else:
            # 单元格里有多个数值（例如 "0.956 0.724"），无法确定取哪个，保守取首个。
            confidence = 0.4
        return normalized_value, unit, round(confidence, 4)

    def _detect_unit(self, text: str, col_name: str) -> Optional[str]:
        """根据单元格文本与列名推断单位/指标类型。"""
        lowered_text = str(text or "").lower()
        for pattern, unit in _EXPLICIT_UNIT_PATTERNS:
            if pattern in lowered_text:
                return unit

        lowered_col = str(col_name or "").lower()
        for keyword, unit in _METRIC_UNIT_KEYWORDS:
            if keyword in lowered_col:
                return unit
        return None

    # ------------------------------------------------------------------
    # 记录抽取与匹配
    # ------------------------------------------------------------------
    def _resolve_table_records(
        self,
        metadata: Dict[str, Any],
        docling_item: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """按优先级解析完整表格行记录：磁盘 JSON > 预览记录。"""
        # chunk 元数据里带有 asset_json_path 时优先使用，避免仅靠 asset_path/order_index
        # 匹配 Docling item 导致同页多表或文件路径变化时误关联。
        metadata_records = self._load_records_from_file(metadata.get("asset_json_path"))
        if metadata_records:
            return metadata_records

        if docling_item:
            json_records = self._load_records_from_file(docling_item.get("asset_json_path"))
            if json_records:
                return json_records
            # asset_json_path 缺失时尝试从 csv 路径推导同名 json。
            json_records = self._load_records_from_file(
                self._derive_json_path(docling_item.get("asset_path") or docling_item.get("asset_abs_path"))
            )
            if json_records:
                return json_records
            preview = docling_item.get("asset_preview")
            if isinstance(preview, list) and preview:
                return [row for row in preview if isinstance(row, dict)]
        return []

    def _load_records_from_file(self, raw_path: Any) -> List[Dict[str, Any]]:
        """从结构化表格 JSON 文件读取完整行记录。"""
        path = self._resolve_existing_path(raw_path)
        if path is None:
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read structured table json %s: %s", path, exc)
            return []
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        return []

    def _match_docling_item(
        self,
        metadata: Dict[str, Any],
        docling_items: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """为表格 chunk 找到对应的 docling table item。

        匹配优先级：asset_path 完全一致 > (页码 + order_index) 一致。
        """
        if not docling_items:
            return None

        chunk_asset_path = self._normalize_path_key(metadata.get("asset_path"))
        if chunk_asset_path:
            for item in docling_items:
                if self._normalize_path_key(item.get("asset_path")) == chunk_asset_path:
                    return item

        chunk_page = self._safe_int(metadata.get("page_start") or metadata.get("page_number"))
        chunk_order = self._safe_int(metadata.get("order_index"))
        if chunk_page is not None and chunk_order is not None:
            for item in docling_items:
                if (
                    self._safe_int(item.get("page_start") or item.get("page")) == chunk_page
                    and self._safe_int(item.get("order_index")) == chunk_order
                ):
                    return item
        return None

    def _collect_docling_table_items(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从文档对象中收集 docling 表格项，兼容文档级与页级两种存放方式。"""
        if not isinstance(document, dict):
            return []

        items = document.get("docling_table_items")
        if isinstance(items, list) and items:
            return [item for item in items if isinstance(item, dict)]

        collected: List[Dict[str, Any]] = []
        for page in document.get("pages", []) or []:
            if not isinstance(page, dict):
                continue
            page_items = page.get("docling_table_items")
            if isinstance(page_items, list):
                collected.extend(item for item in page_items if isinstance(item, dict))
        return collected

    # ------------------------------------------------------------------
    # 行列归一化
    # ------------------------------------------------------------------
    def _records_to_columns_and_rows(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """把行记录列表归一化为列名列表与行字典列表。

        处理两种 Docling 导出形态：
        1. 列名即真实表头（例如 ``{"Dataset": ..., "Method": ...}``）。
        2. 列名为位置序号 ``"0"/"1"/...``，真实表头在第一行。
        """
        cleaned = [row for row in (records or []) if isinstance(row, dict) and row]
        if not cleaned:
            return [], []

        raw_columns = list(cleaned[0].keys())
        if self._columns_are_positional(raw_columns) and len(cleaned) >= 2:
            header_row = cleaned[0]
            columns = self._dedupe_columns([str(header_row.get(key, "") or "").strip() or str(key) for key in raw_columns])
            data_rows = cleaned[1:]
            rows = [
                {columns[idx]: self._stringify(row.get(raw_columns[idx])) for idx in range(len(raw_columns))}
                for row in data_rows
            ]
            return columns, rows

        columns = self._dedupe_columns([str(col).strip() for col in raw_columns])
        rows = [
            {columns[idx]: self._stringify(row.get(raw_columns[idx])) for idx in range(len(raw_columns))}
            for row in cleaned
        ]
        return columns, rows

    def _columns_are_positional(self, columns: List[Any]) -> bool:
        """判断列名是否为 0/1/2... 这类位置序号（说明真实表头在首行）。"""
        if not columns:
            return False
        return all(re.fullmatch(r"\d+", str(col).strip() or "") for col in columns)

    def _dedupe_columns(self, columns: List[str]) -> List[str]:
        """对重复列名做稳定去重，避免后续行字典键冲突。"""
        deduped: List[str] = []
        seen: Dict[str, int] = {}
        for col in columns:
            name = str(col).strip() or "column"
            count = seen.get(name, 0) + 1
            seen[name] = count
            deduped.append(name if count == 1 else f"{name}__{count}")
        return deduped

    # ------------------------------------------------------------------
    # 保守文本回退解析
    # ------------------------------------------------------------------
    def _fallback_parse_records(
        self,
        chunk: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """当缺少结构化记录时，从预览文本里保守恢复明显的行列结构。

        仅处理分隔符规整、列数一致的简单表格，不强行猜测跨行跨列结构。
        """
        preview_text = str(metadata.get("asset_preview_text", "") or "").strip()
        rows_text = [segment.strip() for segment in preview_text.split("|") if segment.strip()]
        parsed_rows: List[Dict[str, Any]] = []
        for segment in rows_text:
            pairs = [pair.strip() for pair in segment.split(";") if pair.strip()]
            row: Dict[str, Any] = {}
            for pair in pairs:
                if ":" not in pair:
                    return []  # 结构不规整，放弃保守解析。
                key, _, value = pair.partition(":")
                key = key.strip()
                if not key:
                    return []
                row[key] = value.strip()
            if row:
                parsed_rows.append(row)

        # 至少要有一行且各行列名一致才认为解析成功。
        if len(parsed_rows) >= 1:
            first_keys = set(parsed_rows[0].keys())
            if all(set(row.keys()) == first_keys for row in parsed_rows):
                return parsed_rows
        return []

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _build_table_id(
        self,
        arxiv_id: str,
        page_number: Optional[int],
        order_index: int,
        used_table_ids: set[str],
    ) -> str:
        """构造稳定且唯一的 table_id。

        基于 (arxiv_id, 页码, order_index) 生成，不依赖带时间戳的资产路径，
        因此可在多次重建之间保持稳定；如遇冲突再追加序号兜底。
        """
        base = f"{arxiv_id}-p{page_number if page_number is not None else 'NA'}-t{order_index}"
        table_id = base
        suffix = 1
        while table_id in used_table_ids:
            suffix += 1
            table_id = f"{base}-{suffix}"
        used_table_ids.add(table_id)
        return table_id

    def _derive_json_path(self, csv_like_path: Any) -> Optional[str]:
        """从 csv 资产路径推导同名 json 路径。"""
        text = str(csv_like_path or "").strip()
        if not text:
            return None
        if text.lower().endswith(".csv"):
            return text[:-4] + ".json"
        if text.lower().endswith(".json"):
            return text
        return None

    def _resolve_existing_path(self, raw_path: Any) -> Optional[Path]:
        """把相对/绝对资产路径解析为存在的本地文件路径。"""
        text = str(raw_path or "").strip()
        if not text:
            return None
        # Docling 在 Windows 上可能写入反斜杠路径，这里统一处理。
        candidate = Path(text)
        candidates = [candidate]
        if not candidate.is_absolute():
            root = Path(self._workspace_root) if self._workspace_root else Path(os.getcwd())
            candidates.append(root / candidate)
        for path in candidates:
            try:
                if path.is_file():
                    return path
            except OSError:
                continue
        return None

    def _normalize_path_key(self, raw_path: Any) -> str:
        """把资产路径归一化为可比较的 key（统一分隔符与大小写）。"""
        text = str(raw_path or "").strip()
        if not text:
            return ""
        return text.replace("\\", "/").lower()

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _stable_identifier(self, value: Any) -> Any:
        """保留 chunk 标识原始语义：数字 ID 转 int，字符串 ID 不强行丢弃。"""
        numeric_value = self._safe_int(value)
        if numeric_value is not None:
            return numeric_value
        text = str(value or "").strip()
        return text or None

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
