"""把 v2 表格证据渲染成 prompt / rerank 文本，以及给 trace 生成精简摘要。

三种决策对应三种渲染：compute 输出确定的单元格与计算结果；defer_to_llm 输出候选行列、
候选计算和原表上下文，交给 LLM 判断；fallback_table_context 只输出原表上下文。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from services.retrieval.table_evidence_schema import validate_table_evidence_payload


def render_table_evidence_prompt_block(
    evidence: Dict[str, Any],
    *,
    source_id: str = "",
    max_table_context_rows: Optional[int] = None,
) -> str:
    """只负责把 v2 表格证据渲染进 prompt，不在格式化层重新做行列或数值决策。"""
    validate_table_evidence_payload(evidence)
    decision = str(evidence.get("decision") or "")
    if decision == "compute":
        return _render_compute_block(evidence, source_id=source_id)
    if decision == "defer_to_llm":
        return _render_candidate_block(evidence, source_id=source_id, max_table_context_rows=max_table_context_rows)
    return _render_context_block(evidence, source_id=source_id, max_table_context_rows=max_table_context_rows)


def render_table_evidence_rerank_text(evidence: Dict[str, Any], *, max_chars: int = 4096) -> str:
    """给 rerank 用的单行文本：只带前 4 行原表上下文，压缩空白后截断到 max_chars。"""
    block = render_table_evidence_prompt_block(evidence, max_table_context_rows=4)
    return _limit_text(block, max_chars)


def summarize_table_evidence_debug(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """trace/debug 只保留决策摘要，避免把完整表格上下文在多个阶段反复膨胀。"""
    validate_table_evidence_payload(evidence)
    final = _dict_field(evidence, "final_evidence")
    candidate = _dict_field(evidence, "candidate_evidence")
    calculation = _dict_field(final, "calculation")
    return {
        "schema_version": evidence.get("schema_version"),
        "decision": evidence.get("decision"),
        "operation_hint": evidence.get("operation_hint"),
        "confidence": evidence.get("confidence"),
        "table_id": (evidence.get("table") or {}).get("table_id"),
        "final_rows": final.get("rows", []),
        "final_columns": final.get("columns", []),
        "final_cell_count": len(final.get("cells") or []),
        "calculation_display_value": calculation.get("display_value"),
        "candidate_rows": candidate.get("candidate_rows", []),
        "candidate_columns": candidate.get("candidate_columns", []),
        "candidate_calculation_count": len(candidate.get("candidate_calculations") or []),
        "reasons": evidence.get("reasons", {}),
    }


def _dict_field(container: Dict[str, Any], key: str) -> Dict[str, Any]:
    """取 container[key]，不是字典（缺失、None 或类型异常）时返回空字典。"""
    value = container.get(key)
    return value if isinstance(value, dict) else {}


def _text_items(values: Optional[Iterable[Any]]) -> List[str]:
    """转成字符串列表并丢弃空白项，用于行名、列名这类展示字段。"""
    return [str(item) for item in (values or []) if str(item).strip()]


def _optional_text(value: Any) -> str:
    """None 视为空串，其余值转字符串后去掉首尾空白（保留 0 / False 等有效值）。"""
    return "" if value is None else str(value).strip()


def _join_lines(lines: List[str]) -> str:
    """丢弃空行后按换行拼接；各渲染函数用空串占位“条件不满足的行”，在这里统一过滤。"""
    return "\n".join(line for line in lines if str(line).strip()).strip()


def _calculation_display(calculation: Dict[str, Any]) -> str:
    """计算结果的展示值：优先 display_value，没有再用原始 value。"""
    return str(calculation.get("display_value") or calculation.get("value") or "").strip()


def _render_common_header(title: str, evidence: Dict[str, Any], *, source_id: str = "") -> List[str]:
    """三种渲染共用的表头：标题、来源、表格元信息与决策信息；缺失的可选字段整行省略。"""
    table = _dict_field(evidence, "table")
    lines = [
        title,
        f"- source_id: {source_id}" if source_id else "",
        f"- table_id: {table.get('table_id') or ''}",
        f"- caption: {table.get('caption') or ''}" if table.get("caption") else "",
        f"- page: {table.get('page_number')}" if table.get("page_number") not in (None, "") else "",
        f"- section: {table.get('section_path') or table.get('section_title') or ''}" if (table.get("section_path") or table.get("section_title")) else "",
        f"- decision: {evidence.get('decision')}",
        f"- operation_hint: {evidence.get('operation_hint')}",
        f"- confidence: {evidence.get('confidence')}",
        f"- source chunk: {table.get('source_chunk_id') or table.get('original_chunk_id') or ''}",
    ]
    return [line for line in lines if str(line).strip()]


def _render_compute_block(evidence: Dict[str, Any], *, source_id: str = "") -> str:
    """compute 决策：列出命中的行列、单元格以及已算好的结果，LLM 直接引用即可。"""
    final = _dict_field(evidence, "final_evidence")
    rows = _text_items(final.get("rows"))
    columns = _text_items(final.get("columns"))
    lines = _render_common_header("Table Evidence:", evidence, source_id=source_id)
    lines.extend(
        [
            f"- matched row: {', '.join(rows)}" if rows else "",
            f"- matched column: {', '.join(columns)}" if columns else "",
            f"- operation: {final.get('operation') or evidence.get('operation_hint')}",
        ]
    )
    cell_lines = _cell_lines(final.get("cells") or [])
    if cell_lines:
        lines.append("Cell Evidence:")
        lines.extend(cell_lines)
    calculation = _dict_field(final, "calculation")
    if calculation:
        lines.append("Calculation Evidence:")
        expression = str(calculation.get("expression") or "").strip()
        display = _calculation_display(calculation)
        lines.append(f"- {expression} = {display}" if expression else f"- result = {display}")
    return _join_lines(lines)


def _render_candidate_block(
    evidence: Dict[str, Any],
    *,
    source_id: str = "",
    max_table_context_rows: Optional[int] = None,
) -> str:
    """defer_to_llm 决策：规则无法唯一确定行列，给出候选和原表上下文，并明确说明没有预计算答案。"""
    candidate = _dict_field(evidence, "candidate_evidence")
    rows = _text_items(candidate.get("candidate_rows"))
    columns = _text_items(candidate.get("candidate_columns"))
    lines = _render_common_header("Table Evidence Candidates:", evidence, source_id=source_id)
    lines.extend(
        [
            f"- candidate rows: {', '.join(rows)}" if rows else "",
            f"- candidate columns: {', '.join(columns)}" if columns else "",
            "- rule decision: metric or row/column evidence is ambiguous, so no final answer was precomputed.",
        ]
    )
    calculations = [item for item in (candidate.get("candidate_calculations") or []) if isinstance(item, dict)]
    if calculations:
        lines.append("Candidate Calculations:")
        for item in calculations:
            rows_text = ", ".join(_text_items(item.get("rows")))
            display = _calculation_display(_dict_field(item, "calculation"))
            lines.append(f"- {item.get('column')}: rows={rows_text}; result={display}")
    cell_lines = _cell_lines(candidate.get("candidate_cells") or [])
    if cell_lines:
        lines.append("Candidate Cells:")
        lines.extend(cell_lines)
    context_lines = _table_context_lines(evidence, max_rows=max_table_context_rows)
    if context_lines:
        lines.append("Original Table Context:")
        lines.extend(context_lines)
    return _join_lines(lines)


def _render_context_block(
    evidence: Dict[str, Any],
    *,
    source_id: str = "",
    max_table_context_rows: Optional[int] = None,
) -> str:
    """fallback_table_context 决策：只给表头信息和原表上下文。"""
    lines = _render_common_header("Relevant Table Context:", evidence, source_id=source_id)
    lines.extend(_table_context_lines(evidence, max_rows=max_table_context_rows))
    return _join_lines(lines)


def _cell_lines(cells: List[Any]) -> List[str]:
    """把单元格渲染成 "- 行 / 列 = 原值 (normalized_value=..; unit=..)"；没有标签或原值的单元格跳过。"""
    lines: List[str] = []
    for cell in [item for item in cells if isinstance(item, dict)]:
        row = str(cell.get("row_label") or "").strip()
        column = str(cell.get("col_name") or "").strip()
        raw_value = _optional_text(cell.get("raw_value"))
        normalized = cell.get("normalized_value")
        unit = _optional_text(cell.get("unit"))
        suffix_parts = []
        if normalized not in (None, ""):
            suffix_parts.append(f"normalized_value={normalized}")
        if unit:
            suffix_parts.append(f"unit={unit}")
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        label = " / ".join(part for part in (row, column) if part)
        if label and raw_value:
            lines.append(f"- {label} = {raw_value}{suffix}")
    return lines


def _table_context_lines(evidence: Dict[str, Any], *, max_rows: Optional[int]) -> List[str]:
    """渲染原表上下文：列名、行列数、是否截断，以及每行 "列=值" 列表（最多 max_rows 行）。

    row_count / column_count 优先用 payload 中记录的原表规模，而不是截断后的行数，
    让 LLM 知道看到的只是表格的一部分。
    """
    context = _dict_field(evidence, "table_context")
    columns = _text_items(context.get("columns"))
    rows = [row for row in (context.get("rows") or []) if isinstance(row, dict)]
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    lines = [
        f"- columns: {', '.join(columns)}" if columns else "",
        f"- row_count: {context.get('row_count', len(rows))}",
        f"- column_count: {context.get('column_count', len(columns))}",
        f"- truncated: {bool(context.get('truncated'))}",
    ]
    for row in rows:
        row_label = str(row.get("row_label") or "").strip()
        cell_values = []
        cells = _dict_field(row, "cells")
        for column in columns:
            value = cells.get(column)
            if isinstance(value, dict):
                # 结构化单元格优先展示表格里的原始写法，没有时退到归一化数值。
                display = value.get("raw_value") if value.get("raw_value") not in (None, "") else value.get("normalized_value")
            else:
                display = value
            if display not in (None, ""):
                cell_values.append(f"{column}={display}")
        if cell_values:
            lines.append(f"- row {row_label or row.get('row_index', '')}: {'; '.join(cell_values)}")
    return [line for line in lines if str(line).strip()]


def _limit_text(text: str, max_chars: int) -> str:
    """把所有空白压成单个空格；超过 max_chars 时保留前 max_chars - 3 个字符并追加 "..."。"""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, int(max_chars) - 3)].rstrip() + "..."
