from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

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
    block = render_table_evidence_prompt_block(evidence, max_table_context_rows=4)
    return _limit_text(block, max_chars)


def summarize_table_evidence_debug(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """trace/debug 只保留决策摘要，避免把完整表格上下文在多个阶段反复膨胀。"""
    validate_table_evidence_payload(evidence)
    final = evidence.get("final_evidence") if isinstance(evidence.get("final_evidence"), dict) else {}
    candidate = evidence.get("candidate_evidence") if isinstance(evidence.get("candidate_evidence"), dict) else {}
    calculation = final.get("calculation") if isinstance(final.get("calculation"), dict) else {}
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


def _render_common_header(title: str, evidence: Dict[str, Any], *, source_id: str = "") -> List[str]:
    table = evidence.get("table") if isinstance(evidence.get("table"), dict) else {}
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
    final = evidence.get("final_evidence") if isinstance(evidence.get("final_evidence"), dict) else {}
    rows = [str(item) for item in (final.get("rows") or []) if str(item).strip()]
    columns = [str(item) for item in (final.get("columns") or []) if str(item).strip()]
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
    calculation = final.get("calculation") if isinstance(final.get("calculation"), dict) else {}
    if calculation:
        lines.append("Calculation Evidence:")
        expression = str(calculation.get("expression") or "").strip()
        display = str(calculation.get("display_value") or calculation.get("value") or "").strip()
        lines.append(f"- {expression} = {display}" if expression else f"- result = {display}")
    return "\n".join(line for line in lines if str(line).strip()).strip()


def _render_candidate_block(
    evidence: Dict[str, Any],
    *,
    source_id: str = "",
    max_table_context_rows: Optional[int] = None,
) -> str:
    candidate = evidence.get("candidate_evidence") if isinstance(evidence.get("candidate_evidence"), dict) else {}
    rows = [str(item) for item in (candidate.get("candidate_rows") or []) if str(item).strip()]
    columns = [str(item) for item in (candidate.get("candidate_columns") or []) if str(item).strip()]
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
            calculation = item.get("calculation") if isinstance(item.get("calculation"), dict) else {}
            rows_text = ", ".join(str(row) for row in (item.get("rows") or []) if str(row).strip())
            display = str(calculation.get("display_value") or calculation.get("value") or "").strip()
            lines.append(f"- {item.get('column')}: rows={rows_text}; result={display}")
    cell_lines = _cell_lines(candidate.get("candidate_cells") or [])
    if cell_lines:
        lines.append("Candidate Cells:")
        lines.extend(cell_lines)
    context_lines = _table_context_lines(evidence, max_rows=max_table_context_rows)
    if context_lines:
        lines.append("Original Table Context:")
        lines.extend(context_lines)
    return "\n".join(line for line in lines if str(line).strip()).strip()


def _render_context_block(
    evidence: Dict[str, Any],
    *,
    source_id: str = "",
    max_table_context_rows: Optional[int] = None,
) -> str:
    lines = _render_common_header("Relevant Table Context:", evidence, source_id=source_id)
    context_lines = _table_context_lines(evidence, max_rows=max_table_context_rows)
    lines.extend(context_lines)
    return "\n".join(line for line in lines if str(line).strip()).strip()


def _cell_lines(cells: List[Any]) -> List[str]:
    lines: List[str] = []
    for cell in [item for item in cells if isinstance(item, dict)]:
        row = str(cell.get("row_label") or "").strip()
        column = str(cell.get("col_name") or "").strip()
        raw_value = str(cell.get("raw_value") if cell.get("raw_value") is not None else "").strip()
        normalized = cell.get("normalized_value")
        unit = str(cell.get("unit") if cell.get("unit") is not None else "").strip()
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
    context = evidence.get("table_context") if isinstance(evidence.get("table_context"), dict) else {}
    columns = [str(item) for item in (context.get("columns") or []) if str(item).strip()]
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
        cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        for column in columns:
            value = cells.get(column)
            if isinstance(value, dict):
                display = value.get("raw_value") if value.get("raw_value") not in (None, "") else value.get("normalized_value")
            else:
                display = value
            if display not in (None, ""):
                cell_values.append(f"{column}={display}")
        if cell_values:
            lines.append(f"- row {row_label or row.get('row_index', '')}: {'; '.join(cell_values)}")
    return [line for line in lines if str(line).strip()]


def _limit_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, int(max_chars) - 3)].rstrip() + "..."
