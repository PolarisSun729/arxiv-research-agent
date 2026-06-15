from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


TABLE_TRIGGER_TERMS: Tuple[str, ...] = (
    "table",
    "tab.",
    "result",
    "results",
    "score",
    "scores",
    "metric",
    "metrics",
    "accuracy",
    "f1",
    "bleu",
    "rouge",
    "precision",
    "recall",
    "baseline",
    "dataset",
    "ablation",
    "highest",
    "lowest",
    "best",
    "worst",
    "increase",
    "decrease",
    "drop",
    "improve",
    "compare",
    "comparison",
    "memory",
    "表",
    "结果",
    "分数",
    "指标",
    "准确率",
    "召回率",
    "精确率",
    "基线",
    "数据集",
    "消融",
    "最高",
    "最低",
    "最大",
    "最小",
    "提升",
    "下降",
    "降低",
    "增加",
    "去掉",
    "去除",
    "移除",
    "比较",
)

METRIC_HINT_TERMS: Tuple[str, ...] = (
    "accuracy",
    "acc",
    "f1",
    "bleu",
    "rouge",
    "precision",
    "recall",
    "em",
    "score",
    "scores",
    "metric",
    "metrics",
    "准确率",
    "分数",
    "指标",
)

MAX_TERMS: Tuple[str, ...] = ("highest", "highest value", "best", "top", "max", "maximum", "最高", "最大", "最强")
MIN_TERMS: Tuple[str, ...] = ("lowest", "worst", "min", "minimum", "最低", "最小", "最弱")
DIFF_TERMS: Tuple[str, ...] = (
    "difference",
    "diff",
    "gap",
    "increase",
    "decrease",
    "drop",
    "improve",
    "improvement",
    "下降",
    "提升",
    "提高",
    "降低",
    "增加",
    "差值",
    "相差",
)
NEGATIVE_ROW_TERMS: Tuple[str, ...] = ("w/o", "without", "remove", "removed", "no ", "drop ", "ablation", "去掉", "去除", "移除")
REFERENCE_ROW_TERMS: Tuple[str, ...] = ("full", "all", "base", "baseline", "ours", "our", "default", "complete", "完整", "全部", "原始")
NON_BASELINE_ROW_TERMS: Tuple[str, ...] = ("ours", "our", "proposed", "full model", "本文", "我们")


class TableStructuredRetriever:
    """基于结构化表格索引做精确匹配，专门服务结果表、消融表和对比表问题。"""

    def __init__(
        self,
        *,
        query_tools: Any,
        route_confidence_builder: Any,
        structural_bonus_builder: Any,
        config: Dict[str, Any],
    ) -> None:
        self.query_tools = query_tools
        self.route_confidence_builder = route_confidence_builder
        self.structural_bonus_builder = structural_bonus_builder
        self.config = config

    def retrieve(
        self,
        *,
        user_query: str,
        query_profile: Any,
        retrieval_index: Any,
        top_k: int,
    ) -> Dict[str, Any]:
        """按 caption、section、列名、行名和单元格值逐层收敛候选表格。"""
        signal = self._build_query_signal(user_query, query_profile)
        debug: Dict[str, Any] = {
            "enabled": False,
            "triggered_terms": signal["triggered_terms"],
            "operation": signal["operation"],
            "table_refs": signal["table_refs"],
            "metric_terms": signal["metric_terms"],
            "candidate_tables": [],
            "matched_tables": [],
            "matched_cells": [],
            "reason": "",
        }
        if not bool(self.config.get("enable_table_structured_route", True)):
            debug["reason"] = "disabled_by_runtime_config"
            return {"results": [], "debug": debug}
        if not signal["enabled"]:
            debug["reason"] = "query_not_table_like"
            return {"results": [], "debug": debug}

        structured_tables = [item for item in getattr(retrieval_index, "structured_tables", []) or [] if isinstance(item, dict)]
        debug["enabled"] = True
        debug["structured_table_count"] = len(structured_tables)
        if not structured_tables:
            debug["reason"] = "no_structured_tables"
            return {"results": [], "debug": debug}

        scored_candidates: List[Dict[str, Any]] = []
        for table in structured_tables[: max(1, int(self.config.get("table_structured_candidate_limit", 8)))]:
            candidate = self._score_table(table, signal, query_profile)
            if float(candidate.get("score", 0.0) or 0.0) < float(self.config.get("table_structured_match_score_floor", 0.22)):
                continue
            scored_candidates.append(candidate)

        scored_candidates.sort(
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
                len(item.get("matched_cells", []) or []),
            ),
            reverse=True,
        )
        debug["candidate_tables"] = [self._candidate_debug_row(item) for item in scored_candidates[:5]]
        if not scored_candidates:
            debug["reason"] = "no_table_match"
            return {"results": [], "debug": debug}

        raw_results: List[Dict[str, Any]] = []
        for candidate in scored_candidates[: max(1, top_k)]:
            chunk = self._resolve_source_chunk(candidate["table"], retrieval_index)
            if chunk is None:
                continue
            evidence = self._build_evidence_payload(candidate)
            raw_results.append(
                {
                    **chunk,
                    "score": float(candidate["score"]),
                    "table_id": candidate["table"].get("table_id"),
                    "table_structured_text": candidate["evidence_text"],
                    "table_structured_evidence": evidence,
                    "table_structured_reason": candidate["reason"],
                    "table_structured_confidence": float(candidate["confidence"]),
                }
            )
            debug["matched_tables"].append(
                {
                    "table_id": candidate["table"].get("table_id"),
                    "score": round(float(candidate["score"]), 4),
                    "confidence": round(float(candidate["confidence"]), 4),
                    "reason": candidate["reason"],
                    "matched_columns": candidate.get("matched_columns", [])[:4],
                    "matched_rows": candidate.get("matched_rows", [])[:4],
                }
            )
            debug["matched_cells"].extend(evidence.get("matched_cells", [])[:4])

        debug["route_result_count"] = len(raw_results)
        debug["reason"] = "table_candidates_available" if raw_results else "source_chunk_not_found"
        return {"results": raw_results, "debug": debug}

    def _build_query_signal(self, user_query: str, query_profile: Any) -> Dict[str, Any]:
        raw_query = str(user_query or "")
        normalized_query = self._normalize_text(raw_query)
        profile_normalized = self._normalize_text(getattr(query_profile, "normalized_query", "") or "")
        # 中文问句在上游归一化后可能被压成问号，这里必须保留原始问句做触发识别。
        signal_text = " ".join([raw_query, normalized_query, profile_normalized]).strip().lower()
        tokens = self._tokenize(signal_text)
        triggered_terms = self._matched_terms(signal_text, TABLE_TRIGGER_TERMS)
        metric_terms = self._matched_terms(signal_text, METRIC_HINT_TERMS)
        table_refs = self._extract_table_refs(raw_query) or self._extract_table_refs(normalized_query)
        enabled = bool(triggered_terms or metric_terms or table_refs)
        if self._has_any(signal_text, DIFF_TERMS):
            operation = "difference"
        elif self._has_any(signal_text, MAX_TERMS):
            operation = "max"
        elif self._has_any(signal_text, MIN_TERMS):
            operation = "min"
        else:
            operation = "lookup"
        return {
            "enabled": enabled,
            "normalized_query": signal_text,
            "tokens": tokens,
            "triggered_terms": triggered_terms[:10],
            "metric_terms": metric_terms[:6],
            "table_refs": table_refs,
            "operation": operation,
            "baseline_requested": "baseline" in signal_text or "基线" in signal_text,
            "component_terms": self._extract_component_terms(raw_query),
        }

    def _score_table(self, table: Dict[str, Any], signal: Dict[str, Any], query_profile: Any) -> Dict[str, Any]:
        columns = [str(item).strip() for item in (table.get("columns") or []) if str(item).strip()]
        row_labels = self._collect_row_labels(table)
        matched_columns = self._match_named_items(signal, columns)
        matched_rows = self._match_named_items(signal, row_labels)
        caption_text = self._normalize_text(" ".join([str(table.get("caption", "") or ""), str(table.get("table_id", "") or "")]))
        # section_path 可能来自 OCR/Docling 结构化恢复，只有在锚点可信时才参与打分。
        section_text = (
            self._normalize_text(" ".join([str(table.get("section_path", "") or ""), str(table.get("section_title", "") or "")]))
            if self._asset_section_anchor_allowed(table)
            else ""
        )

        score = 0.0
        reasons: List[str] = []
        if signal["table_refs"] and any(ref in caption_text for ref in signal["table_refs"]):
            score += 1.1
            reasons.append("table_ref_match")
        caption_overlap = self._overlap_score(signal["tokens"], self._tokenize(caption_text))
        if caption_overlap:
            score += min(0.45, caption_overlap * 0.6)
            reasons.append("caption_overlap")
        section_overlap = self._overlap_score(signal["tokens"], self._tokenize(section_text))
        if section_overlap:
            score += min(0.28, section_overlap * 0.35)
            reasons.append("section_overlap")
        if matched_columns:
            score += min(0.8, 0.26 * len(matched_columns))
            reasons.append("column_match")
        if matched_rows:
            score += min(0.55, 0.2 * len(matched_rows))
            reasons.append("row_match")
        if "ablation" in signal["normalized_query"] or "消融" in signal["normalized_query"]:
            if "ablation" in caption_text or "ablation" in section_text or "消融" in section_text:
                score += 0.25
                reasons.append("ablation_context")
        if "dataset" in signal["normalized_query"] or "数据集" in signal["normalized_query"]:
            if "dataset" in caption_text or "benchmark" in caption_text or "dataset" in section_text:
                score += 0.18
                reasons.append("dataset_context")

        evidence = self._select_evidence(
            table=table,
            signal=signal,
            matched_columns=matched_columns,
            matched_rows=matched_rows,
        )
        score += min(0.85, float(evidence.get("confidence", 0.0) or 0.0) * 0.9)
        if evidence.get("matched_cells"):
            reasons.append("cell_match")
        return {
            "table": table,
            "score": round(score, 4),
            "confidence": round(float(evidence.get("confidence", 0.0) or 0.0), 4),
            "reason": ", ".join(reasons) or "weak_match",
            "matched_columns": matched_columns,
            "matched_rows": matched_rows,
            **evidence,
        }

    def _select_evidence(
        self,
        *,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        if signal["operation"] == "difference":
            return self._difference_evidence(table, signal, matched_columns, matched_rows)
        if signal["operation"] in {"max", "min"}:
            return self._extreme_value_evidence(table, signal, matched_columns, matched_rows)
        return self._lookup_evidence(table, signal, matched_columns, matched_rows)

    def _extreme_value_evidence(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        candidate_rows = matched_rows or self._collect_row_labels(table)
        candidate_columns = matched_columns or self._preferred_metric_columns(table)
        numeric_cells = self._collect_numeric_cells(table, candidate_columns, candidate_rows, baseline_only=signal["baseline_requested"])
        if not numeric_cells:
            return self._lookup_evidence(table, signal, matched_columns, matched_rows)
        reverse = signal["operation"] == "max"
        ranked = sorted(numeric_cells, key=lambda item: float(item.get("normalized_value") or 0.0), reverse=reverse)
        best_cell = ranked[0]
        row_label = str(best_cell.get("row_label", "") or "").strip()
        col_name = str(best_cell.get("col_name", "") or "").strip()
        evidence_text = (
            f"表格命中 {table.get('table_id')}: {col_name} 的{'最高' if reverse else '最低'}值位于 {row_label or '未命名行'}，"
            f"原始值 {best_cell.get('raw_value') or '-'}。"
        )
        return {
            "matched_rows": [row_label] if row_label else matched_rows[:1],
            "matched_columns": [col_name] if col_name else matched_columns[:1],
            "matched_cells": [self._compact_cell(best_cell)],
            "evidence_type": "cell_lookup",
            "numeric_operation": signal["operation"],
            "confidence": min(0.96, 0.72 + float(best_cell.get("confidence", 0.0) or 0.0) * 0.25),
            "reason": "best_baseline_from_numeric_cells" if signal["baseline_requested"] else "extreme_value_from_numeric_cells",
            "computed_value": best_cell.get("normalized_value"),
            "evidence_text": evidence_text,
        }

    def _difference_evidence(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        candidate_columns = matched_columns or self._preferred_metric_columns(table)
        if not candidate_columns:
            return self._lookup_evidence(table, signal, matched_columns, matched_rows)

        focus_row = self._find_focus_row_for_difference(table, signal, matched_rows)
        if not focus_row:
            return self._lookup_evidence(table, signal, matched_columns, matched_rows)
        reference_row = self._find_reference_row_for_difference(table, focus_row, signal, matched_rows)
        if not reference_row:
            return self._lookup_evidence(table, signal, matched_columns, [focus_row])

        best_column = candidate_columns[0]
        focus_cell = self._find_row_column_cell(table, focus_row, best_column)
        reference_cell = self._find_row_column_cell(table, reference_row, best_column)
        if not focus_cell or not reference_cell:
            return self._lookup_evidence(table, signal, matched_columns, [focus_row, reference_row])
        if focus_cell.get("normalized_value") is None or reference_cell.get("normalized_value") is None:
            return self._lookup_evidence(table, signal, matched_columns, [focus_row, reference_row])

        difference = float(reference_cell["normalized_value"]) - float(focus_cell["normalized_value"])
        evidence_text = (
            f"表格命中 {table.get('table_id')}: {best_column} 在 {reference_row} 与 {focus_row} 之间相差 "
            f"{round(difference, 4)}，原始值分别为 {reference_cell.get('raw_value') or '-'} 和 {focus_cell.get('raw_value') or '-'}。"
        )
        return {
            "matched_rows": [focus_row, reference_row],
            "matched_columns": [best_column],
            "matched_cells": [self._compact_cell(reference_cell), self._compact_cell(focus_cell)],
            "evidence_type": "cell_comparison",
            "numeric_operation": "difference",
            "confidence": min(
                0.94,
                0.68
                + float(reference_cell.get("confidence", 0.0) or 0.0) * 0.12
                + float(focus_cell.get("confidence", 0.0) or 0.0) * 0.12,
            ),
            "reason": "difference_between_reference_and_focus_row",
            "computed_value": round(difference, 6),
            "evidence_text": evidence_text,
        }

    def _lookup_evidence(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        matched_cells = self._collect_matching_cells(table, matched_columns, matched_rows)
        evidence_text_parts = [f"表格命中 {table.get('table_id')}"]
        if matched_columns:
            evidence_text_parts.append(f"列: {', '.join(matched_columns[:3])}")
        if matched_rows:
            evidence_text_parts.append(f"行: {', '.join(matched_rows[:3])}")
        if matched_cells:
            preview = matched_cells[0]
            evidence_text_parts.append(
                f"示例单元格: {preview.get('row_label') or '未命名行'} / {preview.get('col_name')} = {preview.get('raw_value') or '-'}"
            )
        return {
            "matched_rows": matched_rows[:4],
            "matched_columns": matched_columns[:4],
            "matched_cells": [self._compact_cell(item) for item in matched_cells[:4]],
            "evidence_type": "table_lookup",
            "numeric_operation": "lookup",
            "confidence": 0.45 + min(0.35, 0.1 * len(matched_columns) + 0.08 * len(matched_rows) + 0.06 * len(matched_cells)),
            "reason": "table_lookup_match",
            "computed_value": None,
            "evidence_text": "；".join(evidence_text_parts),
        }

    def _collect_numeric_cells(
        self,
        table: Dict[str, Any],
        columns: Sequence[str],
        rows: Sequence[str],
        *,
        baseline_only: bool,
    ) -> List[Dict[str, Any]]:
        row_set = {str(item).strip() for item in rows if str(item).strip()}
        column_set = {str(item).strip() for item in columns if str(item).strip()}
        numeric_cells: List[Dict[str, Any]] = []
        for cell in table.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            if cell.get("normalized_value") is None:
                continue
            row_label = str(cell.get("row_label", "") or "").strip()
            col_name = str(cell.get("col_name", "") or "").strip()
            if row_set and row_label not in row_set:
                continue
            if column_set and col_name not in column_set:
                continue
            if baseline_only and any(term in self._normalize_text(row_label) for term in NON_BASELINE_ROW_TERMS):
                continue
            numeric_cells.append(cell)
        return numeric_cells

    def _collect_matching_cells(self, table: Dict[str, Any], columns: Sequence[str], rows: Sequence[str]) -> List[Dict[str, Any]]:
        row_set = {str(item).strip() for item in rows if str(item).strip()}
        column_set = {str(item).strip() for item in columns if str(item).strip()}
        cells: List[Dict[str, Any]] = []
        for cell in table.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            row_label = str(cell.get("row_label", "") or "").strip()
            col_name = str(cell.get("col_name", "") or "").strip()
            if row_set and row_label not in row_set:
                continue
            if column_set and col_name not in column_set:
                continue
            if not row_set and not column_set:
                continue
            cells.append(cell)
        return cells

    def _find_focus_row_for_difference(self, table: Dict[str, Any], signal: Dict[str, Any], matched_rows: List[str]) -> Optional[str]:
        if matched_rows:
            negative_rows = [row for row in matched_rows if any(term in self._normalize_text(row) for term in NEGATIVE_ROW_TERMS)]
            if negative_rows:
                return negative_rows[0]
            return matched_rows[0]
        component_terms = signal.get("component_terms", []) or []
        row_labels = self._collect_row_labels(table)
        scored: List[Tuple[float, str]] = []
        for row in row_labels:
            normalized = self._normalize_text(row)
            score = 0.0
            if any(term in normalized for term in NEGATIVE_ROW_TERMS):
                score += 0.5
            score += 0.18 * sum(1 for term in component_terms if term and term in normalized)
            if score > 0:
                scored.append((score, row))
        scored.sort(reverse=True)
        return scored[0][1] if scored else None

    def _find_reference_row_for_difference(
        self,
        table: Dict[str, Any],
        focus_row: str,
        signal: Dict[str, Any],
        matched_rows: List[str],
    ) -> Optional[str]:
        row_labels = [row for row in self._collect_row_labels(table) if row != focus_row]
        if matched_rows:
            preferred = [row for row in matched_rows if row != focus_row]
            if preferred:
                row_labels = preferred + [row for row in row_labels if row not in preferred]
        component_terms = signal.get("component_terms", []) or []
        scored: List[Tuple[float, str]] = []
        for row in row_labels:
            normalized = self._normalize_text(row)
            score = 0.0
            if any(term in normalized for term in REFERENCE_ROW_TERMS):
                score += 0.45
            if any(term in normalized for term in component_terms) and not any(term in normalized for term in NEGATIVE_ROW_TERMS):
                score += 0.15
            scored.append((score, row))
        scored.sort(reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]

        numeric_cells = self._collect_numeric_cells(table, self._preferred_metric_columns(table), row_labels, baseline_only=False)
        if not numeric_cells:
            return row_labels[0] if row_labels else None
        best_cell = sorted(numeric_cells, key=lambda item: float(item.get("normalized_value") or 0.0), reverse=True)[0]
        return str(best_cell.get("row_label", "") or "").strip() or (row_labels[0] if row_labels else None)

    def _find_row_column_cell(self, table: Dict[str, Any], row_label: str, col_name: str) -> Optional[Dict[str, Any]]:
        for cell in table.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            if str(cell.get("row_label", "") or "").strip() == str(row_label or "").strip() and str(cell.get("col_name", "") or "").strip() == str(col_name or "").strip():
                return cell
        return None

    def _preferred_metric_columns(self, table: Dict[str, Any]) -> List[str]:
        columns = [str(item).strip() for item in (table.get("columns") or []) if str(item).strip()]
        metric_columns = [col for col in columns if any(term in self._normalize_text(col) for term in METRIC_HINT_TERMS)]
        if metric_columns:
            return metric_columns
        return columns[1:2] if len(columns) > 1 else columns[:1]

    def _match_named_items(self, signal: Dict[str, Any], items: Sequence[str]) -> List[str]:
        matches: List[Tuple[float, str]] = []
        query_text = signal["normalized_query"]
        query_tokens = signal["tokens"]
        for item in items:
            normalized_item = self._normalize_text(item)
            if not normalized_item:
                continue
            score = 0.0
            if normalized_item in query_text:
                score += 0.9
            overlap = self._overlap_score(query_tokens, self._tokenize(normalized_item))
            if overlap:
                score += overlap
            if any(term in normalized_item for term in signal.get("metric_terms", [])):
                score += 0.35
            if score > 0:
                matches.append((score, item))
        matches.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in matches[:4]]

    def _resolve_source_chunk(self, table: Dict[str, Any], retrieval_index: Any) -> Optional[Dict[str, Any]]:
        source_ids = [
            str(table.get("source_chunk_id", "") or "").strip(),
            str(table.get("original_chunk_id", "") or "").strip(),
        ]
        for source_id in source_ids:
            if source_id and getattr(retrieval_index, "by_chunk_id", {}).get(source_id):
                return dict(retrieval_index.documents[retrieval_index.by_chunk_id[source_id][0]].chunk)
            if source_id and getattr(retrieval_index, "by_original_chunk_id", {}).get(source_id):
                return dict(retrieval_index.documents[retrieval_index.by_original_chunk_id[source_id][0]].chunk)

        # 兜底构造一个最小 table chunk，保证结构化表格索引不依赖向量库补全字段。
        return {
            "content": str(table.get("caption", "") or "").strip(),
            "chunk_id": table.get("source_chunk_id") or table.get("table_id"),
            "parent_chunk_id": table.get("original_chunk_id") or table.get("source_chunk_id") or table.get("table_id"),
            "original_chunk_id": table.get("original_chunk_id") or table.get("source_chunk_id") or table.get("table_id"),
            "chunk_type": "table",
            "page_number": table.get("page_number"),
            "section_path": table.get("section_path", ""),
            "section_title": table.get("section_title", ""),
            "asset_section_match_type": table.get("asset_section_match_type", ""),
            "asset_section_match_confidence": table.get("asset_section_match_confidence", 0.0),
            "asset_section_match_reason": table.get("asset_section_match_reason", ""),
            "asset_section_match_is_heuristic": table.get("asset_section_match_is_heuristic", False),
            "asset_section_match_allow_embedding": table.get("asset_section_match_allow_embedding", False),
            "asset_kind": "table",
            "asset_path": table.get("asset_path", ""),
            "asset_summary": table.get("caption", ""),
            "asset_preview_text": "",
            "asset_caption": table.get("caption", ""),
            "order_index": table.get("order_index", 0),
        }

    def _collect_row_labels(self, table: Dict[str, Any]) -> List[str]:
        labels: List[str] = []
        for cell in table.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            label = str(cell.get("row_label", "") or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _build_evidence_payload(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "table_id": candidate["table"].get("table_id"),
            "matched_columns": candidate.get("matched_columns", [])[:4],
            "matched_rows": candidate.get("matched_rows", [])[:4],
            "matched_cells": candidate.get("matched_cells", [])[:6],
            "evidence_type": candidate.get("evidence_type", "table_lookup"),
            "numeric_operation": candidate.get("numeric_operation", "lookup"),
            "confidence": round(float(candidate.get("confidence", 0.0) or 0.0), 4),
            "reason": candidate.get("reason", ""),
            "computed_value": candidate.get("computed_value"),
        }

    def _candidate_debug_row(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "table_id": candidate["table"].get("table_id"),
            "caption": candidate["table"].get("caption", ""),
            "section_path": candidate["table"].get("section_path", ""),
            "asset_section_match_type": candidate["table"].get("asset_section_match_type", ""),
            "asset_section_match_confidence": candidate["table"].get("asset_section_match_confidence", 0.0),
            "asset_section_match_is_heuristic": candidate["table"].get("asset_section_match_is_heuristic", False),
            "asset_section_match_allow_embedding": candidate["table"].get("asset_section_match_allow_embedding", False),
            "score": round(float(candidate.get("score", 0.0) or 0.0), 4),
            "confidence": round(float(candidate.get("confidence", 0.0) or 0.0), 4),
            "matched_columns": candidate.get("matched_columns", [])[:4],
            "matched_rows": candidate.get("matched_rows", [])[:4],
            "matched_cells": candidate.get("matched_cells", [])[:3],
            "reason": candidate.get("reason", ""),
        }

    @staticmethod
    def _asset_section_anchor_allowed(table: Dict[str, Any]) -> bool:
        """section_path 只有在显式可靠时才参与匹配，避免 OCR 锚点误导表格召回。"""
        match_type = str(table.get("asset_section_match_type", "") or "").strip()
        if not match_type:
            return True
        return bool(table.get("asset_section_match_allow_embedding", False))

    @staticmethod
    def _extract_table_refs(text: str) -> List[str]:
        refs: List[str] = []
        for pattern in (r"\btable\s*([0-9]+[a-z]?)\b", r"表\s*([0-9]+[a-zA-Z]?)"):
            for match in re.findall(pattern, str(text or ""), flags=re.IGNORECASE):
                label = str(match or "").strip().lower()
                if label:
                    refs.append(label)
                    refs.append(f"table {label}")
                    refs.append(f"表{label}")
        deduped: List[str] = []
        for ref in refs:
            if ref not in deduped:
                deduped.append(ref)
        return deduped

    def _extract_component_terms(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        patterns = (
            r"(?:without|w/o|remove|removed|drop)\s+([a-z0-9_\-]+)",
            r"(?:去掉|去除|移除|无)\s*([a-zA-Z0-9_\-\u4e00-\u9fff]+)",
        )
        terms: List[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, normalized, flags=re.IGNORECASE):
                term = str(match or "").strip().lower()
                if term and term not in terms:
                    terms.append(term)
        return terms[:3]

    def _normalize_text(self, text: Any) -> str:
        normalizer = getattr(self.query_tools, "normalize_query_text", None)
        value = str(text or "").strip()
        if callable(normalizer):
            return str(normalizer(value) or "").strip().lower()
        return value.lower()

    def _tokenize(self, text: str) -> List[str]:
        tokenizer = getattr(self.query_tools, "tokenize_for_keyword_search", None)
        if callable(tokenizer):
            return [str(item).strip().lower() for item in tokenizer(text) if str(item).strip()]
        return [item for item in re.split(r"\W+", str(text or "").lower()) if item]

    @staticmethod
    def _matched_terms(text: str, terms: Sequence[str]) -> List[str]:
        normalized = str(text or "").lower()
        return [term for term in terms if term in normalized]

    @staticmethod
    def _has_any(text: str, terms: Sequence[str]) -> bool:
        normalized = str(text or "").lower()
        return any(term in normalized for term in terms)

    @staticmethod
    def _overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
        left_set = {str(item).strip() for item in left if str(item).strip()}
        right_set = {str(item).strip() for item in right if str(item).strip()}
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / max(1, len(right_set))

    @staticmethod
    def _compact_cell(cell: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "row_index": cell.get("row_index"),
            "row_label": cell.get("row_label"),
            "col_name": cell.get("col_name"),
            "raw_value": cell.get("raw_value"),
            "normalized_value": cell.get("normalized_value"),
            "unit": cell.get("unit"),
            "confidence": cell.get("confidence"),
        }
