from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.retrieval.table_evidence_schema import (
    TableCandidateCalculation,
    TableCandidateEvidence,
    TableContext,
    TableEvidenceCalculation,
    TableEvidenceCell,
    TableEvidencePayload,
    TableEvidenceTable,
    TableFinalEvidence,
)


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
    "f1-score",
    "bleu",
    "rouge",
    "precision",
    "recall",
    "em",
    "exact match",
    "准确率",
    "召回率",
    "精确率",
)

GENERIC_METRIC_TERMS: Tuple[str, ...] = ("score", "scores", "metric", "metrics", "performance", "result", "results", "best", "指标", "分数", "性能", "结果")
LABEL_COLUMN_TERMS: Tuple[str, ...] = ("model", "method", "approach", "dataset", "setting", "variant", "模型", "方法", "数据集", "设置")
MAX_TERMS: Tuple[str, ...] = ("highest", "best", "top", "max", "maximum", "最高", "最大", "最佳")
MIN_TERMS: Tuple[str, ...] = ("lowest", "worst", "min", "minimum", "最低", "最小", "最差")
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
NEGATIVE_ROW_TERMS: Tuple[str, ...] = ("w/o", "without", "remove", "removed", "no ", "drop ", "ablation", "去掉", "去除", "移除", "无")
REFERENCE_ROW_TERMS: Tuple[str, ...] = ("full", "all", "base", "baseline", "ours", "our", "default", "complete", "完整", "全部", "原始")
NON_BASELINE_ROW_TERMS: Tuple[str, ...] = ("ours", "our", "proposed", "full model", "本文", "我们")

CONSERVATIVE_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "accuracy": ("accuracy", "acc", "准确率"),
    "f1": ("f1", "f1-score", "f1 score"),
    "bleu": ("bleu",),
    "rouge": ("rouge",),
    "precision": ("precision", "精确率"),
    "recall": ("recall", "召回率"),
    "em": ("em", "exact match"),
}


class TableStructuredRetriever:
    """基于结构化表格索引抽取可追踪证据，规则层只在高置信条件下计算最终数值。"""

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
        """按 table/caption/section/行列/cell 召回候选表，并输出 table_evidence_v2。"""
        signal = self._build_query_signal(user_query, query_profile)
        debug: Dict[str, Any] = {
            "enabled": False,
            "triggered_terms": signal["triggered_terms"],
            "operation_hint": signal["operation_hint"],
            "table_refs": signal["table_refs"],
            "metric_terms": signal["metric_terms"],
            "candidate_tables": [],
            "matched_tables": [],
            "matched_cells": [],
            "decision_counts": {},
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
        candidate_limit = max(1, int(self.config.get("table_structured_candidate_limit", 8)))
        for table in structured_tables[:candidate_limit]:
            candidate = self._score_table(table, signal, query_profile)
            if float(candidate.get("score", 0.0) or 0.0) < float(self.config.get("table_structured_match_score_floor", 0.22)):
                continue
            scored_candidates.append(candidate)

        scored_candidates.sort(
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                float(item.get("confidence", 0.0) or 0.0),
                len(self._evidence_cells(item.get("table_evidence") or {})),
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
            table_evidence = candidate["table_evidence"]
            raw_results.append(
                {
                    **chunk,
                    "score": float(candidate["score"]),
                    "table_id": candidate["table"].get("table_id"),
                    "table_evidence": table_evidence,
                }
            )
            decision = str(table_evidence.get("decision") or "")
            debug["decision_counts"][decision] = int(debug["decision_counts"].get(decision, 0)) + 1
            debug["matched_tables"].append(
                {
                    "table_id": candidate["table"].get("table_id"),
                    "score": round(float(candidate["score"]), 4),
                    "confidence": round(float(candidate["confidence"]), 4),
                    "decision": decision,
                    "operation_hint": table_evidence.get("operation_hint"),
                    "matched_columns": candidate.get("matched_columns", [])[:4],
                    "matched_rows": candidate.get("matched_rows", [])[:4],
                    "reasons": table_evidence.get("reasons", {}),
                }
            )
            debug["matched_cells"].extend(self._evidence_cells(table_evidence)[:4])

        debug["route_result_count"] = len(raw_results)
        debug["reason"] = "table_candidates_available" if raw_results else "source_chunk_not_found"
        return {"results": raw_results, "debug": debug}

    def _build_query_signal(self, user_query: str, query_profile: Any) -> Dict[str, Any]:
        raw_query = str(user_query or "")
        normalized_query = self._normalize_text(raw_query)
        profile_normalized = self._normalize_text(getattr(query_profile, "normalized_query", "") or "")
        # 原始问句和归一化问句一起参与触发，避免中文符号或 query profile 改写导致表格意图丢失。
        signal_text = " ".join([raw_query, normalized_query, profile_normalized]).strip().lower()
        tokens = self._tokenize(signal_text)
        triggered_terms = self._matched_terms(signal_text, TABLE_TRIGGER_TERMS)
        metric_terms = self._matched_terms(signal_text, METRIC_HINT_TERMS)
        table_refs = self._extract_table_refs(raw_query) or self._extract_table_refs(normalized_query)
        enabled = bool(triggered_terms or metric_terms or table_refs)
        if self._has_any(signal_text, DIFF_TERMS):
            operation_hint = "difference"
        elif self._has_any(signal_text, MAX_TERMS):
            operation_hint = "max"
        elif self._has_any(signal_text, MIN_TERMS):
            operation_hint = "min"
        else:
            operation_hint = "lookup"
        return {
            "enabled": enabled,
            "normalized_query": signal_text,
            "tokens": tokens,
            "triggered_terms": triggered_terms[:10],
            "metric_terms": metric_terms[:6],
            "table_refs": table_refs,
            "operation_hint": operation_hint,
            "baseline_requested": "baseline" in signal_text or "基线" in signal_text,
            "component_terms": self._extract_component_terms(raw_query),
        }

    def _score_table(self, table: Dict[str, Any], signal: Dict[str, Any], query_profile: Any) -> Dict[str, Any]:
        columns = [str(item).strip() for item in (table.get("columns") or []) if str(item).strip()]
        row_labels = self._collect_row_labels(table)
        matched_columns = self._match_named_items(signal, columns)
        matched_rows = self._match_named_items(signal, row_labels)
        caption_text = self._normalize_text(" ".join([str(table.get("caption", "") or ""), str(table.get("table_id", "") or "")]))
        # 弱 OCR/启发式章节锚点不参与表级打分，避免错误 section_path 把表格召回带偏。
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

        table_evidence = self._select_table_evidence(
            table=table,
            signal=signal,
            matched_columns=matched_columns,
            matched_rows=matched_rows,
        )
        confidence = float(table_evidence.get("confidence", 0.0) or 0.0)
        score += min(0.85, confidence * 0.9)
        if self._evidence_cells(table_evidence):
            reasons.append("cell_match")
        return {
            "table": table,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "reason": ", ".join(reasons) or "weak_match",
            "matched_columns": matched_columns,
            "matched_rows": matched_rows,
            "table_evidence": table_evidence,
        }

    def _select_table_evidence(
        self,
        *,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        operation_hint = signal["operation_hint"]
        if operation_hint == "difference":
            return self._difference_evidence(table, signal, matched_columns, matched_rows)
        if operation_hint in {"max", "min"}:
            return self._extreme_value_evidence(table, signal, matched_columns, matched_rows)
        return self._lookup_evidence(table, signal, matched_columns, matched_rows)

    def _extreme_value_evidence(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        metric_columns, metric_state, metric_reasons = self._resolve_metric_columns(table, signal, matched_columns)
        candidate_rows = matched_rows or self._collect_row_labels(table)
        candidate_columns = metric_columns or self._numeric_metric_columns(table)
        table_context = self._build_table_context(table, focus_rows=candidate_rows)
        reasons = self._empty_reasons()
        reasons["matched"].extend(metric_reasons)
        if metric_state != "clear" or len(metric_columns) != 1:
            reasons["ambiguity"].append("metric_column_not_specified")
            return self._build_payload(
                table=table,
                decision="defer_to_llm",
                operation_hint=signal["operation_hint"],
                confidence=0.58 if candidate_columns else 0.38,
                final_evidence=None,
                candidate_evidence=self._extreme_candidate_evidence(
                    table,
                    signal["operation_hint"],
                    candidate_columns,
                    candidate_rows,
                ),
                table_context=table_context,
                reasons=reasons,
                debug={"metric_state": metric_state, "matched_columns": matched_columns, "matched_rows": matched_rows},
            )

        numeric_cells = self._collect_numeric_cells(table, metric_columns, candidate_rows, baseline_only=signal["baseline_requested"])
        if not numeric_cells:
            reasons["fallback"].append("numeric_cells_not_found")
            return self._fallback_payload(table, signal, table_context, reasons, matched_columns, matched_rows)

        reverse = signal["operation_hint"] == "max"
        ranked = sorted(numeric_cells, key=lambda item: float(item.get("normalized_value") or 0.0), reverse=reverse)
        best_cell = ranked[0]
        calculation = TableEvidenceCalculation(
            operation=signal["operation_hint"],
            value=best_cell.get("normalized_value"),
            display_value=self._display_cell_value(best_cell),
            unit=str(best_cell.get("unit") or ""),
            expression=self._cell_label(best_cell),
        )
        final = TableFinalEvidence(
            operation=signal["operation_hint"],
            rows=[str(best_cell.get("row_label") or "").strip()],
            columns=[str(best_cell.get("col_name") or "").strip()],
            cells=[TableEvidenceCell.from_cell(best_cell)],
            calculation=calculation,
            reason="extreme_value_from_numeric_cells",
        )
        return self._build_payload(
            table=table,
            decision="compute",
            operation_hint=signal["operation_hint"],
            confidence=min(0.96, 0.72 + float(best_cell.get("confidence", 0.0) or 0.0) * 0.25),
            final_evidence=final,
            candidate_evidence=None,
            table_context=table_context,
            reasons=reasons,
            debug={"metric_state": metric_state, "ranked_cell_count": len(ranked)},
        )

    def _difference_evidence(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        reasons = self._empty_reasons()
        metric_columns, metric_state, metric_reasons = self._resolve_metric_columns(table, signal, matched_columns)
        reasons["matched"].extend(metric_reasons)

        focus_row = self._find_focus_row_for_difference(table, signal, matched_rows)
        reference_row = self._find_reference_row_for_difference(table, focus_row, signal, matched_rows) if focus_row else None
        focus_rows = [row for row in (focus_row, reference_row) if row]
        table_context = self._build_table_context(table, focus_rows=focus_rows or matched_rows)

        if not focus_row:
            reasons["fallback"].append("focus_row_not_found")
            return self._fallback_payload(table, signal, table_context, reasons, matched_columns, matched_rows)
        if not reference_row:
            reasons["fallback"].append("reference_row_not_found")
            return self._fallback_payload(table, signal, table_context, reasons, matched_columns, [focus_row])

        candidate_columns = metric_columns or self._numeric_metric_columns(table)
        candidate_calculations = self._difference_candidate_calculations(table, focus_row, reference_row, candidate_columns)
        candidate = TableCandidateEvidence(
            candidate_rows=[focus_row, reference_row],
            candidate_columns=candidate_columns,
            candidate_cells=[cell for item in candidate_calculations for cell in item.cells],
            candidate_calculations=candidate_calculations,
        )
        if metric_state != "clear" or len(metric_columns) != 1:
            # 这里是规则膨胀阀门：metric 不唯一时只交候选和上下文，不把任一列伪装成最终答案。
            reasons["ambiguity"].append("metric_column_not_specified")
            return self._build_payload(
                table=table,
                decision="defer_to_llm",
                operation_hint="difference",
                confidence=0.64 if candidate_calculations else 0.46,
                final_evidence=None,
                candidate_evidence=candidate,
                table_context=table_context,
                reasons=reasons,
                debug={"metric_state": metric_state, "matched_columns": matched_columns, "matched_rows": matched_rows},
            )

        metric_column = metric_columns[0]
        focus_cell = self._find_row_column_cell(table, focus_row, metric_column)
        reference_cell = self._find_row_column_cell(table, reference_row, metric_column)
        if not self._cell_has_numeric_value(focus_cell) or not self._cell_has_numeric_value(reference_cell):
            reasons["fallback"].append("difference_numeric_cells_not_found")
            return self._fallback_payload(table, signal, table_context, reasons, matched_columns, [focus_row, reference_row])

        calculation = self._difference_calculation(reference_cell, focus_cell)
        final = TableFinalEvidence(
            operation="difference",
            rows=[focus_row, reference_row],
            columns=[metric_column],
            cells=[TableEvidenceCell.from_cell(reference_cell), TableEvidenceCell.from_cell(focus_cell)],
            calculation=calculation,
            reason="difference_between_reference_and_focus_row",
        )
        return self._build_payload(
            table=table,
            decision="compute",
            operation_hint="difference",
            confidence=min(
                0.94,
                0.68
                + float(reference_cell.get("confidence", 0.0) or 0.0) * 0.12
                + float(focus_cell.get("confidence", 0.0) or 0.0) * 0.12,
            ),
            final_evidence=final,
            candidate_evidence=None,
            table_context=table_context,
            reasons=reasons,
            debug={"metric_state": metric_state},
        )

    def _lookup_evidence(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        reasons = self._empty_reasons()
        table_context = self._build_table_context(table, focus_rows=matched_rows)
        matched_cells = self._collect_matching_cells(table, matched_columns, matched_rows)
        if len(matched_rows) == 1 and len(matched_columns) == 1 and matched_cells:
            # lookup 只有行列都唯一时才算高置信证据；否则交给模型结合表格上下文消歧。
            primary_cell = matched_cells[0]
            final = TableFinalEvidence(
                operation="lookup",
                rows=[matched_rows[0]],
                columns=[matched_columns[0]],
                cells=[TableEvidenceCell.from_cell(primary_cell)],
                calculation=TableEvidenceCalculation(
                    operation="lookup",
                    value=primary_cell.get("normalized_value"),
                    display_value=self._display_cell_value(primary_cell),
                    unit=str(primary_cell.get("unit") or ""),
                    expression=self._cell_label(primary_cell),
                ),
                reason="unique_row_and_column_lookup",
            )
            return self._build_payload(
                table=table,
                decision="compute",
                operation_hint="lookup",
                confidence=0.72 + min(0.18, float(primary_cell.get("confidence", 0.0) or 0.0) * 0.18),
                final_evidence=final,
                candidate_evidence=None,
                table_context=table_context,
                reasons=reasons,
                debug={"matched_columns": matched_columns, "matched_rows": matched_rows},
            )

        candidate = TableCandidateEvidence(
            candidate_rows=matched_rows[:4],
            candidate_columns=matched_columns[:4],
            candidate_cells=[TableEvidenceCell.from_cell(item) for item in matched_cells[:6]],
            candidate_calculations=[],
        )
        if matched_rows or matched_columns or matched_cells:
            reasons["ambiguity"].append("lookup_row_or_column_not_unique")
            return self._build_payload(
                table=table,
                decision="defer_to_llm",
                operation_hint="lookup",
                confidence=0.5 + min(0.18, 0.04 * len(matched_cells)),
                final_evidence=None,
                candidate_evidence=candidate,
                table_context=table_context,
                reasons=reasons,
                debug={"matched_columns": matched_columns, "matched_rows": matched_rows},
            )

        reasons["fallback"].append("lookup_evidence_not_found")
        return self._fallback_payload(table, signal, table_context, reasons, matched_columns, matched_rows)

    def _fallback_payload(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        table_context: TableContext,
        reasons: Dict[str, List[str]],
        matched_columns: List[str],
        matched_rows: List[str],
    ) -> Dict[str, Any]:
        return self._build_payload(
            table=table,
            decision="fallback_table_context",
            operation_hint=signal["operation_hint"],
            confidence=0.34 + min(0.16, 0.04 * len(matched_columns) + 0.035 * len(matched_rows)),
            final_evidence=None,
            candidate_evidence=None,
            table_context=table_context,
            reasons=reasons,
            debug={"matched_columns": matched_columns, "matched_rows": matched_rows},
        )

    def _build_payload(
        self,
        *,
        table: Dict[str, Any],
        decision: str,
        operation_hint: str,
        confidence: float,
        final_evidence: Optional[TableFinalEvidence],
        candidate_evidence: Optional[TableCandidateEvidence],
        table_context: TableContext,
        reasons: Dict[str, List[str]],
        debug: Dict[str, Any],
    ) -> Dict[str, Any]:
        table_info = TableEvidenceTable(
            table_id=str(table.get("table_id") or ""),
            caption=str(table.get("caption") or ""),
            page_number=table.get("page_number"),
            section_path=str(table.get("section_path") or ""),
            section_title=str(table.get("section_title") or ""),
            source_chunk_id=str(table.get("source_chunk_id") or ""),
            original_chunk_id=str(table.get("original_chunk_id") or ""),
        )
        # TableEvidencePayload.to_dict 会立即校验 v2 不变量；坏证据应尽早失败，不做旧字段兼容。
        return TableEvidencePayload(
            table=table_info,
            decision=decision,
            operation_hint=operation_hint,
            confidence=confidence,
            final_evidence=final_evidence,
            candidate_evidence=candidate_evidence,
            table_context=table_context,
            reasons=reasons,
            debug=debug,
        ).to_dict()

    def _resolve_metric_columns(
        self,
        table: Dict[str, Any],
        signal: Dict[str, Any],
        matched_columns: List[str],
    ) -> Tuple[List[str], str, List[str]]:
        numeric_columns = self._numeric_metric_columns(table)
        if not numeric_columns:
            return [], "missing", ["numeric_metric_columns_missing"]

        query_text = signal["normalized_query"]
        query_tokens = set(signal["tokens"])
        explicit_matches: List[str] = []
        for column in numeric_columns:
            normalized_column = self._normalize_metric_name(column)
            if normalized_column and self._metric_name_explicitly_mentioned(normalized_column, column, query_text, query_tokens):
                explicit_matches.append(column)

        if explicit_matches:
            return self._dedupe(explicit_matches), "clear" if len(explicit_matches) == 1 else "ambiguous", ["explicit_metric_column_match"]

        alias_matches: List[str] = []
        for alias_group in CONSERVATIVE_METRIC_ALIASES.values():
            if not any(self._term_present(query_text, query_tokens, alias) for alias in alias_group):
                continue
            for column in numeric_columns:
                normalized_column = self._normalize_metric_name(column)
                if any(self._normalize_metric_name(alias) == normalized_column for alias in alias_group):
                    alias_matches.append(column)
        if alias_matches:
            deduped = self._dedupe(alias_matches)
            return deduped, "clear" if len(deduped) == 1 else "ambiguous", ["conservative_metric_alias_match"]

        matched_numeric = [column for column in matched_columns if column in numeric_columns and not self._is_generic_metric_column_match(column, signal)]
        if matched_numeric:
            deduped = self._dedupe(matched_numeric)
            return deduped, "clear" if len(deduped) == 1 else "ambiguous", ["matched_numeric_column"]

        if len(numeric_columns) == 1:
            return numeric_columns, "clear", ["single_numeric_metric_column"]
        return [], "ambiguous", ["metric_column_not_specified"]

    def _difference_candidate_calculations(
        self,
        table: Dict[str, Any],
        focus_row: str,
        reference_row: str,
        candidate_columns: Sequence[str],
    ) -> List[TableCandidateCalculation]:
        calculations: List[TableCandidateCalculation] = []
        for column in candidate_columns:
            focus_cell = self._find_row_column_cell(table, focus_row, column)
            reference_cell = self._find_row_column_cell(table, reference_row, column)
            if not self._cell_has_numeric_value(focus_cell) or not self._cell_has_numeric_value(reference_cell):
                continue
            calculation = self._difference_calculation(reference_cell, focus_cell)
            calculations.append(
                TableCandidateCalculation(
                    operation="difference",
                    column=column,
                    rows=[focus_row, reference_row],
                    cells=[TableEvidenceCell.from_cell(reference_cell), TableEvidenceCell.from_cell(focus_cell)],
                    calculation=calculation,
                    reason="candidate_difference_same_metric_column",
                )
            )
        return calculations

    def _extreme_candidate_evidence(
        self,
        table: Dict[str, Any],
        operation_hint: str,
        candidate_columns: Sequence[str],
        candidate_rows: Sequence[str],
    ) -> TableCandidateEvidence:
        calculations: List[TableCandidateCalculation] = []
        candidate_cells: List[TableEvidenceCell] = []
        reverse = operation_hint == "max"
        for column in candidate_columns:
            numeric_cells = self._collect_numeric_cells(table, [column], candidate_rows, baseline_only=False)
            if not numeric_cells:
                continue
            best_cell = sorted(numeric_cells, key=lambda item: float(item.get("normalized_value") or 0.0), reverse=reverse)[0]
            cell = TableEvidenceCell.from_cell(best_cell)
            candidate_cells.append(cell)
            calculations.append(
                TableCandidateCalculation(
                    operation=operation_hint,
                    column=column,
                    rows=[str(best_cell.get("row_label") or "").strip()],
                    cells=[cell],
                    calculation=TableEvidenceCalculation(
                        operation=operation_hint,
                        value=best_cell.get("normalized_value"),
                        display_value=self._display_cell_value(best_cell),
                        unit=str(best_cell.get("unit") or ""),
                        expression=self._cell_label(best_cell),
                    ),
                    reason="candidate_extreme_value_for_metric_column",
                )
            )
        return TableCandidateEvidence(
            candidate_rows=[str(row) for row in candidate_rows if str(row).strip()][:6],
            candidate_columns=[str(column) for column in candidate_columns if str(column).strip()],
            candidate_cells=candidate_cells,
            candidate_calculations=calculations,
        )

    def _build_table_context(self, table: Dict[str, Any], *, focus_rows: Sequence[str]) -> TableContext:
        columns = [str(item).strip() for item in (table.get("columns") or []) if str(item).strip()]
        rows_by_index: Dict[Any, Dict[str, Any]] = {}
        for cell in table.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            row_index = cell.get("row_index")
            row_label = str(cell.get("row_label") or "").strip()
            row = rows_by_index.setdefault(row_index, {"row_index": row_index, "row_label": row_label, "cells": {}})
            if row_label and not row.get("row_label"):
                row["row_label"] = row_label
            col_name = str(cell.get("col_name") or "").strip()
            if col_name:
                row["cells"][col_name] = TableEvidenceCell.from_cell(cell).to_dict()
        rows = list(rows_by_index.values())
        rows.sort(key=lambda item: (self._row_sort_key(item.get("row_index")), str(item.get("row_label") or "")))

        short_cell_limit = max(1, int(self.config.get("table_structured_context_short_table_cell_limit", 48)))
        max_rows = max(1, int(self.config.get("table_structured_context_max_rows", 12)))
        focus_window = max(0, int(self.config.get("table_structured_context_focus_window", 2)))
        total_cells = max(1, len(rows)) * max(1, len(columns))
        if total_cells <= short_cell_limit:
            return TableContext(columns=columns, rows=rows, truncated=False, row_count=len(rows), column_count=len(columns))

        # 中长表只给命中行附近窗口，但仍保留完整列名和 truncated 标记，让 LLM 知道上下文被裁剪过。
        focus_set = {str(row).strip() for row in focus_rows if str(row).strip()}
        focus_indexes = [idx for idx, row in enumerate(rows) if str(row.get("row_label") or "").strip() in focus_set]
        selected_indexes = set()
        if focus_indexes:
            for idx in focus_indexes:
                start = max(0, idx - focus_window)
                end = min(len(rows), idx + focus_window + 1)
                selected_indexes.update(range(start, end))
        else:
            selected_indexes.update(range(min(max_rows, len(rows))))
        selected_rows = [rows[idx] for idx in sorted(selected_indexes)[:max_rows]]
        return TableContext(columns=columns, rows=selected_rows, truncated=True, row_count=len(rows), column_count=len(columns))

    def _numeric_metric_columns(self, table: Dict[str, Any]) -> List[str]:
        columns = [str(item).strip() for item in (table.get("columns") or []) if str(item).strip()]
        numeric_columns: List[str] = []
        for column in columns:
            normalized = self._normalize_metric_name(column)
            if any(term == normalized or term in normalized.split() for term in LABEL_COLUMN_TERMS):
                continue
            has_numeric = any(
                isinstance(cell, dict)
                and str(cell.get("col_name") or "").strip() == column
                and cell.get("normalized_value") is not None
                for cell in table.get("cells", []) or []
            )
            if has_numeric:
                numeric_columns.append(column)
        return numeric_columns

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
            if not isinstance(cell, dict) or cell.get("normalized_value") is None:
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
            # 如果 query 明确写了 without/remove 的组件，优先找包含组件的消融行，而不是拿任意命中行计算。
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
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else None

    def _find_reference_row_for_difference(
        self,
        table: Dict[str, Any],
        focus_row: Optional[str],
        signal: Dict[str, Any],
        matched_rows: List[str],
    ) -> Optional[str]:
        if not focus_row:
            return None
        row_labels = [row for row in self._collect_row_labels(table) if row != focus_row]
        if matched_rows:
            preferred = [row for row in matched_rows if row != focus_row]
            if preferred:
                row_labels = preferred + [row for row in row_labels if row not in preferred]
        component_terms = signal.get("component_terms", []) or []
        all_rows = self._collect_row_labels(table)
        focus_index = all_rows.index(focus_row) if focus_row in all_rows else -1
        scored: List[Tuple[float, str]] = []
        for row in row_labels:
            normalized = self._normalize_text(row)
            score = 0.0
            if any(term in normalized for term in ("ours", "our", "proposed", "full", "complete", "完整")):
                score += 0.58
            elif any(term in normalized for term in ("baseline", "base", "default", "原始")):
                score += 0.42
            if any(term in normalized for term in component_terms) and not any(term in normalized for term in NEGATIVE_ROW_TERMS):
                score += 0.15
            # 消融表常把 w/o 行放在完整模型下一行；相邻参考行是高置信线索，但不能替代显式参考词。
            row_index = all_rows.index(row) if row in all_rows else -1
            if focus_index > 0 and row_index == focus_index - 1:
                score += 0.2
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else None

    def _find_row_column_cell(self, table: Dict[str, Any], row_label: str, col_name: str) -> Optional[Dict[str, Any]]:
        for cell in table.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            if str(cell.get("row_label", "") or "").strip() == str(row_label or "").strip() and str(cell.get("col_name", "") or "").strip() == str(col_name or "").strip():
                return cell
        return None

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

        # 结构化表存在但找不到原 chunk 时构造最小 table chunk，保证调试仍能看到表格证据来源。
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

    def _candidate_debug_row(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        evidence = candidate.get("table_evidence") if isinstance(candidate.get("table_evidence"), dict) else {}
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
            "decision": evidence.get("decision"),
            "operation_hint": evidence.get("operation_hint"),
            "matched_columns": candidate.get("matched_columns", [])[:4],
            "matched_rows": candidate.get("matched_rows", [])[:4],
            "matched_cells": self._evidence_cells(evidence)[:3],
            "reason": candidate.get("reason", ""),
            "reasons": evidence.get("reasons", {}),
        }

    @staticmethod
    def _asset_section_anchor_allowed(table: Dict[str, Any]) -> bool:
        """section_path 只有在显式可信时参与匹配，避免 OCR 锚点误导表格召回。"""
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
                    refs.append(f"表 {label}")
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

    def _difference_calculation(self, reference_cell: Dict[str, Any], focus_cell: Dict[str, Any]) -> TableEvidenceCalculation:
        difference = float(reference_cell["normalized_value"]) - float(focus_cell["normalized_value"])
        display_value, display_unit = self._display_difference(reference_cell, focus_cell, difference)
        column = str(reference_cell.get("col_name") or focus_cell.get("col_name") or "").strip()
        expression = f"{reference_cell.get('row_label') or 'reference'} / {column} - {focus_cell.get('row_label') or 'focus'} / {column}".strip()
        return TableEvidenceCalculation(
            operation="difference",
            value=round(difference, 6),
            display_value=display_value,
            unit=display_unit,
            expression=expression,
        )

    def _display_difference(self, reference_cell: Dict[str, Any], focus_cell: Dict[str, Any], difference: float) -> Tuple[str, str]:
        units = {str(reference_cell.get("unit") or "").strip().lower(), str(focus_cell.get("unit") or "").strip().lower()}
        raw_joined = f"{reference_cell.get('raw_value') or ''} {focus_cell.get('raw_value') or ''}"
        if "percent" in units or "%" in raw_joined:
            return f"{self._format_number(difference * 100.0, decimals=1)} percentage points", "percentage_points"
        return self._format_number(difference, decimals=6), str(reference_cell.get("unit") or focus_cell.get("unit") or "")

    @staticmethod
    def _display_cell_value(cell: Dict[str, Any]) -> str:
        raw = str(cell.get("raw_value") if cell.get("raw_value") is not None else "").strip()
        if raw:
            return raw
        return TableStructuredRetriever._format_number(cell.get("normalized_value"), decimals=6)

    @staticmethod
    def _format_number(value: Any, *, decimals: int) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value or "")
        text = f"{round(number, decimals):.{decimals}f}".rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _cell_label(cell: Dict[str, Any]) -> str:
        row = str(cell.get("row_label") or "").strip()
        column = str(cell.get("col_name") or "").strip()
        return " / ".join(part for part in (row, column) if part)

    @staticmethod
    def _cell_has_numeric_value(cell: Optional[Dict[str, Any]]) -> bool:
        return isinstance(cell, dict) and cell.get("normalized_value") is not None

    def _metric_name_explicitly_mentioned(self, normalized_column: str, raw_column: str, query_text: str, query_tokens: set[str]) -> bool:
        if not normalized_column:
            return False
        raw_normalized = self._normalize_text(raw_column)
        if raw_normalized and raw_normalized in query_text:
            return True
        if normalized_column in query_tokens:
            return True
        compact = normalized_column.replace(" ", "")
        return bool(compact and compact in query_text.replace(" ", ""))

    def _is_generic_metric_column_match(self, column: str, signal: Dict[str, Any]) -> bool:
        normalized_column = self._normalize_metric_name(column)
        query_text = signal["normalized_query"]
        return normalized_column in GENERIC_METRIC_TERMS or not self._metric_name_explicitly_mentioned(
            normalized_column,
            column,
            query_text,
            set(signal["tokens"]),
        )

    def _normalize_metric_name(self, text: Any) -> str:
        normalized = self._normalize_text(text)
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized).strip()
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def _term_present(query_text: str, query_tokens: set[str], term: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(term or "").lower()).strip()
        if not normalized:
            return False
        return normalized in query_tokens or normalized in query_text

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
        return [term for term in terms if term and term in normalized]

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
    def _dedupe(items: Sequence[str]) -> List[str]:
        deduped: List[str] = []
        for item in items:
            normalized = str(item or "").strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    @staticmethod
    def _empty_reasons() -> Dict[str, List[str]]:
        return {"matched": [], "ambiguity": [], "fallback": []}

    @staticmethod
    def _row_sort_key(value: Any) -> Tuple[int, Any]:
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value or ""))

    @staticmethod
    def _evidence_cells(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(evidence, dict):
            return []
        final = evidence.get("final_evidence") if isinstance(evidence.get("final_evidence"), dict) else {}
        if final.get("cells"):
            return [cell for cell in final.get("cells") or [] if isinstance(cell, dict)]
        candidate = evidence.get("candidate_evidence") if isinstance(evidence.get("candidate_evidence"), dict) else {}
        return [cell for cell in candidate.get("candidate_cells") or [] if isinstance(cell, dict)]
