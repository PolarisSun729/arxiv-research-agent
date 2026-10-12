"""结构化表格路由：在解析好的表格（行 / 列 / 单元格）上直接定位证据。

面向“某指标最高/最低是谁”“去掉某组件后下降多少”“某行某列是多少”这类问题：
- 行列都能唯一确定时，由规则直接算出答案（decision = compute）；
- 存在歧义（如没说清是哪个指标列）时，只给候选单元格与表格上下文，交给 LLM 判断（defer_to_llm）；
- 完全定位不到时，退回提供整表上下文（fallback_table_context）。
规则层宁可保守地交给 LLM，也不把不确定的值伪装成最终答案。
"""

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


# 问题中出现任一词，就认为可能需要查表（触发本路由）。
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

# 常见评测指标名，用于识别问题里点名的指标。
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

# 泛化的“分数/结果”类词：问题里出现它们并不能确定是哪一个指标列。
GENERIC_METRIC_TERMS: Tuple[str, ...] = ("score", "scores", "metric", "metrics", "performance", "result", "results", "best", "指标", "分数", "性能", "结果")
# 这类列名通常是行标签列（模型名、方法名等），不当作数值指标列。
LABEL_COLUMN_TERMS: Tuple[str, ...] = ("model", "method", "approach", "dataset", "setting", "variant", "模型", "方法", "数据集", "设置")
# 以下三组词决定运算类型（优先级：差值 > 最大 > 最小 > 查值）。
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
# 消融行标记（w/o、去掉某组件……）：差值计算时作为被比较的 focus 行。
NEGATIVE_ROW_TERMS: Tuple[str, ...] = ("w/o", "without", "remove", "removed", "no ", "drop ", "ablation", "去掉", "去除", "移除", "无")
# 差值计算的参考行：完整模型/本文方法优先，其次是 baseline/默认设置。
STRONG_REFERENCE_ROW_TERMS: Tuple[str, ...] = ("ours", "our", "proposed", "full", "complete", "完整")
WEAK_REFERENCE_ROW_TERMS: Tuple[str, ...] = ("baseline", "base", "default", "原始")
# 问 baseline 时需要排除的“本文方法”行。
NON_BASELINE_ROW_TERMS: Tuple[str, ...] = ("ours", "our", "proposed", "full model", "本文", "我们")

# 保守的指标别名表：只收录不会误配的同义写法，问题提到别名时才能确定对应指标列。
CONSERVATIVE_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "accuracy": ("accuracy", "acc", "准确率"),
    "f1": ("f1", "f1-score", "f1 score"),
    "bleu": ("bleu",),
    "rouge": ("rouge",),
    "precision": ("precision", "精确率"),
    "recall": ("recall", "召回率"),
    "em": ("em", "exact match"),
}

# 指标名比较前把非字母数字、非中文字符统一视为分隔符（"F1-score" 与 "f1 score" 视为相同）。
_NON_WORD_RE = re.compile(r"[^a-z0-9一-鿿]+")


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
        # 提供文本归一化与分词（normalize_query_text / tokenize_for_keyword_search）。
        self.query_tools = query_tools
        # 下面两个构建器当前未被本类使用（表格路由的置信度由 route_retriever 统一计算），
        # 保留是为了与其它路由检索器的构造参数保持一致。
        self.route_confidence_builder = route_confidence_builder
        self.structural_bonus_builder = structural_bonus_builder
        # 运行期配置：开关、候选表数量、匹配分下限、表格上下文裁剪参数等。
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
        """从问题中提取表格信号：是否触发本路由、运算类型、表号引用、指标词、消融组件词等。"""
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
        """给一张候选表打匹配分，并同时抽取该表的证据（table_evidence）。

        分数由表号引用、caption/章节词重合、行列名命中、消融/数据集语境加分组成，
        最后再叠加证据置信度：能抽出可靠单元格的表排名更靠前。
        """
        columns = self._table_columns(table)
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
        """按运算类型分派到差值 / 极值 / 查值三种证据抽取逻辑。"""
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
        """最大/最小值问题：指标列唯一确定时直接取极值单元格，否则给出每个候选列的极值交给 LLM。"""
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
        final = TableFinalEvidence(
            operation=signal["operation_hint"],
            rows=[str(best_cell.get("row_label") or "").strip()],
            columns=[str(best_cell.get("col_name") or "").strip()],
            cells=[TableEvidenceCell.from_cell(best_cell)],
            calculation=self._single_cell_calculation(signal["operation_hint"], best_cell),
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
        """差值问题（如“去掉 X 后下降多少”）：找到消融行（focus）和参考行（完整模型/baseline），
        指标列唯一时计算 参考值 − 消融值，否则给出各候选列的差值交给 LLM。"""
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
        """查值问题：行、列各唯一命中一个时直接给出单元格，部分命中时给候选，都没命中则退回表格上下文。"""
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
                calculation=self._single_cell_calculation("lookup", primary_cell),
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
        """无法定位具体单元格时只提供表格上下文；置信度随命中的行列数小幅上调。"""
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
        """组装 table_evidence_v2 载荷并序列化为 dict（序列化时会校验 v2 不变量）。"""
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
        """确定问题针对的数值指标列，返回 (列名列表, 状态, 判定原因)。

        判定顺序由严到宽：列名被显式提及 → 保守别名命中 → 行列匹配阶段命中的数值列 → 表中只有一个数值列。
        状态为 clear（唯一）、ambiguous（多个或无法确定）或 missing（表中没有数值列）。
        """
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
        """对每个候选指标列计算 参考行 − focus 行 的差值（两格都有数值才计算）。"""
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
        """指标列不确定时，为每个候选列各找出极值单元格，作为候选证据交给 LLM 选择。"""
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
                    calculation=self._single_cell_calculation(operation_hint, best_cell),
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
        """按行重建表格作为 LLM 上下文；短表给全表，中长表只给命中行上下 focus_window 行（最多 max_rows 行）。"""
        columns = self._table_columns(table)
        rows_by_index: Dict[Any, Dict[str, Any]] = {}
        for cell in self._table_cells(table):
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
        """返回至少有一个数值单元格、且不像行标签列（模型名、方法名等）的列。"""
        cells = self._table_cells(table)
        numeric_columns: List[str] = []
        for column in self._table_columns(table):
            normalized = self._normalize_metric_name(column)
            if any(term == normalized or term in normalized.split() for term in LABEL_COLUMN_TERMS):
                continue
            has_numeric = any(
                str(cell.get("col_name") or "").strip() == column and cell.get("normalized_value") is not None
                for cell in cells
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
        """收集指定行列范围内的数值单元格（行/列集合为空表示不限）；baseline_only 时排除“本文方法”行。"""
        row_set = self._label_set(rows)
        column_set = self._label_set(columns)
        numeric_cells: List[Dict[str, Any]] = []
        for cell in self._table_cells(table):
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
        """收集落在命中行/列上的单元格（不要求是数值）；行列都没命中时返回空。"""
        row_set = self._label_set(rows)
        column_set = self._label_set(columns)
        cells: List[Dict[str, Any]] = []
        for cell in self._table_cells(table):
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
        """找差值计算的 focus 行（通常是消融行）：命中行里的消融行优先，否则按消融标记和组件词给全部行打分。"""
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
        """找差值计算的参考行：完整模型/本文方法 > baseline/默认设置，含被消融组件且非消融行、紧邻 focus 行上方者加分。"""
        if not focus_row:
            return None
        all_rows = self._collect_row_labels(table)
        row_labels = [row for row in all_rows if row != focus_row]
        if matched_rows:
            # 问题里点名的行排在前面，同分时优先被选中。
            preferred = [row for row in matched_rows if row != focus_row]
            if preferred:
                row_labels = preferred + [row for row in row_labels if row not in preferred]
        component_terms = signal.get("component_terms", []) or []
        focus_index = all_rows.index(focus_row) if focus_row in all_rows else -1
        scored: List[Tuple[float, str]] = []
        for row in row_labels:
            normalized = self._normalize_text(row)
            score = 0.0
            if any(term in normalized for term in STRONG_REFERENCE_ROW_TERMS):
                score += 0.58
            elif any(term in normalized for term in WEAK_REFERENCE_ROW_TERMS):
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
        """按行标签和列名精确定位单元格，找不到返回 None。"""
        target_row = str(row_label or "").strip()
        target_column = str(col_name or "").strip()
        for cell in self._table_cells(table):
            if str(cell.get("row_label", "") or "").strip() == target_row and str(cell.get("col_name", "") or "").strip() == target_column:
                return cell
        return None

    def _match_named_items(self, signal: Dict[str, Any], items: Sequence[str]) -> List[str]:
        """在行标签或列名中找问题提到的项：整串出现、token 重合、含指标词分别加分，返回得分最高的 4 个。"""
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
        """找到表格所在的原始 chunk（先按 source_chunk_id，再按 original_chunk_id）；找不到时构造最小 table chunk。"""
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
        original_id = table.get("original_chunk_id") or table.get("source_chunk_id") or table.get("table_id")
        return {
            "content": str(table.get("caption", "") or "").strip(),
            "chunk_id": table.get("source_chunk_id") or table.get("table_id"),
            "parent_chunk_id": original_id,
            "original_chunk_id": original_id,
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
        """按单元格出现顺序返回去重后的非空行标签。"""
        labels = (str(cell.get("row_label", "") or "").strip() for cell in self._table_cells(table))
        return list(dict.fromkeys(label for label in labels if label))

    def _candidate_debug_row(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """候选表的 debug 摘要：匹配分、证据决策、命中行列/单元格以及章节锚点可信度。"""
        evidence = candidate.get("table_evidence") if isinstance(candidate.get("table_evidence"), dict) else {}
        table = candidate["table"]
        return {
            "table_id": table.get("table_id"),
            "caption": table.get("caption", ""),
            "section_path": table.get("section_path", ""),
            "asset_section_match_type": table.get("asset_section_match_type", ""),
            "asset_section_match_confidence": table.get("asset_section_match_confidence", 0.0),
            "asset_section_match_is_heuristic": table.get("asset_section_match_is_heuristic", False),
            "asset_section_match_allow_embedding": table.get("asset_section_match_allow_embedding", False),
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
        """提取 "Table 3" / "表 3" 这类表号引用，展开成 "3"、"table 3"、"表 3" 三种写法便于与 caption 匹配。"""
        refs: List[str] = []
        for pattern in (r"\btable\s*([0-9]+[a-z]?)\b", r"表\s*([0-9]+[a-zA-Z]?)"):
            for match in re.findall(pattern, str(text or ""), flags=re.IGNORECASE):
                label = str(match or "").strip().lower()
                if label:
                    refs.extend([label, f"table {label}", f"表 {label}"])
        return list(dict.fromkeys(refs))

    def _extract_component_terms(self, text: str) -> List[str]:
        """提取 without/w/o/去掉 等词后面的被消融组件名（最多 3 个）。"""
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
        """计算 参考值 − focus 值，并生成可读的表达式（如 "Full / F1 - w/o X / F1"）。"""
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
        """差值的展示文本与单位：百分比类指标的归一化值是小数，换算成“百分点”展示（0.012 → 1.2 percentage points）。"""
        units = {str(reference_cell.get("unit") or "").strip().lower(), str(focus_cell.get("unit") or "").strip().lower()}
        raw_joined = f"{reference_cell.get('raw_value') or ''} {focus_cell.get('raw_value') or ''}"
        if "percent" in units or "%" in raw_joined:
            return f"{self._format_number(difference * 100.0, decimals=1)} percentage points", "percentage_points"
        return self._format_number(difference, decimals=6), str(reference_cell.get("unit") or focus_cell.get("unit") or "")

    def _single_cell_calculation(self, operation: str, cell: Dict[str, Any]) -> TableEvidenceCalculation:
        """由单个单元格直接得出结果的计算记录（查值、极值共用）。"""
        return TableEvidenceCalculation(
            operation=operation,
            value=cell.get("normalized_value"),
            display_value=self._display_cell_value(cell),
            unit=str(cell.get("unit") or ""),
            expression=self._cell_label(cell),
        )

    @staticmethod
    def _display_cell_value(cell: Dict[str, Any]) -> str:
        """单元格展示值：优先用原文（保留 % 等写法），没有原文时格式化数值。"""
        raw = str(cell.get("raw_value") if cell.get("raw_value") is not None else "").strip()
        if raw:
            return raw
        return TableStructuredRetriever._format_number(cell.get("normalized_value"), decimals=6)

    @staticmethod
    def _format_number(value: Any, *, decimals: int) -> str:
        """四舍五入到指定小数位并去掉末尾多余的 0；非数值原样转成字符串。"""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value or "")
        text = f"{round(number, decimals):.{decimals}f}".rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _cell_label(cell: Dict[str, Any]) -> str:
        """单元格的可读位置 "行标签 / 列名"。"""
        row = str(cell.get("row_label") or "").strip()
        column = str(cell.get("col_name") or "").strip()
        return " / ".join(part for part in (row, column) if part)

    @staticmethod
    def _cell_has_numeric_value(cell: Optional[Dict[str, Any]]) -> bool:
        """单元格存在且解析出了数值。"""
        return isinstance(cell, dict) and cell.get("normalized_value") is not None

    def _metric_name_explicitly_mentioned(self, normalized_column: str, raw_column: str, query_text: str, query_tokens: set[str]) -> bool:
        """列名是否被问题显式提及：原列名子串、归一化后的整词 token，或去空格后的紧凑写法任一命中即可。"""
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
        """列名是泛化词（score/result 等）或没有被问题显式提及时，不能凭它确定指标列。"""
        normalized_column = self._normalize_metric_name(column)
        query_text = signal["normalized_query"]
        return normalized_column in GENERIC_METRIC_TERMS or not self._metric_name_explicitly_mentioned(
            normalized_column,
            column,
            query_text,
            set(signal["tokens"]),
        )

    def _normalize_metric_name(self, text: Any) -> str:
        """\u628a\u5217\u540d/\u6307\u6807\u540d\u5f52\u4e00\u6210\u4ee5\u5355\u4e2a\u7a7a\u683c\u5206\u9694\u7684\u5c0f\u5199\u8bcd\u4e32\uff0c\u4fbf\u4e8e\u8de8\u5199\u6cd5\u6bd4\u8f83\u3002"""
        normalized = self._normalize_text(text)
        normalized = _NON_WORD_RE.sub(" ", normalized).strip()
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def _term_present(query_text: str, query_tokens: set[str], term: str) -> bool:
        """\u6307\u6807\u522b\u540d\uff08\u5f52\u4e00\u5316\u540e\uff09\u662f\u5426\u4f5c\u4e3a token \u6216\u5b50\u4e32\u51fa\u73b0\u5728\u95ee\u9898\u4e2d\u3002"""
        normalized = _NON_WORD_RE.sub(" ", str(term or "").lower()).strip()
        if not normalized:
            return False
        return normalized in query_tokens or normalized in query_text

    def _normalize_text(self, text: Any) -> str:
        """小写化文本；query_tools 提供 normalize_query_text 时先用它统一全半角、标点等写法。"""
        normalizer = getattr(self.query_tools, "normalize_query_text", None)
        value = str(text or "").strip()
        if callable(normalizer):
            return str(normalizer(value) or "").strip().lower()
        return value.lower()

    def _tokenize(self, text: str) -> List[str]:
        """优先复用关键词检索的分词器，保证与 BM25 路由的切词一致；缺失时按非单词字符切分。"""
        tokenizer = getattr(self.query_tools, "tokenize_for_keyword_search", None)
        if callable(tokenizer):
            return [str(item).strip().lower() for item in tokenizer(text) if str(item).strip()]
        return [item for item in re.split(r"\W+", str(text or "").lower()) if item]

    @staticmethod
    def _matched_terms(text: str, terms: Sequence[str]) -> List[str]:
        """返回在文本中以子串形式出现的词（保持词表顺序）。"""
        normalized = str(text or "").lower()
        return [term for term in terms if term and term in normalized]

    @staticmethod
    def _has_any(text: str, terms: Sequence[str]) -> bool:
        """文本中是否出现任一词（子串匹配）。"""
        normalized = str(text or "").lower()
        return any(term in normalized for term in terms)

    @staticmethod
    def _overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
        """right 中有多大比例的 token 出现在 left 中（以 right 为分母，衡量目标文本被问题覆盖的程度）。"""
        left_set = {str(item).strip() for item in left if str(item).strip()}
        right_set = {str(item).strip() for item in right if str(item).strip()}
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / max(1, len(right_set))

    @staticmethod
    def _dedupe(items: Sequence[str]) -> List[str]:
        """去空白后保序去重，丢弃空值。"""
        deduped: List[str] = []
        for item in items:
            normalized = str(item or "").strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    @staticmethod
    def _empty_reasons() -> Dict[str, List[str]]:
        """证据判定原因的三个分类：命中依据、歧义来源、降级原因。"""
        return {"matched": [], "ambiguity": [], "fallback": []}

    @staticmethod
    def _table_columns(table: Dict[str, Any]) -> List[str]:
        """表头列名（去空白，丢弃空列名）。"""
        return [str(item).strip() for item in (table.get("columns") or []) if str(item).strip()]

    @staticmethod
    def _table_cells(table: Dict[str, Any]) -> List[Dict[str, Any]]:
        """表格中结构合法（dict）的单元格；解析结果里偶有坏数据，统一在这里过滤。"""
        return [cell for cell in table.get("cells", []) or [] if isinstance(cell, dict)]

    @staticmethod
    def _label_set(labels: Sequence[str]) -> set[str]:
        """行标签/列名集合（去空白、丢弃空值），用于单元格过滤。"""
        return {str(item).strip() for item in labels if str(item).strip()}

    @staticmethod
    def _row_sort_key(value: Any) -> Tuple[int, Any]:
        """行号排序键：数字行号按数值排在前面，非数字行号按字符串排在后面。"""
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value or ""))

    @staticmethod
    def _evidence_cells(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """证据中的单元格：有最终证据时取最终证据的单元格，否则取候选单元格。"""
        if not isinstance(evidence, dict):
            return []
        final = evidence.get("final_evidence") if isinstance(evidence.get("final_evidence"), dict) else {}
        if final.get("cells"):
            return [cell for cell in final.get("cells") or [] if isinstance(cell, dict)]
        candidate = evidence.get("candidate_evidence") if isinstance(evidence.get("candidate_evidence"), dict) else {}
        return [cell for cell in candidate.get("candidate_cells") or [] if isinstance(cell, dict)]
