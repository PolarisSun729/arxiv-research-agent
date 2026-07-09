from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


TABLE_EVIDENCE_SCHEMA_VERSION = "table_evidence_v2"
TABLE_EVIDENCE_DECISIONS = {"compute", "defer_to_llm", "fallback_table_context"}
TABLE_EVIDENCE_OPERATIONS = {"lookup", "max", "min", "difference"}


@dataclass
class TableEvidenceCell:
    row_index: Any = None
    row_label: str = ""
    col_name: str = ""
    raw_value: Any = None
    normalized_value: Any = None
    unit: Any = None
    confidence: Any = None

    @classmethod
    def from_cell(cls, cell: Dict[str, Any]) -> "TableEvidenceCell":
        return cls(
            row_index=cell.get("row_index"),
            row_label=str(cell.get("row_label") or ""),
            col_name=str(cell.get("col_name") or ""),
            raw_value=cell.get("raw_value"),
            normalized_value=cell.get("normalized_value"),
            unit=cell.get("unit"),
            confidence=cell.get("confidence"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableEvidenceCalculation:
    operation: str
    value: Any
    display_value: str
    unit: str = ""
    expression: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableEvidenceTable:
    table_id: str = ""
    caption: str = ""
    page_number: Any = None
    section_path: str = ""
    section_title: str = ""
    source_chunk_id: str = ""
    original_chunk_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableFinalEvidence:
    operation: str
    rows: List[str]
    columns: List[str]
    cells: List[TableEvidenceCell]
    calculation: Optional[TableEvidenceCalculation] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["cells"] = [cell.to_dict() for cell in self.cells]
        payload["calculation"] = self.calculation.to_dict() if self.calculation else None
        return payload


@dataclass
class TableCandidateCalculation:
    operation: str
    column: str
    rows: List[str]
    cells: List[TableEvidenceCell]
    calculation: TableEvidenceCalculation
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["cells"] = [cell.to_dict() for cell in self.cells]
        payload["calculation"] = self.calculation.to_dict()
        return payload


@dataclass
class TableCandidateEvidence:
    candidate_rows: List[str] = field(default_factory=list)
    candidate_columns: List[str] = field(default_factory=list)
    candidate_cells: List[TableEvidenceCell] = field(default_factory=list)
    candidate_calculations: List[TableCandidateCalculation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["candidate_cells"] = [cell.to_dict() for cell in self.candidate_cells]
        payload["candidate_calculations"] = [item.to_dict() for item in self.candidate_calculations]
        return payload


@dataclass
class TableContext:
    columns: List[str]
    rows: List[Dict[str, Any]]
    truncated: bool
    row_count: int = 0
    column_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TableEvidencePayload:
    table: TableEvidenceTable
    decision: str
    operation_hint: str
    confidence: float
    final_evidence: Optional[TableFinalEvidence]
    candidate_evidence: Optional[TableCandidateEvidence]
    table_context: TableContext
    reasons: Dict[str, List[str]]
    debug: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = TABLE_EVIDENCE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "table": self.table.to_dict(),
            "decision": self.decision,
            "operation_hint": self.operation_hint,
            "confidence": round(float(self.confidence or 0.0), 4),
            "final_evidence": self.final_evidence.to_dict() if self.final_evidence else None,
            "candidate_evidence": self.candidate_evidence.to_dict() if self.candidate_evidence else None,
            "table_context": self.table_context.to_dict(),
            "reasons": {key: list(value or []) for key, value in dict(self.reasons or {}).items()},
            "debug": dict(self.debug or {}),
        }
        validate_table_evidence_payload(payload)
        return payload


def validate_table_evidence_payload(payload: Dict[str, Any]) -> None:
    """校验 v2 证据的决策不变量，避免旧 schema 或半成品 payload 静默流入生成链路。"""
    if not isinstance(payload, dict):
        raise ValueError("table_evidence must be a dict")
    if payload.get("schema_version") != TABLE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("table_evidence schema_version must be table_evidence_v2")

    decision = str(payload.get("decision") or "").strip()
    operation_hint = str(payload.get("operation_hint") or "").strip()
    if decision not in TABLE_EVIDENCE_DECISIONS:
        raise ValueError(f"invalid table_evidence decision: {decision}")
    if operation_hint not in TABLE_EVIDENCE_OPERATIONS:
        raise ValueError(f"invalid table_evidence operation_hint: {operation_hint}")

    for key in ("table", "table_context", "reasons", "debug"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"table_evidence.{key} must be a dict")
    if not isinstance(payload["table_context"].get("columns"), list):
        raise ValueError("table_evidence.table_context.columns must be a list")
    if "truncated" not in payload["table_context"]:
        raise ValueError("table_evidence.table_context.truncated is required")

    final_evidence = payload.get("final_evidence")
    candidate_evidence = payload.get("candidate_evidence")
    if decision == "compute":
        if not isinstance(final_evidence, dict):
            raise ValueError("compute table_evidence requires final_evidence")
        final_operation = str(final_evidence.get("operation") or "").strip()
        if final_operation not in TABLE_EVIDENCE_OPERATIONS:
            raise ValueError(f"invalid final_evidence operation: {final_operation}")
        if not isinstance(final_evidence.get("cells"), list):
            raise ValueError("final_evidence.cells must be a list")
    elif decision == "defer_to_llm":
        if final_evidence is not None:
            raise ValueError("defer_to_llm table_evidence must not include final_evidence")
        if not isinstance(candidate_evidence, dict):
            raise ValueError("defer_to_llm table_evidence requires candidate_evidence")
    elif decision == "fallback_table_context":
        if final_evidence is not None:
            raise ValueError("fallback_table_context must not include final_evidence")
