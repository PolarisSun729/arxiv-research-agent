"""线上记录和离线运行器共用的评测契约；运行失败与业务拒答严格分离。"""

from typing import Any, Literal

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

EVAL_SCHEMA_VERSION = "paper_evidence_eval_v1"
METRICS_VERSION = "claim_metrics_v1"


class GoldenCase(BaseModel):
    case_id: str = Field(min_length=1)
    arxiv_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    difficulty: str = "unknown"
    main_intent: str = "unknown"
    language: str = "zh"
    expected_answer_points: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] | None = None
    answerable: StrictBool | None = None
    gold_answer: str | None = None

    @field_validator("case_id", "arxiv_id", "question")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("字段不能仅包含空白")
        return value.strip()

    @field_validator("expected_answer_points", "expected_chunk_ids")
    @classmethod
    def normalize_labels(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if any(not value.strip() for value in values):
            raise ValueError("标注项不能为空")
        return list(dict.fromkeys(value.strip() for value in values))

    def missing_annotations(self) -> list[str]:
        missing = [name for name in ("expected_chunk_ids", "answerable", "gold_answer") if getattr(self, name) is None]
        if self.answerable is True:
            if not self.expected_chunk_ids and "expected_chunk_ids" not in missing:
                missing.append("expected_chunk_ids")
            if not self.expected_answer_points:
                missing.append("expected_answer_points")
            if not (self.gold_answer or "").strip() and "gold_answer" not in missing:
                missing.append("gold_answer")
        return missing


class EvaluationRecord(BaseModel):
    schema_version: Literal["paper_evidence_eval_v1"] = EVAL_SCHEMA_VERSION
    record_id: str
    timestamp: str
    app_version: str
    run_status: Literal["success", "error"]
    user_id: str
    session_id: str
    turn_id: str = ""
    paper_context: dict[str, Any]
    configuration: dict[str, Any] = Field(default_factory=dict)
    query: dict[str, Any]
    outcome: Literal["completed", "partial", "abstained"] | None = None
    answer: str = ""
    termination_reason: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    research_summary: dict[str, Any] = Field(default_factory=dict)
    efficiency: dict[str, Any] = Field(default_factory=dict)
    trace_ref: str
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_run_outcome(self):
        # 技术错误没有研究终态；否则超时会被错误地算成一次正确拒答。
        if self.run_status == "error" and (self.outcome is not None or self.error is None):
            raise ValueError("运行失败必须携带 error，且不能携带业务 outcome")
        if self.run_status == "success" and (self.outcome is None or self.error is not None):
            raise ValueError("成功运行必须有明确研究终态，且不能携带 error")
        return self
