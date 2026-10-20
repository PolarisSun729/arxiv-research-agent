from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator


class _ResearchActionBase(BaseModel):
    # 决策器读完已召回证据全文后标记的无关候选，动作通过门禁后移出证据池。
    # 用"标记丢弃"而不是"列出保留"：模型漏写时证据仍在，不会因输出不全丢掉有用证据。
    discard_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("discard_evidence_ids", mode="before")
    @classmethod
    def _numeric_ids_as_strings(cls, value: Any) -> Any:
        # source_id 多是 "54" 这样的数字串，模型常把引号丢掉写成 54；不能因此判整次决策无效。
        if isinstance(value, list):
            return [str(item) if isinstance(item, int) and not isinstance(item, bool) else item for item in value]
        return value


class SearchPaperAction(_ResearchActionBase):
    action: Literal["search_paper"]
    target_need_id: str
    target_claim_id: str | None = None
    objective: Literal["discover", "verify_claim", "resolve_conflict", "check_absence"]
    retrieval_mode: Literal["broad", "definition", "method", "experiment", "comparison", "numeric"]
    query: str
    section_hints: list[str] = Field(default_factory=list)
    reason_code: str


class DraftAnswerAction(_ResearchActionBase):
    action: Literal["draft_answer"]
    addressed_need_ids: list[str] = Field(default_factory=list)
    reason_code: str


class FinalizeAnswerAction(_ResearchActionBase):
    action: Literal["finalize_answer"]
    draft_id: str | None = None
    reason_code: str


class AbstainAction(_ResearchActionBase):
    action: Literal["abstain"]
    reason_code: str


ResearchAction = Annotated[
    Union[SearchPaperAction, DraftAnswerAction, FinalizeAnswerAction, AbstainAction],
    Field(discriminator="action"),
]

_ACTION_ADAPTER = TypeAdapter(ResearchAction)


def parse_research_action(value: Any) -> ResearchAction:
    return _ACTION_ADAPTER.validate_python(value)
