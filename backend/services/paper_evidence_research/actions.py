from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class SearchPaperAction(BaseModel):
    action: Literal["search_paper"]
    target_need_id: str
    target_claim_id: str | None = None
    objective: Literal["discover", "verify_claim", "resolve_conflict", "check_absence"]
    retrieval_mode: Literal["broad", "definition", "method", "experiment", "comparison", "numeric"]
    query: str
    section_hints: list[str] = Field(default_factory=list)
    reason_code: str


class DraftAnswerAction(BaseModel):
    action: Literal["draft_answer"]
    addressed_need_ids: list[str] = Field(default_factory=list)
    reason_code: str


class FinalizeAnswerAction(BaseModel):
    action: Literal["finalize_answer"]
    draft_id: str | None = None
    reason_code: str


class AbstainAction(BaseModel):
    action: Literal["abstain"]
    reason_code: str


ResearchAction = Annotated[
    Union[SearchPaperAction, DraftAnswerAction, FinalizeAnswerAction, AbstainAction],
    Field(discriminator="action"),
]

_ACTION_ADAPTER = TypeAdapter(ResearchAction)


def parse_research_action(value: Any) -> ResearchAction:
    return _ACTION_ADAPTER.validate_python(value)
