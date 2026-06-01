from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


@lru_cache(maxsize=1)
def get_valid_arxiv_categories() -> Set[str]:
    """
    Load arXiv subject categories from the project's existing taxonomy source.

    This keeps the agent aligned with the repository's own category list instead
    of inventing a separate taxonomy.
    """
    try:
        from services.arxiv.arxiv_search_service import ArxivSearchService

        categories: Set[str] = set()
        for item in ArxivSearchService.get_subject_categories():
            code = str((item or {}).get("code", "")).strip()
            if code:
                categories.add(code)
        if categories:
            return categories
    except Exception:
        pass

    # Conservative fallback if the service import is unavailable.
    return {"cs.AI", "cs.CL", "cs.IR", "cs.LG"}


@lru_cache(maxsize=1)
def get_default_agent_arxiv_categories() -> List[str]:
    """
    Return the fixed category scope for the arXiv search agent.

    The agent should search within the repository-configured category set rather
    than asking the LLM to invent or choose categories per request.
    """
    valid_categories = get_valid_arxiv_categories()

    try:
        from utils.config import get_arxiv_oai_runtime_config

        configured_categories = [
            str(item).strip()
            for item in get_arxiv_oai_runtime_config().get("target_categories", [])
            if str(item).strip()
        ]
    except Exception:
        configured_categories = []

    normalized: List[str] = []
    seen = set()
    for category in configured_categories:
        if category in valid_categories and category not in seen:
            seen.add(category)
            normalized.append(category)

    if normalized:
        return normalized

    return [category for category in ("cs.CL", "cs.LG", "cs.IR", "cs.AI") if category in valid_categories]


class ArxivSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: str

    @field_validator("user_id", "session_id", "message", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text

    @model_validator(mode="after")
    def _validate_message(self) -> "ArxivSearchRequest":
        if not self.message or not str(self.message).strip():
            raise ValueError("message cannot be empty")
        self.message = str(self.message).strip()
        return self


class ArxivSearchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["arxiv_search"]
    query: Optional[str] = None
    title_query: Optional[str] = None
    abstract_query: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    submitted_days_ago: Optional[int] = None
    max_results: int = 10
    sort_by: str = "submittedDate"
    sort_order: str = "descending"
    field_operator: str = "AND"
    category_operator: str = "OR"
    reasoning_summary: Optional[str] = None

    @field_validator("query", "title_query", "abstract_query", "reasoning_summary", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("categories", mode="before")
    @classmethod
    def _normalize_categories(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raise ValueError("categories must be a list of category codes")

        valid_categories = get_valid_arxiv_categories()
        canonical_map = {category.lower(): category for category in valid_categories}
        normalized: List[str] = []
        invalid: List[str] = []
        seen = set()
        for item in raw_items:
            code = str(item).strip()
            if not code:
                continue
            canonical_code = canonical_map.get(code.lower())
            if canonical_code is None:
                invalid.append(code)
                continue
            if canonical_code not in seen:
                seen.add(canonical_code)
                normalized.append(canonical_code)

        if invalid:
            raise ValueError(f"invalid arxiv categories: {', '.join(invalid)}")
        return normalized

    @field_validator("submitted_days_ago", mode="before")
    @classmethod
    def _normalize_submitted_days_ago(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        days = int(value)
        if days < 0:
            raise ValueError("submitted_days_ago must be greater than or equal to 0")
        return days

    @field_validator("max_results", mode="before")
    @classmethod
    def _normalize_max_results(cls, value: Any) -> Any:
        if value is None or value == "":
            return 10
        results = int(value)
        if not (1 <= results <= 20):
            raise ValueError("max_results must be between 1 and 20")
        return results

    @field_validator("sort_by", mode="before")
    @classmethod
    def _normalize_sort_by(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "submittedDate"
        if text not in {"relevance", "lastUpdatedDate", "submittedDate"}:
            raise ValueError("sort_by must be one of relevance, lastUpdatedDate, submittedDate")
        return text

    @field_validator("sort_order", mode="before")
    @classmethod
    def _normalize_sort_order(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "descending"
        if text not in {"ascending", "descending"}:
            raise ValueError("sort_order must be one of ascending, descending")
        return text

    @field_validator("field_operator", mode="before")
    @classmethod
    def _normalize_field_operator(cls, value: Any) -> str:
        text = str(value or "").strip().upper()
        if not text:
            return "AND"
        if text not in {"AND", "OR", "ANDNOT"}:
            raise ValueError("field_operator must be one of AND, OR, ANDNOT")
        return text

    @field_validator("category_operator", mode="before")
    @classmethod
    def _normalize_category_operator(cls, value: Any) -> str:
        text = str(value or "").strip().upper()
        if not text:
            return "OR"
        if text not in {"AND", "OR"}:
            raise ValueError("category_operator must be one of AND, OR")
        return text

    @model_validator(mode="after")
    def _validate_search_fields(self) -> "ArxivSearchSpec":
        if not any([self.query, self.title_query, self.abstract_query, self.categories]):
            raise ValueError("at least one search field or category must be provided")
        return self


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: str
    summary: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    status: Literal["success", "failed", "skipped"]
    action: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class AgentStreamEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal[
        "run_start",
        "step_start",
        "step_end",
        "tool_call_start",
        "tool_call_end",
        "final_response",
        "exception",
        "stream_end",
    ]
    sequence: int
    run_id: str
    timestamp: str
    data: Dict[str, Any] = Field(default_factory=dict)


class ArxivSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "arxiv_search",
        "paper_detail",
        "paper_summary",
        "paper_qa",
        "recommendation",
        "preference_action",
        "reading_list_action",
        "unclear",
        "unsupported",
    ]
    intent_source: Optional[str] = None
    fallback_reason: Optional[str] = None
    llm_confidence: Optional[float] = None
    answer: str
    search_spec: Optional[ArxivSearchSpec] = None
    plan: List[str] = Field(default_factory=list)
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)
