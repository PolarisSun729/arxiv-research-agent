from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from utils.config import get_default_user_id


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    summary: str
    data: Optional[Dict[str, Any]] = None
    trace: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


class SearchArxivRawInput(BaseModel):
    search_query: Optional[str] = None
    id_list: Optional[List[str]] = None
    max_results: int = 10
    start: int = 0
    sort_by: str = "relevance"
    sort_order: str = "descending"
    submitted_days_ago: Optional[int] = None


class SearchArxivStructuredInput(BaseModel):
    query: Optional[str] = None
    title_query: Optional[str] = None
    author_query: Optional[str] = None
    abstract_query: Optional[str] = None
    categories: Optional[List[str]] = None
    comment_query: Optional[str] = None
    journal_ref_query: Optional[str] = None
    report_number_query: Optional[str] = None
    id_list: Optional[List[str]] = None
    field_operator: str = "AND"
    category_operator: str = "OR"
    submitted_days_ago: Optional[int] = None
    max_results: int = 10
    start: int = 0
    sort_by: str = "submittedDate"
    sort_order: str = "descending"


class SearchArxivPapersInput(BaseModel):
    search_query: Optional[str] = None
    id_list: Optional[List[str]] = None
    title: Optional[str] = None
    author: Optional[str] = None
    abstract: Optional[str] = None
    category: Optional[str] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    report_number: Optional[str] = None
    operator: str = "AND"
    field_operator: str = "AND"
    category_operator: str = "OR"
    query: Optional[str] = None
    title_query: Optional[str] = None
    author_query: Optional[str] = None
    abstract_query: Optional[str] = None
    categories: Optional[List[str]] = None
    comment_query: Optional[str] = None
    journal_ref_query: Optional[str] = None
    report_number_query: Optional[str] = None
    max_results: int = 10
    start: int = 0
    sort_by: str = "relevance"
    sort_order: str = "descending"
    submitted_days_ago: Optional[int] = None


class GetPaperMetadataInput(BaseModel):
    arxiv_id: str


class RecommendPapersInput(BaseModel):
    user_id: str = Field(default_factory=get_default_user_id)
    top_n: int = 10
    max_age_months: int = 6
    message: Optional[str] = None
    topic_hint: Optional[str] = None
    user_memory_summary: Optional[str] = None
    research_profile: Optional[Dict[str, Any]] = None
    request_context: Optional[Dict[str, Any]] = None


class RecordPaperPreferenceInput(BaseModel):
    user_id: str = Field(default_factory=get_default_user_id)
    arxiv_id: str
    liked: bool
    paper: Optional[Dict[str, Any]] = None


class RemovePaperPreferenceInput(BaseModel):
    user_id: str = Field(default_factory=get_default_user_id)
    arxiv_id: str
    remove_scope: str = "both"


class CheckPaperQAIndexInput(BaseModel):
    arxiv_id: str


class BuildPaperQAIndexInput(BaseModel):
    arxiv_id: str
    loading_method: str = "docling"


class AnswerPaperQuestionInput(BaseModel):
    arxiv_id: str
    question: str
    top_k: Optional[int] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    debug: Optional[bool] = None
