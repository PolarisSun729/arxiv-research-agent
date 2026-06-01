from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schemas import AgentStep, AgentToolCall, ArxivSearchSpec


class AgentState(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    intent: Optional[str] = None
    intent_source: Optional[str] = None
    fallback_reason: Optional[str] = None
    llm_confidence: Optional[float] = None
    search_spec: Optional[ArxivSearchSpec] = None
    preference_action_result: Optional[Dict[str, Any]] = None
    debug: Dict[str, Any] = Field(default_factory=dict)
    plan: List[str] = Field(default_factory=list)
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    answer: Optional[str] = None
    next_actions: List[str] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    personalized_rerank_applied: bool = False
