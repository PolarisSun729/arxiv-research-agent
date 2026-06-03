from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreferenceSummary:
    user_id: str
    liked_papers: List[str] = field(default_factory=list)
    disliked_papers: List[str] = field(default_factory=list)
    paper_actions: Dict[str, List[str]] = field(default_factory=dict)
    interest_vector: Optional[Dict[str, Any]] = None
    counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperChatHistory:
    user_id: str
    arxiv_id: str
    requested_session_id: Optional[str] = None
    selected_session: Optional[Dict[str, Any]] = None
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    message_limit: int = 5
    total_messages: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackendMemorySnapshot:
    user_profile: Optional[Dict[str, Any]] = None
    preference_summary: Optional[Dict[str, Any]] = None
    paper_notes: List[Dict[str, Any]] = field(default_factory=list)
    paper_chat_history: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSessionMemory:
    session_id: Optional[str] = None
    agent_session: Optional[Dict[str, Any]] = None
    backend_memory: Dict[str, Any] = field(default_factory=dict)
    merged_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryDebugPayload:
    user_id: str
    arxiv_id: Optional[str] = None
    loaded_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    frontend_context_keys: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
