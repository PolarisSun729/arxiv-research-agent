from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreferenceSummary:
    """封装用户偏好摘要，包括点赞/点踩、动作映射与兴趣向量快照。"""
    user_id: str
    liked_papers: List[str] = field(default_factory=list)
    disliked_papers: List[str] = field(default_factory=list)
    paper_actions: Dict[str, List[str]] = field(default_factory=dict)
    interest_vector: Optional[Dict[str, Any]] = None
    counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """把偏好摘要转换为普通字典，便于接口返回与调试输出。"""
        return asdict(self)


@dataclass
class PaperChatHistory:
    """封装单篇论文下的问答会话历史与消息列表。"""
    user_id: str
    arxiv_id: str
    requested_session_id: Optional[str] = None
    selected_session: Optional[Dict[str, Any]] = None
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    message_limit: int = 5
    total_messages: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """把论文会话历史转换为普通字典，便于跨层传递。"""
        return asdict(self)


@dataclass
class BackendMemorySnapshot:
    """封装从后端存储层读取出的记忆快照，用于统一聚合输出。"""
    user_profile: Optional[Dict[str, Any]] = None
    preference_summary: Optional[Dict[str, Any]] = None
    paper_notes: List[Dict[str, Any]] = field(default_factory=list)
    paper_chat_history: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """把后端记忆快照转换为普通字典。"""
        return asdict(self)


@dataclass
class AgentSessionMemory:
    """封装 Agent 会话态记忆、后端记忆与合并后的上下文结果。"""
    session_id: Optional[str] = None
    agent_session: Optional[Dict[str, Any]] = None
    backend_memory: Dict[str, Any] = field(default_factory=dict)
    merged_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """把 Agent 会话记忆结构转换为普通字典。"""
        return asdict(self)


@dataclass
class MemoryDebugPayload:
    """封装面向调试展示的记忆观测载荷。"""
    user_id: str
    arxiv_id: Optional[str] = None
    loaded_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    frontend_context_keys: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """把调试载荷转换为普通字典，便于接口返回。"""
        return asdict(self)
