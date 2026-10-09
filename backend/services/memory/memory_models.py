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
class AgentSessionMemory:
    """封装 Agent 会话态记忆、后端记忆与合并后的上下文结果。"""
    session_id: Optional[str] = None
    agent_session: Optional[Dict[str, Any]] = None
    backend_memory: Dict[str, Any] = field(default_factory=dict)
    merged_context: Dict[str, Any] = field(default_factory=dict)
    context_merge_debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """把 Agent 会话记忆结构转换为普通字典。"""
        return asdict(self)

