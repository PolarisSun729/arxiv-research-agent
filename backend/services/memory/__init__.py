"""记忆服务模块集合。

该包负责承载与用户记忆相关的核心能力，包括：
- 用户长期研究画像与偏好摘要读取
- Agent 会话态记忆提取与上下文合并
- 论文对话历史、笔记与偏好数据的整理
- 面向调试与观测的记忆快照和调试载荷构造
"""

from services.memory.memory_models import (
    AgentSessionMemory,
    BackendMemorySnapshot,
    MemoryDebugPayload,
    PaperChatHistory,
    PreferenceSummary,
)
from services.memory.memory_service import MemoryService

__all__ = [
    "AgentSessionMemory",
    "BackendMemorySnapshot",
    "MemoryDebugPayload",
    "MemoryService",
    "PaperChatHistory",
    "PreferenceSummary",
]
