"""Prompt 上下文组装模块。

数据流（论文问答最终回答阶段）::

    PromptContextBuilder           把记忆 / 会话摘要 / 历史对话 / RAG 证据拆成 PromptBlock
        └─ PromptBudgetPlanner
             ├─ RuleCompactor      按 block 类型做结构保真压缩（截断、字段白名单、保留表格单元格）
             ├─ LLMCompactor       可选的语义压缩，默认关闭
             └─ 硬保护 + 分 section 软预算 + 共享池回收，选出最终 blocks
        └─ PromptRenderer          按固定 section 顺序渲染为单段 prompt 文本

TokenCounter 贯穿全程，统一 token 计数口径（有本地 tokenizer 时精确计数，否则保守估算）。
"""

from .blocks import PromptBlock
from .budget_planner import PromptBudgetConfig, PromptBudgetPlan, PromptBudgetPlanner
from .builder import PromptContextBuilder, PromptSection
from .compactor import RuleCompactionConfig, RuleCompactor
from .llm_compactor import LLMCompactionConfig, LLMCompactor
from .renderer import PromptRenderer
from .token_counter import TokenCounter, build_token_counter

__all__ = [
    "PromptBlock",
    "PromptBudgetConfig",
    "PromptBudgetPlan",
    "PromptBudgetPlanner",
    "PromptContextBuilder",
    "PromptSection",
    "RuleCompactionConfig",
    "RuleCompactor",
    "LLMCompactionConfig",
    "LLMCompactor",
    "PromptRenderer",
    "TokenCounter",
    "build_token_counter",
]
