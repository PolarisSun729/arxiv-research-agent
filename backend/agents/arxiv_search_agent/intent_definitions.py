"""Agent 意图分类的单一事实来源（零依赖叶子模块）。

历史上 `SUPPORTED_INTENTS` 在 node/intent_support、utils/search_spec_builder、
clarification_analysis 三处各维护一份副本，已经发生漂移（clarification 多出
confirmation 运行时确认流意图）；LLM prompt 又把同一批 intent 名单硬编码两处。

本模块把"有哪些 intent、各自属于哪一层、最小可执行条件是什么"集中成一份配置，
其它模块只从这里派生集合，不再各自维护名单。模块刻意零依赖（只用标准库），
可被 node/utils/包根任意模块安全导入，不会引入循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class IntentDefinition:
    """单个 intent 的统一定义。

    - parse_selectable: parse 阶段 LLM/规则意图分类是否可以直接产出该 intent；
      confirmation 是运行时确认流专用意图，parse 阶段不产出，因此为 False。
    - is_search: 是否搜索类意图（决定是否解析结构化 search_spec）。
    - is_non_search_task: 是否属于"非搜索但带任务语义"的意图（旧 NON_SEARCH_INTENTS）。
    - minimum_requirements: 该意图最小可执行条件，供澄清 trace 解释"为何还不能继续"。
    """

    name: str
    description: str
    parse_selectable: bool
    is_search: bool
    is_non_search_task: bool
    minimum_requirements: Tuple[str, ...] = field(default_factory=tuple)


# 顺序即 LLM prompt 中列出 intent 的顺序，确保 prompt 文案可从配置稳定派生。
_ORDERED_DEFINITIONS: Tuple[IntentDefinition, ...] = (
    IntentDefinition("arxiv_search", "按主题/关键词搜索 arXiv 论文。", True, True, False, ("search_topic",)),
    IntentDefinition("paper_detail", "解释某篇具体论文的方法/贡献/章节细节。", True, False, True, ("target_paper",)),
    IntentDefinition("paper_summary", "总结/概述某篇具体论文。", True, False, True, ("target_paper",)),
    IntentDefinition("paper_qa", "就某篇具体论文提出问题并基于全文回答。", True, False, True, ("target_paper", "user_question")),
    IntentDefinition("recommendation", "基于用户画像/兴趣生成个性化论文推荐。", True, False, True, ("user_identity_or_profile_or_constraints",)),
    IntentDefinition("preference_action", "对某篇论文执行喜欢/不喜欢/取消偏好。", True, False, True, ("preference_action", "preference_target", "user_identity")),
    IntentDefinition("unclear", "想要论文相关能力但主题不够明确，需要澄清。", True, False, False, ("task_intent",)),
    IntentDefinition("unsupported", "超出 arXiv 搜索 Agent 能力范围的请求。", True, False, False, tuple()),
    IntentDefinition("confirmation", "运行时等待用户确认/拒绝某个待执行动作。", False, False, False, ("confirmation_request", "confirmation_decision")),
)


INTENT_DEFINITIONS: Dict[str, IntentDefinition] = {definition.name: definition for definition in _ORDERED_DEFINITIONS}

# parse 阶段可产出的 intent 全集（不含运行时 confirmation），用于校验 LLM/规则输出。
PARSE_SUPPORTED_INTENTS = frozenset(definition.name for definition in _ORDERED_DEFINITIONS if definition.parse_selectable)

# 包含运行时确认流在内的 intent 全集，供澄清/确认链路使用。
ALL_INTENTS = frozenset(INTENT_DEFINITIONS)

# 非搜索但带任务语义的 intent（旧 NON_SEARCH_INTENTS）。
NON_SEARCH_INTENTS = frozenset(definition.name for definition in _ORDERED_DEFINITIONS if definition.is_non_search_task)

# 各 intent 最小可执行条件（旧 clarification._MINIMUM_EXECUTION_REQUIREMENTS）。
MINIMUM_EXECUTION_REQUIREMENTS: Dict[str, List[str]] = {
    definition.name: list(definition.minimum_requirements) for definition in _ORDERED_DEFINITIONS
}

# LLM prompt 中"可选 intent"的有序名单，prompt 从这里派生而非硬编码。
LLM_INTENT_CHOICES: Tuple[str, ...] = tuple(definition.name for definition in _ORDERED_DEFINITIONS if definition.parse_selectable)

# LLM 分类的跨意图判别规则，集中维护 prompt guidance 文案。
LLM_CLASSIFICATION_GUIDANCE: Tuple[str, ...] = (
    "If the user is asking to summarize/explain/QA a specific paper, do not classify as arxiv_search.",
    "If the user is expressing like/dislike or cancelling like/dislike about a specific paper, classify as preference_action.",
    "If the user asks to save/bookmark/add to a reading list/read later/list saved papers, classify as unsupported.",
    "If the user is asking for personalized recommendations, classify as recommendation.",
    "If the topic is too vague, classify as unclear.",
)


__all__ = [
    "IntentDefinition",
    "INTENT_DEFINITIONS",
    "PARSE_SUPPORTED_INTENTS",
    "ALL_INTENTS",
    "NON_SEARCH_INTENTS",
    "MINIMUM_EXECUTION_REQUIREMENTS",
    "LLM_INTENT_CHOICES",
    "LLM_CLASSIFICATION_GUIDANCE",
]
