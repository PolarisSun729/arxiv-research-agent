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
    - llm_definition: 写进意图识别 prompt 的英文定义，按"用户要 Agent 做什么"描述类别边界，
      模型据此分类，而不是只看标签名去匹配用户用词。
    """

    name: str
    description: str
    parse_selectable: bool
    is_search: bool
    is_non_search_task: bool
    minimum_requirements: Tuple[str, ...] = field(default_factory=tuple)
    llm_definition: str = ""


# 顺序即 LLM prompt 中列出 intent 的顺序，确保 prompt 文案可从配置稳定派生。
_ORDERED_DEFINITIONS: Tuple[IntentDefinition, ...] = (
    IntentDefinition(
        "arxiv_search", "按主题/关键词搜索 arXiv 论文。", True, True, False, ("search_topic",),
        llm_definition="The user specifies what the papers should be about (a topic, task, problem, method or similar) "
        "and wants papers matching it found on arXiv. Search results are also ranked with the user's research profile.",
    ),
    IntentDefinition(
        "paper_detail", "解释某篇具体论文的方法/贡献/章节细节。", True, False, True, ("target_paper",),
        llm_definition="About one specific paper: explain its method, contributions or a particular section.",
    ),
    IntentDefinition(
        "paper_summary", "总结/概述某篇具体论文。", True, False, True, ("target_paper",),
        llm_definition="About one specific paper: give an overall summary of it.",
    ),
    IntentDefinition(
        "paper_qa", "就某篇具体论文提出问题并基于全文回答。", True, False, True, ("target_paper", "user_question"),
        llm_definition="About one specific paper: answer a concrete question about its content from the full text.",
    ),
    IntentDefinition(
        "recommendation", "基于用户画像/兴趣生成个性化论文推荐。", True, False, True, ("user_identity_or_profile_or_constraints",),
        llm_definition="The user does not specify what the papers should be about and leaves the choice to the agent, "
        "which picks papers according to the user's own research profile, interests or reading history.",
    ),
    IntentDefinition(
        "preference_action", "对某篇论文执行喜欢/不喜欢/取消偏好。", True, False, True, ("preference_action", "preference_target", "user_identity"),
        llm_definition="About one specific paper: mark it as liked, mark it as disliked / not interested, or cancel such a mark.",
    ),
    IntentDefinition(
        "unclear", "想要论文相关能力但主题不够明确，需要澄清。", True, False, False, ("task_intent",),
        llm_definition="The user wants papers or paper-related help, but the request is too vague to act on.",
    ),
    IntentDefinition(
        "unsupported", "超出 arXiv 搜索 Agent 能力范围的请求。", True, False, False, tuple(),
        llm_definition="Anything else, including saving/bookmarking papers, reading lists, read-later, listing saved papers, "
        "and requests unrelated to papers.",
    ),
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

# LLM prompt 中逐条列出的 intent 定义，与 LLM_INTENT_CHOICES 同序。
LLM_INTENT_DEFINITIONS: Tuple[Tuple[str, str], ...] = tuple(
    (definition.name, definition.llm_definition) for definition in _ORDERED_DEFINITIONS if definition.parse_selectable
)


__all__ = [
    "IntentDefinition",
    "INTENT_DEFINITIONS",
    "PARSE_SUPPORTED_INTENTS",
    "ALL_INTENTS",
    "NON_SEARCH_INTENTS",
    "MINIMUM_EXECUTION_REQUIREMENTS",
    "LLM_INTENT_CHOICES",
    "LLM_INTENT_DEFINITIONS",
]
