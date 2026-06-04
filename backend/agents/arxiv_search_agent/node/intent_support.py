"""意图识别阶段共用的规则、提示词和轻量启发式工具。

这个模块不直接改写 AgentState，而是为 parse_node 提供一套稳定的识别支撑：
1. 定义支持的 intent 集合、硬规则模式和搜索触发词；
2. 提供论文总结、详情、问答、偏好、推荐等请求的启发式判断函数；
3. 负责把研究画像压缩进 prompt，并生成约束 JSON 输出的 LLM 提示词。

输入/输出说明：
- 输入：用户消息、研究画像、规则模式与异常对象；
- 输出：布尔判断结果、计划/建议/warnings、prompt 文本，以及调试用摘要信息。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from ..utils.search_spec_builder import ABSTRACT_HINT_PATTERNS
from ..utils.text_utils import _extract_json_block, _matches_any, _normalize_text

SUPPORTED_INTENTS = {
    "arxiv_search",
    "paper_detail",
    "paper_summary",
    "paper_qa",
    "recommendation",
    "preference_action",
    "reading_list_action",
    "unclear",
    "unsupported",
}

NON_SEARCH_INTENTS = {
    "paper_detail",
    "paper_summary",
    "paper_qa",
    "recommendation",
    "preference_action",
    "reading_list_action",
}

LLM_CONFIDENCE_THRESHOLD = 0.55
HARD_RULE_PATTERNS: Dict[str, Sequence[str]] = {}
SEARCH_TRIGGER_PATTERNS: Sequence[str] = ()


def _build_intent_guidance(intent: str) -> Tuple[List[str], List[str], List[str]]:
    """为不同 intent 生成计划说明、下一步建议和附加 warning。
    
    主要用途：
    1. 给 parse 节点补齐 plan / next_actions，保证后续展示层始终有解释性信息；
    2. 对非搜索或不支持的分支统一产出边界说明，避免不同调用方重复拼文案。
    
    输入/输出说明：
    - 输入：已经归一化的 intent 字符串；
    - 输出：三元组 (plan, next_actions, warnings)；不同 intent 会走不同分支，但不会修改外部状态。
    """
    if intent == "arxiv_search":
        return (
            [
                "LLM 已识别为 arXiv 搜索请求",
                "规则会补全和校验检索参数，然后调用搜索工具",
                "整理结果并返回给用户",
            ],
            [
                "继续细化检索范围",
                "选择一篇论文查看详情",
                "如果需要，可以继续接论文总结或 QA 能力",
            ],
            [],
        )

    if intent in NON_SEARCH_INTENTS:
        if intent == "preference_action":
            return (
                [
                    "解析用户偏好动作",
                    "解析被操作的目标论文",
                    "保留 intent，等待后续偏好更新节点执行",
                ],
                [
                    "继续对其他论文执行喜欢、不喜欢或收藏动作",
                    "后续也可以继续搜索、查看推荐或打开论文详情",
                ],
                [f"identified non-search intent: {intent}"],
            )
        if intent == "reading_list_action":
            return (
                [
                    "识别到用户在查询阅读列表或收藏列表",
                    "当前入口先保留 intent，等待后续列表查询节点接入",
                ],
                [
                    "如果你想找论文，请改成明确的 arXiv 搜索需求",
                    "如果你想看收藏内容，可以后续直接进入阅读列表页面",
                ],
                [f"identified non-search intent: {intent}"],
            )
        if intent in {"paper_summary", "paper_detail", "paper_qa"}:
            return (
                [
                    "识别到论文详情相关请求",
                    "当前入口先保留 intent，等待论文详情、摘要或 QA 节点接入",
                ],
                [
                    "如果你是在找论文，请改成明确的 arXiv 搜索需求",
                    "如果你已经有目标论文标题或 arXiv ID，可以直接提供给后续详情能力",
                ],
                [f"identified non-search intent: {intent}"],
            )
        if intent == "recommendation":
            return (
                [
                    "识别到论文推荐请求",
                    "当前入口先保留 intent，等待个性化推荐服务接入",
                ],
                [
                    "如果你要的是普通搜索，请直接描述论文主题或关键词",
                    "如果你想看推荐结果，可以继续补充偏好方向或兴趣主题",
                ],
                [f"identified non-search intent: {intent}"],
            )
        return (
            [
                "LLM 已识别出论文系统内的非搜索请求类型",
                "当前 agent 暂未把该能力完整接入执行链路",
            ],
            [
                "如果你是在找论文，请改成明确的 arXiv 搜索需求",
                "如果你要总结、解释或问答某篇论文，请提供标题或 arXiv ID",
            ],
            [f"identified non-search intent: {intent}"],
        )

    if intent == "unclear":
        return (
            [
                "LLM 判断用户想要论文相关能力，但主题还不够明确",
                "需要补充研究方向、关键词或时间范围后再继续",
            ],
            [
                "补充主题、关键词或类别",
                "例如：RAG、LLM、Agent、NLP、推荐系统",
            ],
            ["search topic is unclear"],
        )

    return (
        [
            "当前请求超出 arXiv 搜索 Agent 的处理范围",
            "先返回可解释的边界说明，再等待用户改写请求",
        ],
        [
            "改写成 arXiv 论文搜索需求",
            "后续可以接论文总结或 QA 功能",
        ],
        ["request is outside the supported search workflow"],
    )


def _contains_any_term(message: str, terms: Sequence[str]) -> bool:
    """判断消息中是否包含任意一个候选词。
    
    核心步骤：
    1. 先把原消息转成 lower-case 版本，便于 ASCII 词统一比较；
    2. 对英文词走大小写不敏感子串匹配，对中文等非 ASCII 词保留原文匹配；
    3. 只要命中任一有效词就立即返回 True，避免无意义的后续遍历。
    
    输入/输出说明：
    - 输入：原始消息文本和候选词列表；
    - 输出：是否命中任一候选词。
    """
    lowered = message.lower()
    for term in terms:
        normalized = str(term).strip()
        if not normalized:
            continue
        if normalized.isascii():
            if normalized.lower() in lowered:
                return True
        elif normalized in message:
            return True
    return False


def _references_specific_paper(message: str) -> bool:
    """判断用户是否在引用某一篇具体论文，而不是泛泛谈一个主题。
    
    判断逻辑分两层：
    1. 先检测 arXiv ID 这类强指示信号；
    2. 再检查“这篇论文”“title”“paper id”等弱指示词，兼顾中英文表达。
    
    输出含义：
    - True 表示后续可以优先按“已有目标论文”处理；
    - False 表示更可能仍处于主题搜索阶段。
    """
    lowered = message.lower()
    if bool(re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", lowered)):
        return True
    return _contains_any_term(
        message,
        (
            "这篇论文",
            "本文",
            "该论文",
            "这篇 paper",
            "this paper",
            "the paper",
            "paper id",
            "arxiv id",
            "标题",
            "title",
        ),
    )


def _looks_like_preference_action_request(message: str) -> bool:
    """启发式识别“喜欢 / 不喜欢 / 收藏 / 取消标记”类请求。
    
    主要步骤：
    1. 先排除“阅读列表 / 收藏夹”这类列表查询，避免和偏好动作混淆；
    2. 再检查是否出现 like / dislike / 收藏 等动作词；
    3. 最后要求消息中能定位到具体论文或结果序号，降低误标风险。
    
    输出说明：
    - True 表示请求更像对某篇论文做显式偏好操作；
    - False 表示应交给其他 intent 分支继续判断。
    """
    if _contains_any_term(message, ("阅读列表", "收藏夹", "reading list", "favorites", "bookmarks")):
        return False
    # 先判断是否出现了明确动作词；没有动作词时直接排除，不继续做更昂贵的论文引用判断。
    action_hit = _contains_any_term(
        message,
        (
            "喜欢",
            "不喜欢",
            "收藏",
            "取消收藏",
            "加入收藏",
            "加入待读",
            "取消喜欢",
            "bookmark",
            "favorite",
            "favourite",
            "dislike",
            "thumbs up",
            "thumbs down",
        ),
    )
    if not action_hit:
        return False
    # 只有在动作词之外还能识别出具体论文目标时，才认为这是可执行的偏好操作请求。
    return _references_specific_paper(message) or _contains_any_term(message, ("第一篇", "第二篇", "第1篇", "第2篇", "这篇"))


def _looks_like_reading_list_action_request(message: str) -> bool:
    """启发式识别阅读列表或收藏列表查询请求。
    
    这个函数只负责做轻量词面判断，不解析列表过滤条件。
    当命中“阅读列表”“我的收藏”“saved papers”等表达时，后续流程可直接走列表类 intent。
    """
    return _contains_any_term(
        message,
        (
            "阅读列表",
            "待读",
            "待读列表",
            "收藏夹",
            "我的收藏",
            "favorites",
            "bookmarks",
            "reading list",
            "saved papers",
        ),
    )


def _looks_like_paper_summary_request(message: str) -> bool:
    """判断用户是否在要求总结某篇具体论文。
    
    分支逻辑：
    1. 先排除摘要字段过滤提示词，避免把“摘要里包含 XXX 的论文”误判成阅读总结；
    2. 再检查“总结 / summarize / tl;dr”等总结类表达；
    3. 最后结合 _looks_search_like 结果，避免把普通搜索需求归到论文阅读链路。
    
    输出：True 表示更适合走论文总结能力，False 表示继续交由其他规则判断。
    """
    if _matches_any(message, ABSTRACT_HINT_PATTERNS):
        return False
    return _contains_any_term(
        message,
        (
            "总结",
            "概述",
            "概括",
            "摘要一下",
            "summarize",
            "summary",
            "tl;dr",
        ),
    ) and not _looks_search_like(message)


def _looks_like_paper_detail_request(message: str) -> bool:
    """判断用户是否想解释论文方法、贡献或章节细节。
    
    核心思路：
    1. 搜索风格请求先直接排除；
    2. 命中“解释”“方法”“贡献”“section”等细节词后，仍要求消息能指向某篇具体论文；
    3. 只有“细节词 + 具体论文引用”同时成立时才返回 True。
    """
    if _looks_search_like(message):
        return False
    detail_hit = _contains_any_term(
        message,
        (
            "解释",
            "讲讲",
            "介绍一下",
            "方法",
            "贡献",
            "细节",
            "流程",
            "第一节",
            "第二节",
            "method",
            "approach",
            "contribution",
            "details",
            "section",
        ),
    )
    return detail_hit and (_references_specific_paper(message) or _contains_any_term(message, ("这篇", "本文", "paper")))


def _looks_like_paper_qa_request(message: str) -> bool:
    """判断输入是否更像围绕某篇论文进行问答，而不是搜索。
    
    判断步骤：
    1. 先检查问句符号或“为什么 / 怎么 / 是否”等问答特征；
    2. 若整体更像搜索请求则直接排除；
    3. 再确认消息里引用了具体论文，避免把泛问句误送入全文 QA。
    """
    question_hit = (
        "?" in message
        or "？" in message
        or _contains_any_term(message, ("为什么", "怎么", "是否", "能否", "区别", "question", "ask"))
    )
    if not question_hit or _looks_search_like(message):
        return False
    return _references_specific_paper(message) or _contains_any_term(message, ("这篇", "本文", "paper"))


def _looks_like_recommendation_request(message: str) -> bool:
    """识别个性化推荐请求，并避免把“推荐系统”主题误判成推荐意图。
    
    这个函数会先排除 recommender system 研究主题，再检查“根据我的兴趣”“为我推荐”等个性化表达，
    并补充英文正则，以覆盖更自然的推荐式问法。
    """
    lowered = message.lower()
    if "推荐系统" in message or "recommender system" in lowered or "recommendation system" in lowered:
        return False
    return bool(
        _contains_any_term(
            message,
            (
                "根据我的兴趣",
                "基于我的兴趣",
                "根据我的偏好",
                "基于我的偏好",
                "个性化推荐",
                "为我推荐",
                "适合我",
                "我可能感兴趣",
                "我感兴趣",
            ),
        )
        or re.search(r"recommend(?: me)?(?: papers?| articles?)? based on my (?:interests?|preferences?)", lowered)
        or re.search(r"personalized (?:paper )?recommendations?", lowered)
    )


def _compact_research_profile_for_prompt(research_profile: Optional[Mapping[str, Any]]) -> str:
    """把研究画像压缩成适合塞进 prompt 的单行摘要。
    
    主要步骤：
    1. 只抽取最能影响意图判断和搜索补全的关键字段；
    2. 对列表字段做清洗、截断和拼接，避免 prompt 过长；
    3. 额外补充 preferred_answer_style，帮助模型理解用户偏好。
    
    输出：返回供 prompt 直接插入的单行字符串；没有有效画像时返回 "N/A"。
    """
    if not isinstance(research_profile, Mapping):
        return "N/A"

    parts: List[str] = []
    # 只挑选最能影响意图判断和搜索补全的关键字段，避免 prompt 冗长。
    for key, label in (
        ("positive_topics", "positive_topics"),
        ("negative_topics", "negative_topics"),
        ("recent_topics", "recent_topics"),
        ("preferred_categories", "preferred_categories"),
        ("common_question_types", "common_question_types"),
        ("representative_papers", "representative_papers"),
    ):
        raw_value = research_profile.get(key)
        if isinstance(raw_value, list):
            values = [str(item).strip() for item in raw_value if str(item).strip()]
            if values:
                parts.append(f"{label}: {', '.join(values[:5])}")

    preferred_answer_style = str(research_profile.get("preferred_answer_style") or "").strip()
    if preferred_answer_style:
        parts.append(f"preferred_answer_style: {preferred_answer_style}")

    return " | ".join(parts) if parts else "N/A"


def _build_llm_prompt_with_profile(message: str, research_profile: Optional[Mapping[str, Any]] = None) -> str:
    """构造给 LLM 的意图识别提示词。
    
    主要内容包括：
    1. 限定可选 intent，减少模型自由发挥空间；
    2. 明确 search spec 的 JSON schema，约束字段名和取值；
    3. 注入研究画像摘要和关键规则，帮助模型在搜索、阅读、偏好、推荐之间做更稳定的区分。
    
    输出：返回完整 prompt 字符串，不直接触发调用。
    """
    profile_hint = _compact_research_profile_for_prompt(research_profile)
    return (
        "You are an intent parser for a natural-language arXiv paper agent.\n"
        "Return JSON only.\n"
        "Classify the message into one of: arxiv_search, paper_detail, paper_summary, paper_qa, recommendation, preference_action, reading_list_action, unclear, unsupported.\n"
        "If it is a search request, extract a structured search spec.\n"
        "Schema:\n"
        "{"
        '\"intent\":\"arxiv_search|paper_detail|paper_summary|paper_qa|recommendation|preference_action|reading_list_action|unclear|unsupported\",'
        '\"confidence\":0.0,'
        '\"query\":null|string,'
        '\"title_query\":null|string,'
        '\"abstract_query\":null|string,'
        '\"submitted_days_ago\":null|int,'
        '\"max_results\":10,'
        '\"sort_by\":\"submittedDate|relevance|lastUpdatedDate\",'
        '\"sort_order\":\"ascending|descending\",'
        '\"field_operator\":\"AND|OR|ANDNOT\",'
        '\"category_operator\":\"AND|OR\",'
        '\"reasoning_summary\":null|string,'
        '\"cleaned_topic_cn\":null|string,'
        '\"cleaned_topic_en\":null|string,'
        '\"missing_info\":[],'
        '\"warnings\":[],'
        '\"next_actions\":[]'
        "}\n"
        "Guidance:\n"
        "- If the user is asking to summarize/explain/QA a specific paper, do not classify as arxiv_search.\n"
        "- If the user is expressing like/dislike/favorite about a specific paper, classify as preference_action.\n"
        "- If the user is asking for personalized recommendations, classify as recommendation.\n"
        "- If the topic is too vague, classify as unclear.\n"
        f"Research profile hint: {profile_hint}\n"
        f"User message: {message}\n"
    )


def _looks_search_like(message: str) -> bool:
    """判断消息整体是否更像 arXiv 搜索请求。
    
    判断时会综合搜索触发词、搜索语气和若干排除条件：
    - 命中明显的论文阅读/偏好表达时会降权；
    - 命中查找、检索、最近 N 天、类别等线索时会升高为搜索意图。
    这个函数的输出常被上层规则当作 unclear 与 unsupported 的分界信号。
    """
    lowered = message.lower()
    explicit_cn_patterns = (
        r"找.*论文",
        r"搜.*论文",
        r"检索.*论文",
        r"查找.*论文",
        r"最近.*论文",
        r"新论文",
        r"新工作",
        r"最新进展",
        r"值得读",
        r"研究.*论文",
    )
    return _matches_any(message, SEARCH_TRIGGER_PATTERNS) or _matches_any(message, explicit_cn_patterns) or any(
        hint in lowered for hint in ("search", "find", "look for", "recent paper", "recent papers", "latest", "newest", "recent")
    )


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """按出现顺序去重字符串列表。
    
    用途主要是清理 warnings、next_actions 等展示字段，既避免重复提示，
    又保持第一次出现时的语义顺序，便于前端按原始推理轨迹展示。
    """
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = _normalize_text(item).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(_normalize_text(item))
    return result


def _validation_error_summary(exc: ValidationError) -> str:
    """把 Pydantic 校验异常压缩成短摘要字符串。
    
    这个函数不会改变异常本身，只负责提炼 location 和 message，
    方便 parse 节点把 schema 校验失败原因写入 warning、fallback_reason 或调试信息。
    """
    parts: List[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc") or []) or "unknown"
        parts.append(f"{location}: {error.get('msg')}")
    return "; ".join(parts) if parts else str(exc)


__all__ = [
    "HARD_RULE_PATTERNS",
    "LLM_CONFIDENCE_THRESHOLD",
    "SEARCH_TRIGGER_PATTERNS",
    "SUPPORTED_INTENTS",
    "_build_intent_guidance",
    "_build_llm_prompt_with_profile",
    "_contains_any_term",
    "_dedupe_preserve_order",
    "_extract_json_block",
    "_looks_like_paper_detail_request",
    "_looks_like_paper_qa_request",
    "_looks_like_paper_summary_request",
    "_looks_like_preference_action_request",
    "_looks_like_reading_list_action_request",
    "_looks_like_recommendation_request",
    "_looks_search_like",
    "_matches_any",
    "_references_specific_paper",
    "_validation_error_summary",
]
