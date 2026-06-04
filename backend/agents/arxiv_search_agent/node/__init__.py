"""arXiv 搜索 Agent 的节点导出入口。

这个模块本身不承载执行逻辑，主要负责两件事：
1. 按功能分组 re-export 各节点、常量和辅助函数，减少上层编排代码的导入成本；
2. 用统一出口暴露 parse、search、preference、paper reading 等阶段需要的公共符号。

输入/输出说明：
- 输入：来自同目录各子模块的函数、常量与节点实现；
- 输出：通过 __all__ 明确声明可复用接口，供工作流或测试按需导入。
"""

from __future__ import annotations

# 意图识别辅助工具：供 parse 节点和测试直接复用。
# 意图识别辅助工具：供 parse 节点和测试直接复用。
# 意图识别辅助工具：供 parse 节点和测试直接复用。
# 意图识别辅助工具：供 parse 节点和测试直接复用。
# 意图识别辅助工具：供 parse 节点和测试直接复用。
from .intent_support import (
    HARD_RULE_PATTERNS,
    LLM_CONFIDENCE_THRESHOLD,
    SEARCH_TRIGGER_PATTERNS,
    SUPPORTED_INTENTS,
    _build_intent_guidance,
    _build_llm_prompt_with_profile,
    _contains_any_term,
    _dedupe_preserve_order,
    _extract_json_block,
    _looks_like_paper_detail_request,
    _looks_like_paper_qa_request,
    _looks_like_paper_summary_request,
    _looks_like_preference_action_request,
    _looks_like_reading_list_action_request,
    _looks_like_recommendation_request,
    _looks_search_like,
    _matches_any,
    _references_specific_paper,
    _validation_error_summary,
)
from ..utils.state_utils import _coerce_state
# 论文阅读类节点：负责把“总结 / 解释 / QA”请求接入论文全文问答链路，
# 并根据是否已有索引决定直接回答还是进入待确认解析流程。
from .paper_reading_node import handle_paper_reading_request
# parse 节点：负责把自然语言请求收敛成结构化 intent 与 search_spec。
# parse 节点：负责把自然语言请求收敛成结构化 intent 与 search_spec。
# parse 节点：负责把自然语言请求收敛成结构化 intent 与 search_spec。
# parse 节点：负责把自然语言请求收敛成结构化 intent 与 search_spec。
# parse 节点：负责把自然语言请求收敛成结构化 intent 与 search_spec。
from .parse_node import parse_search_request
from .pending_action_node import (
    classify_pending_action_confirmation,
    handle_pending_action_confirmation,
)
# 偏好、回复、搜索等节点分别承担独立阶段的状态加工工作：
# - preference_node 负责写入用户偏好；
# - response_node 负责把状态收口成最终自然语言答复；
# - search_node 负责真正的 arXiv 检索执行和结果重排。
from .preference_node import apply_preference_action
from .response_node import synthesize_response
from .search_node import (
    SEARCH_TOOL_NAME,
    build_search_tool_args,
    check_search_result,
    invoke_search_tool,
    personalized_rank_and_annotate_papers,
    relax_search_for_retry,
)

__all__ = [
    "HARD_RULE_PATTERNS",
    "LLM_CONFIDENCE_THRESHOLD",
    "SEARCH_TOOL_NAME",
    "SEARCH_TRIGGER_PATTERNS",
    "SUPPORTED_INTENTS",
    "_build_intent_guidance",
    "_build_llm_prompt_with_profile",
    "_contains_any_term",
    "_coerce_state",
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
    "apply_preference_action",
    "build_search_tool_args",
    "check_search_result",
    "classify_pending_action_confirmation",
    "handle_paper_reading_request",
    "handle_pending_action_confirmation",
    "invoke_search_tool",
    "parse_search_request",
    "personalized_rank_and_annotate_papers",
    "relax_search_for_retry",
    "synthesize_response",
]
