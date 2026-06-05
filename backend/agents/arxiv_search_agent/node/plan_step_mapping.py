"""计划步骤到工具调用请求的白名单映射层。

这个模块的职责很克制：
1. 读取当前 execution_plan 中“下一可执行步骤”；
2. 根据 step_type + intent + state 中的可靠字段生成标准 ToolCallRequest；
3. 不直接执行工具，也不决定复杂重试策略；
4. 对暂不支持的步骤安全返回空映射结果。

这样后续如果要扩展更多计划驱动能力，只需要继续往这里增加白名单映射，
而不是把工具构造逻辑分散在各业务节点内部。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ..schemas import ToolCallRequest
from ..state import AgentState
from ..utils.state_utils import _coerce_state, _compact_search_spec, _get_next_executable_plan_step

SEARCH_TOOL_NAME = "search_arxiv_structured"


def _build_search_execution_request(state: AgentState, *, step_id: str) -> Optional[ToolCallRequest]:
    """把 arXiv 搜索执行步骤映射成标准搜索工具请求。"""
    spec = state.search_spec
    if state.intent != "arxiv_search" or spec is None:
        return None

    arguments = {
        "query": spec.query,
        "title_query": spec.title_query,
        "abstract_query": spec.abstract_query,
        "categories": list(spec.categories or []),
        "submitted_days_ago": spec.submitted_days_ago,
        "max_results": spec.max_results or 10,
        "start": 0,
        "sort_by": spec.sort_by or "submittedDate",
        "sort_order": spec.sort_order or "descending",
        "field_operator": spec.field_operator or "AND",
        "category_operator": spec.category_operator or "OR",
    }
    return ToolCallRequest(
        tool_name=SEARCH_TOOL_NAME,
        arguments=arguments,
        reason="根据当前计划步骤执行 arXiv 结构化搜索",
        expected_result="返回与当前搜索主题相关的 arXiv 论文候选列表",
        plan_step_id=step_id,
        fallback_tools=["search_arxiv_raw"],
    )


def build_tool_call_request_from_plan_step(
    state: Union[AgentState, Mapping[str, Any]],
) -> Tuple[Optional[ToolCallRequest], Dict[str, Any]]:
    """根据当前可执行计划步骤构造工具请求。

    返回值：
    - ToolCallRequest: 成功映射时返回标准工具请求；
    - debug payload: 记录当前命中的步骤、映射结果和跳过原因，便于调试。
    """
    current_state = _coerce_state(state)
    next_plan_step = _get_next_executable_plan_step(current_state)
    if next_plan_step is None:
        return None, {
            "status": "skipped",
            "reason": "no_executable_plan_step",
            "intent": current_state.intent,
        }

    step_id = str(getattr(next_plan_step, "step_id", "") or "").strip()
    step_type = str(getattr(next_plan_step, "step_type", "") or "").strip()
    intent = str(current_state.intent or "").strip()

    if step_type == "search_execution" and intent == "arxiv_search":
        request = _build_search_execution_request(current_state, step_id=step_id)
        if request is None:
            return None, {
                "status": "skipped",
                "reason": "missing_search_spec",
                "intent": intent,
                "step_id": step_id,
                "step_type": step_type,
                "search_spec": _compact_search_spec(current_state.search_spec),
            }
        return request, {
            "status": "mapped",
            "intent": intent,
            "step_id": step_id,
            "step_type": step_type,
            "tool_name": request.tool_name,
            "search_spec": _compact_search_spec(current_state.search_spec),
        }

    return None, {
        "status": "skipped",
        "reason": "unsupported_plan_step_mapping",
        "intent": intent,
        "step_id": step_id,
        "step_type": step_type,
    }


__all__ = ["SEARCH_TOOL_NAME", "build_tool_call_request_from_plan_step"]
