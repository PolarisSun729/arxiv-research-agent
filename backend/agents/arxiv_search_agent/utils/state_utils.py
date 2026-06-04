"""AgentState 压缩、追加步骤和状态统一化工具。

这个模块主要负责把较重的状态对象整理成更适合记录、调试和跨节点传递的形式，
避免每个 node 都各自实现一套类似的状态拼装逻辑。

职责重点包括：
1. 把复杂对象压缩成轻量快照；
2. 统一追加执行步骤轨迹；
3. 把外部传入的 state 规范化为 AgentState。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, List, Union

from ..schemas import AgentStep, ArxivSearchSpec
from ..state import AgentState


def _compact_search_spec(spec: Optional[ArxivSearchSpec]) -> Dict[str, Any]:
    """把搜索规格对象压缩成适合日志、调试和 step 输出的轻量字典。

    这里不会保留完整模型对象，而是提取出最有解释价值的字段，
    便于写入 debug、step outputs 或前端展示区域。
    """
    if spec is None:
        return {}
    payload = {
        "intent": spec.intent,
        "query": spec.query,
        "title_query": spec.title_query,
        "abstract_query": spec.abstract_query,
        "categories": list(spec.categories or []),
        "submitted_days_ago": spec.submitted_days_ago,
        "max_results": spec.max_results,
        "sort_by": spec.sort_by,
        "sort_order": spec.sort_order,
        "field_operator": spec.field_operator,
        "category_operator": spec.category_operator,
        "reasoning_summary": spec.reasoning_summary,
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


def _compact_paper_summaries(papers: Sequence[Mapping[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    """从论文列表中抽取少量摘要字段，避免在轨迹里保存完整论文数据。

    完整论文对象通常字段很多、体积较大，直接写入状态轨迹会让调试信息过重。
    因此这里只挑选标题、arXiv ID、发布时间和简单排序字段作为摘要。
    """
    summaries: List[Dict[str, Any]] = []
    for paper in list(papers or [])[: max(0, limit)]:
        if not isinstance(paper, Mapping):
            continue
        summary: Dict[str, Any] = {}
        for key in ("arxiv_id", "title", "published", "primary_category", "score", "rank"):
            value = paper.get(key)
            if value not in (None, ""):
                summary[key] = value
        if summary:
            summaries.append(summary)
    return summaries


def _append_step(
    state: AgentState,
    *,
    step: str,
    status: str,
    action: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> AgentState:
    """向状态轨迹中追加一个新的执行步骤，并返回新的 state 副本。

    该函数统一维护步骤追加方式，确保所有节点记录的 step 结构一致，
    这样前端、日志系统和调试工具都可以稳定消费这些轨迹数据。
    """
    next_state = state.model_copy(deep=True)
    next_state.steps = list(next_state.steps or []) + [
        AgentStep(
            step=step,
            status=status,
            action=action,
            inputs=inputs or {},
            outputs=outputs or {},
            error=error,
        )
    ]
    return next_state


def _compact_tool_args(tool_args: Mapping[str, Any]) -> Dict[str, Any]:
    """筛选并压缩工具参数，生成适合展示和追踪的参数快照。

    只保留与搜索执行和问题定位最相关的字段，避免把过多冗余参数写入 trace。
    """
    payload: Dict[str, Any] = {}
    for key in (
        "query",
        "title_query",
        "abstract_query",
        "author_query",
        "categories",
        "comment_query",
        "journal_ref_query",
        "report_number_query",
        "id_list",
        "field_operator",
        "category_operator",
        "submitted_days_ago",
        "max_results",
        "start",
        "sort_by",
        "sort_order",
    ):
        value = tool_args.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    return payload


def _coerce_state(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """把节点收到的 state 统一转换成 AgentState 副本。

    这样各个节点就不需要关心调用方传进来的是完整模型实例还是普通字典，
    只需要面对统一的 AgentState 接口即可。
    """
    if isinstance(state, AgentState):
        return state.model_copy(deep=True)
    return AgentState.model_validate(dict(state))
