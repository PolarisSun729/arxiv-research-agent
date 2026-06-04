"""执行 arXiv 搜索、结果检查、重试放宽和个性化重排。

这个模块承接 parse_node 产出的 search_spec，负责把结构化条件真正落到执行层：
1. 构造搜索工具参数并调用 arXiv 工具；
2. 检查结果是否报错、为空或明显偏少；
3. 在空结果时自动放宽 query 后重试；
4. 在具备用户偏好时，对结果做可选的个性化重排。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

try:  # pragma: no cover - import path differs between backend cwd and package import
    from dependencies import get_recommendation_service
except ModuleNotFoundError:  # pragma: no cover
    from backend.dependencies import get_recommendation_service

from ..schemas import AgentToolCall, ArxivSearchSpec, ToolCallRequest
from ..state import AgentState
from ..utils.result_utils import (
    _extract_error_message,
    _extract_papers_from_tool_result,
    _result_mapping,
    _result_ok,
    _result_text,
    _to_plain_dict,
)
from ..utils.search_spec_builder import _build_reasoning_summary
from ..utils.state_utils import _append_step, _coerce_state, _compact_paper_summaries, _compact_search_spec, _compact_tool_args
from ..utils.text_utils import _contains_chinese, _normalize_text
from .tool_node import execute_tool

SEARCH_TOOL_NAME = "search_arxiv_structured"
MAX_SEARCH_RETRIES = 3


def _append_structured_error(
    state: AgentState,
    *,
    step: str,
    code: str,
    message: str,
    detail: Optional[str] = None,
    recoverable: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """向 errors/debug 追加结构化排查信息，不直接暴露给最终用户。"""
    error_payload: Dict[str, Any] = {
        "step": step,
        "code": code,
        "message": message,
        "recoverable": recoverable,
    }
    if detail:
        error_payload["detail"] = detail
    if extra:
        error_payload["extra"] = dict(extra)

    state.errors = list(state.errors or []) + [error_payload]

    debug = dict(state.debug or {})
    recoverable_errors = list(debug.get("recoverable_errors", []))
    recoverable_errors.append(error_payload)
    debug["recoverable_errors"] = recoverable_errors
    debug["last_error"] = error_payload
    state.debug = debug


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """按顺序去重字符串列表，主要用于 warning 和提示文案去重。
    
    实现上会先做归一化再去重，保证“同义大小写差异”的提示不会重复出现；
    同时保留第一次出现的顺序，方便前端展示真实执行轨迹。
    """
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = _normalize_text(item).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(_normalize_text(item))
    return result


def _build_tool_call_trace(
    *,
    tool_name: str,
    tool_args: Mapping[str, Any],
    result: Optional[Mapping[str, Any]] = None,
    paper_count: Optional[int] = None,
    source: Optional[str] = None,
    normalized_inputs: Optional[Dict[str, Any]] = None,
    final_search_query: Optional[str] = None,
) -> Dict[str, Any]:
    """构造统一的工具调用 trace，便于日志、调试和前端展示。
    
    主要步骤：
    1. 先吸收工具层已经给出的 trace；
    2. 再补齐 agent 侧压缩后的 inputs、排序参数、返回数量等元信息；
    3. 若工具结果中缺少 returned_count 或 error，则在这里兜底补全。
    
    输出：返回一个尽量结构稳定的 trace 字典。
    """
    trace: Dict[str, Any] = {}
    if isinstance(result, Mapping):
        trace.update(_result_mapping(result, "trace") or {})

    # 先把输入压缩成适合记录的紧凑结构，再补齐 agent 侧派生出的元信息。
    compact_inputs = _compact_tool_args(tool_args)
    trace.setdefault("tool_name", tool_name)
    trace.setdefault("inputs", compact_inputs)
    trace.setdefault("raw_inputs", compact_inputs)
    trace.setdefault("normalized_inputs", normalized_inputs or compact_inputs)
    trace.setdefault("final_search_query", final_search_query or compact_inputs.get("query"))
    trace.setdefault("source", source or trace.get("source") or "agent")
    trace.setdefault("sort_by", compact_inputs.get("sort_by"))
    trace.setdefault("sort_order", compact_inputs.get("sort_order"))
    trace.setdefault("start", compact_inputs.get("start"))
    trace.setdefault("max_results", compact_inputs.get("max_results"))
    trace.setdefault("returned_count", paper_count if paper_count is not None else 0)
    trace.setdefault("id_list", list(compact_inputs.get("id_list") or []))
    if isinstance(result, Mapping):
        if "returned_count" not in trace:
            trace["returned_count"] = len(_extract_papers_from_tool_result(_to_plain_dict(result)))
        trace.setdefault("result_ok", _result_ok(result))
        error = _result_mapping(result, "error")
        if error is not None:
            trace.setdefault("error", error)
    return trace


def _relax_query_for_fallback(query: Optional[str], retry_count: int) -> Optional[str]:
    """在自动重试时逐步放宽查询词，尽量提升召回率。
    
    分支行为：
    1. 空 query 或重试轮次过深时直接返回 None；
    2. 英文多 token 查询通过逐轮删减末尾 token 放宽；
    3. 连续中文查询则按长度截短，避免一下子把主题语义全部清空。
    """
    if not query or not str(query).strip():
        return None

    query = str(query).strip()

    if retry_count >= 3:
        return None

    # 英文/空格分词查询优先通过“逐轮减少末尾 token”来放宽条件。
    tokens = query.split()
    if len(tokens) > 1:
        keep = max(1, len(tokens) - retry_count)
        return " ".join(tokens[:keep])

    # 对连续中文查询采用截短策略，避免直接清空查询导致语义损失过大。
    if _contains_chinese(query) and len(query) > 2:
        keep = max(2, len(query) - retry_count * 2)
        return query[:keep]

    return query


def _format_tool_failure_warning(result: Mapping[str, Any]) -> str:
    """把工具失败结果转成更适合给用户/日志看的 warning 文案。
    
    warnings 只保留用户可理解的降级提示，具体底层异常写入 errors/debug。
    """
    return "工具调用失败，请检查搜索参数或 arXiv 服务状态"


def _papers_are_significantly_fewer_than_requested(actual_count: int, max_results: int) -> bool:
    """判断结果数是否明显偏少，用于提示用户搜索条件可能过窄。
    
    这个函数不会判断“空结果”，只关注“有结果但远低于期望值”的情况：
    - max_results 很小时按严格阈值判断；
    - 其余情况按一半左右的经验阈值判断。
    """
    if actual_count <= 0 or max_results <= 0:
        return False
    if max_results <= 3:
        return actual_count < max_results
    return actual_count <= max(1, max_results // 2)


def _summarize_search_spec(spec: Optional[ArxivSearchSpec]) -> str:
    """把 search spec 压缩成可读的中文摘要，用于最终回复。
    
    会按 query、title_query、abstract_query、categories、时间范围、排序方式和数量上限依次拼接，
    输出适合直接放进 answer 的一句中文说明。
    """
    if spec is None:
        return "当前搜索条件"

    parts: List[str] = []
    if spec.query:
        parts.append(f"主题 {spec.query}")
    if spec.title_query:
        parts.append(f"标题 {spec.title_query}")
    if spec.abstract_query:
        parts.append(f"摘要 {spec.abstract_query}")
    if spec.categories:
        parts.append(f"类别 {', '.join(spec.categories)}")
    if spec.submitted_days_ago is not None:
        parts.append(f"最近 {spec.submitted_days_ago} 天")
    parts.append(f"排序 {spec.sort_by} / {spec.sort_order}")
    parts.append(f"最多 {spec.max_results} 篇")
    return "，".join(parts)


def _determine_requested_max_results(state: AgentState) -> int:
    """统一计算本轮搜索真正期望返回的最大论文数。
    
    优先级为：
    1. 优先读取 search_spec.max_results；
    2. 若缺失则回退到已有 tool_args；
    3. 两者都没有时使用默认值 10。
    
    这样可以保证结果检查、个性化重排和回复生成使用同一套上限口径。
    """
    if state.search_spec is not None:
        return max(1, int(state.search_spec.max_results or 10))
    if isinstance(state.tool_args, dict) and state.tool_args.get("max_results") is not None:
        return max(1, int(state.tool_args.get("max_results") or 10))
    return 10


def _collect_priority_titles(papers: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    """从结果集中挑出最值得优先展示的论文标题。
    
    排序时会综合 priority、final_score、query_match_score 和 arxiv_id，
    尽量让真正被推荐或匹配度更高的论文排在前面，最后只返回标题列表供回复层使用。
    """
    prioritized = sorted(
        [paper for paper in papers if isinstance(paper, dict)],
        key=lambda paper: (
            float(paper.get("priority", 0) or 0) if float(paper.get("priority", 0) or 0) > 0 else 10_000.0,
            -float(paper.get("final_score", 0.0) or 0.0),
            -float(paper.get("query_match_score", 0.0) or 0.0),
            str(paper.get("arxiv_id", "") or paper.get("id", "") or ""),
        ),
    )
    titles: List[str] = []
    for paper in prioritized[: max(1, int(limit or 3))]:
        title = str(paper.get("title", "") or "").strip()
        if not title:
            continue
        titles.append(title)
    return titles


def _get_search_execution_plan_step_id(state: AgentState) -> Optional[str]:
    """从 execution_plan 中恢复搜索执行步骤 ID，供 tool_call_request 关联使用。"""
    for step in list(state.execution_plan or []):
        if str(getattr(step, "step_type", "") or "").strip() == "search_execution":
            step_id = str(getattr(step, "step_id", "") or "").strip()
            return step_id or None
    return None


def build_search_tool_args(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """根据 search_spec 构造 arXiv 搜索工具参数。
    
    主要步骤：
    1. 先确认当前 intent 和 search_spec 都合法；
    2. 把业务对象字段显式映射成工具层需要的纯参数；
    3. 同步写入阶段 2 的 tool_call_request，作为统一工具协议入口；
    4. 在 skipped 分支下清空 tool_name/tool_args/tool_call_request，避免误复用上一轮状态。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search" or next_state.search_spec is None:
        next_state.tool_name = None
        next_state.tool_args = {}
        next_state.tool_call_request = None
        return _append_step(
            next_state,
            step="tool_argument_construction",
            status="skipped",
            action="基于搜索条件构造 arXiv 工具参数",
            inputs={"intent": next_state.intent, "search_spec": _compact_search_spec(next_state.search_spec)},
            outputs={"reason": "非 arXiv 搜索或搜索条件缺失"},
        )

    # 这里把 schema 字段显式映射成工具调用参数，避免后续工具层感知业务对象。
    spec = next_state.search_spec
    next_state.tool_name = SEARCH_TOOL_NAME
    next_state.tool_args = {
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
    next_state.tool_call_request = ToolCallRequest(
        tool_name=SEARCH_TOOL_NAME,
        arguments=dict(next_state.tool_args),
        reason="根据用户目标和结构化搜索条件检索 arXiv 论文候选结果",
        expected_result="返回与当前搜索主题相关的 arXiv 论文候选列表",
        plan_step_id=_get_search_execution_plan_step_id(next_state),
        fallback_tools=["search_arxiv_raw"],
    )
    return _append_step(
        next_state,
        step="tool_argument_construction",
        status="success",
        action="基于搜索条件构造 arXiv 工具参数",
        inputs={"search_spec": _compact_search_spec(spec)},
        outputs={
            "tool_name": SEARCH_TOOL_NAME,
            "tool_args": {key: value for key, value in next_state.tool_args.items() if key != "query" or value},
            "tool_call_request": next_state.tool_call_request.model_dump(),
        },
    )


def invoke_search_tool(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """通过通用 execute_tool 执行 arXiv 搜索工具，并保留旧的 step 命名兼容性。"""
    next_state = execute_tool(state)
    if next_state.steps and next_state.steps[-1].step == "execute_tool":
        next_state.steps[-1].step = "search_tool_call"
        next_state.steps[-1].action = "调用 arXiv 搜索工具"
    return next_state


def adapt_search_tool_result(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """把通用工具执行结果适配回搜索链路所需的 papers/trace 结构。

    execute_tool 负责统一执行和记录观察；本节点则负责：
    1. 从通用 tool_result 中提取 arXiv 搜索论文列表；
    2. 补齐搜索链路历史上依赖的统一 trace 结构；
    3. 在失败场景下继续生成 warnings/errors，保持 check_search_result 和后续节点兼容。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search":
        return _append_step(
            next_state,
            step="search_tool_result_adaptation",
            status="skipped",
            action="适配通用工具结果为搜索链路状态",
            inputs={"intent": next_state.intent},
            outputs={"reason": "当前意图不是 arXiv 搜索"},
        )

    tool_result = _to_plain_dict(next_state.tool_result)
    if not tool_result:
        next_state.papers = []
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + ["搜索工具结果不存在，请先执行工具调用"])
        return _append_step(
            next_state,
            step="search_tool_result_adaptation",
            status="failed",
            action="适配通用工具结果为搜索链路状态",
            inputs={"tool_name": next_state.tool_name, "tool_args": dict(next_state.tool_args or {})},
            outputs={"paper_count": 0},
            error="搜索工具结果不存在",
        )

    extracted_papers = _extract_papers_from_tool_result(tool_result) if _result_ok(tool_result) else []
    trace_payload = _result_mapping(tool_result, "trace")
    enriched_trace = _build_tool_call_trace(
        tool_name=SEARCH_TOOL_NAME,
        tool_args=next_state.tool_args,
        result=tool_result,
        paper_count=len(extracted_papers),
        source=trace_payload.get("source") if trace_payload else None,
        normalized_inputs=trace_payload.get("normalized_inputs") if trace_payload else None,
        final_search_query=trace_payload.get("final_search_query") if trace_payload else None,
    )
    tool_result["trace"] = enriched_trace
    next_state.tool_result = tool_result

    if next_state.tool_calls and str(next_state.tool_calls[-1].tool_name or "") == SEARCH_TOOL_NAME:
        next_state.tool_calls[-1].trace = enriched_trace
        next_state.tool_calls[-1].summary = _result_text(tool_result, "summary")
        next_state.tool_calls[-1].status = "success" if _result_ok(tool_result) else "failed"
        next_state.tool_calls[-1].error = _result_mapping(tool_result, "error")
    else:
        next_state.tool_calls = list(next_state.tool_calls) + [
            AgentToolCall(
                tool_name=SEARCH_TOOL_NAME,
                arguments=dict(next_state.tool_args),
                status="success" if _result_ok(tool_result) else "failed",
                summary=_result_text(tool_result, "summary"),
                trace=enriched_trace,
                error=_result_mapping(tool_result, "error"),
            )
        ]

    if next_state.tool_observations and str(next_state.tool_observations[-1].tool_name or "") == SEARCH_TOOL_NAME:
        next_state.tool_observations[-1].raw_trace = enriched_trace
        next_state.tool_observations[-1].is_sufficient = bool(_result_ok(tool_result) and extracted_papers)
        next_state.tool_observations[-1].result_ref = {
            "paper_count": len(extracted_papers),
            "top_papers": _compact_paper_summaries(extracted_papers, limit=3),
        }

    if _result_ok(tool_result):
        next_state.papers = extracted_papers
    else:
        _append_structured_error(
            next_state,
            step="search_tool_call",
            code="search_tool_result_failed",
            message="搜索工具返回失败状态，已按可恢复错误处理",
            detail=_extract_error_message(tool_result),
            recoverable=True,
            extra={"tool_name": SEARCH_TOOL_NAME, "tool_args": dict(next_state.tool_args)},
        )
        next_state.papers = []
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [_format_tool_failure_warning(tool_result)])

    return _append_step(
        next_state,
        step="search_tool_result_adaptation",
        status="success" if _result_ok(tool_result) else "failed",
        action="适配通用工具结果为搜索链路状态",
        inputs={"tool_name": SEARCH_TOOL_NAME, "tool_args": dict(next_state.tool_args or {})},
        outputs={
            "paper_count": len(next_state.papers or []),
            "tool_call_status": "success" if _result_ok(tool_result) else "failed",
            "tool_summary": _result_text(tool_result, "summary"),
        },
        error="搜索工具返回失败状态" if not _result_ok(tool_result) else None,
    )


def check_search_result(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """检查搜索结果是否为空、报错或明显偏少，并追加提示信息。
    
    判断顺序为：
    1. 先确认是否存在 tool_result；
    2. 再区分工具失败、空结果、结果偏少等不同提示；
    3. 对空结果场景额外结合 search_retry_count 提示是否会继续自动放宽。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "arxiv_search":
        return _append_step(
            next_state,
            step="search_result_check",
            status="skipped",
            action="检查搜索结果质量并补充提示",
            inputs={"intent": next_state.intent},
            outputs={"reason": "当前意图不是 arXiv 搜索"},
        )

    warnings = list(next_state.warnings)
    tool_result = _to_plain_dict(next_state.tool_result)
    papers = list(next_state.papers or [])
    max_results = _determine_requested_max_results(next_state)

    if not tool_result:
        warnings.append("搜索工具结果不存在，请先执行工具调用")
    else:
        if not _result_ok(tool_result):
            warnings.append("工具调用失败，请检查搜索参数或 arXiv 服务状态")
            _append_structured_error(
                next_state,
                step="search_result_check",
                code="search_result_unavailable",
                message="搜索结果检查发现工具调用失败，后续将按失败结果继续响应",
                detail=_extract_error_message(tool_result),
                recoverable=True,
                extra={"paper_count": len(papers), "requested_max_results": max_results},
            )

    # 只有工具成功时，才基于论文数判断是否需要提示自动放宽或缩窄查询。
    if _result_ok(tool_result):
        if not papers:
            retry_count = int(next_state.search_retry_count or 0)
            if retry_count < MAX_SEARCH_RETRIES:
                warnings.append(f"搜索结果为空（第 {retry_count + 1} 轮），将自动放宽关键词后重试")
            else:
                warnings.append("搜索结果为空，已尝试多轮放宽关键词，建议扩大时间范围或减少关键词")
        elif _papers_are_significantly_fewer_than_requested(len(papers), max_results):
            warnings.append("结果数量较少，可能是查询条件过窄")

    next_state.warnings = _dedupe_preserve_order(warnings)
    return _append_step(
        next_state,
        step="search_result_check",
        status="success" if _result_ok(tool_result) else "failed",
        action="检查搜索结果质量并补充提示",
        inputs={
            "paper_count": len(papers),
            "tool_result_ok": _result_ok(tool_result),
            "requested_max_results": max_results,
        },
        outputs={
            "warning_count": len(next_state.warnings),
            "paper_count": len(papers),
        },
        error="搜索结果检查发现工具调用失败" if tool_result and not _result_ok(tool_result) else None,
    )


def relax_search_for_retry(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """在搜索结果为空时放宽 query，并重置执行态以便下一轮重试。
    
    这个函数会：
    1. 递增 search_retry_count；
    2. 记录 fallback_specs 和 debug 中的每轮查询变化；
    3. 直接修改 search_spec.query 与 reasoning_summary；
    4. 清空上一轮 tool_name、tool_args、tool_result 和 papers，避免污染重试结果。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    retry_count = int(next_state.search_retry_count or 0)
    next_state.search_retry_count = retry_count + 1

    # 只有保留 search_spec 时，后续重试链路才有可放宽的查询上下文。
    if next_state.search_spec is not None:
        original_query = next_state.search_spec.query
        relaxed_query = _relax_query_for_fallback(original_query, next_state.search_retry_count)

        # fallback_specs/debug 会记录每一轮放宽前后的查询词，便于回放问题定位。
        fallback_record = {
            "round": next_state.search_retry_count,
            "original_query": original_query,
            "relaxed_query": relaxed_query,
            "categories": list(next_state.search_spec.categories or []),
        }
        next_state.fallback_specs = list(next_state.fallback_specs) + [fallback_record]

        debug = dict(next_state.debug or {})
        debug["fallback_round"] = next_state.search_retry_count
        fallback_queries = list(debug.get("fallback_queries", []))
        fallback_queries.append({
            "round": next_state.search_retry_count,
            "query": relaxed_query,
        })
        debug["fallback_queries"] = fallback_queries
        next_state.debug = debug

        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + [f"自动放宽关键词: \"{original_query}\" -> \"{relaxed_query or '(仅按类别搜索)'}\""]
        )

        # 直接修改 search_spec，使下一轮 build_search_tool_args 使用放宽后的查询。
        next_state.search_spec.query = relaxed_query
        next_state.search_spec.reasoning_summary = _build_reasoning_summary(
            relaxed_query,
            list(next_state.search_spec.categories or []),
            None,
            next_state.search_spec.max_results or 10,
            next_state.search_spec.sort_by or "submittedDate",
        ) + f" (fallback round {next_state.search_retry_count})"

    # 重试前清空上一轮工具执行残留，避免错误结果污染后续节点。
    next_state.tool_name = None
    next_state.tool_args = {}
    next_state.tool_call_request = None
    next_state.tool_result = None
    next_state.papers = []

    return _append_step(
        next_state,
        step="search_fallback_retry",
        status="success",
        action=f"搜索结果为空，自动放宽关键词后重试（第 {next_state.search_retry_count} 轮）",
        inputs={
            "retry_count": next_state.search_retry_count,
            "relaxed_query": next_state.search_spec.query if next_state.search_spec else None,
        },
        outputs={
            "new_query": next_state.search_spec.query if next_state.search_spec else None,
            "fallback_specs": list(next_state.fallback_specs),
        },
    )


def personalized_rank_and_annotate_papers(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """基于用户偏好对搜索结果做可选的个性化重排。
    
    主要分支：
    1. 缺少搜索结果或 user_id 时直接跳过；
    2. 推荐服务初始化失败时保留普通排序并补 warning；
    3. 服务调用成功后，用 reranked papers 覆盖结果，并标记是否真正应用了个性化；
    4. 若最终没有生效，仍会显式告知已退化为普通搜索排序。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    query_text = next_state.search_spec.query if next_state.search_spec is not None else None
    search_spec_payload = next_state.search_spec.model_dump() if next_state.search_spec is not None else None
    compact_search_spec = _compact_search_spec(next_state.search_spec)
    user_id = str(next_state.user_id) if next_state.user_id is not None else None
    paper_count = len(next_state.papers or [])

    def _build_failed_rerank_state(exc: Exception, warning_prefix: str) -> AgentState:
        next_state.personalized_rerank_applied = False
        _append_structured_error(
            next_state,
            step="personalized_rerank",
            code="personalized_rerank_failed",
            message=f"{warning_prefix}，已降级为普通搜索排序",
            detail=str(exc),
            recoverable=True,
            extra={
                "user_id": user_id,
                "paper_count": paper_count,
                "query": query_text,
                "search_spec": compact_search_spec,
            },
        )
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + [f"{warning_prefix}，已保留普通搜索排序"],
        )
        return _append_step(
            next_state,
            step="personalized_rerank",
            status="failed",
            action="基于用户偏好对搜索结果做个性化重排",
            inputs={
                "user_id": user_id,
                "paper_count": paper_count,
                "query": query_text,
                "search_spec": compact_search_spec,
            },
            outputs={
                "personalized_rerank_applied": False,
                "paper_count": len(next_state.papers or []),
                "papers_preserved": True,
            },
            error=warning_prefix,
        )

    # 个性化重排依赖三个前提：搜索意图、已有候选论文、以及有效 user_id。
    if next_state.intent != "arxiv_search" or not next_state.papers or not next_state.user_id:
        next_state.personalized_rerank_applied = False
        return _append_step(
            next_state,
            step="personalized_rerank",
            status="skipped",
            action="基于用户偏好对搜索结果做个性化重排",
            inputs={
                "intent": next_state.intent,
                "user_id_present": bool(next_state.user_id),
                "paper_count": len(next_state.papers or []),
            },
            outputs={"personalized_rerank_applied": False},
        )

    try:
        recommendation_service = get_recommendation_service()
    except Exception as exc:
        return _build_failed_rerank_state(exc, "无法初始化推荐服务")

    try:
        # 重排服务只调整排序和附加打分，不会改变“本轮搜索主题”的语义边界。
        rerank_result = recommendation_service.rerank_search_results_for_user(
            user_id=user_id,
            papers=list(next_state.papers or []),
            query=query_text,
            top_n=_determine_requested_max_results(next_state),
            search_spec=search_spec_payload,
        )
    except Exception as exc:
        return _build_failed_rerank_state(exc, "个性化重排失败")

    reranked_papers = rerank_result.get("papers") if isinstance(rerank_result, Mapping) else None
    if isinstance(reranked_papers, list) and reranked_papers:
        next_state.papers = [paper for paper in reranked_papers if isinstance(paper, dict)]

    # 即使服务调用成功，也要区分“真正做了个性化”与“退化为普通排序”两种情况。
    next_state.personalized_rerank_applied = bool(rerank_result.get("personalized_applied")) if isinstance(rerank_result, Mapping) else False
    rerank_warnings = rerank_result.get("warnings", []) if isinstance(rerank_result, Mapping) else []
    if isinstance(rerank_warnings, list):
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [str(item) for item in rerank_warnings if str(item).strip()])

    if not next_state.personalized_rerank_applied:
        next_state.warnings = _dedupe_preserve_order(
            list(next_state.warnings) + ["用户兴趣向量不可用或个性化重排未生效，已退化为普通搜索结果"],
        )

    return _append_step(
        next_state,
        step="personalized_rerank",
        status="success",
        action="基于用户偏好对搜索结果做个性化重排",
        inputs={
            "user_id": user_id,
            "paper_count": len(current_state.papers or []),
            "query": query_text,
            "search_spec": compact_search_spec,
        },
        outputs={
            "personalized_rerank_applied": bool(next_state.personalized_rerank_applied),
            "paper_count": len(next_state.papers or []),
            "top_papers": _compact_paper_summaries(next_state.papers, limit=3),
        },
    )


__all__ = [
    "MAX_SEARCH_RETRIES",
    "SEARCH_TOOL_NAME",
    "adapt_search_tool_result",
    "build_search_tool_args",
    "check_search_result",
    "invoke_search_tool",
    "personalized_rank_and_annotate_papers",
    "relax_search_for_retry",
]
