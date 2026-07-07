from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Mapping, Optional

from dependencies import get_paper_qa_service as get_dependency_paper_qa_service
from services.paper_qa.repair_actions import (
    ASK_USER_TO_REBUILD_INDEX,
    RETRY_WITH_EXPANDED_CONTEXT,
    RETRY_WITH_KEYWORD_EMPHASIS,
    RETRY_WITH_QUERY_REWRITE,
    RETRY_WITH_SECTION_FOCUS,
    RETRY_WITHOUT_HYDE,
    normalize_repair_actions,
    repair_action_input_params,
)
from services.paper_qa.qa_observation import build_error_qa_observation
from .tool_result import make_tool_error, make_tool_result, make_tool_trace


@lru_cache(maxsize=1)
def _get_paper_qa_service():
    return get_dependency_paper_qa_service()


def check_paper_qa_index(arxiv_id: str, run_id: Optional[str] = None) -> Dict[str, Any]:
    tool_name = "check_paper_qa_index"
    # run_id 只进入工具 trace，方便把 Agent 工具调用和后端状态检查串联。
    trace_inputs = {"arxiv_id": arxiv_id, "run_id": run_id}
    try:
        data = _get_paper_qa_service().get_qa_status(arxiv_id)
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已检查论文 {arxiv_id} 的 QA 索引状态",
            data=data,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="检查 QA 索引失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
            error=make_tool_error("qa_status_failed", str(exc)),
        )


def build_paper_qa_index(arxiv_id: str, loading_method: str = "docling", run_id: Optional[str] = None) -> Dict[str, Any]:
    tool_name = "build_paper_qa_index"
    # run_id 只进入工具 trace 和索引构建日志，不影响加载方式或索引内容。
    trace_inputs = {"arxiv_id": arxiv_id, "loading_method": loading_method, "run_id": run_id}
    try:
        service = _get_paper_qa_service()
        status = service.get_qa_status(arxiv_id)
        if bool(status.get("has_index")):
            # Agent 的可恢复构建链路会先异步建索引，job 成功后再 resume；
            # 此时工具只需要确认索引已存在，避免重复解析 PDF 和重写向量库。
            data = {
                **status,
                "status": "already_indexed",
                "arxiv_id": arxiv_id,
                "skipped_rebuild": True,
            }
            return make_tool_result(
                ok=True,
                tool_name=tool_name,
                summary=f"论文 {arxiv_id} 已有 QA 索引，跳过重复构建",
                data=data,
                trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
            )
        service_kwargs: Dict[str, Any] = {"loading_method": loading_method}
        if run_id:
            service_kwargs["run_id"] = run_id
        data = service.build_qa_index(arxiv_id, **service_kwargs)
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已为论文 {arxiv_id} 构建 QA 索引",
            data=data,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
        )
    except Exception as exc:
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="构建 QA 索引失败",
            data=None,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
            error=make_tool_error("qa_index_failed", str(exc)),
        )


def _apply_qa_recovery_strategy(payload: Dict[str, Any], strategy: Optional[Mapping[str, Any]]) -> None:
    """把 Agent 的修复决策收敛成现有 QA 参数，避免在检索链路里新增临时分支。"""
    if not isinstance(strategy, Mapping):
        return
    actions = set(normalize_repair_actions(strategy.get("repair_actions") or strategy.get("actions") or []))
    reason = str(strategy.get("reason") or strategy.get("failure_category") or "").strip()
    input_params = dict(strategy.get("input_params") or {})
    input_params.update(repair_action_input_params(actions))
    if "top_k_min" not in input_params and strategy.get("retrieval_top_k") is not None:
        input_params["top_k_min"] = strategy.get("retrieval_top_k")
    if RETRY_WITH_QUERY_REWRITE in actions or reason in {"qa_low_grounding", "qa_no_sources", "qa_no_answer"}:
        payload["enable_query_rewrite"] = True
    if RETRY_WITH_EXPANDED_CONTEXT in actions or RETRY_WITH_SECTION_FOCUS in actions:
        payload["enable_context_expansion"] = True
    if RETRY_WITHOUT_HYDE in actions:
        payload["enable_hyde"] = False
    if RETRY_WITH_KEYWORD_EMPHASIS in actions:
        payload["enable_keyword_search"] = True
    if ASK_USER_TO_REBUILD_INDEX in actions:
        # 重建索引是高成本动作；这里仅保留意图给 trace，真正执行必须走确认链路。
        payload["ask_user_to_rebuild_index"] = True
    top_k_min = input_params.get("top_k_min")
    if top_k_min is not None:
        payload["top_k"] = max(int(payload.get("top_k") or 0), int(top_k_min))
    if actions or reason:
        # 修复重试默认打开更严格 grounding 和 debug，便于后续质量门解释为什么仍然降级或兜底。
        payload["stricter_grounding"] = bool(strategy.get("grounding_required", True))
        payload["debug"] = True
        payload["qa_recovery_strategy"] = {**dict(strategy), "repair_actions": sorted(actions), "input_params": input_params}


def answer_paper_question(
    arxiv_id: str,
    question: str,
    stricter_grounding: Optional[bool] = None,
    top_k: Optional[int] = None,
    enable_query_rewrite: Optional[bool] = None,
    enable_hyde: Optional[bool] = None,
    enable_keyword_search: Optional[bool] = None,
    enable_llm_rerank: Optional[bool] = None,
    enable_context_expansion: Optional[bool] = None,
    debug: Optional[bool] = None,
    qa_recovery_strategy: Optional[Dict[str, Any]] = None,
    index_strategy: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    tool_name = "answer_paper_question"
    trace_inputs = {
        "arxiv_id": arxiv_id,
        "question": question,
        "stricter_grounding": stricter_grounding,
        "top_k": top_k,
        "enable_query_rewrite": enable_query_rewrite,
        "enable_hyde": enable_hyde,
        "enable_keyword_search": enable_keyword_search,
        "enable_llm_rerank": enable_llm_rerank,
        "enable_context_expansion": enable_context_expansion,
        "debug": debug,
        "qa_recovery_strategy": qa_recovery_strategy,
        "index_strategy": index_strategy,
        "run_id": run_id,
    }
    payload = {
        "question": question,
        # Observer 发现 grounding 不足后会打开该开关，提示下游生成阶段收紧证据引用要求。
        "stricter_grounding": stricter_grounding,
        "top_k": top_k,
        "enable_query_rewrite": enable_query_rewrite,
        "enable_hyde": enable_hyde,
        "enable_keyword_search": enable_keyword_search,
        "enable_llm_rerank": enable_llm_rerank,
        "enable_context_expansion": enable_context_expansion,
        "debug": debug,
        # Agent 工具调用传入的 run_id 只用于日志/trace 串联，不改变 QA 参数语义。
        "run_id": run_id,
    }
    _apply_qa_recovery_strategy(payload, qa_recovery_strategy)
    if index_strategy:
        # index_strategy 当前只作为 trace/debug 透传，避免恢复路径丢失“为什么复用旧索引”的上下文。
        payload["index_strategy"] = dict(index_strategy)
    try:
        data = _get_paper_qa_service().answer_question(arxiv_id, payload)
        return make_tool_result(
            ok=True,
            tool_name=tool_name,
            summary=f"已回答论文 {arxiv_id} 的问题",
            data=data,
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
        )
    except Exception as exc:
        error_payload_builder = getattr(exc, "to_payload", None)
        error_payload = error_payload_builder() if callable(error_payload_builder) else {}
        qa_observation = error_payload.get("qa_observation") if isinstance(error_payload, dict) else None
        if qa_observation is None:
            # 工具层是 Agent 的稳定边界；即使服务层抛出未知异常，也要补齐最小观察结构。
            qa_observation = build_error_qa_observation(
                error_code="paper_qa_answer_failed",
                error_stage="answer_paper_question",
                error_reason=str(exc),
            )
        return make_tool_result(
            ok=False,
            tool_name=tool_name,
            summary="回答论文问题失败",
            data={
                "status": "failed",
                "arxiv_id": arxiv_id,
                "question": question,
                "answer": "",
                "sources": [],
                "retrieval_debug": None,
                "qa_observation": qa_observation,
                "error": (error_payload.get("message") if isinstance(error_payload, dict) else None) or str(exc),
            },
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
            error=make_tool_error("paper_qa_answer_failed", str(exc), detail={"qa_observation": qa_observation}),
        )
