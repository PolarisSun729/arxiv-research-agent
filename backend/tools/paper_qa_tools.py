from __future__ import annotations

import logging
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
from .tool_progress import report_tool_progress
from .tool_result import make_tool_error, make_tool_result, make_tool_trace

logger = logging.getLogger(__name__)

# 未知异常对 Agent 暴露的固定文案：异常字符串可能包含 provider 响应体或服务器路径，只写入服务端日志。
PAPER_QA_ANSWER_FAILED_MESSAGE = "论文问答执行失败，请稍后重试"


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


def _clip(text: Any, limit: int = 60) -> str:
    value = str(text or "").strip()
    return f"{value[:limit]}…" if len(value) > limit else value


_ACTION_LABELS = {"search_paper": "检索", "draft_answer": "撰写回答", "finalize_answer": "定稿", "abstain": "放弃作答"}
# 门禁拒绝决策的原因；预算类原因出现时研究随即结束，用户需要知道为什么停在这一版。
_REJECTION_REASONS = {
    "RETRIEVAL_BUDGET_EXHAUSTED": "检索轮数已用完",
    "DRAFT_BUDGET_EXHAUSTED": "修订次数已用完",
    "NO_PROGRESS_LIMIT_REACHED": "继续检索也找不到新证据",
    "TARGET_NEED_NOT_OPEN": "这个需求已经有证据覆盖",
    "EVIDENCE_CANDIDATES_EMPTY": "还没有可用的证据",
    "SEARCH_QUERY_EMPTY": "没有给出检索词",
}


def _need_list(numbers: Any) -> str:
    return "、".join(f"需求 {int(number)}" for number in numbers or [])


def _decision_detail(event: Mapping[str, Any]) -> str:
    action = event.get("action")
    if action == "search_paper":
        return f"→ 检索需求 {event['need_number']}" if event.get("need_number") else "→ 继续检索"
    if action == "draft_answer":
        covered = _need_list(event.get("need_numbers"))
        return ("→ 修订回答" if event.get("revision") else "→ 撰写回答") + (f"，覆盖{covered}" if covered else "")
    if action == "finalize_answer":
        return "→ 定稿"
    if action == "abstain":
        return "→ 证据不足，放弃作答"
    return ""


def _report_research_progress(event: Mapping[str, Any]) -> None:
    """把证据研究引擎的阶段事件翻译成给用户看的工具进度；未列出的事件不上报。"""
    kind = event.get("event")
    stage = str(event.get("stage") or "")
    if kind == "stage_started":
        draft_attempt = int(event.get("draft_attempt") or 1)
        if stage == "prepare":
            report_tool_progress(stage, "分析问题、拆分证据需求")
        elif stage == "decide":
            report_tool_progress(stage, "规划下一步")
        elif stage == "search":
            need = f"需求 {event['need_number']}：" if event.get("need_number") else "查找："
            report_tool_progress(
                stage,
                f"检索论文片段（第 {int(event.get('retrieval_round') or 1)} 轮）",
                detail=f"{need}{_clip(event['need_description'])}" if event.get("need_description") else None,
            )
        elif stage == "draft":
            report_tool_progress(stage, "撰写回答" if draft_attempt <= 1 else f"修订回答（第 {draft_attempt} 版）")
        elif stage == "extract_claims":
            report_tool_progress(stage, "拆分回答要点")
        elif stage == "verify_claims":
            report_tool_progress(stage, "逐条核对原文证据")
    elif kind == "question_analyzed":
        needs = [need for need in event.get("needs") or [] if isinstance(need, Mapping)]
        report_tool_progress(
            "prepare", "", status="done", detail=f"拆出 {int(event.get('need_count') or 0)} 个证据需求",
            items=[
                f"需求 {need.get('number')}（{'核心' if need.get('importance') == 'core' else '辅助'}）：{_clip(need.get('description'), 80)}"
                for need in needs
            ],
        )
    elif kind == "action_decided":
        detail = _decision_detail(event)
        if detail:
            report_tool_progress("decide", "", status="done", detail=detail)
    elif kind == "action_rejected":
        action = _ACTION_LABELS.get(str(event.get("action") or ""), "这一步")
        reason = _REJECTION_REASONS.get(str(event.get("rejection_code") or ""), "决策不合规，需要重新规划")
        report_tool_progress("decide", "", status="done", detail=f"→ 想{action}，但{reason}")
    elif kind == "retrieval_completed":
        if event.get("query"):
            report_tool_progress("search", "", status="done", detail=f"检索词：{_clip(event['query'])}")
        duplicates = int(event.get("duplicate_candidate_count") or 0)
        report_tool_progress(
            "search", "", status="done",
            detail=f"新增 {int(event.get('new_candidate_count') or 0)} 段证据" + (f"，{duplicates} 段已有" if duplicates else ""),
        )
    elif kind == "draft_created":
        report_tool_progress("draft", "", status="done")
    elif kind == "claims_extracted":
        detail = f"{int(event.get('claim_count') or 0)} 条要点"
        if event.get("extractor_source") == "rule":
            # 模型拆分失败退回按标点切句，标题、半句也会当成要点送检，核对结果会偏低。
            detail += "（模型拆分失败，已按标点切句兜底）"
        report_tool_progress("extract_claims", "", status="done", detail=detail)
    elif kind == "claim_verification_completed":
        supported = int(event.get("supported_count") or 0)
        total = int(event.get("claim_count") or 0)
        report_tool_progress(
            "verify_claims", "", status="done",
            detail=f"{supported}/{total} 条有原文支持" if total else f"{supported} 条有原文支持",
        )
    elif kind == "coverage_projected":
        total = int(event.get("need_count") or 0)
        satisfied = len(event.get("satisfied_need_numbers") or [])
        missing = _need_list(event.get("unsatisfied_need_numbers"))
        if total:
            report_tool_progress(
                "verify_claims", "", status="done",
                detail=f"需求覆盖 {satisfied}/{total}" + (f"（未覆盖：{missing}）" if missing else ""),
            )


def _answer_with_progress(arxiv_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # 与 PaperQAService.answer_question 消费同一条执行流，区别只是沿途把阶段进度转给调用方。
    stream = _get_paper_qa_service().answer_question_with_research_stream(arxiv_id, payload)
    while True:
        try:
            item = next(stream)
        except StopIteration as stop:
            return stop.value
        data = item.get("data") if isinstance(item, Mapping) else None
        if isinstance(data, Mapping):
            _report_research_progress(data)


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
        data = _answer_with_progress(arxiv_id, payload)
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
        # AppError 的 message 是面向用户的受控文案；其他异常一律使用固定文案，原始异常只进日志。
        error_message = (error_payload.get("message") if isinstance(error_payload, dict) else None) or PAPER_QA_ANSWER_FAILED_MESSAGE
        logger.exception("answer_paper_question failed: arxiv_id=%s run_id=%s", arxiv_id, run_id)
        if qa_observation is None:
            # 工具层是 Agent 的稳定边界；即使服务层抛出未知异常，也要补齐最小观察结构。
            # error_reason 会原样进入 observation_reason，必须是稳定错误码而不是 str(exc)。
            qa_observation = build_error_qa_observation(
                error_code="paper_qa_answer_failed",
                error_stage="answer_paper_question",
                error_reason="paper_qa_answer_failed",
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
                "error": error_message,
            },
            trace=make_tool_trace(tool_name, inputs=trace_inputs, source="paper_qa_service"),
            error=make_tool_error("paper_qa_answer_failed", error_message, detail={"qa_observation": qa_observation}),
        )
