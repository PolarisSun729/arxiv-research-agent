"""构造可重算的评测记录，并按日期落盘；观测写入失败不影响问答。"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.paper_evidence_research.contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult
from .contracts import EvaluationRecord
from .types import LLMUsage, EvalError

logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[2]
EVAL_RECORDS_DIR = BACKEND_DIR / "06-evaluation-result" / "records"


def _get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND_DIR,
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        logger.debug("获取评测版本失败", exc_info=True)
    return "unknown"


def _extract_main_intent(retrieval_debug: dict[str, Any] | None) -> str:
    profile = (retrieval_debug or {}).get("intent_profile")
    return str(profile.get("main_intent") or "unknown").strip().lower() if isinstance(profile, dict) else "unknown"



def _extract_configuration_from_trace(events: list[dict[str, Any]], request: PaperEvidenceResearchRequest) -> dict[str, Any]:
    """从 trace 统一提取配置，失败时返回最小契约"""
    config_event = next(
        (e for e in events if e.get("event_type") == "research_started" and e.get("configuration")),
        None
    )
    if config_event:
        return config_event["configuration"]

    # Fallback: 最小契约
    return {
        "research_limits": request.limits.model_dump(mode="json"),
        "engine": {},
    }


def _build_paper_context_from_trace(events: list[dict[str, Any]], arxiv_id: str) -> dict[str, Any]:
    """提取索引快照"""
    indexes = []
    for event in events:
        snapshot = event.get("index_snapshot")
        if isinstance(snapshot, dict) and snapshot not in indexes:
            indexes.append(snapshot)
    return {"arxiv_id": arxiv_id, "indexes": indexes}


def _extract_main_intent_from_trace(events: list[dict[str, Any]]) -> str:
    """从 trace 的 research_started 事件提取 main_intent"""
    config_event = next(
        (e for e in events if e.get("event_type") == "research_started"),
        None
    )
    if config_event:
        main_intent = config_event.get("main_intent")
        if main_intent:
            return main_intent.strip().lower()

    # Fallback: 从任意事件中查找
    main_intent = next((e["main_intent"] for e in events if e.get("main_intent")), None)
    return main_intent.strip().lower() if main_intent else "unknown"


def build_success_eval_record(
    *,
    request: PaperEvidenceResearchRequest,
    result: PaperEvidenceResearchResult,
    trace_events: list[dict[str, Any]],
    turn_id: str = "",
    latency_ms: float,
    llm_usage: dict[str, Any] | LLMUsage,
) -> dict[str, Any]:
    """构造成功执行的评测记录

    Args:
        request: 研究请求
        result: 研究结果
        trace_events: 轨迹事件列表（至少1个事件）
        turn_id: 对话轮次ID
        latency_ms: 延迟毫秒数（非负）
        llm_usage: LLM使用统计（dict或LLMUsage）

    Returns:
        评测记录字典

    Raises:
        ValueError: 参数验证失败
    """
    # 入口验证
    if not trace_events:
        raise ValueError("trace_events cannot be empty")
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")

    # 归一化 llm_usage
    usage = LLMUsage(**llm_usage) if isinstance(llm_usage, dict) else llm_usage
    usage_dict = usage.model_dump()

    # 提取数据（减少防御性代码）
    configuration = _extract_configuration_from_trace(trace_events, request)
    summary = result.research_summary.model_dump(mode="json")

    return EvaluationRecord(
        record_id=f"eval-{request.research_run_id}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=_get_git_commit_hash(),
        run_status="success",
        user_id=request.user_id,
        session_id=request.session_id,
        turn_id=turn_id,
        paper_context=_build_paper_context_from_trace(trace_events, request.arxiv_id),
        configuration=configuration,
        query={
            "raw": request.original_question,
            "rewritten": request.original_question,
            "main_intent": _extract_main_intent_from_trace(trace_events),
        },
        outcome=result.outcome,
        answer=result.answer,
        termination_reason=result.research_summary.termination_reason,
        citations=[c.model_dump(mode="json") for c in result.citations],
        research_summary=summary,
        efficiency={
            "latency_ms": round(latency_ms, 1),
            **usage_dict,
            "retrieval_count": summary.get("retrieval_count"),
            "draft_attempt_count": summary.get("draft_attempt_count"),
        },
        trace_ref=request.research_run_id,
        trace_events=trace_events,
        error=None,
    ).model_dump(mode="json")


def build_error_eval_record(
    *,
    request: PaperEvidenceResearchRequest,
    error: dict[str, Any] | EvalError,
    trace_events: list[dict[str, Any]],
    latency_ms: float,
    llm_usage: dict[str, Any] | LLMUsage | None = None,
) -> dict[str, Any]:
    """构造失败执行的评测记录

    Args:
        request: 研究请求
        error: 错误信息（dict或EvalError）
        trace_events: 轨迹事件列表（至少1个事件）
        latency_ms: 延迟毫秒数（非负）
        llm_usage: LLM使用统计（可选）

    Returns:
        评测记录字典

    Raises:
        ValueError: 参数验证失败
    """
    # 入口验证
    if not trace_events:
        raise ValueError("trace_events cannot be empty")
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")

    # 归一化
    error_obj = EvalError(**error) if isinstance(error, dict) else error
    usage = LLMUsage(**llm_usage) if isinstance(llm_usage, dict) else llm_usage if llm_usage else None

    configuration = _extract_configuration_from_trace(trace_events, request)

    return EvaluationRecord(
        record_id=f"eval-{request.research_run_id}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=_get_git_commit_hash(),
        run_status="error",
        user_id=request.user_id,
        session_id=request.session_id,
        turn_id="",
        paper_context=_build_paper_context_from_trace(trace_events, request.arxiv_id),
        configuration=configuration,
        query={
            "raw": request.original_question,
            "rewritten": request.original_question,
            "main_intent": _extract_main_intent_from_trace(trace_events),
        },
        outcome=None,
        answer="",
        termination_reason=None,
        citations=[],
        research_summary={},
        efficiency={
            "latency_ms": round(latency_ms, 1),
            **(usage.model_dump() if usage else {}),
        },
        trace_ref=request.research_run_id,
        trace_events=trace_events,
        error=error_obj.model_dump(),
    ).model_dump(mode="json")


def build_eval_record(
    *, request: PaperEvidenceResearchRequest, result: PaperEvidenceResearchResult | None = None,
    turn_id: str | None = None, retrieval_debug: dict[str, Any] | None = None,
    trace_events: list[dict[str, Any]] | None = None, latency_ms: float = 0.0,
    llm_usage: dict[str, Any] | None = None, raw_question: str | None = None,
    main_intent: str | None = None, error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """【已废弃】向后兼容的构造器；新代码应使用 build_success_eval_record 或 build_error_eval_record"""
    events = list(trace_events or [])
    if main_intent is None:
        main_intent = _extract_main_intent(retrieval_debug)
        if main_intent == "unknown":
            main_intent = next((str(e["main_intent"]) for e in events if e.get("main_intent")), "unknown")
    summary = result.research_summary.model_dump(mode="json") if result else {}
    indexes = []
    for event in events:
        snapshot = event.get("index_snapshot")
        if isinstance(snapshot, dict) and snapshot not in indexes:
            indexes.append(snapshot)
    configuration = next((event["configuration"] for event in events
                          if event.get("event_type") == "research_started" and event.get("configuration")),
                         {"research_limits": request.limits.model_dump(mode="json"), "engine": {}})
    return EvaluationRecord(
        record_id=f"eval-{request.research_run_id}", timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=_get_git_commit_hash(), run_status="error" if error is not None else "success",
        user_id=request.user_id, session_id=request.session_id, turn_id=turn_id or "",
        paper_context={"arxiv_id": request.arxiv_id, "indexes": indexes}, configuration=configuration,
        query={"raw": request.original_question if raw_question is None else raw_question,
               "rewritten": request.original_question, "main_intent": main_intent},
        outcome=result.outcome if result and error is None else None,
        answer=result.answer if result else "",
        termination_reason=summary.get("termination_reason"),
        citations=[c.model_dump(mode="json") for c in result.citations] if result else [],
        research_summary=summary,
        efficiency={"latency_ms": round(latency_ms, 1), "llm_calls": None,
                    "input_tokens": None, "output_tokens": None, "total_tokens": None,
                    **(llm_usage or {}), "retrieval_count": summary.get("retrieval_count"),
                    "draft_attempt_count": summary.get("draft_attempt_count")},
        trace_ref=request.research_run_id, trace_events=events, error=error,
    ).model_dump(mode="json")


def write_eval_record(*, record: dict[str, Any] | None = None, **kwargs: Any) -> str | None:
    """构造和写盘均在容错边界内，避免评测异常破坏已经完成的问答。"""
    try:
        payload = EvaluationRecord.model_validate(record).model_dump(mode="json") if record is not None else build_eval_record(**kwargs)
        output_dir = EVAL_RECORDS_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_dir.mkdir(parents=True, exist_ok=True)
        # run_id 允许由调用方提供，只用于文件名，不能改变评测产物目录。
        safe_id = re.sub(r"[^0-9A-Za-z._-]+", "_", payload["trace_ref"]).strip(".") or "research"
        output_path = output_dir / f"{safe_id}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output_path)
    except Exception:
        logger.warning("评测记录写入失败，保留问答结果", exc_info=True)
        return None
