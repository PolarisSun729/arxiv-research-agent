"""每轮规划的结构化记录（设计见 notes/agent-redesign.md D4-7 ⑤），用来在重构前拿基线。

记录只含元数据：来源、成败、失败类型、简短原因、耗时、步数、工具名，不含用户原话和 LLM 原始输出。
每轮在 planner 收尾时写进 planning_debug["planning_record"]，同时打一行
``agent_planning_record {json}`` 的 INFO 日志，由 scripts/planning_baseline_stats.py 统计。
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import ValidationError

logger = logging.getLogger(__name__)

PLANNING_RECORD_LOG_PREFIX = "agent_planning_record"
PLANNING_RECORD_SCHEMA_VERSION = 1

# 失败类型：timeout / llm_error / parse_error / validation_failed / none。
FAILURE_NONE = "none"
FAILURE_TIMEOUT = "timeout"
FAILURE_LLM_ERROR = "llm_error"
FAILURE_PARSE_ERROR = "parse_error"
FAILURE_VALIDATION_FAILED = "validation_failed"

# LLMPlanDraftGenerator.last_debug["stage"] 记录失败时走到哪一步，阶段决定失败类型。
_STAGE_FAILURE_KIND = {
    "llm_call": FAILURE_LLM_ERROR,
    "parse": FAILURE_PARSE_ERROR,
    "validate": FAILURE_VALIDATION_FAILED,
}

_MAX_REASONS = 5
_MAX_REASON_CHARS = 160


def _is_timeout_error(exc: BaseException) -> bool:
    """沿异常链找超时：openai.APITimeoutError、httpx.TimeoutException、内置 TimeoutError 都按类名认。"""
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or any("Timeout" in cls.__name__ for cls in type(current).__mro__):
            return True
        current = current.__cause__ or current.__context__
    return False


def classify_llm_plan_failure(exc: BaseException, stage: Optional[str]) -> str:
    if _is_timeout_error(exc):
        return FAILURE_TIMEOUT
    # 阶段未知说明连生成器都没建起来（例如拿不到 generation service），算 LLM 调用失败。
    return _STAGE_FAILURE_KIND.get(str(stage or ""), FAILURE_LLM_ERROR)


def _short(text: Any) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= _MAX_REASON_CHARS else value[: _MAX_REASON_CHARS - 1] + "…"


def failure_reasons(exc: BaseException, failure_kind: str, *, invalid_reasons: Sequence[str] = ()) -> List[str]:
    """把异常压成几条短原因；超时和调用报错只留异常类名，避免把服务端回显的内容写进日志。"""
    if failure_kind in {FAILURE_TIMEOUT, FAILURE_LLM_ERROR}:
        return [type(exc).__name__]
    if isinstance(exc, ValidationError):
        # pydantic 的报错文本带 input_value（LLM 原始片段），只保留字段路径和错误类型。
        reasons = [f"{'.'.join(str(part) for part in error.get('loc') or ())}: {error.get('type')}" for error in exc.errors()]
    elif invalid_reasons:
        reasons = list(invalid_reasons)
    else:
        reasons = str(exc).split("; ")
    return [_short(reason) for reason in reasons[:_MAX_REASONS] if str(reason).strip()]


def _elapsed_ms(started_at: float) -> int:
    return max(0, int(round((perf_counter() - started_at) * 1000)))


class PlanningRecorder:
    """一轮规划的记录器：planner 每走一条路径就 add_attempt 一次，收尾时 finish。"""

    def __init__(self, *, intent: Optional[str], goal_type: Optional[str], run_id: Optional[str], session_id: Optional[str]) -> None:
        self.started_at = perf_counter()
        self.intent = intent
        self.goal_type = goal_type
        self.run_id = run_id
        self.session_id = session_id
        self.attempts: List[Dict[str, Any]] = []

    def add_attempt(
        self,
        source: str,
        *,
        started_at: float,
        ok: bool,
        failure_kind: str = FAILURE_NONE,
        reasons: Iterable[str] = (),
        tools: Sequence[Any] = (),
        step_count: Optional[int] = None,
    ) -> None:
        tool_names = [str(name) for name in tools if str(name or "").strip()]
        self.attempts.append(
            {
                "source": source,
                "ok": bool(ok),
                "failure_kind": FAILURE_NONE if ok else failure_kind,
                "reasons": [] if ok else [_short(reason) for reason in list(reasons)[:_MAX_REASONS]],
                "elapsed_ms": _elapsed_ms(started_at),
                "step_count": len(tool_names) if step_count is None else int(step_count),
                "tools": tool_names,
            }
        )

    def finish(self, plan_source: Optional[str], *, plan_source_detail: Optional[str] = None) -> Dict[str, Any]:
        return {
            "schema_version": PLANNING_RECORD_SCHEMA_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "intent": self.intent,
            "goal_type": self.goal_type,
            "plan_attempts": list(self.attempts),
            "plan_source": plan_source,
            "plan_source_detail": plan_source_detail,
            "total_planning_ms": _elapsed_ms(self.started_at),
        }


def emit_planning_record(record: Mapping[str, Any]) -> None:
    """打一行固定前缀 + JSON 的 INFO 日志；记录失败不能影响规划本身。"""
    try:
        logger.info("%s %s", PLANNING_RECORD_LOG_PREFIX, json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
    except Exception as exc:  # pragma: no cover - 日志序列化失败只降级，不抛出
        logger.debug("agent planning record emit failed: %s", exc)


def state_run_id(state: Any) -> Optional[str]:
    context = getattr(state, "context", None)
    debug = getattr(state, "debug", None)
    value = (context.get("run_id") if isinstance(context, Mapping) else None) or (
        debug.get("run_id") if isinstance(debug, Mapping) else None
    )
    text = str(value or "").strip()
    return text or None


__all__ = [
    "FAILURE_LLM_ERROR",
    "FAILURE_NONE",
    "FAILURE_PARSE_ERROR",
    "FAILURE_TIMEOUT",
    "FAILURE_VALIDATION_FAILED",
    "PLANNING_RECORD_LOG_PREFIX",
    "PlanningRecorder",
    "classify_llm_plan_failure",
    "emit_planning_record",
    "failure_reasons",
    "state_run_id",
]
