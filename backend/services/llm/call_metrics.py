"""按请求统计真实模型调用；计数器不挂在全局服务上，避免并发问答互相污染。"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator


@dataclass
class LLMCallStats:
    calls: int = 0
    usage_count: int = 0
    total_usage_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    models: set[str] = field(default_factory=set)
    tasks: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            # 缺少 usage 的调用不能当成零 token；部分用量只在 observed 字段呈现。
            complete = self.calls == self.usage_count
            return {
                "llm_calls": self.calls,
                "input_tokens": self.input_tokens if complete else None,
                "output_tokens": self.output_tokens if complete else None,
                "total_tokens": self.total_tokens if self.calls == self.total_usage_count else None,
                "usage_reported_calls": self.total_usage_count,
                "observed_input_tokens": self.input_tokens,
                "observed_output_tokens": self.output_tokens,
                "observed_total_tokens": self.total_tokens,
                "models": sorted(self.models), "task_calls": dict(self.tasks),
            }


_CURRENT_STATS: ContextVar[LLMCallStats | None] = ContextVar("llm_call_stats", default=None)


@contextmanager
def use_call_stats(stats: LLMCallStats) -> Iterator[LLMCallStats]:
    """只包住实际执行；流式调用须在 yield 前退出，避免跨线程恢复 ContextVar。"""
    token = _CURRENT_STATS.set(stats)
    try:
        yield stats
    finally:
        _CURRENT_STATS.reset(token)


def record_llm_call(*, model: str, task_type: str) -> None:
    stats = _CURRENT_STATS.get()
    if stats is not None:
        with stats._lock:
            # 在 provider 调用前计数，超时和失败的真实请求同样计入成本。
            stats.calls += 1
            stats.models.add(str(model))
            stats.tasks[task_type] = stats.tasks.get(task_type, 0) + 1


def record_llm_usage(usage: Any) -> None:
    stats = _CURRENT_STATS.get()
    if stats is None or usage is None:
        return
    payload = usage if isinstance(usage, dict) else {
        key: getattr(usage, key, None)
        for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens")
    }
    input_tokens = payload.get("input_tokens")
    output_tokens = payload.get("output_tokens")
    if input_tokens is None:
        input_tokens = payload.get("prompt_tokens")
    if output_tokens is None:
        output_tokens = payload.get("completion_tokens")
    def valid(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    has_breakdown = valid(input_tokens) and valid(output_tokens)
    total_tokens = payload.get("total_tokens")
    if not valid(total_tokens) and has_breakdown:
        total_tokens = input_tokens + output_tokens
    with stats._lock:
        if has_breakdown:
            stats.usage_count += 1
            stats.input_tokens += input_tokens
            stats.output_tokens += output_tokens
        if valid(total_tokens):
            # 远程 rerank 可能只返回总 token；保留已知总量，不捏造输入/输出拆分。
            stats.total_usage_count += 1
            stats.total_tokens += total_tokens
