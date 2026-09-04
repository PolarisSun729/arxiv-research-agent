"""LLM 依赖降级的结构化留痕。

降级是研究图的正常保险丝而非系统错误：LLM 调用失败或输出不合法时，器官必须回到
规则兜底并留下可审计的事件。事件通过监听器外抛，评测与排查直接消费，不进入图状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# LLM 调用本身抛错（网络、超时、鉴权等）。
REASON_LLM_CALL_FAILED = "LLM_CALL_FAILED"
# LLM 返回了内容，但不是合法 JSON 或未通过 schema 校验。
REASON_LLM_OUTPUT_INVALID = "LLM_OUTPUT_INVALID"


@dataclass(frozen=True)
class DependencyDegradation:
    organ: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {"organ": self.organ, "reason": self.reason, "detail": dict(self.detail)}


DegradationListener = Callable[[DependencyDegradation], None]


class DegradationLedger:
    """收集降级事件的最小容器，可直接作为监听器注入器官。

    器官实例与服务同生命周期、跨请求复用，因此用 max_events 封顶防止无界增长。
    """

    def __init__(self, *, max_events: int = 200) -> None:
        self.max_events = max_events
        self.events: list[DependencyDegradation] = []

    def __call__(self, event: DependencyDegradation) -> None:
        if len(self.events) < self.max_events:
            self.events.append(event)
