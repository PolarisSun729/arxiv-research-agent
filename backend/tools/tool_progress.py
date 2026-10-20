"""工具执行中途的进度上报。

长耗时工具（如论文全文问答）内部有多个阶段，调用方需要在工具返回前就看到"现在做到哪了"。
工具只管调用 report_tool_progress；谁在听、怎么转发由调用方用 use_tool_progress_reporter 决定，
没有调用方监听时上报是空操作，工具在 Agent 之外（路由、脚本、测试）照常运行。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

ToolProgressReporter = Callable[[Dict[str, Any]], None]

_reporter: ContextVar[Optional[ToolProgressReporter]] = ContextVar("tool_progress_reporter", default=None)


@contextmanager
def use_tool_progress_reporter(reporter: Optional[ToolProgressReporter]) -> Iterator[None]:
    token = _reporter.set(reporter)
    try:
        yield
    finally:
        _reporter.reset(token)


def report_tool_progress(
    stage: str,
    label: str,
    *,
    status: str = "running",
    detail: Optional[str] = None,
    items: Optional[List[str]] = None,
) -> None:
    """上报一个阶段的开始（running）或结束（done）；label/detail/items 是直接给用户看的中文文案。

    detail 是跟在阶段名后面的一句说明，items 是可展开查看的明细列表（如拆出的各个证据需求）。
    """
    reporter = _reporter.get()
    if reporter is None:
        return
    payload: Dict[str, Any] = {"stage": stage, "label": label, "status": status}
    if detail:
        payload["detail"] = detail
    if items:
        payload["items"] = list(items)
    try:
        reporter(payload)
    except Exception:
        # 进度只是展示，转发失败不能打断工具本身。
        logger.warning("工具进度上报失败: stage=%s", stage, exc_info=True)
