"""在真正产生外部成本的位置扣减配额：账号模式按配额类型，密钥模式计入该密钥的日配额。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Callable

from starlette.concurrency import run_in_threadpool

from auth.context import current_auth


# 由密钥模式的限流中间件按请求绑定；可信本地调用和账号模式下为空。
api_key_charger: ContextVar[Callable[[], None] | None] = ContextVar("api_key_charger", default=None)


async def charge(quota_type: str) -> None:
    """超额时抛出 AuthError；没有认证上下文的可信本地调用不计费。"""
    # 先在当前协程取出上下文，再到线程池执行存储写入，不依赖线程池是否复制 ContextVar。
    context = current_auth.get()
    if context is not None:
        await run_in_threadpool(context.consume, quota_type)
        return
    charger = api_key_charger.get()
    if charger is not None:
        await run_in_threadpool(charger)
