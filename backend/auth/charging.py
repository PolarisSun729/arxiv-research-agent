"""在真正产生外部成本的位置扣减配额：账号模式按配额类型，密钥模式计入该密钥的日配额。

业务代码只需调用 ``await charge("papers")`` 等，不必关心当前是哪种认证模式：
- JWT 账号模式：通过 current_auth 中的 AuthContext 扣减该账号对应类型的每日配额；
- API Key 模式：调用 RateLimitMiddleware 绑定的 api_key_charger，计入该密钥的 daily_quota
  （密钥配额不区分类型，quota_type 参数在此模式下不使用）；
- 两者都没有（可信的本地调用，例如脚本或测试）：不计费。

典型用法见 routers.paper_router：论文详情只有在本地缺失、需要回源 arXiv 时才调用 charge。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Callable

from starlette.concurrency import run_in_threadpool

from auth.context import current_auth


# 由密钥模式的限流中间件按请求绑定；可信本地调用和账号模式下为空。
# 绑定的函数在额度不足时抛出 AuthError（daily_quota_exceeded）。
api_key_charger: ContextVar[Callable[[], None] | None] = ContextVar("api_key_charger", default=None)


async def charge(quota_type: str) -> None:
    """超额时抛出 AuthError；没有认证上下文的可信本地调用不计费。

    quota_type 取值见 auth.models.QUOTA_TYPES（papers / qa_queries / agent_runs）。
    """
    # 先在当前协程取出上下文，再到线程池执行存储写入，不依赖线程池是否复制 ContextVar。
    context = current_auth.get()
    if context is not None:
        # 账号模式：在 SQLite 写事务中扣减，同步 IO 放到线程池，避免阻塞事件循环。
        await run_in_threadpool(context.consume, quota_type)
        return
    charger = api_key_charger.get()
    if charger is not None:
        # 密钥模式：在共享限流存储中原子递增该密钥当天的计数。
        await run_in_threadpool(charger)
