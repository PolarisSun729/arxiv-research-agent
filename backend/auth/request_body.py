"""两种认证模式共用的有界请求体读取：先限制大小与读取时长，再交给业务解析。

ASGI 的请求体只能从 receive 读取一次。中间件读完后，需要用 replay_receive 构造一个新的 receive
交给下游应用，让业务代码仍然能正常读到同一份请求体。
"""

from __future__ import annotations

import anyio
from starlette.types import Message, Receive

from auth.errors import AuthError


# 读取整个请求体的总时长上限（秒），而不是单个分块的上限，
# 防止客户端每次只发送少量字节、长时间占用连接（slowloris 类攻击）。
_READ_TIMEOUT_SECONDS = 30


async def read_bounded_body(receive: Receive, max_bytes: int) -> bytes:
    """完整读取请求体并返回。

    - 累计超过 max_bytes：抛出 request_too_large，已收到的数据不再继续缓存；
    - 总耗时超过 _READ_TIMEOUT_SECONDS，或客户端中途断开：抛出 request_timeout。
    """
    chunks, size = [], 0
    try:
        with anyio.fail_after(_READ_TIMEOUT_SECONDS):
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    raise AuthError("request_timeout")
                chunk = message.get("body", b"")
                size += len(chunk)
                # 先检查累计大小再缓存，超限的分块不会进入内存中的列表。
                if size > max_bytes:
                    raise AuthError("request_too_large")
                chunks.append(chunk)
                # more_body 为 False 表示这是最后一个分块。
                if not message.get("more_body", False):
                    break
    except TimeoutError:
        raise AuthError("request_timeout") from None
    return b"".join(chunks)


def replay_receive(body: bytes, receive: Receive) -> Receive:
    """返回一个新的 receive：第一次调用一次性返回完整的 body，之后转发给原始 receive。"""
    consumed = False

    async def replay() -> Message:
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        # 后续仍读取真实 disconnect，不能用重复的空 body 让 SSE 的断线观察陷入忙循环。
        return await receive()

    return replay
