"""两种认证模式共用的有界请求体读取：先限制大小与读取时长，再交给业务解析。"""

from __future__ import annotations

import anyio
from starlette.types import Message, Receive

from auth.errors import AuthError


_READ_TIMEOUT_SECONDS = 30


async def read_bounded_body(receive: Receive, max_bytes: int) -> bytes:
    chunks, size = [], 0
    try:
        with anyio.fail_after(_READ_TIMEOUT_SECONDS):
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    raise AuthError("request_timeout")
                chunk = message.get("body", b"")
                size += len(chunk)
                if size > max_bytes:
                    raise AuthError("request_too_large")
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
    except TimeoutError:
        raise AuthError("request_timeout") from None
    return b"".join(chunks)


def replay_receive(body: bytes, receive: Receive) -> Receive:
    consumed = False

    async def replay() -> Message:
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        # 后续仍读取真实 disconnect，不能用重复的空 body 让 SSE 的断线观察陷入忙循环。
        return await receive()

    return replay
