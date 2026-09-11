"""先认证与授权，再读取有界请求体；流式响应保持原始 ASGI 生命周期。"""

from __future__ import annotations

import json

import anyio
from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders, QueryParams
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from auth.api_key_middleware import SECURITY_HEADERS
from auth.context import AuthContext, bind_identity_fields, bind_user_id, current_auth
from auth.errors import AuthError
from auth.jwt_handler import JwtAuthenticator
from auth.permissions import PUBLIC_AUTH_ROUTES, ROUTE_POLICIES
from middleware.common import public_health


async def require_jwt_identity(request: Request) -> None:
    if not request.scope.get("auth_session"):
        raise AuthError("missing_token")


async def _validated_body(scope: Scope, receive: Receive, max_bytes: int) -> Receive:
    chunks, size = [], 0
    try:
        with anyio.fail_after(30):
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
    body = b"".join(chunks)
    if body:
        try:
            decoded = json.loads(body)
            bind_identity_fields(decoded)
            # 兼容原来单个 Body 字段的标量形式，也不能让它绕过 user_id 绑定。
            if scope.get("security_route") == "/api/user/generate-interest-vector" and isinstance(decoded, str):
                decoded = bind_user_id(decoded)
            if current_auth.get() is not None:
                # 显式空身份字段也必须规范化为本人，不能再被底层旧 Store 的默认值兜底成演示用户。
                body = json.dumps(decoded, ensure_ascii=False).encode("utf-8")
                MutableHeaders(scope=scope)["content-length"] = str(len(body))
        except (ValueError, UnicodeError, RecursionError):
            raise AuthError("request_validation_error") from None
    consumed = False

    async def replay() -> Message:
        nonlocal consumed
        if not consumed:
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        # 后续仍读取真实 disconnect，不能用重复的空 body 让 SSE 的断线观察陷入忙循环。
        return await receive()

    return replay


class JwtAuthMiddleware:
    def __init__(self, app: ASGIApp, *, authenticator: JwtAuthenticator) -> None:
        self.app, self.authenticator = app, authenticator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # 当前没有业务 WebSocket；默认拒绝，避免以后新增传输绕过 HTTP 权限矩阵。
            await send({"type": "websocket.close", "code": 1008})
            return

        response_started = False

        async def send_secure(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        route_key = (scope.get("method"), scope.get("path"))
        if public_health(scope) or route_key in PUBLIC_AUTH_ROUTES:
            scope["security_auth_public"] = True
            try:
                if route_key == ("POST", "/api/auth/register") and not self.authenticator.settings.registration_enabled:
                    raise AuthError("registration_disabled")
                if scope.get("method") == "POST":
                    receive = await _validated_body(scope, receive, self.authenticator.settings.max_body_bytes)
                await self.app(scope, receive, send_secure)
            except AuthError as exc:
                scope["security_error_code"] = exc.code
                if response_started:
                    raise
                await exc.to_response()(scope, receive, send_secure)
            return
        context_token = None
        try:
            candidates = Headers(scope=scope).getlist("authorization")
            if not candidates:
                raise AuthError("missing_token")
            if len(candidates) != 1:
                raise AuthError("invalid_token")
            parts = candidates[0].split(" ")
            if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
                raise AuthError("invalid_token")
            session = await run_in_threadpool(self.authenticator.authenticate, parts[1])
            scope["auth_session"] = session
            context = AuthContext(session, self.authenticator.store)
            context_token = current_auth.set(context)
            policy = ROUTE_POLICIES.get((scope.get("method"), scope.get("security_route")))
            if policy is None or session.user.role not in policy.roles:
                raise AuthError("insufficient_permissions")
            scope["auth_route_policy"] = policy
            for value in QueryParams(scope.get("query_string", b"").decode("latin-1")).getlist("user_id"):
                if value != context.user_id:
                    raise AuthError("identity_mismatch")
            path_user = scope.get("security_path_params", {}).get("user_id")
            if path_user is not None:
                if path_user != context.user_id:
                    raise AuthError("identity_mismatch")
            if scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
                receive = await _validated_body(scope, receive, self.authenticator.settings.max_body_bytes)
            await self.app(scope, receive, send_secure)
        except AuthError as exc:
            scope["security_error_code"] = exc.code
            if response_started:
                # 流已经开始后只能终止，不得追加第二套响应头伪装成完整的 401/429 JSON。
                raise
            await exc.to_response()(scope, receive, send_secure)
        finally:
            if context_token is not None:
                current_auth.reset(context_token)


class UserQuotaMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        context = current_auth.get()
        policy = scope.get("auth_route_policy")
        if scope["type"] == "http" and context and policy and policy.quota:
            try:
                # 流在发送响应头之前预留一次用量；失败/断线不退还已获准执行的资源预算。
                await run_in_threadpool(context.consume, policy.quota)
            except AuthError as exc:
                scope["security_error_code"] = exc.code
                await exc.to_response()(scope, receive, send)
                return
        await self.app(scope, receive, send)
