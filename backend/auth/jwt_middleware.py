"""先认证与授权，再读取有界请求体；流式响应保持原始 ASGI 生命周期。

仅在 AUTH_MODE=jwt 时启用。main.py 中的执行顺序为：
审计 → CORS → IP 过滤 → 认证前限流 → JwtAuthMiddleware → 用户速率限流 → UserQuotaMiddleware → 业务路由。

JwtAuthMiddleware 对每个非公开请求依次执行：
1. 解析 Authorization: Bearer <token>，校验令牌并在数据库中确认会话有效；
2. 把 AuthContext 绑定到 current_auth，供下游的路由、服务和 Agent 工具使用；
3. 按 (方法, 路由模板) 查询权限清单 ROUTE_POLICIES，未登记的路由一律拒绝；
4. 校验查询参数和路径参数中的 user_id 必须是本人；
5. 对写方法有界读取 JSON 请求体，把其中所有 user_id 字段规范化为本人后再交给业务。
"""

from __future__ import annotations

import json

from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders, QueryParams
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from auth.context import AuthContext, bind_identity_fields, bind_user_id, current_auth
from auth.errors import AuthError
from auth.jwt_handler import JwtAuthenticator
from auth.permissions import PUBLIC_AUTH_ROUTES, ROUTE_POLICIES
from auth.request_body import read_bounded_body, replay_receive
from middleware.common import SECURITY_HEADERS, public_health, require_request_context


async def require_jwt_identity(request: Request) -> None:
    """业务路由的依赖项：确认请求已由 JwtAuthMiddleware 认证，作为中间件之外的第二道防线。"""
    if not request.scope.get("auth_session"):
        raise AuthError("missing_token")


async def _validated_body(scope: Scope, receive: Receive, max_bytes: int) -> Receive:
    """有界读取请求体，绑定其中的身份字段，返回一个能重放（可能已改写的）请求体的 receive。

    请求体非空时必须是合法 JSON，否则抛出 request_validation_error。
    """
    body = await read_bounded_body(receive, max_bytes)
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
                # 请求体被重新序列化后长度可能变化，同步更新 content-length，避免下游按旧长度解析。
                MutableHeaders(scope=scope)["content-length"] = str(len(body))
        except (ValueError, UnicodeError, RecursionError):
            # ValueError 覆盖 JSON 格式错误；RecursionError 来自嵌套过深的 JSON。
            raise AuthError("request_validation_error") from None
    return replay_receive(body, receive)


class JwtAuthMiddleware:
    """JWT 模式的入口认证与授权中间件（纯 ASGI 实现，不缓冲流式响应）。

    写入 scope 的键：
    - auth_session：认证成功的 AuthSession，供限流中间件、审计日志和 require_jwt_identity 使用；
    - auth_route_policy：本次请求命中的 AccessPolicy，UserQuotaMiddleware 据此扣减配额；
    - security_auth_public：公开路由标记，限流中间件对其改用匿名入口的预算；
    - security_error_code：拒绝原因，供审计日志和 IP 滥用统计读取。

    依赖外层 RequestContextMiddleware 预先写入的 security_route（路由模板）和 security_path_params（路径参数），
    缺失时直接抛出 RuntimeError（见 middleware.common.require_request_context）。
    """

    def __init__(self, app: ASGIApp, *, authenticator: JwtAuthenticator) -> None:
        self.app, self.authenticator = app, authenticator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # lifespan 等非请求事件直接放行。
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # 当前没有业务 WebSocket；默认拒绝，避免以后新增传输绕过 HTTP 权限矩阵。
            await send({"type": "websocket.close", "code": 1008})
            return
        # 授权依赖路由模板与路径参数；请求上下文缺失时直接失败，不能带着缺省值去查权限清单。
        require_request_context(scope)

        # 记录响应头是否已经发出；发出之后就不能再改为返回错误响应。
        response_started = False

        async def send_secure(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                # 所有经过本中间件的响应都补上统一的安全响应头。
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        # ---- 公开路由：健康检查、查询认证模式、登录、注册，无需令牌 ----
        # 公开路由都没有路径参数，可以直接用原始 path 匹配。
        route_key = (scope.get("method"), scope.get("path"))
        if public_health(scope) or route_key in PUBLIC_AUTH_ROUTES:
            scope["security_auth_public"] = True
            try:
                # 公开注册默认关闭，需显式设置 ALLOW_PUBLIC_REGISTRATION=true。
                if route_key == ("POST", "/api/auth/register") and not self.authenticator.settings.registration_enabled:
                    raise AuthError("registration_disabled")
                # 登录/注册同样先有界读取请求体，防止匿名用户用慢速或超大上传占用 worker。
                if scope.get("method") == "POST":
                    receive = await _validated_body(scope, receive, self.authenticator.settings.max_body_bytes)
                await self.app(scope, receive, send_secure)
            except AuthError as exc:
                scope["security_error_code"] = exc.code
                if response_started:
                    raise
                await exc.to_response()(scope, receive, send_secure)
            return

        # ---- 受保护路由 ----
        context_token = None
        try:
            # 1. 解析 Authorization 头：必须恰好一个，格式为 "Bearer <token>"（scheme 不区分大小写）。
            candidates = Headers(scope=scope).getlist("authorization")
            if not candidates:
                raise AuthError("missing_token")
            if len(candidates) != 1:
                raise AuthError("invalid_token")
            parts = candidates[0].split(" ")
            if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
                raise AuthError("invalid_token")
            # 2. 校验令牌；其中包含 SQLite 查询，放到线程池执行，避免阻塞事件循环。
            session = await run_in_threadpool(self.authenticator.authenticate, parts[1])
            scope["auth_session"] = session
            context = AuthContext(session, self.authenticator.store)
            context_token = current_auth.set(context)
            # 3. 按路由模板查权限清单；未登记的路由（包括 "<unmatched>"）一律拒绝。
            policy = ROUTE_POLICIES.get((scope.get("method"), scope.get("security_route")))
            if policy is None or session.user.role not in policy.roles:
                raise AuthError("insufficient_permissions")
            scope["auth_route_policy"] = policy
            # 4. 查询参数中的每一个 user_id 都必须是本人（防止 ?user_id=a&user_id=b 这类重复参数绕过）。
            for value in QueryParams(scope.get("query_string", b"").decode("latin-1")).getlist("user_id"):
                if value != context.user_id:
                    raise AuthError("identity_mismatch")
            # 路径参数中的 user_id（如 /api/user/preferences/{user_id}）同样必须是本人。
            path_user = scope.get("security_path_params", {}).get("user_id")
            if path_user is not None:
                if path_user != context.user_id:
                    raise AuthError("identity_mismatch")
            # 5. 写方法读取并改写请求体中的身份字段。
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
            # 请求结束后恢复 ContextVar，避免身份泄漏到复用同一上下文的后续任务。
            if context_token is not None:
                current_auth.reset(context_token)


class UserQuotaMiddleware:
    """JWT 模式下按路由扣减每日配额。

    位于用户速率限流之后、业务路由之前：只有当前路由在 ROUTE_POLICIES 中标记了 quota 时，
    才在调用业务之前扣减一次对应类型的配额。未标记的路由（以及没有认证上下文的公开路由）直接放行。
    这里收到的 send 已由外层 JwtAuthMiddleware 包装，拒绝响应同样带有安全响应头。
    """

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
