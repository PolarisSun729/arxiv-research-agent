"""在读取业务请求体之前校验访问密钥，统一保护普通、流式和文件接口。

仅在 AUTH_MODE=api_key 时启用。处理流程：
1. 从 X-API-Key 请求头取出密钥，按 SHA-256 摘要与 key_config 加载的密钥策略逐一比较；
2. 命中后把策略写入 scope["api_key_policy"]，由后续的 RateLimitMiddleware 按该密钥做速率与日配额控制；
3. 写方法（POST/PUT/PATCH/DELETE）在认证通过后先有界读取请求体，再把读到的内容重放给业务应用。

本模块通过 ASGI scope 与其他组件约定以下键：
- api_key_policy：命中的密钥策略；即使随后因禁用或过期被拒绝，也已经写入；
- api_key_authenticated：入口认证已通过，路由依赖 verify_api_key 据此跳过重复比较，限流中间件据此确认身份；
- security_auth_public：公开路由标记，限流中间件对其改用匿名入口的预算；
- security_error_code：拒绝原因，供审计日志和 IP 滥用统计读取。
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from fastapi import HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.errors import ErrorCode, make_error_payload
from auth.errors import AuthError
from auth.key_config import ApiKeyPolicy, load_key_policies
from auth.request_body import read_bounded_body, replay_receive
from middleware.common import SECURITY_HEADERS, get_allowed_origins, positive_env_int, public_health


# 主要用于在 OpenAPI 文档中声明 X-API-Key 安全方案；auto_error=False 表示缺少请求头时
# 不由 FastAPI 抛出默认错误，而是交给本模块返回统一格式的错误体。
_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False, description="管理员分配的后端访问密钥")


@dataclass(frozen=True)
class ApiKeySettings:
    """API Key 模式的运行配置：应用启动时构造一次，之后只读。"""

    # 所有已配置密钥的策略（只保存摘要，不保存明文）。
    policies: tuple[ApiKeyPolicy, ...]
    # 允许跨域访问的前端 Origin 白名单。
    allowed_origins: tuple[str, ...]
    # 写请求请求体的字节上限，默认 1 MiB。
    max_body_bytes: int = 1048576

    @property
    def key_hashes(self) -> tuple[bytes, ...]:
        """所有已配置密钥的 SHA-256 摘要。"""
        return tuple(policy.digest for policy in self.policies)

    @classmethod
    def from_environment(cls) -> ApiKeySettings:
        """从环境变量和密钥配置文件加载配置；任何一项不合法都会抛出 RuntimeError，阻止应用启动。"""
        return cls(
            policies=load_key_policies(),
            allowed_origins=get_allowed_origins(),
            max_body_bytes=positive_env_int("AUTH_MAX_REQUEST_BYTES", 1048576),
        )

    def match(self, candidate: str) -> ApiKeyPolicy | None:
        """返回与候选密钥匹配的策略；没有匹配时返回 None。"""
        # 合法密钥最长 256 个字符（见 key_config._validate_secret），更长的输入直接拒绝，不再计算哈希。
        if not candidate or len(candidate) > 256:
            return None
        candidate_hash = hashlib.sha256(candidate.encode("utf-8")).digest()
        matched = None
        for policy in self.policies:
            # 固定长度摘要逐一比较，避免 Unicode 头引发异常以及短路比较泄露密钥位置。
            # 命中后也不提前 break，使比较次数与命中的是第几个密钥无关。
            if secrets.compare_digest(candidate_hash, policy.digest):
                matched = policy
        return matched


def _authentication_error(headers: Headers, settings: ApiKeySettings, scope: Scope) -> HTTPException | None:
    """检查请求头中的访问密钥，返回应当拒绝时的 HTTPException；认证通过则返回 None。

    中间件和路由依赖共用这一套判断，保证两处返回的状态码和错误码一致：
    - 缺少密钥或密钥为空：401 missing_api_key；
    - 存在多个 X-API-Key 头或密钥不匹配：403 invalid_api_key；
    - 密钥已禁用或已过期：403，错误码由 ApiKeyPolicy.rejection_code 给出。

    副作用：只要密钥匹配到策略，就会把该策略写入 scope["api_key_policy"]。
    """
    candidates = headers.getlist("x-api-key")
    if not candidates or not candidates[0]:
        return HTTPException(
            status_code=401,
            detail=make_error_payload(code=ErrorCode.MISSING_API_KEY),
            headers={"WWW-Authenticate": "ApiKey"},
        )
    # 重复头可能被代理和框架用不同方式合并；拒绝歧义，不能只挑其中一个有效值。
    if len(candidates) != 1:
        return HTTPException(status_code=403, detail=make_error_payload(code=ErrorCode.INVALID_API_KEY))
    policy = settings.match(candidates[0])
    if policy is None:
        return HTTPException(status_code=403, detail=make_error_payload(code=ErrorCode.INVALID_API_KEY))
    scope["api_key_policy"] = policy
    # 密钥存在但被禁用或已过期时，返回更具体的错误码。
    code = policy.rejection_code()
    if code:
        return HTTPException(status_code=403, detail=make_error_payload(code=code))
    return None


async def verify_api_key(request: Request, x_api_key: str | None = Security(_KEY_HEADER)) -> None:
    """路由依赖同时声明 OpenAPI 安全契约；已通过入口认证的请求不重复比较。

    x_api_key 参数只用于让接口文档显示需要 X-API-Key，实际判断仍读取完整请求头，
    这样才能识别重复的 X-API-Key 头。入口中间件出现遗漏时，这里作为第二道防线。
    """
    if request.scope.get("api_key_authenticated"):
        return
    error = _authentication_error(request.headers, request.app.state.api_key_settings, request.scope)
    if error:
        raise error


class ApiKeyMiddleware:
    """API Key 模式的入口认证中间件。

    采用纯 ASGI 实现而不是 BaseHTTPMiddleware，不缓冲响应体，SSE 等流式响应能保持原有的发送节奏。
    """

    def __init__(self, app: ASGIApp, *, settings: ApiKeySettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # lifespan 等非请求事件直接放行。
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        async def send_secure(message: Message) -> None:
            # 所有经过本中间件发出的响应（包括拒绝响应）都在响应头阶段补上统一的安全头。
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        # 仅健康检查与登录模式公开；OPTIONS 预检由外层 CORS 处理，不整体豁免其他路径。
        health = scope["type"] == "http" and public_health(scope)
        # 前端通过 GET /api/auth/config 查询当前认证模式，此时还没有凭据，因此必须公开。
        public_config = scope["type"] == "http" and scope.get("path") == "/api/auth/config" and scope.get("method") == "GET"
        if public_config:
            scope["security_auth_public"] = True
        if not health and not public_config:
            error = _authentication_error(Headers(scope=scope), self.settings, scope)
            if error:
                scope["security_error_code"] = error.detail["code"]
                if scope["type"] == "websocket":
                    # 1008 = Policy Violation：WebSocket 握手阶段无法返回 JSON，只能直接关闭连接。
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    response = JSONResponse(status_code=error.status_code, content=error.detail, headers=error.headers)
                    await response(scope, receive, send_secure)
                return
            scope["api_key_authenticated"] = True
            if scope["type"] == "http" and scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
                # 与 JWT 模式相同：认证通过后先有界读取请求体，慢速或超大上传不能长期占用 worker。
                try:
                    body = await read_bounded_body(receive, self.settings.max_body_bytes)
                except AuthError as exc:
                    scope["security_error_code"] = exc.code
                    await exc.to_response()(scope, receive, send_secure)
                    return
                # 请求体已被读走，替换 receive，让下游仍能按 ASGI 协议读到同一份内容。
                receive = replay_receive(body, receive)

        await self.app(scope, receive, send_secure)
