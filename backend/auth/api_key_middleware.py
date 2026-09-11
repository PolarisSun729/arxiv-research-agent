"""在读取业务请求体之前校验访问密钥，统一保护普通、流式和文件接口。"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.errors import ErrorCode, make_error_payload
from auth.key_config import ApiKeyPolicy, get_valid_api_keys as get_valid_api_keys, load_key_policies


_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False, description="管理员分配的后端访问密钥")
SECURITY_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}


def get_allowed_origins() -> tuple[str, ...]:
    origins = tuple(dict.fromkeys(origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if origin.strip()))
    if not origins:
        raise RuntimeError("ALLOWED_ORIGINS 不能为空，请配置明确的前端 Origin。")
    for origin in origins:
        try:
            parsed = urlsplit(origin)
            _ = parsed.port
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and "*" not in origin
                and not parsed.username
                and not parsed.password
                and not parsed.path
                and not parsed.query
                and not parsed.fragment
                and not any(char.isspace() for char in origin)
            )
        except ValueError:
            valid = False
        if not valid:
            # 不回显错误配置，防止误把带凭据的 URL 写入启动日志。
            raise RuntimeError("ALLOWED_ORIGINS 仅允许完整的 http/https Origin，不能含通配符、路径或凭据。")
    return origins


@dataclass(frozen=True)
class ApiKeySettings:
    policies: tuple[ApiKeyPolicy, ...]
    allowed_origins: tuple[str, ...]

    @property
    def key_hashes(self) -> tuple[bytes, ...]:
        return tuple(policy.digest for policy in self.policies)

    @classmethod
    def from_environment(cls) -> ApiKeySettings:
        return cls(
            policies=load_key_policies(),
            allowed_origins=get_allowed_origins(),
        )

    def accepts(self, candidate: str) -> bool:
        policy = self.match(candidate)
        return policy is not None and policy.rejection_code() is None

    def match(self, candidate: str) -> ApiKeyPolicy | None:
        if not candidate or len(candidate) > 256:
            return None
        candidate_hash = hashlib.sha256(candidate.encode("utf-8")).digest()
        matched = None
        for policy in self.policies:
            # 固定长度摘要逐一比较，避免 Unicode 头引发异常以及短路比较泄露密钥位置。
            if secrets.compare_digest(candidate_hash, policy.digest):
                matched = policy
        return matched


def _authentication_error(headers: Headers, settings: ApiKeySettings, scope: Scope) -> HTTPException | None:
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
    code = policy.rejection_code()
    if code:
        return HTTPException(status_code=403, detail=make_error_payload(code=code))
    return None


async def verify_api_key(request: Request, x_api_key: str | None = Security(_KEY_HEADER)) -> None:
    """路由依赖同时声明 OpenAPI 安全契约；已通过入口认证的请求不重复比较。"""
    if request.scope.get("api_key_authenticated"):
        return
    error = _authentication_error(request.headers, request.app.state.api_key_settings, request.scope)
    if error:
        raise error


class ApiKeyMiddleware:
    def __init__(self, app: ASGIApp, *, settings: ApiKeySettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        async def send_secure(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        # 仅健康检查与登录模式公开；OPTIONS 预检由外层 CORS 处理，不整体豁免其他路径。
        public_health = scope["type"] == "http" and scope.get("path") == "/health" and scope.get("method") in {"GET", "HEAD"}
        public_config = scope["type"] == "http" and scope.get("path") == "/api/auth/config" and scope.get("method") == "GET"
        if public_config:
            scope["security_auth_public"] = True
        if not public_health and not public_config:
            error = _authentication_error(Headers(scope=scope), self.settings, scope)
            if error:
                scope["security_error_code"] = error.detail["code"]
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    response = JSONResponse(status_code=error.status_code, content=error.detail, headers=error.headers)
                    await response(scope, receive, send_secure)
                return
            scope["api_key_authenticated"] = True

        await self.app(scope, receive, send_secure)
