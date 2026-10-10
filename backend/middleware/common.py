"""安全中间件共用的响应与严格配置校验。"""

import os
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse
from starlette.types import Scope

from core.errors import make_error_payload


_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
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


def positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise RuntimeError(f"{name} 必须是正整数。") from None


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise RuntimeError(f"{name} 必须是 true 或 false。")
    return value in {"true", "1"}


def public_health(scope: Scope) -> bool:
    return scope.get("path") == "/health" and scope.get("method") in {"GET", "HEAD"}


def policy_error(scope: Scope, code: str, status: int, *, retry_after: int | None = None, headers: dict | None = None) -> JSONResponse:
    scope["security_error_code"] = code
    payload = make_error_payload(code=code)
    response_headers = {**SECURITY_HEADERS, **(headers or {})}
    if retry_after is not None:
        # Retry-After 是整数秒，不能返回诸如 60/minute 的限流规则字符串。
        payload["retry_after"] = retry_after
        response_headers["Retry-After"] = str(retry_after)
    return JSONResponse(status_code=status, content=payload, headers=response_headers)
