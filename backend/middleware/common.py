"""安全中间件共用的响应与严格配置校验。"""

import os

from fastapi.responses import JSONResponse
from starlette.types import Scope

from auth.api_key_middleware import SECURITY_HEADERS
from core.errors import make_error_payload


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
