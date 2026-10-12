"""安全中间件共用的响应与严格配置校验。

这里的环境变量读取函数都是"严格"的：取值不合法时直接抛出 RuntimeError 阻止应用启动，
而不是静默回退到默认值——安全配置写错后悄悄放宽，比启动失败更危险。
"""

import os
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse
from starlette.types import Scope

from core.errors import make_error_payload


# 未配置 ALLOWED_ORIGINS 时的默认值：只放行本机前端开发服务器（3000 / 5173 端口）。
_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
# 附加到每个响应上的安全头（审计中间件对所有响应强制写入，policy_error 也会带上）：
# - Cache-Control: no-store       响应可能含个人数据，禁止浏览器和中间代理缓存；
# - X-Content-Type-Options: nosniff  禁止浏览器猜测内容类型，防止把 JSON 当作脚本 / HTML 执行；
# - Referrer-Policy: no-referrer  跳转到外部链接时不泄露当前 URL。
SECURITY_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}
# 请求路径没有匹配到任何路由时，scope["security_route"] 的取值。
ROUTE_UNMATCHED = "<unmatched>"


def _normalize_origin(parsed) -> str:
    """把已校验的 Origin 规范为浏览器发送的形式：协议与主机小写，省略协议默认端口。

    浏览器的 Origin 头总是小写且不带默认端口；不规范化的话，配置成 HTTPS://Example.com 或
    https://example.com:443 能通过启动校验，却永远匹配不上真实请求，CORS 会静默失效。
    """
    netloc = parsed.netloc.lower()
    if parsed.port == {"http": 80, "https": 443}[parsed.scheme]:
        netloc = netloc.rsplit(":", 1)[0]
    return f"{parsed.scheme}://{netloc}"


def get_allowed_origins() -> tuple[str, ...]:
    """读取并校验 CORS 允许的来源列表（ALLOWED_ORIGINS，逗号分隔）。

    每一项都必须是完整的 Origin：http/https 协议 + 主机名（可带端口），不能含通配符、
    用户名密码、路径、查询串或片段。任意一项不合法都拒绝启动。
    合法项规范化（见 _normalize_origin）后按出现顺序去重。
    """
    candidates = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if origin.strip()]
    if not candidates:
        raise RuntimeError("ALLOWED_ORIGINS 不能为空，请配置明确的前端 Origin。")
    normalized = []
    for origin in candidates:
        try:
            parsed = urlsplit(origin)
            # 访问 port 属性会触发端口解析：端口不是数字或超出范围时抛出 ValueError。
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
        normalized.append(_normalize_origin(parsed))
    # dict.fromkeys 在去重的同时保留原始顺序。
    return tuple(dict.fromkeys(normalized))


def positive_env_int(name: str, default: int) -> int:
    """读取正整数环境变量；未设置时使用 default，取值不是正整数时拒绝启动。"""
    try:
        value = int(os.getenv(name, str(default)))
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        raise RuntimeError(f"{name} 必须是正整数。") from None


def env_bool(name: str, default: bool) -> bool:
    """读取布尔环境变量；只接受 true / false / 1 / 0（不区分大小写），其他取值拒绝启动。"""
    value = os.getenv(name, str(default)).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise RuntimeError(f"{name} 必须是 true 或 false。")
    return value in {"true", "1"}


def public_health(scope: Scope) -> bool:
    """判断请求是否为公开健康检查（GET / HEAD /health）。

    健康检查需要始终可达（负载均衡器、部署脚本依赖它），因此 IP 过滤、封禁和限流都会放行它。
    """
    return scope.get("path") == "/health" and scope.get("method") in {"GET", "HEAD"}


def require_request_context(scope: Scope) -> None:
    """确认外层的 RequestContextMiddleware 已写入客户端 IP 与路由模板，否则拒绝处理请求。

    IP 过滤、限流与认证授权都依赖 security_client_ip / security_route；缺失时如果按默认值继续，
    授权会匹配不到路由、配额与限流类别会被误判。这里抛出 RuntimeError，请求以 500 失败（失败即关闭），
    中间件注册顺序被改错时能立刻暴露。
    """
    if "security_route" not in scope or "security_client_ip" not in scope:
        raise RuntimeError("缺少请求上下文：RequestContextMiddleware 必须注册在所有安全中间件的外层。")


def policy_error(scope: Scope, code: str, status: int, *, retry_after: int | None = None, headers: dict | None = None) -> JSONResponse:
    """构造中间件层的统一错误响应（与业务层 AppError 使用同一套错误载荷格式）。

    同时把错误码写入 scope["security_error_code"]，供审计日志和外层限流中间件读取。
    返回的 JSONResponse 本身是 ASGI 应用，调用方式为 ``await policy_error(...)(scope, receive, send)``。
    """
    scope["security_error_code"] = code
    payload = make_error_payload(code=code)
    # 调用方传入的 headers（如 X-RateLimit-*）可以覆盖同名的默认安全头。
    response_headers = {**SECURITY_HEADERS, **(headers or {})}
    if retry_after is not None:
        # Retry-After 是整数秒，不能返回诸如 60/minute 的限流规则字符串。
        payload["retry_after"] = retry_after
        response_headers["Retry-After"] = str(retry_after)
    return JSONResponse(status_code=status, content=payload, headers=response_headers)
