"""在所有安全中间件之前，一次性确定请求的客户端 IP、协议与路由模板。

写入 scope 的键：
- security_client_ip：按可信代理规则解析出的客户端 IP；无法确定时为 "unknown"；
- security_proxy_headers_invalid：可信代理发来的 X-Forwarded-For / X-Forwarded-Proto 不合法时为 True，
  IPFilterMiddleware 据此拒绝请求；
- scheme：来自可信代理时改写为客户端实际使用的协议（http / https）；
- security_route：匹配到的路由模板（如 /api/paper/{arxiv_id}），未匹配时为 "<unmatched>"；
- security_path_params：完全匹配时的路径参数。

这些值是 IP 过滤、限流、认证授权和审计日志的共同输入。下游中间件通过 common.require_request_context
确认本中间件已经运行，注册顺序被改错时请求直接失败，而不是带着缺省值继续执行。
"""

from __future__ import annotations

import logging

from starlette.routing import Match
from starlette.types import ASGIApp, Receive, Scope, Send

from middleware.common import ROUTE_UNMATCHED
from middleware.ip_filter import IPFilterSettings


logger = logging.getLogger(__name__)
# proxy_misconfiguration 返回的问题标识 → 日志说明；每种问题在每个进程中只告警一次。
_MISCONFIGURATION_WARNINGS = {
    "client_address_missing": "ASGI 服务器没有提供客户端地址，所有请求的来源都记为 unknown，"
                              "IP 限流、登录限流与临时封禁将共用同一个计数。",
    "loopback_proxy_untrusted": "收到来自本机回环地址、带 X-Forwarded-For 的请求，但该地址不在 TRUSTED_PROXY_IPS 中："
                                "所有客户端都会被识别为代理地址，IP 限流、登录限流与临时封禁将共用同一个计数。"
                                "经同机 Nginx 转发时请设置 TRUSTED_PROXY_IPS=127.0.0.1,::1。",
}


class RequestContextMiddleware:
    """解析请求上下文的中间件（纯 ASGI）。注册在审计中间件之内、CORS 之外。

    位于 CORS 之外，CORS 直接应答的预检请求在审计日志里也有来源 IP 和路由模板。
    本中间件只写 scope，不拒绝请求；拒绝由 IPFilterMiddleware 等下游中间件决定。
    """

    def __init__(self, app: ASGIApp, *, routes: list, ip_settings: IPFilterSettings) -> None:
        # routes 传入 app.routes（同一个列表对象），之后注册的路由也能匹配到。
        self.app, self.routes, self.ip_settings = app, routes, ip_settings
        # 尚未告警过的配置问题。
        self._pending_warnings = set(_MISCONFIGURATION_WARNINGS)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # 只处理 HTTP 请求；lifespan 等其他类型直接透传。
        if scope["type"] == "http":
            self._resolve_client(scope)
            self._resolve_route(scope)
        await self.app(scope, receive, send)

    def _resolve_client(self, scope: Scope) -> None:
        """解析客户端 IP 与协议；代理头不合法时记为 unknown 并打上标记，由 IPFilterMiddleware 拒绝。"""
        try:
            client_ip = self.ip_settings.client_ip(scope)
            scheme = self.ip_settings.forwarded_scheme(scope)
        except ValueError:
            scope["security_client_ip"] = "unknown"
            scope["security_proxy_headers_invalid"] = True
            return
        scope["security_client_ip"] = client_ip
        if scheme is not None:
            # 禁用服务器自动代理解析后，由同一可信对端补回协议，避免 HTTPS 重定向降级为 HTTP。
            scope["scheme"] = scheme
        if self._pending_warnings:
            issue = self.ip_settings.proxy_misconfiguration(scope)
            if issue in self._pending_warnings:
                self._pending_warnings.discard(issue)
                logger.warning("security_proxy_misconfiguration code=%s %s", issue, _MISCONFIGURATION_WARNINGS[issue])

    def _resolve_route(self, scope: Scope) -> None:
        """匹配路由模板：与 Starlette 路由相同的规则，按注册顺序取第一个完全匹配。"""
        scope["security_route"] = ROUTE_UNMATCHED
        for route in self.routes:
            # FULL = 路径与方法都匹配；PARTIAL = 路径匹配但方法不匹配（如 OPTIONS 预检或 405）。
            match, matched_scope = route.matches(scope)
            if match is Match.PARTIAL and scope["security_route"] == ROUTE_UNMATCHED:
                # 预检和错误方法也保留匹配到的模板，不把原始路径参数写入日志。
                scope["security_route"] = getattr(route, "path", ROUTE_UNMATCHED)
            if match is Match.FULL:
                scope["security_route"] = getattr(route, "path", ROUTE_UNMATCHED)
                # 参数仅供身份绑定，不写入 path 审计字段，避免个人内容或凭据出现在 URL 日志中。
                scope["security_path_params"] = matched_scope.get("path_params", {})
                break
