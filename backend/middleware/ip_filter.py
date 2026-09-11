"""解析可信代理后的客户端 IP，并执行 IPv4/IPv6 地址及网段策略。"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from middleware.common import policy_error, public_health


def _networks(value: str) -> tuple:
    try:
        # 配置错误必须阻止启动，不能忽略拼写错误后意外放宽访问范围。
        networks = []
        for item in value.split(","):
            if not item.strip():
                continue
            if "%" in item:
                raise ValueError
            network = ipaddress.ip_network(item.strip(), strict=False)
            if isinstance(network, ipaddress.IPv6Network) and network.prefixlen >= 96 and network.network_address.ipv4_mapped:
                # 配置与请求地址使用相同的 IPv4 规范形式，防止映射地址在黑白名单里失配。
                network = ipaddress.IPv4Network((network.network_address.ipv4_mapped, network.prefixlen - 96))
            networks.append(network)
        return tuple(networks)
    except ValueError:
        raise RuntimeError("IP 配置必须是合法的 IPv4/IPv6 地址或 CIDR 网段，不能使用通配符。") from None


def _address(value: str):
    if "%" in value:
        raise ValueError("不接受带接口作用域的客户端地址")
    address = ipaddress.ip_address(value)
    return address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address


def _contains(address, networks: tuple) -> bool:
    return any(address.version == network.version and address in network for network in networks)


def _peer_address(scope: Scope):
    try:
        return _address(scope.get("client", ("", 0))[0])
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class IPFilterSettings:
    mode: str
    blacklist: tuple
    whitelist: tuple
    trusted_proxies: tuple

    @classmethod
    def from_environment(cls) -> IPFilterSettings:
        mode = os.getenv("IP_FILTER_MODE", "disabled").strip().lower()
        if mode not in {"disabled", "blacklist", "whitelist"}:
            raise RuntimeError("IP_FILTER_MODE 必须是 disabled、blacklist 或 whitelist。")
        settings = cls(mode, _networks(os.getenv("IP_BLACKLIST", "")), _networks(os.getenv("IP_WHITELIST", "")), _networks(os.getenv("TRUSTED_PROXY_IPS", "")))
        if mode == "whitelist" and not settings.whitelist:
            raise RuntimeError("白名单模式要求非空 IP_WHITELIST，避免配置遗漏导致服务无法使用。")
        return settings

    def client_ip(self, scope: Scope) -> str:
        peer = _peer_address(scope)
        if peer is None:
            # 测试或不提供地址的 ASGI 服务器共享 unknown 桶，不能靠随机头创建无限个计数桶。
            return "unknown"
        values = Headers(scope=scope).getlist("x-forwarded-for")
        if not values or not _contains(peer, self.trusted_proxies):
            return str(peer)
        if len(values) != 1 or len(values[0]) > 2048:
            raise ValueError("代理地址头不合法")
        hops = values[0].split(",")
        if not 1 <= len(hops) <= 20:
            raise ValueError("代理跳数不合法")
        chain = [_address(hop.strip()) for hop in hops] + [peer]
        # 从 TCP 对端向左剥离可信代理，首个不可信节点才是客户端，不能盲信最左边的值。
        while len(chain) > 1 and _contains(chain[-1], self.trusted_proxies):
            chain.pop()
        return str(chain[-1])

    def forwarded_scheme(self, scope: Scope) -> str | None:
        peer = _peer_address(scope)
        if peer is None or not _contains(peer, self.trusted_proxies):
            return None
        values = Headers(scope=scope).getlist("x-forwarded-proto")
        if not values:
            return None
        if len(values) != 1 or values[0] not in {"http", "https"}:
            raise ValueError("代理协议头不合法")
        # 禁用服务器自动代理解析后，由同一可信对端补回协议，避免 HTTPS 重定向降级为 HTTP。
        return values[0]

    def rejection_code(self, client_ip: str) -> str | None:
        if self.mode == "disabled":
            return None
        if client_ip == "unknown":
            return "ip_not_allowed"
        address = _address(client_ip)
        if _contains(address, self.blacklist):
            return "ip_blocked"
        if self.mode == "whitelist" and not _contains(address, self.whitelist):
            return "ip_not_allowed"
        return None


class IPFilterMiddleware:
    def __init__(self, app: ASGIApp, *, settings: IPFilterSettings) -> None:
        self.app, self.settings = app, settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            scope["security_client_ip"] = self.settings.client_ip(scope)
            scheme = self.settings.forwarded_scheme(scope)
            if scheme is not None:
                scope["scheme"] = scheme
            code = self.settings.rejection_code(scope["security_client_ip"])
        except ValueError:
            # 可信代理发送歧义或无效头时拒绝，不能退回代理 IP 以绕过白名单。
            scope["security_client_ip"] = "unknown"
            code = "ip_not_allowed"
        if code and not public_health(scope):
            await policy_error(scope, code, 403)(scope, receive, send)
            return
        await self.app(scope, receive, send)
