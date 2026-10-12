"""解析可信代理后的客户端 IP，并执行 IPv4/IPv6 地址及网段策略。

相关环境变量：
- IP_FILTER_MODE：disabled（默认，不过滤）、blacklist（拒绝黑名单）、whitelist（只允许白名单）；
- IP_BLACKLIST / IP_WHITELIST：逗号分隔的 IP 或 CIDR 网段；
- TRUSTED_PROXY_IPS：可信反向代理地址。只有 TCP 对端属于这些地址时，才解析 X-Forwarded-For /
  X-Forwarded-Proto；否则这些头一律忽略，防止客户端伪造来源 IP 绕过限流和黑白名单。

uvicorn / gunicorn 启动时关闭了自带的代理头解析（proxy_headers=False、--forwarded-allow-ips ""），
scope["client"] 始终是原始 TCP 对端，代理头只由本模块按 TRUSTED_PROXY_IPS 解释。

IPFilterSettings 负责解析与判定；客户端 IP 与协议由 RequestContextMiddleware 在最外层统一计算一次，
IPFilterMiddleware 只读取结果并执行黑 / 白名单。
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from middleware.common import policy_error, public_health, require_request_context


def _networks(value: str) -> tuple:
    """把逗号分隔的 IP / CIDR 配置解析为网段元组；任何一项不合法都拒绝启动。

    单个 IP 会被解析为 /32（IPv4）或 /128（IPv6）的网段。strict=False 允许 "10.0.0.1/8"
    这类主机位非零的写法（自动归一为 10.0.0.0/8）。
    """
    try:
        # 配置错误必须阻止启动，不能忽略拼写错误后意外放宽访问范围。
        networks = []
        for item in value.split(","):
            if not item.strip():
                continue
            # "%" 是 IPv6 接口作用域标识（如 fe80::1%eth0），不接受。
            if "%" in item:
                raise ValueError
            network = ipaddress.ip_network(item.strip(), strict=False)
            if isinstance(network, ipaddress.IPv6Network) and network.prefixlen >= 96 and network.network_address.ipv4_mapped:
                # 配置与请求地址使用相同的 IPv4 规范形式，防止映射地址在黑白名单里失配。
                # 例如 ::ffff:10.0.0.0/104 转换为 10.0.0.0/8。
                network = ipaddress.IPv4Network((network.network_address.ipv4_mapped, network.prefixlen - 96))
            networks.append(network)
        return tuple(networks)
    except ValueError:
        raise RuntimeError("IP 配置必须是合法的 IPv4/IPv6 地址或 CIDR 网段，不能使用通配符。") from None


def _address(value: str):
    """把字符串解析为 IP 地址对象；IPv4 映射的 IPv6 地址（::ffff:a.b.c.d）还原为 IPv4。

    不合法时抛出 ValueError，由调用方决定是拒绝请求还是回退。
    """
    if "%" in value:
        raise ValueError("不接受带接口作用域的客户端地址")
    address = ipaddress.ip_address(value)
    return address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address


def _contains(address, networks: tuple) -> bool:
    """判断地址是否落在任一网段内。先比较版本号：IPv4 地址与 IPv6 网段之间不做比较。"""
    return any(address.version == network.version and address in network for network in networks)


def _peer_address(scope: Scope):
    """取 ASGI scope 中的 TCP 对端地址（scope["client"][0]）；缺失或不合法时返回 None。"""
    try:
        # scope["client"] 可能不存在（返回默认值 ("", 0)，空串解析失败），也可能是 None（取下标抛 TypeError）。
        return _address(scope.get("client", ("", 0))[0])
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class IPFilterSettings:
    """IP 过滤与可信代理配置：应用启动时构造一次，之后只读。

    RequestContextMiddleware 用它解析客户端 IP 与协议，IPFilterMiddleware 用它执行黑 / 白名单。
    """

    # disabled / blacklist / whitelist
    mode: str
    blacklist: tuple
    whitelist: tuple
    trusted_proxies: tuple

    @classmethod
    def from_environment(cls) -> IPFilterSettings:
        """从环境变量加载配置；模式不合法、网段不合法或白名单模式缺少白名单时拒绝启动。"""
        mode = os.getenv("IP_FILTER_MODE", "disabled").strip().lower()
        if mode not in {"disabled", "blacklist", "whitelist"}:
            raise RuntimeError("IP_FILTER_MODE 必须是 disabled、blacklist 或 whitelist。")
        settings = cls(mode, _networks(os.getenv("IP_BLACKLIST", "")), _networks(os.getenv("IP_WHITELIST", "")), _networks(os.getenv("TRUSTED_PROXY_IPS", "")))
        if mode == "whitelist" and not settings.whitelist:
            raise RuntimeError("白名单模式要求非空 IP_WHITELIST，避免配置遗漏导致服务无法使用。")
        return settings

    def client_ip(self, scope: Scope) -> str:
        """确定请求的真实客户端 IP（字符串形式）。

        - TCP 对端未知：返回 "unknown"；
        - 没有 X-Forwarded-For，或对端不是可信代理：直接返回对端地址（忽略代理头）；
        - 对端是可信代理：从 X-Forwarded-For 链的右端开始，依次剥离可信代理，
          第一个不可信的地址就是客户端。

        可信代理发来的 X-Forwarded-For 出现多个头、过长、跳数异常或地址不合法时抛出 ValueError。
        """
        peer = _peer_address(scope)
        if peer is None:
            # 测试或不提供地址的 ASGI 服务器共享 unknown 桶，不能靠随机头创建无限个计数桶。
            return "unknown"
        values = Headers(scope=scope).getlist("x-forwarded-for")
        if not values or not _contains(peer, self.trusted_proxies):
            return str(peer)
        # 只接受恰好一个 X-Forwarded-For 头：多个同名头的合并顺序有歧义。长度上限防止超长头消耗 CPU。
        if len(values) != 1 or len(values[0]) > 2048:
            raise ValueError("代理地址头不合法")
        hops = values[0].split(",")
        if not 1 <= len(hops) <= 20:
            raise ValueError("代理跳数不合法")
        # 链条从左到右为：客户端, 代理1, 代理2, ...，最后追加真实 TCP 对端。
        chain = [_address(hop.strip()) for hop in hops] + [peer]
        # 从 TCP 对端向左剥离可信代理，首个不可信节点才是客户端，不能盲信最左边的值。
        # （最左边的值由客户端自己填写，可以任意伪造；只有可信代理追加的部分可信。）
        while len(chain) > 1 and _contains(chain[-1], self.trusted_proxies):
            chain.pop()
        return str(chain[-1])

    def forwarded_scheme(self, scope: Scope) -> str | None:
        """从可信代理的 X-Forwarded-Proto 头取得客户端实际使用的协议（http / https）。

        对端不可信或没有该头时返回 None（保持 scope 原有协议）；头重复或取值不合法时抛出 ValueError。
        """
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

    def proxy_misconfiguration(self, scope: Scope) -> str | None:
        """检测会让所有客户端共用同一个 IP 身份的部署配置问题；未发现问题时返回 None。

        - client_address_missing：ASGI 服务器没有提供 TCP 对端地址（例如经 Unix socket 接入），
          所有请求都被识别为 unknown；
        - loopback_proxy_untrusted：请求来自本机回环地址且带 X-Forwarded-For，但该地址不在
          TRUSTED_PROXY_IPS 中——典型情况是 Nginx 同机转发却忘了配置可信代理，
          所有客户端都会被识别为 127.0.0.1。
        两种情况下 IP 限流、登录限流和临时封禁都共用一个计数，一个攻击者就能让所有人被封。
        """
        peer = _peer_address(scope)
        if peer is None:
            return "client_address_missing"
        if peer.is_loopback and not _contains(peer, self.trusted_proxies) and Headers(scope=scope).get("x-forwarded-for"):
            return "loopback_proxy_untrusted"
        return None

    def rejection_code(self, client_ip: str) -> str | None:
        """根据过滤模式判断是否拒绝该客户端 IP；返回错误码，放行时返回 None。

        - disabled：一律放行；
        - IP 未知：拒绝（ip_not_allowed），无法确认来源时不能放行；
        - 命中黑名单：拒绝（ip_blocked）。白名单模式下黑名单同样生效，可在大网段中排除个别地址；
        - 白名单模式且不在白名单中：拒绝（ip_not_allowed）。
        """
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
    """IP 过滤中间件（纯 ASGI）：按黑 / 白名单拒绝请求。

    读取外层 RequestContextMiddleware 写入的 security_client_ip，以及代理头不合法时的
    security_proxy_headers_invalid 标记。被拒绝的请求返回 403；健康检查即使命中规则也放行。
    """

    def __init__(self, app: ASGIApp, *, settings: IPFilterSettings) -> None:
        self.app, self.settings = app, settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # 只处理 HTTP 请求；lifespan 等其他类型直接透传。
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        require_request_context(scope)
        if scope.get("security_proxy_headers_invalid"):
            # 可信代理发送歧义或无效头时拒绝，不能退回代理 IP 以绕过白名单。
            # 注意：即使 IP_FILTER_MODE=disabled，代理头不合法也会被拒绝。
            code = "ip_not_allowed"
        else:
            code = self.settings.rejection_code(scope["security_client_ip"])
        if code and not public_health(scope):
            await policy_error(scope, code, 403)(scope, receive, send)
            return
        await self.app(scope, receive, send)
