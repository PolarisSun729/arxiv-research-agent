"""在解析业务参数之前执行共享限流、每日配额及临时封禁，支持完整 SSE 生命周期。

同一个 RateLimitMiddleware 类在中间件栈里注册两次：
- before_auth=True（认证之前）：先检查来源 IP 是否被临时封禁，再按 IP 限流（RATE_LIMIT_IP），挡洪水和扫描；
- before_auth=False（认证之后），按请求类型分三种：
  - 登录 / 注册等公开认证接口：按 IP 限流（RATE_LIMIT_AUTH）；
  - JWT 用户：按 user_id 限流（RATE_LIMIT_USER），并按操作类别（read / write / expensive）再限一次；
  - API Key：按密钥自身的速率限流，同样按操作类别再限一次，另外扣减密钥的每日配额。

封禁机制：同一 IP 认证失败（AUTH_FAILURE_LIMIT）或被限流拒绝（RATE_LIMIT_VIOLATION_LIMIT）过于频繁时，
封禁该 IP ABUSE_BLOCK_SECONDS 秒，期间所有请求（健康检查除外）返回 403。

所有计数都存放在 RATE_LIMIT_STORAGE 指定的共享存储中（memory:// 或 Redis），
多 worker 部署必须使用 Redis，否则每个进程各自计数。存储不可用时请求返回 503，不会退回进程内计数。
规则字符串使用 limits 库的格式，例如 "60/minute;10/second"（分号分隔的多条规则需同时满足）。

一个请求同时受多份预算约束时（多条规则、身份 + 类别、速率 + 日配额），先确认全部预算都有余量再统一计数，
被任一预算拒绝的请求不会消耗其他预算。
"""

from __future__ import annotations

import logging
import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from limits import RateLimitItem, RateLimitItemPerDay, parse_many
from limits.storage import storage_from_string
from limits.strategies import FixedWindowRateLimiter
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from auth.charging import api_key_charger
from auth.errors import AuthError
from auth.key_config import ApiKeyPolicy, validate_rate
from auth.permissions import ROUTE_POLICIES, rate_category
from core.errors import AUTHENTICATION_FAILURE_CODES
from middleware.common import policy_error, positive_env_int, public_health, require_request_context


logger = logging.getLogger(__name__)
# 所有计数键的公共前缀。
_KEY_PREFIX = "arxiv-stage2"


@dataclass(frozen=True)
class RateLimitSettings:
    """限流配置：应用启动时构造一次，之后只读。各字段（除 storage_uri 与 block_seconds）均为限流规则字符串。"""

    # 计数存储地址；可能包含 Redis 密码，不出现在 repr 中。
    storage_uri: str = field(repr=False)
    # 认证前按来源 IP 的总请求上限。
    ip: str
    # 认证后按操作类别的上限：读取 / 普通写入 / 昂贵操作。
    read: str
    write: str
    expensive: str
    # 单个 IP 认证失败的频率上限，超过后封禁该 IP。
    auth_failures: str
    # 单个 IP 被限流拒绝的频率上限，超过后封禁该 IP。
    violations: str
    # 封禁时长（秒）。
    block_seconds: int
    # JWT 用户（按 user_id）的总请求上限。
    user: str
    # 登录 / 注册等公开认证接口按 IP 的上限。
    auth: str
    # 登录接口按用户名的上限，防止换 IP 对同一账号猜密码。
    login_account: str

    @classmethod
    def from_environment(cls) -> RateLimitSettings:
        """从环境变量加载并校验限流配置；存储地址或任一规则不合法时拒绝启动。"""
        uri = os.getenv("RATE_LIMIT_STORAGE", "memory://").strip()
        try:
            parsed = urlsplit(uri)
            # 只接受 memory://，或带主机名、端口不为 0、无片段、无空白的 redis:// / rediss:// 地址。
            # 访问 parsed.port 时端口不合法会抛出 ValueError。
            valid = uri == "memory://" or (
                parsed.scheme in {"redis", "rediss"} and parsed.hostname
                and parsed.port != 0 and not parsed.fragment and not any(char.isspace() for char in uri)
            )
        except ValueError:
            valid = False
        if not valid:
            # URL 可能包含 Redis 密码，格式错误只能返回固定说明，不能回显原值或解析异常。
            raise RuntimeError("RATE_LIMIT_STORAGE 只支持 memory://、redis:// 或 rediss://。")
        if uri == "memory://":
            # 进程内计数感知不到 worker 数量，只能提示：多 worker / 多实例时各进程各算各的，实际额度成倍放大。
            logger.warning("RATE_LIMIT_STORAGE=memory:// 只在当前进程内计数，多 worker 或多实例部署必须改用 Redis。")
        # 参数按字段声明顺序传入：ip, read, write, expensive, auth_failures, violations, block_seconds, user, auth, login_account。
        return cls(
            # 认证前的 IP 预算只防洪水与暴力破解；NAT 后的多个已登录用户由各自的身份预算区分，不能被它卡住。
            uri, validate_rate(os.getenv("RATE_LIMIT_IP", "300/minute;30/second")),
            validate_rate(os.getenv("RATE_LIMIT_READ", "60/minute")), validate_rate(os.getenv("RATE_LIMIT_WRITE", "20/minute")),
            validate_rate(os.getenv("RATE_LIMIT_EXPENSIVE", "5/minute")),
            validate_rate(os.getenv("AUTH_FAILURE_LIMIT", "20/minute")),
            validate_rate(os.getenv("RATE_LIMIT_VIOLATION_LIMIT", "30/minute")),
            positive_env_int("ABUSE_BLOCK_SECONDS", 900),
            validate_rate(os.getenv("RATE_LIMIT_USER", "60/minute")),
            validate_rate(os.getenv("RATE_LIMIT_AUTH", "10/minute")),
            validate_rate(os.getenv("RATE_LIMIT_LOGIN_ACCOUNT", "10/minute")),
        )


@dataclass(frozen=True)
class LimitDecision:
    """一次限流判定的结果。

    code 为 None 表示放行；否则是拒绝原因（rate_limit_exceeded、daily_quota_exceeded、
    ip_temporarily_blocked），retry_after 为建议的重试等待秒数。
    headers 是要附加到响应上的限流 / 配额信息头（放行和拒绝时都有）。
    """

    code: str | None = None
    retry_after: int | None = None
    headers: dict[str, str] = field(default_factory=dict)


class RateLimitController:
    """限流判定逻辑：封装共享计数存储，提供各类限流、配额与封禁操作。

    所有方法都是同步阻塞调用（可能访问 Redis），中间件通过 run_in_threadpool 调用。
    存储访问失败时直接抛出异常，由中间件转换为 503。
    """

    def __init__(self, settings: RateLimitSettings, *, storage_options: dict | None = None, clock=None) -> None:
        self.settings = settings
        # 时间源可注入，便于测试控制时间。
        self.clock = clock or time.time
        # Redis 连接与读写超时 1 秒：存储故障时尽快返回 503，而不是让请求长时间挂起。
        options = {} if settings.storage_uri == "memory://" else {"socket_connect_timeout": 1, "socket_timeout": 1}
        options.update(storage_options or {})
        # 直接使用 limits 的计数存储与固定窗口限流器；规则由统一 ASGI 入口应用，避免装饰器遗漏和 SSE 缓冲。
        # 存储故障时异常直接抛出，由中间件转换为 503，不会被吞掉或悄悄改用进程内计数。
        try:
            self.storage = storage_from_string(settings.storage_uri, **options)
        except Exception:
            raise RuntimeError("无法初始化限流存储，请检查 RATE_LIMIT_STORAGE 配置。") from None
        # 提供 test（只检查不计数）、hit（原子加一并检查）与 get_window_stats（查询窗口状态）。
        self.backend = FixedWindowRateLimiter(self.storage)

    def _window(self, item: RateLimitItem, label: str, identifier: str) -> tuple[dict[str, str], int]:
        """查询一条规则的当前窗口，返回 (X-RateLimit-* 信息头, 距窗口重置的秒数)。"""
        # 返回 (窗口重置时间戳, 剩余次数)。
        reset, remaining = self.backend.get_window_stats(item, _KEY_PREFIX, label, identifier)
        headers = {
            "X-RateLimit-Limit": str(item.amount), "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(math.ceil(reset)), "X-RateLimit-Scope": label,
        }
        return headers, max(1, math.ceil(reset - self.clock()))

    def _consume(self, *budgets: tuple[str, str, str]) -> LimitDecision:
        """对一组预算统一计一次数；每份预算为 (规则, label, identifier)。

        每份规则可以包含多条（如 "300/minute;30/second"）。分两步：
        1. 逐条检查是否还有余量，任一条已耗尽就拒绝并返回该条的信息头，此时任何预算都不计数；
        2. 全部有余量时逐条计数，返回剩余次数最少的那条规则的信息头。
        计数键形如 arxiv-stage2/<label>/<identifier>，label 区分不同用途的预算。
        """
        items = [(item, label, identifier) for rule, label, identifier in budgets for item in parse_many(rule)]
        for item, label, identifier in items:
            if not self.backend.test(item, _KEY_PREFIX, label, identifier):
                headers, retry_after = self._window(item, label, identifier)
                return LimitDecision("rate_limit_exceeded", retry_after, headers)
        chosen = {}
        for item, label, identifier in items:
            # hit 原子地加一并返回是否仍在限额内。
            allowed = self.backend.hit(item, _KEY_PREFIX, label, identifier)
            headers, retry_after = self._window(item, label, identifier)
            if not allowed:
                # 检查与计数之间被并发请求抢先用完（窗口极短，很少见）：已计数的预算不回退，仍按超限拒绝。
                return LimitDecision("rate_limit_exceeded", retry_after, headers)
            if not chosen or int(headers["X-RateLimit-Remaining"]) < int(chosen["X-RateLimit-Remaining"]):
                chosen = headers
        return LimitDecision(headers=chosen)

    def _block_key(self, client_ip: str) -> str:
        """IP 临时封禁标记在存储中的键。"""
        return f"arxiv-stage2:block:{client_ip}"

    def before_auth(self, client_ip: str) -> LimitDecision:
        """认证之前的检查：该 IP 处于封禁期则直接拒绝（返回剩余封禁秒数），否则按 IP 规则计数。"""
        block_key = self._block_key(client_ip)
        if self.storage.get(block_key):
            seconds = max(1, math.ceil(self.storage.get_expiry(block_key) - self.clock()))
            return LimitDecision("ip_temporarily_blocked", seconds)
        return self._consume((self.settings.ip, "ip", client_ip))

    def after_auth(self, policy: ApiKeyPolicy, scope: Scope) -> LimitDecision:
        """API Key 模式的认证后检查：先按密钥速率和操作类别限流，再处理该密钥的每日配额。"""
        # 日配额只计入权限清单中标记了配额的付费操作；登录校验、列表轮询等免费请求只回显剩余额度。
        route_policy = ROUTE_POLICIES.get((scope.get("method", ""), scope.get("security_route", "")))
        paid = policy.daily_quota is not None and bool(route_policy and route_policy.quota)
        if paid:
            # 先确认日配额还有余量：配额已用完的付费请求直接拒绝，不再消耗速率预算。
            precheck = self._daily_quota(policy, consume=False, require=True)
            if precheck.code:
                return precheck
        decision = self._after_identity(policy.identifier, policy.rate_limit, scope)
        if decision.code or policy.daily_quota is None:
            return decision
        daily = self._daily_quota(policy, consume=paid)
        # 合并限流头与配额头一起返回。
        return LimitDecision(daily.code, daily.retry_after, {**decision.headers, **daily.headers})

    def charge_key(self, policy: ApiKeyPolicy) -> LimitDecision:
        """业务层在真正回源或调用模型时扣减密钥日配额，例如论文详情只在本地缺失时计费。"""
        if policy.daily_quota is None:
            return LimitDecision()
        return self._daily_quota(policy, consume=True)

    def after_user_auth(self, user_id: str, scope: Scope) -> LimitDecision:
        """JWT 模式的认证后检查：按 user_id 的总预算和操作类别预算限流（每日配额由 UserQuotaMiddleware 负责）。"""
        # JWT 可重复登录和轮换；计数永远绑定稳定 user_id，不按 token 或来源 IP 拆分。
        return self._after_identity(user_id, self.settings.user, scope, identity_label="user")

    def public_auth(self, client_ip: str) -> LimitDecision:
        """登录 / 注册等公开认证接口：按来源 IP 共用 RATE_LIMIT_AUTH 预算。"""
        return self._consume((self.settings.auth, "auth", client_ip))

    def login_attempt(self, username: str) -> LimitDecision:
        """登录接口按用户名限流（由 auth_router 在登录时调用），防止攻击者换 IP 对同一账号猜密码。

        用户名不区分大小写，并以 SHA-256 摘要作为计数键，存储中不出现明文用户名。
        """
        identifier = hashlib.sha256(username.casefold().encode("utf-8")).hexdigest()
        return self._consume((self.settings.login_account, "login-account", identifier))

    def _after_identity(self, identifier: str, rate_limit: str, scope: Scope, *, identity_label: str = "key") -> LimitDecision:
        """已认证主体的两级限流：主体总预算与该操作类别的预算同时有余量才放行，并一起计数。

        类别由权限清单决定（auth.permissions.rate_category）：标记了配额的付费操作为 expensive，
        其余按方法归入 read / write，个别接口在清单中显式指定。
        任一级拒绝时两级都不计数；都通过时返回剩余次数较少的那一级的信息头。
        """
        category = rate_category(scope.get("method", ""), scope.get("security_route", ""))
        # 按已认证主体和操作类别共享预算，切换论文 ID、URL 或 IP 都不会得到新的额度。
        return self._consume((rate_limit, identity_label, identifier), (getattr(self.settings, category), category, identifier))

    def _daily_quota(self, policy: ApiKeyPolicy, *, consume: bool, require: bool = False) -> LimitDecision:
        """API Key 的每日配额。

        - consume=True：扣减一次，额度不足时拒绝；
        - consume=False, require=True：只检查能否再扣一次，不计数，额度不足时拒绝；
        - 两者都为 False：只查询剩余额度，从不拒绝。
        按 UTC 自然日计算，下一个 UTC 零点重置。返回 X-DailyQuota-* 信息头；额度用完时返回 daily_quota_exceeded。
        """
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        # 下一个 UTC 零点的时间戳。
        reset = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp()
        item = RateLimitItemPerDay(policy.daily_quota)
        identity = (_KEY_PREFIX, "daily", policy.identifier, now.date().isoformat())
        # 日期加入计数键，按 UTC 自然日切换；存储原子递增确保并发和多 worker 无法超发。
        if consume:
            allowed = self.backend.hit(item, *identity)
        else:
            allowed = not require or self.backend.test(item, *identity)
        _, remaining = self.backend.get_window_stats(item, *identity)
        headers = {"X-DailyQuota-Limit": str(policy.daily_quota), "X-DailyQuota-Remaining": str(remaining), "X-DailyQuota-Reset": str(int(reset))}
        if not allowed:
            return LimitDecision("daily_quota_exceeded", max(1, math.ceil(reset - self.clock())), headers)
        return LimitDecision(headers=headers)

    def observe_rejection(self, client_ip: str, code: str | None) -> None:
        """记录一次可能属于滥用的拒绝，频率超限时封禁该 IP。

        - 认证失败类错误码（AUTHENTICATION_FAILURE_CODES）计入 auth-failures 预算；
        - rate_limit_exceeded 计入 rate-violations 预算；
        - 其他错误码忽略。
        """
        if code in AUTHENTICATION_FAILURE_CODES:
            rule, label = self.settings.auth_failures, "auth-failures"
        elif code == "rate_limit_exceeded":
            rule, label = self.settings.violations, "rate-violations"
        else:
            return
        if self._consume((rule, label, client_ip)).code:
            # 临时封禁使用同一个共享存储；持续攻击不会延长已开始的封禁窗口。
            # （incr 只在键首次创建时设置过期时间，之后的递增不会刷新过期时间。）
            self.storage.incr(self._block_key(client_ip), self.settings.block_seconds)


class RateLimitMiddleware:
    """限流中间件（纯 ASGI，不缓冲响应，SSE 流式响应可以正常工作）。

    before_auth=True 时注册在认证中间件之外，执行 IP 封禁检查与 IP 限流，
    并在响应时把认证失败等事件计入滥用统计；before_auth=False 时注册在认证中间件之内，
    根据认证结果按身份限流。被拒绝的请求返回 429（封禁时返回 403），存储故障返回 503。
    健康检查和非 HTTP 请求直接放行。
    """

    def __init__(self, app: ASGIApp, *, controller: RateLimitController, before_auth: bool = False) -> None:
        self.app, self.controller, self.before_auth = app, controller, before_auth

    def _key_charger(self, policy: ApiKeyPolicy):
        """生成绑定到当前 API Key 的扣费函数，供 auth.charging.charge 在业务层调用；额度不足时抛出 AuthError。"""
        def charge() -> None:
            decision = self.controller.charge_key(policy)
            if decision.code:
                raise AuthError(decision.code, retry_after=decision.retry_after)

        return charge

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or public_health(scope):
            await self.app(scope, receive, send)
            return
        # 客户端 IP 与路由模板由最外层的 RequestContextMiddleware 写入。缺失时直接失败（不转成 503）：
        # 不能把所有请求算进同一个 unknown 计数，也不能因路由缺失而错判类别、漏扣配额。
        require_request_context(scope)
        client_ip = scope["security_client_ip"]
        key_policy = None
        try:
            # ---- 1. 根据所处位置和认证结果选择限流方式 ----
            if self.before_auth:
                decision = await run_in_threadpool(self.controller.before_auth, client_ip)
            elif scope.get("security_auth_public"):
                # 认证中间件标记的公开认证接口（登录 / 注册等）。
                decision = await run_in_threadpool(self.controller.public_auth, client_ip)
            elif scope.get("auth_session"):
                # JWT 认证成功，scope 中有会话。
                decision = await run_in_threadpool(self.controller.after_user_auth, scope["auth_session"].user.user_id, scope)
            else:
                # API Key 模式：必须已通过 ApiKeyMiddleware 认证，否则按缺少密钥拒绝（兜底，正常不会走到）。
                key_policy = scope.get("api_key_policy")
                if key_policy is None or not scope.get("api_key_authenticated"):
                    await policy_error(scope, "missing_api_key", 401)(scope, receive, send)
                    return
                decision = await run_in_threadpool(self.controller.after_auth, key_policy, scope)
            # ---- 2. 判定为拒绝时直接返回错误响应 ----
            if decision.code:
                if self.before_auth:
                    # 认证前的 IP 超限也计入该 IP 的违规次数，持续超限会被临时封禁。
                    await run_in_threadpool(self.controller.observe_rejection, client_ip, decision.code)
                status = 403 if decision.code == "ip_temporarily_blocked" else 429
                await policy_error(scope, decision.code, status, retry_after=decision.retry_after, headers=decision.headers)(scope, receive, send)
                return
        except Exception:
            # 共享存储故障必须拒绝昂贵业务，不能悄悄回退到每个进程独立的内存限流。
            logger.error("security_storage_unavailable")
            await policy_error(scope, "security_storage_unavailable", 503, retry_after=5)(scope, receive, send)
            return

        async def send_limited(message: Message) -> None:
            """包装 send：给响应加上限流信息头；认证前实例还要据内层的错误码记录滥用事件。"""
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                # 不覆盖已有同名头：内层（认证后）实例先写入更具体的身份限流头，外层（认证前）的 IP 头不再覆盖。
                for name, value in decision.headers.items():
                    if name not in headers:
                        headers[name] = value
                # 内层中间件或业务代码通过 policy_error / AppError 处理器写入的错误码。
                code = scope.get("security_error_code")
                # 凭据无效、匿名登录/注册入口超限才计入 IP 违规；已认证身份的超限由该身份承担，不能封掉同一出口的其他用户。
                anonymous_over_limit = code == "rate_limit_exceeded" and scope.get("security_auth_public")
                if self.before_auth and (code in AUTHENTICATION_FAILURE_CODES or anonymous_over_limit):
                    # 在认证拒绝的响应头发出前记录失败，不等待读取攻击者可能永不发送的请求体。
                    try:
                        await run_in_threadpool(self.controller.observe_rejection, client_ip, code)
                    except Exception:
                        # 滥用统计失败不影响已经生成的响应。
                        logger.error("security_abuse_tracking_unavailable")
            await send(message)

        # 密钥模式没有账号上下文；业务层通过 auth.charging 在真正付费的位置扣减该密钥的日配额。
        charger_token = api_key_charger.set(self._key_charger(key_policy)) if key_policy is not None else None
        try:
            await self.app(scope, receive, send_limited)
        finally:
            # 请求结束后恢复 ContextVar，避免扣费函数泄漏到同一协程上下文的后续请求。
            if charger_token is not None:
                api_key_charger.reset(charger_token)
