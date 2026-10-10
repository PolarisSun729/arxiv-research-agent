"""在解析业务参数之前执行共享限流、每日配额及临时封禁，支持完整 SSE 生命周期。"""

from __future__ import annotations

import logging
import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from limits import RateLimitItemPerDay, parse_many
from slowapi import Limiter
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from auth.charging import api_key_charger
from auth.errors import AuthError
from auth.key_config import ApiKeyPolicy, validate_rate
from auth.permissions import ROUTE_POLICIES
from core.errors import AUTHENTICATION_FAILURE_CODES
from middleware.common import policy_error, positive_env_int, public_health


logger = logging.getLogger(__name__)
_EXPENSIVE_OPERATIONS = {
    ("POST", "/api/agent/chat"),
    ("POST", "/api/agent/chat/stream"),
    ("POST", "/api/agent/work-continuations/{continuation_id}/resume/stream"),
    ("POST", "/api/paper"),
    ("POST", "/api/arxiv/download"),
    ("POST", "/api/paper/{arxiv_id}/create-qa-index"),
    ("POST", "/api/paper/{arxiv_id}/qa"),
    ("POST", "/api/paper/{arxiv_id}/qa/stream"),
    ("POST", "/api/user/research-profile/rebuild"),
    ("POST", "/api/user/generate-interest-vector"),
    ("POST", "/api/user/recommend-papers"),
    # 详情回源和偏好/行为记录可能物化论文并生成 embedding，不能因 GET 或收藏语义而降级预算。
    ("GET", "/api/paper/{arxiv_id}"),
    ("POST", "/api/user/like-paper"),
    ("POST", "/api/user/dislike-paper"),
    ("POST", "/api/user/paper-action"),
}


@dataclass(frozen=True)
class RateLimitSettings:
    storage_uri: str = field(repr=False)
    ip: str
    read: str
    write: str
    expensive: str
    auth_failures: str
    violations: str
    block_seconds: int
    user: str
    auth: str
    login_account: str

    @classmethod
    def from_environment(cls) -> RateLimitSettings:
        uri = os.getenv("RATE_LIMIT_STORAGE", "memory://").strip()
        try:
            parsed = urlsplit(uri)
            valid = uri == "memory://" or (
                parsed.scheme in {"redis", "rediss"} and parsed.hostname
                and parsed.port != 0 and not parsed.fragment and not any(char.isspace() for char in uri)
            )
        except ValueError:
            valid = False
        if not valid:
            # URL 可能包含 Redis 密码，格式错误只能返回固定说明，不能回显原值或解析异常。
            raise RuntimeError("RATE_LIMIT_STORAGE 只支持 memory://、redis:// 或 rediss://。")
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
    code: str | None = None
    retry_after: int | None = None
    headers: dict[str, str] = field(default_factory=dict)


class RateLimitController:
    def __init__(self, settings: RateLimitSettings, *, storage_options: dict | None = None, clock=None) -> None:
        self.settings = settings
        self.clock = clock or time.time
        options = {} if settings.storage_uri == "memory://" else {"socket_connect_timeout": 1, "socket_timeout": 1}
        options.update(storage_options or {})
        # 使用 SlowAPI 的公共计数后端；统一 ASGI 入口负责应用规则，避免装饰器遗漏和 SSE 缓冲。
        try:
            self.limiter = Limiter(
                key_func=lambda request: request.scope.get("security_client_ip", "unknown"),
                storage_uri=settings.storage_uri, storage_options=options, strategy="fixed-window",
                auto_check=False, swallow_errors=False, in_memory_fallback_enabled=False,
            )
        except Exception:
            raise RuntimeError("无法初始化限流存储，请检查 RATE_LIMIT_STORAGE 配置。") from None
        self.backend = self.limiter.limiter

    def _consume(self, rule: str, label: str, identifier: str) -> LimitDecision:
        chosen = {}
        for item in parse_many(rule):
            allowed = self.backend.hit(item, "arxiv-stage2", label, identifier)
            reset, remaining = self.backend.get_window_stats(item, "arxiv-stage2", label, identifier)
            headers = {
                "X-RateLimit-Limit": str(item.amount), "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(math.ceil(reset)), "X-RateLimit-Scope": label,
            }
            if not allowed:
                return LimitDecision("rate_limit_exceeded", max(1, math.ceil(reset - self.clock())), headers)
            if not chosen or remaining < int(chosen["X-RateLimit-Remaining"]):
                chosen = headers
        return LimitDecision(headers=chosen)

    def _block_key(self, client_ip: str) -> str:
        return f"arxiv-stage2:block:{client_ip}"

    def before_auth(self, client_ip: str) -> LimitDecision:
        block_key = self._block_key(client_ip)
        if self.backend.storage.get(block_key):
            seconds = max(1, math.ceil(self.backend.storage.get_expiry(block_key) - self.clock()))
            return LimitDecision("ip_temporarily_blocked", seconds)
        return self._consume(self.settings.ip, "ip", client_ip)

    def after_auth(self, policy: ApiKeyPolicy, scope: Scope) -> LimitDecision:
        decision = self._after_identity(policy.identifier, policy.rate_limit, scope)
        if decision.code or policy.daily_quota is None:
            return decision
        # 日配额只计入权限清单中标记了配额的付费操作；登录校验、列表轮询等免费请求只回显剩余额度。
        route_policy = ROUTE_POLICIES.get((scope.get("method", ""), scope.get("security_route", "")))
        daily = self._daily_quota(policy, consume=bool(route_policy and route_policy.quota))
        return LimitDecision(daily.code, daily.retry_after, {**decision.headers, **daily.headers})

    def charge_key(self, policy: ApiKeyPolicy) -> LimitDecision:
        """业务层在真正回源或调用模型时扣减密钥日配额，例如论文详情只在本地缺失时计费。"""
        if policy.daily_quota is None:
            return LimitDecision()
        return self._daily_quota(policy, consume=True)

    def after_user_auth(self, user_id: str, scope: Scope) -> LimitDecision:
        # JWT 可重复登录和轮换；计数永远绑定稳定 user_id，不按 token 或来源 IP 拆分。
        return self._after_identity(user_id, self.settings.user, scope, identity_label="user")

    def public_auth(self, client_ip: str) -> LimitDecision:
        return self._consume(self.settings.auth, "auth", client_ip)

    def login_attempt(self, username: str) -> LimitDecision:
        identifier = hashlib.sha256(username.casefold().encode("utf-8")).hexdigest()
        return self._consume(self.settings.login_account, "login-account", identifier)

    def _after_identity(self, identifier: str, rate_limit: str, scope: Scope, *, identity_label: str = "key") -> LimitDecision:
        headers = {}
        route = scope.get("security_route", "")
        method = scope.get("method", "")
        if (method, route) in _EXPENSIVE_OPERATIONS:
            category = "expensive"
        elif method in {"GET", "HEAD", "OPTIONS"} or (method == "POST" and route == "/api/arxiv/search"):
            category = "read"
        else:
            category = "write"
        # 按已认证主体和操作类别共享预算，切换论文 ID、URL 或 IP 都不会得到新的额度。
        for label, rule in ((identity_label, rate_limit), (category, getattr(self.settings, category))):
            decision = self._consume(rule, label, identifier)
            if decision.code:
                return decision
            if not headers or int(decision.headers["X-RateLimit-Remaining"]) < int(headers["X-RateLimit-Remaining"]):
                headers = decision.headers
        return LimitDecision(headers=headers)

    def _daily_quota(self, policy: ApiKeyPolicy, *, consume: bool) -> LimitDecision:
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        reset = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp()
        item = RateLimitItemPerDay(policy.daily_quota)
        identity = ("arxiv-stage2", "daily", policy.identifier, now.date().isoformat())
        # 日期加入计数键，按 UTC 自然日切换；存储原子递增确保并发和多 worker 无法超发。
        allowed = self.backend.hit(item, *identity) if consume else True
        _, remaining = self.backend.get_window_stats(item, *identity)
        headers = {"X-DailyQuota-Limit": str(policy.daily_quota), "X-DailyQuota-Remaining": str(remaining), "X-DailyQuota-Reset": str(int(reset))}
        if not allowed:
            return LimitDecision("daily_quota_exceeded", max(1, math.ceil(reset - self.clock())), headers)
        return LimitDecision(headers=headers)

    def observe_rejection(self, client_ip: str, code: str | None) -> None:
        if code in AUTHENTICATION_FAILURE_CODES:
            rule, label = self.settings.auth_failures, "auth-failures"
        elif code == "rate_limit_exceeded":
            rule, label = self.settings.violations, "rate-violations"
        else:
            return
        if self._consume(rule, label, client_ip).code:
            # 临时封禁使用同一个共享存储；持续攻击不会延长已开始的封禁窗口。
            self.backend.storage.incr(self._block_key(client_ip), self.settings.block_seconds)


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp, *, controller: RateLimitController, before_auth: bool = False) -> None:
        self.app, self.controller, self.before_auth = app, controller, before_auth

    def _key_charger(self, policy: ApiKeyPolicy):
        def charge() -> None:
            decision = self.controller.charge_key(policy)
            if decision.code:
                raise AuthError(decision.code, retry_after=decision.retry_after)

        return charge

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or public_health(scope):
            await self.app(scope, receive, send)
            return
        client_ip = scope.get("security_client_ip", "unknown")
        key_policy = None
        try:
            if self.before_auth:
                decision = await run_in_threadpool(self.controller.before_auth, client_ip)
            elif scope.get("security_auth_public"):
                decision = await run_in_threadpool(self.controller.public_auth, client_ip)
            elif scope.get("auth_session"):
                decision = await run_in_threadpool(self.controller.after_user_auth, scope["auth_session"].user.user_id, scope)
            else:
                key_policy = scope.get("api_key_policy")
                if key_policy is None or not scope.get("api_key_authenticated"):
                    await policy_error(scope, "missing_api_key", 401)(scope, receive, send)
                    return
                decision = await run_in_threadpool(self.controller.after_auth, key_policy, scope)
            if decision.code:
                if self.before_auth:
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
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in decision.headers.items():
                    if name not in headers:
                        headers[name] = value
                code = scope.get("security_error_code")
                # 凭据无效、匿名登录/注册入口超限才计入 IP 违规；已认证身份的超限由该身份承担，不能封掉同一出口的其他用户。
                anonymous_over_limit = code == "rate_limit_exceeded" and scope.get("security_auth_public")
                if self.before_auth and (code in AUTHENTICATION_FAILURE_CODES or anonymous_over_limit):
                    # 在认证拒绝的响应头发出前记录失败，不等待读取攻击者可能永不发送的请求体。
                    try:
                        await run_in_threadpool(self.controller.observe_rejection, client_ip, code)
                    except Exception:
                        logger.error("security_abuse_tracking_unavailable")
            await send(message)

        # 密钥模式没有账号上下文；业务层通过 auth.charging 在真正付费的位置扣减该密钥的日配额。
        charger_token = api_key_charger.set(self._key_charger(key_policy)) if key_policy is not None else None
        try:
            await self.app(scope, receive, send_limited)
        finally:
            if charger_token is not None:
                api_key_charger.reset(charger_token)
