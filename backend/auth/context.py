"""请求和后台续跑共享的已验证身份上下文，客户端与 LLM 都不能替换它。"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from auth.errors import AuthError
from auth.jwt_handler import AuthSession
from auth.store import AuthStore


@dataclass(frozen=True)
class AuthContext:
    session: AuthSession
    store: AuthStore

    @property
    def user_id(self) -> str:
        return self.session.user.user_id

    def require_roles(self, roles) -> None:
        # 长时间 Agent 执行必须重新检查撤销和角色，不能沿用最初请求中的权限快照。
        user = self.store.check_session(**self.session.store_arguments())
        if user.role not in roles:
            raise AuthError("insufficient_permissions")

    def consume(self, quota_type: str, amount: int = 1) -> None:
        self.store.consume(**self.session.store_arguments(), amounts={quota_type: amount})


current_auth: ContextVar[AuthContext | None] = ContextVar("current_auth", default=None)


def authenticated_user_id() -> str | None:
    context = current_auth.get()
    return context.user_id if context else None


def bind_user_id(value: str | None, *, fallback: str = "local_user") -> str:
    context = current_auth.get()
    if context:
        if value is not None and (not isinstance(value, str) or value.strip() not in {"", context.user_id}):
            raise AuthError("identity_mismatch")
        return context.user_id
    # 无 JWT 上下文只存在于显式 API Key 兼容模式或可信本地调用中。
    return str(value or fallback).strip() or fallback


def bind_identity_fields(value) -> None:
    """规范化参数树中的身份字段，LLM 参数与 JSON 请求复用同一个约束。"""
    if current_auth.get() is None:
        return
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            for name, child in item.items():
                if name == "user_id":
                    item[name] = bind_user_id(child)
                elif isinstance(child, (dict, list)):
                    pending.append(child)
        elif isinstance(item, list):
            pending.extend(item)
