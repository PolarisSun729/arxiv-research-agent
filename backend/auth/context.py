"""请求和后台续跑共享的已验证身份上下文，客户端与 LLM 都不能替换它。

JwtAuthMiddleware 认证成功后把 AuthContext 写入 ContextVar current_auth，
同一请求内的路由、服务层和 Agent 工具都通过它取得"当前是谁"。
请求参数或 LLM 生成的工具参数里出现的 user_id 只能与之一致，不能用来切换身份。

current_auth 为 None 有两种情况：API Key 兼容模式，或者没有经过 HTTP 入口的可信本地调用。
此时本模块的各个函数会退化为不做身份约束。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from auth.errors import AuthError
from auth.jwt_handler import AuthSession
from auth.store import AuthStore


@dataclass(frozen=True)
class AuthContext:
    """一次已认证请求的身份上下文：令牌解析出的会话，加上可以实时复核的认证存储。"""

    session: AuthSession
    store: AuthStore

    @property
    def user_id(self) -> str:
        """当前登录用户的业务 user_id（服务端生成，不可由客户端指定）。"""
        return self.session.user.user_id

    def require_roles(self, roles) -> None:
        """要求当前用户在数据库中的最新角色属于 roles，否则抛出 insufficient_permissions。

        会话已被撤销、令牌版本已变化或账号已停用时，check_session 会抛出 invalid_token。
        """
        # 长时间 Agent 执行必须重新检查撤销和角色，不能沿用最初请求中的权限快照。
        user = self.store.check_session(**self.session.store_arguments())
        if user.role not in roles:
            raise AuthError("insufficient_permissions")

    def consume(self, quota_type: str, amount: int = 1) -> None:
        """扣减当前用户 quota_type 类型的每日配额；超额抛出 quota_exceeded。

        扣减与会话有效性检查在同一个写事务中完成，已撤销的会话无法继续消耗配额。
        """
        self.store.consume(**self.session.store_arguments(), amounts={quota_type: amount})


# 当前请求的身份上下文；由 JwtAuthMiddleware 设置并在请求结束时重置，
# Agent 图节点通过 tool_access.bind_graph_node 在执行时重新设置。
current_auth: ContextVar[AuthContext | None] = ContextVar("current_auth", default=None)


def authenticated_user_id() -> str | None:
    """返回当前已认证用户的 user_id；没有 JWT 上下文时返回 None。"""
    context = current_auth.get()
    return context.user_id if context else None


def bind_user_id(value: str | None, *, fallback: str = "local_user") -> str:
    """把调用方传入的 user_id 规范化为"实际应使用的 user_id"。

    有 JWT 上下文时：
    - value 为 None、空字符串或只有空白：视为未指定，返回当前用户；
    - value 等于当前用户（允许首尾空白）：返回当前用户；
    - 其他任何值（包括非字符串）：抛出 identity_mismatch，不允许借参数访问他人数据。

    没有 JWT 上下文时：沿用旧行为，返回去掉首尾空白的 value，为空则返回 fallback。
    """
    context = current_auth.get()
    if context:
        if value is not None and (not isinstance(value, str) or value.strip() not in {"", context.user_id}):
            raise AuthError("identity_mismatch")
        return context.user_id
    # 无 JWT 上下文只存在于显式 API Key 兼容模式或可信本地调用中。
    return str(value or fallback).strip() or fallback


def bind_identity_fields(value) -> None:
    """规范化参数树中的身份字段，LLM 参数与 JSON 请求复用同一个约束。

    原地遍历 value 中所有嵌套的 dict / list，把每个名为 user_id 的字段替换为
    bind_user_id 的结果；与当前用户不一致时抛出 identity_mismatch。
    使用显式栈而不是递归，深层嵌套的输入不会触发 Python 递归深度限制。
    没有 JWT 上下文时不做任何修改。
    """
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
