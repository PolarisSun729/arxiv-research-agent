"""JWT 仅携带身份和会话版本，权限始终读取当前数据库记录。

令牌中的声明（claims）：
- sub：用户 user_id；
- ver：用户的 token_version，改密码、改角色或停用账号时递增，使该用户已签发的令牌全部失效；
- jti：本次登录会话的唯一 ID，对应 auth_sessions 表的一行，登出时删除即可单独撤销；
- iat / nbf / exp：签发时间、生效时间、过期时间（Unix 秒）；
- iss / aud：签发方与受众，防止其他系统用同一密钥签发的令牌被本服务接受。

令牌中不包含角色，每次请求都从数据库读取最新的角色和状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import jwt

from auth.errors import AuthError
from auth.models import AuthUser
from auth.passwords import verify_password
from auth.settings import JwtSettings
from auth.store import AuthStore


@dataclass(frozen=True)
class AuthSession:
    """一次已通过校验的登录会话：数据库中的用户记录，加上令牌里的 jti 和过期时间。"""

    user: AuthUser
    jti: str
    expires_at: int

    def store_arguments(self) -> dict:
        """返回 AuthStore.check_session / consume 所需的会话定位参数。"""
        return {"user_id": self.user.user_id, "version": self.user.token_version, "jti": self.jti}


class JwtAuthenticator:
    """负责登录签发令牌，以及校验请求携带的令牌。"""

    def __init__(self, settings: JwtSettings, store: AuthStore) -> None:
        self.settings, self.store = settings, store

    def login(self, username: str, password: str) -> dict:
        """校验用户名和密码，成功后创建会话并返回 OAuth2 风格的令牌响应。

        用户不存在、密码错误、账号停用都统一返回 invalid_credentials，不透露具体原因。
        """
        user = self.store.get_by_username(username)
        # 即使用户不存在也执行一次 bcrypt 比较（见 passwords.verify_password），使响应耗时一致。
        matched = verify_password(password, user.hashed_password if user else None)
        if not matched or not user or not user.is_active:
            raise AuthError("invalid_credentials")
        now, jti = int(self.store.clock()), uuid4().hex
        expires = now + self.settings.expire_seconds
        # open_session 在写事务中再次确认密码哈希和令牌版本未变，并登记会话。
        user = self.store.open_session(user, jti=jti, expires_at=expires)
        payload = {"sub": user.user_id, "ver": user.token_version, "jti": jti, "iat": now, "nbf": now,
                   "exp": expires, "iss": self.settings.issuer, "aud": self.settings.audience}
        return {"access_token": jwt.encode(payload, self.settings.secret, algorithm="HS256"),
                "token_type": "bearer", "expires_in": self.settings.expire_seconds}

    def authenticate(self, token: str) -> AuthSession:
        """校验 Bearer 令牌，返回对应的会话；任何不合法的情况都抛出 invalid_token。

        校验分两步：
        1. 无状态校验：签名、算法、iss/aud、时间窗口，以及各声明的类型；
        2. 有状态校验：在数据库中确认用户仍有效、token_version 一致、会话未被撤销且未过期。
        """
        try:
            # 限制令牌长度，避免攻击者用超长输入消耗解码资源。
            if not token or len(token) > 4096:
                raise ValueError
            payload = jwt.decode(token, self.settings.secret, algorithms=["HS256"], issuer=self.settings.issuer,
                                 audience=self.settings.audience,
                                 options={"require": ["sub", "ver", "jti", "iat", "nbf", "exp", "iss", "aud"], "strict_aud": True})
            # 声明类型逐项检查：使用 type(...) is int 而不是 isinstance，排除 True/False 这类 bool 值；
            # 令牌有效期不能超过当前配置的 expire_seconds，调低过期时长后，旧的长期令牌也会立即失效。
            if (not isinstance(payload["sub"], str) or not payload["sub"] or not isinstance(payload["jti"], str)
                    or not payload["jti"] or type(payload["ver"]) is not int
                    or any(type(payload[key]) is not int for key in ("iat", "nbf", "exp"))
                    or payload["exp"] <= payload["iat"] or payload["exp"] - payload["iat"] > self.settings.expire_seconds):
                raise ValueError
        except (jwt.InvalidTokenError, ValueError, TypeError, KeyError):
            # 固定算法与 issuer/audience 防止其他系统令牌混用；异常原文不回显攻击者输入。
            raise AuthError("invalid_token") from None
        user = self.store.check_session(user_id=payload["sub"], version=payload["ver"], jti=payload["jti"])
        return AuthSession(user, payload["jti"], payload["exp"])
