"""JWT 仅携带身份和会话版本，权限始终读取当前数据库记录。"""

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
    user: AuthUser
    jti: str
    expires_at: int

    def store_arguments(self) -> dict:
        return {"user_id": self.user.user_id, "version": self.user.token_version, "jti": self.jti}


class JwtAuthenticator:
    def __init__(self, settings: JwtSettings, store: AuthStore) -> None:
        self.settings, self.store = settings, store

    def login(self, username: str, password: str) -> dict:
        user = self.store.get_by_username(username)
        matched = verify_password(password, user.hashed_password if user else None)
        if not matched or not user or not user.is_active:
            raise AuthError("invalid_credentials")
        now, jti = int(self.store.clock()), uuid4().hex
        expires = now + self.settings.expire_seconds
        user = self.store.open_session(user, jti=jti, expires_at=expires)
        payload = {"sub": user.user_id, "ver": user.token_version, "jti": jti, "iat": now, "nbf": now,
                   "exp": expires, "iss": self.settings.issuer, "aud": self.settings.audience}
        return {"access_token": jwt.encode(payload, self.settings.secret, algorithm="HS256"),
                "token_type": "bearer", "expires_in": self.settings.expire_seconds}

    def authenticate(self, token: str) -> AuthSession:
        try:
            if not token or len(token) > 4096:
                raise ValueError
            payload = jwt.decode(token, self.settings.secret, algorithms=["HS256"], issuer=self.settings.issuer,
                                 audience=self.settings.audience,
                                 options={"require": ["sub", "ver", "jti", "iat", "nbf", "exp", "iss", "aud"], "strict_aud": True})
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
