"""每个应用独立加载认证配置；JWT 与共享 API Key 模式不能隐式混用。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from auth.key_config import REPO_ROOT
from middleware.common import env_bool, positive_env_int
from utils.secret_redaction import register_sensitive_values


def auth_mode() -> str:
    mode = os.getenv("AUTH_MODE", "jwt").strip().lower()
    if mode not in {"jwt", "api_key"}:
        raise RuntimeError("AUTH_MODE 必须显式选择 jwt 或 api_key。")
    return mode


def auth_database_path() -> Path:
    configured = os.getenv("AUTH_DATABASE_PATH", "backend/data/auth/auth.sqlite3").strip()
    if not configured or configured == ":memory:":
        raise RuntimeError("AUTH_DATABASE_PATH 必须指向持久化的独立认证数据库。")
    path = Path(configured)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class JwtSettings:
    secret: str = field(repr=False)
    database_path: Path
    issuer: str
    audience: str
    expire_seconds: int
    registration_enabled: bool
    max_body_bytes: int

    @classmethod
    def from_environment(cls) -> JwtSettings:
        secret = os.getenv("JWT_SECRET_KEY", "").strip()
        secret_file = os.getenv("JWT_SECRET_FILE", "").strip()
        if secret_file:
            if secret:
                raise RuntimeError("JWT_SECRET_KEY 与 JWT_SECRET_FILE 只能设置一个。")
            path = Path(secret_file)
            if not path.is_absolute():
                path = REPO_ROOT / path
            try:
                if os.name != "nt" and path.stat().st_mode & 0o077:
                    raise OSError
                secret = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                raise RuntimeError("无法读取 JWT 私有密钥文件，请检查路径、编码和权限（0600）。") from None
        # 覆盖各模型供应商及兼容访问密钥，禁止将已交给第三方的凭据复用为 JWT 签名材料。
        external_secrets = {
            value.strip() for name, raw in os.environ.items()
            if name.upper().endswith(("_API_KEY", "_API_KEYS", "_ACCESS_KEY"))
            for value in raw.split(",") if value.strip()
        }
        # 长度校验不能证明随机性，但应拒绝明显占位值、重复字符和文档中的旧示例密钥。
        invalid = (
            not 43 <= len(secret) <= 512 or len(set(secret)) < 16
            or any(ord(char) < 33 or ord(char) > 126 for char in secret)
            or any(word in secret.lower() for word in ("change-me", "changeme", "example", "replace", "tz9xk2pl8mn4"))
            or secret in external_secrets
        )
        if invalid:
            raise RuntimeError("请为 JWT 配置独立强随机密钥：使用 secrets.token_urlsafe(48) 生成 JWT_SECRET_KEY 或 JWT_SECRET_FILE。")
        expires = positive_env_int("JWT_EXPIRE_MINUTES", 60)
        if expires > 1440:
            raise RuntimeError("JWT_EXPIRE_MINUTES 不能超过 1440 分钟。")
        issuer = os.getenv("JWT_ISSUER", "arxiv-research-agent").strip()
        audience = os.getenv("JWT_AUDIENCE", "arxiv-research-api").strip()
        if not issuer or not audience or len(issuer) > 200 or len(audience) > 200:
            raise RuntimeError("JWT_ISSUER 与 JWT_AUDIENCE 必须是 1 至 200 字符的非空标识。")
        register_sensitive_values((secret,))
        return cls(secret, auth_database_path(), issuer, audience, expires * 60,
                   env_bool("ALLOW_PUBLIC_REGISTRATION", False), positive_env_int("AUTH_MAX_REQUEST_BYTES", 1048576))
