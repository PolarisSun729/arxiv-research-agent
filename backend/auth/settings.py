"""每个应用独立加载认证配置；JWT 与共享 API Key 模式不能隐式混用。

相关环境变量：
- AUTH_MODE：jwt（默认）或 api_key；
- AUTH_DATABASE_PATH：认证 SQLite 数据库路径，相对路径以仓库根目录为基准；
- JWT_SECRET_KEY / JWT_SECRET_FILE：JWT 签名密钥（二选一）；
- JWT_EXPIRE_MINUTES：令牌有效期，默认 60，最长 1440；
- JWT_ISSUER / JWT_AUDIENCE：令牌的签发方与受众标识；
- ALLOW_PUBLIC_REGISTRATION：是否开放公开注册，默认 false；
- AUTH_MAX_REQUEST_BYTES：写请求请求体上限，默认 1 MiB。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from auth.key_config import REPO_ROOT, external_secret_values
from middleware.common import env_bool, positive_env_int
from utils.secret_redaction import register_sensitive_values


def auth_mode() -> str:
    """返回当前认证模式（"jwt" 或 "api_key"）；取值不合法时拒绝启动。"""
    mode = os.getenv("AUTH_MODE", "jwt").strip().lower()
    if mode not in {"jwt", "api_key"}:
        raise RuntimeError("AUTH_MODE 必须显式选择 jwt 或 api_key。")
    return mode


def auth_database_path() -> Path:
    """返回认证数据库的绝对路径。

    不允许使用 :memory:：AuthStore 每次操作都会新开连接，内存库在连接之间不共享，
    账号和会话会随时丢失。
    """
    configured = os.getenv("AUTH_DATABASE_PATH", "backend/data/auth/auth.sqlite3").strip()
    if not configured or configured == ":memory:":
        raise RuntimeError("AUTH_DATABASE_PATH 必须指向持久化的独立认证数据库。")
    path = Path(configured)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class JwtSettings:
    """JWT 模式的运行配置：应用启动时构造一次，之后只读。"""

    # HS256 签名密钥，不出现在 repr 中。
    secret: str = field(repr=False)
    database_path: Path
    issuer: str
    audience: str
    # 令牌有效期（秒）。
    expire_seconds: int
    # 是否开放 POST /api/auth/register。
    registration_enabled: bool
    # 写请求请求体的字节上限。
    max_body_bytes: int

    @classmethod
    def from_environment(cls) -> JwtSettings:
        """从环境变量加载并严格校验 JWT 配置；任何一项不合法都抛出 RuntimeError 阻止启动。"""
        # ---- 1. 读取签名密钥：JWT_SECRET_KEY 与 JWT_SECRET_FILE 只能二选一 ----
        secret = os.getenv("JWT_SECRET_KEY", "").strip()
        secret_file = os.getenv("JWT_SECRET_FILE", "").strip()
        if secret_file:
            if secret:
                raise RuntimeError("JWT_SECRET_KEY 与 JWT_SECRET_FILE 只能设置一个。")
            path = Path(secret_file)
            if not path.is_absolute():
                path = REPO_ROOT / path
            try:
                # 非 Windows 系统要求密钥文件只有属主可读写（同组和其他用户的权限位必须全为 0）；
                # Windows 没有 POSIX 权限位，跳过此检查。
                if os.name != "nt" and path.stat().st_mode & 0o077:
                    raise OSError
                secret = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                raise RuntimeError("无法读取 JWT 私有密钥文件，请检查路径、编码和权限（0600）。") from None
        # ---- 2. 校验密钥强度 ----
        # 覆盖各模型供应商及兼容访问密钥，禁止将已交给第三方的凭据复用为 JWT 签名材料。
        external_secrets = external_secret_values()
        # 长度校验不能证明随机性，但应拒绝明显占位值、重复字符和文档中的旧示例密钥。
        # 43 个 URL-safe base64 字符最多可承载约 256 位熵；至少 16 种不同字符用于排除 "aaaa..." 之类的值。
        invalid = (
            not 43 <= len(secret) <= 512 or len(set(secret)) < 16
            or any(ord(char) < 33 or ord(char) > 126 for char in secret)
            or any(word in secret.lower() for word in ("change-me", "changeme", "example", "replace", "tz9xk2pl8mn4"))
            or secret in external_secrets
        )
        if invalid:
            raise RuntimeError("请为 JWT 配置独立强随机密钥：使用 secrets.token_urlsafe(48) 生成 JWT_SECRET_KEY 或 JWT_SECRET_FILE。")
        # ---- 3. 有效期、签发方与受众 ----
        expires = positive_env_int("JWT_EXPIRE_MINUTES", 60)
        if expires > 1440:
            raise RuntimeError("JWT_EXPIRE_MINUTES 不能超过 1440 分钟。")
        issuer = os.getenv("JWT_ISSUER", "arxiv-research-agent").strip()
        audience = os.getenv("JWT_AUDIENCE", "arxiv-research-api").strip()
        if not issuer or not audience or len(issuer) > 200 or len(audience) > 200:
            raise RuntimeError("JWT_ISSUER 与 JWT_AUDIENCE 必须是 1 至 200 字符的非空标识。")
        # 注册到日志脱敏器，之后任何日志中出现该密钥都会被遮盖。
        register_sensitive_values((secret,))
        return cls(secret, auth_database_path(), issuer, audience, expires * 60,
                   env_bool("ALLOW_PUBLIC_REGISTRATION", False), positive_env_int("AUTH_MAX_REQUEST_BYTES", 1048576))
