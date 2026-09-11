"""访问密钥及其策略的启动配置；文件配置显式失败，不降级为更宽松的环境配置。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from limits import parse_many

from utils.secret_redaction import register_sensitive_values


REPO_ROOT = Path(__file__).resolve().parents[2]


def validate_rate(value: str) -> str:
    try:
        limits = parse_many(value) if isinstance(value, str) else []
        if not limits or any(item.amount <= 0 or item.multiples <= 0 for item in limits):
            raise ValueError
    except (ValueError, TypeError):
        raise RuntimeError("速率限制必须是正数规则，例如 60/minute;10/second。") from None
    return value


def _validate_secret(key: str) -> str:
    if not isinstance(key, str) or not 32 <= len(key) <= 256 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise RuntimeError("每个访问密钥必须包含 32 至 256 个无空白的可打印 ASCII 字符。")
    if key == os.getenv("ALIYUN_API_KEY", "").strip():
        raise RuntimeError("访问密钥必须与 ALIYUN_API_KEY 分开生成，模型供应商密钥不能交给浏览器。")
    return key


def get_valid_api_keys() -> set[str]:
    keys = {key.strip() for key in os.getenv("BACKEND_API_KEYS", "").split(",") if key.strip()}
    if not keys:
        raise RuntimeError("必须配置 BACKEND_API_KEYS 或 API_KEY_CONFIG_FILE，不能以匿名模式启动。")
    return {_validate_secret(key) for key in keys}


def _parse_date(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
        if parsed is None:
            raise ValueError
        # 不带时区的 ISO 时间和日期统一按 UTC 解释，避免服务器时区改变密钥有效期。
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        raise RuntimeError("密钥日期必须是合法 ISO 日期或时间，建议明确使用 UTC 时区。") from None


@dataclass(frozen=True)
class ApiKeyPolicy:
    digest: bytes = field(repr=False)
    name: str
    rate_limit: str
    daily_quota: int | None = None
    enabled: bool = True
    expires_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def identifier(self) -> str:
        # 完整摘要用于计数和审计，既不暴露密钥前缀，也不会把同前缀的不同密钥合并。
        return self.digest.hex()

    def rejection_code(self) -> str | None:
        if not self.enabled:
            return "api_key_disabled"
        if self.expires_at is not None and datetime.now(timezone.utc) >= self.expires_at:
            return "api_key_expired"
        return None


def load_key_policies() -> tuple[ApiKeyPolicy, ...]:
    configured_path = os.getenv("API_KEY_CONFIG_FILE", "").strip()
    default_path = REPO_ROOT / "backend" / "config" / "api_keys.json"
    path = Path(configured_path) if configured_path else default_path
    if not path.is_absolute():
        path = REPO_ROOT / path
    default_rate = validate_rate(os.getenv("RATE_LIMIT_KEY", "60/minute"))
    if not configured_path and not path.exists():
        records = [{"key": key, "name": "环境变量密钥"} for key in sorted(get_valid_api_keys())]
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise RuntimeError("无法读取密钥配置文件，请检查路径、权限和 JSON 格式。") from None
        records = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(records, list) or not records:
            raise RuntimeError("密钥配置文件必须包含非空 keys 列表。")

    policies = []
    digests = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("每项密钥配置必须是对象。")
        # 示例使用环境变量引用；私有文件也兼容 key 字段，但不能同时配置两个来源。
        if "key" in record and "key_env" in record:
            raise RuntimeError("密钥配置只能选择 key 或 key_env 中的一项。")
        env_name = record.get("key_env")
        if env_name is not None and (not isinstance(env_name, str) or not env_name):
            raise RuntimeError("key_env 必须是非空环境变量名。")
        secret = _validate_secret(os.getenv(env_name, "") if env_name else record.get("key"))
        digest = hashlib.sha256(secret.encode("ascii")).digest()
        if digest in digests:
            raise RuntimeError("密钥配置存在重复项，不能为同一密钥设置冲突的策略。")
        name = record.get("name", "未命名密钥")
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or any(ord(char) < 32 for char in name):
            raise RuntimeError("密钥名称必须是 1 至 80 个字符，且不能包含控制字符。")
        enabled = record.get("enabled", True)
        quota = record.get("daily_quota")
        if not isinstance(enabled, bool) or (quota is not None and (type(quota) is not int or quota <= 0)):
            raise RuntimeError("enabled 必须是布尔值，daily_quota 必须是正整数或 null。")
        policies.append(ApiKeyPolicy(
            digest=digest, name=name.strip(), rate_limit=validate_rate(record.get("rate_limit", default_rate)),
            daily_quota=quota, enabled=enabled, expires_at=_parse_date(record.get("expires_at")),
            created_at=_parse_date(record.get("created_at")),
        ))
        digests.add(digest)
        # 文件中的凭据也要进入公共脱敏器，不能只保护 os.environ 中的密钥。
        register_sensitive_values((secret,))
    return tuple(policies)
