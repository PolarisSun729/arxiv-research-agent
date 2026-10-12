"""访问密钥及其策略的启动配置；文件配置显式失败，不降级为更宽松的环境配置。

密钥来源的优先级（见 load_key_policies）：
1. 设置了 API_KEY_CONFIG_FILE：必须能读取该 JSON 文件，否则启动失败；
2. 未设置但默认文件 backend/config/api_keys.json 存在：读取默认文件；
3. 都没有：退回环境变量 BACKEND_API_KEYS（逗号分隔），所有密钥使用默认速率、不限日配额。

配置文件格式示例：
    {"keys": [{"key_env": "TEAM_A_KEY", "name": "团队 A", "rate_limit": "60/minute",
               "daily_quota": 1000, "enabled": true, "expires_at": "2026-12-31"}]}

内存中只保存密钥的 SHA-256 摘要，明文在加载时注册到日志脱敏器后即不再保留。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from limits import parse_many

from utils.secret_redaction import register_sensitive_values


# 仓库根目录（本文件位于 <repo>/backend/auth/），相对路径的配置都以它为基准解析。
REPO_ROOT = Path(__file__).resolve().parents[2]
# 单条密钥配置允许出现的字段白名单。
_RECORD_FIELDS = frozenset({"key", "key_env", "name", "rate_limit", "daily_quota", "enabled", "created_at", "expires_at"})
# 合法的环境变量名：字母或下划线开头，最长 128 个字符。
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")


def external_secret_values(*, exclude: frozenset[str] | set[str] = frozenset()) -> dict[str, str]:
    """收集已交给第三方或其他用途的环境变量凭据，返回 值 → 变量名，供各类本地密钥做复用检查。

    凡是变量名以 _API_KEY / _API_KEYS / _ACCESS_KEY 结尾（不区分大小写）的环境变量都会被收集，
    多值变量按逗号拆分。exclude 中列出的变量名不参与收集。
    """
    excluded = {name.upper() for name in exclude}
    secrets_by_value = {}
    for name, raw in os.environ.items():
        upper = name.upper()
        if upper in excluded or not upper.endswith(("_API_KEY", "_API_KEYS", "_ACCESS_KEY")):
            continue
        for value in raw.split(","):
            if value.strip():
                # 同一个值出现在多个变量中时，保留最先遇到的变量名用于报错提示。
                secrets_by_value.setdefault(value.strip(), name)
    return secrets_by_value


def validate_rate(value: str) -> str:
    """校验 limits 库格式的速率规则（如 "60/minute;10/second"），合法则原样返回。

    每一条规则的次数和时间倍数都必须是正数，否则抛出 RuntimeError。
    """
    try:
        limits = parse_many(value) if isinstance(value, str) else []
        if not limits or any(item.amount <= 0 or item.multiples <= 0 for item in limits):
            raise ValueError
    except (ValueError, TypeError):
        raise RuntimeError("速率限制必须是正数规则，例如 60/minute;10/second。") from None
    return value


def _validate_secret(key: str, external_secrets: dict[str, str]) -> str:
    """校验单个访问密钥：32~256 个可打印 ASCII 字符（不含空白），且不能与第三方凭据相同。"""
    # 33~126 是除空格外的可打印 ASCII 字符范围。
    if not isinstance(key, str) or not 32 <= len(key) <= 256 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise RuntimeError("每个访问密钥必须包含 32 至 256 个无空白的可打印 ASCII 字符。")
    if key in external_secrets:
        # 只报告变量名，不回显密钥值。
        raise RuntimeError(f"访问密钥不能与环境变量 {external_secrets[key]} 相同，模型供应商等第三方密钥不能交给浏览器。")
    return key


def get_valid_api_keys() -> set[str]:
    """读取并校验 BACKEND_API_KEYS 中的密钥（逗号分隔）；一个都没有时拒绝启动。"""
    keys = {key.strip() for key in os.getenv("BACKEND_API_KEYS", "").split(",") if key.strip()}
    if not keys:
        raise RuntimeError("必须配置 BACKEND_API_KEYS 或 API_KEY_CONFIG_FILE，不能以匿名模式启动。")
    # BACKEND_API_KEYS 本身就是访问密钥，需要排除在"第三方凭据"之外，否则会与自己比较而误报。
    external = external_secret_values(exclude={"BACKEND_API_KEYS"})
    return {_validate_secret(key, external) for key in keys}


def _parse_date(value: object, *, end_of_day: bool = False) -> datetime | None:
    """把配置中的日期/时间字符串解析为 UTC datetime；None 表示未配置。

    - 纯日期（如 "2026-12-31"）：end_of_day=False 时取当天 00:00 UTC；
      end_of_day=True（用于失效时间）时取次日 00:00 UTC，即当天全天有效；
    - 带时间的 ISO 字符串：不带时区按 UTC 解释，带时区则换算为 UTC。
    """
    if value is None:
        return None
    try:
        if not isinstance(value, str):
            raise ValueError
        try:
            day = date.fromisoformat(value)
        except ValueError:
            day = None
        if day is not None:
            # 只写日期的失效时间表示当天 UTC 全天有效，次日 00:00 UTC 才失效；生效时间仍取当天 0 点。
            start = datetime.combine(day, datetime.min.time(), timezone.utc)
            return start + timedelta(days=1) if end_of_day else start
        parsed = datetime.fromisoformat(value)
        # 不带时区的 ISO 时间统一按 UTC 解释，避免服务器时区改变密钥有效期。
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        raise RuntimeError("密钥日期必须是合法 ISO 日期或时间，建议明确使用 UTC 时区。") from None


def _field_names(names) -> str:
    """把字段名集合格式化为报错文案中的逗号分隔列表。"""
    # 字段名来自配置文件；过长的名字可能是误粘贴的密钥，只提示存在而不回显。
    return ", ".join(sorted(name if isinstance(name, str) and len(name) <= 40 else "<过长字段名>" for name in names))


@dataclass(frozen=True)
class ApiKeyPolicy:
    """单个访问密钥的策略。

    - digest：密钥明文的 SHA-256 摘要，用于匹配；不出现在 repr 中；
    - name：便于管理员识别的名称；
    - rate_limit：该密钥的速率规则（limits 格式）；
    - daily_quota：每个 UTC 自然日可执行的付费操作次数，None 表示不限；
    - enabled：是否启用；
    - expires_at：失效时间（UTC），None 表示永不过期；
    - created_at：创建时间（UTC），仅作记录。
    """

    digest: bytes = field(repr=False)
    name: str
    rate_limit: str
    daily_quota: int | None = None
    enabled: bool = True
    expires_at: datetime | None = None
    created_at: datetime | None = None

    @property
    def identifier(self) -> str:
        """限流计数和审计日志中用来标识该密钥的字符串。"""
        # 完整摘要用于计数和审计，既不暴露密钥前缀，也不会把同前缀的不同密钥合并。
        return self.digest.hex()

    def rejection_code(self) -> str | None:
        """密钥当前不可用时返回对应的错误码（禁用 / 过期），可用时返回 None。"""
        if not self.enabled:
            return "api_key_disabled"
        if self.expires_at is not None and datetime.now(timezone.utc) >= self.expires_at:
            return "api_key_expired"
        return None


def load_key_policies() -> tuple[ApiKeyPolicy, ...]:
    """按模块文档描述的优先级加载所有密钥策略，任何配置错误都抛出 RuntimeError 阻止启动。"""
    # ---- 1. 确定配置来源 ----
    configured_path = os.getenv("API_KEY_CONFIG_FILE", "").strip()
    default_path = REPO_ROOT / "backend" / "config" / "api_keys.json"
    path = Path(configured_path) if configured_path else default_path
    if not path.is_absolute():
        path = REPO_ROOT / path
    # 未单独配置 rate_limit 的密钥使用 RATE_LIMIT_KEY 作为默认速率。
    default_rate = validate_rate(os.getenv("RATE_LIMIT_KEY", "60/minute"))
    if not configured_path and not path.exists():
        # 既没有显式配置文件、也没有默认文件：退回环境变量，转换成与配置文件相同的记录格式统一校验。
        records = [{"key": key, "name": "环境变量密钥"} for key in sorted(get_valid_api_keys())]
    else:
        try:
            # utf-8-sig 兼容 Windows 记事本保存时写入的 BOM。
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise RuntimeError("无法读取密钥配置文件，请检查路径、权限和 JSON 格式。") from None
        records = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(records, list) or not records:
            raise RuntimeError("密钥配置文件必须包含非空 keys 列表。")
        if set(data) - {"keys"}:
            raise RuntimeError(f"密钥配置文件顶层只允许 keys，发现未知字段：{_field_names(set(data) - {'keys'})}。")

    # ---- 2. 结构校验：先检查所有记录的字段，再逐条解析 ----
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise RuntimeError("每项密钥配置必须是对象。")
        unknown = set(record) - _RECORD_FIELDS
        if unknown:
            # 拼错的 enabled/daily_quota 会被当成默认值（启用、无限额），必须启动失败而不是静默放宽。
            raise RuntimeError(f"第 {index} 项密钥配置包含未知字段：{_field_names(unknown)}；允许的字段：{_field_names(_RECORD_FIELDS)}。")
    # 本文件引用的变量本来就是访问密钥，不参与第三方密钥复用检查；重复密钥由摘要去重拦截。
    referenced_envs = {record["key_env"] for record in records if isinstance(record.get("key_env"), str)}
    external = external_secret_values(exclude={"BACKEND_API_KEYS", *referenced_envs})

    # ---- 3. 逐条解析为 ApiKeyPolicy ----
    policies = []
    digests = set()
    for record in records:
        # 示例使用环境变量引用；私有文件也兼容 key 字段，但不能同时配置两个来源。
        if "key" in record and "key_env" in record:
            raise RuntimeError("密钥配置只能选择 key 或 key_env 中的一项。")
        env_name = record.get("key_env")
        if env_name is not None and (not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name)):
            raise RuntimeError("key_env 必须是合法的环境变量名（字母、数字和下划线，不能以数字开头）。")
        if env_name and not os.getenv(env_name, ""):
            raise RuntimeError(f"key_env 引用的环境变量 {env_name} 未设置或为空。")
        secret = _validate_secret(os.getenv(env_name, "") if env_name else record.get("key"), external)
        # 已校验为纯 ASCII，可以安全地按 ascii 编码。
        digest = hashlib.sha256(secret.encode("ascii")).digest()
        if digest in digests:
            raise RuntimeError("密钥配置存在重复项，不能为同一密钥设置冲突的策略。")
        name = record.get("name", "未命名密钥")
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or any(ord(char) < 32 for char in name):
            raise RuntimeError("密钥名称必须是 1 至 80 个字符，且不能包含控制字符。")
        enabled = record.get("enabled", True)
        quota = record.get("daily_quota")
        # type(quota) is not int 排除了 bool 和浮点数等看似数字的值。
        if not isinstance(enabled, bool) or (quota is not None and (type(quota) is not int or quota <= 0)):
            raise RuntimeError("enabled 必须是布尔值，daily_quota 必须是正整数或 null。")
        policies.append(ApiKeyPolicy(
            digest=digest, name=name.strip(), rate_limit=validate_rate(record.get("rate_limit", default_rate)),
            daily_quota=quota, enabled=enabled, expires_at=_parse_date(record.get("expires_at"), end_of_day=True),
            created_at=_parse_date(record.get("created_at")),
        ))
        digests.add(digest)
        # 文件中的凭据也要进入公共脱敏器，不能只保护 os.environ 中的密钥。
        register_sensitive_values((secret,))
    return tuple(policies)
