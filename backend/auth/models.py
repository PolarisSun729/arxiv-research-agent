"""账号身份与默认配额；业务 user_id 使用服务端生成的字符串，不认领旧演示数据。"""

from dataclasses import dataclass, field
from typing import Literal


UserRole = Literal["admin", "researcher", "viewer", "guest"]
QuotaType = Literal["papers", "qa_queries", "agent_runs"]
QUOTA_TYPES = ("papers", "qa_queries", "agent_runs")
DEFAULT_QUOTAS: dict[str, dict[str, int]] = {
    "admin": {"papers": -1, "qa_queries": -1, "agent_runs": -1},
    "researcher": {"papers": 500, "qa_queries": 200, "agent_runs": 20},
    "viewer": {"papers": 50, "qa_queries": 20, "agent_runs": 0},
    "guest": {"papers": 10, "qa_queries": 5, "agent_runs": 0},
}


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: str
    last_login: str | None
    token_version: int = field(repr=False)
    hashed_password: str = field(repr=False)

    def public_view(self) -> dict:
        # 白名单投影避免未来新增密码恢复字段时被自动序列化到响应。
        return {name: getattr(self, name) for name in (
            "user_id", "username", "email", "role", "is_active", "created_at", "last_login",
        )}
