"""账号身份与默认配额；业务 user_id 使用服务端生成的字符串，不认领旧演示数据。"""

from dataclasses import dataclass, field
from typing import Literal


# 角色由高到低：admin 管理员；researcher 研究员（可使用 Agent）；viewer 只读用户；guest 访客（新注册默认角色）。
UserRole = Literal["admin", "researcher", "viewer", "guest"]
# 配额类型：papers 论文检索/下载/回源等；qa_queries 论文问答；agent_runs Agent 对话。
QuotaType = Literal["papers", "qa_queries", "agent_runs"]
# 与 QuotaType 一致的运行时元组，用于遍历和校验（Literal 类型本身不便在运行时使用）。
QUOTA_TYPES = ("papers", "qa_queries", "agent_runs")
# 各角色的默认每日配额：-1 表示不限，0 表示禁止使用该类操作。
# 创建账号或变更角色时按此表写入 auth_quotas；管理员之后可以按用户单独调整。
DEFAULT_QUOTAS: dict[str, dict[str, int]] = {
    "admin": {"papers": -1, "qa_queries": -1, "agent_runs": -1},
    "researcher": {"papers": 500, "qa_queries": 200, "agent_runs": 20},
    "viewer": {"papers": 50, "qa_queries": 20, "agent_runs": 0},
    "guest": {"papers": 10, "qa_queries": 5, "agent_runs": 0},
}


@dataclass(frozen=True)
class AuthUser:
    """auth_users 表中的一行；字段名与表列名一一对应（AuthStore._user 直接按列名构造）。

    - user_id：服务端生成的 uuid4 hex，作为业务数据的归属标识；
    - username / email：保存时已统一转为小写（casefold）；
    - created_at / last_login：UTC 时间的 ISO 字符串；
    - token_version：会话版本号，改密、改角色、停用时递增，使旧令牌全部失效；
    - hashed_password：bcrypt 哈希。

    token_version 与 hashed_password 不出现在 repr 中，避免被意外写入日志。
    """

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
        """返回可以安全返回给客户端的字段。"""
        # 白名单投影避免未来新增密码恢复字段时被自动序列化到响应。
        return {name: getattr(self, name) for name in (
            "user_id", "username", "email", "role", "is_active", "created_at", "last_login",
        )}
