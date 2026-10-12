"""账号接口严格拒绝未声明字段，公开注册无法借额外 role/user_id 字段提权。

密码字段统一使用 SecretStr：打印对象或记录校验错误时只显示 '**********'，不会泄露明文。
"""

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator

from auth.models import QuotaType, UserRole
from auth.passwords import validate_password


class StrictModel(BaseModel):
    """所有账号接口模型的基类：出现未声明的字段时直接校验失败（422），而不是静默忽略。"""

    model_config = ConfigDict(extra="forbid")


class UserCreate(StrictModel):
    """公开注册的请求体。不含 role 字段，新用户角色由服务端决定。"""

    # 用户名只允许字母、数字、下划线和连字符。
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    # 254 是 RFC 5321 规定的邮箱地址最大长度。
    email: EmailStr = Field(max_length=254)
    password: SecretStr

    @field_validator("password")
    @classmethod
    def password_policy(cls, value: SecretStr) -> SecretStr:
        """在入参校验阶段执行密码强度规则，不合格时返回 422 并给出具体原因。"""
        validate_password(value.get_secret_value())
        return value


class AdminUserCreate(UserCreate):
    """管理员创建账号的请求体：在注册字段之外，可以指定角色（默认 researcher）。"""

    role: UserRole = "researcher"


class UserLogin(StrictModel):
    """登录请求体。只限制长度、不检查密码强度，避免向未登录的调用方透露密码策略。"""

    username: str = Field(min_length=1, max_length=50)
    # 72 与 bcrypt 的输入上限一致，更长的密码不可能是合法密码。
    password: SecretStr = Field(min_length=1, max_length=72)


class PasswordChange(StrictModel):
    """修改本人密码的请求体：需要提供当前密码，新密码必须满足强度规则。"""

    current_password: SecretStr = Field(min_length=1, max_length=72)
    new_password: SecretStr

    @field_validator("new_password")
    @classmethod
    def password_policy(cls, value: SecretStr) -> SecretStr:
        """新密码同样在入参校验阶段执行强度规则。"""
        validate_password(value.get_secret_value())
        return value


class AdminUserUpdate(StrictModel):
    """管理员修改账号的请求体；字段为 None 表示保持不变。"""

    role: UserRole | None = None
    is_active: bool | None = None


class QuotaUpdate(StrictModel):
    """管理员调整某个用户某一类配额的请求体。"""

    quota_type: QuotaType
    # -1 表示不限；strict=True 拒绝 "100"、1.0、true 这类会被隐式转换为整数的值。
    daily_limit: int = Field(ge=-1, le=1000000, strict=True)
