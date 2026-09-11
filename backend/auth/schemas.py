"""账号接口严格拒绝未声明字段，公开注册无法借额外 role/user_id 字段提权。"""

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator

from auth.models import QuotaType, UserRole
from auth.passwords import validate_password


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserCreate(StrictModel):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr = Field(max_length=254)
    password: SecretStr

    @field_validator("password")
    @classmethod
    def password_policy(cls, value: SecretStr) -> SecretStr:
        validate_password(value.get_secret_value())
        return value


class AdminUserCreate(UserCreate):
    role: UserRole = "researcher"


class UserLogin(StrictModel):
    username: str = Field(min_length=1, max_length=50)
    password: SecretStr = Field(min_length=1, max_length=72)


class PasswordChange(StrictModel):
    current_password: SecretStr = Field(min_length=1, max_length=72)
    new_password: SecretStr

    @field_validator("new_password")
    @classmethod
    def password_policy(cls, value: SecretStr) -> SecretStr:
        validate_password(value.get_secret_value())
        return value


class AdminUserUpdate(StrictModel):
    role: UserRole | None = None
    is_active: bool | None = None


class QuotaUpdate(StrictModel):
    quota_type: QuotaType
    daily_limit: int = Field(ge=-1, le=1000000, strict=True)
