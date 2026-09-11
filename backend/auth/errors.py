"""认证、授权和配额共用的稳定错误契约，不携带凭据或数据库异常原文。"""

from fastapi.responses import JSONResponse

from core.errors import AppError


AUTH_ERRORS = {
    "missing_token": (401, "请先登录。"),
    "invalid_token": (401, "登录已失效，请重新登录。"),
    "invalid_credentials": (401, "用户名或密码错误，或账号已停用。"),
    "insufficient_permissions": (403, "当前账号没有执行此操作的权限。"),
    "identity_mismatch": (403, "只能访问当前登录账号的数据。"),
    "registration_disabled": (403, "当前服务未开放注册，请联系管理员创建账号。"),
    "account_conflict": (409, "无法创建账号，请检查用户名和邮箱是否已被使用。"),
    "user_not_found": (404, "账号不存在。"),
    "private_resource_not_found": (404, "请求的个人资源不存在。"),
    "last_admin_required": (409, "必须保留至少一个启用的管理员账号。"),
    "quota_exceeded": (429, "今日配额已用完，请在重置后重试。"),
    "security_storage_unavailable": (503, "访问控制服务暂时不可用，请稍后重试。"),
    "request_too_large": (413, "请求体超过允许的大小。"),
    "request_timeout": (408, "读取请求超时，请重试。"),
    "request_validation_error": (422, "请求参数不合法，请检查后重试。"),
}


class AuthError(AppError):
    def __init__(self, code: str, *, retry_after: int | None = None, quota_type: str | None = None) -> None:
        status, message = AUTH_ERRORS[code]
        super().__init__(code, message=message, status_code=status, recoverable=status in {408, 429, 503})
        self.retry_after, self.quota_type = retry_after, quota_type

    def to_payload(self) -> dict:
        payload = super().to_payload()
        if self.retry_after is not None:
            payload["retry_after"] = self.retry_after
        if self.quota_type:
            payload["quota_type"] = self.quota_type
        return payload

    def to_response(self) -> JSONResponse:
        headers = {"Cache-Control": "no-store"}
        if self.status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return JSONResponse(self.to_payload(), status_code=self.status_code, headers=headers)
