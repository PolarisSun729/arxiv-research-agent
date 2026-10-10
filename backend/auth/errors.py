"""认证、授权和配额异常；错误码、状态与文案统一登记在 core.errors，不携带凭据或数据库异常原文。"""

from fastapi.responses import JSONResponse

from core.errors import ERROR_HTTP_STATUS, AppError


class AuthError(AppError):
    def __init__(self, code: str, *, retry_after: int | None = None, quota_type: str | None = None) -> None:
        if code not in ERROR_HTTP_STATUS:
            # 未登记的码会退化成 500 与通用文案，开发期直接暴露拼写错误。
            raise KeyError(code)
        super().__init__(code)
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
