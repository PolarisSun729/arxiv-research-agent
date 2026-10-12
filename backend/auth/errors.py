"""认证、授权和配额异常；错误码、状态与文案统一登记在 core.errors，不携带凭据或数据库异常原文。"""

from fastapi.responses import JSONResponse

from core.errors import ERROR_HTTP_STATUS, AppError


class AuthError(AppError):
    """访问控制相关的业务异常。

    参数：
    - code：必须是 core.errors 中已登记的错误码，HTTP 状态码和对外文案由登记表决定；
    - retry_after：建议客户端等待的秒数，同时写入响应体和 Retry-After 响应头；
    - quota_type：配额超限时标明是哪一类配额，方便前端提示。

    中间件在发送响应头之前捕获它并调用 to_response()；
    路由内抛出的 AuthError 由 main.py 中注册的全局异常处理器统一转换。
    """

    def __init__(self, code: str, *, retry_after: int | None = None, quota_type: str | None = None) -> None:
        if code not in ERROR_HTTP_STATUS:
            # 未登记的码会退化成 500 与通用文案，开发期直接暴露拼写错误。
            raise KeyError(code)
        super().__init__(code)
        self.retry_after, self.quota_type = retry_after, quota_type

    def to_payload(self) -> dict:
        """在通用错误体的基础上补充 retry_after / quota_type（仅在有值时）。"""
        payload = super().to_payload()
        if self.retry_after is not None:
            payload["retry_after"] = self.retry_after
        if self.quota_type:
            payload["quota_type"] = self.quota_type
        return payload

    def to_response(self) -> JSONResponse:
        """生成可直接作为 ASGI 应用调用的 JSON 错误响应。"""
        # 认证错误可能包含与具体用户相关的信息，禁止浏览器和代理缓存。
        headers = {"Cache-Control": "no-store"}
        if self.status_code == 401:
            # RFC 6750：401 响应需告知客户端应使用 Bearer 令牌认证。
            headers["WWW-Authenticate"] = "Bearer"
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return JSONResponse(self.to_payload(), status_code=self.status_code, headers=headers)
