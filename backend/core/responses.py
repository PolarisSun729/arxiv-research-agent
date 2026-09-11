"""HTTP 输出边界的统一密钥保护。"""

from typing import Any

from fastapi.responses import JSONResponse

from utils.secret_redaction import redact_sensitive_value


class RedactedJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # 正常业务响应也可能夹带 SDK 诊断信息，不能只依赖异常处理器做脱敏。
        return super().render(redact_sensitive_value(content))
