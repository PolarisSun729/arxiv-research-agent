"""HTTP 输出边界的统一密钥保护。"""

import json
from typing import Any

from fastapi.responses import JSONResponse

from utils.secret_redaction import redact_sensitive_value


class RedactedJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        # 正常业务响应也可能夹带 SDK 诊断信息，不能只依赖异常处理器做脱敏。
        return super().render(redact_sensitive_value(content))


def format_sse_event(event_name: str, payload: Any) -> str:
    """把一条事件格式化为 SSE 文本帧，并在序列化前统一脱敏。

    StreamingResponse 不经过 RedactedJSONResponse，流式出口必须在逐条序列化时自行套用同一脱敏边界。
    """
    return f"event: {event_name}\ndata: {json.dumps(redact_sensitive_value(payload), ensure_ascii=False)}\n\n"
