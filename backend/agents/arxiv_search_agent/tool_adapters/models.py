from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolError(BaseModel):
    """统一工具错误模型。

    adapter 不再用异常字符串表达失败；这些字段给 executor、observer 和 replanner
    提供稳定恢复语义，便于区分输入校验、输出校验、业务失败和外部服务异常。
    """

    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str
    detail: Dict[str, Any] = Field(default_factory=dict)
    recoverable: bool = True
    retryable: bool = False
    suggested_recovery: Optional[str] = None
    failed_stage: Optional[str] = None
    raw_exception_type: Optional[str] = None
    safe_debug: Dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    """所有 ToolAdapter 的统一返回 envelope。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    data: Any = None
    error: Optional[ToolError] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    adapter_name: str
    tool_name: str
