from __future__ import annotations

import time
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from .models import ToolError, ToolExecutionResult


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseToolAdapter(Generic[InputT, OutputT]):
    """强类型 adapter 基类。

    子类只实现 `_run()`；基类统一负责延迟统计、异常转 ToolError、输出模型封装。
    这样 executor 永远只消费 ToolExecutionResult，不需要理解各业务工具的异常形态。
    """

    input_model: type[InputT]
    output_model: type[OutputT]
    tool_name: str

    def execute(self, tool_input: InputT) -> ToolExecutionResult:
        started = time.perf_counter()
        try:
            output = self._run(tool_input)
            if not isinstance(output, self.output_model):
                output = self.output_model.model_validate(output)
            return self._result(ok=True, data=output, started=started)
        except ValidationError as exc:
            return self._error_result(
                started=started,
                error_code="output_validation_error",
                message="工具输出未通过 Pydantic 模型校验",
                detail={"errors": exc.errors()},
                failed_stage="output_validation",
                recoverable=False,
                retryable=False,
                raw_exception_type=exc.__class__.__name__,
            )
        except Exception as exc:  # pragma: no cover - 具体业务异常形态由底层服务决定。
            return self._error_result(
                started=started,
                error_code="tool_runtime_error",
                message=str(exc) or "工具执行失败",
                detail={},
                failed_stage="adapter_execution",
                recoverable=True,
                retryable=False,
                suggested_recovery="fallback_answer",
                raw_exception_type=exc.__class__.__name__,
            )

    def _run(self, tool_input: InputT) -> OutputT:
        raise NotImplementedError

    def _result(
        self,
        *,
        ok: bool,
        data: Optional[OutputT],
        started: float,
        error: Optional[ToolError] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            ok=ok,
            data=data,
            error=error,
            metadata=dict(metadata or {}),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            adapter_name=self.__class__.__name__,
            tool_name=self.tool_name,
        )

    def _error_result(
        self,
        *,
        started: float,
        error_code: str,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retryable: bool = False,
        suggested_recovery: Optional[str] = None,
        failed_stage: Optional[str] = None,
        raw_exception_type: Optional[str] = None,
        safe_debug: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        return self._result(
            ok=False,
            data=None,
            started=started,
            error=ToolError(
                error_code=error_code,
                message=message,
                detail=dict(detail or {}),
                recoverable=recoverable,
                retryable=retryable,
                suggested_recovery=suggested_recovery,
                failed_stage=failed_stage,
                raw_exception_type=raw_exception_type,
                safe_debug=dict(safe_debug or {}),
            ),
        )


def backend_tool_error(
    *,
    error_code: str,
    message: str,
    detail: Optional[Dict[str, Any]] = None,
    retryable: bool = False,
    suggested_recovery: Optional[str] = None,
    failed_stage: str = "backend_tool",
    safe_debug: Optional[Dict[str, Any]] = None,
) -> ToolError:
    """把旧 backend tool 的失败 envelope 转成统一 ToolError。"""
    return ToolError(
        error_code=error_code,
        message=message,
        detail=dict(detail or {}),
        recoverable=True,
        retryable=retryable,
        suggested_recovery=suggested_recovery,
        failed_stage=failed_stage,
        safe_debug=dict(safe_debug or {}),
    )
