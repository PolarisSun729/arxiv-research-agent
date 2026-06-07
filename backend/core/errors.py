from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ErrorCode:
    REQUEST_VALIDATION_ERROR = "request_validation_error"
    PAPER_NOT_FOUND = "paper_not_found"
    QA_INDEX_NOT_FOUND = "qa_index_not_found"
    QA_INDEX_BUILD_FAILED = "qa_index_build_failed"
    VECTOR_STORE_ERROR = "vector_store_error"
    LLM_GENERATION_FAILED = "llm_generation_failed"
    RESUME_CHECKPOINT_NOT_FOUND = "resume_checkpoint_not_found"
    AGENT_RUNTIME_ERROR = "agent_runtime_error"
    DATABASE_WRITE_FAILED = "database_write_failed"
    UNKNOWN_ERROR = "unknown_error"


ERROR_MESSAGES: Dict[str, str] = {
    ErrorCode.REQUEST_VALIDATION_ERROR: "请求参数不合法，请检查后重试。",
    ErrorCode.PAPER_NOT_FOUND: "未找到对应论文，请确认论文 ID 是否正确。",
    ErrorCode.QA_INDEX_NOT_FOUND: "这篇论文还没有 QA 索引，请先构建索引。",
    ErrorCode.QA_INDEX_BUILD_FAILED: "论文 QA 索引构建失败，可以稍后重新构建。",
    ErrorCode.VECTOR_STORE_ERROR: "检索服务暂时不可用，请稍后重试。",
    ErrorCode.LLM_GENERATION_FAILED: "答案生成失败，请稍后重试。",
    ErrorCode.RESUME_CHECKPOINT_NOT_FOUND: "原执行现场已失效，请重新发起请求。",
    ErrorCode.AGENT_RUNTIME_ERROR: "Agent 运行失败，请稍后重试。",
    ErrorCode.DATABASE_WRITE_FAILED: "数据保存失败，请稍后重试。",
    ErrorCode.UNKNOWN_ERROR: "系统内部错误，请稍后重试。",
}

ERROR_HTTP_STATUS: Dict[str, int] = {
    ErrorCode.REQUEST_VALIDATION_ERROR: 422,
    ErrorCode.PAPER_NOT_FOUND: 404,
    ErrorCode.QA_INDEX_NOT_FOUND: 404,
    ErrorCode.QA_INDEX_BUILD_FAILED: 500,
    ErrorCode.VECTOR_STORE_ERROR: 503,
    ErrorCode.LLM_GENERATION_FAILED: 502,
    ErrorCode.RESUME_CHECKPOINT_NOT_FOUND: 404,
    ErrorCode.AGENT_RUNTIME_ERROR: 500,
    ErrorCode.DATABASE_WRITE_FAILED: 500,
    ErrorCode.UNKNOWN_ERROR: 500,
}

RECOVERABLE_DEFAULTS: Dict[str, bool] = {
    ErrorCode.REQUEST_VALIDATION_ERROR: True,
    ErrorCode.PAPER_NOT_FOUND: False,
    ErrorCode.QA_INDEX_NOT_FOUND: True,
    ErrorCode.QA_INDEX_BUILD_FAILED: True,
    ErrorCode.VECTOR_STORE_ERROR: True,
    ErrorCode.LLM_GENERATION_FAILED: True,
    ErrorCode.RESUME_CHECKPOINT_NOT_FOUND: False,
    ErrorCode.AGENT_RUNTIME_ERROR: True,
    ErrorCode.DATABASE_WRITE_FAILED: True,
    ErrorCode.UNKNOWN_ERROR: True,
}


def sanitize_detail(detail: Any, *, max_length: int = 500) -> Optional[str]:
    """保留短调试信息，避免把完整堆栈或底层库大段输出直接暴露给前端。"""
    if detail in (None, ""):
        return None
    if isinstance(detail, dict):
        compact = {
            key: value
            for key, value in detail.items()
            if key in {"code", "message", "detail", "stage", "arxiv_id", "user_id", "session_id", "job_id"}
        }
        text = str(compact or detail)
    else:
        text = str(detail)
    text = " ".join(text.split())
    if len(text) > max_length:
        return text[: max_length - 3].rstrip() + "..."
    return text


def make_error_payload(
    *,
    code: str,
    message: Optional[str] = None,
    detail: Any = None,
    recoverable: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "code": code,
        "message": message or ERROR_MESSAGES.get(code, ERROR_MESSAGES[ErrorCode.UNKNOWN_ERROR]),
        "detail": sanitize_detail(detail),
        "recoverable": RECOVERABLE_DEFAULTS.get(code, True) if recoverable is None else bool(recoverable),
    }


class AppError(Exception):
    """核心链路使用的轻量业务异常，负责携带稳定 code 而不是裸字符串。"""

    def __init__(
        self,
        code: str,
        *,
        message: Optional[str] = None,
        detail: Any = None,
        recoverable: Optional[bool] = None,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, ERROR_MESSAGES[ErrorCode.UNKNOWN_ERROR])
        self.detail = sanitize_detail(detail)
        self.recoverable = RECOVERABLE_DEFAULTS.get(code, True) if recoverable is None else bool(recoverable)
        self.status_code = status_code or ERROR_HTTP_STATUS.get(code, 500)
        self.context = dict(context or {})
        super().__init__(self.message)

    def to_payload(self) -> Dict[str, Any]:
        return make_error_payload(
            code=self.code,
            message=self.message,
            detail=self.detail,
            recoverable=self.recoverable,
        )

    def to_response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status_code, content=self.to_payload())

    def to_http_exception(self) -> HTTPException:
        return HTTPException(status_code=self.status_code, detail=self.to_payload())


def error_response(error: AppError) -> JSONResponse:
    return error.to_response()


def http_exception_to_app_error(exc: HTTPException) -> AppError:
    """把旧 HTTPException 收敛成统一错误对象，作为渐进迁移期间的兜底适配。"""
    detail = exc.detail
    if isinstance(detail, dict) and detail.get("code"):
        return AppError(
            str(detail.get("code")),
            message=str(detail.get("message") or ERROR_MESSAGES.get(str(detail.get("code")), "")) or None,
            detail=detail.get("detail"),
            recoverable=bool(detail.get("recoverable", RECOVERABLE_DEFAULTS.get(str(detail.get("code")), True))),
            status_code=exc.status_code,
        )

    code = ErrorCode.UNKNOWN_ERROR
    text = str(detail or "")
    if exc.status_code == 422:
        code = ErrorCode.REQUEST_VALIDATION_ERROR
    elif exc.status_code == 404 and "paper" in text.lower():
        code = ErrorCode.PAPER_NOT_FOUND
    elif "qa index" in text.lower():
        code = ErrorCode.QA_INDEX_NOT_FOUND

    return AppError(
        code,
        message=ERROR_MESSAGES.get(code),
        detail=detail,
        status_code=exc.status_code,
    )
