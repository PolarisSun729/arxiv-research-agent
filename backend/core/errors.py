from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from utils.secret_redaction import redact_sensitive_value, redact_text

logger = logging.getLogger(__name__)


class ErrorCode:
    MISSING_API_KEY = "missing_api_key"
    INVALID_API_KEY = "invalid_api_key"
    API_KEY_DISABLED = "api_key_disabled"
    API_KEY_EXPIRED = "api_key_expired"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    DAILY_QUOTA_EXCEEDED = "daily_quota_exceeded"
    IP_BLOCKED = "ip_blocked"
    IP_NOT_ALLOWED = "ip_not_allowed"
    IP_TEMPORARILY_BLOCKED = "ip_temporarily_blocked"
    SECURITY_STORAGE_UNAVAILABLE = "security_storage_unavailable"
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
    ErrorCode.MISSING_API_KEY: "缺少访问密钥，请在请求头中提供 X-API-Key。",
    ErrorCode.INVALID_API_KEY: "访问密钥无效，请重新输入。",
    ErrorCode.API_KEY_DISABLED: "访问密钥已被禁用，请联系管理员。",
    ErrorCode.API_KEY_EXPIRED: "访问密钥已过期，请联系管理员。",
    ErrorCode.RATE_LIMIT_EXCEEDED: "请求过于频繁，请稍后重试。",
    ErrorCode.DAILY_QUOTA_EXCEEDED: "此访问密钥今日配额已用尽，请在 UTC 次日重试。",
    ErrorCode.IP_BLOCKED: "当前 IP 已被禁止访问。",
    ErrorCode.IP_NOT_ALLOWED: "当前 IP 不在允许的访问范围内。",
    ErrorCode.IP_TEMPORARILY_BLOCKED: "检测到过多失败或超限请求，当前 IP 已被临时封禁。",
    ErrorCode.SECURITY_STORAGE_UNAVAILABLE: "访问控制服务暂时不可用，请稍后重试。",
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
    ErrorCode.MISSING_API_KEY: 401,
    ErrorCode.INVALID_API_KEY: 403,
    ErrorCode.API_KEY_DISABLED: 403,
    ErrorCode.API_KEY_EXPIRED: 403,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.DAILY_QUOTA_EXCEEDED: 429,
    ErrorCode.IP_BLOCKED: 403,
    ErrorCode.IP_NOT_ALLOWED: 403,
    ErrorCode.IP_TEMPORARILY_BLOCKED: 403,
    ErrorCode.SECURITY_STORAGE_UNAVAILABLE: 503,
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
    ErrorCode.MISSING_API_KEY: False,
    ErrorCode.INVALID_API_KEY: False,
    ErrorCode.API_KEY_DISABLED: False,
    ErrorCode.API_KEY_EXPIRED: False,
    ErrorCode.IP_BLOCKED: False,
    ErrorCode.IP_NOT_ALLOWED: False,
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
    """先移除凭据，再保留短调试信息，防止截断后留下无法匹配的密钥前缀。"""
    if detail in (None, ""):
        return None
    detail = redact_sensitive_value(detail)
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
        "message": redact_text(message or ERROR_MESSAGES.get(code, ERROR_MESSAGES[ErrorCode.UNKNOWN_ERROR])),
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
        # detail 对外会被压缩成短文本；原始 detail 单独保留，用来透传结构化观察信息等白名单字段。
        self.raw_detail = detail
        self.detail = sanitize_detail(detail)
        self.recoverable = RECOVERABLE_DEFAULTS.get(code, True) if recoverable is None else bool(recoverable)
        self.status_code = status_code or ERROR_HTTP_STATUS.get(code, 500)
        self.context = dict(context or {})
        super().__init__(self.message)

    def to_payload(self) -> Dict[str, Any]:
        payload = make_error_payload(
            code=self.code,
            message=self.message,
            detail=self.detail,
            recoverable=self.recoverable,
        )
        qa_observation = None
        if isinstance(self.raw_detail, dict):
            qa_observation = self.raw_detail.get("qa_observation")
        if qa_observation is None and isinstance(self.context, dict):
            qa_observation = self.context.get("qa_observation")
        if qa_observation is not None:
            # Paper QA 失败时 Agent 需要结构化读取质量原因，不能只依赖 detail 文本。
            payload["qa_observation"] = redact_sensitive_value(qa_observation)
        return payload

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
        # 旧路由经常把 SDK 异常原文放入 5xx detail；公网响应只返回稳定消息。
        detail=None if exc.status_code >= 500 else detail,
        status_code=exc.status_code,
    )
