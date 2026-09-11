"""记录入口判定和完整响应生命周期，日志只保留凭据摘要及路由模板。"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from auth.api_key_middleware import SECURITY_HEADERS
from middleware.common import env_bool, positive_env_int
from middleware.ip_filter import IPFilterSettings
from utils.secret_redaction import redact_sensitive_value


class _AuditFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # 先对字段脱敏再编码 JSON，避免对序列化文本替换引号后损坏日志，亦避免截断密钥后漏检。
        return json.dumps(redact_sensitive_value(record.audit_event), ensure_ascii=False, separators=(",", ":"))


class _AuditWriteFailure(RuntimeError):
    pass


class _AuditFailureReporting:
    def handleError(self, record: logging.LogRecord) -> None:
        # 标准 handler 会吞掉写入错误；改成固定异常，让审计出口显式选择备用日志通道。
        raise _AuditWriteFailure("audit_log_write_failed") from None


class _AuditStreamHandler(_AuditFailureReporting, logging.StreamHandler):
    pass


class _AuditFileHandler(_AuditFailureReporting, RotatingFileHandler):
    def _open(self):
        # 轮转后创建的新文件也必须只有部署账号可读，不能重新依赖进程默认 umask。
        stream = open(
            self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors,
            opener=lambda path, flags: os.open(path, flags, 0o600),
        )
        try:
            os.chmod(self.baseFilename, 0o600)
        except OSError:
            stream.close()
            raise
        return stream


@lru_cache(maxsize=32)
def _audit_logger(path: str, max_bytes: int, backups: int, process_id: int) -> logging.Logger:
    logger = logging.Logger(f"backend.security.audit.{process_id}", level=logging.INFO)
    logger.propagate = False
    if path == "-":
        handler = _AuditStreamHandler(sys.stdout)
    else:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handler = _AuditFileHandler(log_path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setFormatter(_AuditFormatter())
    logger.addHandler(handler)
    return logger


class AuditSink:
    def __init__(self) -> None:
        self.logger = None
        if env_bool("AUDIT_LOG_ENABLED", True):
            path = os.getenv("AUDIT_LOG_FILE", "logs/audit.log").strip()
            if not path:
                raise RuntimeError("启用审计时 AUDIT_LOG_FILE 不能为空。")
            if path != "-":
                path = str((Path(__file__).resolve().parents[2] / path).resolve())
            try:
                # 同进程同配置复用 handler，避免 create_app 多次调用导致一条请求写入多份日志。
                self.logger = _audit_logger(path, positive_env_int("AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024), positive_env_int("AUDIT_LOG_BACKUP_COUNT", 5), os.getpid())
            except OSError:
                raise RuntimeError("无法初始化审计日志，请检查 AUDIT_LOG_FILE 目录权限。") from None

    def write(self, event: dict) -> None:
        if self.logger is not None:
            try:
                self.logger.info("", extra={"audit_event": event})
            except _AuditWriteFailure:
                # 此时响应可能已经发完，不能伪装成业务失败；将脱敏事件和故障标记送到 stderr 供收集告警。
                fallback = {"event": "audit_log_write_failed", "audit_event": redact_sensitive_value(event)}
                try:
                    sys.stderr.write(json.dumps(fallback, ensure_ascii=False, separators=(",", ":")) + "\n")
                    sys.stderr.flush()
                except (OSError, ValueError):
                    # 两个日志通道均不可写时无法补记，不覆盖原请求的异常或诱发重复执行。
                    pass


class AuditLogMiddleware:
    def __init__(self, app: ASGIApp, *, sink: AuditSink, routes: list, ip_settings: IPFilterSettings) -> None:
        self.app, self.sink, self.routes = app, sink, routes
        self.ip_settings = ip_settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started_at = time.perf_counter()
        try:
            # 预检会被 CORS 提前处理，审计仍需要记录按照同一可信代理规则解析的来源。
            scope["security_client_ip"] = self.ip_settings.client_ip(scope)
        except ValueError:
            scope["security_client_ip"] = "unknown"
        scope["security_request_id"] = uuid4().hex
        scope["security_route"] = "<unmatched>"
        for route in self.routes:
            match, matched_scope = route.matches(scope)
            if match is Match.PARTIAL and scope["security_route"] == "<unmatched>":
                # 预检和错误方法也保留匹配到的模板，不把原始路径参数写入日志。
                scope["security_route"] = getattr(route, "path", "<unmatched>")
            if match is Match.FULL:
                scope["security_route"] = getattr(route, "path", "<unmatched>")
                # 参数仅供身份绑定，不写入 path 审计字段，避免个人内容或凭据出现在 URL 日志中。
                scope["security_path_params"] = matched_scope.get("path_params", {})
                break
        status, outcome, response_started, complete, disconnected = 500, "completed", False, False, False
        size = 0

        async def receive_audited() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def send_audited(message: Message) -> None:
            nonlocal status, response_started, complete, size
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = scope["security_request_id"]
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)
            # 只有下游成功接收后才计入完成状态，发送失败不能被误记为完整响应。
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
            elif message["type"] == "http.response.body":
                size += len(message.get("body", b""))
                complete = not message.get("more_body", False)

        try:
            await self.app(scope, receive_audited, send_audited)
        except BaseException as exc:
            outcome = "cancelled" if isinstance(exc, (GeneratorExit, anyio.get_cancelled_exc_class())) else "error"
            if not response_started:
                status = 499 if outcome == "cancelled" else 500
            raise
        finally:
            policy = scope.get("api_key_policy")
            session = scope.get("auth_session")
            user = session.user if session else scope.get("security_login_user")
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": scope["security_request_id"],
                "client_ip": scope.get("security_client_ip", "unknown"), "method": scope.get("method"),
                # 路由模板可追踪端点，同时避免 query、路径参数、认证头和请求体进入审计文件。
                "path": scope["security_route"], "status_code": status,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "credential_id": policy.identifier if policy else None, "credential_name": policy.name if policy else None,
                "authenticated": bool(scope.get("api_key_authenticated") or user), "code": scope.get("security_error_code"),
                "auth_mode": getattr(getattr(scope.get("app"), "state", None), "auth_mode", None),
                "user_id": user.user_id if user else None, "username": user.username if user else None,
                "role": user.role if user else None, "target_user_id": scope.get("security_target_user_id"),
                "outcome": "disconnected" if outcome == "completed" and disconnected and not complete else outcome,
                "response_complete": complete, "bytes_sent": size,
            }
            # 等待流真正结束后写一条审计；取消请求也必须完成这一步，但不会缓冲或复制响应正文。
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self.sink.write, event)
