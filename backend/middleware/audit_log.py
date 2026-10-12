"""记录入口判定和完整响应生命周期，日志只保留凭据摘要及路由模板。

每个 HTTP 请求在响应流真正结束（或失败、被取消）后写一条 JSON 审计记录，包括来源 IP、
路由模板、状态码、耗时、认证主体、错误码、是否完整发送等。不记录 query、路径参数、请求头和请求体。

相关环境变量：
- AUDIT_LOG_ENABLED：是否写审计日志，默认 true；
- AUDIT_LOG_FILE：日志文件路径，默认 logs/audit.log（相对路径以仓库根目录为基准），"-" 表示写到标准输出；
- AUDIT_LOG_MAX_BYTES：单个日志文件的轮转大小，默认 10 MiB；
- AUDIT_LOG_BACKUP_COUNT：保留的轮转文件个数，默认 5。

来源 IP、路由模板由内层的 RequestContextMiddleware 写入 scope，本模块只读取，
因此关闭审计日志（或移除本中间件）不影响认证授权与限流。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from uuid import uuid4

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from auth.key_config import REPO_ROOT
from middleware.common import ROUTE_UNMATCHED, SECURITY_HEADERS, env_bool, positive_env_int
from utils.secret_redaction import redact_sensitive_value


class _AuditFormatter(logging.Formatter):
    """把 LogRecord 上附带的 audit_event 字典脱敏后输出为单行紧凑 JSON（忽略日志消息本身）。"""

    def format(self, record: logging.LogRecord) -> str:
        # 先对字段脱敏再编码 JSON，避免对序列化文本替换引号后损坏日志，亦避免截断密钥后漏检。
        return json.dumps(redact_sensitive_value(record.audit_event), ensure_ascii=False, separators=(",", ":"))


class _AuditWriteFailure(RuntimeError):
    """审计日志写入失败的内部信号，由 AuditSink.write 捕获后转用备用通道。"""

    pass


class _AuditFailureReporting:
    """混入类：覆盖 logging.Handler.handleError，让写入失败以异常形式抛给调用方。"""

    def handleError(self, record: logging.LogRecord) -> None:
        # 标准 handler 会吞掉写入错误；改成固定异常，让审计出口显式选择备用日志通道。
        raise _AuditWriteFailure("audit_log_write_failed") from None


class _AuditStreamHandler(_AuditFailureReporting, logging.StreamHandler):
    """写到流（标准输出）的审计 handler，写入失败时抛出 _AuditWriteFailure。"""

    pass


class _AuditFileHandler(_AuditFailureReporting, RotatingFileHandler):
    """按大小轮转的审计文件 handler：日志文件只允许属主读写，写入失败时抛出 _AuditWriteFailure。"""

    def _open(self):
        # 轮转后创建的新文件也必须只有部署账号可读，不能重新依赖进程默认 umask。
        # opener 让新建文件以 0600 创建；文件已存在时创建权限不生效，所以再补一次 chmod。
        stream = open(
            self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors,
            opener=lambda path, flags: os.open(path, flags, 0o600),
        )
        # Windows 没有 POSIX 权限位，chmod 只能切换只读属性，起不到隔离作用；
        # 那里的访问控制依赖目录 ACL，这里不再假装设置权限。
        if os.name != "nt":
            try:
                os.chmod(self.baseFilename, 0o600)
            except OSError:
                stream.close()
                raise
        return stream


# 已创建的审计 logger：键为 (path, max_bytes, backups, process_id)，按最近使用排序。
_AUDIT_LOGGERS: OrderedDict[tuple, logging.Logger] = OrderedDict()
_AUDIT_LOGGERS_LOCK = threading.Lock()
# 缓存上限。正常运行只有一种配置；测试反复用不同路径创建应用时，靠它限制同时打开的文件数。
_AUDIT_LOGGERS_MAX = 32


def _audit_logger(path: str, max_bytes: int, backups: int, process_id: int) -> logging.Logger:
    """返回审计专用的 Logger；相同参数复用同一个实例。

    参数中包含进程号：fork 出的子进程会得到自己的 logger 和文件句柄，不复用父进程的。
    缓存超过上限时淘汰最久未用的 logger 并关闭其 handler，释放文件句柄；handler 仍挂在 logger 上，
    万一仍有 AuditSink 引用它，下次写入时 FileHandler 会自动重新打开文件，记录不会丢失。
    """
    key = (path, max_bytes, backups, process_id)
    with _AUDIT_LOGGERS_LOCK:
        logger = _AUDIT_LOGGERS.get(key)
        if logger is not None:
            _AUDIT_LOGGERS.move_to_end(key)
            return logger
        logger = _create_audit_logger(path, max_bytes, backups, process_id)
        _AUDIT_LOGGERS[key] = logger
        while len(_AUDIT_LOGGERS) > _AUDIT_LOGGERS_MAX:
            _, evicted = _AUDIT_LOGGERS.popitem(last=False)
            for handler in evicted.handlers:
                handler.close()
        return logger


def _create_audit_logger(path: str, max_bytes: int, backups: int, process_id: int) -> logging.Logger:
    """创建审计专用的 Logger。

    直接实例化 logging.Logger 而不是 logging.getLogger，不注册到全局 logger 树，
    其他代码对日志系统的配置不会影响它；propagate=False 保证审计记录不会混入普通应用日志。
    """
    logger = logging.Logger(f"backend.security.audit.{process_id}", level=logging.INFO)
    logger.propagate = False
    if path == "-":
        handler = _AuditStreamHandler(sys.stdout)
    else:
        log_path = Path(path)
        # 日志目录只允许属主访问（目录已存在时 mode 不生效）。
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handler = _AuditFileHandler(log_path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
    handler.setFormatter(_AuditFormatter())
    logger.addHandler(handler)
    return logger


class AuditSink:
    """审计日志的写入出口：启动时按环境变量完成初始化，运行时只负责写事件。

    AUDIT_LOG_ENABLED=false 时 logger 为 None，write 不做任何事。
    """

    def __init__(self) -> None:
        self.logger = None
        if env_bool("AUDIT_LOG_ENABLED", True):
            path = os.getenv("AUDIT_LOG_FILE", "logs/audit.log").strip()
            if not path:
                raise RuntimeError("启用审计时 AUDIT_LOG_FILE 不能为空。")
            if path != "-":
                # 相对路径以仓库根目录为基准，绝对路径与其拼接后保持不变。
                path = str((REPO_ROOT / path).resolve())
            try:
                # 同进程同配置复用 handler，避免 create_app 多次调用导致一条请求写入多份日志。
                self.logger = _audit_logger(path, positive_env_int("AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024), positive_env_int("AUDIT_LOG_BACKUP_COUNT", 5), os.getpid())
            except OSError:
                raise RuntimeError("无法初始化审计日志，请检查 AUDIT_LOG_FILE 目录权限。") from None

    def write(self, event: dict) -> None:
        """同步写入一条审计事件（会阻塞，调用方应放到线程池执行）。

        主通道写入失败时，改为把脱敏后的事件连同故障标记写到 stderr；
        stderr 也不可写时放弃，不抛出异常。
        """
        if self.logger is not None:
            try:
                # 消息体留空，实际内容通过 extra 挂到 record.audit_event 上，由 _AuditFormatter 输出。
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
    """最外层的审计中间件（纯 ASGI）。

    请求进入时：
    - 生成请求 ID（security_request_id）；来源 IP 与路由模板由内层 RequestContextMiddleware 写入。
    响应发送时：
    - 给每个响应加上 X-Request-ID 和 SECURITY_HEADERS 安全头；
    - 统计状态码、发送字节数、响应是否完整发送，以及客户端是否中途断开。
    请求结束后（包括异常和取消）：
    - 汇总以上信息及认证主体，写一条审计事件。
    """

    def __init__(self, app: ASGIApp, *, sink: AuditSink) -> None:
        self.app, self.sink = app, sink

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # 只处理 HTTP 请求；lifespan 等其他类型直接透传。
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started_at = time.perf_counter()
        scope["security_request_id"] = uuid4().hex
        # 状态码默认 500：应用没有发出响应头就结束时，按服务器错误记录。
        status, outcome, response_started, complete, disconnected = 500, "completed", False, False, False
        size = 0

        async def receive_audited() -> Message:
            """包装 receive：记录客户端是否已断开连接。"""
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def send_audited(message: Message) -> None:
            """包装 send：在响应头中加入请求 ID 与安全头，并统计状态码、字节数和完成情况。"""
            nonlocal status, response_started, complete, size
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = scope["security_request_id"]
                # 强制覆盖：即使业务代码设置了不同的值，也以统一安全头为准。
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)
            # 只有下游成功接收后才计入完成状态，发送失败不能被误记为完整响应。
            if message["type"] == "http.response.start":
                status, response_started = message["status"], True
            elif message["type"] == "http.response.body":
                size += len(message.get("body", b""))
                # more_body=False 表示这是最后一块正文；SSE 流在结束前一直是 True。
                complete = not message.get("more_body", False)

        try:
            await self.app(scope, receive_audited, send_audited)
        except BaseException as exc:
            # 任务被取消（客户端断开、服务关闭）记为 cancelled，其他异常记为 error；异常继续向上抛出。
            outcome = "cancelled" if isinstance(exc, (GeneratorExit, anyio.get_cancelled_exc_class())) else "error"
            if not response_started:
                # 499 借用 Nginx 的约定，表示客户端在响应开始前关闭了连接。
                status = 499 if outcome == "cancelled" else 500
            raise
        finally:
            # ---- 汇总认证主体：API Key 策略、JWT 会话用户，或登录接口写入的登录用户 ----
            policy = scope.get("api_key_policy")
            session = scope.get("auth_session")
            user = session.user if session else scope.get("security_login_user")
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(), "request_id": scope["security_request_id"],
                "client_ip": scope.get("security_client_ip", "unknown"), "method": scope.get("method"),
                # 路由模板可追踪端点，同时避免 query、路径参数、认证头和请求体进入审计文件。
                "path": scope.get("security_route", ROUTE_UNMATCHED), "status_code": status,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "credential_id": policy.identifier if policy else None, "credential_name": policy.name if policy else None,
                "authenticated": bool(scope.get("api_key_authenticated") or user), "code": scope.get("security_error_code"),
                "auth_mode": getattr(getattr(scope.get("app"), "state", None), "auth_mode", None),
                "user_id": user.user_id if user else None, "username": user.username if user else None,
                # target_user_id：管理员操作或注册等接口中被操作的账号。
                "role": user.role if user else None, "target_user_id": scope.get("security_target_user_id"),
                # 应用正常返回、但客户端已断开且响应没有发完时，记为 disconnected。
                "outcome": "disconnected" if outcome == "completed" and disconnected and not complete else outcome,
                "response_complete": complete, "bytes_sent": size,
            }
            # 等待流真正结束后写一条审计；取消请求也必须完成这一步，但不会缓冲或复制响应正文。
            # shield=True 屏蔽外部取消，保证日志写完；写文件是阻塞操作，放到线程池执行。
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self.sink.write, event)
