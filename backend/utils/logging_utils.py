from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from utils.config import get_backend_logging_runtime_config
from utils.secret_redaction import install_log_redaction, is_secret_field, redact_sensitive_value, redact_text


_SAFE_EVENT_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def configure_backend_logging(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """统一配置后端 CMD 日志层级，并允许按 logger 前缀局部打开 DEBUG。"""
    runtime = dict(config or get_backend_logging_runtime_config())
    install_log_redaction()
    level_name = str(runtime.get("level") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    for logger_name in runtime.get("debug_loggers") or []:
        # 模块级 DEBUG 只放大指定前缀，避免为了看检索细节把数据库、HTTP access 等全部打开。
        logging.getLogger(str(logger_name)).setLevel(logging.DEBUG)
    return runtime


def _runtime_config() -> Dict[str, Any]:
    return get_backend_logging_runtime_config()


def _should_redact_key(key: Any, *, config: Mapping[str, Any]) -> bool:
    # 保留调用签名，但不允许旧的 redact_secrets=False 绕过部署后的密钥保护。
    return is_secret_field(key)


def redact_log_value(value: Any, *, config: Optional[Mapping[str, Any]] = None) -> Any:
    """递归脱敏字段和文本中的凭据，完整 trace 与控制台使用相同规则。"""
    return redact_sensitive_value(value)


def _to_log_text(value: Any, *, config: Mapping[str, Any]) -> str:
    safe_value = redact_log_value(value, config=config)
    if isinstance(safe_value, str):
        text = safe_value
    else:
        try:
            text = json.dumps(safe_value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            text = str(safe_value)
    return redact_text(text).replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


def format_log_preview(value: Any, *, limit: Optional[int] = None, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """把任意输入/输出压成 INFO 友好的单行预览，并返回原始长度和截断标记。"""
    runtime = dict(config or _runtime_config())
    text = _to_log_text(value, config=runtime)
    full_io = bool(runtime.get("full_io", False))
    max_chars = int(limit if limit is not None else runtime.get("io_preview_chars", 1200) or 1200)
    truncated = False
    preview = text
    if not full_io and max_chars > 0 and len(text) > max_chars:
        truncated = True
        preview = text[:max_chars] + "..."
    return {
        "preview": preview,
        "chars": len(text),
        "truncated": truncated,
    }


def _format_scalar(value: Any, *, config: Mapping[str, Any]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = _to_log_text(value, config=config)
    max_chars = int(config.get("io_preview_chars", 1200) or 1200)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "..."
    return json.dumps(text, ensure_ascii=False)


def format_log_kv(**fields: Any) -> str:
    runtime = _runtime_config()
    parts: List[str] = []
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        if _should_redact_key(key, config=runtime):
            value = "***REDACTED***"
        if key in {"input", "output"}:
            preview = format_log_preview(value, config=runtime)
            parts.append(f"{key}_preview={_format_scalar(preview['preview'], config=runtime)}")
            parts.append(f"{key}_chars={preview['chars']}")
            parts.append(f"{key}_truncated={'true' if preview['truncated'] else 'false'}")
            continue
        parts.append(f"{key}={_format_scalar(value, config=runtime)}")
    return " ".join(parts)


def info_event(logger: logging.Logger, event_name: str, **fields: Any) -> None:
    """输出稳定事件名和 key=value 字段，作为 INFO 主链路时间线的唯一格式。"""
    if not logger.isEnabledFor(logging.INFO):
        return
    event = _SAFE_EVENT_NAME_RE.sub("_", str(event_name or "event")).strip("_") or "event"
    payload = format_log_kv(**fields)
    logger.info("%s%s", event, f" {payload}" if payload else "")


@dataclass
class RequestTrace:
    """请求级 trace 只承载完整 IO 和事件摘要，检索候选等重对象继续留在 retrieval trace。"""

    run_id: str
    route: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    input: Any = None
    output: Any = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    fallback: bool = False
    failed: bool = False

    def add_event(self, event_name: str, **fields: Any) -> None:
        self.events.append(
            {
                "event": event_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fields": redact_log_value(fields),
            }
        )

    def set_output(self, output: Any) -> None:
        self.output = output

    def mark_fallback(self) -> None:
        self.fallback = True

    def mark_failed(self) -> None:
        self.failed = True

    def should_write(self) -> bool:
        config = _runtime_config()
        mode = str(config.get("request_trace") or "auto").lower()
        if mode in {"1", "true", "yes", "on", "always"}:
            return True
        if mode in {"0", "false", "no", "off", "never"}:
            # 显式关闭时尊重配置，不再因为失败强制落完整 IO；默认 auto 才负责失败兜底追踪。
            return False
        input_preview = format_log_preview(self.input, config=config)
        output_preview = format_log_preview(self.output, config=config)
        return bool(input_preview["truncated"] or output_preview["truncated"] or self.failed or self.fallback)

    def write(self, *, reason: str = "auto") -> Optional[str]:
        if not self.should_write():
            return None
        config = _runtime_config()
        trace_dir = Path(str(config.get("request_trace_dir") or "temp/backend-request-traces"))
        trace_dir.mkdir(parents=True, exist_ok=True)
        safe_run_id = _SAFE_EVENT_NAME_RE.sub("_", self.run_id).strip("_") or "request"
        trace_path = trace_dir / f"{safe_run_id}.json"
        # trace 与 CMD 日志使用同一套脱敏，避免完整 IO 落盘时重新暴露密钥类字段。
        payload = redact_log_value(
            {
                "run_id": self.run_id,
                "route": self.route,
                "session_id": self.session_id,
                "user_id": self.user_id,
                "reason": reason,
                "failed": self.failed,
                "fallback": self.fallback,
                "input": self.input,
                "output": self.output,
                "events": self.events,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
            config=config,
        )
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(trace_path)
