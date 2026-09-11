"""日志、错误响应与 trace 共用的密钥脱敏，禁止通过调试开关关闭。"""

from __future__ import annotations

import logging
import os
import re
from threading import RLock
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, quote_plus


REDACTED = "***REDACTED***"
_registered_secrets: set[str] = set()
_registered_secrets_lock = RLock()
_CREDENTIAL_FIELD_PATTERN = (
    r"api[_-]?keys?|authorization|cookies?|password|passwd|secret|access[_-]?token|refresh[_-]?token"
)
_SECRET_FIELD_RE = re.compile(rf"(?:{_CREDENTIAL_FIELD_PATTERN}|^token$)", re.IGNORECASE)
_CREDENTIAL_FIELD_RE = re.compile(_CREDENTIAL_FIELD_PATTERN, re.IGNORECASE)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_PROVIDER_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
# JWT 可能作为普通文本被模型或 SDK 回显，不只出现在 Authorization 字段中。
_JWT_SHAPE_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?P<header>[A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
)
_JWT_START_RE = re.compile(r"\beyJ[A-Za-z0-9_-]")
_TRAILING_BEARER_RE = re.compile(r"\bBearer\Z", re.IGNORECASE)
_VALUE_DELIMITERS = frozenset(",;}]&")


def _is_field_char(char: str) -> bool:
    return char.isalnum() or char in "_-"


class _AssignmentRedactor:
    """单向扫描字段和值；长字段只判断一次，未闭合的凭据值直接丢弃。"""

    def __init__(self) -> None:
        self._state = "text"
        self._pending: list[str] = []
        self._allow_quote = False
        self._quote = ""
        self._escaped = False
        self._bearer_size = -1

    def feed(self, text: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(text):
            char = text[index]
            if self._state == "text":
                if _is_field_char(char):
                    self._pending.append(char)
                    self._state = "word"
                else:
                    output.append(char)
            elif self._state == "word":
                if _is_field_char(char):
                    self._pending.append(char)
                else:
                    # 不在长词内部反复尝试“任意前缀 + 敏感词 + 任意后缀”，避免匿名输入耗尽 CPU。
                    word = "".join(self._pending)
                    self._pending = [word]
                    if _CREDENTIAL_FIELD_RE.search(word):
                        self._state = "head"
                        self._allow_quote = True
                    else:
                        output.append(word)
                        self._pending.clear()
                        self._state = "text"
                    continue
            elif self._state == "head":
                if char in ":=":
                    output.extend(self._pending)
                    output.append(char)
                    self._pending.clear()
                    self._state = "value_start"
                elif char.isspace() or (self._allow_quote and char in "\"'"):
                    self._pending.append(char)
                    self._allow_quote = False
                else:
                    output.extend(self._pending)
                    self._pending.clear()
                    self._state = "text"
                    continue
            elif self._state == "value_start":
                if char.isspace():
                    output.append(char)
                elif char in _VALUE_DELIMITERS:
                    self._state = "text"
                    continue
                else:
                    output.append(REDACTED)
                    if char in "\"'":
                        self._quote = char
                        self._escaped = False
                        self._state = "quoted_value"
                    else:
                        self._bearer_size = 1 if char.lower() == "b" else -1
                        self._state = "value"
            elif self._state == "quoted_value":
                # 一旦确认凭据字段，不保留原值；引号跨分片或最终未闭合也不能回退输出原文。
                if char in "\r\n":
                    self._escaped = False
                    self._state = "text"
                    continue
                elif self._escaped:
                    self._escaped = False
                elif char == "\\":
                    # JSON/SDK 表示中的转义引号不是字段结束，且转义状态需要跨 SSE 分片保留。
                    self._escaped = True
                elif char == self._quote:
                    self._state = "text"
            elif self._state == "value":
                if char.isspace() and self._bearer_size == 6:
                    self._pending.append(char)
                    self._state = "bearer_space"
                elif char.isspace() or char in _VALUE_DELIMITERS:
                    self._state = "text"
                    continue
                elif 0 <= self._bearer_size < 6 and char.lower() == "bearer"[self._bearer_size]:
                    self._bearer_size += 1
                else:
                    self._bearer_size = -1
            elif self._state == "bearer_space":
                if char.isspace():
                    self._pending.append(char)
                elif char in _VALUE_DELIMITERS:
                    output.extend(self._pending)
                    self._pending.clear()
                    self._state = "text"
                    continue
                else:
                    # Authorization: Bearer <值> 的空白属于凭据；没有后续值时在 finish 中保留普通空白。
                    self._pending.clear()
                    self._bearer_size = -1
                    self._state = "value"
            index += 1
        return "".join(output)

    def finish(self) -> str:
        tail = "".join(self._pending)
        self.__init__()
        return tail

    def discard(self) -> None:
        self.__init__()


def _redact_bare_credentials(text: str) -> str:
    text = _BEARER_RE.sub("Bearer " + REDACTED, text)
    parts: list[str] = []
    cursor = position = 0
    while match := _JWT_SHAPE_RE.search(text, position):
        # 先扫描完整的 base64url 段，再查 eyJ 起点；重复 eyJ- 串不能触发逐后缀回溯。
        start = _JWT_START_RE.search(text, match.start("header"), match.end("header"))
        if start:
            parts.extend((text[cursor:start.start()], REDACTED))
            cursor = position = match.end()
        else:
            # 第一个段不是 JWT 时仅滑动一段，保留 a.eyJxxx.payload.signature 的识别能力。
            position = match.end("header") + 1
    parts.append(text[cursor:])
    return _PROVIDER_KEY_RE.sub(REDACTED, "".join(parts))


class _CredentialTokenRedactor:
    """完整普通词只扫描一次；Bearer 的空白前缀最多与后一个词合并。"""

    def __init__(self) -> None:
        self._pending: list[str] = []
        self._in_token = False
        self._awaiting_token = False
        self._joined_bearer = False

    def feed(self, text: str) -> str:
        output: list[str] = []
        for char in text:
            if char.isalnum() or char in "._~+/=-":
                if self._awaiting_token:
                    self._awaiting_token = False
                    self._joined_bearer = True
                self._pending.append(char)
                self._in_token = True
                continue
            if self._in_token:
                token = "".join(self._pending)
                if char.isspace() and not self._joined_bearer and _TRAILING_BEARER_RE.search(token):
                    self._pending = [token, char]
                    self._in_token = False
                    self._awaiting_token = True
                    continue
                output.append(_redact_bare_credentials(token))
                self.__init__()
            elif self._awaiting_token:
                if char.isspace():
                    self._pending.append(char)
                    continue
                output.append(_redact_bare_credentials("".join(self._pending)))
                self.__init__()
            output.append(char)
        return "".join(output)

    def finish(self) -> str:
        tail = _redact_bare_credentials("".join(self._pending))
        self.__init__()
        return tail

    def discard(self) -> None:
        self.__init__()


class _LiteralRedactor:
    """只保留已配置密钥的有限长度前缀，不随普通长词增长反复扫描整个尾部。"""

    def __init__(self, variants: tuple[str, ...]) -> None:
        self._pattern = re.compile("|".join(re.escape(secret) for secret in variants)) if variants else None
        self._prefixes = {secret[:size] for secret in variants for size in range(1, len(secret))}
        self._max_prefix_length = max(map(len, self._prefixes), default=0)
        self._pending = ""

    def feed(self, delta: str) -> str:
        text = self._pending + delta
        if self._pattern is not None:
            # 先消除完整匹配，避免重复前缀的密钥被拆分成已发送的原文与未决尾部。
            text = self._pattern.sub(lambda _match: REDACTED, text)
        safe_end = len(text)
        for size in range(min(len(text), self._max_prefix_length), 0, -1):
            if text[-size:] in self._prefixes:
                safe_end -= size
                break
        self._pending = text[safe_end:]
        return text[:safe_end]

    def finish(self) -> str:
        tail, self._pending = self._pending, ""
        return tail

    def discard(self) -> None:
        self._pending = ""


def is_secret_field(name: Any) -> bool:
    return bool(_SECRET_FIELD_RE.search(str(name)))


def register_sensitive_values(values) -> None:
    """补充从私有配置文件加载的凭据；旧凭据继续脱敏，避免轮换后历史错误重新泄漏。"""
    with _registered_secrets_lock:
        _registered_secrets.update(value for value in values if isinstance(value, str) and value)


def _configured_secret_variants() -> tuple[str, ...]:
    variants: set[str] = set()
    with _registered_secrets_lock:
        for secret in _registered_secrets:
            variants.update((secret, quote(secret, safe=""), quote_plus(secret)))
    for name, raw_value in os.environ.items():
        if not name.upper().endswith(("_API_KEY", "_API_KEYS", "_TOKEN", "_PASSWORD", "_SECRET", "_SECRET_KEY", "_ACCESS_KEY")) or not raw_value:
            continue
        values = raw_value.split(",") if name == "BACKEND_API_KEYS" else [raw_value]
        for item in values:
            secret = item.strip()
            if secret:
                # SDK 可能将密钥放进 URL；同时覆盖原文和百分号编码，避免换一种表示就绕过脱敏。
                variants.update((secret, quote(secret, safe=""), quote_plus(secret)))
    return tuple(sorted(variants, key=len, reverse=True))


def redact_text(value: str) -> str:
    """先按真实配置值脱敏，再处理未知凭据；截断必须在脱敏之后执行。"""
    for secret in _configured_secret_variants():
        value = value.replace(secret, REDACTED)
    assignments = _AssignmentRedactor()
    return _redact_bare_credentials(assignments.feed(value) + assignments.finish())


def redact_sensitive_value(value: Any, *, _parent_field: str | None = None) -> Any:
    """保留业务数据结构与数值，仅移除凭据字段和文本中的密钥。"""
    if isinstance(value, Mapping):
        # BM25 的 token 表示匹配词；仅在已知匹配词结构内豁免字段名，值本身仍按密钥规则处理。
        is_matched_term = (
            _parent_field in {"keyword_matched_terms", "matched_terms"}
            and {"idf", "term_score", "fields", "query_sources"}.issubset(value)
        )
        return {
            redact_text(str(key)): (
                REDACTED if is_secret_field(key) and not (is_matched_term and key == "token")
                else redact_sensitive_value(item, _parent_field=str(key))
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_value(item, _parent_field=_parent_field) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return redact_text(str(value))


class StreamingSecretRedactor:
    """每条答案流独享未决尾部，避免逐事件脱敏后仍能拼出完整凭据。"""

    def __init__(self) -> None:
        self._stages = (_LiteralRedactor(_configured_secret_variants()), _AssignmentRedactor(), _CredentialTokenRedactor())

    def feed(self, delta: str) -> str:
        # 已确定的文本单向通过各层，未决词采用列表追加；不会随 SSE 分片重复扫描或复制增长中的长词。
        for stage in self._stages:
            delta = stage.feed(delta)
        return delta

    def finish(self) -> str:
        """仅在正常结束时发送已脱敏的尾部，保证普通答案不会丢字。"""
        output: list[str] = []
        for index, stage in enumerate(self._stages):
            tail = stage.finish()
            # 上游普通尾部仍必须经过下游凭据识别，不能在正常结束时旁路脱敏。
            for remaining in self._stages[index + 1:]:
                tail = remaining.feed(tail)
            output.append(tail)
        return "".join(output)

    def discard(self) -> None:
        """异常或断开连接时放弃未决片段，清理阶段不得重新向客户端发送原文。"""
        for stage in self._stages:
            stage.discard()


class _RedactedLogRecord(logging.LogRecord):
    def getMessage(self) -> str:
        # 在格式化后处理消息，避免把模板里的 %s 当成密钥值替换而破坏日志插值。
        return redact_text(super().getMessage())


def install_log_redaction() -> None:
    """在 LogRecord 出口脱敏，覆盖业务、SDK 以及随后创建的 Uvicorn handler。"""
    previous_factory = logging.getLogRecordFactory()
    if getattr(previous_factory, "_redacts_backend_secrets", False):
        return

    def safe_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        original = previous_factory(*args, **kwargs)
        record = _RedactedLogRecord.__new__(_RedactedLogRecord)
        record.__dict__.update(original.__dict__)
        # 保留参数的元组/字典形状；Uvicorn AccessFormatter 会解包 args 构造访问日志。
        if isinstance(record.args, tuple):
            record.args = tuple(redact_sensitive_value(item) for item in record.args)
        elif isinstance(record.args, Mapping):
            record.args = redact_sensitive_value(record.args)
        if record.exc_info:
            # traceback 不属于 msg；提前生成安全文本，防止标准 Formatter 再输出原始异常。
            record.exc_text = redact_text(logging.Formatter().formatException(record.exc_info))
        if record.stack_info:
            record.stack_info = redact_text(record.stack_info)
        return record

    safe_record_factory._redacts_backend_secrets = True
    logging.setLogRecordFactory(safe_record_factory)
