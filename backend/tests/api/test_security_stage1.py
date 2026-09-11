"""通过真实 ASGI 入口验证公网基础防护，不调用模型、数据库或外部服务。"""

from __future__ import annotations

import importlib
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from fastapi import Depends, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from auth.api_key_middleware import ApiKeySettings, get_allowed_origins, get_valid_api_keys
from core.errors import AppError, ErrorCode, sanitize_detail
from utils import logging_utils
from utils.secret_redaction import StreamingSecretRedactor, install_log_redaction, redact_sensitive_value, redact_text


TEST_KEY = "stage1-test-" + "a" * 40
ROTATED_KEY = "stage1-test-" + "b" * 40
PROVIDER_KEY = "sk-provider-" + "c" * 36


class ProbeBody(BaseModel):
    count: int


@pytest.fixture
def security_app(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv("BACKEND_API_KEYS", f"{TEST_KEY},{ROTATED_KEY}")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://research.example")
    monkeypatch.setenv("ALIYUN_API_KEY", PROVIDER_KEY)
    main = importlib.import_module("main")
    app = main.create_app(load_mode="lazy", enable_debug_routes=True)
    calls = []

    def costly_dependency():
        calls.append("called")

    @app.post("/api/security-probe", dependencies=[Depends(costly_dependency)])
    async def probe(body: ProbeBody):
        return {"count": body.count, "sdk_diagnostic": PROVIDER_KEY}

    @app.get("/api/security-failure/{kind}")
    async def failure(kind: str):
        if kind == "http":
            raise HTTPException(status_code=503, detail=f"SDK failed: {PROVIDER_KEY}")
        if kind == "app":
            raise AppError(ErrorCode.LLM_GENERATION_FAILED, detail=PROVIDER_KEY, context={"qa_observation": {"reason": PROVIDER_KEY}})
        if kind == "challenge":
            raise HTTPException(status_code=401, detail={"code": "missing_api_key"}, headers={"WWW-Authenticate": "ApiKey"})
        raise RuntimeError(f"SDK failed: {PROVIDER_KEY}")

    app.state.security_test_calls = calls
    return app


@pytest.fixture
def client(security_app):
    return TestClient(security_app, raise_server_exceptions=False)


@pytest.mark.parametrize("raw", [None, "", " , , ", "too-short", "x" * 257, "密" * 40, "x" * 32 + " y"])
def test_invalid_keys_fail_closed_without_echoing_values(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("BACKEND_API_KEYS", raising=False)
    else:
        monkeypatch.setenv("BACKEND_API_KEYS", raw)
    with pytest.raises(RuntimeError) as error:
        get_valid_api_keys()
    if raw and raw.strip():
        assert raw not in str(error.value)


def test_config_rejects_reusing_the_provider_key(monkeypatch):
    monkeypatch.setenv("BACKEND_API_KEYS", PROVIDER_KEY)
    monkeypatch.setenv("ALIYUN_API_KEY", PROVIDER_KEY)
    with pytest.raises(RuntimeError):
        get_valid_api_keys()


@pytest.mark.parametrize("origin", ["", "*", "https://*.example", "null", "https://research.example/", "https://user:password@example.com", "https://example.com?key=secret", "https://example.com:bad"])
def test_invalid_origins_fail_closed(monkeypatch, origin):
    monkeypatch.setenv("ALLOWED_ORIGINS", origin)
    with pytest.raises(RuntimeError):
        get_allowed_origins()


def test_settings_deduplicate_keys_and_do_not_expose_them_in_repr(monkeypatch):
    monkeypatch.setenv("BACKEND_API_KEYS", f" {TEST_KEY},,{TEST_KEY},{ROTATED_KEY} ")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://research.example, https://research.example")
    settings = ApiKeySettings.from_environment()
    assert len(settings.key_hashes) == 2
    assert settings.allowed_origins == ("https://research.example",)
    assert TEST_KEY not in repr(settings)
    assert settings.accepts(TEST_KEY)
    assert settings.accepts(ROTATED_KEY)
    assert not settings.accepts("密" * 40)


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/arxiv/fields"), ("GET", "/api/papers"), ("DELETE", "/api/paper/2401.00001"),
    ("GET", "/api/user/preferences/u1"), ("POST", "/api/agent/chat/stream"),
    ("POST", "/api/agent/work-continuations/c1/resume/stream"),
    ("POST", "/api/paper/2401.00001/qa/stream"),
    ("GET", "/api/paper/2401.00001/evidence-assets/figure-1"),
    ("GET", "/api/paper/2401.00001/notes/export"), ("GET", "/api/paper/2401.00001/qa-trace/latest"),
    ("GET", "/api/debug/chunks/files"), ("GET", "/health/extra"),
])
def test_all_api_entry_types_require_authentication(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 401
    assert response.json()["code"] == "missing_api_key"
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_authentication_precedes_json_parsing_and_service_dependencies(client, security_app):
    response = client.post("/api/security-probe", content="{not-json", headers={"Content-Type": "application/json"})
    assert response.status_code == 401
    assert security_app.state.security_test_calls == []


@pytest.mark.parametrize("key", [TEST_KEY, ROTATED_KEY])
def test_valid_keys_reach_the_protected_endpoint(client, key):
    response = client.get("/api/auth/check", headers={"X-API-Key": key})
    assert response.status_code == 200
    assert response.json() == {"status": "authenticated"}
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_invalid_duplicate_and_query_credentials_are_rejected(client):
    for headers in [{"X-API-Key": "wrong"}, [("X-API-Key", TEST_KEY), ("X-API-Key", ROTATED_KEY)], [(b"X-API-Key", b"\xff" * 40)]]:
        response = client.get("/api/auth/check", headers=headers)
        assert response.status_code == 403
        assert response.json()["code"] == "invalid_api_key"
    assert client.get("/api/auth/check", params={"api_key": TEST_KEY}).status_code == 401


def test_health_is_public_and_documentation_is_not_published(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "arxiv-research-backend", "version": "1.0.0"}
    for path in ["/docs", "/redoc", "/openapi.json"]:
        assert client.get(path, headers={"X-API-Key": TEST_KEY}).status_code == 404


def test_cors_preflight_and_authentication_errors(client):
    preflight = {"Origin": "https://research.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-API-Key, Content-Type"}
    response = client.options("/api/agent/chat/stream", headers=preflight)
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://research.example"
    assert "x-api-key" in response.headers["Access-Control-Allow-Headers"].lower()
    denied = client.options("/api/agent/chat/stream", headers={**preflight, "Origin": "https://evil.example"})
    assert denied.status_code == 400
    assert "Access-Control-Allow-Origin" not in denied.headers
    missing = client.get("/api/auth/check", headers={"Origin": "https://research.example"})
    assert missing.status_code == 401
    assert missing.headers["Access-Control-Allow-Origin"] == "https://research.example"
    # CORS 不能代替认证：伪造一个允许的 Origin 仍然无法调用业务接口。
    assert client.get("/api/auth/check", headers={"Origin": "https://evil.example"}).status_code == 401
    assert client.options("/api/auth/check").status_code == 401


def test_app_instances_keep_independent_credentials(client, monkeypatch):
    monkeypatch.setenv("BACKEND_API_KEYS", ROTATED_KEY)
    second = TestClient(importlib.import_module("main").create_app(load_mode="lazy"))
    assert client.get("/api/auth/check", headers={"X-API-Key": TEST_KEY}).status_code == 200
    assert second.get("/api/auth/check", headers={"X-API-Key": TEST_KEY}).status_code == 403
    monkeypatch.delenv("BACKEND_API_KEYS")
    with pytest.raises(RuntimeError):
        importlib.import_module("main").create_app(load_mode="lazy")


def test_validation_does_not_echo_request_values(client):
    response = client.post("/api/security-probe", json={"count": "unconfigured-private-input"}, headers={"X-API-Key": TEST_KEY})
    assert response.status_code == 422
    assert "unconfigured-private-input" not in response.text
    assert "count" in response.json()["detail"]


@pytest.mark.parametrize("kind,status", [("http", 503), ("app", 502), ("unknown", 500)])
def test_errors_never_expose_provider_credentials(client, kind, status):
    response = client.get(f"/api/security-failure/{kind}", headers={"X-API-Key": TEST_KEY})
    assert response.status_code == status
    assert PROVIDER_KEY not in response.text
    assert response.json()["status"] == "failed"


@pytest.mark.parametrize("origin", ["https://research.example", "https://evil.example", None])
def test_unhandled_errors_keep_security_headers_and_origin_policy(client, origin):
    headers = {"X-API-Key": TEST_KEY}
    if origin:
        headers["Origin"] = origin
    response = client.get("/api/security-failure/unknown", headers=headers)
    assert response.status_code == 500
    assert response.json()["code"] == ErrorCode.UNKNOWN_ERROR
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert PROVIDER_KEY not in response.text
    if origin == "https://research.example":
        assert response.headers["Access-Control-Allow-Origin"] == origin
        assert "Origin" in response.headers["Vary"]
    else:
        assert "Access-Control-Allow-Origin" not in response.headers


def test_normal_json_responses_and_challenge_headers_are_protected(client):
    response = client.post("/api/security-probe", json={"count": 2}, headers={"X-API-Key": TEST_KEY})
    assert response.json()["count"] == 2
    assert PROVIDER_KEY not in response.text
    challenge = client.get("/api/security-failure/challenge", headers={"X-API-Key": TEST_KEY})
    assert challenge.status_code == 401
    assert challenge.headers["WWW-Authenticate"] == "ApiKey"


def test_redaction_covers_nested_headers_text_and_truncated_details(monkeypatch):
    monkeypatch.setenv("ALIYUN_API_KEY", PROVIDER_KEY)
    monkeypatch.setenv("BACKEND_API_KEYS", f"{TEST_KEY},{ROTATED_KEY}")
    monkeypatch.setenv("BACKEND_LOG_REDACT_SECRETS", "false")
    result = redact_sensitive_value({"X-API-Key": "unconfigured-private-key", "nested": [f"failed: {PROVIDER_KEY} {TEST_KEY} {ROTATED_KEY}"], "total_tokens": 300})
    rendered = json.dumps(result)
    for secret in ["unconfigured-private-key", PROVIDER_KEY, TEST_KEY, ROTATED_KEY]:
        assert secret not in rendered
    assert result["total_tokens"] == 300
    assert "bearer-secret" not in redact_text("Authorization: Bearer bearer-secret")
    assert PROVIDER_KEY[:12] not in sanitize_detail("x" * 490 + PROVIDER_KEY)
    assert "unconfigured-private-key" not in json.dumps(logging_utils.redact_log_value({"api_key": "unconfigured-private-key"}, config={"redact_secrets": False}))


def test_bm25_matched_words_survive_response_and_trace_redaction(tmp_path, monkeypatch):
    from core.responses import RedactedJSONResponse
    from services.retrieval.keyword_backend import build_keyword_matched_index_entry, build_keyword_matched_terms

    monkeypatch.setenv("ALIYUN_API_KEY", PROVIDER_KEY)
    term_traces = {
        "dataset": {"token": "dataset", "idf": 1.2, "best_term_score": 2.3, "fields": ["body"], "query_sources": ["original"]},
    }
    matched_terms = build_keyword_matched_terms(term_traces)
    index_entry = build_keyword_matched_index_entry(SimpleNamespace(), {"term_traces": term_traces})
    payload = {
        "retrieval_debug": {"keyword_matched_terms": matched_terms},
        "keyword_debug": {"matched_chunks": [index_entry]},
        "auth": {"token": "unconfigured-token", "access_token": "private-access", "refresh_token": "private-refresh"},
    }
    response = json.loads(RedactedJSONResponse(payload).body)
    assert response["retrieval_debug"]["keyword_matched_terms"] == matched_terms
    assert response["keyword_debug"]["matched_chunks"][0]["matched_terms"] == matched_terms
    assert set(response["auth"].values()) == {"***REDACTED***"}

    # 只豁免检索词的字段名；真正的密钥即使混入检索结果，也必须按值脱敏。
    sensitive_terms = {"keyword_matched_terms": [{**matched_terms[0], "token": PROVIDER_KEY, "auth": {"token": "nested-secret"}}]}
    safe_terms = redact_sensitive_value(sensitive_terms)["keyword_matched_terms"][0]
    assert safe_terms["token"] == "***REDACTED***"
    assert safe_terms["auth"]["token"] == "***REDACTED***"

    monkeypatch.setattr(logging_utils, "_runtime_config", lambda: {"request_trace": "always", "request_trace_dir": str(tmp_path)})
    trace = logging_utils.RequestTrace(run_id="bm25-security-test", route="/api/probe", input={})
    trace.set_output(payload)
    written = json.loads(Path(trace.write()).read_text(encoding="utf-8"))
    assert written["output"]["retrieval_debug"]["keyword_matched_terms"] == matched_terms
    assert written["output"]["keyword_debug"]["matched_chunks"][0]["matched_terms"] == matched_terms
    assert "unconfigured-token" not in json.dumps(written)


@pytest.mark.parametrize("text", [
    "Normal words and 中文回答。",
    "Provider: sk-unconfigured-provider-value; finished.",
    "Bearer unknown.header/payload+signature==; finished.",
    "Authorization: Bearer unknown.header.signature; finished.",
    "MY_API_KEY = \"unknown credential with spaces\"; finished.",
    "{'refresh_token': 'unknown token value'}; finished.",
    "password=unknown-value&next=public",
])
def test_streaming_redaction_covers_unknown_credential_syntax_at_every_boundary(text):
    # 穷举二分边界并测试逐字符分片，验证凭据语法与值分开到达时仍与整段脱敏一致。
    chunkings = [[text[:index], text[index:]] for index in range(1, len(text))]
    chunkings.append(list(text))
    for chunks in chunkings:
        redactor = StreamingSecretRedactor()
        actual = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
        assert actual == redact_text(text)


def test_streaming_redaction_handles_encoded_keys_and_preserves_live_text(monkeypatch):
    secret = "provider+/=" + "z" * 32
    monkeypatch.setenv("ALIYUN_API_KEY", secret)
    redactor = StreamingSecretRedactor()
    assert redactor.feed("这段普通回答应立即显示。") == "这段普通回答应立即显示。"
    encoded = quote(secret, safe="")
    actual = "".join(redactor.feed(char) for char in encoded) + redactor.finish()
    assert actual == "***REDACTED***"


@pytest.mark.parametrize("text", [
    r'password="unknown \"quoted\" credential"; done',
    r"password='unknown \'quoted\' credential'; done",
])
def test_quoted_credentials_remain_hidden_when_the_value_contains_escaped_quotes(text):
    expected = "password=***REDACTED***; done"
    assert redact_text(text) == expected
    # 转义符和引号可能处在相邻 SSE 事件中，不能把后半段凭据当作普通答案提前发送。
    redactor = StreamingSecretRedactor()
    assert "".join(redactor.feed(char) for char in text) + redactor.finish() == expected


@pytest.mark.parametrize("case", ["plain", "repeated_field", "jwt_prefixes", "stream", "stream_quoted", "stream_spaces"])
def test_redaction_long_input_has_bounded_processing_cost(case):
    # 在子进程中重放攻击形状；即使重新引入灾难性回溯，也只失败当前用例而不挂住整套回归。
    probe = """
import sys
sys.path.insert(0, 'backend')
from utils.secret_redaction import StreamingSecretRedactor, redact_text
case = sys.argv[1]
text = {'repeated_field': 'password' * 8192, 'jwt_prefixes': 'eyJa-' * 13107,
        'stream_quoted': 'password="' + ('quoted value ' * 5462) + '"; done',
        'stream_spaces': 'password' + (' ' * 65536) + '=hidden-value; done'}.get(case, 'a' * 65536)
if case.startswith('stream'):
    redactor = StreamingSecretRedactor()
    result = ''.join(redactor.feed(text[pos:pos + 64]) for pos in range(0, len(text), 64)) + redactor.finish()
else:
    result = redact_text(text)
assert result == redact_text(text)
if case not in {'stream_quoted', 'stream_spaces'}:
    assert result == text
assert 'hidden-value' not in redact_text(text + '_api_key=hidden-value; done')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", probe, case],
            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=8,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{case}: 64 KiB 文本脱敏超过 8 秒，疑似回溯或重复扫描未决分片。")
    assert result.returncode == 0, result.stderr


def test_plain_logging_and_tracebacks_are_redacted(monkeypatch):
    monkeypatch.setenv("ALIYUN_API_KEY", PROVIDER_KEY)
    install_log_redaction()
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger = logging.getLogger("security-test-sdk")
    logger.addHandler(handler)
    try:
        try:
            raise RuntimeError(PROVIDER_KEY)
        except RuntimeError:
            logger.exception("provider error: %s", {"api_key": "unconfigured-private-key"})
    finally:
        logger.removeHandler(handler)
    assert PROVIDER_KEY not in output.getvalue()
    assert "unconfigured-private-key" not in output.getvalue()
    assert "RuntimeError" in output.getvalue()


def test_request_trace_is_redacted_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIYUN_API_KEY", PROVIDER_KEY)
    monkeypatch.setattr(logging_utils, "_runtime_config", lambda: {"request_trace": "always", "request_trace_dir": str(tmp_path), "redact_secrets": False})
    trace = logging_utils.RequestTrace(run_id="security-test", route="/api/probe", input={"X-API-Key": TEST_KEY})
    trace.set_output({"error": PROVIDER_KEY})
    path = Path(trace.write())
    text = path.read_text(encoding="utf-8")
    assert TEST_KEY not in text
    assert PROVIDER_KEY not in text
    assert json.loads(text)["run_id"] == "security-test"


def test_uvicorn_access_formatter_keeps_its_argument_contract(monkeypatch):
    from uvicorn.logging import AccessFormatter

    monkeypatch.setenv("BACKEND_API_KEYS", TEST_KEY)
    install_log_redaction()
    record = logging.getLogger("uvicorn.access").makeRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", "GET", f"/api/probe?api_key={TEST_KEY}", "1.1", 401), None,
    )
    rendered = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(record)
    assert TEST_KEY not in rendered
    assert "401" in rendered


@pytest.mark.parametrize("format", ["md", "json"])
def test_historical_trace_downloads_are_redacted_without_rewriting_files(client, security_app, tmp_path, format):
    from routers import qa_router

    security_app.dependency_overrides[qa_router.get_enhanced_retrieval_service] = lambda: SimpleNamespace(trace_export_dir=tmp_path)
    paper_dir = tmp_path / qa_router.sanitize_trace_slug("2401.00001")
    paper_dir.mkdir()
    raw_text = json.dumps({"error": PROVIDER_KEY, "api_key": TEST_KEY}) if format == "json" else f"# Error\n{PROVIDER_KEY}"
    trace_file = paper_dir / f"old-trace.{format}"
    trace_file.write_text(raw_text, encoding="utf-8")
    response = client.get("/api/paper/2401.00001/qa-trace/latest", params={"format": format, "trace_name": trace_file.name}, headers={"X-API-Key": TEST_KEY})
    assert response.status_code == 200
    assert PROVIDER_KEY not in response.text
    assert TEST_KEY not in response.text
    assert "attachment" in response.headers["Content-Disposition"]
    if format == "json":
        assert isinstance(response.json(), dict)
    assert trace_file.read_text(encoding="utf-8") == raw_text
