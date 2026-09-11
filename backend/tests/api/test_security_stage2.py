"""以真实 ASGI 入口和原子计数后端验证第二阶段防护，不调用外部模型或数据库。"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import fakeredis
import limits.storage.memory as memory_storage
import pytest
import redis
from fastapi import Depends
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from auth.api_key_middleware import ApiKeySettings
from middleware.audit_log import AuditLogMiddleware, AuditSink
from middleware.ip_filter import IPFilterSettings
from middleware.rate_limit import RateLimitController, RateLimitSettings
from utils.secret_redaction import StreamingSecretRedactor, redact_sensitive_value


KEY_A = "stage2-shared-prefix-" + "a" * 40
KEY_B = "stage2-shared-prefix-" + "b" * 40


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    settings = {
        "AUTH_MODE": "api_key",
        "BACKEND_API_KEYS": f"{KEY_A},{KEY_B}", "API_KEY_CONFIG_FILE": "", "ALLOWED_ORIGINS": "https://research.example",
        "RATE_LIMIT_STORAGE": "memory://", "IP_FILTER_MODE": "disabled", "IP_BLACKLIST": "", "IP_WHITELIST": "", "TRUSTED_PROXY_IPS": "",
        "AUDIT_LOG_ENABLED": "true", "AUDIT_LOG_FILE": str(tmp_path / "audit.log"),
        "AUDIT_LOG_MAX_BYTES": "10485760", "AUDIT_LOG_BACKUP_COUNT": "5",
        "RATE_LIMIT_IP": "10000/minute", "RATE_LIMIT_KEY": "10000/minute", "RATE_LIMIT_READ": "10000/minute",
        "RATE_LIMIT_WRITE": "10000/minute", "RATE_LIMIT_EXPENSIVE": "10000/minute",
        "AUTH_FAILURE_LIMIT": "10000/minute", "RATE_LIMIT_VIOLATION_LIMIT": "10000/minute", "ABUSE_BLOCK_SECONDS": "90",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)

    def factory(**overrides):
        for name, value in overrides.items():
            monkeypatch.setenv(name, str(value))
        app = importlib.import_module("main").create_app(load_mode="lazy")
        app.state.probe_calls = []

        def costly_dependency():
            app.state.probe_calls.append("dependency")

        @app.post("/api/security-probe", dependencies=[Depends(costly_dependency)])
        async def probe(body: dict):
            return {"ok": True}

        @app.get("/api/security-failure")
        async def failure():
            raise RuntimeError("private failure")

        @app.get("/api/security-stream")
        async def stream():
            return StreamingResponse(iter(["data: one\n\n", "data: two\n\n"]), media_type="text/event-stream")

        return app

    return factory


def client(app, ip="198.51.100.10", key=KEY_A):
    return TestClient(app, client=(ip, 12345), headers={"X-API-Key": key} if key else {}, raise_server_exceptions=False)


def write_keys(tmp_path, monkeypatch, records):
    path = tmp_path / "private-keys.json"
    path.write_text(json.dumps({"keys": records}), encoding="utf-8")
    monkeypatch.setenv("API_KEY_CONFIG_FILE", str(path))
    return path


def test_ip_budget_cannot_be_reset_by_changing_invalid_keys(app_factory):
    app = app_factory(RATE_LIMIT_IP="2/minute")
    http = client(app, key=None)
    assert http.get("/api/auth/check").status_code == 401
    assert http.get("/api/auth/check", headers={"X-API-Key": "unknown-1"}).status_code == 403
    denied = http.post("/api/security-probe", content="{invalid", headers={"X-API-Key": KEY_A, "Origin": "https://research.example"})
    assert denied.status_code == 429
    assert denied.json()["code"] == "rate_limit_exceeded"
    assert 1 <= int(denied.headers["Retry-After"]) <= 60
    assert denied.json()["retry_after"] == int(denied.headers["Retry-After"])
    assert denied.headers["Access-Control-Allow-Origin"] == "https://research.example"
    assert denied.headers["Cache-Control"] == "no-store"
    assert app.state.probe_calls == []
    assert client(app, "198.51.100.11").get("/api/auth/check").status_code == 200


def test_per_second_burst_budget_applies_with_minute_budget(app_factory):
    http = client(app_factory(RATE_LIMIT_IP="100/minute;1/second"))
    assert http.get("/api/auth/check").status_code == 200
    result = http.get("/api/auth/check")
    assert result.status_code == 429
    assert result.headers["Retry-After"] == "1"


def test_key_budgets_are_shared_across_ips_but_not_key_prefixes(app_factory):
    app = app_factory(RATE_LIMIT_KEY="1/minute")
    assert client(app).get("/api/auth/check").status_code == 200
    assert client(app, "198.51.100.11").get("/api/auth/check").status_code == 429
    assert client(app, key=KEY_B).get("/api/auth/check").status_code == 200


def test_write_and_expensive_budgets_are_shared_across_real_routes(app_factory):
    app = app_factory(RATE_LIMIT_WRITE="1/minute", RATE_LIMIT_EXPENSIVE="2/minute")
    http = client(app)
    assert http.post("/api/security-probe", json={}).status_code == 200
    assert http.post("/api/security-probe", json={}).status_code == 429
    assert http.get("/api/auth/check").status_code == 200
    # 无效 JSON 不会启动真实模型；准入之后的校验失败仍算一次已接纳的昂贵请求。
    bad_json = {"content": "{broken", "headers": {"Content-Type": "application/json"}}
    assert http.post("/api/agent/chat/stream", **bad_json).status_code == 422
    assert http.post("/api/paper/2401.00001/qa/stream", **bad_json).status_code == 422
    for path in ("/api/paper/2401.00002/create-qa-index", "/api/agent/work-continuations/c1/resume/stream", "/api/user/research-profile/rebuild"):
        response = http.post(path, **bad_json)
        assert response.status_code == 429
        assert response.headers["X-RateLimit-Scope"] == "expensive"
    assert app.state.probe_calls == ["dependency"]


def test_health_and_preflight_do_not_consume_budgets(app_factory):
    app = app_factory(RATE_LIMIT_IP="1/minute", RATE_LIMIT_KEY="1/minute")
    http = client(app, key=None)
    for _ in range(3):
        assert http.get("/health").status_code == 200
        assert http.options("/api/agent/chat/stream", headers={"Origin": "https://research.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-API-Key"}).status_code == 200
    assert client(app).get("/api/auth/check").status_code == 200
    assert client(app).get("/api/auth/check").status_code == 429


def test_paper_backfill_and_preference_routes_cannot_bypass_expensive_budget(app_factory):
    app = app_factory(RATE_LIMIT_EXPENSIVE="1/minute")
    calls = []
    def materialization_dependency():
        calls.append("materialization")
        raise AssertionError("昂贵预算耗尽后不能初始化论文物化依赖")
    from routers import paper_router, user_router
    app.dependency_overrides[paper_router.get_recommendation_service] = materialization_dependency
    app.dependency_overrides[user_router.get_recommendation_service] = materialization_dependency
    http = client(app)
    # 创建论文的入口声明了 JSON 请求体；用解析失败耗尽预算，不初始化真实 embedding 或文档服务。
    assert http.post("/api/paper", content="{broken", headers={"Content-Type": "application/json"}).status_code == 422
    operations = [
        ("GET", "/api/paper/2401.00002"), ("GET", "/api/paper/2401.00003"),
        ("POST", "/api/user/like-paper"), ("POST", "/api/user/dislike-paper"), ("POST", "/api/user/paper-action"),
    ]
    for method, path in operations:
        response = http.request(method, path, json={"arxiv_id": "2401.00004", "action_type": "view"})
        assert response.status_code == 429
        assert response.headers["X-RateLimit-Scope"] == "expensive"
    assert calls == []
    assert http.get("/api/auth/check").status_code == 200


@pytest.mark.parametrize("metadata,code", [({"enabled": False}, "api_key_disabled"), ({"expires_at": "2020-01-01"}, "api_key_expired")])
def test_disabled_and_expired_file_keys_cannot_fall_back_to_environment(app_factory, monkeypatch, tmp_path, metadata, code):
    write_keys(tmp_path, monkeypatch, [{"key": KEY_A, **metadata}])
    app = app_factory()
    response = client(app).get("/api/auth/check")
    assert response.status_code == 403
    assert response.json()["code"] == code
    assert client(app, key=KEY_B).get("/api/auth/check").status_code == 403


def test_expiry_is_rechecked_on_each_request(app_factory, monkeypatch, tmp_path):
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    write_keys(tmp_path, monkeypatch, [{"key": KEY_A, "expires_at": expiry.isoformat()}])
    app = app_factory()
    assert client(app).get("/api/auth/check").status_code == 200
    key_config = importlib.import_module("auth.key_config")
    monkeypatch.setattr(key_config, "datetime", SimpleNamespace(now=lambda _tz: expiry + timedelta(seconds=1)))
    assert client(app).get("/api/auth/check").json()["code"] == "api_key_expired"


@pytest.mark.parametrize("override", [
    {"daily_quota": -1}, {"daily_quota": True}, {"enabled": "false"}, {"rate_limit": "0/minute"},
    {"expires_at": "bad-date"}, {"key": "short"}, {"key_env": "MISSING"},
])
def test_invalid_key_metadata_fails_closed(app_factory, monkeypatch, tmp_path, override):
    write_keys(tmp_path, monkeypatch, [{"key": KEY_A, **override}])
    with pytest.raises(RuntimeError) as error:
        app_factory()
    assert KEY_A not in str(error.value)


def test_explicit_missing_or_invalid_config_file_does_not_fall_back(app_factory, tmp_path):
    with pytest.raises(RuntimeError):
        app_factory(API_KEY_CONFIG_FILE=tmp_path / "missing.json")


@pytest.mark.parametrize("contents", ["{broken", "{}", '{"keys": []}', '{"keys": [null]}'])
def test_malformed_key_file_does_not_fall_back_to_valid_environment(app_factory, tmp_path, contents):
    path = tmp_path / "invalid-keys.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError):
        app_factory(API_KEY_CONFIG_FILE=path)


def test_duplicate_key_policies_are_rejected(app_factory, monkeypatch, tmp_path):
    write_keys(tmp_path, monkeypatch, [{"key": KEY_A}, {"key": KEY_A, "enabled": False}])
    with pytest.raises(RuntimeError, match="重复"):
        app_factory()


def test_file_key_env_reference_and_per_key_policies(app_factory, monkeypatch, tmp_path):
    monkeypatch.setenv("TEAM_ACCESS_KEY", KEY_A)
    write_keys(tmp_path, monkeypatch, [{"key_env": "TEAM_ACCESS_KEY", "name": "Team A", "rate_limit": "1/minute"}, {"key": KEY_B, "rate_limit": "2/minute"}])
    app = app_factory()
    assert client(app).get("/api/auth/check").status_code == 200
    assert client(app).get("/api/auth/check").status_code == 429
    assert client(app, key=KEY_B).get("/api/auth/check").status_code == 200
    assert client(app, key=KEY_B).get("/api/auth/check").status_code == 200
    assert client(app, key=KEY_B).get("/api/auth/check").status_code == 429


def test_daily_quota_is_atomic_and_resets_on_utc_day(app_factory, monkeypatch, tmp_path):
    write_keys(tmp_path, monkeypatch, [{"key": KEY_A, "daily_quota": 3}])
    app = app_factory()
    http = client(app)
    for _ in range(3):
        assert http.get("/api/auth/check").status_code == 200
    exhausted = http.get("/api/auth/check")
    assert exhausted.status_code == 429
    assert exhausted.json()["code"] == "daily_quota_exceeded"
    assert 0 < exhausted.json()["retry_after"] <= 86400
    controller = app.state.rate_limit_controller
    now = controller.clock()
    controller.clock = lambda: now + 86400
    assert http.get("/api/auth/check").status_code == 200


def test_concurrent_requests_do_not_overspend_daily_quota(app_factory, monkeypatch, tmp_path):
    write_keys(tmp_path, monkeypatch, [{"key": KEY_A, "daily_quota": 7}])
    app = app_factory()
    controller = app.state.rate_limit_controller
    policy = app.state.api_key_settings.match(KEY_A)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: controller.after_auth(policy, {"method": "GET", "security_route": "/api/auth/check"}), range(40)))
    assert sum(result.code is None for result in results) == 7


def test_redis_shares_key_quota_and_abuse_blocks_across_controllers(app_factory, monkeypatch, tmp_path):
    write_keys(tmp_path, monkeypatch, [{"key": KEY_A, "daily_quota": 1}])
    monkeypatch.setenv("RATE_LIMIT_STORAGE", "redis://localhost:6379/15")
    monkeypatch.setenv("AUTH_FAILURE_LIMIT", "1/minute")
    settings = RateLimitSettings.from_environment()
    server = fakeredis.FakeServer()
    pool = redis.ConnectionPool(connection_class=fakeredis.FakeRedisConnection, server=server)
    first = RateLimitController(settings, storage_options={"connection_pool": pool})
    second = RateLimitController(settings, storage_options={"connection_pool": pool})
    policy = ApiKeySettings.from_environment().match(KEY_A)
    scope = {"method": "GET", "security_route": "/api/auth/check"}
    assert first.after_auth(policy, scope).code is None
    assert second.after_auth(policy, scope).code == "daily_quota_exceeded"
    first.observe_rejection("198.51.100.10", "invalid_api_key")
    first.observe_rejection("198.51.100.10", "invalid_api_key")
    assert second.before_auth("198.51.100.10").code == "ip_temporarily_blocked"


@pytest.mark.parametrize("mode,networks,ip,allowed", [
    ("blacklist", "198.51.100.0/24", "198.51.100.10", False),
    ("whitelist", "198.51.100.0/24", "198.51.100.10", True),
    ("whitelist", "2001:db8::/32", "2001:db8::10", True),
    ("blacklist", "2001:db8::/32", "2001:db8::10", False),
    ("blacklist", "198.51.100.0/24", "::ffff:198.51.100.10", False),
    ("blacklist", "::ffff:198.51.100.0/120", "198.51.100.10", False),
    ("whitelist", "::ffff:198.51.100.10", "::ffff:198.51.100.10", True),
])
def test_ip_filters_support_ipv4_ipv6_and_cidr(app_factory, mode, networks, ip, allowed):
    app = app_factory(**{"IP_FILTER_MODE": mode, "IP_BLACKLIST" if mode == "blacklist" else "IP_WHITELIST": networks})
    http = client(app, ip)
    assert http.get("/api/auth/check").status_code == (200 if allowed else 403)
    assert http.get("/health").status_code == 200


@pytest.mark.parametrize("overrides", [{"IP_FILTER_MODE": "unknown"}, {"IP_FILTER_MODE": "whitelist"}, {"IP_BLACKLIST": "bad-ip"}, {"TRUSTED_PROXY_IPS": "*"}, {"RATE_LIMIT_IP": "unlimited"}, {"ABUSE_BLOCK_SECONDS": "0"}])
def test_invalid_security_configuration_cannot_silently_disable_controls(app_factory, overrides):
    with pytest.raises(RuntimeError):
        app_factory(**overrides)


def test_only_trusted_proxies_can_supply_client_ip(app_factory):
    app = app_factory(IP_FILTER_MODE="blacklist", IP_BLACKLIST="198.51.100.0/24", TRUSTED_PROXY_IPS="127.0.0.1,10.0.0.0/8")
    direct = client(app)
    assert direct.get("/api/auth/check", headers={"X-Forwarded-For": "203.0.113.10"}).status_code == 403
    proxy = client(app, "127.0.0.1")
    assert proxy.get("/api/auth/check", headers={"X-Forwarded-For": "203.0.113.10,198.51.100.10,10.0.0.2"}).status_code == 403
    assert proxy.get("/api/auth/check", headers={"X-Forwarded-For": "203.0.113.10"}).status_code == 200
    assert proxy.get("/api/auth/check", headers={"X-Forwarded-For": "invalid-ip"}).status_code == 403


@pytest.mark.parametrize("headers", [
    [("X-Forwarded-For", "203.0.113.10"), ("X-Forwarded-For", "198.51.100.10")],
    {"X-Forwarded-For": ",".join(["203.0.113.10"] * 21)},
    {"X-Forwarded-For": "203.0.113.10,"},
])
def test_ambiguous_forwarded_headers_are_rejected_before_body_parsing(app_factory, headers):
    app = app_factory(TRUSTED_PROXY_IPS="127.0.0.1")
    response = client(app, "127.0.0.1").post("/api/security-probe", content="{broken", headers=headers)
    assert response.status_code == 403
    assert response.json()["code"] == "ip_not_allowed"
    assert app.state.probe_calls == []


def test_blacklist_takes_priority_over_whitelist(app_factory):
    app = app_factory(IP_FILTER_MODE="whitelist", IP_WHITELIST="198.51.100.0/24", IP_BLACKLIST="198.51.100.10")
    assert client(app).get("/api/auth/check").json()["code"] == "ip_blocked"
    assert client(app, "198.51.100.11").get("/api/auth/check").status_code == 200


@pytest.mark.parametrize("peer,scheme", [("127.0.0.1", "https"), ("198.51.100.10", "http")])
def test_only_trusted_proxies_can_preserve_https_redirect_scheme(app_factory, peer, scheme):
    app = app_factory(TRUSTED_PROXY_IPS="127.0.0.1")
    response = client(app, peer).get(
        "/api/auth/check/", follow_redirects=False,
        headers={"Host": "research.example", "X-Forwarded-Proto": "https", "X-Forwarded-For": "203.0.113.10"},
    )
    assert response.status_code == 307
    assert response.headers["Location"] == f"{scheme}://research.example/api/auth/check"


@pytest.mark.parametrize("values", [["http", "https"], ["https,http"], ["ftp"]])
def test_invalid_trusted_proxy_scheme_is_rejected(app_factory, values):
    app = app_factory(TRUSTED_PROXY_IPS="127.0.0.1")
    response = client(app, "127.0.0.1").post(
        "/api/security-probe", json={}, headers=[("X-Forwarded-Proto", value) for value in values],
    )
    assert response.status_code == 403
    assert app.state.probe_calls == []


@pytest.mark.parametrize("cause", ["auth", "rate"])
def test_abnormal_requests_temporarily_block_only_the_source_ip(app_factory, cause):
    app = app_factory(AUTH_FAILURE_LIMIT="1/minute", RATE_LIMIT_VIOLATION_LIMIT="1/minute", RATE_LIMIT_KEY="1/minute")
    http = client(app, key=None if cause == "auth" else KEY_A)
    count = 2 if cause == "auth" else 3
    for _ in range(count):
        http.get("/api/auth/check")
    denied = client(app).get("/api/auth/check")
    assert denied.json()["code"] == "ip_temporarily_blocked"
    assert 1 <= int(denied.headers["Retry-After"]) <= 90
    assert client(app, "198.51.100.11", KEY_B).get("/api/auth/check").status_code == 200
    assert http.get("/health").status_code == 200


def test_rate_window_and_temporary_block_expire_without_extending_on_rejection(app_factory, monkeypatch):
    now = [datetime.now(timezone.utc).timestamp()]
    # 只替换计数后端的时钟，不修改进程全局时间或等待真实封禁结束。
    monkeypatch.setattr(memory_storage, "time", SimpleNamespace(time=lambda: now[0]))
    app = app_factory(RATE_LIMIT_IP="1/minute", RATE_LIMIT_VIOLATION_LIMIT="1/minute")
    app.state.rate_limit_controller.clock = lambda: now[0]
    http = client(app)
    assert http.get("/api/auth/check").status_code == 200
    assert http.get("/api/auth/check").status_code == 429
    assert http.get("/api/auth/check").status_code == 429
    now[0] += 89
    blocked = http.get("/api/auth/check")
    assert blocked.json()["code"] == "ip_temporarily_blocked"
    assert blocked.headers["Retry-After"] == "1"
    now[0] += 2
    assert http.get("/api/auth/check").status_code == 200


@pytest.mark.parametrize("uri", [
    "redis://user:private-redis-password@[invalid", "redis://localhost:bad", "redis://localhost:70000",
    "redis://localhost:0", "redis://local host/0", "memory://ignored", "https://redis.example",
])
def test_invalid_storage_uri_fails_startup_without_leaking_credentials(app_factory, uri):
    with pytest.raises(RuntimeError) as error:
        app_factory(RATE_LIMIT_STORAGE=uri)
    assert "private-redis-password" not in str(error.value)


def test_storage_outage_fails_closed_before_business_dependencies(app_factory, monkeypatch):
    app = app_factory()
    monkeypatch.setattr(app.state.rate_limit_controller.backend.storage, "get", lambda *_: (_ for _ in ()).throw(RuntimeError("private storage diagnostic")))
    response = client(app).post("/api/security-probe", json={})
    assert response.status_code == 503
    assert response.json()["code"] == "security_storage_unavailable"
    assert "private storage diagnostic" not in response.text
    assert app.state.probe_calls == []


def test_audit_includes_denials_failures_and_completed_stream_without_credentials(app_factory, tmp_path):
    app = app_factory()
    http = client(app)
    response = http.get("/api/security-stream", params={"private": KEY_A})
    assert response.text == "data: one\n\ndata: two\n\n"
    assert client(app, key=None).get("/api/auth/check").status_code == 401
    assert http.get("/api/security-failure").status_code == 500
    events = [json.loads(line) for line in (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 3
    assert events[0]["request_id"] == response.headers["X-Request-ID"]
    assert events[0]["response_complete"] is True
    assert events[0]["credential_id"] == app.state.api_key_settings.match(KEY_A).identifier
    assert events[1]["code"] == "missing_api_key"
    assert events[2]["status_code"] == 500 and events[2]["outcome"] == "error"
    assert all(event["client_ip"] == "198.51.100.10" for event in events)
    assert KEY_A[:10] not in json.dumps(events)


def test_audit_uses_route_templates_for_preflight_and_records_rate_denial(app_factory, tmp_path):
    app = app_factory(RATE_LIMIT_IP="1/minute")
    http = client(app)
    http.options("/api/paper/2401.00001/qa", headers={"Origin": "https://research.example", "Access-Control-Request-Method": "POST"})
    assert http.get("/api/auth/check").status_code == 200
    assert http.get("/api/auth/check").status_code == 429
    events = [json.loads(line) for line in (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()]
    assert events[0]["path"] == "/api/paper/{arxiv_id}/qa"
    assert events[0]["client_ip"] == "198.51.100.10"
    assert events[-1]["code"] == "rate_limit_exceeded"
    assert "2401.00001" not in json.dumps(events)


def test_reusing_audit_configuration_does_not_duplicate_records(app_factory, tmp_path):
    first, second = app_factory(), app_factory()
    assert client(first).get("/api/auth/check").status_code == 200
    assert client(second).get("/api/auth/check").status_code == 200
    assert len((tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()) == 2


def test_audit_rotation_bounds_retention_and_keeps_posix_files_private(app_factory, monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIT_LOG_MAX_BYTES", "200")
    monkeypatch.setenv("AUDIT_LOG_BACKUP_COUNT", "2")
    sink = AuditSink()
    for index in range(20):
        sink.write({"request_id": str(index), "message": "rotation-test" * 5})
    files = list(tmp_path.glob("audit.log*"))
    assert len(files) == 3
    assert '"request_id":"19"' in (tmp_path / "audit.log").read_text(encoding="utf-8")
    if os.name == "posix":
        # Windows 使用部署目录 ACL；POSIX 下验证轮转出的旧文件和新文件均不可被其他账号读取。
        assert all(path.stat().st_mode & 0o077 == 0 for path in files)


def test_audit_write_failure_falls_back_to_sanitized_stderr(app_factory, monkeypatch, capsys):
    monkeypatch.setenv("AUDIT_LOG_MAX_BYTES", "1")
    sink = AuditSink()
    # 标准轮转器不会轮转空文件，先写一条记录再模拟下一次轮转的磁盘故障。
    sink.write({"message": "before-failure"})
    def disk_failure():
        raise OSError(f"disk error: {KEY_A}")
    monkeypatch.setattr(sink.logger.handlers[0], "doRollover", disk_failure)
    sink.write({"message": KEY_A, "status_code": 200})
    output = capsys.readouterr().err
    fallback = json.loads(output)
    assert fallback["event"] == "audit_log_write_failed"
    assert fallback["audit_event"]["message"] == "***REDACTED***"
    assert KEY_A not in output and "disk error" not in output


def test_audit_can_be_explicitly_disabled(app_factory, tmp_path):
    app = app_factory(AUDIT_LOG_ENABLED="false")
    assert client(app).get("/api/auth/check").status_code == 200
    assert not (tmp_path / "audit.log").exists()


def test_audit_records_cancelled_stream_once(app_factory):
    records = []
    async def stream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        raise asyncio.CancelledError
    async def receive():
        return {"type": "http.disconnect"}
    async def send(message):
        pass
    middleware = AuditLogMiddleware(stream, sink=SimpleNamespace(write=records.append), routes=[], ip_settings=IPFilterSettings.from_environment())
    scope = {"type": "http", "method": "GET", "path": "/api/probe", "headers": [], "client": ("198.51.100.10", 1)}
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(middleware(scope, receive, send))
    assert len(records) == 1
    assert records[0]["outcome"] == "cancelled"
    assert records[0]["response_complete"] is False


def test_audit_does_not_mark_failed_final_send_as_complete(app_factory):
    records = []
    async def stream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        await send({"type": "http.response.body", "body": b"final", "more_body": False})
    async def receive():
        return {"type": "http.disconnect"}
    async def send(message):
        if message.get("body") == b"final":
            raise OSError("transport closed")
    middleware = AuditLogMiddleware(stream, sink=SimpleNamespace(write=records.append), routes=[], ip_settings=IPFilterSettings.from_environment())
    scope = {"type": "http", "method": "GET", "path": "/api/probe", "headers": [], "client": ("198.51.100.10", 1)}
    with pytest.raises(OSError):
        asyncio.run(middleware(scope, receive, send))
    assert len(records) == 1
    assert records[0]["outcome"] == "error" and records[0]["status_code"] == 200
    assert records[0]["response_complete"] is False
    assert records[0]["bytes_sent"] == len(b"partial")


def test_file_only_keys_are_redacted_in_json_and_split_stream(app_factory, monkeypatch, tmp_path):
    secret = "only-in-private-file-" + "x" * 40
    write_keys(tmp_path, monkeypatch, [{"key": secret}])
    app_factory()
    assert redact_sensitive_value({"diagnostic": secret})["diagnostic"] == "***REDACTED***"
    redactor = StreamingSecretRedactor()
    actual = redactor.feed(secret[:10]) + redactor.feed(secret[10:]) + redactor.finish()
    assert actual == "***REDACTED***"
