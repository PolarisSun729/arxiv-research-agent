"""真实 JWT/SQLite/ASGI 安全回归；所有账号、存储和凭据均为隔离测试数据。"""

from __future__ import annotations

import importlib
import asyncio
import json
from pathlib import Path
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace

import jwt
import httpx
import pytest
from fastapi import Depends
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from auth import passwords
from auth.context import AuthContext, current_auth
from auth.errors import AuthError
from auth.permissions import ALL_ROLES, AccessPolicy, PUBLIC_AUTH_ROUTES, ROUTE_POLICIES
from auth.settings import JwtSettings
from auth.store import AuthStore


# 合成签名串：必须满足 JwtSettings 的长度与熵要求，因此形状与真实密钥一致，仅用于隔离测试。
TEST_SECRET = "S3-ehrUM7OynSbi5Q_82LKHfq0ZVBAtwkgWtcXdvNTpxsJR61"  # gitleaks:allow
PASSWORD = "ResearchPass123"


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    values = {
        "AUTH_MODE": "jwt", "JWT_SECRET_KEY": TEST_SECRET, "JWT_SECRET_FILE": "", "JWT_EXPIRE_MINUTES": "60",
        "JWT_ISSUER": "arxiv-test", "JWT_AUDIENCE": "arxiv-test-api", "ALLOW_PUBLIC_REGISTRATION": "true",
        "AUTH_DATABASE_PATH": str(tmp_path / "auth" / "accounts.sqlite3"), "AUTH_MAX_REQUEST_BYTES": "1048576",
        "ALLOWED_ORIGINS": "https://research.example", "AUDIT_LOG_ENABLED": "true", "AUDIT_LOG_FILE": str(tmp_path / "audit.log"),
        "RATE_LIMIT_STORAGE": "memory://", "IP_FILTER_MODE": "disabled", "IP_BLACKLIST": "", "IP_WHITELIST": "", "TRUSTED_PROXY_IPS": "",
        **{name: "10000/minute" for name in ("RATE_LIMIT_IP", "RATE_LIMIT_USER", "RATE_LIMIT_AUTH", "RATE_LIMIT_LOGIN_ACCOUNT",
           "RATE_LIMIT_READ", "RATE_LIMIT_WRITE", "RATE_LIMIT_EXPENSIVE", "AUTH_FAILURE_LIMIT", "RATE_LIMIT_VIOLATION_LIMIT")},
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    # 降低测试成本而不开放生产配置降级入口；算法、长度和比较行为仍使用真实 bcrypt。
    monkeypatch.setattr(passwords, "BCRYPT_ROUNDS", 4)

    def factory(**overrides):
        for name, value in overrides.items():
            monkeypatch.setenv(name, str(value))
        return importlib.import_module("main").create_app(load_mode="lazy", enable_debug_routes=True)

    return factory


@pytest.fixture
def app(app_factory):
    return app_factory()


def http(app, token=None, *, ip="198.51.100.9"):
    return TestClient(app, client=(ip, 12000), headers={"Authorization": f"Bearer {token}"} if token else {}, raise_server_exceptions=False)


def account(app, *, username="researcher", role="researcher"):
    user = app.state.authenticator.store.create_user(username=username, email=f"{username}@example.com",
                                                     hashed_password=passwords.hash_password(PASSWORD), role=role)
    response = http(app).post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    assert token.startswith("eyJ") and "REDACTED" not in token
    return user, token


def test_registration_is_guest_only_and_never_returns_credentials(app):
    client = http(app)
    payload = {"username": "new_user", "email": "new@example.com", "password": PASSWORD}
    rejected = client.post("/api/auth/register", json={**payload, "role": "admin"})
    assert rejected.status_code == 422
    assert PASSWORD not in rejected.text
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    assert response.json()["role"] == "guest"
    assert "password" not in response.text and "token" not in response.text
    user = app.state.authenticator.store.get_user(response.json()["user_id"])
    assert user.hashed_password.startswith("$2b$")
    assert passwords.verify_password(PASSWORD, user.hashed_password)
    quotas = app.state.authenticator.store.quotas(user.user_id)
    assert {q["quota_type"]: q["daily_limit"] for q in quotas} == {"papers": 10, "qa_queries": 5, "agent_runs": 0}
    assert client.post("/api/auth/register", json={**payload, "username": "NEW_USER"}).status_code == 409


def test_registration_disabled_is_public_but_rejected_before_body(app_factory):
    client = http(app_factory(ALLOW_PUBLIC_REGISTRATION="false"))
    assert client.get("/api/auth/config").json() == {"mode": "jwt", "registration_enabled": False}
    response = client.post("/api/auth/register", content="{bad")
    assert response.status_code == 403 and response.json()["code"] == "registration_disabled"
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("password", ["weak", "alllower123", "ALLUPPER123", "NoNumbersHere", "A1" + "密" * 24, "Aa1" + "x" * 70])
def test_password_policy_rejects_weak_and_overlong_bytes(app, password):
    response = http(app).post("/api/auth/register", json={"username": "new_user", "email": "u@example.com", "password": password})
    assert response.status_code == 422
    assert password not in response.text


@pytest.mark.parametrize("path", ["/api/papers", "/api/auth/me", "/api/paper/p1/evidence-assets/img", "/api/paper/p1/notes/export", "/api/debug/chunks/files"])
def test_api_key_cannot_bypass_jwt_for_any_transport(app, path):
    response = http(app).get(path, headers={"X-API-Key": "old-compatibility-key-" + "a" * 40})
    assert response.status_code == 401 and response.json()["code"] == "missing_token"
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("change", [
    {"exp": 1}, {"nbf": 4102444800}, {"iat": 4102444800}, {"iss": "another-app"}, {"aud": "another-api"},
    {"sub": "another-user"}, {"jti": "unknown-session"}, {"ver": True}, {"exp": "9999999999"},
])
def test_claim_validation_and_session_lookup_fail_closed(app, change):
    _, token = account(app)
    claims = jwt.decode(token, options={"verify_signature": False})
    forged = jwt.encode({**claims, **change}, TEST_SECRET, algorithm="HS256")
    response = http(app, forged).get("/api/auth/me")
    assert response.status_code == 401 and response.json()["code"] == "invalid_token"


def test_signature_algorithm_required_claims_and_duplicate_headers(app):
    _, token = account(app)
    claims = jwt.decode(token, options={"verify_signature": False})
    bad_tokens = [jwt.encode(claims, "different-secret-with-sufficient-length-to-sign", algorithm="HS256"),
                  jwt.encode(claims, TEST_SECRET, algorithm="HS384"), jwt.encode(claims, "", algorithm="none")]
    for name in ("sub", "jti", "ver", "iat", "nbf", "exp", "iss", "aud"):
        bad_tokens.append(jwt.encode({key: value for key, value in claims.items() if key != name}, TEST_SECRET, algorithm="HS256"))
    for bad in bad_tokens:
        assert http(app, bad).get("/api/auth/me").status_code == 401
    response = http(app).get("/api/auth/me", headers=[("Authorization", f"Bearer {token}"), ("Authorization", f"Bearer {token}")])
    assert response.status_code == 401


def test_logout_revokes_only_current_session_and_password_revokes_all(app):
    user, first = account(app)
    second = http(app).post("/api/auth/login", json={"username": user.username, "password": PASSWORD}).json()["access_token"]
    assert http(app, first).post("/api/auth/logout").status_code == 200
    assert http(app, first).get("/api/auth/me").status_code == 401
    assert http(app, second).get("/api/auth/me").status_code == 200
    response = http(app, second).post("/api/auth/password", json={"current_password": PASSWORD, "new_password": "ChangedPassword123"})
    assert response.status_code == 200
    assert http(app, second).get("/api/auth/me").status_code == 401
    assert http(app).post("/api/auth/login", json={"username": user.username, "password": PASSWORD}).status_code == 401
    assert http(app).post("/api/auth/login", json={"username": user.username, "password": "ChangedPassword123"}).status_code == 200


def test_admin_management_revokes_existing_tokens_and_preserves_last_admin(app):
    admin, admin_token = account(app, username="administrator", role="admin")
    user, token = account(app)
    client = http(app, admin_token)
    assert http(app, token).get("/api/auth/users").status_code == 403
    assert client.get("/api/auth/users").json()["total"] == 2
    assert client.patch(f"/api/auth/users/{user.user_id}", json={"role": "viewer"}).status_code == 200
    assert http(app, token).get("/api/auth/me").status_code == 401
    new_token = http(app).post("/api/auth/login", json={"username": user.username, "password": PASSWORD}).json()["access_token"]
    assert client.patch(f"/api/auth/users/{user.user_id}", json={"is_active": False}).status_code == 200
    assert http(app, new_token).get("/api/auth/me").status_code == 401
    assert http(app).post("/api/auth/login", json={"username": user.username, "password": PASSWORD}).status_code == 401
    assert client.patch(f"/api/auth/users/{admin.user_id}", json={"is_active": False}).status_code == 409
    assert client.patch(f"/api/auth/users/{admin.user_id}", json={"role": "researcher"}).status_code == 409


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/agent/chat"), ("POST", "/api/agent/chat/stream"),
    ("POST", "/api/agent/work-continuations/work/resume/stream"),
    ("POST", "/api/paper/p1/create-qa-index"), ("DELETE", "/api/paper/p1"),
    ("GET", "/api/paper/p1/qa-trace/latest"), ("GET", "/api/paper/p1/qa-diagnose"),
    ("GET", "/api/debug/chunks/files"), ("POST", "/api/user/research-profile/rebuild"),
])
@pytest.mark.parametrize("role", ["viewer", "guest"])
def test_roles_deny_sensitive_routes_before_invalid_body_or_dependencies(app, role, method, path):
    _, token = account(app, role=role)
    response = http(app, token).request(method, path, content="{bad", headers={"Content-Type": "application/json"})
    assert response.status_code == 403 and response.json()["code"] == "insufficient_permissions"


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/api/user/preferences/someone-else", None),
    ("GET", "/api/stats?user_id=someone-else", None),
    ("GET", "/api/paper/p1/notes?user_id=someone-else", None),
    ("POST", "/api/user/like-paper", {"arxiv_id": "p1", "user_id": "someone-else"}),
    ("POST", "/api/agent/chat", {"message": "test", "context": {"user_id": "someone-else"}}),
    ("POST", "/api/paper/p1/qa/stream", {"question": "test", "user_id": "someone-else"}),
])
def test_identity_forgery_rejected_before_business(app, method, path, body):
    _, token = account(app)
    response = http(app, token).request(method, path, json=body)
    assert response.status_code == 403 and response.json()["code"] == "identity_mismatch"


def test_every_registered_route_has_explicit_policy_and_unknown_routes_deny(app):
    for route in app.routes:
        if route.path == "/health":
            continue
        for method in route.methods:
            assert (method, route.path) in ROUTE_POLICIES or (method, route.path) in PUBLIC_AUTH_ROUTES
    calls = []

    @app.get("/api/new-endpoint", dependencies=[Depends(lambda: calls.append(True))])
    def new_endpoint():
        return {"ok": True}

    _, token = account(app, role="admin")
    assert http(app, token).get("/api/new-endpoint").status_code == 403
    assert calls == []


def test_quota_atomic_across_connections_and_fixed_utc_reset(app):
    user, token = account(app)
    authenticator = app.state.authenticator
    store, session = authenticator.store, authenticator.authenticate(token)
    store.set_quota(user.user_id, quota_type="qa_queries", daily_limit=3)

    def consume(_):
        other_store = AuthStore(store.path)
        try:
            other_store.consume(**session.store_arguments(), amounts={"qa_queries": 1})
            return True
        except AuthError as exc:
            assert exc.code == "quota_exceeded"
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(consume, range(20))) == 3
    assert next(q for q in store.quotas(user.user_id) if q["quota_type"] == "qa_queries")["used_today"] == 3
    # /me 读取和角色修改都不清零当天记录；UTC 跨天投影为零，而非读取后滚动 24 小时。
    store.update_user(user.user_id, role="viewer")
    assert next(q for q in store.quotas(user.user_id) if q["quota_type"] == "qa_queries")["used_today"] == 3
    store.clock = lambda: datetime(2090, 1, 2, 23, 59, tzinfo=timezone.utc).timestamp()
    projected = store.quotas(user.user_id)
    assert all(q["used_today"] == 0 and q["reset_at"].startswith("2090-01-03T00:00:00") for q in projected)


def test_quota_rejects_sse_before_business_and_records_user_audit(app, monkeypatch, tmp_path):
    user, token = account(app)
    store = app.state.authenticator.store
    store.set_quota(user.user_id, quota_type="qa_queries", daily_limit=0)
    response = http(app, token).post("/api/paper/p1/qa/stream", json={"question": "test"})
    assert response.status_code == 429 and response.json()["quota_type"] == "qa_queries"
    assert int(response.headers["Retry-After"]) > 0
    events = [json.loads(line) for line in (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["user_id"] == user.user_id and events[-1]["role"] == "researcher"
    assert events[-1]["code"] == "quota_exceeded"
    assert token not in json.dumps(events) and PASSWORD not in json.dumps(events)


def test_graph_worker_identity_is_bound_and_rechecked_after_revocation(app):
    from auth.tool_access import bind_graph_node
    user, token = account(app)
    authenticator = app.state.authenticator
    context = AuthContext(authenticator.authenticate(token), authenticator.store)
    binding = current_auth.set(context)
    try:
        node = bind_graph_node(lambda state: current_auth.get().user_id)
    finally:
        current_auth.reset(binding)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(node, {"user_id": user.user_id}).result() == user.user_id
        authenticator.store.revoke_session(user_id=user.user_id, jti=context.session.jti)
        with pytest.raises(AuthError, match="登录已失效"):
            pool.submit(node, {"user_id": user.user_id}).result()
    assert current_auth.get() is None


def test_auth_rate_limits_and_oversized_public_body(app_factory):
    app = app_factory(RATE_LIMIT_AUTH="1/minute", AUTH_MAX_REQUEST_BYTES="100")
    client = http(app)
    assert client.post("/api/auth/login", content="x" * 101).status_code == 413
    payload = {"username": "missing", "password": PASSWORD}
    assert client.post("/api/auth/login", json=payload).status_code == 401
    assert client.post("/api/auth/login", json=payload).status_code == 429


def test_anonymous_login_long_field_names_remain_responsive(app_factory):
    # 走真实公开路由和 Pydantic 错误出口；子进程继承本 fixture 的临时库，不读取开发者的 .env。
    probe = """
import os, sys
from pathlib import Path
sys.path.insert(0, 'backend')
from scripts.test_security import isolate_configuration_reads
isolate_configuration_reads(Path(os.environ['AUTH_DATABASE_PATH']).parent)
from fastapi.testclient import TestClient
from main import app
client = TestClient(app, raise_server_exceptions=False)
for field in ('a' * 65536, 'password' * 8192):
    response = client.post('/api/auth/login', json={'username': 'test', 'password': 'TestPass123', field: 'private-input'})
    assert response.status_code == 422
    assert 'private-input' not in response.text
    assert len(response.content) < 2048
assert client.get('/health').status_code == 200
"""
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", probe], cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("匿名登录的 64 KiB 非法字段导致错误出口超过 15 秒，且阻塞健康请求。")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("operation", ["search", "download"])
def test_arxiv_blocking_io_does_not_block_health_or_lose_identity(app, operation):
    import dependencies

    user, token = account(app, username="io_researcher")
    entered, release, finished = Event(), Event(), Event()
    identities = []

    def blocking_service(*args, **kwargs):
        identities.append(current_auth.get().user_id)
        entered.set()
        try:
            # 有界等待使旧的同步调用确定性失败；不访问 arXiv、不使用付费模型。
            if not release.wait(2):
                raise TimeoutError("事件循环未能在同步 I/O 期间响应。")
            return "download.pdf" if operation == "download" else {"papers": []}
        finally:
            finished.set()

    dependency = dependencies.get_arxiv_api_service if operation == "download" else dependencies.get_arxiv_search_backend
    app.dependency_overrides[dependency] = lambda: SimpleNamespace(**{
        "download_pdf" if operation == "download" else "search": blocking_service,
    })
    payload = {"arxiv_id": "2301.01234", "pdf_url": "https://arxiv.org/pdf/2301.01234"} if operation == "download" else {"search_query": "test"}

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            pending = asyncio.create_task(client.post(f"/api/arxiv/{operation}", json=payload, headers={"Authorization": f"Bearer {token}"}))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                response = await client.get("/health")
                assert response.status_code == 200
                assert not finished.is_set(), "健康请求只能在同步 I/O 超时后运行。"
            finally:
                release.set()
                result = await pending
            assert result.status_code == 200

    asyncio.run(scenario())
    assert identities == [user.user_id]


def test_dashboard_keeps_sync_diagnostics_for_admin_only(app, monkeypatch, tmp_path):
    import dependencies
    import routers.paper_router as paper_router

    meta = tmp_path / "sync.json"
    diagnostic = "internal-file-path /private/provider-trace.txt"
    meta.write_text(json.dumps({"status": "failed", "error_message": diagnostic}), encoding="utf-8")
    monkeypatch.setattr(paper_router, "SYNC_META_FILE", meta)
    monkeypatch.setattr(paper_router, "SYNC_STATE_FILE", tmp_path / "missing.state")
    app.dependency_overrides[dependencies.get_user_preference_store] = lambda: SimpleNamespace(get_user_labeled_paper_count=lambda **kwargs: 0)
    app.dependency_overrides[dependencies.get_oai_database_service] = lambda: SimpleNamespace(get_total_paper_count=lambda: 0)
    _, guest_token = account(app, username="stats_guest", role="guest")
    _, admin_token = account(app, username="stats_admin", role="admin")
    guest = http(app, guest_token).get("/api/stats")
    assert guest.status_code == 200
    assert guest.json()["lastSyncStatus"] == "failed"
    assert guest.json()["syncErrorMessage"]
    assert diagnostic not in guest.text
    assert http(app, guest_token).get("/api/sync-status").status_code == 403
    assert http(app, admin_token).get("/api/stats").json()["syncErrorMessage"] == diagnostic
    assert http(app, admin_token).get("/api/sync-status").json()["syncErrorMessage"] == diagnostic


def test_weak_or_reused_jwt_configuration_fails_closed(app_factory, monkeypatch):
    app_factory()
    # 最后一个值是 settings.py 明确拒绝的文档旧示例密钥，测试即验证其被拒绝。
    for value in ("", "x" * 64, "change-me-" + TEST_SECRET, "tZ9xK2pL8mN4qR6wY0zA3bC5dE7fG9hJ1kM4nP6qS8uW0yA2"):  # gitleaks:allow
        monkeypatch.setenv("JWT_SECRET_KEY", value)
        with pytest.raises(RuntimeError):
            JwtSettings.from_environment()
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_SECRET)
    monkeypatch.setenv("ALIYUN_API_KEY", TEST_SECRET)
    with pytest.raises(RuntimeError):
        JwtSettings.from_environment()


@pytest.fixture
def business(app, monkeypatch, tmp_path):
    import dependencies
    from services.storage.sqlite import StorageContainer
    from services.storage.sqlite.connection import SqliteConnectionProvider
    storage = StorageContainer(connection_provider=SqliteConnectionProvider(str(tmp_path / "business.sqlite3")))

    def getter(value):
        return lambda: value

    for name, value in {
        "get_paper_chat_session_store": storage.paper_chat_sessions,
        "get_paper_chat_message_store": storage.paper_chat_messages,
        "get_paper_note_store": storage.paper_notes,
        "get_paper_catalog_store": storage.paper_catalog,
        "get_user_preference_store": storage.user_preferences,
        "get_agent_session_store": storage.agent_sessions,
    }.items():
        original = getattr(dependencies, name)
        replacement = getter(value)
        app.dependency_overrides[original] = replacement
        monkeypatch.setattr(dependencies, name, replacement)
    import routers.agent_router as agent_router
    monkeypatch.setattr(agent_router, "get_agent_session_store", getter(storage.agent_sessions))
    app.dependency_overrides[dependencies.get_paper_qa_service] = getter(SimpleNamespace(
        answer_question=lambda arxiv_id, payload: {"user_id": payload.user_id, "session_id": payload.session_id},
    ))
    app.dependency_overrides[dependencies.get_generation_service] = getter(object())
    return storage


def test_real_notes_sessions_and_qa_are_isolated(app, business):
    alice, alice_token = account(app, username="alice")
    bob, bob_token = account(app, username="bob")
    first, second = http(app, alice_token), http(app, bob_token)
    alice_session = first.post("/api/paper/p1/chat-sessions", json={}).json()["item"]
    bob_session = second.post("/api/paper/p1/chat-sessions", json={}).json()["item"]
    assert alice_session["user_id"] == alice.user_id and bob_session["user_id"] == bob.user_id
    note = first.post("/api/paper/p1/notes", json={"content": "private-alice-note", "user_id": None}).json()["item"]
    assert note["user_id"] == alice.user_id
    assert second.get("/api/paper/p1/notes").json()["items"] == []
    assert "private-alice-note" not in second.get("/api/paper/p1/notes/export").text
    assert first.get("/api/paper/p1/notes/export").status_code == 200
    assert second.patch(f"/api/paper/p1/notes/{note['note_id']}", json={"content": "overwrite"}).status_code == 404
    assert second.delete(f"/api/paper/p1/notes/{note['note_id']}").status_code == 404
    path = f"/api/paper/p1/chat-sessions/{alice_session['session_id']}"
    assert second.get(path).status_code == 404
    assert second.get(path + "/messages").status_code == 404
    assert second.post(path + "/clear", json={}).status_code == 404
    assert second.delete(path).status_code == 404
    for endpoint in ("qa", "qa/stream"):
        assert second.post(f"/api/paper/p1/{endpoint}", json={"question": "test", "session_id": alice_session["session_id"]}).status_code == 404
    answer = second.post("/api/paper/p1/qa", json={"question": "test", "session_id": bob_session["session_id"]})
    assert answer.json()["user_id"] == bob.user_id
    assert second.post("/api/paper/p2/qa", json={"question": "test", "session_id": bob_session["session_id"]}).status_code == 404


def test_note_sources_cannot_reference_other_user_or_paper(app, business):
    alice, alice_token = account(app, username="alice")
    bob, bob_token = account(app, username="bob")
    alice_session = business.paper_chat_sessions.create_paper_chat_session(arxiv_id="p1", user_id=alice.user_id)
    bob_other_paper = business.paper_chat_sessions.create_paper_chat_session(arxiv_id="p2", user_id=bob.user_id)
    with business.connection_provider.connect() as conn:
        conn.executemany("INSERT INTO paper_chat_messages(message_id,session_id,role,content,turn_id) VALUES(?,?,?,?,?)", [
            ("alice-message", alice_session["session_id"], "assistant", "private-alice-message", "alice-turn"),
            ("bob-other-paper", bob_other_paper["session_id"], "assistant", "other-paper", "bob-turn"),
        ])
        conn.commit()
    client = http(app, bob_token)
    for link in ({"session_id": alice_session["session_id"]}, {"source_message_id": "alice-message"},
                 {"source_message_id": "bob-other-paper"}, {"source_turn_id": "alice-turn"}):
        assert client.post("/api/paper/p1/notes", json={"content": "note", **link}).status_code == 404
    own = http(app, alice_token).post("/api/paper/p1/notes", json={"content": "linked", "source_message_id": "alice-message"})
    assert own.status_code == 200 and own.json()["item"]["source_turn_id"] == "alice-turn"


def test_notes_markdown_export_redacts_credentials_from_all_sources(app, business, monkeypatch):
    from utils.secret_redaction import register_sensitive_values

    user, token = account(app, username="note_export")
    client = http(app, token)
    secret = "export-fixture-credential-" + "c" * 32
    register_sensitive_values([secret])
    session = business.paper_chat_sessions.create_paper_chat_session(arxiv_id="p1", user_id=user.user_id)
    # 保留真实存储与所有权检查，模拟历史问答和用户笔记已经包含凭据的场景。
    with business.connection_provider.connect() as conn:
        conn.execute(
            "INSERT INTO paper_chat_messages(message_id,session_id,role,content,turn_id,sources) VALUES(?,?,?,?,?,?)",
            ("export-source", session["session_id"], "assistant", "来源回答", "export-turn",
             json.dumps([{"content": f"关联来源 {secret}", "page_number": 1}], ensure_ascii=False)),
        )
        conn.commit()
    monkeypatch.setattr(business.paper_catalog, "get_paper", lambda _arxiv_id: {"title": f"论文标题 {secret}"})
    created = client.post("/api/paper/p1/notes", json={
        "title": f"笔记标题 {secret}", "content": f"正常正文 {secret}\npassword=unknown-export-password",
        "source_message_id": "export-source",
    })
    assert created.status_code == 200
    response = client.get("/api/paper/p1/notes/export")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    assert secret not in response.text
    assert "unknown-export-password" not in response.text
    for preserved in ("论文标题", "笔记标题", "正常正文", "关联来源"):
        assert preserved in response.text


def test_agent_session_cannot_claim_legacy_or_foreign_thread(app, business, monkeypatch):
    import routers.agent_router as agent_router
    alice, _ = account(app, username="alice")
    bob, token = account(app, username="bob")
    foreign = business.agent_sessions.create_or_get_agent_session(user_id=alice.user_id)
    calls = []

    def stream(payload):
        calls.append(payload)
        return StreamingResponse(iter([f"data: {payload.user_id}\n\n"]), media_type="text/event-stream")

    monkeypatch.setattr(agent_router, "stream_arxiv_search_agent", stream)
    client = http(app, token)
    for session_id in (foreign["session_id"], "unclaimed-legacy-checkpoint"):
        assert client.post("/api/agent/chat/stream", json={"message": "hello", "session_id": session_id}).status_code == 404
    assert calls == []
    response = client.post("/api/agent/chat/stream", json={"message": "hello"})
    assert response.status_code == 200 and bob.user_id in response.text
    assert business.agent_sessions.get_agent_session(calls[0].session_id, user_id=bob.user_id)


def test_profile_jobs_require_the_authenticated_owner(app, monkeypatch):
    import dependencies
    alice, _ = account(app, username="alice")
    _, token = account(app, username="bob")
    service = SimpleNamespace(get_profile_build_job=lambda job_id: {"job_id": job_id, "user_id": alice.user_id, "private": "alice-profile"})
    app.dependency_overrides[dependencies.get_memory_service] = lambda: service
    response = http(app, token).get("/api/user/research-profile/build-jobs/alice-job")
    assert response.status_code == 404 and "alice-profile" not in response.text


def test_tool_aliases_share_quota_and_reject_llm_identity_override(app, monkeypatch):
    from tools import tool_registry
    user, token = account(app)
    authenticator = app.state.authenticator
    authenticator.store.set_quota(user.user_id, quota_type="papers", daily_limit=1)
    calls = []
    original = tool_registry.TOOL_REGISTRY["record_paper_preference"]

    def write(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "data": {"user_id": kwargs["user_id"]}, "summary": "recorded"}

    from dataclasses import replace
    monkeypatch.setitem(tool_registry.TOOL_REGISTRY, "record_paper_preference", replace(original, func=write))
    binding = current_auth.set(AuthContext(authenticator.authenticate(token), authenticator.store))
    try:
        denied = tool_registry.invoke_tool("record_paper_preference", user_id="another-user", arxiv_id="p1", liked=True)
        assert denied["ok"] is False and calls == []
        assert tool_registry.invoke_tool("record_user_paper_preference", arxiv_id="p1", liked=True)["ok"] is True
        assert tool_registry.invoke_tool("record_paper_preference", arxiv_id="p2", liked=True)["ok"] is False
        assert len(calls) == 1 and calls[0]["user_id"] == user.user_id
    finally:
        current_auth.reset(binding)


def test_user_rate_is_shared_by_rotated_tokens_and_not_shared_between_users(app_factory):
    app = app_factory(RATE_LIMIT_USER="1/minute")
    user, first = account(app)
    second = http(app).post("/api/auth/login", json={"username": user.username, "password": PASSWORD}).json()["access_token"]
    assert http(app, first).get("/api/auth/check").status_code == 200
    assert http(app, second, ip="198.51.100.10").get("/api/auth/check").status_code == 429
    _, other = account(app, username="other")
    assert http(app, other).get("/api/auth/check").status_code == 200


def test_quota_and_auth_store_failures_prevent_business(app, monkeypatch):
    _, token = account(app)
    calls = []
    monkeypatch.setitem(ROUTE_POLICIES, ("GET", "/api/quota-probe"), AccessPolicy(ALL_ROLES, "papers"))

    @app.get("/api/quota-probe")
    def probe():
        calls.append(True)
        return {"ok": True}

    def unavailable(**kwargs):
        raise AuthError("security_storage_unavailable", retry_after=5)

    monkeypatch.setattr(app.state.authenticator.store, "consume", unavailable)
    assert http(app, token).get("/api/quota-probe").status_code == 503
    monkeypatch.setattr(app.state.authenticator.store, "check_session", unavailable)
    assert http(app, token).get("/api/auth/check").status_code == 503
    assert calls == []


def test_bare_jwt_redaction_across_stream_fragments(app):
    from utils.secret_redaction import StreamingSecretRedactor, redact_text
    _, token = account(app)
    assert token not in redact_text(f"copied {token} into diagnostics")
    redactor = StreamingSecretRedactor()
    output = "".join(redactor.feed(char) for char in f"copied {token} done") + redactor.finish()
    assert token not in output and "REDACTED" in output


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "QWEN_API_KEY", "DEEPSEEK_API_KEY", "BACKEND_API_KEYS"])
def test_jwt_signing_secret_cannot_reuse_any_provider_or_shared_key(app_factory, monkeypatch, name):
    monkeypatch.setenv(name, f" {TEST_SECRET} " if name == "BACKEND_API_KEYS" else TEST_SECRET)
    with pytest.raises(RuntimeError):
        JwtSettings.from_environment()


def test_bootstrap_cli_uses_selected_auth_database_and_hidden_password(app_factory, monkeypatch, tmp_path, capsys):
    import getpass

    cli = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts" / "manage_users.py"))
    selected_path = tmp_path / "selected" / "auth.sqlite3"
    config_path = tmp_path / "deployment.env"
    config_path.write_text(f"AUTH_DATABASE_PATH={selected_path.as_posix()}\n", encoding="utf-8")
    monkeypatch.delenv("AUTH_DATABASE_PATH")
    monkeypatch.setattr(sys, "argv", ["manage_users.py", "create-admin", "--username", "bootstrap",
                                    "--email", "bootstrap@example.com", "--env-file", str(config_path)])
    monkeypatch.setattr(getpass, "getpass", lambda prompt: PASSWORD)
    assert cli["main"]() == 0
    user = AuthStore(selected_path).get_by_username("bootstrap")
    assert user.role == "admin" and passwords.verify_password(PASSWORD, user.hashed_password)
    assert PASSWORD not in capsys.readouterr().out


def test_bootstrap_cli_missing_config_fails_before_password_or_database(monkeypatch, tmp_path, capsys):
    import getpass

    cli = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts" / "manage_users.py"))
    monkeypatch.setattr(sys, "argv", ["manage_users.py", "create-admin", "--username", "bootstrap",
                                    "--env-file", str(tmp_path / "missing.env")])
    monkeypatch.setattr(getpass, "getpass", lambda prompt: pytest.fail("missing config must not prompt"))
    assert cli["main"]() == 1
    assert "配置文件不存在" in capsys.readouterr().err


@pytest.mark.parametrize("arxiv_id,url", [
    ("../private", "https://arxiv.org/pdf/../private"),
    ("2301.00001", "http://169.254.169.254/latest/meta-data"),
])
def test_researcher_download_rejects_unsafe_targets_at_service_boundary(app, monkeypatch, tmp_path, arxiv_id, url):
    import dependencies
    from services.arxiv.arxiv_search_service import ArxivSearchService

    # 使用真实下载实现，只替换远程传输；证明已登录研究者也不能触达任意地址。
    service = ArxivSearchService()
    service.papers_dir = str(tmp_path)
    monkeypatch.setattr(service, "_make_request_with_retry", lambda *args, **kwargs: pytest.fail("unsafe target reached network"))
    app.dependency_overrides[dependencies.get_arxiv_api_service] = lambda: service
    _, token = account(app)
    response = http(app, token).post("/api/arxiv/download", json={"arxiv_id": arxiv_id, "pdf_url": url})
    assert response.status_code == 400
    assert response.json()["code"] == "arxiv_invalid_download"


@pytest.mark.parametrize("scalar", [True, False])
def test_interest_vector_request_preserves_both_shapes_and_binds_user(app, scalar):
    from routers import user_router

    called = []
    def generate(*, user_id):
        called.append(user_id)
        return {"user_id": user_id}

    # 旧单测的轻量加载器会替换 dependencies 模块属性；覆盖路由注册时绑定的对象，避免误调真实业务。
    app.dependency_overrides[user_router.get_recommendation_service] = lambda: SimpleNamespace(generate_user_interest_vector=generate)
    user, token = account(app)
    client = http(app, token)
    response = client.post("/api/user/generate-interest-vector", json=user.user_id if scalar else {"user_id": user.user_id})
    assert response.status_code == 200 and response.json()["user_id"] == user.user_id
    forged = client.post("/api/user/generate-interest-vector", json="legacy-other" if scalar else {"user_id": "legacy-other"})
    assert forged.status_code == 403 and called == [user.user_id]


@pytest.mark.parametrize("scalar", [True, False])
def test_interest_vector_legacy_api_key_shape_remains_compatible(app_factory, scalar):
    from routers import user_router

    app = app_factory(AUTH_MODE="api_key", BACKEND_API_KEYS=TEST_SECRET, API_KEY_CONFIG_FILE="")
    app.dependency_overrides[user_router.get_recommendation_service] = lambda: SimpleNamespace(
        generate_user_interest_vector=lambda **kwargs: kwargs,
    )
    client = http(app)
    response = client.post("/api/user/generate-interest-vector", headers={"X-API-Key": TEST_SECRET},
                           json="legacy-user" if scalar else {"user_id": "legacy-user"})
    assert response.status_code == 200 and response.json()["user_id"] == "legacy-user"


def test_quota_waiting_for_write_lock_uses_the_day_when_lock_is_acquired(tmp_path, monkeypatch):
    """模拟另一 worker 持锁跨零点，准入不能把新一天的配额记录倒写为昨天。"""
    current_time = [datetime(2030, 1, 1, 23, 59, 59, tzinfo=timezone.utc).timestamp()]
    next_day = datetime(2030, 1, 2, tzinfo=timezone.utc).timestamp()
    store = AuthStore(tmp_path / "midnight" / "auth.sqlite3", clock=lambda: current_time[0])
    other_worker = AuthStore(store.path)
    user = store.create_user(username="midnight", email="midnight@example.com", hashed_password="unused", role="researcher")
    store.open_session(user, jti="midnight-session", expires_at=int(next_day + 3600))
    store.set_quota(user.user_id, quota_type="papers", daily_limit=1)
    connection = store.connection
    waiting = Event()

    @contextmanager
    def signal_connection(*, write=False):
        if write:
            waiting.set()
        with connection(write=write) as conn:
            yield conn

    monkeypatch.setattr(store, "connection", signal_connection)
    arguments = {"user_id": user.user_id, "version": user.token_version, "jti": "midnight-session", "amounts": {"papers": 1}}
    with ThreadPoolExecutor(max_workers=1) as pool:
        with other_worker.connection(write=True):
            future = pool.submit(store.consume, **arguments)
            assert waiting.wait(2)
            current_time[0] = next_day
        future.result(timeout=3)
    quota = next(item for item in store.quotas(user.user_id) if item["quota_type"] == "papers")
    assert quota["used_today"] == 1 and quota["remaining"] == 0
    with pytest.raises(AuthError) as rejected:
        store.consume(**arguments)
    assert rejected.value.code == "quota_exceeded"
