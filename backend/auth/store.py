"""独立 SQLite 认证存储：账号、会话撤销和 UTC 配额共用原子事务。"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from auth.errors import AuthError
from auth.models import AuthUser, DEFAULT_QUOTAS, QUOTA_TYPES


class AuthStore:
    def __init__(self, path: Path, *, clock=None) -> None:
        self.path = Path(path)
        self.clock = clock or time.time
        try:
            # 认证库放在独立私有目录中，包含 WAL/SHM 的所有副本也不能被静态服务器公开。
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.name != "nt" and self.path.parent.stat().st_mode & 0o077:
                raise OSError
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
            except FileExistsError:
                if os.name != "nt" and self.path.stat().st_mode & 0o077:
                    raise OSError
            with self.connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS auth_users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        hashed_password TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('admin','researcher','viewer','guest')),
                        is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
                        token_version INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_login TEXT
                    );
                    CREATE TABLE IF NOT EXISTS auth_quotas (
                        user_id TEXT NOT NULL REFERENCES auth_users(user_id) ON DELETE CASCADE,
                        quota_type TEXT NOT NULL CHECK(quota_type IN ('papers','qa_queries','agent_runs')),
                        daily_limit INTEGER NOT NULL CHECK(daily_limit >= -1),
                        used_today INTEGER NOT NULL DEFAULT 0 CHECK(used_today >= 0),
                        usage_day TEXT NOT NULL,
                        PRIMARY KEY(user_id, quota_type)
                    );
                    CREATE TABLE IF NOT EXISTS auth_sessions (
                        jti TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES auth_users(user_id) ON DELETE CASCADE,
                        token_version INTEGER NOT NULL,
                        expires_at INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id);
                    CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at);
                """)
        except (OSError, AuthError):
            raise RuntimeError("认证数据库初始化失败，请检查 AUTH_DATABASE_PATH 和私有目录权限（0700/0600）。") from None

    @contextmanager
    def connection(self, *, write: bool = False):
        conn = None
        try:
            conn = sqlite3.connect(str(self.path), timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            if write:
                # 先取得写锁，再读取和更新；多线程/多 worker 不能同时消费同一份剩余额度。
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.commit()
        except sqlite3.Error:
            raise AuthError("security_storage_unavailable", retry_after=5) from None
        finally:
            if conn is not None:
                conn.close()

    def _now(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), timezone.utc)

    @staticmethod
    def _user(row) -> AuthUser | None:
        if row is None:
            return None
        values = dict(row)
        values["is_active"] = bool(values["is_active"])
        return AuthUser(**values)

    def get_user(self, user_id: str) -> AuthUser | None:
        with self.connection() as conn:
            return self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())

    def get_by_username(self, username: str) -> AuthUser | None:
        with self.connection() as conn:
            return self._user(conn.execute("SELECT * FROM auth_users WHERE username=?", (username.casefold(),)).fetchone())

    def create_user(self, *, username: str, email: str, hashed_password: str, role: str = "guest") -> AuthUser:
        if role not in DEFAULT_QUOTAS:
            raise ValueError("unsupported role")
        user_id, now = uuid4().hex, self._now()
        with self.connection(write=True) as conn:
            try:
                conn.execute(
                    "INSERT INTO auth_users(user_id,username,email,hashed_password,role,created_at) VALUES(?,?,?,?,?,?)",
                    (user_id, username.casefold(), email.casefold(), hashed_password, role, now.isoformat()),
                )
            except sqlite3.IntegrityError:
                # 用户与全部配额一起提交，冲突时不留下没有配额的半成品账号。
                raise AuthError("account_conflict") from None
            conn.executemany(
                "INSERT INTO auth_quotas(user_id,quota_type,daily_limit,usage_day) VALUES(?,?,?,?)",
                [(user_id, kind, limit, now.date().isoformat()) for kind, limit in DEFAULT_QUOTAS[role].items()],
            )
            return self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())

    def list_users(self, *, offset: int, limit: int) -> dict:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM auth_users ORDER BY created_at,user_id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return {"items": [self._user(row).public_view() for row in rows],
                    "total": conn.execute("SELECT COUNT(*) FROM auth_users").fetchone()[0]}

    def open_session(self, user: AuthUser, *, jti: str, expires_at: int) -> AuthUser:
        with self.connection(write=True) as conn:
            current = self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user.user_id,)).fetchone())
            # 密码比较较慢，放在事务外；提交登录时再校验版本，避免改密/停用竞态签出新令牌。
            if not current or not current.is_active or current.token_version != user.token_version or current.hashed_password != user.hashed_password:
                raise AuthError("invalid_credentials")
            conn.execute("DELETE FROM auth_sessions WHERE expires_at<=?", (int(self.clock()),))
            conn.execute("INSERT INTO auth_sessions VALUES(?,?,?,?)", (jti, user.user_id, user.token_version, expires_at))
            conn.execute("UPDATE auth_users SET last_login=? WHERE user_id=?", (self._now().isoformat(), user.user_id))
            return self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user.user_id,)).fetchone())

    def _check_session(self, conn, *, user_id: str, version: int, jti: str) -> AuthUser:
        user = self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())
        session = conn.execute("SELECT * FROM auth_sessions WHERE jti=? AND user_id=?", (jti, user_id)).fetchone()
        if (not user or not user.is_active or user.token_version != version or not session
                or session["token_version"] != version or session["expires_at"] <= self.clock()):
            raise AuthError("invalid_token")
        return user

    def check_session(self, *, user_id: str, version: int, jti: str) -> AuthUser:
        with self.connection() as conn:
            return self._check_session(conn, user_id=user_id, version=version, jti=jti)

    def revoke_session(self, *, user_id: str, jti: str) -> None:
        with self.connection(write=True) as conn:
            conn.execute("DELETE FROM auth_sessions WHERE user_id=? AND jti=?", (user_id, jti))

    def change_password(self, user: AuthUser, *, hashed_password: str) -> None:
        with self.connection(write=True) as conn:
            changed = conn.execute(
                "UPDATE auth_users SET hashed_password=?,token_version=token_version+1 WHERE user_id=? AND hashed_password=? AND token_version=? AND is_active=1",
                (hashed_password, user.user_id, user.hashed_password, user.token_version),
            ).rowcount
            if changed != 1:
                raise AuthError("invalid_token")
            conn.execute("DELETE FROM auth_sessions WHERE user_id=?", (user.user_id,))

    def update_user(self, user_id: str, *, role: str | None = None, is_active: bool | None = None) -> AuthUser:
        with self.connection(write=True) as conn:
            current = self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())
            if current is None:
                raise AuthError("user_not_found")
            new_role = role if role is not None else current.role
            active = current.is_active if is_active is None else is_active
            if new_role not in DEFAULT_QUOTAS or type(active) is not bool:
                raise ValueError("invalid account update")
            if current.role == "admin" and current.is_active and (new_role != "admin" or not active):
                count = conn.execute("SELECT COUNT(*) FROM auth_users WHERE role='admin' AND is_active=1").fetchone()[0]
                if count <= 1:
                    raise AuthError("last_admin_required")
            if new_role != current.role or active != current.is_active:
                conn.execute("UPDATE auth_users SET role=?,is_active=?,token_version=token_version+1 WHERE user_id=?", (new_role, int(active), user_id))
                conn.execute("DELETE FROM auth_sessions WHERE user_id=?", (user_id,))
                if new_role != current.role:
                    # 角色切换应用新角色限额，但保留当日已用量，避免来回切换角色刷配额。
                    conn.executemany("UPDATE auth_quotas SET daily_limit=? WHERE user_id=? AND quota_type=?",
                                     [(limit, user_id, kind) for kind, limit in DEFAULT_QUOTAS[new_role].items()])
            return self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())

    def set_quota(self, user_id: str, *, quota_type: str, daily_limit: int) -> None:
        if quota_type not in QUOTA_TYPES or type(daily_limit) is not int or not -1 <= daily_limit <= 1000000:
            raise ValueError("invalid quota")
        with self.connection(write=True) as conn:
            if conn.execute("UPDATE auth_quotas SET daily_limit=? WHERE user_id=? AND quota_type=?", (daily_limit, user_id, quota_type)).rowcount != 1:
                raise AuthError("user_not_found")

    def quotas(self, user_id: str) -> list[dict]:
        now = self._now()
        reset = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM auth_quotas WHERE user_id=? ORDER BY quota_type", (user_id,)).fetchall()
        if len(rows) != len(QUOTA_TYPES):
            raise AuthError("security_storage_unavailable", retry_after=5)
        result = []
        for row in rows:
            # 查询只投影当天用量；窗口固定在 UTC 零点，不因为 /me 轮询而滚动延后。
            used = row["used_today"] if row["usage_day"] == now.date().isoformat() else 0
            limit = row["daily_limit"]
            result.append({"quota_type": row["quota_type"], "daily_limit": limit, "used_today": used,
                           "remaining": -1 if limit == -1 else max(0, limit - used), "reset_at": reset.isoformat()})
        return result

    def consume(self, *, user_id: str, version: int, jti: str, amounts: dict[str, int]) -> None:
        if not amounts or any(kind not in QUOTA_TYPES or type(amount) is not int or amount <= 0 for kind, amount in amounts.items()):
            raise ValueError("quota amounts must be positive integers")
        with self.connection(write=True) as conn:
            self._check_session(conn, user_id=user_id, version=version, jti=jti)
            # 等待写锁可能跨过 UTC 零点；取得锁后再取日期，避免旧请求把已重置的用量写回前一天。
            now = self._now()
            day = now.date().isoformat()
            reset = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
            for kind, amount in amounts.items():
                row = conn.execute("SELECT * FROM auth_quotas WHERE user_id=? AND quota_type=?", (user_id, kind)).fetchone()
                if row is None:
                    raise AuthError("security_storage_unavailable", retry_after=5)
                used = row["used_today"] if row["usage_day"] == day else 0
                if row["daily_limit"] != -1 and used + amount > row["daily_limit"]:
                    # 任一额度不足会回滚整个事务，不能部分扣费，也不把拒绝请求算作业务用量。
                    raise AuthError("quota_exceeded", retry_after=max(1, int((reset - now).total_seconds()) + 1), quota_type=kind)
                conn.execute("UPDATE auth_quotas SET used_today=?,usage_day=? WHERE user_id=? AND quota_type=?", (used + amount, day, user_id, kind))
