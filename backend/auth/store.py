"""独立 SQLite 认证存储：账号、会话撤销和 UTC 配额共用原子事务。

表结构：
- auth_users：账号。username/email 唯一且不区分大小写；token_version 用于让旧令牌整体失效；
- auth_quotas：每个用户每类配额一行。daily_limit 为 -1 表示不限；used_today 只在 usage_day 等于
  当天（UTC）时有效，跨天后按 0 计算，下次扣减时再惰性重置，不需要定时任务；
- auth_sessions：已签发且未撤销的登录会话，以 JWT 的 jti 为主键。登出即删除对应行。

每个方法都新开一个短连接，用完即关，因此可以在线程池中并发调用；
多线程、多 worker 之间依靠 SQLite 文件锁保证写操作互斥。
"""

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
    """账号、会话与配额的持久化存储。

    clock 返回当前 Unix 时间戳（秒），默认 time.time；测试中可以注入假时钟来模拟跨天、过期等情况。
    """

    def __init__(self, path: Path, *, clock=None) -> None:
        """准备数据库文件和私有目录，并建表（幂等）；失败时抛出 RuntimeError 阻止启动。"""
        self.path = Path(path)
        self.clock = clock or time.time
        try:
            # 认证库放在独立私有目录中，包含 WAL/SHM 的所有副本也不能被静态服务器公开。
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # 非 Windows 系统上，目录对同组和其他用户必须没有任何权限。
            if os.name != "nt" and self.path.parent.stat().st_mode & 0o077:
                raise OSError
            try:
                # O_EXCL 保证"创建文件"与"设置 0600 权限"一步完成，不存在先以宽松权限创建、再收紧的时间窗口。
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
            except FileExistsError:
                # 已有的数据库文件同样必须只有属主可读写。
                if os.name != "nt" and self.path.stat().st_mode & 0o077:
                    raise OSError
            with self.connection() as conn:
                # WAL 模式允许读操作与一个写事务并发进行，减少登录校验被配额写入阻塞的情况。
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
        """打开一个短连接；write=True 时立即开启写事务，并在 with 块正常结束时提交。

        with 块内抛出任何异常都会跳过 commit，连接关闭时未提交的修改全部回滚。
        SQLite 错误（包括等待锁超过 5 秒）统一转换为 security_storage_unavailable，不向外暴露数据库细节。
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.path), timeout=5)
            # 让查询结果可以按列名访问。
            conn.row_factory = sqlite3.Row
            # SQLite 默认不启用外键约束，需要在每个连接上单独开启，ON DELETE CASCADE 才会生效。
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
        """当前 UTC 时间（基于注入的 clock）。"""
        return datetime.fromtimestamp(self.clock(), timezone.utc)

    @staticmethod
    def _user(row) -> AuthUser | None:
        """把 auth_users 的一行转换为 AuthUser；row 为 None 时返回 None。"""
        if row is None:
            return None
        values = dict(row)
        # SQLite 用 0/1 存布尔值，这里转回 bool。
        values["is_active"] = bool(values["is_active"])
        return AuthUser(**values)

    def get_user(self, user_id: str) -> AuthUser | None:
        """按 user_id 查询账号。"""
        with self.connection() as conn:
            return self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())

    def get_by_username(self, username: str) -> AuthUser | None:
        """按用户名查询账号（不区分大小写）。"""
        with self.connection() as conn:
            return self._user(conn.execute("SELECT * FROM auth_users WHERE username=?", (username.casefold(),)).fetchone())

    def create_user(self, *, username: str, email: str, hashed_password: str, role: str = "guest") -> AuthUser:
        """创建账号，并按角色写入默认配额；用户名或邮箱已存在时抛出 account_conflict。

        hashed_password 必须是已经哈希过的密码（见 passwords.hash_password）。
        """
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
        """分页列出账号（只含公开字段），按创建时间排序，user_id 作为次序键保证分页稳定。"""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM auth_users ORDER BY created_at,user_id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return {"items": [self._user(row).public_view() for row in rows],
                    "total": conn.execute("SELECT COUNT(*) FROM auth_users").fetchone()[0]}

    def open_session(self, user: AuthUser, *, jti: str, expires_at: int) -> AuthUser:
        """登录成功后登记会话，并更新最后登录时间；返回最新的账号记录。

        user 是校验密码前读出的账号快照。
        """
        with self.connection(write=True) as conn:
            current = self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user.user_id,)).fetchone())
            # 密码比较较慢，放在事务外；提交登录时再校验版本，避免改密/停用竞态签出新令牌。
            if not current or not current.is_active or current.token_version != user.token_version or current.hashed_password != user.hashed_password:
                raise AuthError("invalid_credentials")
            # 顺带清理已过期的会话，避免会话表无限增长。
            conn.execute("DELETE FROM auth_sessions WHERE expires_at<=?", (int(self.clock()),))
            conn.execute("INSERT INTO auth_sessions VALUES(?,?,?,?)", (jti, user.user_id, user.token_version, expires_at))
            conn.execute("UPDATE auth_users SET last_login=? WHERE user_id=?", (self._now().isoformat(), user.user_id))
            return self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user.user_id,)).fetchone())

    def _check_session(self, conn, *, user_id: str, version: int, jti: str) -> AuthUser:
        """在给定连接内校验会话，通过时返回最新的账号记录，否则抛出 invalid_token。

        以下任一情况都视为无效：账号不存在或已停用；令牌版本与账号当前版本不一致
        （已改密/改角色/停用过）；会话已被撤销（登出）；会话已过期。
        """
        user = self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())
        session = conn.execute("SELECT * FROM auth_sessions WHERE jti=? AND user_id=?", (jti, user_id)).fetchone()
        if (not user or not user.is_active or user.token_version != version or not session
                or session["token_version"] != version or session["expires_at"] <= self.clock()):
            raise AuthError("invalid_token")
        return user

    def check_session(self, *, user_id: str, version: int, jti: str) -> AuthUser:
        """校验会话是否仍然有效，返回数据库中最新的账号记录（角色以此为准）。"""
        with self.connection() as conn:
            return self._check_session(conn, user_id=user_id, version=version, jti=jti)

    def revoke_session(self, *, user_id: str, jti: str) -> None:
        """撤销单个会话（登出），该令牌此后无法再通过校验。"""
        with self.connection(write=True) as conn:
            conn.execute("DELETE FROM auth_sessions WHERE user_id=? AND jti=?", (user_id, jti))

    def change_password(self, user: AuthUser, *, hashed_password: str) -> None:
        """修改密码，并使该用户所有已签发的令牌失效（包括发起请求的这一个）。

        user 是调用方校验当前密码时读出的账号快照。
        """
        with self.connection(write=True) as conn:
            # 乐观并发控制：只有密码哈希和令牌版本都与快照一致、且账号仍启用时才更新；
            # 期间账号若被其他请求改过密码或停用，影响行数为 0，本次修改失败。
            changed = conn.execute(
                "UPDATE auth_users SET hashed_password=?,token_version=token_version+1 WHERE user_id=? AND hashed_password=? AND token_version=? AND is_active=1",
                (hashed_password, user.user_id, user.hashed_password, user.token_version),
            ).rowcount
            if changed != 1:
                raise AuthError("invalid_token")
            conn.execute("DELETE FROM auth_sessions WHERE user_id=?", (user.user_id,))

    def update_user(self, user_id: str, *, role: str | None = None, is_active: bool | None = None) -> AuthUser:
        """管理员修改账号的角色或启用状态；参数为 None 表示保持不变。返回修改后的账号记录。

        角色或状态发生变化时，递增 token_version 并删除该用户全部会话，强制其重新登录。
        系统中必须至少保留一个启用的管理员。
        """
        with self.connection(write=True) as conn:
            current = self._user(conn.execute("SELECT * FROM auth_users WHERE user_id=?", (user_id,)).fetchone())
            if current is None:
                raise AuthError("user_not_found")
            new_role = role if role is not None else current.role
            active = current.is_active if is_active is None else is_active
            if new_role not in DEFAULT_QUOTAS or type(active) is not bool:
                raise ValueError("invalid account update")
            # 降级或停用一个启用中的管理员前，确认它不是最后一个启用的管理员。
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
        """单独调整某个用户某一类配额的每日上限（-1 表示不限），不影响当日已用量。"""
        if quota_type not in QUOTA_TYPES or type(daily_limit) is not int or not -1 <= daily_limit <= 1000000:
            raise ValueError("invalid quota")
        with self.connection(write=True) as conn:
            # 影响行数不为 1 说明该用户不存在。
            if conn.execute("UPDATE auth_quotas SET daily_limit=? WHERE user_id=? AND quota_type=?", (daily_limit, user_id, quota_type)).rowcount != 1:
                raise AuthError("user_not_found")

    def quotas(self, user_id: str) -> list[dict]:
        """返回用户各类配额的当日使用情况（只读，不修改数据库）。

        每项包含：quota_type、daily_limit、used_today、remaining（不限时为 -1）、
        reset_at（下一个 UTC 零点）。
        """
        now = self._now()
        reset = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM auth_quotas WHERE user_id=? ORDER BY quota_type", (user_id,)).fetchall()
        # 每个用户都应当恰好有每类配额各一行，缺行说明数据不一致，按存储不可用处理。
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
        """在一个写事务中校验会话并扣减配额；任一类型额度不足时整体不扣，抛出 quota_exceeded。

        amounts 为 配额类型 → 扣减数量（正整数）。quota_exceeded 携带 retry_after（距下一个 UTC 零点的秒数）
        和 quota_type（不足的类型）。
        """
        if not amounts or any(kind not in QUOTA_TYPES or type(amount) is not int or amount <= 0 for kind, amount in amounts.items()):
            raise ValueError("quota amounts must be positive integers")
        with self.connection(write=True) as conn:
            # 与扣减在同一事务中检查会话，已登出或已被停用的令牌不能继续消耗配额。
            self._check_session(conn, user_id=user_id, version=version, jti=jti)
            # 等待写锁可能跨过 UTC 零点；取得锁后再取日期，避免旧请求把已重置的用量写回前一天。
            now = self._now()
            day = now.date().isoformat()
            reset = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
            for kind, amount in amounts.items():
                row = conn.execute("SELECT * FROM auth_quotas WHERE user_id=? AND quota_type=?", (user_id, kind)).fetchone()
                if row is None:
                    raise AuthError("security_storage_unavailable", retry_after=5)
                # 记录的日期不是今天，说明已跨天，从 0 开始计。
                used = row["used_today"] if row["usage_day"] == day else 0
                if row["daily_limit"] != -1 and used + amount > row["daily_limit"]:
                    # 任一额度不足会回滚整个事务，不能部分扣费，也不把拒绝请求算作业务用量。
                    raise AuthError("quota_exceeded", retry_after=max(1, int((reset - now).total_seconds()) + 1), quota_type=kind)
                # 同时写入 usage_day，完成跨天后的惰性重置。
                conn.execute("UPDATE auth_quotas SET used_today=?,usage_day=? WHERE user_id=? AND quota_type=?", (used + amount, day, user_id, kind))
