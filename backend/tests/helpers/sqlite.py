"""Lightweight SQLite helpers for backend unittest suites."""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from services.storage.sqlite import StorageContainer
from services.storage.sqlite.connection import SqliteConnectionProvider


class TemporarySqliteDatabase:
    """Create an isolated temporary SQLite database on disk."""

    def __init__(self, file_name: str = "test.sqlite3") -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="backend-test-db-")
        self.root_path = Path(self._temp_dir.name)
        self.db_path = self.root_path / file_name

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            yield connection
        finally:
            connection.close()

    def cleanup(self) -> None:
        self._temp_dir.cleanup()

    def __enter__(self) -> "TemporarySqliteDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()


def build_storage_container(
    *,
    db_path: Optional[str] = None,
    check_same_thread: bool = False,
    initialize_schema: bool = True,
) -> StorageContainer:
    """创建指向临时 SQLite 文件的真实存储组合根。

    测试只负责提供隔离数据库路径，schema 初始化仍由生产 StorageContainer 执行；
    这样可以验证真实 Store 行为，而不是继续依赖旧的统一存储入口。
    """
    temp_db: Optional[TemporarySqliteDatabase] = None
    resolved_path = db_path
    if not resolved_path:
        temp_db = TemporarySqliteDatabase()
        resolved_path = str(temp_db.db_path)

    connection_provider = SqliteConnectionProvider(
        db_path=str(resolved_path),
        check_same_thread=check_same_thread,
    )
    storage = StorageContainer(
        connection_provider=connection_provider,
        initialize_schema=initialize_schema,
    )
    setattr(storage, "_test_temp_db", temp_db)
    return storage
