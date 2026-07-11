import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

import logging

from utils.config import SQLITE_CONFIG
from utils.storage_paths import BACKEND_DATA_ROOT, resolve_storage_path

logger = logging.getLogger(__name__)


class SqliteConnectionProvider:
    """统一创建 SQLite 短连接，让业务 store 专注维护自己的事务不变量。"""

    def __init__(self, db_path: str | None = None, check_same_thread: bool | None = None) -> None:
        configured_path = db_path or SQLITE_CONFIG["database_path"]
        # 显式注入路径也走统一规则，防止新调用点绕开配置层重新写入错误目录。
        self.db_path = resolve_storage_path(
            configured_path,
            default_path=BACKEND_DATA_ROOT / "recommendation.db",
            option_name="db_path" if db_path else "SQLITE_DATABASE_PATH",
        )
        self.check_same_thread = (
            SQLITE_CONFIG["check_same_thread"] if check_same_thread is None else check_same_thread
        )
        self.ensure_database_directory()

    def ensure_database_directory(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info("Created database directory: %s", db_dir)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, check_same_thread=self.check_same_thread)
        try:
            yield connection
        finally:
            connection.close()
