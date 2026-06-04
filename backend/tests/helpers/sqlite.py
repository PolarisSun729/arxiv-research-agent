"""Lightweight SQLite helpers for backend unittest suites."""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Type, TypeVar


T = TypeVar("T")


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


def build_database_service(
    service_cls: Type[T],
    *,
    db_path: Optional[str] = None,
    check_same_thread: bool = False,
    initialize: bool = True,
    **extra_attributes: Any,
) -> T:
    """Instantiate a database-like service against a temporary SQLite path.

    This bypasses constructors that hard-code production config while still
    reusing the service's own initialization hooks.
    """

    temp_db: Optional[TemporarySqliteDatabase] = None
    resolved_path = db_path
    if not resolved_path:
        temp_db = TemporarySqliteDatabase()
        resolved_path = str(temp_db.db_path)

    service = service_cls.__new__(service_cls)
    setattr(service, "db_path", resolved_path)
    setattr(service, "check_same_thread", check_same_thread)
    setattr(service, "_test_temp_db", temp_db)

    for key, value in extra_attributes.items():
        setattr(service, key, value)

    ensure_dir = getattr(service, "_ensure_database_directory", None)
    if callable(ensure_dir):
        ensure_dir()

    if initialize:
        initialize_db = getattr(service, "_initialize_database", None)
        if callable(initialize_db):
            initialize_db()

    return service
