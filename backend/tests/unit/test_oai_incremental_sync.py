from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from services.arxiv.oai_incremental_sync import (
    IncrementalSyncOptions,
    exclusive_sync_lock,
    run_incremental_sync,
)

TODAY = date(2026, 9, 14)


class FakeStats:
    def __init__(self, *, written: int = 1, errors: int = 0) -> None:
        self.data = {"records_written": written, "records_matched": written, "errors": errors, "requests_made": 1}

    def to_dict(self) -> Dict[str, int]:
        return dict(self.data)


class FakeDaySync:
    """按天记录调用；failures 中的日期返回 errors=1，raises 中的日期抛出指定异常。"""

    def __init__(
        self,
        *,
        failures: tuple[str, ...] = (),
        raises: Optional[Dict[str, BaseException]] = None,
        on_call: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.calls: List[str] = []
        self.failures = failures
        self.raises = raises or {}
        self.on_call = on_call

    def __call__(self, day: str) -> FakeStats:
        self.calls.append(day)
        if self.on_call:
            self.on_call(day)
        if day in self.raises:
            raise self.raises[day]
        return FakeStats(errors=1 if day in self.failures else 0)


def _options(tmp_path: Path, **overrides) -> IncrementalSyncOptions:
    values = dict(
        state_file=tmp_path / "sync.state",
        meta_file=tmp_path / "sync.meta.json",
        lock_file=tmp_path / "sync.lock",
    )
    values.update(overrides)
    return IncrementalSyncOptions(**values)


def _run(options: IncrementalSyncOptions, sync_day: FakeDaySync, *, latest: Optional[date] = None) -> int:
    return run_incremental_sync(options, sync_day=sync_day, latest_datestamp=lambda: latest, today=TODAY)


def _state(tmp_path: Path) -> str:
    return (tmp_path / "sync.state").read_text(encoding="utf-8").strip()


def _meta(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "sync.meta.json").read_text(encoding="utf-8"))


def test_syncs_each_day_after_cursor_and_advances_cursor(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    sync_day = FakeDaySync()

    assert _run(_options(tmp_path), sync_day) == 0

    # 每天单独一次 ListRecords，区间截止到昨天（今天未结束，不同步）。
    assert sync_day.calls == ["2026-09-11", "2026-09-12", "2026-09-13"]
    assert _state(tmp_path) == "2026-09-13"
    meta = _meta(tmp_path)
    assert meta["status"] == "success"
    assert meta["last_successful_until"] == "2026-09-13"
    assert meta["days_completed"] == 3
    assert meta["records_written"] == 3


def test_failed_day_stops_run_and_keeps_cursor_at_last_success(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    sync_day = FakeDaySync(failures=("2026-09-12",))

    assert _run(_options(tmp_path), sync_day) == 1

    # 游标表示“这一天及之前全部同步完成”，失败日之后的日期不能跳着同步。
    assert sync_day.calls == ["2026-09-11", "2026-09-12"]
    assert _state(tmp_path) == "2026-09-11"
    meta = _meta(tmp_path)
    assert meta["status"] == "failed"
    assert meta["last_successful_until"] == "2026-09-11"
    assert "2026-09-12" in meta["error_message"]


def test_next_run_resumes_from_failed_day(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    _run(_options(tmp_path), FakeDaySync(failures=("2026-09-12",)))

    sync_day = FakeDaySync()
    assert _run(_options(tmp_path), sync_day) == 0

    assert sync_day.calls == ["2026-09-12", "2026-09-13"]
    assert _state(tmp_path) == "2026-09-13"


def test_exception_during_day_is_treated_as_failed_day(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    sync_day = FakeDaySync(raises={"2026-09-12": RuntimeError("database is locked")})

    assert _run(_options(tmp_path), sync_day) == 1

    assert _state(tmp_path) == "2026-09-11"
    assert "database is locked" in _meta(tmp_path)["error_message"]


def test_interrupt_keeps_completed_days_and_marks_meta_failed(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    sync_day = FakeDaySync(raises={"2026-09-12": KeyboardInterrupt()})

    assert _run(_options(tmp_path), sync_day) == 130

    assert _state(tmp_path) == "2026-09-11"
    meta = _meta(tmp_path)
    assert meta["status"] == "failed"
    assert "中断" in meta["error_message"]


def test_meta_reports_running_progress_while_syncing(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    snapshots: List[dict] = []
    sync_day = FakeDaySync(on_call=lambda day: snapshots.append(_meta(tmp_path)))

    _run(_options(tmp_path), sync_day)

    second_day = snapshots[1]
    assert second_day["status"] == "running"
    assert second_day["current_date"] == "2026-09-12"
    assert second_day["days_completed"] == 1
    assert second_day["days_total"] == 3
    assert second_day["last_successful_until"] == "2026-09-11"


def test_missing_cursor_falls_back_to_latest_datestamp_in_database(tmp_path: Path) -> None:
    sync_day = FakeDaySync()

    # 库里最后一天可能只同步了一部分，所以从最大 datestamp 当天重新开始。
    assert _run(_options(tmp_path), sync_day, latest=date(2026, 9, 12)) == 0

    assert sync_day.calls == ["2026-09-12", "2026-09-13"]
    assert _state(tmp_path) == "2026-09-13"


def test_invalid_cursor_falls_back_to_latest_datestamp(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("garbage\n", encoding="utf-8")
    sync_day = FakeDaySync()

    assert _run(_options(tmp_path), sync_day, latest=date(2026, 9, 13)) == 0

    assert sync_day.calls == ["2026-09-13"]


def test_missing_cursor_and_empty_database_requires_explicit_from(tmp_path: Path) -> None:
    sync_day = FakeDaySync()

    assert _run(_options(tmp_path), sync_day, latest=None) == 2

    assert sync_day.calls == []
    assert not (tmp_path / "sync.state").exists()


def test_nothing_to_sync_when_cursor_is_yesterday(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-13\n", encoding="utf-8")
    sync_day = FakeDaySync()

    assert _run(_options(tmp_path), sync_day) == 0

    assert sync_day.calls == []
    # 没有新的完整日期时不覆盖首页上一次的同步摘要。
    assert not (tmp_path / "sync.meta.json").exists()


def test_backfill_of_older_range_does_not_move_cursor(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    sync_day = FakeDaySync()
    options = _options(tmp_path, from_date=date(2026, 8, 1), until_date=date(2026, 8, 2))

    assert _run(options, sync_day) == 0

    assert sync_day.calls == ["2026-08-01", "2026-08-02"]
    assert _state(tmp_path) == "2026-09-10"


def test_backfill_overlapping_cursor_advances_it(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    options = _options(tmp_path, from_date=date(2026, 9, 9), until_date=date(2026, 9, 12))

    assert _run(options, FakeDaySync()) == 0

    assert _state(tmp_path) == "2026-09-12"


def test_backfill_leaving_gap_after_cursor_does_not_move_cursor(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-01\n", encoding="utf-8")
    options = _options(tmp_path, from_date=date(2026, 9, 10), until_date=date(2026, 9, 12))

    assert _run(options, FakeDaySync()) == 0

    # 09-02 ~ 09-09 没同步过，推进游标会把缺口当成已完成。
    assert _state(tmp_path) == "2026-09-01"


def test_explicit_from_after_until_is_rejected(tmp_path: Path) -> None:
    options = _options(tmp_path, from_date=date(2026, 9, 12), until_date=date(2026, 9, 11))
    sync_day = FakeDaySync()

    assert _run(options, sync_day) == 2
    assert sync_day.calls == []


@pytest.mark.parametrize("mode", ["dry_run", "count_only"])
def test_preview_modes_never_move_cursor(tmp_path: Path, mode: str) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    sync_day = FakeDaySync()

    assert _run(_options(tmp_path, **{mode: True}), sync_day) == 0

    assert sync_day.calls == ["2026-09-11", "2026-09-12", "2026-09-13"]
    assert _state(tmp_path) == "2026-09-10"


def test_concurrent_run_is_rejected_by_lock(tmp_path: Path) -> None:
    (tmp_path / "sync.state").write_text("2026-09-10\n", encoding="utf-8")
    sync_day = FakeDaySync()

    with exclusive_sync_lock(tmp_path / "sync.lock"):
        assert _run(_options(tmp_path), sync_day) == 1

    assert sync_day.calls == []
    assert _state(tmp_path) == "2026-09-10"
    # 锁释放后可以正常运行。
    assert _run(_options(tmp_path), sync_day) == 0


def test_manual_run_without_state_file_requires_from(tmp_path: Path) -> None:
    options = IncrementalSyncOptions(lock_file=tmp_path / "sync.lock")
    sync_day = FakeDaySync()

    assert run_incremental_sync(options, sync_day=sync_day, latest_datestamp=lambda: None, today=TODAY) == 2
    assert sync_day.calls == []
