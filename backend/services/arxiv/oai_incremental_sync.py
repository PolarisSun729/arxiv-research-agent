"""arXiv OAI 增量同步的按天编排：断点续传、游标推进、运行摘要和单实例锁。

一次运行把 [起点, 截止日] 拆成逐天的 ListRecords 请求；某一天的全部页抓完并落库后，
立刻原子写入游标 = 这一天。中断或失败时游标停在最后一个成功日，下次运行从失败的那天接着同步。
不持久化 resumptionToken：arXiv 的 token 会过期，而一天通常只有 1~3 页，按天续传的损失可以忽略。

游标语义：游标日期及之前的所有日期都已完整同步。因此
- 某一天失败必须立即停止，不能跳过它继续同步后面的日期；
- 手动补一段与游标不相连的旧区间时，不推进游标，否则会把中间的缺口当成已完成。
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# 参数错误/无法确定起点，与“同步失败(1)”区分开，方便外层脚本和 systemd 日志排查。
EXIT_SUCCESS = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


class SyncLockError(RuntimeError):
    """另一个同步进程正在运行。"""


@dataclass(frozen=True)
class IncrementalSyncOptions:
    # 显式起止日期；from 为空时由游标或库内最大 datestamp 推断，until 为空时取昨天。
    from_date: Optional[date] = None
    until_date: Optional[date] = None
    # 游标文件：单行 YYYY-MM-DD。为空表示纯手动运行，既不读也不写游标。
    state_file: Optional[Path] = None
    # 首页 /sync-status 读取的运行摘要；为空时不写，避免手动补数据覆盖最近一次增量同步的信息。
    meta_file: Optional[Path] = None
    # 单实例锁；只在真正写库时持有。
    lock_file: Optional[Path] = None
    dry_run: bool = False
    count_only: bool = False

    @property
    def writes_database(self) -> bool:
        return not (self.dry_run or self.count_only)

    @property
    def mode(self) -> str:
        return "count_only" if self.count_only else "dry_run" if self.dry_run else "sync"


@contextmanager
def exclusive_sync_lock(lock_file: Path) -> Iterator[None]:
    """非阻塞地独占锁文件；进程退出（包括被杀）时操作系统自动释放，不会留下需要手动清理的死锁。"""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_file, "a+")
    try:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SyncLockError(f"另一个 arXiv OAI 同步正在运行（锁文件 {lock_file}）") from None
        yield
    finally:
        handle.close()


def read_cursor(state_file: Path) -> Optional[date]:
    """读取游标；文件不存在或内容不是合法日期时返回 None（后者额外告警）。"""
    try:
        text = state_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        logger.warning("Ignoring invalid sync cursor %r in %s", text, state_file)
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写同目录临时文件再替换，读取方（首页、下一次同步）永远看不到半截内容。
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_cursor(state_file: Path, day: date) -> None:
    _atomic_write_text(state_file, f"{day.isoformat()}\n")


def _add_summaries(total: Dict[str, Any], day_summary: Dict[str, Any]) -> None:
    for key, value in day_summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value


class _Run:
    """单次运行的可变状态，负责把进度写进 meta。"""

    def __init__(self, options: IncrementalSyncOptions, from_date: date, until_date: date) -> None:
        self.options = options
        self.from_date = from_date
        self.until_date = until_date
        self.started_at = datetime.now().astimezone()
        self.days_total = (until_date - from_date).days + 1
        self.days_completed = 0
        self.last_completed: Optional[date] = None
        self.current: Optional[date] = None
        self.summary: Dict[str, Any] = {}
        self.error_message: Optional[str] = None

    def write_meta(self, status: str, exit_code: Optional[int] = None) -> None:
        if not self.options.meta_file:
            return
        payload = {
            "status": status,
            "mode": self.options.mode,
            "from_date": self.from_date.isoformat(),
            "until_date": self.until_date.isoformat(),
            "started_at": self.started_at.isoformat(),
            "finished_at": None if status == "running" else datetime.now().astimezone().isoformat(),
            "exit_code": exit_code,
            # 本次运行中连续完成到的最后一天；首页据此显示“已同步到”。
            "last_successful_until": self.last_completed.isoformat() if self.last_completed else None,
            "current_date": self.current.isoformat() if status == "running" and self.current else None,
            "days_total": self.days_total,
            "days_completed": self.days_completed,
            "records_written": int(self.summary.get("records_written", 0) or 0),
            "records_matched": int(self.summary.get("records_matched", 0) or 0),
            "errors": int(self.summary.get("errors", 0) or 0),
            "error_message": self.error_message,
            "summary": self.summary,
        }
        try:
            _atomic_write_text(self.options.meta_file, json.dumps(payload, ensure_ascii=False, indent=2))
        except OSError:
            # 摘要只用于展示，写失败不能影响同步本身和游标。
            logger.exception("Failed to write sync metadata file: %s", self.options.meta_file)


def _resolve_range(
    options: IncrementalSyncOptions,
    *,
    latest_datestamp: Callable[[], Optional[date]],
    today: date,
) -> tuple[Optional[date], date, Optional[date]]:
    """返回 (起点, 截止日, 基准游标)。起点为 None 表示无法确定，由调用方按参数错误处理。"""
    until_date = options.until_date or today - timedelta(days=1)
    cursor = read_cursor(options.state_file) if options.state_file else None
    if options.state_file and cursor is None:
        latest = latest_datestamp()
        if latest is not None:
            # 库里最大 datestamp 那天可能只同步了一部分，把基准退一天，让那天重新同步（upsert 幂等）。
            cursor = latest - timedelta(days=1)
            logger.warning(
                "Sync cursor missing in %s; inferred %s from latest OAI datestamp %s",
                options.state_file,
                cursor,
                latest,
            )

    from_date = options.from_date or (cursor + timedelta(days=1) if cursor else None)
    if cursor is None and from_date is not None and options.state_file:
        # 空库首次同步：以显式起点为基准，同步成功后游标就能接上。
        cursor = from_date - timedelta(days=1)
    return from_date, until_date, cursor


def run_incremental_sync(
    options: IncrementalSyncOptions,
    *,
    sync_day: Callable[[str], Any],
    latest_datestamp: Callable[[], Optional[date]],
    today: Optional[date] = None,
    pause: Callable[[], None] = lambda: None,
) -> int:
    """按天执行同步并返回进程退出码。sync_day(YYYY-MM-DD) 返回带 to_dict() 的统计对象。"""
    if options.from_date is None and options.state_file is None:
        logger.error("Either --from or --state-file is required.")
        return EXIT_USAGE
    if options.from_date and options.until_date and options.from_date > options.until_date:
        logger.error("--from must be less than or equal to --until")
        return EXIT_USAGE

    with ExitStack() as stack:
        if options.writes_database and options.lock_file:
            try:
                stack.enter_context(exclusive_sync_lock(options.lock_file))
            except SyncLockError as exc:
                logger.error("%s", exc)
                return EXIT_FAILED

        from_date, until_date, cursor = _resolve_range(
            options, latest_datestamp=latest_datestamp, today=today or date.today()
        )
        if from_date is None:
            logger.error("No sync cursor and the OAI database is empty; pass --from to choose a start date.")
            return EXIT_USAGE
        if from_date > until_date:
            logger.info("No new full day to sync; cursor=%s latest complete day=%s", cursor, until_date)
            return EXIT_SUCCESS
        return _run_days(options, from_date, until_date, cursor, sync_day=sync_day, pause=pause)


def _run_days(
    options: IncrementalSyncOptions,
    from_date: date,
    until_date: date,
    cursor: Optional[date],
    *,
    sync_day: Callable[[str], Any],
    pause: Callable[[], None],
) -> int:
    run = _Run(options, from_date, until_date)
    advance_cursor = options.writes_database and options.state_file is not None
    logger.info(
        "Starting daily arXiv OAI sync: from=%s until=%s days=%s cursor=%s mode=%s",
        from_date,
        until_date,
        run.days_total,
        cursor,
        options.mode,
    )

    day = from_date
    try:
        while day <= until_date:
            run.current = day
            run.write_meta("running")
            try:
                day_summary = dict(sync_day(day.isoformat()).to_dict())
            except Exception as exc:
                logger.exception("arXiv OAI sync failed on %s", day)
                day_summary = {"errors": 1}
                run.error_message = f"{day.isoformat()} 同步失败：{exc}"
            _add_summaries(run.summary, day_summary)

            if int(day_summary.get("errors", 0) or 0) > 0:
                run.error_message = run.error_message or f"{day.isoformat()} 同步失败，详见同步日志。"
                logger.error("Stopping at %s; cursor stays at %s", day, cursor)
                run.write_meta("failed", EXIT_FAILED)
                return EXIT_FAILED

            run.days_completed += 1
            run.last_completed = day
            # 只有与游标相连时才推进；cursor 之前的日期重跑不回退游标。
            if advance_cursor and cursor is not None and day <= cursor + timedelta(days=1):
                if day > cursor:
                    cursor = day
                    write_cursor(options.state_file, cursor)
            logger.info(
                "Day %s synced (%s/%s): written=%s matched=%s cursor=%s",
                day,
                run.days_completed,
                run.days_total,
                day_summary.get("records_written", 0),
                day_summary.get("records_matched", 0),
                cursor,
            )
            day += timedelta(days=1)
            if day <= until_date:
                pause()
    except KeyboardInterrupt:
        run.error_message = f"同步在 {day.isoformat()} 被中断，游标停在 {cursor}，下次运行会从中断的那天继续。"
        logger.warning("%s", run.error_message)
        run.write_meta("failed", EXIT_INTERRUPTED)
        return EXIT_INTERRUPTED

    logger.info("Daily arXiv OAI sync finished: %s", run.summary)
    print(json.dumps(run.summary, ensure_ascii=False, indent=2))
    run.write_meta("success", EXIT_SUCCESS)
    return EXIT_SUCCESS
