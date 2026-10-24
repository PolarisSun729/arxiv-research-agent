"""论文向量维护任务：在 Web 进程内补齐保留窗口内缺向量的论文，并每天清理一次窗口外的向量。

为什么放在 Web 进程:
    生产用的 Milvus Lite 在进程内运行，打开即对数据目录加独占锁，进程退出才释放。Web 进程要读论文向量做推荐，
    一旦打开就持锁到重启，别的进程（如每日 OAI 同步）再打开都会失败。所以论文向量只由 Web 进程写：
    同步只写 SQLite，这里按轮次把新入库的论文补上向量。

运行方式:
    - 单个后台线程，lifespan 启动、退出时停止；启动后延迟一段时间再跑第一轮，避开预热高峰；
    - 每轮先一次取出向量库已有的 arxiv_id，再按入库顺序从新到旧扫描窗口内论文，只给没有向量的论文生成 embedding，一次只发一批请求；
    - 不存游标：是否缺向量始终以向量库为准，中断或重启后下一轮重新扫描即可接着补，已写的不会重复花钱；
    - 单轮写入达到上限就提前结束、短暂休息后继续，长时间的补齐被拆成可中断的短轮次；
    - 整批失败（接口报错、限流、向量库写入失败）按 1、2、4…分钟退避，最长 30 分钟；
    - 同一篇累计失败 3 次就当天不再尝试（换天或重启清空），免得一篇坏数据卡在最前面、让整轮反复退避；
    - 扫完整个窗口后，每天最多清理一次窗口外向量（点赞 / 点踩过的论文保留）；
    - 除了定时，同步游标文件被更新（每日同步成功推进）或管理员手动触发时也会提前开始下一轮。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from services.arxiv.arxiv_embedding_handler import ArxivEmbeddingHandler, EmbeddingStats
from utils.logging_utils import info_event

logger = logging.getLogger(__name__)

# 退避从 1 分钟起翻倍，封顶 30 分钟。
BACKOFF_BASE_SECONDS = 60.0
BACKOFF_MAX_SECONDS = 1800.0
# 单轮触顶上限后，休息这么久再开下一轮，给停止信号和管理员查看留出间隙。
CATCH_UP_PAUSE_SECONDS = 10.0
# 等待下一轮期间，每隔这么久检查一次同步游标文件有没有更新。
SYNC_CHECK_INTERVAL_SECONDS = 60.0
# 同一篇论文当天累计失败这么多次后，当天不再尝试。
MAX_PAPER_FAILURES = 3
# 连续这么多轮出错时写 ERROR 日志，而不是只累加计数。
ERROR_ROUNDS_ALERT_THRESHOLD = 3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _current_rss_mb() -> Optional[float]:
    """当前进程常驻内存（MB）；只在 Linux 上可读，其他平台返回 None。用来观察向量增多后 Web 进程的内存走势。"""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


class OaiEmbeddingMaintainer:
    """论文向量维护任务；run_round 可单独调用（测试、手动补一轮），start/stop 管理常驻线程。"""

    def __init__(
        self,
        *,
        handler_factory: Callable[[], ArxivEmbeddingHandler],
        database_service_factory: Callable[[], Any],
        prune_out_of_window: Callable[[ArxivEmbeddingHandler], int],
        sync_state_file: Optional[Path] = None,
        startup_delay_seconds: float = 120.0,
        interval_seconds: float = 1800.0,
        max_per_round: int = 1000,
        page_size: int = 500,
        today: Callable[[], date] = date.today,
    ) -> None:
        """参数:
        handler_factory: 每轮新建 embedding 处理器；保留窗口起点在创建时按当天计算，跨天运行也不会过期。
        database_service_factory: 返回 OAI SQLite 服务，用来分页读取窗口内论文。
        prune_out_of_window: 删除窗口外向量并返回条数（内部会读取需要保留的点赞 / 点踩论文）。
        sync_state_file: 每日同步的游标文件；修改时间变新说明有新论文入库，提前开始下一轮。
        """
        self._handler_factory = handler_factory
        self._database_service_factory = database_service_factory
        self._prune_out_of_window = prune_out_of_window
        self._sync_state_file = sync_state_file
        self.startup_delay_seconds = max(0.0, float(startup_delay_seconds))
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.max_per_round = max(1, int(max_per_round))
        self.page_size = max(1, int(page_size))
        self._today = today

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        # 只保护状态快照；轮次本身由单个工作线程串行执行。
        self._status_lock = threading.Lock()
        self._round_running = False
        self._consecutive_failed_batches = 0
        self._consecutive_error_rounds = 0
        self._paper_failures: Dict[str, int] = {}
        self._skipped_arxiv_ids: set[str] = set()
        self._failures_date: Optional[date] = None
        self._last_prune_date: Optional[date] = None
        self._last_round_started_monotonic: Optional[float] = None
        self._status: Dict[str, Any] = {
            "started_at": None,
            "last_round": None,
            "last_full_scan": None,
            "last_prune": None,
            "last_error": None,
            "next_round_at": None,
            "written_since_start": 0,
            "cost_yuan_since_start": 0.0,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动常驻线程；重复调用保持幂等。"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        with self._status_lock:
            self._status["started_at"] = _utc_now_iso()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="oai-embedding-maintainer")
        self._worker_thread.start()

    def stop(self, *, join_timeout_seconds: float = 5.0) -> None:
        """停止任务：所有等待立即打断，正在进行的一批（最多一次 embedding 请求）写完后退出。"""
        self._stop_event.set()
        self._wake_event.set()
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, float(join_timeout_seconds)))

    def request_run(self) -> bool:
        """请求尽快开始下一轮（管理员手动触发）；返回常驻线程是否在运行，不在运行时请求不会生效。"""
        self._wake_event.set()
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def get_status(self) -> Dict[str, Any]:
        """返回给管理员接口展示的状态快照。"""
        with self._status_lock:
            snapshot = {key: (dict(value) if isinstance(value, dict) else value) for key, value in self._status.items()}
            snapshot["running"] = self._worker_thread is not None and self._worker_thread.is_alive()
            snapshot["round_in_progress"] = self._round_running
            snapshot["skipped_papers"] = len(self._skipped_arxiv_ids)
        return snapshot

    def _worker_loop(self) -> None:
        if self._stop_event.wait(self.startup_delay_seconds):
            return
        while not self._stop_event.is_set():
            try:
                result = self.run_round()
                delay = self._next_delay(result["status"])
            except Exception as exc:
                # 读 SQLite、查向量库或清理失败都不能让常驻线程退出，记录后按退避重试。
                logger.exception("OAI embedding maintenance round failed")
                self._record_error(f"{exc.__class__.__name__}: {exc}")
                self._consecutive_failed_batches += 1
                delay = self._backoff_seconds()
            self._wait_for_next_round(delay)

    def _next_delay(self, status: str) -> float:
        if status == "backoff":
            return self._backoff_seconds()
        if status == "limit_reached":
            return CATCH_UP_PAUSE_SECONDS
        return self.interval_seconds

    def _backoff_seconds(self) -> float:
        exponent = max(0, self._consecutive_failed_batches - 1)
        return min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2**exponent))

    def _wait_for_next_round(self, delay_seconds: float) -> None:
        """等到下一轮的时间；停止、手动触发或同步游标更新都会提前结束等待。"""
        deadline = time.monotonic() + delay_seconds
        with self._status_lock:
            self._status["next_round_at"] = datetime.fromtimestamp(time.time() + delay_seconds, timezone.utc).isoformat(timespec="seconds")
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._wake_event.wait(min(remaining, SYNC_CHECK_INTERVAL_SECONDS)):
                self._wake_event.clear()
                return
            if self._sync_state_updated_since_last_round():
                logger.info("OAI sync cursor advanced, starting embedding maintenance round early")
                return

    def _sync_state_updated_since_last_round(self) -> bool:
        if self._sync_state_file is None or self._last_round_started_monotonic is None:
            return False
        try:
            modified_at = self._sync_state_file.stat().st_mtime
        except OSError:
            return False
        # 游标文件记的是墙钟时间，这里换算成本轮开始时的墙钟时间再比较。
        round_started_wall = time.time() - (time.monotonic() - self._last_round_started_monotonic)
        return modified_at > round_started_wall

    # ------------------------------------------------------------------
    # 一轮
    # ------------------------------------------------------------------

    def run_round(self) -> Dict[str, Any]:
        """跑一轮：扫描窗口内论文补向量，扫完整个窗口时按天清理窗口外向量；返回本轮摘要。

        status: covered（扫完整个窗口）/ limit_reached（达到单轮上限）/ backoff（整批失败，需退避）/ stopped（收到停止信号）。
        """
        started = time.perf_counter()
        self._last_round_started_monotonic = time.monotonic()
        if self._failures_date != self._today():
            # 跳过名单按天清空：接口长时间故障时被连带跳过的论文，第二天会重新尝试。
            self._paper_failures.clear()
            self._skipped_arxiv_ids.clear()
            self._failures_date = self._today()
        with self._status_lock:
            self._round_running = True
        try:
            result = self._scan_and_embed()
            if result["status"] == "covered":
                self._maybe_prune(result)
        finally:
            with self._status_lock:
                self._round_running = False
        result["duration_seconds"] = round(time.perf_counter() - started, 1)
        result["rss_mb"] = _current_rss_mb()
        result["finished_at"] = _utc_now_iso()
        self._record_round(result)
        return result

    def _scan_and_embed(self) -> Dict[str, Any]:
        handler = self._handler_factory()
        database_service = self._database_service_factory()
        window_start = handler.min_paper_date
        stats = EmbeddingStats()
        scanned = 0
        missing_found = 0
        status = "covered"
        # 开轮时一次取出向量库里已有的 arxiv_id，扫描时在内存里比对：逐页按 id 查向量库每次都要筛全表，已补的越多越慢。
        # 本轮期间别处新写的论文可能仍被当成缺失，写入前 store_paper_embeddings_batch 还会再查一次，不会重复生成。
        embedded_arxiv_ids = handler.list_embedded_arxiv_ids()
        for page in database_service.iter_papers_newest_first(since=window_start, page_size=self.page_size):
            if self._stop_event.is_set():
                status = "stopped"
                break
            scanned += len(page)
            missing_papers = [
                paper
                for paper in handler.find_papers_missing_embeddings(page, embedded_arxiv_ids=embedded_arxiv_ids)
                if self._paper_id(paper) not in self._skipped_arxiv_ids
            ]
            missing_found += len(missing_papers)
            status = self._embed_missing_papers(handler, missing_papers, stats)
            if status != "covered":
                break
        return {"status": status, "window_start": window_start, "scanned": scanned, "missing_found": missing_found, **asdict(stats)}

    def _embed_missing_papers(self, handler: ArxivEmbeddingHandler, papers: List[Dict[str, Any]], stats: EmbeddingStats) -> str:
        """按 embedding 批大小逐批写入；一次只发一批请求，不和用户请求抢 embedding 并发。返回 covered 表示这一页处理完。"""
        batch_size = max(1, int(handler.embedding_batch_size))
        for start in range(0, len(papers), batch_size):
            if self._stop_event.is_set():
                return "stopped"
            remaining = self.max_per_round - stats.embeddings_attempted
            if remaining <= 0:
                return "limit_reached"
            batch = papers[start : start + min(batch_size, remaining)]
            written_before, errors_before = stats.embeddings_written, stats.embedding_errors
            handler.store_paper_embeddings_batch(batch, stats)
            written = stats.embeddings_written - written_before
            errors = stats.embedding_errors - errors_before
            if errors and not written:
                # 整批都没写进去，多半是接口或向量库出了问题，继续下一批也会失败：结束本轮，退避后重试。
                # 也可能是这批里全是坏数据，所以照样给每篇记一次失败，避免它们永远卡在最前面。
                self._count_paper_failures(batch)
                self._consecutive_failed_batches += 1
                return "backoff"
            self._consecutive_failed_batches = 0
            if errors:
                self._record_partial_failures(handler, batch)
        if self.max_per_round - stats.embeddings_attempted <= 0:
            return "limit_reached"
        return "covered"

    def _record_partial_failures(self, handler: ArxivEmbeddingHandler, batch: List[Dict[str, Any]]) -> None:
        """批内有的写进去了、有的没写进去：重新查一次，只给仍缺向量的论文记失败。"""
        try:
            still_missing = handler.find_papers_missing_embeddings(batch)
        except Exception as exc:  # pragma: no cover - vector store runtime dependent
            logger.warning("Failed to recheck OAI papers after partial embedding failure: %s", exc)
            return
        self._count_paper_failures(still_missing)

    def _count_paper_failures(self, papers: List[Dict[str, Any]]) -> None:
        for paper in papers:
            arxiv_id = self._paper_id(paper)
            self._paper_failures[arxiv_id] = self._paper_failures.get(arxiv_id, 0) + 1
            if self._paper_failures[arxiv_id] == MAX_PAPER_FAILURES:
                self._skipped_arxiv_ids.add(arxiv_id)
                logger.warning("Skipping OAI paper %s for embedding for the rest of the day after %s failures", arxiv_id, MAX_PAPER_FAILURES)

    def _maybe_prune(self, result: Dict[str, Any]) -> None:
        today = self._today()
        if self._last_prune_date == today:
            return
        # 窗口已全部扫过才清理：补齐期间先补后删，没扫完就删可能误判。
        handler = self._handler_factory()
        pruned = int(self._prune_out_of_window(handler))
        self._last_prune_date = today
        result["pruned_out_of_window"] = pruned
        with self._status_lock:
            self._status["last_prune"] = {"finished_at": _utc_now_iso(), "cutoff_date": handler.min_paper_date, "pruned": pruned}

    # ------------------------------------------------------------------
    # 状态与日志
    # ------------------------------------------------------------------

    def _record_round(self, result: Dict[str, Any]) -> None:
        info_event(
            logger,
            "oai_embedding_maintenance.round",
            status=result["status"],
            window_start=result["window_start"],
            scanned=result["scanned"],
            missing_found=result["missing_found"],
            attempted=result["embeddings_attempted"],
            written=result["embeddings_written"],
            errors=result["embedding_errors"],
            total_tokens=result["embedding_total_tokens"],
            cost_yuan=round(result["embedding_cost_yuan"], 4),
            pruned=result.get("pruned_out_of_window"),
            skipped_papers=len(self._skipped_arxiv_ids),
            duration_seconds=result["duration_seconds"],
            rss_mb=result["rss_mb"],
        )
        if result["embedding_errors"]:
            self._consecutive_error_rounds += 1
            if self._consecutive_error_rounds >= ERROR_ROUNDS_ALERT_THRESHOLD:
                logger.error(
                    "OAI embedding maintenance has had errors for %s consecutive rounds (latest: %s errors)",
                    self._consecutive_error_rounds,
                    result["embedding_errors"],
                )
        else:
            self._consecutive_error_rounds = 0
        with self._status_lock:
            self._status["last_round"] = dict(result)
            self._status["written_since_start"] += int(result["embeddings_written"])
            self._status["cost_yuan_since_start"] = round(self._status["cost_yuan_since_start"] + float(result["embedding_cost_yuan"]), 4)
            if result["status"] == "covered":
                # 只有扫完整个窗口时，“还缺多少”才是全窗口口径。
                self._status["last_full_scan"] = {
                    "finished_at": result["finished_at"],
                    "window_start": result["window_start"],
                    "window_papers": result["scanned"],
                    "missing_after_round": max(0, result["missing_found"] - result["embeddings_written"]) + len(self._skipped_arxiv_ids),
                }
            if result["embedding_errors"] or result["status"] == "backoff":
                self._status["last_error"] = {"at": result["finished_at"], "message": f"{result['embedding_errors']} embedding errors in round ({result['status']})"}

    def _record_error(self, message: str) -> None:
        with self._status_lock:
            self._status["last_error"] = {"at": _utc_now_iso(), "message": message}

    @staticmethod
    def _paper_id(paper: Dict[str, Any]) -> str:
        return str(paper.get("arxiv_id", "") or "").strip()
