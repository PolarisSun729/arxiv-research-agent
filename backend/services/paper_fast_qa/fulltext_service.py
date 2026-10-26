"""快速问答的论文全文：解析任务排队、隔离子进程解析、按论文缓存。

缓存目录：``{cache_dir}/{论文 ID}/{缓存键}/``，缓存键由解析结构版本、pymupdf4llm 版本和 PDF 内容哈希组成，
任何一项变化都会得到新目录，旧目录在新结果写好后清理。目录内：

- ``manifest.json``：解析出的全文块及元数据；
- ``images/``：图片、公式和表格截图。

结果先写进同级临时目录，写完整后再 rename，读取方看到的缓存目录一定是完整的。

解析任务在进程内排队（生产环境只有一个 Web worker）：

- 同一篇论文同时只有一个任务，重复请求直接合并；
- 用户提问触发的任务排在打开详情页触发的预解析前面，已在排队的预解析被提问时会提到前面；
- 预解析的排队数有上限，满了直接拒绝，之后提问时还会再触发；
- 解析失败后冷却一段时间才允许重试，避免坏 PDF 被反复解析占满名额。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from services.document.isolated_subprocess import IsolatedSubprocessError, run_in_isolated_subprocess
from services.paper_fast_qa.fulltext_parser import FULLTEXT_SCHEMA_VERSION

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"
_IMAGE_DIR_NAME = "images"
_TEMP_DIR_PREFIX = ".tmp-"
_SAFE_PAPER_DIR = re.compile(r"^[A-Za-z0-9._-]+$")
# 全文块 ID：正文等为 p{页}-{序号}，图片为 fig-{序号}；研究引擎的 chunk_id 是整数，两者不会冲突。
FULLTEXT_BLOCK_ID_PATTERN = re.compile(r"^(?:fig-\d+|p\d+-\d+)$")
_PDF_HASH_CHUNK_BYTES = 1024 * 1024


class FulltextParseError(RuntimeError):
    """全文解析失败；reason 为稳定的失败类型，message 只用于日志。

    reason 取值：
        cooling_down  最近解析失败，还在冷却期内
        queue_full    预解析排队已满
        empty         PDF 没有可提取的文字（多为扫描版）
        timeout       解析超时
        failed        下载 PDF 或解析过程出错
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PaperFulltext:
    """一篇论文已缓存的全文解析结果。"""

    arxiv_id: str
    directory: Path
    manifest: Dict[str, Any]

    @property
    def blocks(self) -> List[Dict[str, Any]]:
        return list(self.manifest.get("blocks") or [])

    def image_path(self, block_id: str) -> Optional[Path]:
        """按块 ID 找图片文件；ID 不在本论文的块里时返回 None，不接受任意文件名。"""
        for block in self.blocks:
            if block.get("id") == block_id and block.get("image"):
                path = self.directory / _IMAGE_DIR_NAME / str(block["image"])
                return path if path.is_file() else None
        return None


@dataclass
class _ParseJob:
    arxiv_id: str
    priority: bool
    enqueued_at: float
    started_at: Optional[float] = None
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[PaperFulltext] = None
    error: Optional[FulltextParseError] = None


def parse_pdf_in_child(pdf_path: str, image_dir: str, image_dpi: int) -> Dict[str, Any]:
    """子进程入口：调低调度优先级后解析，让 Web 进程仍能及时拿到 CPU。"""
    from utils.logging_utils import configure_backend_logging

    configure_backend_logging()
    if hasattr(os, "nice"):
        os.nice(10)
    from services.paper_fast_qa.fulltext_parser import parse_pdf_fulltext

    return parse_pdf_fulltext(pdf_path, image_dir, image_dpi)


class PaperFulltextService:
    """管理论文全文解析任务与缓存；线程安全。"""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        pdf_provider: Callable[[str], str],
        parse_concurrency: int,
        parse_queue_limit: int,
        parse_timeout_seconds: float,
        failure_cooldown_seconds: float,
        image_dpi: int,
        parse_function: Callable[[str, str, int], Dict[str, Any]] = parse_pdf_in_child,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cache_dir = Path(cache_dir)
        self._pdf_provider = pdf_provider
        self._parse_concurrency = max(1, int(parse_concurrency))
        self._parse_queue_limit = max(0, int(parse_queue_limit))
        self._parse_timeout_seconds = float(parse_timeout_seconds)
        self._failure_cooldown_seconds = float(failure_cooldown_seconds)
        self._image_dpi = int(image_dpi)
        self._parse_function = parse_function
        self._clock = clock
        # 子进程名额与 Docling 建索引的全局互斥锁分开，两类解析互不阻塞。
        self._process_slots = threading.BoundedSemaphore(self._parse_concurrency)
        self._condition = threading.Condition()
        self._pending: List[_ParseJob] = []
        self._jobs: Dict[str, _ParseJob] = {}
        self._failures: Dict[str, tuple[float, FulltextParseError]] = {}
        self._workers: List[threading.Thread] = []

    # ---- 查询 ----

    def load_cached(self, arxiv_id: str) -> Optional[PaperFulltext]:
        """读取当前解析器版本下已完成的缓存；没有时返回 None。"""
        paper_dir = self._paper_dir(arxiv_id)
        if not paper_dir.is_dir():
            return None
        prefix = self._cache_key_prefix()
        candidates = [
            path for path in paper_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix) and (path / _MANIFEST_NAME).is_file()
        ]
        if not candidates:
            return None
        directory = max(candidates, key=lambda path: (path / _MANIFEST_NAME).stat().st_mtime)
        try:
            manifest = json.loads((directory / _MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Fulltext cache manifest unreadable: %s", directory, exc_info=True)
            return None
        return PaperFulltext(arxiv_id=arxiv_id, directory=directory, manifest=manifest)

    def resolve_image_path(self, arxiv_id: str, block_id: str) -> Optional[str]:
        """按块 ID 返回缓存里的图片路径；不是全文块 ID 时返回 None，交给详细模式的证据查找。

        块 ID 格式正确但找不到图片时抛 FileNotFoundError；只认缓存清单里的块，不接受任意文件名。
        """
        if not FULLTEXT_BLOCK_ID_PATTERN.match(str(block_id or "")):
            return None
        fulltext = self.load_cached(arxiv_id)
        path = fulltext.image_path(block_id) if fulltext is not None else None
        if path is None:
            raise FileNotFoundError("evidence asset not found")
        return str(path)

    def get_status(self, arxiv_id: str) -> Dict[str, Any]:
        """返回全文准备状态：ready / queued / parsing / failed / not_parsed。"""
        cached = self.load_cached(arxiv_id)
        if cached is not None:
            return _ready_status(cached)
        with self._condition:
            job = self._jobs.get(arxiv_id)
            if job is not None:
                return {"status": "parsing" if job.started_at is not None else "queued"}
            failure = self._active_failure(arxiv_id)
        if failure is not None:
            return {"status": "failed", "reason": failure.reason}
        return {"status": "not_parsed"}

    # ---- 触发与等待 ----

    def request_parse(self, arxiv_id: str, *, priority: bool) -> Dict[str, Any]:
        """确保论文已解析或已在排队，返回当前状态；不等待解析完成。"""
        cached = self.load_cached(arxiv_id)
        if cached is not None:
            return _ready_status(cached)
        try:
            job = self._enqueue(arxiv_id, priority=priority)
        except FulltextParseError as exc:
            return {"status": "failed" if exc.reason != "queue_full" else "busy", "reason": exc.reason}
        return {"status": "parsing" if job.started_at is not None else "queued"}

    def wait_for_fulltext(self, arxiv_id: str, timeout_seconds: float) -> Optional[PaperFulltext]:
        """按提问优先级确保解析并等待最多 timeout_seconds；超时返回 None，解析失败抛 FulltextParseError。

        调用方可以分多次短时间调用，在等待间隙向客户端发送进度。
        """
        cached = self.load_cached(arxiv_id)
        if cached is not None:
            return cached
        job = self._enqueue(arxiv_id, priority=True)
        if not job.done.wait(max(0.0, timeout_seconds)):
            return None
        if job.error is not None:
            raise job.error
        return job.result

    def _enqueue(self, arxiv_id: str, *, priority: bool) -> _ParseJob:
        self._paper_dir(arxiv_id)  # 先校验论文 ID 能安全地作为目录名。
        with self._condition:
            job = self._jobs.get(arxiv_id)
            if job is not None:
                if priority and not job.priority:
                    job.priority = True
                    self._condition.notify_all()
                return job
            failure = self._active_failure(arxiv_id)
            if failure is not None:
                raise failure
            if not priority:
                waiting = sum(1 for pending in self._pending if not pending.priority)
                if waiting >= self._parse_queue_limit:
                    raise FulltextParseError("queue_full", f"预解析排队已满（{waiting}）")
            job = _ParseJob(arxiv_id=arxiv_id, priority=priority, enqueued_at=self._clock())
            self._jobs[arxiv_id] = job
            self._pending.append(job)
            self._ensure_workers()
            self._condition.notify_all()
            return job

    def _active_failure(self, arxiv_id: str) -> Optional[FulltextParseError]:
        failure = self._failures.get(arxiv_id)
        if failure is None:
            return None
        failed_at, error = failure
        if self._clock() - failed_at >= self._failure_cooldown_seconds:
            self._failures.pop(arxiv_id, None)
            return None
        return FulltextParseError("cooling_down", f"最近一次解析失败（{error.reason}），冷却中")

    # ---- 后台执行 ----

    def _ensure_workers(self) -> None:
        """首次有任务时才启动工作线程；调用方持有 _condition。"""
        self._workers = [worker for worker in self._workers if worker.is_alive()]
        while len(self._workers) < self._parse_concurrency:
            worker = threading.Thread(
                target=self._worker_loop, name=f"fulltext-parse-{len(self._workers) + 1}", daemon=True,
            )
            self._workers.append(worker)
            worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    self._condition.wait()
                # 提问触发的任务优先，同优先级按入队先后。
                job = min(self._pending, key=lambda item: (not item.priority, item.enqueued_at))
                self._pending.remove(job)
                job.started_at = self._clock()
            try:
                job.result = self._parse_and_store(job.arxiv_id)
            except FulltextParseError as exc:
                job.error = exc
            except Exception as exc:
                logger.exception("Fulltext parse failed: arxiv_id=%s", job.arxiv_id)
                job.error = FulltextParseError("failed", f"{type(exc).__name__}: {exc}")
            with self._condition:
                self._jobs.pop(job.arxiv_id, None)
                if job.error is not None:
                    logger.warning("Fulltext parse failed: arxiv_id=%s reason=%s detail=%s",
                                   job.arxiv_id, job.error.reason, job.error)
                    self._failures[job.arxiv_id] = (self._clock(), job.error)
            job.done.set()

    def _parse_and_store(self, arxiv_id: str) -> PaperFulltext:
        pdf_path = self._pdf_provider(arxiv_id)
        pdf_sha256 = _file_sha256(pdf_path)
        paper_dir = self._paper_dir(arxiv_id)
        target_dir = paper_dir / f"{self._cache_key_prefix()}{pdf_sha256[:16]}"
        if (target_dir / _MANIFEST_NAME).is_file():
            cached = self.load_cached(arxiv_id)
            if cached is not None and cached.directory == target_dir:
                return cached

        temp_dir = paper_dir / f"{_TEMP_DIR_PREFIX}{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True)
        try:
            started = time.monotonic()
            try:
                parsed = run_in_isolated_subprocess(
                    self._parse_function,
                    pdf_path,
                    str(temp_dir / _IMAGE_DIR_NAME),
                    self._image_dpi,
                    timeout_seconds=self._parse_timeout_seconds,
                    lock_path=None,
                    lock_timeout_seconds=self._parse_timeout_seconds,
                    task_label=f"Fulltext parse {arxiv_id}",
                    process_slots=self._process_slots,
                )
            except IsolatedSubprocessError as exc:
                reason = "timeout" if exc.reason == "timeout" else "failed"
                raise FulltextParseError(reason, str(exc)) from exc
            if not parsed.get("blocks"):
                raise FulltextParseError("empty", "PDF 没有可提取的文字")
            manifest = {
                **parsed,
                "arxiv_id": arxiv_id,
                "pdf_sha256": pdf_sha256,
                "parser": {"pymupdf4llm": _pymupdf4llm_version()},
                "parse_seconds": round(time.monotonic() - started, 1),
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            }
            (temp_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            try:
                temp_dir.rename(target_dir)
            except OSError:
                # 同一份 PDF 已被其他进程写好，保留已有结果。
                if not (target_dir / _MANIFEST_NAME).is_file():
                    raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._remove_stale_versions(paper_dir, keep=target_dir)
        logger.info("Fulltext parsed: arxiv_id=%s blocks=%s images=%s seconds=%s",
                    arxiv_id, len(manifest["blocks"]), manifest.get("image_count"), manifest["parse_seconds"])
        return PaperFulltext(arxiv_id=arxiv_id, directory=target_dir, manifest=manifest)

    @staticmethod
    def _remove_stale_versions(paper_dir: Path, *, keep: Path) -> None:
        for path in paper_dir.iterdir():
            if path.is_dir() and path != keep and not path.name.startswith(_TEMP_DIR_PREFIX):
                shutil.rmtree(path, ignore_errors=True)

    # ---- 路径 ----

    def _paper_dir(self, arxiv_id: str) -> Path:
        # 旧式 ID 中的斜杠属于论文标识，不是目录。
        name = str(arxiv_id or "").strip().replace("/", "_")
        if not _SAFE_PAPER_DIR.match(name) or name in {".", ".."}:
            raise ValueError(f"invalid paper id for fulltext cache: {arxiv_id!r}")
        return self.cache_dir / name

    @staticmethod
    def _cache_key_prefix() -> str:
        return f"v{FULLTEXT_SCHEMA_VERSION}-p4l{_pymupdf4llm_version()}-"


def _ready_status(fulltext: PaperFulltext) -> Dict[str, Any]:
    manifest = fulltext.manifest
    return {
        "status": "ready",
        "page_count": manifest.get("page_count"),
        "image_count": manifest.get("image_count"),
        "char_count": manifest.get("char_count"),
    }


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_PDF_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pymupdf4llm_version() -> str:
    try:
        return metadata.version("pymupdf4llm")
    except metadata.PackageNotFoundError:
        return "unknown"
