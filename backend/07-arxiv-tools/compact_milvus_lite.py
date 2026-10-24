"""离线整理 Milvus Lite 数据文件：清掉已删除的行，并把大文件切成小段。

为什么需要:
    生产的 Milvus Lite 在 Web 进程内运行，会把每个数据文件整份加载进内存；删除只是记一条删除标记，
    被删的行要等合并时才真正清掉。为了防止合并时 OOM，Web 进程限制了单次合并的行数
    （MILVUS_LITE_MAX_SEGMENT_ROWS，默认 8000），超过上限的文件不再参与合并，
    论文向量每天按窗口清理删掉的旧论文就会一直占着内存。这个脚本在服务停止时单独运行，
    把带删除行或超过上限的文件重新合并一遍：删除行被清掉，输出按上限切段，之后都在可合并的范围内。

怎么运行:
    整理必须在 Web 服务停止时运行：脚本先获取 Milvus Lite 的独占锁，服务还在运行时直接退出（退出码 1）。
    生产由 deploy/arxiv-milvus-compact.timer 每天凌晨触发 compact_milvus_lite.sh：先用 --check 只读检查，
    有删除行要清才停服务、整理、再启动服务（无论成败），没有就不停服务直接结束。
    手动执行：sudo systemctl start arxiv-milvus-compact.service

    # 只看统计，不改文件（服务运行中也可以执行，只读）
    python 07-arxiv-tools/compact_milvus_lite.py --dry-run
    # 只读检查是否需要整理：需要时退出码 3
    python 07-arxiv-tools/compact_milvus_lite.py --check
    # 整理（需先停服务）
    python 07-arxiv-tools/compact_milvus_lite.py

内存:
    合并时会把一组输入文件整份读进内存再复制几份，2048 维下峰值约每行 40–55KB。
    脚本按组合并，每组不超过 --group-rows 行（默认 24000，约 +1.3GB）；单个文件超过这个数时单独成组。
    服务停止时机器剩余内存足够，而在 Web 进程里做同样的事会和进程本身的内存叠加。

实现说明:
    用的是 milvus-lite 自己的合并函数（CompactionManager._compact_files / _gc_tombstones），
    写新文件、原子更新 manifest、删旧文件都和它后台合并走同一条路径，中途崩溃也能由它的恢复逻辑处理。
    这些是内部接口，所以只在 lock 文件锁定的版本上运行，版本不符直接退出。
    新文件的向量索引由 Web 进程启动加载时补建（FLAT 索引只是拷贝向量，很快）。

退出码: 0 = 完成（含没有需要整理的文件）；1 = 服务未停止或版本不符；2 = 整理过程中出错；
       3 = 仅 --check：有文件需要整理。
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.config import MILVUS_CONFIG  # noqa: E402

logger = logging.getLogger("compact_milvus_lite")

# 用到了 milvus-lite 的内部接口，只在验证过的版本上运行（与 requirements.lock.txt 一致）。
SUPPORTED_MILVUS_LITE_VERSION = "3.2.1"


@dataclass
class FileStats:
    name: str
    rows: int
    deleted_rows: int


@dataclass
class CollectionReport:
    name: str
    files_before: int = 0
    files_after: int = 0
    rows_before: int = 0
    deleted_rows: int = 0
    compacted_groups: int = 0
    skipped_reason: str = ""
    groups: List[List[str]] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compact Milvus Lite data files offline: drop deleted rows, split oversized files.")
    parser.add_argument("--db", default=MILVUS_CONFIG["uri"], help="Milvus Lite 数据目录，默认取 MILVUS_URI / 默认路径")
    parser.add_argument("--segment-rows", type=int, default=MILVUS_CONFIG["lite_max_segment_rows"], help="输出文件的最大行数，与 Web 进程的合并上限一致")
    parser.add_argument("--group-rows", type=int, default=24000, help="一次合并的最大行数（单个文件超过时单独成组），控制内存峰值")
    parser.add_argument("--dry-run", action="store_true", help="只统计各文件的行数和删除行数，不改任何文件")
    parser.add_argument("--check", action="store_true", help="同 --dry-run，有文件需要整理时退出码为 3，供外层脚本决定要不要停服务")
    parser.add_argument("--flush-only", action="store_true", help=argparse.SUPPRESS)
    return parser


def _peak_rss_mb() -> float:
    import resource  # 只在 Linux 上有；脚本只在生产服务器运行

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _check_version() -> bool:
    from importlib import metadata

    version = metadata.version("milvus-lite")
    if version != SUPPORTED_MILVUS_LITE_VERSION:
        logger.error("milvus-lite %s is not supported (expected %s); refusing to touch data files", version, SUPPORTED_MILVUS_LITE_VERSION)
        return False
    return True


def _collection_dirs(db_path: str) -> List[str]:
    """默认数据库下的所有 collection 目录（项目只用默认数据库）。"""
    return sorted(path for path in glob.glob(os.path.join(db_path, "collections", "*")) if os.path.isdir(path))


def _has_wal(col_dir: str) -> bool:
    from milvus_lite.storage.wal import WAL

    return bool(WAL.find_wal_files(os.path.join(col_dir, "wal")))


def flush_pending_wal(db_path: str) -> None:
    """把还在 WAL 里的写入落成数据文件。打开 collection 会把它的数据整份加载进内存，所以放在单独的子进程里做。"""
    from milvus_lite.db import MilvusLite

    with MilvusLite(db_path) as db:
        for col_dir in _collection_dirs(db_path):
            if _has_wal(col_dir):
                name = os.path.basename(col_dir)
                logger.info("Flushing pending WAL of collection %s", name)
                db.get_collection(name).close()


def _file_stats(partition_dir: str, rel: str, pk_name: str, delta_index) -> FileStats:
    import pyarrow.parquet as pq

    table = pq.read_table(os.path.join(partition_dir, rel), columns=[pk_name, "_seq"])
    pks = table.column(pk_name).to_pylist()
    seqs = table.column("_seq").to_pylist()
    deleted = sum(1 for pk, seq in zip(pks, seqs) if delta_index.is_deleted(pk, seq))
    return FileStats(name=rel, rows=table.num_rows, deleted_rows=deleted)


def plan_groups(stats: List[FileStats], segment_rows: int, group_rows: int) -> List[List[str]]:
    """挑出需要整理的文件（有删除行，或行数超过输出段上限），按 manifest 顺序打包成不超过 group_rows 的组。"""
    groups: List[List[str]] = []
    current: List[str] = []
    current_rows = 0
    for item in stats:
        if item.deleted_rows == 0 and item.rows <= segment_rows:
            continue
        if current and current_rows + item.rows > group_rows:
            groups.append(current)
            current, current_rows = [], 0
        current.append(item.name)
        current_rows += item.rows
    if current:
        groups.append(current)
    return groups


def _remove_index_sidecars(partition_dir: str, rels: List[str]) -> None:
    """删掉已被合并掉的数据文件的索引文件；Web 进程也会清理孤儿索引，这里只是让磁盘立刻回收。"""
    for rel in rels:
        stem = os.path.splitext(os.path.basename(rel))[0]
        for path in glob.glob(os.path.join(partition_dir, "indexes", f"{stem}.*")):
            os.remove(path)


def compact_collection(col_dir: str, segment_rows: int, group_rows: int, dry_run: bool) -> CollectionReport:
    from milvus_lite.db import SCHEMA_FILENAME
    from milvus_lite.engine.compaction import CompactionManager
    from milvus_lite.schema.arrow_builder import get_primary_field
    from milvus_lite.schema.persistence import load_schema
    from milvus_lite.storage.delta_index import DeltaIndex
    from milvus_lite.storage.manifest import Manifest

    report = CollectionReport(name=os.path.basename(col_dir))
    _name, schema = load_schema(os.path.join(col_dir, SCHEMA_FILENAME))
    pk_name = get_primary_field(schema).name
    manifest = Manifest.load(col_dir)
    if not dry_run and _has_wal(col_dir):
        # 删除可能还在 WAL 里没落成删除标记，这时整理会漏删；正常流程下已在子进程里刷过，这里只是兜底。
        # 只读统计不跳过：服务运行时 WAL 里总有数据，统计只是少算还没落盘的删除。
        report.skipped_reason = "pending WAL"
        return report

    delta_files = {
        partition: [os.path.join(col_dir, "partitions", partition, rel) for rel in rels]
        for partition, rels in manifest.get_all_delta_files().items()
    }
    delta_index = DeltaIndex.rebuild_from(pk_name, {partition: [p for p in paths if os.path.exists(p)] for partition, paths in delta_files.items()})
    manager = CompactionManager(col_dir, schema)

    for partition in manifest.list_partitions():
        partition_dir = os.path.join(col_dir, "partitions", partition)
        stats = [_file_stats(partition_dir, rel, pk_name, delta_index) for rel in manifest.get_data_files(partition)]
        report.files_before += len(stats)
        report.rows_before += sum(item.rows for item in stats)
        report.deleted_rows += sum(item.deleted_rows for item in stats)
        groups = plan_groups(stats, segment_rows, group_rows)
        report.groups.extend(groups)
        if dry_run:
            continue
        for group in groups:
            logger.info("Compacting %s/%s: %s file(s)", report.name, partition, len(group))
            manager._compact_files(partition, partition_dir, group, manifest, delta_index)
            _remove_index_sidecars(partition_dir, group)
            report.compacted_groups += 1

    if not dry_run and report.compacted_groups:
        # 合并后不再被任何数据文件引用的删除标记（含磁盘上的 delta 文件）一并回收。
        manager._gc_tombstones(manifest, delta_index)
    report.files_after = sum(len(manifest.get_data_files(partition)) for partition in manifest.list_partitions())
    return report


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not _check_version():
        return 1

    from milvus_lite.engine import compaction

    # 合并输出按这个行数切段（与 Web 进程的合并上限保持一致）。
    compaction.MAX_SEGMENT_ROWS = args.segment_rows

    if args.flush_only:
        flush_pending_wal(args.db)
        return 0

    dry_run = args.dry_run or args.check
    if dry_run:
        reports = [compact_collection(col_dir, args.segment_rows, args.group_rows, dry_run=True) for col_dir in _collection_dirs(args.db)]
    else:
        from milvus_lite.db import MilvusLite
        from milvus_lite.exceptions import DataDirLockedError

        if any(_has_wal(col_dir) for col_dir in _collection_dirs(args.db)):
            result = subprocess.run([sys.executable, __file__, "--db", args.db, "--flush-only"], check=False)
            if result.returncode != 0:
                logger.error("Flushing pending WAL failed (exit %s); is the web service still running?", result.returncode)
                return 1
        try:
            # 只为持锁：拿到 Milvus Lite 的独占锁才说明没有别的进程（Web 服务）打开这个库。
            lock_holder = MilvusLite(args.db)
        except DataDirLockedError as exc:
            logger.error("Cannot lock %s (%s); stop the web service first", args.db, exc)
            return 1
        try:
            reports = [compact_collection(col_dir, args.segment_rows, args.group_rows, dry_run=False) for col_dir in _collection_dirs(args.db)]
        except Exception:
            logger.exception("Compaction failed; Milvus Lite recovery will clean up partial output on next open")
            return 2
        finally:
            lock_holder.close()

    for report in reports:
        logger.info(
            "milvus_lite_compact collection=%s files=%s->%s rows=%s deleted_rows=%s groups=%s%s",
            report.name,
            report.files_before,
            report.files_after if not dry_run else "-",
            report.rows_before,
            report.deleted_rows,
            len(report.groups),
            f" skipped={report.skipped_reason}" if report.skipped_reason else "",
        )
    logger.info("milvus_lite_compact done dry_run=%s peak_rss_mb=%s", dry_run, _peak_rss_mb())
    if args.check and any(report.groups for report in reports):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
