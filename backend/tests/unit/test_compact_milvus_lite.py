from __future__ import annotations

import importlib.util
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("compact_milvus_lite", BACKEND_ROOT / "07-arxiv-tools" / "compact_milvus_lite.py")
compact_milvus_lite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compact_milvus_lite)

FileStats = compact_milvus_lite.FileStats
plan_groups = compact_milvus_lite.plan_groups


def test_only_files_with_deleted_rows_or_oversized_are_compacted() -> None:
    stats = [
        FileStats("big", rows=47669, deleted_rows=0),  # 超过输出段上限：要切段
        FileStats("clean", rows=8000, deleted_rows=0),  # 不动
        FileStats("dirty", rows=6000, deleted_rows=120),
        FileStats("tiny", rows=20, deleted_rows=0),  # 小文件交给 Milvus Lite 自己合并
    ]

    assert plan_groups(stats, segment_rows=8000, group_rows=24000) == [["big"], ["dirty"]]


def test_groups_stay_under_group_rows_to_bound_memory() -> None:
    stats = [FileStats(f"f{i}", rows=8000, deleted_rows=10) for i in range(5)]

    assert plan_groups(stats, segment_rows=8000, group_rows=24000) == [["f0", "f1", "f2"], ["f3", "f4"]]


def test_nothing_to_do_when_no_deleted_rows() -> None:
    stats = [FileStats("a", rows=8000, deleted_rows=0), FileStats("b", rows=300, deleted_rows=0)]

    assert plan_groups(stats, segment_rows=8000, group_rows=24000) == []
