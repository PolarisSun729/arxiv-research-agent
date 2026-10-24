from __future__ import annotations

import sys
from types import ModuleType

import pytest

from services.storage.vector_store_service import cap_milvus_lite_compaction


@pytest.fixture
def fake_compaction_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """用假的 milvus_lite.engine.compaction 代替真实包（本地 Windows 装不了 milvus-lite）。"""
    package = ModuleType("milvus_lite")
    engine = ModuleType("milvus_lite.engine")
    compaction = ModuleType("milvus_lite.engine.compaction")
    compaction.MAX_SEGMENT_ROWS = 100_000
    engine.compaction = compaction
    package.engine = engine
    monkeypatch.setitem(sys.modules, "milvus_lite", package)
    monkeypatch.setitem(sys.modules, "milvus_lite.engine", engine)
    monkeypatch.setitem(sys.modules, "milvus_lite.engine.compaction", compaction)
    cap_milvus_lite_compaction.cache_clear()
    yield compaction
    cap_milvus_lite_compaction.cache_clear()


def test_cap_overrides_compaction_segment_rows(fake_compaction_module: ModuleType) -> None:
    assert cap_milvus_lite_compaction(8000) is True
    assert fake_compaction_module.MAX_SEGMENT_ROWS == 8000


def test_cap_is_skipped_when_milvus_lite_layout_changes(fake_compaction_module: ModuleType) -> None:
    # 升级后常量改名时不能静默写一个没人读的属性，返回 False 并保持原样。
    del fake_compaction_module.MAX_SEGMENT_ROWS

    assert cap_milvus_lite_compaction(8000) is False
    assert not hasattr(fake_compaction_module, "MAX_SEGMENT_ROWS")
