from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from services.storage.paper_vector_store import PAPER_EMBEDDING_COLLECTION as COLLECTION
from services.storage.paper_vector_store import FilePaperVectorStore

BACKEND_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("export_paper_vectors", BACKEND_ROOT / "07-arxiv-tools" / "export_paper_vectors.py")
export_paper_vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_paper_vectors)

DIM = 8


class _Iterator:
    def __init__(self, rows, batch_size):
        self._batches = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]

    def next(self):
        return self._batches.pop(0) if self._batches else []

    def close(self):
        pass


class _FakeMilvusClient:
    """按 Milvus Lite 的返回形状模拟 query_iterator / search（COSINE 精确检索）。"""

    rows: list = []

    def __init__(self, uri):
        self.uri = uri

    def has_collection(self, collection_name):
        return collection_name == COLLECTION

    def load_collection(self, collection_name):
        pass

    def query_iterator(self, collection_name, batch_size, filter, output_fields):
        rows = [row for row in self.rows if filter == "id >= 0" or row["arxiv_id"]]
        return _Iterator([{field: row[field] for field in output_fields} for row in rows], batch_size)

    def search(self, collection_name, data, limit, **_kwargs):
        query = np.asarray(data[0], dtype=np.float32)
        best = {}
        for row in self.rows:
            vector = np.asarray(row["vector"], dtype=np.float32)
            score = float(vector @ query / (np.linalg.norm(vector) * np.linalg.norm(query)))
            if row["arxiv_id"] and score > best.get(row["arxiv_id"], -2.0):
                best[row["arxiv_id"]] = score
        ranked = sorted(best.items(), key=lambda pair: -pair[1])[:limit]
        return [[{"entity": {"arxiv_id": arxiv_id}, "distance": score} for arxiv_id, score in ranked]]

    def close(self):
        pass


def _row(row_id, arxiv_id, vector):
    return {
        "id": row_id,
        "arxiv_id": arxiv_id,
        "title": f"Title {arxiv_id}",
        "authors": "Alice, Bob",
        "categories": "cs.CL",
        "published_date": "2026-10-01",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "embedding_model": "test-model",
        "content": f"Abstract {arxiv_id}",
        "vector": [float(value) for value in vector],
    }


@pytest.fixture
def fake_milvus(monkeypatch):
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((30, DIM)).astype(np.float32)
    rows = [_row(1000 + index, f"p{index}", vector) for index, vector in enumerate(vectors)]
    rows.append(_row(2000, "p3", vectors[7]))  # 同一篇的重复向量：只保留第一条
    rows.append(_row(2001, "", vectors[8]))  # 空 arxiv_id：跳过
    _FakeMilvusClient.rows = rows
    pymilvus = ModuleType("pymilvus")
    pymilvus.MilvusClient = _FakeMilvusClient
    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)
    monkeypatch.setattr(export_paper_vectors, "_disable_milvus_lite_compaction", lambda: True)
    monkeypatch.setattr(export_paper_vectors, "_prefer_as_oom_victim", lambda: None)
    monkeypatch.setattr(export_paper_vectors, "_peak_rss_mb", lambda: 0.0)
    return rows


def _argv(tmp_path):
    return ["--milvus-db", str(tmp_path / "milvus.db"), "--vectors", str(tmp_path / "v.f32"), "--db", str(tmp_path / "v.db"), "--batch-size", "7", "--verify-queries", "5", "--verify-top-k", "10"]


def test_export_copies_vectors_verifies_and_renames(tmp_path, fake_milvus):
    assert export_paper_vectors.main(_argv(tmp_path)) == 0

    assert not (tmp_path / "v.f32.tmp").exists() and not (tmp_path / "v.db.tmp").exists()
    store = FilePaperVectorStore(vectors_path=str(tmp_path / "v.f32"), db_path=str(tmp_path / "v.db"))
    try:
        assert store.list_paper_arxiv_ids(COLLECTION) == {f"p{index}" for index in range(30)}
        stored = store.get_paper_embeddings_by_arxiv_ids(COLLECTION, ["p3"])[0]
        # 沿用 Milvus 主键，重复向量保留先读到的那条。
        assert stored["embedding_id"] == 1003
        assert stored["vector"] == fake_milvus[3]["vector"]
    finally:
        store.close()

    # 目标已存在：拒绝覆盖。
    assert export_paper_vectors.main(_argv(tmp_path)) == 1


def test_export_keeps_tmp_files_when_search_disagrees(tmp_path, fake_milvus, monkeypatch):
    original = _FakeMilvusClient.search

    def shifted(self, *args, **kwargs):
        hits = original(self, *args, **kwargs)
        return [[{**hit, "distance": hit["distance"] + 1e-3} for hit in hits[0]]]

    monkeypatch.setattr(_FakeMilvusClient, "search", shifted)

    assert export_paper_vectors.main(_argv(tmp_path)) == 2
    assert (tmp_path / "v.f32.tmp").exists() and not (tmp_path / "v.f32").exists()


def test_disable_compaction_patches_milvus_lite(monkeypatch):
    package = ModuleType("milvus_lite")
    engine = ModuleType("milvus_lite.engine")
    compaction = ModuleType("milvus_lite.engine.compaction")

    class CompactionManager:
        def maybe_compact(self, partition, manifest, delta_index):
            raise AssertionError("compaction must not run during export")

    compaction.CompactionManager = CompactionManager
    engine.compaction = compaction
    package.engine = engine
    monkeypatch.setitem(sys.modules, "milvus_lite", package)
    monkeypatch.setitem(sys.modules, "milvus_lite.engine", engine)
    monkeypatch.setitem(sys.modules, "milvus_lite.engine.compaction", compaction)

    assert export_paper_vectors._disable_milvus_lite_compaction() is True
    assert CompactionManager().maybe_compact("p", None, None) is False
