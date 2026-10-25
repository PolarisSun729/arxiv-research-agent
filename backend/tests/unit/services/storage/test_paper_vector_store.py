from __future__ import annotations

import os

import numpy as np
import pytest

from services.storage import paper_vector_store as store_module
from services.storage.paper_vector_store import PAPER_EMBEDDING_COLLECTION as COLLECTION
from services.storage.paper_vector_store import FilePaperVectorStore

DIM = 8


def _item(arxiv_id: str, vector, published_date: str = "2026-10-01", **extra):
    return {
        "embedding": [float(value) for value in vector],
        "metadata": {
            "arxiv_id": arxiv_id,
            "title": f"Title {arxiv_id}",
            "authors": "Alice, Bob",
            "categories": "cs.CL cs.AI",
            "published_date": published_date,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "embedding_model": "test-model",
            "content": f"Abstract {arxiv_id}",
        },
        **extra,
    }


@pytest.fixture
def store(tmp_path):
    instance = FilePaperVectorStore(vectors_path=str(tmp_path / "vectors.f32"), db_path=str(tmp_path / "vectors.db"))
    yield instance
    instance.close()


def _reopen(store: FilePaperVectorStore) -> FilePaperVectorStore:
    store.close()
    return FilePaperVectorStore(vectors_path=store.vectors_path, db_path=store.db_path)


def test_insert_skips_existing_and_reads_back_exact_vectors(store):
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((3, DIM)).astype(np.float32)

    assert store.insert_embeddings(COLLECTION, [_item("a", vectors[0]), _item("b", vectors[1]), _item("a", vectors[2])]) == 2
    assert store.insert_embeddings(COLLECTION, [_item("b", vectors[2]), _item("c", vectors[2])]) == 1

    assert store.list_paper_arxiv_ids(COLLECTION) == {"a", "b", "c"}
    assert store.get_existing_paper_arxiv_ids(COLLECTION, ["a", " c ", "zzz", ""]) == {"a", "c"}
    rows = store.get_paper_embeddings_by_arxiv_ids(COLLECTION, ["c", "missing", "a"])
    assert [row["arxiv_id"] for row in rows] == ["c", "a"]
    # 向量原样保存（不归一化），与写入时逐位相等。
    assert np.array_equal(np.asarray(rows[0]["vector"], dtype=np.float32), vectors[2])
    assert np.array_equal(np.asarray(rows[1]["vector"], dtype=np.float32), vectors[0])
    assert rows[1]["title"] == "Title a" and rows[1]["published_date"] == "2026-10-01" and rows[1]["embedding_model"] == "test-model"


def test_search_ranks_by_cosine_and_honours_filter(store):
    rng = np.random.default_rng(1)
    # 模长各不相同，验证按余弦而不是内积排序。
    vectors = rng.standard_normal((40, DIM)).astype(np.float32) * rng.uniform(0.5, 3.0, size=(40, 1)).astype(np.float32)
    store.insert_embeddings(COLLECTION, [_item(f"p{index}", vector) for index, vector in enumerate(vectors)])
    query = rng.standard_normal(DIM).astype(np.float32)

    cosine = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
    expected = [f"p{index}" for index in np.argsort(-cosine)]

    results = store.search_similar_papers(COLLECTION, query.tolist(), top_k=5)
    assert [paper["arxiv_id"] for paper in results] == expected[:5]
    assert results[0]["score"] == pytest.approx(float(cosine.max()), abs=1e-5)
    assert results[0]["distance"] == results[0]["score"]
    assert results[0]["authors"] == ["Alice", "Bob"]
    assert results[0]["abstract"] == f"Abstract {expected[0]}"

    filtered = store.search_similar_papers(COLLECTION, query.tolist(), top_k=5, filter_arxiv_ids=expected[:2])
    assert [paper["arxiv_id"] for paper in filtered] == expected[2:7]
    assert len(store.search_similar_papers(COLLECTION, query.tolist(), top_k=100)) == 40


def test_delete_respects_keep_list_and_reuses_slots(store):
    store.insert_embeddings(
        COLLECTION,
        [
            _item("old", np.ones(DIM), "2026-01-01"),
            _item("old-liked", np.ones(DIM), "2026-01-02"),
            _item("undated", np.ones(DIM), ""),
            _item("new", np.ones(DIM), "2026-10-01"),
        ],
    )
    size_before = os.path.getsize(store.vectors_path)

    assert store.delete_paper_embeddings_published_before(COLLECTION, "2026-05-01", {"old-liked"}, dry_run=True) == 1
    assert store.list_paper_arxiv_ids(COLLECTION) == {"old", "old-liked", "undated", "new"}
    assert store.delete_paper_embeddings_published_before(COLLECTION, "2026-05-01", {"old-liked"}) == 1
    assert store.list_paper_arxiv_ids(COLLECTION) == {"old-liked", "undated", "new"}
    assert store.search_similar_papers(COLLECTION, np.ones(DIM).tolist(), top_k=10, filter_arxiv_ids=[]) != []
    assert "old" not in {paper["arxiv_id"] for paper in store.search_similar_papers(COLLECTION, np.ones(DIM).tolist(), top_k=10)}

    store.insert_embeddings(COLLECTION, [_item("fresh", np.full(DIM, 2.0))])
    assert os.path.getsize(store.vectors_path) == size_before
    assert np.array_equal(store.get_paper_embeddings_by_arxiv_ids(COLLECTION, ["fresh"])[0]["vector"], np.full(DIM, 2.0))


def test_reopen_restores_state_and_keeps_explicit_ids(store):
    store.insert_embeddings(COLLECTION, [_item("migrated", np.arange(DIM), id=460000000000000123)])
    new_id = store.insert_single_embedding(COLLECTION, np.arange(DIM)[::-1].tolist(), _item("new", np.zeros(DIM))["metadata"])
    assert new_id == 460000000000000124
    assert store.insert_single_embedding(COLLECTION, np.ones(DIM).tolist(), _item("new", np.zeros(DIM))["metadata"]) == new_id

    reopened = _reopen(store)
    try:
        assert reopened.list_paper_arxiv_ids(COLLECTION) == {"migrated", "new"}
        rows = reopened.get_paper_embeddings_by_arxiv_ids(COLLECTION, ["migrated", "new"])
        assert [row["embedding_id"] for row in rows] == [460000000000000123, 460000000000000124]
        assert rows[1]["vector"] == [float(value) for value in np.arange(DIM)[::-1]]
        assert reopened.search_similar_papers(COLLECTION, np.arange(DIM).tolist(), top_k=1)[0]["arxiv_id"] == "migrated"
    finally:
        reopened.close()


def test_vector_written_without_record_stays_free_after_restart(store, monkeypatch):
    store.insert_embeddings(COLLECTION, [_item("kept", np.ones(DIM))])

    # 向量已写进文件、SQLite 记录还没提交时崩溃：重启后该槽位仍是空闲的，不会出现半条记录。
    monkeypatch.setattr(store_module.FilePaperVectorStore, "_connect", lambda self: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        store.insert_embeddings(COLLECTION, [_item("lost", np.full(DIM, 5.0))])
    monkeypatch.undo()

    reopened = _reopen(store)
    try:
        assert reopened.list_paper_arxiv_ids(COLLECTION) == {"kept"}
        assert reopened.insert_embeddings(COLLECTION, [_item("lost", np.full(DIM, 5.0))]) == 1
        assert reopened.list_paper_arxiv_ids(COLLECTION) == {"kept", "lost"}
    finally:
        reopened.close()


def test_rejects_wrong_dimension_and_collection(store):
    store.insert_embeddings(COLLECTION, [_item("a", np.ones(DIM))])

    with pytest.raises(ValueError, match="dimension"):
        store.insert_embeddings(COLLECTION, [_item("b", np.ones(DIM + 1))])
    with pytest.raises(ValueError, match="dimension"):
        store.search_similar_papers(COLLECTION, np.ones(DIM + 1).tolist())
    with pytest.raises(ValueError, match="arxiv_paper_embeddings"):
        store.list_paper_arxiv_ids("arxiv_abstracts")


def test_empty_store_reads_return_empty(store):
    assert store.list_paper_arxiv_ids(COLLECTION) == set()
    assert store.search_similar_papers(COLLECTION, np.ones(DIM).tolist()) == []
    assert store.get_paper_embeddings_by_arxiv_ids(COLLECTION, ["a"]) == []
    assert store.delete_paper_embeddings_published_before(COLLECTION, "2026-05-01", set()) == 0
