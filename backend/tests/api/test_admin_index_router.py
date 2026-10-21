import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import admin_index_router
from services.paper_qa import index_inspection


def _chunk(chunk_id, content, *, chunk_type="text", **metadata):
    return {"content": content, "metadata": {"chunk_id": chunk_id, "chunk_type": chunk_type, **metadata}}


class _FakeIndexStore:
    def __init__(self):
        self.indexes = {}
        self.builds = {}

    def get_paper_qa_index(self, arxiv_id):
        return self.indexes.get(arxiv_id)

    def list_paper_qa_index_builds(self, arxiv_id, **_kwargs):
        return self.builds.get(arxiv_id, [])

    def list_paper_qa_indexes(self, **kwargs):
        self.list_kwargs = kwargs
        return [{"arxiv_id": arxiv_id} for arxiv_id in self.indexes], len(self.indexes)


@pytest.fixture
def env(tmp_path, monkeypatch):
    loaded_docs = tmp_path / "01-loaded-docs"
    assets = tmp_path / "03-docling-assets"
    loaded_docs.mkdir()
    (assets / "p1").mkdir(parents=True)
    (assets / "p1" / "fig.png").write_bytes(b"png-bytes")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"secret")
    monkeypatch.setattr(index_inspection, "LOADED_DOCS_ROOT", loaded_docs)
    monkeypatch.setattr(index_inspection, "BACKEND_DOCLING_ASSETS_ROOT", assets)

    def write_chunk_file(name, payload):
        (loaded_docs / name).write_text(json.dumps(payload), encoding="utf-8")
        # 记录里是别的机器/旧发布目录下的绝对路径，查看时只按文件名定位。
        return f"/old/release/backend/01-loaded-docs/{name}"

    new_file = write_chunk_file("p1_new.json", {
        "loading_method": "docling",
        "chunking_strategy": "docling_sections",
        "total_pages": 2,
        "chunking_params": {"chunk_size": 500, "max_chunk_content_length": 8000, "chunk_overlap_length": 200, "split_search_window": 400},
        "pages": [{"page": 1, "text": "page one"}, {"page_number": 2, "text": "page two"}],
        "document_markdown": "# Title",
        "chunks": [
            _chunk(1, "Intro", section_path="1 Intro", section_part_count=1),
            _chunk(2, "x" * 600, section_path="2 Method", section_part_index=1, section_part_count=2),
            _chunk(3, "y" * 300, section_path="2 Method", section_part_index=2, section_part_count=2, subchunk_index=1, subchunk_count=2),
            _chunk(4, "Figure 1", chunk_type="figure", asset_path=str(assets / "p1" / "fig.png"), asset_abs_path="/abs/fig.png"),
            _chunk(5, "Figure 2", chunk_type="figure", asset_path=str(outside)),
        ],
    })
    old_file = write_chunk_file("p1_old.json", {"chunks": [_chunk(1, "old chunk")]})

    store = _FakeIndexStore()
    store.indexes["p1"] = {"arxiv_id": "p1", "status": "indexed", "active_build_id": "b-new"}
    store.builds["p1"] = [
        {"build_id": "b-new", "status": "active", "is_active": True, "chunk_file": new_file, "error_message": "/srv/x failed"},
        {"build_id": "b-old", "status": "build_failed", "is_active": False, "chunk_file": old_file},
    ]
    store.indexes["legacy"] = {"arxiv_id": "legacy", "status": "indexed", "active_build_id": "legacy-1"}
    store.builds["legacy"] = [{"build_id": "legacy-1", "status": "active", "is_active": True, "chunk_file": None}]
    store.indexes["gone"] = {"arxiv_id": "gone", "status": "indexed", "chunk_file": "01-loaded-docs\\gone.json"}

    app = FastAPI()
    app.include_router(admin_index_router.router, prefix="/api")
    app.dependency_overrides[dependencies.get_paper_qa_index_store] = lambda: store
    return TestClient(app), store


def test_list_passes_filters_to_store(env):
    client, store = env
    response = client.get("/api/admin/paper-indexes", params={"q": "rag", "status": "indexed", "offset": 50, "limit": 50})
    assert response.status_code == 200
    assert response.json()["total"] == 3
    assert store.list_kwargs == {"query": "rag", "status": "indexed", "offset": 50, "limit": 50}


def test_detail_defaults_to_active_build_with_stats_and_full_chunks(env):
    client, _ = env
    body = client.get("/api/admin/paper-indexes/p1").json()

    assert body["selected_build_id"] == "b-new"
    assert [build["build_id"] for build in body["builds"]] == ["b-new", "b-old"]
    assert "error_message" not in body["builds"][0] and "chunk_file" not in body["builds"][0]
    artifact = body["artifact"]
    assert artifact["available"] is True
    assert artifact["overview"]["chunking_params"]["chunk_size"] == 500
    chunks = artifact["chunks"]
    assert chunks[1]["content"] == "x" * 600 and chunks[1]["char_count"] == 600
    assert chunks[3]["has_image"] is True and chunks[0]["has_image"] is False
    assert "asset_path" not in chunks[3]["metadata"] and "asset_abs_path" not in chunks[3]["metadata"]

    stats = artifact["stats"]
    assert stats["by_type"] == {"text": 3, "figure": 2}
    assert stats["text_chars"]["max"] == 600 and stats["text_chars"]["p50"] == 300
    assert stats["short_text_count"] == 1
    assert stats["split_by_size_count"] == 2
    assert stats["split_overlong_count"] == 1
    assert stats["section_count"] == 2
    assert sum(bucket["count"] for bucket in stats["length_histogram"]) == 3


def test_detail_can_select_older_build_and_rejects_unknown(env):
    client, _ = env
    body = client.get("/api/admin/paper-indexes/p1", params={"build_id": "b-old"}).json()
    assert body["selected_build_id"] == "b-old"
    assert body["artifact"]["chunks"][0]["content"] == "old chunk"
    # 旧文件没有参数快照，前端据此提示「未记录」。
    assert body["artifact"]["overview"]["chunking_params"] is None

    assert client.get("/api/admin/paper-indexes/p1", params={"build_id": "nope"}).status_code == 404
    assert client.get("/api/admin/paper-indexes/unknown").status_code == 404


def test_detail_reports_why_artifact_is_unavailable(env):
    client, _ = env
    assert client.get("/api/admin/paper-indexes/legacy").json()["artifact"] == {"available": False, "reason": "not_recorded"}
    # 没有版本记录的旧索引退回主表的 chunk_file。
    assert client.get("/api/admin/paper-indexes/gone").json()["artifact"] == {"available": False, "reason": "missing"}


def test_parsed_document_returns_pages_and_markdown(env):
    client, _ = env
    body = client.get("/api/admin/paper-indexes/p1/parsed").json()
    assert body == {"pages": [{"page_number": 1, "text": "page one"}, {"page_number": 2, "text": "page two"}], "markdown": "# Title"}
    assert client.get("/api/admin/paper-indexes/legacy/parsed").status_code == 404


def test_figure_asset_only_serves_figures_inside_asset_root(env):
    client, _ = env
    response = client.get("/api/admin/paper-indexes/p1/assets/4")
    assert response.status_code == 200 and response.content == b"png-bytes"
    # 文本 chunk、资产根目录之外的路径、不存在的 chunk 一律 404。
    assert client.get("/api/admin/paper-indexes/p1/assets/1").status_code == 404
    assert client.get("/api/admin/paper-indexes/p1/assets/5").status_code == 404
    assert client.get("/api/admin/paper-indexes/p1/assets/99").status_code == 404


def test_chunk_file_lookup_cannot_escape_loaded_docs(env, tmp_path):
    (tmp_path / "escape.json").write_text(json.dumps({"chunks": []}), encoding="utf-8")
    with pytest.raises(index_inspection.ChunkArtifactUnavailable) as exc:
        index_inspection.load_chunk_artifact("01-loaded-docs/../escape.json/..")
    assert exc.value.reason == "missing"
    with pytest.raises(index_inspection.ChunkArtifactUnavailable):
        index_inspection.load_chunk_artifact("..")
