from __future__ import annotations

import os
from pathlib import Path

import pytest

import services.paper_qa.paper_qa_index_builder as builder_module
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder


def _link_directory(link: Path, target: Path) -> None:
    """建目录软链接；Windows 没有建软链接权限时改用目录联接，Path.resolve 同样会跟随。"""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("当前环境不允许创建软链接")
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


@pytest.fixture
def release_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """模拟生产发布目录：releases/<版本>/backend 下的产物目录是指向 shared 的软链接。"""
    backend = tmp_path / "releases" / "r1" / "backend"
    shared = tmp_path / "shared" / "backend"
    backend.mkdir(parents=True)
    for dir_name in ("01-loaded-docs", "02-retrieval-indexes", "02-sparse-indexes"):
        (shared / dir_name).mkdir(parents=True)
        _link_directory(backend / dir_name, shared / dir_name)
    monkeypatch.setattr(PaperQAIndexBuilder, "_workspace_root", lambda self: backend)
    monkeypatch.setattr(builder_module, "resolve_backend_artifact_path", lambda path: (backend / path).resolve())
    return {"backend": backend, "shared": shared}


def _builder() -> PaperQAIndexBuilder:
    # 删除逻辑只依赖路径解析，不需要构造完整的索引构建依赖。
    return PaperQAIndexBuilder.__new__(PaperQAIndexBuilder)


def test_delete_follows_symlinked_artifact_dirs(release_layout: dict[str, Path]) -> None:
    backend, shared = release_layout["backend"], release_layout["shared"]
    chunk_file = shared / "01-loaded-docs" / "paper.json"
    retrieval_file = shared / "02-retrieval-indexes" / "paper_retrieval_indexes.json"
    sparse_dir = shared / "02-sparse-indexes" / "paper" / "v1"
    for file_path in (chunk_file, retrieval_file, sparse_dir / "manifest.json"):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("{}", encoding="utf-8")

    builder = _builder()
    # 记录里既可能是 shared 下的真实路径，也可能是经由发布目录软链接的路径，都要能删。
    assert builder._safe_delete_file(str(chunk_file)) is True
    assert builder._safe_delete_file(str(backend / "02-retrieval-indexes" / retrieval_file.name)) is True
    assert builder._safe_delete_directory(str(sparse_dir)) is True
    assert not chunk_file.exists() and not retrieval_file.exists() and not sparse_dir.exists()


def test_delete_rejects_paths_outside_artifact_dirs(release_layout: dict[str, Path], tmp_path: Path) -> None:
    shared = release_layout["shared"]
    outside_file = tmp_path / "shared" / "other.json"
    outside_file.write_text("{}", encoding="utf-8")

    builder = _builder()
    assert builder._safe_delete_file(str(outside_file)) is False
    assert outside_file.exists()
    # 产物根目录本身不能被整个删掉。
    assert builder._safe_delete_directory(str(shared / "02-sparse-indexes")) is False
    assert (shared / "02-sparse-indexes").exists()
