"""管理端只读查看论文索引的切分与解析产物。

数据源是索引构建时落盘的 chunk 文件（backend/01-loaded-docs/*.json）：
- 由索引版本记录里的 chunk_file 定位，调用方只能指定 arxiv_id / build_id，不能传文件路径；
- chunk_file 可能是相对路径、本机绝对路径或旧发布目录下的绝对路径，统一只取文件名在 01-loaded-docs 下查找；
- 对外视图去掉本地路径字段，正文保留全文：管理员要看的正是完整的切分结果。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.storage_paths import BACKEND_DOCLING_ASSETS_ROOT, resolve_backend_artifact_path

LOADED_DOCS_ROOT = resolve_backend_artifact_path("01-loaded-docs")
# 正文 chunk 少于这个字符数视为过短：多半是孤立标题或残句，检索时主要贡献噪声。
SHORT_TEXT_CHUNK_CHARS = 200
# 正文 chunk 字符数直方图的分桶上界（左闭右开），最后一桶收纳所有更长的 chunk。
LENGTH_BUCKET_EDGES = (100, 200, 500, 1000, 2000, 4000, 8000)
# chunk 元数据里指向本地文件的字段不对外返回，是否有图由 has_image 表达。
_LOCAL_PATH_METADATA_KEYS = frozenset({"asset_path", "asset_abs_path", "asset_json_path", "image_path", "source_path"})
# 版本列表对外展示的字段；error_message 可能夹带本地路径，只给 failed_stage。
_BUILD_FIELDS = (
    "build_id", "status", "is_active", "artifact_status", "chunk_count", "loading_method",
    "chunking_strategy", "failed_stage", "indexed_at", "created_at",
)


class ChunkArtifactUnavailable(Exception):
    """所选版本没有可读的 chunk 文件；reason 为 not_recorded（未记录）、missing（已清理）或 invalid（内容损坏）。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def select_build(
    index_record: Mapping[str, Any],
    builds: Sequence[Mapping[str, Any]],
    build_id: str = "",
) -> Optional[Mapping[str, Any]]:
    """选出要查看的构建版本：显式指定的 build_id → 当前生效版本 → 最近一次构建。"""
    if build_id:
        return next((build for build in builds if build.get("build_id") == build_id), None)
    active_build_id = str(index_record.get("active_build_id") or "")
    active = next((build for build in builds if build.get("is_active") or build.get("build_id") == active_build_id), None)
    return active or (builds[0] if builds else None)


def project_build(build: Mapping[str, Any]) -> Dict[str, Any]:
    """版本列表里的一行；has_chunk_file 只看记录，不检查文件是否还在。"""
    return {**{field: build.get(field) for field in _BUILD_FIELDS}, "has_chunk_file": bool(build.get("chunk_file"))}


def resolve_chunk_file(raw_chunk_file: Any) -> Path:
    """把记录里的 chunk_file 映射到 01-loaded-docs 下的同名文件，不跟随记录里的目录部分。"""
    name = str(raw_chunk_file or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not name:
        raise ChunkArtifactUnavailable("not_recorded")
    root = LOADED_DOCS_ROOT.resolve()
    path = (root / name).resolve()
    if path.parent != root or path.suffix.lower() != ".json" or not path.is_file():
        raise ChunkArtifactUnavailable("missing")
    return path


def load_chunk_artifact(raw_chunk_file: Any) -> Dict[str, Any]:
    """读取并校验 chunk 文件；结构不对时按 invalid 处理，不把解析异常带给调用方。"""
    path = resolve_chunk_file(raw_chunk_file)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        raise ChunkArtifactUnavailable("invalid")
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        raise ChunkArtifactUnavailable("invalid")
    return payload


def project_chunk(chunk: Mapping[str, Any]) -> Dict[str, Any]:
    """单个 chunk 的对外视图：全文、字符数、是否有原图，以及去掉本地路径后的元数据。"""
    metadata = dict(chunk.get("metadata") or {})
    content = str(chunk.get("content") or "")
    has_image = metadata.get("chunk_type") == "figure" and bool(metadata.get("asset_path") or metadata.get("asset_abs_path"))
    return {
        "content": content,
        "char_count": len(content),
        "has_image": has_image,
        "metadata": {key: value for key, value in metadata.items() if key not in _LOCAL_PATH_METADATA_KEYS},
    }


def _percentile(sorted_values: Sequence[int], ratio: float) -> int:
    """最近秩法取分位数；样本少时比插值更直观，返回的一定是某个真实 chunk 的长度。"""
    if not sorted_values:
        return 0
    rank = max(1, math.ceil(len(sorted_values) * ratio))
    return int(sorted_values[rank - 1])


def summarize_chunks(chunks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """统计切分效果，入参是 project_chunk 的输出。

    长度分布只统计正文 chunk：图片/表格 chunk 的长度由资产本身决定，与切分参数无关。
    - split_by_size_count：所在章节因超过 chunk_size 被拆成多段的 chunk 数；
    - split_overlong_count：因超过 max_chunk_content_length 被二次硬切的 chunk 数。
    """
    by_type: Dict[str, int] = {}
    text_lengths: List[int] = []
    sections = set()
    split_by_size = 0
    split_overlong = 0
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        chunk_type = str(metadata.get("chunk_type") or "text")
        by_type[chunk_type] = by_type.get(chunk_type, 0) + 1
        if int(metadata.get("subchunk_count") or 1) > 1:
            split_overlong += 1
        if chunk_type != "text":
            continue
        text_lengths.append(int(chunk.get("char_count") or 0))
        if metadata.get("section_path"):
            sections.add(str(metadata["section_path"]))
        if int(metadata.get("section_part_count") or 1) > 1:
            split_by_size += 1

    text_lengths.sort()
    histogram = [
        {"min": lower, "max": upper, "count": sum(1 for length in text_lengths if lower <= length and (upper is None or length < upper))}
        for lower, upper in zip((0, *LENGTH_BUCKET_EDGES), (*LENGTH_BUCKET_EDGES, None))
    ]
    return {
        "total": len(chunks),
        "by_type": by_type,
        "text_count": len(text_lengths),
        "section_count": len(sections),
        "text_chars": {
            "min": text_lengths[0] if text_lengths else 0,
            "p50": _percentile(text_lengths, 0.5),
            "p90": _percentile(text_lengths, 0.9),
            "max": text_lengths[-1] if text_lengths else 0,
            "mean": round(sum(text_lengths) / len(text_lengths)) if text_lengths else 0,
        },
        "short_text_threshold": SHORT_TEXT_CHUNK_CHARS,
        "short_text_count": sum(1 for length in text_lengths if length < SHORT_TEXT_CHUNK_CHARS),
        "split_by_size_count": split_by_size,
        "split_overlong_count": split_overlong,
        "length_histogram": histogram,
    }


def build_overview(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """文件级信息：加载/切分方式、切分参数快照，以及解析阶段的计数。

    chunking_params 为 None 表示该文件生成于开始记录参数之前，当时用的参数已无从得知。
    """
    document_metadata = payload.get("document_metadata") or {}
    return {
        "loading_method": payload.get("loading_method"),
        "chunking_strategy": payload.get("chunking_strategy") or payload.get("chunking_method"),
        "total_pages": payload.get("total_pages"),
        "saved_at": document_metadata.get("saved_at") or payload.get("timestamp"),
        "chunking_params": payload.get("chunking_params"),
        "parsing": {
            "docling_item_count": payload.get("docling_item_count"),
            "docling_picture_count": payload.get("docling_picture_item_count"),
            "docling_table_count": payload.get("docling_table_item_count"),
            "markdown_length": payload.get("document_markdown_length"),
        },
        "table_structure": payload.get("table_structure_debug"),
        "retrieval_index": payload.get("retrieval_index_debug"),
    }


def describe_chunk_artifact(raw_chunk_file: Any) -> Dict[str, Any]:
    """详情页主体：文件不可读时返回 available=False 和原因，可读时返回概览、统计和全部 chunk。"""
    try:
        payload = load_chunk_artifact(raw_chunk_file)
    except ChunkArtifactUnavailable as exc:
        return {"available": False, "reason": exc.reason}
    chunks = [project_chunk(chunk) for chunk in payload["chunks"] if isinstance(chunk, Mapping)]
    return {
        "available": True,
        "overview": build_overview(payload),
        "stats": summarize_chunks(chunks),
        "chunks": chunks,
    }


def project_parsed_document(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """解析结果：逐页文本（PyMuPDF 与 Docling 都有）和 Docling 导出的整篇 Markdown（PyMuPDF 为空）。"""
    pages = []
    for position, page in enumerate(payload.get("pages") or [], start=1):
        if not isinstance(page, Mapping):
            continue
        pages.append({
            "page_number": int(page.get("page_number") or page.get("page") or position),
            "text": str(page.get("text") or ""),
        })
    return {"pages": pages, "markdown": str(payload.get("document_markdown") or "")}


def resolve_figure_asset(payload: Mapping[str, Any], chunk_id: int) -> Path:
    """找到图片 chunk 对应的原图，真实路径必须落在 Docling 资产根目录内。

    优先用相对 backend 的 asset_path，换机器或换发布目录后仍能定位；旧文件只有绝对路径时再退回 asset_abs_path。
    """
    for chunk in payload.get("chunks") or []:
        metadata = (chunk.get("metadata") or {}) if isinstance(chunk, Mapping) else {}
        if str(metadata.get("chunk_id")) != str(chunk_id):
            continue
        if metadata.get("chunk_type") != "figure":
            break
        asset_root = os.path.realpath(BACKEND_DOCLING_ASSETS_ROOT)
        for raw_path in (metadata.get("asset_path"), metadata.get("asset_abs_path")):
            text = str(raw_path or "").strip().replace("\\", "/")
            if not text:
                continue
            candidate = os.path.realpath(resolve_backend_artifact_path(text))
            try:
                inside_root = os.path.commonpath([candidate, asset_root]) == asset_root
            except ValueError:
                # Windows 下不同盘符无法求公共前缀，视为越界。
                inside_root = False
            if inside_root and os.path.isfile(candidate):
                return Path(candidate)
        break
    raise FileNotFoundError("figure asset not found")
