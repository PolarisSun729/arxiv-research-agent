from __future__ import annotations

"""受控 chunk 调试路由。

该模块只在 ENABLE_DEBUG_ROUTES 显式开启时注册，用于本地排查 PDF 解析、
chunk 切分和 RAG 召回质量；它不是面向普通用户的正式业务 API。
"""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from utils.config import get_debug_routes_runtime_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug/chunks", tags=["debug-chunks"])

BASE_DIR = Path(__file__).resolve().parents[1]
# 约定分块后的调试产物保存在 backend/01-loaded-docs 目录中。
CHUNK_DOCS_DIR = BASE_DIR / "01-loaded-docs"
DEBUG_CONFIG = get_debug_routes_runtime_config()
CONTENT_PREVIEW_CHARS = int(DEBUG_CONFIG.get("chunk_content_preview_chars") or 4000)

_PATH_FIELD_NAMES = {
    "path",
    "pdf_path",
    "chunk_file",
    "embedding_file",
    "asset_path",
    "asset_abs_path",
    "asset_json_path",
    "image_path",
    "source_path",
}
_LARGE_VECTOR_FIELD_NAMES = {"embedding", "vector", "vector_data"}
_RAW_PAGE_FIELD_NAMES = {"pages", "raw_pages"}
_LONG_TEXT_FIELD_NAMES = {"content", "text", "raw_text", "markdown"}


def _basename(value: Any) -> str:
    text = str(value or "")
    return text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _resolve_chunk_debug_file(filename: str) -> Path:
    """只允许读取调试目录下的 JSON 文件，避免路径穿越读取任意本地文件。"""
    normalized_name = _basename(filename)
    if not normalized_name or normalized_name != filename or Path(normalized_name).suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Invalid debug chunk filename")
    base_dir = CHUNK_DOCS_DIR.resolve()
    file_path = (base_dir / normalized_name).resolve()
    if file_path.parent != base_dir:
        raise HTTPException(status_code=400, detail="Invalid debug chunk filename")
    return file_path


def _truncate_debug_text(value: str) -> str:
    if len(value) <= CONTENT_PREVIEW_CHARS:
        return value
    return value[:CONTENT_PREVIEW_CHARS] + "\n...[debug preview truncated]"


def _sanitize_debug_payload(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """裁剪 chunk 调试响应，只保留定位问题所需字段，避免泄露本地路径和大体量向量。"""
    key_lower = str(key or "").lower()
    if depth > 10:
        return "[debug payload depth limit]"
    if key_lower in _RAW_PAGE_FIELD_NAMES:
        return "[redacted: raw pages omitted from debug API]"
    if key_lower in _LARGE_VECTOR_FIELD_NAMES and isinstance(value, list):
        return {"redacted": True, "vector_length": len(value)}
    if key_lower in _PATH_FIELD_NAMES:
        return _basename(value)
    if key_lower in {"filename", "source"} and isinstance(value, str):
        return _basename(value)
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_debug_payload(item_value, key=str(item_key), depth=depth + 1)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_debug_payload(item, key=key, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if key_lower in _LONG_TEXT_FIELD_NAMES:
            return _truncate_debug_text(value)
        if value.startswith(("/", "\\")) or ":\\" in value or ":/" in value:
            return _basename(value)
    return value


@router.get("/files")
async def list_chunk_files():
    """列出本地 chunk 调试结果目录中的所有 JSON 文件。

    返回的数据除了文件名，还附带文件大小和最后修改时间，
    便于前端按时间排序展示，也方便开发者判断最新一次切分产物是否已经生成。
    """
    try:
        chunk_files = []
        # 如果目录不存在，不视为接口错误，而是返回空列表，
        # 这样前端可以自然地展示“暂无数据”的状态。
        if not CHUNK_DOCS_DIR.exists():
            logger.warning("Chunk docs directory does not exist: %s", CHUNK_DOCS_DIR)
            return {"status": "success", "debug": True, "files": []}

        for file_path in CHUNK_DOCS_DIR.iterdir():
            # 这里只暴露 JSON 文件，因为该目录中真正有业务价值的是分块结果文件；
            # 其他临时文件或隐藏文件不应进入接口返回。
            if file_path.is_file() and file_path.suffix.lower() == ".json":
                file_size = file_path.stat().st_size
                modified_time = file_path.stat().st_mtime
                chunk_files.append(
                    {
                        "filename": file_path.name,
                        "size": file_size,
                        "modified_time": modified_time,
                        "debug_only": True,
                    }
                )

        # 按最近修改时间倒序排列，让最新生成的 chunk 文件优先显示。
        chunk_files.sort(key=lambda item: item["modified_time"], reverse=True)
        return {"status": "success", "debug": True, "files": chunk_files}
    except Exception as exc:
        logger.error("Error listing chunk files: %s", str(exc))
        # 调试接口仍不把本地异常细节直接返回给调用方，避免把内部路径或文件结构带到响应里。
        raise HTTPException(status_code=500, detail="Failed to list debug chunk files")


@router.get("/file/{filename}")
async def get_chunk_file(filename: str):
    """读取指定的 chunk JSON 文件内容，并返回裁剪后的调试视图。

    该接口用于验证页码、层级、文本内容和元数据是否符合预期；响应会移除本地路径、
    原始页面全文和 embedding 向量，避免把内部调试产物当成正式数据接口暴露。
    """
    try:
        file_path = _resolve_chunk_debug_file(filename)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # 分块文件为标准 JSON，因此直接反序列化后返回给前端。
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return {
            "status": "success",
            "debug": True,
            "filename": file_path.name,
            "sanitized": True,
            "data": _sanitize_debug_payload(data),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading chunk file: %s", str(exc))
        # 读取失败时仅暴露稳定错误语义；具体文件路径和解析异常保留在后端日志中。
        raise HTTPException(status_code=500, detail="Failed to read debug chunk file")
