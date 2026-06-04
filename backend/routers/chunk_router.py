from __future__ import annotations

"""分块结果查看相关路由。

该模块负责把本地落盘的 chunk JSON 文件以 HTTP 接口形式暴露出去，
主要用于调试文档切分结果、排查解析问题，以及给前端提供已生成 chunk 的文件列表。
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chunks", tags=["chunks"])

BASE_DIR = Path(__file__).resolve().parents[1]
# 约定分块后的调试产物保存在 backend/01-loaded-docs 目录中。
CHUNK_DOCS_DIR = BASE_DIR / "01-loaded-docs"


@router.get("/files")
async def list_chunk_files():
    """列出本地 chunk 结果目录中的所有 JSON 文件。

    返回的数据除了文件名，还附带文件大小和最后修改时间，
    便于前端按时间排序展示，也方便开发者判断最新一次切分产物是否已经生成。
    """
    try:
        chunk_files = []
        # 如果目录不存在，不视为接口错误，而是返回空列表，
        # 这样前端可以自然地展示“暂无数据”的状态。
        if not CHUNK_DOCS_DIR.exists():
            logger.warning("Chunk docs directory does not exist: %s", CHUNK_DOCS_DIR)
            return {"status": "success", "files": []}

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
                    }
                )

        # 按最近修改时间倒序排列，让最新生成的 chunk 文件优先显示。
        chunk_files.sort(key=lambda item: item["modified_time"], reverse=True)
        return {"status": "success", "files": chunk_files}
    except Exception as exc:
        logger.error("Error listing chunk files: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/file/{filename}")
async def get_chunk_file(filename: str):
    """读取指定的 chunk JSON 文件内容。

    该接口通常用于查看某一篇论文或某一次切分任务的完整 chunk 结构，
    方便验证页码、层级、文本内容、元数据是否符合预期。
    """
    try:
        # 直接基于文件名拼接目标路径；这里默认调用方传入的是目录中的具体文件名。
        file_path = CHUNK_DOCS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # 分块文件为标准 JSON，因此直接反序列化后返回给前端。
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return {"status": "success", "filename": filename, "data": data}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading chunk file: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
