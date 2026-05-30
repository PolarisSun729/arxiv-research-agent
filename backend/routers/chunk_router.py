from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chunks", tags=["chunks"])

BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_DOCS_DIR = BASE_DIR / "01-loaded-docs"


@router.get("/files")
async def list_chunk_files():
    try:
        chunk_files = []
        if not CHUNK_DOCS_DIR.exists():
            logger.warning("Chunk docs directory does not exist: %s", CHUNK_DOCS_DIR)
            return {"status": "success", "files": []}

        for file_path in CHUNK_DOCS_DIR.iterdir():
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

        chunk_files.sort(key=lambda item: item["modified_time"], reverse=True)
        return {"status": "success", "files": chunk_files}
    except Exception as exc:
        logger.error("Error listing chunk files: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/file/{filename}")
async def get_chunk_file(filename: str):
    try:
        file_path = CHUNK_DOCS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        return {"status": "success", "filename": filename, "data": data}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading chunk file: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

