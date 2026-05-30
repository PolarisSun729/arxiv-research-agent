from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import VectorDBProvider


def sanitize_trace_slug(text: str, max_length: int = 40) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "query"
    return slug[:max_length]


def get_latest_retrieval_trace(
    enhanced_retrieval_service: Any,
    arxiv_id: str,
    format_name: str = "md",
) -> Optional[Path]:
    trace_root = Path(str(enhanced_retrieval_service.trace_export_dir))
    paper_dir = trace_root / sanitize_trace_slug(arxiv_id)
    if not paper_dir.exists() or not paper_dir.is_dir():
        return None

    suffix = ".json" if format_name == "json" else ".md"
    trace_files = sorted(
        paper_dir.glob(f"*{suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return trace_files[0] if trace_files else None


def build_qa_diagnostic(
    *,
    db_service: Any,
    vector_store_service: Any,
    arxiv_id: str,
    sample_limit: int = 3,
) -> Dict[str, Any]:
    qa_index = db_service.get_paper_qa_index(arxiv_id)
    all_collections = vector_store_service.list_collections(VectorDBProvider.MILVUS.value)

    diagnostic: Dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "qa_index": qa_index,
        "milvus": {
            "provider": VectorDBProvider.MILVUS.value,
            "collections": all_collections,
        },
        "collection": None,
        "sample_chunks": [],
        "checks": {},
    }

    if not qa_index:
        diagnostic["checks"] = {
            "has_qa_index": False,
            "collection_exists": False,
            "entity_count_matches_metadata": False,
        }
        return diagnostic

    collection_name = qa_index.get("collection_name", "")
    collection_exists = vector_store_service.collection_exists(VectorDBProvider.MILVUS.value, collection_name)
    collection_info: Dict[str, Any] = {}
    sample_chunks: List[Dict[str, Any]] = []
    collection_error: Optional[str] = None

    if collection_name and collection_exists:
        try:
            collection_info = vector_store_service.get_collection_info(VectorDBProvider.MILVUS.value, collection_name)
            num_entities = int(collection_info.get("num_entities") or 0)
            if num_entities > 0:
                sample_chunks = vector_store_service.get_all_chunks(
                    collection_name,
                    limit=min(max(sample_limit, 1), num_entities),
                )
        except Exception as exc:
            collection_error = str(exc)
    elif collection_name:
        collection_error = "collection_name not found in Milvus list_collections()"

    num_entities = int(collection_info.get("num_entities") or 0)
    chunk_count = int(qa_index.get("chunk_count") or 0)

    diagnostic["collection"] = {
        "name": collection_name,
        "exists_in_milvus": collection_exists,
        "info": collection_info or None,
        "error": collection_error,
    }
    diagnostic["sample_chunks"] = sample_chunks
    diagnostic["checks"] = {
        "has_qa_index": True,
        "indexed_status": qa_index.get("status") == "indexed",
        "collection_exists": collection_exists,
        "qa_chunk_count": chunk_count,
        "milvus_num_entities": num_entities,
        "entity_count_matches_metadata": chunk_count == num_entities,
        "milvus_has_entities": num_entities > 0,
        "sample_chunks_returned": len(sample_chunks),
        "likely_keyword_search_will_work": collection_exists and num_entities > 0,
    }
    return diagnostic
