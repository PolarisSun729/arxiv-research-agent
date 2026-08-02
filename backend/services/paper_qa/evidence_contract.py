from __future__ import annotations

import os
from typing import Any, Dict, List
from urllib.parse import quote

from services.paper_qa.context_pack_builder import ContextPackBuilder


def build_public_source_payload(
    search_results: List[Dict[str, Any]],
    *,
    arxiv_id: str | None = None,
) -> List[Dict[str, Any]]:
    """生成无服务器路径的来源契约，图片只保留受控回放 URL。"""
    raw_sources = ContextPackBuilder().build_source_payload(search_results)
    public_sources: List[Dict[str, Any]] = []
    paper_id = str(arxiv_id or "").strip()
    for source in raw_sources:
        item = dict(source)
        had_asset = any(str(item.get(key) or "").strip() for key in ("asset_path", "asset_abs_path"))
        # 绝对路径只允许留在生成器内部的 asset_metadata，不能进入前端或持久化消息。
        for private_key in ("asset_path", "asset_abs_path", "image_path", "asset_json_path"):
            item.pop(private_key, None)
        source_id = str(item.get("source_id") or "").strip()
        if paper_id and source_id and str(item.get("chunk_type") or "").lower() == "figure" and had_asset:
            item["asset_url"] = f"/api/paper/{quote(paper_id, safe='')}/evidence-assets/{quote(source_id, safe='')}"
        public_sources.append(item)
    return public_sources


def resolve_evidence_asset_path(
    *,
    arxiv_id: str,
    source_id: str,
    paper_qa_index_store: Any,
    enhanced_retrieval_service: Any,
) -> str:
    """从已索引论文反查图片，并限制路径在 docling 资产根目录内。"""
    paper_id = str(arxiv_id or "").strip()
    requested_source_id = str(source_id or "").strip()
    if not paper_id or not requested_source_id or any(token in requested_source_id for token in ("/", "\\", "..")):
        raise FileNotFoundError("evidence asset not found")

    qa_index = paper_qa_index_store.get_paper_qa_index(paper_id)
    if not qa_index or qa_index.get("status") != "indexed":
        raise FileNotFoundError("evidence asset not found")
    provider = getattr(enhanced_retrieval_service, "collection_retrieval_index_provider", None)
    if provider is None:
        raise FileNotFoundError("evidence asset not found")
    index = provider.get_index(qa_index["collection_name"], index_record=qa_index)
    for document in getattr(index, "documents", []) or []:
        chunk = getattr(document, "chunk", {}) or {}
        source_candidates = {
            str(chunk.get(key) or "").strip()
            for key in ("source_id", "chunk_id", "parent_chunk_id", "original_chunk_id", "id")
            if str(chunk.get(key) or "").strip()
        }
        if requested_source_id not in source_candidates:
            continue
        if str(chunk.get("chunk_type") or "").strip().lower() != "figure":
            raise FileNotFoundError("evidence asset not found")
        raw_path = str(chunk.get("asset_abs_path") or "").strip()
        if not raw_path:
            raise FileNotFoundError("evidence asset not found")
        resolved = os.path.realpath(raw_path)
        asset_root = os.path.realpath(os.path.join(os.getcwd(), "03-docling-assets"))
        try:
            is_inside_root = os.path.commonpath([resolved, asset_root]) == asset_root
        except ValueError:
            is_inside_root = False
        if not is_inside_root or not os.path.isfile(resolved):
            raise FileNotFoundError("evidence asset not found")
        return resolved
    raise FileNotFoundError("evidence asset not found")
