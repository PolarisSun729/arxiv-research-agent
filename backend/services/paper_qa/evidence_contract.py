"""Paper QA 证据的对外契约：来源 payload 脱敏，以及图片证据的受控回放路径解析。

安全约束：
- 检索结果里的 asset_path / asset_abs_path 等服务器本地路径只允许在生成链路内部使用，
  绝不能进入 API 响应、会话消息表或 Agent 输出；
- 前端只能拿到 `/api/paper/{arxiv_id}/evidence-assets/{source_id}` 形式的回放 URL，
  真实文件路径由服务端根据已索引论文反查，客户端无法指定任意路径。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from urllib.parse import quote

from services.paper_qa.context_pack_builder import ContextPackBuilder
from utils.storage_paths import BACKEND_DOCLING_ASSETS_ROOT


def build_public_source_payload(
    search_results: List[Dict[str, Any]],
    *,
    arxiv_id: str | None = None,
) -> List[Dict[str, Any]]:
    """生成无服务器路径的来源契约，图片只保留受控回放 URL。"""
    # 复用 ContextPackBuilder 的 source_id 生成规则，保证这里的 source_id 与生成阶段 prompt 中的引用 ID 一致。
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
        # 只有图片证据且原本确实有资产文件时才生成回放 URL；表格/文本证据不暴露资产入口。
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
    # 第一道防线：source_id 不允许携带路径分隔符或上跳目录，直接拒绝可疑输入。
    # 所有失败分支统一抛同一条 FileNotFoundError，避免通过错误信息差异探测服务器上的索引/文件状态。
    if not paper_id or not requested_source_id or any(token in requested_source_id for token in ("/", "\\", "..")):
        raise FileNotFoundError("evidence asset not found")

    qa_index = paper_qa_index_store.get_paper_qa_index(paper_id)
    if not qa_index or qa_index.get("status") != "indexed":
        raise FileNotFoundError("evidence asset not found")
    provider = getattr(enhanced_retrieval_service, "collection_retrieval_index_provider", None)
    if provider is None:
        raise FileNotFoundError("evidence asset not found")
    index = provider.get_index(qa_index["collection_name"], index_record=qa_index)
    # 线性扫描当前 active 索引的文档；source_id 可能来自 source_id/chunk_id 等多个历史字段，全部纳入匹配。
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
        # 第二道防线：解析符号链接后的真实路径必须落在 docling 资产根目录内。
        # 资产根目录与 LoadingService 写入端共用 storage_paths.BACKEND_DOCLING_ASSETS_ROOT（固定在 backend 下），
        # 不再依赖进程当前工作目录，从仓库根目录或其他目录启动后端时也能正确校验。
        resolved = os.path.realpath(raw_path)
        asset_root = os.path.realpath(str(BACKEND_DOCLING_ASSETS_ROOT))
        try:
            is_inside_root = os.path.commonpath([resolved, asset_root]) == asset_root
        except ValueError:
            # Windows 下不同盘符的路径无法求公共前缀，直接视为越界。
            is_inside_root = False
        if not is_inside_root or not os.path.isfile(resolved):
            raise FileNotFoundError("evidence asset not found")
        return resolved
    raise FileNotFoundError("evidence asset not found")
