from __future__ import annotations

"""QA 路由辅助工具函数。

这里放的是多个 QA 相关路由都会复用的小工具，
包括 trace 文件名清洗、检索 trace 定位，以及 QA 索引健康检查信息构建。
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import VectorDBProvider


def sanitize_trace_slug(text: str, max_length: int = 40) -> str:
    """把任意输入文本转换成适合当作目录名/文件名前缀的安全字符串。"""
    # 仅保留英文、数字、中文，其余字符统一替换为下划线，
    # 这样既兼顾可读性，也避免路径中出现空格或特殊字符带来的兼容性问题。
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip())
    # 连续下划线压缩成一个，避免生成过长且难读的名称。
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        # 完全为空时给一个兜底值，避免出现空文件名。
        slug = "query"
    return slug[:max_length]


def get_latest_retrieval_trace(
    enhanced_retrieval_service: Any,
    arxiv_id: str,
    format_name: str = "md",
) -> Optional[Path]:
    """返回指定论文最近一次导出的检索 trace 文件路径。"""
    trace_root = Path(str(enhanced_retrieval_service.trace_export_dir))
    paper_dir = trace_root / sanitize_trace_slug(arxiv_id)
    if not paper_dir.exists() or not paper_dir.is_dir():
        return None

    # 根据调用方希望下载的格式决定查找 .md 或 .json 文件。
    suffix = ".json" if format_name == "json" else ".md"
    trace_files = sorted(
        paper_dir.glob(f"*{suffix}"),
        # 以修改时间倒序排序，列表首项即最近一次 trace。
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
    """构建 QA 索引诊断信息。

    该函数会同时查看数据库中的 QA 索引元数据，以及向量库 Milvus 中的真实集合状态，
    帮助开发者快速判断“索引是否建成功”“库里是否真的有向量”“元数据和实际数量是否一致”。
    """
    # 先读取数据库里的论文 QA 索引记录，这是系统认知中的“应该存在什么”。
    qa_index = db_service.get_paper_qa_index(arxiv_id)
    # 再读取 Milvus 当前的所有集合，这是向量库中的“实际存在什么”。
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
        # 若数据库侧根本没有索引元数据，则无需继续检查集合详情。
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
            # 读取集合统计信息，例如实体总数、schema 等。
            collection_info = vector_store_service.get_collection_info(VectorDBProvider.MILVUS.value, collection_name)
            num_entities = int(collection_info.get("num_entities") or 0)
            if num_entities > 0:
                # 抽样读取少量 chunk，验证该集合不只是“存在”，而且“可查询、可返回”。
                sample_chunks = vector_store_service.get_all_chunks(
                    collection_name,
                    limit=min(max(sample_limit, 1), num_entities),
                )
        except Exception as exc:
            collection_error = str(exc)
    elif collection_name:
        # 数据库里记录了集合名，但 Milvus 中找不到，通常说明索引构建或迁移不完整。
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
        # 最关键的核对项之一：数据库记录的 chunk 数是否与 Milvus 实体数一致。
        "entity_count_matches_metadata": chunk_count == num_entities,
        "milvus_has_entities": num_entities > 0,
        "sample_chunks_returned": len(sample_chunks),
        # 经验判断：有集合且至少有实体时，关键词/向量召回通常才具备正常工作基础。
        "likely_keyword_search_will_work": collection_exists and num_entities > 0,
    }
    return diagnostic
