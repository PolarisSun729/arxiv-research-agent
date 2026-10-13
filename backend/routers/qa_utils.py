"""QA 路由辅助工具函数。

这里放的是 qa_router 用到的几个无状态小工具（单独成模块便于单元测试），
包括 trace 文件名清洗、检索 trace 定位，以及 QA 索引健康检查信息构建。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from services.retrieval.retrieval_index import iter_retrieval_indexes_for_embedding, load_retrieval_index_artifact
from services.retrieval.trace_builder import RetrievalTraceBuilder
from utils.config import VectorDBProvider

# 把任意文本转换成可安全用作目录名/文件名前缀的 slug：只保留英文、数字、中文，
# 其余字符替换为下划线并压缩，结果为空时回退为 "query"，长度上限来自检索配置。
# 直接复用写 trace 时的同一实现：trace 目录名由检索服务生成、由这里的下载接口查找，
# 两端规则（包括长度上限）一旦不一致就会找不到文件，因此不能各自维护一份拷贝。
sanitize_trace_slug = RetrievalTraceBuilder.sanitize_trace_slug


def get_latest_retrieval_trace(
    enhanced_retrieval_service: Any,
    arxiv_id: str,
    format_name: str = "md",
) -> Optional[Path]:
    """返回指定论文最近一次导出的检索 trace 文件路径；目录不存在或没有对应格式的文件时返回 None。

    trace 目录结构：<trace_export_dir>/<arxiv_id 的 slug>/<按查询生成的文件名>.{md,json}
    """
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


def _expected_vector_count(qa_index: Dict[str, Any]) -> Optional[int]:
    """按建库时的同一筛选规则，从 retrieval index 文件算出应写入 Milvus 的条数；读不到时返回 None。

    Milvus 里存的是 RetrievalIndex 向量而不是原始 chunk：一个 chunk 会派生 body/summary/question 等多条，
    section_anchor 这类只走关键词路线的索引又不入向量库，所以既不能拿 chunk_count，也不能拿 retrieval_index_count 去比。
    """
    try:
        artifact = load_retrieval_index_artifact(qa_index.get("retrieval_index_file"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    raw_indexes = artifact.get("retrieval_indexes") or artifact.get("indexes") or []
    if not isinstance(raw_indexes, list):
        return None
    return len(iter_retrieval_indexes_for_embedding(raw_indexes))


def build_qa_diagnostic(
    *,
    paper_qa_index_store: Any,
    vector_store_service: Any,
    arxiv_id: str,
    sample_limit: int = 3,
) -> Dict[str, Any]:
    """构建 QA 索引诊断信息。

    该函数会同时查看数据库中的 QA 索引元数据，以及向量库 Milvus 中的真实集合状态，
    帮助开发者快速判断“索引是否建成功”“库里是否真的有向量”“元数据和实际数量是否一致”。
    """
    # 先读取 SQLite 中的论文 QA 索引记录，这是系统认知中的“应该存在什么”。
    qa_index = paper_qa_index_store.get_paper_qa_index(arxiv_id)
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
            "entity_count_matches_metadata": None,
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
    expected_vector_count = _expected_vector_count(qa_index)

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
        "expected_vector_count": expected_vector_count,
        "milvus_num_entities": num_entities,
        # 最关键的核对项之一：应入向量库的索引条数是否与 Milvus 实体数一致；索引文件读不到时为 None（无法判断）。
        "entity_count_matches_metadata": None if expected_vector_count is None else expected_vector_count == num_entities,
        "milvus_has_entities": num_entities > 0,
        "sample_chunks_returned": len(sample_chunks),
        # 经验判断：有集合且至少有实体时，关键词/向量召回通常才具备正常工作基础。
        "likely_keyword_search_will_work": collection_exists and num_entities > 0,
    }
    return diagnostic
