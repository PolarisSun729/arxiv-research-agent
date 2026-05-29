import os
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Query, Request, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from numpy import False_
from services.loading_service import LoadingService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService, EmbeddingConfig
from services.vector_store_service import VectorStoreService, VectorDBConfig
from services.search_service import SearchService
from services.parsing_service import ParsingService
from services.arxiv_search_service import (
    ArxivSearchService,
    ArxivSearchValidationError,
    build_arxiv_query_from_structured_params,
    build_arxiv_submitted_date_query,
    validate_arxiv_search_request,
)
from services.local_arxiv_service import LocalArxivService
from services.database_service import DatabaseService
from services.arxiv_oai_service import ArxivOaiDatabaseService
from services.recommendation_service import RecommendationService
from services.enhanced_retrieval_service import EnhancedRetrievalService, RetrievalOptions
from services.paper_qa_service import PaperQAService
import logging
from enum import Enum
from utils.config import CORE_CONFIG, VectorDBProvider, get_recommendation_clustering_runtime_config
import pandas as pd
from pathlib import Path
from services.generation_service import GenerationService
from typing import List, Dict, Optional, Any
import requests
from pydantic import BaseModel

# # 设置 Clash 代理 (默认端口 7890)
# PROXY_URL = "http://127.0.0.1:7897"
# os.environ["HTTP_PROXY"] = PROXY_URL
# os.environ["HTTPS_PROXY"] = PROXY_URL
# os.environ["http_proxy"] = PROXY_URL
# os.environ["https_proxy"] = PROXY_URL

# # 配置 requests 使用代理
# requests.Session.proxies = {
#     'http': PROXY_URL,
#     'https': PROXY_URL,
# }

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CHUNK_DOCS_DIR = BASE_DIR / "01-loaded-docs"

app = FastAPI()

# 确保必要的目录存在
os.makedirs("temp", exist_ok=True)
os.makedirs("01-chunked-docs", exist_ok=True)
os.makedirs("02-embedded-docs", exist_ok=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据源配置
DATA_SOURCE = CORE_CONFIG["arxiv_data_source"]
ARXIV_PROXY_URL = CORE_CONFIG.get("arxiv_proxy_url", "")

# 初始化服务
db_service = DatabaseService()
oai_db_service = ArxivOaiDatabaseService()
embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()
generation_service = GenerationService()
enhanced_retrieval_service = EnhancedRetrievalService(
    embedding_service=embedding_service,
    vector_store_service=vector_store_service,
    generation_service=generation_service,
)

# 初始化 arXiv 服务
local_arxiv_service = LocalArxivService()


def get_current_embedding_config() -> EmbeddingConfig:
    return embedding_service.get_default_embedding_config()


def get_current_recommendation_clustering_config() -> dict:
    return get_recommendation_clustering_runtime_config()


recommendation_service = RecommendationService(
    db_service=db_service,
    embedding_service=embedding_service,
    vector_store_service=vector_store_service,
    get_embedding_config=get_current_embedding_config,
    get_clustering_config=get_current_recommendation_clustering_config,
    arxiv_service_factory=lambda: get_arxiv_service(),
    oai_db_service=oai_db_service,
)

paper_qa_service = PaperQAService(
    db_service=db_service,
    embedding_service=embedding_service,
    vector_store_service=vector_store_service,
    generation_service=generation_service,
    enhanced_retrieval_service=enhanced_retrieval_service,
    arxiv_service_factory=lambda: get_arxiv_api_service(),
    get_embedding_config=get_current_embedding_config,
)


def embed_text_with_current_config(text: str) -> tuple[list, EmbeddingConfig]:
    config = get_current_embedding_config()
    embedding = embedding_service.create_single_embedding(
        text,
        provider=config.provider,
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.base_url,
        dimension=config.dimension,
    )
    return embedding, config


def get_arxiv_service():
    """根据配置获取当前使用的 arXiv 服务"""
    if DATA_SOURCE == "api":
        return ArxivSearchService(proxy_url=ARXIV_PROXY_URL)
    else:
        return local_arxiv_service


def get_arxiv_api_service() -> ArxivSearchService:
    """始终返回走代理配置的 arXiv API 服务。"""
    return ArxivSearchService(proxy_url=ARXIV_PROXY_URL)


class QaRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    enable_llm_rerank: Optional[bool] = None
    debug: Optional[bool] = None


def build_qa_context(arxiv_id: str, payload: QaRequest):
    return paper_qa_service.build_qa_context(arxiv_id, payload)


def build_generation_context(search_results: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    return paper_qa_service.build_generation_context(search_results)


def build_source_payload(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return paper_qa_service.build_source_payload(search_results)


def _sanitize_trace_slug(text: str, max_length: int = 40) -> str:
    import re

    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "query"
    return slug[:max_length]


def _get_latest_retrieval_trace(arxiv_id: str, format_name: str = "md") -> Optional[Path]:
    trace_root = Path(str(enhanced_retrieval_service.trace_export_dir))
    paper_dir = trace_root / _sanitize_trace_slug(arxiv_id)
    if not paper_dir.exists() or not paper_dir.is_dir():
        return None

    suffix = ".json" if format_name == "json" else ".md"
    trace_files = sorted(
        paper_dir.glob(f"*{suffix}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return trace_files[0] if trace_files else None


def build_qa_diagnostic(arxiv_id: str, sample_limit: int = 3) -> Dict[str, Any]:
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
    collection_info = {}
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


# arXiv 论文搜索接口
@app.post("/arxiv/search")
async def arxiv_search(
    search_query: Optional[str] = Body(None, description="搜索查询字符串，支持字段前缀语法如 ti:deep learning"),
    id_list: Optional[List[str]] = Body(None, description="arXiv论文ID列表，用于精确匹配"),
    title: Optional[str] = Body(None, description="标题关键词"),
    author: Optional[str] = Body(None, description="作者姓名"),
    abstract: Optional[str] = Body(None, description="摘要关键词"),
    category: Optional[str] = Body(None, description="学科分类代码，如 cs.AI"),
    comment: Optional[str] = Body(None, description="评论关键词"),
    journal_ref: Optional[str] = Body(None, description="期刊引用关键词"),
    report_number: Optional[str] = Body(None, description="报告编号关键词"),
    operator: Optional[str] = Body("AND", description="逻辑操作符：AND 或 OR"),
    max_results: int = Body(10),
    start: int = Body(0),
    sort_by: str = Body("relevance"),
    sort_order: str = Body("descending"),
    submitted_days_ago: Optional[int] = Body(None, description="搜索提交日期在多少天内的文章（本地数据源暂不支持）")
):
    """
    搜索 arXiv 论文
    支持两种方式：
    1. 直接传入 search_query 字符串，如 "ti:deep learning+AND+au:John"
    2. 传入各字段自动构建查询，如 title="deep learning", author="John", category="cs.AI"
    
    支持的数据源：
    - 本地数据集（默认）：从 Kaggle 下载的 JSON 文件
    - API：直接调用 arXiv 官方 API
    
    通过环境变量 ARXIV_DATA_SOURCE 切换数据源（local/api）
    """
    try:
        arxiv_service = get_arxiv_service()

        structured_fields_present = any([title, author, abstract, category, comment, journal_ref, report_number])
        if structured_fields_present:
            structured = build_arxiv_query_from_structured_params(
                query=search_query,
                title_query=title,
                author_query=author,
                abstract_query=abstract,
                categories=[category] if category else None,
                comment_query=comment,
                journal_ref_query=journal_ref,
                report_number_query=report_number,
                id_list=id_list,
                field_operator=operator if operator and operator.strip() else "AND",
                category_operator="OR",
                submitted_days_ago=submitted_days_ago,
            )
            validate_arxiv_search_request(
                search_query=structured["final_search_query"],
                id_list=structured["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            results = arxiv_service.search(
                search_query=structured["final_search_query"],
                id_list=structured["id_list"],
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        else:
            validate_arxiv_search_request(
                search_query=search_query,
                id_list=id_list,
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            normalized_search_query = search_query
            if submitted_days_ago is not None and submitted_days_ago >= 0 and normalized_search_query and not id_list:
                normalized_search_query = f"({normalized_search_query}) AND {build_arxiv_submitted_date_query(submitted_days_ago)}"
            results = arxiv_service.search(
                search_query=normalized_search_query,
                id_list=id_list,
                max_results=max_results,
                start=start,
                sort_by=sort_by,
                sort_order=sort_order,
            )

        return results
    except ArxivSearchValidationError as exc:
        logger.error(f"Invalid arXiv search query: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        logger.error(f"Error searching arXiv: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/arxiv/fields")
async def arxiv_get_fields():
    """获取支持的搜索字段列表"""
    try:
        arxiv_service = get_arxiv_service()
        fields = arxiv_service.get_available_fields()
        return {"fields": fields}
    except Exception as e:
        logger.error(f"Error getting arXiv fields: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/arxiv/categories")
async def arxiv_get_categories():
    """获取常用的arXiv学科分类"""
    try:
        arxiv_service = get_arxiv_service()
        categories = arxiv_service.get_subject_categories()
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error getting arXiv categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/arxiv/download")
async def arxiv_download(
    arxiv_id: str = Body(...),
    pdf_url: str = Body(...)
):
    """下载 arXiv 论文 PDF"""
    try:
        arxiv_service = get_arxiv_api_service()
        filepath = arxiv_service.download_pdf(pdf_url, arxiv_id)
        return {"status": "success", "filepath": filepath}
    except Exception as e:
        logger.error(f"Error downloading arXiv paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/arxiv/search-and-save")
async def arxiv_search_and_save(
    search_query: str = Body(""),
    id_list: Optional[List[str]] = Body(None),
    max_results: int = Body(10),
    download_pdfs: bool = Body(False),
    **kwargs
):
    """搜索 arXiv 论文并保存结果，可选择下载 PDF"""
    try:
        arxiv_service = get_arxiv_api_service()
        results = await arxiv_service.search_and_save(
            search_query=search_query,
            id_list=id_list,
            max_results=max_results,
            download_pdfs=download_pdfs,
            **kwargs
        )
        return results
    except Exception as e:
        logger.error(f"Error in arXiv search and save: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 用户偏好管理 API
@app.post("/user/preferences")
async def upsert_user_preferences(
    user_id: str = Body("local_user")
):
    """创建或更新用户偏好"""
    try:
        preferences = db_service.get_user_preferences(user_id=user_id)
        return {"status": "success", "message": "User preferences retrieved", "preferences": preferences}
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """获取用户偏好"""
    try:
        preferences = db_service.get_user_preferences(user_id=user_id)
        return preferences
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/like-paper")
async def like_paper(
    arxiv_id: str = Body(...),
    user_id: str = Body("local_user"),
    paper: Optional[Dict[str, Any]] = Body(None)
):
    """标记论文为喜欢"""
    try:
        return recommendation_service.record_user_paper_preference(
            user_id=user_id,
            arxiv_id=arxiv_id,
            liked=True,
            paper_payload=paper,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error liking paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/dislike-paper")
async def dislike_paper(
    arxiv_id: str = Body(...),
    user_id: str = Body("local_user"),
    paper: Optional[Dict[str, Any]] = Body(None)
):
    """标记论文为不喜欢"""
    try:
        return recommendation_service.record_user_paper_preference(
            user_id=user_id,
            arxiv_id=arxiv_id,
            liked=False,
            paper_payload=paper,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disliking paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user/like-paper")
async def remove_like(
    arxiv_id: str = Body(...),
    user_id: str = Body("local_user")
):
    """从喜欢列表中移除论文"""
    try:
        success = db_service.remove_liked_paper(user_id=user_id, arxiv_id=arxiv_id)
        if success:
            return {"status": "success", "message": "Paper removed from liked list"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove from liked list")
    except Exception as e:
        logger.error(f"Error removing liked paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user/dislike-paper")
async def remove_dislike(
    arxiv_id: str = Body(...),
    user_id: str = Body("local_user")
):
    """从不喜欢列表中移除论文"""
    try:
        success = db_service.remove_disliked_paper(user_id=user_id, arxiv_id=arxiv_id)
        if success:
            return {"status": "success", "message": "Paper removed from disliked list"}
        else:
            raise HTTPException(status_code=500, detail="Failed to remove from disliked list")
    except Exception as e:
        logger.error(f"Error removing disliked paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/generate-interest-vector")
async def generate_user_interest_vector(
    user_id: str = Body("local_user")
):
    try:
        return recommendation_service.generate_user_interest_vector(user_id=user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating interest vector: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/interest-vector")
async def get_user_interest_vector(
    user_id: str = "local_user"
):
    """获取用户兴趣向量"""
    try:
        result = db_service.get_user_interest_vector(user_id=user_id)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="User interest vector not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting interest vector: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/recommend-papers")
async def recommend_papers(
    user_id: str = Body("local_user"),
    top_n: int = Body(10),
    max_age_months: int = Body(6)
):
    try:
        return recommendation_service.recommend_papers(user_id=user_id, top_n=top_n, max_age_months=max_age_months)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 论文元数据管理 API
@app.post("/paper")
async def add_paper(
    arxiv_id: str = Body(...),
    title: str = Body(...),
    authors: str = Body(...),
    abstract: str = Body(...),
    categories: str = Body(...),
    published_date: str = Body(...),
    url: str = Body(...),
    collection_name: str = Body("arxiv_abstracts")
):
    """
    添加论文元数据，并自动使用当前配置的 embedding 模型生成 embedding
    将embedding向量存储到向量数据库，并将embedding_id保存到SQLite
    """
    try:
        logger.info(f"Adding paper with embedding: {arxiv_id}")
        
        embedding_config = get_current_embedding_config()
        logger.info(
            "Creating embedding for abstract using %s / %s",
            embedding_config.provider,
            embedding_config.model_name,
        )
        text_to_embed = embedding_service.build_paper_embedding_text(title, abstract)
        embedding = embedding_service.create_single_embedding(
            text_to_embed,
            provider=embedding_config.provider,
            model=embedding_config.model_name,
            api_key=embedding_config.api_key,
            base_url=embedding_config.base_url,
            dimension=embedding_config.dimension,
        )
        
        logger.info(f"Embedding created, dimension: {len(embedding)}")
        
        metadata = {
            "content": abstract,
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "categories": categories,
            "published_date": published_date,
            "url": url,
            "embedding_model": embedding_config.model_name
        }
        
        logger.info(f"Inserting embedding to collection: {collection_name}")
        embedding_id = vector_store_service.insert_single_embedding(collection_name, embedding, metadata)
        
        logger.info(f"Embedding inserted with ID: {embedding_id}")
        
        # 打印要添加的论文元数据
        logger.info(f"Adding paper to database: {metadata}, embedding_id: {embedding_id}")

        success = db_service.add_paper({
            'arxiv_id': arxiv_id,
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'categories': categories,
            'published_date': published_date,
            'url': url,
            'embedding_id': str(embedding_id),
            'embedding_model': embedding_config.model_name
        })
        
        if success:
            logger.info(f"Paper {arxiv_id} added successfully with embedding")
            return {
                "status": "success",
                "message": "Paper added with embedding",
                "arxiv_id": arxiv_id,
                "embedding_id": embedding_id,
                "embedding_model": embedding_config.model_name,
                "vector_dimension": len(embedding),
                "collection_name": collection_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add paper")
    except Exception as e:
        logger.error(f"Error adding paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/paper/{arxiv_id}")
async def get_paper(arxiv_id: str):
    """获取论文元数据"""
    try:
        paper = db_service.get_paper(arxiv_id)
        if paper:
            return paper
        source_paper = recommendation_service._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
        if source_paper:
            return recommendation_service._materialize_paper_from_source(source_paper, arxiv_id)
        raise HTTPException(status_code=404, detail="Paper not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/paper/{arxiv_id}")
async def delete_paper(arxiv_id: str):
    """删除论文元数据"""
    try:
        success = db_service.delete_paper(arxiv_id)
        if success:
            return {"status": "success", "message": "Paper deleted"}
        else:
            raise HTTPException(status_code=404, detail="Paper not found")
    except Exception as e:
        logger.error(f"Error deleting paper: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers")
async def get_all_papers():
    """获取所有论文列表"""
    try:
        papers = db_service.get_all_papers()
        return {"papers": papers}
    except Exception as e:
        logger.error(f"Error getting all papers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/category/{category}")
async def search_papers_by_category(category: str):
    """按学科分类搜索论文"""
    try:
        papers = db_service.search_papers_by_category(category)
        return {"papers": papers}
    except Exception as e:
        logger.error(f"Error searching papers by category: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper/{arxiv_id}/qa-status")
async def get_paper_qa_status(arxiv_id: str):
    """检查论文是否已有问答索引"""
    try:
        return paper_qa_service.get_qa_status(arxiv_id)
    except Exception as e:
        logger.error(f"Error getting paper QA status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper/{arxiv_id}/qa-diagnose")
async def diagnose_paper_qa(arxiv_id: str, sample_limit: int = Query(3, ge=0, le=20)):
    """返回论文 QA 索引与 Milvus 现场状态的诊断信息。"""
    try:
        return build_qa_diagnostic(arxiv_id, sample_limit=sample_limit)
    except Exception as e:
        logger.error(f"Error diagnosing paper QA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper/{arxiv_id}/qa-trace/latest")
async def download_latest_qa_trace(
    arxiv_id: str,
    format: str = Query("md"),
    trace_name: Optional[str] = Query(None),
):
    """下载该论文最近一次检索的 trace 文件，或下载指定文件名的 trace。"""
    try:
        normalized_format = str(format or "md").strip().lower()
        if normalized_format not in {"md", "json"}:
            raise HTTPException(status_code=400, detail="format must be md or json")

        trace_root = Path(str(enhanced_retrieval_service.trace_export_dir))
        paper_dir = trace_root / _sanitize_trace_slug(arxiv_id)
        trace_file: Optional[Path] = None

        if trace_name:
            safe_name = Path(str(trace_name)).name
            if safe_name != trace_name:
                raise HTTPException(status_code=400, detail="Invalid trace_name")
            candidate = paper_dir / safe_name
            if candidate.exists() and candidate.is_file():
                trace_file = candidate
        else:
            trace_file = _get_latest_retrieval_trace(arxiv_id, normalized_format)

        if trace_file is None:
            raise HTTPException(status_code=404, detail="No retrieval trace found for this paper")

        return FileResponse(
            path=str(trace_file),
            filename=f"{arxiv_id}_retrieval_trace.{normalized_format}",
            media_type="application/json" if normalized_format == "json" else "text/markdown",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading QA trace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper/{arxiv_id}/create-qa-index")
async def create_paper_qa_index(arxiv_id: str, loading_method: str = Query("docling")):
    """
    为论文创建问答索引
    流程：下载PDF -> 解析正文 -> 切分chunks -> 计算embeddings -> 保存到向量数据库
    """
    try:
        return paper_qa_service.build_qa_index(arxiv_id, loading_method=loading_method)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating QA index: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper/{arxiv_id}/qa")
async def qa_paper(arxiv_id: str, payload: QaRequest):
    """
    对论文进行问答
    流程：检查索引 -> 搜索相似chunks -> 生成回答
    """
    try:
        return paper_qa_service.answer_question(arxiv_id, payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in QA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/paper/{arxiv_id}/qa/stream")
async def qa_paper_stream(arxiv_id: str, payload: QaRequest):
    """
    流式论文问答，SSE 输出。
    """
    question = payload.question.strip()
    logger.info(f"QA stream request for paper: {arxiv_id}, question: {question}")

    _, search_results, qa_context, retrieval_debug = paper_qa_service.build_qa_context(arxiv_id, payload)

    source_payload = paper_qa_service.build_source_payload(search_results)

    def sse_event(event_name: str, data: dict) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def event_stream():
        try:
            yield sse_event(
                "meta",
                {
                    "status": "started",
                    "arxiv_id": arxiv_id,
                    "question": question,
                    "sources": source_payload,
                    "image_inputs": qa_context["image_inputs"],
                    "asset_metadata": qa_context["asset_metadata"],
                    "retrieval_debug": retrieval_debug,
                },
            )

            for chunk in generation_service.stream_qwen_responses(
                query=question,
                context=qa_context["text_context"],
                model_name="qwen3.6-plus",
                image_inputs=qa_context["image_inputs"],
                asset_metadata=[item for item in qa_context["asset_metadata"] if item.get("chunk_type") == "figure"],
            ):
                if chunk.get("type") == "delta":
                    yield sse_event("delta", {"delta": chunk.get("delta", "")})
                elif chunk.get("type") == "completed":
                    yield sse_event(
                        "done",
                        {
                            "status": "success",
                            "answer": chunk.get("answer", ""),
                            "sources": source_payload,
                            "image_inputs": qa_context["image_inputs"],
                            "asset_metadata": qa_context["asset_metadata"],
                            "retrieval_debug": retrieval_debug,
                            "usage": chunk.get("usage"),
                        },
                    )
                    return

            yield sse_event(
                "done",
                {
                    "status": "success",
                    "answer": "",
                    "sources": source_payload,
                    "image_inputs": qa_context["image_inputs"],
                    "asset_metadata": qa_context["asset_metadata"],
                    "retrieval_debug": retrieval_debug,
                    "usage": None,
                },
            )

        except Exception as e:
            logger.error(f"Error in QA stream: {str(e)}")
            yield sse_event(
                "error",
                {
                    "status": "error",
                    "detail": str(e),
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/chunks/files")
async def list_chunk_files():
    """
    获取所有已切片的文档文件列表
    """
    try:
        chunk_files = []
        if not CHUNK_DOCS_DIR.exists():
            logger.warning(f"Chunk docs directory does not exist: {CHUNK_DOCS_DIR}")
            return {"status": "success", "files": []}

        for file_path in CHUNK_DOCS_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == ".json":
                file_size = file_path.stat().st_size
                modified_time = file_path.stat().st_mtime
                chunk_files.append({
                    "filename": file_path.name,
                    "size": file_size,
                    "modified_time": modified_time
                })
        
        chunk_files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        return {
            "status": "success",
            "files": chunk_files
        }
        
    except Exception as e:
        logger.error(f"Error listing chunk files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks/file/{filename}")
async def get_chunk_file(filename: str):
    """
    获取指定切片文件的内容
    """
    try:
        file_path = CHUNK_DOCS_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return {
            "status": "success",
            "filename": filename,
            "data": data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading chunk file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="debug"
    )

