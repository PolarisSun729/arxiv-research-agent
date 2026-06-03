import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent


class VectorDBProvider(str, Enum):
    MILVUS = "milvus"


class DatabaseType(str, Enum):
    SQLITE = "sqlite"


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, str(default)).lower()
    return raw not in {"0", "false", "no", "off", ""}


ALIYUN_API_KEY = _env_str("ALIYUN_API_KEY", "<REMOVED_API_KEY>")


CORE_CONFIG: Dict[str, Any] = {
    "arxiv_data_source": _env_str("ARXIV_DATA_SOURCE", "local"),
    "arxiv_proxy_url": _env_str(
        "ARXIV_PROXY_URL",
        "http://127.0.0.1:7897",
    ),
    # "service_load_mode": _env_str("BACKEND_SERVICE_LOAD_MODE", "lazy"),
    "service_load_mode": _env_str("BACKEND_SERVICE_LOAD_MODE", "preload"),
}

DOCLING_CONFIG: Dict[str, Any] = {
    "do_ocr_enabled": _env_bool("DOCLING_OCR_ENABLED", False),
    "annotated_pdf_export_enabled": _env_bool("DOCLING_ANNOTATED_PDF_EXPORT_ENABLED", True),
    "images_scale": float(_env_str("DOCLING_IMAGES_SCALE", "2.0")),
    "legend_padding": float(_env_str("DOCLING_LEGEND_PADDING", "8.0")),
    "legend_width": float(_env_str("DOCLING_LEGEND_WIDTH", "165.0")),
    "legend_row_height": float(_env_str("DOCLING_LEGEND_ROW_HEIGHT", "13.0")),
    "legend_title_height": float(_env_str("DOCLING_LEGEND_TITLE_HEIGHT", "12.0")),
    "legend_margin": float(_env_str("DOCLING_LEGEND_MARGIN", "12.0")),
    "table_caption_short_text_limit": _env_int("DOCLING_TABLE_CAPTION_SHORT_TEXT_LIMIT", 8),
    "normalized_upper_short_length": _env_int("DOCLING_NORMALIZED_UPPER_SHORT_LENGTH", 4),
    "normalized_heading_length": _env_int("DOCLING_NORMALIZED_HEADING_LENGTH", 80),
    "vertical_gap_multiplier": float(_env_str("DOCLING_VERTICAL_GAP_MULTIPLIER", "1.6")),
    "x_aligned_tolerance": float(_env_str("DOCLING_X_ALIGNED_TOLERANCE", "24.0")),
    "same_column_tolerance": float(_env_str("DOCLING_SAME_COLUMN_TOLERANCE", "80.0")),
}

MILVUS_CONFIG: Dict[str, Any] = {
    "uri": _env_str("MILVUS_URI", "http://localhost:19530"),
    "index_types": {
        "flat": "FLAT",
        "ivf_flat": "IVF_FLAT",
        "ivf_sq8": "IVF_SQ8",
        "hnsw": "HNSW",
    },
    "index_params": {
        "flat": {},
        "ivf_flat": {"nlist": 1024},
        "ivf_sq8": {"nlist": 1024},
        "hnsw": {"M": 16, "efConstruction": 500},
    },
}

SQLITE_CONFIG: Dict[str, Any] = {
    "database_path": _env_str("SQLITE_DATABASE_PATH", "06-database/recommendation.db"),
    "check_same_thread": _env_bool("SQLITE_CHECK_SAME_THREAD", False),
}

USER_CONFIG: Dict[str, Any] = {
    "default_user_id": _env_str("DEFAULT_USER_ID", "local_user"),
}

OAI_SQLITE_CONFIG: Dict[str, Any] = {
    # Keep the OAI database under backend/06-database so backend launches and tools share the same store.
    "database_path": _env_str("OAI_SQLITE_DATABASE_PATH", str(BASE_DIR.parent / "06-database" / "arxiv_oai.db")),
    "check_same_thread": _env_bool("OAI_SQLITE_CHECK_SAME_THREAD", False),
}

EMBEDDING_CONFIG: Dict[str, Any] = {
    "provider": _env_str("EMBEDDING_PROVIDER", "dashscope"),
    "model_name": _env_str("EMBEDDING_MODEL", "qwen3-vl-embedding"),
    "api_key": _env_str("EMBEDDING_API_KEY", ALIYUN_API_KEY),
    "dashscope_api_key": _env_str("EMBEDDING_DASHSCOPE_API_KEY", ALIYUN_API_KEY),
    "base_url": _env_str(
        "EMBEDDING_BASE_URL",
        _env_str(
            "DASHSCOPE_EMBEDDING_URL",
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding",
        ),
    ),
    "dimension": _env_int("EMBEDDING_DIMENSION", 2048),
    "batch_size": _env_int("EMBEDDING_BATCH_SIZE", 20),
    "local_model_path": _env_str("LOCAL_EMBEDDING_MODEL_PATH", str(REPO_ROOT / "00-models" / "Qwen3-VL-Embedding-2B")),
    "local_model_scripts_path": _env_str(
        "LOCAL_EMBEDDING_MODEL_SCRIPTS_PATH",
        str(REPO_ROOT / "00-models" / "Qwen3-VL-Embedding-2B" / "scripts"),
    ),
    "openai_model": _env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    "openai_api_key": _env_str("OPENAI_API_KEY", ""),
    "openai_base_url": _env_str("OPENAI_EMBEDDING_BASE_URL", ""),
    "aws_access_key_id": _env_str("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": _env_str("AWS_SECRET_ACCESS_KEY", ""),
}

CHUNKING_CONFIG: Dict[str, Any] = {
    "max_chunk_content_length": _env_int("CHUNKING_MAX_CHUNK_CONTENT_LENGTH", 8000),
    "chunk_overlap_length": _env_int("CHUNKING_CHUNK_OVERLAP_LENGTH", 200),
    "split_search_window": _env_int("CHUNKING_SPLIT_SEARCH_WINDOW", 400),
}

ARXIV_SEARCH_CONFIG: Dict[str, Any] = {
    "max_allowed_results": _env_int("ARXIV_SEARCH_MAX_ALLOWED_RESULTS", 100),
}

ARXIV_OAI_CONFIG: Dict[str, Any] = {
    "endpoint": _env_str("ARXIV_OAI_ENDPOINT", "https://oaipmh.arxiv.org/oai"),
    "target_categories": [
        item.strip()
        for item in _env_str("ARXIV_OAI_TARGET_CATEGORIES", "cs.CL,cs.LG,cs.IR,cs.AI").split(",")
        if item.strip()
    ],
    "embedding_batch_size": _env_int("ARXIV_OAI_EMBEDDING_BATCH_SIZE", 20),
    "vector_query_batch_size": _env_int("ARXIV_OAI_VECTOR_QUERY_BATCH_SIZE", 100),
    "dashscope_text_token_price_per_1k": float(_env_str("ARXIV_OAI_DASHSCOPE_TEXT_TOKEN_PRICE_PER_1K", "0.0007")),
}

VECTOR_STORE_CONFIG: Dict[str, Any] = {
    "content_max_length": _env_int("VECTOR_STORE_CONTENT_MAX_LENGTH", 12000),
    "rerank_text_max_length": _env_int("VECTOR_STORE_RERANK_TEXT_MAX_LENGTH", 6000),
    "asset_path_max_length": _env_int("VECTOR_STORE_ASSET_PATH_MAX_LENGTH", 2048),
    "asset_summary_max_length": _env_int("VECTOR_STORE_ASSET_SUMMARY_MAX_LENGTH", 4096),
    "asset_preview_max_length": _env_int("VECTOR_STORE_ASSET_PREVIEW_MAX_LENGTH", 6000),
}

RECOMMENDATION_CLUSTERING_CONFIG: Dict[str, Any] = {
    "hdbscan_min_cluster_size": _env_int("RECOMMENDATION_HDBSCAN_MIN_CLUSTER_SIZE", 2),
    "hdbscan_min_samples": _env_int("RECOMMENDATION_HDBSCAN_MIN_SAMPLES", 1),
    "hdbscan_metric": _env_str("RECOMMENDATION_HDBSCAN_METRIC", "cosine"),
    "hdbscan_cluster_selection_method": _env_str("RECOMMENDATION_HDBSCAN_CLUSTER_SELECTION_METHOD", "eom"),
    "hdbscan_allow_single_cluster": _env_bool("RECOMMENDATION_HDBSCAN_ALLOW_SINGLE_CLUSTER", True),
}

RECOMMENDATION_CONFIG: Dict[str, Any] = {
    "backfill_request_interval_seconds": float(_env_str("RECOMMENDATION_BACKFILL_REQUEST_INTERVAL_SECONDS", "8.0")),
    "min_liked_papers_for_clustering": _env_int("RECOMMENDATION_MIN_LIKED_PAPERS_FOR_CLUSTERING", 4),
    "max_interest_clusters": _env_int("RECOMMENDATION_MAX_INTEREST_CLUSTERS", 4),
    "negative_weight_default": float(_env_str("RECOMMENDATION_NEGATIVE_WEIGHT_DEFAULT", "0.3")),
    "default_top_n": _env_int("RECOMMENDATION_DEFAULT_TOP_N", 10),
    "default_max_age_months": _env_int("RECOMMENDATION_DEFAULT_MAX_AGE_MONTHS", 6),
    "category_query_max_categories": _env_int("RECOMMENDATION_CATEGORY_QUERY_MAX_CATEGORIES", 5),
    "score_weights": {
        "semantic": float(_env_str("RECOMMENDATION_SCORE_WEIGHT_SEMANTIC", "0.65")),
        "category": float(_env_str("RECOMMENDATION_SCORE_WEIGHT_CATEGORY", "0.08")),
        "recency": float(_env_str("RECOMMENDATION_SCORE_WEIGHT_RECENCY", "0.05")),
        "disliked_penalty": float(_env_str("RECOMMENDATION_SCORE_WEIGHT_DISLIKED_PENALTY", "0.15")),
    },
    "semantic_similarity_penalty_threshold": float(
        _env_str("RECOMMENDATION_SEMANTIC_SIMILARITY_PENALTY_THRESHOLD", "0.75")
    ),
    "cluster_diversity_component_weight": float(_env_str("RECOMMENDATION_CLUSTER_DIVERSITY_COMPONENT_WEIGHT", "0.3")),
    "semantic_diversity_component_weight": float(_env_str("RECOMMENDATION_SEMANTIC_DIVERSITY_COMPONENT_WEIGHT", "0.7")),
    "selection_relevance_weight": float(_env_str("RECOMMENDATION_SELECTION_RELEVANCE_WEIGHT", "0.75")),
    "selection_diversity_weight": float(_env_str("RECOMMENDATION_SELECTION_DIVERSITY_WEIGHT", "0.25")),
}

MEMORY_RUNTIME_CONFIG: Dict[str, Any] = {
    "enable_short_term_memory": _env_bool("ENABLE_SHORT_TERM_MEMORY", True),
    "enable_paper_chat_session": _env_bool("ENABLE_PAPER_CHAT_SESSION", True),
    "enable_memory_aware_retrieval": _env_bool("ENABLE_MEMORY_AWARE_RETRIEVAL", True),
    "enable_user_research_profile": _env_bool("ENABLE_USER_RESEARCH_PROFILE", False),
    "short_term_memory_max_turns": _env_int("SHORT_TERM_MEMORY_MAX_TURNS", 5),
    "short_term_memory_max_chars": _env_int("SHORT_TERM_MEMORY_MAX_CHARS", 280),
    "memory_source_boost_weight": float(_env_str("MEMORY_SOURCE_BOOST_WEIGHT", "0.12")),
    "memory_context_debug": _env_bool("MEMORY_CONTEXT_DEBUG", True),
}

ENHANCED_RETRIEVAL_CONFIG: Dict[str, Any] = {
    "query_view_limit": _env_int("ENHANCED_RETRIEVAL_QUERY_VIEW_LIMIT", 6),
    "query_plan_limit": _env_int("ENHANCED_RETRIEVAL_QUERY_PLAN_LIMIT", 5),
    "recall_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RECALL_CANDIDATE_LIMIT", 30),
    "rrf_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RRF_CANDIDATE_LIMIT", 30),
    "rerank_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RERANK_CANDIDATE_LIMIT", 30),
    "final_context_top_k": _env_int("ENHANCED_RETRIEVAL_FINAL_CONTEXT_TOP_K", 15),
    "max_final_context_top_k": _env_int("ENHANCED_RETRIEVAL_MAX_FINAL_CONTEXT_TOP_K", 30),
    "sample_limit": _env_int("ENHANCED_RETRIEVAL_SAMPLE_LIMIT", 24),
    "merge_candidate_terms_limit": _env_int("ENHANCED_RETRIEVAL_MERGE_CANDIDATE_TERMS_LIMIT", 24),
    "extract_paper_terms_limit": _env_int("ENHANCED_RETRIEVAL_EXTRACT_PAPER_TERMS_LIMIT", 10),
    "compact_terms_limit": _env_int("ENHANCED_RETRIEVAL_COMPACT_TERMS_LIMIT", 5),
    "extract_query_keywords_limit": _env_int("ENHANCED_RETRIEVAL_EXTRACT_QUERY_KEYWORDS_LIMIT", 8),
    "build_query_keywords_limit": _env_int("ENHANCED_RETRIEVAL_BUILD_QUERY_KEYWORDS_LIMIT", 12),
    "language_weight_zh_mixed": float(_env_str("ENHANCED_RETRIEVAL_LANGUAGE_WEIGHT_ZH_MIXED", "0.1")),
    "specificity_content_weight": float(_env_str("ENHANCED_RETRIEVAL_SPECIFICITY_CONTENT_WEIGHT", "0.45")),
    "specificity_intent_weight": float(_env_str("ENHANCED_RETRIEVAL_SPECIFICITY_INTENT_WEIGHT", "0.3")),
    "specificity_length_weight": float(_env_str("ENHANCED_RETRIEVAL_SPECIFICITY_LENGTH_WEIGHT", "0.15")),
    "specificity_floor": float(_env_str("ENHANCED_RETRIEVAL_SPECIFICITY_FLOOR", "0.1")),
    "bm25_k1": float(_env_str("ENHANCED_RETRIEVAL_BM25_K1", "1.5")),
    "bm25_b": float(_env_str("ENHANCED_RETRIEVAL_BM25_B", "0.75")),
    "section_bonus_weight": float(_env_str("ENHANCED_RETRIEVAL_SECTION_BONUS_WEIGHT", "0.045")),
    "figure_table_bonus_weight": float(_env_str("ENHANCED_RETRIEVAL_FIGURE_TABLE_BONUS_WEIGHT", "0.05")),
    "noisy_section_penalty_weight": float(_env_str("ENHANCED_RETRIEVAL_NOISY_SECTION_PENALTY_WEIGHT", "0.02")),
    "preferred_section_bonus_weight": float(_env_str("ENHANCED_RETRIEVAL_PREFERRED_SECTION_BONUS_WEIGHT", "0.03")),
    "section_path_bonus_weight": float(_env_str("ENHANCED_RETRIEVAL_SECTION_PATH_BONUS_WEIGHT", "0.02")),
    "semantic_similarity_penalty_threshold": float(
        _env_str("ENHANCED_RETRIEVAL_SEMANTIC_SIMILARITY_PENALTY_THRESHOLD", "0.75")
    ),
    "route_similarity_weight": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_SIMILARITY_WEIGHT", "0.35")),
    "route_base_similarity_weight": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_BASE_SIMILARITY_WEIGHT", "0.65")),
    "memory_source_boost_weight": float(
        _env_str("MEMORY_SOURCE_BOOST_WEIGHT", str(MEMORY_RUNTIME_CONFIG["memory_source_boost_weight"]))
    ),
    "bm25_token_boost": float(_env_str("ENHANCED_RETRIEVAL_BM25_TOKEN_BOOST", "0.2")),
    "cluster_repeat_divisor": float(_env_str("ENHANCED_RETRIEVAL_CLUSTER_REPEAT_DIVISOR", "4.0")),
    "category_repeat_divisor": float(_env_str("ENHANCED_RETRIEVAL_CATEGORY_REPEAT_DIVISOR", "4.0")),
    "short_text_preview_limit": _env_int("ENHANCED_RETRIEVAL_SHORT_TEXT_PREVIEW_LIMIT", 100),
    "preview_text_limit": _env_int("ENHANCED_RETRIEVAL_PREVIEW_TEXT_LIMIT", 100),
    "rerank_document_preview_limit": _env_int("ENHANCED_RETRIEVAL_RERANK_DOCUMENT_PREVIEW_LIMIT", 220),
    "candidate_debug_limit": _env_int("ENHANCED_RETRIEVAL_CANDIDATE_DEBUG_LIMIT", 10),
    "source_sample_limit": _env_int("ENHANCED_RETRIEVAL_SOURCE_SAMPLE_LIMIT", 6),
    "source_sample_primary_limit": _env_int("ENHANCED_RETRIEVAL_SOURCE_SAMPLE_PRIMARY_LIMIT", 260),
    "source_sample_secondary_limit": _env_int("ENHANCED_RETRIEVAL_SOURCE_SAMPLE_SECONDARY_LIMIT", 180),
    "paper_terms_preview_limit": _env_int("ENHANCED_RETRIEVAL_PAPER_TERMS_PREVIEW_LIMIT", 8),
    "keyword_parts_limit": _env_int("ENHANCED_RETRIEVAL_KEYWORD_PARTS_LIMIT", 8),
    "sanitize_trace_slug_max_length": _env_int("ENHANCED_RETRIEVAL_SANITIZE_TRACE_SLUG_MAX_LENGTH", 40),
    "query_weight_base_summary_other": float(_env_str("ENHANCED_RETRIEVAL_QUERY_WEIGHT_BASE_SUMMARY_OTHER", "1.05")),
    "query_weight_base_default": float(_env_str("ENHANCED_RETRIEVAL_QUERY_WEIGHT_BASE_DEFAULT", "0.95")),
    "query_weight_base_ambiguous_keyword": float(_env_str("ENHANCED_RETRIEVAL_QUERY_WEIGHT_BASE_AMBIGUOUS_KEYWORD", "0.72")),
    "query_weight_base_ambiguous_other": float(_env_str("ENHANCED_RETRIEVAL_QUERY_WEIGHT_BASE_AMBIGUOUS_OTHER", "0.45")),
    "query_weight_base_keyword": float(_env_str("ENHANCED_RETRIEVAL_QUERY_WEIGHT_BASE_KEYWORD", "0.52")),
    "query_weight_base_fallback": float(_env_str("ENHANCED_RETRIEVAL_QUERY_WEIGHT_BASE_FALLBACK", "0.5")),
    "route_focus_bonus": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_FOCUS_BONUS", "0.08")),
    "route_summary_bonus": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_SUMMARY_BONUS", "0.05")),
    "route_keyword_bonus": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_KEYWORD_BONUS", "0.08")),
    "route_default_floor": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_DEFAULT_FLOOR", "0.2")),
    "route_confidence_multiplier": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_CONFIDENCE_MULTIPLIER", "0.65")),
    "route_confidence_similarity_weight": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_CONFIDENCE_SIMILARITY_WEIGHT", "0.35")),
}

INTENT_ROUTING_CONFIG: Dict[str, Any] = {
    "default_route_weights": {
        "vector_original": 1.0,
        "vector_rewrite": 0.9,
        "vector_hyde": 0.85,
        "keyword": 0.75,
    },
    "intent_route_weights": {
        "contribution": {
            "vector_original": 1.15,
            "vector_rewrite": 1.1,
            "vector_hyde": 0.9,
            "keyword": 0.6,
        },
        "paper_overview": {
            "vector_original": 1.15,
            "vector_rewrite": 1.1,
            "vector_hyde": 0.9,
            "keyword": 0.6,
        },
        "method_flow": {
            "vector_original": 0.95,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.85,
            "keyword": 1.0,
        },
        "experiment_setup": {
            "vector_original": 0.9,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.8,
            "keyword": 1.1,
        },
        "result_analysis": {
            "vector_original": 0.88,
            "vector_rewrite": 1.18,
            "vector_hyde": 0.82,
            "keyword": 1.08,
        },
        "comparison": {
            "vector_original": 0.85,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.8,
            "keyword": 1.0,
        },
        "dataset": {
            "vector_original": 0.9,
            "vector_rewrite": 1.1,
            "vector_hyde": 0.8,
            "keyword": 1.05,
        },
        "limitation": {
            "vector_original": 0.85,
            "vector_rewrite": 1.05,
            "vector_hyde": 0.8,
            "keyword": 0.95,
        },
        "definition": {
            "vector_original": 0.9,
            "vector_rewrite": 1.05,
            "vector_hyde": 0.85,
            "keyword": 1.0,
        },
        "implementation_detail": {
            "vector_original": 0.95,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.85,
            "keyword": 1.0,
        },
        "figure_table": {
            "vector_original": 0.75,
            "vector_rewrite": 0.95,
            "vector_hyde": 0.7,
            "keyword": 1.2,
        },
        "other": {
            "vector_original": 1.0,
            "vector_rewrite": 0.9,
            "vector_hyde": 0.85,
            "keyword": 0.75,
        },
    },
}

RERANK_CONFIG: Dict[str, Any] = {
    "provider": _env_str("RERANK_PROVIDER", "dashscope"),
    "model_name": _env_str("RERANK_MODEL_NAME", "qwen3-vl-rerank"),
    "local_model_name_or_path": _env_str(
        "RERANK_LOCAL_MODEL_NAME_OR_PATH",
        str(REPO_ROOT / "00-models" / "Qwen3-VL-Reranker-2B"),
    ),
    "api_key": _env_str("RERANK_API_KEY", ALIYUN_API_KEY),
    "dashscope_api_key": _env_str("RERANK_DASHSCOPE_API_KEY", ALIYUN_API_KEY),
    "base_url": _env_str("RERANK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"),
    "fallback_local": _env_bool("RERANK_FALLBACK_LOCAL", False),
    "prompt": _env_str("RERANK_PROMPT", "Retrieve text relevant to the user's query."),
    "batch_size": _env_int("RERANK_BATCH_SIZE", 8),
    "candidate_limit": _env_int("RERANK_CANDIDATE_LIMIT", 24),
    "max_doc_chars": _env_int("RERANK_MAX_DOC_CHARS", 4096),
    "enable_query_rewrite": _env_bool("ENABLE_QUERY_REWRITE", True),
    "enable_hyde": _env_bool("ENABLE_HYDE", False),
    "enable_keyword_search": _env_bool("ENABLE_KEYWORD_SEARCH", True),
    "enable_llm_rerank": _env_bool("ENABLE_LLM_RERANK", True),
    "debug": _env_bool("RETRIEVAL_DEBUG", False),
    "rrf_k": _env_int("RRF_K", 60),
    "default_top_k": _env_int("DEFAULT_TOP_K", 15),
    "candidate_multiplier": _env_int("CANDIDATE_MULTIPLIER", 3),
    "route_weights": {
        "vector_original": 1.0,
        "vector_rewrite": 0.9,
        "vector_hyde": 0.85,
        "keyword": 0.75,
        "memory_context": float(_env_str("RETRIEVAL_ROUTE_WEIGHT_MEMORY_CONTEXT", "0.35")),
    },
    "trace_export_enabled": _env_bool("RETRIEVAL_TRACE_EXPORT_ENABLED", True),
    "trace_export_dir": _env_str(
        "RETRIEVAL_TRACE_EXPORT_DIR",
        str(REPO_ROOT / "temp" / "retrieval-traces"),
    ),
}

RETRIEVAL_CONFIG: Dict[str, Any] = {
    "default_top_k": RERANK_CONFIG["default_top_k"],
    "enable_query_rewrite": RERANK_CONFIG["enable_query_rewrite"],
    "enable_hyde": RERANK_CONFIG["enable_hyde"],
    "enable_keyword_search": RERANK_CONFIG["enable_keyword_search"],
    "enable_llm_rerank": RERANK_CONFIG["enable_llm_rerank"],
    "debug": RERANK_CONFIG["debug"],
    "rrf_k": RERANK_CONFIG["rrf_k"],
    "candidate_multiplier": RERANK_CONFIG["candidate_multiplier"],
    "route_weights": RERANK_CONFIG["route_weights"],
    "llm_rerank_provider": RERANK_CONFIG["provider"],
    "llm_rerank_api_key": RERANK_CONFIG["dashscope_api_key"] or RERANK_CONFIG["api_key"],
    "llm_rerank_base_url": RERANK_CONFIG["base_url"],
    "llm_rerank_fallback_local": RERANK_CONFIG["fallback_local"],
    "llm_rerank_model_name_or_path": RERANK_CONFIG["model_name"],
    "llm_rerank_local_model_name_or_path": RERANK_CONFIG["local_model_name_or_path"],
    "llm_rerank_batch_size": RERANK_CONFIG["batch_size"],
    "llm_rerank_candidate_limit": RERANK_CONFIG["candidate_limit"],
    "llm_rerank_max_doc_chars": RERANK_CONFIG["max_doc_chars"],
    "llm_rerank_prompt": RERANK_CONFIG["prompt"],
    "trace_export_enabled": RERANK_CONFIG["trace_export_enabled"],
    "trace_export_dir": RERANK_CONFIG["trace_export_dir"],
}

GENERATION_CONFIG: Dict[str, Any] = {
    "qwen_api_key": _env_str("QWEN_API_KEY", ALIYUN_API_KEY),
    "qwen_base_url": _env_str("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    # 只保留生成链路的小/大模型配置，不再混用旧别名。
    "small_qwen_model_name": _env_str("QWEN_SMALL_MODEL_NAME", "qwen3.6-flash"),
    "large_qwen_model_name": _env_str("QWEN_LARGE_MODEL_NAME", "qwen3.6-plus"),
    "openai_api_key": _env_str("OPENAI_API_KEY", ALIYUN_API_KEY),
    "deepseek_api_key": _env_str("DEEPSEEK_API_KEY", ALIYUN_API_KEY),
    # rerank 前的 chunk 压缩/摘要使用独立的生成模型配置。
    "qwen_rerank_compress_model_name": _env_str("QWEN_RERANK_COMPRESS_MODEL_NAME", "qwen3.6-flash"),
    "qwen_rerank_compress_enable_thinking": _env_bool("QWEN_RERANK_COMPRESS_ENABLE_THINKING", False),
    "huggingface_generate_max_length": _env_int("HF_GENERATE_MAX_LENGTH", 512),
    "huggingface_generate_temperature": float(_env_str("HF_GENERATE_TEMPERATURE", "0.7")),
    "huggingface_generate_do_sample": _env_bool("HF_GENERATE_DO_SAMPLE", True),
    "openai_chat_temperature": float(_env_str("OPENAI_CHAT_TEMPERATURE", "0.7")),
    "openai_chat_max_tokens": _env_int("OPENAI_CHAT_MAX_TOKENS", 512),
    "rewrite_query_max_queries_default": _env_int("REWRITE_QUERY_MAX_QUERIES_DEFAULT", 3),
    "plan_query_max_queries_default": _env_int("PLAN_QUERY_MAX_QUERIES_DEFAULT", 5),
    # 任务到模型角色的路由表只保留在配置层，业务代码只负责读取，不再硬编码模型名。
    "task_model_roles": {
        "intent_recognition": "small",  # 解析用户原始输入以识别高层意图
        "intent_routing": "small",  # 决定将意图路由到哪个子 agent 或处理流水线
        "search_spec_parse": "small",  # 将自然语言搜索请求转换为结构化的搜索规范
        "preference_action_parse": "small",  # 解析用户的喜欢/不喜欢/收藏等偏好操作
        "pending_action_confirmation": "small",  # 判断用户是否在确认待执行的论文解析任务
        "query_planning": "small",  # 规划多步或多查询的检索策略
        "query_rewrite": "small",  # 重写或规范化查询以提升检索效果
        "rerank_query": "small",  # 为重排阶段生成或调整查询
        "hyde_generation": "small",  # HYDE 风格的伪文档生成（用于查询扩展）
        "paper_qa_final_answer": "large",  # 为论文问答生成最终高质量答案
        "paper_summary": "large",  # 生成论文的综合性摘要
        "paper_detail": "large",  # 生成方法或技术细节的详尽说明
        "paper_qa": "large",  # 处理关于具体论文的问答
        "recommendation_generation": "large",  # 生成个性化推荐文本或推荐理由
        "general_generation": "large",  # 用于复杂自然语言输出的一般大模型生成
        "default": "large",  # 当无特定任务映射时的默认模型角色
    },
}

def get_embedding_runtime_config() -> Dict[str, Any]:
    return dict(EMBEDDING_CONFIG)


def get_chunking_runtime_config() -> Dict[str, Any]:
    return dict(CHUNKING_CONFIG)


def get_arxiv_search_runtime_config() -> Dict[str, Any]:
    return dict(ARXIV_SEARCH_CONFIG)


def get_arxiv_oai_runtime_config() -> Dict[str, Any]:
    return dict(ARXIV_OAI_CONFIG)


def get_vector_store_runtime_config() -> Dict[str, Any]:
    return dict(VECTOR_STORE_CONFIG)


def get_recommendation_clustering_runtime_config() -> Dict[str, Any]:
    return dict(RECOMMENDATION_CLUSTERING_CONFIG)


def get_recommendation_runtime_config() -> Dict[str, Any]:
    return dict(RECOMMENDATION_CONFIG)


def get_memory_runtime_config() -> Dict[str, Any]:
    return dict(MEMORY_RUNTIME_CONFIG)


def get_default_user_id() -> str:
    return str(USER_CONFIG["default_user_id"] or "local_user").strip() or "local_user"


def get_user_runtime_config() -> Dict[str, Any]:
    return dict(USER_CONFIG)


def get_enhanced_retrieval_runtime_config() -> Dict[str, Any]:
    return dict(ENHANCED_RETRIEVAL_CONFIG)


def get_intent_routing_runtime_config() -> Dict[str, Any]:
    return dict(INTENT_ROUTING_CONFIG)


def get_rerank_runtime_config() -> Dict[str, Any]:
    return dict(RERANK_CONFIG)


def get_generation_runtime_config() -> Dict[str, Any]:
    return dict(GENERATION_CONFIG)
