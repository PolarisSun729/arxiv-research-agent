from __future__ import annotations

"""Example project configuration.

This file is a readable Python example showing the configuration structure used
by the project. It is not imported by the application by default.

The real runtime configuration currently lives in `backend/utils/config.py` and
mainly reads values from environment variables.

Use this file as:
1. a reference for what can be configured,
2. a starting point if you want to maintain a local Python config variant,
3. documentation for collaborators.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent


CORE_CONFIG = {
    "arxiv_data_source": "local",
    "arxiv_proxy_url": "http://127.0.0.1:7897",
    "service_load_mode": "lazy",  # lazy | preload
}


DOCLING_CONFIG = {
    "do_ocr_enabled": False,
    "annotated_pdf_export_enabled": True,
    "images_scale": 2.0,
    "legend_padding": 8.0,
    "legend_width": 165.0,
    "legend_row_height": 13.0,
    "legend_title_height": 12.0,
    "legend_margin": 12.0,
    "table_caption_short_text_limit": 8,
    "normalized_upper_short_length": 4,
    "normalized_heading_length": 80,
    "vertical_gap_multiplier": 1.6,
    "x_aligned_tolerance": 24.0,
    "same_column_tolerance": 80.0,
}


MILVUS_CONFIG = {
    "uri": "http://localhost:19530",
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


SQLITE_CONFIG = {
    "database_path": "06-database/recommendation.db",
    "check_same_thread": False,
}


USER_CONFIG = {
    "default_user_id": "local_user",
}


OAI_SQLITE_CONFIG = {
    "database_path": str(BASE_DIR.parent / "06-database" / "arxiv_oai.db"),
    "check_same_thread": False,
}


EMBEDDING_CONFIG = {
    "provider": "dashscope",
    "model_name": "qwen3-vl-embedding",
    "api_key": "",
    "dashscope_api_key": "",
    "base_url": (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    ),
    "dimension": 2048,
    "batch_size": 20,
    "local_model_path": str(REPO_ROOT / "00-models" / "Qwen3-VL-Embedding-2B"),
    "local_model_scripts_path": str(REPO_ROOT / "00-models" / "Qwen3-VL-Embedding-2B" / "scripts"),
    "openai_model": "text-embedding-3-small",
    "openai_api_key": "",
    "openai_base_url": "",
    "aws_access_key_id": "",
    "aws_secret_access_key": "",
}


CHUNKING_CONFIG = {
    "max_chunk_content_length": 8000,
    "chunk_overlap_length": 200,
    "split_search_window": 400,
}


ARXIV_SEARCH_CONFIG = {
    "max_allowed_results": 100,
}


ARXIV_OAI_CONFIG = {
    "endpoint": "https://oaipmh.arxiv.org/oai",
    "target_categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"],
    "embedding_batch_size": 20,
    "vector_query_batch_size": 100,
    "dashscope_text_token_price_per_1k": 0.0007,
}


VECTOR_STORE_CONFIG = {
    "content_max_length": 12000,
    "rerank_text_max_length": 6000,
    "asset_path_max_length": 2048,
    "asset_summary_max_length": 4096,
    "asset_preview_max_length": 6000,
}


RECOMMENDATION_CLUSTERING_CONFIG = {
    "hdbscan_min_cluster_size": 2,
    "hdbscan_min_samples": 1,
    "hdbscan_metric": "cosine",
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_allow_single_cluster": True,
}


RECOMMENDATION_CONFIG = {
    "backfill_request_interval_seconds": 8.0,
    "min_liked_papers_for_clustering": 4,
    "max_interest_clusters": 4,
    "negative_weight_default": 0.3,
    "default_top_n": 10,
    "default_max_age_months": 6,
    "category_query_max_categories": 5,
    "score_weights": {
        "semantic": 0.65,
        "category": 0.08,
        "recency": 0.05,
        "disliked_penalty": 0.15,
    },
    "semantic_similarity_penalty_threshold": 0.75,
    "cluster_diversity_component_weight": 0.3,
    "semantic_diversity_component_weight": 0.7,
    "selection_relevance_weight": 0.75,
    "selection_diversity_weight": 0.25,
}


MEMORY_RUNTIME_CONFIG = {
    "enable_short_term_memory": True,
    "enable_paper_chat_session": True,
    "enable_memory_aware_retrieval": True,
    "enable_user_research_profile": False,
    "short_term_memory_max_turns": 5,
    "short_term_memory_max_chars": 280,
    "memory_source_boost_weight": 0.12,
    "memory_context_debug": True,
}


ENHANCED_RETRIEVAL_CONFIG = {
    "query_view_limit": 6,
    "query_plan_limit": 5,
    "recall_candidate_limit": 30,
    "rrf_candidate_limit": 30,
    "rerank_candidate_limit": 30,
    "final_context_top_k": 15,
    "max_final_context_top_k": 30,
    "sample_limit": 24,
    "merge_candidate_terms_limit": 24,
    "extract_paper_terms_limit": 10,
    "compact_terms_limit": 5,
    "extract_query_keywords_limit": 8,
    "build_query_keywords_limit": 12,
    "language_weight_zh_mixed": 0.1,
    "specificity_content_weight": 0.45,
    "specificity_intent_weight": 0.3,
    "specificity_length_weight": 0.15,
    "specificity_floor": 0.1,
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
    "section_bonus_weight": 0.045,
    "figure_table_bonus_weight": 0.05,
    "noisy_section_penalty_weight": 0.02,
    "preferred_section_bonus_weight": 0.03,
    "section_path_bonus_weight": 0.02,
    "semantic_similarity_penalty_threshold": 0.75,
    "route_similarity_weight": 0.35,
    "route_base_similarity_weight": 0.65,
    "memory_source_boost_weight": 0.12,
    "bm25_token_boost": 0.2,
    "cluster_repeat_divisor": 4.0,
}


# Optional provider placeholders for teams that prefer a Python config view.
API_KEYS = {
    "aliyun_api_key": "",
    "openai_api_key": "",
    "deepseek_api_key": "",
    "qwen_api_key": "",
}


# If you later want to move from env-driven config to Python-driven config,
# this file can serve as the initial template.
