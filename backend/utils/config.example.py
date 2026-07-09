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


DEBUG_ROUTES_CONFIG = {
    "enable_debug_routes": False,
    "chunk_content_preview_chars": 4000,
}


BACKEND_LOGGING_CONFIG = {
    # 默认 INFO 作为 CMD 主时间线；需要细节时通过 debug_loggers 局部放大模块。
    "level": "INFO",
    "debug_loggers": [],
    # access log 与业务日志分离，避免前端轮询请求淹没 Agent/QA 事件。
    "access_log": False,
    # INFO 中只展示输入输出预览，完整内容按 request_trace 策略写入本地 trace。
    "io_preview_chars": 1200,
    "full_io": False,
    "request_trace": "auto",
    "request_trace_dir": str(REPO_ROOT / "temp" / "backend-request-traces"),
    "redact_secrets": True,
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


RECOMMENDATION_PROFILE_CONFIG = {
    "positive": {
        "min_liked_for_vector": 1,
        "min_liked_for_clustering": 4,
        "max_interest_clusters": 4,
        "clustering": {
            "min_cluster_size": 2,
            "min_samples": 1,
            "metric": "cosine",
            "cluster_selection_method": "eom",
            "allow_single_cluster": True,
        },
    },
    "negative": {
        "enabled": True,
        "store_disliked_examples": True,
        "min_disliked_for_instance_feedback": 1,
        "max_disliked_examples": 20,
        "enable_negative_clustering": True,
        "min_disliked_for_clustering": 4,
        "max_negative_clusters": 4,
        "clustering": {
            "min_cluster_size": 2,
            "min_samples": 1,
            "metric": "cosine",
            "cluster_selection_method": "eom",
            "allow_single_cluster": True,
        },
        "fallback_to_examples_when_cluster_failed": True,
    },
}


RECOMMENDATION_CLUSTERING_CONFIG = {
    **RECOMMENDATION_PROFILE_CONFIG["positive"]["clustering"],
}


RECOMMENDATION_RANKING_CONFIG = {
    "negative": {
        "enabled": True,
        "similarity_threshold": 0.82,
        "margin": 0.05,
        "penalty_weight": 0.15,
        "confidence_min_count": 6,
        "max_penalty": 0.22,
        "use_margin_penalty": True,
        "use_threshold_penalty": True,
        "hard_filter_threshold": 0.97,
        "enable_hard_filter": False,
        "debug_enabled": True,
    },
}


RECOMMENDATION_CONFIG = {
    "backfill_request_interval_seconds": 8.0,
    "profile": RECOMMENDATION_PROFILE_CONFIG,
    "ranking": RECOMMENDATION_RANKING_CONFIG,
    "min_liked_papers_for_clustering": RECOMMENDATION_PROFILE_CONFIG["positive"]["min_liked_for_clustering"],
    "max_interest_clusters": RECOMMENDATION_PROFILE_CONFIG["positive"]["max_interest_clusters"],
    # deprecated: 负向反馈不再参与主兴趣向量生成，仅保留旧配置读取兼容。
    "negative_weight_default": 0.3,
    "default_top_n": 10,
    "default_max_age_months": 6,
    "category_query_max_categories": 5,
    "candidate_concept_enrichment": {
        "enabled": True,
        "top_k": 20,
        "max_llm_calls": 5,
        "allow_lazy_generation": False,
        "allow_async_generation": False,
        "cache_version": "llm_paper_evidence_v1",
        "score_field": "base_score",
    },
    "score_weights": {
        "semantic": 0.65,
        "category": 0.08,
        "recency": 0.05,
        # deprecated: 负向扣分已迁移到 recommendation.ranking.negative.penalty_weight / max_penalty。
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


PROMPT_CONTEXT_CONFIG = {
    "enable_block_prompt_context": True,
    "prompt_max_input_tokens": 64000,
    "prompt_safety_margin_tokens": 2048,
    "rag_target_ratio": 0.78,
    "token_counter_provider": "auto",
    "tokenizer_name_or_path": "",
    "token_counter_fallback_chars_per_token": 3.0,
    "enable_rule_compaction": True,
    "enable_llm_compaction": False,
    # LLM 压缩默认关闭；开启后只处理超过阈值的长文本块，表格/数值证据仍由规则层保护。
    "llm_compaction_min_block_tokens": 1200,
    "llm_compaction_target_block_tokens": 700,
    "llm_compaction_max_blocks": 8,
    "llm_compaction_model_name": "",
    "llm_compaction_task_type": "prompt_context_compaction",
    "llm_compaction_enable_thinking": False,
    "recent_turns_recent_full_count": 2,
    "recent_turns_source_id_limit": 6,
    "rag_max_block_tokens": 4000,
    "rag_sibling_context_target_tokens": 350,
    "rag_section_context_target_tokens": 250,
    "rag_fallback_preview_tokens": 160,
    "table_candidate_cell_limit": 8,
    "figure_preview_tokens": 220,
}


ENHANCED_RETRIEVAL_CONFIG = {
    "query_view_limit": 6,
    "query_plan_limit": 5,
    "recall_candidate_limit": 30,
    "rrf_candidate_limit": 30,
    "rerank_candidate_limit": 30,
    "final_context_top_k": 15,
    "max_final_context_top_k": 30,
    "enable_context_expansion": True,
    # 灰度开关：开启后 dense embedding 按 retrieval index 生成；关闭时回退旧 chunk.content 向量。
    "enable_multi_index_embedding": True,
    # 灰度开关：开启后 keyword/BM25 使用 retrieval index 作为 sparse document，关闭时回退旧 chunk-level BM25。
    "enable_index_level_bm25": True,
    # 灰度开关：控制 LLM/规则生成的问题型 retrieval index，便于定位 question index 对召回的影响。
    "enable_generated_question_index": True,
    # 灰度开关：index-level 输入为空或旧 collection 未迁移时，是否允许回退 chunk-level 检索。
    "enable_chunk_level_retrieval_fallback": True,
    "context_budget_max_chars": 24000,
    "retrieval_candidate_max_blocks": 60,
    "retrieval_candidate_max_tokens_soft": 90000,
    "retrieval_candidate_token_chars_per_token": 3.0,
    "enable_table_structured_route": True,
    "table_structured_candidate_limit": 8,
    "table_structured_match_score_floor": 0.22,
    # 表格证据不确定时只交短表全量或长表局部上下文给 LLM，避免继续扩写硬规则。
    "table_structured_context_short_table_cell_limit": 48,
    "table_structured_context_max_rows": 12,
    "table_structured_context_focus_window": 2,
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
