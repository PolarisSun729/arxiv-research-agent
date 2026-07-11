import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DEFAULT_RECOMMENDATION_CONCEPT_CACHE_VERSION = "llm_paper_evidence_v1"


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


# API 密钥只允许通过环境变量注入，避免任何代码兜底值再次进入 Git 历史。
ALIYUN_API_KEY = _env_str("ALIYUN_API_KEY", "")


CORE_CONFIG: Dict[str, Any] = {
    "arxiv_data_source": _env_str("ARXIV_DATA_SOURCE", "local"),
    "arxiv_proxy_url": _env_str(
        "ARXIV_PROXY_URL",
        "http://127.0.0.1:7897",
    ),
    # "service_load_mode": _env_str("BACKEND_SERVICE_LOAD_MODE", "lazy"),
    "service_load_mode": _env_str("BACKEND_SERVICE_LOAD_MODE", "preload"),
}

DEBUG_ROUTES_CONFIG: Dict[str, Any] = {
    # 调试路由会读取本地诊断产物，默认关闭，避免生产或演示环境暴露内部文件结构。
    "enable_debug_routes": _env_bool("ENABLE_DEBUG_ROUTES", False),
    # chunk 调试接口只返回文本预览，防止一次请求把完整 PDF 解析文本或超大字段暴露出去。
    "chunk_content_preview_chars": _env_int("DEBUG_CHUNK_CONTENT_PREVIEW_CHARS", 4000),
}

BACKEND_LOGGING_CONFIG: Dict[str, Any] = {
    # CMD 默认展示请求主时间线；需要深挖时再按模块打开 DEBUG，避免全后端日志一起刷屏。
    "level": (_env_str("BACKEND_LOG_LEVEL", "INFO") or "INFO").upper(),
    "debug_loggers": [
        item.strip()
        for item in _env_str("BACKEND_DEBUG_LOGGERS", "").split(",")
        if item.strip()
    ],
    # access log 与业务事件分开控制，避免前端轮询/静态请求淹没真正的 Agent/QA 进度。
    "access_log": _env_bool("BACKEND_ACCESS_LOG", False),
    # INFO 只放可读预览；完整输入输出按请求 trace 策略落盘。
    "io_preview_chars": _env_int("BACKEND_LOG_IO_PREVIEW_CHARS", 1200),
    "full_io": _env_bool("BACKEND_LOG_FULL_IO", False),
    "request_trace": (_env_str("BACKEND_REQUEST_TRACE", "auto") or "auto").lower(),
    "request_trace_dir": _env_str(
        "BACKEND_REQUEST_TRACE_DIR",
        str(REPO_ROOT / "temp" / "backend-request-traces"),
    ),
    # 日志和 trace 共用脱敏策略，避免 CMD 安全但本地 trace 泄漏密钥类字段。
    "redact_secrets": _env_bool("BACKEND_LOG_REDACT_SECRETS", True),
}

DOCLING_CONFIG: Dict[str, Any] = {
    "do_ocr_enabled": _env_bool("DOCLING_OCR_ENABLED", False),
    "annotated_pdf_export_enabled": _env_bool("DOCLING_ANNOTATED_PDF_EXPORT_ENABLED", True),
    "images_scale": float(_env_str("DOCLING_IMAGES_SCALE", "13.9")),
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
    # 公共仓库使用非个人化的本地用户标识，部署时可通过环境变量覆盖。
    "default_user_id": _env_str("DEFAULT_USER_ID", "local_user"),
}

QA_INDEX_JOB_CONFIG: Dict[str, Any] = {
    # 后台线程在进程重启后会丢失，心跳超时用于把旧 pending/running 任务恢复成可重试状态。
    "timeout_seconds": _env_int("QA_INDEX_JOB_TIMEOUT_SECONDS", 30 * 60),
}

PAPER_QA_BUILD_CACHE_CONFIG: Dict[str, Any] = {
    # QA 建库里的 LLM 增强和 embedding 都是高成本外部调用；持久缓存用于失败重试和重复重建时复用确定性结果。
    "enabled": _env_bool("PAPER_QA_BUILD_CACHE_ENABLED", True),
    "root_dir": _env_str("PAPER_QA_BUILD_CACHE_DIR", str(BASE_DIR.parent / "06-database")),
    "llm_cache_name": _env_str("PAPER_QA_LLM_CACHE_NAME", "paper_qa_llm_cache"),
    "embedding_cache_name": _env_str("PAPER_QA_EMBEDDING_CACHE_NAME", "paper_qa_embedding_cache"),
    "llm_size_limit": _env_int("PAPER_QA_LLM_CACHE_SIZE_LIMIT", 512 * 1024 * 1024),
    "embedding_size_limit": _env_int("PAPER_QA_EMBEDDING_CACHE_SIZE_LIMIT", 2 * 1024 * 1024 * 1024),
    # per-chunk LLM 调用互不依赖，默认小并发提速，同时避免把 provider 限流压力放大到不可控。
    "llm_max_workers": _env_int("PAPER_QA_BUILD_LLM_MAX_WORKERS", 4),
}

PROFILE_EVIDENCE_CONFIG: Dict[str, Any] = {
    # evidence card 会调用外部 LLM；默认只开小并发，避免把限流和 SQLite 写入压力放大。
    "max_workers": _env_int("PROFILE_EVIDENCE_MAX_WORKERS", 2),
    # 命中限流/服务繁忙时放慢补充新任务的速度，让当前构建可以降速而不是持续压请求。
    "rate_limit_backoff_seconds": _env_int("PROFILE_EVIDENCE_RATE_LIMIT_BACKOFF_SECONDS", 3),
}

AGENT_PLANNER_CONFIG: Dict[str, Any] = {
    # planner_runtime_mode 决定主路径策略：
    # - llm_preferred: 先走 LLM draft，再做本地校验，失败时回退到 rule/template。
    # - rule_only: 只走规则 planner。
    # - llm_only_strict: 只测试 LLM draft，失败直接报错，不做业务兜底。
    # - demo_rule: 演示/稳定场景优先固定规则路径，避免引入 LLM 波动。
    "planner_runtime_mode": _env_str("AGENT_PLANNER_MODE", _env_str("PLANNER_RUNTIME_MODE", "llm_preferred")).lower() or "llm_preferred",
    # 这些显式开关仍保留，用于兼容历史环境变量；最终是否生效由 planner_runtime_mode 统一裁决。
    "enable_rule_based_planner": _env_bool("ENABLE_RULE_BASED_PLANNER", _env_bool("ENABLE_TOOL_AWARE_PLANNER", True)),
    "enable_tool_aware_planner": _env_bool("ENABLE_TOOL_AWARE_PLANNER", _env_bool("ENABLE_RULE_BASED_PLANNER", True)),
    "enable_experimental_llm_planner": _env_bool("ENABLE_EXPERIMENTAL_LLM_PLANNER", _env_bool("ENABLE_LLM_PLAN_DRAFT", True)),
    "enable_llm_plan_draft": _env_bool("ENABLE_LLM_PLAN_DRAFT", _env_bool("ENABLE_EXPERIMENTAL_LLM_PLANNER", True)),
    # Recovery 诊断默认关闭；即使开启也只提供语义诊断/排序建议，不能直接改写计划或执行工具。
    "enable_llm_recovery_diagnosis": _env_bool("ENABLE_LLM_RECOVERY_DIAGNOSIS", False),
    "llm_recovery_timeout": _env_int("LLM_RECOVERY_TIMEOUT_SECONDS", 6),
    "llm_plan_timeout": _env_int("LLM_PLAN_TIMEOUT_SECONDS", 8),
    "llm_plan_max_steps": _env_int("LLM_PLAN_MAX_STEPS", 8),
    # Artifact refinement 只允许 LLM 输出受控 patch；默认关闭，避免 profile-aware 路径无意增加模型成本。
    "enable_artifact_refinement": _env_bool("ENABLE_ARTIFACT_REFINEMENT", _env_bool("ENABLE_LLM_ARTIFACT_REFINEMENT", False)),
    "artifact_refinement_timeout": _env_int("ARTIFACT_REFINEMENT_TIMEOUT_SECONDS", 6),
    "artifact_refinement_max_patches": _env_int("ARTIFACT_REFINEMENT_MAX_PATCHES", 12),
    # LLM draft 失败后先回退到规则型 planner；显式开关便于日志和 trace 说明真实兜底顺序。
    "enable_rule_fallback_after_llm_planner": _env_bool("ENABLE_RULE_FALLBACK_AFTER_LLM_PLANNER", _env_bool("LLM_PLAN_FALLBACK_TO_RULE", True)),
    "llm_plan_fallback_to_rule": _env_bool("LLM_PLAN_FALLBACK_TO_RULE", _env_bool("ENABLE_RULE_FALLBACK_AFTER_LLM_PLANNER", True)),
    "enable_template_fallback_planner": _env_bool("ENABLE_TEMPLATE_FALLBACK_PLANNER", _env_bool("LLM_PLAN_FALLBACK_TO_TEMPLATE", True)),
    "llm_plan_fallback_to_template": _env_bool("LLM_PLAN_FALLBACK_TO_TEMPLATE", _env_bool("ENABLE_TEMPLATE_FALLBACK_PLANNER", True)),
    "expose_planner_debug": _env_bool("EXPOSE_PLANNER_DEBUG", True),
}

AGENT_RUNTIME_CHECKPOINT_CONFIG: Dict[str, Any] = {
    # 生产路径默认使用 SQLite 持久化 checkpoint；只有显式设置为 memory 时才退回进程内开发模式。
    "backend": _env_str("AGENT_RUNTIME_CHECKPOINT_BACKEND", "sqlite").lower() or "sqlite",
    # confirmation 等待现场需要有明确生命周期，避免用户长期不处理导致数据库无限增长。
    "ttl_seconds": _env_int("AGENT_RUNTIME_CHECKPOINT_TTL_SECONDS", 24 * 60 * 60),
    "cleanup_retention_days": _env_int("AGENT_RUNTIME_CHECKPOINT_CLEANUP_RETENTION_DAYS", 7),
}

CONTEXT_LIFECYCLE_CONFIG: Dict[str, Any] = {
    # 原始聊天消息长期保留；这里的限制只作用于 debug/trace 快照，避免诊断字段随会话无限膨胀。
    "debug_snapshot_mode": _env_str("CONTEXT_DEBUG_SNAPSHOT_MODE", "summary").lower() or "summary",
    "max_debug_string_chars": _env_int("CONTEXT_MAX_DEBUG_STRING_CHARS", 1200),
    "max_debug_list_items": _env_int("CONTEXT_MAX_DEBUG_LIST_ITEMS", 8),
    "max_debug_depth": _env_int("CONTEXT_MAX_DEBUG_DEPTH", 5),
    "max_trace_files_per_paper": _env_int("CONTEXT_TRACE_FILES_PER_PAPER", 20),
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

RECOMMENDATION_PROFILE_CONFIG: Dict[str, Any] = {
    "positive": {
        # 正向画像只表达“用户喜欢什么”，主向量和兴趣簇阈值都集中在这里，避免业务代码散落默认值。
        "min_liked_for_vector": _env_int("RECOMMENDATION_PROFILE_POSITIVE_MIN_LIKED_FOR_VECTOR", 1),
        "min_liked_for_clustering": _env_int(
            "RECOMMENDATION_PROFILE_POSITIVE_MIN_LIKED_FOR_CLUSTERING",
            _env_int("RECOMMENDATION_MIN_LIKED_PAPERS_FOR_CLUSTERING", 4),
        ),
        "max_interest_clusters": _env_int(
            "RECOMMENDATION_PROFILE_POSITIVE_MAX_INTEREST_CLUSTERS",
            _env_int("RECOMMENDATION_MAX_INTEREST_CLUSTERS", 4),
        ),
        # 自动行为画像必须比推荐主向量更保守：稳定簇和长期 topic 都要求跨论文重复出现。
        "min_cluster_paper_count_for_profile": _env_int("RECOMMENDATION_PROFILE_POSITIVE_MIN_CLUSTER_PAPER_COUNT_FOR_PROFILE", 2),
        "min_topic_source_papers_for_profile": _env_int("RECOMMENDATION_PROFILE_POSITIVE_MIN_TOPIC_SOURCE_PAPERS_FOR_PROFILE", 2),
        "clustering": {
            "min_cluster_size": _env_int(
                "RECOMMENDATION_PROFILE_POSITIVE_CLUSTERING_MIN_CLUSTER_SIZE",
                _env_int("RECOMMENDATION_HDBSCAN_MIN_CLUSTER_SIZE", 2),
            ),
            "min_samples": _env_int(
                "RECOMMENDATION_PROFILE_POSITIVE_CLUSTERING_MIN_SAMPLES",
                _env_int("RECOMMENDATION_HDBSCAN_MIN_SAMPLES", 1),
            ),
            "metric": _env_str(
                "RECOMMENDATION_PROFILE_POSITIVE_CLUSTERING_METRIC",
                _env_str("RECOMMENDATION_HDBSCAN_METRIC", "cosine"),
            ),
            "cluster_selection_method": _env_str(
                "RECOMMENDATION_PROFILE_POSITIVE_CLUSTERING_CLUSTER_SELECTION_METHOD",
                _env_str("RECOMMENDATION_HDBSCAN_CLUSTER_SELECTION_METHOD", "eom"),
            ),
            "allow_single_cluster": _env_bool(
                "RECOMMENDATION_PROFILE_POSITIVE_CLUSTERING_ALLOW_SINGLE_CLUSTER",
                _env_bool("RECOMMENDATION_HDBSCAN_ALLOW_SINGLE_CLUSTER", True),
            ),
        },
    },
    "negative": {
        # 负向反馈只作为“不要再推荐什么”的独立信号保存，后续由排序/过滤阶段消费。
        "enabled": _env_bool("RECOMMENDATION_PROFILE_NEGATIVE_ENABLED", True),
        "store_disliked_examples": _env_bool("RECOMMENDATION_PROFILE_NEGATIVE_STORE_DISLIKED_EXAMPLES", True),
        "min_disliked_for_instance_feedback": _env_int(
            "RECOMMENDATION_PROFILE_NEGATIVE_MIN_DISLIKED_FOR_INSTANCE_FEEDBACK",
            1,
        ),
        "max_disliked_examples": _env_int("RECOMMENDATION_PROFILE_NEGATIVE_MAX_DISLIKED_EXAMPLES", 20),
        "enable_negative_clustering": _env_bool("RECOMMENDATION_PROFILE_NEGATIVE_ENABLE_NEGATIVE_CLUSTERING", True),
        "min_disliked_for_clustering": _env_int("RECOMMENDATION_PROFILE_NEGATIVE_MIN_DISLIKED_FOR_CLUSTERING", 4),
        "max_negative_clusters": _env_int("RECOMMENDATION_PROFILE_NEGATIVE_MAX_NEGATIVE_CLUSTERS", 4),
        # 负向长期画像会影响后续过滤/降权，默认同样要求稳定负向簇，避免单次误点踩污染画像。
        "min_cluster_paper_count_for_profile": _env_int("RECOMMENDATION_PROFILE_NEGATIVE_MIN_CLUSTER_PAPER_COUNT_FOR_PROFILE", 2),
        "min_topic_source_papers_for_profile": _env_int("RECOMMENDATION_PROFILE_NEGATIVE_MIN_TOPIC_SOURCE_PAPERS_FOR_PROFILE", 2),
        "clustering": {
            "min_cluster_size": _env_int("RECOMMENDATION_PROFILE_NEGATIVE_CLUSTERING_MIN_CLUSTER_SIZE", 2),
            "min_samples": _env_int("RECOMMENDATION_PROFILE_NEGATIVE_CLUSTERING_MIN_SAMPLES", 1),
            "metric": _env_str("RECOMMENDATION_PROFILE_NEGATIVE_CLUSTERING_METRIC", "cosine"),
            "cluster_selection_method": _env_str(
                "RECOMMENDATION_PROFILE_NEGATIVE_CLUSTERING_CLUSTER_SELECTION_METHOD",
                "eom",
            ),
            "allow_single_cluster": _env_bool(
                "RECOMMENDATION_PROFILE_NEGATIVE_CLUSTERING_ALLOW_SINGLE_CLUSTER",
                True,
            ),
        },
        "fallback_to_examples_when_cluster_failed": _env_bool(
            "RECOMMENDATION_PROFILE_NEGATIVE_FALLBACK_TO_EXAMPLES_WHEN_CLUSTER_FAILED",
            True,
        ),
    },
}

RECOMMENDATION_CLUSTERING_CONFIG: Dict[str, Any] = {
    **RECOMMENDATION_PROFILE_CONFIG["positive"]["clustering"],
}

RECOMMENDATION_RANKING_CONFIG: Dict[str, Any] = {
    "negative": {
        # 负向反馈只在排序阶段生效；默认不开 hard filter，避免一两次点踩误伤相近但可能有价值的论文。
        "enabled": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_ENABLED", True),
        "similarity_threshold": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_SIMILARITY_THRESHOLD", "0.82")),
        "margin": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_MARGIN", "0.05")),
        "penalty_weight": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_PENALTY_WEIGHT", "0.15")),
        "confidence_min_count": _env_int("RECOMMENDATION_RANKING_NEGATIVE_CONFIDENCE_MIN_COUNT", 6),
        "max_penalty": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_MAX_PENALTY", "0.22")),
        "use_margin_penalty": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_USE_MARGIN_PENALTY", True),
        "use_threshold_penalty": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_USE_THRESHOLD_PENALTY", True),
        "hard_filter_threshold": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_HARD_FILTER_THRESHOLD", "0.97")),
        "enable_hard_filter": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_ENABLE_HARD_FILTER", False),
        "debug_enabled": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_DEBUG_ENABLED", True),
    },
}

RECOMMENDATION_CONFIG: Dict[str, Any] = {
    "backfill_request_interval_seconds": float(_env_str("RECOMMENDATION_BACKFILL_REQUEST_INTERVAL_SECONDS", "8.0")),
    "profile": RECOMMENDATION_PROFILE_CONFIG,
    "ranking": RECOMMENDATION_RANKING_CONFIG,
    # 兼容旧环境变量/读取路径：这些字段已迁移到 recommendation.profile.positive，不再作为业务读取入口。
    "min_liked_papers_for_clustering": RECOMMENDATION_PROFILE_CONFIG["positive"]["min_liked_for_clustering"],
    "max_interest_clusters": RECOMMENDATION_PROFILE_CONFIG["positive"]["max_interest_clusters"],
    # deprecated: 负向反馈不再参与主兴趣向量生成，仅保留旧配置读取兼容。
    "negative_weight_default": float(_env_str("RECOMMENDATION_NEGATIVE_WEIGHT_DEFAULT", "0.3")),
    "default_top_n": _env_int("RECOMMENDATION_DEFAULT_TOP_N", 10),
    "default_max_age_months": _env_int("RECOMMENDATION_DEFAULT_MAX_AGE_MONTHS", 6),
    "category_query_max_categories": _env_int("RECOMMENDATION_CATEGORY_QUERY_MAX_CATEGORIES", 5),
    "candidate_concept_enrichment": {
        # 只对基础相关性靠前的候选补 evidence concepts，避免在大召回池上无差别调用 LLM。
        "enabled": _env_bool("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_ENABLED", True),
        "top_k": _env_int("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_TOP_K", 20),
        "max_llm_calls": _env_int("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_MAX_LLM_CALLS", 5),
        # 当前推荐接口优先保障稳定延迟；打开后只复用缓存，不在主链路同步等待新的概念抽取。
        "allow_lazy_generation": _env_bool("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_ALLOW_LAZY_GENERATION", False),
        # 预留异步开关；当前实现与 lazy 一致，都用于禁止主链路继续发起新的 LLM 调用。
        "allow_async_generation": _env_bool("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_ALLOW_ASYNC_GENERATION", False),
        "cache_version": _env_str(
            "RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_CACHE_VERSION",
            DEFAULT_RECOMMENDATION_CONCEPT_CACHE_VERSION,
        ),
        "score_field": _env_str("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_SCORE_FIELD", "base_score"),
    },
    "score_weights": {
        "semantic": float(_env_str("RECOMMENDATION_SCORE_WEIGHT_SEMANTIC", "0.65")),
        "category": float(_env_str("RECOMMENDATION_SCORE_WEIGHT_CATEGORY", "0.08")),
        "recency": float(_env_str("RECOMMENDATION_SCORE_WEIGHT_RECENCY", "0.05")),
        # deprecated: 负向扣分已迁移到 recommendation.ranking.negative.penalty_weight / max_penalty。
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

PROMPT_CONTEXT_CONFIG: Dict[str, Any] = {
    "enable_block_prompt_context": _env_bool("ENABLE_BLOCK_PROMPT_CONTEXT", True),
    "prompt_max_input_tokens": _env_int("PROMPT_MAX_INPUT_TOKENS", 64000),
    "prompt_safety_margin_tokens": _env_int("PROMPT_SAFETY_MARGIN_TOKENS", 2048),
    "rag_target_ratio": float(_env_str("PROMPT_RAG_TARGET_RATIO", "0.78")),
    "token_counter_provider": _env_str("PROMPT_TOKEN_COUNTER_PROVIDER", "auto"),
    "tokenizer_name_or_path": _env_str("PROMPT_TOKENIZER_NAME_OR_PATH", ""),
    "token_counter_fallback_chars_per_token": float(_env_str("PROMPT_TOKEN_FALLBACK_CHARS_PER_TOKEN", "3.0")),
    "enable_rule_compaction": _env_bool("ENABLE_PROMPT_RULE_COMPACTION", True),
    "enable_llm_compaction": _env_bool("ENABLE_PROMPT_LLM_COMPACTION", False),
    # LLM 压缩是规则压缩后的二级优化，默认关闭；阈值限制可控调用量，避免长证据批量触发外部模型。
    "llm_compaction_min_block_tokens": _env_int("PROMPT_LLM_COMPACTION_MIN_BLOCK_TOKENS", 1200),
    "llm_compaction_target_block_tokens": _env_int("PROMPT_LLM_COMPACTION_TARGET_BLOCK_TOKENS", 700),
    "llm_compaction_max_blocks": _env_int("PROMPT_LLM_COMPACTION_MAX_BLOCKS", 8),
    "llm_compaction_model_name": _env_str("PROMPT_LLM_COMPACTION_MODEL_NAME", ""),
    "llm_compaction_task_type": _env_str("PROMPT_LLM_COMPACTION_TASK_TYPE", "prompt_context_compaction"),
    "llm_compaction_enable_thinking": _env_bool("PROMPT_LLM_COMPACTION_ENABLE_THINKING", False),
    "recent_turns_recent_full_count": _env_int("PROMPT_RECENT_TURNS_RECENT_FULL_COUNT", 2),
    "recent_turns_source_id_limit": _env_int("PROMPT_RECENT_TURNS_SOURCE_ID_LIMIT", 6),
    "rag_max_block_tokens": _env_int("PROMPT_RAG_MAX_BLOCK_TOKENS", 4000),
    "rag_sibling_context_target_tokens": _env_int("PROMPT_RAG_SIBLING_CONTEXT_TARGET_TOKENS", 350),
    "rag_section_context_target_tokens": _env_int("PROMPT_RAG_SECTION_CONTEXT_TARGET_TOKENS", 250),
    "rag_fallback_preview_tokens": _env_int("PROMPT_RAG_FALLBACK_PREVIEW_TOKENS", 160),
    "table_candidate_cell_limit": _env_int("PROMPT_TABLE_CANDIDATE_CELL_LIMIT", 8),
    "figure_preview_tokens": _env_int("PROMPT_FIGURE_PREVIEW_TOKENS", 220),
}

ENHANCED_RETRIEVAL_CONFIG: Dict[str, Any] = {
    "query_view_limit": _env_int("ENHANCED_RETRIEVAL_QUERY_VIEW_LIMIT", 6),
    "query_plan_limit": _env_int("ENHANCED_RETRIEVAL_QUERY_PLAN_LIMIT", 5),
    "recall_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RECALL_CANDIDATE_LIMIT", 30),
    "rrf_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RRF_CANDIDATE_LIMIT", 30),
    "rerank_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RERANK_CANDIDATE_LIMIT", 30),
    "final_context_top_k": _env_int("ENHANCED_RETRIEVAL_FINAL_CONTEXT_TOP_K", 15),
    "max_final_context_top_k": _env_int("ENHANCED_RETRIEVAL_MAX_FINAL_CONTEXT_TOP_K", 30),
    "enable_context_expansion": _env_bool("ENHANCED_RETRIEVAL_ENABLE_CONTEXT_EXPANSION", True),
    # Multi-index 灰度开关分开控制 dense、sparse、生成式问题索引和旧 chunk 兜底，便于逐段回滚定位问题。
    "enable_multi_index_embedding": _env_bool("ENABLE_MULTI_INDEX_EMBEDDING", True),
    "enable_index_level_bm25": _env_bool("ENABLE_INDEX_LEVEL_BM25", True),
    "enable_generated_question_index": _env_bool("ENABLE_GENERATED_QUESTION_INDEX", True),
    "enable_chunk_level_retrieval_fallback": _env_bool("ENABLE_CHUNK_LEVEL_RETRIEVAL_FALLBACK", True),
    "context_budget_max_chars": _env_int("ENHANCED_RETRIEVAL_CONTEXT_BUDGET_MAX_CHARS", 24000),
    "retrieval_candidate_max_blocks": _env_int("RETRIEVAL_CANDIDATE_MAX_BLOCKS", 60),
    "retrieval_candidate_max_tokens_soft": _env_int("RETRIEVAL_CANDIDATE_MAX_TOKENS_SOFT", 90000),
    "retrieval_candidate_token_chars_per_token": float(_env_str("RETRIEVAL_CANDIDATE_TOKEN_CHARS_PER_TOKEN", "3.0")),
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
    "enable_table_structured_route": _env_bool("ENHANCED_RETRIEVAL_ENABLE_TABLE_STRUCTURED_ROUTE", True),
    "table_structured_candidate_limit": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CANDIDATE_LIMIT", 8),
    "table_structured_match_score_floor": float(_env_str("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_MATCH_SCORE_FLOOR", "0.22")),
    # 表格 route 在不确定时会把局部表格上下文交给 LLM；这些参数限制上下文体积，避免规则层为了覆盖长表继续膨胀。
    "table_structured_context_short_table_cell_limit": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CONTEXT_SHORT_TABLE_CELL_LIMIT", 48),
    "table_structured_context_max_rows": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CONTEXT_MAX_ROWS", 12),
    "table_structured_context_focus_window": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CONTEXT_FOCUS_WINDOW", 2),
    "route_focus_bonus": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_FOCUS_BONUS", "0.08")),
    "route_summary_bonus": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_SUMMARY_BONUS", "0.05")),
    "route_keyword_bonus": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_KEYWORD_BONUS", "0.08")),
    "route_default_floor": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_DEFAULT_FLOOR", "0.2")),
    "route_confidence_multiplier": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_CONFIDENCE_MULTIPLIER", "0.65")),
    "route_confidence_similarity_weight": float(_env_str("ENHANCED_RETRIEVAL_ROUTE_CONFIDENCE_SIMILARITY_WEIGHT", "0.35")),
    "route_timeout_default_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_DEFAULT_SECONDS", "8")),
    "route_timeout_vector_original_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_VECTOR_ORIGINAL_SECONDS", "8")),
    "route_timeout_vector_rewrite_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_VECTOR_REWRITE_SECONDS", "8")),
    "route_timeout_vector_hyde_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_VECTOR_HYDE_SECONDS", "8")),
    "route_timeout_keyword_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_KEYWORD_SECONDS", "4")),
    "route_timeout_table_structured_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_TABLE_STRUCTURED_SECONDS", "3")),
    "route_timeout_memory_context_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_MEMORY_CONTEXT_SECONDS", "3")),
    "route_timeout_rerank_seconds": float(_env_str("RETRIEVAL_ROUTE_TIMEOUT_RERANK_SECONDS", "12")),
    "route_max_workers": _env_int("RETRIEVAL_ROUTE_MAX_WORKERS", 4),
    "query_embedding_batch_size": _env_int("RETRIEVAL_QUERY_EMBEDDING_BATCH_SIZE", 20),
}

INTENT_ROUTING_CONFIG: Dict[str, Any] = {
    "default_route_weights": {
        "vector_original": 1.0,
        "vector_rewrite": 0.9,
        "vector_hyde": 0.85,
        "keyword": 0.75,
        "table_structured": 0.95,
    },
    "intent_route_weights": {
        "contribution": {
            "vector_original": 1.15,
            "vector_rewrite": 1.1,
            "vector_hyde": 0.9,
            "keyword": 0.6,
            "table_structured": 0.6,
        },
        "paper_overview": {
            "vector_original": 1.15,
            "vector_rewrite": 1.1,
            "vector_hyde": 0.9,
            "keyword": 0.6,
            "table_structured": 0.6,
        },
        "method_flow": {
            "vector_original": 0.95,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.85,
            "keyword": 1.0,
            "table_structured": 0.8,
        },
        "experiment_setup": {
            "vector_original": 0.9,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.8,
            "keyword": 1.1,
            "table_structured": 1.05,
        },
        "result_analysis": {
            "vector_original": 0.88,
            "vector_rewrite": 1.18,
            "vector_hyde": 0.82,
            "keyword": 1.08,
            "table_structured": 1.2,
        },
        "comparison": {
            "vector_original": 0.85,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.8,
            "keyword": 1.0,
            "table_structured": 1.18,
        },
        "dataset": {
            "vector_original": 0.9,
            "vector_rewrite": 1.1,
            "vector_hyde": 0.8,
            "keyword": 1.05,
            "table_structured": 1.08,
        },
        "limitation": {
            "vector_original": 0.85,
            "vector_rewrite": 1.05,
            "vector_hyde": 0.8,
            "keyword": 0.95,
            "table_structured": 0.75,
        },
        "definition": {
            "vector_original": 0.9,
            "vector_rewrite": 1.05,
            "vector_hyde": 0.85,
            "keyword": 1.0,
            "table_structured": 0.8,
        },
        "implementation_detail": {
            "vector_original": 0.95,
            "vector_rewrite": 1.15,
            "vector_hyde": 0.85,
            "keyword": 1.0,
            "table_structured": 0.78,
        },
        "figure_table": {
            "vector_original": 0.75,
            "vector_rewrite": 0.95,
            "vector_hyde": 0.7,
            "keyword": 1.2,
            "table_structured": 1.22,
        },
        "other": {
            "vector_original": 1.0,
            "vector_rewrite": 0.9,
            "vector_hyde": 0.85,
            "keyword": 0.75,
            "table_structured": 0.88,
        },
    },
}

_default_rerank_model_name = _env_str("RERANK_MODEL_NAME", "qwen3-vl-rerank")
_default_rerank_base_url = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    if _default_rerank_model_name in {"qwen3-vl-rerank", "gte-rerank-v2"}
    else "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
)

RERANK_CONFIG: Dict[str, Any] = {
    "provider": _env_str("RERANK_PROVIDER", "dashscope"),
    "model_name": _default_rerank_model_name,
    "local_model_name_or_path": _env_str(
        "RERANK_LOCAL_MODEL_NAME_OR_PATH",
        str(REPO_ROOT / "00-models" / "Qwen3-VL-Reranker-2B"),
    ),
    "api_key": _env_str("RERANK_API_KEY", ALIYUN_API_KEY),
    "dashscope_api_key": _env_str("RERANK_DASHSCOPE_API_KEY", ALIYUN_API_KEY),
    "base_url": _env_str("RERANK_BASE_URL", _default_rerank_base_url),
    "fallback_local": _env_bool("RERANK_FALLBACK_LOCAL", False),
    "prompt": _env_str("RERANK_PROMPT", "Retrieve text relevant to the user's query."),
    "batch_size": _env_int("RERANK_BATCH_SIZE", 8),
    "candidate_limit": _env_int("RERANK_CANDIDATE_LIMIT", 24),
    "max_doc_chars": _env_int("RERANK_MAX_DOC_CHARS", 4096),
    "enable_query_rewrite": _env_bool("ENABLE_QUERY_REWRITE", True),
    "enable_hyde": _env_bool("ENABLE_HYDE", False),
    "enable_keyword_search": _env_bool("ENABLE_KEYWORD_SEARCH", True),
    "keyword_backend": _env_str("KEYWORD_BACKEND", "bm25s"),  # bm25s or internal_bm25
    "enable_multi_index_embedding": _env_bool("ENABLE_MULTI_INDEX_EMBEDDING", True),
    "enable_index_level_bm25": _env_bool("ENABLE_INDEX_LEVEL_BM25", True),
    "enable_generated_question_index": _env_bool("ENABLE_GENERATED_QUESTION_INDEX", True),
    "enable_chunk_level_retrieval_fallback": _env_bool("ENABLE_CHUNK_LEVEL_RETRIEVAL_FALLBACK", True),
    "enable_table_structured_route": _env_bool("ENABLE_TABLE_STRUCTURED_ROUTE", True),
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
        "table_structured": float(_env_str("RETRIEVAL_ROUTE_WEIGHT_TABLE_STRUCTURED", "0.95")),
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
    "keyword_backend": RERANK_CONFIG["keyword_backend"],
    "enable_multi_index_embedding": RERANK_CONFIG["enable_multi_index_embedding"],
    "enable_index_level_bm25": RERANK_CONFIG["enable_index_level_bm25"],
    "enable_generated_question_index": RERANK_CONFIG["enable_generated_question_index"],
    "enable_chunk_level_retrieval_fallback": RERANK_CONFIG["enable_chunk_level_retrieval_fallback"],
    "enable_table_structured_route": RERANK_CONFIG["enable_table_structured_route"],
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
    # 跨厂商调用必须显式提供对应密钥，避免错误路由时把阿里云凭据发送给第三方端点。
    "openai_api_key": _env_str("OPENAI_API_KEY", ""),
    "deepseek_api_key": _env_str("DEEPSEEK_API_KEY", ""),
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
        "query_planning": "small",  # 规划多步或多查询的检索策略
        "query_rewrite": "small",  # 重写或规范化查询以提升检索效果
        "rerank_query": "small",  # 为重排阶段生成或调整查询
        "prompt_context_compaction": "small",  # 对长证据块做语义压缩，结果仍需最终回答模型基于证据使用
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


def get_debug_routes_runtime_config() -> Dict[str, Any]:
    return dict(DEBUG_ROUTES_CONFIG)


def get_backend_logging_runtime_config() -> Dict[str, Any]:
    return dict(BACKEND_LOGGING_CONFIG)


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


def get_qa_index_job_runtime_config() -> Dict[str, Any]:
    return dict(QA_INDEX_JOB_CONFIG)


def get_paper_qa_build_cache_config() -> Dict[str, Any]:
    return dict(PAPER_QA_BUILD_CACHE_CONFIG)


def get_agent_planner_runtime_config() -> Dict[str, Any]:
    return dict(AGENT_PLANNER_CONFIG)


def get_agent_runtime_checkpoint_config() -> Dict[str, Any]:
    return dict(AGENT_RUNTIME_CHECKPOINT_CONFIG)


def get_context_lifecycle_config() -> Dict[str, Any]:
    return dict(CONTEXT_LIFECYCLE_CONFIG)


def get_enhanced_retrieval_runtime_config() -> Dict[str, Any]:
    return dict(ENHANCED_RETRIEVAL_CONFIG)


def get_intent_routing_runtime_config() -> Dict[str, Any]:
    return dict(INTENT_ROUTING_CONFIG)


def get_rerank_runtime_config() -> Dict[str, Any]:
    return dict(RERANK_CONFIG)


def get_generation_runtime_config() -> Dict[str, Any]:
    return dict(GENERATION_CONFIG)
