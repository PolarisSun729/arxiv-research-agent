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
    "arxiv_data_source": _env_str("ARXIV_DATA_SOURCE", "api"),
    "arxiv_proxy_url": _env_str(
        "ARXIV_PROXY_URL",
        "http://127.0.0.1:7897",
    ),
}

DOCLING_CONFIG: Dict[str, Any] = {
    "do_ocr_enabled": _env_bool("DOCLING_OCR_ENABLED", False),
    "annotated_pdf_export_enabled": _env_bool("DOCLING_ANNOTATED_PDF_EXPORT_ENABLED", True),
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

RERANK_CONFIG: Dict[str, Any] = {
    "provider": _env_str("RERANK_PROVIDER", "dashscope"),
    "model_name": _env_str("RERANK_MODEL_NAME", "qwen3-rerank"),
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
    "qwen_model_name": _env_str("QWEN_MODEL_NAME", "qwen3.6-plus"),
    "openai_api_key": _env_str("OPENAI_API_KEY", ALIYUN_API_KEY),
    "deepseek_api_key": _env_str("DEEPSEEK_API_KEY", ALIYUN_API_KEY),
}


def get_embedding_runtime_config() -> Dict[str, Any]:
    return dict(EMBEDDING_CONFIG)


def get_rerank_runtime_config() -> Dict[str, Any]:
    return dict(RERANK_CONFIG)


def get_generation_runtime_config() -> Dict[str, Any]:
    return dict(GENERATION_CONFIG)
