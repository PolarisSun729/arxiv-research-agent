import os
import platform
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from utils.agent_planner_limits import AGENT_PLANNER_LIMITS
from utils.storage_paths import BACKEND_DATA_ROOT, BACKEND_FULLTEXT_CACHE_ROOT, resolve_storage_path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent

# 与启动目录无关地加载配置：进程环境优先，其次根目录 .env，最后 backend/.env。
# 必须在各项配置首次读取之前完成，否则 API 认证与模型客户端可能拿到不同的环境值。
load_dotenv(REPO_ROOT / ".env", override=False, encoding="utf-8-sig")
load_dotenv(REPO_ROOT / "backend" / ".env", override=False, encoding="utf-8-sig")

DEFAULT_RECOMMENDATION_CONCEPT_CACHE_VERSION = "llm_paper_evidence_v1"

# 平台检测：自动适配 Milvus 配置
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


class VectorDBProvider(str, Enum):
    MILVUS = "milvus"


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量；非法值回退默认配置，避免配置拼写错误阻断服务启动。"""
    raw = _env_str(name, str(default))
    try:
        return float(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, str(default)).lower()
    return raw not in {"0", "false", "no", "off", ""}


def _env_optional_int(name: str) -> Any:
    """读取可选整数环境变量；未设置、为空或非法时返回 None，表示“沿用代码中的默认策略”。"""
    raw = _env_str(name, "")
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


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
    # 密钥脱敏是固定安全边界，调高日志级别或旧环境开关都不能关闭。
    "redact_secrets": True,
}

DOCLING_CONFIG: Dict[str, Any] = {
    # Docling 解析放进独立子进程，避免长时间占住 Web 进程的 GIL 导致 gunicorn worker 心跳超时被杀。
    "subprocess_enabled": _env_bool("DOCLING_SUBPROCESS_ENABLED", True),
    # 单次解析的最长时间，超时强制结束子进程；内存不足时进程会卡在换页上，必须有上限才不会无限等待。
    "subprocess_timeout_seconds": _env_int("DOCLING_SUBPROCESS_TIMEOUT_SECONDS", 15 * 60),
    # 同一时间只允许一个 Docling 子进程；后来的解析最多排队等待这么久，超时则本次建索引失败。
    "subprocess_lock_timeout_seconds": _env_int("DOCLING_SUBPROCESS_LOCK_TIMEOUT_SECONDS", 15 * 60),
    "do_ocr_enabled": _env_bool("DOCLING_OCR_ENABLED", False),
    "annotated_pdf_export_enabled": _env_bool("DOCLING_ANNOTATED_PDF_EXPORT_ENABLED", True),
    # Docling 会按 images_scale 渲染页面/图片资产；默认保持低倍率，避免 30+ 页论文在建索引时生成超大图。
    # 整页截图默认关闭：表格资产由 PyMuPDF 从原 PDF 裁剪，图片资产只依赖 generate_picture_images。
    "generate_page_images": _env_bool("DOCLING_GENERATE_PAGE_IMAGES", False),
    "generate_picture_images": _env_bool("DOCLING_GENERATE_PICTURE_IMAGES", True),
    "images_scale": float(_env_str("DOCLING_IMAGES_SCALE", "2.0")),
    # Docling 流水线默认每批 4 页、阶段间最多缓存 100 页、4 个推理线程；
    # 4G 机器上解析单篇论文时子进程曾超过 2.5G 被 OOM 杀掉，这里压到逐页处理以降低内存峰值。
    "layout_batch_size": _env_int("DOCLING_LAYOUT_BATCH_SIZE", 1),
    "table_batch_size": _env_int("DOCLING_TABLE_BATCH_SIZE", 1),
    "ocr_batch_size": _env_int("DOCLING_OCR_BATCH_SIZE", 1),
    "queue_max_size": _env_int("DOCLING_QUEUE_MAX_SIZE", 4),
    "num_threads": _env_int("DOCLING_NUM_THREADS", 2),
    "legend_padding": float(_env_str("DOCLING_LEGEND_PADDING", "8.0")),
    "legend_width": float(_env_str("DOCLING_LEGEND_WIDTH", "165.0")),
    "legend_row_height": float(_env_str("DOCLING_LEGEND_ROW_HEIGHT", "13.0")),
    "legend_title_height": float(_env_str("DOCLING_LEGEND_TITLE_HEIGHT", "12.0")),
    "legend_margin": float(_env_str("DOCLING_LEGEND_MARGIN", "12.0")),
    "table_caption_short_text_limit": _env_int("DOCLING_TABLE_CAPTION_SHORT_TEXT_LIMIT", 8),
    "vertical_gap_multiplier": float(_env_str("DOCLING_VERTICAL_GAP_MULTIPLIER", "1.6")),
    "x_aligned_tolerance": float(_env_str("DOCLING_X_ALIGNED_TOLERANCE", "24.0")),
    "same_column_tolerance": float(_env_str("DOCLING_SAME_COLUMN_TOLERANCE", "80.0")),
}

# Milvus 平台自适应配置
# Windows: 使用完整 Milvus Standalone (Docker)
# Linux: 使用轻量级 Milvus Lite (本地文件)
def _get_default_milvus_uri() -> str:
    """根据操作系统返回默认 Milvus URI"""
    # 优先使用环境变量
    env_uri = _env_str("MILVUS_URI", "")
    if env_uri:
        return env_uri

    # Windows 默认使用完整 Milvus
    if IS_WINDOWS:
        return "http://localhost:19530"

    # Linux 默认使用 Milvus Lite
    milvus_lite_path = BACKEND_DATA_ROOT / "milvus_lite.db"
    return str(milvus_lite_path)

MILVUS_CONFIG: Dict[str, Any] = {
    "uri": _get_default_milvus_uri(),
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
    # Milvus Lite 合并数据文件时把输入整份读进内存再复制几份，峰值约每行 40–55KB（2048 维）；
    # 一次最多合并 2 倍这么多行，更大的文件不再参与合并。默认 8000 → 单次峰值约 +0.9GB。
    "lite_max_segment_rows": _env_int("MILVUS_LITE_MAX_SEGMENT_ROWS", 8000),
}

# 论文级向量（arxiv_paper_embeddings）的存储后端：milvus 沿用 Milvus Lite；
# file 存成内存映射的向量文件 + SQLite 元数据，向量不再常驻 Web 进程内存（论文问答的分块向量不受影响）。
PAPER_VECTOR_CONFIG: Dict[str, Any] = {
    "backend": _env_str("PAPER_VECTOR_BACKEND", "milvus").lower(),
    "vectors_path": resolve_storage_path(
        _env_str("PAPER_VECTOR_FILE_PATH"),
        default_path=BACKEND_DATA_ROOT / "paper_vectors.f32",
    ),
    "db_path": resolve_storage_path(
        _env_str("PAPER_VECTOR_DB_PATH"),
        default_path=BACKEND_DATA_ROOT / "paper_vectors.db",
    ),
}

SQLITE_CONFIG: Dict[str, Any] = {
    # 持久化路径在配置加载时固定解析，避免启动目录不同导致状态库分裂。
    "database_path": resolve_storage_path(
        _env_str("SQLITE_DATABASE_PATH"),
        default_path=BACKEND_DATA_ROOT / "recommendation.db",
    ),
    "check_same_thread": _env_bool("SQLITE_CHECK_SAME_THREAD", False),
}

USER_CONFIG: Dict[str, Any] = {
    # 公共仓库使用非个人化的本地用户标识，部署时可通过环境变量覆盖。
    "default_user_id": _env_str("DEFAULT_USER_ID", "local_user"),
}

QA_INDEX_JOB_CONFIG: Dict[str, Any] = {
    # timeout_seconds 保留给旧查询兼容；真正的失联判定由 lease 与独立 heartbeat 驱动。
    "timeout_seconds": _env_int("QA_INDEX_JOB_TIMEOUT_SECONDS", 30 * 60),
    "lease_seconds": _env_int("QA_INDEX_JOB_LEASE_SECONDS", 90),
    "heartbeat_interval_seconds": _env_int("QA_INDEX_JOB_HEARTBEAT_INTERVAL_SECONDS", 15),
    "poll_interval_seconds": _env_int("QA_INDEX_JOB_POLL_INTERVAL_SECONDS", 1),
    # 自动 attempt 只覆盖 worker 崩溃等基础设施中断；明确业务失败仍要求新的用户授权。
    "max_attempts": _env_int("QA_INDEX_JOB_MAX_ATTEMPTS", 3),
    "recipe_version": _env_str("QA_INDEX_JOB_RECIPE_VERSION", "paper_qa_index_v1"),
}

PAPER_QA_BUILD_CACHE_CONFIG: Dict[str, Any] = {
    # QA 建库里的 LLM 增强和 embedding 都是高成本外部调用；持久缓存用于失败重试和重复重建时复用确定性结果。
    "enabled": _env_bool("PAPER_QA_BUILD_CACHE_ENABLED", True),
    "root_dir": resolve_storage_path(
        _env_str("PAPER_QA_BUILD_CACHE_DIR"),
        default_path=BACKEND_DATA_ROOT,
    ),
    "llm_cache_name": _env_str("PAPER_QA_LLM_CACHE_NAME", "paper_qa_llm_cache"),
    "embedding_cache_name": _env_str("PAPER_QA_EMBEDDING_CACHE_NAME", "paper_qa_embedding_cache"),
    "llm_size_limit": _env_int("PAPER_QA_LLM_CACHE_SIZE_LIMIT", 512 * 1024 * 1024),
    "embedding_size_limit": _env_int("PAPER_QA_EMBEDDING_CACHE_SIZE_LIMIT", 2 * 1024 * 1024 * 1024),
    # per-chunk LLM 调用互不依赖，默认小并发提速，同时避免把 provider 限流压力放大到不可控。
    "llm_max_workers": _env_int("PAPER_QA_BUILD_LLM_MAX_WORKERS", 4),
}

PAPER_FAST_QA_CONFIG: Dict[str, Any] = {
    # 快速问答把整篇论文全文和图片直接送入模型，不建索引；全文解析结果按论文缓存，所有用户共用。
    "cache_dir": resolve_storage_path(
        _env_str("PAPER_FAST_QA_CACHE_DIR"),
        default_path=BACKEND_FULLTEXT_CACHE_ROOT,
    ),
    # 解析在独立子进程中运行，与 Docling 建索引各用各的并发名额；单篇峰值内存约 0.4～0.7G，并发数要按机器内存调整。
    "parse_concurrency": _env_int("PAPER_FAST_QA_PARSE_CONCURRENCY", 2),
    # 只限制打开详情页触发的预解析排队数；用户提问触发的解析不受限，并排在预解析前面。
    "parse_queue_limit": _env_int("PAPER_FAST_QA_PARSE_QUEUE_LIMIT", 20),
    # 实测单篇最慢 60 秒（49 页）；两篇同时解析时 CPU 争用使单篇耗时涨到原来的约 6 倍（19 秒 → 2 分钟），
    # 按最慢单篇乘 6 约 6 分钟，再给比开发机慢的服务器留余量。
    "parse_timeout_seconds": _env_int("PAPER_FAST_QA_PARSE_TIMEOUT_SECONDS", 600),
    # 提问时全文还没解析好，最多等这么久；超时本轮报错，解析任务继续在后台跑。
    "ask_wait_seconds": _env_int("PAPER_FAST_QA_ASK_WAIT_SECONDS", 120),
    # 解析失败后这段时间内不再重试，避免同一篇坏 PDF 被反复解析占满名额。
    "failure_cooldown_seconds": _env_int("PAPER_FAST_QA_FAILURE_COOLDOWN_SECONDS", 300),
    # 图片、公式按 2 倍分辨率（144 dpi）渲染，保证图中小字可读。
    "image_dpi": _env_int("PAPER_FAST_QA_IMAGE_DPI", 144),
    # 带入最近几轮对话，供追问理解指代。
    "history_turns": _env_int("PAPER_FAST_QA_HISTORY_TURNS", 3),
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
    # 规划预算与恢复上限：配置项、环境变量和默认值统一定义在 utils/agent_planner_limits.py。
    **{key: _env_int(env_name, default) for key, (env_name, default) in AGENT_PLANNER_LIMITS.items()},
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
    # OAI 与推荐状态库共享同一条路径规则，避免工具脚本因 cwd 不同重建另一份索引库。
    "database_path": resolve_storage_path(
        _env_str("OAI_SQLITE_DATABASE_PATH"),
        default_path=BACKEND_DATA_ROOT / "arxiv_oai.db",
    ),
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
    # 远程 embedding 会连续执行大量请求；有限重试用于吸收 TLS 断连、限流和短暂 5xx，不能掩盖永久配置错误。
    "request_max_retries": _env_int("EMBEDDING_REQUEST_MAX_RETRIES", 3),
    "backoff_base_seconds": _env_float("EMBEDDING_REQUEST_BACKOFF_BASE_SECONDS", 1.0),
    "backoff_max_seconds": _env_float("EMBEDDING_REQUEST_BACKOFF_MAX_SECONDS", 30.0),
    "jitter_ratio": _env_float("EMBEDDING_REQUEST_JITTER_RATIO", 0.2),
    # 一次建索引里同时在途的远程 embedding 请求数；调大更快，但更容易触发供应商限流。
    "request_max_workers": _env_int("EMBEDDING_REQUEST_MAX_WORKERS", 4),
    # 进程内 embedding 缓存的最大条数，超出后按最近最少使用淘汰；2048 维向量每条约 16KB。
    "memory_cache_max_entries": _env_int("EMBEDDING_MEMORY_CACHE_MAX_ENTRIES", 2048),
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
    # 论文向量只保留首次提交在最近 N 个月内的论文（按 N*30 天算，和推荐近期论文池的口径一致，默认值也与推荐的 6 个月对齐）；
    # 维护任务不给更早的论文生成向量，并每天删掉移出窗口的向量（点赞/点踩过的除外）。0 表示不限、也不清理。
    "embedding_retention_months": _env_int("ARXIV_OAI_EMBEDDING_RETENTION_MONTHS", 6),
    # 论文向量维护任务：Web 进程内的后台线程，补齐窗口内缺向量的论文并清理窗口外向量。
    # Milvus Lite 只允许一个进程打开，论文向量只能由 Web 进程写；默认关闭，避免本地起服务就调用付费 embedding。
    "embedding_maintainer_enabled": _env_bool("ARXIV_OAI_EMBEDDING_MAINTAINER_ENABLED", False),
    "embedding_maintainer_startup_delay_seconds": _env_float("ARXIV_OAI_EMBEDDING_MAINTAINER_STARTUP_DELAY_SECONDS", 120.0),
    "embedding_maintainer_interval_seconds": _env_float("ARXIV_OAI_EMBEDDING_MAINTAINER_INTERVAL_SECONDS", 1800.0),
    # 单轮最多生成多少篇的向量；补齐阶段触顶后短暂休息就开下一轮，把长时间补齐拆成可中断的短轮次。
    "embedding_maintainer_max_per_round": _env_int("ARXIV_OAI_EMBEDDING_MAINTAINER_MAX_PER_ROUND", 1000),
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
        # 相似度阈值都基于 max(0, cos) 口径（与 Milvus COSINE 分数一致）；
        # 旧版 (cos+1)/2 口径下的 0.82 / 0.05 / 0.97 分别等价于这里的 0.64 / 0.10 / 0.94。
        "enabled": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_ENABLED", True),
        "similarity_threshold": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_SIMILARITY_THRESHOLD", "0.64")),
        "margin": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_MARGIN", "0.10")),
        "penalty_weight": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_PENALTY_WEIGHT", "0.15")),
        "confidence_min_count": _env_int("RECOMMENDATION_RANKING_NEGATIVE_CONFIDENCE_MIN_COUNT", 6),
        "max_penalty": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_MAX_PENALTY", "0.22")),
        "use_margin_penalty": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_USE_MARGIN_PENALTY", True),
        "use_threshold_penalty": _env_bool("RECOMMENDATION_RANKING_NEGATIVE_USE_THRESHOLD_PENALTY", True),
        "hard_filter_threshold": float(_env_str("RECOMMENDATION_RANKING_NEGATIVE_HARD_FILTER_THRESHOLD", "0.94")),
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
        # 单次请求最多发起（同步执行或提交后台）的概念抽取次数。
        "max_llm_calls": _env_int("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_MAX_LLM_CALLS", 5),
        # 打开后主链路只复用缓存，缺失的概念抽取提交到后台线程执行并写回缓存，下次请求即可命中。
        # 默认打开：同步抽取每篇约 18 秒，5 篇会让推荐请求超过 gunicorn 超时、worker 被杀。
        # 搜索重排链路无论该开关如何都走后台抽取，不在请求里同步等待 LLM。
        "allow_lazy_generation": _env_bool("RECOMMENDATION_CANDIDATE_CONCEPT_ENRICHMENT_ALLOW_LAZY_GENERATION", True),
        # 与 allow_lazy_generation 等价（任一开启即走后台抽取），保留以兼容旧配置。
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
    # 多样性选择中“语义重复”的判定阈值，基于 max(0, cos) 口径（旧 (cos+1)/2 口径的 0.75 等价于 0.5）。
    "semantic_similarity_penalty_threshold": float(
        _env_str("RECOMMENDATION_SEMANTIC_SIMILARITY_PENALTY_THRESHOLD", "0.5")
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
    # 主力模型上下文 256K（262,144 token），扣 16K 给输出。
    "prompt_max_input_tokens": _env_int("PROMPT_MAX_INPUT_TOKENS", 245760),
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
    # 最近 N 轮历史对话保留全文，更早的轮次把回答摘要压缩到 recent_turns_older_answer_tokens 以内。
    "recent_turns_recent_full_count": _env_int("PROMPT_RECENT_TURNS_RECENT_FULL_COUNT", 2),
    "recent_turns_older_answer_tokens": _env_int("PROMPT_RECENT_TURNS_OLDER_ANSWER_TOKENS", 120),
    "recent_turns_source_id_limit": _env_int("PROMPT_RECENT_TURNS_SOURCE_ID_LIMIT", 6),
    # 各 section 第一轮软预算（token）的显式覆盖；未设置的 section 沿用按比例计算的默认值。
    "section_targets": {
        section: value
        for section, value in {
            "rag_evidence": _env_optional_int("PROMPT_SECTION_TARGET_RAG_EVIDENCE"),
            "session_summary": _env_optional_int("PROMPT_SECTION_TARGET_SESSION_SUMMARY"),
            "recent_turns": _env_optional_int("PROMPT_SECTION_TARGET_RECENT_TURNS"),
            "user_memory": _env_optional_int("PROMPT_SECTION_TARGET_USER_MEMORY"),
            "agent_state": _env_optional_int("PROMPT_SECTION_TARGET_AGENT_STATE"),
        }.items()
        if value is not None
    },
    "rag_max_block_tokens": _env_int("PROMPT_RAG_MAX_BLOCK_TOKENS", 4000),
    "rag_sibling_context_target_tokens": _env_int("PROMPT_RAG_SIBLING_CONTEXT_TARGET_TOKENS", 350),
    "rag_section_context_target_tokens": _env_int("PROMPT_RAG_SECTION_CONTEXT_TARGET_TOKENS", 250),
    "rag_fallback_preview_tokens": _env_int("PROMPT_RAG_FALLBACK_PREVIEW_TOKENS", 160),
    "table_candidate_cell_limit": _env_int("PROMPT_TABLE_CANDIDATE_CELL_LIMIT", 8),
    "figure_preview_tokens": _env_int("PROMPT_FIGURE_PREVIEW_TOKENS", 220),
}

PAPER_EVIDENCE_RESEARCH_CONFIG: Dict[str, Any] = {
    # 证据池上限按 256K 上下文推导：输入预算 245,760 token，扣掉 30 张图（约 45K）与指令/状态（约 10K），
    # 证据文本约 190K token，按 3 字符/token 折算约 57 万字符。一篇论文约 8 万字符，整篇都能进池。
    "max_evidence_items": _env_int("RESEARCH_MAX_EVIDENCE_ITEMS", 300),
    "max_total_context_chars": _env_int("RESEARCH_MAX_TOTAL_CONTEXT_CHARS", 560_000),
}

ENHANCED_RETRIEVAL_CONFIG: Dict[str, Any] = {
    "recall_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RECALL_CANDIDATE_LIMIT", 30),
    # 融合结果不截断，按融合顺序取前 rerank_candidate_limit 条送 rerank，其余接在尾部；
    # 三路各召回 30 条时候选最多约 90 条，默认 90 即几乎总是全量 rerank（rerank 一次请求打完，90 条约 0.9s）。
    "rerank_candidate_limit": _env_int("ENHANCED_RETRIEVAL_RERANK_CANDIDATE_LIMIT", 90),
    # 上下文扩展的锚点数按「论文块数 × 扩展策略比例 × 本篇份额」逐次计算，再夹在 [min, limit] 之间；
    # limit 是上限（拿不到块数时直接用它），min 防止短论文或多篇分摊后只剩一两个锚点。
    "context_expansion_anchor_limit": _env_int("ENHANCED_RETRIEVAL_CONTEXT_EXPANSION_ANCHOR_LIMIT", 30),
    "context_expansion_anchor_min": _env_int("ENHANCED_RETRIEVAL_CONTEXT_EXPANSION_ANCHOR_MIN", 4),
    # rerank 排在末尾这一比例的块被视为 rerank 已判不相关，结构扩展不能再把它们拉回上下文。
    "context_expansion_reject_tail_ratio": float(_env_str("ENHANCED_RETRIEVAL_CONTEXT_EXPANSION_REJECT_TAIL_RATIO", "0.3")),
    "final_context_top_k": _env_int("ENHANCED_RETRIEVAL_FINAL_CONTEXT_TOP_K", 15),
    "max_final_context_top_k": _env_int("ENHANCED_RETRIEVAL_MAX_FINAL_CONTEXT_TOP_K", 30),
    "enable_context_expansion": _env_bool("ENHANCED_RETRIEVAL_ENABLE_CONTEXT_EXPANSION", True),
    # Multi-index 灰度开关分开控制 dense、sparse、生成式问题索引和旧 chunk 兜底，便于逐段回滚定位问题。
    "enable_multi_index_embedding": _env_bool("ENABLE_MULTI_INDEX_EMBEDDING", True),
    "enable_index_level_bm25": _env_bool("ENABLE_INDEX_LEVEL_BM25", True),
    "enable_generated_question_index": _env_bool("ENABLE_GENERATED_QUESTION_INDEX", True),
    "enable_chunk_level_retrieval_fallback": _env_bool("ENABLE_CHUNK_LEVEL_RETRIEVAL_FALLBACK", True),
    # 进程内最多缓存多少篇论文的关键词检索索引，超出后按最近最少使用淘汰，被淘汰的论文下次查询时重建。
    "collection_index_cache_max_entries": _env_int("RETRIEVAL_INDEX_CACHE_MAX_ENTRIES", 16),
    "context_budget_max_chars": _env_int("ENHANCED_RETRIEVAL_CONTEXT_BUDGET_MAX_CHARS", 24000),
    "retrieval_candidate_max_blocks": _env_int("RETRIEVAL_CANDIDATE_MAX_BLOCKS", 60),
    "retrieval_candidate_max_tokens_soft": _env_int("RETRIEVAL_CANDIDATE_MAX_TOKENS_SOFT", 90000),
    "retrieval_candidate_token_chars_per_token": float(_env_str("RETRIEVAL_CANDIDATE_TOKEN_CHARS_PER_TOKEN", "3.0")),
    "sample_limit": _env_int("ENHANCED_RETRIEVAL_SAMPLE_LIMIT", 24),
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
    "sanitize_trace_slug_max_length": _env_int("ENHANCED_RETRIEVAL_SANITIZE_TRACE_SLUG_MAX_LENGTH", 40),
    "enable_table_structured_route": _env_bool("ENHANCED_RETRIEVAL_ENABLE_TABLE_STRUCTURED_ROUTE", True),
    "table_structured_candidate_limit": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CANDIDATE_LIMIT", 8),
    "table_structured_match_score_floor": float(_env_str("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_MATCH_SCORE_FLOOR", "0.22")),
    # 表格 route 在不确定时会把局部表格上下文交给 LLM；这些参数限制上下文体积，避免规则层为了覆盖长表继续膨胀。
    "table_structured_context_short_table_cell_limit": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CONTEXT_SHORT_TABLE_CELL_LIMIT", 48),
    "table_structured_context_max_rows": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CONTEXT_MAX_ROWS", 12),
    "table_structured_context_focus_window": _env_int("ENHANCED_RETRIEVAL_TABLE_STRUCTURED_CONTEXT_FOCUS_WINDOW", 2),
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
    # 原问向量路由的查询侧任务说明（DashScope 多模态向量接口的 instruct 参数，官方建议用英文）；
    # 文档侧建索引时不带说明，留空则原问与其它路由一样不带说明。
    "original_query_embedding_instruct": _env_str(
        "RETRIEVAL_ORIGINAL_QUERY_EMBEDDING_INSTRUCT",
        "Given a question about an academic paper, retrieve passages from the paper that answer the question.",
    ),
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
    "batch_size": _env_int("RERANK_BATCH_SIZE", 8),
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
    "llm_rerank_max_doc_chars": RERANK_CONFIG["max_doc_chars"],
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
    "qwen_rerank_compress_model_name": _env_str("QWEN_RERANK_COMPRESS_MODEL_NAME", "qwen3.6-plus"),
    "qwen_rerank_compress_enable_thinking": _env_bool("QWEN_RERANK_COMPRESS_ENABLE_THINKING", False),
    # 建库期（rerank 压缩、检索索引 summary/question）单次 LLM 请求的超时秒数；正常 3~5 秒返回，
    # 不设时 SDK 默认等 600 秒，一个卡住的请求会拖住整篇建库。超时后由 SDK 自动重试（默认 2 次）。
    "qwen_index_build_timeout_seconds": float(_env_str("QWEN_INDEX_BUILD_TIMEOUT_SECONDS", "60")),
    "openai_chat_temperature": float(_env_str("OPENAI_CHAT_TEMPERATURE", "0.7")),
    "openai_chat_max_tokens": _env_int("OPENAI_CHAT_MAX_TOKENS", 512),
    "rewrite_query_max_queries_default": _env_int("REWRITE_QUERY_MAX_QUERIES_DEFAULT", 3),
    # 原问改写的采样温度：低于默认值让改写贴着原问，但不设为 0，保留一点随机性。
    "query_rewrite_temperature": float(_env_str("QUERY_REWRITE_TEMPERATURE", "0.3")),
    # 每次生成的完整上下文与回答落盘到 05-generation-results，仅供调试复盘；
    # 文件含用户问题和回答原文且没有清理机制，默认关闭，排查时再临时打开。
    "save_generation_results": _env_bool("SAVE_GENERATION_RESULTS", False),
    # Responses API 多模态请求按百炼官方限制设置（https://help.aliyun.com/zh/model-studio/vision）：
    # dashscope.aliyuncs.com 域名请求体上限 16 MiB（工作空间域名 64MB），Base64 传图单次最多 250 张；
    # 软上限预留 1 MiB 给 JSON 包装和文本，单图目标字节会按剩余请求体预算自适应压缩。
    "qwen_responses_max_image_count": _env_int("QWEN_RESPONSES_MAX_IMAGE_COUNT", 250),
    "qwen_responses_request_body_limit_bytes": _env_int("QWEN_RESPONSES_REQUEST_BODY_LIMIT_BYTES", 16 * 1024 * 1024),
    "qwen_responses_request_body_soft_limit_bytes": _env_int("QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES", 15 * 1024 * 1024),
    "qwen_responses_image_target_bytes": _env_int("QWEN_RESPONSES_IMAGE_TARGET_BYTES", 1_500_000),
    # 任务到模型角色的路由表只保留在配置层，业务代码只负责读取，不再硬编码模型名。
    "task_model_roles": {
        "intent_recognition": "large",  # 解析用户原始输入以识别高层意图
        "agent_request_contextualization": "large",  # 借最近对话把 Agent 追问改写成可独立理解的请求
        "intent_routing": "large",  # 决定将意图路由到哪个子 agent 或处理流水线
        "search_spec_parse": "large",  # 将自然语言搜索请求转换为结构化的搜索规范
        "preference_action_parse": "large",  # 解析用户的喜欢/不喜欢/收藏等偏好操作
        "query_rewrite": "large",  # 把问题改写成贴合论文术语的英文检索问句（具体化指代、拆分复合问题）
        "query_translation": "large",  # 把用户问题译成英文，供原问向量路由检索英文论文
        "rerank_query": "large",  # 为重排阶段生成或调整查询
        "prompt_context_compaction": "large",  # 对长证据块做语义压缩，结果仍需最终回答模型基于证据使用
        "hyde_generation": "large",  # HYDE 风格的伪文档生成（用于查询扩展）
        "paper_qa_final_answer": "large",  # 为论文问答生成最终高质量答案
        "paper_summary": "large",  # 生成论文的综合性摘要
        "paper_detail": "large",  # 生成方法或技术细节的详尽说明
        "paper_qa": "large",  # 处理关于具体论文的问答
        "paper_fast_qa": "large",  # 快速问答：整篇论文全文送入模型直接作答
        "recommendation_generation": "large",  # 生成个性化推荐文本或推荐理由
        "general_generation": "large",  # 用于复杂自然语言输出的一般大模型生成
        "research_claim_extraction": "large",  # 研究引擎把草稿拆成可校验主张并绑定证据需求
        "research_claim_verification": "large",  # 研究引擎逐条对照引用证据核对主张
        "abstract_translation": "small",  # 论文卡片上把英文摘要流式翻译成中文，首字时延优先
        "default": "large",  # 当无特定任务映射时的默认模型角色
    },
}


def get_chunking_runtime_config() -> Dict[str, Any]:
    return dict(CHUNKING_CONFIG)


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
    from auth.context import authenticated_user_id

    # 请求上下文的身份先于演示配置，避免省略 user_id 时读写到共享的 local_user 数据。
    actor_id = authenticated_user_id()
    if actor_id:
        return actor_id
    return get_legacy_default_user_id()


def get_legacy_default_user_id() -> str:
    """仅供本地兼容常量使用，不能在模块导入时捕获第一个 JWT 请求的身份。"""
    return str(USER_CONFIG["default_user_id"] or "local_user").strip() or "local_user"


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


