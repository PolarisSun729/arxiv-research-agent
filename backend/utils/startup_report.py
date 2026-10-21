"""后端启动时打印一份分组的关键配置快照，便于本地调试确认实际生效的路径与开关。

只输出路径、开关、模型名等非敏感值；密钥一律只显示“已配置/未配置”，URL 中的账号密码会被遮蔽。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from utils import config as cfg
from utils.storage_paths import (
    ARXIV_OAI_SYNC_STATE_FILE,
    BACKEND_DATA_ROOT,
    BACKEND_DOCLING_ASSETS_ROOT,
    BACKEND_ROOT,
)

Section = Tuple[str, List[Tuple[str, Any]]]


def _path(value: Any) -> str:
    """路径后附带是否存在，排查“读错库/目录没建”时不用再手动去看磁盘。"""
    if not value:
        return "<未设置>"
    text = str(value)
    if text == "-":
        return "- (stdout)"
    return f"{text}  [{'存在' if Path(text).exists() else '不存在'}]"


def _secret(value: Any) -> str:
    # 标签不能含 api_key 等字样：全局日志脱敏会把整行值替换掉，看不到是否已配置。
    return "已配置" if str(value or "").strip() else "未配置"


def _mask_url(value: Any) -> str:
    """遮蔽 URL 中的 user:password，Redis/Milvus 地址可能携带凭据。"""
    text = str(value or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    if not parts.password and not parts.username:
        return text
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


def _join(values: Iterable[Any]) -> str:
    items = [str(item) for item in values]
    return ", ".join(items) if items else "<空>"


def build_startup_sections(
    *,
    load_mode: str,
    auth_mode: str,
    jwt_settings: Any = None,
    api_key_count: int = 0,
    allowed_origins: Sequence[str] = (),
    ip_settings: Any = None,
    rate_limit_settings: Any = None,
) -> List[Section]:
    """按主题收集启动配置；只读取已解析好的配置对象，不触发任何服务初始化。"""
    logging_cfg = cfg.BACKEND_LOGGING_CONFIG
    embedding = cfg.EMBEDDING_CONFIG
    rerank = cfg.RERANK_CONFIG
    generation = cfg.GENERATION_CONFIG
    retrieval = cfg.ENHANCED_RETRIEVAL_CONFIG
    build_cache = cfg.PAPER_QA_BUILD_CACHE_CONFIG

    audit_enabled = os.getenv("AUDIT_LOG_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off", ""}
    audit_file = os.getenv("AUDIT_LOG_FILE", "logs/audit.log").strip()
    if audit_file and audit_file != "-":
        audit_file = str((cfg.REPO_ROOT / audit_file).resolve())

    security: List[Tuple[str, Any]] = [("auth_mode", auth_mode)]
    if jwt_settings is not None:
        security += [
            ("auth_database_path", _path(jwt_settings.database_path)),
            ("jwt_issuer / audience", f"{jwt_settings.issuer} / {jwt_settings.audience}"),
            ("jwt_expire_minutes", jwt_settings.expire_seconds // 60),
            ("registration_enabled", jwt_settings.registration_enabled),
        ]
    else:
        security.append(("api_key_count", api_key_count))
    security += [
        ("allowed_origins", _join(allowed_origins)),
        ("ip_filter_mode", getattr(ip_settings, "mode", "<未知>")),
        ("trusted_proxies", _join(getattr(ip_settings, "trusted_proxies", ()))),
        ("rate_limit_storage", _mask_url(getattr(rate_limit_settings, "storage_uri", "<未知>"))),
        ("audit_log", _path(audit_file) if audit_enabled else "已关闭"),
    ]

    return [
        (
            "运行环境",
            [
                ("python", f"{sys.executable} ({sys.version.split()[0]})"),
                ("pid", os.getpid()),
                ("cwd", os.getcwd()),
                ("backend_root", BACKEND_ROOT),
                (".env (repo)", _path(cfg.REPO_ROOT / ".env")),
                (".env (backend)", _path(cfg.REPO_ROOT / "backend" / ".env")),
                ("load_mode", load_mode),
                ("arxiv_data_source", cfg.CORE_CONFIG["arxiv_data_source"]),
                ("arxiv_proxy_url", _mask_url(cfg.CORE_CONFIG["arxiv_proxy_url"])),
                ("default_user_id", cfg.USER_CONFIG["default_user_id"]),
            ],
        ),
        (
            "数据库与存储",
            [
                ("sqlite (main)", _path(cfg.SQLITE_CONFIG["database_path"])),
                ("sqlite (arxiv_oai)", _path(cfg.OAI_SQLITE_CONFIG["database_path"])),
                ("milvus_uri", _mask_url(cfg.MILVUS_CONFIG["uri"])),
                ("agent_checkpoint_backend", cfg.AGENT_RUNTIME_CHECKPOINT_CONFIG["backend"]),
                ("data_root", _path(BACKEND_DATA_ROOT)),
                ("docling_assets_root", _path(BACKEND_DOCLING_ASSETS_ROOT)),
                ("loaded_docs_dir", _path(BACKEND_ROOT / "01-loaded-docs")),
                ("retrieval_index_dir", _path(BACKEND_ROOT / "02-retrieval-indexes")),
                ("sparse_index_dir", _path(BACKEND_ROOT / "02-sparse-indexes")),
                ("oai_sync_state_file", _path(ARXIV_OAI_SYNC_STATE_FILE)),
                (
                    "qa_build_cache",
                    f"{'启用' if build_cache['enabled'] else '关闭'}  {_path(build_cache['root_dir'])}",
                ),
                ("retrieval_trace_dir", _path(rerank["trace_export_dir"]) if rerank["trace_export_enabled"] else "已关闭"),
                ("request_trace_dir", f"{logging_cfg['request_trace']}  {_path(logging_cfg['request_trace_dir'])}"),
            ],
        ),
        ("认证与安全", security),
        (
            "模型",
            [
                ("embedding", f"{embedding['provider']} / {embedding['model_name']} (dim={embedding['dimension']})"),
                ("embedding_base_url", embedding["base_url"]),
                ("embedding_credential", _secret(embedding["api_key"] or embedding["dashscope_api_key"])),
                ("rerank", f"{rerank['provider']} / {rerank['model_name']} (fallback_local={rerank['fallback_local']})"),
                ("rerank_base_url", rerank["base_url"]),
                ("rerank_credential", _secret(rerank["api_key"] or rerank["dashscope_api_key"])),
                ("llm_base_url", generation["qwen_base_url"]),
                ("llm_small / large", f"{generation['small_qwen_model_name']} / {generation['large_qwen_model_name']}"),
                ("qwen_credential", _secret(generation["qwen_api_key"])),
                ("openai / deepseek credential", f"{_secret(generation['openai_api_key'])} / {_secret(generation['deepseek_api_key'])}"),
            ],
        ),
        (
            "检索与问答",
            [
                ("planner_mode", cfg.AGENT_PLANNER_CONFIG["planner_runtime_mode"]),
                ("query_rewrite / hyde", f"{rerank['enable_query_rewrite']} / {rerank['enable_hyde']}"),
                ("keyword_search", f"{rerank['enable_keyword_search']} ({rerank['keyword_backend']})"),
                ("llm_rerank", rerank["enable_llm_rerank"]),
                ("final_context_top_k", retrieval["final_context_top_k"]),
                ("prompt_max_input_tokens", cfg.PROMPT_CONTEXT_CONFIG["prompt_max_input_tokens"]),
                ("short_term_memory", cfg.MEMORY_RUNTIME_CONFIG["enable_short_term_memory"]),
                ("user_research_profile", cfg.MEMORY_RUNTIME_CONFIG["enable_user_research_profile"]),
            ],
        ),
        (
            "日志",
            [
                ("level", logging_cfg["level"]),
                ("debug_loggers", _join(logging_cfg["debug_loggers"])),
                ("access_log", logging_cfg["access_log"]),
                ("full_io", logging_cfg["full_io"]),
            ],
        ),
    ]


def format_startup_report(sections: Sequence[Section]) -> str:
    width = max((len(key) for _, items in sections for key, _ in items), default=0)
    lines = ["", "=" * 24 + " 后端启动配置 " + "=" * 24]
    for title, items in sections:
        lines.append(f"[{title}]")
        lines.extend(f"  {key.ljust(width)} : {value}" for key, value in items)
    lines.append("=" * 62)
    return "\n".join(lines)


def log_startup_report(logger: logging.Logger, **kwargs: Any) -> None:
    """一次 logger 调用输出整块多行文本，避免与并发启动日志交错。"""
    if not logger.isEnabledFor(logging.INFO):
        return
    logger.info("backend.startup_config%s", format_startup_report(build_startup_sections(**kwargs)))
