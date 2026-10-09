"""本地持久化路径的统一解析规则。"""

from __future__ import annotations

from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DATA_ROOT = BACKEND_ROOT / "06-database"
# OAI 增量游标和最近一次运行摘要是跨版本保留的运行时状态。
# 目录位于 backend/data：生产发布时该目录会由 deploy/apply_release.py 链接到 shared/backend/data。
BACKEND_ARXIV_OAI_SYNC_ROOT = BACKEND_ROOT / "data" / "arxiv-oai-sync"
ARXIV_OAI_SYNC_STATE_FILE = BACKEND_ARXIV_OAI_SYNC_ROOT / "sync_arxiv_oai_since_last_run.state"
ARXIV_OAI_SYNC_META_FILE = BACKEND_ARXIV_OAI_SYNC_ROOT / "sync_arxiv_oai_since_last_run.meta.json"


def resolve_storage_path(
    raw_path: str | Path | None,
    *,
    default_path: str | Path,
) -> str:
    """将持久化路径固定到后端根目录，避免启动目录决定写入位置。"""
    selected_path = Path(str(raw_path).strip()).expanduser() if raw_path else Path(default_path).expanduser()
    if not selected_path.is_absolute():
        # 运行方式不应影响持久化位置；相对配置统一以 backend 为基准解释。
        selected_path = BACKEND_ROOT / selected_path
    return str(selected_path.resolve(strict=False))


def resolve_backend_artifact_path(path: str | Path) -> Path:
    """将后端本地产物目录固定到 backend 下，避免启动目录决定写入位置。"""
    selected_path = Path(path).expanduser()
    if not selected_path.is_absolute():
        # 这些产物目录属于后端运行资产；即使从仓库根目录启动，也不能写回仓库根目录。
        selected_path = BACKEND_ROOT / selected_path
    return selected_path.resolve(strict=False)
