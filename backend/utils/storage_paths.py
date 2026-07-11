"""本地持久化路径的统一解析规则。"""

from __future__ import annotations

from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
BACKEND_LOADED_DOCS_ROOT = BACKEND_ROOT / "01-loaded-docs"
BACKEND_EMBEDDED_DOCS_ROOT = BACKEND_ROOT / "02-embedded-docs"
BACKEND_DATA_ROOT = BACKEND_ROOT / "06-database"
LEGACY_DATABASE_ROOT = REPO_ROOT / "06-database"
BACKEND_VECTOR_STORE_ROOT = BACKEND_ROOT / "03-vector-store"
BACKEND_GENERATION_RESULTS_ROOT = BACKEND_ROOT / "05-generation-results"
BACKEND_DAILY_ARXIV_ROOT = BACKEND_ROOT / "06-daily-arxiv-paper"

LEGACY_BACKEND_ARTIFACT_ROOTS = {
    "01-loaded-docs": BACKEND_LOADED_DOCS_ROOT,
    "02-embedded-docs": BACKEND_EMBEDDED_DOCS_ROOT,
    "03-vector-store": BACKEND_VECTOR_STORE_ROOT,
    "05-generation-results": BACKEND_GENERATION_RESULTS_ROOT,
    "06-daily-arxiv-paper": BACKEND_DAILY_ARXIV_ROOT,
}


class StoragePathConfigurationError(ValueError):
    """持久化路径违反仓库目录约束时抛出。"""


def is_legacy_database_path(path: Path) -> bool:
    """判断路径是否落在已废弃的仓库根目录数据库目录内。"""
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_legacy_root = LEGACY_DATABASE_ROOT.resolve(strict=False)
    try:
        return resolved_path.is_relative_to(resolved_legacy_root)
    except ValueError:
        return False


def resolve_storage_path(
    raw_path: str | Path | None,
    *,
    default_path: str | Path,
    option_name: str,
) -> str:
    """将持久化路径固定到后端根目录，并阻止历史错误目录重新被写入。"""
    selected_path = Path(str(raw_path).strip()).expanduser() if raw_path else Path(default_path).expanduser()
    if not selected_path.is_absolute():
        # 运行方式不应影响持久化位置；相对配置统一以 backend 为基准解释。
        selected_path = BACKEND_ROOT / selected_path

    resolved_path = selected_path.resolve(strict=False)
    if is_legacy_database_path(resolved_path):
        raise StoragePathConfigurationError(
            f"{option_name} 不能指向已废弃的仓库根目录 06-database：{resolved_path}。"
            f"请改用 {BACKEND_DATA_ROOT} 或仓库外的绝对路径。"
        )
    return str(resolved_path)


def resolve_backend_artifact_path(path: str | Path, *, option_name: str) -> Path:
    """将后端本地产物目录固定到 backend 下，避免启动目录决定写入位置。"""
    selected_path = Path(path).expanduser()
    if not selected_path.is_absolute():
        # 这些产物目录属于后端运行资产；即使从仓库根目录启动，也不能写回仓库根目录。
        selected_path = BACKEND_ROOT / selected_path

    resolved_path = selected_path.resolve(strict=False)
    for legacy_name, backend_path in LEGACY_BACKEND_ARTIFACT_ROOTS.items():
        legacy_root = (REPO_ROOT / legacy_name).resolve(strict=False)
        try:
            is_legacy_path = resolved_path.is_relative_to(legacy_root)
        except ValueError:
            is_legacy_path = False
        if is_legacy_path:
            return backend_path.resolve(strict=False) / resolved_path.relative_to(legacy_root)
    return resolved_path
