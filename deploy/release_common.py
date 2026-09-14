"""发布包与服务器共用的目录约定；不导入业务模块或生产配置。"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


SHARED_DIRECTORIES = (
    "backend/01-loaded-docs", "backend/01-chunked-docs",
    "backend/02-embedded-docs", "backend/02-retrieval-indexes", "backend/02-sparse-indexes",
    "backend/03-docling-assets", "backend/03-vector-store", "backend/04-search-results",
    "backend/05-generation-results", "backend/06-daily-arxiv-paper", "backend/06-database",
    "backend/06-evaluation-result", "backend/data", "backend/temp", "temp", "logs",
)
EXPECTED_RUNTIME = {"system": "Linux", "distribution": "debian", "version": "12", "python": "3.11", "machine": "x86_64"}


class ReleaseError(RuntimeError):
    """校验或发布失败时使用稳定错误，不输出环境变量和凭据。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_commit(commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseError("发布提交必须是完整的 Git SHA。")
    return commit


def verify_runtime() -> None:
    release = {}
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        release = dict(line.split("=", 1) for line in os_release.read_text().splitlines() if "=" in line)
    actual = {
        "system": platform.system(), "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "distribution": release.get("ID", "").strip('"'),
        "version": release.get("VERSION_ID", "").strip('"'),
    }
    if actual != EXPECTED_RUNTIME:
        raise ReleaseError("当前发布包要求 Debian 12 x86_64 / Python 3.11；请使构建与服务器环境一致。")


def is_runtime_path(name: str) -> bool:
    path = PurePosixPath(name)
    if any(path == PurePosixPath(item) or PurePosixPath(item) in path.parents for item in SHARED_DIRECTORIES):
        return True
    # 即使未来误把运行配置加入 Git，也不能让代码发布包覆盖服务器上的凭据或虚拟环境。
    return (
        any(part in {".git", ".venv", ".conda", ".codex", ".claude", "__pycache__", "node_modules"} for part in path.parts)
        or path == PurePosixPath("backend/config/api_keys.json")
        or (path.name.startswith(".env") and path.name != ".env.example")
        or path.name.endswith((".jwt-secret", ".db", ".sqlite", ".sqlite3",
                               ".db-wal", ".db-shm", ".db-journal", ".sqlite-wal", ".sqlite-shm",
                               ".sqlite3-wal", ".sqlite3-shm", ".sqlite3-journal"))
        # 即使旧 release 中残留了游标文件，也不能把它重新打进发布包。
        or (path.name.startswith("sync_arxiv_oai_since_last_run")
            and path.suffix in {".state", ".json"})
    )


def checked_members(archive: tarfile.TarFile, *, source_archive: bool = False) -> list[tarfile.TarInfo]:
    members = []
    names = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or "\\" in member.name or not path.parts:
            raise ReleaseError("发布包包含不安全的路径。")
        if not (member.isfile() or member.isdir()):
            raise ReleaseError("发布包不能包含符号链接、硬链接或设备文件。")
        # tar 可用 a/./b 等别名描述同一路径；按规范化路径去重，避免先校验后覆盖。
        normalized = path.as_posix()
        if normalized != member.name.rstrip("/") or ":" in member.name:
            raise ReleaseError("发布包路径必须规范化。")
        if normalized in names:
            raise ReleaseError("发布包包含重复路径。")
        names.add(normalized)
        if is_runtime_path(member.name):
            if source_archive:
                continue
            raise ReleaseError("发布包不能携带持久化目录或生产配置。")
        members.append(member)
    return members


def extract_checked(archive_path: Path, destination: Path, *, source_archive: bool = False) -> None:
    # 先检查完整目录表再写文件；只接收普通文件，避免归档链接穿透至 shared 数据目录。
    with tarfile.open(archive_path) as archive:
        members = checked_members(archive, source_archive=source_archive)
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)


def validate_release(directory: Path, commit: str) -> dict:
    manifest = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ReleaseError("发布清单版本不受支持。")
    if manifest.get("commit") != validate_commit(commit) or manifest.get("runtime") != EXPECTED_RUNTIME:
        raise ReleaseError("发布包版本或运行环境与请求不一致。")
    lock = directory / ".release/requirements.lock"
    if manifest.get("dependency_hash") != sha256_file(lock):
        raise ReleaseError("依赖锁文件校验失败。")
    for name in ("backend/main.py", "new_frontend/dist/index.html"):
        if not (directory / name).is_file():
            raise ReleaseError(f"发布包缺少必要文件：{name}")
    if not (directory / ".release/wheels").is_dir():
        raise ReleaseError("发布包缺少离线依赖。")
    return manifest
