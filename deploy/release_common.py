"""发布包与服务器共用的目录约定；不导入业务模块或生产配置。

本模块同时被两端使用：
- CI 打包端：package_release.py 用它校验运行环境、过滤源码归档、计算哈希；
- 服务器安装端：apply_release.py 用它校验并解压发布包、核对 release.json 清单。

publish_ssh.sh 会把本文件和 apply_release.py 一起 scp 到服务器的 incoming 目录，
服务器上只有系统 python3、没有项目虚拟环境，因此这里只能依赖标准库，
也不能 import 任何业务代码或读取生产配置。
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


# 跨版本持久化的运行时目录（相对项目根目录）。
# 这些目录保存向量库、索引、数据库、日志等"用户数据"，不属于代码：
# - 打包时：即使它们出现在 Git 源码归档中也会被剔除，不进入发布包；
# - 安装时：每个 releases/<id>/ 下的同名路径都会被替换成指向 shared/<路径> 的符号链接，
#   这样切换或回退代码版本时数据保持不变。
SHARED_DIRECTORIES = (
    "backend/01-loaded-docs",
    "backend/02-embedded-docs", "backend/02-retrieval-indexes", "backend/02-sparse-indexes",
    "backend/03-docling-assets", "backend/03-vector-store",
    "backend/05-generation-results", "backend/06-daily-arxiv-paper", "backend/06-database",
    "backend/06-evaluation-result", "backend/data", "backend/temp", "temp", "logs",
)
# 构建机与生产服务器必须完全一致的运行环境。发布包内含预编译 wheel（torch、milvus-lite 等原生扩展），
# 操作系统、CPU 架构或 Python 小版本不同都会导致 wheel 无法安装或运行时崩溃。
EXPECTED_RUNTIME = {"system": "Linux", "distribution": "debian", "version": "12", "python": "3.11", "machine": "x86_64"}


class ReleaseError(RuntimeError):
    """校验或发布失败时使用稳定错误，不输出环境变量和凭据。"""


def sha256_file(path: Path) -> str:
    """按 1 MiB 分块计算文件 SHA-256，避免把数百 MB 的发布包一次读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        # iter(callable, sentinel)：反复调用 read，直到返回空字节串（文件末尾）为止。
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_commit(commit: str) -> str:
    """只接受 40 位小写十六进制的完整 Git SHA，拒绝分支名、短 SHA 等可变引用。"""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ReleaseError("发布提交必须是完整的 Git SHA。")
    return commit


def verify_runtime() -> None:
    """检查当前机器是否为 Debian 12 / x86_64 / Python 3.11，不一致则拒绝打包或安装。"""
    release = {}
    # /etc/os-release 是 KEY=VALUE 格式（值可能带引号），用来识别发行版及版本号。
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
    """判断归档中的某个路径是否属于"运行时数据或敏感配置"，这类路径绝不能进入发布包。"""
    path = PurePosixPath(name)
    # 路径本身是共享目录，或位于某个共享目录之下。
    if any(path == PurePosixPath(item) or PurePosixPath(item) in path.parents for item in SHARED_DIRECTORIES):
        return True
    # 即使未来误把运行配置加入 Git，也不能让代码发布包覆盖服务器上的凭据或虚拟环境。
    return (
        # 版本库、虚拟环境、AI 工具配置、字节码缓存、前端依赖目录。
        any(part in {".git", ".venv", ".conda", ".codex", ".claude", "__pycache__", "node_modules"} for part in path.parts)
        # API 密钥文件。
        or path == PurePosixPath("backend/config/api_keys.json")
        # 任何 .env* 文件（.env.production 等），只放行示例文件 .env.example。
        or (path.name.startswith(".env") and path.name != ".env.example")
        # JWT 签名密钥，以及 SQLite 数据库及其 WAL/SHM/journal 附属文件。
        or path.name.endswith((".jwt-secret", ".db", ".sqlite", ".sqlite3",
                               ".db-wal", ".db-shm", ".db-journal", ".sqlite-wal", ".sqlite-shm",
                               ".sqlite3-wal", ".sqlite3-shm", ".sqlite3-journal"))
        # 即使旧 release 中残留了游标文件，也不能把它重新打进发布包。
        or (path.name.startswith("sync_arxiv_oai_since_last_run")
            and path.suffix in {".state", ".json"})
    )


def checked_members(archive: tarfile.TarFile, *, source_archive: bool = False) -> list[tarfile.TarInfo]:
    """在解压前逐项审查 tar 目录表，返回允许写入磁盘的成员列表。

    source_archive=True 表示正在处理 CI 端的 git archive 源码包：遇到运行时路径时静默跳过；
    source_archive=False 表示正在处理服务器端收到的发布包：发布包本不该含有运行时路径，
    一旦出现就说明打包异常或被篡改，直接报错。
    """
    members = []
    names = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        # 防止路径穿越（Zip Slip / Tar Slip）：绝对路径、".."、Windows 反斜杠、空路径都可能写到目标目录之外。
        if path.is_absolute() or ".." in path.parts or "\\" in member.name or not path.parts:
            raise ReleaseError("发布包包含不安全的路径。")
        # 只允许普通文件和目录；符号链接/硬链接可指向任意位置，设备文件更不应出现。
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
    """安全解压 tar 包到一个全新的目录；目标目录已存在时报错（exist_ok=False），绝不覆盖旧版本。"""
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
                # "xb" 模式：文件已存在则失败，作为重复路径检查之外的第二道保险。
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                # 不信任归档里的原始权限位（可能带 setuid 等），只保留"是否可执行"这一信息。
                target.chmod(0o755 if member.mode & 0o111 else 0o644)


def validate_release(directory: Path, commit: str) -> dict:
    """校验解压后的发布目录：清单版本、提交、运行环境、依赖锁哈希和必要文件都必须匹配。

    返回 release.json 的内容，供调用方读取 dependency_hash 等字段。
    """
    manifest = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ReleaseError("发布清单版本不受支持。")
    # 防止把 A 提交的包当作 B 提交部署，或把为其他环境构建的包装到本机。
    if manifest.get("commit") != validate_commit(commit) or manifest.get("runtime") != EXPECTED_RUNTIME:
        raise ReleaseError("发布包版本或运行环境与请求不一致。")
    # 依赖锁文件的哈希写在清单中，用于确认锁文件未被替换，也作为服务器端 venv 缓存的键。
    lock = directory / ".release/requirements.lock"
    if manifest.get("dependency_hash") != sha256_file(lock):
        raise ReleaseError("依赖锁文件校验失败。")
    # 后端入口和前端构建产物缺一不可。
    for name in ("backend/main.py", "frontend/dist/index.html"):
        if not (directory / name).is_file():
            raise ReleaseError(f"发布包缺少必要文件：{name}")
    # 服务器不联网安装依赖，必须携带离线 wheel 目录。
    if not (directory / ".release/wheels").is_dir():
        raise ReleaseError("发布包缺少离线依赖。")
    return manifest
