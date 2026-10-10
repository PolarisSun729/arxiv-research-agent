"""项目运行环境体检：逐项检查本机能否运行测试和真实服务，并给出每项的原因与修复建议。

用途：
    新机器配环境、依赖升级后或排查“为什么跑不起来”时运行。
    basic 模式也是 check_quality.py 的 doctor-basic 阶段（包含在 all / ci 目标中）。

用法（任意目录运行均可）：
    python scripts/doctor.py                     # 等同 basic：只检查本地环境，不访问网络
    python scripts/doctor.py full                # basic + 真实连接 Milvus、arXiv API、arXiv OAI
    python scripts/doctor.py full --check-paid   # 再加上真实调用 Embedding / LLM / Rerank（可能产生费用）
    python scripts/doctor.py --json              # 以 JSON 输出，便于 CI 或其他工具解析

basic 检查项（见 BASIC_CHECKS）：
    Python 版本、后端关键包、Node/npm、前端依赖、本地数据目录、后端配置加载、
    SQLite 临时写入、API Key 环境变量、FastAPI lazy app 创建、PDF 解析依赖。
full 额外检查项（见 run_full_checks）：
    Milvus 连接、arXiv API、arXiv OAI；Embedding / LLM / Rerank 默认 SKIP，需 --check-paid。

每项结果的状态：
    PASS 正常；WARN 可以继续但建议处理；FAIL 需要修复；SKIP 主动跳过。
    同时标注“是否必需”“是否影响默认离线测试”“是否影响真实运行”，帮助判断优先级。

注意：
    - 体检会读取与后端相同的配置（导入 backend/utils/config.py，即会加载 .env）；
    - 只报告密钥是否已配置，任何情况下都不打印密钥值；
    - 退出码：存在任一 FAIL 返回 1，否则返回 0（WARN / SKIP 不影响退出码）。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
FRONTEND_ROOT = REPO_ROOT / "frontend"

# 必须先把 backend 加入模块搜索路径，Doctor 作为独立脚本运行时才能复用后端路径规则。
for path in (REPO_ROOT, BACKEND_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@dataclass(frozen=True)
class DoctorResult:
    """单项体检结果。

    name：检查项名称；status：PASS / WARN / FAIL / SKIP；reason：判断依据；fix：修复建议；
    required：是否为必需项；affects_default_tests：是否影响默认离线测试；
    affects_real_runtime：是否影响真实运行；mode：所属模式 basic / full。
    """

    name: str
    status: str
    reason: str
    fix: str
    required: bool
    affects_default_tests: bool
    affects_real_runtime: bool
    mode: str


def _result(
    name: str,
    status: str,
    reason: str,
    fix: str = "无需处理。",
    *,
    required: bool,
    affects_default_tests: bool,
    affects_real_runtime: bool,
    mode: str,
) -> DoctorResult:
    """构造 DoctorResult 的便捷函数；除名称、状态、原因外的字段强制使用关键字参数，避免传错位置。"""
    return DoctorResult(
        name=name,
        status=status,
        reason=reason,
        fix=fix,
        required=required,
        affects_default_tests=affects_default_tests,
        affects_real_runtime=affects_real_runtime,
        mode=mode,
    )


def _safe_check(
    name: str,
    check: Callable[[], DoctorResult],
    *,
    required: bool,
    affects_default_tests: bool,
    affects_real_runtime: bool,
    mode: str,
) -> DoctorResult:
    """执行单项检查；检查函数抛出任何异常都转成一条 FAIL 结果，保证一项出错不影响其他项。"""
    try:
        return check()
    except Exception as exc:
        # doctor 的职责是生成可读体检报告，不把底层 traceback 直接抛给使用者。
        return _result(
            name,
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            "按失败项单独复现；若是依赖缺失，先安装 requirements / npm 依赖。",
            required=required,
            affects_default_tests=affects_default_tests,
            affects_real_runtime=affects_real_runtime,
            mode=mode,
        )


def _run_command(command: list[str], cwd: Path, timeout: int = 15) -> tuple[int, str]:
    """运行外部命令，返回 (退出码, 合并后的 stdout+stderr)；超时会抛出异常，由 _safe_check 兜底。"""
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip()


def _find_executable(name: str) -> str | None:
    """在 PATH 中查找可执行文件；Windows 上依次尝试 .cmd/.exe/.bat 后缀（npm 在 Windows 上是 npm.cmd）。"""
    suffixes = [".cmd", ".exe", ".bat", ""] if os.name == "nt" else [""]
    for suffix in suffixes:
        candidate = name + suffix
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            path = Path(directory) / candidate
            if path.exists():
                return str(path)
    return None


def _importable(module_name: str) -> bool:
    """只查找模块是否存在而不真正导入，避免加载 docling 等重型包拖慢体检。"""
    return importlib.util.find_spec(module_name) is not None


def _format_version_tuple(version_info: Any) -> str:
    """把 sys.version_info 这类元组格式化为 "3.11.9"。"""
    return ".".join(str(part) for part in version_info[:3])


def _is_env_set(name: str) -> bool:
    """环境变量存在且去掉空白后非空才算已设置。"""
    return bool(os.getenv(name, "").strip())


def _mask_state(env_names: list[str], configured_value: str = "") -> tuple[str, str]:
    """判断一组候选环境变量的配置状态，只返回状态和说明，不返回密钥值。

    env_names 中任意一个显式设置即 PASS；都没设置但配置层有值（如从 ALIYUN_API_KEY 兜底）为 WARN；
    完全没有值也是 WARN。
    """
    explicit = [name for name in env_names if _is_env_set(name)]
    if explicit:
        return "PASS", f"已显式配置环境变量：{', '.join(explicit)}。"
    if configured_value:
        return "WARN", f"未显式设置 {', '.join(env_names)}，当前配置存在默认值或代码内兜底值。"
    return "WARN", f"未设置 {', '.join(env_names)}。"


def check_python_version() -> DoctorResult:
    """项目要求 Python >= 3.10，低于此版本直接 FAIL。"""
    version = _format_version_tuple(sys.version_info)
    passed = sys.version_info >= (3, 10)
    return _result(
        "Python 版本",
        "PASS" if passed else "FAIL",
        f"当前 Python {version}，项目要求 Python 3.10+。",
        "无需处理。" if passed else "安装 Python 3.10+ 并确认当前虚拟环境已激活。",
        required=True,
        affects_default_tests=True,
        affects_real_runtime=True,
        mode="basic",
    )


def check_python_packages() -> DoctorResult:
    """检查后端运行所需的关键第三方包是否已安装。"""
    # 键为展示名，值为实际导入名（如 PyMuPDF 的导入名是 fitz）。
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "requests": "requests",
        "numpy": "numpy",
        "langgraph": "langgraph",
        "pymilvus": "pymilvus",
        "openai": "openai",
        "fitz(PyMuPDF)": "fitz",
        "docling": "docling",
    }
    missing = [label for label, module in packages.items() if not _importable(module)]
    if missing:
        return _result(
            "后端关键 Python 包",
            "FAIL",
            f"缺少包：{', '.join(missing)}。",
            "安装依赖：pip install -r requirements.txt",
            required=True,
            affects_default_tests=True,
            affects_real_runtime=True,
            mode="basic",
        )
    return _result(
        "后端关键 Python 包",
        "PASS",
        "FastAPI、Milvus、OpenAI、PDF 解析等关键包均可导入。",
        required=True,
        affects_default_tests=True,
        affects_real_runtime=True,
        mode="basic",
    )


def check_node_version() -> DoctorResult:
    """检查 node 与 npm 是否可用；Node 主版本低于 18 只给 WARN，因为未必立即无法构建。"""
    node = _find_executable("node")
    npm = _find_executable("npm")
    if not node or not npm:
        return _result(
            "Node / npm",
            "FAIL",
            "node 或 npm 不在 PATH 中。",
            "安装 Node.js 18+，并重新打开终端确认 node -v / npm -v 可用。",
            required=True,
            affects_default_tests=True,
            affects_real_runtime=True,
            mode="basic",
        )
    node_code, node_out = _run_command([node, "-v"], REPO_ROOT)
    npm_code, npm_out = _run_command([npm, "-v"], REPO_ROOT)
    if node_code != 0 or npm_code != 0:
        return _result(
            "Node / npm",
            "FAIL",
            f"node/npm 执行失败：node={node_out or node_code}; npm={npm_out or npm_code}。",
            "检查 Node.js 安装和 PATH。",
            required=True,
            affects_default_tests=True,
            affects_real_runtime=True,
            mode="basic",
        )
    major = int(node_out.lstrip("v").split(".", 1)[0])
    status = "PASS" if major >= 18 else "WARN"
    return _result(
        "Node / npm",
        status,
        f"Node {node_out}，npm {npm_out}。",
        "建议升级到 Node.js 18+。" if status == "WARN" else "无需处理。",
        required=True,
        affects_default_tests=True,
        affects_real_runtime=True,
        mode="basic",
    )


def check_frontend_dependencies() -> DoctorResult:
    """以 node_modules/.bin/vite 是否存在为标志，判断前端是否执行过 npm install。"""
    node_modules = FRONTEND_ROOT / "node_modules"
    vite_bin = node_modules / ".bin" / ("vite.cmd" if os.name == "nt" else "vite")
    if node_modules.exists() and vite_bin.exists():
        return _result(
            "前端依赖",
            "PASS",
            "frontend/node_modules 存在，Vite 本地依赖可见。",
            required=True,
            affects_default_tests=True,
            affects_real_runtime=True,
            mode="basic",
        )
    return _result(
        "前端依赖",
        "FAIL",
        "frontend/node_modules 或 Vite 本地可执行文件不存在。",
        "进入 frontend 后执行 npm install。",
        required=True,
        affects_default_tests=True,
        affects_real_runtime=True,
        mode="basic",
    )


def check_local_directories() -> DoctorResult:
    """检查后端运行时使用的数据、索引、结果目录是否存在且可写。

    目录不存在但父目录可写只给 WARN（服务首次运行时可自动创建）；不可写才是 FAIL。
    """
    dirs = [
        BACKEND_ROOT / "01-loaded-docs",
        BACKEND_ROOT / "02-embedded-docs",
        BACKEND_ROOT / "03-vector-store",
        BACKEND_ROOT / "03-docling-assets",
        BACKEND_ROOT / "05-generation-results",
        BACKEND_ROOT / "06-daily-arxiv-paper",
        BACKEND_ROOT / "06-database",
        REPO_ROOT / "temp",
    ]
    missing = [path.relative_to(REPO_ROOT).as_posix() for path in dirs if not path.exists()]
    unwritable: list[str] = []
    for path in dirs:
        probe_dir = path if path.exists() else path.parent
        if not probe_dir.exists():
            unwritable.append(path.relative_to(REPO_ROOT).as_posix())
            continue
        # 真正创建并立即删除一个临时文件，比检查权限位更可靠（Windows ACL、只读挂载等情况都能覆盖）。
        try:
            with tempfile.NamedTemporaryFile(prefix=".doctor-", dir=probe_dir, delete=True):
                pass
        except Exception:
            unwritable.append(path.relative_to(REPO_ROOT).as_posix())

    if unwritable:
        return _result(
            "本地数据目录",
            "FAIL",
            f"目录不可写或父目录不存在：{', '.join(unwritable)}。",
            "检查目录权限；必要时手动创建对应目录。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="basic",
        )
    if missing:
        return _result(
            "本地数据目录",
            "WARN",
            f"部分运行目录尚不存在，但父目录可写：{', '.join(missing)}。",
            "真实运行前可手动创建，或由相关服务在首次运行时创建。",
            required=False,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="basic",
        )
    return _result(
        "本地数据目录",
        "PASS",
        "数据、索引、trace 相关目录存在且可写。",
        required=True,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="basic",
    )


def check_backend_config_loads() -> DoctorResult:
    """确认 backend/utils/config.py 能正常导入，且核心配置字典都存在。"""
    # 导入 config 会按后端规则加载 .env；配置非法时这里会抛异常，由 _safe_check 记为 FAIL。
    import utils.config as config

    required_attrs = [
        "CORE_CONFIG",
        "SQLITE_CONFIG",
        "OAI_SQLITE_CONFIG",
        "MILVUS_CONFIG",
        "EMBEDDING_CONFIG",
        "RERANK_CONFIG",
        "GENERATION_CONFIG",
    ]
    missing = [name for name in required_attrs if not hasattr(config, name)]
    if missing:
        return _result(
            "后端配置加载",
            "FAIL",
            f"配置缺少字段：{', '.join(missing)}。",
            "检查 backend/utils/config.py 是否被误删或字段命名是否变更。",
            required=True,
            affects_default_tests=True,
            affects_real_runtime=True,
            mode="basic",
        )
    return _result(
        "后端配置加载",
        "PASS",
        "backend/utils/config.py 可导入，核心运行配置字段存在。",
        required=True,
        affects_default_tests=True,
        affects_real_runtime=True,
        mode="basic",
    )


def check_sqlite_temp_access() -> DoctorResult:
    """在业务 SQLite 所在目录新建一个临时数据库做建表和写入，验证权限后删除；不触碰真实数据库文件。"""
    import utils.config as config

    # 配置层已经输出绝对路径，Doctor 与真实运行直接检查同一个文件位置。
    db_path = Path(str(config.SQLITE_CONFIG["database_path"])).expanduser().resolve(strict=False)
    db_dir = db_path.parent
    if not db_dir.exists():
        return _result(
            "SQLite 目录与临时写入",
            "WARN",
            f"配置的 SQLite 目录不存在：{db_dir}。",
            "真实运行前创建目录，或确认服务启动时有权限自动创建。",
            required=False,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="basic",
        )
    temp_name = ""
    try:
        fd, temp_name = tempfile.mkstemp(prefix=".doctor-sqlite-", suffix=".db", dir=db_dir)
        os.close(fd)
        # Windows 会锁住已打开的临时文件句柄，因此先关闭文件，再让 sqlite 独占打开探针文件。
        conn = sqlite3.connect(temp_name)
        try:
            conn.execute("CREATE TABLE doctor_probe (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO doctor_probe (value) VALUES ('ok')")
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        return _result(
            "SQLite 目录与临时写入",
            "FAIL",
            f"无法在 SQLite 目录创建临时数据库：{type(exc).__name__}: {exc}",
            "检查 SQLITE_DATABASE_PATH 所在目录权限，避免把数据库放在不可写位置。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="basic",
        )
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    return _result(
        "SQLite 目录与临时写入",
        "PASS",
        f"可在 {db_dir} 创建并写入临时 SQLite 文件；未修改真实数据库。",
        required=True,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="basic",
    )


def check_environment_variables() -> DoctorResult:
    """检查 Embedding、LLM、Rerank 三类服务的密钥环境变量。

    三类都至少有一个候选变量被显式设置才是 PASS，任意一类没有就整体 WARN。
    注意：只设置 ALIYUN_API_KEY 时，配置层会用它兜底三类密钥，服务可以正常运行，
    但这里仍会报 WARN（提示“存在代码内兜底值”），这是预期行为，不是故障。
    """
    import utils.config as config

    # (类别, 候选环境变量, 配置层最终得到的值)
    checks = [
        ("Embedding", ["EMBEDDING_API_KEY", "EMBEDDING_DASHSCOPE_API_KEY", "OPENAI_API_KEY"], str(config.EMBEDDING_CONFIG.get("api_key", ""))),
        ("LLM", ["QWEN_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"], str(config.GENERATION_CONFIG.get("qwen_api_key", ""))),
        ("Rerank", ["RERANK_API_KEY", "RERANK_DASHSCOPE_API_KEY"], str(config.RERANK_CONFIG.get("api_key", ""))),
    ]
    warnings = []
    for label, env_names, configured in checks:
        status, reason = _mask_state(env_names, configured)
        if status != "PASS":
            warnings.append(f"{label}: {reason}")
    if warnings:
        return _result(
            "API Key 环境变量",
            "WARN",
            "；".join(warnings),
            "在本机安全位置配置对应环境变量；doctor 不会打印任何密钥值。",
            required=False,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="basic",
        )
    return _result(
        "API Key 环境变量",
        "PASS",
        "Embedding、LLM、rerank 至少有一个相关 API key 环境变量显式存在；未输出密钥值。",
        required=False,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="basic",
    )


def check_fastapi_lazy_app() -> DoctorResult:
    """以 lazy 模式创建 FastAPI app（不初始化向量库和模型客户端），确认路由能正常注册。"""
    import main

    app = main.create_app(load_mode="lazy")
    route_count = len(getattr(app, "routes", []) or [])
    if route_count <= 0:
        return _result(
            "FastAPI lazy app 创建",
            "FAIL",
            "create_app(load_mode='lazy') 返回的 app 没有路由。",
            "检查 backend/main.py 的 router 注册逻辑。",
            required=True,
            affects_default_tests=True,
            affects_real_runtime=True,
            mode="basic",
        )
    return _result(
        "FastAPI lazy app 创建",
        "PASS",
        f"FastAPI app 可在 lazy 模式创建，已注册路由数：{route_count}。",
        required=True,
        affects_default_tests=True,
        affects_real_runtime=True,
        mode="basic",
    )


def check_pdf_dependencies() -> DoctorResult:
    """检查 PDF 下载解析链路依赖的三个库；只影响真实运行，默认离线测试使用替身。"""
    modules = {
        "PyMuPDF(fitz)": "fitz",
        "docling": "docling",
        "pypdf": "pypdf",
    }
    missing = [label for label, module in modules.items() if not _importable(module)]
    if missing:
        return _result(
            "PDF 解析依赖",
            "FAIL",
            f"缺少 PDF 解析依赖：{', '.join(missing)}。",
            "安装 requirements 中的 PDF / Docling 相关依赖。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="basic",
        )
    return _result(
        "PDF 解析依赖",
        "PASS",
        "PyMuPDF、Docling、pypdf 均可导入。",
        required=True,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="basic",
    )


def _tcp_probe(uri: str, timeout: float = 3.0) -> tuple[bool, str]:
    """只尝试建立 TCP 连接，判断目标端口是否在监听；URI 不带协议时按 http:// 解析。"""
    parsed = urllib.parse.urlparse(uri if "://" in uri else f"http://{uri}")
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} TCP 可连接。"
    except Exception as exc:
        return False, f"{host}:{port} TCP 连接失败：{type(exc).__name__}: {exc}"


def check_milvus_connection() -> DoctorResult:
    """分两步检查 Milvus：先 TCP 探测端口，再用 MilvusClient 列出集合，确认端口上确实是可用的 Milvus。

    注意：配置为 Milvus Lite 本地文件（Linux 默认）时 URI 不是网络地址，TCP 探测会失败，
    此时 FAIL 不代表真实运行有问题。
    """
    import utils.config as config

    uri = str(config.MILVUS_CONFIG.get("uri", ""))
    if not _importable("pymilvus"):
        return _result(
            "Milvus / 向量库连接",
            "FAIL",
            "pymilvus 未安装，无法继续检查 Milvus 服务连接。",
            "先安装后端依赖：pip install -r requirements.txt，之后再运行 python scripts/doctor.py full。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    ok, message = _tcp_probe(uri)
    if not ok:
        return _result(
            "Milvus / 向量库连接",
            "FAIL",
            f"{message} 当前配置 MILVUS_URI={uri or '<empty>'}。",
            "启动 Milvus，或修正 MILVUS_URI 后重试 full doctor。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=uri)
        collections = client.list_collections(timeout=5)
        return _result(
            "Milvus / 向量库连接",
            "PASS",
            f"{message} Milvus list_collections 成功，集合数：{len(collections)}。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    except Exception as exc:
        return _result(
            "Milvus / 向量库连接",
            "FAIL",
            f"TCP 可连接，但 Milvus 客户端查询失败：{type(exc).__name__}: {exc}",
            "确认目标端口确实是 Milvus 服务，并检查版本兼容性。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )


def _urlopen_probe(url: str, timeout: int = 8) -> tuple[bool, str]:
    """发起一次 HTTP GET，只读取前 256 字节，判断外部服务是否可访问。"""
    request = urllib.request.Request(url, headers={"User-Agent": "rag-project-doctor/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            sample = response.read(256)
            return True, f"HTTP {response.status}，返回 {len(sample)} bytes 预览。"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_arxiv_api_connection() -> DoctorResult:
    """请求一条 arXiv 搜索结果，验证网络能访问 arXiv API（ARXIV_DATA_SOURCE=api 时使用）。"""
    # 注意：这里直接用 urllib 访问，不经过 ARXIV_PROXY_URL；需要代理时可通过系统 HTTP(S)_PROXY 环境变量生效。
    url = "https://export.arxiv.org/api/query?search_query=cat:cs.CL&start=0&max_results=1"
    ok, message = _urlopen_probe(url)
    return _result(
        "arXiv API 连接",
        "PASS" if ok else "FAIL",
        message,
        "检查网络、代理或 arXiv 访问限制；必要时配置 ARXIV_PROXY_URL。" if not ok else "无需处理。",
        required=False,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="full",
    )


def check_arxiv_oai_connection() -> DoctorResult:
    """调用 OAI-PMH 的 Identify 接口，验证能访问 arXiv OAI 元数据服务（同步本地元数据镜像时使用）。"""
    import utils.config as config

    endpoint = str(config.ARXIV_OAI_CONFIG.get("endpoint", "https://oaipmh.arxiv.org/oai"))
    separator = "&" if "?" in endpoint else "?"
    ok, message = _urlopen_probe(f"{endpoint}{separator}verb=Identify")
    return _result(
        "arXiv OAI 连接",
        "PASS" if ok else "FAIL",
        message,
        "检查网络、代理或 ARXIV_OAI_ENDPOINT 配置。" if not ok else "无需处理。",
        required=False,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="full",
    )


def _skip_paid(name: str, mode: str) -> DoctorResult:
    """未加 --check-paid 时，可能计费的模型检查统一返回 SKIP。"""
    return _result(
        name,
        "SKIP",
        "该检查会真实调用可能计费的模型服务，默认跳过。",
        "如需确认真实模型连接，显式运行：python scripts/doctor.py full --check-paid。",
        required=False,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode=mode,
    )


def check_embedding_connection(check_paid: bool) -> DoctorResult:
    """发送一条最短文本的向量化请求，验证 Embedding 服务地址、模型名和密钥；仅支持 DashScope。"""
    if not check_paid:
        return _skip_paid("Embedding 服务连接", "full")
    import requests
    import utils.config as config

    embedding_config = config.EMBEDDING_CONFIG
    provider = str(embedding_config.get("provider", "")).lower()
    if provider != "dashscope":
        return _result(
            "Embedding 服务连接",
            "SKIP",
            f"当前 provider={provider or '<empty>'}，doctor 仅内置 DashScope 最小健康检查。",
            "如使用其他 provider，请单独用供应商 SDK 做最小 embedding 请求。",
            required=False,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    api_key = str(embedding_config.get("dashscope_api_key") or embedding_config.get("api_key") or "")
    if not api_key:
        return _result(
            "Embedding 服务连接",
            "FAIL",
            "Embedding API key 未配置。",
            "设置 EMBEDDING_DASHSCOPE_API_KEY 或 EMBEDDING_API_KEY。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    payload = {
        "model": embedding_config.get("model_name"),
        "input": {"contents": [{"text": "doctor connectivity probe"}]},
        "parameters": {"dimension": int(embedding_config.get("dimension") or 2048)},
    }
    response = requests.post(
        str(embedding_config.get("base_url")),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if response.status_code >= 400:
        return _result(
            "Embedding 服务连接",
            "FAIL",
            f"HTTP {response.status_code}: {response.text[:240]}",
            "检查 Embedding endpoint、模型名和 API key 权限。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    return _result(
        "Embedding 服务连接",
        "PASS",
        "最小 DashScope embedding 请求成功；未输出密钥。",
        required=True,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="full",
    )


def check_llm_connection(check_paid: bool) -> DoctorResult:
    """用小模型发送 "ping"（最多 4 个 token），以最低成本验证 LLM 服务可用。"""
    if not check_paid:
        return _skip_paid("LLM 服务连接", "full")
    import utils.config as config
    from openai import OpenAI

    generation_config = config.GENERATION_CONFIG
    api_key = str(generation_config.get("qwen_api_key") or "")
    if not api_key:
        return _result(
            "LLM 服务连接",
            "FAIL",
            "Qwen API key 未配置。",
            "设置 QWEN_API_KEY。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    client = OpenAI(api_key=api_key, base_url=str(generation_config.get("qwen_base_url")))
    response = client.chat.completions.create(
        model=str(generation_config.get("small_qwen_model_name")),
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=4,
        temperature=0,
    )
    content = response.choices[0].message.content if response.choices else ""
    if not content:
        return _result(
            "LLM 服务连接",
            "WARN",
            "模型请求返回成功但内容为空。",
            "检查模型名和供应商返回格式。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    return _result(
        "LLM 服务连接",
        "PASS",
        "最小 chat completion 请求成功；未输出密钥。",
        required=True,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="full",
    )


def check_rerank_connection(check_paid: bool) -> DoctorResult:
    """用一个查询和两篇短文档发起重排请求，验证 Rerank 服务可用。"""
    if not check_paid:
        return _skip_paid("Rerank 服务连接", "full")
    import requests
    import utils.config as config

    rerank_config = config.RERANK_CONFIG
    api_key = str(rerank_config.get("dashscope_api_key") or rerank_config.get("api_key") or "")
    if not api_key:
        return _result(
            "Rerank 服务连接",
            "FAIL",
            "Rerank API key 未配置。",
            "设置 RERANK_DASHSCOPE_API_KEY 或 RERANK_API_KEY。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    model_name = str(rerank_config.get("model_name") or "qwen3-vl-rerank")
    # DashScope 原生重排模型使用 input/parameters 嵌套格式，其余按 OpenAI 兼容的扁平格式发送。
    if model_name in {"qwen3-vl-rerank", "gte-rerank-v2"}:
        payload: dict[str, Any] = {
            "model": model_name,
            "input": {
                "query": {"text": "doctor"},
                "documents": [{"text": "doctor connectivity probe"}, {"text": "unrelated"}],
            },
            "parameters": {"return_documents": False, "top_n": 1},
        }
    else:
        payload = {
            "model": model_name,
            "query": "doctor",
            "documents": ["doctor connectivity probe", "unrelated"],
            "top_n": 1,
        }
    response = requests.post(
        str(rerank_config.get("base_url")),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if response.status_code >= 400:
        return _result(
            "Rerank 服务连接",
            "FAIL",
            f"HTTP {response.status_code}: {response.text[:240]}",
            "检查 Rerank endpoint、模型名和 API key 权限。",
            required=True,
            affects_default_tests=False,
            affects_real_runtime=True,
            mode="full",
        )
    return _result(
        "Rerank 服务连接",
        "PASS",
        "最小 rerank 请求成功；未输出密钥。",
        required=True,
        affects_default_tests=False,
        affects_real_runtime=True,
        mode="full",
    )


# basic 模式检查清单：(名称, 检查函数, 是否必需, 是否影响默认测试, 是否影响真实运行)。
# 后三个标记用于检查函数自身抛异常时生成 FAIL 结果（见 _safe_check）。
BASIC_CHECKS: list[tuple[str, Callable[[], DoctorResult], bool, bool, bool]] = [
    ("Python 版本", check_python_version, True, True, True),
    ("后端关键 Python 包", check_python_packages, True, True, True),
    ("Node / npm", check_node_version, True, True, True),
    ("前端依赖", check_frontend_dependencies, True, True, True),
    ("本地数据目录", check_local_directories, True, False, True),
    ("后端配置加载", check_backend_config_loads, True, True, True),
    ("SQLite 目录与临时写入", check_sqlite_temp_access, True, False, True),
    ("API Key 环境变量", check_environment_variables, False, False, True),
    ("FastAPI lazy app 创建", check_fastapi_lazy_app, True, True, True),
    ("PDF 解析依赖", check_pdf_dependencies, True, False, True),
]


def run_basic_checks() -> list[DoctorResult]:
    """按顺序执行全部 basic 检查。"""
    return [
        _safe_check(
            name,
            check,
            required=required,
            affects_default_tests=affects_default_tests,
            affects_real_runtime=affects_real_runtime,
            mode="basic",
        )
        for name, check, required, affects_default_tests, affects_real_runtime in BASIC_CHECKS
    ]


def run_full_checks(check_paid: bool) -> list[DoctorResult]:
    """执行 full 模式额外的真实连接检查；check_paid 为 False 时模型类检查返回 SKIP。"""
    checks: list[tuple[str, Callable[[], DoctorResult], bool, bool, bool]] = [
        ("Milvus / 向量库连接", check_milvus_connection, True, False, True),
        ("arXiv API 连接", check_arxiv_api_connection, False, False, True),
        ("arXiv OAI 连接", check_arxiv_oai_connection, False, False, True),
        ("Embedding 服务连接", lambda: check_embedding_connection(check_paid), True, False, True),
        ("LLM 服务连接", lambda: check_llm_connection(check_paid), True, False, True),
        ("Rerank 服务连接", lambda: check_rerank_connection(check_paid), True, False, True),
    ]
    return [
        _safe_check(
            name,
            check,
            required=required,
            affects_default_tests=affects_default_tests,
            affects_real_runtime=affects_real_runtime,
            mode="full",
        )
        for name, check, required, affects_default_tests, affects_real_runtime in checks
    ]


def _status_counts(results: list[DoctorResult]) -> dict[str, int]:
    """统计各状态的数量。"""
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def print_report(results: list[DoctorResult], mode: str, elapsed: float) -> None:
    """打印人类可读的体检报告。

    每项以 "[01] FAIL 名称" 开头、下一行为 "原因:"，check_quality.py 依赖这个格式提取失败摘要，修改时需同步。
    """
    counts = _status_counts(results)
    print("=== 环境体检报告 ===")
    print(f"仓库根目录: {REPO_ROOT}")
    print(f"体检模式: {mode}")
    print(f"耗时: {elapsed:.1f}s")
    print(
        "状态汇总: "
        f"PASS={counts.get('PASS', 0)} "
        f"WARN={counts.get('WARN', 0)} "
        f"FAIL={counts.get('FAIL', 0)} "
        f"SKIP={counts.get('SKIP', 0)}"
    )
    print()
    for index, result in enumerate(results, start=1):
        print(f"[{index:02d}] {result.status} {result.name}")
        print(f"     原因: {result.reason}")
        print(f"     建议: {result.fix}")
        print(f"     必需项: {'是' if result.required else '否'}")
        print(f"     影响默认测试: {'是' if result.affects_default_tests else '否'}")
        print(f"     影响真实运行: {'是' if result.affects_real_runtime else '否'}")
        print()

    if counts.get("FAIL", 0):
        print("最终结论: FAIL，存在需要修复的环境问题。")
    elif counts.get("WARN", 0):
        print("最终结论: WARN，基础环境可继续排查，但真实运行前建议处理警告项。")
    else:
        print("最终结论: PASS，选中模式下未发现阻断项。")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="项目运行环境体检工具。")
    parser.add_argument("mode", nargs="?", choices=["basic", "full"], default="basic", help="默认 basic；full 会检查真实连接。")
    parser.add_argument(
        "--check-paid",
        action="store_true",
        help="允许 full 模式真实调用可能计费的 Embedding / LLM / rerank API。",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于复制给 AI 或 CI 解析。")
    return parser.parse_args()


def main() -> int:
    """命令行入口：full 模式先跑 basic 再跑 full；存在 FAIL 返回 1，否则返回 0。"""
    args = parse_args()
    start = time.perf_counter()
    results = run_basic_checks()
    if args.mode == "full":
        results.extend(run_full_checks(check_paid=bool(args.check_paid)))
    elapsed = time.perf_counter() - start

    if args.json:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "elapsed_seconds": round(elapsed, 3),
                    "summary": _status_counts(results),
                    "results": [asdict(result) for result in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_report(results, mode=args.mode, elapsed=elapsed)

    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
