from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
FRONTEND_SRC_ROOT = REPO_ROOT / "new_frontend" / "src"
DOCS_ROOT = REPO_ROOT / "docs"
STATIC_LEGACY_SCAN_ROOTS = (BACKEND_ROOT, REPO_ROOT / "scripts", FRONTEND_SRC_ROOT, DOCS_ROOT)
STATIC_LEGACY_SCAN_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".md"}
REMOVED_VECTOR_STORE_FILENAME = "vector_store_service_" + "langchain.py"
REMOVED_LEGACY_PATHS = (
    (
        BACKEND_ROOT / "services" / "archive" / REMOVED_VECTOR_STORE_FILENAME,
        "旧 LangChain/Milvus VectorStore 实现已删除；向量存储统一使用 services.storage.vector_store_service.VectorStoreService。",
    ),
)

# 已删除的兼容入口不能重新出现在源码里；这些字符串用分段拼接保存，
# 避免本检查脚本本身被普通文本搜索误判为旧入口残留。
REMOVED_LEGACY_ENTRY_MARKERS = (
    (
        "build_" + "legacy_" + "pending_action",
        "旧 pending_action 构造函数已经删除，请使用当前确认恢复链路。",
    ),
    (
        "arxiv_search_agent" + ".compat",
        "旧 compat 包路径已经退出主流程，请不要新增独立 legacy 映射入口。",
    ),
    (
        "arxiv_search_agent" + ".compat" + ".legacy",
        "旧 compat 模块路径已经删除，请不要新增独立 legacy 映射入口。",
    ),
    (
        "agents." + "arxiv_search_agent" + ".compat" + ".legacy",
        "旧 compat 导入路径已经删除，请不要绕过 service/graph 的出站投影。",
    ),
    (
        "from agents." + "arxiv_search_agent" + ".compat import",
        "compat 包不再导出确认展示构造器，请改用当前结构化确认链路。",
    ),
    (
        "from backend.agents." + "arxiv_search_agent" + ".compat import",
        "compat 包不再导出确认展示构造器，请改用当前结构化确认链路。",
    ),
    (
        "vector_store_service_" + "langchain",
        "旧 VectorStore 文件级实现已删除，请使用 services.storage.vector_store_service.VectorStoreService。",
    ),
    (
        "services." + "archive" + ".vector_store",
        "archive 下不再保留 VectorStore 入口，请通过 dependencies.get_vector_store_service() 或正式 storage 层获取服务。",
    ),
    (
        "backend." + "services" + ".archive" + ".vector_store",
        "archive 下不再保留 VectorStore 入口，请通过 dependencies.get_vector_store_service() 或正式 storage 层获取服务。",
    ),
)

# 用户偏好读取已经收敛为 GET /user/preferences/{user_id}；这里用正则兜住常见回流形态，
# 包括后端重新注册 POST 路由、前端重新调用 POST 读取、文档/测试重新声明旧兼容入口。
REMOVED_USER_PREFERENCE_POST_PATTERNS = (
    (
        re.compile(r"legacy_" + r"post_" + r"user_" + r"preferences"),
        "旧 POST 偏好读取函数已删除，请只保留 get_user_preferences() 作为读取入口。",
    ),
    (
        re.compile(r"@\s*router\s*\.\s*post\s*\(\s*[\"']/" + r"preferences[\"']", re.MULTILINE),
        "禁止恢复 /user/preferences 的 POST 路由；偏好读取必须使用 GET /user/preferences/{user_id}。",
    ),
    (
        re.compile(r"\.\s*post\s*\(\s*[`\"'](?:/api)?/user/" + r"preferences[`\"']"),
        "禁止前端或测试通过 POST 读取用户偏好；请调用 GET /user/preferences/{user_id}。",
    ),
    (
        re.compile(r"POST\s+`?(?:/api)?/user/" + r"preferences`?"),
        "文档和测试计划不应再声明旧 POST 偏好读取入口；请统一记录 GET 读取入口。",
    ),
    (
        re.compile(
            r"successor-" + r"version|Deprecated:\s*read\s+user\s+preferences|deprecated\s+POST\s+compatibility\s+endpoint",
            re.IGNORECASE,
        ),
        "旧 deprecated 兼容入口的响应头和 successor 文案已删除，不应重新出现。",
    ),
)


IMPORT_MODULES = [
    "utils.config",
    "core.errors",
    "dependencies",
    "main",
    "routers.agent_router",
    "routers.arxiv_router",
    "routers.chunk_router",
    "routers.paper_router",
    "routers.qa_router",
    "routers.qa_utils",
    "routers.user_router",
    "services.arxiv.arxiv_query_builder",
    "services.arxiv.arxiv_search_service",
    # 本地检索已经收敛到 OAI 正式服务，静态检查不得重新依赖已删除的兼容入口。
    "services.arxiv.arxiv_oai_service",
    "services.document.chunking_service",
    "services.document.loading_service",
    "services.embedding.embedding_service",
    "services.intent.intent_service",
    "services.llm.generation_service",
    "services.memory",
    "services.paper_qa.index_job_manager",
    "services.paper_qa.paper_qa_index_builder",
    "services.paper_qa.paper_qa_service",
    "services.recommendation.recommendation_service",
    "services.retrieval.enhanced_retrieval_service",
    "services.retrieval.query_planner",
    "services.retrieval.rerank_service",
    "services.retrieval.route_retriever",
    # SQLite 组合包是当前持久化公共入口，旧 DatabaseService 已按存储重构契约删除。
    "services.storage.sqlite",
    "services.storage.vector_store_service",
    "tools.arxiv_tools",
    "tools.paper_qa_tools",
    "tools.recommendation_tools",
    "tools.schemas",
    "tools.tool_registry",
    "tools.tool_result",
    "agents.arxiv_search_agent.graph",
    "agents.arxiv_search_agent.plan_executor",
    "agents.arxiv_search_agent.planner",
    "agents.arxiv_search_agent.replanner",
    "agents.arxiv_search_agent.schemas",
    "agents.arxiv_search_agent.service",
    "agents.arxiv_search_agent.tool_registry",
]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    duration_seconds: float
    detail: str = ""


def _prepare_environment() -> None:
    # 静态检查只验证导入边界，不允许借由 FastAPI preload 去实例化 Milvus、模型或远程客户端。
    os.environ.setdefault("RAG_QUALITY_GATE", "offline")
    os.environ.setdefault("BACKEND_SERVICE_LOAD_MODE", "lazy")
    # import-smoke 只验证模块边界；强制使用内存 checkpoint，避免旧本地 SQLite schema 影响静态检查。
    os.environ["AGENT_RUNTIME_CHECKPOINT_BACKEND"] = "memory"
    for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    _install_optional_dependency_stubs()


def _install_optional_dependency_stubs() -> None:
    """为缺失的重型可选依赖安装最小导入桩。

    static 阶段的职责是验证本仓库模块能否在 lazy/offline 模式下完成导入，
    不是验证 Milvus、LangGraph、PDF 或本地模型运行时是否真的可用。
    """

    if "langgraph.graph" not in sys.modules:
        langgraph_module = sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
        graph_module = types.ModuleType("langgraph.graph")

        class _CompiledGraph:
            def get_graph(self):
                return types.SimpleNamespace(draw_mermaid=lambda: "graph TD\n    START --> END")

        class _StateGraph:
            def __init__(self, *_args, **_kwargs):
                pass

            def add_node(self, *_args, **_kwargs):
                return None

            def add_edge(self, *_args, **_kwargs):
                return None

            def add_conditional_edges(self, *_args, **_kwargs):
                return None

            def compile(self, *_args, **_kwargs):
                return _CompiledGraph()

        graph_module.START = "START"
        graph_module.END = "END"
        graph_module.StateGraph = _StateGraph
        langgraph_module.graph = graph_module
        sys.modules["langgraph.graph"] = graph_module

    if "langgraph.types" not in sys.modules:
        types_module = types.ModuleType("langgraph.types")

        class _Command:
            def __init__(self, *, resume=None):
                self.resume = resume

        types_module.Command = _Command
        types_module.interrupt = lambda payload: payload
        sys.modules["langgraph.types"] = types_module

    if "langgraph.checkpoint.memory" not in sys.modules:
        checkpoint_module = sys.modules.setdefault("langgraph.checkpoint", types.ModuleType("langgraph.checkpoint"))
        memory_module = types.ModuleType("langgraph.checkpoint.memory")

        class _MemorySaver:
            pass

        memory_module.MemorySaver = _MemorySaver
        memory_module.InMemorySaver = _MemorySaver
        checkpoint_module.memory = memory_module
        sys.modules["langgraph.checkpoint.memory"] = memory_module

    if "pymilvus" not in sys.modules:
        pymilvus_module = types.ModuleType("pymilvus")

        class _DataType:
            INT64 = "INT64"
            VARCHAR = "VARCHAR"
            FLOAT_VECTOR = "FLOAT_VECTOR"

        class _MilvusClient:
            @staticmethod
            def create_schema(*_args, **_kwargs):
                return types.SimpleNamespace(add_field=lambda *_a, **_k: None)

            @staticmethod
            def prepare_index_params(*_args, **_kwargs):
                return types.SimpleNamespace(add_index=lambda *_a, **_k: None)

        pymilvus_module.DataType = _DataType
        pymilvus_module.MilvusClient = _MilvusClient
        pymilvus_module.Collection = object
        pymilvus_module.connections = types.SimpleNamespace()
        pymilvus_module.utility = types.SimpleNamespace()
        sys.modules["pymilvus"] = pymilvus_module

    if "fitz" not in sys.modules:
        fitz_module = types.ModuleType("fitz")
        fitz_module.open = lambda *_args, **_kwargs: None
        fitz_module.Rect = lambda *args, **_kwargs: types.SimpleNamespace(x0=0, y0=0, x1=0, y1=0, args=args)
        fitz_module.Point = lambda *args, **_kwargs: types.SimpleNamespace(args=args)
        sys.modules["fitz"] = fitz_module

    if "transformers" not in sys.modules:
        transformers_module = types.ModuleType("transformers")
        transformers_module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: None)
        transformers_module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: None)
        sys.modules["transformers"] = transformers_module

    if "openai" not in sys.modules:
        openai_module = types.ModuleType("openai")

        class _OpenAI:
            def __init__(self, *_args, **_kwargs):
                pass

        openai_module.OpenAI = _OpenAI
        sys.modules["openai"] = openai_module

    if "pypinyin" not in sys.modules:
        pypinyin_module = types.ModuleType("pypinyin")
        pypinyin_module.lazy_pinyin = lambda text, *_args, **_kwargs: list(str(text or ""))
        pypinyin_module.Style = types.SimpleNamespace(NORMAL="normal")
        sys.modules["pypinyin"] = pypinyin_module


def _run_named_check(name: str, fn: Callable[[], str]) -> CheckResult:
    print(f"\n--- {name} ---")
    started_at = time.perf_counter()
    try:
        detail = fn()
    except Exception as exc:
        duration = time.perf_counter() - started_at
        print(f"结果: FAIL ({duration:.1f}s)")
        print(str(exc))
        return CheckResult(name=name, passed=False, duration_seconds=duration, detail=str(exc))

    duration = time.perf_counter() - started_at
    print(f"结果: PASS ({duration:.1f}s)")
    if detail:
        print(detail)
    return CheckResult(name=name, passed=True, duration_seconds=duration, detail=detail)


def _check_compileall() -> str:
    completed = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(BACKEND_ROOT)],
        cwd=REPO_ROOT,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part.strip())
        raise RuntimeError(output.strip() or f"compileall failed with exit={completed.returncode}")
    return "已编译检查 backend 下所有 Python 文件。"


def _check_imports() -> str:
    imported: list[str] = []
    failures: list[str] = []
    for module_name in IMPORT_MODULES:
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    if failures:
        # import smoke 的职责是提前暴露入口模块缺失、重命名遗漏和导出契约断裂。
        raise RuntimeError("关键模块导入失败:\n" + "\n".join(failures))
    return f"已导入 {len(imported)} 个关键模块。"


def _iter_static_legacy_scan_files() -> list[Path]:
    files: list[Path] = []
    for root in STATIC_LEGACY_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in STATIC_LEGACY_SCAN_SUFFIXES:
                continue
            files.append(path)
    return sorted(files)


def _check_removed_legacy_entry_markers() -> str:
    """阻止已删除的 legacy 入口再次被接回源码。

    当前确认恢复的执行真源是 pending_confirmation / runtime_state / resume；
    pending_action 只允许由现有 service/graph 出站投影生成展示镜像。
    VectorStore 只保留 storage 层正式实现，避免 archive 旧实现恢复后形成双入口。
    用户偏好读取只允许 GET 入口，避免 POST 读取语义回流为伪 upsert。
    """
    hits: list[str] = []
    for removed_path, guidance in REMOVED_LEGACY_PATHS:
        if removed_path.exists():
            relative_path = removed_path.relative_to(REPO_ROOT)
            hits.append(f"{relative_path}: 禁止恢复已删除路径。{guidance}")

    for path in _iter_static_legacy_scan_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker, guidance in REMOVED_LEGACY_ENTRY_MARKERS:
            if marker not in text:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if marker in line:
                    relative_path = path.relative_to(REPO_ROOT)
                    hits.append(f"{relative_path}:{line_number}: 禁止出现 `{marker}`。{guidance}")
        for pattern, guidance in REMOVED_USER_PREFERENCE_POST_PATTERNS:
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                relative_path = path.relative_to(REPO_ROOT)
                snippet = " ".join(match.group(0).split())
                hits.append(f"{relative_path}:{line_number}: 禁止恢复旧 POST 偏好读取入口 `{snippet}`。{guidance}")

    if hits:
        # 这里直接失败，避免旧兼容入口和当前正式服务链路并存后产生双入口维护成本。
        raise RuntimeError(
            "检测到已删除的 legacy 入口；请使用当前正式服务链路。\n"
            + "\n".join(hits[:50])
        )
    return "未发现已删除的 legacy 入口；确认恢复、VectorStore 和用户偏好读取均保持当前正式入口。"


def _check_ruff() -> str:
    ruff = shutil.which("ruff")
    if not ruff:
        return "未检测到 ruff，跳过可选 lint 扩展。安装 ruff 后会启用 E9/F821/F822/F823。"

    command = [
        ruff,
        "check",
        str(BACKEND_ROOT),
        "--select",
        "E9,F821,F822,F823",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part.strip())
    if completed.returncode != 0:
        # 第一版只启用语法和未定义名称类规则，避免风格债阻塞门禁落地。
        raise RuntimeError(output.strip() or f"ruff failed with exit={completed.returncode}")
    return output.strip() or "ruff 基础规则通过。"


def _print_summary(results: list[CheckResult]) -> None:
    print("\n=== 后端静态检查汇总 ===")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:4} {result.name:20} {result.duration_seconds:6.1f}s")
        if not result.passed and result.detail:
            first_line = next((line for line in result.detail.splitlines() if line.strip()), "")
            if first_line:
                print(f"     摘要: {first_line}")

    if all(result.passed for result in results):
        print("\n最终结果: PASS，后端静态检查通过。")
    else:
        print("\n最终结果: FAIL，后端静态检查存在失败项。")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="后端静态检查：编译、关键模块导入和可选 ruff lint。")
    parser.add_argument("--skip-ruff", action="store_true", help="跳过可选 ruff lint 检查。")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _prepare_environment()
    print(f"仓库根目录: {REPO_ROOT}")
    print("运行模式: offline static")

    checks: list[tuple[str, Callable[[], str]]] = [
        ("legacy-entry-guard", _check_removed_legacy_entry_markers),
        ("compileall", _check_compileall),
        ("import-smoke", _check_imports),
    ]
    if not args.skip_ruff:
        checks.append(("ruff-optional", _check_ruff))

    results = [_run_named_check(name, fn) for name, fn in checks]
    _print_summary(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
