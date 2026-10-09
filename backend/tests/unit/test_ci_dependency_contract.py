import ast
import re
from importlib.metadata import PackageNotFoundError, packages_distributions, requires
from pathlib import Path

from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[3]
# 生产服务器使用 Debian 12 自带解释器；CI 镜像是同一 minor 的更新补丁版本。
SERVER_PYTHON = "3.11.2"
CI_PYTHON = "3.11.99"


def test_ci_pins_fastapi_to_router_compatible_version() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    fastapi_entries = [line.strip() for line in requirements if line.strip().lower().startswith("fastapi")]

    # CI 会从零解析依赖；锁定已通过 startup smoke 的版本，避免新版 include_router 回归清空路由路径。
    assert fastapi_entries == ["fastapi==0.136.1"]


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-r")):
            continue
        names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def _doctor_required_modules() -> set[str]:
    tree = ast.parse((REPO_ROOT / "scripts" / "doctor.py").read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {"check_python_packages", "check_pdf_dependencies"}:
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    modules.update(value.value for value in child.values if isinstance(value, ast.Constant))
    return modules


def test_doctor_required_modules_are_declared_in_requirements() -> None:
    # 本机环境往往装了比 requirements 更多的包；doctor 若检查未声明的依赖，只会在 CI 全新安装时暴露。
    declared = _requirement_names(REPO_ROOT / "requirements.txt")
    distributions = packages_distributions()
    modules = _doctor_required_modules()
    assert modules, "未能从 doctor.py 解析出依赖检查列表"

    undeclared = sorted(
        module
        for module in modules
        if not any(dist.lower().replace("_", "-") in declared for dist in distributions.get(module, []))
        # docling 在部分本地环境以 docling-slim 发行版提供，requirements 中声明的是 docling。
        and not (module == "docling" and "docling" in declared)
    )
    assert undeclared == []


def _marker_env(python_full_version: str) -> dict[str, str]:
    return {
        "python_full_version": python_full_version, "python_version": "3.11", "sys_platform": "linux",
        "platform_system": "Linux", "platform_machine": "x86_64", "os_name": "posix",
        "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": "",
    }


def test_patch_version_dependent_requirements_are_declared() -> None:
    # 发布包按 CI 环境冻结依赖、服务器 --no-index 安装；只在服务器补丁版本上生效的依赖不会进入 wheels，
    # 必须在 requirements.txt 显式声明才会被 CI 安装并冻结（曾因 redis -> async-timeout 部署失败）。
    declared = _requirement_names(REPO_ROOT / "requirements.txt")
    server_env, ci_env = _marker_env(SERVER_PYTHON), _marker_env(CI_PYTHON)
    pending, visited, missing = list(declared), set(), []
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        try:
            dependencies = requires(name) or []
        except PackageNotFoundError:
            continue
        for raw in dependencies:
            dependency = Requirement(raw)
            child = dependency.name.lower().replace("_", "-")
            marker = dependency.marker
            server_needs = marker is None or marker.evaluate(server_env)
            if server_needs:
                pending.append(child)
            if marker is not None and server_needs and not marker.evaluate(ci_env) and child not in declared:
                missing.append(f"{name} -> {raw}")
    assert missing == []
