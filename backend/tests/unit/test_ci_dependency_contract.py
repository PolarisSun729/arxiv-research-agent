import ast
import re
from importlib.metadata import PackageNotFoundError, packages_distributions, requires
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[3]
# 生产服务器使用 Debian 12 自带解释器；CI 镜像是同一 minor 的更新补丁版本。
SERVER_PYTHON = "3.11.2"
CI_PYTHON = "3.11.99"


def test_ci_pins_fastapi_to_router_compatible_version() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    fastapi_entries = [line.strip() for line in requirements if line.strip().lower().startswith("fastapi")]

    # 重新生成锁文件时也不能越过已通过 startup smoke 的版本，避免新版 include_router 回归清空路由路径。
    assert fastapi_entries == ["fastapi==0.136.1"]


def _locked_versions(path: Path) -> dict[str, Version]:
    """解析 uv pip compile 生成的锁文件，返回 {规范化包名: 版本}；哈希、注释和续行都会跳过。"""
    locked = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0] in " #-":
            continue
        requirement = Requirement(line.rstrip(" \\"))
        (specifier,) = requirement.specifier
        locked[canonicalize_name(requirement.name)] = Version(specifier.version)
    return locked


def _direct_requirements(path: Path) -> list[Requirement]:
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [Requirement(line) for line in lines if line and not line.startswith("-")]


def _assert_lock_satisfies(requirements: list[Requirement], locked: dict[str, Version]) -> None:
    unsatisfied = [
        str(requirement) for requirement in requirements
        if canonicalize_name(requirement.name) not in locked
        or not requirement.specifier.contains(locked[canonicalize_name(requirement.name)], prereleases=True)
    ]
    # 修改 requirements*.txt 后必须按锁文件头部命令重新生成，否则 CI 仍安装旧锁，改动不会生效。
    assert unsatisfied == []


def test_runtime_lock_satisfies_requirements() -> None:
    locked = _locked_versions(REPO_ROOT / "requirements.lock.txt")
    _assert_lock_satisfies(_direct_requirements(REPO_ROOT / "requirements.txt"), locked)


def test_dev_lock_keeps_runtime_versions_and_satisfies_dev_requirements() -> None:
    runtime = _locked_versions(REPO_ROOT / "requirements.lock.txt")
    development = _locked_versions(REPO_ROOT / "requirements-dev.lock.txt")
    # CI 先装运行时锁并冻结进发布包，再装测试锁；两者版本不一致会让测试环境偏离发布内容。
    assert {name: development.get(name) for name in runtime} == runtime
    _assert_lock_satisfies(_direct_requirements(REPO_ROOT / "requirements-dev.txt"), development)


def test_bootstrap_lock_pins_install_tools_without_changing_runtime_versions() -> None:
    runtime = _locked_versions(REPO_ROOT / "requirements.lock.txt")
    bootstrap = _locked_versions(REPO_ROOT / "requirements-bootstrap.lock.txt")
    # 发布锁来自 pip freeze --all，pip 与 wheel 不锁定时，镜像更新就会改变发布锁哈希并使服务器 venv 缓存失效。
    assert {"pip", "wheel"} <= bootstrap.keys()
    shared = bootstrap.keys() & runtime.keys()
    assert {name: bootstrap[name] for name in shared} == {name: runtime[name] for name in shared}
    _assert_lock_satisfies(_direct_requirements(REPO_ROOT / "requirements-bootstrap.txt"), bootstrap)


def test_runtime_lock_uses_cpu_torch_only() -> None:
    locked = _locked_versions(REPO_ROOT / "requirements.lock.txt")
    # 服务器没有 GPU，package_release.py 也会拒绝 CUDA 运行库；在锁文件阶段提前发现。
    assert {name: locked[name].local for name in ("torch", "torchvision")} == {"torch": "cpu", "torchvision": "cpu"}
    assert [name for name in locked if name.startswith("nvidia-") or name == "triton"] == []


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
