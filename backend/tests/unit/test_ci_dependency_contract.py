import ast
import re
from importlib.metadata import packages_distributions
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


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
