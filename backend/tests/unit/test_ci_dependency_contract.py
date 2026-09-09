from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_ci_pins_fastapi_to_router_compatible_version() -> None:
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    fastapi_entries = [line.strip() for line in requirements if line.strip().lower().startswith("fastapi")]

    # CI 会从零解析依赖；锁定已通过 startup smoke 的版本，避免新版 include_router 回归清空路由路径。
    assert fastapi_entries == ["fastapi==0.136.1"]
