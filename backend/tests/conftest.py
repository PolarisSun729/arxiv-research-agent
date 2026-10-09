from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import dotenv
from starlette.config import Config


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"

for path in (REPO_ROOT, BACKEND_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# 开发者 .env 里的真实模型密钥会让 planner 调用付费 LLM，结果随本机配置漂移；只放行测试自建的配置文件。
_DEVELOPER_ENV_FILES = {(REPO_ROOT / name).resolve() for name in (".env", ".env.production", "backend/.env")}
_original_load_dotenv = dotenv.load_dotenv
_original_read_file = Config._read_file


def _is_developer_env_file(path) -> bool:
    return path is None or Path(path).resolve() in _DEVELOPER_ENV_FILES


def _load_test_dotenv(dotenv_path=None, *args, **kwargs):
    if _is_developer_env_file(dotenv_path):
        return False
    return _original_load_dotenv(dotenv_path, *args, **kwargs)


def _read_test_config_file(self, file_name, encoding="utf-8"):
    # SlowAPI 通过 Starlette Config 直接读取 .env，绕过 python-dotenv。
    if _is_developer_env_file(file_name):
        return {}
    return _original_read_file(self, file_name, encoding)


dotenv.load_dotenv = _load_test_dotenv
Config._read_file = _read_test_config_file

# 模型凭据也可能来自系统级环境变量；与 CI 一样在无凭据环境下运行，需要凭据的测试自行注入假值。
for _name in [name for name in os.environ if name.upper().endswith("_API_KEY")]:
    os.environ.pop(_name)
for _name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    os.environ.pop(_name, None)


def _preload_real_module(module_name: str) -> None:
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, "__file__", None):
        return
    if existing is not None:
        sys.modules.pop(module_name, None)
    importlib.import_module(module_name)


for module_name in ("utils.config", "services.memory", "services.storage.sqlite", "dependencies"):
    _preload_real_module(module_name)


def pytest_collect_file(file_path, parent):
    _ = file_path, parent
    for module_name in ("utils.config", "services.memory", "services.storage.sqlite", "dependencies"):
        _preload_real_module(module_name)
    return None
