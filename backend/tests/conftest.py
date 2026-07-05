from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"

for path in (REPO_ROOT, BACKEND_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


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
