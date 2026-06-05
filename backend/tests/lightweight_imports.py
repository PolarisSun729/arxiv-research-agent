from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


_LIGHTWEIGHT_UTILS_PACKAGE = "_pytest_arxiv_search_agent_utils"


def load_arxiv_utils_module(module_name: str):
    """Load arxiv_search_agent utility modules without importing the full agent package."""
    repo_root = Path(__file__).resolve().parents[2]
    utils_dir = repo_root / "backend" / "agents" / "arxiv_search_agent" / "utils"

    if _LIGHTWEIGHT_UTILS_PACKAGE not in sys.modules:
        package = types.ModuleType(_LIGHTWEIGHT_UTILS_PACKAGE)
        package.__path__ = [str(utils_dir)]
        sys.modules[_LIGHTWEIGHT_UTILS_PACKAGE] = package

    return importlib.import_module(f"{_LIGHTWEIGHT_UTILS_PACKAGE}.{module_name}")
