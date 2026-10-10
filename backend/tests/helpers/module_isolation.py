"""在导入期加载 Agent 源码的测试桩隔离：加载完成后把项目公共模块恢复原样，避免污染后续测试。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator

# 这些顶层包会被测试桩替换或在桩环境下重新加载；backend.* 是测试专用的命名空间，允许跨文件复用。
_PROJECT_PACKAGES = ("dependencies", "tools", "utils", "services", "langgraph")
# tools.* 在桩环境下加载时会绑定到假的工具和依赖函数，即使来自真实源码文件也不能留给后续测试。
_ALWAYS_DISCARD_PACKAGES = ("tools",)
# 测试桩会直接给这些真实模块补属性或覆盖函数，需要连同属性一起恢复。
_ATTRIBUTE_SNAPSHOT_MODULES = ("dependencies", "utils.config")


def _top_level(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def _is_project_module(module_name: str) -> bool:
    return _top_level(module_name) in _PROJECT_PACKAGES


@contextmanager
def isolated_project_modules() -> Iterator[None]:
    saved_modules: Dict[str, Any] = {name: module for name, module in sys.modules.items() if _is_project_module(name)}
    saved_attributes = {
        name: dict(vars(sys.modules[name])) for name in _ATTRIBUTE_SNAPSHOT_MODULES if name in sys.modules
    }
    try:
        yield
    finally:
        for name in [name for name in sys.modules if _is_project_module(name) and name not in saved_modules]:
            module = sys.modules[name]
            # 加载期间正常导入的真实模块可以保留，避免后续重复导入产生两份类对象；桩模块和 tools.* 一律丢弃。
            if _top_level(name) in _ALWAYS_DISCARD_PACKAGES or not getattr(module, "__file__", None):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        for name, attributes in saved_attributes.items():
            module = sys.modules[name]
            vars(module).clear()
            vars(module).update(attributes)


__all__ = ["isolated_project_modules"]
