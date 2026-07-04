from __future__ import annotations

import builtins
import importlib
import sys
import unittest
from unittest import mock


class ArxivOaiImportGuardTests(unittest.TestCase):
    def test_importing_oai_database_module_does_not_require_vector_store_dependency(self) -> None:
        module_name = "services.arxiv.arxiv_oai_service"
        saved_module = sys.modules.pop(module_name, None)
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            # 首页统计只需要 OAI SQLite 能力；如果这里又触发向量库导入，就说明懒加载边界被破坏了。
            if name in {"services.storage.vector_store_service", "pymilvus"}:
                raise AssertionError(f"unexpected eager import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                imported_module = importlib.import_module(module_name)
            self.assertTrue(hasattr(imported_module, "ArxivOaiDatabaseService"))
        finally:
            sys.modules.pop(module_name, None)
            if saved_module is not None:
                sys.modules[module_name] = saved_module


if __name__ == "__main__":
    unittest.main()
