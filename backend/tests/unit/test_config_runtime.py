import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_config_module(module_name: str, env: dict[str, str] | None = None):
    backend_dir = Path(__file__).resolve().parents[2]
    utils_dir = backend_dir / "utils"

    if "utils" not in sys.modules:
        utils_package = types.ModuleType("utils")
        utils_package.__path__ = [str(utils_dir)]
        sys.modules["utils"] = utils_package

    spec = importlib.util.spec_from_file_location(module_name, utils_dir / "config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    with mock.patch.dict(os.environ, env or {}, clear=True):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class ConfigRuntimeUnitTests(unittest.TestCase):
    def test_env_helpers_parse_string_int_and_bool(self) -> None:
        module = _load_config_module("tests.unit._config_parse")

        with mock.patch.dict(
            os.environ,
            {
                "TEST_STR": "  hello  ",
                "TEST_INT": "42",
                "TEST_INT_BAD": "not-a-number",
                "TEST_BOOL_TRUE": "yes",
                "TEST_BOOL_FALSE": "off",
            },
            clear=True,
        ):
            self.assertEqual(module._env_str("TEST_STR", "x"), "hello")
            self.assertEqual(module._env_int("TEST_INT", 7), 42)
            self.assertEqual(module._env_int("TEST_INT_BAD", 7), 7)
            self.assertTrue(module._env_bool("TEST_BOOL_TRUE", False))
            self.assertFalse(module._env_bool("TEST_BOOL_FALSE", True))

    def test_default_config_values_are_applied_when_env_missing(self) -> None:
        module = _load_config_module("tests.unit._config_defaults")

        self.assertEqual(module.CORE_CONFIG["arxiv_data_source"], "local")
        self.assertEqual(module.CORE_CONFIG["service_load_mode"], "preload")
        self.assertEqual(module.get_default_user_id(), "local_user")
        self.assertFalse(module.SQLITE_CONFIG["check_same_thread"])

    def test_runtime_config_accessors_return_copies(self) -> None:
        module = _load_config_module("tests.unit._config_copies", {"ARXIV_SEARCH_MAX_ALLOWED_RESULTS": "12"})

        runtime_config = module.get_arxiv_search_runtime_config()
        runtime_config["max_allowed_results"] = 99

        self.assertEqual(module.ARXIV_SEARCH_CONFIG["max_allowed_results"], 12)


if __name__ == "__main__":
    unittest.main()
