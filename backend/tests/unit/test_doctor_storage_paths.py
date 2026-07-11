from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils.storage_paths import LEGACY_DATABASE_ROOT


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCTOR_PATH = REPO_ROOT / "scripts" / "doctor.py"
DOCTOR_MODULE_NAME = "tests.unit._doctor_storage_paths"


def _load_doctor_module():
    spec = importlib.util.spec_from_file_location(DOCTOR_MODULE_NAME, DOCTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[DOCTOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


class DoctorStoragePathUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doctor = _load_doctor_module()

    def test_legacy_database_directory_is_a_non_blocking_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_root = Path(temp_dir) / "06-database"
            legacy_root.mkdir()
            with mock.patch.object(self.doctor, "LEGACY_DATABASE_ROOT", legacy_root):
                result = self.doctor.check_legacy_database_directory()

        self.assertEqual(result.status, "WARN")
        self.assertFalse(result.required)

    def test_legacy_backend_artifact_directories_are_non_blocking_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            legacy_root = temp_root / "03-vector-store"
            backend_root = temp_root / "backend" / "03-vector-store"
            legacy_root.mkdir()
            with (
                mock.patch.object(self.doctor, "REPO_ROOT", temp_root),
                mock.patch.object(
                    self.doctor,
                    "LEGACY_BACKEND_ARTIFACT_ROOTS",
                    {"03-vector-store": backend_root},
                ),
            ):
                result = self.doctor.check_legacy_backend_artifact_directories()

        self.assertEqual(result.status, "WARN")
        self.assertFalse(result.required)
        self.assertIn("03-vector-store", result.reason)

    def test_explicit_legacy_database_configuration_fails_doctor(self) -> None:
        previous_config_module = sys.modules.pop("utils.config", None)
        try:
            with mock.patch.dict(
                os.environ,
                {"SQLITE_DATABASE_PATH": str(LEGACY_DATABASE_ROOT / "recommendation.db")},
                clear=False,
            ):
                result = self.doctor.check_backend_config_loads()
        finally:
            sys.modules.pop("utils.config", None)
            if previous_config_module is not None:
                sys.modules["utils.config"] = previous_config_module

        self.assertEqual(result.status, "FAIL")
        self.assertIn("持久化路径配置无效", result.reason)
