from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from services.storage.sqlite import SqliteConnectionProvider
from utils.storage_paths import (
    ARXIV_OAI_SYNC_META_FILE,
    ARXIV_OAI_SYNC_STATE_FILE,
    BACKEND_ARXIV_OAI_SYNC_ROOT,
    BACKEND_DAILY_ARXIV_ROOT,
    BACKEND_DATA_ROOT,
    BACKEND_EMBEDDED_DOCS_ROOT,
    BACKEND_GENERATION_RESULTS_ROOT,
    BACKEND_LOADED_DOCS_ROOT,
    BACKEND_ROOT,
    BACKEND_VECTOR_STORE_ROOT,
    LEGACY_BACKEND_ARTIFACT_ROOTS,
    LEGACY_DATABASE_ROOT,
    REPO_ROOT,
    StoragePathConfigurationError,
    resolve_backend_artifact_path,
    resolve_storage_path,
)


class StoragePathUnitTests(unittest.TestCase):
    def test_arxiv_oai_sync_state_is_under_backend_runtime_data(self) -> None:
        """增量同步游标属于运行时数据，生产部署时应随 backend/data 进入 shared。"""
        self.assertEqual(BACKEND_ARXIV_OAI_SYNC_ROOT.parent, BACKEND_ROOT / "data")
        self.assertEqual(ARXIV_OAI_SYNC_STATE_FILE.parent, BACKEND_ARXIV_OAI_SYNC_ROOT)
        self.assertEqual(ARXIV_OAI_SYNC_META_FILE.parent, BACKEND_ARXIV_OAI_SYNC_ROOT)

    def test_incremental_launchers_target_runtime_data_not_tools_directory(self) -> None:
        tools_dir = BACKEND_ROOT / "07-arxiv-tools"
        for name in ("sync_arxiv_oai_since_last_run.cmd", "sync_arxiv_oai_since_last_run_count_only.cmd", "sync_arxiv_oai_since_last_run.sh"):
            content = (tools_dir / name).read_text(encoding="utf-8")
            self.assertIn("arxiv-oai-sync", content)
            self.assertNotIn("STATE_FILE=%PROJECT_ROOT%sync_arxiv_oai_since_last_run", content)

    def test_linux_incremental_sync_has_systemd_timer_entrypoint(self) -> None:
        deploy_dir = REPO_ROOT / "deploy"
        service = (deploy_dir / "arxiv-oai-sync.service").read_text(encoding="utf-8")
        timer = (deploy_dir / "arxiv-oai-sync.timer").read_text(encoding="utf-8")
        self.assertIn("sync_arxiv_oai_since_last_run.sh", service)
        self.assertIn("OnCalendar=", timer)

    def test_relative_path_is_independent_of_current_working_directory(self) -> None:
        original_cwd = Path.cwd()
        try:
            os.chdir(REPO_ROOT)
            root_started_path = resolve_storage_path(
                "custom-state/recommendation.db",
                default_path=BACKEND_DATA_ROOT / "recommendation.db",
                option_name="SQLITE_DATABASE_PATH",
            )
            os.chdir(BACKEND_ROOT)
            backend_started_path = resolve_storage_path(
                "custom-state/recommendation.db",
                default_path=BACKEND_DATA_ROOT / "recommendation.db",
                option_name="SQLITE_DATABASE_PATH",
            )
        finally:
            os.chdir(original_cwd)

        expected_path = str((BACKEND_ROOT / "custom-state" / "recommendation.db").resolve())
        self.assertEqual(root_started_path, expected_path)
        self.assertEqual(backend_started_path, expected_path)

    def test_external_absolute_path_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            external_path = Path(temp_dir) / "recommendation.db"
            resolved_path = resolve_storage_path(
                external_path,
                default_path=BACKEND_DATA_ROOT / "recommendation.db",
                option_name="SQLITE_DATABASE_PATH",
            )

        self.assertEqual(resolved_path, str(external_path.resolve()))

    def test_legacy_root_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(StoragePathConfigurationError, "SQLITE_DATABASE_PATH"):
            resolve_storage_path(
                LEGACY_DATABASE_ROOT / "recommendation.db",
                default_path=BACKEND_DATA_ROOT / "recommendation.db",
                option_name="SQLITE_DATABASE_PATH",
            )

    def test_direct_sqlite_path_uses_the_same_resolution_rule(self) -> None:
        with tempfile.TemporaryDirectory(dir=BACKEND_ROOT) as temp_dir:
            relative_path = Path(temp_dir).relative_to(BACKEND_ROOT) / "nested" / "recommendation.db"
            provider = SqliteConnectionProvider(db_path=str(relative_path), check_same_thread=False)

            self.assertEqual(provider.db_path, str((BACKEND_ROOT / relative_path).resolve()))

    def test_backend_artifact_relative_paths_are_backend_scoped(self) -> None:
        cases = {
            "01-loaded-docs": BACKEND_LOADED_DOCS_ROOT,
            "02-embedded-docs": BACKEND_EMBEDDED_DOCS_ROOT,
            "03-vector-store": BACKEND_VECTOR_STORE_ROOT,
            "05-generation-results": BACKEND_GENERATION_RESULTS_ROOT,
            "06-daily-arxiv-paper": BACKEND_DAILY_ARXIV_ROOT,
        }

        original_cwd = Path.cwd()
        try:
            os.chdir(REPO_ROOT)
            for relative_path, expected_root in cases.items():
                with self.subTest(relative_path=relative_path):
                    resolved_path = resolve_backend_artifact_path(
                        relative_path,
                        option_name="TEST_ARTIFACT_DIR",
                    )
                    self.assertEqual(resolved_path, expected_root.resolve())
        finally:
            os.chdir(original_cwd)

    def test_legacy_root_artifact_path_is_redirected_to_backend(self) -> None:
        for legacy_name, backend_root in LEGACY_BACKEND_ARTIFACT_ROOTS.items():
            with self.subTest(legacy_name=legacy_name):
                resolved_path = resolve_backend_artifact_path(
                    REPO_ROOT / legacy_name / "nested",
                    option_name="TEST_ARTIFACT_DIR",
                )
                self.assertEqual(resolved_path, (backend_root / "nested").resolve())
