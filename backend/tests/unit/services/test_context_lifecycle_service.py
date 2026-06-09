from __future__ import annotations

import gc
import unittest

from services.context_lifecycle import ContextLifecycleService
from services.storage.database_service import DatabaseService
from tests.helpers import build_database_service


class ContextLifecycleServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_service = build_database_service(DatabaseService)
        self.user_id = "user-1"
        self.service = ContextLifecycleService(
            db_service=self.db_service,
            lifecycle_config={
                "debug_snapshot_mode": "summary",
                "max_debug_string_chars": 40,
                "max_debug_list_items": 2,
                "max_debug_depth": 4,
                "max_trace_files_per_paper": 3,
            },
            checkpoint_config={"cleanup_retention_days": 7},
        )

    def tearDown(self) -> None:
        temp_db = getattr(self.db_service, "_test_temp_db", None)
        self.service = None
        self.db_service = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _create_paper_session(self) -> dict:
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00001",
                "title": "Lifecycle Paper",
                "authors": ["Alice"],
                "abstract": "A paper.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00001",
            }
        )
        return self.db_service.create_paper_chat_session(
            arxiv_id="2401.00001",
            user_id=self.user_id,
            session_id="paper-session-1",
            title="Lifecycle",
        )

    def test_debug_snapshot_summarizes_large_fields_without_removing_shape(self) -> None:
        payload = {
            "original_question": "Q",
            "raw_retrieval_top30": [{"content": "x" * 200}, {"content": "y" * 200}, {"content": "z" * 200}],
            "generation": {"prompt_context": {"used_chars": 100}},
        }

        snapshot = self.service.prepare_debug_snapshot(payload)

        self.assertEqual(snapshot["original_question"], "Q")
        self.assertEqual(len(snapshot["raw_retrieval_top30"]), 3)
        self.assertIn("truncated_items", snapshot["raw_retrieval_top30"][-1])
        self.assertTrue(snapshot["raw_retrieval_top30"][0]["content"]["truncated"])
        self.assertEqual(snapshot["generation"]["prompt_context"]["used_chars"], 100)

    def test_paper_qa_health_debug_reports_runtime_read_scale_and_policy(self) -> None:
        session = self._create_paper_session()
        self.db_service.append_paper_qa_turn(
            session_id=session["session_id"],
            user_id=self.user_id,
            question="Q",
            answer="A",
        )

        debug = self.service.build_paper_qa_health_debug(
            user_id=self.user_id,
            session_id=session["session_id"],
            short_term_debug={
                "db_message_read_count": 2,
                "db_message_read_limit": 8,
                "total_message_count": 20,
                "merged_turn_count": 1,
                "filtered_incomplete_turn_count": 0,
            },
            session_summary={"topic": "method", "summary_turn_count": 3},
            prompt_context_debug={"used_chars": 1200, "total_budget_chars": 28000, "estimated_tokens": 300},
        )

        self.assertEqual(debug["paper_chat"]["stored_message_count"], 2)
        self.assertEqual(debug["runtime_context"]["db_message_read_count"], 2)
        self.assertEqual(debug["runtime_context"]["total_message_count"], 20)
        self.assertTrue(debug["runtime_context"]["summary_loaded"])
        self.assertEqual(debug["prompt_budget"]["used_chars"], 1200)
        self.assertEqual(debug["policy"]["paper_chat_messages"]["retention"], "long_term")


if __name__ == "__main__":
    unittest.main()
