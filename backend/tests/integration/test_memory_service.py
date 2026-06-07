import gc
import unittest
from unittest import mock

from services.memory.memory_debug import build_memory_debug_payload
from services.memory.memory_service import MemoryService
from services.storage.database_service import DatabaseService
from tests.helpers import build_database_service


class MemoryServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_service = build_database_service(DatabaseService)
        self.memory_service = MemoryService(db_service=self.db_service)
        self.user_id = "user-1"
        self.arxiv_id = "2401.00001"
        self._add_paper(self.arxiv_id)

    def tearDown(self) -> None:
        temp_db = getattr(self.db_service, "_test_temp_db", None)
        self.memory_service = None
        self.db_service = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _add_paper(self, arxiv_id: str, *, title: str = "Test Paper", categories=None) -> None:
        self.db_service.add_paper(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": ["Alice"],
                "abstract": f"Abstract for {title}",
                "categories": categories or ["cs.CL"],
                "published_date": "2024-01-01",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )

    def test_load_preference_summary_returns_stable_counts(self) -> None:
        self.db_service.add_liked_paper(self.user_id, self.arxiv_id)
        self.db_service.record_user_paper_action(self.user_id, "2401.00002", "bookmark")
        self.db_service.save_user_interest_vector(
            user_id=self.user_id,
            vector_data=[0.1, 0.2, 0.3],
            paper_count=1,
            embedding_model="fake-model",
            vector_dimension=3,
        )

        summary = self.memory_service.load_preference_summary(self.user_id)

        self.assertEqual(summary["user_id"], self.user_id)
        self.assertEqual(summary["counts"]["liked_papers"], 1)
        self.assertEqual(summary["counts"]["disliked_papers"], 0)
        self.assertIn("favorite", summary["paper_actions"])
        self.assertIsNotNone(summary["interest_vector"])

    def test_load_paper_chat_history_reads_recent_messages(self) -> None:
        session = self.db_service.create_paper_chat_session(arxiv_id=self.arxiv_id, user_id=self.user_id, title="QA")
        self.db_service.append_paper_chat_message(session["session_id"], "user", "What is the idea?", user_id=self.user_id, turn_id="turn-1")
        self.db_service.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "The paper studies retrieval.",
            user_id=self.user_id,
            turn_id="turn-1",
            sources=[{"source_id": "s1"}],
        )

        history = self.memory_service.load_paper_chat_history(self.user_id, self.arxiv_id, session_id=session["session_id"], limit=5)
        context = self.memory_service.load_paper_conversation_context(self.user_id, self.arxiv_id, session_id=session["session_id"], limit=5)

        self.assertEqual(history["selected_session"]["session_id"], session["session_id"])
        self.assertEqual(history["total_messages"], 2)
        self.assertEqual(len(history["messages"]), 2)
        self.assertEqual(context["turn_count"], 1)
        self.assertEqual(context["turns"][0]["turn_id"], "turn-1")

    def test_conversation_context_filters_incomplete_and_invalid_turns(self) -> None:
        session = self.db_service.create_paper_chat_session(arxiv_id=self.arxiv_id, user_id=self.user_id, title="QA")
        self.db_service.append_paper_chat_message(session["session_id"], "user", "Valid question?", user_id=self.user_id, turn_id="valid-turn")
        self.db_service.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "Valid answer.",
            user_id=self.user_id,
            turn_id="valid-turn",
            sources=[{"source_id": "s1"}],
        )
        self.db_service.append_paper_chat_message(session["session_id"], "user", "Only user", user_id=self.user_id, turn_id="user-only")
        self.db_service.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "Only assistant",
            user_id=self.user_id,
            turn_id="assistant-only",
        )
        self.db_service.append_paper_chat_message(session["session_id"], "user", "Duplicate one", user_id=self.user_id, turn_id="duplicate-role")
        self.db_service.append_paper_chat_message(session["session_id"], "user", "Duplicate two", user_id=self.user_id, turn_id="duplicate-role")
        self.db_service.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "Duplicate answer",
            user_id=self.user_id,
            turn_id="duplicate-role",
        )

        context = self.memory_service.load_paper_conversation_context(
            self.user_id,
            self.arxiv_id,
            session_id=session["session_id"],
            limit=5,
        )

        self.assertEqual(context["turn_count"], 1)
        self.assertEqual(context["turns"][0]["turn_id"], "valid-turn")
        self.assertEqual(context["filtered_incomplete_turn_count"], 2)
        self.assertEqual(context["invalid_turn_count"], 1)
        self.assertEqual(context["filtered_turn_count"], 3)

    def test_update_profile_from_note_merges_note_and_paper_signals(self) -> None:
        note = self.db_service.create_paper_note(
            user_id=self.user_id,
            arxiv_id=self.arxiv_id,
            title="Important finding",
            content="Remember this paper.",
            note_type="summary",
            tags=["rag", "retrieval"],
            include_in_profile=True,
        )

        profile = self.memory_service.update_profile_from_note(self.user_id, note)

        self.assertIn("rag", profile["positive_topics"])
        self.assertIn("Important finding", profile["recent_topics"])
        self.assertIn(self.arxiv_id, profile["representative_papers"])

    def test_build_memory_debug_payload_has_stable_fields(self) -> None:
        debug_payload = build_memory_debug_payload(
            user_id=self.user_id,
            arxiv_id=self.arxiv_id,
            user_profile={"positive_topics": ["rag"], "negative_topics": [], "recent_topics": [], "preferred_categories": ["cs.CL"]},
            preference_summary={"liked_papers": [self.arxiv_id], "disliked_papers": [], "paper_actions": {"favorite": [self.arxiv_id]}, "interest_vector": {"vector_data": [0.1]} , "counts": {"liked_papers": 1, "disliked_papers": 0}},
            paper_notes=[{"note_type": "summary"}],
            paper_chat_history={"sessions": [{"session_id": "s1"}], "messages": [{"message_id": "m1"}], "total_messages": 1, "selected_session": {"session_id": "s1"}},
            frontend_context={"selected_paper": {"arxiv_id": self.arxiv_id}},
            extra={"source": "test"},
        )

        self.assertEqual(debug_payload["user_id"], self.user_id)
        self.assertIn("loaded_sources", debug_payload)
        self.assertIn("preference_summary", debug_payload["loaded_sources"])
        self.assertEqual(debug_payload["loaded_sources"]["paper_chat_history"]["selected_session_id"], "s1")
        self.assertIn("selected_paper", debug_payload["frontend_context_keys"])
        self.assertEqual(debug_payload["extra"]["source"], "test")

    def test_db_exception_falls_back_to_safe_empty_memory_summary(self) -> None:
        with mock.patch.object(self.db_service, "_get_connection", side_effect=RuntimeError("db boom")):
            profile = self.memory_service.load_user_profile(self.user_id)
            summary = self.memory_service.load_preference_summary(self.user_id)

        self.assertEqual(profile["user_id"], self.user_id)
        self.assertEqual(summary["liked_papers"], [])
        self.assertEqual(summary["disliked_papers"], [])
        self.assertEqual(summary["paper_actions"], {})
        self.assertIsNone(summary["interest_vector"])

    def test_agent_memory_save_and_load_merges_backend_and_frontend_context(self) -> None:
        final_state = {
            "intent": "paper_qa",
            "answer": "The paper uses a retrieval pipeline.",
            "context": {
                "selected_paper": {"arxiv_id": self.arxiv_id, "title": "Test Paper"},
                "active_paper_session_id": "paper-session-1",
            },
            "paper_qa_result": {
                "arxiv_id": self.arxiv_id,
                "session_id": "paper-session-1",
                "answer": "retrieval pipeline",
            },
            "tool_calls": [
                {"tool_name": "check_paper_qa_index", "status": "success", "summary": "index ready"},
                {"tool_name": "answer_paper_question", "status": "success", "summary": "grounded answer"},
            ],
        }

        saved = self.memory_service.save_agent_memory(self.user_id, "agent-session-1", final_state)
        loaded = self.memory_service.load_agent_memory(
            self.user_id,
            "agent-session-1",
            frontend_context={"selected_paper": {"arxiv_id": "frontend-paper"}, "ui_state": "detail"},
        )

        self.assertIsNotNone(saved)
        self.assertEqual(saved["active_arxiv_id"], self.arxiv_id)
        self.assertEqual(saved["active_paper_session_id"], "paper-session-1")
        self.assertEqual(saved["last_intent"], "paper_qa")
        self.assertEqual(saved["last_tool_calls_summary"][0]["tool_name"], "check_paper_qa_index")
        self.assertEqual(loaded["session_id"], "agent-session-1")
        self.assertEqual(loaded["backend_memory"]["selected_paper"]["arxiv_id"], self.arxiv_id)
        # 前端上下文代表当前 UI 现场，应覆盖同名后端记忆字段，但后端记忆仍单独保留。
        self.assertEqual(loaded["merged_context"]["selected_paper"]["arxiv_id"], "frontend-paper")
        self.assertEqual(loaded["merged_context"]["ui_state"], "detail")


if __name__ == "__main__":
    unittest.main()
