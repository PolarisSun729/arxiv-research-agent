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

    def test_update_profile_from_note_merges_clean_note_tags_only(self) -> None:
        note = self.db_service.create_paper_note(
            user_id=self.user_id,
            arxiv_id=self.arxiv_id,
            title="Important finding about a single paper",
            content="Remember this paper.",
            note_type="summary",
            tags=["rag", "retrieval", "cs.CL", "paper", self.arxiv_id, "https://arxiv.org/abs/2401.00001"],
            include_in_profile=True,
        )

        profile = self.memory_service.update_profile_from_note(self.user_id, note)

        self.assertIn("rag", profile["positive_topics"])
        self.assertIn("retrieval", profile["recent_topics"])
        self.assertNotIn("Important finding about a single paper", profile["positive_topics"])
        self.assertNotIn("cs.CL", profile["positive_topics"])
        self.assertNotIn("paper", profile["positive_topics"])
        self.assertIn(self.arxiv_id, profile["representative_papers"])

    def test_like_paper_keeps_categories_and_titles_out_of_topics(self) -> None:
        profile = self.memory_service.update_profile_from_preference(
            self.user_id,
            self.arxiv_id,
            "like",
            paper_payload={
                "arxiv_id": self.arxiv_id,
                "title": "A Complete Paper Title That Should Not Become A Topic",
                "categories": ["cs.CL", "cs.AI"],
                "topics": ["RAG", "agent memory", "method", self.arxiv_id],
            },
        )

        self.assertIn("RAG", profile["positive_topics"])
        self.assertIn("agent memory", profile["recent_topics"])
        self.assertNotIn("cs.CL", profile["positive_topics"])
        self.assertNotIn("cs.AI", profile["recent_topics"])
        self.assertNotIn("A Complete Paper Title That Should Not Become A Topic", profile["positive_topics"])
        self.assertNotIn("method", profile["positive_topics"])
        self.assertIn("cs.CL", profile["preferred_categories"])
        self.assertIn("cs.AI", profile["preferred_categories"])
        self.assertIn(self.arxiv_id, profile["representative_papers"])

    def test_like_paper_without_explicit_topics_only_updates_category_and_representative_paper(self) -> None:
        profile = self.memory_service.update_profile_from_preference(
            self.user_id,
            "2401.00002",
            "like",
            paper_payload={
                "arxiv_id": "2401.00002",
                "title": "A Long Paper Title Should Not Become Recent Research Interest",
                "categories": "cs.CL cs.LG",
            },
        )

        self.assertEqual(profile["positive_topics"], [])
        self.assertEqual(profile["recent_topics"], [])
        self.assertIn("cs.CL", profile["preferred_categories"])
        self.assertIn("cs.LG", profile["preferred_categories"])
        self.assertIn("2401.00002", profile["representative_papers"])

    def test_dislike_paper_does_not_store_categories_or_titles_as_negative_topics(self) -> None:
        profile = self.memory_service.update_profile_from_preference(
            self.user_id,
            self.arxiv_id,
            "dislike",
            paper_payload={
                "arxiv_id": self.arxiv_id,
                "title": "Another Full Paper Title That Should Stay Out",
                "categories": ["cs.CL"],
                "topics": ["vision-only generation", "framework"],
            },
        )

        self.assertIn("vision-only generation", profile["negative_topics"])
        self.assertNotIn("cs.CL", profile["negative_topics"])
        self.assertNotIn("Another Full Paper Title That Should Stay Out", profile["negative_topics"])
        self.assertNotIn("framework", profile["negative_topics"])

    def test_generate_user_research_profile_summarizes_behavior_evidence(self) -> None:
        self._add_paper(
            "2401.00010",
            title="RAG Retrieval Optimization for Long-Context Reasoning Agents",
            categories=["cs.CL", "cs.AI"],
        )
        self._add_paper(
            "2401.00011",
            title="Agent Memory for Tool-Using Language Models",
            categories=["cs.AI"],
        )
        self._add_paper(
            "2401.00012",
            title="Memory-Augmented RAG Agents",
            categories=["cs.CL", "cs.LG"],
        )
        self._add_paper(
            "2401.00013",
            title="Diffusion Models for Image Generation",
            categories=["cs.CV"],
        )
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00014",
                "title": "Knowledge Graph Construction for Scientific QA",
                "authors": ["Alice"],
                "abstract": "Entity extraction and relation extraction build knowledge graph evidence for question answering.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00014",
            }
        )
        # 用完整摘要作为证据输入，生成器应抽象出稳定主题，而不是保存这些标题。
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00010",
                "title": "RAG Retrieval Optimization for Long-Context Reasoning Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation systems improve hybrid retrieval, reranking, and long context reasoning for LLM agents.",
                "categories": ["cs.CL", "cs.AI"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00010",
            }
        )
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00011",
                "title": "Agent Memory for Tool-Using Language Models",
                "authors": ["Alice"],
                "abstract": "Long-term memory and episodic memory help autonomous agents plan with tool use and function calling.",
                "categories": ["cs.AI"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00011",
            }
        )
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00012",
                "title": "Memory-Augmented RAG Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval augmented generation and agent memory support long-context reasoning workflows.",
                "categories": ["cs.CL", "cs.LG"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00012",
            }
        )
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00013",
                "title": "Diffusion Models for Image Generation",
                "authors": ["Alice"],
                "abstract": "Diffusion models focus on text-to-image generation and visual synthesis.",
                "categories": ["cs.CV"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00013",
            }
        )
        self.db_service.upsert_user_research_profile(
            self.user_id,
            {"preferred_answer_style": "concise", "common_question_types": ["summary"]},
        )
        for arxiv_id in ("2401.00010", "2401.00011", "2401.00012"):
            self.db_service.add_liked_paper(self.user_id, arxiv_id)
        self.db_service.add_disliked_paper(self.user_id, "2401.00013")
        self.db_service.record_user_paper_action(self.user_id, "2401.00014", "favorite")
        self.db_service.create_paper_note(
            user_id=self.user_id,
            arxiv_id="2401.00014",
            title="Do not store this note title as a topic",
            content="Use this note as supporting evidence for knowledge graph construction.",
            note_type="method",
            tags=["knowledge graph construction", "RAG", "paper"],
            include_in_profile=True,
        )

        profile = self.memory_service.generate_user_research_profile(self.user_id)
        repeated = self.memory_service.generate_user_research_profile(self.user_id)

        self.assertIn("RAG retrieval optimization", profile["positive_topics"])
        self.assertIn("agent memory", profile["positive_topics"])
        self.assertIn("long-context reasoning", profile["positive_topics"])
        self.assertIn("knowledge graph construction", profile["recent_topics"])
        self.assertIn("diffusion models", profile["negative_topics"])
        self.assertIn("cs.CL", profile["preferred_categories"])
        self.assertIn("cs.AI", profile["preferred_categories"])
        self.assertNotIn("cs.CL", profile["positive_topics"])
        self.assertNotIn("cs.CV", profile["negative_topics"])
        self.assertNotIn("Do not store this note title as a topic", profile["positive_topics"])
        self.assertIn("2401.00010", profile["representative_papers"])
        self.assertIn("method", profile["common_question_types"])
        self.assertEqual(profile["preferred_answer_style"], "concise")
        self.assertEqual(profile["positive_topics"], repeated["positive_topics"])
        self.assertEqual(profile["negative_topics"], repeated["negative_topics"])

    def test_rebuild_user_research_profile_cleans_existing_dirty_profile(self) -> None:
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00110",
                "title": "RAG Retrieval Optimization for Long-Context Reasoning Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and long context reasoning for agents.",
                "categories": ["cs.CL", "cs.AI"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00110",
            }
        )
        self.db_service.add_paper(
            {
                "arxiv_id": "2401.00111",
                "title": "Diffusion Models for Image Generation",
                "authors": ["Alice"],
                "abstract": "Diffusion models focus on text-to-image visual generation.",
                "categories": ["cs.CV"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00111",
            }
        )
        self.db_service.upsert_user_research_profile(
            self.user_id,
            {
                "positive_topics": ["cs.CL", "RAG Retrieval Optimization for Long-Context Reasoning Agents", "manual retrieval topic"],
                "negative_topics": ["cs.AI", "Diffusion Models for Image Generation"],
                "recent_topics": ["A Complete Paper Title That Should Be Removed"],
                "preferred_categories": ["cs.SE"],
                "preferred_answer_style": "先结论后细节",
                "common_question_types": ["summary"],
                "representative_papers": ["RAG Retrieval Optimization for Long-Context Reasoning Agents"],
            },
        )
        self.db_service.add_liked_paper(self.user_id, "2401.00110")
        self.db_service.add_disliked_paper(self.user_id, "2401.00111")

        profile = self.memory_service.rebuild_user_research_profile(self.user_id)
        repeated = self.memory_service.rebuild_user_research_profile(self.user_id)

        self.assertIn("RAG retrieval optimization", profile["positive_topics"])
        self.assertIn("manual retrieval topic", profile["positive_topics"])
        self.assertNotIn("cs.CL", profile["positive_topics"])
        self.assertNotIn("RAG Retrieval Optimization for Long-Context Reasoning Agents", profile["positive_topics"])
        self.assertNotIn("cs.AI", profile["negative_topics"])
        self.assertNotIn("Diffusion Models for Image Generation", profile["negative_topics"])
        self.assertIn("diffusion models", profile["negative_topics"])
        self.assertIn("cs.CL", profile["preferred_categories"])
        self.assertIn("cs.AI", profile["preferred_categories"])
        self.assertIn("cs.SE", profile["preferred_categories"])
        self.assertEqual(profile["representative_papers"], ["2401.00110"])
        self.assertEqual(profile["preferred_answer_style"], "先结论后细节")
        self.assertEqual(profile["positive_topics"], repeated["positive_topics"])
        self.assertEqual(profile["representative_papers"], repeated["representative_papers"])

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

    def test_agent_memory_prefers_current_paper_qa_target_over_stale_selected_paper(self) -> None:
        final_state = {
            "intent": "paper_qa",
            "answer": "Second paper answer.",
            "context": {
                # 前端默认 selected_paper 可能仍是搜索列表第一篇；会话焦点应以本轮 QA 目标为准。
                "selected_paper": {"arxiv_id": "2401.00001", "title": "First Paper"},
                "last_papers": [
                    {"arxiv_id": "2401.00001", "title": "First Paper"},
                    {"arxiv_id": "2401.00002", "title": "Second Paper"},
                ],
            },
            "paper_qa_result": {
                "arxiv_id": "2401.00002",
                "title": "Second Paper",
                "answer": "grounded answer",
            },
        }

        saved = self.memory_service.save_agent_memory(self.user_id, "agent-session-ordinal", final_state)

        self.assertEqual(saved["active_arxiv_id"], "2401.00002")
        self.assertEqual(saved["selected_paper"]["arxiv_id"], "2401.00002")
        self.assertEqual(saved["selected_paper"]["title"], "Second Paper")


if __name__ == "__main__":
    unittest.main()
