import gc
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from services.storage.sqlite.shared import PaperQATurnPersistenceError
from tests.helpers import build_storage_container


class StorageContainerSqliteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = build_storage_container()
        self.user_id = "user-1"

    def tearDown(self) -> None:
        temp_db = getattr(self.storage, "_test_temp_db", None)
        self.storage = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _add_sample_paper(self, arxiv_id: str = "2401.00001") -> dict:
        paper = {
            "arxiv_id": arxiv_id,
            "title": f"Paper {arxiv_id}",
            "authors": ["Alice", "Bob"],
            "abstract": "A sample abstract.",
            "categories": ["cs.CL", "cs.IR"],
            "published_date": "2024-01-01",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }
        self.assertTrue(self.storage.paper_catalog.add_paper(paper))
        return paper

    def _create_session(self, arxiv_id: str = "2401.00001", session_id: str = "session-1") -> dict:
        self._add_sample_paper(arxiv_id)
        session = self.storage.paper_chat_sessions.create_paper_chat_session(
            arxiv_id=arxiv_id,
            user_id=self.user_id,
            title="Initial title",
            session_id=session_id,
        )
        self.assertIsNotNone(session)
        return session

    def test_initialization_creates_expected_tables(self) -> None:
        with self.storage.connection_provider.connect() as conn:
            table_names = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }

        self.assertIn("arxiv_papers", table_names)
        self.assertIn("paper_qa_index", table_names)
        self.assertIn("paper_index_jobs", table_names)
        self.assertIn("paper_chat_sessions", table_names)
        self.assertIn("paper_chat_messages", table_names)
        self.assertIn("paper_notes", table_names)
        self.assertIn("user_research_profiles", table_names)
        self.assertIn("user_profile_events", table_names)
        self.assertIn("paper_profile_evidence", table_names)
        self.assertIn("user_manual_profiles", table_names)
        self.assertIn("user_generated_profiles", table_names)
        self.assertIn("user_effective_profiles", table_names)
        self.assertIn("user_profile_snapshots", table_names)
        self.assertIn("user_profile_build_jobs", table_names)

    def test_arxiv_papers_support_add_get_and_delete(self) -> None:
        paper = self._add_sample_paper()

        stored = self.storage.paper_catalog.get_paper(paper["arxiv_id"])

        self.assertEqual(stored["arxiv_id"], paper["arxiv_id"])
        self.assertEqual(stored["authors"], ["Alice", "Bob"])
        self.assertEqual(stored["categories"], ["cs.CL", "cs.IR"])
        self.assertEqual(len(self.storage.paper_catalog.get_all_papers()), 1)
        self.assertTrue(self.storage.paper_catalog.delete_paper(paper["arxiv_id"]))
        self.assertIsNone(self.storage.paper_catalog.get_paper(paper["arxiv_id"]))

    def test_liked_and_disliked_papers_are_mutually_exclusive(self) -> None:
        paper = self._add_sample_paper()

        self.assertTrue(self.storage.user_preferences.add_liked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"]))
        self.assertEqual(self.storage.user_preferences.get_liked_papers(self.user_id), [paper["arxiv_id"]])
        self.assertEqual(self.storage.user_preferences.get_disliked_papers(self.user_id), [])

        self.assertTrue(self.storage.user_preferences.add_disliked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"]))
        self.assertEqual(self.storage.user_preferences.get_liked_papers(self.user_id), [])
        self.assertEqual(self.storage.user_preferences.get_disliked_papers(self.user_id), [paper["arxiv_id"]])

        self.assertTrue(self.storage.user_preferences.add_liked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"]))
        self.assertEqual(self.storage.user_preferences.get_liked_papers(self.user_id), [paper["arxiv_id"]])
        self.assertEqual(self.storage.user_preferences.get_disliked_papers(self.user_id), [])

    def test_record_user_paper_action_supports_aliases_and_rejects_invalid_values(self) -> None:
        paper = self._add_sample_paper("2401.00002")

        self.assertFalse(self.storage.user_preferences.record_user_paper_action(self.user_id, paper["arxiv_id"], "liked"))
        self.assertTrue(self.storage.user_preferences.record_user_paper_action(self.user_id, paper["arxiv_id"], "bookmark"))
        self.assertTrue(self.storage.user_preferences.record_user_paper_action(self.user_id, paper["arxiv_id"], "save_for_later"))
        self.assertFalse(self.storage.user_preferences.record_user_paper_action(self.user_id, paper["arxiv_id"], "disliked"))
        self.assertFalse(self.storage.user_preferences.record_user_paper_action(self.user_id, paper["arxiv_id"], "invalid_action"))

        actions = self.storage.user_preferences.get_user_paper_actions(self.user_id)
        action_types = {item["action_type"] for item in actions}

        self.assertIn("favorite", action_types)
        self.assertIn("later", action_types)
        self.assertNotIn("dislike", action_types)
        self.assertNotIn("like", action_types)

    def test_profile_events_are_append_only_and_deduped_by_semantic_target(self) -> None:
        paper = self._add_sample_paper("2401.00999")

        self.assertTrue(self.storage.user_preferences.add_liked_paper(self.user_id, paper["arxiv_id"]))
        self.assertTrue(self.storage.user_preferences.add_liked_paper(self.user_id, paper["arxiv_id"]))
        self.assertTrue(self.storage.user_preferences.record_user_paper_action(self.user_id, paper["arxiv_id"], "read"))
        events = self.storage.profile_events.list_user_profile_events(self.user_id)
        liked_events = [event for event in events if event["event_type"] == "liked"]
        read_events = [event for event in events if event["event_type"] == "read"]

        self.assertEqual(len(liked_events), 1)
        self.assertEqual(len(read_events), 1)
        self.assertEqual(liked_events[0]["arxiv_id"], paper["arxiv_id"])
        self.assertGreater(liked_events[0]["action_strength"], read_events[0]["action_strength"])
        self.assertTrue(liked_events[0]["include_in_profile"])
        self.assertNotIn("like", self.storage.user_preferences.get_user_paper_action_map(self.user_id))

    def test_removing_preference_deactivates_profile_event_without_action_residue(self) -> None:
        paper = self._add_sample_paper("2401.00998")

        self.assertTrue(self.storage.user_preferences.add_liked_paper(self.user_id, paper["arxiv_id"]))
        self.assertTrue(self.storage.user_preferences.remove_liked_paper(self.user_id, paper["arxiv_id"]))

        self.assertEqual(self.storage.user_preferences.get_liked_papers(self.user_id), [])
        self.assertNotIn("like", self.storage.user_preferences.get_user_paper_action_map(self.user_id))
        active_events = self.storage.profile_events.list_user_profile_events(self.user_id)
        self.assertNotIn("liked", {event["event_type"] for event in active_events})
        self.assertIn("liked_removed", {event["event_type"] for event in active_events})

    def test_get_user_preferences_returns_stable_structure(self) -> None:
        paper = self._add_sample_paper("2401.00003")
        self.storage.user_preferences.add_liked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"])

        preferences = self.storage.user_preferences.get_user_preferences(self.user_id)

        self.assertEqual(
            set(preferences.keys()),
            {"user_id", "liked_papers", "disliked_papers", "paper_actions", "research_profile"},
        )
        self.assertEqual(preferences["user_id"], self.user_id)
        self.assertEqual(preferences["liked_papers"], [paper["arxiv_id"]])
        self.assertIsInstance(preferences["paper_actions"], dict)
        self.assertIsInstance(preferences["research_profile"], dict)

    def test_paper_qa_index_can_be_inserted_updated_and_queried(self) -> None:
        arxiv_id = "2401.00004"

        self.assertTrue(
            self.storage.paper_qa_index.insert_paper_qa_index(
                arxiv_id,
                collection_name="paper_2401_00004",
                status="building",
                chunk_count=2,
                embedding_model="fake-model",
                pdf_path="/tmp/paper.pdf",
                chunk_file="/tmp/chunks.json",
                retrieval_index_file="/tmp/retrieval_indexes.json",
                retrieval_index_count=5,
                retrieval_index_types='["body","section_anchor"]',
                retrieval_index_version="retrieval-v1",
                sparse_index_dir="/tmp/sparse-index",
                sparse_index_manifest_file="/tmp/sparse-index/manifest.json",
                sparse_index_document_count=6,
                sparse_index_token_count=42,
                sparse_index_backend="internal_bm25",
                sparse_index_schema_version="sparse_index_artifact_v1",
                sparse_index_source_file="/tmp/chunks.json",
                sparse_index_source_hash="abc123",
                sparse_index_avgdl=12.5,
                embedding_file="/tmp/embeddings.json",
                loading_method="docling",
                chunking_strategy="docling_sections",
                current_stage="save_embeddings",
                failed_stage="",
                error_message="",
            )
        )
        self.assertTrue(
            self.storage.paper_qa_index.update_paper_qa_index(
                arxiv_id,
                status="indexed",
                chunk_count=3,
                collection_name="paper_2401_00004_v2",
                indexed_at="2024-01-01T00:00:00",
            )
        )

        qa_index = self.storage.paper_qa_index.get_paper_qa_index(arxiv_id)

        self.assertEqual(qa_index["collection_name"], "paper_2401_00004_v2")
        self.assertEqual(qa_index["status"], "indexed")
        self.assertEqual(qa_index["chunk_count"], 3)
        self.assertEqual(qa_index["chunk_file"], "/tmp/chunks.json")
        self.assertEqual(qa_index["retrieval_index_file"], "/tmp/retrieval_indexes.json")
        self.assertEqual(qa_index["retrieval_index_count"], 5)
        self.assertEqual(qa_index["retrieval_index_types"], '["body","section_anchor"]')
        self.assertEqual(qa_index["retrieval_index_version"], "retrieval-v1")
        self.assertEqual(qa_index["sparse_index_dir"], "/tmp/sparse-index")
        self.assertEqual(qa_index["sparse_index_manifest_file"], "/tmp/sparse-index/manifest.json")
        self.assertEqual(qa_index["sparse_index_document_count"], 6)
        self.assertEqual(qa_index["sparse_index_token_count"], 42)
        self.assertEqual(qa_index["sparse_index_backend"], "internal_bm25")
        self.assertEqual(qa_index["sparse_index_schema_version"], "sparse_index_artifact_v1")
        self.assertEqual(qa_index["sparse_index_source_file"], "/tmp/chunks.json")
        self.assertEqual(qa_index["sparse_index_source_hash"], "abc123")
        self.assertEqual(qa_index["sparse_index_avgdl"], 12.5)
        self.assertEqual(qa_index["embedding_file"], "/tmp/embeddings.json")
        self.assertEqual(qa_index["loading_method"], "docling")
        self.assertEqual(qa_index["chunking_strategy"], "docling_sections")
        self.assertEqual(qa_index["current_stage"], "save_embeddings")
        self.assertEqual(qa_index["indexed_at"], "2024-01-01T00:00:00")

    def test_paper_qa_index_build_activation_keeps_single_active_version(self) -> None:
        arxiv_id = "2401.01000"
        self.assertTrue(
            self.storage.paper_qa_index.insert_paper_qa_index(
                arxiv_id,
                collection_name="qa_old",
                status="indexed",
                chunk_count=1,
                embedding_model="old-model",
            )
        )
        old_active = self.storage.paper_qa_index.get_active_paper_qa_index_build(arxiv_id)
        build = self.storage.paper_qa_index.create_paper_qa_index_build(arxiv_id, "docling")
        self.assertTrue(
            self.storage.paper_qa_index.update_paper_qa_index_build(
                build["build_id"],
                status="build_success",
                collection_name="qa_new",
                chunk_count=2,
                embedding_model="new-model",
                retrieval_index_file="retrieval-new.json",
                retrieval_index_count=4,
                retrieval_index_types='["body","summary"]',
                retrieval_index_version="retrieval-new",
                sparse_index_dir="sparse-new",
                sparse_index_manifest_file="sparse-new/manifest.json",
                sparse_index_document_count=4,
                sparse_index_token_count=33,
                sparse_index_backend="internal_bm25",
                sparse_index_schema_version="sparse_index_artifact_v1",
                sparse_index_source_file="chunks-new.json",
                sparse_index_source_hash="hash-new",
                sparse_index_avgdl=9.25,
                current_stage="activate_index",
            )
        )

        self.assertTrue(self.storage.paper_qa_index.activate_paper_qa_index_build(build["build_id"]))

        active = self.storage.paper_qa_index.get_paper_qa_index(arxiv_id)
        active_versions = self.storage.paper_qa_index.list_paper_qa_index_builds(arxiv_id, statuses=["active"], limit=10)
        cleanup_versions = self.storage.paper_qa_index.list_paper_qa_index_builds(arxiv_id, statuses=["cleanup_pending"], limit=10)
        self.assertEqual(active["collection_name"], "qa_new")
        self.assertEqual(active["active_build_id"], build["build_id"])
        self.assertEqual(active["retrieval_index_file"], "retrieval-new.json")
        self.assertEqual(active["retrieval_index_count"], 4)
        self.assertEqual(active["sparse_index_manifest_file"], "sparse-new/manifest.json")
        self.assertEqual(active["sparse_index_document_count"], 4)
        self.assertEqual(active["sparse_index_token_count"], 33)
        self.assertEqual(active["sparse_index_source_hash"], "hash-new")
        self.assertEqual(active_versions[0]["retrieval_index_file"], "retrieval-new.json")
        self.assertEqual(active_versions[0]["sparse_index_manifest_file"], "sparse-new/manifest.json")
        self.assertEqual(active_versions[0]["sparse_index_token_count"], 33)
        self.assertEqual(len(active_versions), 1)
        self.assertEqual(cleanup_versions[0]["build_id"], old_active["build_id"])

    def test_paper_qa_index_build_activation_requires_sparse_artifact(self) -> None:
        arxiv_id = "2401.01002"
        self.assertTrue(
            self.storage.paper_qa_index.insert_paper_qa_index(
                arxiv_id,
                collection_name="qa_old_sparse_guard",
                status="indexed",
                chunk_count=1,
                embedding_model="old-model",
            )
        )
        old_active = self.storage.paper_qa_index.get_active_paper_qa_index_build(arxiv_id)
        build = self.storage.paper_qa_index.create_paper_qa_index_build(arxiv_id, "docling")
        self.assertTrue(
            self.storage.paper_qa_index.update_paper_qa_index_build(
                build["build_id"],
                status="build_success",
                collection_name="qa_new_missing_sparse",
                chunk_count=2,
                embedding_model="new-model",
                retrieval_index_file="retrieval-new.json",
                retrieval_index_count=4,
                retrieval_index_types='["body","summary"]',
                retrieval_index_version="retrieval-new",
                current_stage="activate_index",
            )
        )

        self.assertFalse(self.storage.paper_qa_index.activate_paper_qa_index_build(build["build_id"]))

        active = self.storage.paper_qa_index.get_paper_qa_index(arxiv_id)
        new_build = self.storage.paper_qa_index.get_paper_qa_index_build(build["build_id"])
        cleanup_versions = self.storage.paper_qa_index.list_paper_qa_index_builds(arxiv_id, statuses=["cleanup_pending"], limit=10)
        self.assertEqual(active["collection_name"], "qa_old_sparse_guard")
        self.assertEqual(active["active_build_id"], old_active["build_id"])
        self.assertEqual(new_build["status"], "build_success")
        self.assertEqual(cleanup_versions, [])

    def test_legacy_qa_index_row_is_recognized_as_active_version(self) -> None:
        arxiv_id = "2401.01001"
        self.assertTrue(
            self.storage.paper_qa_index.insert_paper_qa_index(
                arxiv_id,
                collection_name="qa_legacy",
                status="indexed",
                chunk_count=1,
                embedding_model="legacy-model",
            )
        )

        active = self.storage.paper_qa_index.get_active_paper_qa_index_build(arxiv_id)

        self.assertIsNotNone(active)
        self.assertTrue(active["is_active"])
        self.assertEqual(active["collection_name"], "qa_legacy")

    def test_paper_index_jobs_support_create_update_and_latest_query(self) -> None:
        job = self.storage.paper_qa_index.create_paper_index_job("2401.00005", loading_method="docling")

        self.assertIsNotNone(job)
        self.assertTrue(
            self.storage.paper_qa_index.update_paper_index_job(
                job["job_id"],
                status="completed",
                current_stage="finished",
                progress=150,
                error_message="",
            )
        )

        stored = self.storage.paper_qa_index.get_paper_index_job(job["job_id"])
        latest = self.storage.paper_qa_index.get_latest_paper_index_job("2401.00005")

        self.assertEqual(stored["status"], "completed")
        self.assertEqual(stored["current_stage"], "finished")
        self.assertEqual(stored["progress"], 100)
        self.assertEqual(stored["idempotency_key"], "2401.00005:docling")
        self.assertIsNotNone(stored["heartbeat_at"])
        self.assertEqual(latest["job_id"], job["job_id"])

    def test_paper_chat_sessions_support_create_list_clear_and_delete(self) -> None:
        session = self._create_session(arxiv_id="2401.00006", session_id="session-clear")
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "user", "question", user_id=self.user_id)
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "assistant", "answer", user_id=self.user_id)

        sessions = self.storage.paper_chat_sessions.list_paper_chat_sessions("2401.00006", user_id=self.user_id)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["message_count"], 2)

        self.assertTrue(self.storage.paper_chat_messages.clear_paper_chat_session(session["session_id"], user_id=self.user_id))
        cleared_messages = self.storage.paper_chat_messages.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        cleared_session = self.storage.paper_chat_sessions.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertEqual(cleared_messages, [])
        self.assertEqual(cleared_session["message_count"], 0)

        self.assertTrue(self.storage.paper_chat_sessions.delete_paper_chat_session(session["session_id"], user_id=self.user_id))
        self.assertIsNone(self.storage.paper_chat_sessions.get_paper_chat_session(session["session_id"], user_id=self.user_id))

    def test_paper_chat_messages_preserve_turn_id_and_json_payloads(self) -> None:
        session = self._create_session(arxiv_id="2401.00007", session_id="session-msg")
        turn_id = "turn-1"
        user_message = self.storage.paper_chat_messages.append_paper_chat_message(
            session["session_id"],
            "user",
            "What is the method?",
            user_id=self.user_id,
            turn_id=turn_id,
            contextualized_question="ctx question",
            question_contextualization={"history": 1},
        )
        assistant_message = self.storage.paper_chat_messages.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "It uses retrieval.",
            user_id=self.user_id,
            turn_id=turn_id,
            sources=[{"chunk_id": "c1"}],
            retrieval_debug_snapshot={"score": 0.8},
        )

        messages = self.storage.paper_chat_messages.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        by_turn = self.storage.paper_chat_messages.get_paper_chat_message_by_turn(session["session_id"], turn_id, role="assistant", user_id=self.user_id)

        self.assertEqual(len(messages), 2)
        self.assertEqual(user_message["turn_id"], turn_id)
        self.assertEqual(assistant_message["sources"], [{"chunk_id": "c1"}])
        self.assertEqual(assistant_message["retrieval_debug_snapshot"], {"score": 0.8})
        self.assertEqual(by_turn["message_id"], assistant_message["message_id"])

    def test_paper_chat_session_summary_round_trips_on_session_only(self) -> None:
        session = self._create_session(arxiv_id="2401.00015", session_id="session-summary")
        other = self._create_session(arxiv_id="2401.00015", session_id="session-summary-other")
        summary = {
            "topic": "method details",
            "confirmed_facts": ["fact 1"],
            "user_preferences": ["focus on experiments"],
            "task_progress": ["compared baselines"],
            "source_clues": [{"source_id": "s1", "section_path": "Experiments"}],
        }

        updated = self.storage.paper_chat_sessions.update_paper_chat_session_summary(
            session["session_id"],
            user_id=self.user_id,
            summary=summary,
            summary_turn_count=3,
            summary_last_turn_id="turn-3",
            summary_updated_at="2026-06-09T00:00:00+00:00",
        )
        refreshed = self.storage.paper_chat_sessions.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        other_refreshed = self.storage.paper_chat_sessions.get_paper_chat_session(other["session_id"], user_id=self.user_id)

        self.assertTrue(updated)
        self.assertEqual(refreshed["summary"]["topic"], "method details")
        self.assertEqual(refreshed["summary_turn_count"], 3)
        self.assertEqual(refreshed["summary_last_turn_id"], "turn-3")
        self.assertEqual(refreshed["summary_updated_at"], "2026-06-09T00:00:00+00:00")
        self.assertIsNone(other_refreshed["summary"])

    def test_recent_paper_chat_messages_limit_at_database_layer_and_restore_order(self) -> None:
        session = self._create_session(arxiv_id="2401.00016", session_id="session-recent-window")
        for index in range(6):
            self.storage.paper_chat_messages.append_paper_chat_message(
                session["session_id"],
                "user",
                f"question {index}",
                user_id=self.user_id,
                turn_id=f"turn-{index}",
            )
            self.storage.paper_chat_messages.append_paper_chat_message(
                session["session_id"],
                "assistant",
                f"answer {index}",
                user_id=self.user_id,
                turn_id=f"turn-{index}",
            )

        recent_messages = self.storage.paper_chat_messages.list_recent_paper_chat_messages(
            session["session_id"],
            user_id=self.user_id,
            limit=4,
        )

        self.assertEqual(self.storage.paper_chat_messages.count_paper_chat_messages(session["session_id"], user_id=self.user_id), 12)
        self.assertEqual(len(recent_messages), 4)
        self.assertEqual([item["turn_id"] for item in recent_messages], ["turn-4", "turn-4", "turn-5", "turn-5"])
        self.assertEqual([item["role"] for item in recent_messages], ["user", "assistant", "user", "assistant"])

    def test_append_paper_qa_turn_writes_complete_turn_atomically(self) -> None:
        session = self._create_session(arxiv_id="2401.00017", session_id="session-atomic-turn")

        result = self.storage.paper_qa_turns.append_paper_qa_turn(
            session_id=session["session_id"],
            user_id=self.user_id,
            question="What is the method?",
            answer="It uses retrieval.",
            sources=[{"chunk_id": "c1"}],
            retrieval_debug_snapshot={"score": 0.8},
            contextualized_question="What is the method?",
            question_contextualization={"used_short_term_memory": False},
        )
        messages = self.storage.paper_chat_messages.list_paper_chat_messages(session["session_id"], user_id=self.user_id)

        self.assertEqual(len(messages), 2)
        self.assertEqual(result["turn_id"], messages[0]["turn_id"])
        self.assertEqual(messages[0]["turn_id"], messages[1]["turn_id"])
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["sources"], [{"chunk_id": "c1"}])
        self.assertEqual(result["refreshed_session"]["message_count"], 2)

    def test_append_paper_qa_turn_rolls_back_when_assistant_payload_fails(self) -> None:
        session = self._create_session(arxiv_id="2401.00018", session_id="session-assistant-rollback")
        circular_sources = []
        circular_sources.append(circular_sources)

        with self.assertRaises(PaperQATurnPersistenceError):
            self.storage.paper_qa_turns.append_paper_qa_turn(
                session_id=session["session_id"],
                user_id=self.user_id,
                question="Q",
                answer="A",
                sources=circular_sources,
            )

        messages = self.storage.paper_chat_messages.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        refreshed_session = self.storage.paper_chat_sessions.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertEqual(messages, [])
        self.assertEqual(refreshed_session["message_count"], 0)

    def test_append_paper_qa_turn_rolls_back_when_session_stats_fail(self) -> None:
        session = self._create_session(arxiv_id="2401.00019", session_id="session-stats-rollback")

        with mock.patch.object(
            self.storage.paper_qa_turns,
            "_refresh_paper_chat_session_stats",
            side_effect=RuntimeError("stats failed"),
        ):
            with self.assertRaises(PaperQATurnPersistenceError):
                self.storage.paper_qa_turns.append_paper_qa_turn(
                    session_id=session["session_id"],
                    user_id=self.user_id,
                    question="Q",
                    answer="A",
                )

        messages = self.storage.paper_chat_messages.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        refreshed_session = self.storage.paper_chat_sessions.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertEqual(messages, [])
        self.assertEqual(refreshed_session["message_count"], 0)

    def test_append_paper_qa_turn_raises_for_missing_session(self) -> None:
        with self.assertRaises(PaperQATurnPersistenceError):
            self.storage.paper_qa_turns.append_paper_qa_turn(
                session_id="missing-session",
                user_id=self.user_id,
                question="Q",
                answer="A",
            )

    def test_cleanup_langgraph_checkpoints_follows_terminal_runtime_retention(self) -> None:
        old_time = (datetime.now(timezone.utc) - timedelta(days=9)).isoformat()
        self.storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
            user_id=self.user_id,
            session_id="agent-session-old",
            thread_id="agent-session-old",
            runtime_state={"step": "done"},
            status="completed",
        )
        self.storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
            user_id=self.user_id,
            session_id="agent-session-waiting",
            thread_id="agent-session-waiting",
            runtime_state={"step": "wait"},
            interaction={
                "interaction_id": "interaction-waiting",
                "kind": "target_selection",
                "status": "pending",
                "plan_id": "plan-waiting",
                "step_id": "resolve_paper",
                "payload": {"candidates": [], "reference_hint": {}},
                "created_at": old_time,
                "expires_at": "2999-01-01T00:00:00+00:00",
            },
            status="waiting_interaction",
        )
        with self.storage.connection_provider.connect() as conn:
            conn.execute(
                "UPDATE agent_runtime_checkpoints SET updated_at = ? WHERE session_id = ?",
                (old_time, "agent-session-old"),
            )
            conn.commit()
        self.assertTrue(
            self.storage.langgraph_checkpoints.put_langgraph_checkpoint(
                thread_id="agent-session-old",
                checkpoint_ns="",
                checkpoint_id="cp-old",
                checkpoint={"id": "cp-old"},
            )
        )
        self.assertTrue(
            self.storage.langgraph_checkpoints.put_langgraph_checkpoint_writes(
                thread_id="agent-session-old",
                checkpoint_ns="",
                checkpoint_id="cp-old",
                task_id="task-1",
                writes=[{"channel": "state", "value": {"ok": True}}],
            )
        )
        self.assertTrue(
            self.storage.langgraph_checkpoints.put_langgraph_checkpoint(
                thread_id="agent-session-waiting",
                checkpoint_ns="",
                checkpoint_id="cp-waiting",
                checkpoint={"id": "cp-waiting"},
            )
        )

        cleanup_result = self.storage.agent_runtime_checkpoints.cleanup_langgraph_checkpoints_for_terminal_runtime(retention_days=7)
        runtime_deleted = self.storage.agent_runtime_checkpoints.cleanup_agent_runtime_checkpoints(retention_days=7)

        self.assertEqual(cleanup_result, {"threads": 1, "checkpoints": 1, "writes": 1})
        self.assertEqual(runtime_deleted, 1)
        self.assertIsNone(self.storage.langgraph_checkpoints.get_langgraph_checkpoint(thread_id="agent-session-old"))
        self.assertIsNotNone(self.storage.langgraph_checkpoints.get_langgraph_checkpoint(thread_id="agent-session-waiting"))
        self.assertIsNone(
            self.storage.agent_runtime_checkpoints.get_agent_runtime_checkpoint(
                user_id=self.user_id,
                session_id="agent-session-old",
                thread_id="agent-session-old",
            )
        )

    def test_paper_notes_support_create_update_list_and_delete(self) -> None:
        session = self._create_session(arxiv_id="2401.00008", session_id="session-note")
        message = self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "assistant", "answer", user_id=self.user_id)

        note = self.storage.paper_notes.create_paper_note(
            user_id=self.user_id,
            arxiv_id="2401.00008",
            session_id=session["session_id"],
            source_message_id=message["message_id"],
            title="Method note",
            content="Important detail",
            note_type="method",
            source_chunk_ids=[1, "2"],
            tags=["rag", "method"],
            include_in_profile=True,
        )
        updated = self.storage.paper_notes.update_paper_note(
            note["note_id"],
            user_id=self.user_id,
            title="Updated title",
            note_type="unknown-type",
            tags=["updated"],
            include_in_profile=False,
        )
        notes = self.storage.paper_notes.list_paper_notes(arxiv_id="2401.00008", user_id=self.user_id)
        profile_notes = self.storage.paper_notes.list_user_profile_notes(user_id=self.user_id)

        self.assertEqual(note["source_chunk_ids"], ["1", "2"])
        self.assertEqual(updated["title"], "Updated title")
        self.assertEqual(updated["note_type"], "custom")
        self.assertEqual(updated["tags"], ["updated"])
        self.assertFalse(updated["include_in_profile"])
        self.assertEqual(len(notes), 1)
        self.assertEqual(profile_notes, [])
        self.assertTrue(self.storage.paper_notes.delete_paper_note(note["note_id"], user_id=self.user_id))
        self.assertIsNone(self.storage.paper_notes.get_paper_note(note["note_id"], user_id=self.user_id))

    def test_list_user_profile_notes_returns_only_included_notes(self) -> None:
        self.storage.paper_notes.create_paper_note(
            user_id=self.user_id,
            arxiv_id="2401.00008",
            title="Profile note",
            content="Important profile signal",
            note_type="method",
            tags=["rag"],
            include_in_profile=True,
        )
        self.storage.paper_notes.create_paper_note(
            user_id=self.user_id,
            arxiv_id="2401.00009",
            title="Private note",
            content="Do not include",
            note_type="summary",
            tags=["private"],
            include_in_profile=False,
        )

        notes = self.storage.paper_notes.list_user_profile_notes(user_id=self.user_id)

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["title"], "Profile note")
        self.assertTrue(notes[0]["include_in_profile"])

    def test_user_research_profiles_support_upsert_patch_and_get(self) -> None:
        profile = self.storage.research_profiles.upsert_user_research_profile(
            user_id=self.user_id,
            profile={
                "positive_topics": ["rag", "agents"],
                "negative_topics": "vision",
                "preferred_categories": ["cs.CL"],
                "preferred_answer_style": "concise",
            },
        )
        patched = self.storage.research_profiles.patch_user_research_profile(
            user_id=self.user_id,
            profile={"recent_topics": ["memory"], "preferred_answer_style": "detailed"},
        )
        loaded = self.storage.research_profiles.get_user_research_profile(self.user_id)

        self.assertEqual(profile["positive_topics"], ["rag", "agents"])
        self.assertEqual(profile["negative_topics"], ["vision"])
        self.assertEqual(patched["recent_topics"], ["memory"])
        self.assertEqual(loaded["preferred_answer_style"], "detailed")

    def test_profile_layers_keep_manual_generated_and_effective_separate(self) -> None:
        manual = self.storage.research_profiles.upsert_user_manual_profile(
            user_id=self.user_id,
            profile={"positive_topics": ["manual retrieval topic"], "preferred_categories": ["cs.SE"]},
        )
        job_id = self.storage.profile_build_jobs.create_user_profile_build_job(self.user_id)
        snapshot = self.storage.research_profiles.save_generated_profile_snapshot(
            user_id=self.user_id,
            generated_profile={
                "positive_topics": ["RAG retrieval optimization"],
                "negative_topics": ["diffusion models"],
                "preferred_categories": ["cs.CL"],
                "representative_papers": ["2401.00001"],
            },
            evidence_summary={"liked_paper_count": 1},
            quality_report={"positive_topic_count": 1},
            build_config={"reason": "unit-test"},
            job_id=job_id,
        )
        layers = self.storage.research_profiles.get_user_profile_layers(self.user_id)
        effective = self.storage.research_profiles.get_user_research_profile(self.user_id)

        self.assertEqual(manual["positive_topics"], ["manual retrieval topic"])
        self.assertIn("manual retrieval topic", effective["positive_topics"])
        self.assertIn("RAG retrieval optimization", effective["positive_topics"])
        self.assertEqual(layers["manual_profile"]["positive_topics"], ["manual retrieval topic"])
        self.assertEqual(layers["generated_profile"]["positive_topics"], ["RAG retrieval optimization"])
        self.assertEqual(layers["generated_profile"]["snapshot_id"], snapshot["snapshot_id"])
        self.assertIn("diffusion models", layers["effective_profile"]["negative_topics"])

    def test_low_quality_snapshot_does_not_activate_generated_profile(self) -> None:
        active_snapshot = self.storage.research_profiles.save_generated_profile_snapshot(
            user_id=self.user_id,
            generated_profile={
                "positive_topics": ["RAG retrieval optimization"],
                "preferred_categories": ["cs.CL"],
                "representative_papers": ["2401.00001"],
            },
            evidence_summary={"liked_paper_count": 1},
            quality_report={"approved": True, "quality_score": 0.9},
            build_config={"reason": "active"},
        )
        job_id = self.storage.profile_build_jobs.create_user_profile_build_job(self.user_id)
        inactive_snapshot = self.storage.research_profiles.save_generated_profile_snapshot(
            user_id=self.user_id,
            generated_profile={"positive_topics": ["paper"], "representative_papers": []},
            evidence_summary={"liked_paper_count": 1},
            quality_report={"approved": False, "quality_score": 0.2},
            build_config={"reason": "low-quality"},
            job_id=job_id,
            activate=False,
        )
        layers = self.storage.research_profiles.get_user_profile_layers(self.user_id)

        self.assertNotEqual(active_snapshot["snapshot_id"], inactive_snapshot["snapshot_id"])
        self.assertEqual(layers["generated_profile"]["snapshot_id"], active_snapshot["snapshot_id"])
        self.assertEqual(layers["generated_profile"]["positive_topics"], ["RAG retrieval optimization"])
        with self.storage.connection_provider.connect() as conn:
            row = conn.execute("SELECT status, snapshot_id FROM user_profile_build_jobs WHERE job_id = ?", (job_id,)).fetchone()
        self.assertEqual(row[0], "needs_review")
        self.assertEqual(row[1], inactive_snapshot["snapshot_id"])

    def test_json_serialization_helpers_handle_boundary_values(self) -> None:
        self.assertEqual(self.storage.paper_catalog._serialize_json_field(None), "")
        self.assertEqual(self.storage.paper_catalog._deserialize_json_field(""), None)
        self.assertEqual(self.storage.paper_catalog._deserialize_json_field("not-json"), "not-json")
        self.assertEqual(self.storage.paper_catalog._deserialize_json_field('{"a": 1}'), {"a": 1})
        self.assertEqual(self.storage.paper_catalog._deserialize_paper_db_value('["x", "y"]'), ["x", "y"])
        self.assertEqual(self.storage.paper_catalog._deserialize_paper_db_value('{"nested": true}'), {"nested": True})


if __name__ == "__main__":
    unittest.main()
