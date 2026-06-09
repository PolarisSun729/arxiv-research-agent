import gc
import unittest
from unittest import mock

from services.storage.database_service import DatabaseService, PaperQATurnPersistenceError
from tests.helpers import build_database_service


class DatabaseServiceSqliteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = build_database_service(DatabaseService)
        self.user_id = "user-1"

    def tearDown(self) -> None:
        temp_db = getattr(self.service, "_test_temp_db", None)
        self.service = None
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
        self.assertTrue(self.service.add_paper(paper))
        return paper

    def _create_session(self, arxiv_id: str = "2401.00001", session_id: str = "session-1") -> dict:
        self._add_sample_paper(arxiv_id)
        session = self.service.create_paper_chat_session(
            arxiv_id=arxiv_id,
            user_id=self.user_id,
            title="Initial title",
            session_id=session_id,
        )
        self.assertIsNotNone(session)
        return session

    def test_initialization_creates_expected_tables(self) -> None:
        with self.service._get_connection() as conn:
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

        stored = self.service.get_paper(paper["arxiv_id"])

        self.assertEqual(stored["arxiv_id"], paper["arxiv_id"])
        self.assertEqual(stored["authors"], ["Alice", "Bob"])
        self.assertEqual(stored["categories"], ["cs.CL", "cs.IR"])
        self.assertEqual(len(self.service.get_all_papers()), 1)
        self.assertTrue(self.service.delete_paper(paper["arxiv_id"]))
        self.assertIsNone(self.service.get_paper(paper["arxiv_id"]))

    def test_liked_and_disliked_papers_are_mutually_exclusive(self) -> None:
        paper = self._add_sample_paper()

        self.assertTrue(self.service.add_liked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"]))
        self.assertEqual(self.service.get_liked_papers(self.user_id), [paper["arxiv_id"]])
        self.assertEqual(self.service.get_disliked_papers(self.user_id), [])

        self.assertTrue(self.service.add_disliked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"]))
        self.assertEqual(self.service.get_liked_papers(self.user_id), [])
        self.assertEqual(self.service.get_disliked_papers(self.user_id), [paper["arxiv_id"]])

        self.assertTrue(self.service.add_liked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"]))
        self.assertEqual(self.service.get_liked_papers(self.user_id), [paper["arxiv_id"]])
        self.assertEqual(self.service.get_disliked_papers(self.user_id), [])

    def test_record_user_paper_action_supports_aliases_and_rejects_invalid_values(self) -> None:
        paper = self._add_sample_paper("2401.00002")

        self.assertTrue(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "liked"))
        self.assertTrue(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "bookmark"))
        self.assertTrue(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "save_for_later"))
        self.assertTrue(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "disliked"))
        self.assertFalse(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "invalid_action"))

        actions = self.service.get_user_paper_actions(self.user_id)
        action_types = {item["action_type"] for item in actions}

        self.assertIn("dislike", action_types)
        self.assertIn("favorite", action_types)
        self.assertIn("later", action_types)
        self.assertNotIn("like", action_types)

    def test_profile_events_are_append_only_and_deduped_by_semantic_target(self) -> None:
        paper = self._add_sample_paper("2401.00999")

        self.assertTrue(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "liked"))
        self.assertTrue(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "liked"))
        self.assertTrue(self.service.record_user_paper_action(self.user_id, paper["arxiv_id"], "read"))
        events = self.service.list_user_profile_events(self.user_id)
        liked_events = [event for event in events if event["event_type"] == "liked"]
        read_events = [event for event in events if event["event_type"] == "read"]

        self.assertEqual(len(liked_events), 1)
        self.assertEqual(len(read_events), 1)
        self.assertEqual(liked_events[0]["arxiv_id"], paper["arxiv_id"])
        self.assertGreater(liked_events[0]["action_strength"], read_events[0]["action_strength"])
        self.assertTrue(liked_events[0]["include_in_profile"])

    def test_get_user_preferences_returns_stable_structure(self) -> None:
        paper = self._add_sample_paper("2401.00003")
        self.service.add_liked_paper(user_id=self.user_id, arxiv_id=paper["arxiv_id"])

        preferences = self.service.get_user_preferences(self.user_id)

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
            self.service.insert_paper_qa_index(
                arxiv_id,
                collection_name="paper_2401_00004",
                status="building",
                chunk_count=2,
                embedding_model="fake-model",
                pdf_path="/tmp/paper.pdf",
                chunk_file="/tmp/chunks.json",
                embedding_file="/tmp/embeddings.json",
                loading_method="docling",
                chunking_strategy="docling_sections",
                current_stage="save_embeddings",
                failed_stage="",
                error_message="",
            )
        )
        self.assertTrue(
            self.service.update_paper_qa_index(
                arxiv_id,
                status="indexed",
                chunk_count=3,
                collection_name="paper_2401_00004_v2",
                indexed_at="2024-01-01T00:00:00",
            )
        )

        qa_index = self.service.get_paper_qa_index(arxiv_id)

        self.assertEqual(qa_index["collection_name"], "paper_2401_00004_v2")
        self.assertEqual(qa_index["status"], "indexed")
        self.assertEqual(qa_index["chunk_count"], 3)
        self.assertEqual(qa_index["chunk_file"], "/tmp/chunks.json")
        self.assertEqual(qa_index["embedding_file"], "/tmp/embeddings.json")
        self.assertEqual(qa_index["loading_method"], "docling")
        self.assertEqual(qa_index["chunking_strategy"], "docling_sections")
        self.assertEqual(qa_index["current_stage"], "save_embeddings")
        self.assertEqual(qa_index["indexed_at"], "2024-01-01T00:00:00")

    def test_paper_qa_index_build_activation_keeps_single_active_version(self) -> None:
        arxiv_id = "2401.01000"
        self.assertTrue(
            self.service.insert_paper_qa_index(
                arxiv_id,
                collection_name="qa_old",
                status="indexed",
                chunk_count=1,
                embedding_model="old-model",
            )
        )
        old_active = self.service.get_active_paper_qa_index_build(arxiv_id)
        build = self.service.create_paper_qa_index_build(arxiv_id, "docling")
        self.assertTrue(
            self.service.update_paper_qa_index_build(
                build["build_id"],
                status="build_success",
                collection_name="qa_new",
                chunk_count=2,
                embedding_model="new-model",
                current_stage="activate_index",
            )
        )

        self.assertTrue(self.service.activate_paper_qa_index_build(build["build_id"]))

        active = self.service.get_paper_qa_index(arxiv_id)
        active_versions = self.service.list_paper_qa_index_builds(arxiv_id, statuses=["active"], limit=10)
        cleanup_versions = self.service.list_paper_qa_index_builds(arxiv_id, statuses=["cleanup_pending"], limit=10)
        self.assertEqual(active["collection_name"], "qa_new")
        self.assertEqual(active["active_build_id"], build["build_id"])
        self.assertEqual(len(active_versions), 1)
        self.assertEqual(cleanup_versions[0]["build_id"], old_active["build_id"])

    def test_legacy_qa_index_row_is_recognized_as_active_version(self) -> None:
        arxiv_id = "2401.01001"
        self.assertTrue(
            self.service.insert_paper_qa_index(
                arxiv_id,
                collection_name="qa_legacy",
                status="indexed",
                chunk_count=1,
                embedding_model="legacy-model",
            )
        )

        active = self.service.get_active_paper_qa_index_build(arxiv_id)

        self.assertIsNotNone(active)
        self.assertTrue(active["is_active"])
        self.assertEqual(active["collection_name"], "qa_legacy")

    def test_paper_index_jobs_support_create_update_and_latest_query(self) -> None:
        job = self.service.create_paper_index_job("2401.00005", loading_method="docling")

        self.assertIsNotNone(job)
        self.assertTrue(
            self.service.update_paper_index_job(
                job["job_id"],
                status="completed",
                current_stage="finished",
                progress=150,
                error_message="",
            )
        )

        stored = self.service.get_paper_index_job(job["job_id"])
        latest = self.service.get_latest_paper_index_job("2401.00005")

        self.assertEqual(stored["status"], "completed")
        self.assertEqual(stored["current_stage"], "finished")
        self.assertEqual(stored["progress"], 100)
        self.assertEqual(stored["idempotency_key"], "2401.00005:docling")
        self.assertIsNotNone(stored["heartbeat_at"])
        self.assertEqual(latest["job_id"], job["job_id"])

    def test_paper_chat_sessions_support_create_list_clear_and_delete(self) -> None:
        session = self._create_session(arxiv_id="2401.00006", session_id="session-clear")
        self.service.append_paper_chat_message(session["session_id"], "user", "question", user_id=self.user_id)
        self.service.append_paper_chat_message(session["session_id"], "assistant", "answer", user_id=self.user_id)

        sessions = self.service.list_paper_chat_sessions("2401.00006", user_id=self.user_id)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["message_count"], 2)

        self.assertTrue(self.service.clear_paper_chat_session(session["session_id"], user_id=self.user_id))
        cleared_messages = self.service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        cleared_session = self.service.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertEqual(cleared_messages, [])
        self.assertEqual(cleared_session["message_count"], 0)

        self.assertTrue(self.service.delete_paper_chat_session(session["session_id"], user_id=self.user_id))
        self.assertIsNone(self.service.get_paper_chat_session(session["session_id"], user_id=self.user_id))

    def test_paper_chat_messages_preserve_turn_id_and_json_payloads(self) -> None:
        session = self._create_session(arxiv_id="2401.00007", session_id="session-msg")
        turn_id = "turn-1"
        user_message = self.service.append_paper_chat_message(
            session["session_id"],
            "user",
            "What is the method?",
            user_id=self.user_id,
            turn_id=turn_id,
            contextualized_question="ctx question",
            question_contextualization={"history": 1},
        )
        assistant_message = self.service.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "It uses retrieval.",
            user_id=self.user_id,
            turn_id=turn_id,
            sources=[{"chunk_id": "c1"}],
            retrieval_debug_snapshot={"score": 0.8},
        )

        messages = self.service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        by_turn = self.service.get_paper_chat_message_by_turn(session["session_id"], turn_id, role="assistant", user_id=self.user_id)

        self.assertEqual(len(messages), 2)
        self.assertEqual(user_message["turn_id"], turn_id)
        self.assertEqual(assistant_message["sources"], [{"chunk_id": "c1"}])
        self.assertEqual(assistant_message["retrieval_debug_snapshot"], {"score": 0.8})
        self.assertEqual(by_turn["message_id"], assistant_message["message_id"])

    def test_append_paper_qa_turn_writes_complete_turn_atomically(self) -> None:
        session = self._create_session(arxiv_id="2401.00017", session_id="session-atomic-turn")

        result = self.service.append_paper_qa_turn(
            session_id=session["session_id"],
            user_id=self.user_id,
            question="What is the method?",
            answer="It uses retrieval.",
            sources=[{"chunk_id": "c1"}],
            retrieval_debug_snapshot={"score": 0.8},
            contextualized_question="What is the method?",
            question_contextualization={"used_short_term_memory": False},
        )
        messages = self.service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)

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
            self.service.append_paper_qa_turn(
                session_id=session["session_id"],
                user_id=self.user_id,
                question="Q",
                answer="A",
                sources=circular_sources,
            )

        messages = self.service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        refreshed_session = self.service.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertEqual(messages, [])
        self.assertEqual(refreshed_session["message_count"], 0)

    def test_append_paper_qa_turn_rolls_back_when_session_stats_fail(self) -> None:
        session = self._create_session(arxiv_id="2401.00019", session_id="session-stats-rollback")

        with mock.patch.object(self.service, "_refresh_paper_chat_session_stats", side_effect=RuntimeError("stats failed")):
            with self.assertRaises(PaperQATurnPersistenceError):
                self.service.append_paper_qa_turn(
                    session_id=session["session_id"],
                    user_id=self.user_id,
                    question="Q",
                    answer="A",
                )

        messages = self.service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        refreshed_session = self.service.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertEqual(messages, [])
        self.assertEqual(refreshed_session["message_count"], 0)

    def test_append_paper_qa_turn_raises_for_missing_session(self) -> None:
        with self.assertRaises(PaperQATurnPersistenceError):
            self.service.append_paper_qa_turn(
                session_id="missing-session",
                user_id=self.user_id,
                question="Q",
                answer="A",
            )

    def test_paper_notes_support_create_update_list_and_delete(self) -> None:
        session = self._create_session(arxiv_id="2401.00008", session_id="session-note")
        message = self.service.append_paper_chat_message(session["session_id"], "assistant", "answer", user_id=self.user_id)

        note = self.service.create_paper_note(
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
        updated = self.service.update_paper_note(
            note["note_id"],
            user_id=self.user_id,
            title="Updated title",
            note_type="unknown-type",
            tags=["updated"],
            include_in_profile=False,
        )
        notes = self.service.list_paper_notes(arxiv_id="2401.00008", user_id=self.user_id)
        profile_notes = self.service.list_user_profile_notes(user_id=self.user_id)

        self.assertEqual(note["source_chunk_ids"], ["1", "2"])
        self.assertEqual(updated["title"], "Updated title")
        self.assertEqual(updated["note_type"], "custom")
        self.assertEqual(updated["tags"], ["updated"])
        self.assertFalse(updated["include_in_profile"])
        self.assertEqual(len(notes), 1)
        self.assertEqual(profile_notes, [])
        self.assertTrue(self.service.delete_paper_note(note["note_id"], user_id=self.user_id))
        self.assertIsNone(self.service.get_paper_note(note["note_id"], user_id=self.user_id))

    def test_list_user_profile_notes_returns_only_included_notes(self) -> None:
        self.service.create_paper_note(
            user_id=self.user_id,
            arxiv_id="2401.00008",
            title="Profile note",
            content="Important profile signal",
            note_type="method",
            tags=["rag"],
            include_in_profile=True,
        )
        self.service.create_paper_note(
            user_id=self.user_id,
            arxiv_id="2401.00009",
            title="Private note",
            content="Do not include",
            note_type="summary",
            tags=["private"],
            include_in_profile=False,
        )

        notes = self.service.list_user_profile_notes(user_id=self.user_id)

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["title"], "Profile note")
        self.assertTrue(notes[0]["include_in_profile"])

    def test_user_research_profiles_support_upsert_patch_and_get(self) -> None:
        profile = self.service.upsert_user_research_profile(
            user_id=self.user_id,
            profile={
                "positive_topics": ["rag", "agents"],
                "negative_topics": "vision",
                "preferred_categories": ["cs.CL"],
                "preferred_answer_style": "concise",
            },
        )
        patched = self.service.patch_user_research_profile(
            user_id=self.user_id,
            profile={"recent_topics": ["memory"], "preferred_answer_style": "detailed"},
        )
        loaded = self.service.get_user_research_profile(self.user_id)

        self.assertEqual(profile["positive_topics"], ["rag", "agents"])
        self.assertEqual(profile["negative_topics"], ["vision"])
        self.assertEqual(patched["recent_topics"], ["memory"])
        self.assertEqual(loaded["preferred_answer_style"], "detailed")

    def test_profile_layers_keep_manual_generated_and_effective_separate(self) -> None:
        manual = self.service.upsert_user_manual_profile(
            user_id=self.user_id,
            profile={"positive_topics": ["manual retrieval topic"], "preferred_categories": ["cs.SE"]},
        )
        job_id = self.service.create_user_profile_build_job(self.user_id)
        snapshot = self.service.save_generated_profile_snapshot(
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
        layers = self.service.get_user_profile_layers(self.user_id)
        effective = self.service.get_user_research_profile(self.user_id)

        self.assertEqual(manual["positive_topics"], ["manual retrieval topic"])
        self.assertIn("manual retrieval topic", effective["positive_topics"])
        self.assertIn("RAG retrieval optimization", effective["positive_topics"])
        self.assertEqual(layers["manual_profile"]["positive_topics"], ["manual retrieval topic"])
        self.assertEqual(layers["generated_profile"]["positive_topics"], ["RAG retrieval optimization"])
        self.assertEqual(layers["generated_profile"]["snapshot_id"], snapshot["snapshot_id"])
        self.assertIn("diffusion models", layers["effective_profile"]["negative_topics"])

    def test_low_quality_snapshot_does_not_activate_generated_profile(self) -> None:
        active_snapshot = self.service.save_generated_profile_snapshot(
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
        job_id = self.service.create_user_profile_build_job(self.user_id)
        inactive_snapshot = self.service.save_generated_profile_snapshot(
            user_id=self.user_id,
            generated_profile={"positive_topics": ["paper"], "representative_papers": []},
            evidence_summary={"liked_paper_count": 1},
            quality_report={"approved": False, "quality_score": 0.2},
            build_config={"reason": "low-quality"},
            job_id=job_id,
            activate=False,
        )
        layers = self.service.get_user_profile_layers(self.user_id)

        self.assertNotEqual(active_snapshot["snapshot_id"], inactive_snapshot["snapshot_id"])
        self.assertEqual(layers["generated_profile"]["snapshot_id"], active_snapshot["snapshot_id"])
        self.assertEqual(layers["generated_profile"]["positive_topics"], ["RAG retrieval optimization"])
        with self.service._get_connection() as conn:
            row = conn.execute("SELECT status, snapshot_id FROM user_profile_build_jobs WHERE job_id = ?", (job_id,)).fetchone()
        self.assertEqual(row[0], "needs_review")
        self.assertEqual(row[1], inactive_snapshot["snapshot_id"])

    def test_json_serialization_helpers_handle_boundary_values(self) -> None:
        self.assertEqual(self.service._serialize_json_field(None), "")
        self.assertEqual(self.service._deserialize_json_field(""), None)
        self.assertEqual(self.service._deserialize_json_field("not-json"), "not-json")
        self.assertEqual(self.service._deserialize_json_field('{"a": 1}'), {"a": 1})
        self.assertEqual(self.service._deserialize_paper_db_value('["x", "y"]'), ["x", "y"])
        self.assertEqual(self.service._deserialize_paper_db_value('{"nested": true}'), {"nested": True})


if __name__ == "__main__":
    unittest.main()
