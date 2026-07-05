import gc
import unittest

from services.storage.sqlite import SqliteConnectionProvider, StorageContainer
from tests.helpers.sqlite import TemporarySqliteDatabase


class StorageSqliteContainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_db = TemporarySqliteDatabase()
        provider = SqliteConnectionProvider(db_path=str(self.temp_db.db_path), check_same_thread=False)
        self.storage = StorageContainer(connection_provider=provider)
        self.user_id = "user-1"

    def tearDown(self) -> None:
        self.storage = None
        gc.collect()
        self.temp_db.cleanup()

    def _add_sample_paper(self, arxiv_id: str = "2401.00001") -> None:
        self.assertTrue(
            self.storage.paper_catalog.add_paper(
                {
                    "arxiv_id": arxiv_id,
                    "title": f"Paper {arxiv_id}",
                    "authors": ["Alice", "Bob"],
                    "abstract": "A sample abstract.",
                    "categories": ["cs.CL", "cs.IR"],
                    "published_date": "2024-01-01",
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                }
            )
        )

    def test_schema_initialization_creates_core_tables(self) -> None:
        with self.temp_db.connect() as conn:
            table_names = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }

        self.assertIn("arxiv_papers", table_names)
        self.assertIn("paper_qa_index", table_names)
        self.assertIn("agent_runtime_checkpoints", table_names)
        self.assertIn("langgraph_checkpoints", table_names)

    def test_paper_catalog_store_adds_reads_and_deletes_paper(self) -> None:
        self._add_sample_paper()

        paper = self.storage.paper_catalog.get_paper("2401.00001")

        self.assertEqual(paper["title"], "Paper 2401.00001")
        self.assertEqual(paper["authors"], ["Alice", "Bob"])
        self.assertEqual(self.storage.paper_catalog.get_total_paper_count(), 1)
        self.assertTrue(self.storage.paper_catalog.delete_paper("2401.00001"))
        self.assertIsNone(self.storage.paper_catalog.get_paper("2401.00001"))

    def test_user_preference_store_keeps_strong_feedback_mutually_exclusive(self) -> None:
        self._add_sample_paper()

        self.assertTrue(self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00001"))
        self.assertEqual(self.storage.user_preferences.get_liked_papers(self.user_id), ["2401.00001"])
        self.assertEqual(self.storage.user_preferences.get_disliked_papers(self.user_id), [])
        liked_events = self.storage.profile_events.list_user_profile_events(self.user_id)
        self.assertEqual({event["event_type"] for event in liked_events}, {"liked"})

        self.assertTrue(self.storage.user_preferences.add_disliked_paper(self.user_id, "2401.00001"))
        self.assertEqual(self.storage.user_preferences.get_liked_papers(self.user_id), [])
        self.assertEqual(self.storage.user_preferences.get_disliked_papers(self.user_id), ["2401.00001"])

        events = self.storage.profile_events.list_user_profile_events(self.user_id)
        self.assertEqual({event["event_type"] for event in events}, {"disliked"})

    def test_paper_qa_index_store_activates_single_complete_build(self) -> None:
        arxiv_id = "2401.01000"
        self.assertTrue(
            self.storage.paper_qa_index.insert_paper_qa_index(
                arxiv_id,
                collection_name="qa_old",
                status="indexed",
                chunk_count=1,
                embedding_model="old-model",
                retrieval_index_file="retrieval-old.json",
                retrieval_index_count=1,
                retrieval_index_types='["body"]',
                retrieval_index_version="retrieval-old",
                sparse_index_dir="sparse-old",
                sparse_index_manifest_file="sparse-old/manifest.json",
                sparse_index_document_count=1,
                sparse_index_token_count=10,
                sparse_index_backend="internal_bm25",
                sparse_index_schema_version="sparse_index_artifact_v1",
                sparse_index_source_file="chunks-old.json",
                sparse_index_source_hash="hash-old",
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

        active = self.storage.paper_qa_index.get_active_paper_qa_index_build(arxiv_id)
        old_build = self.storage.paper_qa_index.get_paper_qa_index_build(old_active["build_id"])
        qa_index = self.storage.paper_qa_index.get_paper_qa_index(arxiv_id)

        self.assertEqual(active["build_id"], build["build_id"])
        self.assertEqual(old_build["status"], "cleanup_pending")
        self.assertEqual(qa_index["collection_name"], "qa_new")
        self.assertEqual(qa_index["active_build_id"], build["build_id"])
        self.assertEqual(qa_index["previous_build_id"], old_active["build_id"])

    def test_agent_runtime_checkpoint_store_consumes_confirmation_once(self) -> None:
        checkpoint = self.storage.agent_runtime_checkpoints.upsert_agent_runtime_checkpoint(
            user_id=self.user_id,
            session_id="agent-session-1",
            thread_id="thread-1",
            runtime_state={
                "pending_confirmation": {"step_id": "step-1", "tool_name": "search"},
                "turn_status": "waiting_confirmation",
            },
            pending_confirmation={"step_id": "step-1", "tool_name": "search"},
            status="waiting_confirmation",
        )
        self.assertEqual(checkpoint["status"], "waiting_confirmation")

        first_consume = self.storage.agent_runtime_checkpoints.consume_agent_runtime_pending_confirmation(
            user_id=self.user_id,
            session_id="agent-session-1",
            thread_id="thread-1",
            decision="approve",
            step_id="step-1",
            tool_name="search",
        )
        second_consume = self.storage.agent_runtime_checkpoints.consume_agent_runtime_pending_confirmation(
            user_id=self.user_id,
            session_id="agent-session-1",
            thread_id="thread-1",
            decision="approve",
            step_id="step-1",
            tool_name="search",
        )
        stored = self.storage.agent_runtime_checkpoints.get_agent_runtime_checkpoint(
            user_id=self.user_id,
            session_id="agent-session-1",
            thread_id="thread-1",
        )

        self.assertTrue(first_consume)
        self.assertFalse(second_consume)
        self.assertEqual(stored["status"], "running")
        self.assertIsNone(stored["pending_confirmation"])
        self.assertEqual(stored["runtime_state"]["approved_step_ids"], ["step-1"])

    def test_langgraph_checkpoint_store_round_trips_raw_checkpoint(self) -> None:
        self.assertTrue(
            self.storage.langgraph_checkpoints.put_langgraph_checkpoint(
                thread_id="thread-1",
                checkpoint_ns="",
                checkpoint_id="checkpoint-1",
                checkpoint={"channel_values": {"messages": []}},
                metadata={"source": "unit-test"},
                parent_checkpoint_id=None,
            )
        )

        stored = self.storage.langgraph_checkpoints.get_langgraph_checkpoint(
            thread_id="thread-1",
            checkpoint_ns="",
            checkpoint_id="checkpoint-1",
        )

        self.assertEqual(stored["checkpoint_id"], "checkpoint-1")
        self.assertEqual(stored["checkpoint"]["channel_values"], {"messages": []})
        self.assertEqual(stored["metadata"], {"source": "unit-test"})

    def test_research_profile_store_merges_manual_and_generated_layers(self) -> None:
        self.storage.research_profiles.upsert_user_manual_profile(
            user_id=self.user_id,
            profile={
                "positive_topics": ["retrieval"],
                "pinned_topics": ["RAG"],
                "hidden_topics": ["noise"],
                "preferred_answer_style": "concise",
            },
        )
        snapshot = self.storage.research_profiles.save_generated_profile_snapshot(
            user_id=self.user_id,
            generated_profile={
                "positive_topics": ["noise", "reranking"],
                "preferred_categories": ["cs.IR"],
                "representative_papers": ["2401.00001"],
            },
            evidence_summary={"paper_count": 1},
            quality_report={"status": "accepted"},
            build_config={"mode": "unit"},
            activate=True,
        )

        layers = self.storage.research_profiles.get_user_profile_layers(self.user_id)
        effective = layers["effective_profile"]

        self.assertEqual(snapshot["snapshot_id"], layers["generated_profile"]["snapshot_id"])
        self.assertEqual(effective["positive_topics"], ["RAG", "retrieval", "reranking"])
        self.assertNotIn("noise", effective["positive_topics"])
        self.assertEqual(effective["preferred_categories"], ["cs.IR"])
        self.assertEqual(effective["preferred_answer_style"], "concise")
