import gc
import json
import unittest
from unittest import mock

from services.memory.memory_debug import build_memory_debug_payload
from services.memory.memory_service import MemoryService
from tests.helpers import build_storage_container


class FakeEvidenceGenerationService:
    def __init__(self, payload: str = ""):
        self.payload = payload
        self.call_count = 0

    def complete_with_qwen(self, prompt, *args, **kwargs):
        self.call_count += 1
        if self.payload:
            return self.payload
        text = str(prompt or "").lower()
        concepts = []
        if "diffusion" in text:
            concepts.append("diffusion models")
        if "knowledge graph" in text:
            concepts.append("knowledge graph construction")
        if "agent memory" in text or "memory-augmented" in text:
            concepts.append("agent memory")
        if "long context" in text or "long-context" in text:
            concepts.append("long-context reasoning")
        if "retrieval-augmented generation" in text or "retrieval augmented generation" in text or "hybrid retrieval" in text or "reranking" in text:
            concepts.append("RAG retrieval optimization")
        if "vision-only generation" in text:
            concepts.append("vision-only generation")
        if not concepts:
            concepts.append("question answering")
        return json.dumps({
            "main_research_area": concepts[0],
            "research_objects": ["LLM agents"] if "agent" in text else [],
            "methods": [item for item in concepts if item in {"RAG retrieval optimization", "diffusion models"}],
            "tasks": ["question answering"] if "question answering" in text or "qa" in text else [],
            "application_domains": ["scientific literature search"] if "scientific" in text else [],
            "technical_concepts": concepts,
            "evaluation_focus": ["retrieval quality"] if "retrieval" in text else [],
            "system_type": "agentic RAG system" if "agent" in text else "",
            "candidate_concepts": [
                {
                    "label": concept,
                    "type": "technical_concept",
                    "confidence": 0.9,
                    "evidence_text": "fake evidence",
                    "source": "llm",
                    "whether_generalizable": True,
                }
                for concept in concepts
            ],
            "excluded_concepts": [],
            "extraction_confidence": 0.9,
        }, ensure_ascii=False)


class FakeInterestModelRefresher:
    def __init__(self, storage, *, stable_positive_ids=None, weak_positive_ids=None, stable_negative_ids=None, weak_negative_ids=None):
        self.storage = storage
        self.stable_positive_ids = stable_positive_ids
        self.weak_positive_ids = weak_positive_ids
        self.stable_negative_ids = stable_negative_ids
        self.weak_negative_ids = weak_negative_ids
        self.call_count = 0

    def __call__(self, user_id: str):
        self.call_count += 1
        liked_ids = self.storage.user_preferences.get_liked_papers(user_id)
        disliked_ids = self.storage.user_preferences.get_disliked_papers(user_id)
        stable_positive_ids = list(self.stable_positive_ids) if self.stable_positive_ids is not None else (liked_ids if len(liked_ids) >= 2 else [])
        weak_positive_ids = list(self.weak_positive_ids) if self.weak_positive_ids is not None else [item for item in liked_ids if item not in stable_positive_ids]
        stable_negative_ids = list(self.stable_negative_ids) if self.stable_negative_ids is not None else (disliked_ids if len(disliked_ids) >= 2 else [])
        weak_negative_ids = list(self.weak_negative_ids) if self.weak_negative_ids is not None else [item for item in disliked_ids if item not in stable_negative_ids]
        interest_clusters = [
            {
                "cluster_id": "cluster_0",
                "paper_count": len(stable_positive_ids),
                "paper_ids": stable_positive_ids,
                "centroid_vector": [1.0, 0.0, 0.0],
            }
        ] if stable_positive_ids else []
        weak_interest_pool = {
            "pool_id": "weak_interest_pool",
            "paper_count": len(weak_positive_ids),
            "paper_ids": weak_positive_ids,
            "centroid_vector": [0.0, 1.0, 0.0],
        } if weak_positive_ids else None
        negative_feedback_profile = {
            "version": "negative_feedback_profile_v1",
            "enabled": True,
            "mode": "clustered" if stable_negative_ids else ("examples" if weak_negative_ids else "none"),
            "hard_exclude_ids": disliked_ids,
            "examples": [{"arxiv_id": item} for item in disliked_ids],
            "clusters": [
                {
                    "cluster_id": "negative_cluster_0",
                    "paper_count": len(stable_negative_ids),
                    "paper_ids": stable_negative_ids,
                    "centroid_vector": [0.0, 0.0, 1.0],
                }
            ] if stable_negative_ids else [],
            "stats": {
                "total_disliked": len(disliked_ids),
                "usable_disliked": len(disliked_ids),
                "unresolved_disliked": 0,
                "negative_cluster_count": 1 if stable_negative_ids else 0,
            },
        }
        self.storage.interest_vectors.save_user_interest_vector(
            user_id=user_id,
            vector_data=[1.0, 0.0, 0.0],
            paper_count=len(liked_ids),
            embedding_model="fake-model",
            vector_dimension=3,
            cluster_count=len(interest_clusters),
            profile_mode="clustered" if interest_clusters else ("mean_with_weak_pool" if weak_interest_pool else "mean"),
            interest_clusters=interest_clusters,
            weak_interest_pool=weak_interest_pool,
            negative_feedback_profile=negative_feedback_profile,
        )
        return {"status": "success", "cluster_count": len(interest_clusters)}


class MemoryServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = build_storage_container()
        self.memory_service = self._make_memory_service(generation_service=FakeEvidenceGenerationService())
        self.user_id = "user-1"
        self.arxiv_id = "2401.00001"
        self._add_paper(self.arxiv_id)

    def tearDown(self) -> None:
        temp_db = getattr(self.storage, "_test_temp_db", None)
        self.memory_service = None
        self.storage = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _make_memory_service(self, *, generation_service=None, interest_model_refresher=None) -> MemoryService:
        if interest_model_refresher is None:
            interest_model_refresher = FakeInterestModelRefresher(self.storage)
        return MemoryService(
            paper_catalog_store=self.storage.paper_catalog,
            user_preference_store=self.storage.user_preferences,
            interest_vector_store=self.storage.interest_vectors,
            paper_profile_evidence_store=self.storage.paper_profile_evidence,
            paper_chat_session_store=self.storage.paper_chat_sessions,
            paper_chat_message_store=self.storage.paper_chat_messages,
            paper_note_store=self.storage.paper_notes,
            profile_event_store=self.storage.profile_events,
            profile_build_job_store=self.storage.profile_build_jobs,
            research_profile_store=self.storage.research_profiles,
            agent_session_store=self.storage.agent_sessions,
            generation_service=generation_service,
            interest_model_refresher=interest_model_refresher,
        )

    def _add_paper(self, arxiv_id: str, *, title: str = "Test Paper", categories=None) -> None:
        self.storage.paper_catalog.add_paper(
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
        self.storage.user_preferences.add_liked_paper(self.user_id, self.arxiv_id)
        self.storage.user_preferences.record_user_paper_action(self.user_id, "2401.00002", "bookmark")
        self.storage.interest_vectors.save_user_interest_vector(
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
        session = self.storage.paper_chat_sessions.create_paper_chat_session(arxiv_id=self.arxiv_id, user_id=self.user_id, title="QA")
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "user", "What is the idea?", user_id=self.user_id, turn_id="turn-1")
        self.storage.paper_chat_messages.append_paper_chat_message(
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
        session = self.storage.paper_chat_sessions.create_paper_chat_session(arxiv_id=self.arxiv_id, user_id=self.user_id, title="QA")
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "user", "Valid question?", user_id=self.user_id, turn_id="valid-turn")
        self.storage.paper_chat_messages.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "Valid answer.",
            user_id=self.user_id,
            turn_id="valid-turn",
            sources=[{"source_id": "s1"}],
        )
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "user", "Only user", user_id=self.user_id, turn_id="user-only")
        self.storage.paper_chat_messages.append_paper_chat_message(
            session["session_id"],
            "assistant",
            "Only assistant",
            user_id=self.user_id,
            turn_id="assistant-only",
        )
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "user", "Duplicate one", user_id=self.user_id, turn_id="duplicate-role")
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "user", "Duplicate two", user_id=self.user_id, turn_id="duplicate-role")
        self.storage.paper_chat_messages.append_paper_chat_message(
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

    def test_conversation_context_reads_recent_message_window_instead_of_full_history(self) -> None:
        session = self.storage.paper_chat_sessions.create_paper_chat_session(arxiv_id=self.arxiv_id, user_id=self.user_id, title="Long QA")
        for index in range(12):
            self.storage.paper_chat_messages.append_paper_chat_message(
                session["session_id"],
                "user",
                f"Question {index}",
                user_id=self.user_id,
                turn_id=f"turn-{index}",
            )
            self.storage.paper_chat_messages.append_paper_chat_message(
                session["session_id"],
                "assistant",
                f"Answer {index}",
                user_id=self.user_id,
                turn_id=f"turn-{index}",
            )

        context = self.memory_service.load_paper_conversation_context(
            self.user_id,
            self.arxiv_id,
            session_id=session["session_id"],
            limit=3,
        )

        self.assertEqual(context["total_message_count"], 24)
        self.assertEqual(context["db_message_read_limit"], 16)
        self.assertEqual(context["db_message_read_count"], 16)
        self.assertEqual(context["turn_count"], 3)
        self.assertEqual([turn["turn_id"] for turn in context["turns"]], ["turn-9", "turn-10", "turn-11"])

    def test_update_profile_from_note_merges_clean_note_tags_only(self) -> None:
        note = self.storage.paper_notes.create_paper_note(
            user_id=self.user_id,
            arxiv_id=self.arxiv_id,
            title="Important finding about a single paper",
            content="Remember this paper.",
            note_type="summary",
            tags=["rag", "retrieval", "cs.CL", "paper", self.arxiv_id, "https://arxiv.org/abs/2401.00001"],
            include_in_profile=True,
        )

        immediate_profile = self.memory_service.update_profile_from_note(self.user_id, note)
        events = self.storage.profile_events.list_user_profile_events(self.user_id, event_types=["note_saved"])
        profile = self.memory_service.rebuild_user_research_profile(self.user_id)

        self.assertEqual(immediate_profile["positive_topics"], [])
        self.assertEqual(len(events), 1)
        self.assertIn("RAG retrieval optimization", profile["positive_topics"])
        self.assertEqual(profile["canonical_topics"][0]["label"], "RAG retrieval optimization")
        self.assertIn("retrieval", profile["recent_topics"])
        self.assertNotIn("Important finding about a single paper", profile["positive_topics"])
        self.assertNotIn("cs.CL", profile["positive_topics"])
        self.assertNotIn("paper", profile["positive_topics"])
        self.assertIn(self.arxiv_id, profile["representative_papers"])

    def test_note_and_qa_write_profile_events_without_sync_generation(self) -> None:
        note = self.storage.paper_notes.create_paper_note(
            user_id=self.user_id,
            arxiv_id=self.arxiv_id,
            title="Profile note",
            content="Use retrieval signals",
            note_type="summary",
            tags=["rag"],
            include_in_profile=True,
        )
        session = self.storage.paper_chat_sessions.create_paper_chat_session(arxiv_id=self.arxiv_id, user_id=self.user_id, title="QA")
        self.storage.paper_chat_messages.append_paper_chat_message(session["session_id"], "user", "How does retrieval work?", user_id=self.user_id)

        events = self.storage.profile_events.list_user_profile_events(self.user_id)
        event_types = {event["event_type"] for event in events}

        self.assertIsNotNone(note)
        self.assertIn("note_saved", event_types)
        self.assertIn("qa_asked", event_types)
        self.assertEqual(self.storage.research_profiles.get_user_generated_profile(self.user_id)["positive_topics"], [])
        self.memory_service.rebuild_user_research_profile(self.user_id)
        self.assertIsNotNone(self.storage.paper_profile_evidence.get_paper_profile_evidence(self.arxiv_id))

    def test_like_paper_keeps_categories_and_titles_out_of_topics(self) -> None:
        immediate_profile = self.memory_service.update_profile_from_preference(
            self.user_id,
            self.arxiv_id,
            "like",
            paper_payload={
                "arxiv_id": self.arxiv_id,
                "title": "A Complete Paper Title That Should Not Become A Topic",
                "abstract": "Retrieval-augmented generation uses hybrid retrieval, reranking, and agent memory for research workflows.",
                "categories": ["cs.CL", "cs.AI"],
                "topics": ["RAG", "agent memory", "method", self.arxiv_id],
            },
        )
        events = self.storage.profile_events.list_user_profile_events(self.user_id, event_types=["liked"])
        profile = self.memory_service.rebuild_user_research_profile(self.user_id)

        self.assertEqual(immediate_profile["positive_topics"], [])
        self.assertEqual(len(events), 1)
        self.assertEqual(profile["positive_topics"], [])
        self.assertEqual(profile["recent_topics"], [])
        self.assertNotIn("cs.CL", profile["positive_topics"])
        self.assertNotIn("cs.AI", profile["recent_topics"])
        self.assertNotIn("A Complete Paper Title That Should Not Become A Topic", profile["positive_topics"])
        self.assertNotIn("method", profile["positive_topics"])
        self.assertEqual(profile["preferred_categories"], [])
        self.assertNotIn(self.arxiv_id, profile["representative_papers"])
        self.assertIn("insufficient_stable_interest_evidence", {issue["code"] for issue in profile["quality_report"]["issues"]})

    def test_like_paper_without_explicit_topics_only_updates_category_and_representative_paper(self) -> None:
        self.memory_service.update_profile_from_preference(
            self.user_id,
            "2401.00002",
            "like",
            paper_payload={
                "arxiv_id": "2401.00002",
                "title": "A Long Paper Title Should Not Become Recent Research Interest",
                "categories": "cs.CL cs.LG",
            },
        )
        profile = self.memory_service.rebuild_user_research_profile(self.user_id)

        self.assertEqual(profile["positive_topics"], [])
        self.assertEqual(profile["recent_topics"], [])
        self.assertEqual(profile["preferred_categories"], [])
        self.assertNotIn("2401.00002", profile["representative_papers"])

    def test_single_liked_paper_is_gated_as_weak_behavior_profile_evidence(self) -> None:
        fake_llm = FakeEvidenceGenerationService()
        refresher = FakeInterestModelRefresher(self.storage, stable_positive_ids=[], weak_positive_ids=[self.arxiv_id])
        service = self._make_memory_service(generation_service=fake_llm, interest_model_refresher=refresher)
        self.storage.user_preferences.add_liked_paper(self.user_id, self.arxiv_id)

        profile = service.rebuild_user_research_profile(self.user_id)
        latest_job = service.list_profile_build_jobs(self.user_id, limit=1)[0]
        gating = profile["quality_report"]["behavior_profile_gating"]

        self.assertEqual(refresher.call_count, 1)
        self.assertEqual(fake_llm.call_count, 0)
        self.assertEqual(profile["positive_topics"], [])
        self.assertEqual(gating["weak_positive_paper_count"], 1)
        self.assertIn(self.arxiv_id, gating["skipped_positive_papers"])
        self.assertEqual(latest_job["metrics"]["behavior_profile_gating"]["weak_positive_paper_count"], 1)
        self.assertIsNone(self.storage.paper_profile_evidence.get_paper_profile_evidence(self.arxiv_id))
        self.assertIn("insufficient_stable_interest_evidence", {issue["code"] for issue in profile["quality_report"]["issues"]})

    def test_stable_interest_cluster_triggers_evidence_and_profile_topics(self) -> None:
        self._add_paper(
            "2401.01001",
            title="Hybrid Retrieval and Reranking for Long-Context Agents",
            categories=["cs.CL"],
        )
        self._add_paper(
            "2401.01002",
            title="Hybrid Retrieval and Reranking for Research Agents",
            categories=["cs.CL"],
        )
        fake_llm = FakeEvidenceGenerationService()
        service = self._make_memory_service(generation_service=fake_llm)
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.01001")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.01002")

        profile = service.rebuild_user_research_profile(self.user_id)
        gating = profile["quality_report"]["behavior_profile_gating"]

        self.assertEqual(fake_llm.call_count, 2)
        self.assertIn("RAG retrieval optimization", profile["positive_topics"])
        self.assertEqual(gating["stable_positive_paper_count"], 2)
        self.assertEqual(gating["weak_positive_paper_count"], 0)
        self.assertEqual(gating["skipped_positive_papers"], [])

    def test_dislike_paper_does_not_store_categories_or_titles_as_negative_topics(self) -> None:
        self.memory_service.update_profile_from_preference(
            self.user_id,
            self.arxiv_id,
            "dislike",
            paper_payload={
                "arxiv_id": self.arxiv_id,
                "title": "Another Full Paper Title That Should Stay Out",
                "abstract": "The paper studies vision-only generation systems and their limitations.",
                "categories": ["cs.CL"],
                "topics": ["vision-only generation", "framework"],
            },
        )
        profile = self.memory_service.rebuild_user_research_profile(self.user_id)

        self.assertEqual(profile["negative_topics"], [])
        self.assertNotIn("cs.CL", profile["negative_topics"])
        self.assertNotIn("Another Full Paper Title That Should Stay Out", profile["negative_topics"])
        self.assertNotIn("framework", profile["negative_topics"])
        self.assertEqual(profile["quality_report"]["behavior_profile_gating"]["weak_negative_paper_count"], 1)

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
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00015",
                "title": "Diffusion Models for Visual Synthesis",
                "authors": ["Alice"],
                "abstract": "Diffusion models focus on text-to-image visual generation and image synthesis.",
                "categories": ["cs.CV"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00015",
            }
        )
        self.storage.research_profiles.upsert_user_research_profile(
            self.user_id,
            {"preferred_answer_style": "concise", "common_question_types": ["summary"]},
        )
        for arxiv_id in ("2401.00010", "2401.00011", "2401.00012"):
            self.storage.user_preferences.add_liked_paper(self.user_id, arxiv_id)
        self.storage.user_preferences.add_disliked_paper(self.user_id, "2401.00013")
        self.storage.user_preferences.add_disliked_paper(self.user_id, "2401.00015")
        self.storage.user_preferences.record_user_paper_action(self.user_id, "2401.00014", "favorite")
        self.storage.paper_notes.create_paper_note(
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
        rag_evidence = profile["topic_evidence"]["RAG retrieval optimization"]
        self.assertGreater(rag_evidence["positive_score"], rag_evidence["negative_score"])
        self.assertIn("2401.00010", rag_evidence["source_papers"])
        self.assertIn("liked", rag_evidence["source_actions"])
        self.assertEqual(profile["aggregation_report"]["aggregator_version"], "profile_aggregator_v1")
        self.assertTrue(profile["review_status"]["approved"])
        self.assertTrue(profile["quality_report"]["approved"])
        self.assertIsNotNone(profile.get("snapshot_id"))
        layers = self.memory_service.load_user_profile_layers(self.user_id)
        self.assertIn("RAG retrieval optimization", layers["generated_profile"]["positive_topics"])
        self.assertIn("RAG retrieval optimization", layers["effective_profile"]["positive_topics"])

    def test_read_only_recent_actions_do_not_pollute_recent_topics(self) -> None:
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00021",
                "title": "RAG Systems",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00021",
            }
        )
        self.storage.user_preferences.record_user_paper_action(self.user_id, "2401.00021", "read")

        profile = self.memory_service.rebuild_user_research_profile(self.user_id)

        self.assertEqual(profile["recent_topics"], [])
        self.assertEqual(profile["positive_topics"], [])
        self.assertIn("recent_topic_pollution", {issue["code"] for issue in profile["quality_report"]["issues"]})

    def test_rebuild_user_research_profile_cleans_existing_dirty_profile(self) -> None:
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00112",
                "title": "Hybrid Retrieval and Reranking for Research Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and long context reasoning for agents.",
                "categories": ["cs.CL", "cs.AI"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00112",
            }
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00113",
                "title": "Diffusion Models for Visual Synthesis",
                "authors": ["Alice"],
                "abstract": "Diffusion models focus on text-to-image visual generation.",
                "categories": ["cs.CV"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00113",
            }
        )
        self.storage.research_profiles.upsert_user_research_profile(
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
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00110")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00112")
        self.storage.user_preferences.add_disliked_paper(self.user_id, "2401.00111")
        self.storage.user_preferences.add_disliked_paper(self.user_id, "2401.00113")

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
        self.assertIn("2401.00110", profile["representative_papers"])
        self.assertEqual(profile["preferred_answer_style"], "先结论后细节")
        self.assertEqual(profile["positive_topics"], repeated["positive_topics"])
        self.assertEqual(profile["representative_papers"], repeated["representative_papers"])
        layers = self.memory_service.load_user_profile_layers(self.user_id)
        self.assertIn("manual retrieval topic", layers["manual_profile"]["positive_topics"])
        self.assertNotIn("manual retrieval topic", layers["generated_profile"]["positive_topics"])
        self.assertIn("manual retrieval topic", layers["effective_profile"]["positive_topics"])
        self.assertIsNotNone(layers["generated_profile"]["snapshot_id"])

    def test_manual_profile_is_not_overwritten_by_generated_rebuild(self) -> None:
        self.storage.research_profiles.upsert_user_manual_profile(
            self.user_id,
            {"positive_topics": ["manual retrieval topic"], "preferred_categories": ["cs.SE"]},
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00210",
                "title": "RAG Retrieval Optimization for Long-Context Reasoning Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00210",
            }
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00211",
                "title": "Hybrid Retrieval and Reranking for Research Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00211",
            }
        )
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00210")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00211")

        profile = self.memory_service.rebuild_user_research_profile(self.user_id)
        layers = self.memory_service.load_user_profile_layers(self.user_id)

        self.assertIn("manual retrieval topic", layers["manual_profile"]["positive_topics"])
        self.assertNotIn("manual retrieval topic", layers["generated_profile"]["positive_topics"])
        self.assertIn("manual retrieval topic", profile["positive_topics"])
        self.assertIn("RAG retrieval optimization", profile["positive_topics"])

    def test_rebuild_consumes_profile_events_and_uses_event_stream(self) -> None:
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00310",
                "title": "RAG Retrieval Optimization for Long-Context Reasoning Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00310",
            }
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00311",
                "title": "Hybrid Retrieval and Reranking for Research Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00311",
            }
        )
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00310")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00310")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00311")
        self.storage.user_preferences.record_user_paper_action(self.user_id, "2401.00310", "read")

        before_events = self.storage.profile_events.list_user_profile_events(self.user_id, include_consumed=True)
        self.assertGreater(self.storage.profile_events.get_user_profile_dirty_event_count(self.user_id), 0)
        profile = self.memory_service.rebuild_user_research_profile(self.user_id)
        after_events = self.storage.profile_events.list_user_profile_events(self.user_id, include_consumed=True)

        self.assertEqual(len([event for event in before_events if event["event_type"] == "liked"]), 2)
        self.assertIn("RAG retrieval optimization", profile["positive_topics"])
        self.assertTrue(all(event["consumed_by_job_id"] for event in after_events))
        self.assertEqual(self.storage.profile_events.get_user_profile_dirty_event_count(self.user_id), 0)
        self.assertIn("read", {event["event_type"] for event in after_events})

    def test_manual_profile_updates_emit_manual_events(self) -> None:
        self.storage.research_profiles.upsert_user_manual_profile(
            self.user_id,
            {
                "positive_topics": ["manual retrieval topic"],
                "preferred_answer_style": "concise",
            },
        )
        self.storage.research_profiles.patch_user_manual_profile(
            self.user_id,
            {
                "positive_topics": [],
                "negative_topics": ["vision-only generation"],
            },
        )

        events = self.storage.profile_events.list_user_profile_events(self.user_id)
        event_types = [event["event_type"] for event in events]

        self.assertIn("manual_topic_added", event_types)
        self.assertIn("manual_topic_removed", event_types)
        self.assertIn("manual_style_updated", event_types)

    def test_legacy_dirty_topics_do_not_migrate_into_new_profile_layers(self) -> None:
        with self.storage.connection_provider.connect() as conn:
            self.storage.research_profiles._upsert_legacy_research_profile_cache(
                conn,
                "legacy-user",
                {
                    "positive_topics": ["cs.CL", "A Complete Paper Title That Should Be Removed", "manual retrieval topic"],
                    "negative_topics": ["https://arxiv.org/abs/2401.00001", "diffusion models"],
                    "preferred_categories": ["cs.AI"],
                    "representative_papers": ["A Complete Paper Title That Should Be Removed"],
                },
            )
            conn.commit()
            self.storage.research_profiles._migrate_legacy_research_profiles(conn)

        layers = self.storage.research_profiles.get_user_profile_layers("legacy-user")

        self.assertEqual(layers["manual_profile"]["positive_topics"], ["manual retrieval topic"])
        self.assertEqual(layers["manual_profile"]["negative_topics"], ["diffusion models"])
        self.assertEqual(layers["manual_profile"]["preferred_categories"], ["cs.AI"])
        self.assertEqual(layers["manual_profile"]["representative_papers"], [])

    def test_llm_evidence_card_drives_profile_topics_and_reuses_cache(self) -> None:
        fake_llm = FakeEvidenceGenerationService(
            """
            {
              "main_research_area": "retrieval-augmented generation",
              "research_objects": ["long-context agents"],
              "methods": ["hybrid retrieval", "reranking"],
              "tasks": ["question answering"],
              "application_domains": ["scientific literature search"],
              "technical_concepts": ["RAG retrieval optimization", "agent memory"],
              "evaluation_focus": ["retrieval quality"],
              "system_type": "agentic RAG system",
              "candidate_concepts": [
                {"label": "RAG retrieval optimization", "type": "technical_concept", "confidence": 0.92, "evidence_text": "hybrid retrieval and reranking", "source": "llm", "whether_generalizable": true},
                {"label": "cs.CL", "type": "technical_concept", "confidence": 0.99, "evidence_text": "category", "source": "llm", "whether_generalizable": true},
                {"label": "RAG Retrieval Optimization for Long-Context Reasoning Agents", "type": "technical_concept", "confidence": 0.99, "evidence_text": "title", "source": "llm", "whether_generalizable": true}
              ],
              "excluded_concepts": [{"label": "cs.CL", "reason": "arXiv category"}],
              "extraction_confidence": 0.9
            }
            """
        )
        service = self._make_memory_service(generation_service=fake_llm)
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00410",
                "title": "RAG Retrieval Optimization for Long-Context Reasoning Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00410",
            }
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00412",
                "title": "Hybrid Retrieval and Reranking for Research Agents",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00412",
            }
        )
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00410")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00412")

        profile = service.rebuild_user_research_profile(self.user_id)
        repeated = service.rebuild_user_research_profile(self.user_id)
        card = self.storage.paper_profile_evidence.get_paper_profile_evidence("2401.00410")

        self.assertEqual(fake_llm.call_count, 2)
        self.assertTrue(card["schema_valid"])
        self.assertIn("RAG retrieval optimization", profile["positive_topics"])
        self.assertEqual(profile["positive_topics"], repeated["positive_topics"])
        self.assertNotIn("cs.CL", card["technical_concepts"])
        self.assertNotIn("RAG Retrieval Optimization for Long-Context Reasoning Agents", profile["positive_topics"])

    def test_invalid_llm_evidence_card_records_error_and_does_not_use_title_ngrams(self) -> None:
        fake_llm = FakeEvidenceGenerationService("not json")
        service = self._make_memory_service(generation_service=fake_llm)
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00411",
                "title": "A Complete Paper Title That Should Not Become Topic",
                "authors": ["Alice"],
                "abstract": "This abstract mentions retrieval augmented generation but the extractor output is invalid.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00411",
            }
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00413",
                "title": "Hybrid Retrieval and Reranking Failure Case",
                "authors": ["Alice"],
                "abstract": "Retrieval augmented generation and hybrid retrieval are present but extractor output is invalid.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00413",
            }
        )
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00411")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00413")

        profile = service.rebuild_user_research_profile(self.user_id)
        card = self.storage.paper_profile_evidence.get_paper_profile_evidence("2401.00411")

        self.assertFalse(card["schema_valid"])
        self.assertIn("llm_output_not_json", card["error_message"])
        self.assertEqual(profile["positive_topics"], [])
        self.assertNotIn("A Complete Paper Title That Should Not Become Topic", profile["positive_topics"])

    def test_concept_normalization_merges_aliases_and_keeps_sources(self) -> None:
        fake_llm = FakeEvidenceGenerationService(
            """
            {
              "main_research_area": "agent memory",
              "research_objects": ["LLM agents"],
              "methods": ["graph-structured session memory"],
              "tasks": [],
              "application_domains": [],
              "technical_concepts": ["agent memory", "LLM long-term memory", "memory-augmented agents"],
              "evaluation_focus": [],
              "system_type": "agentic memory system",
              "candidate_concepts": [
                {"label": "agent memory", "type": "technical_concept", "confidence": 0.91, "evidence_text": "agent memory", "source": "llm", "whether_generalizable": true},
                {"label": "LLM long-term memory", "type": "technical_concept", "confidence": 0.88, "evidence_text": "long-term memory", "source": "llm", "whether_generalizable": true},
                {"label": "memory-augmented agents", "type": "technical_concept", "confidence": 0.86, "evidence_text": "memory augmented agents", "source": "llm", "whether_generalizable": true}
              ],
              "excluded_concepts": [],
              "extraction_confidence": 0.9
            }
            """
        )
        service = self._make_memory_service(generation_service=fake_llm)
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00420",
                "title": "Memory Systems for Agents",
                "authors": ["Alice"],
                "abstract": "LLM long-term memory and memory-augmented agents use graph-structured session memory.",
                "categories": ["cs.AI"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00420",
            }
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00421",
                "title": "Memory-Augmented Agents",
                "authors": ["Alice"],
                "abstract": "LLM long-term memory and memory-augmented agents use graph-structured session memory.",
                "categories": ["cs.AI"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00421",
            }
        )
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00420")
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00421")

        profile = service.rebuild_user_research_profile(self.user_id)
        canonical = profile["canonical_topics"][0]

        self.assertEqual(profile["positive_topics"].count("agent memory"), 1)
        self.assertNotIn("LLM long-term memory", profile["positive_topics"])
        self.assertNotIn("memory-augmented agents", profile["positive_topics"])
        self.assertEqual(canonical["label"], "agent memory")
        self.assertIn("LLM long-term memory", canonical["aliases"])
        self.assertIn("2401.00420", canonical["source_papers"])
        self.assertTrue(canonical["source_concepts"])
        self.assertEqual(profile["normalizer_version"], "profile_normalizer_v1")

    def test_manual_hidden_and_pinned_topics_affect_effective_profile(self) -> None:
        self.storage.research_profiles.upsert_user_manual_profile(
            self.user_id,
            {
                "pinned_topics": ["manual agent memory"],
                "hidden_topics": ["RAG retrieval optimization"],
            },
        )
        self.storage.paper_catalog.add_paper(
            {
                "arxiv_id": "2401.00430",
                "title": "RAG Systems",
                "authors": ["Alice"],
                "abstract": "Retrieval-augmented generation improves hybrid retrieval and reranking.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00430",
            }
        )
        self.storage.user_preferences.add_liked_paper(self.user_id, "2401.00430")

        profile = self.memory_service.rebuild_user_research_profile(self.user_id)
        events = self.storage.profile_events.list_user_profile_events(self.user_id, include_consumed=True)
        event_types = {event["event_type"] for event in events}

        self.assertIn("manual agent memory", profile["positive_topics"])
        self.assertNotIn("RAG retrieval optimization", profile["positive_topics"])
        self.assertEqual(profile["canonical_topics"][0]["label"], "manual agent memory")
        self.assertTrue(profile["canonical_topics"][0]["pinned"])
        self.assertIn("manual_topic_pinned", event_types)
        self.assertIn("manual_topic_hidden", event_types)

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

    def test_storage_exception_falls_back_to_safe_empty_memory_summary(self) -> None:
        with mock.patch.object(self.storage.connection_provider, "connect", side_effect=RuntimeError("db boom")):
            profile = self.memory_service.load_user_profile(self.user_id)
            summary = self.memory_service.load_preference_summary(self.user_id)

        self.assertEqual(profile["user_id"], self.user_id)
        self.assertEqual(summary["liked_papers"], [])
        self.assertEqual(summary["disliked_papers"], [])
        self.assertEqual(summary["paper_actions"], {})
        self.assertIsNone(summary["interest_vector"])

    def test_agent_memory_save_and_load_keeps_backend_context_authoritative(self) -> None:
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
            "pending_action": {"status": "waiting_confirmation", "step_id": "backend-step"},
            "tool_calls": [
                {"tool_name": "check_paper_qa_index", "status": "success", "summary": "index ready"},
                {"tool_name": "answer_paper_question", "status": "success", "summary": "grounded answer"},
            ],
        }

        saved = self.memory_service.save_agent_memory(self.user_id, "agent-session-1", final_state)
        loaded = self.memory_service.load_agent_memory(
            self.user_id,
            "agent-session-1",
            frontend_context={
                "selected_paper": {"arxiv_id": "frontend-paper"},
                "pending_action": {"status": "waiting_confirmation", "step_id": "stale"},
                "research_profile": {"positive_topics": ["stale"]},
                "ui_tab": "detail",
                "unknown_cache": "old",
            },
        )

        self.assertIsNotNone(saved)
        self.assertEqual(saved["active_arxiv_id"], self.arxiv_id)
        self.assertEqual(saved["active_paper_session_id"], "paper-session-1")
        self.assertEqual(saved["last_intent"], "paper_qa")
        self.assertEqual(saved["last_tool_calls_summary"][0]["tool_name"], "check_paper_qa_index")
        self.assertEqual(loaded["session_id"], "agent-session-1")
        self.assertEqual(loaded["backend_memory"]["selected_paper"]["arxiv_id"], self.arxiv_id)
        # 前端选中论文只能作为候选输入，不能覆盖后端持久化的会话焦点和待确认状态。
        self.assertEqual(loaded["merged_context"]["selected_paper"]["arxiv_id"], self.arxiv_id)
        self.assertEqual(loaded["merged_context"]["pending_action"], loaded["backend_memory"]["pending_action"])
        self.assertEqual(loaded["merged_context"]["frontend_visible_paper"]["arxiv_id"], "frontend-paper")
        self.assertEqual(loaded["merged_context"]["ui_tab"], "detail")
        self.assertNotIn("unknown_cache", loaded["merged_context"])
        merge_debug = loaded["context_merge_debug"]
        self.assertEqual(merge_debug["frontend_accepted_fields"]["selected_paper"], "frontend_visible_paper")
        self.assertEqual(merge_debug["frontend_ignored_fields"]["pending_action"], "backend_authoritative")
        self.assertEqual(merge_debug["frontend_ignored_fields"]["research_profile"], "backend_authoritative")
        self.assertEqual(merge_debug["frontend_ignored_fields"]["unknown_cache"], "not_allowlisted")

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
