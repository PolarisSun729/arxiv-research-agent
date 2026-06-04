import gc
import importlib.util
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from services.storage.database_service import DatabaseService
from tests.helpers import FakeEmbeddingService, FakeGenerationService, FakeVectorStoreService, build_database_service


def _load_paper_qa_service_class():
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root

    packages = {
        "services": backend_dir / "services",
        "services.arxiv": backend_dir / "services" / "arxiv",
        "services.document": backend_dir / "services" / "document",
        "services.paper_qa": backend_dir / "services" / "paper_qa",
        "services.retrieval": backend_dir / "services" / "retrieval",
        "services.storage": backend_dir / "services" / "storage",
        "services.embedding": backend_dir / "services" / "embedding",
        "services.llm": backend_dir / "services" / "llm",
    }
    for package_name, package_path in packages.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]
        sys.modules[package_name] = module

    if "services.arxiv.arxiv_search_service" not in sys.modules:
        module = types.ModuleType("services.arxiv.arxiv_search_service")

        class _ArxivSearchService:
            pass

        module.ArxivSearchService = _ArxivSearchService
        sys.modules[module.__name__] = module

    if "services.arxiv.arxiv_oai_service" not in sys.modules:
        module = types.ModuleType("services.arxiv.arxiv_oai_service")

        class _ArxivOaiDatabaseService:
            pass

        module.ArxivOaiDatabaseService = _ArxivOaiDatabaseService
        sys.modules[module.__name__] = module

    if "services.document.chunking_service" not in sys.modules:
        module = types.ModuleType("services.document.chunking_service")

        class _ChunkingService:
            pass

        module.ChunkingService = _ChunkingService
        sys.modules[module.__name__] = module

    if "services.document.loading_service" not in sys.modules:
        module = types.ModuleType("services.document.loading_service")

        class _LoadingService:
            pass

        module.LoadingService = _LoadingService
        sys.modules[module.__name__] = module

    if "services.memory" not in sys.modules:
        module = types.ModuleType("services.memory")

        class _MemoryService:
            pass

        module.MemoryService = _MemoryService
        sys.modules[module.__name__] = module

    if "services.embedding.embedding_service" not in sys.modules:
        module = types.ModuleType("services.embedding.embedding_service")

        class _EmbeddingConfig:
            pass

        class _EmbeddingService:
            pass

        module.EmbeddingConfig = _EmbeddingConfig
        module.EmbeddingService = _EmbeddingService
        sys.modules[module.__name__] = module

    if "services.retrieval.enhanced_retrieval_service" not in sys.modules:
        module = types.ModuleType("services.retrieval.enhanced_retrieval_service")

        class _EnhancedRetrievalService:
            pass

        class _RetrievalOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        module.EnhancedRetrievalService = _EnhancedRetrievalService
        module.RetrievalOptions = _RetrievalOptions
        sys.modules[module.__name__] = module

    if "services.llm.generation_service" not in sys.modules:
        module = types.ModuleType("services.llm.generation_service")

        class _GenerationService:
            pass

        module.GenerationService = _GenerationService
        sys.modules[module.__name__] = module

    if "services.paper_qa.paper_qa_index_builder" not in sys.modules:
        module = types.ModuleType("services.paper_qa.paper_qa_index_builder")

        class _PaperQAIndexBuilder:
            def __init__(self, *args, **kwargs):
                return None

        module.PaperQAIndexBuilder = _PaperQAIndexBuilder
        sys.modules[module.__name__] = module

    if "services.storage.vector_store_service" not in sys.modules:
        module = types.ModuleType("services.storage.vector_store_service")

        class _VectorStoreService:
            pass

        module.VectorStoreService = _VectorStoreService
        sys.modules[module.__name__] = module

    module_name = "services.paper_qa.paper_qa_service"
    if module_name in sys.modules:
        return sys.modules[module_name].PaperQAService

    spec = importlib.util.spec_from_file_location(module_name, backend_dir / "services" / "paper_qa" / "paper_qa_service.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.PaperQAService


PaperQAService = _load_paper_qa_service_class()


class _FakeMemoryService:
    def __init__(self) -> None:
        self.updated_notes = []

    def load_paper_conversation_context(self, *, user_id: str, arxiv_id: str, session_id=None, limit: int = 5):
        return {"turns": [], "selected_session_id": session_id}

    def merge_conversation_context(self, db_turns, payload_turns, limit: int = 5):
        merged = list(db_turns or []) + list(payload_turns or [])
        return merged[-limit:]

    def update_profile_from_note(self, user_id: str, note):
        self.updated_notes.append((user_id, deepcopy(note)))


class _FakeRetrievalService:
    def __init__(self, *, chunks=None, debug=None, raise_error: Exception | None = None) -> None:
        self.chunks = list(chunks or [])
        self.debug = deepcopy(debug) if debug is not None else {"provider": "fake-retrieval"}
        self.raise_error = raise_error
        self._temp_dir = tempfile.TemporaryDirectory(prefix="paper-qa-trace-")
        self.trace_export_dir = self._temp_dir.name

    def enhanced_retrieve(self, **_kwargs):
        if self.raise_error:
            raise self.raise_error
        return {"chunks": deepcopy(self.chunks), "debug": deepcopy(self.debug)}

    def cleanup(self) -> None:
        self._temp_dir.cleanup()


class _FakeGenerationWithResponse(FakeGenerationService):
    def __init__(self, *, response_text: str = "fake answer", raise_error: Exception | None = None) -> None:
        super().__init__(response_text=response_text)
        self.raise_error = raise_error

    def generate(self, **kwargs):
        self._record("generate", kwargs=kwargs)
        if self.raise_error:
            raise self.raise_error
        return {"response": self.response_text, "usage": {"output_tokens": len(self.response_text)}}


class PaperQAServiceComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_service = build_database_service(DatabaseService)
        self.memory_service = _FakeMemoryService()
        self.embedding_service = FakeEmbeddingService()
        self.vector_store_service = FakeVectorStoreService()
        self.generation_service = _FakeGenerationWithResponse(response_text="component answer")
        self.retrieval_service = _FakeRetrievalService()
        self.service = self._make_service()
        self.user_id = "user-1"
        self.arxiv_id = "2401.00001"
        self._add_sample_paper(self.arxiv_id)

    def tearDown(self) -> None:
        self.retrieval_service.cleanup()
        temp_db = getattr(self.db_service, "_test_temp_db", None)
        self.service = None
        self.db_service = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _make_service(self, *, retrieval_service=None, generation_service=None, memory_service=None, db_service=None):
        return PaperQAService(
            db_service=db_service or self.db_service,
            memory_service=memory_service or self.memory_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=generation_service or self.generation_service,
            enhanced_retrieval_service=retrieval_service or self.retrieval_service,
            qa_index_builder=object(),
        )

    def _add_sample_paper(self, arxiv_id: str) -> None:
        self.db_service.add_paper(
            {
                "arxiv_id": arxiv_id,
                "title": f"Paper {arxiv_id}",
                "authors": ["Alice", "Bob"],
                "abstract": "A paper abstract.",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )

    def _insert_index(self, *, arxiv_id=None, status="indexed") -> None:
        self.db_service.insert_paper_qa_index(
            arxiv_id or self.arxiv_id,
            collection_name="paper_2401",
            status=status,
            chunk_count=2,
            embedding_model="fake-model",
            pdf_path="/tmp/paper.pdf",
        )

    def _create_session(self, *, arxiv_id=None, session_id=None, status="active"):
        return self.db_service.create_paper_chat_session(
            arxiv_id=arxiv_id or self.arxiv_id,
            user_id=self.user_id,
            title="Session",
            status=status,
            session_id=session_id,
        )

    def test_get_qa_status_for_not_indexed_indexed_and_failed_states(self) -> None:
        not_indexed = self.service.get_qa_status(self.arxiv_id)

        self._insert_index(status="indexed")
        indexed = self.service.get_qa_status(self.arxiv_id)

        failed_arxiv_id = "2401.00002"
        self._add_sample_paper(failed_arxiv_id)
        self._insert_index(arxiv_id=failed_arxiv_id, status="failed")
        failed = self.service.get_qa_status(failed_arxiv_id)

        self.assertEqual(not_indexed["status"], "not_indexed")
        self.assertFalse(not_indexed["has_index"])
        self.assertEqual(indexed["status"], "indexed")
        self.assertTrue(indexed["has_index"])
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["has_index"])

    def test_resolve_chat_session_creates_or_reuses_sessions(self) -> None:
        created = self.service._resolve_chat_session(self.arxiv_id, {"user_id": self.user_id, "question": "What is this paper about?"})
        self.assertTrue(created["session_id"])

        reused_active = self.service._resolve_chat_session(self.arxiv_id, {"user_id": self.user_id, "question": "follow up"})
        self.assertEqual(reused_active["session_id"], created["session_id"])

        explicit = self.service._resolve_chat_session(
            self.arxiv_id,
            {"user_id": self.user_id, "session_id": created["session_id"], "question": "explicit reuse"},
        )
        self.assertEqual(explicit["session_id"], created["session_id"])

        other_session = self._create_session(arxiv_id="2401.99999", session_id="other-session")
        different_arxiv = self.service._resolve_chat_session(
            self.arxiv_id,
            {"user_id": self.user_id, "session_id": other_session["session_id"], "question": "wrong paper"},
        )
        self.assertNotEqual(different_arxiv["session_id"], other_session["session_id"])
        self.assertEqual(different_arxiv["arxiv_id"], self.arxiv_id)

    def test_resolve_chat_session_falls_back_to_stateless_when_db_errors(self) -> None:
        with mock.patch.object(self.db_service, "list_paper_chat_sessions", side_effect=RuntimeError("db boom")):
            session = self.service._resolve_chat_session(self.arxiv_id, {"user_id": self.user_id, "question": "fallback"})

        self.assertEqual(session, {})

    def test_persist_completed_turn_writes_two_messages_with_shared_turn_id(self) -> None:
        session = self._create_session(session_id="persist-session")

        result = self.service.persist_completed_turn(
            chat_session=session,
            question="What is the method?",
            answer="It uses retrieval.",
            source_payload=[{"source_id": "s1", "content": "source text"}],
            retrieval_debug={"score": 0.8},
            contextualized_question="What is the method?",
            question_contextualization={"used_short_term_memory": False},
        )

        messages = self.db_service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["turn_id"], messages[1]["turn_id"])
        self.assertEqual(result["turn_id"], messages[0]["turn_id"])
        self.assertEqual(messages[1]["sources"], [{"source_id": "s1", "content": "source text"}])
        self.assertEqual(messages[1]["retrieval_debug_snapshot"], {"score": 0.8})

    def test_persist_completed_turn_swallows_db_failures(self) -> None:
        session = self._create_session(session_id="failing-session")

        with mock.patch.object(self.db_service, "append_paper_chat_message", side_effect=RuntimeError("write failed")):
            result = self.service.persist_completed_turn(
                chat_session=session,
                question="Q",
                answer="A",
                source_payload=[],
                retrieval_debug=None,
                contextualized_question="Q",
                question_contextualization={},
            )

        self.assertTrue(result["turn_id"])
        self.assertIsNone(result["user_message"])
        self.assertIsNone(result["assistant_message"])

    def test_build_generation_context_and_source_payload_cover_text_assets_and_fields(self) -> None:
        long_content = "x" * 400
        search_results = [
            {
                "content": long_content,
                "chunk_type": "text",
                "page_number": 1,
                "source": "body",
                "subchunk_label": "1.1",
                "section_path": "Intro",
                "parent_chunk_id": "p1",
            },
            {
                "chunk_type": "figure",
                "asset_kind": "image",
                "asset_abs_path": "/tmp/figure.png",
                "asset_path": "figure.png",
                "asset_summary": "Figure summary",
                "page_number": 2,
                "section_path": "Method/Figure",
                "source": "figure-source",
            },
            {
                "chunk_type": "table",
                "asset_kind": "table",
                "asset_summary": "Table summary",
                "asset_preview_text": "cell a | cell b",
                "page_number": 3,
                "section_path": "Results/Table",
                "source": "table-source",
            },
        ]

        text_context, image_inputs, asset_metadata = self.service.build_generation_context(search_results)
        source_payload = self.service.build_source_payload(search_results)
        truncated = self.service._truncate_text(long_content, 32)

        self.assertIn(long_content, text_context)
        self.assertIn("[Table 3]", text_context)
        self.assertEqual(image_inputs[0]["image_path"], "/tmp/figure.png")
        self.assertEqual(len(asset_metadata), 2)
        self.assertEqual(source_payload[0]["parent_chunk_id"], "p1")
        self.assertEqual(source_payload[1]["asset_summary"], "Figure summary")
        self.assertIn("chunk_type", source_payload[2])
        self.assertTrue(truncated.endswith("..."))

    def test_answer_question_success_generates_answer_and_persists_turn(self) -> None:
        self._insert_index(status="indexed")
        session = self._create_session(session_id="qa-success")
        retrieval_service = _FakeRetrievalService(
            chunks=[
                {
                    "content": "Relevant chunk content",
                    "chunk_type": "text",
                    "page_number": 1,
                    "source": "body",
                }
            ],
            debug={"provider": "fake-retrieval", "score": 0.9},
        )
        generation_service = _FakeGenerationWithResponse(response_text="generated answer")
        service = self._make_service(retrieval_service=retrieval_service, generation_service=generation_service)

        result = service.answer_question(
            self.arxiv_id,
            {"question": "What is the contribution?", "user_id": self.user_id, "session_id": session["session_id"]},
        )

        messages = self.db_service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        retrieval_service.cleanup()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "generated answer")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "generated answer")

    def test_answer_question_raises_400_when_retrieval_returns_no_chunks(self) -> None:
        self._insert_index(status="indexed")
        retrieval_service = _FakeRetrievalService(chunks=[])
        service = self._make_service(retrieval_service=retrieval_service)

        with self.assertRaises(HTTPException) as ctx:
            service.answer_question(self.arxiv_id, {"question": "Empty?", "user_id": self.user_id})

        retrieval_service.cleanup()
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("No relevant chunks found", ctx.exception.detail)

    def test_answer_question_uses_fallback_answer_when_generation_fails(self) -> None:
        self._insert_index(status="indexed")
        session = self._create_session(session_id="qa-fallback")
        retrieval_service = _FakeRetrievalService(
            chunks=[
                {
                    "content": "Context block " * 120,
                    "chunk_type": "text",
                    "page_number": 1,
                    "source": "body",
                }
            ]
        )
        generation_service = _FakeGenerationWithResponse(raise_error=RuntimeError("llm boom"))
        service = self._make_service(retrieval_service=retrieval_service, generation_service=generation_service)

        result = service.answer_question(
            self.arxiv_id,
            {"question": "Why does it work?", "user_id": self.user_id, "session_id": session["session_id"]},
        )

        messages = self.db_service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        retrieval_service.cleanup()

        self.assertEqual(result["status"], "success")
        self.assertIn("根据论文内容", result["answer"])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], result["answer"])


if __name__ == "__main__":
    unittest.main()
