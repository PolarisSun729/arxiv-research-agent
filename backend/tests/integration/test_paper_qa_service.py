import gc
import importlib.util
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock


from core.errors import AppError, ErrorCode
from services.storage.sqlite.shared import PaperQATurnPersistenceError
from tests.helpers import FakeEmbeddingService, FakeGenerationService, FakeVectorStoreService, build_storage_container


def _load_paper_qa_service_class():
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root
    created_stub_modules = []

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
        created_stub_modules.append(module.__name__)

    if "services.arxiv.arxiv_oai_service" not in sys.modules:
        module = types.ModuleType("services.arxiv.arxiv_oai_service")

        class _ArxivOaiDatabaseService:
            pass

        module.ArxivOaiDatabaseService = _ArxivOaiDatabaseService
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)

    if "services.document.chunking_service" not in sys.modules:
        module = types.ModuleType("services.document.chunking_service")

        class _ChunkingService:
            pass

        module.ChunkingService = _ChunkingService
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)

    if "services.document.loading_service" not in sys.modules:
        module = types.ModuleType("services.document.loading_service")

        class _LoadingService:
            pass

        module.LoadingService = _LoadingService
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)

    if "services.memory" not in sys.modules:
        module = types.ModuleType("services.memory")

        class _MemoryService:
            pass

        module.MemoryService = _MemoryService
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)

    module = sys.modules.get("services.embedding.embedding_service")
    if module is None:
        module = types.ModuleType("services.embedding.embedding_service")
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)
    if not hasattr(module, "EmbeddingConfig"):
        # 同一 pytest 进程里其它测试可能已注入轻量 stub；这里补齐 PaperQAService 真实导入契约。
        class _EmbeddingConfig:
            pass

        module.EmbeddingConfig = _EmbeddingConfig
    if not hasattr(module, "EmbeddingService"):
        class _EmbeddingService:
            pass

        module.EmbeddingService = _EmbeddingService

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
        created_stub_modules.append(module.__name__)

    if "services.llm.generation_service" not in sys.modules:
        module = types.ModuleType("services.llm.generation_service")

        class _GenerationService:
            pass

        module.GenerationService = _GenerationService
        # PaperQAService 测试桩只隔离生成服务本体，但仍需保留真实模块的常量导出形状。
        module.QWEN_RERANK_COMPRESS_MODEL_NAME = "fake-rerank-compress"
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)

    if "services.paper_qa.paper_qa_index_builder" not in sys.modules:
        module = types.ModuleType("services.paper_qa.paper_qa_index_builder")

        class _PaperQAIndexBuilder:
            def __init__(self, *args, **kwargs):
                return None

        module.PaperQAIndexBuilder = _PaperQAIndexBuilder
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)

    if "services.storage.vector_store_service" not in sys.modules:
        module = types.ModuleType("services.storage.vector_store_service")

        class _VectorDBConfig:
            def __init__(self, provider="milvus", index_mode="default"):
                self.provider = provider
                self.index_mode = index_mode

        class _VectorStoreService:
            pass

        # 真实建索引模块会导入 VectorDBConfig；测试桩保留该导出，避免污染其他集成测试。
        module.VectorDBConfig = _VectorDBConfig
        module.VectorStoreService = _VectorStoreService
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)

    module_name = "services.paper_qa.paper_qa_service"
    if module_name in sys.modules:
        return sys.modules[module_name].PaperQAService

    spec = importlib.util.spec_from_file_location(module_name, backend_dir / "services" / "paper_qa" / "paper_qa_service.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    # 这些子模块 stub 只服务 PaperQAService 加载隔离；加载完成后清理，避免 full-suite 后续真实单测拿到假模块。
    for stub_name in created_stub_modules:
        sys.modules.pop(stub_name, None)
    return module.PaperQAService


PaperQAService = _load_paper_qa_service_class()


class _FakeMemoryService:
    def __init__(self) -> None:
        self.paper_conversation_turns = []

    def load_paper_conversation_context(self, *, user_id: str, arxiv_id: str, session_id=None, limit: int = 5):
        turns = deepcopy(self.paper_conversation_turns[-limit:])
        return {
            "turns": turns,
            "selected_session_id": session_id,
            "db_message_read_count": len(turns) * 2,
            "db_message_read_limit": limit * 4 + 4,
            "total_message_count": len(self.paper_conversation_turns) * 2,
            "filtered_incomplete_turn_count": 0,
            "invalid_turn_count": 0,
            "filtered_turn_count": 0,
        }

    def merge_conversation_context(self, db_turns, payload_turns, limit: int = 5):
        merged = list(db_turns or []) + list(payload_turns or [])
        return merged[-limit:]

    def build_user_memory_summary(self, _user_id: str):
        # 该组件用例不关注长期记忆内容，但 fake 仍需与生产接口一致，避免误走异常兜底。
        return {}


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
        self.storage = build_storage_container()
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
        temp_db = getattr(self.storage, "_test_temp_db", None)
        self.service = None
        self.storage = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _make_service(self, *, retrieval_service=None, generation_service=None, memory_service=None):
        return PaperQAService(
            paper_qa_index_store=self.storage.paper_qa_index,
            paper_catalog_store=self.storage.paper_catalog,
            paper_chat_session_store=self.storage.paper_chat_sessions,
            paper_qa_turn_store=self.storage.paper_qa_turns,
            research_profile_store=self.storage.research_profiles,
            agent_runtime_checkpoint_store=self.storage.agent_runtime_checkpoints,
            memory_service=memory_service or self.memory_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=generation_service or self.generation_service,
            enhanced_retrieval_service=retrieval_service or self.retrieval_service,
            qa_index_builder=object(),
            research_service=object(),
        )

    def _add_sample_paper(self, arxiv_id: str) -> None:
        self.storage.paper_catalog.add_paper(
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
        self.storage.paper_qa_index.insert_paper_qa_index(
            arxiv_id or self.arxiv_id,
            collection_name="paper_2401",
            status=status,
            chunk_count=2,
            embedding_model="fake-model",
            pdf_path="/tmp/paper.pdf",
        )

    def _create_session(self, *, arxiv_id=None, session_id=None, status="active"):
        return self.storage.paper_chat_sessions.create_paper_chat_session(
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

    def test_get_qa_status_reports_active_index_while_rebuild_is_building(self) -> None:
        self._insert_index(status="indexed")
        building = self.storage.paper_qa_index.create_paper_qa_index_build(self.arxiv_id, "docling")
        self.storage.paper_qa_index.update_paper_qa_index_build(
            building["build_id"],
            status="building",
            current_stage="create_chunk_embeddings",
        )
        failed = self.storage.paper_qa_index.create_paper_qa_index_build(self.arxiv_id, "docling")
        self.storage.paper_qa_index.update_paper_qa_index_build(
            failed["build_id"],
            status="build_failed",
            current_stage="chunk_document",
            failed_stage="chunk_document",
            error_message="chunking failed",
            artifact_status="cleanup_pending",
        )

        status = self.service.get_qa_status(self.arxiv_id)

        self.assertTrue(status["has_index"])
        self.assertEqual(status["active_collection_name"], "paper_2401")
        self.assertEqual(status["building_status"], "building")
        self.assertEqual(status["last_failed_build"]["failed_stage"], "chunk_document")
        self.assertGreaterEqual(status["cleanup_pending_count"], 1)

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
        with mock.patch.object(self.storage.paper_chat_sessions, "list_paper_chat_sessions", side_effect=RuntimeError("db boom")):
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

        messages = self.storage.paper_chat_messages.list_paper_chat_messages(session["session_id"], user_id=self.user_id)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["turn_id"], messages[1]["turn_id"])
        self.assertEqual(result["turn_id"], messages[0]["turn_id"])
        self.assertEqual(messages[1]["sources"], [{"source_id": "s1", "content": "source text"}])
        self.assertEqual(messages[1]["retrieval_debug_snapshot"], {"score": 0.8})
        refreshed_session = self.storage.paper_chat_sessions.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertTrue(result["session_summary_update"]["updated"])
        self.assertEqual(refreshed_session["summary_turn_count"], 1)
        self.assertIn("What is the method?", refreshed_session["summary"]["confirmed_facts"][0])

    def test_persist_completed_turn_raises_db_error_without_half_turn(self) -> None:
        session = self._create_session(session_id="failing-session")

        with mock.patch.object(
            self.storage.paper_qa_turns,
            "append_paper_qa_turn",
            side_effect=PaperQATurnPersistenceError("write failed"),
        ):
            with self.assertRaises(AppError) as ctx:
                self.service.persist_completed_turn(
                    chat_session=session,
                    question="Q",
                    answer="A",
                    source_payload=[],
                    retrieval_debug=None,
                    contextualized_question="Q",
                    question_contextualization={},
                )

        messages = self.storage.paper_chat_messages.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        self.assertEqual(ctx.exception.code, ErrorCode.DATABASE_WRITE_FAILED)
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
