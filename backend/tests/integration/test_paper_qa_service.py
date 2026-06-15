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

from core.errors import AppError, ErrorCode
from services.storage.database_service import DatabaseService, PaperQATurnPersistenceError
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

    module = sys.modules.get("services.embedding.embedding_service")
    if module is None:
        module = types.ModuleType("services.embedding.embedding_service")
        sys.modules[module.__name__] = module
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

    if "services.llm.generation_service" not in sys.modules:
        module = types.ModuleType("services.llm.generation_service")

        class _GenerationService:
            pass

        module.GenerationService = _GenerationService
        # PaperQAService 测试桩只隔离生成服务本体，但仍需保留真实模块的常量导出形状。
        module.QWEN_RERANK_COMPRESS_MODEL_NAME = "fake-rerank-compress"
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

    def test_get_qa_status_reports_active_index_while_rebuild_is_building(self) -> None:
        self._insert_index(status="indexed")
        building = self.db_service.create_paper_qa_index_build(self.arxiv_id, "docling")
        self.db_service.update_paper_qa_index_build(
            building["build_id"],
            status="building",
            current_stage="create_chunk_embeddings",
        )
        failed = self.db_service.create_paper_qa_index_build(self.arxiv_id, "docling")
        self.db_service.update_paper_qa_index_build(
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
        with mock.patch.object(self.db_service, "list_paper_chat_sessions", side_effect=RuntimeError("db boom")):
            session = self.service._resolve_chat_session(self.arxiv_id, {"user_id": self.user_id, "question": "fallback"})

        self.assertEqual(session, {})

    def test_build_qa_context_debugs_db_first_conversation_context_merge(self) -> None:
        self._insert_index(status="indexed")
        self.retrieval_service.chunks = [
            {
                "content": "retrieval evidence",
                "metadata": {"chunk_id": "chunk-1", "page_number": "1", "section_path": "Intro"},
            }
        ]
        self.memory_service.paper_conversation_turns = [
            {
                "turn_id": "db-turn",
                "question": "What was the previous answer?",
                "answer_summary": "It discussed retrieval.",
                "sources": [{"source_id": "db-source", "content": "db evidence"}],
            }
        ]
        payload = {
            "user_id": self.user_id,
            "question": "What about the method?",
            "conversation_context": [
                {
                    "turn_id": "payload-turn",
                    "question": "Payload question",
                    "answer_summary": "Payload answer",
                }
            ],
        }

        _qa_index, _search_results, qa_context, retrieval_debug = self.service.build_qa_context(self.arxiv_id, payload)

        short_term_debug = retrieval_debug["memory_modules"]["short_term_memory"]
        self.assertEqual(short_term_debug["merge_policy"], "db_authoritative_payload_compat_supplement")
        self.assertEqual(short_term_debug["backend_loaded_fields"], ["conversation_context"])
        self.assertEqual(short_term_debug["payload_received_fields"], ["conversation_context"])
        self.assertEqual(short_term_debug["payload_accepted_fields"], ["conversation_context"])
        self.assertEqual(short_term_debug["db_turn_count"], 1)
        self.assertEqual(short_term_debug["payload_turn_count"], 1)
        self.assertEqual(short_term_debug["merged_turn_count"], 2)
        self.assertEqual([turn["turn_id"] for turn in qa_context["conversation_context"]], ["db-turn", "payload-turn"])

    def test_build_qa_context_includes_session_summary_before_recent_turns(self) -> None:
        self._insert_index(status="indexed")
        session = self._create_session(session_id="summary-context-session")
        self.db_service.update_paper_chat_session_summary(
            session["session_id"],
            user_id=self.user_id,
            summary={
                "topic": "focus on experiments",
                "confirmed_facts": ["baseline A was already compared"],
                "user_preferences": ["skip background"],
                "task_progress": ["now asking about methods"],
                "source_clues": [{"source_id": "summary-source", "section_path": "Experiments"}],
            },
            summary_turn_count=4,
            summary_last_turn_id="turn-4",
            summary_updated_at="2026-06-09T00:00:00+00:00",
        )
        self.memory_service.paper_conversation_turns = [
            {"turn_id": "turn-5", "question": "Recent?", "answer_summary": "Recent answer."}
        ]
        self.retrieval_service.chunks = [{"content": "retrieval evidence", "metadata": {"chunk_id": "chunk-1"}}]

        _qa_index, _search_results, qa_context, retrieval_debug = self.service.build_qa_context(
            self.arxiv_id,
            {"user_id": self.user_id, "session_id": session["session_id"], "question": "continue"},
        )

        context_turn_ids = [turn["turn_id"] for turn in qa_context["conversation_context"]]
        summary_debug = retrieval_debug["memory_modules"]["short_term_memory"]["summary"]
        self.assertEqual(context_turn_ids, ["__session_summary__", "turn-5"])
        self.assertEqual(qa_context["conversation_context"][0]["context_type"], "session_summary")
        self.assertTrue(summary_debug["loaded"])
        self.assertEqual(summary_debug["summary_turn_count"], 4)
        self.assertEqual(summary_debug["summary_updated_at"], "2026-06-09T00:00:00+00:00")

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
        refreshed_session = self.db_service.get_paper_chat_session(session["session_id"], user_id=self.user_id)
        self.assertTrue(result["session_summary_update"]["updated"])
        self.assertEqual(refreshed_session["summary_turn_count"], 1)
        self.assertIn("What is the method?", refreshed_session["summary"]["confirmed_facts"][0])

    def test_persist_completed_turn_raises_db_error_without_half_turn(self) -> None:
        session = self._create_session(session_id="failing-session")

        with mock.patch.object(
            self.db_service,
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

        messages = self.db_service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        self.assertEqual(ctx.exception.code, ErrorCode.DATABASE_WRITE_FAILED)
        self.assertEqual(messages, [])

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
        self.assertEqual(source_payload[0]["source_id"], "p1")
        self.assertEqual(image_inputs[0]["source_id"], source_payload[1]["source_id"])
        self.assertEqual(source_payload[0]["parent_chunk_id"], "p1")
        self.assertEqual(source_payload[1]["asset_summary"], "Figure summary")
        self.assertIn("chunk_type", source_payload[2])
        self.assertTrue(truncated.endswith("..."))

    def test_structured_table_evidence_enters_prompt_sources_and_numeric_verifier(self) -> None:
        table_result = {
            "chunk_id": "chunk-table-results",
            "original_chunk_id": "orig-table-results",
            "chunk_type": "table",
            "asset_kind": "table",
            "asset_summary": "Table 2 Main results on the benchmark.",
            "asset_preview_text": "Full model 87.5 | w/o memory 84.1",
            "table_id": "paper-table-2",
            "page_number": 5,
            "section_path": "Results/Ablation",
            "source": "table-source",
            "table_structured_evidence": {
                "table_id": "paper-table-2",
                "matched_rows": ["Full model", "w/o memory"],
                "matched_columns": ["Accuracy"],
                "matched_cells": [
                    {
                        "row_index": 0,
                        "row_label": "Full model",
                        "col_name": "Accuracy",
                        "raw_value": "87.5",
                        "normalized_value": 87.5,
                        "unit": "%",
                        "confidence": 0.94,
                    },
                    {
                        "row_index": 1,
                        "row_label": "w/o memory",
                        "col_name": "Accuracy",
                        "raw_value": "84.1",
                        "normalized_value": 84.1,
                        "unit": "%",
                        "confidence": 0.92,
                    },
                ],
                "evidence_type": "cell_comparison",
                "numeric_operation": "difference",
                "computed_value": 3.4,
                "confidence": 0.91,
            },
        }

        context_pack = self.service.context_pack_builder.build([table_result])
        source_payload = context_pack["source_payload"]
        text_context = context_pack["text_context"]

        self.assertTrue(source_payload[0]["source_id"].startswith("table-structured-paper-table-2-chunk-table-results"))
        self.assertIn("Table Evidence:", text_context)
        self.assertIn("table_id: paper-table-2", text_context)
        self.assertIn("matched row: Full model, w/o memory", text_context)
        self.assertIn("matched column: Accuracy", text_context)
        self.assertIn("value: 87.5", text_context)
        self.assertIn("difference = 3.4", text_context)
        self.assertIn("source chunk: chunk-table-results", text_context)
        self.assertIn("Table Supplemental Context:", text_context)
        self.assertEqual(source_payload[0]["table_cell_citations"][1]["row_label"], "w/o memory")
        self.assertEqual(context_pack["context_budget_debug"]["table_evidence_count"], 1)
        self.assertEqual(context_pack["context_budget_debug"]["table_cell_evidence_count"], 2)
        self.assertTrue(context_pack["context_budget_debug"]["table_numeric_calculation_used"])

        prompt_assembly = self.service.answer_generator.prompt_context_builder.build_paper_qa_final_answer_context(
            question="Ablation 里去掉 memory 后下降多少？",
            contextualized_question="Ablation 里去掉 memory 后下降多少？",
            context_pack=context_pack,
        )
        self.assertIn("prefer structured Table Evidence blocks", prompt_assembly["text"])
        self.assertEqual(prompt_assembly["debug"]["table_evidence_count"], 1)
        self.assertEqual(prompt_assembly["debug"]["table_cell_evidence_count"], 2)

        passed = self.service.evidence_verifier.verify(
            answer=f"去掉 memory 后下降 3.4 个点，来源 {source_payload[0]['source_id']}。",
            sources=source_payload,
        )
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(passed["unsupported_numeric_values"], [])

        failed = self.service.evidence_verifier.verify(
            answer="去掉 memory 后下降 3.7 个点。",
            sources=source_payload,
        )
        self.assertEqual(failed["status"], "insufficient_evidence")
        self.assertIn("answer_numeric_value_not_in_table_evidence", failed["warnings"])

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
        self.assertEqual(result["verification_debug"]["status"], "passed")
        self.assertEqual(result["qa_observation"]["schema_version"], "paper_qa_observation_v1")
        self.assertEqual(result["qa_observation"]["retrieval_quality"], "weak")
        self.assertEqual(result["qa_observation"]["retrieval_quality_reason"], "source_chunks_too_few:1")
        self.assertEqual(result["retrieval_debug"]["qa_observation"]["source_count"], 1)
        self.assertIn("generation", result["retrieval_debug"])
        self.assertIn("verification", result["retrieval_debug"])
        self.assertIn("context_pack", result["retrieval_debug"])
        prompt_context_debug = result["retrieval_debug"]["generation"]["prompt_context"]
        self.assertEqual(
            prompt_context_debug["section_order"],
            ["system_instruction", "rag_evidence", "current_question"],
        )
        generate_call = next(call for call in generation_service.calls if call["method"] == "generate")
        generation_search_results = generate_call["kwargs"]["search_results"]
        self.assertEqual(len(generation_search_results), 1)
        self.assertEqual(generation_search_results[0]["chunk_type"], "prompt_context")
        self.assertEqual(generation_search_results[0]["text"].count("Relevant chunk content"), 1)
        self.assertEqual(result["sources"][0]["source_id"], "source-1-text-p1")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], "generated answer")
        self.assertEqual(messages[1]["retrieval_debug_snapshot"]["verification"]["status"], "passed")

    def test_build_qa_context_records_prompt_context_debug_for_question_rewrite(self) -> None:
        self._insert_index(status="indexed")
        session = self._create_session(session_id="qa-contextualization-debug")
        self.memory_service.paper_conversation_turns = [
            {
                "turn_id": "turn-1",
                "question": "What baseline did the paper compare?",
                "answer_summary": "It compared baseline A.",
                "sources": [{"source_id": "source-1", "content": "baseline evidence"}],
            }
        ]
        self.retrieval_service.chunks = [{"content": "retrieval evidence", "metadata": {"chunk_id": "chunk-1"}}]
        generation_service = _FakeGenerationWithResponse(
            response_text=(
                '{"contextualized_question":"What result follows from baseline A?",'
                '"is_follow_up":true,'
                '"referenced_turn_ids":["turn-1"],'
                '"referenced_source_ids":["source-1"],'
                '"memory_reason":"Resolved the follow-up from the previous turn."}'
            )
        )
        service = self._make_service(generation_service=generation_service)

        _qa_index, _search_results, _qa_context, retrieval_debug = service.build_qa_context(
            self.arxiv_id,
            {"user_id": self.user_id, "session_id": session["session_id"], "question": "那结果呢？"},
        )

        contextualization_debug = retrieval_debug["question_contextualization"]["prompt_context"]
        complete_call = next(call for call in generation_service.calls if call["method"] == "complete_with_qwen")
        prompt = complete_call["prompt"]
        self.assertEqual(
            contextualization_debug["section_order"],
            ["system_instruction", "recent_turns", "agent_state", "current_question"],
        )
        self.assertIn("Prompt context sections:", prompt)
        self.assertIn("Allowed recent source references", prompt)
        self.assertNotIn("Recent QA turns:", prompt)

    def test_answer_question_marks_empty_answer_as_insufficient_evidence(self) -> None:
        self._insert_index(status="indexed")
        session = self._create_session(session_id="qa-empty-answer")
        retrieval_service = _FakeRetrievalService(
            chunks=[
                {
                    "content": "Relevant chunk content",
                    "chunk_type": "text",
                    "page_number": 1,
                    "source": "body",
                }
            ],
            debug={"provider": "fake-retrieval"},
        )
        generation_service = _FakeGenerationWithResponse(response_text="")
        service = self._make_service(retrieval_service=retrieval_service, generation_service=generation_service)

        result = service.answer_question(
            self.arxiv_id,
            {"question": "What is the contribution?", "user_id": self.user_id, "session_id": session["session_id"]},
        )

        messages = self.db_service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        retrieval_service.cleanup()

        self.assertIn("当前检索证据不足", result["answer"])
        self.assertEqual(result["verification_debug"]["status"], "insufficient_evidence")
        self.assertEqual(result["qa_observation"]["answer_insufficient_evidence"], "yes")
        self.assertIn("retry_with_expanded_context", result["qa_observation"]["recommended_repair_actions"])
        self.assertTrue(result["qa_observation"]["repair_action_details"])
        self.assertEqual(messages[1]["content"], result["answer"])

    def test_answer_question_reports_persistence_failure_without_half_turn(self) -> None:
        self._insert_index(status="indexed")
        session = self._create_session(session_id="qa-persistence-fail")
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

        with mock.patch.object(
            self.db_service,
            "append_paper_qa_turn",
            side_effect=PaperQATurnPersistenceError("write failed"),
        ):
            with self.assertRaises(AppError) as ctx:
                service.answer_question(
                    self.arxiv_id,
                    {"question": "What is the contribution?", "user_id": self.user_id, "session_id": session["session_id"]},
                )

        messages = self.db_service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        retrieval_service.cleanup()
        self.assertEqual(ctx.exception.code, ErrorCode.DATABASE_WRITE_FAILED)
        self.assertEqual(messages, [])

    def test_answer_question_raises_vector_error_when_retrieval_returns_no_chunks(self) -> None:
        self._insert_index(status="indexed")
        retrieval_service = _FakeRetrievalService(chunks=[])
        service = self._make_service(retrieval_service=retrieval_service)

        with self.assertRaises(AppError) as ctx:
            service.answer_question(self.arxiv_id, {"question": "Empty?", "user_id": self.user_id})

        retrieval_service.cleanup()
        self.assertEqual(ctx.exception.code, ErrorCode.VECTOR_STORE_ERROR)
        self.assertIn("检索服务没有返回可用的论文片段", str(ctx.exception))

        error_payload = ctx.exception.to_payload()
        self.assertEqual(error_payload["qa_observation"]["retrieval_quality"], "failed")
        self.assertIn("retry_with_query_rewrite", error_payload["qa_observation"]["recommended_repair_actions"])

    def test_answer_question_observation_reports_rerank_fallback(self) -> None:
        self._insert_index(status="indexed")
        session = self._create_session(session_id="qa-rerank-fallback")
        retrieval_service = _FakeRetrievalService(
            chunks=[
                {
                    "content": "Relevant method evidence with enough detail to support a grounded answer. " * 2,
                    "chunk_type": "text",
                    "page_number": 1,
                    "source": "body",
                },
                {
                    "content": "Additional experiment evidence with enough detail to avoid short evidence classification. " * 2,
                    "chunk_type": "text",
                    "page_number": 2,
                    "source": "body",
                },
            ],
            debug={
                "provider": "fake-retrieval",
                "llm_rerank": {
                    "enabled": True,
                    "applied": False,
                    "mode": "fallback_fused",
                    "reason": "rerank timed out",
                },
            },
        )
        generation_service = _FakeGenerationWithResponse(response_text="generated answer")
        service = self._make_service(retrieval_service=retrieval_service, generation_service=generation_service)

        result = service.answer_question(
            self.arxiv_id,
            {"question": "What is the method?", "user_id": self.user_id, "session_id": session["session_id"]},
        )

        retrieval_service.cleanup()
        self.assertEqual(result["qa_observation"]["retrieval_stage_status"]["rerank"]["status"], "fallback")
        self.assertEqual(result["qa_observation"]["rerank_failed_reason"], "rerank timed out")
        self.assertIn("retry_with_query_rewrite", result["qa_observation"]["recommended_repair_actions"])

    def test_answer_question_reports_generation_failure_without_fallback_answer(self) -> None:
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

        with self.assertRaises(AppError) as ctx:
            service.answer_question(
                self.arxiv_id,
                {"question": "Why does it work?", "user_id": self.user_id, "session_id": session["session_id"]},
            )

        messages = self.db_service.list_paper_chat_messages(session["session_id"], user_id=self.user_id)
        retrieval_service.cleanup()

        self.assertEqual(ctx.exception.code, ErrorCode.LLM_GENERATION_FAILED)
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
