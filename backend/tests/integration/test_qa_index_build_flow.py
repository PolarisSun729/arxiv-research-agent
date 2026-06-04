import gc
import importlib.util
import json
import re
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from services.storage.database_service import DatabaseService
from tests.helpers import FakeGenerationService, build_database_service


def _load_builder_and_manager():
    backend_dir = Path(__file__).resolve().parents[2]

    packages = {
        "services": backend_dir / "services",
        "services.arxiv": backend_dir / "services" / "arxiv",
        "services.document": backend_dir / "services" / "document",
        "services.paper_qa": backend_dir / "services" / "paper_qa",
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
            def search(self, **_kwargs):
                return {"papers": []}

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

    if "services.embedding.embedding_service" not in sys.modules:
        module = types.ModuleType("services.embedding.embedding_service")

        class _EmbeddingConfig:
            def __init__(self, provider="fake", model_name="fake-embedding-model", dimension=8):
                self.provider = provider
                self.model_name = model_name
                self.dimension = dimension

        class _EmbeddingService:
            pass

        module.EmbeddingConfig = _EmbeddingConfig
        module.EmbeddingService = _EmbeddingService
        sys.modules[module.__name__] = module

    if "services.llm.generation_service" not in sys.modules:
        module = types.ModuleType("services.llm.generation_service")

        class _GenerationService:
            pass

        module.GenerationService = _GenerationService
        module.QWEN_RERANK_COMPRESS_MODEL_NAME = "fake-rerank-compress"
        sys.modules[module.__name__] = module

    if "services.storage.vector_store_service" not in sys.modules:
        module = types.ModuleType("services.storage.vector_store_service")

        class _VectorDBConfig:
            def __init__(self, provider="milvus", index_mode="default"):
                self.provider = provider
                self.index_mode = index_mode

        class _VectorStoreService:
            pass

        module.VectorDBConfig = _VectorDBConfig
        module.VectorStoreService = _VectorStoreService
        sys.modules[module.__name__] = module

    builder_module_name = "services.paper_qa.paper_qa_index_builder"
    if builder_module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            builder_module_name,
            backend_dir / "services" / "paper_qa" / "paper_qa_index_builder.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[builder_module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)

    manager_module_name = "services.paper_qa.index_job_manager"
    if manager_module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            manager_module_name,
            backend_dir / "services" / "paper_qa" / "index_job_manager.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[manager_module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)

    return sys.modules[builder_module_name], sys.modules[manager_module_name]


builder_module, manager_module = _load_builder_and_manager()
PaperQAIndexBuilder = builder_module.PaperQAIndexBuilder
IndexJobManager = manager_module.IndexJobManager


class _FakeOaiDbService:
    def __init__(self, paper=None):
        self.paper = dict(paper) if paper else None
        self.calls = []

    def get_paper(self, arxiv_id: str):
        self.calls.append({"method": "get_paper", "arxiv_id": arxiv_id})
        if self.paper is None:
            return None
        payload = dict(self.paper)
        payload.setdefault("arxiv_id", arxiv_id)
        return payload


class _FakeLoadingService:
    def __init__(self, *, document=None, page_map=None, chunk_file_path="chunk-output.json", fail_stage=None):
        self.document = document or {"sections": [{"title": "Intro", "text": "Chunk text"}]}
        self.page_map = page_map or [{"page_number": 1}]
        self.chunk_file_path = chunk_file_path
        self.fail_stage = fail_stage
        self.calls = []

    def load_pdf(self, pdf_path: str, method: str = "docling"):
        self.calls.append({"method": "load_pdf", "pdf_path": pdf_path, "loading_method": method})
        if self.fail_stage == "load_pdf_document":
            raise RuntimeError("loading failed")
        return self.document

    def get_page_map(self):
        self.calls.append({"method": "get_page_map"})
        return list(self.page_map)

    def save_document(self, **kwargs):
        self.calls.append({"method": "save_document", "kwargs": dict(kwargs)})
        return self.chunk_file_path


class _FakeChunkingService:
    def __init__(self, *, chunks=None, fail_stage=None):
        self.chunks = list(
            chunks
            or [
                {"content": "Chunk text", "metadata": {"chunk_type": "text", "chunk_id": "chunk-1"}},
                {"content": "Figure summary", "metadata": {"chunk_type": "figure", "chunk_id": "chunk-2"}},
            ]
        )
        self.fail_stage = fail_stage
        self.calls = []

    def chunk_docling(self, document, metadata=None, page_map=None):
        self.calls.append({"method": "chunk_docling", "metadata": dict(metadata or {}), "page_count": len(page_map or [])})
        if self.fail_stage == "chunk_document":
            raise RuntimeError("chunking failed")
        return {"chunks": [dict(chunk) for chunk in self.chunks], "document": document}

    def chunk_pymupdf(self, document, method="by_titles", metadata=None, page_map=None):
        self.calls.append(
            {
                "method": "chunk_pymupdf",
                "chunk_method": method,
                "metadata": dict(metadata or {}),
                "page_count": len(page_map or []),
            }
        )
        if self.fail_stage == "chunk_document":
            raise RuntimeError("chunking failed")
        return {"chunks": [dict(chunk) for chunk in self.chunks], "document": document}


class _FakeEmbeddingConfig:
    def __init__(self, provider="fake", model_name="fake-embedding-model", dimension=8):
        self.provider = provider
        self.model_name = model_name
        self.dimension = dimension


class _FakeEmbeddingService:
    def __init__(self, *, model_name="fake-embedding-model", fail_stage=None):
        self.model_name = model_name
        self.fail_stage = fail_stage
        self.calls = []

    def get_default_embedding_config(self):
        return _FakeEmbeddingConfig(model_name=self.model_name)

    def create_embeddings(self, input_data, config):
        self.calls.append({"method": "create_embeddings", "config_model": config.model_name})
        if self.fail_stage == "create_chunk_embeddings":
            raise RuntimeError("embedding failed")
        chunks = list((input_data or {}).get("chunks") or [])
        embeddings = []
        for index, chunk in enumerate(chunks, start=1):
            embeddings.append(
                {
                    "id": chunk.get("metadata", {}).get("chunk_id", f"chunk-{index}"),
                    "content": chunk.get("content", ""),
                    "embedding": [float(index), float(index) / 10.0, 0.5],
                    "metadata": dict(chunk.get("metadata", {}) or {}),
                }
            )
        return embeddings, {"input_count": len(embeddings)}

    def save_embeddings(self, filename: str, embeddings):
        self.calls.append({"method": "save_embeddings", "filename": filename, "count": len(embeddings)})
        return json.dumps({"filename": filename, "embeddings": embeddings}, ensure_ascii=True)


class _FakeVectorStoreService:
    def __init__(self, *, fail_stage=None):
        self.fail_stage = fail_stage
        self.calls = []

    @staticmethod
    def _stable_collection_name(filename: str) -> str:
        slug = re.sub(r"[^0-9A-Za-z]+", "_", str(filename or "").strip()).strip("_").lower()
        return f"qa_{slug or 'default'}"

    def index_embeddings(self, embedding_file: str, config):
        self.calls.append(
            {
                "method": "index_embeddings",
                "provider": getattr(config, "provider", None),
                "index_mode": getattr(config, "index_mode", None),
            }
        )
        if self.fail_stage == "index_embeddings_to_vector_store":
            raise RuntimeError("vector write failed")
        payload = json.loads(embedding_file)
        collection_name = self._stable_collection_name(payload.get("filename", ""))
        return {
            "collection_name": collection_name,
            "database": getattr(config, "provider", "fake"),
            "index_mode": getattr(config, "index_mode", "default"),
            "total_vectors": len(payload.get("embeddings") or []),
        }


class _InlineThread:
    def __init__(self, *, target=None, args=None, kwargs=None, **_thread_kwargs):
        self.target = target
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.started = False

    def start(self):
        self.started = True
        if self.target is not None:
            self.target(*self.args, **self.kwargs)


class _NoopThread:
    instances = []

    def __init__(self, *, target=None, args=None, kwargs=None, **_thread_kwargs):
        self.target = target
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.started = False
        _NoopThread.instances.append(self)

    def start(self):
        self.started = True


class _FakeJobBuilder:
    def __init__(self, *, result=None, error=None):
        self.result = result or {"status": "indexed"}
        self.error = error
        self.calls = []

    def validate_loading_method(self, loading_method: str) -> str:
        self.calls.append({"method": "validate_loading_method", "loading_method": loading_method})
        return str(loading_method or "docling").strip().lower()

    def build_qa_index(self, arxiv_id: str, loading_method: str = "docling", progress_callback=None):
        self.calls.append({"method": "build_qa_index", "arxiv_id": arxiv_id, "loading_method": loading_method})
        if progress_callback is not None:
            progress_callback(current_stage="build", progress=50, message="building")
        if self.error is not None:
            raise self.error
        return dict(self.result)


class _BaseIndexTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.db_service = build_database_service(DatabaseService)
        self.arxiv_id = "2401.00001"

    def tearDown(self) -> None:
        temp_db = getattr(self.db_service, "_test_temp_db", None)
        self.db_service = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _paper_payload(self, arxiv_id=None, title="Paper Title", abstract="Paper abstract"):
        value = arxiv_id or self.arxiv_id
        return {
            "arxiv_id": value,
            "title": title,
            "authors": ["Alice", "Bob"],
            "abstract": abstract,
            "categories": ["cs.CL"],
            "published_date": "2024-01-01",
            "url": f"https://arxiv.org/abs/{value}",
            "embedding_id": "",
            "embedding_model": "",
        }

    def _make_builder(self, **overrides):
        loading_service = overrides.pop("loading_service", None) or _FakeLoadingService()
        chunking_service = overrides.pop("chunking_service", None) or _FakeChunkingService()
        embedding_service = overrides.pop("embedding_service", None) or _FakeEmbeddingService()
        vector_store_service = overrides.pop("vector_store_service", None) or _FakeVectorStoreService()
        generation_service = overrides.pop("generation_service", None) or FakeGenerationService(response_text="compressed")
        arxiv_service_factory = overrides.pop("arxiv_service_factory", None) or (lambda: types.SimpleNamespace(download_pdf=lambda _url, _id: "tmp/fake-paper.pdf"))
        oai_db_service = overrides.pop("oai_db_service", None)

        builder = PaperQAIndexBuilder(
            db_service=overrides.pop("db_service", self.db_service),
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            generation_service=generation_service,
            arxiv_service_factory=arxiv_service_factory,
            oai_db_service=oai_db_service,
            get_embedding_config=overrides.pop("get_embedding_config", None),
            loading_service_factory=lambda: loading_service,
            chunking_service_factory=lambda: chunking_service,
        )
        return builder, loading_service, chunking_service, embedding_service, vector_store_service


class PaperQAIndexBuilderFlowTests(_BaseIndexTestCase):
    def test_load_paper_metadata_uses_primary_database_first(self) -> None:
        paper = self._paper_payload(title="Primary DB Paper")
        self.db_service.add_paper(paper)
        oai_db = _FakeOaiDbService(paper={"title": "Should not be used", "abstract": "unused"})
        builder, *_ = self._make_builder(oai_db_service=oai_db)

        loaded = builder.load_paper_metadata(self.arxiv_id)

        self.assertEqual(loaded["title"], "Primary DB Paper")
        self.assertEqual(oai_db.calls, [])

    def test_load_paper_metadata_falls_back_to_oai_and_persists(self) -> None:
        oai_db = _FakeOaiDbService(
            paper={
                "title": "OAI Paper",
                "authors": ["OAI Author"],
                "abstract": "OAI abstract",
                "categories": ["cs.AI"],
                "created": "2024-02-01",
                "abs_url": f"https://arxiv.org/abs/{self.arxiv_id}",
            }
        )
        builder, *_ = self._make_builder(oai_db_service=oai_db)

        loaded = builder.load_paper_metadata(self.arxiv_id)
        persisted = self.db_service.get_paper(self.arxiv_id)

        self.assertEqual(loaded["title"], "OAI Paper")
        self.assertEqual(persisted["abstract"], "OAI abstract")
        self.assertEqual(len(oai_db.calls), 1)

    def test_load_paper_metadata_falls_back_to_arxiv_and_persists(self) -> None:
        builder, *_ = self._make_builder(oai_db_service=_FakeOaiDbService())

        class _PatchedArxivSearchService:
            def search(self, **_kwargs):
                return {
                    "papers": [
                        {
                            "arxiv_id": self_arxiv_id,
                            "title": "arXiv Paper",
                            "authors": ["Remote Author"],
                            "summary": "Remote abstract",
                            "categories": ["cs.IR"],
                            "published": "2024-03-03",
                            "abs_url": f"https://arxiv.org/abs/{self_arxiv_id}",
                        }
                    ]
                }

        self_arxiv_id = self.arxiv_id
        with mock.patch.object(builder_module, "ArxivSearchService", _PatchedArxivSearchService):
            loaded = builder.load_paper_metadata(self.arxiv_id)

        persisted = self.db_service.get_paper(self.arxiv_id)
        self.assertEqual(loaded["title"], "arXiv Paper")
        self.assertEqual(persisted["abstract"], "Remote abstract")

    def test_build_qa_index_success_updates_index_record_and_pipeline_metadata(self) -> None:
        self.db_service.add_paper(self._paper_payload())
        builder, loading_service, chunking_service, embedding_service, vector_store_service = self._make_builder()

        result = builder.build_qa_index(self.arxiv_id, loading_method="docling")
        record = self.db_service.get_paper_qa_index(self.arxiv_id)

        self.assertEqual(result["status"], "success")
        self.assertEqual(record["status"], "indexed")
        self.assertEqual(record["chunk_count"], 2)
        self.assertEqual(record["embedding_model"], "fake-embedding-model")
        self.assertEqual(record["pdf_path"], "tmp/fake-paper.pdf")
        self.assertEqual(record["collection_name"], "qa_2401_00001_pdf")
        self.assertEqual(loading_service.calls[0]["method"], "load_pdf")
        self.assertEqual(chunking_service.calls[0]["method"], "chunk_docling")
        self.assertEqual(embedding_service.calls[0]["method"], "create_embeddings")
        self.assertEqual(vector_store_service.calls[0]["method"], "index_embeddings")

    def test_collection_name_generation_is_stable(self) -> None:
        vector_store_service = _FakeVectorStoreService()

        first = vector_store_service.index_embeddings(json.dumps({"filename": "2401.00001.pdf", "embeddings": []}), types.SimpleNamespace(provider="milvus", index_mode="default"))
        second = vector_store_service.index_embeddings(json.dumps({"filename": "2401.00001.pdf", "embeddings": []}), types.SimpleNamespace(provider="milvus", index_mode="default"))

        self.assertEqual(first["collection_name"], second["collection_name"])
        self.assertEqual(first["collection_name"], "qa_2401_00001_pdf")

    def test_build_qa_index_records_failure_when_loading_fails(self) -> None:
        self.db_service.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(loading_service=_FakeLoadingService(fail_stage="load_pdf_document"))

        with self.assertRaises(RuntimeError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.db_service.get_paper_qa_index(self.arxiv_id)
        self.assertEqual(str(ctx.exception), "loading failed")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(getattr(ctx.exception, "error_stage", ""), "load_pdf_document")

    def test_build_qa_index_records_failure_when_chunking_fails(self) -> None:
        self.db_service.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(chunking_service=_FakeChunkingService(fail_stage="chunk_document"))

        with self.assertRaises(RuntimeError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.db_service.get_paper_qa_index(self.arxiv_id)
        self.assertEqual(str(ctx.exception), "chunking failed")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(getattr(ctx.exception, "error_stage", ""), "chunk_document")

    def test_build_qa_index_records_failure_when_embedding_fails(self) -> None:
        self.db_service.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(embedding_service=_FakeEmbeddingService(fail_stage="create_chunk_embeddings"))

        with self.assertRaises(RuntimeError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.db_service.get_paper_qa_index(self.arxiv_id)
        self.assertEqual(str(ctx.exception), "embedding failed")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(getattr(ctx.exception, "error_stage", ""), "create_chunk_embeddings")

    def test_build_qa_index_records_failure_when_vector_write_fails(self) -> None:
        self.db_service.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(vector_store_service=_FakeVectorStoreService(fail_stage="index_embeddings_to_vector_store"))

        with self.assertRaises(RuntimeError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.db_service.get_paper_qa_index(self.arxiv_id)
        self.assertEqual(str(ctx.exception), "vector write failed")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(getattr(ctx.exception, "error_stage", ""), "index_embeddings_to_vector_store")


class IndexJobManagerFlowTests(_BaseIndexTestCase):
    def test_submit_job_creates_pending_job_and_reuses_active_duplicate(self) -> None:
        _NoopThread.instances = []
        builder = _FakeJobBuilder()
        manager = IndexJobManager(db_service=self.db_service, qa_index_builder=builder)

        with mock.patch.object(manager_module.threading, "Thread", _NoopThread):
            first_job = manager.submit_job(self.arxiv_id, "docling")
            second_job = manager.submit_job(self.arxiv_id, "docling")

        latest = self.db_service.get_latest_paper_index_job(self.arxiv_id)

        self.assertEqual(first_job["status"], "pending")
        self.assertEqual(second_job["job_id"], first_job["job_id"])
        self.assertEqual(latest["job_id"], first_job["job_id"])
        self.assertEqual(len(_NoopThread.instances), 1)
        self.assertTrue(_NoopThread.instances[0].started)

    def test_run_job_success_updates_job_status_and_latest_job(self) -> None:
        builder = _FakeJobBuilder(result={"status": "success"})
        manager = IndexJobManager(db_service=self.db_service, qa_index_builder=builder)
        job = self.db_service.create_paper_index_job(self.arxiv_id, "docling")

        manager.run_job(job["job_id"], self.arxiv_id, "docling")

        stored = self.db_service.get_paper_index_job(job["job_id"])
        latest = self.db_service.get_latest_paper_index_job(self.arxiv_id)
        self.assertEqual(stored["status"], "success")
        self.assertEqual(stored["current_stage"], "mark_index_success")
        self.assertEqual(stored["progress"], 100)
        self.assertEqual(latest["job_id"], job["job_id"])

    def test_run_job_failure_updates_failed_status_and_error_message(self) -> None:
        error = HTTPException(status_code=500, detail="builder failed")
        setattr(error, "error_stage", "chunk_document")
        builder = _FakeJobBuilder(error=error)
        manager = IndexJobManager(db_service=self.db_service, qa_index_builder=builder)
        job = self.db_service.create_paper_index_job(self.arxiv_id, "docling")

        manager.run_job(job["job_id"], self.arxiv_id, "docling")

        stored = self.db_service.get_paper_index_job(job["job_id"])
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["current_stage"], "chunk_document")
        self.assertEqual(stored["error_message"], "builder failed")

    def test_submit_job_with_inline_thread_reaches_terminal_state(self) -> None:
        builder = _FakeJobBuilder(result={"status": "success"})
        manager = IndexJobManager(db_service=self.db_service, qa_index_builder=builder)

        with mock.patch.object(manager_module.threading, "Thread", _InlineThread):
            job = manager.submit_job(self.arxiv_id, "docling")

        latest = self.db_service.get_latest_paper_index_job(self.arxiv_id)
        self.assertEqual(job["job_id"], latest["job_id"])
        self.assertEqual(latest["status"], "success")


if __name__ == "__main__":
    unittest.main()
