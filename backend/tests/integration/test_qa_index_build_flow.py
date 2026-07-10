import gc
import importlib.util
import json
import re
import shutil
import sys
import threading
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from tests.helpers import FakeGenerationService, build_storage_container


def _needs_reload(module_name: str, required_attrs: tuple[str, ...]) -> bool:
    module = sys.modules.get(module_name)
    if module is None:
        return True
    # 其他集成测试会注册轻量 stub 隔离重依赖；这里必须重新加载真实源码，才能验证建索引链路本身。
    return not getattr(module, "__file__", None) or any(not hasattr(module, attr) for attr in required_attrs)


def _load_builder_and_manager():
    backend_dir = Path(__file__).resolve().parents[2]
    created_stub_modules = []

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
        created_stub_modules.append(module.__name__)

    if "services.llm.generation_service" not in sys.modules:
        module = types.ModuleType("services.llm.generation_service")

        class _GenerationService:
            pass

        module.GenerationService = _GenerationService
        module.QWEN_RERANK_COMPRESS_MODEL_NAME = "fake-rerank-compress"
        sys.modules[module.__name__] = module
        created_stub_modules.append(module.__name__)
    elif not hasattr(sys.modules["services.llm.generation_service"], "QWEN_RERANK_COMPRESS_MODEL_NAME"):
        # 混跑时可能复用其他测试留下的生成服务 stub；补齐常量即可保持 builder 导入契约。
        sys.modules["services.llm.generation_service"].QWEN_RERANK_COMPRESS_MODEL_NAME = "fake-rerank-compress"

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
        created_stub_modules.append(module.__name__)
    elif not hasattr(sys.modules["services.storage.vector_store_service"], "VectorDBConfig"):
        class _VectorDBConfig:
            def __init__(self, provider="milvus", index_mode="default"):
                self.provider = provider
                self.index_mode = index_mode

        # 混跑时可能复用其他测试留下的向量库 stub；补齐配置类型即可保持 builder 导入契约。
        sys.modules["services.storage.vector_store_service"].VectorDBConfig = _VectorDBConfig

    builder_module_name = "services.paper_qa.paper_qa_index_builder"
    if _needs_reload(builder_module_name, ("PaperQAIndexBuilder", "AppError", "ErrorCode")):
        sys.modules.pop(builder_module_name, None)
        spec = importlib.util.spec_from_file_location(
            builder_module_name,
            backend_dir / "services" / "paper_qa" / "paper_qa_index_builder.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[builder_module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)

    manager_module_name = "services.paper_qa.index_job_manager"
    if _needs_reload(manager_module_name, ("IndexJobManager",)):
        sys.modules.pop(manager_module_name, None)
        spec = importlib.util.spec_from_file_location(
            manager_module_name,
            backend_dir / "services" / "paper_qa" / "index_job_manager.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[manager_module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)

    # builder/manager 已完成真实源码加载；移除本 loader 创建的轻量依赖 stub，避免污染后续真实单测。
    for stub_name in created_stub_modules:
        sys.modules.pop(stub_name, None)
    return sys.modules[builder_module_name], sys.modules[manager_module_name]


builder_module, manager_module = _load_builder_and_manager()
PaperQAIndexBuilder = builder_module.PaperQAIndexBuilder
IndexJobManager = manager_module.IndexJobManager
AppError = builder_module.AppError
ErrorCode = builder_module.ErrorCode


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
        retrieval_indexes = list((input_data or {}).get("retrieval_indexes") or [])
        self.calls.append(
            {
                "method": "create_embeddings",
                "config_model": config.model_name,
                "retrieval_index_count": len(retrieval_indexes),
            }
        )
        if self.fail_stage == "create_chunk_embeddings":
            raise RuntimeError("embedding failed")
        chunks = list((input_data or {}).get("chunks") or [])
        chunk_lookup = {
            str((chunk.get("metadata", {}) or {}).get("chunk_id") or chunk.get("chunk_id") or index): chunk
            for index, chunk in enumerate(chunks, start=1)
        }
        embeddings = []
        vector_routes = {"vector", "vector_original", "vector_rewrite", "vector_hyde"}
        if retrieval_indexes:
            # fake 与真实 embedding service 一样消费 retrieval index，避免集成测试继续绑定旧的一 chunk 一向量假设。
            embedding_sources = []
            for index_payload in retrieval_indexes:
                index_text = str(index_payload.get("index_text") or "").strip()
                routes = {str(route) for route in index_payload.get("enabled_routes", [])}
                if not index_text or (routes and not routes & vector_routes):
                    continue
                chunk = chunk_lookup.get(str(index_payload.get("chunk_id"))) or {}
                metadata = {
                    **dict((chunk.get("metadata", {}) or {})),
                    "content": chunk.get("content", ""),
                    "retrieval_index_id": index_payload.get("index_id", ""),
                    "retrieval_index_type": index_payload.get("index_type", ""),
                    "retrieval_index_text": index_text,
                    "retrieval_index_weight": index_payload.get("index_weight", 1.0),
                    "retrieval_index_enabled_routes": list(index_payload.get("enabled_routes", [])),
                }
                embedding_sources.append(
                    {
                        "id": index_payload.get("index_id") or index_payload.get("chunk_id"),
                        "content": index_text,
                        "metadata": metadata,
                    }
                )
        else:
            embedding_sources = [
                {
                    "id": chunk.get("metadata", {}).get("chunk_id", f"chunk-{index}"),
                    "content": chunk.get("content", ""),
                    "metadata": dict(chunk.get("metadata", {}) or {}),
                }
                for index, chunk in enumerate(chunks, start=1)
            ]
        for index, source in enumerate(embedding_sources, start=1):
            embeddings.append(
                {
                    "id": source.get("id", f"chunk-{index}"),
                    "content": source.get("content", ""),
                    "embedding": [float(index), float(index) / 10.0, 0.5],
                    "metadata": dict(source.get("metadata", {}) or {}),
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

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        self.calls.append({"method": "delete_collection", "provider": provider, "collection_name": collection_name})
        return bool(collection_name)


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
        self.storage = build_storage_container()
        self.arxiv_id = "2401.00001"

    def tearDown(self) -> None:
        self._cleanup_retrieval_index_artifacts()
        temp_db = getattr(self.storage, "_test_temp_db", None)
        self.storage = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _cleanup_retrieval_index_artifacts(self) -> None:
        """测试会真实落 retrieval index 文件；用例结束时只清理当前测试数据库记录到的文件。"""
        if self.storage is None:
            return
        paths = set()
        active = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)
        if active and active.get("retrieval_index_file"):
            paths.add(str(active["retrieval_index_file"]))
        if active and active.get("sparse_index_dir"):
            paths.add(str(active["sparse_index_dir"]))
        for build in self.storage.paper_qa_index.list_paper_qa_index_builds(self.arxiv_id, limit=20):
            if build.get("retrieval_index_file"):
                paths.add(str(build["retrieval_index_file"]))
            if build.get("sparse_index_dir"):
                paths.add(str(build["sparse_index_dir"]))
        for path_text in paths:
            path = Path(path_text)
            if not path.is_absolute():
                # 构建器将相对 artifact 路径解释为 backend 相对路径，测试清理必须遵循同一契约。
                path = Path(__file__).resolve().parents[2] / path
            try:
                if path.is_file() and "02-retrieval-indexes" in path.parts:
                    path.unlink()
                elif path.is_dir() and "02-sparse-indexes" in path.parts:
                    # sparse artifact 是目录布局，测试结束只清理本用例登记到 DB 的目录。
                    shutil.rmtree(path)
            except OSError:
                pass

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
            paper_qa_index_store=self.storage.paper_qa_index,
            paper_catalog_store=self.storage.paper_catalog,
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
        self.storage.paper_catalog.add_paper(paper)
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
        persisted = self.storage.paper_catalog.get_paper(self.arxiv_id)

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

        persisted = self.storage.paper_catalog.get_paper(self.arxiv_id)
        self.assertEqual(loaded["title"], "arXiv Paper")
        self.assertEqual(persisted["abstract"], "Remote abstract")

    def test_build_qa_index_success_updates_index_record_and_pipeline_metadata(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        builder, loading_service, chunking_service, embedding_service, vector_store_service = self._make_builder()

        result = builder.build_qa_index(self.arxiv_id, loading_method="docling")
        record = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)

        self.assertEqual(result["status"], "success")
        self.assertEqual(record["status"], "indexed")
        self.assertEqual(record["chunk_count"], 2)
        self.assertEqual(record["embedding_model"], "fake-embedding-model")
        self.assertEqual(record["pdf_path"], "tmp/fake-paper.pdf")
        self.assertEqual(record["chunk_file"], "chunk-output.json")
        self.assertEqual(record["retrieval_index_file"], result["retrieval_index_file"])
        self.assertEqual(record["retrieval_index_count"], result["retrieval_index_count"])
        self.assertEqual(json.loads(record["retrieval_index_types"]), result["retrieval_index_types"])
        self.assertEqual(record["retrieval_index_version"], result["retrieval_index_version"])
        self.assertEqual(record["sparse_index_manifest_file"], result["sparse_index_manifest_file"])
        self.assertEqual(record["sparse_index_document_count"], result["sparse_index_document_count"])
        self.assertEqual(record["sparse_index_token_count"], result["sparse_index_token_count"])
        self.assertEqual(record["sparse_index_schema_version"], result["sparse_index_schema_version"])
        self.assertEqual(record["sparse_index_source_file"], result["sparse_index_source_file"])
        self.assertEqual(record["sparse_index_source_hash"], result["sparse_index_source_hash"])
        self.assertEqual(result["sparse_index_source_file"], result["chunk_file"])
        self.assertEqual(result["sparse_index_document_count"], result["chunk_count"])
        self.assertIn('"filename": "2401.00001_', record["embedding_file"])
        self.assertIn("_pdf", record["collection_name"])
        self.assertEqual(record["loading_method"], "docling")
        self.assertEqual(record["chunking_strategy"], "docling_sections")
        self.assertEqual(record["current_stage"], "activate_index")
        self.assertEqual(record["failed_stage"], "")
        self.assertEqual(record["error_message"], "")
        self.assertIsNotNone(record["indexed_at"])
        self.assertIsNotNone(record["active_build_id"])
        self.assertNotEqual(record["active_index_version"], "legacy")
        self.assertGreaterEqual(result["retrieval_index_count"], 4)
        self.assertEqual(result["retrieval_index_type_counts"]["body"], 2)
        self.assertEqual(result["retrieval_index_type_counts"]["section_anchor"], 2)
        self.assertGreaterEqual(result["retrieval_index_generation_error_count"], 1)
        artifact_path = Path(result["retrieval_index_file"])
        self.assertTrue(artifact_path.is_file())
        artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact_payload["index_count"], result["retrieval_index_count"])
        self.assertEqual(len(artifact_payload["retrieval_indexes"]), result["retrieval_index_count"])
        self.assertTrue(all(item["paper_id"] == self.arxiv_id for item in artifact_payload["retrieval_indexes"]))
        self.assertTrue(all(item["chunk_id"] in artifact_payload["chunk_ids"] for item in artifact_payload["retrieval_indexes"]))
        sparse_manifest_path = Path(result["sparse_index_manifest_file"])
        self.assertTrue(sparse_manifest_path.is_file())
        sparse_manifest = json.loads(sparse_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(sparse_manifest["paper_id"], self.arxiv_id)
        self.assertEqual(sparse_manifest["build_id"], result["build_id"])
        self.assertEqual(sparse_manifest["index_version"], result["index_version"])
        self.assertEqual(sparse_manifest["source_type"], "chunk")
        self.assertEqual(sparse_manifest["source_file"], result["chunk_file"])
        self.assertEqual(sparse_manifest["source_hash"], result["sparse_index_source_hash"])
        self.assertEqual(sparse_manifest["schema_version"], result["sparse_index_schema_version"])
        self.assertEqual(sparse_manifest["token_count"], result["sparse_index_token_count"])
        documents_file = sparse_manifest_path.parent / sparse_manifest["files"]["documents"]
        self.assertEqual(len(documents_file.read_text(encoding="utf-8").splitlines()), result["sparse_index_document_count"])
        token_stats = json.loads((sparse_manifest_path.parent / sparse_manifest["files"]["token_stats"]).read_text(encoding="utf-8"))
        self.assertEqual(token_stats["document_count"], result["sparse_index_document_count"])
        self.assertEqual(token_stats["total_document_length"], result["sparse_index_token_count"])
        self.assertIn("field_stats", token_stats)
        postings = json.loads((sparse_manifest_path.parent / sparse_manifest["files"]["postings"]).read_text(encoding="utf-8"))
        document_frequency = json.loads((sparse_manifest_path.parent / sparse_manifest["files"]["document_frequency"]).read_text(encoding="utf-8"))
        self.assertTrue(postings)
        self.assertTrue(document_frequency)
        self.assertNotIn("", postings)
        self.assertNotIn("", document_frequency)
        self.assertEqual(loading_service.calls[0]["method"], "load_pdf")
        self.assertEqual(chunking_service.calls[0]["method"], "chunk_docling")
        self.assertEqual(embedding_service.calls[0]["method"], "create_embeddings")
        self.assertEqual(embedding_service.calls[0]["retrieval_index_count"], result["retrieval_index_count"])
        self.assertEqual(vector_store_service.calls[0]["method"], "index_embeddings")

    def test_build_qa_index_persists_structured_table_objects_in_chunk_file(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        document = {
            "docling_table_items": [
                {
                    "asset_kind": "table",
                    "asset_path": "03-docling-assets/paper/tables/table-1.csv",
                    "asset_caption": "Table 1: Main results",
                    "asset_preview": [
                        {"Method": "Baseline", "Accuracy": "82.5%", "F1": "0.71"},
                        {"Method": "Ours", "Accuracy": "91.0%", "F1": "0.83"},
                    ],
                    "page_start": 5,
                    "order_index": 1,
                }
            ]
        }
        table_chunk = {
            "content": "Table evidence\nMain results\nMethod: Baseline; Accuracy: 82.5% | Method: Ours; Accuracy: 91.0%",
            "metadata": {
                "chunk_type": "table",
                "chunk_id": 7,
                "page_start": 5,
                "section_path": "Experiments > Results",
                "asset_path": "03-docling-assets/paper/tables/table-1.csv",
                "asset_caption": "Table 1: Main results",
                "order_index": 1,
            },
        }
        chunking_service = _FakeChunkingService(chunks=[table_chunk])
        loading_service = _FakeLoadingService(document=document)
        builder, loading_service, *_ = self._make_builder(
            loading_service=loading_service,
            chunking_service=chunking_service,
        )

        result = builder.build_qa_index(self.arxiv_id, loading_method="docling")
        save_call = next(call for call in loading_service.calls if call["method"] == "save_document")
        saved_document = save_call["kwargs"]["document_data"]
        saved_chunks = save_call["kwargs"]["chunks"]

        self.assertEqual(result["status"], "success")
        self.assertEqual(saved_document["table_structure_debug"]["table_count"], 1)
        self.assertEqual(saved_document["table_structure_debug"]["structured_table_count"], 1)
        self.assertEqual(saved_document["table_structure_debug"]["failed_table_parse_count"], 0)
        self.assertEqual(saved_chunks[0]["metadata"]["table_id"], saved_document["structured_tables"][0]["table_id"])
        table_object = saved_document["structured_tables"][0]
        self.assertEqual(table_object["source_chunk_id"], 7)
        self.assertEqual(table_object["page_number"], 5)
        self.assertEqual(table_object["section_path"], "Experiments > Results")
        self.assertEqual(table_object["columns"], ["Method", "Accuracy", "F1"])
        ours_accuracy = next(
            cell
            for cell in table_object["cells"]
            if cell["row_label"] == "Ours" and cell["col_name"] == "Accuracy"
        )
        self.assertEqual(ours_accuracy["raw_value"], "91.0%")
        self.assertAlmostEqual(ours_accuracy["normalized_value"], 0.91)
        self.assertEqual(ours_accuracy["unit"], "percent")

    def test_rebuild_keeps_old_collection_until_new_index_activates(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        self.storage.paper_qa_index.insert_paper_qa_index(
            self.arxiv_id,
            collection_name="qa_old_collection",
            status="indexed",
            pdf_path="",
            chunk_file="",
            embedding_file="",
        )
        builder, *_services, vector_store_service = self._make_builder()

        builder.build_qa_index(self.arxiv_id, loading_method="docling")

        self.assertEqual(vector_store_service.calls[0]["method"], "index_embeddings")
        self.assertFalse(any(call["method"] == "delete_collection" for call in vector_store_service.calls))
        builds = self.storage.paper_qa_index.list_paper_qa_index_builds(self.arxiv_id, statuses=["cleanup_pending"], limit=5)
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0]["collection_name"], "qa_old_collection")

    def test_collection_name_generation_is_stable(self) -> None:
        vector_store_service = _FakeVectorStoreService()

        first = vector_store_service.index_embeddings(json.dumps({"filename": "2401.00001.pdf", "embeddings": []}), types.SimpleNamespace(provider="milvus", index_mode="default"))
        second = vector_store_service.index_embeddings(json.dumps({"filename": "2401.00001.pdf", "embeddings": []}), types.SimpleNamespace(provider="milvus", index_mode="default"))

        self.assertEqual(first["collection_name"], second["collection_name"])
        self.assertEqual(first["collection_name"], "qa_2401_00001_pdf")

    def test_build_qa_index_records_failure_when_loading_fails(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(loading_service=_FakeLoadingService(fail_stage="load_pdf_document"))

        with self.assertRaises(AppError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)
        failed_build = self.storage.paper_qa_index.get_latest_paper_qa_index_build(self.arxiv_id, statuses=["build_failed"])
        self.assertEqual(ctx.exception.code, ErrorCode.QA_INDEX_BUILD_FAILED)
        self.assertIn("loading failed", str(ctx.exception.__cause__))
        self.assertEqual(record["status"], "not_indexed")
        self.assertEqual(failed_build["status"], "build_failed")
        self.assertEqual(failed_build["failed_stage"], "load_pdf_document")
        self.assertEqual(failed_build["error_message"], "loading failed")
        self.assertEqual(failed_build["pdf_path"], "tmp/fake-paper.pdf")
        self.assertEqual(ctx.exception.context.get("stage"), "load_pdf_document")

    def test_build_qa_index_records_failure_when_chunking_fails(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(chunking_service=_FakeChunkingService(fail_stage="chunk_document"))

        with self.assertRaises(AppError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)
        failed_build = self.storage.paper_qa_index.get_latest_paper_qa_index_build(self.arxiv_id, statuses=["build_failed"])
        self.assertEqual(ctx.exception.code, ErrorCode.QA_INDEX_BUILD_FAILED)
        self.assertIn("chunking failed", str(ctx.exception.__cause__))
        self.assertEqual(record["status"], "not_indexed")
        self.assertEqual(failed_build["status"], "build_failed")
        self.assertEqual(failed_build["failed_stage"], "chunk_document")
        self.assertEqual(failed_build["error_message"], "chunking failed")
        self.assertEqual(failed_build["pdf_path"], "tmp/fake-paper.pdf")
        self.assertEqual(ctx.exception.context.get("stage"), "chunk_document")

    def test_build_qa_index_records_failure_when_embedding_fails(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(embedding_service=_FakeEmbeddingService(fail_stage="create_chunk_embeddings"))

        with self.assertRaises(AppError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)
        failed_build = self.storage.paper_qa_index.get_latest_paper_qa_index_build(self.arxiv_id, statuses=["build_failed"])
        self.assertEqual(ctx.exception.code, ErrorCode.QA_INDEX_BUILD_FAILED)
        self.assertIn("embedding failed", str(ctx.exception.__cause__))
        self.assertEqual(record["status"], "not_indexed")
        self.assertEqual(failed_build["status"], "build_failed")
        self.assertEqual(failed_build["failed_stage"], "create_chunk_embeddings")
        self.assertEqual(failed_build["error_message"], "embedding failed")
        self.assertEqual(failed_build["chunk_file"], "chunk-output.json")
        self.assertTrue(failed_build["retrieval_index_file"])
        self.assertGreater(failed_build["retrieval_index_count"], 0)
        self.assertEqual(failed_build["artifact_status"], "cleanup_pending")
        self.assertEqual(ctx.exception.context.get("stage"), "create_chunk_embeddings")

    def test_build_qa_index_records_failure_when_vector_write_fails(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        builder, *_ = self._make_builder(vector_store_service=_FakeVectorStoreService(fail_stage="index_embeddings_to_vector_store"))

        with self.assertRaises(AppError) as ctx:
            builder.build_qa_index(self.arxiv_id)

        record = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)
        failed_build = self.storage.paper_qa_index.get_latest_paper_qa_index_build(self.arxiv_id, statuses=["build_failed"])
        self.assertEqual(ctx.exception.code, ErrorCode.VECTOR_STORE_ERROR)
        self.assertIn("vector write failed", str(ctx.exception.__cause__))
        self.assertEqual(record["status"], "not_indexed")
        self.assertEqual(failed_build["status"], "build_failed")
        self.assertEqual(failed_build["failed_stage"], "index_embeddings_to_vector_store")
        self.assertEqual(failed_build["error_message"], "vector write failed")
        self.assertIn('"filename": "2401.00001_', failed_build["embedding_file"])
        self.assertEqual(ctx.exception.context.get("stage"), "index_embeddings_to_vector_store")

    def test_rebuild_failure_keeps_previous_active_index(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        self.storage.paper_qa_index.insert_paper_qa_index(
            self.arxiv_id,
            collection_name="qa_old_collection",
            status="indexed",
            chunk_count=2,
            embedding_model="old-model",
            pdf_path="old.pdf",
            chunk_file="old-chunks.json",
            embedding_file="old-embeddings.json",
        )
        builder, *_ = self._make_builder(embedding_service=_FakeEmbeddingService(fail_stage="create_chunk_embeddings"))

        with self.assertRaises(AppError):
            builder.build_qa_index(self.arxiv_id)

        active = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)
        failed_build = self.storage.paper_qa_index.get_latest_paper_qa_index_build(self.arxiv_id, statuses=["build_failed"])
        self.assertEqual(active["status"], "indexed")
        self.assertEqual(active["collection_name"], "qa_old_collection")
        self.assertEqual(failed_build["failed_stage"], "create_chunk_embeddings")

    def test_cleanup_pending_builds_does_not_delete_active_collection(self) -> None:
        self.storage.paper_catalog.add_paper(self._paper_payload())
        backend_root = Path(__file__).resolve().parents[2]
        old_retrieval_index_path = Path("02-retrieval-indexes") / "old_active_retrieval_indexes.json"
        old_retrieval_index_file = backend_root / old_retrieval_index_path
        old_retrieval_index_file.parent.mkdir(exist_ok=True)
        old_retrieval_index_file.write_text('{"retrieval_indexes":[]}', encoding="utf-8")
        old_sparse_path = Path("02-sparse-indexes") / "old_active_sparse"
        old_sparse_dir = backend_root / old_sparse_path
        old_sparse_dir.mkdir(parents=True, exist_ok=True)
        old_sparse_manifest = old_sparse_dir / "manifest.json"
        old_sparse_manifest.write_text("{}", encoding="utf-8")
        self.storage.paper_qa_index.insert_paper_qa_index(
            self.arxiv_id,
            collection_name="qa_old_collection",
            status="indexed",
            chunk_count=2,
            embedding_model="old-model",
            # 旧记录可能仍是相对路径；清理流程应以 backend 为基准兼容这些记录。
            retrieval_index_file=str(old_retrieval_index_path),
            retrieval_index_count=2,
            retrieval_index_types=json.dumps(["body"], ensure_ascii=False),
            retrieval_index_version="old-version",
            sparse_index_dir=str(old_sparse_path),
            sparse_index_manifest_file=str(old_sparse_path / "manifest.json"),
            sparse_index_document_count=2,
            sparse_index_token_count=12,
            sparse_index_backend="internal_bm25",
            sparse_index_schema_version="sparse_index_artifact_v1",
            sparse_index_source_file="old-chunks.json",
            sparse_index_source_hash="old-sparse-hash",
        )
        builder, *_services, vector_store_service = self._make_builder()
        builder.build_qa_index(self.arxiv_id, loading_method="docling")
        active = self.storage.paper_qa_index.get_paper_qa_index(self.arxiv_id)
        active_retrieval_index_file = Path(active["retrieval_index_file"])
        active_sparse_dir = Path(active["sparse_index_dir"])
        self.assertTrue(active_retrieval_index_file.is_file())
        self.assertTrue(active_sparse_dir.is_dir())

        cleanup_result = builder.cleanup_pending_index_builds(self.arxiv_id)

        deleted_collections = [
            call["collection_name"]
            for call in vector_store_service.calls
            if call["method"] == "delete_collection"
        ]
        self.assertIn("qa_old_collection", deleted_collections)
        self.assertNotIn(active["collection_name"], deleted_collections)
        self.assertFalse(old_retrieval_index_file.exists())
        self.assertFalse(old_sparse_dir.exists())
        self.assertTrue(active_retrieval_index_file.exists())
        self.assertTrue(active_sparse_dir.exists())
        active_cleanup = builder.cleanup_qa_index_artifacts(self.arxiv_id, active)
        self.assertTrue(active_cleanup["skipped_active_build"])
        self.assertTrue(active_sparse_dir.exists())
        self.assertEqual(len(cleanup_result["failed"]), 0)


class IndexJobManagerFlowTests(_BaseIndexTestCase):
    def test_submit_job_creates_pending_job_and_reuses_active_duplicate(self) -> None:
        _NoopThread.instances = []
        builder = _FakeJobBuilder()
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder)

        with mock.patch.object(manager_module.threading, "Thread", _NoopThread):
            first_job = manager.submit_job(self.arxiv_id, "docling")
            second_job = manager.submit_job(self.arxiv_id, "docling")

        latest = self.storage.paper_qa_index.get_latest_paper_index_job(self.arxiv_id)

        self.assertEqual(first_job["status"], "pending")
        self.assertEqual(second_job["job_id"], first_job["job_id"])
        self.assertEqual(latest["job_id"], first_job["job_id"])
        self.assertEqual(len(_NoopThread.instances), 1)
        self.assertTrue(_NoopThread.instances[0].started)

    def test_submit_job_marks_stale_running_job_and_creates_new_job(self) -> None:
        _NoopThread.instances = []
        builder = _FakeJobBuilder()
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder, timeout_seconds=10)
        old_job = self.storage.paper_qa_index.create_paper_index_job(self.arxiv_id, "docling")
        old_heartbeat = (datetime.now(timezone.utc) - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
        self.storage.paper_qa_index.update_paper_index_job(
            old_job["job_id"],
            status="running",
            current_stage="build",
            progress=40,
            heartbeat_at=old_heartbeat,
        )

        with mock.patch.object(manager_module.threading, "Thread", _NoopThread):
            new_job = manager.submit_job(self.arxiv_id, "docling")

        stored_old = self.storage.paper_qa_index.get_paper_index_job(old_job["job_id"])
        self.assertEqual(stored_old["status"], "stale")
        self.assertNotEqual(new_job["job_id"], old_job["job_id"])
        self.assertEqual(new_job["previous_job_id"], old_job["job_id"])
        self.assertEqual(new_job["recovery_action"], "marked_stale_and_created")
        self.assertEqual(len(_NoopThread.instances), 1)

    def test_submit_job_reuses_unexpired_running_job(self) -> None:
        _NoopThread.instances = []
        builder = _FakeJobBuilder()
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder, timeout_seconds=300)
        old_job = self.storage.paper_qa_index.create_paper_index_job(self.arxiv_id, "docling")
        self.storage.paper_qa_index.update_paper_index_job(
            old_job["job_id"],
            status="running",
            current_stage="build",
            progress=40,
        )

        with mock.patch.object(manager_module.threading, "Thread", _NoopThread):
            reused_job = manager.submit_job(self.arxiv_id, "docling")

        self.assertEqual(reused_job["job_id"], old_job["job_id"])
        self.assertEqual(reused_job["recovery_action"], "reused_active")
        self.assertEqual(len(_NoopThread.instances), 0)

    def test_submit_job_marks_stale_pending_job_and_creates_new_job(self) -> None:
        _NoopThread.instances = []
        builder = _FakeJobBuilder()
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder, timeout_seconds=5)
        old_job = self.storage.paper_qa_index.create_paper_index_job(self.arxiv_id, "docling")
        old_heartbeat = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        self.storage.paper_qa_index.update_paper_index_job(old_job["job_id"], heartbeat_at=old_heartbeat)

        with mock.patch.object(manager_module.threading, "Thread", _NoopThread):
            new_job = manager.submit_job(self.arxiv_id, "docling")

        stored_old = self.storage.paper_qa_index.get_paper_index_job(old_job["job_id"])
        self.assertEqual(stored_old["status"], "stale")
        self.assertNotEqual(new_job["job_id"], old_job["job_id"])

    def test_update_job_refreshes_heartbeat(self) -> None:
        job = self.storage.paper_qa_index.create_paper_index_job(self.arxiv_id, "docling")
        old_heartbeat = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        self.storage.paper_qa_index.update_paper_index_job(job["job_id"], heartbeat_at=old_heartbeat)

        self.assertTrue(self.storage.paper_qa_index.update_paper_index_job(job["job_id"], current_stage="build", progress=30))

        stored = self.storage.paper_qa_index.get_paper_index_job(job["job_id"])
        self.assertNotEqual(stored["heartbeat_at"], old_heartbeat)

    def test_polling_recovery_marks_stale_job_retryable(self) -> None:
        job = self.storage.paper_qa_index.create_paper_index_job(self.arxiv_id, "docling")
        old_heartbeat = (datetime.now(timezone.utc) - timedelta(seconds=60)).strftime("%Y-%m-%d %H:%M:%S")
        self.storage.paper_qa_index.update_paper_index_job(
            job["job_id"],
            status="running",
            current_stage="build",
            heartbeat_at=old_heartbeat,
        )

        marked_count = self.storage.paper_qa_index.mark_stale_paper_index_jobs(
            arxiv_id=self.arxiv_id,
            job_id=job["job_id"],
            timeout_seconds=10,
        )

        stored = self.storage.paper_qa_index.get_paper_index_job(job["job_id"])
        self.assertEqual(marked_count, 1)
        self.assertEqual(stored["status"], "stale")
        self.assertIn("heartbeat timed out", stored["error_message"])

    def test_run_job_success_updates_job_status_and_latest_job(self) -> None:
        builder = _FakeJobBuilder(result={"status": "success"})
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder)
        job = self.storage.paper_qa_index.create_paper_index_job(self.arxiv_id, "docling")

        manager.run_job(job["job_id"], self.arxiv_id, "docling")

        stored = self.storage.paper_qa_index.get_paper_index_job(job["job_id"])
        latest = self.storage.paper_qa_index.get_latest_paper_index_job(self.arxiv_id)
        self.assertEqual(stored["status"], "success")
        self.assertEqual(stored["current_stage"], "mark_index_success")
        self.assertEqual(stored["progress"], 100)
        self.assertEqual(latest["job_id"], job["job_id"])

    def test_run_job_failure_updates_failed_status_and_error_message(self) -> None:
        error = HTTPException(status_code=500, detail="builder failed")
        setattr(error, "error_stage", "chunk_document")
        builder = _FakeJobBuilder(error=error)
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder)
        job = self.storage.paper_qa_index.create_paper_index_job(self.arxiv_id, "docling")

        manager.run_job(job["job_id"], self.arxiv_id, "docling")

        stored = self.storage.paper_qa_index.get_paper_index_job(job["job_id"])
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["current_stage"], "chunk_document")
        self.assertEqual(stored["error_message"], "builder failed")

    def test_concurrent_submit_reuses_single_active_job(self) -> None:
        _NoopThread.instances = []
        builder = _FakeJobBuilder()
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder, timeout_seconds=300)
        results = []

        def _submit_once() -> None:
            results.append(manager.submit_job(self.arxiv_id, "docling"))

        with mock.patch.object(manager, "run_job", lambda *_args, **_kwargs: None):
            threads = [threading.Thread(target=_submit_once) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        job_ids = {item["job_id"] for item in results}
        active_jobs = [
            job
            for job in self.storage.paper_qa_index.list_paper_index_jobs(arxiv_id=self.arxiv_id, limit=10)
            if job["status"] in {"pending", "running", "retrying"}
        ]
        self.assertEqual(len(job_ids), 1)
        self.assertEqual(len(active_jobs), 1)

    def test_submit_job_with_inline_thread_reaches_terminal_state(self) -> None:
        builder = _FakeJobBuilder(result={"status": "success"})
        manager = IndexJobManager(paper_qa_index_store=self.storage.paper_qa_index, qa_index_builder=builder)

        with mock.patch.object(manager_module.threading, "Thread", _InlineThread):
            job = manager.submit_job(self.arxiv_id, "docling")

        latest = self.storage.paper_qa_index.get_latest_paper_index_job(self.arxiv_id)
        self.assertEqual(job["job_id"], latest["job_id"])
        self.assertEqual(latest["status"], "success")


if __name__ == "__main__":
    unittest.main()
