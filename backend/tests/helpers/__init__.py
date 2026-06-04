"""Reusable unittest helpers for backend tests."""

from .fake_arxiv_service import FakeArxivService
from .fake_embedding_service import FakeEmbeddingService, FakeEmbeddingConfig
from .fake_generation_service import FakeGenerationService
from .fake_paper_qa_service import FakePaperQAService
from .fake_vector_store_service import FakeVectorStoreService
from .retrieval import (
    ControlledFakeEmbeddingService,
    ControlledFakeVectorStoreService,
    build_retrieval_service,
    build_sample_chunks,
    load_retrieval_modules,
    seed_collection,
)
from .sqlite import TemporarySqliteDatabase, build_database_service

__all__ = [
    "FakeArxivService",
    "FakeEmbeddingConfig",
    "FakeEmbeddingService",
    "FakeGenerationService",
    "FakePaperQAService",
    "FakeVectorStoreService",
    "ControlledFakeEmbeddingService",
    "ControlledFakeVectorStoreService",
    "build_retrieval_service",
    "build_sample_chunks",
    "load_retrieval_modules",
    "seed_collection",
    "TemporarySqliteDatabase",
    "build_database_service",
]
