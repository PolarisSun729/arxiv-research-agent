from __future__ import annotations

import sys
import types

from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService


if "pymilvus" not in sys.modules:
    pymilvus_stub = types.ModuleType("pymilvus")

    class _DataType:
        VARCHAR = "VARCHAR"
        INT64 = "INT64"
        DOUBLE = "DOUBLE"
        BOOL = "BOOL"
        FLOAT_VECTOR = "FLOAT_VECTOR"

    class _MilvusClient:
        pass

    pymilvus_stub.DataType = _DataType
    pymilvus_stub.MilvusClient = _MilvusClient
    sys.modules["pymilvus"] = pymilvus_stub

if "pypinyin" not in sys.modules:
    pypinyin_stub = types.ModuleType("pypinyin")

    class _Style:
        NORMAL = "NORMAL"

    def _lazy_pinyin(value, style=None):
        return [str(value or "")]

    pypinyin_stub.Style = _Style
    pypinyin_stub.lazy_pinyin = _lazy_pinyin
    sys.modules["pypinyin"] = pypinyin_stub

from services.storage.vector_store_service import VectorStoreService


class _RecordingEmbeddingService(EmbeddingService):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_text_batches: list[list[str]] = []

    def _create_dashscope_embeddings(self, texts: list, config: EmbeddingConfig) -> list:
        self.recorded_text_batches.append([str(text) for text in texts])
        return [[float(index + 1), 0.0, 0.0] for index, _text in enumerate(texts)]


def test_create_embeddings_uses_retrieval_index_text_and_preserves_chunk_content() -> None:
    service = _RecordingEmbeddingService()
    chunk = {
        "content": "Original chunk body should remain evidence text.",
        "metadata": {"chunk_id": 7, "chunk_type": "text", "page_number": 3, "source": "paper.pdf"},
    }
    retrieval_indexes = [
        {
            "index_id": "7:summary:1",
            "chunk_id": "7",
            "index_type": "summary",
            "index_text": "Index-level summary text used for embedding.",
            "index_weight": 0.78,
            "enabled_routes": ["vector_original"],
        }
    ]

    embeddings, _usage = service.create_embeddings(
        {"chunks": [chunk], "retrieval_indexes": retrieval_indexes, "metadata": {"filename": "paper.pdf"}},
        EmbeddingConfig(provider="dashscope", model_name="fake", dimension=3, batch_size=8),
    )

    assert service.recorded_text_batches == [["Index-level summary text used for embedding."]]
    assert len(embeddings) == 1
    metadata = embeddings[0]["metadata"]
    assert metadata["content"] == "Original chunk body should remain evidence text."
    assert metadata["index_id"] == "7:summary:1"
    assert metadata["retrieval_index_id"] == "7:summary:1"
    assert metadata["index_text"] == "Index-level summary text used for embedding."


def test_create_embeddings_can_fallback_to_legacy_chunk_level_mode() -> None:
    service = _RecordingEmbeddingService()
    chunk = {
        "content": "Legacy chunk content used directly.",
        "metadata": {"chunk_id": 3, "chunk_type": "text", "page_number": 1, "source": "paper.pdf"},
    }

    embeddings, _usage = service.create_embeddings(
        {"chunks": [chunk], "embedding_mode": "chunk_level", "metadata": {"filename": "paper.pdf"}},
        EmbeddingConfig(provider="dashscope", model_name="fake", dimension=3, batch_size=8),
    )

    assert service.recorded_text_batches == [["Legacy chunk content used directly."]]
    metadata = embeddings[0]["metadata"]
    assert metadata["embedding_source_level"] == "chunk_level_fallback"
    assert metadata["index_id"] == "3:body:legacy"
    assert metadata["retrieval_index_type"] == "body"


def test_create_embeddings_can_disable_multi_index_embedding_by_config_metadata() -> None:
    service = _RecordingEmbeddingService()
    chunk = {
        "content": "Config disabled multi-index embedding uses chunk evidence.",
        "metadata": {"chunk_id": 4, "chunk_type": "text", "page_number": 1, "source": "paper.pdf"},
    }
    retrieval_indexes = [
        {
            "index_id": "4:question:1",
            "chunk_id": "4",
            "index_type": "question",
            "index_text": "Index question text should not be embedded when disabled.",
            "index_weight": 0.82,
            "enabled_routes": ["vector_original"],
        }
    ]

    embeddings, _usage = service.create_embeddings(
        {
            "chunks": [chunk],
            "retrieval_indexes": retrieval_indexes,
            "metadata": {
                "filename": "paper.pdf",
                "enable_multi_index_embedding": False,
                "enable_chunk_level_retrieval_fallback": True,
            },
        },
        EmbeddingConfig(provider="dashscope", model_name="fake", dimension=3, batch_size=8),
    )

    assert service.recorded_text_batches == [["Config disabled multi-index embedding uses chunk evidence."]]
    metadata = embeddings[0]["metadata"]
    assert metadata["embedding_source_level"] == "chunk_level_fallback"
    assert metadata["matched_index_id"] == "4:body:legacy"


def test_vector_store_payload_returns_matched_index_aliases_and_legacy_fallback() -> None:
    service = VectorStoreService()

    payload = service._build_chunk_payload(
        entity={
            "content": "Chunk evidence",
            "index_id": "5:question:1",
            "index_type": "question",
            "index_text": "What does the method do?",
            "index_weight": 0.82,
            "chunk_id": 5,
            "parent_chunk_id": 5,
            "original_chunk_id": 5,
            "chunk_index": 5,
            "page_start": 2,
            "page_end": 2,
        },
        score=0.91,
        distance=None,
    )

    assert payload["index_id"] == "5:question:1"
    assert payload["matched_index_id"] == "5:question:1"
    assert payload["matched_index_type"] == "question"
    assert payload["matched_index_score"] == 0.91
    assert payload["chunk_id"] == 5

    legacy_payload = service._build_chunk_payload(
        entity={
            "content": "Old chunk-level evidence",
            "chunk_id": 9,
            "parent_chunk_id": 9,
            "original_chunk_id": 9,
            "chunk_index": 9,
        },
        score=0.7,
        distance=None,
    )

    assert legacy_payload["matched_index_id"] == "9:body:legacy"
    assert legacy_payload["matched_index_type"] == "body"
    assert legacy_payload["matched_index_text"] == "Old chunk-level evidence"


def test_vector_store_truncates_index_preview_fields_before_validation() -> None:
    service = VectorStoreService()
    fields = [{"name": "index_text", "dtype": "VARCHAR", "max_length": 5}]
    entities = [{"index_text": "abcdef"}]

    truncated = service._truncate_entities_to_varchar_limits(entities, fields)

    assert truncated[0]["index_text"] == "abcde"
    service._validate_varchar_lengths(truncated, fields)
