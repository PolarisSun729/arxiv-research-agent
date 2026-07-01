from __future__ import annotations

import json
import tempfile
from pathlib import Path

from services.retrieval.retrieval_index import (
    CollectionRetrievalIndexProvider,
    build_retrieval_index_payload,
    build_retrieval_indexes,
    iter_retrieval_indexes_for_embedding,
    save_retrieval_index_artifact,
    summarize_retrieval_indexes,
)
import services.retrieval.retrieval_index as retrieval_index_module


def _tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in str(text or "").replace("\n", " ").split() if token.strip()]


class _JsonGenerationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def complete_with_qwen(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if "\"questions\"" in prompt:
            return '{"questions":["How is the retriever trained?","What loss is used?"]}'
        return '{"summary":"This chunk describes contrastive retrieval training."}'


class _FailingGenerationService:
    def complete_with_qwen(self, *_args, **_kwargs):
        raise RuntimeError("llm unavailable")


def test_build_retrieval_indexes_keeps_body_index_and_stable_ids() -> None:
    chunk = {
        "content": "The method uses a two-stage retrieval pipeline.",
        "metadata": {
            "chunk_id": "chunk-method",
            "chunk_type": "text",
            "section_title": "Method",
            "section_path": "2 Method",
            "page_number": 2,
            "source": "paper.pdf",
        },
        "rerank_text": "Summary: two-stage retrieval method.",
        "retrieval_questions": ["How does the retrieval method work?"],
    }

    first = build_retrieval_indexes([chunk])
    second = build_retrieval_indexes([chunk])

    assert [item["index_id"] for item in first] == [item["index_id"] for item in second]
    assert {item["index_type"] for item in first} >= {"body", "section_anchor", "summary", "question"}
    body = next(item for item in first if item["index_type"] == "body")
    assert body["chunk_id"] == "chunk-method"
    assert body["index_text"] == chunk["content"]
    assert all(item["chunk_id"] for item in first)


def test_empty_index_text_is_kept_for_debug_but_skipped_for_embedding() -> None:
    indexes = build_retrieval_indexes(
        [
            {
                "content": "",
                "metadata": {"chunk_id": "empty-chunk", "chunk_type": "text"},
            }
        ]
    )

    assert {item["index_type"] for item in indexes} == {"body", "section_anchor"}
    body = next(item for item in indexes if item["index_type"] == "body")
    assert body["index_text"] == ""
    assert summarize_retrieval_indexes(indexes)["empty_index_text_count"] == 1
    assert iter_retrieval_indexes_for_embedding(indexes) == []


def test_generative_indexes_can_add_summary_and_questions() -> None:
    indexes, debug = build_retrieval_index_payload(
        [
            {
                "content": "The retriever is trained with contrastive learning over positive and negative passages.",
                "metadata": {
                    "chunk_id": "chunk-training",
                    "chunk_type": "text",
                    "section_title": "Retriever Training",
                    "page_number": 4,
                },
            }
        ],
        generation_service=_JsonGenerationService(),
        enable_generative_indexes=True,
    )

    index_types = [item["index_type"] for item in indexes]
    assert index_types.count("body") == 1
    assert index_types.count("section_anchor") == 1
    assert index_types.count("summary") == 1
    assert index_types.count("question") == 2
    assert len(indexes) <= 6
    assert debug["generated_summary_count"] == 1
    assert debug["generated_question_count"] == 2
    assert debug["generation_error_count"] == 0
    assert next(item for item in indexes if item["index_type"] == "summary")["index_text"].startswith("This chunk")


def test_generated_question_indexes_can_be_disabled() -> None:
    indexes, debug = build_retrieval_index_payload(
        [
            {
                "content": "The retriever is trained with contrastive learning over positive and negative passages.",
                "metadata": {
                    "chunk_id": "chunk-training",
                    "chunk_type": "text",
                    "section_title": "Retriever Training",
                },
            }
        ],
        generation_service=_JsonGenerationService(),
        enable_generative_indexes=True,
        max_questions_per_chunk=0,
    )

    assert "summary" in {item["index_type"] for item in indexes}
    assert "question" not in {item["index_type"] for item in indexes}
    assert debug["generated_question_count"] == 0


def test_generative_failure_keeps_rule_indexes_and_records_debug() -> None:
    indexes, debug = build_retrieval_index_payload(
        [
            {
                "content": "The paper introduces a dual encoder retrieval baseline.",
                "metadata": {
                    "chunk_id": "chunk-fallback",
                    "chunk_type": "text",
                    "section_title": "Method",
                },
            }
        ],
        generation_service=_FailingGenerationService(),
        enable_generative_indexes=True,
    )

    index_types = {item["index_type"] for item in indexes}
    assert {"body", "section_anchor"} <= index_types
    assert debug["generation_error_count"] == 2
    assert all(item["chunk_id"] == "chunk-fallback" for item in indexes)
    assert any(item["index_type"] == "question" for item in indexes)


def test_asset_chunk_prioritizes_caption_summary_preview_for_indexes() -> None:
    indexes = build_retrieval_indexes(
        [
            {
                "content": "OCR-only table body",
                "metadata": {
                    "chunk_id": "table-1",
                    "chunk_type": "table",
                    "section_title": "Results",
                    "page_number": 6,
                    "asset_caption": "Table 1: Main benchmark results.",
                    "asset_summary": "The proposed method outperforms the baseline.",
                    "asset_preview_text": "Baseline 82.5%; Ours 91.0%",
                    "table_structured_text": "Method | Accuracy\nBaseline | 82.5%\nOurs | 91.0%",
                },
            }
        ]
    )

    index_types = {item["index_type"] for item in indexes}
    assert {"body", "section_anchor", "table_or_figure"} <= index_types
    body = next(item for item in indexes if item["index_type"] == "body")
    table_index = next(item for item in indexes if item["index_type"] == "table_or_figure")
    assert "Table 1: Main benchmark results." in body["index_text"]
    assert "Method | Accuracy" in table_index["index_text"]
    assert body["index_text"].find("Table 1") < body["index_text"].find("OCR-only")


def test_collection_keyword_index_uses_index_text_and_returns_source_chunk() -> None:
    source_chunk = {
        "content": "Original chunk body does not mention the anchor term.",
        "chunk_id": "chunk-1",
        "chunk_type": "text",
        "source": "paper.pdf",
    }
    rows = [
        {
            **source_chunk,
            "retrieval_index_id": "chunk-1:section_anchor:1",
            "retrieval_index_type": "section_anchor",
            "retrieval_index_text": "Method Anchor",
            "retrieval_index_weight": 0.45,
            "retrieval_index_enabled_routes": ["keyword"],
        },
        {
            **source_chunk,
            "retrieval_index_id": "chunk-1:summary:1",
            "retrieval_index_type": "summary",
            "retrieval_index_text": "",
            "retrieval_index_weight": 0.78,
            "retrieval_index_enabled_routes": ["keyword"],
        },
    ]

    class Store:
        def get_all_chunks(self, _collection_name: str):
            return list(rows)

    provider = CollectionRetrievalIndexProvider(
        vector_store_service=Store(),
        chunk_normalizer=lambda chunk: dict(chunk),
        tokenizer=_tokenize,
    )

    index = provider.get_index("paper_collection")

    # 空 index_text 只保留在上游模型里，不进入 BM25 postings；命中的 doc 仍回填原 chunk 内容。
    assert len(index.documents) == 1
    document = index.documents[0]
    assert document.retrieval_index_id == "chunk-1:section_anchor:1"
    assert document.retrieval_index_type == "section_anchor"
    assert document.chunk["content"] == source_chunk["content"]
    assert index.postings["method"] == {0}


def test_collection_keyword_index_rebuilds_from_retrieval_index_artifact() -> None:
    source_chunk = {
        "content": "Original chunk content is still used as final evidence.",
        "metadata": {"chunk_id": "chunk-1", "chunk_type": "text", "section_title": "Method"},
    }
    retrieval_indexes = [
        {
            "index_id": "chunk-1:question:1",
            "chunk_id": "chunk-1",
            "index_type": "question",
            "index_text": "How does the artifact backed retriever work?",
            "index_weight": 0.82,
            "enabled_routes": ["keyword"],
            "metadata": {"source": "artifact-test.pdf"},
        }
    ]
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        artifact_dir = Path(temp_dir)
        chunk_file = artifact_dir / "chunks.json"
        chunk_file.write_text(json.dumps({"chunks": [source_chunk]}, ensure_ascii=False), encoding="utf-8")
        retrieval_index_file = save_retrieval_index_artifact(
            paper_id="paper-artifact",
            retrieval_indexes=retrieval_indexes,
            retrieval_index_debug=summarize_retrieval_indexes(retrieval_indexes),
            chunks=[source_chunk],
            index_version="v1",
            output_dir=str(artifact_dir),
        )

        class Store:
            def get_all_chunks(self, _collection_name: str):
                raise AssertionError("artifact path should not fall back to Milvus chunk scan")

        provider = CollectionRetrievalIndexProvider(
            vector_store_service=Store(),
            chunk_normalizer=lambda chunk: dict(chunk),
            tokenizer=_tokenize,
        )

        index = provider.get_index(
            "paper_collection",
            index_record={
                "arxiv_id": "paper-artifact",
                "chunk_file": str(chunk_file),
                "retrieval_index_file": retrieval_index_file,
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
        )

        assert index.build_source == "retrieval_index_artifact"
        assert len(index.documents) == 1
        assert index.documents[0].retrieval_index_id == "chunk-1:question:1"


def test_collection_keyword_index_defaults_to_index_level_artifact_documents() -> None:
    source_chunk = {
        "content": "Original chunk body is returned after sparse matching.",
        "metadata": {"chunk_id": "chunk-multi", "chunk_type": "text", "section_title": "Method"},
    }
    retrieval_indexes = [
        {
            "index_id": "chunk-multi:question:1",
            "chunk_id": "chunk-multi",
            "index_type": "question",
            "index_text": "Which contrastive objective is used?",
            "index_weight": 0.82,
            "enabled_routes": ["keyword"],
            "metadata": {},
        },
        {
            "index_id": "chunk-multi:summary:1",
            "chunk_id": "chunk-multi",
            "index_type": "summary",
            "index_text": "The chunk summarizes contrastive objective training.",
            "index_weight": 0.78,
            "enabled_routes": ["keyword"],
            "metadata": {},
        },
    ]
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        artifact_dir = Path(temp_dir)
        chunk_file = artifact_dir / "chunks.json"
        chunk_file.write_text(json.dumps({"chunks": [source_chunk]}, ensure_ascii=False), encoding="utf-8")
        retrieval_index_file = save_retrieval_index_artifact(
            paper_id="paper-index-level",
            retrieval_indexes=retrieval_indexes,
            retrieval_index_debug=summarize_retrieval_indexes(retrieval_indexes),
            chunks=[source_chunk],
            index_version="v1",
            output_dir=str(artifact_dir),
        )

        class Store:
            def get_all_chunks(self, _collection_name: str):
                raise AssertionError("index-level artifact should avoid Milvus chunk scan")

        provider = CollectionRetrievalIndexProvider(
            vector_store_service=Store(),
            chunk_normalizer=lambda chunk: dict(chunk),
            tokenizer=_tokenize,
        )

        index = provider.get_index(
            "paper_collection",
            index_record={
                "arxiv_id": "paper-index-level",
                "chunk_file": str(chunk_file),
                "retrieval_index_file": retrieval_index_file,
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
        )

        assert index.build_source == "retrieval_index_artifact"
        assert len(index.documents) == 2
        assert {document.retrieval_index_type for document in index.documents} == {"question", "summary"}
        assert {document.chunk["content"] for document in index.documents} == {source_chunk["content"]}
        assert index.retrieval_index_artifact_debug["index_level_bm25_enabled"] is True


def test_collection_keyword_index_can_disable_index_level_bm25_for_legacy_fallback(monkeypatch) -> None:
    monkeypatch.setitem(retrieval_index_module.ENHANCED_RETRIEVAL_CONFIG, "enable_index_level_bm25", False)
    source_chunk = {
        "content": "Legacy body sparse retrieval keeps the original chunk text.",
        "metadata": {"chunk_id": "chunk-legacy", "chunk_type": "text", "section_title": "Legacy"},
    }
    retrieval_indexes = [
        {
            "index_id": "chunk-legacy:question:1",
            "chunk_id": "chunk-legacy",
            "index_type": "question",
            "index_text": "Index-only question text should be ignored in legacy mode.",
            "index_weight": 0.82,
            "enabled_routes": ["keyword"],
            "metadata": {},
        },
        {
            "index_id": "chunk-legacy:summary:1",
            "chunk_id": "chunk-legacy",
            "index_type": "summary",
            "index_text": "Index-only summary text should also be ignored.",
            "index_weight": 0.78,
            "enabled_routes": ["keyword"],
            "metadata": {},
        },
    ]
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        artifact_dir = Path(temp_dir)
        chunk_file = artifact_dir / "chunks.json"
        chunk_file.write_text(json.dumps({"chunks": [source_chunk]}, ensure_ascii=False), encoding="utf-8")
        retrieval_index_file = save_retrieval_index_artifact(
            paper_id="paper-legacy",
            retrieval_indexes=retrieval_indexes,
            retrieval_index_debug=summarize_retrieval_indexes(retrieval_indexes),
            chunks=[source_chunk],
            index_version="v1",
            output_dir=str(artifact_dir),
        )

        class Store:
            def get_all_chunks(self, _collection_name: str):
                raise AssertionError("legacy chunk-file path should not scan Milvus")

        provider = CollectionRetrievalIndexProvider(
            vector_store_service=Store(),
            chunk_normalizer=lambda chunk: dict(chunk),
            tokenizer=_tokenize,
        )

        index = provider.get_index(
            "paper_collection",
            index_record={
                "arxiv_id": "paper-legacy",
                "chunk_file": str(chunk_file),
                "retrieval_index_file": retrieval_index_file,
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
        )

        assert index.build_source == "legacy_chunk_file"
        assert len(index.documents) == 1
        assert index.documents[0].retrieval_index_type == "body"
        assert index.documents[0].retrieval_index_text == source_chunk["content"]
        assert "legacy" in index.postings
        assert "index-only" not in index.postings
        assert index.retrieval_index_artifact_debug["index_level_bm25_enabled"] is False
        assert "index_level_bm25_disabled" in index.fallback_reason
        assert index.documents[0].chunk["content"] == source_chunk["content"]
        assert index.retrieval_index_artifact_debug["row_count"] == 1
