from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from services.retrieval.keyword_backend import InternalBM25Backend
from services.paper_qa.build_cache import PaperQABuildCache
from services.retrieval.retrieval_index import (
    CollectionRetrievalIndexProvider,
    build_retrieval_index_payload,
    build_retrieval_indexes,
    iter_retrieval_indexes_for_embedding,
    save_retrieval_index_artifact,
    save_sparse_index_artifact,
    summarize_retrieval_indexes,
)
import services.retrieval.retrieval_index as retrieval_index_module


def _tokenize(text: str) -> list[str]:
    return [token.strip().lower() for token in str(text or "").replace("\n", " ").split() if token.strip()]


class _KeywordQueryTools:
    @staticmethod
    def tokenize_for_keyword_search(text: str) -> list[str]:
        return _tokenize(text)

    @staticmethod
    def extract_query_keywords(tokens: list[str], limit: int) -> list[str]:
        return list(tokens)[:limit]

    @staticmethod
    def expand_keyword_query_tokens(tokens: list[str]) -> list[str]:
        return list(dict.fromkeys(tokens))

    @staticmethod
    def is_informative_keyword_token(token: str) -> bool:
        return len(str(token or "").strip()) > 1

def _keyword_query_profile(query: str) -> SimpleNamespace:
    return SimpleNamespace(
        original_query=query,
        keyword_query=query,
        intent_tags=[],
        intent_profile=SimpleNamespace(main_intent="other"),
    )


def _make_internal_bm25_backend() -> InternalBM25Backend:
    return InternalBM25Backend(
        query_tools=_KeywordQueryTools(),
        route_confidence_builder=lambda *_args, **_kwargs: 0.7,
        structural_bonus_builder=lambda *_args, **_kwargs: 0.0,
        fusion_service=None,
    )


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


def test_generative_indexes_reuse_cached_summary_and_questions(tmp_path: Path) -> None:
    cache = PaperQABuildCache(
        {
            "enabled": True,
            "root_dir": str(tmp_path),
            "llm_cache_name": "llm",
            "embedding_cache_name": "embedding",
            "llm_size_limit": 1024 * 1024,
            "embedding_size_limit": 1024 * 1024,
        }
    )
    chunk = {
        "content": "The retriever is trained with contrastive learning over positive and negative passages.",
        "metadata": {
            "chunk_id": "chunk-cache",
            "chunk_type": "text",
            "section_title": "Retriever Training",
            "page_number": 4,
        },
    }
    first_service = _JsonGenerationService()

    first_indexes, first_debug = build_retrieval_index_payload(
        [chunk],
        generation_service=first_service,
        enable_generative_indexes=True,
        generation_cache=cache,
        generation_model_name="fake-qwen",
    )
    second_service = _JsonGenerationService()
    second_indexes, second_debug = build_retrieval_index_payload(
        [chunk],
        generation_service=second_service,
        enable_generative_indexes=True,
        generation_cache=cache,
        generation_model_name="fake-qwen",
    )

    assert len(first_service.calls) == 2
    assert second_service.calls == []
    assert [item["index_text"] for item in second_indexes] == [item["index_text"] for item in first_indexes]
    assert first_debug["generated_summary_count"] == 1
    assert first_debug["generated_question_count"] == 2
    assert second_debug["cached_summary_count"] == 1
    assert second_debug["cached_question_count"] == 2


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


def test_collection_keyword_index_loads_persisted_sparse_artifact() -> None:
    source_chunk = {
        "chunk_id": "chunk-sparse",
        "chunk_type": "figure",
        "content": "Persisted sparse retriever body evidence remains available.",
        "section_title": "Method",
        "section_path": "2 Method",
        "asset_caption": "Figure 1: Sparse artifact diagram.",
        "metadata": {"chunk_id": "chunk-sparse", "chunk_type": "figure", "section_title": "Method"},
    }
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        artifact_dir = Path(temp_dir)
        chunk_file = artifact_dir / "chunks.json"
        chunk_file.write_text(json.dumps({"chunks": [source_chunk]}, ensure_ascii=False), encoding="utf-8")

        class Store:
            def __init__(self) -> None:
                self.calls = 0

            def get_all_chunks(self, _collection_name: str):
                self.calls += 1
                return [dict(source_chunk)]

        store = Store()
        provider = CollectionRetrievalIndexProvider(
            vector_store_service=store,
            chunk_normalizer=lambda chunk: dict(chunk),
            tokenizer=_tokenize,
        )
        built_index = provider.get_index(
            "paper_collection",
            index_record={
                "arxiv_id": "paper-sparse",
                "chunk_file": str(chunk_file),
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
            force_refresh=True,
        )
        sparse_artifact = save_sparse_index_artifact(
            index=built_index,
            paper_id="paper-sparse",
            build_id="build-v1",
            index_version="v1",
            source_type="chunk_file",
            source_file=str(chunk_file),
            backend="internal_bm25",
            output_dir=str(artifact_dir / "sparse"),
        )
        manifest = json.loads(Path(sparse_artifact["manifest_file"]).read_text(encoding="utf-8"))

        loaded_index = provider.get_index(
            "paper_collection_sparse",
            index_record={
                "arxiv_id": "paper-sparse",
                "chunk_file": str(chunk_file),
                "sparse_index_dir": sparse_artifact["artifact_dir"],
                "sparse_index_manifest_file": sparse_artifact["manifest_file"],
                "sparse_index_source_hash": sparse_artifact["source_hash"],
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
            force_refresh=True,
        )

        assert loaded_index.build_source == "sparse_index_artifact"
        assert len(loaded_index.documents) == 1
        assert loaded_index.documents[0].retrieval_index_id == "chunk-sparse:body:1"
        assert loaded_index.postings["persisted"] == {0}
        assert loaded_index.postings["method"] == {0}
        assert loaded_index.document_frequency == built_index.document_frequency
        assert loaded_index.avgdl == built_index.avgdl
        assert loaded_index.to_keyword_debug(1)["keyword_route_index_source"] == "persistent_sparse_artifact"
        assert not loaded_index.to_keyword_debug(1)["keyword_index_fallback_used"]
        assert manifest["source_type"] == "chunk"
        assert manifest["source_file"] == str(chunk_file)
        assert manifest["token_count"] == sparse_artifact["token_count"]
        backend = _make_internal_bm25_backend()
        query_views = [
            {"view_id": "original:0", "source": "original", "query": "persisted sparse method", "weight": 1.0}
        ]
        query_profile = _keyword_query_profile("persisted sparse method")
        runtime_result = backend.retrieve(
            query_views=query_views,
            top_k=5,
            query_profile=query_profile,
            retrieval_index=built_index,
        )
        artifact_result = backend.retrieve(
            query_views=query_views,
            top_k=5,
            query_profile=query_profile,
            retrieval_index=loaded_index,
        )
        assert [item["chunk_id"] for item in artifact_result.results] == [item["chunk_id"] for item in runtime_result.results]
        assert artifact_result.debug["sparse_index"]["load_source"] == "persistent_sparse_artifact"
        assert artifact_result.debug["sparse_index"]["backend"] == "internal_bm25"
        assert store.calls == 1
        assert loaded_index.retrieval_index_artifact_debug["source_hash"] == sparse_artifact["source_hash"]
        Path(sparse_artifact["manifest_file"]).write_text("not-json", encoding="utf-8")
        cached_index = provider.get_index(
            "paper_collection_sparse",
            index_record={
                "arxiv_id": "paper-sparse",
                "chunk_file": str(chunk_file),
                "sparse_index_dir": sparse_artifact["artifact_dir"],
                "sparse_index_manifest_file": sparse_artifact["manifest_file"],
                "sparse_index_source_hash": sparse_artifact["source_hash"],
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
        )
        assert cached_index.cache_hit
        assert cached_index.to_keyword_debug(1)["keyword_route_index_source"] == "persistent_sparse_artifact"


def test_sparse_artifact_manifest_version_mismatch_falls_back_to_runtime_build() -> None:
    source_chunk = {
        "content": "Runtime fallback chunk content remains searchable.",
        "metadata": {"chunk_id": "chunk-runtime-fallback", "chunk_type": "text", "section_title": "Fallback"},
    }
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        artifact_dir = Path(temp_dir)
        chunk_file = artifact_dir / "chunks.json"
        chunk_file.write_text(json.dumps({"chunks": [source_chunk]}, ensure_ascii=False), encoding="utf-8")

        class Store:
            def get_all_chunks(self, _collection_name: str):
                return [dict(source_chunk)]

        provider = CollectionRetrievalIndexProvider(
            vector_store_service=Store(),
            chunk_normalizer=lambda chunk: dict(chunk),
            tokenizer=_tokenize,
        )
        built_index = provider.get_index(
            "paper_runtime_fallback_seed",
            index_record={
                "arxiv_id": "paper-runtime-fallback",
                "chunk_file": str(chunk_file),
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
            force_refresh=True,
        )
        sparse_artifact = save_sparse_index_artifact(
            index=built_index,
            paper_id="paper-runtime-fallback",
            build_id="build-v1",
            index_version="v1",
            source_type="chunk_file",
            source_file=str(chunk_file),
            backend="internal_bm25",
            output_dir=str(artifact_dir / "sparse"),
        )
        manifest_path = Path(sparse_artifact["manifest_file"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["tokenizer_version"] = "stale-tokenizer"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        index = provider.get_index(
            "paper_runtime_fallback",
            index_record={
                "arxiv_id": "paper-runtime-fallback",
                "chunk_file": str(chunk_file),
                "sparse_index_manifest_file": sparse_artifact["manifest_file"],
                "sparse_index_source_hash": sparse_artifact["source_hash"],
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
            force_refresh=True,
        )

        debug = index.to_keyword_debug(1)
        assert index.build_source == "milvus_chunk_scan"
        assert debug["keyword_route_index_source"] == "runtime_build_fallback"
        assert debug["keyword_index_fallback_used"]
        assert "sparse_index_artifact_unavailable:tokenizer_version_mismatch" in index.fallback_reason
        assert debug["retrieval_index_artifact"]["sparse_index_artifact"]["reason"].startswith("tokenizer_version_mismatch")


def test_sparse_artifact_source_hash_mismatch_falls_back_to_retrieval_index_artifact() -> None:
    source_chunk = {
        "content": "Fallback chunk content remains available.",
        "metadata": {"chunk_id": "chunk-fallback-sparse", "chunk_type": "text", "section_title": "Fallback"},
    }
    retrieval_indexes = [
        {
            "index_id": "chunk-fallback-sparse:question:1",
            "chunk_id": "chunk-fallback-sparse",
            "index_type": "question",
            "index_text": "Which fallback path handles stale sparse artifacts?",
            "index_weight": 0.82,
            "enabled_routes": ["keyword"],
            "metadata": {},
        }
    ]
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        artifact_dir = Path(temp_dir)
        chunk_file = artifact_dir / "chunks.json"
        chunk_file.write_text(json.dumps({"chunks": [source_chunk]}, ensure_ascii=False), encoding="utf-8")
        retrieval_index_file = save_retrieval_index_artifact(
            paper_id="paper-sparse-fallback",
            retrieval_indexes=retrieval_indexes,
            retrieval_index_debug=summarize_retrieval_indexes(retrieval_indexes),
            chunks=[source_chunk],
            index_version="v1",
            output_dir=str(artifact_dir),
        )

        class Store:
            def get_all_chunks(self, _collection_name: str):
                raise AssertionError("hash mismatch should fall back to retrieval index artifact before vector scan")

        provider = CollectionRetrievalIndexProvider(
            vector_store_service=Store(),
            chunk_normalizer=lambda chunk: dict(chunk),
            tokenizer=_tokenize,
        )
        built_index = provider.get_index(
            "paper_collection",
            index_record={
                "arxiv_id": "paper-sparse-fallback",
                "chunk_file": str(chunk_file),
                "retrieval_index_file": retrieval_index_file,
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
            force_refresh=True,
        )
        sparse_artifact = save_sparse_index_artifact(
            index=built_index,
            paper_id="paper-sparse-fallback",
            build_id="build-v1",
            index_version="v1",
            source_type="retrieval_index_artifact",
            source_file=retrieval_index_file,
            backend="internal_bm25",
            output_dir=str(artifact_dir / "sparse"),
        )

        index = provider.get_index(
            "paper_collection_fallback",
            index_record={
                "arxiv_id": "paper-sparse-fallback",
                "chunk_file": str(chunk_file),
                "retrieval_index_file": retrieval_index_file,
                "sparse_index_manifest_file": sparse_artifact["manifest_file"],
                "sparse_index_source_hash": "stale-hash",
                "chunk_count": 1,
                "active_index_version": "v1",
                "active_build_id": "build-v1",
            },
            force_refresh=True,
        )

        assert index.build_source == "retrieval_index_artifact"
        assert len(index.documents) == 1
        assert "sparse_index_artifact_unavailable:source_hash_mismatch" in index.fallback_reason


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
