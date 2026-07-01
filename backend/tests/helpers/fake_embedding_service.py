"""Deterministic fake embedding service for backend unittests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple


@dataclass
class FakeEmbeddingConfig:
    provider: str = "fake"
    model_name: str = "fake-embedding-model"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    dimension: int = 8
    batch_size: int = 16
    enable_fusion: bool = False


class FakeEmbeddingService:
    """Small deterministic embedding helper with no external dependencies."""

    def __init__(self, *, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, Any]] = []

    def get_default_embedding_config(self) -> FakeEmbeddingConfig:
        return FakeEmbeddingConfig(dimension=self.dimension)

    @staticmethod
    def build_paper_embedding_text(title: str, abstract: str) -> str:
        title_value = str(title or "").strip()
        abstract_value = str(abstract or "").strip()
        return f"{title_value}\n\nAbstract: {abstract_value}".strip()

    def _record(self, method: str, **payload: Any) -> None:
        self.calls.append({"method": method, **payload})

    def _vector_for_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(str(text or "").encode("utf-8")).digest()
        values: list[float] = []
        for index in range(self.dimension):
            byte_value = digest[index % len(digest)]
            values.append(round(byte_value / 255.0, 6))
        return values

    def create_single_embedding(self, text: str, *_args: Any, **_kwargs: Any) -> list[float]:
        self._record("create_single_embedding", text=text)
        return self._vector_for_text(text)

    def create_single_embedding_with_usage(
        self,
        text: str,
        *_args: Any,
        **_kwargs: Any,
    ) -> Tuple[list[float], dict[str, Any]]:
        embedding = self.create_single_embedding(text)
        usage = {"input_count": 1, "token_count": len(str(text or ""))}
        self._record("create_single_embedding_with_usage", text=text, usage=usage)
        return embedding, usage

    def create_text_embeddings_with_usage(
        self,
        texts: Sequence[str],
        *_args: Any,
        **_kwargs: Any,
    ) -> Tuple[list[list[float]], dict[str, Any]]:
        embeddings = [self._vector_for_text(text) for text in texts]
        usage = {"input_count": len(texts), "token_count": sum(len(str(text or "")) for text in texts)}
        self._record("create_text_embeddings_with_usage", texts=list(texts), usage=usage)
        return embeddings, usage

    def create_embeddings(
        self,
        input_data: Iterable[Any],
        _config: Optional[Any] = None,
    ) -> Tuple[list[dict[str, Any]], dict[str, Any]]:
        retrieval_indexes = []
        if isinstance(input_data, dict):
            chunks = list(input_data.get("chunks") or [])
            retrieval_indexes = list(input_data.get("retrieval_indexes") or [])
            chunk_lookup = {
                str((chunk.get("metadata", {}) or {}).get("chunk_id") or chunk.get("chunk_id") or index): chunk
                for index, chunk in enumerate(chunks, start=1)
            }
            # QA 索引构建现在按 RetrievalIndex 生成向量；fake 同步该输入形态，避免测试仍假设一 chunk 一向量。
            if retrieval_indexes:
                normalized = [
                    {
                        "content": str(index.get("index_text") or ""),
                        "metadata": {
                            **dict((chunk_lookup.get(str(index.get("chunk_id"))) or {}).get("metadata", {}) or {}),
                            "content": (chunk_lookup.get(str(index.get("chunk_id"))) or {}).get("content", ""),
                            "retrieval_index_id": index.get("index_id", ""),
                            "retrieval_index_type": index.get("index_type", ""),
                            "retrieval_index_text": index.get("index_text", ""),
                            "retrieval_index_weight": index.get("index_weight", 1.0),
                            "retrieval_index_enabled_routes": index.get("enabled_routes", []),
                        },
                    }
                    for index in retrieval_indexes
                    if str(index.get("index_text") or "").strip()
                ]
            else:
                normalized = chunks
        else:
            normalized = list(input_data)
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(normalized):
            if isinstance(item, dict):
                text = str(item.get("content", item.get("text", "")) or "")
                metadata = dict(item.get("metadata", {}) or {})
            else:
                text = str(item or "")
                metadata = {}
            rows.append(
                {
                    "id": metadata.get("chunk_id", f"chunk-{index + 1}"),
                    "content": text,
                    "embedding": self._vector_for_text(text),
                    "metadata": metadata,
                }
            )
        usage = {"input_count": len(rows)}
        self._record("create_embeddings", row_count=len(rows))
        return rows, usage

    def save_embeddings(self, filename: str, embeddings: list[dict[str, Any]]) -> str:
        self._record("save_embeddings", filename=filename, row_count=len(embeddings))
        safe_name = str(filename or "embeddings.json").replace("/", "_").replace("\\", "_")
        payload = {"filename": safe_name, "embeddings": embeddings}
        return json.dumps(payload, ensure_ascii=True)

    def embed_query(self, text: str) -> list[float]:
        self._record("embed_query", text=text)
        return self._vector_for_text(text)

    def embed_documents(self, texts: Sequence[str]) -> List[list[float]]:
        self._record("embed_documents", texts=list(texts))
        return [self._vector_for_text(text) for text in texts]
