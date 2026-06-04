"""In-memory fake vector store used by backend unittests."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List


class FakeVectorStoreService:
    """Minimal in-memory vector store with deterministic behavior."""

    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []

    def _record(self, method: str, **payload: Any) -> None:
        self.calls.append({"method": method, **payload})

    @staticmethod
    def _normalize_collection_name(collection_name: str) -> str:
        return str(collection_name or "default").strip() or "default"

    @staticmethod
    def _distance(left: List[float], right: List[float]) -> float:
        size = min(len(left), len(right))
        if size == 0:
            return 0.0
        return math.sqrt(sum((float(left[index]) - float(right[index])) ** 2 for index in range(size)))

    def collection_exists(self, _provider: str, collection_name: str) -> bool:
        normalized = self._normalize_collection_name(collection_name)
        self._record("collection_exists", collection_name=normalized)
        return normalized in self.collections

    def list_collections(self, _provider: str) -> list[str]:
        self._record("list_collections")
        return sorted(self.collections)

    def delete_collection(self, _provider: str, collection_name: str) -> bool:
        normalized = self._normalize_collection_name(collection_name)
        self._record("delete_collection", collection_name=normalized)
        return self.collections.pop(normalized, None) is not None

    def insert_single_embedding(self, collection_name: str, embedding: List[float], metadata: Dict[str, Any]) -> int:
        normalized = self._normalize_collection_name(collection_name)
        rows = self.collections.setdefault(normalized, [])
        row_id = len(rows) + 1
        rows.append({"id": row_id, "embedding": list(embedding), "metadata": dict(metadata or {})})
        self._record("insert_single_embedding", collection_name=normalized, row_id=row_id)
        return row_id

    def insert_embeddings(self, collection_name: str, items: Iterable[Dict[str, Any]]) -> int:
        normalized = self._normalize_collection_name(collection_name)
        inserted = 0
        for item in items:
            metadata = dict(item.get("metadata", {}) or {})
            embedding = list(item.get("embedding", []) or [])
            self.insert_single_embedding(normalized, embedding, metadata)
            inserted += 1
        self._record("insert_embeddings", collection_name=normalized, count=inserted)
        return inserted

    def index_embeddings(self, embedding_payload: Any, _config: Any) -> Dict[str, Any]:
        self._record("index_embeddings", payload_type=type(embedding_payload).__name__)
        return {
            "database": "fake",
            "index_mode": "fake",
            "total_vectors": 0,
            "index_size": 0,
            "processing_time": 0.0,
            "collection_name": "fake_collection",
        }

    def get_collection_info(self, _provider: str, collection_name: str) -> Dict[str, Any]:
        normalized = self._normalize_collection_name(collection_name)
        rows = self.collections.get(normalized, [])
        self._record("get_collection_info", collection_name=normalized)
        return {"collection_name": normalized, "row_count": len(rows)}

    def get_all_chunks(self, collection_name: str, limit: int | None = None) -> list[dict[str, Any]]:
        normalized = self._normalize_collection_name(collection_name)
        rows = list(self.collections.get(normalized, []))
        self._record("get_all_chunks", collection_name=normalized, limit=limit)
        if limit is None:
            return rows
        return rows[:limit]

    def search_similar_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        normalized = self._normalize_collection_name(collection_name)
        rows = self.collections.get(normalized, [])
        ranked = sorted(
            rows,
            key=lambda row: self._distance(list(query_vector or []), list(row.get("embedding", []) or [])),
        )
        self._record("search_similar_vectors", collection_name=normalized, top_k=top_k)
        results: list[dict[str, Any]] = []
        for row in ranked[:top_k]:
            result = dict(row)
            result["score"] = self._distance(list(query_vector or []), list(row.get("embedding", []) or []))
            results.append(result)
        return results

    def search_similar_papers(self, collection_name: str, query_vector: List[float], top_k: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
        self._record("search_similar_papers", collection_name=collection_name, top_k=top_k, kwargs=kwargs)
        return self.search_similar_vectors(collection_name, query_vector, top_k=top_k)

    def get_paper_embeddings_by_arxiv_ids(self, collection_name: str, arxiv_ids: Iterable[str]) -> list[dict[str, Any]]:
        normalized = self._normalize_collection_name(collection_name)
        wanted = {str(value) for value in arxiv_ids}
        rows = []
        for row in self.collections.get(normalized, []):
            metadata = dict(row.get("metadata", {}) or {})
            if str(metadata.get("arxiv_id", "")) in wanted:
                rows.append(dict(row))
        self._record("get_paper_embeddings_by_arxiv_ids", collection_name=normalized, count=len(rows))
        return rows
