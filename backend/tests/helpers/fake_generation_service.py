"""Deterministic fake generation service for backend unittests."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional


class FakeGenerationService:
    """Return canned text while recording calls for assertions."""

    def __init__(self, *, response_text: str = "fake generation response") -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def _record(self, method: str, **payload: Any) -> None:
        self.calls.append({"method": method, **payload})

    def complete_with_qwen(self, prompt: str, **kwargs: Any) -> str:
        self._record("complete_with_qwen", prompt=prompt, kwargs=kwargs)
        return self.response_text

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self._record("generate", prompt=prompt, kwargs=kwargs)
        return {
            "result": self.response_text,
            "text": self.response_text,
            "answer": self.response_text,
            "model": kwargs.get("model_name", "fake-generation-model"),
        }

    def stream_qwen_responses(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        self._record("stream_qwen_responses", prompt=prompt, kwargs=kwargs)
        for chunk in self.response_text.split():
            yield chunk

    def rewrite_query_for_retrieval(self, query: str, **kwargs: Any) -> list[str]:
        self._record("rewrite_query_for_retrieval", query=query, kwargs=kwargs)
        return [query]

    def plan_queries_for_retrieval(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self._record("plan_queries_for_retrieval", query=query, kwargs=kwargs)
        return {
            "query": query,
            "rewrites": [query],
            "strategy": "fake",
        }

    def build_rerank_query(self, query: str, **kwargs: Any) -> str:
        self._record("build_rerank_query", query=query, kwargs=kwargs)
        return query

    def generate_hyde_document(self, query: str, **kwargs: Any) -> str:
        self._record("generate_hyde_document", query=query, kwargs=kwargs)
        return f"HyDE: {query}"

    def compress_chunks_for_rerank(self, chunks: Iterable[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        normalized_chunks = [dict(chunk) for chunk in chunks]
        self._record("compress_chunks_for_rerank", chunk_count=len(normalized_chunks), kwargs=kwargs)
        return normalized_chunks
