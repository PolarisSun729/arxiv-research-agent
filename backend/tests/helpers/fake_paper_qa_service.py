"""Fake paper QA service for backend unittests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional


class FakePaperQAService:
    """Return canned QA payloads without invoking retrieval or models."""

    def __init__(
        self,
        *,
        answer_text: str = "fake paper answer",
        qa_status: Optional[Dict[str, Any]] = None,
        search_results: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        self.answer_text = answer_text
        self.qa_status = dict(qa_status or {"status": "ready", "chunk_count": 0})
        self.search_results = [dict(item) for item in (search_results or [])]
        self.calls: list[dict[str, Any]] = []

    def _record(self, method: str, **payload: Any) -> None:
        self.calls.append({"method": method, **payload})

    def get_qa_status(self, arxiv_id: str) -> Dict[str, Any]:
        self._record("get_qa_status", arxiv_id=arxiv_id)
        payload = deepcopy(self.qa_status)
        payload.setdefault("arxiv_id", arxiv_id)
        return payload

    def build_qa_index(self, arxiv_id: str, **kwargs: Any) -> Dict[str, Any]:
        self._record("build_qa_index", arxiv_id=arxiv_id, kwargs=kwargs)
        return {"job_id": f"job-{arxiv_id}", "status": "queued", "arxiv_id": arxiv_id}

    def answer_question(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        self._record("answer_question", arxiv_id=arxiv_id, payload=payload)
        return {
            "arxiv_id": arxiv_id,
            "answer": self.answer_text,
            "result": self.answer_text,
            "sources": self.build_source_payload(self.search_results),
            "retrieval_debug": {"provider": "fake"},
        }

    def build_qa_context(self, arxiv_id: str, payload: Any):
        self._record("build_qa_context", arxiv_id=arxiv_id, payload=payload)
        source_payload = self.build_source_payload(self.search_results)
        retrieval_debug = {"provider": "fake", "result_count": len(source_payload)}
        return payload, deepcopy(self.search_results), "fake qa context", retrieval_debug

    def build_source_payload(self, search_results: Iterable[Dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for index, item in enumerate(search_results, start=1):
            normalized.append(
                {
                    "source_id": item.get("source_id", f"source-{index}"),
                    "title": item.get("title", f"Paper {index}"),
                    "content": item.get("content", item.get("snippet", "")),
                }
            )
        self._record("build_source_payload", count=len(normalized))
        return normalized
