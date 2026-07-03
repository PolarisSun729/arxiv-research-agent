"""Fake arXiv service for backend unittests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional


class FakeArxivService:
    """Serve canned paper payloads without touching network or disk."""

    def __init__(self, *, papers: Optional[Iterable[Dict[str, Any]]] = None, pdf_path: str = "fake-paper.pdf") -> None:
        self.papers = [dict(paper) for paper in (papers or [])]
        self.pdf_path = pdf_path
        self.calls: list[dict[str, Any]] = []

    def _record(self, method: str, **payload: Any) -> None:
        self.calls.append({"method": method, **payload})

    def _select_papers(self, id_list: Optional[Iterable[str]] = None, max_results: int = 10) -> list[dict[str, Any]]:
        if id_list:
            wanted = {str(value) for value in id_list}
            filtered = [paper for paper in self.papers if str(paper.get("arxiv_id", "")) in wanted]
        else:
            filtered = list(self.papers)
        return [deepcopy(paper) for paper in filtered[:max_results]]

    def search(self, search_query: str = "", id_list: Optional[Iterable[str]] = None, max_results: int = 10, **kwargs: Any) -> Dict[str, Any]:
        self._record("search", search_query=search_query, id_list=list(id_list or []), max_results=max_results, kwargs=kwargs)
        papers = self._select_papers(id_list=id_list, max_results=max_results)
        return {"papers": papers, "total_results": len(papers), "query": search_query}

    def download_pdf(self, pdf_url: str, arxiv_id: str, *_args: Any, **_kwargs: Any) -> str:
        self._record("download_pdf", pdf_url=pdf_url, arxiv_id=arxiv_id)
        return self.pdf_path

    def get_available_fields(self) -> list[dict[str, str]]:
        self._record("get_available_fields")
        return [{"prefix": "all", "field": "All Fields", "description": "Search all fields"}]

    def get_subject_categories(self) -> list[dict[str, str]]:
        self._record("get_subject_categories")
        return [{"code": "cs.CL", "name": "Computation and Language"}]
