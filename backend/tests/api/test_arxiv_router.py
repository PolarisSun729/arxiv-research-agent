import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import arxiv_router
from services.arxiv.local_oai_search_contract import LocalArxivSearchIndexUnavailable


class _FailingArxivBackend:
    def search(self, *args, **kwargs):
        # 路由层只负责透传稳定错误契约，这里直接模拟底层索引未就绪场景。
        raise LocalArxivSearchIndexUnavailable(
            "本地 OAI 镜像分类索引未就绪，当前无法执行 cat 分类检索。",
            query="cat:cs.CL",
            reason="category_index_missing",
            search_index_status="category_index_missing",
            suggested_action="请先运行 07-arxiv-tools\\rebuild_arxiv_oai_search_index.cmd 后重试。",
        )

    def get_available_fields(self):
        return []

    def get_subject_categories(self):
        return []


class ArxivRouterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(arxiv_router.router, prefix="/api")
        app.dependency_overrides[dependencies.get_arxiv_search_backend] = lambda: _FailingArxivBackend()
        self.app = app
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    def test_search_returns_503_when_local_search_index_is_unavailable(self) -> None:
        response = self.client.post("/api/arxiv/search", json={"search_query": "cat:cs.CL"})

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["detail"]["code"], "local_search_index_unavailable")
        self.assertEqual(payload["detail"]["details"]["reason"], "category_index_missing")
        self.assertEqual(
            payload["detail"]["details"]["query_capability"]["search_index_status"],
            "category_index_missing",
        )



class _CountingArxivBackend:
    """模拟本地 OAI 镜像：按最终查询串返回命中数，并记录被计数的查询。"""

    def __init__(self, totals):
        self.totals = totals
        self.counted_queries = []

    def search(self, *args, **kwargs):
        raise AssertionError("放宽建议只计数，不应执行完整搜索")

    def count_results(self, search_query="", id_list=None):
        self.counted_queries.append(search_query)
        for fragment, total in self.totals.items():
            if search_query.startswith(fragment):
                return total
        return 0

    def get_available_fields(self):
        return []

    def get_subject_categories(self):
        return []


class _RemoteLikeArxivBackend:
    """模拟远程 arXiv API：没有 count_results，放宽建议不逐个计数。"""

    def search(self, *args, **kwargs):
        raise AssertionError("远程后端不应为放宽建议发起搜索")

    def get_available_fields(self):
        return []

    def get_subject_categories(self):
        return []


class ArxivSearchSuggestionsApiTests(unittest.TestCase):
    def _client(self, backend) -> TestClient:
        app = FastAPI()
        app.include_router(arxiv_router.router, prefix="/api")
        app.dependency_overrides[dependencies.get_arxiv_search_backend] = lambda: backend
        return TestClient(app)

    def test_suggestions_include_totals_and_only_widen_results_when_backend_can_count(self) -> None:
        backend = _CountingArxivBackend({"(knowledge graph rag)": 25, "(knowledge graph)": 293})

        response = self._client(backend).post(
            "/api/arxiv/search/suggestions",
            json={"search_query": "knowledge graph rag", "submitted_days_ago": 30},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["baseline_total"], 25)
        self.assertEqual(
            [(item["kind"], item["search_query"], item["total"]) for item in payload["suggestions"]],
            [("drop_term", "knowledge graph", 293)],
        )
        # 计数用的是与 /search 相同的最终查询串，时间窗口按候选各自的天数拼接。
        self.assertTrue(all("submittedDate:[" in query for query in backend.counted_queries))

    def test_structured_fields_only_get_time_suggestions(self) -> None:
        backend = _CountingArxivBackend({})

        response = self._client(backend).post(
            "/api/arxiv/search/suggestions",
            json={"search_query": "knowledge graph", "author": "Alice", "submitted_days_ago": 30},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["suggestions"], [])
        self.assertEqual(len(backend.counted_queries), 3)

    def test_backend_without_counter_returns_candidates_without_totals(self) -> None:
        response = self._client(_RemoteLikeArxivBackend()).post(
            "/api/arxiv/search/suggestions",
            json={"search_query": "knowledge graph rag", "submitted_days_ago": 30},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIsNone(payload["baseline_total"])
        self.assertEqual(len(payload["suggestions"]), 7)
        self.assertTrue(all(item["total"] is None for item in payload["suggestions"]))

    def test_invalid_request_is_rejected_instead_of_returning_empty_suggestions(self) -> None:
        response = self._client(_RemoteLikeArxivBackend()).post(
            "/api/arxiv/search/suggestions",
            json={"search_query": "rag", "submitted_days_ago": -1},
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
