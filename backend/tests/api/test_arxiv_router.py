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


if __name__ == "__main__":
    unittest.main()
