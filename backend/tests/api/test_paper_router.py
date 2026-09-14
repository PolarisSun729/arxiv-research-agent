import types
import unittest
from unittest import mock
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import paper_router


class _FakePaperStorage:
    def __init__(self) -> None:
        self.papers = {}

    def get_user_labeled_paper_count(self, user_id: str = "") -> int:
        return 2

    def add_paper(self, paper):
        self.papers[paper["arxiv_id"]] = dict(paper)
        return True

    def get_paper(self, arxiv_id: str):
        return self.papers.get(arxiv_id)

    def delete_paper(self, arxiv_id: str) -> bool:
        return self.papers.pop(arxiv_id, None) is not None

    def get_all_papers(self):
        return list(self.papers.values())

    def search_papers_by_category(self, category: str):
        return [paper for paper in self.papers.values() if category in str(paper.get("categories", ""))]


class _FakeOaiDatabaseService:
    def get_total_paper_count(self) -> int:
        return 123


class _FakeEmbeddingService:
    def build_paper_embedding_text(self, title: str, abstract: str) -> str:
        return f"{title}\n{abstract}"

    def create_single_embedding(self, *_args, **_kwargs):
        return [0.1, 0.2, 0.3]


class _FakeVectorStoreService:
    def insert_single_embedding(self, collection_name: str, embedding, metadata):
        return 99


class _FakeRecommendationService:
    def __init__(self) -> None:
        self.source_paper = None

    def _fetch_paper_from_arxiv_with_rate_limit(self, arxiv_id: str):
        return self.source_paper

    def _materialize_paper_from_source(self, source_paper, arxiv_id: str):
        payload = dict(source_paper)
        payload.setdefault("arxiv_id", arxiv_id)
        return payload

    def _ensure_paper_materialized(self, arxiv_id: str, paper_payload=None):
        payload = dict(paper_payload or {})
        payload.setdefault("arxiv_id", arxiv_id)
        return payload


class _FakePaperQAService:
    def delete_qa_index(self, arxiv_id: str):
        # 删除论文时路由会先清理 QA 索引；测试只关心该依赖存在且不触达真实向量库。
        return {"status": "success", "arxiv_id": arxiv_id}


class PaperRouterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paper_storage = _FakePaperStorage()
        self.oai_db_service = _FakeOaiDatabaseService()
        self.embedding_service = _FakeEmbeddingService()
        self.vector_store_service = _FakeVectorStoreService()
        self.recommendation_service = _FakeRecommendationService()
        self.paper_qa_service = _FakePaperQAService()

        app = FastAPI()
        app.include_router(paper_router.router, prefix="/api")
        for dependency in (dependencies.get_paper_catalog_store, paper_router.get_paper_catalog_store):
            app.dependency_overrides[dependency] = lambda: self.paper_storage
        for dependency in (dependencies.get_user_preference_store, paper_router.get_user_preference_store):
            app.dependency_overrides[dependency] = lambda: self.paper_storage
        for dependency in (dependencies.get_oai_database_service, paper_router.get_oai_database_service):
            app.dependency_overrides[dependency] = lambda: self.oai_db_service
        app.dependency_overrides[dependencies.get_embedding_service] = lambda: self.embedding_service
        for dependency in (dependencies.get_recommendation_service, paper_router.get_recommendation_service):
            app.dependency_overrides[dependency] = lambda: self.recommendation_service
        for dependency in (dependencies.get_paper_qa_service, paper_router.get_paper_qa_service):
            app.dependency_overrides[dependency] = lambda: self.paper_qa_service

        self.sync_patch = mock.patch.object(
            paper_router,
            "_get_sync_status_payload",
            return_value={
                "status": "success",
                "mode": "sync",
                "lastSyncRunAt": "2026-06-04T10:00:00",
                "lastSyncedDate": "2026-06-03",
                "latestSyncNewPapers": 8,
                "latestSyncMatchedPapers": 13,
                "syncErrors": 0,
                "syncErrorMessage": None,
            },
        )
        self.config_patch = mock.patch.object(
            paper_router,
            "get_current_embedding_config",
            return_value=types.SimpleNamespace(
                provider="fake",
                model_name="fake-embedding-model",
                api_key=None,
                base_url=None,
                dimension=3,
            ),
        )
        self.vector_patch = mock.patch.object(
            paper_router,
            "get_vector_store_service",
            return_value=self.vector_store_service,
        )

        self.sync_patch.start()
        self.config_patch.start()
        self.vector_patch.start()
        self.addCleanup(self.sync_patch.stop)
        self.addCleanup(self.config_patch.stop)
        self.addCleanup(self.vector_patch.stop)
        self.client = TestClient(app)

    def test_get_stats_returns_dashboard_fields(self) -> None:
        response = self.client.get("/api/stats", params={"user_id": "u1"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["totalPapers"], 123)
        self.assertEqual(payload["labeledPapers"], 2)
        self.assertEqual(payload["latestSyncNewPapers"], 8)
        self.assertEqual(payload["lastSyncStatus"], "success")

    def test_get_sync_status_returns_latest_sync_payload(self) -> None:
        response = self.client.get("/api/sync-status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lastSyncedDate"], "2026-06-03")

    def test_sync_status_reads_persistent_files_and_degrades_on_corrupt_values(self) -> None:
        # 本类的 API 测试默认替换聚合函数；这里恢复真实实现以覆盖文件读取边界。
        self.sync_patch.stop()
        self.addCleanup(self.sync_patch.start)
        with mock.patch.object(paper_router, "SYNC_META_FILE") as meta, mock.patch.object(
            paper_router, "SYNC_STATE_FILE"
        ) as state:
            meta.exists.return_value = True
            meta.is_file.return_value = True
            meta.read_text.return_value = json.dumps(
                {
                    "status": "success",
                    "last_successful_until": "2026-09-12",
                    "records_written": "not-a-number",
                }
            )
            state.exists.return_value = False
            state.is_file.return_value = False
            payload = paper_router._get_sync_status_payload()

        self.assertEqual(payload["lastSyncedDate"], "2026-09-12")
        self.assertEqual(payload["latestSyncNewPapers"], 0)

    def test_sync_status_defaults_live_under_backend_data(self) -> None:
        self.assertEqual(paper_router.SYNC_STATE_FILE.parent.name, "arxiv-oai-sync")
        self.assertEqual(paper_router.SYNC_STATE_FILE.parent.parent.name, "data")

    def test_add_paper_returns_embedding_metadata(self) -> None:
        response = self.client.post(
            "/api/paper",
            json={
                "arxiv_id": "2401.00001",
                "title": "RAG Paper",
                "authors": "Alice, Bob",
                "abstract": "Abstract",
                "categories": "cs.CL",
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.00001",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["embedding_id"], 99)
        self.assertEqual(payload["vector_dimension"], 3)

    def test_add_paper_returns_422_when_required_field_missing(self) -> None:
        response = self.client.post(
            "/api/paper",
            json={
                "arxiv_id": "2401.00001",
                "title": "RAG Paper",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_get_paper_returns_404_when_not_found_anywhere(self) -> None:
        response = self.client.get("/api/paper/9999.00001")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Paper not found")

    def test_get_paper_returns_local_paper_when_present(self) -> None:
        self.paper_storage.add_paper({"arxiv_id": "2401.00002", "title": "Stored Paper"})

        response = self.client.get("/api/paper/2401.00002")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Stored Paper")

    def test_get_paper_repairs_incomplete_local_metadata_from_source(self) -> None:
        self.paper_storage.add_paper(
            {
                "arxiv_id": "2401.00004",
                "title": "",
                "abstract": "",
                "embedding_id": "7",
            }
        )
        self.recommendation_service.source_paper = {
            "arxiv_id": "2401.00004",
            "title": "Recovered Paper",
            "abstract": "Recovered abstract",
        }

        response = self.client.get("/api/paper/2401.00004")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Recovered Paper")
        self.assertEqual(response.json()["abstract"], "Recovered abstract")

    def test_delete_paper_returns_success_and_404(self) -> None:
        self.paper_storage.add_paper({"arxiv_id": "2401.00003", "title": "Delete Me"})

        success = self.client.delete("/api/paper/2401.00003")
        missing = self.client.delete("/api/paper/2401.00003")

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["status"], "success")
        self.assertEqual(missing.status_code, 404)

    def test_get_all_papers_and_category_filter_return_stable_shape(self) -> None:
        self.paper_storage.add_paper({"arxiv_id": "2401.00004", "title": "CL Paper", "categories": "cs.CL"})
        self.paper_storage.add_paper({"arxiv_id": "2401.00005", "title": "IR Paper", "categories": "cs.IR"})

        all_response = self.client.get("/api/papers")
        category_response = self.client.get("/api/papers/category/cs.CL")

        self.assertEqual(all_response.status_code, 200)
        self.assertEqual(len(all_response.json()["papers"]), 2)
        self.assertEqual(category_response.status_code, 200)
        self.assertEqual(category_response.json()["papers"][0]["arxiv_id"], "2401.00004")


if __name__ == "__main__":
    unittest.main()
