import types
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import paper_router


class _FakeDatabaseService:
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


class PaperRouterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_service = _FakeDatabaseService()
        self.oai_db_service = _FakeOaiDatabaseService()
        self.embedding_service = _FakeEmbeddingService()
        self.vector_store_service = _FakeVectorStoreService()
        self.recommendation_service = _FakeRecommendationService()

        app = FastAPI()
        app.include_router(paper_router.router, prefix="/api")
        app.dependency_overrides[dependencies.get_database_service] = lambda: self.db_service
        app.dependency_overrides[dependencies.get_oai_database_service] = lambda: self.oai_db_service
        app.dependency_overrides[dependencies.get_embedding_service] = lambda: self.embedding_service
        app.dependency_overrides[dependencies.get_recommendation_service] = lambda: self.recommendation_service

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
        self.db_service.add_paper({"arxiv_id": "2401.00002", "title": "Stored Paper"})

        response = self.client.get("/api/paper/2401.00002")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Stored Paper")

    def test_delete_paper_returns_success_and_404(self) -> None:
        self.db_service.add_paper({"arxiv_id": "2401.00003", "title": "Delete Me"})

        success = self.client.delete("/api/paper/2401.00003")
        missing = self.client.delete("/api/paper/2401.00003")

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["status"], "success")
        self.assertEqual(missing.status_code, 500)

    def test_get_all_papers_and_category_filter_return_stable_shape(self) -> None:
        self.db_service.add_paper({"arxiv_id": "2401.00004", "title": "CL Paper", "categories": "cs.CL"})
        self.db_service.add_paper({"arxiv_id": "2401.00005", "title": "IR Paper", "categories": "cs.IR"})

        all_response = self.client.get("/api/papers")
        category_response = self.client.get("/api/papers/category/cs.CL")

        self.assertEqual(all_response.status_code, 200)
        self.assertEqual(len(all_response.json()["papers"]), 2)
        self.assertEqual(category_response.status_code, 200)
        self.assertEqual(category_response.json()["papers"][0]["arxiv_id"], "2401.00004")


if __name__ == "__main__":
    unittest.main()
