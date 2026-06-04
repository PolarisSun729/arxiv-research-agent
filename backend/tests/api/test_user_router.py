import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import user_router


class _FakeDatabaseService:
    def __init__(self) -> None:
        self.preferences = {
            "user_id": "u1",
            "liked_papers": ["2401.00001"],
            "disliked_papers": [],
            "paper_actions": {"favorite": ["2401.00001"]},
            "research_profile": {"preferred_answer_style": "concise"},
        }
        self.actions = [
            {
                "user_id": "u1",
                "arxiv_id": "2401.00001",
                "action_type": "favorite",
                "metadata": {"source": "test"},
            }
        ]
        self.remove_action_result = True

    def get_user_preferences(self, user_id: str):
        payload = dict(self.preferences)
        payload["user_id"] = user_id
        return payload

    def remove_user_paper_action(self, user_id: str, arxiv_id: str, action_type: str) -> bool:
        return self.remove_action_result

    def get_user_paper_actions(self, user_id: str, action_type=None):
        if action_type:
            return [item for item in self.actions if item["action_type"] == action_type]
        return list(self.actions)

    def get_user_paper_action_map(self, user_id: str):
        return {"favorite": ["2401.00001"]}


class _FakeMemoryService:
    def load_user_profile(self, user_id: str):
        return {"user_id": user_id, "preferred_answer_style": "concise"}

    def patch_user_profile(self, user_id: str, patch, source: str):
        payload = dict(patch)
        payload["user_id"] = user_id
        payload["source"] = source
        return payload


class _FakeRecommendationService:
    def __init__(self) -> None:
        self.raise_error = False

    def record_user_paper_preference(self, user_id: str, arxiv_id: str, liked: bool, paper_payload=None):
        if self.raise_error:
            raise RuntimeError("preference failed")
        return {"status": "success", "user_id": user_id, "arxiv_id": arxiv_id, "liked": liked}

    def record_user_paper_action(self, user_id: str, arxiv_id: str, action_type: str, paper_payload=None, metadata=None):
        return {
            "status": "success",
            "user_id": user_id,
            "arxiv_id": arxiv_id,
            "action_type": action_type,
            "metadata": metadata or {},
        }

    def recommend_papers(self, user_id: str, top_n: int, max_age_months: int):
        if self.raise_error:
            raise RuntimeError("recommend failed")
        return {
            "user_id": user_id,
            "top_n": top_n,
            "max_age_months": max_age_months,
            "papers": [{"arxiv_id": "2401.00001"}],
        }


class UserRouterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_service = _FakeDatabaseService()
        self.memory_service = _FakeMemoryService()
        self.recommendation_service = _FakeRecommendationService()

        app = FastAPI()
        app.include_router(user_router.router, prefix="/api")
        app.dependency_overrides[dependencies.get_database_service] = lambda: self.db_service
        app.dependency_overrides[dependencies.get_memory_service] = lambda: self.memory_service
        app.dependency_overrides[dependencies.get_recommendation_service] = lambda: self.recommendation_service
        self.client = TestClient(app)

    def test_get_user_preferences_returns_stable_payload(self) -> None:
        response = self.client.get("/api/user/preferences/u1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["user_id"], "u1")
        self.assertIn("liked_papers", payload)
        self.assertIn("research_profile", payload)

    def test_like_and_dislike_routes_forward_to_recommendation_service(self) -> None:
        like_response = self.client.post("/api/user/like-paper", json={"user_id": "u1", "arxiv_id": "2401.00001"})
        dislike_response = self.client.post("/api/user/dislike-paper", json={"user_id": "u1", "arxiv_id": "2401.00001"})

        self.assertEqual(like_response.status_code, 200)
        self.assertTrue(like_response.json()["liked"])
        self.assertEqual(dislike_response.status_code, 200)
        self.assertFalse(dislike_response.json()["liked"])

    def test_like_paper_maps_runtime_error_to_500(self) -> None:
        self.recommendation_service.raise_error = True

        response = self.client.post("/api/user/like-paper", json={"user_id": "u1", "arxiv_id": "2401.00001"})

        self.assertEqual(response.status_code, 500)
        self.assertIn("preference failed", response.json()["detail"])

    def test_record_paper_action_and_validation_error(self) -> None:
        success = self.client.post(
            "/api/user/paper-action",
            json={"user_id": "u1", "arxiv_id": "2401.00001", "action_type": "favorite", "metadata": {"source": "ui"}},
        )
        invalid = self.client.post("/api/user/paper-action", json={"user_id": "u1", "arxiv_id": "2401.00001"})

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["action_type"], "favorite")
        self.assertEqual(invalid.status_code, 422)

    def test_delete_paper_action_returns_success_and_404(self) -> None:
        success = self.client.request(
            "DELETE",
            "/api/user/paper-action",
            json={"user_id": "u1", "arxiv_id": "2401.00001", "action_type": "favorite"},
        )
        self.db_service.remove_action_result = False
        missing = self.client.request(
            "DELETE",
            "/api/user/paper-action",
            json={"user_id": "u1", "arxiv_id": "2401.00001", "action_type": "favorite"},
        )

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["status"], "success")
        self.assertEqual(missing.status_code, 404)

    def test_get_user_paper_actions_returns_items_and_map(self) -> None:
        response = self.client.get("/api/user/paper-actions/u1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(len(payload["actions"]), 1)
        self.assertIn("action_map", payload)

    def test_recommend_papers_returns_payload_and_500_mapping(self) -> None:
        success = self.client.post("/api/user/recommend-papers", json={"user_id": "u1", "top_n": 3, "max_age_months": 6})
        self.recommendation_service.raise_error = True
        failure = self.client.post("/api/user/recommend-papers", json={"user_id": "u1", "top_n": 3, "max_age_months": 6})

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["top_n"], 3)
        self.assertEqual(failure.status_code, 500)
        self.assertIn("recommend failed", failure.json()["detail"])


if __name__ == "__main__":
    unittest.main()
