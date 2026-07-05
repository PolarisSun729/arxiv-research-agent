import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import user_router


class _FakeUserPreferenceStore:
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

    def remove_liked_paper(self, user_id: str, arxiv_id: str) -> bool:
        return True

    def remove_disliked_paper(self, user_id: str, arxiv_id: str) -> bool:
        return True

    def get_user_interest_vector(self, user_id: str):
        return {"user_id": user_id, "interest_vector": [0.1, 0.2]}


class _FakeMemoryService:
    def load_user_profile(self, user_id: str):
        return {
            "user_id": user_id,
            "positive_topics": ["RAG retrieval optimization"],
            "topic_evidence": {"RAG retrieval optimization": {"source_papers": ["2401.00001"]}},
            "preferred_answer_style": "concise",
        }

    def load_user_profile_detail(self, user_id: str):
        return {
            "user_id": user_id,
            "manual_profile": {"positive_topics": ["manual topic"]},
            "generated_profile": {"positive_topics": ["RAG retrieval optimization"]},
            "effective_profile": self.load_user_profile(user_id),
            "build_jobs": [{"job_id": "job-1", "status": "completed"}],
            "snapshots": [{"snapshot_id": "snap-1", "active": True}],
        }

    def get_profile_topic_evidence(self, user_id: str, topic: str):
        return {"user_id": user_id, "topic": topic, "found": True, "evidence": {"source_papers": ["2401.00001"]}}

    def create_profile_rebuild_job(self, user_id: str, build_config=None):
        return {"job_id": "job-1", "user_id": user_id, "status": "running", "progress": 0, "build_config": build_config or {}}

    def run_profile_rebuild_job(self, user_id: str, job_id: str, build_mode: str = "incremental", max_papers=None):
        return self.rebuild_user_research_profile(user_id, build_mode=build_mode, max_papers=max_papers)

    def get_profile_build_job(self, job_id: str):
        if job_id == "missing":
            return None
        return {"job_id": job_id, "status": "completed", "snapshot_id": "snap-1"}

    def list_profile_build_jobs(self, user_id: str, limit: int = 20):
        return [{"job_id": "job-1", "user_id": user_id, "status": "completed"}]

    def activate_profile_snapshot(self, user_id: str, snapshot_id: str):
        if snapshot_id == "missing":
            raise ValueError("profile_snapshot_not_found")
        return {"user_id": user_id, "positive_topics": ["RAG retrieval optimization"], "snapshot_id": snapshot_id}

    def patch_user_profile(self, user_id: str, patch, source: str):
        payload = dict(patch)
        payload["user_id"] = user_id
        payload["source"] = source
        return payload

    def rebuild_user_research_profile(self, user_id: str, build_mode: str = "incremental", max_papers=None):
        return {
            "user_id": user_id,
            "positive_topics": ["RAG retrieval optimization"],
            "negative_topics": [],
            "recent_topics": ["agent memory"],
            "preferred_categories": ["cs.CL"],
            "preferred_answer_style": "concise",
            "common_question_types": ["method"],
            "representative_papers": ["2401.00001"],
        }


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
        self.user_preference_store = _FakeUserPreferenceStore()
        self.memory_service = _FakeMemoryService()
        self.recommendation_service = _FakeRecommendationService()

        app = FastAPI()
        app.include_router(user_router.router, prefix="/api")
        for dependency in (dependencies.get_user_preference_store, user_router.get_user_preference_store):
            app.dependency_overrides[dependency] = lambda: self.user_preference_store
        for dependency in (dependencies.get_interest_vector_store, user_router.get_interest_vector_store):
            app.dependency_overrides[dependency] = lambda: self.user_preference_store
        for dependency in (dependencies.get_memory_service, user_router.get_memory_service):
            app.dependency_overrides[dependency] = lambda: self.memory_service
        for dependency in (dependencies.get_recommendation_service, user_router.get_recommendation_service):
            app.dependency_overrides[dependency] = lambda: self.recommendation_service
        self.client = TestClient(app)

    def test_get_user_preferences_returns_stable_payload(self) -> None:
        response = self.client.get("/api/user/preferences/u1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload.keys()),
            {"user_id", "liked_papers", "disliked_papers", "paper_actions", "research_profile"},
        )
        self.assertEqual(payload["user_id"], "u1")
        self.assertEqual(payload["liked_papers"], ["2401.00001"])
        self.assertEqual(payload["disliked_papers"], [])
        self.assertEqual(payload["paper_actions"], {"favorite": ["2401.00001"]})
        self.assertEqual(payload["research_profile"]["preferred_answer_style"], "concise")
        self.assertNotIn("deprecated", payload)
        self.assertNotIn("preferences", payload)

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
        explicit_preference = self.client.post(
            "/api/user/paper-action",
            json={"user_id": "u1", "arxiv_id": "2401.00001", "action_type": "like"},
        )

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["action_type"], "favorite")
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(explicit_preference.status_code, 422)
        self.assertIn("like/dislike", str(explicit_preference.json()["detail"]))

    def test_delete_paper_action_returns_success_and_404(self) -> None:
        success = self.client.request(
            "DELETE",
            "/api/user/paper-action",
            json={"user_id": "u1", "arxiv_id": "2401.00001", "action_type": "favorite"},
        )
        self.user_preference_store.remove_action_result = False
        missing = self.client.request(
            "DELETE",
            "/api/user/paper-action",
            json={"user_id": "u1", "arxiv_id": "2401.00001", "action_type": "favorite"},
        )

        self.assertEqual(success.status_code, 200)
        self.assertEqual(success.json()["status"], "success")
        self.assertEqual(missing.status_code, 404)

    def test_delete_paper_action_rejects_explicit_preference_action(self) -> None:
        response = self.client.request(
            "DELETE",
            "/api/user/paper-action",
            json={"user_id": "u1", "arxiv_id": "2401.00001", "action_type": "dislike"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("DELETE /user/dislike-paper", response.json()["detail"])

    def test_get_user_paper_actions_returns_items_and_map(self) -> None:
        response = self.client.get("/api/user/paper-actions/u1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(len(payload["actions"]), 1)
        self.assertIn("action_map", payload)

    def test_rebuild_research_profile_returns_regenerated_profile(self) -> None:
        response = self.client.post("/api/user/research-profile/rebuild", json={"user_id": "u1", "async_build": False})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["profile"]["positive_topics"], ["RAG retrieval optimization"])
        self.assertEqual(payload["profile"]["preferred_answer_style"], "concise")

    def test_research_profile_detail_job_evidence_and_snapshot_routes(self) -> None:
        detail = self.client.get("/api/user/research-profile/u1/detail")
        evidence = self.client.get("/api/user/research-profile/u1/topic-evidence", params={"topic": "RAG retrieval optimization"})
        rebuild = self.client.post("/api/user/research-profile/rebuild", json={"user_id": "u1"})
        job = self.client.get("/api/user/research-profile/build-jobs/job-1")
        jobs = self.client.get("/api/user/research-profile/u1/build-jobs")
        activate = self.client.post("/api/user/research-profile/snapshots/activate", json={"user_id": "u1", "snapshot_id": "snap-1"})

        self.assertEqual(detail.status_code, 200)
        self.assertIn("manual_profile", detail.json()["detail"])
        self.assertEqual(evidence.json()["evidence"]["source_papers"], ["2401.00001"])
        self.assertEqual(rebuild.json()["status"], "accepted")
        self.assertEqual(job.json()["job"]["status"], "completed")
        self.assertEqual(len(jobs.json()["items"]), 1)
        self.assertEqual(activate.json()["profile"]["snapshot_id"], "snap-1")

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
