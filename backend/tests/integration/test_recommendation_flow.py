import gc
import importlib.util
import sys
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from services.storage.database_service import DatabaseService
from tests.helpers import FakeEmbeddingService, build_database_service


def _load_recommendation_service_class():
    backend_dir = Path(__file__).resolve().parents[2]

    packages = {
        "services": backend_dir / "services",
        "services.arxiv": backend_dir / "services" / "arxiv",
        "services.recommendation": backend_dir / "services" / "recommendation",
        "services.storage": backend_dir / "services" / "storage",
        "services.embedding": backend_dir / "services" / "embedding",
        "services.memory": backend_dir / "services" / "memory",
    }
    for package_name, package_path in packages.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]
        sys.modules[package_name] = module

    if "services.storage.vector_store_service" not in sys.modules:
        module = types.ModuleType("services.storage.vector_store_service")

        class _VectorStoreService:
            pass

        module.VectorStoreService = _VectorStoreService
        sys.modules[module.__name__] = module

    if "services.embedding.embedding_service" not in sys.modules:
        module = types.ModuleType("services.embedding.embedding_service")

        class _EmbeddingConfig:
            def __init__(self, provider="fake", model_name="fake-embedding-model", dimension=3, api_key=None, base_url=None):
                self.provider = provider
                self.model_name = model_name
                self.dimension = dimension
                self.api_key = api_key
                self.base_url = base_url

        class _EmbeddingService:
            pass

        module.EmbeddingConfig = _EmbeddingConfig
        module.EmbeddingService = _EmbeddingService
        sys.modules[module.__name__] = module

    if "services.arxiv.arxiv_search_service" not in sys.modules:
        module = types.ModuleType("services.arxiv.arxiv_search_service")

        class _ArxivSearchService:
            pass

        module.ArxivSearchService = _ArxivSearchService
        sys.modules[module.__name__] = module

    if "services.arxiv.arxiv_oai_service" not in sys.modules:
        module = types.ModuleType("services.arxiv.arxiv_oai_service")

        class _ArxivOaiDatabaseService:
            pass

        module.ArxivOaiDatabaseService = _ArxivOaiDatabaseService
        sys.modules[module.__name__] = module

    if "services.memory" in sys.modules and not hasattr(sys.modules["services.memory"], "MemoryService"):
        spec = importlib.util.spec_from_file_location("services.memory", backend_dir / "services" / "memory" / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["services.memory"] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)

    module_name = "services.recommendation.recommendation_service"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, backend_dir / "services" / "recommendation" / "recommendation_service.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)

    return sys.modules[module_name].RecommendationService


RecommendationService = _load_recommendation_service_class()


class _FakeOaiDbService:
    def __init__(self, papers=None):
        self.papers = list(papers or [])

    def get_recent_papers(self, categories=None, max_age_months=12, max_results=50):
        _ = categories, max_age_months
        return list(self.papers)[:max_results]


class _RecommendationVectorStore:
    def __init__(self) -> None:
        self.paper_embeddings = {}
        self.similar_papers = []

    def seed_paper_embedding(self, arxiv_id: str, vector, *, title="", abstract="", categories=None, published_date="2024-01-01") -> None:
        self.paper_embeddings[str(arxiv_id)] = {
            "arxiv_id": str(arxiv_id),
            "vector": [float(value) for value in vector],
            "title": title,
            "abstract": abstract,
            "categories": categories or ["cs.CL"],
            "published_date": published_date,
        }

    def get_paper_embeddings_by_arxiv_ids(self, collection_name: str, arxiv_ids):
        _ = collection_name
        rows = []
        for arxiv_id in arxiv_ids:
            payload = self.paper_embeddings.get(str(arxiv_id))
            if payload is not None:
                rows.append(dict(payload))
        return rows

    def search_similar_papers(self, collection_name: str, query_vector, top_k=5, filter_arxiv_ids=None, **_kwargs):
        _ = collection_name, query_vector
        excluded = {str(value) for value in (filter_arxiv_ids or [])}
        rows = [dict(item) for item in self.similar_papers if str(item.get("arxiv_id")) not in excluded]
        return rows[:top_k]


class RecommendationFlowIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_service = build_database_service(DatabaseService)
        self.embedding_service = FakeEmbeddingService(dimension=3)
        self.vector_store_service = _RecommendationVectorStore()
        self.user_id = "user-1"
        self.collection_name = "test_paper_embeddings"
        self.recent_papers = [
            {
                "arxiv_id": "2401.10001",
                "title": "Candidate A",
                "authors": ["A"],
                "abstract": "retrieval augmented generation benchmark",
                "categories": ["cs.CL"],
                "created": "2024-01-01",
                "abs_url": "https://arxiv.org/abs/2401.10001",
            },
            {
                "arxiv_id": "2401.10002",
                "title": "Candidate B",
                "authors": ["B"],
                "abstract": "vision language evaluation",
                "categories": ["cs.AI"],
                "created": "2024-02-01",
                "abs_url": "https://arxiv.org/abs/2401.10002",
            },
        ]
        self.oai_db_service = _FakeOaiDbService(papers=self.recent_papers)
        self.service = RecommendationService(
            db_service=self.db_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            get_embedding_config=self.embedding_service.get_default_embedding_config,
            oai_db_service=self.oai_db_service,
            arxiv_service_factory=lambda: types.SimpleNamespace(),
            collection_name=self.collection_name,
        )
        self._add_paper("2401.00001", title="Liked Paper A", abstract="retrieval augmented generation", categories=["cs.CL"])
        self._add_paper("2401.00002", title="Liked Paper B", abstract="benchmark evaluation", categories=["cs.CL", "cs.IR"])
        self._add_paper("2401.00003", title="Disliked Paper", abstract="prompt engineering only", categories=["cs.AI"])

    def tearDown(self) -> None:
        temp_db = getattr(self.db_service, "_test_temp_db", None)
        self.service = None
        self.db_service = None
        gc.collect()
        if temp_db is not None:
            temp_db.cleanup()

    def _add_paper(self, arxiv_id: str, *, title: str, abstract: str, categories) -> None:
        self.db_service.add_paper(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": ["Author"],
                "abstract": abstract,
                "categories": list(categories),
                "published_date": "2024-01-01",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }
        )

    def _seed_interest_vectors(self) -> None:
        self.vector_store_service.seed_paper_embedding("2401.00001", [1.0, 0.0, 0.0], title="Liked Paper A", abstract="retrieval augmented generation", categories=["cs.CL"])
        self.vector_store_service.seed_paper_embedding("2401.00002", [0.8, 0.2, 0.0], title="Liked Paper B", abstract="benchmark evaluation", categories=["cs.CL", "cs.IR"])
        self.vector_store_service.seed_paper_embedding("2401.00003", [0.0, 1.0, 0.0], title="Disliked Paper", abstract="prompt engineering only", categories=["cs.AI"])

    def test_generate_user_interest_vector_from_liked_papers(self) -> None:
        self._seed_interest_vectors()
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        self.db_service.add_liked_paper(self.user_id, "2401.00002")

        result = self.service.generate_user_interest_vector(self.user_id)
        stored = self.db_service.get_user_interest_vector(self.user_id)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["liked_count"], 2)
        self.assertEqual(stored["paper_count"], 2)
        self.assertEqual(len(stored["vector_data"]), 3)

    def test_generate_user_interest_vector_tracks_disliked_feedback(self) -> None:
        self._seed_interest_vectors()
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        self.db_service.add_disliked_paper(self.user_id, "2401.00003")

        result = self.service.generate_user_interest_vector(self.user_id)
        stored = self.db_service.get_user_interest_vector(self.user_id)

        self.assertEqual(result["disliked_count"], 1)
        self.assertIsNotNone(stored["disliked_vector_data"])

    def test_candidate_recall_returns_fixed_candidates(self) -> None:
        candidates = self.service._fetch_recent_db_candidates(Counter(), max_age_months=12, max_results=10)

        self.assertEqual([item["arxiv_id"] for item in candidates], ["2401.10001", "2401.10002"])
        self.assertEqual(candidates[0]["score"], 0.0)

    def test_ranker_score_applies_negative_feedback_penalty(self) -> None:
        candidate = {
            "arxiv_id": "2401.10001",
            "title": "Candidate A",
            "abstract": "retrieval augmented generation benchmark",
            "categories": ["cs.CL"],
            "published_date": "2024-01-01",
        }
        stored_vector = self.embedding_service.create_single_embedding(
            self.embedding_service.build_paper_embedding_text(candidate["title"], candidate["abstract"])
        )
        candidate["_stored_vector"] = stored_vector
        liked_category_freq = Counter({"cs.CL": 2})
        config = self.embedding_service.get_default_embedding_config()
        scored_without_penalty = self.service._build_candidate_score(
            candidate=dict(candidate),
            liked_category_freq=liked_category_freq,
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[],
            disliked_vector=None,
            embedding_config=config,
        )
        scored_with_penalty = self.service._build_candidate_score(
            candidate=dict(candidate),
            liked_category_freq=liked_category_freq,
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[{"cluster_id": "c1", "centroid_vector": [1.0, 0.0, 0.0]}],
            disliked_vector=[1.0, 0.0, 0.0],
            embedding_config=config,
        )

        self.assertGreater(scored_with_penalty["disliked_penalty"], 0.0)
        self.assertLess(scored_with_penalty["final_score"], scored_without_penalty["final_score"])

    def test_recommend_papers_applies_top_n_and_stable_fields(self) -> None:
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        self.db_service.add_liked_paper(self.user_id, "2401.00002")
        self.db_service.upsert_user_research_profile(self.user_id, {"positive_topics": ["retrieval"], "preferred_categories": ["cs.CL"]})

        candidate_pool = [
            {
                "arxiv_id": "2401.10001",
                "title": "Candidate A",
                "authors": ["A"],
                "abstract": "retrieval augmented generation benchmark",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.10001",
                "score": 0.8,
            },
            {
                "arxiv_id": "2401.10002",
                "title": "Candidate B",
                "authors": ["B"],
                "abstract": "vision language evaluation",
                "categories": ["cs.AI"],
                "published_date": "2024-02-01",
                "url": "https://arxiv.org/abs/2401.10002",
                "score": 0.6,
            },
            {
                "arxiv_id": "2401.10003",
                "title": "Candidate C",
                "authors": ["C"],
                "abstract": "long context retrieval pipeline",
                "categories": ["cs.IR"],
                "published_date": "2024-03-01",
                "url": "https://arxiv.org/abs/2401.10003",
                "score": 0.7,
            },
        ]

        with mock.patch.object(self.service, "_get_or_refresh_interest_vector", return_value={"vector_data": [1.0, 0.0, 0.0], "interest_clusters": [], "disliked_vector_data": None, "profile_mode": "mean", "cluster_count": 0}), \
             mock.patch.object(self.service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(self.service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 3, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = self.service.recommend_papers(self.user_id, top_n=2)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["recommendations"]), 2)
        self.assertIn("recall_mode", result)
        first = result["recommendations"][0]
        self.assertIn("arxiv_id", first)
        self.assertIn("title", first)
        self.assertIn("final_score", first)
        self.assertIn("score_breakdown", first)

    def test_context_aware_recommendation_changes_with_query(self) -> None:
        candidate_pool = [
            {
                "arxiv_id": "2401.20001",
                "title": "RAG Pipeline",
                "authors": ["A"],
                "abstract": "retrieval augmented generation for enterprise qa",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.20001",
                "score": 0.5,
            },
            {
                "arxiv_id": "2401.20002",
                "title": "Agent Planning",
                "authors": ["B"],
                "abstract": "llm agent planning and tool orchestration",
                "categories": ["cs.AI"],
                "published_date": "2024-01-02",
                "url": "https://arxiv.org/abs/2401.20002",
                "score": 0.5,
            },
        ]

        with mock.patch.object(self.service, "_get_or_refresh_interest_vector", side_effect=HTTPException(status_code=400, detail="missing vector")), \
             mock.patch.object(self.service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(self.service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 2, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            rag_result = self.service.recommend_papers_with_context(
                self.user_id,
                top_n=1,
                message="推荐 RAG 方向论文",
                topic_hint="RAG",
                request_context={"force_cold_start": True, "positive_topics": ["rag"]},
            )
            agent_result = self.service.recommend_papers_with_context(
                self.user_id,
                top_n=1,
                message="推荐 Agent 方向论文",
                topic_hint="Agent",
                request_context={"force_cold_start": True, "positive_topics": ["agent"]},
            )

        self.assertEqual(rag_result["recommendations"][0]["arxiv_id"], "2401.20001")
        self.assertEqual(agent_result["recommendations"][0]["arxiv_id"], "2401.20002")

    def test_context_aware_recommendation_filters_negative_topics(self) -> None:
        candidate_pool = [
            {
                "arxiv_id": "2401.30001",
                "title": "Vision Agent",
                "authors": ["A"],
                "abstract": "vision language agent benchmark",
                "categories": ["cs.CV"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.30001",
                "score": 0.8,
            },
            {
                "arxiv_id": "2401.30002",
                "title": "RAG Agent",
                "authors": ["B"],
                "abstract": "rag agent retrieval workflow",
                "categories": ["cs.CL"],
                "published_date": "2024-01-02",
                "url": "https://arxiv.org/abs/2401.30002",
                "score": 0.7,
            },
        ]

        with mock.patch.object(self.service, "_get_or_refresh_interest_vector", side_effect=HTTPException(status_code=400, detail="missing vector")), \
             mock.patch.object(self.service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(self.service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 2, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = self.service.recommend_papers_with_context(
                self.user_id,
                top_n=2,
                message="推荐 agent 论文，但不要 vision",
                topic_hint="agent",
                research_profile={"negative_topics": ["vision"]},
                request_context={"force_cold_start": True, "negative_topics": ["vision"]},
            )

        self.assertEqual([item["arxiv_id"] for item in result["recommendations"]], ["2401.30002"])
        self.assertTrue(result["filter_debug"]["filtered_out_by_context"])

    def test_context_aware_recommendation_supports_cold_start_without_user_history(self) -> None:
        candidate_pool = [
            {
                "arxiv_id": "2401.40001",
                "title": "Cold Start RAG",
                "authors": ["A"],
                "abstract": "retrieval augmented generation tutorial",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.40001",
                "score": 0.6,
            },
            {
                "arxiv_id": "2401.40002",
                "title": "General Vision",
                "authors": ["B"],
                "abstract": "image generation benchmark",
                "categories": ["cs.CV"],
                "published_date": "2024-01-02",
                "url": "https://arxiv.org/abs/2401.40002",
                "score": 0.6,
            },
        ]

        with mock.patch.object(self.service, "_get_or_refresh_interest_vector", side_effect=HTTPException(status_code=400, detail="missing vector")), \
             mock.patch.object(self.service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(self.service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 2, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = self.service.recommend_papers_with_context(
                user_id="anonymous",
                top_n=1,
                message="推荐 RAG 论文",
                topic_hint="RAG",
                request_context={"force_cold_start": True, "positive_topics": ["rag"]},
            )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["recommendation_context"]["cold_start"])
        self.assertEqual(result["recommendations"][0]["arxiv_id"], "2401.40001")
        self.assertIn("recommendation_explanation", result["recommendations"][0])

    def test_recommend_papers_raises_for_empty_liked_papers(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self.service.generate_user_interest_vector(self.user_id)

        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
