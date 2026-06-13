import gc
import importlib.util
import json
import sys
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from services.storage.database_service import DatabaseService
from tests.helpers import FakeEmbeddingService, build_database_service


class FakeEvidenceGenerationService:
    def __init__(self, payload: str = ""):
        self.payload = payload
        self.call_count = 0

    def complete_with_qwen(self, prompt, *args, **kwargs):
        self.call_count += 1
        if self.payload:
            return self.payload
        text = str(prompt or "").lower()
        concepts = []
        if "retrieval augmented generation" in text or "rag" in text or "reranking" in text:
            concepts.append("RAG retrieval optimization")
        if "agent" in text:
            concepts.append("agent planning")
        if not concepts:
            concepts.append("question answering")
        return json.dumps(
            {
                "main_research_area": concepts[0],
                "research_objects": ["LLM agents"] if "agent" in text else [],
                "methods": [item for item in concepts if item == "RAG retrieval optimization"],
                "tasks": ["question answering"] if "question answering" in text or "qa" in text else [],
                "application_domains": [],
                "technical_concepts": concepts,
                "evaluation_focus": ["retrieval quality"] if "retrieval" in text else [],
                "system_type": "",
                "candidate_concepts": [
                    {
                        "label": concept,
                        "type": "technical_concept",
                        "confidence": 0.9,
                        "evidence_text": "fake evidence",
                        "source": "llm",
                        "whether_generalizable": True,
                    }
                    for concept in concepts
                ],
                "excluded_concepts": [],
                "extraction_confidence": 0.9,
            },
            ensure_ascii=False,
        )


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
        self.fake_llm = FakeEvidenceGenerationService()
        self.service = RecommendationService(
            db_service=self.db_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            get_embedding_config=self.embedding_service.get_default_embedding_config,
            memory_service=sys.modules["services.memory"].MemoryService(
                db_service=self.db_service,
                generation_service=self.fake_llm,
            ),
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
        self.assertEqual(stored["vector_data"], [1.0, 0.0, 0.0])
        self.assertEqual(stored["paper_count"], 1)
        self.assertEqual(stored["negative_feedback_stats"]["usable_disliked"], 1)
        self.assertFalse(stored["negative_feedback_stats"]["participates_in_main_vector"])
        self.assertEqual(stored["disliked_paper_examples"][0]["arxiv_id"], "2401.00003")
        self.assertEqual(stored["negative_feedback_profile"]["mode"], "examples")
        self.assertEqual(stored["negative_feedback_profile"]["examples"][0]["arxiv_id"], "2401.00003")
        self.assertEqual(len(stored["negative_feedback_profile"]["examples"][0]["vector"]), 3)

    def test_legacy_interest_vector_projects_disliked_examples_to_negative_profile(self) -> None:
        legacy_examples = [
            {
                "arxiv_id": "2401.00003",
                "title": "Disliked Paper",
                "vector": [0.0, 1.0, 0.0],
                "vector_source": "milvus",
            }
        ]
        self.db_service.save_user_interest_vector(
            user_id=self.user_id,
            vector_data=[1.0, 0.0, 0.0],
            paper_count=1,
            embedding_model="fake-embedding-model",
            vector_dimension=3,
            disliked_paper_examples=legacy_examples,
        )

        stored = self.db_service.get_user_interest_vector(self.user_id)

        self.assertTrue(stored["negative_feedback_profile"]["enabled"])
        self.assertEqual(stored["negative_feedback_profile"]["mode"], "examples")
        self.assertEqual(stored["negative_feedback_profile"]["examples"], legacy_examples)
        self.assertEqual(stored["negative_feedback_stats"]["stored_examples"], 1)
        self.assertEqual(stored["negative_feedback_stats"]["usable_disliked"], 1)

    def test_generate_user_interest_vector_builds_negative_clusters(self) -> None:
        self._seed_interest_vectors()
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        extra_disliked = [
            ("2401.00004", [0.0, 0.9, 0.1]),
            ("2401.00005", [0.0, 0.8, 0.2]),
            ("2401.00006", [0.0, 1.0, 0.1]),
        ]
        self.db_service.add_disliked_paper(self.user_id, "2401.00003")
        for arxiv_id, vector in extra_disliked:
            self._add_paper(arxiv_id, title=f"Disliked {arxiv_id}", abstract="vision benchmark", categories=["cs.CV"])
            self.vector_store_service.seed_paper_embedding(arxiv_id, vector, title=f"Disliked {arxiv_id}", abstract="vision benchmark", categories=["cs.CV"])
            self.db_service.add_disliked_paper(self.user_id, arxiv_id)

        cluster_payload = [
            {
                "cluster_id": "negative_cluster_0",
                "cluster_label": 7,
                "centroid_vector": [0.0, 1.0, 0.0],
                "paper_count": 4,
                "paper_ids": ["2401.00003", "2401.00004", "2401.00005", "2401.00006"],
            }
        ]
        with mock.patch.object(self.service, "_cluster_negative_feedback_vectors", return_value=cluster_payload) as cluster_mock:
            result = self.service.generate_user_interest_vector(self.user_id)

        stored = self.db_service.get_user_interest_vector(self.user_id)
        cluster_mock.assert_called_once()
        self.assertEqual(result["negative_cluster_count"], 1)
        self.assertEqual(stored["negative_feedback_profile"]["mode"], "clusters")
        self.assertEqual(stored["negative_feedback_profile"]["clusters"], cluster_payload)
        self.assertEqual(stored["negative_feedback_stats"]["negative_cluster_count"], 1)

    def test_generate_user_interest_vector_falls_back_to_negative_examples_when_clustering_fails(self) -> None:
        self._seed_interest_vectors()
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        for index in range(4, 7):
            arxiv_id = f"2401.0000{index}"
            self._add_paper(arxiv_id, title=f"Disliked {index}", abstract="survey only", categories=["cs.AI"])
            self.vector_store_service.seed_paper_embedding(arxiv_id, [0.0, 1.0, float(index) / 10.0], title=f"Disliked {index}", abstract="survey only", categories=["cs.AI"])
            self.db_service.add_disliked_paper(self.user_id, arxiv_id)
        self.db_service.add_disliked_paper(self.user_id, "2401.00003")

        with mock.patch.object(self.service, "_cluster_negative_feedback_vectors", return_value=[]):
            self.service.generate_user_interest_vector(self.user_id)

        stored = self.db_service.get_user_interest_vector(self.user_id)
        self.assertEqual(stored["negative_feedback_profile"]["mode"], "cluster_fallback_examples")
        self.assertEqual(stored["negative_feedback_stats"]["fallback_reason"], "no_stable_negative_clusters")
        self.assertGreaterEqual(len(stored["negative_feedback_profile"]["examples"]), 1)

    def test_generate_user_interest_vector_builds_positive_clusters(self) -> None:
        self._seed_interest_vectors()
        self._add_paper("2401.00004", title="Liked Paper C", abstract="retrieval reranking", categories=["cs.IR"])
        self._add_paper("2401.00005", title="Liked Paper D", abstract="rag evaluation", categories=["cs.CL"])
        self.vector_store_service.seed_paper_embedding("2401.00004", [0.7, 0.3, 0.0], title="Liked Paper C", abstract="retrieval reranking", categories=["cs.IR"])
        self.vector_store_service.seed_paper_embedding("2401.00005", [0.9, 0.1, 0.0], title="Liked Paper D", abstract="rag evaluation", categories=["cs.CL"])
        for arxiv_id in ("2401.00001", "2401.00002", "2401.00004", "2401.00005"):
            self.db_service.add_liked_paper(self.user_id, arxiv_id)

        cluster_payload = [
            {
                "cluster_id": "cluster_0",
                "centroid_vector": [1.0, 0.0, 0.0],
                "paper_count": 4,
                "paper_ids": ["2401.00001", "2401.00002", "2401.00004", "2401.00005"],
            }
        ]
        with mock.patch.object(self.service, "_cluster_interest_vectors", return_value=(cluster_payload, None)) as cluster_mock:
            result = self.service.generate_user_interest_vector(self.user_id)

        stored = self.db_service.get_user_interest_vector(self.user_id)
        cluster_mock.assert_called_once()
        self.assertEqual(result["profile_mode"], "clustered")
        self.assertEqual(stored["cluster_count"], 1)
        self.assertEqual(stored["interest_clusters"], cluster_payload)

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
        candidate["_stored_vector"] = [1.0, 0.0, 0.0]
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
            interest_clusters=[],
            disliked_vector=[1.0, 0.0, 0.0],
            embedding_config=config,
        )

        self.assertGreater(scored_with_penalty["disliked_penalty"], 0.0)
        self.assertGreater(scored_with_penalty["negative_score"], scored_with_penalty["negative_feedback_debug"]["similarity_threshold"])
        self.assertTrue(scored_with_penalty["negative_penalty_applied"])
        self.assertLess(scored_with_penalty["final_score"], scored_without_penalty["final_score"])

    def test_ranker_uses_negative_examples_and_clusters(self) -> None:
        candidate = {
            "arxiv_id": "2401.10001",
            "title": "Candidate A",
            "abstract": "retrieval augmented generation benchmark",
            "categories": ["cs.CL"],
            "published_date": "2024-01-01",
            "_stored_vector": [0.0, 1.0, 0.0],
        }
        config = self.embedding_service.get_default_embedding_config()
        example_profile = {
            "version": "negative_feedback_profile_v1",
            "enabled": True,
            "mode": "examples",
            "examples": [{"arxiv_id": "bad-1", "vector": [0.0, 1.0, 0.0]}],
            "clusters": [],
            "stats": {"mode": "examples", "usable_disliked": 1},
        }
        cluster_profile = {
            "version": "negative_feedback_profile_v1",
            "enabled": True,
            "mode": "clusters",
            "examples": [],
            "clusters": [{"cluster_id": "negative_cluster_0", "cluster_label": 0, "centroid_vector": [0.0, 1.0, 0.0], "paper_ids": ["bad-1"], "paper_count": 1}],
            "stats": {"mode": "clusters", "usable_disliked": 6},
        }

        scored_by_example = self.service._build_candidate_score(
            candidate=dict(candidate),
            liked_category_freq=Counter(),
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[],
            negative_feedback_profile=example_profile,
            embedding_config=config,
        )
        scored_by_cluster = self.service._build_candidate_score(
            candidate=dict(candidate),
            liked_category_freq=Counter(),
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[],
            negative_feedback_profile=cluster_profile,
            embedding_config=config,
        )

        self.assertEqual(scored_by_example["negative_feedback_match"]["source"], "negative_example")
        self.assertEqual(scored_by_cluster["negative_feedback_match"]["source"], "negative_cluster")
        self.assertGreater(scored_by_example["disliked_penalty"], 0.0)
        self.assertGreater(scored_by_cluster["disliked_penalty"], 0.0)
        self.assertLess(scored_by_example["negative_confidence"], scored_by_cluster["negative_confidence"])

    def test_ranker_negative_feedback_uses_threshold_margin_and_confidence(self) -> None:
        config = self.embedding_service.get_default_embedding_config()
        base_candidate = {
            "arxiv_id": "2401.10001",
            "title": "Candidate A",
            "abstract": "retrieval augmented generation benchmark",
            "categories": ["cs.CL"],
            "published_date": "2024-01-01",
            "_stored_vector": [1.0, 0.0, 0.0],
        }
        low_negative_profile = {
            "version": "negative_feedback_profile_v1",
            "enabled": True,
            "mode": "examples",
            "examples": [{"arxiv_id": "bad-low", "vector": [0.0, 1.0, 0.0]}],
            "clusters": [],
            "stats": {"mode": "examples", "usable_disliked": 6},
        }
        weak_negative_profile = {
            "version": "negative_feedback_profile_v1",
            "enabled": True,
            "mode": "examples",
            "examples": [{"arxiv_id": "bad-weak", "vector": [1.0, 0.0, 0.0]}],
            "clusters": [],
            "stats": {"mode": "examples", "usable_disliked": 1},
        }
        full_negative_profile = {
            **weak_negative_profile,
            "stats": {"mode": "examples", "usable_disliked": 6},
        }
        near_threshold_profile = {
            "version": "negative_feedback_profile_v1",
            "enabled": True,
            "mode": "examples",
            "examples": [{"arxiv_id": "bad-near", "vector": [0.7, 0.714142842854285, 0.0]}],
            "clusters": [],
            "stats": {"mode": "examples", "usable_disliked": 6},
        }

        scored_low = self.service._build_candidate_score(
            candidate=dict(base_candidate),
            liked_category_freq=Counter(),
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[],
            negative_feedback_profile=low_negative_profile,
            embedding_config=config,
        )
        scored_weak = self.service._build_candidate_score(
            candidate=dict(base_candidate),
            liked_category_freq=Counter(),
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[],
            negative_feedback_profile=weak_negative_profile,
            embedding_config=config,
        )
        scored_full = self.service._build_candidate_score(
            candidate=dict(base_candidate),
            liked_category_freq=Counter(),
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[],
            negative_feedback_profile=full_negative_profile,
            embedding_config=config,
        )
        scored_near_threshold = self.service._build_candidate_score(
            candidate=dict(base_candidate),
            liked_category_freq=Counter(),
            user_vector=[1.0, 0.0, 0.0],
            interest_clusters=[],
            negative_feedback_profile=near_threshold_profile,
            embedding_config=config,
        )

        self.assertEqual(scored_low["disliked_penalty"], 0.0)
        self.assertFalse(scored_low["negative_penalty_applied"])
        self.assertLess(scored_weak["disliked_penalty"], scored_full["disliked_penalty"])
        self.assertGreater(scored_full["score_breakdown"]["negative_margin_penalty"], 0.0)
        self.assertGreater(scored_full["disliked_penalty"], scored_near_threshold["disliked_penalty"])

    def test_ranker_negative_hard_filter_is_config_gated(self) -> None:
        config = self.embedding_service.get_default_embedding_config()
        candidate = {
            "arxiv_id": "2401.10001",
            "title": "Candidate A",
            "abstract": "retrieval augmented generation benchmark",
            "categories": ["cs.CL"],
            "published_date": "2024-01-01",
            "_stored_vector": [1.0, 0.0, 0.0],
        }
        negative_profile = {
            "version": "negative_feedback_profile_v1",
            "enabled": True,
            "mode": "examples",
            "examples": [{"arxiv_id": "bad-hard", "vector": [1.0, 0.0, 0.0]}],
            "clusters": [],
            "stats": {"mode": "examples", "usable_disliked": 6},
        }
        negative_config = self.service.RECOMMENDATION_CONFIG["ranking"]["negative"]
        original_config = dict(negative_config)
        try:
            negative_config["enable_hard_filter"] = False
            soft_scored = self.service._build_candidate_score(
                candidate=dict(candidate),
                liked_category_freq=Counter(),
                user_vector=[1.0, 0.0, 0.0],
                interest_clusters=[],
                negative_feedback_profile=negative_profile,
                embedding_config=config,
            )
            negative_config["enable_hard_filter"] = True
            hard_scored = self.service._build_candidate_score(
                candidate=dict(candidate),
                liked_category_freq=Counter(),
                user_vector=[1.0, 0.0, 0.0],
                interest_clusters=[],
                negative_feedback_profile=negative_profile,
                embedding_config=config,
            )
        finally:
            negative_config.clear()
            negative_config.update(original_config)

        self.assertFalse(soft_scored["negative_hard_filter"])
        self.assertTrue(hard_scored["negative_hard_filter"])

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

    def test_recommendation_enriches_top_candidate_concepts_and_reuses_cache(self) -> None:
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        self.db_service.upsert_user_research_profile(
            self.user_id,
            {
                "positive_topics": ["RAG retrieval optimization"],
                "canonical_topics": [{"label": "RAG retrieval optimization", "aliases": []}],
            },
        )
        candidate_pool = [
            {
                "arxiv_id": "2401.11001",
                "title": "Fresh RAG Candidate",
                "authors": ["A"],
                "abstract": "retrieval augmented generation with reranking for enterprise qa",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.11001",
                "score": 0.95,
                "_stored_vector": [1.0, 0.0, 0.0],
            },
            {
                "arxiv_id": "2401.11002",
                "title": "Lower Candidate",
                "authors": ["B"],
                "abstract": "generic benchmark paper",
                "categories": ["cs.AI"],
                "published_date": "2024-01-02",
                "url": "https://arxiv.org/abs/2401.11002",
                "score": 0.30,
                "_stored_vector": [0.2, 0.1, 0.0],
            },
        ]

        with mock.patch.dict(self.service.RECOMMENDATION_CONFIG["candidate_concept_enrichment"], {"top_k": 1}, clear=False), \
             mock.patch.object(self.service, "_build_profile_signal_bundle", return_value={"profile": {"positive_topics": ["RAG retrieval optimization"], "canonical_topics": [{"label": "RAG retrieval optimization", "aliases": []}]}, "actions": {}, "excluded_ids": [], "disabled": False}), \
             mock.patch.object(self.service, "_get_or_refresh_interest_vector", return_value={"vector_data": [1.0, 0.0, 0.0], "interest_clusters": [], "disliked_vector_data": None, "negative_feedback_profile": {}, "profile_mode": "mean", "cluster_count": 0}), \
             mock.patch.object(self.service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(self.service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 2, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = self.service.recommend_papers(self.user_id, top_n=2)
            repeated = self.service.recommend_papers(self.user_id, top_n=2)

        first = result["recommendations"][0]
        self.assertEqual(self.fake_llm.call_count, 1)
        self.assertEqual(first["arxiv_id"], "2401.11001")
        self.assertEqual(first["candidate_concept_source"], "runtime_enriched")
        self.assertIn("RAG retrieval optimization", first["matched_positive_topics"])
        self.assertEqual(first["candidate_concept_debug"]["source"], "runtime_enriched")
        self.assertEqual(first["profile_adjustment_debug"]["profile_score"], first["profile_score"])
        self.assertIn("RAG retrieval optimization", first["profile_adjustment_debug"]["candidate_concept_debug"]["generated_concepts"])
        second = next(item for item in result["recommendations"] if item["arxiv_id"] == "2401.11002")
        self.assertEqual(second["candidate_concept_source"], "title_abstract_fallback")
        self.assertEqual(repeated["ranking_debug"]["candidate_concept_enrichment"]["cache_hit_count"], 1)
        self.assertEqual(result["ranking_debug"]["candidate_concept_enrichment"]["selected_count"], 1)
        cached = self.db_service.get_paper_profile_evidence("2401.11001")
        self.assertTrue(cached["candidate_concepts"])
        self.assertIsNone(self.db_service.get_paper_profile_evidence("2401.11002"))

    def test_recommendation_concept_enrichment_failure_falls_back_without_crashing(self) -> None:
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        self.db_service.upsert_user_research_profile(
            self.user_id,
            {
                "positive_topics": ["question answering"],
                "canonical_topics": [{"label": "question answering", "aliases": []}],
            },
        )
        failing_service = RecommendationService(
            db_service=self.db_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            get_embedding_config=self.embedding_service.get_default_embedding_config,
            memory_service=sys.modules["services.memory"].MemoryService(
                db_service=self.db_service,
                generation_service=FakeEvidenceGenerationService("not json"),
            ),
            oai_db_service=self.oai_db_service,
            arxiv_service_factory=lambda: types.SimpleNamespace(),
            collection_name=self.collection_name,
        )
        candidate_pool = [
            {
                "arxiv_id": "2401.12001",
                "title": "Broken Evidence Candidate",
                "authors": ["A"],
                "abstract": "question answering without cached evidence",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.12001",
                "score": 0.8,
                "_stored_vector": [1.0, 0.0, 0.0],
            }
        ]

        with mock.patch.dict(failing_service.RECOMMENDATION_CONFIG["candidate_concept_enrichment"], {"top_k": 1}, clear=False), \
             mock.patch.object(failing_service, "_build_profile_signal_bundle", return_value={"profile": {"positive_topics": ["question answering"], "canonical_topics": [{"label": "question answering", "aliases": []}]}, "actions": {}, "excluded_ids": [], "disabled": False}), \
             mock.patch.object(failing_service, "_get_or_refresh_interest_vector", return_value={"vector_data": [1.0, 0.0, 0.0], "interest_clusters": [], "disliked_vector_data": None, "negative_feedback_profile": {}, "profile_mode": "mean", "cluster_count": 0}), \
             mock.patch.object(failing_service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(failing_service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 1, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = failing_service.recommend_papers(self.user_id, top_n=1)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ranking_debug"]["candidate_concept_enrichment"]["failed_count"], 1)
        self.assertEqual(result["recommendations"][0]["candidate_concept_source"], "title_abstract_fallback")
        self.assertEqual(result["recommendations"][0]["candidate_concept_debug"]["source_detail"], "generation_failed")
        self.assertIsNotNone(self.db_service.get_paper_profile_evidence("2401.12001"))

    def test_recommendation_concept_enrichment_can_be_disabled(self) -> None:
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        candidate_pool = [
            {
                "arxiv_id": "2401.13001",
                "title": "Disabled Enrichment Candidate",
                "authors": ["A"],
                "abstract": "retrieval augmented generation without cache",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.13001",
                "score": 0.9,
                "_stored_vector": [1.0, 0.0, 0.0],
            }
        ]

        with mock.patch.dict(self.service.RECOMMENDATION_CONFIG["candidate_concept_enrichment"], {"enabled": False, "top_k": 1}, clear=False), \
             mock.patch.object(self.service, "_build_profile_signal_bundle", return_value={"profile": {"positive_topics": ["RAG retrieval optimization"], "canonical_topics": [{"label": "RAG retrieval optimization", "aliases": []}]}, "actions": {}, "excluded_ids": [], "disabled": False}), \
             mock.patch.object(self.service, "_get_or_refresh_interest_vector", return_value={"vector_data": [1.0, 0.0, 0.0], "interest_clusters": [], "disliked_vector_data": None, "negative_feedback_profile": {}, "profile_mode": "mean", "cluster_count": 0}), \
             mock.patch.object(self.service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(self.service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 1, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = self.service.recommend_papers(self.user_id, top_n=1)

        self.assertEqual(self.fake_llm.call_count, 0)
        self.assertFalse(result["ranking_debug"]["candidate_concept_enrichment"]["enabled"])
        self.assertEqual(result["recommendations"][0]["candidate_concept_source"], "title_abstract_fallback")

    def test_recommendation_concept_enrichment_respects_llm_budget(self) -> None:
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        candidate_pool = [
            {
                "arxiv_id": "2401.14001",
                "title": "Budget Candidate A",
                "authors": ["A"],
                "abstract": "retrieval augmented generation",
                "categories": ["cs.CL"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.14001",
                "score": 0.95,
                "_stored_vector": [1.0, 0.0, 0.0],
            },
            {
                "arxiv_id": "2401.14002",
                "title": "Budget Candidate B",
                "authors": ["B"],
                "abstract": "agent planning and retrieval",
                "categories": ["cs.AI"],
                "published_date": "2024-01-02",
                "url": "https://arxiv.org/abs/2401.14002",
                "score": 0.94,
                "_stored_vector": [0.9, 0.1, 0.0],
            },
        ]

        with mock.patch.dict(self.service.RECOMMENDATION_CONFIG["candidate_concept_enrichment"], {"top_k": 2, "max_llm_calls": 1}, clear=False), \
             mock.patch.object(self.service, "_build_profile_signal_bundle", return_value={"profile": {"positive_topics": ["RAG retrieval optimization"], "canonical_topics": [{"label": "RAG retrieval optimization", "aliases": []}]}, "actions": {}, "excluded_ids": [], "disabled": False}), \
             mock.patch.object(self.service, "_get_or_refresh_interest_vector", return_value={"vector_data": [1.0, 0.0, 0.0], "interest_clusters": [], "disliked_vector_data": None, "negative_feedback_profile": {}, "profile_mode": "mean", "cluster_count": 0}), \
             mock.patch.object(self.service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(self.service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 2, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = self.service.recommend_papers(self.user_id, top_n=2)

        self.assertEqual(self.fake_llm.call_count, 1)
        self.assertEqual(result["ranking_debug"]["candidate_concept_enrichment"]["llm_attempt_count"], 1)
        self.assertEqual(result["ranking_debug"]["candidate_concept_enrichment"]["budget_skipped_count"], 1)

    def test_recommendation_concept_enrichment_supports_negative_profile_debug(self) -> None:
        self.db_service.add_liked_paper(self.user_id, "2401.00001")
        negative_llm = FakeEvidenceGenerationService(
            json.dumps(
                {
                    "main_research_area": "vision-only generation",
                    "research_objects": [],
                    "methods": [],
                    "tasks": [],
                    "application_domains": [],
                    "technical_concepts": ["vision-only generation"],
                    "evaluation_focus": [],
                    "system_type": "",
                    "candidate_concepts": [
                        {
                            "label": "vision-only generation",
                            "type": "technical_concept",
                            "confidence": 0.9,
                            "evidence_text": "fake evidence",
                            "source": "llm",
                            "whether_generalizable": True,
                        }
                    ],
                    "excluded_concepts": [],
                    "extraction_confidence": 0.9,
                },
                ensure_ascii=False,
            )
        )
        negative_service = RecommendationService(
            db_service=self.db_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            get_embedding_config=self.embedding_service.get_default_embedding_config,
            memory_service=sys.modules["services.memory"].MemoryService(
                db_service=self.db_service,
                generation_service=negative_llm,
            ),
            oai_db_service=self.oai_db_service,
            arxiv_service_factory=lambda: types.SimpleNamespace(),
            collection_name=self.collection_name,
        )
        candidate_pool = [
            {
                "arxiv_id": "2401.15001",
                "title": "Vision Generator",
                "authors": ["A"],
                "abstract": "vision-only generation system for image synthesis",
                "categories": ["cs.CV"],
                "published_date": "2024-01-01",
                "url": "https://arxiv.org/abs/2401.15001",
                "score": 0.9,
                "_stored_vector": [1.0, 0.0, 0.0],
            }
        ]

        with mock.patch.dict(negative_service.RECOMMENDATION_CONFIG["candidate_concept_enrichment"], {"top_k": 1}, clear=False), \
             mock.patch.object(negative_service, "_build_profile_signal_bundle", return_value={"profile": {"negative_topics": ["vision-only generation"], "canonical_negative_topics": [{"label": "vision-only generation", "aliases": []}]}, "actions": {}, "excluded_ids": [], "disabled": False}), \
             mock.patch.object(negative_service, "_get_or_refresh_interest_vector", return_value={"vector_data": [1.0, 0.0, 0.0], "interest_clusters": [], "disliked_vector_data": None, "negative_feedback_profile": {}, "profile_mode": "mean", "cluster_count": 0}), \
             mock.patch.object(negative_service, "_fetch_recent_db_candidates", return_value=list(candidate_pool)), \
             mock.patch.object(negative_service, "_materialize_candidate_papers_for_recommendation", return_value=(list(candidate_pool), {"total": 1, "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0})):
            result = negative_service.recommend_papers(self.user_id, top_n=1)

        first = result["recommendations"][0]
        self.assertIn("vision-only generation", first["matched_negative_topics"])
        self.assertGreater(first["profile_penalty"], 0.0)
        self.assertLess(first["profile_adjustment_debug"]["profile_adjustment"], 0.0)

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

    def test_search_rerank_hard_excludes_disliked_papers(self) -> None:
        self.db_service.add_disliked_paper(self.user_id, "2401.00003")
        papers = [
            {
                "arxiv_id": "2401.00003",
                "title": "Disliked Paper",
                "abstract": "prompt engineering only",
                "categories": ["cs.AI"],
                "published_date": "2024-01-01",
            },
            {
                "arxiv_id": "2401.50001",
                "title": "Allowed Paper",
                "abstract": "retrieval augmented generation",
                "categories": ["cs.CL"],
                "published_date": "2024-01-02",
            },
        ]
        with mock.patch.object(
            self.service,
            "_get_or_refresh_interest_vector",
            return_value={
                "vector_data": [1.0, 0.0, 0.0],
                "interest_clusters": [],
                "negative_feedback_profile": {"version": "negative_feedback_profile_v1", "enabled": True, "mode": "none", "examples": [], "clusters": [], "stats": {}},
                "profile_mode": "mean",
                "cluster_count": 0,
            },
        ):
            result = self.service.rerank_search_results_for_user(
                user_id=self.user_id,
                papers=papers,
                query="retrieval",
                top_n=2,
            )

        self.assertEqual(result["hard_excluded_count"], 1)
        self.assertEqual([paper["arxiv_id"] for paper in result["papers"]], ["2401.50001"])

    def test_recommend_papers_raises_for_empty_liked_papers(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self.service.generate_user_interest_vector(self.user_id)

        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
