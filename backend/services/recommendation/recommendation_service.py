from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.storage.database_service import DatabaseService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.storage.vector_store_service import VectorStoreService
from utils.config import (
    get_enhanced_retrieval_runtime_config,
    get_recommendation_clustering_runtime_config,
    get_recommendation_runtime_config,
)

from .candidate_materializer import CandidateMaterializer
from .candidate_recall_service import CandidateRecallService
from .interest_profile_service import InterestProfileService
from .recommendation_ranker import RecommendationRanker

logger = logging.getLogger(__name__)


class RecommendationService(InterestProfileService, CandidateRecallService, CandidateMaterializer, RecommendationRanker):
    RECOMMENDATION_CONFIG = get_recommendation_runtime_config()
    ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()
    ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS = RECOMMENDATION_CONFIG["backfill_request_interval_seconds"]
    MIN_LIKED_PAPERS_FOR_CLUSTERING = RECOMMENDATION_CONFIG["min_liked_papers_for_clustering"]
    MAX_INTEREST_CLUSTERS = RECOMMENDATION_CONFIG["max_interest_clusters"]
    RECOMMEND_CANDIDATE_CATEGORIES = [
        "cs.CL",
        "cs.LG",
        "cs.IR",
        "cs.AI",
    ]

    def __init__(
        self,
        db_service: DatabaseService,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        get_embedding_config: Callable[[], EmbeddingConfig],
        get_clustering_config: Optional[Callable[[], Dict[str, Any]]] = None,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        oai_db_service: Optional[ArxivOaiDatabaseService] = None,
        collection_name: str = "arxiv_paper_embeddings",
    ):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.get_embedding_config = get_embedding_config
        self.get_clustering_config = get_clustering_config or get_recommendation_clustering_runtime_config
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
        self.oai_db_service = oai_db_service or ArxivOaiDatabaseService()
        self.collection_name = collection_name
        self._arxiv_backfill_lock = threading.Lock()
        self._arxiv_backfill_next_allowed_time = 0.0

    def _get_or_refresh_interest_vector(self, user_id: str) -> Dict[str, Any]:
        vector_data = self.db_service.get_user_interest_vector(user_id=user_id)
        latest_preference_ts = self.db_service.get_latest_user_preference_timestamp(user_id=user_id)

        if vector_data and latest_preference_ts and self._is_vector_stale(vector_data.get("updated_at"), latest_preference_ts):
            logger.info("Interest vector is stale for user %s, rebuilding", user_id)
            self.generate_user_interest_vector(user_id=user_id)
            vector_data = self.db_service.get_user_interest_vector(user_id=user_id)

        if not vector_data:
            logger.info("Interest vector missing for user %s, rebuilding", user_id)
            self.generate_user_interest_vector(user_id=user_id)
            vector_data = self.db_service.get_user_interest_vector(user_id=user_id)

        if not vector_data:
            raise HTTPException(status_code=400, detail="User interest vector not found. Please generate it first.")

        return vector_data

    def recommend_papers(
        self,
        user_id: str,
        top_n: int = RECOMMENDATION_CONFIG["default_top_n"],
        max_age_months: int = RECOMMENDATION_CONFIG["default_max_age_months"],
    ) -> Dict[str, Any]:
        user_vector_data = self._get_or_refresh_interest_vector(user_id)
        user_vector = user_vector_data["vector_data"]
        interest_clusters = user_vector_data.get("interest_clusters", []) or []
        disliked_vector = user_vector_data.get("disliked_vector_data")

        logger.info("Starting paper recommendation for user %s with top_n=%s max_age_months=%s", user_id, top_n, max_age_months)

        preferences = self.db_service.get_user_preferences(user_id=user_id)
        liked_ids = preferences.get("liked_papers", [])
        disliked_ids = preferences.get("disliked_papers", [])
        excluded_ids = list(dict.fromkeys([*liked_ids, *disliked_ids]))

        liked_details = self.db_service.get_liked_papers_with_details(user_id=user_id)
        liked_category_freq = self._build_liked_category_frequency(liked_details)

        candidate_limit = max(top_n * 5, 50)
        cluster_recall_candidates: List[Dict[str, Any]] = []
        if interest_clusters:
            cluster_recall_candidates = self._fetch_cluster_recall_candidates(
                interest_clusters=interest_clusters,
                excluded_ids=excluded_ids,
                top_k=max(top_n * 3, 20),
            )
            logger.info(
                "Fetched %s cluster recall candidates for user %s from %s interest clusters",
                len(cluster_recall_candidates),
                user_id,
                len(interest_clusters),
            )

        if cluster_recall_candidates:
            candidates = cluster_recall_candidates
            recall_mode = "cluster_recall"
        else:
            candidates = self._fetch_recent_db_candidates(
                liked_category_freq=liked_category_freq,
                max_age_months=max_age_months,
                max_results=max(candidate_limit * 2, candidate_limit),
            )
            recall_mode = "recent_pool"
            logger.info("Fetched %s recent OAI DB candidates for user %s before deduplication", len(candidates), user_id)

        filtered_candidates = self._deduplicate_candidates(candidates, excluded_ids)
        logger.info(
            "Retained %s candidate papers after deduplication against %s excluded papers for user %s",
            len(filtered_candidates),
            len(excluded_ids),
            user_id,
        )
        if not filtered_candidates:
            if recall_mode == "cluster_recall":
                candidates = self._fetch_recent_db_candidates(
                    liked_category_freq=liked_category_freq,
                    max_age_months=max_age_months,
                    max_results=max(candidate_limit * 2, candidate_limit),
                )
                logger.info(
                    "Cluster recall returned no candidates for user %s; falling back to recent OAI DB pool with %s papers",
                    user_id,
                    len(candidates),
                )
                filtered_candidates = self._deduplicate_candidates(candidates, excluded_ids)
                logger.info(
                    "Retained %s fallback candidate papers after deduplication against %s excluded papers for user %s",
                    len(filtered_candidates),
                    len(excluded_ids),
                    user_id,
                )
                recall_mode = "cluster_recall_fallback_recent_pool"
            if not filtered_candidates:
                raise HTTPException(status_code=400, detail=f"No papers found for recommendation within the last {max_age_months} months")

        materialized_candidates, materialize_stats = self._materialize_candidate_papers_for_recommendation(filtered_candidates)
        logger.info(
            "Materialized candidate papers for user %s: total=%s reused=%s db_only=%s batch_embedded=%s batch_inserted=%s unresolved=%s",
            user_id,
            materialize_stats.get("total", 0),
            materialize_stats.get("reused_existing", 0),
            materialize_stats.get("db_only", 0),
            materialize_stats.get("batch_embedded", 0),
            materialize_stats.get("batch_inserted", 0),
            materialize_stats.get("unresolved", 0),
        )

        embedding_config = self.get_embedding_config()
        candidate_ids = [str(candidate.get("arxiv_id", "") or "").strip() for candidate in materialized_candidates]
        existing_candidate_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
            collection_name=self.collection_name,
            arxiv_ids=candidate_ids,
        )
        logger.info("Loaded %s stored candidate embeddings from Milvus for user %s", len(existing_candidate_embeddings), user_id)
        candidate_embedding_map = {
            str(item.get("arxiv_id", "") or "").strip(): item.get("vector", [])
            for item in existing_candidate_embeddings
            if item.get("arxiv_id") and item.get("vector")
        }
        reused_vector_count = 0
        for candidate in materialized_candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if arxiv_id in candidate_embedding_map:
                candidate["_stored_vector"] = candidate_embedding_map[arxiv_id]
                reused_vector_count += 1

        logger.info("Reused %s/%s candidate vectors from Milvus for user %s", reused_vector_count, len(materialized_candidates), user_id)

        scored_candidates = []
        recomputed_vector_count = 0
        missing_vector_count = 0
        for candidate in materialized_candidates:
            scored_candidate = self._build_candidate_score(
                candidate=candidate,
                liked_category_freq=liked_category_freq,
                user_vector=user_vector,
                interest_clusters=interest_clusters,
                disliked_vector=disliked_vector,
                embedding_config=embedding_config,
            )
            scored_candidates.append(scored_candidate)
            embedding_source = str(scored_candidate.get("_embedding_source", "") or "")
            if embedding_source == "recomputed":
                recomputed_vector_count += 1
            elif embedding_source == "missing":
                missing_vector_count += 1

        logger.info(
            "Candidate embedding summary for user %s: total=%s reused=%s recomputed=%s missing=%s",
            user_id,
            len(materialized_candidates),
            reused_vector_count,
            recomputed_vector_count,
            missing_vector_count,
        )

        selected = self._select_diverse_candidates(
            scored_candidates,
            top_n,
            interest_clusters=interest_clusters,
        )

        logger.info(
            "Recommendation finished for user %s: scored=%s selected=%s reused=%s recomputed=%s missing=%s",
            user_id,
            len(scored_candidates),
            len(selected),
            reused_vector_count,
            recomputed_vector_count,
            missing_vector_count,
        )

        return {
            "status": "success",
            "message": f"Generated {len(selected)} recommendations",
            "total_found": len(scored_candidates),
            "interest_profile_mode": user_vector_data.get("profile_mode", "mean"),
            "interest_cluster_count": user_vector_data.get("cluster_count", 0),
            "recall_mode": recall_mode,
            "recommendations": selected,
        }
