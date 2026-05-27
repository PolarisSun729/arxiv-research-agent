import ast
import json
import logging
import math
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from services.arxiv_oai_service import ArxivOaiDatabaseService
from services.database_service import DatabaseService
from services.embedding_service import EmbeddingConfig, EmbeddingService
from services.arxiv_search_service import ArxivSearchService
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class RecommendationService:
    ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS = 8.0
    MIN_LIKED_PAPERS_FOR_CLUSTERING = 4
    MAX_INTEREST_CLUSTERS = 4
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
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        oai_db_service: Optional[ArxivOaiDatabaseService] = None,
        collection_name: str = "arxiv_paper_embeddings",
    ):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.get_embedding_config = get_embedding_config
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
        self.oai_db_service = oai_db_service or ArxivOaiDatabaseService()
        self.collection_name = collection_name
        self._arxiv_backfill_lock = threading.Lock()
        self._arxiv_backfill_next_allowed_time = 0.0

    def generate_user_interest_vector(self, user_id: str, negative_weight: float = 0.3) -> Dict[str, Any]:
        liked_ids = self.db_service.get_liked_papers(user_id=user_id)
        disliked_ids = self.db_service.get_disliked_papers(user_id=user_id)

        if not liked_ids:
            raise HTTPException(status_code=400, detail="No liked papers found for user")

        logger.info(
            "Generating interest vector for user %s with %s liked papers and %s disliked papers",
            user_id,
            len(liked_ids),
            len(disliked_ids),
        )

        config = self.get_embedding_config()
        liked_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
            collection_name=self.collection_name,
            arxiv_ids=liked_ids,
        )
        disliked_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
            collection_name=self.collection_name,
            arxiv_ids=disliked_ids,
        )

        liked_records = self._hydrate_vectors_with_metadata(
            requested_ids=liked_ids,
            milvus_embeddings=liked_embeddings,
            config=config,
            label="liked",
        )
        disliked_records = self._hydrate_vectors_with_metadata(
            requested_ids=disliked_ids,
            milvus_embeddings=disliked_embeddings,
            config=config,
            label="disliked",
        )

        if not liked_records:
            raise HTTPException(status_code=500, detail="No reusable embeddings found for liked papers")

        liked_mean = self._mean_vector([record["vector"] for record in liked_records])
        disliked_mean = self._mean_vector([record["vector"] for record in disliked_records]) if disliked_records else []
        if disliked_mean:
            raw_vector = [
                liked_value - negative_weight * disliked_value
                for liked_value, disliked_value in zip(liked_mean, disliked_mean)
            ]
        else:
            raw_vector = liked_mean

        fallback_interest_vector = self._normalize_vector(raw_vector)
        interest_clusters: List[Dict[str, Any]] = []
        profile_mode = "mean"
        if len(liked_records) >= self.MIN_LIKED_PAPERS_FOR_CLUSTERING:
            try:
                interest_clusters = self._cluster_interest_vectors(liked_records)
                if interest_clusters:
                    profile_mode = "clustered"
                    cluster_summary = [
                        {
                            "cluster_id": cluster.get("cluster_id"),
                            "paper_count": cluster.get("paper_count", 0),
                            "paper_ids": cluster.get("paper_ids", []),
                        }
                        for cluster in interest_clusters
                    ]
                    logger.info(
                        "User %s liked papers clustered into %s interest clusters: %s",
                        user_id,
                        len(interest_clusters),
                        cluster_summary,
                    )
                else:
                    logger.info(
                        "User %s did not produce stable interest clusters; using mean fallback",
                        user_id,
                    )
            except Exception as exc:  # pragma: no cover - clustering should be deterministic but safe to fallback
                logger.warning("Failed to cluster liked papers for user %s, falling back to mean vector: %s", user_id, exc)
                interest_clusters = []

        vector_dimension = len(fallback_interest_vector)
        milvus_used_count = sum(1 for record in liked_records + disliked_records if record.get("source") == "milvus")
        fallback_used_count = sum(1 for record in liked_records + disliked_records if record.get("source") == "fallback")
        used_count = len(liked_records) + len(disliked_records)
        unresolved_count = len(liked_ids) + len(disliked_ids) - used_count

        success = self.db_service.save_user_interest_vector(
            user_id=user_id,
            vector_data=fallback_interest_vector,
            paper_count=used_count,
            embedding_model=config.model_name,
            vector_dimension=vector_dimension,
            cluster_count=len(interest_clusters),
            profile_mode=profile_mode,
            interest_clusters=interest_clusters,
            disliked_vector_data=self._normalize_vector(disliked_mean) if disliked_mean else None,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save interest vector")

        return {
            "status": "success",
            "message": (
                "User interest vector generated successfully "
                f"using {milvus_used_count} Milvus vectors, "
                f"{fallback_used_count} fallback vectors, "
                f"and {unresolved_count} unresolved papers"
            ),
            "paper_count": used_count,
            "used_count": used_count,
            "milvus_used_count": milvus_used_count,
            "fallback_used_count": fallback_used_count,
            "unresolved_count": unresolved_count,
            "liked_count": len(liked_records),
            "disliked_count": len(disliked_records),
            "liked_milvus_count": sum(1 for record in liked_records if record.get("source") == "milvus"),
            "disliked_milvus_count": sum(1 for record in disliked_records if record.get("source") == "milvus"),
            "liked_fallback_count": sum(1 for record in liked_records if record.get("source") == "fallback"),
            "disliked_fallback_count": sum(1 for record in disliked_records if record.get("source") == "fallback"),
            "liked_unresolved_count": len(liked_ids) - len(liked_records),
            "disliked_unresolved_count": len(disliked_ids) - len(disliked_records),
            "vector_dimension": vector_dimension,
            "embedding_model": config.model_name,
            "cluster_count": len(interest_clusters),
            "profile_mode": profile_mode,
            "cluster_summary": [
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "paper_count": cluster.get("paper_count", 0),
                    "paper_ids": cluster.get("paper_ids", []),
                }
                for cluster in interest_clusters
            ],
        }

    def recommend_papers(self, user_id: str, top_n: int = 10, max_age_months: int = 6) -> Dict[str, Any]:
        user_vector_data = self._get_or_refresh_interest_vector(user_id)
        user_vector = user_vector_data["vector_data"]
        interest_clusters = user_vector_data.get("interest_clusters", []) or []
        disliked_vector = user_vector_data.get("disliked_vector_data")

        logger.info(
            "Starting paper recommendation for user %s with top_n=%s max_age_months=%s",
            user_id,
            top_n,
            max_age_months,
        )

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
            logger.info(
                "Fetched %s recent OAI DB candidates for user %s before deduplication",
                len(candidates),
                user_id,
            )

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
                raise HTTPException(
                    status_code=400,
                    detail=f"No papers found for recommendation within the last {max_age_months} months",
                )

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
        logger.info(
            "Loaded %s stored candidate embeddings from Milvus for user %s",
            len(existing_candidate_embeddings),
            user_id,
        )
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

        logger.info(
            "Reused %s/%s candidate vectors from Milvus for user %s",
            reused_vector_count,
            len(materialized_candidates),
            user_id,
        )

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

    def record_user_paper_preference(
        self,
        user_id: str,
        arxiv_id: str,
        liked: bool,
        paper_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ensure the paper exists in the paper store and vector store, then record the user preference.
        """
        normalized_arxiv_id = str(arxiv_id or "").strip()
        if not normalized_arxiv_id:
            raise HTTPException(status_code=400, detail="arxiv_id is required")

        paper = self._ensure_paper_materialized(normalized_arxiv_id, paper_payload=paper_payload)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {normalized_arxiv_id} could not be materialized")

        if liked:
            success = self.db_service.add_liked_paper(user_id=user_id, arxiv_id=normalized_arxiv_id)
            action = "liked"
        else:
            success = self.db_service.add_disliked_paper(user_id=user_id, arxiv_id=normalized_arxiv_id)
            action = "disliked"

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to add paper to {action} list")

        return {
            "status": "success",
            "message": f"Paper added to {action} list and materialized into paper store",
            "arxiv_id": normalized_arxiv_id,
            "paper": paper,
        }

    def _get_or_refresh_interest_vector(self, user_id: str) -> Dict[str, Any]:
        vector_data = self.db_service.get_user_interest_vector(user_id=user_id)
        latest_preference_ts = self.db_service.get_latest_user_preference_timestamp(user_id=user_id)

        if vector_data and latest_preference_ts:
            if self._is_vector_stale(vector_data.get("updated_at"), latest_preference_ts):
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

    def _materialize_candidate_papers_for_recommendation(
        self,
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        stats = {
            "total": len(candidates),
            "reused_existing": 0,
            "db_only": 0,
            "batch_embedded": 0,
            "batch_inserted": 0,
            "unresolved": 0,
        }
        materialized_by_id: Dict[str, Dict[str, Any]] = {}
        batch_jobs: List[Dict[str, Any]] = []
        embedding_config = self.get_embedding_config()
        candidate_ids = [str(candidate.get("arxiv_id", "") or "").strip() for candidate in candidates if str(candidate.get("arxiv_id", "") or "").strip()]
        candidate_embedding_rows = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
            collection_name=self.collection_name,
            arxiv_ids=candidate_ids,
        )
        candidate_embedding_map = {
            str(item.get("arxiv_id", "") or "").strip(): item
            for item in candidate_embedding_rows
            if item.get("arxiv_id")
        }

        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if not arxiv_id:
                stats["unresolved"] += 1
                continue

            try:
                existing_paper = self.db_service.get_paper(arxiv_id)
                existing_embedding = candidate_embedding_map.get(arxiv_id)
            except Exception as exc:  # pragma: no cover - storage/runtime dependent fallback
                logger.warning("Failed to inspect candidate paper %s before materialization: %s", arxiv_id, exc)
                existing_paper = None
                existing_embedding = None

            if existing_paper and existing_paper.get("embedding_id") and existing_embedding:
                materialized_by_id[arxiv_id] = existing_paper
                stats["reused_existing"] += 1
                continue

            if existing_paper and existing_embedding and not existing_paper.get("embedding_id"):
                updated = self.db_service.update_paper_embedding(
                    arxiv_id=arxiv_id,
                    embedding_id=int(existing_embedding.get("embedding_id") or 0),
                    embedding_model=str(existing_embedding.get("embedding_model") or existing_paper.get("embedding_model") or embedding_config.model_name),
                )
                refreshed = self.db_service.get_paper(arxiv_id) if updated else None
                materialized_by_id[arxiv_id] = refreshed or {**existing_paper, "embedding_id": existing_embedding.get("embedding_id")}
                stats["db_only"] += 1
                continue

            if existing_embedding and not existing_paper:
                stored = {
                    "arxiv_id": str(candidate.get("arxiv_id", "") or "").strip(),
                    "title": str(candidate.get("title", "") or "").strip(),
                    "authors": candidate.get("authors", []),
                    "abstract": str(candidate.get("abstract", "") or "").strip(),
                    "categories": candidate.get("categories", []),
                    "published_date": str(candidate.get("published_date", "") or "").strip(),
                    "url": str(candidate.get("url", "") or candidate.get("abs_url", "") or candidate.get("pdf_url", "") or "").strip(),
                    "embedding_id": str(existing_embedding.get("embedding_id") or ""),
                    "embedding_model": str(existing_embedding.get("embedding_model") or embedding_config.model_name),
                }
                if self.db_service.add_paper(stored):
                    materialized_by_id[arxiv_id] = self.db_service.get_paper(arxiv_id) or stored
                    stats["db_only"] += 1
                else:
                    stats["unresolved"] += 1
                continue

            normalized_paper = self._normalize_paper_record(candidate, arxiv_id)
            text_to_embed = self.embedding_service.build_paper_embedding_text(
                normalized_paper["title"],
                normalized_paper["abstract"],
            )
            if not text_to_embed:
                stats["unresolved"] += 1
                continue

            try:
                embedding = self.embedding_service.create_single_embedding(
                    text_to_embed,
                    provider=embedding_config.provider,
                    model=embedding_config.model_name,
                    api_key=embedding_config.api_key,
                    base_url=embedding_config.base_url,
                    dimension=embedding_config.dimension,
                )
            except Exception as exc:  # pragma: no cover - embedding/runtime dependent fallback
                logger.warning("Failed to embed candidate paper %s for recommendation: %s", arxiv_id, exc)
                stats["unresolved"] += 1
                continue

            batch_jobs.append(
                {
                    "arxiv_id": arxiv_id,
                    "normalized_paper": normalized_paper,
                    "embedding": [float(value) for value in embedding],
                }
            )

        if batch_jobs:
            try:
                batch_insert_payload = [
                    {
                        "embedding": job["embedding"],
                        "metadata": {
                            "content": job["normalized_paper"]["abstract"],
                            "arxiv_id": job["normalized_paper"]["arxiv_id"],
                            "title": job["normalized_paper"]["title"],
                            "authors": job["normalized_paper"]["authors"],
                            "categories": job["normalized_paper"]["categories"],
                            "published_date": job["normalized_paper"]["published_date"],
                            "url": job["normalized_paper"]["url"],
                            "embedding_model": job["normalized_paper"]["embedding_model"],
                        },
                    }
                    for job in batch_jobs
                ]
                embedding_count = self.vector_store_service.insert_embeddings(self.collection_name, batch_insert_payload)
                stats["batch_embedded"] = len(batch_jobs)
                stats["batch_inserted"] = int(embedding_count)
                for job in batch_jobs:
                    stored = {
                        "arxiv_id": job["normalized_paper"]["arxiv_id"],
                        "title": job["normalized_paper"]["title"],
                        "authors": job["normalized_paper"]["authors"],
                        "abstract": job["normalized_paper"]["abstract"],
                        "categories": job["normalized_paper"]["categories"],
                        "published_date": job["normalized_paper"]["published_date"],
                        "url": job["normalized_paper"]["url"],
                        "embedding_id": "",
                        "embedding_model": job["normalized_paper"]["embedding_model"],
                    }
                    if self.db_service.add_paper(stored):
                        materialized_by_id[job["arxiv_id"]] = self.db_service.get_paper(job["arxiv_id"]) or stored
                    else:
                        stats["unresolved"] += 1
            except Exception as exc:  # pragma: no cover - storage/runtime dependent fallback
                logger.warning("Failed batch materialization for recommendation candidates: %s", exc)
                stats["unresolved"] += len(batch_jobs)

        ordered_candidates = []
        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if not arxiv_id or arxiv_id not in materialized_by_id:
                continue
            merged_candidate = {**candidate, **materialized_by_id[arxiv_id]}
            for key, value in candidate.items():
                if key.startswith("recall_") or key in {"similarity_score", "score", "distance", "metadata"}:
                    merged_candidate[key] = value
            ordered_candidates.append(merged_candidate)
        return ordered_candidates, stats

    def _ensure_paper_materialized(
        self,
        arxiv_id: str,
        paper_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        existing_paper = self.db_service.get_paper(arxiv_id)
        existing_embedding = self._get_existing_paper_embedding(arxiv_id)

        if existing_paper and existing_paper.get("embedding_id") and existing_embedding:
            return existing_paper

        source_paper = paper_payload or existing_paper or existing_embedding
        if not source_paper:
            raise HTTPException(
                status_code=400,
                detail=f"Paper {arxiv_id} metadata is required from the client to materialize the record",
            )

        normalized_paper = self._normalize_paper_record(source_paper, arxiv_id)
        paper_payload = {
            "arxiv_id": normalized_paper["arxiv_id"],
            "title": normalized_paper["title"],
            "authors": normalized_paper["authors"],
            "abstract": normalized_paper["abstract"],
            "categories": normalized_paper["categories"],
            "published_date": normalized_paper["published_date"],
            "url": normalized_paper["url"],
            "embedding_model": normalized_paper["embedding_model"],
        }

        if existing_paper and existing_embedding and not existing_paper.get("embedding_id"):
            updated = self.db_service.update_paper_embedding(
                arxiv_id=arxiv_id,
                embedding_id=int(existing_embedding.get("embedding_id") or 0),
                embedding_model=str(existing_embedding.get("embedding_model") or normalized_paper["embedding_model"]),
            )
            if updated:
                refreshed = self.db_service.get_paper(arxiv_id)
                if refreshed:
                    return refreshed
            return {**existing_paper, "embedding_id": existing_embedding.get("embedding_id")}

        if existing_paper and existing_paper.get("embedding_id") and not existing_embedding:
            embedding_id = self._insert_paper_embedding(normalized_paper)
            self.db_service.update_paper_embedding(
                arxiv_id=arxiv_id,
                embedding_id=embedding_id,
                embedding_model=normalized_paper["embedding_model"],
            )
            refreshed = self.db_service.get_paper(arxiv_id)
            return refreshed or {**existing_paper, "embedding_id": embedding_id}

        if existing_embedding and not existing_paper:
            stored = dict(paper_payload)
            stored["embedding_id"] = str(existing_embedding.get("embedding_id") or "")
            success = self.db_service.add_paper(stored)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to store paper {arxiv_id}")
            refreshed = self.db_service.get_paper(arxiv_id)
            return refreshed or stored

        embedding_id = self._insert_paper_embedding(normalized_paper)
        stored = dict(paper_payload)
        stored["embedding_id"] = str(embedding_id)
        success = self.db_service.add_paper(stored)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to store paper {arxiv_id}")

        refreshed = self.db_service.get_paper(arxiv_id)
        return refreshed or stored

    def _materialize_paper_from_source(self, source_paper: Dict[str, Any], fallback_arxiv_id: str) -> Dict[str, Any]:
        normalized_paper = self._normalize_paper_record(source_paper, fallback_arxiv_id)
        embedding_id = self._insert_paper_embedding(normalized_paper)
        stored = {
            "arxiv_id": normalized_paper["arxiv_id"],
            "title": normalized_paper["title"],
            "authors": normalized_paper["authors"],
            "abstract": normalized_paper["abstract"],
            "categories": normalized_paper["categories"],
            "published_date": normalized_paper["published_date"],
            "url": normalized_paper["url"],
            "embedding_id": str(embedding_id),
            "embedding_model": normalized_paper["embedding_model"],
        }
        success = self.db_service.add_paper(stored)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to store paper {normalized_paper['arxiv_id']}")

        refreshed = self.db_service.get_paper(normalized_paper["arxiv_id"])
        return refreshed or stored

    def _fetch_paper_from_arxiv_with_rate_limit(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        wait_seconds = 0.0
        with self._arxiv_backfill_lock:
            now = time.monotonic()
            if now < self._arxiv_backfill_next_allowed_time:
                wait_seconds = self._arxiv_backfill_next_allowed_time - now
            self._arxiv_backfill_next_allowed_time = max(self._arxiv_backfill_next_allowed_time, now) + self.ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS

        if wait_seconds > 0:
            logger.info("Waiting %.2f seconds before fetching arXiv paper %s", wait_seconds, arxiv_id)
            time.sleep(wait_seconds)

        arxiv_service = self.arxiv_service_factory()
        search_result = arxiv_service.search_papers(
            id_list=[arxiv_id],
            max_results=1,
            submitted_days_ago=None,
        )
        papers = search_result.get("papers", []) if isinstance(search_result, dict) else []
        return papers[0] if papers else None

    def _normalize_paper_record(self, paper: Dict[str, Any], fallback_arxiv_id: str) -> Dict[str, Any]:
        authors = paper.get("authors", "")
        categories = paper.get("categories", "")
        if isinstance(authors, (list, tuple)):
            authors_value = ", ".join([str(item).strip() for item in authors if str(item).strip()])
        else:
            authors_value = str(authors or "").strip()
        if isinstance(categories, (list, tuple)):
            categories_value = ", ".join([str(item).strip() for item in categories if str(item).strip()])
        else:
            categories_value = str(categories or "").strip()

        title = str(paper.get("title", "") or "").strip()
        abstract = str(paper.get("abstract", "") or paper.get("summary", "") or "").strip()
        published_date = str(
            paper.get("published_date")
            or paper.get("published")
            or paper.get("updated")
            or paper.get("update_date")
            or paper.get("publishedAt")
            or ""
        ).strip()
        url = str(paper.get("url") or paper.get("abs_url") or paper.get("absUrl") or paper.get("pdf_url") or paper.get("pdfUrl") or "").strip()
        arxiv_identifier = str(paper.get("arxiv_id") or paper.get("id") or fallback_arxiv_id or "").strip()
        if arxiv_identifier.startswith("http"):
            arxiv_identifier = arxiv_identifier.rsplit("/", 1)[-1]

        embedding_config = self.get_embedding_config()
        return {
            "arxiv_id": arxiv_identifier,
            "title": title,
            "authors": authors_value,
            "abstract": abstract,
            "categories": categories_value,
            "published_date": published_date,
            "url": url,
            "embedding_model": embedding_config.model_name,
        }

    def _insert_paper_embedding(self, normalized_paper: Dict[str, Any]) -> int:
        embedding_id, _ = self._build_and_insert_paper_embedding(normalized_paper)
        return embedding_id

    def _build_and_insert_paper_embedding(self, normalized_paper: Dict[str, Any]) -> tuple[int, List[float]]:
        embedding_config = self.get_embedding_config()
        text_to_embed = self.embedding_service.build_paper_embedding_text(
            normalized_paper["title"],
            normalized_paper["abstract"],
        )
        if not text_to_embed:
            raise HTTPException(status_code=400, detail=f"Paper {normalized_paper['arxiv_id']} has no text to embed")

        embedding = self.embedding_service.create_single_embedding(
            text_to_embed,
            provider=embedding_config.provider,
            model=embedding_config.model_name,
            api_key=embedding_config.api_key,
            base_url=embedding_config.base_url,
            dimension=embedding_config.dimension,
        )
        embedding_vector = [float(value) for value in embedding]
        metadata = {
            "content": normalized_paper["abstract"],
            "arxiv_id": normalized_paper["arxiv_id"],
            "title": normalized_paper["title"],
            "authors": normalized_paper["authors"],
            "categories": normalized_paper["categories"],
            "published_date": normalized_paper["published_date"],
            "url": normalized_paper["url"],
            "embedding_model": embedding_config.model_name,
        }
        embedding_id = self.vector_store_service.insert_single_embedding(self.collection_name, embedding_vector, metadata)
        return embedding_id, embedding_vector

    def _get_existing_paper_embedding(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
            collection_name=self.collection_name,
            arxiv_ids=[arxiv_id],
        )
        return embeddings[0] if embeddings else None

    def _is_vector_stale(self, vector_updated_at: Optional[str], latest_preference_ts: Optional[str]) -> bool:
        if not vector_updated_at or not latest_preference_ts:
            return False

        vector_dt = self._parse_datetime(vector_updated_at)
        preference_dt = self._parse_datetime(latest_preference_ts)
        if not vector_dt or not preference_dt:
            return False
        return vector_dt < preference_dt

    def _mean_vector(self, vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return []
        dimension = len(vectors[0])
        return [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(dimension)
        ]

    def _hydrate_vectors_with_metadata(
        self,
        requested_ids: List[str],
        milvus_embeddings: List[Dict[str, Any]],
        config: EmbeddingConfig,
        label: str,
    ) -> List[Dict[str, Any]]:
        milvus_map = {
            str(item.get("arxiv_id", "")).strip(): item.get("vector", [])
            for item in milvus_embeddings
            if item.get("arxiv_id") and item.get("vector")
        }
        missing_ids = [arxiv_id for arxiv_id in requested_ids if arxiv_id not in milvus_map]
        vector_records: Dict[str, Dict[str, Any]] = {
            arxiv_id: {
                "arxiv_id": arxiv_id,
                "vector": milvus_map[arxiv_id],
                "source": "milvus",
            }
            for arxiv_id in requested_ids
            if arxiv_id in milvus_map
        }

        if missing_ids:
            logger.warning(
                "Missing paper embeddings for %s papers in collection %s, attempting OAI DB backfill: %s",
                label,
                self.collection_name,
                ", ".join(sorted(missing_ids)),
            )

        recovered_vectors, recovered_ids, _ = self._backfill_missing_vectors_from_arxiv(missing_ids, label)
        for arxiv_id, vector in recovered_vectors.items():
            vector_records[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "vector": vector,
                "source": "milvus",
            }

        remaining_missing = [arxiv_id for arxiv_id in missing_ids if arxiv_id not in recovered_vectors]
        if remaining_missing:
            fallback_papers = [
                paper for paper in (self.db_service.get_paper(arxiv_id) for arxiv_id in remaining_missing)
                if paper
            ]
            embedded_fallbacks = self._embed_papers(fallback_papers, config)
            fallback_map = {
                item["arxiv_id"]: item["vector"]
                for item in embedded_fallbacks
                if item.get("arxiv_id") and item.get("vector")
            }
            fallback_ids = [arxiv_id for arxiv_id in remaining_missing if arxiv_id in fallback_map]
            for arxiv_id in fallback_ids:
                vector_records[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "vector": fallback_map[arxiv_id],
                    "source": "fallback",
                }
            unresolved_ids = [arxiv_id for arxiv_id in remaining_missing if arxiv_id not in fallback_map]

            if unresolved_ids:
                logger.warning(
                    "Unable to rebuild embeddings for %s papers from SQLite metadata: %s",
                    label,
                    ", ".join(sorted(unresolved_ids)),
                )

        ordered_records = [vector_records[arxiv_id] for arxiv_id in requested_ids if arxiv_id in vector_records]
        return ordered_records

    def _hydrate_vectors(
        self,
        requested_ids: List[str],
        milvus_embeddings: List[Dict[str, Any]],
        config: EmbeddingConfig,
        label: str,
    ) -> tuple[List[List[float]], List[str], List[str], List[str]]:
        records = self._hydrate_vectors_with_metadata(
            requested_ids=requested_ids,
            milvus_embeddings=milvus_embeddings,
            config=config,
            label=label,
        )
        vectors = [record["vector"] for record in records]
        milvus_ids = [record["arxiv_id"] for record in records if record.get("source") == "milvus"]
        fallback_ids = [record["arxiv_id"] for record in records if record.get("source") == "fallback"]
        unresolved_ids = [arxiv_id for arxiv_id in requested_ids if arxiv_id not in {record["arxiv_id"] for record in records}]
        return vectors, milvus_ids, fallback_ids, unresolved_ids

    def _backfill_missing_vectors_from_arxiv(
        self,
        missing_ids: List[str],
        label: str,
    ) -> tuple[Dict[str, List[float]], List[str], List[str]]:
        recovered_vectors: Dict[str, List[float]] = {}
        recovered_ids: List[str] = []
        unresolved_ids: List[str] = []

        for arxiv_id in missing_ids:
            existing_paper = self.db_service.get_paper(arxiv_id)
            source_paper = existing_paper
            if source_paper is None:
                try:
                    source_paper = self._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
                except Exception as exc:  # pragma: no cover - depends on network availability
                    logger.warning("Failed to fetch arXiv paper %s for %s backfill: %s", arxiv_id, label, exc)
                    source_paper = None

            if not source_paper:
                unresolved_ids.append(arxiv_id)
                continue

            try:
                normalized_paper = self._normalize_paper_record(source_paper, arxiv_id)
                embedding_id, embedding_vector = self._build_and_insert_paper_embedding(normalized_paper)
                stored = {
                    "arxiv_id": normalized_paper["arxiv_id"],
                    "title": normalized_paper["title"],
                    "authors": normalized_paper["authors"],
                    "abstract": normalized_paper["abstract"],
                    "categories": normalized_paper["categories"],
                    "published_date": normalized_paper["published_date"],
                    "url": normalized_paper["url"],
                    "embedding_id": str(embedding_id),
                    "embedding_model": normalized_paper["embedding_model"],
                }
                if not self.db_service.add_paper(stored):
                    raise HTTPException(status_code=500, detail=f"Failed to store paper {arxiv_id}")
                recovered_vectors[arxiv_id] = embedding_vector
                recovered_ids.append(arxiv_id)
                logger.info("Backfilled missing paper %s into paper store and vector store", arxiv_id)
            except Exception as exc:  # pragma: no cover - depends on embedding/runtime availability
                logger.warning("Failed to materialize missing paper %s for %s backfill: %s", arxiv_id, label, exc)
                unresolved_ids.append(arxiv_id)

        return recovered_vectors, recovered_ids, unresolved_ids

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def _cluster_interest_vectors(self, liked_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(liked_records) < self.MIN_LIKED_PAPERS_FOR_CLUSTERING:
            return []

        valid_records = [record for record in liked_records if record.get("vector")]
        vectors = [
            self._normalize_vector([float(value) for value in record["vector"]])
            for record in valid_records
        ]
        paper_ids = [str(record.get("arxiv_id", "") or "").strip() for record in valid_records]
        if len(vectors) < self.MIN_LIKED_PAPERS_FOR_CLUSTERING:
            return []

        cluster_count = min(self.MAX_INTEREST_CLUSTERS, max(2, len(vectors) // 3))
        cluster_count = min(cluster_count, len(vectors))
        if cluster_count < 2:
            return []

        centroids = self._initialize_cluster_centroids(vectors, cluster_count)
        assignments: List[int] = [-1] * len(vectors)

        for _ in range(20):
            updated_assignments = []
            for vector in vectors:
                best_cluster_index = max(
                    range(len(centroids)),
                    key=lambda index: self._cosine_similarity(vector, centroids[index]),
                )
                updated_assignments.append(best_cluster_index)

            if updated_assignments == assignments:
                break
            assignments = updated_assignments

            new_centroids: List[List[float]] = []
            for cluster_index in range(cluster_count):
                cluster_vectors = [
                    vectors[index]
                    for index, assignment in enumerate(assignments)
                    if assignment == cluster_index
                ]
                if not cluster_vectors:
                    logger.info("Interest clustering produced an empty cluster; using mean fallback")
                    return []
                centroid = self._normalize_vector(self._mean_vector(cluster_vectors))
                new_centroids.append(centroid)
            centroids = new_centroids

        cluster_members: Dict[int, List[int]] = {index: [] for index in range(cluster_count)}
        for vector_index, cluster_index in enumerate(assignments):
            if cluster_index not in cluster_members:
                cluster_members[cluster_index] = []
            cluster_members[cluster_index].append(vector_index)

        if any(not members for members in cluster_members.values()):
            logger.info("Interest clustering ended with an empty cluster; using mean fallback")
            return []

        clusters: List[Dict[str, Any]] = []
        for cluster_index, members in cluster_members.items():
            member_paper_ids = [paper_ids[index] for index in members if paper_ids[index]]
            cluster_vectors = [vectors[index] for index in members]
            clusters.append(
                {
                    "cluster_id": f"cluster_{cluster_index}",
                    "centroid_vector": self._normalize_vector(self._mean_vector(cluster_vectors)),
                    "paper_count": len(member_paper_ids),
                    "paper_ids": member_paper_ids,
                }
            )

        clusters.sort(
            key=lambda item: (
                -int(item.get("paper_count", 0) or 0),
                (item.get("paper_ids") or [""])[0],
            )
        )
        for index, cluster in enumerate(clusters):
            cluster["cluster_id"] = f"cluster_{index}"
        return clusters

    def _initialize_cluster_centroids(self, vectors: List[List[float]], cluster_count: int) -> List[List[float]]:
        if not vectors or cluster_count <= 0:
            return []

        centroids = [vectors[0]]
        chosen_indices = {0}
        while len(centroids) < cluster_count:
            remaining_indices = [index for index in range(len(vectors)) if index not in chosen_indices]
            if not remaining_indices:
                break

            candidate_index = max(
                remaining_indices,
                key=lambda index: min(
                    1.0 - self._cosine_similarity(vectors[index], centroid)
                    for centroid in centroids
                ),
            )
            centroids.append(vectors[candidate_index])
            chosen_indices.add(candidate_index)

        while len(centroids) < cluster_count:
            centroids.append(vectors[0])
        return centroids

    def _build_liked_category_frequency(self, liked_papers: List[Dict[str, Any]]) -> Counter:
        counter: Counter = Counter()
        for paper in liked_papers:
            for category in self._split_categories(paper.get("categories")):
                counter[category] += 1
        return counter

    def _deduplicate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        excluded_ids: List[str],
    ) -> List[Dict[str, Any]]:
        seen = set(excluded_ids)
        deduped = []
        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            deduped.append(candidate)
        return deduped

    def _build_candidate_score(
        self,
        candidate: Dict[str, Any],
        liked_category_freq: Counter,
        user_vector: Optional[List[float]] = None,
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
        disliked_vector: Optional[List[float]] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> Dict[str, Any]:
        semantic_score = float(candidate.get("similarity_score", candidate.get("score", 0.0)) or 0.0)
        best_matched_cluster_id = None
        best_matched_cluster_similarity = None
        cluster_similarities: List[Dict[str, Any]] = []
        disliked_penalty = 0.0
        embedding_source = "none"
        candidate_embedding: Optional[List[float]] = None
        if user_vector and embedding_config:
            stored_vector = candidate.get("_stored_vector") or []
            if stored_vector:
                candidate_embedding = [float(value) for value in stored_vector]
                embedding_source = "stored"
            else:
                text_to_embed = self.embedding_service.build_paper_embedding_text(
                    candidate.get("title", ""),
                    candidate.get("abstract", ""),
                )
                if text_to_embed:
                    candidate_embedding = [
                        float(value)
                        for value in self.embedding_service.create_single_embedding(
                            text_to_embed,
                            provider=embedding_config.provider,
                            model=embedding_config.model_name,
                            api_key=embedding_config.api_key,
                            base_url=embedding_config.base_url,
                            dimension=embedding_config.dimension,
                        )
                    ]
                    embedding_source = "recomputed"
                else:
                    embedding_source = "missing"
            if candidate_embedding:
                if interest_clusters:
                    cluster_similarities = [
                        {
                            "cluster_id": cluster.get("cluster_id"),
                            "similarity": self._cosine_similarity(
                                candidate_embedding,
                                [float(value) for value in cluster.get("centroid_vector", [])],
                            ),
                        }
                        for cluster in interest_clusters
                        if cluster.get("centroid_vector")
                    ]
                    if cluster_similarities:
                        best_cluster = max(
                            cluster_similarities,
                            key=lambda item: float(item.get("similarity", 0.0) or 0.0),
                        )
                        best_matched_cluster_id = best_cluster.get("cluster_id")
                        best_matched_cluster_similarity = float(best_cluster.get("similarity", 0.0) or 0.0)
                        semantic_score = best_matched_cluster_similarity
                    else:
                        semantic_score = self._cosine_similarity(user_vector, candidate_embedding)
                else:
                    semantic_score = self._cosine_similarity(user_vector, candidate_embedding)

                if interest_clusters and disliked_vector:
                    disliked_penalty = self._cosine_similarity(disliked_vector, candidate_embedding)
        categories = self._split_categories(candidate.get("categories"))
        category_score = self._calculate_category_score(categories, liked_category_freq)
        recency_score = self._calculate_recency_score(candidate.get("published_date"))

        base_score = (
            semantic_score * 0.65
            + category_score * 0.08
            + recency_score * 0.05
            - disliked_penalty * 0.15
        )

        return {
            **candidate,
            "similarity_score": semantic_score,
            "semantic_score": semantic_score,
            "relevance_score": base_score,
            "best_matched_cluster_id": best_matched_cluster_id,
            "best_matched_cluster_similarity": best_matched_cluster_similarity,
            "cluster_similarities": cluster_similarities,
            "disliked_penalty": disliked_penalty,
            "_embedding_source": embedding_source,
            "_candidate_embedding": candidate_embedding,
            "final_score": base_score,
            "score_breakdown": {
                "semantic_score": semantic_score,
                "category_score": category_score,
                "recency_score": recency_score,
                "disliked_penalty": disliked_penalty,
                "relevance_score": base_score,
                "diversity_score": 0.0,
            },
            "_candidate_categories": categories,
        }

    def _embed_papers(self, papers: List[Dict[str, Any]], config: EmbeddingConfig) -> List[Dict[str, Any]]:
        embedded: List[Dict[str, Any]] = []
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
            title = str(paper.get("title", "") or "").strip()
            abstract = str(paper.get("abstract", "") or "").strip()
            text_to_embed = self.embedding_service.build_paper_embedding_text(title, abstract)
            if not arxiv_id or not text_to_embed:
                continue
            embedding = self.embedding_service.create_single_embedding(
                text_to_embed,
                provider=config.provider,
                model=config.model_name,
                api_key=config.api_key,
                base_url=config.base_url,
                dimension=config.dimension,
            )
            embedded.append({
                "arxiv_id": arxiv_id,
                "vector": [float(value) for value in embedding],
            })
        return embedded

    def _fetch_recent_db_candidates(
        self,
        liked_category_freq: Counter,
        max_age_months: int,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        categories = self.RECOMMEND_CANDIDATE_CATEGORIES[:]

        logger.info(
            "Fetching recent OAI DB candidates with categories=%s, max_age_months=%s, max_results=%s",
            categories,
            max_age_months,
            max_results,
        )

        papers = self.oai_db_service.get_recent_papers(
            categories=categories,
            max_age_months=max_age_months,
            max_results=max_results,
        )
        candidates: List[Dict[str, Any]] = []
        for paper in papers:
            candidate = {
                "arxiv_id": str(paper.get("arxiv_id", "") or "").strip(),
                "title": str(paper.get("title", "") or "").strip(),
                "authors": paper.get("authors", []),
                "abstract": str(paper.get("abstract", "") or "").strip(),
                "categories": paper.get("categories", []),
                "published_date": str(
                    paper.get("created")
                    or paper.get("updated")
                    or paper.get("oai_datestamp")
                    or ""
                ).strip(),
                "url": str(paper.get("abs_url") or paper.get("pdf_url") or "").strip(),
                "score": 0.0,
                "abs_url": paper.get("abs_url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "primary_category": paper.get("primary_category", ""),
                "oai_datestamp": paper.get("oai_datestamp", ""),
                "fetched_at": paper.get("fetched_at", ""),
            }
            candidates.append(candidate)

        return candidates

    def _build_category_query(self, liked_category_freq: Counter, max_categories: int = 5) -> str:
        categories = self.RECOMMEND_CANDIDATE_CATEGORIES[:max_categories] if max_categories > 0 else self.RECOMMEND_CANDIDATE_CATEGORIES
        return " OR ".join(f"cat:{category}" for category in categories)

    def _select_diverse_candidates(
        self,
        scored_candidates: List[Dict[str, Any]],
        top_n: int,
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if top_n <= 0 or not scored_candidates:
            return []

        ranked_candidates = list(scored_candidates)
        relevance_scores = [
            float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0)
            for candidate in ranked_candidates
        ]
        min_relevance = min(relevance_scores) if relevance_scores else 0.0
        max_relevance = max(relevance_scores) if relevance_scores else 0.0

        def normalize_relevance(score: float) -> float:
            if max_relevance <= min_relevance:
                return 1.0
            normalized = (score - min_relevance) / (max_relevance - min_relevance)
            return max(0.0, min(1.0, normalized))

        def candidate_embedding(candidate: Dict[str, Any]) -> Optional[List[float]]:
            embedding = candidate.get("_candidate_embedding")
            if not embedding:
                return None
            return [float(value) for value in embedding]

        def candidate_cluster_id(candidate: Dict[str, Any]) -> Optional[str]:
            cluster_id = str(candidate.get("best_matched_cluster_id", "") or "").strip()
            if not cluster_id:
                cluster_id = str(candidate.get("recall_cluster_id", "") or "").strip()
            return cluster_id or None

        selected: List[Dict[str, Any]] = []
        selected_embeddings: List[List[float]] = []
        selected_cluster_counts: Counter = Counter()
        selected_category_counts: Counter = Counter()
        remaining = ranked_candidates[:]

        def annotate_selected_candidate(
            candidate: Dict[str, Any],
            selection_rank: int,
            relevance_score: float,
            diversity_score: float,
            semantic_diversity_score: Optional[float],
            cluster_diversity_score: Optional[float],
            category_diversity_score: Optional[float],
            diversity_reason: str,
            diversity_penalty_source: Optional[str] = None,
            diversity_penalty_value: Optional[float] = None,
        ) -> None:
            score_breakdown = dict(candidate.get("score_breakdown", {}))
            score_breakdown.update(
                {
                    "relevance_score": relevance_score,
                    "diversity_score": diversity_score,
                    "semantic_diversity_score": semantic_diversity_score,
                    "cluster_diversity_score": cluster_diversity_score,
                    "category_diversity_score": category_diversity_score,
                    "selection_score": candidate.get("final_score", 0.0),
                }
            )
            candidate["selection_rank"] = selection_rank
            candidate["relevance_score"] = relevance_score
            candidate["selection_score"] = candidate.get("final_score", 0.0)
            candidate["diversity_score"] = diversity_score
            candidate["score_breakdown"] = score_breakdown
            candidate["diversity_debug"] = {
                "diversity_reason": diversity_reason,
                "diversity_penalty_source": diversity_penalty_source,
                "diversity_penalty_value": diversity_penalty_value,
                "semantic_diversity_score": semantic_diversity_score,
                "cluster_diversity_score": cluster_diversity_score,
                "category_diversity_score": category_diversity_score,
            }

        def finalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
            candidate.pop("_candidate_embedding", None)
            candidate.pop("_candidate_categories", None)
            return candidate

        def compute_diversity(
            candidate: Dict[str, Any],
        ) -> tuple[float, Optional[float], Optional[float], Optional[float], str, Optional[str], Optional[float]]:
            embedding = candidate_embedding(candidate)
            semantic_diversity_score: Optional[float] = None
            semantic_similarity_penalty: Optional[float] = None
            if embedding and selected_embeddings:
                max_similarity = max(
                    self._cosine_similarity(embedding, selected_embedding)
                    for selected_embedding in selected_embeddings
                )
                semantic_similarity_penalty = max(0.0, min(1.0, max_similarity))
                semantic_diversity_score = 1.0 - semantic_similarity_penalty

            cluster_id = candidate_cluster_id(candidate)
            cluster_diversity_score: Optional[float] = None
            cluster_repeat_count: Optional[int] = None
            if cluster_id and interest_clusters:
                cluster_repeat_count = int(selected_cluster_counts.get(cluster_id, 0))
                cluster_diversity_score = max(0.0, 1.0 - min(cluster_repeat_count / 4.0, 1.0))

            categories = candidate.get("_candidate_categories", []) or []
            category_diversity_score: Optional[float] = None
            category_repeat_count: Optional[int] = None
            if not embedding and cluster_diversity_score is None and categories:
                category_repeat_count = max((int(selected_category_counts.get(category, 0)) for category in categories), default=0)
                category_diversity_score = max(0.0, 1.0 - min(category_repeat_count / 4.0, 1.0))

            components: List[tuple[float, float]] = []
            if semantic_diversity_score is not None:
                components.append((semantic_diversity_score, 0.7))
            if cluster_diversity_score is not None:
                components.append((cluster_diversity_score, 0.3))
            if not components and category_diversity_score is not None:
                components.append((category_diversity_score, 1.0))

            if components:
                total_weight = sum(weight for _, weight in components)
                diversity_score = sum(value * weight for value, weight in components) / total_weight
                diversity_reason = "semantic_cluster_mix" if len(components) > 1 else (
                    "semantic_guidance" if semantic_diversity_score is not None else "cluster_guidance"
                )
            else:
                diversity_score = 1.0
                diversity_reason = "no_previous_selection"

            if semantic_similarity_penalty is not None and semantic_similarity_penalty >= 0.75:
                diversity_reason = "semantic_repeat"
            elif cluster_repeat_count and cluster_repeat_count > 0:
                diversity_reason = "cluster_repeat"
            elif category_repeat_count and category_repeat_count > 0:
                diversity_reason = "category_repeat"

            return (
                diversity_score,
                semantic_diversity_score,
                cluster_diversity_score,
                category_diversity_score,
                diversity_reason,
                candidate_cluster_id(candidate),
                semantic_similarity_penalty,
            )

        first_candidate = max(
            remaining,
            key=lambda item: (
                normalize_relevance(float(item.get("relevance_score", item.get("final_score", 0.0)) or 0.0)),
                float(item.get("relevance_score", item.get("final_score", 0.0)) or 0.0),
                float(item.get("semantic_score", 0.0) or 0.0),
                str(item.get("arxiv_id", "") or ""),
            ),
        )
        remaining.remove(first_candidate)
        first_relevance = float(first_candidate.get("relevance_score", first_candidate.get("final_score", 0.0)) or 0.0)
        first_selection_score = normalize_relevance(first_relevance)
        first_candidate["final_score"] = first_selection_score
        annotate_selected_candidate(
            candidate=first_candidate,
            selection_rank=1,
            relevance_score=first_relevance,
            diversity_score=1.0,
            semantic_diversity_score=None,
            cluster_diversity_score=None,
            category_diversity_score=None,
            diversity_reason="seed",
        )
        selected_embeddings_candidate = candidate_embedding(first_candidate)
        if selected_embeddings_candidate:
            selected_embeddings.append(selected_embeddings_candidate)
        first_cluster_id = candidate_cluster_id(first_candidate)
        if first_cluster_id:
            selected_cluster_counts[first_cluster_id] += 1
        for category in first_candidate.get("_candidate_categories", []) or []:
            selected_category_counts[category] += 1
        selected.append(finalize_candidate(first_candidate))

        while remaining and len(selected) < top_n:
            best_candidate: Optional[Dict[str, Any]] = None
            best_selection_score = float("-inf")
            best_diversity_data = (1.0, None, None, None, "no_previous_selection", None, None)

            for candidate in remaining:
                relevance_score = float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0)
                normalized_relevance = normalize_relevance(relevance_score)
                (
                    diversity_score,
                    semantic_diversity_score,
                    cluster_diversity_score,
                    category_diversity_score,
                    diversity_reason,
                    diversity_cluster_id,
                    semantic_similarity_penalty,
                ) = compute_diversity(candidate)
                selection_score = normalized_relevance * 0.75 + diversity_score * 0.25
                if selection_score > best_selection_score:
                    best_candidate = candidate
                    best_selection_score = selection_score
                    best_diversity_data = (
                        diversity_score,
                        semantic_diversity_score,
                        cluster_diversity_score,
                        category_diversity_score,
                        diversity_reason,
                        diversity_cluster_id,
                        semantic_similarity_penalty,
                    )

            if best_candidate is None:
                break

            remaining.remove(best_candidate)
            relevance_score = float(best_candidate.get("relevance_score", best_candidate.get("final_score", 0.0)) or 0.0)
            diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, diversity_cluster_id, semantic_similarity_penalty = best_diversity_data
            best_candidate["final_score"] = best_selection_score
            diversity_penalty_value = semantic_similarity_penalty if semantic_similarity_penalty is not None else max(0.0, 1.0 - diversity_score)
            annotate_selected_candidate(
                candidate=best_candidate,
                selection_rank=len(selected) + 1,
                relevance_score=relevance_score,
                diversity_score=diversity_score,
                semantic_diversity_score=semantic_diversity_score,
                cluster_diversity_score=cluster_diversity_score,
                category_diversity_score=category_diversity_score,
                diversity_reason=diversity_reason,
                diversity_penalty_source="semantic" if semantic_similarity_penalty is not None else ("cluster" if cluster_diversity_score is not None else ("category" if category_diversity_score is not None else None)),
                diversity_penalty_value=diversity_penalty_value,
            )
            best_candidate_embedding = candidate_embedding(best_candidate)
            if best_candidate_embedding:
                selected_embeddings.append(best_candidate_embedding)
            best_cluster_id = diversity_cluster_id
            if best_cluster_id:
                selected_cluster_counts[best_cluster_id] += 1
            for category in best_candidate.get("_candidate_categories", []) or []:
                selected_category_counts[category] += 1
            selected.append(finalize_candidate(best_candidate))

        return selected

    def _fetch_cluster_recall_candidates(
        self,
        interest_clusters: List[Dict[str, Any]],
        excluded_ids: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        normalized_excluded_ids = [str(arxiv_id).strip() for arxiv_id in excluded_ids if str(arxiv_id).strip()]
        candidates_by_id: Dict[str, Dict[str, Any]] = {}

        for cluster_index, cluster in enumerate(interest_clusters):
            centroid_vector = cluster.get("centroid_vector") or []
            if not centroid_vector:
                continue

            cluster_id = str(cluster.get("cluster_id") or f"cluster_{cluster_index}").strip() or f"cluster_{cluster_index}"
            try:
                recalled_papers = self.vector_store_service.search_similar_papers(
                    collection_name=self.collection_name,
                    query_vector=[float(value) for value in centroid_vector],
                    top_k=max(1, int(top_k)),
                    filter_arxiv_ids=normalized_excluded_ids,
                )
            except Exception as exc:  # pragma: no cover - vector store runtime dependent
                logger.warning("Failed cluster recall for user cluster %s: %s", cluster_id, exc)
                continue

            for rank, paper in enumerate(recalled_papers, start=1):
                arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
                if not arxiv_id:
                    continue

                similarity = float(paper.get("similarity_score", paper.get("score", 0.0)) or 0.0)
                recall_hit = {
                    "cluster_id": cluster_id,
                    "similarity": similarity,
                    "rank": rank,
                }
                candidate = {
                    **paper,
                    "recall_source": "cluster_recall",
                    "recall_cluster_id": cluster_id,
                    "recall_cluster_similarity": similarity,
                    "recall_cluster_rank": rank,
                    "recall_cluster_hits": [recall_hit],
                }

                existing_candidate = candidates_by_id.get(arxiv_id)
                if existing_candidate is None:
                    candidates_by_id[arxiv_id] = candidate
                    continue

                existing_hits = list(existing_candidate.get("recall_cluster_hits", []))
                combined_hits = self._sort_recall_hits(existing_hits + [recall_hit])
                existing_candidate["recall_cluster_hits"] = combined_hits

                if self._is_better_cluster_recall_candidate(candidate, existing_candidate):
                    candidate["recall_cluster_hits"] = combined_hits
                    candidates_by_id[arxiv_id] = candidate

        candidates = list(candidates_by_id.values())
        candidates.sort(
            key=lambda item: (
                -float(item.get("recall_cluster_similarity", 0.0) or 0.0),
                int(item.get("recall_cluster_rank", 0) or 0),
                str(item.get("arxiv_id", "") or ""),
            )
        )
        return candidates

    def _sort_recall_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique_hits: List[Dict[str, Any]] = []
        seen = set()
        for hit in hits:
            cluster_id = str(hit.get("cluster_id", "") or "").strip()
            rank = int(hit.get("rank", 0) or 0)
            similarity = float(hit.get("similarity", 0.0) or 0.0)
            key = (cluster_id, rank, similarity)
            if key in seen:
                continue
            seen.add(key)
            unique_hits.append(
                {
                    "cluster_id": cluster_id or None,
                    "similarity": similarity,
                    "rank": rank,
                }
            )
        unique_hits.sort(
            key=lambda item: (
                -float(item.get("similarity", 0.0) or 0.0),
                int(item.get("rank", 0) or 0),
                str(item.get("cluster_id", "") or ""),
            )
        )
        return unique_hits

    def _is_better_cluster_recall_candidate(self, candidate: Dict[str, Any], existing_candidate: Dict[str, Any]) -> bool:
        candidate_key = (
            float(candidate.get("recall_cluster_similarity", 0.0) or 0.0),
            -int(candidate.get("recall_cluster_rank", 0) or 0),
        )
        existing_key = (
            float(existing_candidate.get("recall_cluster_similarity", 0.0) or 0.0),
            -int(existing_candidate.get("recall_cluster_rank", 0) or 0),
        )
        return candidate_key > existing_key

    def _calculate_category_score(self, categories: List[str], liked_category_freq: Counter) -> float:
        if not categories or not liked_category_freq:
            return 0.0
        total_frequency = sum(liked_category_freq.values())
        if total_frequency <= 0:
            return 0.0
        matched_frequency = sum(liked_category_freq.get(category, 0) for category in categories)
        return min(1.0, matched_frequency / total_frequency)

    def _calculate_recency_score(self, published_date: Any) -> float:
        parsed_date = self._parse_datetime(str(published_date or ""))
        if not parsed_date:
            return 0.0
        age_days = max(0, (datetime.now(timezone.utc) - parsed_date).days)
        return 1.0 / (1.0 + age_days / 365.0)

    def _cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        if not vector_a or not vector_b:
            return 0.0
        length = min(len(vector_a), len(vector_b))
        if length == 0:
            return 0.0
        dot_product = sum(vector_a[i] * vector_b[i] for i in range(length))
        norm_a = math.sqrt(sum(vector_a[i] * vector_a[i] for i in range(length)))
        norm_b = math.sqrt(sum(vector_b[i] * vector_b[i] for i in range(length)))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        cosine = dot_product / (norm_a * norm_b)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _is_within_max_age(self, published_date: Any, max_age_months: int) -> bool:
        parsed_date = self._parse_datetime(str(published_date or ""))
        if not parsed_date:
            return False

        if max_age_months <= 0:
            return True

        cutoff_days = max_age_months * 30
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=cutoff_days)
        return parsed_date >= cutoff_date

    def _calculate_diversity_score(self, categories: List[str], selected_category_counts: Counter) -> float:
        if not categories:
            return 1.0
        if not selected_category_counts:
            return 1.0
        max_overlap = max((selected_category_counts.get(category, 0) for category in categories), default=0)
        return 1.0 / (1.0 + max_overlap)

    def _split_categories(self, categories: Any) -> List[str]:
        values: List[Any] = []
        if isinstance(categories, list):
            values = categories
        elif isinstance(categories, tuple):
            values = list(categories)
        elif isinstance(categories, str):
            stripped = categories.strip()
            if stripped:
                if stripped.startswith("[") or stripped.startswith("("):
                    try:
                        parsed = ast.literal_eval(stripped)
                        if isinstance(parsed, (list, tuple)):
                            values = list(parsed)
                        else:
                            values = [parsed]
                    except (ValueError, SyntaxError):
                        values = []
                if not values:
                    normalized = stripped.replace(";", ",").replace("|", ",")
                    for token in normalized.replace("\n", " ").split():
                        if "," in token:
                            values.extend(token.split(","))
                        else:
                            values.append(token)
        unique_values = []
        seen = set()
        for category in values:
            normalized_category = str(category).strip()
            if normalized_category and normalized_category not in seen:
                seen.add(normalized_category)
                unique_values.append(normalized_category)
        return unique_values

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        if not value:
            return None

        text = value.strip()
        if not text:
            return None

        candidates = [
            text,
            text.replace("Z", "+00:00"),
            text[:10],
        ]

        for candidate in candidates:
            try:
                parsed = datetime.fromisoformat(candidate)
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                continue

        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text[:10], fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return None
