from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)


class InterestProfileService:
    def generate_user_interest_vector(
        self,
        user_id: str,
        negative_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        if negative_weight is None:
            negative_weight = self.RECOMMENDATION_CONFIG["negative_weight_default"]
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
        weak_interest_pool: Optional[Dict[str, Any]] = None
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
                interest_clusters, weak_interest_pool = self._cluster_interest_vectors(liked_records)
                if interest_clusters:
                    profile_mode = "clustered_with_weak_pool" if weak_interest_pool else "clustered"
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
                    if weak_interest_pool:
                        profile_mode = "mean_with_weak_pool"
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to cluster liked papers for user %s, falling back to mean vector: %s", user_id, exc)
                interest_clusters = []
                weak_interest_pool = None

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
            weak_interest_pool=weak_interest_pool,
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
            "weak_interest_pool": weak_interest_pool,
            "weak_interest_pool_count": weak_interest_pool.get("paper_count", 0) if weak_interest_pool else 0,
            "cluster_summary": [
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "paper_count": cluster.get("paper_count", 0),
                    "paper_ids": cluster.get("paper_ids", []),
                }
                for cluster in interest_clusters
            ],
        }

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
        return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]

    def _hydrate_vectors_with_metadata(
        self,
        requested_ids: List[str],
        milvus_embeddings: List[Dict[str, Any]],
        config: Any,
        label: str,
    ) -> List[Dict[str, Any]]:
        milvus_map = {
            str(item.get("arxiv_id", "")).strip(): item.get("vector", [])
            for item in milvus_embeddings
            if item.get("arxiv_id") and item.get("vector")
        }
        missing_ids = [arxiv_id for arxiv_id in requested_ids if arxiv_id not in milvus_map]
        vector_records: Dict[str, Dict[str, Any]] = {
            arxiv_id: {"arxiv_id": arxiv_id, "vector": milvus_map[arxiv_id], "source": "milvus"}
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

        recovered_vectors, _, _ = self._backfill_missing_vectors_from_arxiv(missing_ids, label)
        for arxiv_id, vector in recovered_vectors.items():
            vector_records[arxiv_id] = {"arxiv_id": arxiv_id, "vector": vector, "source": "milvus"}

        remaining_missing = [arxiv_id for arxiv_id in missing_ids if arxiv_id not in recovered_vectors]
        if remaining_missing:
            fallback_papers = [paper for paper in (self.db_service.get_paper(arxiv_id) for arxiv_id in remaining_missing) if paper]
            embedded_fallbacks = self._embed_papers(fallback_papers, config)
            fallback_map = {item["arxiv_id"]: item["vector"] for item in embedded_fallbacks if item.get("arxiv_id") and item.get("vector")}
            for arxiv_id in [arxiv_id for arxiv_id in remaining_missing if arxiv_id in fallback_map]:
                vector_records[arxiv_id] = {"arxiv_id": arxiv_id, "vector": fallback_map[arxiv_id], "source": "fallback"}

        ordered_records = [vector_records[arxiv_id] for arxiv_id in requested_ids if arxiv_id in vector_records]
        return ordered_records

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
                except Exception as exc:  # pragma: no cover
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
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to materialize missing paper %s for %s backfill: %s", arxiv_id, label, exc)
                unresolved_ids.append(arxiv_id)

        return recovered_vectors, recovered_ids, unresolved_ids

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def _cluster_interest_vectors(self, liked_records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if len(liked_records) < self.MIN_LIKED_PAPERS_FOR_CLUSTERING:
            return [], None

        valid_records = [record for record in liked_records if record.get("vector")]
        vectors = [self._normalize_vector([float(value) for value in record["vector"]]) for record in valid_records]
        paper_ids = [str(record.get("arxiv_id", "") or "").strip() for record in valid_records]
        if len(vectors) < self.MIN_LIKED_PAPERS_FOR_CLUSTERING or len(vectors) < 2:
            return [], None

        clustering_config = self.get_clustering_config() or {}
        min_cluster_size = max(2, int(clustering_config.get("hdbscan_min_cluster_size", 2) or 2))
        min_samples = max(1, int(clustering_config.get("hdbscan_min_samples", 1) or 1))
        metric = str(clustering_config.get("hdbscan_metric", "cosine") or "cosine").strip() or "cosine"
        cluster_selection_method = str(clustering_config.get("hdbscan_cluster_selection_method", "eom") or "eom").strip() or "eom"
        allow_single_cluster = bool(clustering_config.get("hdbscan_allow_single_cluster", True))

        try:
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=cluster_selection_method,
                allow_single_cluster=allow_single_cluster,
            )
            labels = clusterer.fit_predict(vectors)
        except Exception as exc:  # pragma: no cover
            logger.info("HDBSCAN interest clustering failed; using mean fallback: %s", exc)
            return [], None

        cluster_members: Dict[int, List[int]] = {}
        weak_member_indices: List[int] = []
        for vector_index, cluster_label in enumerate(labels):
            if cluster_label < 0:
                weak_member_indices.append(vector_index)
                continue
            cluster_members.setdefault(int(cluster_label), []).append(vector_index)

        weak_interest_pool: Optional[Dict[str, Any]] = None
        if weak_member_indices:
            weak_paper_ids = [paper_ids[index] for index in weak_member_indices if paper_ids[index]]
            weak_vectors = [vectors[index] for index in weak_member_indices]
            if weak_vectors:
                weak_interest_pool = {
                    "pool_id": "weak_interest_pool",
                    "paper_count": len(weak_paper_ids),
                    "paper_ids": weak_paper_ids,
                    "centroid_vector": self._normalize_vector(self._mean_vector(weak_vectors)),
                }

        if not cluster_members:
            if weak_interest_pool:
                logger.info("HDBSCAN produced only weak-interest noise points; keeping weak pool and using mean fallback")
                return [], weak_interest_pool
            logger.info("HDBSCAN produced no stable interest clusters; using mean fallback")
            return [], None

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

        clusters.sort(key=lambda item: (-int(item.get("paper_count", 0) or 0), (item.get("paper_ids") or [""])[0]))
        if len(clusters) > self.MAX_INTEREST_CLUSTERS:
            logger.info(
                "HDBSCAN produced %s interest clusters; keeping top %s by size",
                len(clusters),
                self.MAX_INTEREST_CLUSTERS,
            )
            clusters = clusters[: self.MAX_INTEREST_CLUSTERS]
        for index, cluster in enumerate(clusters):
            cluster["cluster_id"] = f"cluster_{index}"
        return clusters, weak_interest_pool
