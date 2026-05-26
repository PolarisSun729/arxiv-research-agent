import ast
import logging
import math
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from services.database_service import DatabaseService
from services.embedding_service import EmbeddingConfig, EmbeddingService
from services.arxiv_search_service import ArxivSearchService
from services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)


class RecommendationService:
    ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS = 8.0
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
        collection_name: str = "arxiv_paper_embeddings",
    ):
        self.db_service = db_service
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.get_embedding_config = get_embedding_config
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
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

        liked_vectors, liked_milvus_ids, liked_fallback_ids, liked_unresolved_ids = self._hydrate_vectors(
            requested_ids=liked_ids,
            milvus_embeddings=liked_embeddings,
            config=config,
            label="liked",
        )
        disliked_vectors, disliked_milvus_ids, disliked_fallback_ids, disliked_unresolved_ids = self._hydrate_vectors(
            requested_ids=disliked_ids,
            milvus_embeddings=disliked_embeddings,
            config=config,
            label="disliked",
        )

        if not liked_vectors:
            raise HTTPException(status_code=500, detail="No reusable embeddings found for liked papers")

        liked_mean = self._mean_vector(liked_vectors)
        if disliked_vectors:
            disliked_mean = self._mean_vector(disliked_vectors)
            raw_vector = [
                liked_value - negative_weight * disliked_value
                for liked_value, disliked_value in zip(liked_mean, disliked_mean)
            ]
        else:
            raw_vector = liked_mean

        interest_vector = self._normalize_vector(raw_vector)
        vector_dimension = len(interest_vector)
        milvus_used_count = len(liked_milvus_ids) + len(disliked_milvus_ids)
        fallback_used_count = len(liked_fallback_ids) + len(disliked_fallback_ids)
        used_count = len(liked_vectors) + len(disliked_vectors)
        unresolved_count = len(liked_unresolved_ids) + len(disliked_unresolved_ids)

        success = self.db_service.save_user_interest_vector(
            user_id=user_id,
            vector_data=interest_vector,
            paper_count=used_count,
            embedding_model=config.model_name,
            vector_dimension=vector_dimension,
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
            "liked_count": len(liked_vectors),
            "disliked_count": len(disliked_vectors),
            "liked_milvus_count": len(liked_milvus_ids),
            "disliked_milvus_count": len(disliked_milvus_ids),
            "liked_fallback_count": len(liked_fallback_ids),
            "disliked_fallback_count": len(disliked_fallback_ids),
            "liked_unresolved_count": len(liked_unresolved_ids),
            "disliked_unresolved_count": len(disliked_unresolved_ids),
            "vector_dimension": vector_dimension,
            "embedding_model": config.model_name,
        }

    def recommend_papers(self, user_id: str, top_n: int = 10, max_age_months: int = 6) -> Dict[str, Any]:
        user_vector_data = self._get_or_refresh_interest_vector(user_id)
        user_vector = user_vector_data["vector_data"]

        preferences = self.db_service.get_user_preferences(user_id=user_id)
        liked_ids = preferences.get("liked_papers", [])
        disliked_ids = preferences.get("disliked_papers", [])
        excluded_ids = list(dict.fromkeys([*liked_ids, *disliked_ids]))

        liked_details = self.db_service.get_liked_papers_with_details(user_id=user_id)
        liked_category_freq = self._build_liked_category_frequency(liked_details)

        candidate_limit = max(top_n * 5, 50)
        candidates = self._fetch_recent_api_candidates(
            liked_category_freq=liked_category_freq,
            max_age_months=max_age_months,
            max_results=max(candidate_limit * 2, candidate_limit),
        )

        filtered_candidates = self._deduplicate_candidates(candidates, excluded_ids)
        if not filtered_candidates:
            raise HTTPException(
                status_code=400,
                detail=f"No papers found for recommendation within the last {max_age_months} months",
            )

        embedding_config = self.get_embedding_config()
        scored_candidates = []
        for candidate in filtered_candidates:
            scored_candidates.append(
                self._build_candidate_score(
                    candidate=candidate,
                    liked_category_freq=liked_category_freq,
                    user_vector=user_vector,
                    embedding_config=embedding_config,
                )
            )

        selected = self._select_diverse_candidates(scored_candidates, top_n)

        return {
            "status": "success",
            "message": f"Generated {len(selected)} recommendations",
            "total_found": len(scored_candidates),
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
        text_to_embed = f"{normalized_paper['title']}\n\nAbstract: {normalized_paper['abstract']}".strip()
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

    def _hydrate_vectors(
        self,
        requested_ids: List[str],
        milvus_embeddings: List[Dict[str, Any]],
        config: EmbeddingConfig,
        label: str,
    ) -> tuple[List[List[float]], List[str], List[str], List[str]]:
        milvus_map = {
            str(item.get("arxiv_id", "")).strip(): item.get("vector", [])
            for item in milvus_embeddings
            if item.get("arxiv_id") and item.get("vector")
        }
        milvus_used_ids = [arxiv_id for arxiv_id in requested_ids if arxiv_id in milvus_map]
        missing_ids = [arxiv_id for arxiv_id in requested_ids if arxiv_id not in milvus_map]

        if missing_ids:
            logger.warning(
                "Missing paper embeddings for %s papers in collection %s, attempting arXiv backfill: %s",
                label,
                self.collection_name,
                ", ".join(sorted(missing_ids)),
            )

        recovered_vectors, recovered_ids, _ = self._backfill_missing_vectors_from_arxiv(missing_ids, label)
        for arxiv_id, vector in recovered_vectors.items():
            milvus_map[arxiv_id] = vector
        for arxiv_id in recovered_ids:
            if arxiv_id not in milvus_used_ids:
                milvus_used_ids.append(arxiv_id)

        fallback_vectors: List[List[float]] = []
        fallback_ids: List[str] = []
        unresolved_ids: List[str] = []
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
            fallback_vectors = [fallback_map[arxiv_id] for arxiv_id in fallback_ids]
            unresolved_ids = [arxiv_id for arxiv_id in remaining_missing if arxiv_id not in fallback_map]

            if unresolved_ids:
                logger.warning(
                    "Unable to rebuild embeddings for %s papers from SQLite metadata: %s",
                    label,
                    ", ".join(sorted(unresolved_ids)),
                )

        vectors = [milvus_map[arxiv_id] for arxiv_id in milvus_used_ids] + fallback_vectors
        return vectors, milvus_used_ids, fallback_ids, unresolved_ids

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
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> Dict[str, Any]:
        semantic_score = float(candidate.get("similarity_score", candidate.get("score", 0.0)) or 0.0)
        if user_vector and embedding_config:
            text_to_embed = f"{candidate.get('title', '')}\n\nAbstract: {candidate.get('abstract', '')}".strip()
            if text_to_embed:
                candidate_embedding = self.embedding_service.create_single_embedding(
                    text_to_embed,
                    provider=embedding_config.provider,
                    model=embedding_config.model_name,
                    api_key=embedding_config.api_key,
                    base_url=embedding_config.base_url,
                    dimension=embedding_config.dimension,
                )
                semantic_score = self._cosine_similarity(user_vector, [float(value) for value in candidate_embedding])
        categories = self._split_categories(candidate.get("categories"))
        category_score = self._calculate_category_score(categories, liked_category_freq)
        recency_score = 0.0

        base_score = (
            semantic_score * 0.65
            + category_score * 0.08
            + recency_score * 0.0
        )

        return {
            **candidate,
            "similarity_score": semantic_score,
            "semantic_score": semantic_score,
            "final_score": base_score,
            "score_breakdown": {
                "semantic_score": semantic_score,
                "category_score": category_score,
                "recency_score": recency_score,
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
            text_to_embed = f"{title}\n\nAbstract: {abstract}".strip()
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

    def _fetch_recent_api_candidates(
        self,
        liked_category_freq: Counter,
        max_age_months: int,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        arxiv_service = self.arxiv_service_factory()
        category_query = self._build_category_query(liked_category_freq)

        logger.info(
            "Fetching recent API candidates with query=%s, max_age_months=%s, max_results=%s",
            category_query,
            max_age_months,
            max_results,
        )

        search_result = arxiv_service.search_papers(
            search_query=category_query,
            max_results=max_results,
            sort_by="submittedDate",
            sort_order="descending",
            submitted_days_ago=max_age_months * 30,
        )

        papers = search_result.get("papers", []) if isinstance(search_result, dict) else []
        candidates: List[Dict[str, Any]] = []
        for paper in papers:
            candidate = {
                "arxiv_id": paper.get("arxiv_id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "abstract": paper.get("summary", "") or paper.get("abstract", ""),
                "categories": paper.get("categories", []),
                "published_date": paper.get("published", "") or paper.get("published_date", ""),
                "url": paper.get("abs_url", "") or paper.get("url", ""),
                "score": 0.0,
            }
            candidates.append(candidate)

        return candidates

    def _build_category_query(self, liked_category_freq: Counter, max_categories: int = 5) -> str:
        categories = self.RECOMMEND_CANDIDATE_CATEGORIES[:max_categories] if max_categories > 0 else self.RECOMMEND_CANDIDATE_CATEGORIES
        return " OR ".join(f"cat:{category}" for category in categories)

    def _select_diverse_candidates(self, scored_candidates: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        selected = sorted(
            scored_candidates,
            key=lambda item: float(item.get("final_score", 0.0) or 0.0),
            reverse=True,
        )[:top_n]
        selected.sort(key=lambda item: float(item.get("final_score", 0.0) or 0.0), reverse=True)
        for candidate in selected:
            candidate.pop("_candidate_categories", None)
        return selected

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
