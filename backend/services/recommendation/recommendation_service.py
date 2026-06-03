from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.memory import MemoryService
from services.storage.database_service import DatabaseService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.storage.vector_store_service import VectorStoreService
from utils.config import (
    get_memory_runtime_config,
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
        memory_service: Optional[MemoryService],
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        get_embedding_config: Callable[[], EmbeddingConfig],
        get_clustering_config: Optional[Callable[[], Dict[str, Any]]] = None,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        oai_db_service: Optional[ArxivOaiDatabaseService] = None,
        collection_name: str = "arxiv_paper_embeddings",
    ):
        self.db_service = db_service
        self.memory_service = memory_service or MemoryService(db_service=self.db_service)
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.get_embedding_config = get_embedding_config
        self.get_clustering_config = get_clustering_config or get_recommendation_clustering_runtime_config
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
        self.oai_db_service = oai_db_service or ArxivOaiDatabaseService()
        self.collection_name = collection_name
        self._arxiv_backfill_lock = threading.Lock()
        self._arxiv_backfill_next_allowed_time = 0.0
        self.memory_runtime_config = get_memory_runtime_config()

    def _memory_flag(self, key: str, default: Any = None) -> Any:
        return self.memory_runtime_config.get(key, default)

    @staticmethod
    def _normalize_text_terms(values: Any) -> List[str]:
        if not values:
            return []
        if isinstance(values, list):
            source = values
        else:
            source = [values]
        normalized: List[str] = []
        for value in source:
            text = str(value or "").strip().lower()
            if text:
                normalized.append(text)
        return normalized

    def _build_profile_signal_bundle(self, user_id: str) -> Dict[str, Any]:
        if not bool(self._memory_flag("enable_user_research_profile", False)):
            return {
                "profile": {},
                "actions": self.db_service.get_user_paper_action_map(user_id),
                "excluded_ids": [],
                "disabled": True,
            }
        profile = self.db_service.get_user_research_profile(user_id)
        actions = self.db_service.get_user_paper_action_map(user_id)
        excluded_ids = list(
            dict.fromkeys(
                [
                    *self._normalize_text_terms(actions.get("like", [])),
                    *self._normalize_text_terms(actions.get("dislike", [])),
                    *self._normalize_text_terms(actions.get("not_interested", [])),
                    *self._normalize_text_terms(actions.get("archived", [])),
                ]
            )
        )
        return {
            "profile": profile,
            "actions": actions,
            "excluded_ids": excluded_ids,
            "disabled": False,
        }

    def _compute_profile_adjustment(
        self,
        candidate: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        actions: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        profile = profile or {}
        actions = actions or {}
        arxiv_id = str(candidate.get("arxiv_id", "") or candidate.get("id", "") or "").strip()
        title = str(candidate.get("title", "") or "").lower()
        abstract = str(candidate.get("abstract", "") or candidate.get("summary", "") or "").lower()
        haystack = f"{title}\n{abstract}"
        categories = {
            str(item).strip().lower()
            for item in (candidate.get("categories") if isinstance(candidate.get("categories"), list) else str(candidate.get("categories") or "").split(","))
            if str(item).strip()
        }

        positive_topics = self._normalize_text_terms(profile.get("positive_topics"))
        negative_topics = self._normalize_text_terms(profile.get("negative_topics"))
        preferred_categories = {item.lower() for item in self._normalize_text_terms(profile.get("preferred_categories"))}

        matched_positive = [topic for topic in positive_topics if topic and topic in haystack]
        matched_negative = [topic for topic in negative_topics if topic and topic in haystack]
        matched_categories = sorted(categories & preferred_categories)

        action_boost = 0.0
        if arxiv_id and arxiv_id in self._normalize_text_terms(actions.get("favorite", [])):
            action_boost += 0.08
        if arxiv_id and arxiv_id in self._normalize_text_terms(actions.get("later", [])):
            action_boost += 0.04
        if arxiv_id and arxiv_id in self._normalize_text_terms(actions.get("read", [])):
            action_boost -= 0.03

        profile_score = min(0.24, len(matched_positive) * 0.04 + len(matched_categories) * 0.03 + action_boost)
        profile_penalty = min(0.24, len(matched_negative) * 0.06)
        net_adjustment = profile_score - profile_penalty

        reasons: List[str] = []
        if matched_positive:
            reasons.append(f"匹配长期主题: {', '.join(matched_positive[:3])}")
        if matched_categories:
            reasons.append(f"匹配偏好分类: {', '.join(matched_categories[:3])}")
        if matched_negative:
            reasons.append(f"命中负向主题: {', '.join(matched_negative[:3])}")

        return {
            "profile_score": profile_score,
            "profile_penalty": profile_penalty,
            "profile_adjustment": net_adjustment,
            "matched_positive_topics": matched_positive,
            "matched_negative_topics": matched_negative,
            "matched_preferred_categories": matched_categories,
            "profile_reasons": reasons,
        }

    def _get_or_refresh_interest_vector(self, user_id: str) -> Dict[str, Any]:
        vector_data = self.db_service.get_user_interest_vector(user_id=user_id)
        latest_preference_ts = self.db_service.get_latest_user_signal_timestamp(user_id=user_id)

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
        profile_bundle = self._build_profile_signal_bundle(user_id)
        research_profile = profile_bundle.get("profile", {})
        paper_actions = profile_bundle.get("actions", {})
        excluded_ids = list(dict.fromkeys([*liked_ids, *disliked_ids, *profile_bundle.get("excluded_ids", [])]))

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
            profile_adjustment = self._compute_profile_adjustment(
                scored_candidate,
                profile=research_profile,
                actions=paper_actions,
            )
            scored_candidate["profile_score"] = profile_adjustment["profile_score"]
            scored_candidate["profile_penalty"] = profile_adjustment["profile_penalty"]
            scored_candidate["profile_reasons"] = profile_adjustment["profile_reasons"]
            scored_candidate["matched_positive_topics"] = profile_adjustment["matched_positive_topics"]
            scored_candidate["matched_negative_topics"] = profile_adjustment["matched_negative_topics"]
            scored_candidate["matched_preferred_categories"] = profile_adjustment["matched_preferred_categories"]
            scored_candidate["relevance_score"] = float(scored_candidate.get("relevance_score", 0.0) or 0.0) + profile_adjustment["profile_adjustment"]
            scored_candidate["final_score"] = float(scored_candidate.get("final_score", 0.0) or 0.0) + profile_adjustment["profile_adjustment"]
            score_breakdown = dict(scored_candidate.get("score_breakdown", {}))
            score_breakdown["profile_score"] = profile_adjustment["profile_score"]
            score_breakdown["profile_penalty"] = profile_adjustment["profile_penalty"]
            score_breakdown["relevance_score"] = scored_candidate["relevance_score"]
            score_breakdown["final_score"] = scored_candidate["final_score"]
            scored_candidate["score_breakdown"] = score_breakdown
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
            "research_profile": research_profile,
            "paper_actions": paper_actions,
            "recall_mode": recall_mode,
            "recommendations": selected,
        }

    def rerank_search_results_for_user(
        self,
        user_id: str,
        papers: List[Dict[str, Any]],
        query: Optional[str],
        top_n: int,
        search_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Deterministically rerank search results for a specific user.

        The method keeps the existing search results intact when personalization
        cannot be applied, and it reuses the same embedding/vector logic that the
        recommendation pipeline already uses.
        """
        normalized_papers = [paper for paper in papers if isinstance(paper, dict)]
        limit = max(1, min(int(top_n or len(normalized_papers) or 1), len(normalized_papers) or 1))
        warnings: List[str] = []

        if not normalized_papers:
            return {
                "status": "success",
                "message": "No papers to rerank",
                "personalized_applied": False,
                "warnings": warnings,
                "papers": [],
            }

        query_text = str(query or "").strip()
        title_query = str((search_spec or {}).get("title_query") or "").strip()
        abstract_query = str((search_spec or {}).get("abstract_query") or "").strip()
        search_categories = list((search_spec or {}).get("categories") or [])
        profile_bundle = self._build_profile_signal_bundle(user_id)
        research_profile = profile_bundle.get("profile", {})
        paper_actions = profile_bundle.get("actions", {})
        try:
            user_vector_data = self._get_or_refresh_interest_vector(user_id)
            personalized_available = True
        except Exception as exc:
            user_vector_data = None
            personalized_available = False
            warnings.append(f"用户兴趣向量不可用，已退化为普通搜索排序: {exc}")

        try:
            preferences = self.db_service.get_user_preferences(user_id=user_id)
            liked_ids = preferences.get("liked_papers", [])
            disliked_ids = preferences.get("disliked_papers", [])
            liked_details = self.db_service.get_liked_papers_with_details(user_id=user_id)
            liked_category_freq = self._build_liked_category_frequency(liked_details)
        except Exception as exc:
            warnings.append(f"读取用户偏好失败，已退化为普通搜索排序: {exc}")
            preferences = {"liked_papers": [], "disliked_papers": []}
            liked_ids = []
            disliked_ids = []
            liked_category_freq = Counter()
            personalized_available = False
            user_vector = None
            interest_clusters = []
            disliked_vector = None
            embedding_config = None

        user_vector = user_vector_data.get("vector_data") if user_vector_data else None
        interest_clusters = user_vector_data.get("interest_clusters", []) if user_vector_data else []
        disliked_vector = user_vector_data.get("disliked_vector_data") if user_vector_data else None
        embedding_config = self.get_embedding_config() if personalized_available else None

        candidate_ids = [str(paper.get("arxiv_id", "") or paper.get("id", "") or "").strip() for paper in normalized_papers if str(paper.get("arxiv_id", "") or paper.get("id", "") or "").strip()]
        stored_embeddings: Dict[str, List[float]] = {}
        if personalized_available and candidate_ids:
            try:
                candidate_embedding_rows = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
                    collection_name=self.collection_name,
                    arxiv_ids=candidate_ids,
                )
                stored_embeddings = {
                    str(item.get("arxiv_id", "") or "").strip(): [float(value) for value in item.get("vector", [])]
                    for item in candidate_embedding_rows
                    if item.get("arxiv_id") and item.get("vector")
                }
            except Exception as exc:
                warnings.append(f"复用候选论文向量失败，已使用文本特征继续排序: {exc}")
                stored_embeddings = {}

        scored_candidates: List[Dict[str, Any]] = []
        for paper in normalized_papers:
            fallback_arxiv_id = str(paper.get("arxiv_id", "") or paper.get("id", "") or "").strip()
            normalized_paper = self._normalize_paper_record(paper, fallback_arxiv_id or "unknown")
            candidate = {**paper, **normalized_paper}

            if personalized_available and fallback_arxiv_id and fallback_arxiv_id in stored_embeddings:
                candidate["_stored_vector"] = stored_embeddings[fallback_arxiv_id]

            query_breakdown = self._build_query_match_score(
                candidate,
                query=query_text or normalized_paper.get("query", ""),
                title_query=title_query,
                abstract_query=abstract_query,
                search_categories=search_categories,
            )
            candidate["query_match_score"] = query_breakdown["query_match_score"]
            candidate["matched_terms"] = query_breakdown["matched_terms"]
            candidate["query_score_breakdown"] = query_breakdown["query_score_breakdown"]

            if personalized_available and user_vector:
                try:
                    ranked_candidate = self._build_candidate_score(
                        candidate=candidate,
                        liked_category_freq=liked_category_freq,
                        user_vector=user_vector,
                        interest_clusters=interest_clusters,
                        disliked_vector=disliked_vector,
                        embedding_config=embedding_config,
                    )
                    personalization_score = float(ranked_candidate.get("relevance_score", 0.0) or 0.0)
                    score_breakdown = dict(ranked_candidate.get("score_breakdown", {}))
                    score_breakdown.update(query_breakdown["query_score_breakdown"])
                except Exception as exc:
                    warnings.append(f"论文 {fallback_arxiv_id or 'unknown'} 个性化打分失败，已退化为查询排序: {exc}")
                    ranked_candidate = dict(candidate)
                    personalization_score = 0.0
                    score_breakdown = {
                        "semantic_score": 0.0,
                        "category_score": 0.0,
                        "recency_score": 0.0,
                        "disliked_penalty": 0.0,
                        "relevance_score": 0.0,
                        "diversity_score": 0.0,
                    }
                    score_breakdown.update(query_breakdown["query_score_breakdown"])
            else:
                ranked_candidate = dict(candidate)
                personalization_score = 0.0
                score_breakdown = {
                    "semantic_score": 0.0,
                    "category_score": 0.0,
                    "recency_score": 0.0,
                    "disliked_penalty": 0.0,
                    "relevance_score": 0.0,
                    "diversity_score": 0.0,
                }
                score_breakdown.update(query_breakdown["query_score_breakdown"])

            query_match_score = float(query_breakdown["query_match_score"] or 0.0)
            profile_adjustment = self._compute_profile_adjustment(
                ranked_candidate,
                profile=research_profile,
                actions=paper_actions,
            )
            final_score = query_match_score * 0.60 + personalization_score * 0.30 + profile_adjustment["profile_adjustment"]
            ranked_candidate["query_match_score"] = query_match_score
            ranked_candidate["personalization_score"] = personalization_score
            ranked_candidate["profile_score"] = profile_adjustment["profile_score"]
            ranked_candidate["profile_penalty"] = profile_adjustment["profile_penalty"]
            ranked_candidate["profile_reasons"] = profile_adjustment["profile_reasons"]
            ranked_candidate["matched_positive_topics"] = profile_adjustment["matched_positive_topics"]
            ranked_candidate["matched_negative_topics"] = profile_adjustment["matched_negative_topics"]
            ranked_candidate["matched_preferred_categories"] = profile_adjustment["matched_preferred_categories"]
            ranked_candidate["final_score"] = final_score
            ranked_candidate["score_breakdown"] = {
                **score_breakdown,
                "query_match_score": query_match_score,
                "personalization_score": personalization_score,
                "profile_score": profile_adjustment["profile_score"],
                "profile_penalty": profile_adjustment["profile_penalty"],
                "final_score": final_score,
            }
            ranked_candidate["match_reason"] = self._build_match_reason(query_match_score, list(query_breakdown["matched_terms"]))
            ranked_candidate["personalized_reason"] = self._build_personalized_reason(ranked_candidate, ranked_candidate["score_breakdown"])
            ranked_candidate["priority"] = 0

            ranked_candidate.pop("_candidate_embedding", None)
            ranked_candidate.pop("_candidate_categories", None)
            ranked_candidate.pop("_stored_vector", None)
            ranked_candidate.pop("query_score_breakdown", None)
            scored_candidates.append(ranked_candidate)

        scored_candidates.sort(
            key=lambda item: (
                float(item.get("final_score", 0.0) or 0.0),
                float(item.get("query_match_score", 0.0) or 0.0),
                float(item.get("personalization_score", 0.0) or 0.0),
                float(item.get("score_breakdown", {}).get("semantic_score", 0.0) or 0.0),
                str(item.get("published_date", "") or ""),
                str(item.get("arxiv_id", "") or item.get("id", "") or ""),
            ),
            reverse=True,
        )

        for index, paper in enumerate(scored_candidates, start=1):
            paper["priority"] = index
            paper["score_breakdown"]["priority"] = index

        selected = scored_candidates[:limit]
        personalized_applied = bool(personalized_available and user_vector)
        if personalized_applied:
            personalized_count = sum(1 for paper in selected if float(paper.get("personalization_score", 0.0) or 0.0) > 0.0)
            if personalized_count == 0:
                warnings.append("已获取用户兴趣向量，但当前候选论文未形成有效个性化增益")

        return {
            "status": "success",
            "message": "Search results reranked successfully",
            "personalized_applied": personalized_applied,
            "warnings": warnings,
            "papers": selected,
            "total_found": len(scored_candidates),
            "top_n": limit,
            "liked_papers_count": len(liked_ids),
            "disliked_papers_count": len(disliked_ids),
            "research_profile": research_profile,
            "paper_actions": paper_actions,
        }
