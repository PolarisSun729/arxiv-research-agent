from __future__ import annotations

import ast
import logging
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RecommendationRanker:
    def _build_candidate_score(
        self,
        candidate: Dict[str, Any],
        liked_category_freq: Counter,
        user_vector: Optional[List[float]] = None,
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
        disliked_vector: Optional[List[float]] = None,
        embedding_config: Optional[Any] = None,
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
                text_to_embed = self.embedding_service.build_paper_embedding_text(candidate.get("title", ""), candidate.get("abstract", ""))
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
                            "similarity": self._cosine_similarity(candidate_embedding, [float(value) for value in cluster.get("centroid_vector", [])]),
                        }
                        for cluster in interest_clusters
                        if cluster.get("centroid_vector")
                    ]
                    if cluster_similarities:
                        best_cluster = max(cluster_similarities, key=lambda item: float(item.get("similarity", 0.0) or 0.0))
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

        score_weights = self.RECOMMENDATION_CONFIG["score_weights"]
        base_score = (
            semantic_score * score_weights["semantic"]
            + category_score * score_weights["category"]
            + recency_score * score_weights["recency"]
            - disliked_penalty * score_weights["disliked_penalty"]
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

    def _select_diverse_candidates(
        self,
        scored_candidates: List[Dict[str, Any]],
        top_n: int,
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if top_n <= 0 or not scored_candidates:
            return []

        ranked_candidates = list(scored_candidates)
        relevance_scores = [float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0) for candidate in ranked_candidates]
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

        def candidate_priority(candidate: Dict[str, Any]) -> tuple[float, float, float, float, str]:
            relevance_score = float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0)
            return (
                normalize_relevance(relevance_score),
                relevance_score,
                float(candidate.get("semantic_score", 0.0) or 0.0),
                float(candidate.get("final_score", 0.0) or 0.0),
                str(candidate.get("arxiv_id", "") or ""),
            )

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

        def compute_diversity(candidate: Dict[str, Any]) -> tuple[float, Optional[float], Optional[float], Optional[float], str, Optional[str], Optional[float]]:
            embedding = candidate_embedding(candidate)
            semantic_diversity_score: Optional[float] = None
            semantic_similarity_penalty: Optional[float] = None
            if embedding and selected_embeddings:
                max_similarity = max(self._cosine_similarity(embedding, selected_embedding) for selected_embedding in selected_embeddings)
                semantic_similarity_penalty = max(0.0, min(1.0, max_similarity))
                semantic_diversity_score = 1.0 - semantic_similarity_penalty

            cluster_id = candidate_cluster_id(candidate)
            cluster_diversity_score: Optional[float] = None
            cluster_repeat_count: Optional[int] = None
            if cluster_id and interest_clusters:
                cluster_repeat_count = int(selected_cluster_counts.get(cluster_id, 0))
                cluster_diversity_score = max(0.0, 1.0 - min(cluster_repeat_count / self.ENHANCED_RETRIEVAL_CONFIG["cluster_repeat_divisor"], 1.0))

            categories = candidate.get("_candidate_categories", []) or []
            category_diversity_score: Optional[float] = None
            category_repeat_count: Optional[int] = None
            if not embedding and cluster_diversity_score is None and categories:
                category_repeat_count = max((int(selected_category_counts.get(category, 0)) for category in categories), default=0)
                category_diversity_score = max(0.0, 1.0 - min(category_repeat_count / self.ENHANCED_RETRIEVAL_CONFIG["category_repeat_divisor"], 1.0))

            components: List[tuple[float, float]] = []
            if semantic_diversity_score is not None:
                components.append((semantic_diversity_score, self.RECOMMENDATION_CONFIG["semantic_diversity_component_weight"]))
            if cluster_diversity_score is not None:
                components.append((cluster_diversity_score, self.RECOMMENDATION_CONFIG["cluster_diversity_component_weight"]))
            if not components and category_diversity_score is not None:
                components.append((category_diversity_score, 1.0))

            if components:
                total_weight = sum(weight for _, weight in components)
                diversity_score = sum(value * weight for value, weight in components) / total_weight
                diversity_reason = "semantic_cluster_mix" if len(components) > 1 else ("semantic_guidance" if semantic_diversity_score is not None else "cluster_guidance")
            else:
                diversity_score = 1.0
                diversity_reason = "no_previous_selection"

            if semantic_similarity_penalty is not None and semantic_similarity_penalty >= self.RECOMMENDATION_CONFIG["semantic_similarity_penalty_threshold"]:
                diversity_reason = "semantic_repeat"
            elif cluster_repeat_count and cluster_repeat_count > 0:
                diversity_reason = "cluster_repeat"
            elif category_repeat_count and category_repeat_count > 0:
                diversity_reason = "category_repeat"

            return diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, candidate_cluster_id(candidate), semantic_similarity_penalty

        def commit_selected_candidate(
            candidate: Dict[str, Any],
            selection_rank: int,
            diversity_reason: str,
            diversity_score: float,
            semantic_diversity_score: Optional[float],
            cluster_diversity_score: Optional[float],
            category_diversity_score: Optional[float],
            diversity_penalty_source: Optional[str] = None,
            diversity_penalty_value: Optional[float] = None,
        ) -> None:
            relevance_score = float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0)
            annotate_selected_candidate(candidate, selection_rank, relevance_score, diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, diversity_penalty_source, diversity_penalty_value)
            selected_embeddings_candidate = candidate_embedding(candidate)
            if selected_embeddings_candidate:
                selected_embeddings.append(selected_embeddings_candidate)
            cluster_id = candidate_cluster_id(candidate)
            if cluster_id:
                selected_cluster_counts[cluster_id] += 1
            for category in candidate.get("_candidate_categories", []) or []:
                selected_category_counts[category] += 1
            selected.append(finalize_candidate(candidate))

        if interest_clusters:
            cluster_representatives: Dict[str, Dict[str, Any]] = {}
            for candidate in remaining:
                cluster_id = candidate_cluster_id(candidate)
                if not cluster_id:
                    continue
                existing_candidate = cluster_representatives.get(cluster_id)
                if existing_candidate is None or candidate_priority(candidate) > candidate_priority(existing_candidate):
                    cluster_representatives[cluster_id] = candidate

            ordered_cluster_candidates = sorted(cluster_representatives.values(), key=candidate_priority, reverse=True)
            for candidate in ordered_cluster_candidates:
                if len(selected) >= top_n:
                    break
                if candidate not in remaining:
                    continue
                remaining.remove(candidate)
                relevance_score = float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0)
                candidate["final_score"] = normalize_relevance(relevance_score)
                commit_selected_candidate(candidate, len(selected) + 1, "cluster_seed", 1.0, None, None, None)

        if not selected and remaining:
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
            first_candidate["final_score"] = normalize_relevance(first_relevance)
            commit_selected_candidate(first_candidate, 1, "seed", 1.0, None, None, None)

        while remaining and len(selected) < top_n:
            best_candidate: Optional[Dict[str, Any]] = None
            best_selection_score = float("-inf")
            best_diversity_data = (1.0, None, None, None, "no_previous_selection", None, None)
            for candidate in remaining:
                relevance_score = float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0)
                normalized_relevance = normalize_relevance(relevance_score)
                diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, diversity_cluster_id, semantic_similarity_penalty = compute_diversity(candidate)
                selection_score = normalized_relevance * self.RECOMMENDATION_CONFIG["selection_relevance_weight"] + diversity_score * self.RECOMMENDATION_CONFIG["selection_diversity_weight"]
                if selection_score > best_selection_score:
                    best_candidate = candidate
                    best_selection_score = selection_score
                    best_diversity_data = (diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, diversity_cluster_id, semantic_similarity_penalty)

            if best_candidate is None:
                break

            remaining.remove(best_candidate)
            diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, diversity_cluster_id, semantic_similarity_penalty = best_diversity_data
            best_candidate["final_score"] = best_selection_score
            diversity_penalty_value = semantic_similarity_penalty if semantic_similarity_penalty is not None else max(0.0, 1.0 - diversity_score)
            commit_selected_candidate(
                candidate=best_candidate,
                selection_rank=len(selected) + 1,
                diversity_reason=diversity_reason,
                diversity_score=diversity_score,
                semantic_diversity_score=semantic_diversity_score,
                cluster_diversity_score=cluster_diversity_score,
                category_diversity_score=category_diversity_score,
                diversity_penalty_source="semantic" if semantic_similarity_penalty is not None else ("cluster" if cluster_diversity_score is not None else ("category" if category_diversity_score is not None else None)),
                diversity_penalty_value=diversity_penalty_value,
            )

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
        candidates = [text, text.replace("Z", "+00:00"), text[:10]]
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

