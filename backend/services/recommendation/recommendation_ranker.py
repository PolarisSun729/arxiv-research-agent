from __future__ import annotations

import ast
import logging
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
QUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "or",
    "for",
    "with",
    "in",
    "on",
    "about",
    "this",
    "that",
    "paper",
    "papers",
    "arxiv",
}


class RecommendationRanker:
    def _build_candidate_score(
        self,
        candidate: Dict[str, Any],
        liked_category_freq: Counter,
        user_vector: Optional[List[float]] = None,
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
        disliked_vector: Optional[List[float]] = None,
        negative_feedback_profile: Optional[Dict[str, Any]] = None,
        embedding_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """构建单篇候选论文的基础相关性分数及详细分解信息。

        该方法会综合语义相似度、分类偏好、新鲜度以及负反馈惩罚，输出一个
        可解释的中间评分结构，供后续画像修正和多样性选择继续加工。
        """
        semantic_score = float(candidate.get("similarity_score", candidate.get("score", 0.0)) or 0.0)
        best_matched_cluster_id = None
        best_matched_cluster_similarity = None
        cluster_similarities: List[Dict[str, Any]] = []
        disliked_penalty = 0.0
        negative_feedback_match: Dict[str, Any] = {}
        negative_score = 0.0
        negative_penalty_debug = self._empty_negative_penalty_debug()
        embedding_source = "none"
        candidate_embedding: Optional[List[float]] = None
        if user_vector and embedding_config:
            stored_vector = candidate.get("_stored_vector") or []
            if stored_vector:
                # 优先复用候选论文已有向量，避免重复调用 embedding 服务。
                candidate_embedding = [float(value) for value in stored_vector]
                embedding_source = "stored"
            else:
                text_to_embed = self.embedding_service.build_paper_embedding_text(candidate.get("title", ""), candidate.get("abstract", ""))
                if text_to_embed:
                    # 没有现成向量时，按当前 embedding 配置重新生成，保证排序可继续进行。
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
                    # 有兴趣簇时优先比较候选与各簇中心的相似度，以便识别命中的具体子兴趣。
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
                # 负向画像优先使用多簇中心，其次使用实例级样本；旧单均值只作为兼容兜底。
                negative_score, negative_feedback_match = self._calculate_negative_feedback_score(
                    candidate_embedding=candidate_embedding,
                    negative_feedback_profile=negative_feedback_profile,
                    disliked_vector=disliked_vector,
                )
                negative_penalty_debug = self._calculate_negative_feedback_penalty(
                    positive_score=semantic_score,
                    negative_score=negative_score,
                    negative_feedback_profile=negative_feedback_profile,
                    negative_feedback_match=negative_feedback_match,
                )
                disliked_penalty = float(negative_penalty_debug["penalty"])
        categories = self._split_categories(candidate.get("categories"))
        category_score = self._calculate_category_score(categories, liked_category_freq)
        recency_score = self._calculate_recency_score(candidate.get("published_date"))

        score_weights = self.RECOMMENDATION_CONFIG["score_weights"]
        base_score = (
            semantic_score * score_weights["semantic"]
            + category_score * score_weights["category"]
            + recency_score * score_weights["recency"]
            - disliked_penalty
        )

        return {
            **candidate,
            "similarity_score": semantic_score,
            "semantic_score": semantic_score,
            "positive_score": semantic_score,
            "base_score": base_score,
            "relevance_score": base_score,
            "best_matched_cluster_id": best_matched_cluster_id,
            "best_matched_cluster_similarity": best_matched_cluster_similarity,
            "cluster_similarities": cluster_similarities,
            "disliked_penalty": disliked_penalty,
            "negative_score": negative_score,
            "negative_penalty": disliked_penalty,
            "negative_confidence": negative_penalty_debug["confidence"],
            "negative_feedback_hit": bool(negative_penalty_debug["hit"]),
            "negative_penalty_applied": bool(negative_penalty_debug["penalty_applied"]),
            "negative_hard_filter": bool(negative_penalty_debug["hard_filter"]),
            "negative_feedback_mode": (negative_feedback_profile or {}).get("mode") or negative_feedback_match.get("source") or "none",
            "negative_feedback_match": negative_feedback_match,
            "negative_feedback_debug": negative_penalty_debug["debug"],
            "_embedding_source": embedding_source,
            "_candidate_embedding": candidate_embedding,
            "final_score": base_score,
            "score_breakdown": {
                "semantic_score": semantic_score,
                "positive_score": semantic_score,
                "category_score": category_score,
                "recency_score": recency_score,
                "disliked_penalty": disliked_penalty,
                "negative_score": negative_score,
                "negative_penalty": disliked_penalty,
                "negative_confidence": negative_penalty_debug["confidence"],
                "negative_threshold_penalty": negative_penalty_debug["threshold_penalty"],
                "negative_margin_penalty": negative_penalty_debug["margin_penalty"],
                "negative_penalty_applied": bool(negative_penalty_debug["penalty_applied"]),
                "negative_hard_filter": bool(negative_penalty_debug["hard_filter"]),
                "relevance_score": base_score,
                "diversity_score": 0.0,
            },
            "_candidate_categories": categories,
        }

    def _calculate_negative_feedback_score(
        self,
        *,
        candidate_embedding: List[float],
        negative_feedback_profile: Optional[Dict[str, Any]],
        disliked_vector: Optional[List[float]],
    ) -> tuple[float, Dict[str, Any]]:
        """计算候选论文与负向画像的最大相似度，作为排序阶段的独立惩罚信号。"""
        profile = negative_feedback_profile if isinstance(negative_feedback_profile, dict) else {}
        profile_provided = bool(profile)
        if bool(profile.get("enabled", False)):
            clusters = [item for item in profile.get("clusters") or [] if isinstance(item, dict) and item.get("centroid_vector")]
            if clusters:
                # 足量负反馈时优先按负向簇中心惩罚，避免多个 dislike 方向被单一均值压扁。
                matches = [
                    {
                        "source": "negative_cluster",
                        "cluster_id": cluster.get("cluster_id"),
                        "cluster_label": cluster.get("cluster_label"),
                        "paper_ids": cluster.get("paper_ids", []),
                        "similarity": self._cosine_similarity(
                            candidate_embedding,
                            [float(value) for value in cluster.get("centroid_vector", [])],
                        ),
                    }
                    for cluster in clusters
                ]
                best_match = max(matches, key=lambda item: float(item.get("similarity", 0.0) or 0.0))
                return float(best_match.get("similarity", 0.0) or 0.0), best_match

            examples = [item for item in profile.get("examples") or [] if isinstance(item, dict) and item.get("vector")]
            if examples:
                # 样本少或聚类失败时，取与任一点踩样本的最大相似度，表达实例级“不要再类似这个”。
                matches = [
                    {
                        "source": "negative_example",
                        "arxiv_id": example.get("arxiv_id"),
                        "title": example.get("title"),
                        "similarity": self._cosine_similarity(
                            candidate_embedding,
                            [float(value) for value in example.get("vector", [])],
                        ),
                    }
                    for example in examples
                ]
                best_match = max(matches, key=lambda item: float(item.get("similarity", 0.0) or 0.0))
                return float(best_match.get("similarity", 0.0) or 0.0), best_match

        if not profile_provided and disliked_vector:
            # 兼容旧调用方直接传入 disliked_vector 的场景；新画像会通过 profile 进入。
            similarity = self._cosine_similarity(disliked_vector, candidate_embedding)
            return similarity, {"source": "legacy_disliked_vector", "similarity": similarity}

        return 0.0, {}

    def _calculate_negative_feedback_penalty(
        self,
        *,
        positive_score: float,
        negative_score: float,
        negative_feedback_profile: Optional[Dict[str, Any]],
        negative_feedback_match: Dict[str, Any],
    ) -> Dict[str, Any]:
        """按阈值、margin 和置信度把负向相似度转换为实际排序扣分。"""
        config = self.RECOMMENDATION_CONFIG["ranking"]["negative"]
        debug_enabled = bool(config["debug_enabled"])
        empty = self._empty_negative_penalty_debug()
        if not bool(config["enabled"]) or not negative_feedback_match:
            return empty

        threshold = float(config["similarity_threshold"])
        if negative_score <= threshold:
            debug = {
                "reason": "below_negative_similarity_threshold",
                "positive_score": positive_score,
                "negative_score": negative_score,
                "similarity_threshold": threshold,
            } if debug_enabled else {}
            return {**empty, "debug": debug}

        confidence = self._calculate_negative_feedback_confidence(negative_feedback_profile, negative_feedback_match)
        penalty_weight = float(config["penalty_weight"])
        threshold_penalty = 0.0
        if bool(config["use_threshold_penalty"]):
            threshold_range = max(1.0 - threshold, 1e-9)
            threshold_penalty = ((negative_score - threshold) / threshold_range) * penalty_weight

        margin = float(config["margin"])
        margin_penalty = 0.0
        margin_boundary = positive_score - margin
        if bool(config["use_margin_penalty"]) and negative_score > margin_boundary:
            margin_range = max(margin, 1e-9)
            margin_penalty = ((negative_score - margin_boundary) / margin_range) * penalty_weight

        raw_penalty = threshold_penalty + margin_penalty
        scaled_threshold_penalty = threshold_penalty * confidence
        scaled_margin_penalty = margin_penalty * confidence
        scaled_penalty = scaled_threshold_penalty + scaled_margin_penalty
        penalty = min(float(config["max_penalty"]), scaled_penalty)
        if scaled_penalty > penalty and scaled_penalty > 0:
            # max_penalty 截断后同步缩放分项，保证 debug 分解能加总回最终扣分。
            component_scale = penalty / scaled_penalty
            scaled_threshold_penalty *= component_scale
            scaled_margin_penalty *= component_scale
        hard_filter = (
            bool(config["enable_hard_filter"])
            and negative_score >= float(config["hard_filter_threshold"])
        )
        debug = {
            "reason": "negative_feedback_penalty_applied" if penalty > 0 else "negative_feedback_hit_without_penalty",
            "positive_score": positive_score,
            "negative_score": negative_score,
            "similarity_threshold": threshold,
            "margin": margin,
            "margin_boundary": margin_boundary,
            "confidence": confidence,
            "threshold_penalty": scaled_threshold_penalty,
            "margin_penalty": scaled_margin_penalty,
            "raw_penalty": raw_penalty,
            "scaled_penalty_before_cap": scaled_penalty,
            "penalty": penalty,
            "max_penalty": float(config["max_penalty"]),
            "hard_filter": hard_filter,
            "match": negative_feedback_match,
        } if debug_enabled else {}
        return {
            "hit": True,
            "penalty_applied": penalty > 0,
            "penalty": penalty,
            "threshold_penalty": scaled_threshold_penalty,
            "margin_penalty": scaled_margin_penalty,
            "confidence": confidence,
            "hard_filter": hard_filter,
            "debug": debug,
        }

    def _calculate_negative_feedback_confidence(
        self,
        negative_feedback_profile: Optional[Dict[str, Any]],
        negative_feedback_match: Dict[str, Any],
    ) -> float:
        """根据可用 disliked 样本数量降低少量点踩带来的误伤风险。"""
        config = self.RECOMMENDATION_CONFIG["ranking"]["negative"]
        min_count = int(config["confidence_min_count"])
        if min_count <= 0:
            return 1.0
        profile = negative_feedback_profile if isinstance(negative_feedback_profile, dict) else {}
        stats = profile.get("stats") if isinstance(profile.get("stats"), dict) else {}
        usable_count = int(stats.get("usable_disliked") or 0)
        if usable_count <= 0 and negative_feedback_match.get("source") == "negative_cluster":
            usable_count = len(negative_feedback_match.get("paper_ids") or [])
        if usable_count <= 0 and negative_feedback_match:
            usable_count = 1
        return max(0.0, min(1.0, usable_count / min_count))

    @staticmethod
    def _empty_negative_penalty_debug() -> Dict[str, Any]:
        return {
            "hit": False,
            "penalty_applied": False,
            "penalty": 0.0,
            "threshold_penalty": 0.0,
            "margin_penalty": 0.0,
            "confidence": 0.0,
            "hard_filter": False,
            "debug": {},
        }

    def _select_diverse_candidates(
        self,
        scored_candidates: List[Dict[str, Any]],
        top_n: int,
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """从已打分候选集中选择兼顾相关性与多样性的最终结果集。

        该策略分两步：先尽量为每个兴趣簇挑一个代表项作为种子，再在剩余候选中按
        相关性和多样性加权贪心选择，减少推荐列表同质化。
        """
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
            """为已入选候选补齐选择阶段的解释信息与调试字段。"""
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
            """移除仅供排序阶段使用的临时字段，并输出可直接返回给上层的结果。"""
            candidate.pop("_candidate_embedding", None)
            candidate.pop("_candidate_categories", None)
            return candidate

        def compute_diversity(candidate: Dict[str, Any]) -> tuple[float, Optional[float], Optional[float], Optional[float], str, Optional[str], Optional[float]]:
            """计算候选相对当前已选集合的多样性得分，以及对应的惩罚来源。"""
            embedding = candidate_embedding(candidate)
            semantic_diversity_score: Optional[float] = None
            semantic_similarity_penalty: Optional[float] = None
            if embedding and selected_embeddings:
                # 向量越接近已选项，语义多样性越低，需要在选择阶段适度压分。
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
                # 没有向量和簇信息时，用分类重复度作为最后一道保底多样性信号。
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
            """提交一个候选到结果集，并同步更新后续多样性计算所需状态。"""
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

            # 先为每个兴趣簇选一个代表作种子，避免结果被单一兴趣方向垄断。
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
                # 以加权方式融合相关性与多样性，做逐步贪心选择。
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
        """根据候选分类与用户已点赞分类分布的重叠程度，计算分类偏好分。"""
        if not categories or not liked_category_freq:
            return 0.0
        total_frequency = sum(liked_category_freq.values())
        if total_frequency <= 0:
            return 0.0
        matched_frequency = sum(liked_category_freq.get(category, 0) for category in categories)
        return min(1.0, matched_frequency / total_frequency)

    def _calculate_recency_score(self, published_date: Any) -> float:
        """把发布时间衰减为 0 到 1 之间的新鲜度分值。"""
        parsed_date = self._parse_datetime(str(published_date or ""))
        if not parsed_date:
            return 0.0
        age_days = max(0, (datetime.now(timezone.utc) - parsed_date).days)
        return 1.0 / (1.0 + age_days / 365.0)

    def _extract_query_terms(self, text: Any) -> List[str]:
        """从查询文本中抽取去重后的有效词项，并过滤停用词与噪声符号。"""
        normalized = str(text or "").strip().lower()
        if not normalized:
            return []

        raw_terms = re.findall(r"[a-z0-9][a-z0-9+\-_/\.]*|[\u4e00-\u9fff]{2,}", normalized)
        terms: List[str] = []
        seen = set()
        for term in raw_terms:
            cleaned = term.strip("._-+/")
            if not cleaned or cleaned in QUERY_STOPWORDS or cleaned in seen:
                continue
            seen.add(cleaned)
            terms.append(cleaned)
        return terms

    def _build_query_match_score(
        self,
        candidate: Dict[str, Any],
        query: Optional[str] = None,
        title_query: Optional[str] = None,
        abstract_query: Optional[str] = None,
        search_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """计算候选论文与搜索查询之间的匹配分，并输出命中明细。"""
        title = str(candidate.get("title", "") or "").strip()
        abstract = str(candidate.get("abstract", "") or candidate.get("summary", "") or "").strip()
        categories = self._split_categories(candidate.get("categories"))
        candidate_text = " ".join([title, abstract, " ".join(categories)]).lower()

        query_terms = self._extract_query_terms(query)
        title_terms = self._extract_query_terms(title_query)
        abstract_terms = self._extract_query_terms(abstract_query)
        normalized_search_categories = [
            str(item).strip() for item in (search_categories or []) if str(item).strip()
        ]
        normalized_search_categories_set = {item.lower() for item in normalized_search_categories}
        normalized_candidate_categories_set = {item.lower() for item in categories}

        matched_terms: List[str] = []
        query_hits = [term for term in query_terms if term in candidate_text]
        title_hits = [term for term in title_terms if term in title.lower()]
        abstract_hits = [term for term in abstract_terms if term in abstract.lower()]
        category_hits = [
            category
            for category in normalized_search_categories
            if category.lower() in normalized_candidate_categories_set
        ]
        for term in [*query_hits, *title_hits, *abstract_hits, *category_hits]:
            if term not in matched_terms:
                matched_terms.append(term)

        query_term_score = (len(query_hits) / len(query_terms)) if query_terms else 0.0
        title_term_score = (len(title_hits) / len(title_terms)) if title_terms else 0.0
        abstract_term_score = (len(abstract_hits) / len(abstract_terms)) if abstract_terms else 0.0
        category_term_score = (
            len(category_hits) / len(normalized_search_categories_set)
            if normalized_search_categories_set
            else 0.0
        )
        # 保留原搜索服务的得分作为一个弱信号，避免重排完全忽略上游召回质量。
        source_search_score = float(candidate.get("score", candidate.get("similarity_score", 0.0)) or 0.0)
        source_search_score = max(0.0, min(1.0, source_search_score))

        phrase_bonus = 0.0
        if title_query and title_query.lower() in title.lower():
            phrase_bonus += 0.1
        if abstract_query and abstract_query.lower() in abstract.lower():
            phrase_bonus += 0.05

        query_match_score = (
            max(query_term_score, source_search_score) * 0.55
            + title_term_score * 0.2
            + abstract_term_score * 0.15
            + category_term_score * 0.1
            + phrase_bonus
        )
        query_match_score = max(0.0, min(1.0, query_match_score))

        return {
            "query_match_score": query_match_score,
            "matched_terms": matched_terms,
            "query_score_breakdown": {
                "query_term_score": query_term_score,
                "title_term_score": title_term_score,
                "abstract_term_score": abstract_term_score,
                "category_term_score": category_term_score,
                "source_search_score": source_search_score,
                "phrase_bonus": phrase_bonus,
            },
        }

    def _build_match_reason(self, query_match_score: float, matched_terms: List[str]) -> str:
        """把查询命中情况转换为面向前端展示的简短说明文案。"""
        score_text = f"{round(max(0.0, min(1.0, query_match_score)) * 100)}%"
        if matched_terms:
            preview = ", ".join(matched_terms[:3])
            return f"当前查询命中 {preview}，匹配度 {score_text}"
        return f"当前查询相关度 {score_text}"

    def _build_personalized_reason(self, candidate: Dict[str, Any], score_breakdown: Dict[str, Any]) -> str:
        """根据排序分解结果生成个性化推荐理由，方便前端直接展示。"""
        parts: List[str] = []
        semantic_score = float(score_breakdown.get("semantic_score", 0.0) or 0.0)
        category_score = float(score_breakdown.get("category_score", 0.0) or 0.0)
        recency_score = float(score_breakdown.get("recency_score", 0.0) or 0.0)
        disliked_penalty = float(score_breakdown.get("disliked_penalty", 0.0) or 0.0)
        negative_score = float(score_breakdown.get("negative_score", 0.0) or 0.0)
        negative_penalty_applied = bool(score_breakdown.get("negative_penalty_applied", False))
        best_cluster_id = str(candidate.get("best_matched_cluster_id", "") or "").strip()

        if best_cluster_id:
            parts.append(f"命中兴趣簇 {best_cluster_id}")
        if semantic_score >= 0.75:
            parts.append(f"与兴趣向量语义相似度 {round(semantic_score * 100)}%")
        elif semantic_score >= 0.5:
            parts.append(f"与兴趣向量有一定相似度 {round(semantic_score * 100)}%")
        if category_score >= 0.5:
            parts.append(f"类别偏好匹配 {round(category_score * 100)}%")
        elif category_score > 0:
            parts.append(f"部分类别匹配 {round(category_score * 100)}%")
        if recency_score >= 0.5:
            parts.append(f"发布时间较新 {round(recency_score * 100)}%")
        if negative_penalty_applied:
            match = candidate.get("negative_feedback_match") if isinstance(candidate.get("negative_feedback_match"), dict) else {}
            source_text = "负向簇" if match.get("source") == "negative_cluster" else "点踩样本"
            parts.append(f"接近{source_text}，负向相似度 {round(negative_score * 100)}%，扣分 {round(disliked_penalty, 3)}")

        if not parts:
            return "基于用户兴趣向量进行了重排"
        return "；".join(parts)

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
