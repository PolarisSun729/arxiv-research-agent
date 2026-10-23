from __future__ import annotations

import ast
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Pattern

import numpy as np

logger = logging.getLogger(__name__)

# 查询词抽取时忽略的英文停用词；"paper"、"arxiv" 这类领域高频词在本场景下也没有区分度。
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

# 排序过程中挂在候选 dict 上的临时字段，结果返回给上层前必须清理掉。
# 其中 _stored_vector / _candidate_embedding 是完整 embedding，不清理会让接口响应体积暴涨。
RANKING_TEMP_FIELDS = (
    "_stored_vector",
    "_candidate_embedding",
    "_candidate_categories",
    "_embedding_source",
    "_resolved_candidate_concepts",
    "_resolved_candidate_concept_source",
)

# 含中日韩字符的词项没有空格分词，只能退回子串匹配。
_CJK_CHAR_PATTERN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

# 维度不一致的向量对只告警一次，避免每个候选都刷一条日志。
_WARNED_DIMENSION_MISMATCHES: set[tuple[int, int]] = set()


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> Optional[Pattern[str]]:
    """为英文 / 数字词项构造带词边界的正则，CJK 词项返回 None（改用子串匹配）。

    边界用“前后不是字母数字”而不是正则的单词边界，因为词项本身可能以 "." "-" "+" 结尾（如 "cs.cl"、"c++"）。
    允许词尾追加 s / es，兼容最常见的英文复数形式（"agent" 命中 "agents"）。
    """
    if _CJK_CHAR_PATTERN.search(term):
        return None
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?:s|es)?(?![a-z0-9])")


def text_contains_term(text: str, term: str) -> bool:
    """判断小写文本中是否包含某个词项（按词边界匹配，避免 "rag" 误命中 "storage"）。

    text 与 term 都应已转为小写；空词项一律视为不命中。
    """
    normalized_term = str(term or "").strip().lower()
    if not normalized_term or not text:
        return False
    pattern = _term_pattern(normalized_term)
    if pattern is None:
        return normalized_term in text
    return pattern.search(text) is not None


class RecommendationRanker:
    """推荐排序能力（mixin）：基础打分、负反馈惩罚、多样性选择、查询匹配与推荐理由生成。

    依赖宿主 RecommendationService 注入的 ``embedding_service``、``RECOMMENDATION_CONFIG``、
    ``ENHANCED_RETRIEVAL_CONFIG``，以及 CandidateMaterializer 提供的 ``_create_embedding``。

    分数口径说明：本类所有“相似度”都由 _cosine_similarity 计算，取值为 max(0, cos)，
    即与 Milvus COSINE 检索分数同一口径：0 表示无关（或负相关），1 表示方向完全一致。
    配置中的各类相似度阈值（负反馈阈值、语义重复阈值等）也都基于这个口径。
    """

    # ------------------------------------------------------------------
    # 基础打分
    # ------------------------------------------------------------------

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
        可解释的中间评分结构，供后续画像修正和多样性选择继续加工::

            base_score = semantic * w_semantic + category * w_category + recency * w_recency - negative_penalty

        semantic_score 的来源：有兴趣簇时取与各簇中心的最大相似度，否则取与主兴趣向量的相似度；
        未提供 user_vector 或 embedding_config 时不计算向量相似度，沿用候选自带的召回分数。

        返回的新 dict 中会带上以下临时字段，选择阶段结束后由 _strip_ranking_temp_fields 清理：
        _embedding_source（stored / recomputed / missing / none）、_candidate_embedding、_candidate_categories。
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
                    candidate_embedding = self._create_embedding(text_to_embed, embedding_config)
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

        # 多个字段同值（如 similarity_score / semantic_score / positive_score，disliked_penalty / negative_penalty）
        # 是为了兼容前端与旧调用方读取的不同字段名，修改时需要同步保持一致。
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

    @staticmethod
    def _strip_ranking_temp_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
        """原地移除排序阶段的临时字段（见 RANKING_TEMP_FIELDS），返回同一个 dict。"""
        for key in RANKING_TEMP_FIELDS:
            candidate.pop(key, None)
        return candidate

    # ------------------------------------------------------------------
    # 负反馈惩罚
    # ------------------------------------------------------------------

    def _calculate_negative_feedback_score(
        self,
        *,
        candidate_embedding: List[float],
        negative_feedback_profile: Optional[Dict[str, Any]],
        disliked_vector: Optional[List[float]],
    ) -> tuple[float, Dict[str, Any]]:
        """计算候选论文与负向画像的最大相似度，作为排序阶段的独立惩罚信号。

        匹配优先级：负向簇中心 → 点踩样本 → 旧版单一点踩均值向量（仅在未提供 profile 时使用）。

        Returns:
            (负向相似度, 命中详情)；未命中任何负向信号时返回 (0.0, {})。
        """
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
        """按阈值、margin 和置信度把负向相似度转换为实际排序扣分。

        - threshold_penalty：负向相似度超过 similarity_threshold 的部分，线性映射后乘 penalty_weight；
        - margin_penalty：负向相似度逼近（或超过）正向相似度 - margin 时追加的扣分，
          体现“比起喜欢更像讨厌”的情况；
        - 两者之和乘以样本置信度后，再被 max_penalty 截断；
        - hard_filter：负向相似度超过 hard_filter_threshold 时直接过滤该候选（由调用方执行）。
        """
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
        """根据可用 disliked 样本数量降低少量点踩带来的误伤风险。

        confidence = min(1, 可用点踩数 / confidence_min_count)。可用点踩数优先取画像统计，
        缺失时退回命中负向簇的成员数，再不行按 1 计。
        """
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
        """未命中负向信号时的默认惩罚结构。"""
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

    # ------------------------------------------------------------------
    # 多样性选择
    # ------------------------------------------------------------------

    def _select_diverse_candidates(
        self,
        scored_candidates: List[Dict[str, Any]],
        top_n: int,
        interest_clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """从已打分候选集中选择兼顾相关性与多样性的最终结果集（类 MMR 贪心）。

        该策略分两步：先尽量为每个兴趣簇挑一个代表项作为种子，再在剩余候选中按
        相关性和多样性加权贪心选择，减少推荐列表同质化::

            selection_score = 归一化相关性 * selection_relevance_weight + 多样性 * selection_diversity_weight

        多样性信号优先级：语义（与已选项的最大向量相似度）+ 兴趣簇重复度；
        两者都不可用时才使用分类重复度兜底。

        多样性只决定哪些候选入选，不改写 final_score；入选结果最终按 final_score 从高到低返回，
        使展示顺序与卡片上的综合分一致。入选顺序与当轮选择分记在 selection_rank / selection_score 供调试。

        注意：该方法会原地修改传入的候选 dict（写入 selection_rank、selection_score、diversity_debug 等），
        并清理入选候选的排序临时字段。

        性能：候选向量在开始时一次性归一化并按维度分组成矩阵；每入选一篇论文，只用一次矩阵乘法
        增量更新“每个候选与已选集合的最大相似度”。整体约 O(top_n × 候选数 × 向量维度)，
        且向量运算由 numpy 完成，避免了原先每轮对每个候选重算全部已选项的纯 Python 余弦。
        """
        if top_n <= 0 or not scored_candidates:
            return []

        candidates = list(scored_candidates)
        candidate_count = len(candidates)
        relevance_weight = float(self.RECOMMENDATION_CONFIG["selection_relevance_weight"])
        diversity_weight = float(self.RECOMMENDATION_CONFIG["selection_diversity_weight"])
        semantic_component_weight = float(self.RECOMMENDATION_CONFIG["semantic_diversity_component_weight"])
        cluster_component_weight = float(self.RECOMMENDATION_CONFIG["cluster_diversity_component_weight"])
        semantic_repeat_threshold = float(self.RECOMMENDATION_CONFIG["semantic_similarity_penalty_threshold"])
        cluster_repeat_divisor = self.ENHANCED_RETRIEVAL_CONFIG["cluster_repeat_divisor"]
        category_repeat_divisor = self.ENHANCED_RETRIEVAL_CONFIG["category_repeat_divisor"]

        def relevance_of(candidate: Dict[str, Any]) -> float:
            return float(candidate.get("relevance_score", candidate.get("final_score", 0.0)) or 0.0)

        def candidate_cluster_id(candidate: Dict[str, Any]) -> Optional[str]:
            # 优先使用排序阶段算出的最佳匹配簇，其次用召回阶段的主命中簇。
            cluster_id = str(candidate.get("best_matched_cluster_id", "") or "").strip()
            if not cluster_id:
                cluster_id = str(candidate.get("recall_cluster_id", "") or "").strip()
            return cluster_id or None

        # ---- 预计算：相关性归一化、簇 ID、分类、单位向量 ----
        relevance_scores = [relevance_of(candidate) for candidate in candidates]
        min_relevance = min(relevance_scores)
        max_relevance = max(relevance_scores)
        relevance_span = max_relevance - min_relevance
        # 归一化到 [0, 1]，使相关性与多样性在同一量纲上加权；所有候选同分时统一视为 1。
        normalized_relevance = [
            1.0 if relevance_span <= 0 else max(0.0, min(1.0, (score - min_relevance) / relevance_span))
            for score in relevance_scores
        ]
        cluster_ids = [candidate_cluster_id(candidate) for candidate in candidates]
        candidate_categories = [list(candidate.get("_candidate_categories", []) or []) for candidate in candidates]

        unit_vectors: List[Optional[np.ndarray]] = []
        indexes_by_dimension: Dict[int, List[int]] = {}
        for index, candidate in enumerate(candidates):
            embedding = candidate.get("_candidate_embedding")
            if not embedding:
                unit_vectors.append(None)
                continue
            vector = np.asarray(embedding, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            # 零向量与任何向量的相似度都按 0 处理，和 _cosine_similarity 保持一致。
            unit_vectors.append(vector / norm if norm > 0 else np.zeros_like(vector))
            indexes_by_dimension.setdefault(vector.shape[0], []).append(index)
        # 同维度候选堆成矩阵，入选一篇论文时一次矩阵乘法即可得到它与所有同维候选的相似度；
        # 维度不同的候选相似度按 0 处理（与 _cosine_similarity 的维度不一致策略一致）。
        dimension_groups = {
            dimension: (np.asarray(indexes, dtype=np.int64), np.vstack([unit_vectors[index] for index in indexes]))
            for dimension, indexes in indexes_by_dimension.items()
        }
        # 每个候选与已选集合的最大语义相似度；只有已选集合中出现过向量后才有意义。
        max_selected_similarity = np.zeros(candidate_count, dtype=np.float64)
        has_selected_embedding = False

        def candidate_priority(index: int) -> tuple[float, float, float, float, str]:
            candidate = candidates[index]
            return (
                normalized_relevance[index],
                relevance_scores[index],
                float(candidate.get("semantic_score", 0.0) or 0.0),
                float(candidate.get("final_score", 0.0) or 0.0),
                str(candidate.get("arxiv_id", "") or ""),
            )

        selected: List[Dict[str, Any]] = []
        selected_cluster_counts: Counter = Counter()
        selected_category_counts: Counter = Counter()
        remaining: List[int] = list(range(candidate_count))

        def annotate_selected_candidate(
            candidate: Dict[str, Any],
            selection_rank: int,
            selection_score: float,
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
                    "selection_score": selection_score,
                }
            )
            candidate["selection_rank"] = selection_rank
            candidate["relevance_score"] = relevance_score
            candidate["selection_score"] = selection_score
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

        def compute_diversity(index: int) -> tuple[float, Optional[float], Optional[float], Optional[float], str, Optional[float]]:
            """计算候选相对当前已选集合的多样性。

            Returns:
                (综合多样性, 语义多样性, 簇多样性, 分类多样性, 多样性原因, 语义相似度惩罚)。
                不可用的分项为 None。
            """
            has_embedding = unit_vectors[index] is not None
            semantic_diversity_score: Optional[float] = None
            semantic_similarity_penalty: Optional[float] = None
            if has_embedding and has_selected_embedding:
                # 向量越接近已选项，语义多样性越低，需要在选择阶段适度压分。
                semantic_similarity_penalty = float(max_selected_similarity[index])
                semantic_diversity_score = 1.0 - semantic_similarity_penalty

            cluster_id = cluster_ids[index]
            cluster_diversity_score: Optional[float] = None
            cluster_repeat_count: Optional[int] = None
            if cluster_id and interest_clusters:
                # 同簇已选数量越多，簇多样性越低；已选数达到 cluster_repeat_divisor 时降为 0。
                cluster_repeat_count = int(selected_cluster_counts.get(cluster_id, 0))
                cluster_diversity_score = max(0.0, 1.0 - min(cluster_repeat_count / cluster_repeat_divisor, 1.0))

            categories = candidate_categories[index]
            category_diversity_score: Optional[float] = None
            category_repeat_count: Optional[int] = None
            if not has_embedding and cluster_diversity_score is None and categories:
                # 没有向量和簇信息时，用分类重复度作为最后一道保底多样性信号。
                category_repeat_count = max((int(selected_category_counts.get(category, 0)) for category in categories), default=0)
                category_diversity_score = max(0.0, 1.0 - min(category_repeat_count / category_repeat_divisor, 1.0))

            components: List[tuple[float, float]] = []
            if semantic_diversity_score is not None:
                components.append((semantic_diversity_score, semantic_component_weight))
            if cluster_diversity_score is not None:
                components.append((cluster_diversity_score, cluster_component_weight))
            if not components and category_diversity_score is not None:
                components.append((category_diversity_score, 1.0))

            if components:
                total_weight = sum(weight for _, weight in components)
                diversity_score = sum(value * weight for value, weight in components) / total_weight
                diversity_reason = "semantic_cluster_mix" if len(components) > 1 else ("semantic_guidance" if semantic_diversity_score is not None else "cluster_guidance")
            else:
                diversity_score = 1.0
                diversity_reason = "no_previous_selection"

            # 原因标签优先报告“发生了重复”的情况，便于前端 / 调试解释为什么被压分。
            if semantic_similarity_penalty is not None and semantic_similarity_penalty >= semantic_repeat_threshold:
                diversity_reason = "semantic_repeat"
            elif cluster_repeat_count and cluster_repeat_count > 0:
                diversity_reason = "cluster_repeat"
            elif category_repeat_count and category_repeat_count > 0:
                diversity_reason = "category_repeat"

            return diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, semantic_similarity_penalty

        def commit_selected_candidate(
            index: int,
            selection_rank: int,
            selection_score: float,
            diversity_reason: str,
            diversity_score: float,
            semantic_diversity_score: Optional[float],
            cluster_diversity_score: Optional[float],
            category_diversity_score: Optional[float],
            diversity_penalty_source: Optional[str] = None,
            diversity_penalty_value: Optional[float] = None,
        ) -> None:
            """提交一个候选到结果集，并同步更新后续多样性计算所需状态。"""
            nonlocal has_selected_embedding
            candidate = candidates[index]
            remaining.remove(index)
            annotate_selected_candidate(candidate, selection_rank, selection_score, relevance_scores[index], diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, diversity_penalty_source, diversity_penalty_value)
            unit_vector = unit_vectors[index]
            if unit_vector is not None:
                has_selected_embedding = True
                group = dimension_groups.get(unit_vector.shape[0])
                if group is not None:
                    # 增量维护“与已选集合的最大相似度”：只需和新入选的这一篇比较。
                    group_indexes, group_matrix = group
                    similarities = np.clip(group_matrix @ unit_vector, 0.0, 1.0)
                    max_selected_similarity[group_indexes] = np.maximum(max_selected_similarity[group_indexes], similarities)
            cluster_id = cluster_ids[index]
            if cluster_id:
                selected_cluster_counts[cluster_id] += 1
            for category in candidate_categories[index]:
                selected_category_counts[category] += 1
            # 已入选的候选不会再参与多样性计算，此时可以安全地清理临时字段。
            selected.append(self._strip_ranking_temp_fields(candidate))

        if interest_clusters:
            # 第一步：每个兴趣簇挑出优先级最高的候选作为代表。
            cluster_representatives: Dict[str, int] = {}
            for index in remaining:
                cluster_id = cluster_ids[index]
                if not cluster_id:
                    continue
                existing_index = cluster_representatives.get(cluster_id)
                if existing_index is None or candidate_priority(index) > candidate_priority(existing_index):
                    cluster_representatives[cluster_id] = index

            # 先为每个兴趣簇选一个代表作种子，避免结果被单一兴趣方向垄断。
            ordered_cluster_indexes = sorted(cluster_representatives.values(), key=candidate_priority, reverse=True)
            for index in ordered_cluster_indexes:
                if len(selected) >= top_n:
                    break
                commit_selected_candidate(index, len(selected) + 1, normalized_relevance[index], "cluster_seed", 1.0, None, None, None)

        if not selected and remaining:
            # 没有兴趣簇种子时，用相关性最高的候选作为唯一种子。
            first_index = max(remaining, key=candidate_priority)
            commit_selected_candidate(first_index, 1, normalized_relevance[first_index], "seed", 1.0, None, None, None)

        # 第二步：贪心地每轮选出 selection_score 最高的候选。
        while remaining and len(selected) < top_n:
            best_index: Optional[int] = None
            best_selection_score = float("-inf")
            best_diversity_data = (1.0, None, None, None, "no_previous_selection", None)
            for index in remaining:
                diversity_data = compute_diversity(index)
                # 以加权方式融合相关性与多样性，做逐步贪心选择。
                selection_score = normalized_relevance[index] * relevance_weight + diversity_data[0] * diversity_weight
                if selection_score > best_selection_score:
                    best_index = index
                    best_selection_score = selection_score
                    best_diversity_data = diversity_data

            if best_index is None:
                break

            diversity_score, semantic_diversity_score, cluster_diversity_score, category_diversity_score, diversity_reason, semantic_similarity_penalty = best_diversity_data
            if semantic_similarity_penalty is not None:
                diversity_penalty_source: Optional[str] = "semantic"
                diversity_penalty_value = semantic_similarity_penalty
            else:
                diversity_penalty_source = "cluster" if cluster_diversity_score is not None else ("category" if category_diversity_score is not None else None)
                diversity_penalty_value = max(0.0, 1.0 - diversity_score)
            commit_selected_candidate(
                index=best_index,
                selection_rank=len(selected) + 1,
                selection_score=best_selection_score,
                diversity_reason=diversity_reason,
                diversity_score=diversity_score,
                semantic_diversity_score=semantic_diversity_score,
                cluster_diversity_score=cluster_diversity_score,
                category_diversity_score=category_diversity_score,
                diversity_penalty_source=diversity_penalty_source,
                diversity_penalty_value=diversity_penalty_value,
            )

        # 入选集合确定后按综合分展示；同分时保留入选顺序（sort 稳定）。
        selected.sort(key=lambda item: float(item.get("final_score", 0.0) or 0.0), reverse=True)
        return selected

    # ------------------------------------------------------------------
    # 分项分数
    # ------------------------------------------------------------------

    def _calculate_category_score(self, categories: List[str], liked_category_freq: Counter) -> float:
        """根据候选分类与用户已点赞分类分布的重叠程度，计算分类偏好分（命中分类频次之和 / 总频次，上限 1）。"""
        if not categories or not liked_category_freq:
            return 0.0
        total_frequency = sum(liked_category_freq.values())
        if total_frequency <= 0:
            return 0.0
        matched_frequency = sum(liked_category_freq.get(category, 0) for category in categories)
        return min(1.0, matched_frequency / total_frequency)

    def _calculate_recency_score(self, published_date: Any) -> float:
        """把发布时间衰减为 0 到 1 之间的新鲜度分值：1 / (1 + 天数 / 365)，即一年前的论文得 0.5。"""
        parsed_date = self._parse_datetime(str(published_date or ""))
        if not parsed_date:
            return 0.0
        age_days = max(0, (datetime.now(timezone.utc) - parsed_date).days)
        return 1.0 / (1.0 + age_days / 365.0)

    # ------------------------------------------------------------------
    # 查询匹配（搜索重排与上下文推荐共用）
    # ------------------------------------------------------------------

    def _extract_query_terms(self, text: Any) -> List[str]:
        """从查询文本中抽取去重后的有效词项，并过滤停用词与噪声符号。

        英文按字母数字串切分（保留 + - _ / . 以兼容 "gpt-4"、"cs.CL" 这类词），中文取连续 2 字及以上片段。
        """
        normalized = str(text or "").strip().lower()
        if not normalized:
            return []

        raw_terms = re.findall(r"[a-z0-9][a-z0-9+\-_/\.]*|[一-鿿]{2,}", normalized)
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
        """计算候选论文与搜索查询之间的匹配分，并输出命中明细。

        query_match_score = max(查询词覆盖率, 上游搜索分) * 0.55 + 标题词覆盖率 * 0.2
                            + 摘要词覆盖率 * 0.15 + 分类覆盖率 * 0.1 + 整句命中奖励，截断到 [0, 1]。

        词项命中按词边界判断（见 text_contains_term），避免短词（如 "rag"）误命中 "storage"。
        """
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
        title_text = title.lower()
        abstract_text = abstract.lower()
        query_hits = [term for term in query_terms if text_contains_term(candidate_text, term)]
        title_hits = [term for term in title_terms if text_contains_term(title_text, term)]
        abstract_hits = [term for term in abstract_terms if text_contains_term(abstract_text, term)]
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

        # 标题 / 摘要查询作为完整短语出现时给额外奖励。
        phrase_bonus = 0.0
        if title_query and text_contains_term(title_text, title_query):
            phrase_bonus += 0.1
        if abstract_query and text_contains_term(abstract_text, abstract_query):
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

    # ------------------------------------------------------------------
    # 推荐理由文案
    # ------------------------------------------------------------------

    def _build_match_reason(self, query_match_score: float, matched_terms: List[str]) -> str:
        """把查询命中情况转换为面向前端展示的简短说明文案（最多列出 3 个命中词）。"""
        score_text = f"{round(max(0.0, min(1.0, query_match_score)) * 100)}%"
        if matched_terms:
            preview = ", ".join(matched_terms[:3])
            return f"当前查询命中 {preview}，匹配度 {score_text}"
        return f"当前查询相关度 {score_text}"

    def _build_personalized_reason(self, candidate: Dict[str, Any], score_breakdown: Dict[str, Any]) -> str:
        """根据排序分解结果生成个性化推荐理由，方便前端直接展示。

        语义阈值基于 max(0, cos) 口径：≥ 0.5 视为语义相近，≥ 0.3 视为有一定相关。
        """
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
        if semantic_score >= 0.5:
            parts.append(f"与兴趣向量语义相似度 {round(semantic_score * 100)}%")
        elif semantic_score >= 0.3:
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

    # ------------------------------------------------------------------
    # 通用工具
    # ------------------------------------------------------------------

    def _cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """余弦相似度，负值截断为 0，结果落在 [0, 1]：max(0, cos)。

        与 Milvus COSINE 检索分数同口径。之前使用 (cos + 1) / 2 映射，会把“无关”抬到 0.5，
        并把语义差异压缩一半，导致各类阈值都很“松”。

        两向量维度不同通常意味着 embedding 模型被更换过，截断比较没有语义意义，
        因此直接返回 0 并告警一次，提示需要重建兴趣画像与候选向量。
        """
        if not vector_a or not vector_b:
            return 0.0
        if len(vector_a) != len(vector_b):
            dimension_pair = (len(vector_a), len(vector_b))
            if dimension_pair not in _WARNED_DIMENSION_MISMATCHES:
                _WARNED_DIMENSION_MISMATCHES.add(dimension_pair)
                logger.warning(
                    "Embedding dimension mismatch (%s vs %s); similarity treated as 0. "
                    "Rebuild interest vectors / paper embeddings after switching embedding models.",
                    *dimension_pair,
                )
            return 0.0
        dot_product = math.fsum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(math.fsum(a * a for a in vector_a))
        norm_b = math.sqrt(math.fsum(b * b for b in vector_b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        cosine = dot_product / (norm_a * norm_b)
        return max(0.0, min(1.0, cosine))

    def _split_categories(self, categories: Any) -> List[str]:
        """把各种形态的分类字段解析为去重后的分类列表。

        支持：list / tuple；Python 字面量字符串（"['cs.AI', 'cs.LG']"）；
        以逗号、分号、竖线或空白分隔的字符串（"cs.AI cs.LG"、"cs.AI, cs.LG"）。
        """
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
        """宽松解析日期时间，统一返回 UTC aware datetime；无法解析时返回 None。

        依次尝试：完整 ISO 字符串、把 "Z" 换成 "+00:00" 的 ISO 字符串、前 10 位日期（YYYY-MM-DD），
        最后尝试 "YYYY/MM/DD" 格式。无时区信息的时间按 UTC 处理。
        """
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
        try:
            return datetime.strptime(text[:10], "%Y/%m/%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
