from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional

from fastapi import HTTPException

from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.memory import MemoryService
from services.memory.concept_normalizer import ConceptNormalizer
from services.memory.paper_evidence_extractor import PAPER_EVIDENCE_EXTRACTOR_VERSION
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
        memory_service: Optional[MemoryService] = None,
        get_clustering_config: Optional[Callable[[], Dict[str, Any]]] = None,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        oai_db_service: Optional[ArxivOaiDatabaseService] = None,
        collection_name: str = "arxiv_paper_embeddings",
    ):
        """组装推荐链路依赖，并初始化推荐与回填流程所需的运行时状态。"""
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
        """读取记忆模块的运行时开关，避免在推荐主流程中散落硬编码配置访问。"""
        return self.memory_runtime_config.get(key, default)

    @staticmethod
    def _normalize_text_terms(values: Any) -> List[str]:
        """把字符串或字符串列表规范化为小写词项列表，便于后续执行包含匹配。"""
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
        """收集长期研究画像与行为侧信号，并给出需要排除的论文 ID 集合。"""
        if not bool(self._memory_flag("enable_user_research_profile", False)):
            return {
                "profile": {},
                "actions": self.db_service.get_user_paper_action_map(user_id),
                "excluded_ids": [],
                "disabled": True,
            }
        profile = self.db_service.get_user_research_profile(user_id)
        actions = self.db_service.get_user_paper_action_map(user_id)
        explicit_preference_ids = [
            *self.db_service.get_liked_papers(user_id),
            *self.db_service.get_disliked_papers(user_id),
        ]
        excluded_ids = list(
            dict.fromkeys(
                [
                    # like/dislike 是强偏好权威状态，不再从弱行为 action_map 读取。
                    *self._normalize_text_terms(explicit_preference_ids),
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
        """根据 effective profile 的 canonical topic 与候选论文 evidence card 做语义匹配。"""
        profile = profile or {}
        actions = actions or {}
        arxiv_id = str(candidate.get("arxiv_id", "") or candidate.get("id", "") or "").strip()
        categories = {
            str(item).strip().lower()
            for item in (candidate.get("categories") if isinstance(candidate.get("categories"), list) else str(candidate.get("categories") or "").split(","))
            if str(item).strip()
        }

        positive_topics = self._profile_topic_objects(profile, positive=True)
        negative_topics = self._profile_topic_objects(profile, positive=False)
        preferred_categories = {item.lower() for item in self._normalize_text_terms(profile.get("preferred_categories"))}
        candidate_concepts = self._candidate_profile_concepts(candidate)

        matched_positive = self._match_profile_topics_to_candidate(positive_topics, candidate_concepts)
        matched_negative = self._match_profile_topics_to_candidate(negative_topics, candidate_concepts)
        matched_categories = sorted(categories & preferred_categories)

        action_boost = 0.0
        if arxiv_id and arxiv_id in self._normalize_text_terms(actions.get("favorite", [])):
            action_boost += 0.08
        if arxiv_id and arxiv_id in self._normalize_text_terms(actions.get("later", [])):
            action_boost += 0.04
        if arxiv_id and arxiv_id in self._normalize_text_terms(actions.get("read", [])):
            action_boost -= 0.03

        profile_score = min(0.24, sum(float(item.get("match_score") or 0.0) for item in matched_positive) * 0.05 + len(matched_categories) * 0.03 + action_boost)
        profile_penalty = min(0.24, sum(float(item.get("match_score") or 0.0) for item in matched_negative) * 0.07)
        net_adjustment = profile_score - profile_penalty

        reasons: List[str] = []
        if matched_positive:
            reasons.append(self._build_profile_match_reason(matched_positive[:3], profile))
        if matched_categories:
            reasons.append(f"匹配偏好分类: {', '.join(matched_categories[:3])}")
        if matched_negative:
            reasons.append(f"避开负向主题: {', '.join(item['topic'] for item in matched_negative[:3])}")

        return {
            "profile_score": profile_score,
            "profile_penalty": profile_penalty,
            "profile_adjustment": net_adjustment,
            "matched_positive_topics": [item["topic"] for item in matched_positive],
            "matched_negative_topics": [item["topic"] for item in matched_negative],
            "matched_preferred_categories": matched_categories,
            "profile_match_details": matched_positive,
            "profile_negative_match_details": matched_negative,
            "profile_match_score": profile_score,
            "profile_reasons": reasons,
        }

    def _profile_topic_objects(self, profile: Dict[str, Any], *, positive: bool) -> List[Dict[str, Any]]:
        canonical_key = "canonical_topics" if positive else "canonical_negative_topics"
        fallback_key = "positive_topics" if positive else "negative_topics"
        topics: List[Dict[str, Any]] = []
        for item in profile.get(canonical_key) or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if label:
                topics.append({"label": label, "aliases": item.get("aliases") or [], "source": "canonical"})
        known = {item["label"].lower() for item in topics}
        for label in profile.get(fallback_key) or []:
            text = str(label or "").strip()
            if text and text.lower() not in known:
                topics.append({"label": text, "aliases": [], "source": "legacy_projection"})
        return topics

    def _candidate_profile_concepts(self, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        arxiv_id = str(candidate.get("arxiv_id", "") or candidate.get("id", "") or "").strip()
        card = None
        if arxiv_id and hasattr(self.db_service, "get_paper_profile_evidence"):
            card = self.db_service.get_paper_profile_evidence(arxiv_id, extractor_version=PAPER_EVIDENCE_EXTRACTOR_VERSION)
        if not isinstance(card, dict):
            card = candidate.get("evidence_card") if isinstance(candidate.get("evidence_card"), dict) else None
        concepts: List[Dict[str, Any]] = []
        for item in (card or {}).get("candidate_concepts") or []:
            if not isinstance(item, dict) or not item.get("whether_generalizable", True):
                continue
            label = ConceptNormalizer.clean_label(item.get("label"))
            if label:
                concepts.append({"label": label, "confidence": float(item.get("confidence") or 0.6), "source": "paper_evidence_card"})
        for label in candidate.get("technical_concepts") or candidate.get("concepts") or []:
            clean = ConceptNormalizer.clean_label(label)
            if clean:
                concepts.append({"label": clean, "confidence": 0.55, "source": "candidate_payload"})
        if not concepts:
            # 只有 evidence card 缺失时才启用低置信度文本 fallback，避免推荐主路径退回字符串包含匹配。
            text = f"{candidate.get('title', '')} {candidate.get('abstract', '') or candidate.get('summary', '')}"
            for topic in self.memory_service.profile_generator.extract_topics(text):
                concepts.append({"label": topic, "confidence": 0.35, "source": "low_confidence_text_fallback"})
        return concepts

    def _match_profile_topics_to_candidate(self, profile_topics: List[Dict[str, Any]], candidate_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for topic in profile_topics:
            labels = [str(topic.get("label") or "").strip(), *[str(item or "").strip() for item in topic.get("aliases") or []]]
            best: Optional[Dict[str, Any]] = None
            for concept in candidate_concepts:
                concept_label = str(concept.get("label") or "").strip()
                scores = [self._topic_similarity(label, concept_label) for label in labels if label]
                score = max(scores) if scores else 0.0
                if score < 0.55:
                    continue
                candidate_match = {
                    "topic": str(topic.get("label") or "").strip(),
                    "candidate_concept": concept_label,
                    "match_score": round(score * float(concept.get("confidence") or 0.6), 4),
                    "source": concept.get("source") or "paper_evidence_card",
                }
                if best is None or candidate_match["match_score"] > best["match_score"]:
                    best = candidate_match
            if best and best["topic"].lower() not in seen:
                seen.add(best["topic"].lower())
                matches.append(best)
        matches.sort(key=lambda item: (-float(item.get("match_score") or 0.0), item["topic"].lower()))
        return matches

    @staticmethod
    def _topic_similarity(left: str, right: str) -> float:
        left_key = ConceptNormalizer._topic_key(left)
        right_key = ConceptNormalizer._topic_key(right)
        if not left_key or not right_key:
            return 0.0
        if left_key == right_key or left_key in right_key or right_key in left_key:
            return 1.0
        left_tokens = set(left_key.split())
        right_tokens = set(right_key.split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _build_profile_match_reason(self, matches: List[Dict[str, Any]], profile: Dict[str, Any]) -> str:
        topics = [item["topic"] for item in matches if item.get("topic")]
        evidence = profile.get("topic_evidence") if isinstance(profile.get("topic_evidence"), dict) else {}
        evidence_parts: List[str] = []
        for topic in topics[:2]:
            item = evidence.get(topic) or {}
            paper_count = len(item.get("source_papers") or [])
            note_count = len(item.get("source_notes") or [])
            if paper_count or note_count:
                evidence_parts.append(f"{topic}（{paper_count} 篇论文，{note_count} 条笔记）")
        if evidence_parts:
            return f"匹配你的长期兴趣：{', '.join(evidence_parts)}"
        return f"匹配你的长期兴趣：{', '.join(topics[:3])}"

    def _get_or_refresh_interest_vector(self, user_id: str) -> Dict[str, Any]:
        """读取用户兴趣向量；如果缺失或过期，则自动触发重建。"""
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

    def _coerce_mapping(self, value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    def _coerce_candidate_list(self, value: Any) -> List[Dict[str, Any]]:
        return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    def _resolve_effective_research_profile(
        self,
        *,
        stored_profile: Optional[Dict[str, Any]],
        request_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        effective = dict(stored_profile or {})
        # 当前请求显式传入的画像字段优先，避免 Agent 已经识别出的会话约束在服务层被旧缓存覆盖。
        for key, value in dict(request_profile or {}).items():
            if value in (None, "", [], {}):
                continue
            effective[key] = value
        return effective

    def _build_recommendation_context_bundle(
        self,
        *,
        message: Optional[str],
        topic_hint: Optional[str],
        research_profile: Optional[Dict[str, Any]],
        request_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = self._coerce_mapping(request_context)
        query_text = str(topic_hint or message or context.get("query") or "").strip()
        positive_topics = self._normalize_text_terms(context.get("positive_topics"))
        negative_topics = self._normalize_text_terms(context.get("negative_topics"))
        category_constraints = self._normalize_text_terms(context.get("category_constraints"))
        recent_papers = self._coerce_candidate_list(context.get("recent_papers"))
        recent_text = " ".join(
            filter(
                None,
                [
                    f"{paper.get('title', '')} {paper.get('abstract', '') or paper.get('summary', '')}"
                    for paper in recent_papers[:5]
                    if isinstance(paper, dict)
                ],
            )
        )
        recent_topics = self._extract_query_terms(recent_text)
        profile = research_profile or {}
        profile_negative_topics = self._normalize_text_terms(profile.get("negative_topics"))
        profile_positive_topics = self._normalize_text_terms(profile.get("positive_topics"))
        if not positive_topics:
            positive_topics = profile_positive_topics[:]
        cold_start = bool(context.get("force_cold_start")) or (not bool(query_text) and not bool(profile_positive_topics))
        return {
            "query_text": query_text,
            "topic_hint": str(topic_hint or "").strip() or None,
            "positive_topics": positive_topics,
            "negative_topics": list(dict.fromkeys([*negative_topics, *profile_negative_topics])),
            "category_constraints": category_constraints,
            "recent_papers": recent_papers,
            "recent_topics": recent_topics,
            "temporary_requirements": self._normalize_text_terms(context.get("temporary_requirements")),
            "cold_start": cold_start,
            "user_id_provided": bool(context.get("user_id_provided")),
        }

    def _merge_candidate_sources(
        self,
        *,
        recalled_candidates: List[Dict[str, Any]],
        context_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        # 会话里最近出现过的论文会先进入候选池，后续仍由统一排序逻辑决定是否留下，避免 Agent 和 Service 重复做两套推荐。
        for source_name, source_candidates in (("context", context_candidates), ("recall", recalled_candidates)):
            for candidate in source_candidates:
                arxiv_id = str(candidate.get("arxiv_id", "") or candidate.get("id", "") or "").strip()
                if not arxiv_id or arxiv_id in seen:
                    continue
                seen.add(arxiv_id)
                normalized = self._normalize_paper_record(dict(candidate), arxiv_id)
                merged.append({**candidate, **normalized, "candidate_source": source_name})
        return merged

    def _compute_contextual_adjustment(
        self,
        candidate: Dict[str, Any],
        *,
        context_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_breakdown = self._build_query_match_score(
            candidate,
            query=context_bundle.get("query_text"),
            search_categories=context_bundle.get("category_constraints") or [],
        )
        candidate_text = " ".join(
            [
                str(candidate.get("title", "") or ""),
                str(candidate.get("abstract", "") or candidate.get("summary", "") or ""),
                " ".join(self._split_categories(candidate.get("categories"))),
                " ".join(str(item.get("label") or "") for item in self._candidate_profile_concepts(candidate)),
            ]
        ).lower()
        positive_hits = [term for term in context_bundle.get("positive_topics", []) if term and term in candidate_text]
        negative_hits = [term for term in context_bundle.get("negative_topics", []) if term and term in candidate_text]
        recent_hits = [term for term in context_bundle.get("recent_topics", []) if term and term in candidate_text]
        category_constraints = {item.lower() for item in context_bundle.get("category_constraints", [])}
        candidate_categories = {item.lower() for item in self._split_categories(candidate.get("categories"))}
        category_hits = sorted(category_constraints & candidate_categories)
        category_mismatch_penalty = 0.12 if category_constraints and not category_hits else 0.0
        recent_behavior_score = min(0.18, len(recent_hits) * 0.03)
        request_score = min(
            0.38,
            float(query_breakdown.get("query_match_score", 0.0) or 0.0) * 0.24
            + len(positive_hits) * 0.05
            + recent_behavior_score
            + len(category_hits) * 0.03,
        )
        request_penalty = min(0.45, len(negative_hits) * 0.15 + category_mismatch_penalty)
        # 显式负向主题优先级高于泛化查询词命中，避免“用户说不要 vision，但因为也提到 agent 仍被保留”。
        exclude_candidate = bool(negative_hits and not positive_hits)
        negative_filter_reason = None
        if exclude_candidate:
            negative_filter_reason = f"matched negative topics: {', '.join(negative_hits[:3])}"
        elif category_mismatch_penalty > 0:
            negative_filter_reason = "category constraints not matched"
        sources: List[str] = []
        if query_breakdown.get("matched_terms") or positive_hits:
            sources.append("current_request")
        if recent_hits:
            sources.append("recent_behavior")
        if context_bundle.get("cold_start"):
            sources.append("cold_start")
        return {
            "request_score": request_score,
            "request_penalty": request_penalty,
            "query_match_score": float(query_breakdown.get("query_match_score", 0.0) or 0.0),
            "matched_query_terms": list(query_breakdown.get("matched_terms") or []),
            "matched_request_topics": positive_hits,
            "matched_negative_terms": negative_hits,
            "matched_category_constraints": category_hits,
            "recent_behavior_score": recent_behavior_score,
            "recent_behavior_terms": recent_hits,
            "exclude_candidate": exclude_candidate,
            "negative_filter_reason": negative_filter_reason,
            "query_score_breakdown": dict(query_breakdown.get("query_score_breakdown") or {}),
            "context_sources": sources,
        }

    def _build_recommendation_explanation(self, candidate: Dict[str, Any]) -> str:
        parts: List[str] = []
        matched_profile_terms = list(candidate.get("matched_profile_terms") or [])
        matched_query_terms = list(candidate.get("matched_query_terms") or [])
        recent_terms = list(candidate.get("recent_behavior_terms") or [])
        if matched_profile_terms:
            parts.append(f"长期兴趣命中 {', '.join(matched_profile_terms[:3])}")
        if matched_query_terms:
            parts.append(f"当前请求命中 {', '.join(matched_query_terms[:3])}")
        if recent_terms:
            parts.append(f"最近交互相关 {', '.join(recent_terms[:3])}")
        if not parts:
            if candidate.get("recommendation_basis", {}).get("cold_start"):
                parts.append("主要依据当前主题与最新候选进行冷启动推荐")
            else:
                parts.append("依据综合相关度、多样性和用户画像排序")
        return "；".join(parts)

    def recommend_papers_with_context(
        self,
        user_id: str,
        top_n: int = RECOMMENDATION_CONFIG["default_top_n"],
        max_age_months: int = RECOMMENDATION_CONFIG["default_max_age_months"],
        message: Optional[str] = None,
        topic_hint: Optional[str] = None,
        user_memory_summary: Optional[Any] = None,
        research_profile: Optional[Dict[str, Any]] = None,
        request_context: Optional[Dict[str, Any]] = None,
        candidate_papers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """在不破坏旧接口的前提下，为 Agent 推荐链路提供真正消费上下文的推荐实现。"""
        logger.info("Starting context-aware recommendation for user %s with top_n=%s max_age_months=%s", user_id, top_n, max_age_months)

        preferences = self.db_service.get_user_preferences(user_id=user_id)
        liked_ids = preferences.get("liked_papers", [])
        disliked_ids = preferences.get("disliked_papers", [])
        profile_bundle = self._build_profile_signal_bundle(user_id)
        effective_profile = self._resolve_effective_research_profile(
            stored_profile=profile_bundle.get("profile", {}),
            request_profile=research_profile,
        )
        paper_actions = profile_bundle.get("actions", {})
        excluded_ids = list(dict.fromkeys([*liked_ids, *disliked_ids, *profile_bundle.get("excluded_ids", [])]))
        liked_details = self.db_service.get_liked_papers_with_details(user_id=user_id)
        liked_category_freq = self._build_liked_category_frequency(liked_details)
        context_bundle = self._build_recommendation_context_bundle(
            message=message,
            topic_hint=topic_hint,
            research_profile=effective_profile,
            request_context=request_context,
        )

        user_vector_data: Dict[str, Any] = {"profile_mode": "cold_start", "cluster_count": 0}
        user_vector = None
        interest_clusters: List[Dict[str, Any]] = []
        disliked_vector = None
        negative_feedback_profile: Dict[str, Any] = {}
        personalized_available = not bool(context_bundle.get("cold_start"))
        if personalized_available:
            try:
                user_vector_data = self._get_or_refresh_interest_vector(user_id)
                user_vector = user_vector_data["vector_data"]
                interest_clusters = user_vector_data.get("interest_clusters", []) or []
                disliked_vector = user_vector_data.get("disliked_vector_data")
                negative_feedback_profile = user_vector_data.get("negative_feedback_profile", {}) or {}
            except Exception as exc:
                # 只要当前请求能提供足够主题信号，就允许安全退化成冷启动主题推荐，而不是直接报错。
                logger.info("Interest vector unavailable for user %s, falling back to context-driven recommendation: %s", user_id, exc)
                personalized_available = False
                user_vector_data = {"profile_mode": "cold_start", "cluster_count": 0, "fallback_reason": str(exc)}

        candidate_limit = max(top_n * 5, 50)
        cluster_recall_candidates: List[Dict[str, Any]] = []
        if personalized_available and interest_clusters:
            cluster_recall_candidates = self._fetch_cluster_recall_candidates(
                interest_clusters=interest_clusters,
                excluded_ids=excluded_ids,
                top_k=max(top_n * 3, 20),
            )

        if cluster_recall_candidates:
            recalled_candidates = cluster_recall_candidates
            recall_mode = "cluster_recall"
        else:
            recalled_candidates = self._fetch_recent_db_candidates(
                liked_category_freq=liked_category_freq,
                max_age_months=max_age_months,
                max_results=max(candidate_limit * 2, candidate_limit),
            )
            recall_mode = "recent_pool" if personalized_available else "cold_start_recent_pool"

        merged_candidates = self._merge_candidate_sources(
            recalled_candidates=recalled_candidates,
            context_candidates=self._coerce_candidate_list(candidate_papers),
        )
        filtered_candidates = self._deduplicate_candidates(merged_candidates, excluded_ids)
        if not filtered_candidates:
            raise HTTPException(status_code=400, detail=f"No papers found for recommendation within the last {max_age_months} months")

        materialized_candidates, materialize_stats = self._materialize_candidate_papers_for_recommendation(filtered_candidates)
        embedding_config = self.get_embedding_config() if personalized_available else None
        candidate_ids = [str(candidate.get("arxiv_id", "") or "").strip() for candidate in materialized_candidates]
        candidate_embedding_map: Dict[str, List[float]] = {}
        if personalized_available and candidate_ids:
            existing_candidate_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
                collection_name=self.collection_name,
                arxiv_ids=candidate_ids,
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

        scored_candidates = []
        filtered_out_candidates = []
        recomputed_vector_count = 0
        missing_vector_count = 0
        for candidate in materialized_candidates:
            scored_candidate = self._build_candidate_score(
                candidate=candidate,
                liked_category_freq=liked_category_freq,
                user_vector=user_vector,
                interest_clusters=interest_clusters,
                disliked_vector=disliked_vector,
                negative_feedback_profile=negative_feedback_profile,
                embedding_config=embedding_config,
            )
            if scored_candidate.get("negative_hard_filter"):
                filtered_out_candidates.append(
                    {
                        "arxiv_id": scored_candidate.get("arxiv_id"),
                        "title": scored_candidate.get("title"),
                        "reason": "negative_feedback_hard_filter",
                        "negative_score": scored_candidate.get("negative_score"),
                        "negative_feedback_match": scored_candidate.get("negative_feedback_match"),
                    }
                )
                continue
            profile_adjustment = self._compute_profile_adjustment(
                scored_candidate,
                profile=effective_profile,
                actions=paper_actions,
            )
            context_adjustment = self._compute_contextual_adjustment(
                scored_candidate,
                context_bundle=context_bundle,
            )
            if context_adjustment["exclude_candidate"]:
                filtered_out_candidates.append(
                    {
                        "arxiv_id": scored_candidate.get("arxiv_id"),
                        "title": scored_candidate.get("title"),
                        "reason": context_adjustment["negative_filter_reason"],
                    }
                )
                continue

            profile_terms = list(profile_adjustment.get("matched_positive_topics") or [])
            query_terms = list(dict.fromkeys([*context_adjustment["matched_query_terms"], *context_adjustment["matched_request_topics"]]))
            basis = {
                "long_term_profile": bool(profile_terms),
                "current_request": bool(query_terms),
                "recent_behavior": bool(context_adjustment["recent_behavior_terms"]),
                "cold_start": bool(context_bundle.get("cold_start")),
            }

            scored_candidate["profile_score"] = profile_adjustment["profile_score"]
            scored_candidate["profile_penalty"] = profile_adjustment["profile_penalty"]
            scored_candidate["profile_reasons"] = profile_adjustment["profile_reasons"]
            scored_candidate["matched_positive_topics"] = profile_adjustment["matched_positive_topics"]
            scored_candidate["matched_negative_topics"] = profile_adjustment["matched_negative_topics"]
            scored_candidate["matched_preferred_categories"] = profile_adjustment["matched_preferred_categories"]
            scored_candidate["profile_match_details"] = profile_adjustment.get("profile_match_details", [])
            scored_candidate["profile_match_score"] = profile_adjustment.get("profile_match_score", profile_adjustment["profile_score"])
            scored_candidate["matched_profile_terms"] = profile_terms
            scored_candidate["matched_query_terms"] = query_terms
            scored_candidate["matched_request_topics"] = context_adjustment["matched_request_topics"]
            scored_candidate["matched_category_constraints"] = context_adjustment["matched_category_constraints"]
            scored_candidate["recent_behavior_score"] = context_adjustment["recent_behavior_score"]
            scored_candidate["recent_behavior_terms"] = context_adjustment["recent_behavior_terms"]
            scored_candidate["negative_filter_reason"] = context_adjustment["negative_filter_reason"]
            scored_candidate["recommendation_basis"] = basis
            scored_candidate["trace_sources"] = list(
                dict.fromkeys(
                    [
                        *context_adjustment["context_sources"],
                        *(["long_term_profile"] if basis["long_term_profile"] else []),
                    ]
                )
            )
            scored_candidate["relevance_score"] = (
                float(scored_candidate.get("relevance_score", 0.0) or 0.0)
                + profile_adjustment["profile_adjustment"]
                + context_adjustment["request_score"]
                - context_adjustment["request_penalty"]
            )
            scored_candidate["final_score"] = (
                float(scored_candidate.get("final_score", 0.0) or 0.0)
                + profile_adjustment["profile_adjustment"]
                + context_adjustment["request_score"]
                - context_adjustment["request_penalty"]
            )
            scored_candidate["match_reason"] = self._build_match_reason(context_adjustment["query_match_score"], query_terms)
            scored_candidate["personalized_reason"] = self._build_personalized_reason(scored_candidate, dict(scored_candidate.get("score_breakdown", {})))
            scored_candidate["recommendation_explanation"] = self._build_recommendation_explanation(scored_candidate)
            scored_candidate["score_components"] = {
                "semantic_score": float(scored_candidate.get("semantic_score", 0.0) or 0.0),
                "negative_score": float(scored_candidate.get("negative_score", 0.0) or 0.0),
                "negative_penalty": float(scored_candidate.get("negative_penalty", 0.0) or 0.0),
                "negative_confidence": float(scored_candidate.get("negative_confidence", 0.0) or 0.0),
                "profile_adjustment": profile_adjustment["profile_adjustment"],
                "request_score": context_adjustment["request_score"],
                "request_penalty": context_adjustment["request_penalty"],
                "recent_behavior_score": context_adjustment["recent_behavior_score"],
                "final_score": float(scored_candidate.get("final_score", 0.0) or 0.0),
            }
            score_breakdown = dict(scored_candidate.get("score_breakdown", {}))
            score_breakdown.update(context_adjustment["query_score_breakdown"])
            score_breakdown["profile_score"] = profile_adjustment["profile_score"]
            score_breakdown["profile_penalty"] = profile_adjustment["profile_penalty"]
            score_breakdown["request_score"] = context_adjustment["request_score"]
            score_breakdown["request_penalty"] = context_adjustment["request_penalty"]
            score_breakdown["query_match_score"] = context_adjustment["query_match_score"]
            score_breakdown["recent_behavior_score"] = context_adjustment["recent_behavior_score"]
            score_breakdown["relevance_score"] = scored_candidate["relevance_score"]
            score_breakdown["final_score"] = scored_candidate["final_score"]
            scored_candidate["score_breakdown"] = score_breakdown
            scored_candidates.append(scored_candidate)
            embedding_source = str(scored_candidate.get("_embedding_source", "") or "")
            if embedding_source == "recomputed":
                recomputed_vector_count += 1
            elif embedding_source == "missing":
                missing_vector_count += 1

        selected = self._select_diverse_candidates(
            scored_candidates,
            top_n,
            interest_clusters=interest_clusters,
        )
        for item in selected:
            item["score_components"]["diversity_score"] = float(item.get("diversity_score", 0.0) or 0.0)
            item["recommendation_explanation"] = self._build_recommendation_explanation(item)

        logger.info(
            "Context-aware recommendation finished for user %s: scored=%s selected=%s filtered=%s reused=%s recomputed=%s missing=%s",
            user_id,
            len(scored_candidates),
            len(selected),
            len(filtered_out_candidates),
            reused_vector_count,
            recomputed_vector_count,
            missing_vector_count,
        )

        return {
            "status": "success",
            "message": f"Generated {len(selected)} recommendations",
            "total_found": len(scored_candidates),
            "interest_profile_mode": user_vector_data.get("profile_mode", "cold_start"),
            "interest_cluster_count": user_vector_data.get("cluster_count", 0),
            "research_profile": effective_profile,
            "paper_actions": paper_actions,
            "recall_mode": recall_mode,
            "recommendation_context": {
                "query_text": context_bundle.get("query_text"),
                "topic_hint": context_bundle.get("topic_hint"),
                "positive_topics": context_bundle.get("positive_topics"),
                "negative_topics": context_bundle.get("negative_topics"),
                "category_constraints": context_bundle.get("category_constraints"),
                "recent_paper_count": len(context_bundle.get("recent_papers") or []),
                "cold_start": bool(context_bundle.get("cold_start")),
                "used_candidate_papers": bool(candidate_papers),
                "used_user_memory_summary": bool(user_memory_summary),
            },
            "filter_debug": {
                "excluded_ids_count": len(excluded_ids),
                "filtered_out_by_context": filtered_out_candidates,
                "negative_hard_filtered_count": sum(1 for item in filtered_out_candidates if item.get("reason") == "negative_feedback_hard_filter"),
                "materialize_stats": materialize_stats,
            },
            "ranking_debug": {
                "personalized_available": personalized_available,
                "context_candidate_count": len(candidate_papers or []),
                "request_query_used": bool(context_bundle.get("query_text")),
                "profile_topics_used": bool(effective_profile.get("positive_topics") or effective_profile.get("canonical_topics")),
            },
            "recommendations": selected,
        }

    def recommend_papers(
        self,
        user_id: str,
        top_n: int = RECOMMENDATION_CONFIG["default_top_n"],
        max_age_months: int = RECOMMENDATION_CONFIG["default_max_age_months"],
    ) -> Dict[str, Any]:
        """执行完整推荐流程，包括画像读取、候选召回、物化、打分与多样性筛选。"""
        user_vector_data = self._get_or_refresh_interest_vector(user_id)
        user_vector = user_vector_data["vector_data"]
        interest_clusters = user_vector_data.get("interest_clusters", []) or []
        disliked_vector = user_vector_data.get("disliked_vector_data")
        negative_feedback_profile = user_vector_data.get("negative_feedback_profile", {}) or {}

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
            # 有多兴趣簇时优先走簇召回，以便覆盖用户不同研究子方向。
            cluster_recall_candidates = self._fetch_cluster_recall_candidates(
                interest_clusters=interest_clusters,
                excluded_ids=excluded_ids,
                top_k=max(top_n * 3, 20),
            )
            logger.debug(
                "Fetched %s cluster recall candidates for user %s from %s interest clusters",
                len(cluster_recall_candidates),
                user_id,
                len(interest_clusters),
            )

        if cluster_recall_candidates:
            candidates = cluster_recall_candidates
            recall_mode = "cluster_recall"
        else:
            # 没有可用簇召回结果时，退化到近期论文池，保证推荐链路可用性。
            candidates = self._fetch_recent_db_candidates(
                liked_category_freq=liked_category_freq,
                max_age_months=max_age_months,
                max_results=max(candidate_limit * 2, candidate_limit),
            )
            recall_mode = "recent_pool"
            logger.debug("Fetched %s recent OAI DB candidates for user %s before deduplication", len(candidates), user_id)

        filtered_candidates = self._deduplicate_candidates(candidates, excluded_ids)
        logger.debug(
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
                logger.debug(
                    "Cluster recall returned no candidates for user %s; falling back to recent OAI DB pool with %s papers",
                    user_id,
                    len(candidates),
                )
                filtered_candidates = self._deduplicate_candidates(candidates, excluded_ids)
                logger.debug(
                    "Retained %s fallback candidate papers after deduplication against %s excluded papers for user %s",
                    len(filtered_candidates),
                    len(excluded_ids),
                    user_id,
                )
                recall_mode = "cluster_recall_fallback_recent_pool"
            if not filtered_candidates:
                raise HTTPException(status_code=400, detail=f"No papers found for recommendation within the last {max_age_months} months")

        # 统一补齐候选论文的论文表与 embedding 数据，后续排序才有稳定输入。
        materialized_candidates, materialize_stats = self._materialize_candidate_papers_for_recommendation(filtered_candidates)
        logger.debug(
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
        logger.debug("Loaded %s stored candidate embeddings from Milvus for user %s", len(existing_candidate_embeddings), user_id)
        candidate_embedding_map = {
            str(item.get("arxiv_id", "") or "").strip(): item.get("vector", [])
            for item in existing_candidate_embeddings
            if item.get("arxiv_id") and item.get("vector")
        }
        reused_vector_count = 0
        for candidate in materialized_candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if arxiv_id in candidate_embedding_map:
                # 先挂载已存向量，排序阶段会优先复用，减少 embedding API 成本。
                candidate["_stored_vector"] = candidate_embedding_map[arxiv_id]
                reused_vector_count += 1

        logger.info("Reused %s/%s candidate vectors from Milvus for user %s", reused_vector_count, len(materialized_candidates), user_id)

        scored_candidates = []
        negative_hard_filtered_candidates = []
        recomputed_vector_count = 0
        missing_vector_count = 0
        for candidate in materialized_candidates:
            # 先做基础相关性打分，再叠加长期画像修正，最后交给多样性选择器。
            scored_candidate = self._build_candidate_score(
                candidate=candidate,
                liked_category_freq=liked_category_freq,
                user_vector=user_vector,
                interest_clusters=interest_clusters,
                disliked_vector=disliked_vector,
                negative_feedback_profile=negative_feedback_profile,
                embedding_config=embedding_config,
            )
            if scored_candidate.get("negative_hard_filter"):
                negative_hard_filtered_candidates.append(
                    {
                        "arxiv_id": scored_candidate.get("arxiv_id"),
                        "title": scored_candidate.get("title"),
                        "negative_score": scored_candidate.get("negative_score"),
                        "negative_feedback_match": scored_candidate.get("negative_feedback_match"),
                    }
                )
                continue
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
            scored_candidate["profile_match_details"] = profile_adjustment.get("profile_match_details", [])
            scored_candidate["profile_match_score"] = profile_adjustment.get("profile_match_score", profile_adjustment["profile_score"])
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
            "ranking_debug": {
                "negative_hard_filtered_count": len(negative_hard_filtered_candidates),
                "negative_hard_filtered_candidates": negative_hard_filtered_candidates,
            },
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
        """对已有搜索结果做确定性重排，并在可用时叠加个性化排序能力。

        该方法不会改变搜索服务召回出的论文集合，只会在原集合内部重新打分。
        如果用户画像、偏好或候选向量不可用，会自动降级到基于查询匹配的普通重排。
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
            # 个性化能力依赖兴趣向量；读取失败时不报错终止，而是保留普通搜索结果能力。
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
            negative_feedback_profile = {}
            embedding_config = None

        user_vector = user_vector_data.get("vector_data") if user_vector_data else None
        interest_clusters = user_vector_data.get("interest_clusters", []) if user_vector_data else []
        disliked_vector = user_vector_data.get("disliked_vector_data") if user_vector_data else None
        negative_feedback_profile = user_vector_data.get("negative_feedback_profile", {}) if user_vector_data else {}
        embedding_config = self.get_embedding_config() if personalized_available else None
        hard_excluded_ids = set(self._normalize_text_terms(disliked_ids))

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
        hard_excluded_candidates: List[Dict[str, Any]] = []
        negative_hard_filtered_candidates: List[Dict[str, Any]] = []
        for paper in normalized_papers:
            fallback_arxiv_id = str(paper.get("arxiv_id", "") or paper.get("id", "") or "").strip()
            if fallback_arxiv_id.lower() in hard_excluded_ids:
                # 显式点踩是强过滤信号；搜索重排也不能把这类论文重新推荐回结果里。
                hard_excluded_candidates.append({"arxiv_id": fallback_arxiv_id, "title": paper.get("title")})
                continue
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
                    # 查询相关性与个性化相关性分别计算，再在后续阶段做融合。
                    ranked_candidate = self._build_candidate_score(
                        candidate=candidate,
                        liked_category_freq=liked_category_freq,
                        user_vector=user_vector,
                        interest_clusters=interest_clusters,
                        disliked_vector=disliked_vector,
                        negative_feedback_profile=negative_feedback_profile,
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

            if ranked_candidate.get("negative_hard_filter"):
                negative_hard_filtered_candidates.append(
                    {
                        "arxiv_id": fallback_arxiv_id,
                        "title": ranked_candidate.get("title"),
                        "negative_score": ranked_candidate.get("negative_score"),
                        "negative_feedback_match": ranked_candidate.get("negative_feedback_match"),
                    }
                )
                continue

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
            ranked_candidate["profile_match_details"] = profile_adjustment.get("profile_match_details", [])
            ranked_candidate["profile_match_score"] = profile_adjustment.get("profile_match_score", profile_adjustment["profile_score"])
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
            "hard_excluded_count": len(hard_excluded_candidates),
            "negative_hard_filtered_count": len(negative_hard_filtered_candidates),
            "research_profile": research_profile,
            "paper_actions": paper_actions,
        }
