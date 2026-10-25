from __future__ import annotations

import ast
import json
import logging
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from copy import deepcopy
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from fastapi import HTTPException

from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService
from services.arxiv.arxiv_search_service import ArxivSearchService
from services.memory import MemoryService
from services.memory.concept_normalizer import ConceptNormalizer
from services.memory.paper_evidence_extractor import PAPER_EVIDENCE_EXTRACTOR_VERSION
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.storage.sqlite.stores import (
    InterestVectorStore,
    PaperCatalogStore,
    PaperProfileEvidenceStore,
    ProfileEventStore,
    ResearchProfileStore,
    UserPreferenceStore,
)
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
from .recommendation_ranker import RecommendationRanker, text_contains_term

logger = logging.getLogger(__name__)

# 后台候选 concepts 抽取：搜索重排或开启 lazy / async 时，缺失的 evidence card 不在请求里同步生成，
# 而是提交到这个小线程池执行并写回缓存，下一次请求即可命中。线程池按需创建、进程内共享。
_BACKGROUND_CONCEPT_EXECUTOR: Optional[ThreadPoolExecutor] = None
_BACKGROUND_CONCEPT_LOCK = threading.Lock()
# 正在后台抽取的 (cache_version, arxiv_id)，避免并发请求对同一论文重复提交 LLM 调用。
_BACKGROUND_CONCEPT_IN_FLIGHT: set[Tuple[str, str]] = set()
_BACKGROUND_CONCEPT_MAX_WORKERS = 2

# user_memory_summary 以字符串形式传入时，超过该长度不再尝试解析，避免对异常大输入做 literal_eval。
_MEMORY_SUMMARY_PARSE_MAX_CHARS = 200_000


def _get_background_concept_executor() -> ThreadPoolExecutor:
    """懒加载后台 concepts 抽取线程池。"""
    global _BACKGROUND_CONCEPT_EXECUTOR
    with _BACKGROUND_CONCEPT_LOCK:
        if _BACKGROUND_CONCEPT_EXECUTOR is None:
            _BACKGROUND_CONCEPT_EXECUTOR = ThreadPoolExecutor(
                max_workers=_BACKGROUND_CONCEPT_MAX_WORKERS,
                thread_name_prefix="candidate-concepts",
            )
        return _BACKGROUND_CONCEPT_EXECUTOR


class RecommendationService(InterestProfileService, CandidateRecallService, CandidateMaterializer, RecommendationRanker):
    """论文推荐服务的门面类，通过多个 mixin 组合出完整推荐链路。

    各 mixin 的职责：
    - InterestProfileService：维护用户正向兴趣向量 / 兴趣簇与负反馈画像；
    - CandidateRecallService：基于兴趣簇或近期论文池召回候选；
    - CandidateMaterializer：把候选补齐到论文表与向量库；
    - RecommendationRanker：基础打分、负反馈惩罚、多样性选择与理由文案。

    本类负责编排，以及叠加“长期研究画像（canonical topics）+ 当前请求上下文”的修正。
    对外入口有三个：
    - recommend_papers：用户中心的经典推荐（必须有兴趣向量）；
    - recommend_papers_with_context：Agent 推荐链路，消费会话上下文，兴趣向量缺失时退化为冷启动；
    - rerank_search_results_for_user：对搜索结果在原集合内做个性化重排。

    推荐整体流程：
        读取画像 → 召回候选 → 去重/排除 → 物化 → 挂载已存向量 → 基础打分（含负反馈）
        → 补齐 top K concepts → 画像/上下文修正 → 多样性选择。
    """

    RECOMMENDATION_CONFIG = get_recommendation_runtime_config()
    ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()
    ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS = RECOMMENDATION_CONFIG["backfill_request_interval_seconds"]
    # 近期论文池固定使用的分类范围，保证无兴趣簇时的召回池稳定可控。
    RECOMMEND_CANDIDATE_CATEGORIES = [
        "cs.CL",
        "cs.LG",
        "cs.IR",
        "cs.AI",
    ]
    # 后台 concepts 抽取使用的执行器；为 None 时使用模块级共享线程池。测试可注入同步执行器以获得确定性。
    concept_enrichment_executor: Optional[Executor] = None

    def __init__(
        self,
        paper_catalog_store: PaperCatalogStore,
        user_preference_store: UserPreferenceStore,
        interest_vector_store: InterestVectorStore,
        paper_profile_evidence_store: PaperProfileEvidenceStore,
        research_profile_store: ResearchProfileStore,
        profile_event_store: ProfileEventStore,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        get_embedding_config: Callable[[], EmbeddingConfig],
        memory_service: MemoryService,
        get_clustering_config: Optional[Callable[[], Dict[str, Any]]] = None,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        oai_db_service: Optional[ArxivOaiDatabaseService] = None,
        collection_name: str = "arxiv_paper_embeddings",
    ):
        """组装推荐链路依赖，并初始化推荐与回填流程所需的运行时状态。"""
        self.paper_catalog_store = paper_catalog_store
        self.user_preference_store = user_preference_store
        self.interest_vector_store = interest_vector_store
        self.paper_profile_evidence_store = paper_profile_evidence_store
        self.research_profile_store = research_profile_store
        self.profile_event_store = profile_event_store
        self.memory_service = memory_service
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.get_embedding_config = get_embedding_config
        self.get_clustering_config = get_clustering_config or get_recommendation_clustering_runtime_config
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
        self.oai_db_service = oai_db_service or ArxivOaiDatabaseService()
        self.collection_name = collection_name
        # arXiv API 回填限流：多线程共享“下一次允许请求的时间点”。
        self._arxiv_backfill_lock = threading.Lock()
        self._arxiv_backfill_next_allowed_time = 0.0
        self.memory_runtime_config = get_memory_runtime_config()

    # ------------------------------------------------------------------
    # 通用小工具
    # ------------------------------------------------------------------

    def _memory_flag(self, key: str, default: Any = None) -> Any:
        """读取记忆模块的运行时开关，避免在推荐主流程中散落硬编码配置访问。"""
        return self.memory_runtime_config.get(key, default)

    @staticmethod
    def _normalize_text_terms(values: Any) -> List[str]:
        """把字符串或字符串列表规范化为小写词项列表，便于后续执行包含匹配。

        仅用于主题词 / 分类等文本，不要用于 arXiv ID（ID 应保持原样，见 _normalize_id_list）。
        """
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

    @staticmethod
    def _normalize_id_list(values: Any) -> List[str]:
        """把论文 ID 列表规范化为去空白、去重、保序的列表（不改变大小写）。

        _deduplicate_candidates 与向量库过滤都是大小写敏感的精确匹配，
        因此排除列表必须与候选 ID 的原始写法保持一致。
        """
        if not values:
            return []
        source = values if isinstance(values, (list, tuple, set)) else [values]
        return list(dict.fromkeys(str(value).strip() for value in source if str(value or "").strip()))

    @staticmethod
    def _candidate_arxiv_id(candidate: Mapping[str, Any]) -> str:
        """读取候选论文 ID，兼容部分来源使用 "id" 字段的情况。"""
        return str(candidate.get("arxiv_id", "") or candidate.get("id", "") or "").strip()

    def _coerce_mapping(self, value: Any) -> Dict[str, Any]:
        """把任意值安全转换为 dict，非 Mapping 一律视为空。"""
        return dict(value) if isinstance(value, Mapping) else {}

    def _coerce_candidate_list(self, value: Any) -> List[Dict[str, Any]]:
        """把任意值安全转换为 dict 列表，丢弃非 Mapping 元素。"""
        return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    # ------------------------------------------------------------------
    # 用户画像与排除集合
    # ------------------------------------------------------------------

    def _build_profile_signal_bundle(self, user_id: str) -> Dict[str, Any]:
        """收集长期研究画像与行为侧信号，并给出需要排除的论文 ID 集合。

        Returns:
            {
                "profile": 长期研究画像（未开启时为空），
                "actions": 行为 action -> 论文 ID 列表（目前只有 read），
                "excluded_ids": 需要排除的论文 ID（开启长期画像时为点赞、点踩），
                "disabled": 是否未开启长期研究画像，
            }

        未开启画像时点赞 / 点踩由调用方另行排除。
        """
        actions = self.user_preference_store.get_user_paper_action_map(user_id)
        if not bool(self._memory_flag("enable_user_research_profile", False)):
            return {
                "profile": {},
                "actions": actions,
                "excluded_ids": [],
                "disabled": True,
            }
        profile = self.research_profile_store.get_user_research_profile(user_id)
        excluded_ids = self._normalize_id_list(
            [
                # like/dislike 是强偏好权威状态，不再从弱行为 action_map 读取。
                *self._normalize_id_list(self.user_preference_store.get_liked_papers(user_id)),
                *self._normalize_id_list(self.user_preference_store.get_disliked_papers(user_id)),
            ]
        )
        return {
            "profile": profile,
            "actions": actions,
            "excluded_ids": excluded_ids,
            "disabled": False,
        }

    def _collect_excluded_ids(self, liked_ids: List[str], disliked_ids: List[str], profile_bundle: Dict[str, Any]) -> List[str]:
        """合并点赞、点踩与画像侧排除集合，得到推荐时不应再出现的论文 ID。"""
        return self._normalize_id_list([*liked_ids, *disliked_ids, *(profile_bundle.get("excluded_ids") or [])])

    def _resolve_effective_research_profile(
        self,
        *,
        stored_profile: Optional[Dict[str, Any]],
        request_profile: Optional[Dict[str, Any]],
        memory_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """按“记忆摘要 < 存储的长期画像 < 当前请求画像”的优先级逐字段合并出 effective profile。

        每一层只用非空字段覆盖下层：记忆摘要只在长期画像缺失对应字段时兜底，
        当前请求显式传入的字段永远优先。
        """
        effective = dict(stored_profile or {})
        for key, value in dict(memory_profile or {}).items():
            # 记忆摘要只填补长期画像中缺失或为空的字段。
            if value not in (None, "", [], {}) and effective.get(key) in (None, "", [], {}):
                effective[key] = value
        # 当前请求显式传入的画像字段优先，避免 Agent 已经识别出的会话约束在服务层被旧缓存覆盖。
        for key, value in dict(request_profile or {}).items():
            if value in (None, "", [], {}):
                continue
            effective[key] = value
        return effective

    @staticmethod
    def _parse_user_memory_summary(value: Any) -> Dict[str, Any]:
        """把 user_memory_summary 解析为画像信号 dict，解析不了时返回空 dict。

        上游（MemoryService.build_user_memory_summary）产出的是 {"profile": {...}, ...} 结构，
        但经过 Agent 节点时可能被转成字符串（JSON 或 Python repr），这里两种都兼容；纯文本摘要无法结构化，直接忽略。

        Returns:
            只包含 positive_topics / negative_topics / recent_topics / preferred_categories 中非空字段的 dict。
        """
        parsed: Any = value
        if isinstance(value, str):
            text = value.strip()
            parsed = None
            if text.startswith("{") and len(text) <= _MEMORY_SUMMARY_PARSE_MAX_CHARS:
                try:
                    parsed = json.loads(text)
                except ValueError:
                    try:
                        parsed = ast.literal_eval(text)
                    except (ValueError, SyntaxError, MemoryError, RecursionError):
                        parsed = None
        if not isinstance(parsed, Mapping):
            return {}
        profile = parsed.get("profile") if isinstance(parsed.get("profile"), Mapping) else parsed
        signals: Dict[str, Any] = {}
        for key in ("positive_topics", "negative_topics", "recent_topics", "preferred_categories"):
            items = profile.get(key)
            if isinstance(items, (list, tuple)):
                cleaned = [str(item).strip() for item in items if str(item or "").strip()]
                if cleaned:
                    signals[key] = cleaned
        return signals

    def _get_or_refresh_interest_vector(self, user_id: str) -> Dict[str, Any]:
        """读取用户兴趣向量；如果缺失或过期（晚于最近一次偏好信号），则自动触发重建。

        重建失败（例如用户没有任何点赞论文）时 generate_user_interest_vector 会直接抛出 HTTPException；
        重建后仍读不到向量时抛 400。
        """
        vector_data = self.interest_vector_store.get_user_interest_vector(user_id=user_id)
        latest_preference_ts = self.profile_event_store.get_latest_user_signal_timestamp(user_id=user_id)

        if not vector_data:
            logger.info("Interest vector missing for user %s, rebuilding", user_id)
            needs_rebuild = True
        elif latest_preference_ts and self._is_vector_stale(vector_data.get("updated_at"), latest_preference_ts):
            logger.info("Interest vector is stale for user %s, rebuilding", user_id)
            needs_rebuild = True
        else:
            needs_rebuild = False

        if needs_rebuild:
            self.generate_user_interest_vector(user_id=user_id)
            vector_data = self.interest_vector_store.get_user_interest_vector(user_id=user_id)

        if not vector_data:
            raise HTTPException(status_code=400, detail="User interest vector not found. Please generate it first.")

        return vector_data

    # ------------------------------------------------------------------
    # 长期画像修正：canonical topic 与候选 concepts 的语义匹配
    # ------------------------------------------------------------------

    def _action_contains(self, actions: Dict[str, List[str]], action: str, arxiv_id: str) -> bool:
        """判断论文是否出现在某个行为列表中（大小写不敏感）。"""
        if not arxiv_id:
            return False
        target = arxiv_id.lower()
        return any(item.lower() == target for item in self._normalize_id_list(actions.get(action)))

    def _compute_profile_adjustment(
        self,
        candidate: Dict[str, Any],
        profile: Optional[Dict[str, Any]] = None,
        actions: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """根据 effective profile 的 canonical topic 与候选论文 evidence card 做语义匹配。

        profile_score   = min(0.24, Σ正向主题匹配分 × 0.05 + 命中偏好分类数 × 0.03 + 行为加成)
        profile_penalty = min(0.24, Σ负向主题匹配分 × 0.07)
        profile_adjustment = profile_score - profile_penalty

        行为加成：已读 -0.03（已读过的论文略微降权）。
        """
        profile = profile or {}
        actions = actions or {}
        arxiv_id = self._candidate_arxiv_id(candidate)
        categories = {item.lower() for item in self._split_categories(candidate.get("categories"))}

        positive_topics = self._profile_topic_objects(profile, positive=True)
        negative_topics = self._profile_topic_objects(profile, positive=False)
        preferred_categories = set(self._normalize_text_terms(profile.get("preferred_categories")))
        candidate_concepts = self._get_candidate_concepts(candidate)

        matched_positive = self._match_profile_topics_to_candidate(positive_topics, candidate_concepts)
        matched_negative = self._match_profile_topics_to_candidate(negative_topics, candidate_concepts)
        matched_categories = sorted(categories & preferred_categories)

        action_boost = 0.0
        if self._action_contains(actions, "read", arxiv_id):
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
        """从画像中取出正向 / 负向主题对象。

        优先使用 canonical_topics（带 aliases），再用旧版 positive_topics / negative_topics 补齐
        canonical 中没有的主题（按小写去重）。
        """
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
                known.add(text.lower())
                topics.append({"label": text, "aliases": [], "source": "legacy_projection"})
        return topics

    def _get_candidate_concepts(self, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """读取候选 concepts，优先复用本次请求已解析的结果，避免重复查询 evidence 缓存。

        _enrich_top_candidate_concepts 结束时会通过 _attach_candidate_concept_debug 重新解析一次，
        保证缓存的是补全之后的最新结果。
        """
        cached = candidate.get("_resolved_candidate_concepts")
        if isinstance(cached, list):
            return cached
        return self._candidate_profile_concepts(candidate)

    def _candidate_profile_concepts(self, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析候选论文的 concepts，并把结果与来源缓存到候选的临时字段上。

        concepts 来源（置信度从高到低）：
        1. evidence card 的 candidate_concepts（过滤掉不可泛化的概念），默认置信度 0.6；
        2. 候选自带的 technical_concepts / concepts，置信度 0.55；
        3. 以上都没有时，从标题 + 摘要做低置信度（0.35）的主题抽取兜底。

        写入的临时字段：_resolved_candidate_concepts、_resolved_candidate_concept_source
        （cached_evidence_card / runtime_enriched / candidate_payload / title_abstract_fallback / missing）。
        """
        card = self._resolve_candidate_evidence_card(candidate)
        concepts: List[Dict[str, Any]] = []
        concept_source = "missing"
        for item in (card or {}).get("candidate_concepts") or []:
            if not isinstance(item, dict) or not item.get("whether_generalizable", True):
                continue
            label = ConceptNormalizer.clean_label(item.get("label"))
            if label:
                concepts.append({"label": label, "confidence": float(item.get("confidence") or 0.6), "source": "paper_evidence_card"})
        if concepts:
            card_source = str((card or {}).get("_concept_source") or "").strip()
            concept_source = card_source if card_source in {"cached_evidence_card", "runtime_enriched"} else "cached_evidence_card"
        for label in candidate.get("technical_concepts") or candidate.get("concepts") or []:
            clean = ConceptNormalizer.clean_label(label)
            if clean:
                concepts.append({"label": clean, "confidence": 0.55, "source": "candidate_payload"})
        if concepts and concept_source == "missing":
            # evidence card 没有贡献 concepts，全部来自候选自带字段。
            concept_source = "candidate_payload"
        if not concepts:
            # 只有 evidence card 缺失时才启用低置信度文本 fallback，避免推荐主路径退回字符串包含匹配。
            text = f"{candidate.get('title', '')} {candidate.get('abstract', '') or candidate.get('summary', '')}"
            for topic in self.memory_service.profile_generator.extract_topics(text):
                concepts.append({"label": topic, "confidence": 0.35, "source": "low_confidence_text_fallback"})
            if concepts:
                concept_source = "title_abstract_fallback"
        candidate["_resolved_candidate_concepts"] = list(concepts)
        candidate["_resolved_candidate_concept_source"] = concept_source if concepts else "missing"
        return concepts

    def _resolve_candidate_evidence_card(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """按固定优先级解析候选 evidence card，保证画像修正先消费高质量 concepts。

        优先级：已挂载且来源为 runtime_enriched / cached_evidence_card 的卡片
        → 持久化缓存中当前版本的卡片（命中后写回候选）→ 其余已挂载的卡片（可能没有 concepts）。
        """
        attached_card = candidate.get("evidence_card") if isinstance(candidate.get("evidence_card"), dict) else None
        attached_source = str((attached_card or {}).get("_concept_source") or "").strip()
        if attached_source in {"runtime_enriched", "cached_evidence_card"} and (attached_card or {}).get("candidate_concepts"):
            return attached_card

        arxiv_id = self._candidate_arxiv_id(candidate)
        if arxiv_id:
            cached_card = self.paper_profile_evidence_store.get_paper_profile_evidence(
                arxiv_id,
                extractor_version=PAPER_EVIDENCE_EXTRACTOR_VERSION,
            )
            if isinstance(cached_card, dict) and cached_card.get("candidate_concepts"):
                # 命中缓存后写回候选，后续同一请求内的多次匹配都复用同一份 evidence。
                cached_card["_concept_source"] = "cached_evidence_card"
                candidate["evidence_card"] = cached_card
                return cached_card

        # 缓存未命中时退回已挂载卡片（例如 generated_candidate_concepts 或补全失败的错误卡）。
        return attached_card

    def _attach_candidate_concept_debug(self, candidate: Dict[str, Any]) -> None:
        """重新解析 concepts，并把来源与补全状态写回候选，避免后续调试只能靠日志推断。"""
        concepts = self._candidate_profile_concepts(candidate)
        concept_source = str(candidate.get("_resolved_candidate_concept_source") or "missing")
        candidate["candidate_concept_source"] = concept_source
        evidence_card = candidate.get("evidence_card") if isinstance(candidate.get("evidence_card"), dict) else {}
        candidate["candidate_concept_debug"] = {
            "source": concept_source,
            "source_detail": str(evidence_card.get("_concept_source_detail") or concept_source).strip() or concept_source,
            "cache_version": str(evidence_card.get("extractor_version") or "").strip(),
            "enrichment_attempted": bool(evidence_card.get("_enrichment_attempted", False)),
            "enrichment_status": str(evidence_card.get("_enrichment_status") or "").strip() or "not_attempted",
            "used_cache": concept_source == "cached_evidence_card",
            "generated_concepts": [str(item.get("label") or "").strip() for item in concepts if isinstance(item, dict) and str(item.get("label") or "").strip()],
            "concept_count": len(concepts),
            "fallback_used": concept_source == "title_abstract_fallback",
            "missing": concept_source == "missing",
        }
        score_breakdown = dict(candidate.get("score_breakdown", {}))
        score_breakdown["candidate_concept_source"] = concept_source
        candidate["score_breakdown"] = score_breakdown

    def _match_profile_topics_to_candidate(self, profile_topics: List[Dict[str, Any]], candidate_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为每个画像主题找到最匹配的候选 concept。

        主题标签与别名分别和 concept 计算 _topic_similarity，取最大值；低于 0.55 视为不匹配。
        match_score = 相似度 × concept 置信度。结果按 match_score 降序。
        """
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
        """主题相似度：规范化后相等或互相包含记 1.0，否则为词集合的 Jaccard 系数。"""
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
        """生成“匹配你的长期兴趣”文案；画像有 topic_evidence 时附带来源论文数量。"""
        topics = [item["topic"] for item in matches if item.get("topic")]
        evidence = profile.get("topic_evidence") if isinstance(profile.get("topic_evidence"), dict) else {}
        evidence_parts: List[str] = []
        for topic in topics[:2]:
            item = evidence.get(topic) or {}
            paper_count = len(item.get("source_papers") or [])
            if paper_count:
                evidence_parts.append(f"{topic}（{paper_count} 篇论文）")
        if evidence_parts:
            return f"匹配你的长期兴趣：{', '.join(evidence_parts)}"
        return f"匹配你的长期兴趣：{', '.join(topics[:3])}"

    # ------------------------------------------------------------------
    # 候选 concepts 补全（LLM 抽取 evidence card）
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        """把配置值转为 int；None / 空串 / 非法值使用默认值，但保留合法的 0。"""
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _get_candidate_concept_enrichment_config(self) -> Dict[str, Any]:
        """读取并规范化候选 concepts 补全配置。

        - top_k：只对基础分前 top_k 的候选尝试补全（至少 1）；
        - max_llm_calls：单次请求最多发起（同步执行或提交后台）的抽取次数（0 表示完全不调用）；
        - allow_lazy_generation / allow_async_generation：任一开启后主链路只读缓存，缺失的抽取提交到后台执行；
        - score_field：选择 top_k 时使用的分数字段（base_score 或 relevance_score）。
        """
        config = self.RECOMMENDATION_CONFIG.get("candidate_concept_enrichment", {}) or {}
        top_k = self._coerce_int(config.get("top_k"), 20)
        max_llm_calls = self._coerce_int(config.get("max_llm_calls"), 5)
        score_field = str(config.get("score_field") or "base_score").strip() or "base_score"
        return {
            "enabled": bool(config.get("enabled", True)),
            "top_k": max(1, top_k),
            "max_llm_calls": max(0, max_llm_calls),
            "allow_lazy_generation": bool(config.get("allow_lazy_generation", False)),
            "allow_async_generation": bool(config.get("allow_async_generation", False)),
            "cache_version": str(config.get("cache_version") or PAPER_EVIDENCE_EXTRACTOR_VERSION).strip() or PAPER_EVIDENCE_EXTRACTOR_VERSION,
            "score_field": score_field,
        }

    def _get_candidate_concept_sort_score(self, candidate: Dict[str, Any], score_field: str) -> float:
        """读取用于挑选补全对象的分数，字段缺失时在 base_score 与 relevance_score 间互相兜底。"""
        if score_field == "relevance_score":
            return float(candidate.get("relevance_score", candidate.get("base_score", 0.0)) or 0.0)
        return float(candidate.get("base_score", candidate.get("relevance_score", 0.0)) or 0.0)

    def _can_skip_candidate_concept_enrichment(self, candidate: Dict[str, Any]) -> bool:
        """已有可用 evidence card 或候选自带 concepts 时直接跳过，避免重复 LLM 调用。"""
        evidence_card = candidate.get("evidence_card") if isinstance(candidate.get("evidence_card"), dict) else {}
        if evidence_card.get("candidate_concepts"):
            return True
        if candidate.get("technical_concepts") or candidate.get("concepts"):
            return True
        return False

    def _extract_and_cache_evidence_card(self, paper: Dict[str, Any], arxiv_id: str, cache_version: str) -> Optional[Dict[str, Any]]:
        """调用抽取器生成 evidence card，并把结果（包括失败卡）写入持久化缓存。

        同步补全与后台补全共用这一实现。extract 内部已经吞掉了 LLM / JSON 错误，
        这里再兜底抽取器自身崩溃的情况，保证失败结果也按统一 schema 写入缓存，避免同一论文反复失败重试。
        """
        extractor = self.memory_service.paper_evidence_extractor
        try:
            card = extractor.extract(paper)
        except Exception as exc:
            logger.warning("Candidate concept enrichment crashed arxiv_id=%s error=%s", arxiv_id, exc)
            card = extractor.build_error_card(paper, str(exc), extraction_confidence=0.0)
        if not isinstance(card, dict):
            return None
        card["_enrichment_attempted"] = True
        if arxiv_id:
            card["extractor_version"] = cache_version
            self.paper_profile_evidence_store.upsert_paper_profile_evidence(arxiv_id, card)
        return card

    def _schedule_background_concept_extraction(self, candidate: Dict[str, Any], arxiv_id: str, cache_version: str) -> str:
        """把单篇候选的 concepts 抽取提交到后台执行，返回调度状态。

        Returns:
            "scheduled"：已提交新的后台任务；
            "already_scheduled"：同一论文同一版本已有任务在执行，本次不重复提交；
            "schedule_failed"：执行器拒绝提交（例如进程正在退出）。
        """
        key = (cache_version, arxiv_id)
        with _BACKGROUND_CONCEPT_LOCK:
            if key in _BACKGROUND_CONCEPT_IN_FLIGHT:
                return "already_scheduled"
            _BACKGROUND_CONCEPT_IN_FLIGHT.add(key)
        # 只传浅拷贝且去掉排序临时字段（含完整向量），避免后台线程与请求线程共享可变状态。
        paper_snapshot = {name: value for name, value in candidate.items() if not str(name).startswith("_")}

        def run() -> None:
            try:
                self._extract_and_cache_evidence_card(paper_snapshot, arxiv_id, cache_version)
            except Exception:
                logger.exception("Background candidate concept enrichment failed arxiv_id=%s", arxiv_id)
            finally:
                with _BACKGROUND_CONCEPT_LOCK:
                    _BACKGROUND_CONCEPT_IN_FLIGHT.discard(key)

        executor = self.concept_enrichment_executor or _get_background_concept_executor()
        try:
            executor.submit(run)
        except RuntimeError as exc:
            with _BACKGROUND_CONCEPT_LOCK:
                _BACKGROUND_CONCEPT_IN_FLIGHT.discard(key)
            logger.warning("Failed to schedule background concept enrichment arxiv_id=%s error=%s", arxiv_id, exc)
            return "schedule_failed"
        return "scheduled"

    def _enrich_top_candidate_concepts(
        self,
        candidates: List[Dict[str, Any]],
        *,
        allow_sync_generation: bool = True,
    ) -> Dict[str, Any]:
        """在基础分计算后只补齐 top K 候选的 concepts，供长期画像修正阶段稳定复用。

        步骤：
        1. 对全部候选查询 evidence 缓存并挂载（缓存命中 / 失败缓存 / 自带 concepts 分别计数）；
        2. 对 top K 中仍缺 concepts 的候选发起抽取，受 max_llm_calls 预算限制：
           - allow_sync_generation=False（搜索重排）或开启 lazy / async（默认开启）时，提交到后台线程执行，
             本次请求按缺失处理（退回低置信度文本兜底），结果写入缓存后下次请求命中；
           - lazy / async 都关闭时在请求内同步调用 LLM，本次请求即可用上新 concepts；
        3. 抽取结果（包括失败卡）写回持久化缓存，避免同一论文反复失败重试；
        4. 为全部候选写入 concepts 来源调试信息。

        Returns:
            补全统计信息（selected / cache_hit / payload_hit / generated / failed / scheduled 等计数）。
            deferred_count 为“因走后台而本次未补全”的候选数，scheduled_count 为其中新提交的后台任务数。
        """
        config = self._get_candidate_concept_enrichment_config()
        background_generation = (
            not allow_sync_generation
            or config["allow_lazy_generation"]
            or config["allow_async_generation"]
        )
        stats = {
            "enabled": bool(config["enabled"]),
            "top_k": int(config["top_k"]),
            "max_llm_calls": int(config["max_llm_calls"]),
            "allow_lazy_generation": bool(config["allow_lazy_generation"]),
            "allow_async_generation": bool(config["allow_async_generation"]),
            "generation_mode": "background" if background_generation else "sync",
            "cache_version": config["cache_version"],
            "score_field": config["score_field"],
            "selected_count": 0,
            "cache_hit_count": 0,
            "payload_hit_count": 0,
            "generated_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "llm_attempt_count": 0,
            "budget_skipped_count": 0,
            "deferred_count": 0,
            "scheduled_count": 0,
        }
        if not config["enabled"] or not candidates:
            for candidate in candidates:
                self._attach_candidate_concept_debug(candidate)
            return stats

        ranked_candidates = sorted(
            candidates,
            key=lambda item: (
                self._get_candidate_concept_sort_score(item, config["score_field"]),
                self._candidate_arxiv_id(item),
            ),
            reverse=True,
        )
        selected_candidates = ranked_candidates[: config["top_k"]]
        stats["selected_count"] = len(selected_candidates)

        # 第一步：全部候选先查缓存，缓存的命中不占用 LLM 预算。
        for candidate in candidates:
            arxiv_id = self._candidate_arxiv_id(candidate)
            card = None
            if arxiv_id:
                card = self.paper_profile_evidence_store.get_paper_profile_evidence(
                    arxiv_id,
                    extractor_version=config["cache_version"],
                )
            if isinstance(card, dict):
                candidate["evidence_card"] = card
                if card.get("candidate_concepts"):
                    card["_concept_source"] = "cached_evidence_card"
                    card["_concept_source_detail"] = "cache_hit"
                    card["_enrichment_status"] = "cache_hit"
                    stats["cache_hit_count"] += 1
                elif card.get("schema_valid") is False:
                    # 之前抽取失败过的卡：同样挂载，下面会因为没有 concepts 而再次尝试补全。
                    card["_concept_source_detail"] = "failed_cache"
                    card["_enrichment_status"] = str(card.get("error_type") or "failed_cache")
            elif self._can_skip_candidate_concept_enrichment(candidate):
                stats["payload_hit_count"] += 1

        # 第二步：只对 top K 中仍缺 concepts 的候选发起抽取（同步或后台）。
        for candidate in selected_candidates:
            if self._can_skip_candidate_concept_enrichment(candidate):
                continue
            evidence_card = candidate.get("evidence_card") if isinstance(candidate.get("evidence_card"), dict) else {}
            if stats["llm_attempt_count"] + stats["scheduled_count"] >= int(config["max_llm_calls"]):
                stats["budget_skipped_count"] += 1
                evidence_card["_enrichment_attempted"] = False
                evidence_card["_enrichment_status"] = "llm_budget_exhausted"
                evidence_card["_concept_source_detail"] = "llm_budget_exhausted"
                continue

            arxiv_id = self._candidate_arxiv_id(candidate)
            if background_generation:
                # 主链路不等待新的抽取：后台写入缓存后，下一次请求的第一步即可命中。
                # 没有 arxiv_id 的候选无法写缓存，后台抽取没有意义，直接按缺失处理。
                schedule_status = (
                    self._schedule_background_concept_extraction(candidate, arxiv_id, config["cache_version"])
                    if arxiv_id
                    else "missing_arxiv_id"
                )
                if schedule_status == "scheduled":
                    stats["scheduled_count"] += 1
                stats["deferred_count"] += 1
                evidence_card["_enrichment_attempted"] = False
                evidence_card["_enrichment_status"] = schedule_status
                evidence_card["_concept_source_detail"] = "background_generation"
                continue

            stats["llm_attempt_count"] += 1
            card = self._extract_and_cache_evidence_card(candidate, arxiv_id, config["cache_version"])
            if isinstance(card, dict) and card.get("candidate_concepts"):
                card["_concept_source"] = "runtime_enriched"
                card["_concept_source_detail"] = "llm_generated"
                card["_enrichment_status"] = "generated"
                candidate["evidence_card"] = card
                stats["generated_count"] += 1
            else:
                # 失败卡也要挂回候选，后续能明确知道这里已经尝试过补全并安全降级。
                if isinstance(card, dict):
                    card["_concept_source_detail"] = "generation_failed"
                    card["_enrichment_status"] = str(card.get("error_type") or card.get("error_message") or "failed")
                    candidate["evidence_card"] = card
                stats["failed_count"] += 1

        stats["skipped_count"] = max(0, stats["selected_count"] - stats["generated_count"] - stats["failed_count"])
        # 第三步：补全完成后统一刷新 concepts 解析结果与调试信息。
        for candidate in candidates:
            self._attach_candidate_concept_debug(candidate)
        logger.info(
            "Candidate concept enrichment finished mode=%s selected=%s cache_hit=%s payload_hit=%s generated=%s failed=%s scheduled=%s score_field=%s",
            stats["generation_mode"],
            stats["selected_count"],
            stats["cache_hit_count"],
            stats["payload_hit_count"],
            stats["generated_count"],
            stats["failed_count"],
            stats["scheduled_count"],
            stats["score_field"],
        )
        return stats

    # ------------------------------------------------------------------
    # 画像修正 + 上下文修正
    # ------------------------------------------------------------------

    def _apply_profile_adjustment_to_scored_candidates(
        self,
        scored_candidates: List[Dict[str, Any]],
        *,
        profile: Optional[Dict[str, Any]] = None,
        actions: Optional[Dict[str, List[str]]] = None,
        context_bundle: Optional[Dict[str, Any]] = None,
        allow_sync_concept_generation: bool = True,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """先补齐高分候选 concepts，再统一叠加长期画像与上下文修正。

        对每个候选：
            relevance_score += profile_adjustment + request_score - request_penalty
            final_score     += 同上

        提供 context_bundle 时，命中负向主题且没有命中正向主题的候选会被直接过滤。
        allow_sync_concept_generation=False 时缺失的 concepts 只提交后台抽取，不在本次请求里同步调用 LLM。

        Returns:
            (修正后的候选列表（原地修改）, 被上下文过滤掉的候选摘要列表)。
        """
        profile = profile or {}
        actions = actions or {}
        context_bundle = context_bundle or {}
        filtered_out_candidates: List[Dict[str, Any]] = []
        enrichment_stats = self._enrich_top_candidate_concepts(
            scored_candidates,
            allow_sync_generation=allow_sync_concept_generation,
        )

        adjusted_candidates: List[Dict[str, Any]] = []
        for scored_candidate in scored_candidates:
            profile_adjustment = self._compute_profile_adjustment(
                scored_candidate,
                profile=profile,
                actions=actions,
            )
            context_adjustment = (
                self._compute_contextual_adjustment(
                    scored_candidate,
                    context_bundle=context_bundle,
                )
                if context_bundle
                else None
            )
            if context_adjustment and context_adjustment["exclude_candidate"]:
                filtered_out_candidates.append(
                    {
                        "arxiv_id": scored_candidate.get("arxiv_id"),
                        "title": scored_candidate.get("title"),
                        "reason": context_adjustment["negative_filter_reason"],
                    }
                )
                continue

            profile_terms = list(profile_adjustment.get("matched_positive_topics") or [])
            query_terms = list(
                dict.fromkeys(
                    [
                        *(context_adjustment["matched_query_terms"] if context_adjustment else []),
                        *(context_adjustment["matched_request_topics"] if context_adjustment else []),
                    ]
                )
            )
            # 推荐依据标记，供前端展示“为什么推荐”的来源维度。
            basis = {
                "long_term_profile": bool(profile_terms),
                "current_request": bool(query_terms),
                "recent_behavior": bool((context_adjustment or {}).get("recent_behavior_terms")),
                "cold_start": bool(context_bundle.get("cold_start")) if context_bundle else False,
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
            scored_candidate["recommendation_basis"] = basis

            request_score = float((context_adjustment or {}).get("request_score", 0.0) or 0.0)
            request_penalty = float((context_adjustment or {}).get("request_penalty", 0.0) or 0.0)
            total_adjustment = profile_adjustment["profile_adjustment"] + request_score - request_penalty
            # base_score 保持基础打分不变；relevance_score / final_score 叠加修正，供多样性选择使用。
            scored_candidate["base_score"] = float(scored_candidate.get("base_score", scored_candidate.get("relevance_score", 0.0)) or 0.0)
            scored_candidate["relevance_score"] = float(scored_candidate.get("relevance_score", 0.0) or 0.0) + total_adjustment
            scored_candidate["final_score"] = float(scored_candidate.get("final_score", 0.0) or 0.0) + total_adjustment
            scored_candidate["concept_enrichment_debug"] = deepcopy(enrichment_stats)
            scored_candidate["profile_adjustment_debug"] = {
                "concept_enrichment": deepcopy(enrichment_stats),
                "candidate_concept_debug": dict(scored_candidate.get("candidate_concept_debug", {})),
                "matched_positive_topics": list(profile_adjustment["matched_positive_topics"]),
                "matched_negative_topics": list(profile_adjustment["matched_negative_topics"]),
                "profile_score": float(profile_adjustment["profile_score"] or 0.0),
                "profile_penalty": float(profile_adjustment["profile_penalty"] or 0.0),
                "profile_adjustment": float(profile_adjustment["profile_adjustment"] or 0.0),
            }

            if context_adjustment:
                scored_candidate["matched_request_topics"] = context_adjustment["matched_request_topics"]
                scored_candidate["matched_category_constraints"] = context_adjustment["matched_category_constraints"]
                scored_candidate["recent_behavior_score"] = context_adjustment["recent_behavior_score"]
                scored_candidate["recent_behavior_terms"] = context_adjustment["recent_behavior_terms"]
                scored_candidate["negative_filter_reason"] = context_adjustment["negative_filter_reason"]
                scored_candidate["trace_sources"] = list(
                    dict.fromkeys(
                        [
                            *context_adjustment["context_sources"],
                            *(["long_term_profile"] if basis["long_term_profile"] else []),
                        ]
                    )
                )
                scored_candidate["match_reason"] = self._build_match_reason(context_adjustment["query_match_score"], query_terms)
                scored_candidate["score_components"] = {
                    "semantic_score": float(scored_candidate.get("semantic_score", 0.0) or 0.0),
                    "negative_score": float(scored_candidate.get("negative_score", 0.0) or 0.0),
                    "negative_penalty": float(scored_candidate.get("negative_penalty", 0.0) or 0.0),
                    "negative_confidence": float(scored_candidate.get("negative_confidence", 0.0) or 0.0),
                    "profile_adjustment": profile_adjustment["profile_adjustment"],
                    "request_score": request_score,
                    "request_penalty": request_penalty,
                    "recent_behavior_score": context_adjustment["recent_behavior_score"],
                    "final_score": float(scored_candidate.get("final_score", 0.0) or 0.0),
                }
            score_breakdown = dict(scored_candidate.get("score_breakdown", {}))
            score_breakdown["candidate_concept_source"] = scored_candidate.get("candidate_concept_source")
            score_breakdown["profile_score"] = profile_adjustment["profile_score"]
            score_breakdown["profile_penalty"] = profile_adjustment["profile_penalty"]
            score_breakdown["profile_adjustment"] = profile_adjustment["profile_adjustment"]
            score_breakdown["relevance_score"] = scored_candidate["relevance_score"]
            score_breakdown["final_score"] = scored_candidate["final_score"]
            if context_adjustment:
                score_breakdown.update(context_adjustment["query_score_breakdown"])
                score_breakdown["request_score"] = request_score
                score_breakdown["request_penalty"] = request_penalty
                score_breakdown["query_match_score"] = context_adjustment["query_match_score"]
                score_breakdown["recent_behavior_score"] = context_adjustment["recent_behavior_score"]
            scored_candidate["score_breakdown"] = score_breakdown
            adjusted_candidates.append(scored_candidate)
        return adjusted_candidates, filtered_out_candidates

    def _build_recommendation_context_bundle(
        self,
        *,
        message: Optional[str],
        topic_hint: Optional[str],
        research_profile: Optional[Dict[str, Any]],
        request_context: Optional[Dict[str, Any]],
        memory_recent_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """把 Agent 传入的请求上下文整理为统一的上下文信号包。

        - query_text：topic_hint > message > request_context.query；
        - positive_topics：请求未提供时回退到画像正向主题；
        - negative_topics：请求负向主题与画像负向主题的并集；
        - recent_topics：从最近交互的前 5 篇论文标题摘要中抽取的词项，再并上记忆摘要中的近期主题；
        - cold_start：显式要求，或既没有查询文本也没有画像正向主题。
        """
        context = self._coerce_mapping(request_context)
        query_text = str(topic_hint or message or context.get("query") or "").strip()
        positive_topics = self._normalize_text_terms(context.get("positive_topics"))
        negative_topics = self._normalize_text_terms(context.get("negative_topics"))
        category_constraints = self._normalize_text_terms(context.get("category_constraints"))
        recent_papers = self._coerce_candidate_list(context.get("recent_papers"))
        recent_text = " ".join(
            f"{paper.get('title', '')} {paper.get('abstract', '') or paper.get('summary', '')}"
            for paper in recent_papers[:5]
        )
        recent_topics = list(
            dict.fromkeys([*self._extract_query_terms(recent_text), *self._normalize_text_terms(list(memory_recent_topics or []))])
        )
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
        """合并会话上下文候选与召回候选，按 ID 去重并规范化字段，标注 candidate_source。"""
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        embedding_model = self.get_embedding_config().model_name
        # 会话里最近出现过的论文会先进入候选池，后续仍由统一排序逻辑决定是否留下，避免 Agent 和 Service 重复做两套推荐。
        for source_name, source_candidates in (("context", context_candidates), ("recall", recalled_candidates)):
            for candidate in source_candidates:
                arxiv_id = self._candidate_arxiv_id(candidate)
                if not arxiv_id or arxiv_id in seen:
                    continue
                seen.add(arxiv_id)
                normalized = self._normalize_paper_record(dict(candidate), arxiv_id, embedding_model=embedding_model)
                merged.append({**candidate, **normalized, "candidate_source": source_name})
        return merged

    def _compute_contextual_adjustment(
        self,
        candidate: Dict[str, Any],
        *,
        context_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """计算当前请求上下文对候选的加减分。

        request_score   = min(0.38, 查询匹配分 × 0.24 + 正向主题命中数 × 0.05 + 最近行为分 + 分类约束命中数 × 0.03)
        request_penalty = min(0.45, 负向主题命中数 × 0.15 + 分类约束全部未命中时的 0.12)
        最近行为分      = min(0.18, 最近交互词命中数 × 0.03)

        context_bundle["include_query_match_in_request_score"] 为 False 时（搜索重排），
        request_score 不包含“查询匹配分 × 0.24”这一项，因为调用方已经单独按权重使用查询匹配分，避免重复计分。

        主题命中按词边界判断（在标题、摘要、分类与 concepts 拼接的小写文本中查找，见 text_contains_term）。
        """
        query_breakdown = self._build_query_match_score(
            candidate,
            query=context_bundle.get("query_text"),
            search_categories=context_bundle.get("category_constraints") or [],
        )
        include_query_match = bool(context_bundle.get("include_query_match_in_request_score", True))
        candidate_categories = self._split_categories(candidate.get("categories"))
        candidate_text = " ".join(
            [
                str(candidate.get("title", "") or ""),
                str(candidate.get("abstract", "") or candidate.get("summary", "") or ""),
                " ".join(candidate_categories),
                " ".join(str(item.get("label") or "") for item in self._get_candidate_concepts(candidate)),
            ]
        ).lower()
        positive_hits = [term for term in context_bundle.get("positive_topics", []) if text_contains_term(candidate_text, term)]
        negative_hits = [term for term in context_bundle.get("negative_topics", []) if text_contains_term(candidate_text, term)]
        recent_hits = [term for term in context_bundle.get("recent_topics", []) if text_contains_term(candidate_text, term)]
        category_constraints = {str(item).lower() for item in context_bundle.get("category_constraints", [])}
        category_hits = sorted(category_constraints & {item.lower() for item in candidate_categories})
        category_mismatch_penalty = 0.12 if category_constraints and not category_hits else 0.0
        recent_behavior_score = min(0.18, len(recent_hits) * 0.03)
        query_component = float(query_breakdown.get("query_match_score", 0.0) or 0.0) * 0.24 if include_query_match else 0.0
        request_score = min(
            0.38,
            query_component
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
        """汇总长期兴趣、当前请求、最近交互的命中词，生成推荐解释文案。"""
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

    # ------------------------------------------------------------------
    # 推荐主流程的共享步骤
    # ------------------------------------------------------------------

    def _recall_and_filter_candidates(
        self,
        *,
        user_id: str,
        interest_clusters: List[Dict[str, Any]],
        excluded_ids: List[str],
        top_n: int,
        max_age_months: int,
        recent_pool_mode: str = "recent_pool",
        context_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """召回候选并完成去重与排除，返回 (候选列表, recall_mode)。

        - 有兴趣簇时优先走簇召回（每簇 top_k = max(top_n*3, 20)）；
        - 没有簇或簇召回为空时，走近期论文池（max_results = max(top_n*5, 50) * 2），recall_mode 为 recent_pool_mode；
        - 簇召回经过排除后为空时，再回退到近期论文池（recall_mode = cluster_recall_fallback_recent_pool）；
        - context_candidates 不为 None 时，会先与召回结果合并（上下文候选优先）再去重。

        仍然没有候选时抛 400。
        """
        candidate_limit = max(top_n * 5, 50)

        def fetch_recent_pool() -> List[Dict[str, Any]]:
            return self._fetch_recent_db_candidates(
                max_age_months=max_age_months,
                max_results=candidate_limit * 2,
            )

        def merge_and_filter(recalled: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if context_candidates is not None:
                recalled = self._merge_candidate_sources(
                    recalled_candidates=recalled,
                    context_candidates=context_candidates,
                )
            return self._deduplicate_candidates(recalled, excluded_ids)

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
            recall_mode = "cluster_recall"
            filtered_candidates = merge_and_filter(cluster_recall_candidates)
        else:
            # 没有可用簇召回结果时，退化到近期论文池，保证推荐链路可用性。
            recall_mode = recent_pool_mode
            filtered_candidates = merge_and_filter(fetch_recent_pool())

        if not filtered_candidates and recall_mode == "cluster_recall":
            logger.debug("Cluster recall returned no usable candidates for user %s; falling back to recent OAI DB pool", user_id)
            recall_mode = "cluster_recall_fallback_recent_pool"
            filtered_candidates = merge_and_filter(fetch_recent_pool())

        logger.debug(
            "Retained %s candidate papers after deduplication against %s excluded papers for user %s (recall_mode=%s)",
            len(filtered_candidates),
            len(excluded_ids),
            user_id,
            recall_mode,
        )
        if not filtered_candidates:
            raise HTTPException(status_code=400, detail=f"No papers found for recommendation within the last {max_age_months} months")
        return filtered_candidates, recall_mode

    def _load_stored_vector_map(self, arxiv_ids: List[str]) -> Dict[str, List[float]]:
        """批量读取向量库中已存在的论文向量，返回 arxiv_id -> vector。"""
        ids = [arxiv_id for arxiv_id in arxiv_ids if arxiv_id]
        if not ids:
            return {}
        rows = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
            collection_name=self.collection_name,
            arxiv_ids=ids,
        )
        return {
            str(item.get("arxiv_id", "") or "").strip(): [float(value) for value in item.get("vector", [])]
            for item in rows
            if item.get("arxiv_id") and item.get("vector")
        }

    def _attach_stored_vectors(self, candidates: List[Dict[str, Any]], vector_map: Dict[str, List[float]]) -> int:
        """把已存向量挂到候选的 _stored_vector 上，排序阶段会优先复用以减少 embedding API 成本。

        Returns:
            成功挂载向量的候选数量。
        """
        reused_count = 0
        for candidate in candidates:
            arxiv_id = self._candidate_arxiv_id(candidate)
            if arxiv_id in vector_map:
                candidate["_stored_vector"] = vector_map[arxiv_id]
                reused_count += 1
        return reused_count

    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        liked_category_freq: Counter,
        user_vector: Optional[List[float]],
        interest_clusters: List[Dict[str, Any]],
        disliked_vector: Optional[List[float]],
        negative_feedback_profile: Dict[str, Any],
        embedding_config: Optional[EmbeddingConfig],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        """对候选做基础打分，并剔除触发负反馈硬过滤的论文。

        Returns:
            (打分后的候选, 被负反馈硬过滤的候选摘要, 向量来源统计 {"recomputed", "missing"})。
        """
        scored_candidates: List[Dict[str, Any]] = []
        hard_filtered: List[Dict[str, Any]] = []
        vector_stats = {"recomputed": 0, "missing": 0}
        for candidate in candidates:
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
                hard_filtered.append(self._build_negative_hard_filter_record(scored_candidate))
                continue
            scored_candidates.append(scored_candidate)
            embedding_source = str(scored_candidate.get("_embedding_source", "") or "")
            if embedding_source in vector_stats:
                vector_stats[embedding_source] += 1
        return scored_candidates, hard_filtered, vector_stats

    def _build_negative_hard_filter_record(self, scored_candidate: Dict[str, Any]) -> Dict[str, Any]:
        """构造被负反馈硬过滤的候选摘要，供调试信息展示。"""
        return {
            "arxiv_id": self._candidate_arxiv_id(scored_candidate),
            "title": scored_candidate.get("title"),
            "reason": "negative_feedback_hard_filter",
            "negative_score": scored_candidate.get("negative_score"),
            "negative_feedback_match": scored_candidate.get("negative_feedback_match"),
        }

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

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
        """在不破坏旧接口的前提下，为 Agent 推荐链路提供真正消费上下文的推荐实现。

        与 recommend_papers 的区别：
        - 会合并请求传入的画像、会话上下文与候选论文（candidate_papers 优先进入候选池）；
        - 叠加上下文修正，并按负向主题过滤候选；
        - 兴趣向量不可用或判定为冷启动时，不报错而是退化为近期论文池 + 上下文排序。

        user_memory_summary（dict，或其 JSON / Python repr 字符串）中的画像信号作为最低优先级的画像层：
        正向 / 负向主题与偏好分类只在长期画像缺失对应字段时兜底，近期主题并入“最近行为”信号。
        无法结构化解析的纯文本摘要会被忽略。
        """
        logger.info("Starting context-aware recommendation for user %s with top_n=%s max_age_months=%s", user_id, top_n, max_age_months)

        preferences = self.user_preference_store.get_user_preferences(user_id=user_id)
        liked_ids = preferences.get("liked_papers", [])
        disliked_ids = preferences.get("disliked_papers", [])
        profile_bundle = self._build_profile_signal_bundle(user_id)
        memory_signals = self._parse_user_memory_summary(user_memory_summary)
        memory_recent_topics = list(memory_signals.pop("recent_topics", []) or [])
        effective_profile = self._resolve_effective_research_profile(
            stored_profile=profile_bundle.get("profile", {}),
            request_profile=research_profile,
            memory_profile=memory_signals,
        )
        paper_actions = profile_bundle.get("actions", {})
        excluded_ids = self._collect_excluded_ids(liked_ids, disliked_ids, profile_bundle)
        liked_details = self.user_preference_store.get_liked_papers_with_details(user_id=user_id)
        liked_category_freq = self._build_liked_category_frequency(liked_details)
        context_bundle = self._build_recommendation_context_bundle(
            message=message,
            topic_hint=topic_hint,
            research_profile=effective_profile,
            request_context=request_context,
            memory_recent_topics=memory_recent_topics,
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

        filtered_candidates, recall_mode = self._recall_and_filter_candidates(
            user_id=user_id,
            interest_clusters=interest_clusters if personalized_available else [],
            excluded_ids=excluded_ids,
            top_n=top_n,
            max_age_months=max_age_months,
            recent_pool_mode="recent_pool" if personalized_available else "cold_start_recent_pool",
            context_candidates=self._coerce_candidate_list(candidate_papers),
        )

        materialized_candidates, materialize_stats = self._materialize_candidate_papers_for_recommendation(filtered_candidates)
        embedding_config = self.get_embedding_config() if personalized_available else None
        reused_vector_count = 0
        if personalized_available:
            # 冷启动时不做向量打分，也就不需要读取候选向量。
            vector_map = self._load_stored_vector_map([self._candidate_arxiv_id(candidate) for candidate in materialized_candidates])
            reused_vector_count = self._attach_stored_vectors(materialized_candidates, vector_map)

        scored_candidates, filtered_out_candidates, vector_stats = self._score_candidates(
            materialized_candidates,
            liked_category_freq=liked_category_freq,
            user_vector=user_vector,
            interest_clusters=interest_clusters,
            disliked_vector=disliked_vector,
            negative_feedback_profile=negative_feedback_profile,
            embedding_config=embedding_config,
        )

        scored_candidates, concept_filtered_candidates = self._apply_profile_adjustment_to_scored_candidates(
            scored_candidates,
            profile=effective_profile,
            actions=paper_actions,
            context_bundle=context_bundle,
        )
        filtered_out_candidates.extend(concept_filtered_candidates)

        selected = self._select_diverse_candidates(
            scored_candidates,
            top_n,
            interest_clusters=interest_clusters,
        )
        # 理由文案只需为最终入选的候选生成；所依赖的分项在多样性选择前后不变。
        for item in selected:
            item.setdefault("score_components", {})["diversity_score"] = float(item.get("diversity_score", 0.0) or 0.0)
            item["personalized_reason"] = self._build_personalized_reason(item, dict(item.get("score_breakdown", {})))
            item["recommendation_explanation"] = self._build_recommendation_explanation(item)

        logger.info(
            "Context-aware recommendation finished for user %s: scored=%s selected=%s filtered=%s reused=%s recomputed=%s missing=%s",
            user_id,
            len(scored_candidates),
            len(selected),
            len(filtered_out_candidates),
            reused_vector_count,
            vector_stats["recomputed"],
            vector_stats["missing"],
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
                # 只有成功解析出画像信号才算“使用”了记忆摘要；provided 仅表示调用方传了值。
                "used_user_memory_summary": bool(memory_signals or memory_recent_topics),
                "user_memory_summary_provided": bool(user_memory_summary),
            },
            "filter_debug": {
                "excluded_ids_count": len(excluded_ids),
                "filtered_out_by_context": filtered_out_candidates,
                "negative_hard_filtered_count": sum(1 for item in filtered_out_candidates if item.get("reason") == "negative_feedback_hard_filter"),
                "materialize_stats": materialize_stats,
                "candidate_concept_enrichment": scored_candidates[0].get("concept_enrichment_debug", {}) if scored_candidates else {},
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
        """执行完整推荐流程，包括画像读取、候选召回、物化、打分与多样性筛选。

        兴趣向量缺失且无法重建（例如用户还没有点赞任何论文）时直接报错，不做冷启动退化。
        """
        user_vector_data = self._get_or_refresh_interest_vector(user_id)
        user_vector = user_vector_data["vector_data"]
        interest_clusters = user_vector_data.get("interest_clusters", []) or []
        disliked_vector = user_vector_data.get("disliked_vector_data")
        negative_feedback_profile = user_vector_data.get("negative_feedback_profile", {}) or {}

        logger.info("Starting paper recommendation for user %s with top_n=%s max_age_months=%s", user_id, top_n, max_age_months)

        preferences = self.user_preference_store.get_user_preferences(user_id=user_id)
        liked_ids = preferences.get("liked_papers", [])
        disliked_ids = preferences.get("disliked_papers", [])
        profile_bundle = self._build_profile_signal_bundle(user_id)
        research_profile = profile_bundle.get("profile", {})
        paper_actions = profile_bundle.get("actions", {})
        excluded_ids = self._collect_excluded_ids(liked_ids, disliked_ids, profile_bundle)

        liked_details = self.user_preference_store.get_liked_papers_with_details(user_id=user_id)
        liked_category_freq = self._build_liked_category_frequency(liked_details)

        filtered_candidates, recall_mode = self._recall_and_filter_candidates(
            user_id=user_id,
            interest_clusters=interest_clusters,
            excluded_ids=excluded_ids,
            top_n=top_n,
            max_age_months=max_age_months,
        )

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
        vector_map = self._load_stored_vector_map([self._candidate_arxiv_id(candidate) for candidate in materialized_candidates])
        reused_vector_count = self._attach_stored_vectors(materialized_candidates, vector_map)
        logger.info("Reused %s/%s candidate vectors from Milvus for user %s", reused_vector_count, len(materialized_candidates), user_id)

        # 先做基础相关性打分，再叠加长期画像修正，最后交给多样性选择器。
        scored_candidates, negative_hard_filtered_candidates, vector_stats = self._score_candidates(
            materialized_candidates,
            liked_category_freq=liked_category_freq,
            user_vector=user_vector,
            interest_clusters=interest_clusters,
            disliked_vector=disliked_vector,
            negative_feedback_profile=negative_feedback_profile,
            embedding_config=embedding_config,
        )

        scored_candidates, _ = self._apply_profile_adjustment_to_scored_candidates(
            scored_candidates,
            profile=research_profile,
            actions=paper_actions,
        )

        logger.info(
            "Candidate embedding summary for user %s: total=%s reused=%s recomputed=%s missing=%s",
            user_id,
            len(materialized_candidates),
            reused_vector_count,
            vector_stats["recomputed"],
            vector_stats["missing"],
        )

        selected = self._select_diverse_candidates(
            scored_candidates,
            top_n,
            interest_clusters=interest_clusters,
        )
        # 前端推荐卡片直接展示 personalized_reason，这里与上下文推荐链路保持一致，只为入选候选生成文案。
        for item in selected:
            item["personalized_reason"] = self._build_personalized_reason(item, dict(item.get("score_breakdown", {})))
            item["recommendation_explanation"] = self._build_recommendation_explanation(item)

        logger.info(
            "Recommendation finished for user %s: scored=%s selected=%s reused=%s recomputed=%s missing=%s",
            user_id,
            len(scored_candidates),
            len(selected),
            reused_vector_count,
            vector_stats["recomputed"],
            vector_stats["missing"],
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
                "candidate_concept_enrichment": scored_candidates[0].get("concept_enrichment_debug", {}) if scored_candidates else {},
            },
            "recommendations": selected,
        }

    @staticmethod
    def _empty_personalization_breakdown() -> Dict[str, float]:
        """未做个性化打分时使用的默认分项结构。"""
        return {
            "semantic_score": 0.0,
            "category_score": 0.0,
            "recency_score": 0.0,
            "disliked_penalty": 0.0,
            "relevance_score": 0.0,
            "diversity_score": 0.0,
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

        该方法不会引入新论文，只会在原集合内部重新打分；但会剔除用户已点赞 / 点踩的论文
        以及触发负反馈硬过滤的论文。如果用户画像、偏好或候选向量不可用，会自动降级到基于查询匹配的普通重排。

        最终分数::

            final_score = query_match_score × 0.60 + personalization_score × 0.30
                          + (profile_score - profile_penalty) + (request_score - request_penalty)

        其中 request_score 不含查询匹配项（查询匹配分已经以 0.60 的权重单独计入），
        这里主要体现分类约束的命中加分与全部未命中时的扣分。

        候选缺少 concepts 时不会在请求内同步调用 LLM 抽取，而是提交后台任务写入缓存，保证搜索接口延迟稳定。
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
        user_vector_data: Optional[Dict[str, Any]] = None
        try:
            # 个性化能力依赖兴趣向量；读取失败时不报错终止，而是保留普通搜索结果能力。
            user_vector_data = self._get_or_refresh_interest_vector(user_id)
            personalized_available = True
        except Exception as exc:
            personalized_available = False
            warnings.append(f"用户兴趣向量不可用，已退化为普通搜索排序: {exc}")

        liked_ids: List[str] = []
        disliked_ids: List[str] = []
        liked_category_freq: Counter = Counter()
        try:
            preferences = self.user_preference_store.get_user_preferences(user_id=user_id)
            liked_ids = preferences.get("liked_papers", [])
            disliked_ids = preferences.get("disliked_papers", [])
            liked_details = self.user_preference_store.get_liked_papers_with_details(user_id=user_id)
            liked_category_freq = self._build_liked_category_frequency(liked_details)
        except Exception as exc:
            warnings.append(f"读取用户偏好失败，已退化为普通搜索排序: {exc}")
            personalized_available = False

        user_vector = (user_vector_data or {}).get("vector_data") if personalized_available else None
        interest_clusters = (user_vector_data or {}).get("interest_clusters", []) or []
        disliked_vector = (user_vector_data or {}).get("disliked_vector_data")
        negative_feedback_profile = (user_vector_data or {}).get("negative_feedback_profile", {}) or {}
        embedding_config = self.get_embedding_config() if personalized_available else None
        # 搜索结果进入个性化重排前，已明确反馈过的论文都不应再次作为推荐候选；liked 论文虽是正向信号，但它已经被用户处理过。
        hard_excluded_ids = set(self._normalize_text_terms([*liked_ids, *disliked_ids]))

        stored_embeddings: Dict[str, List[float]] = {}
        if personalized_available:
            try:
                stored_embeddings = self._load_stored_vector_map([self._candidate_arxiv_id(paper) for paper in normalized_papers])
            except Exception as exc:
                warnings.append(f"复用候选论文向量失败，已使用文本特征继续排序: {exc}")

        scored_candidates: List[Dict[str, Any]] = []
        hard_excluded_candidates: List[Dict[str, Any]] = []
        negative_hard_filtered_candidates: List[Dict[str, Any]] = []
        # 标准化只需要模型名；个性化不可用时也要读一次配置，但不能每篇论文都读。
        normalize_embedding_model = embedding_config.model_name if embedding_config else self.get_embedding_config().model_name
        for paper in normalized_papers:
            fallback_arxiv_id = self._candidate_arxiv_id(paper)
            if fallback_arxiv_id.lower() in hard_excluded_ids:
                # 显式点赞 / 点踩是强过滤信号；搜索重排也不能把这类论文重新推荐回结果里。
                hard_excluded_candidates.append({"arxiv_id": fallback_arxiv_id, "title": paper.get("title")})
                continue
            normalized_paper = self._normalize_paper_record(
                paper,
                fallback_arxiv_id or "unknown",
                embedding_model=normalize_embedding_model,
            )
            candidate = {**paper, **normalized_paper}

            if fallback_arxiv_id in stored_embeddings:
                candidate["_stored_vector"] = stored_embeddings[fallback_arxiv_id]

            query_breakdown = self._build_query_match_score(
                candidate,
                query=query_text,
                title_query=title_query,
                abstract_query=abstract_query,
                search_categories=search_categories,
            )
            candidate["query_match_score"] = query_breakdown["query_match_score"]
            candidate["matched_terms"] = query_breakdown["matched_terms"]

            ranked_candidate = dict(candidate)
            personalization_score = 0.0
            score_breakdown = self._empty_personalization_breakdown()
            if user_vector:
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
                except Exception as exc:
                    warnings.append(f"论文 {fallback_arxiv_id or 'unknown'} 个性化打分失败，已退化为查询排序: {exc}")
                    ranked_candidate = dict(candidate)
            score_breakdown.update(query_breakdown["query_score_breakdown"])

            if ranked_candidate.get("negative_hard_filter"):
                negative_hard_filtered_candidates.append(self._build_negative_hard_filter_record(ranked_candidate))
                continue
            ranked_candidate["query_match_score"] = float(query_breakdown["query_match_score"] or 0.0)
            ranked_candidate["personalization_score"] = personalization_score
            ranked_candidate["base_score"] = float(ranked_candidate.get("relevance_score", 0.0) or 0.0)
            ranked_candidate["score_breakdown"] = {
                **score_breakdown,
                "query_match_score": float(query_breakdown["query_match_score"] or 0.0),
                "personalization_score": personalization_score,
            }
            scored_candidates.append(ranked_candidate)

        context_bundle = {
            "query_text": query_text,
            "positive_topics": [],
            "negative_topics": [],
            "category_constraints": search_categories,
            "recent_topics": [],
            "cold_start": not personalized_available,
            # 查询匹配分已在最终分数中按 0.60 计入，request_score 里不再重复计算。
            "include_query_match_in_request_score": False,
        }
        # 此处 positive / negative topics 为空，因此上下文修正不会过滤任何候选，被过滤列表可以忽略。
        scored_candidates, _ = self._apply_profile_adjustment_to_scored_candidates(
            scored_candidates,
            profile=research_profile,
            actions=paper_actions,
            context_bundle=context_bundle,
            allow_sync_concept_generation=False,
        )
        for ranked_candidate in scored_candidates:
            self._strip_ranking_temp_fields(ranked_candidate)
            score_breakdown = dict(ranked_candidate.get("score_breakdown", {}))
            query_match_score = float(ranked_candidate.get("query_match_score", 0.0) or 0.0)
            personalization_score = float(ranked_candidate.get("personalization_score", 0.0) or 0.0)
            profile_adjustment = float(ranked_candidate.get("profile_score", 0.0) or 0.0) - float(ranked_candidate.get("profile_penalty", 0.0) or 0.0)
            context_adjustment = float(score_breakdown.get("request_score", 0.0) or 0.0) - float(score_breakdown.get("request_penalty", 0.0) or 0.0)
            final_score = query_match_score * 0.60 + personalization_score * 0.30 + profile_adjustment + context_adjustment
            ranked_candidate["final_score"] = final_score
            ranked_candidate["score_breakdown"] = {
                **score_breakdown,
                "query_match_score": query_match_score,
                "personalization_score": personalization_score,
                "context_adjustment": context_adjustment,
                "final_score": final_score,
            }
            ranked_candidate["match_reason"] = self._build_match_reason(query_match_score, list(ranked_candidate.get("matched_terms") or []))
            ranked_candidate["personalized_reason"] = self._build_personalized_reason(ranked_candidate, ranked_candidate["score_breakdown"])

        scored_candidates.sort(
            key=lambda item: (
                float(item.get("final_score", 0.0) or 0.0),
                float(item.get("query_match_score", 0.0) or 0.0),
                float(item.get("personalization_score", 0.0) or 0.0),
                float(item.get("score_breakdown", {}).get("semantic_score", 0.0) or 0.0),
                str(item.get("published_date", "") or ""),
                self._candidate_arxiv_id(item),
            ),
            reverse=True,
        )

        for index, paper in enumerate(scored_candidates, start=1):
            paper["priority"] = index
            paper["score_breakdown"]["priority"] = index

        selected = scored_candidates[:limit]
        personalized_applied = bool(user_vector)
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
            "candidate_concept_enrichment": scored_candidates[0].get("concept_enrichment_debug", {}) if scored_candidates else {},
        }
