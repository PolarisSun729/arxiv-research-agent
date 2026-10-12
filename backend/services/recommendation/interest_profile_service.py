from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)

# 负向画像结构版本号，写入画像数据中，便于以后结构升级时识别旧数据。
NEGATIVE_FEEDBACK_PROFILE_VERSION = "negative_feedback_profile_v1"


class InterestProfileService:
    """用户兴趣画像能力（mixin）：根据点赞 / 点踩论文生成正向兴趣向量、兴趣簇与负向画像。

    产出的画像通过 ``interest_vector_store.save_user_interest_vector`` 落库，包含：
    - vector_data：所有点赞论文向量的归一化均值，作为兜底的单一兴趣方向；
    - interest_clusters：点赞足够多时由 HDBSCAN 拆出的多个兴趣子方向（簇中心向量）；
    - weak_interest_pool：HDBSCAN 判为噪声点的点赞论文，单独成池，不参与簇召回；
    - negative_feedback_profile：点踩论文的负向簇 / 样本，只用于排序惩罚，不参与主向量计算。

    依赖宿主 RecommendationService 注入：``user_preference_store``、``interest_vector_store``、
    ``paper_catalog_store``、``vector_store_service``、``collection_name``、``get_embedding_config``、
    ``get_clustering_config``、``RECOMMENDATION_CONFIG``；
    并调用其他 mixin 的方法：``_normalize_paper_record`` / ``_build_and_insert_paper_embedding`` /
    ``_build_paper_store_payload`` / ``_fetch_paper_from_arxiv_with_rate_limit`` / ``_embed_papers``
    （CandidateMaterializer）以及 ``_split_categories`` / ``_parse_datetime``（RecommendationRanker）。
    """

    def generate_user_interest_vector(self, user_id: str) -> Dict[str, Any]:
        """根据用户正向反馈生成兴趣向量，并把负向反馈作为独立信号保存。

        该方法是推荐系统个性化画像的入口。主兴趣向量只表达“用户喜欢什么”，
        点踩论文只记录为“不要再推荐什么”的负向信号，供排序、过滤和解释阶段消费。

        流程：
        1. 读取点赞 / 点踩 ID，点赞数不足 min_liked_for_vector 直接报错；
        2. 为每篇论文补齐向量（向量库 → 回填写库 → 临时 embedding 三级兜底）；
        3. 点赞向量求均值得到主向量；数量足够时再做 HDBSCAN 聚类得到兴趣簇；
        4. 点踩向量构建独立的负向画像；
        5. 全部落库并返回统计信息。

        Raises:
            HTTPException(400): 点赞论文数量不足。
            HTTPException(500): 可用向量不足或画像保存失败。
        """
        positive_config = self._get_positive_profile_config()
        negative_config = self._get_negative_profile_config()
        min_liked_for_vector = int(positive_config["min_liked_for_vector"])
        negative_enabled = bool(negative_config["enabled"])
        liked_ids = self.user_preference_store.get_liked_papers(user_id=user_id)
        disliked_ids = self.user_preference_store.get_disliked_papers(user_id=user_id)

        if len(liked_ids) < min_liked_for_vector:
            raise HTTPException(
                status_code=400,
                detail=f"At least {min_liked_for_vector} liked papers are required to generate user interest vector",
            )

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
        liked_records = self._hydrate_vectors_with_metadata(
            requested_ids=liked_ids,
            milvus_embeddings=liked_embeddings,
            config=config,
            label="liked",
        )
        # 负向画像关闭时完全不读取点踩向量，省掉向量库查询和回填开销。
        disliked_records: List[Dict[str, Any]] = []
        if negative_enabled and disliked_ids:
            disliked_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
                collection_name=self.collection_name,
                arxiv_ids=disliked_ids,
            )
            disliked_records = self._hydrate_vectors_with_metadata(
                requested_ids=disliked_ids,
                milvus_embeddings=disliked_embeddings,
                config=config,
                label="disliked",
            )

        if len(liked_records) < min_liked_for_vector:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Only {len(liked_records)} liked papers have usable embeddings, "
                    f"at least {min_liked_for_vector} are required"
                ),
            )

        # 主兴趣方向只来自 liked papers，避免 dislike 原因多样时把正向兴趣中心拉偏。
        fallback_interest_vector = self._normalize_vector(self._mean_vector([record["vector"] for record in liked_records]))
        disliked_vector_data = (
            self._normalize_vector(self._mean_vector([record["vector"] for record in disliked_records]))
            if disliked_records
            else None
        )
        weak_interest_pool: Optional[Dict[str, Any]] = None
        interest_clusters: List[Dict[str, Any]] = []
        profile_mode = "mean"
        if len(liked_records) >= int(positive_config["min_liked_for_clustering"]):
            try:
                # 当用户正反馈足够多时，进一步拆分成多个兴趣簇，提升召回覆盖面和解释性。
                interest_clusters, weak_interest_pool = self._cluster_interest_vectors(liked_records)
                if interest_clusters:
                    profile_mode = "clustered_with_weak_pool" if weak_interest_pool else "clustered"
                    logger.debug(
                        "User %s liked papers clustered into %s interest clusters: %s",
                        user_id,
                        len(interest_clusters),
                        self._summarize_clusters(interest_clusters),
                    )
                else:
                    logger.debug(
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
        liked_milvus_count = sum(1 for record in liked_records if record.get("source") == "milvus")
        liked_fallback_count = sum(1 for record in liked_records if record.get("source") == "fallback")
        liked_unresolved_count = len(liked_ids) - len(liked_records)
        used_count = len(liked_records)
        negative_feedback_profile = self._build_negative_feedback_profile(
            disliked_ids=disliked_ids,
            disliked_records=disliked_records,
            disliked_vector_data=disliked_vector_data,
            negative_config=negative_config,
        )
        disliked_paper_examples = list(negative_feedback_profile.get("examples") or [])
        negative_clusters = list(negative_feedback_profile.get("clusters") or [])
        negative_feedback_stats = dict(negative_feedback_profile.get("stats") or {})

        success = self.interest_vector_store.save_user_interest_vector(
            user_id=user_id,
            vector_data=fallback_interest_vector,
            paper_count=used_count,
            embedding_model=config.model_name,
            vector_dimension=vector_dimension,
            cluster_count=len(interest_clusters),
            profile_mode=profile_mode,
            interest_clusters=interest_clusters,
            weak_interest_pool=weak_interest_pool,
            disliked_vector_data=disliked_vector_data,
            disliked_paper_examples=disliked_paper_examples,
            negative_feedback_stats=negative_feedback_stats,
            negative_feedback_profile=negative_feedback_profile,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save interest vector")

        return {
            "status": "success",
            "message": (
                "User interest vector generated successfully "
                f"using {liked_milvus_count} liked Milvus vectors, "
                f"{liked_fallback_count} liked fallback vectors, "
                f"and {liked_unresolved_count} unresolved liked papers"
            ),
            "paper_count": used_count,
            "used_count": used_count,
            "milvus_used_count": liked_milvus_count,
            "fallback_used_count": liked_fallback_count,
            "unresolved_count": liked_unresolved_count,
            "total_signal_count": len(liked_records) + len(disliked_records),
            "feedback_used_count": len(disliked_records),
            "liked_count": len(liked_records),
            "disliked_count": len(disliked_records),
            "raw_disliked_count": len(disliked_ids),
            "liked_milvus_count": liked_milvus_count,
            "disliked_milvus_count": int(negative_feedback_stats.get("milvus_count", 0)),
            "liked_fallback_count": liked_fallback_count,
            "disliked_fallback_count": int(negative_feedback_stats.get("fallback_count", 0)),
            "liked_unresolved_count": liked_unresolved_count,
            "disliked_unresolved_count": int(negative_feedback_stats.get("unresolved_disliked", len(disliked_ids))),
            "vector_dimension": vector_dimension,
            "embedding_model": config.model_name,
            "cluster_count": len(interest_clusters),
            "profile_mode": profile_mode,
            "weak_interest_pool": weak_interest_pool,
            "weak_interest_pool_count": weak_interest_pool.get("paper_count", 0) if weak_interest_pool else 0,
            "negative_feedback_stats": negative_feedback_stats,
            "disliked_paper_examples": disliked_paper_examples,
            "negative_feedback_profile": negative_feedback_profile,
            "negative_cluster_count": len(negative_clusters),
            "negative_clusters": negative_clusters,
            "disliked_vector_available": disliked_vector_data is not None,
            "cluster_summary": self._summarize_clusters(interest_clusters),
        }

    @staticmethod
    def _summarize_clusters(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成不含中心向量的簇摘要，用于日志与接口返回。"""
        return [
            {
                "cluster_id": cluster.get("cluster_id"),
                "paper_count": cluster.get("paper_count", 0),
                "paper_ids": cluster.get("paper_ids", []),
            }
            for cluster in clusters
        ]

    def _get_positive_profile_config(self) -> Dict[str, Any]:
        """读取正向兴趣画像配置，作为主向量和兴趣簇阈值的唯一来源。"""
        return dict(self.RECOMMENDATION_CONFIG["profile"]["positive"])

    def _get_negative_profile_config(self) -> Dict[str, Any]:
        """读取负向反馈画像配置，避免负向信号保存策略散落在业务代码中。"""
        return dict(self.RECOMMENDATION_CONFIG["profile"]["negative"])

    # ------------------------------------------------------------------
    # 负向画像
    # ------------------------------------------------------------------

    def _build_negative_feedback_profile(
        self,
        *,
        disliked_ids: List[str],
        disliked_records: List[Dict[str, Any]],
        disliked_vector_data: Optional[List[float]],
        negative_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """为点踩论文构建独立负向画像，失败时按配置退回实例级反馈。

        mode 取值：
        - none：没有可用的负向信号；
        - examples：只有点踩样本（数量不够聚类或未开启聚类）；
        - clusters：成功聚出负向簇；
        - cluster_fallback_examples：尝试聚类但没有稳定簇，退回样本；
        - cluster_failed：尝试聚类失败且不允许 / 无法退回样本。
        """
        negative_enabled = bool(negative_config["enabled"])
        if not negative_enabled:
            return self._empty_negative_feedback_profile(disliked_ids, enabled=False)

        examples = self._build_disliked_paper_examples(
            disliked_records=disliked_records,
            negative_config=negative_config,
        )
        clusters: List[Dict[str, Any]] = []
        mode = "examples" if examples else "none"
        fallback_reason: Optional[str] = None

        can_cluster = (
            bool(negative_config["enable_negative_clustering"])
            and len(disliked_records) >= int(negative_config["min_disliked_for_clustering"])
        )
        if can_cluster:
            try:
                clusters = self._cluster_negative_feedback_vectors(
                    disliked_records=disliked_records,
                    negative_config=negative_config,
                )
            except Exception as exc:  # pragma: no cover
                # 负向聚类是增强信号，失败不能阻断画像重建或推荐主链路。
                logger.warning("Failed to cluster disliked papers; falling back to negative examples: %s", exc)
                clusters = []
                fallback_reason = "cluster_failed"

            if clusters:
                mode = "clusters"
            elif bool(negative_config["fallback_to_examples_when_cluster_failed"]) and examples:
                mode = "cluster_fallback_examples"
                fallback_reason = fallback_reason or "no_stable_negative_clusters"
            else:
                mode = "cluster_failed"
                fallback_reason = fallback_reason or "no_stable_negative_clusters"

        stats = self._build_negative_feedback_stats(
            negative_enabled=negative_enabled,
            mode=mode,
            disliked_ids=disliked_ids,
            disliked_records=disliked_records,
            disliked_vector_data=disliked_vector_data,
            disliked_paper_examples=examples,
            negative_clusters=clusters,
            fallback_reason=fallback_reason,
        )
        return {
            "version": NEGATIVE_FEEDBACK_PROFILE_VERSION,
            "enabled": negative_enabled,
            "mode": mode,
            # 无论负向画像能否建模，点踩过的论文 ID 始终作为硬排除名单保留。
            "hard_exclude_ids": self._normalize_paper_ids(disliked_ids),
            "examples": examples,
            "clusters": clusters,
            "stats": stats,
        }

    def _empty_negative_feedback_profile(self, disliked_ids: List[str], *, enabled: bool) -> Dict[str, Any]:
        """生成结构完整的空负向画像（负向画像关闭时使用），保证调用方不需要判空分支。"""
        return {
            "version": NEGATIVE_FEEDBACK_PROFILE_VERSION,
            "enabled": enabled,
            "mode": "none",
            "hard_exclude_ids": self._normalize_paper_ids(disliked_ids),
            "examples": [],
            "clusters": [],
            "stats": self._build_negative_feedback_stats(
                negative_enabled=enabled,
                mode="none",
                disliked_ids=disliked_ids,
                disliked_records=[],
                disliked_vector_data=None,
                disliked_paper_examples=[],
                negative_clusters=[],
            ),
        }

    @staticmethod
    def _normalize_paper_ids(paper_ids: List[str]) -> List[str]:
        """去掉空白与空 ID，保持原顺序。"""
        return [str(arxiv_id).strip() for arxiv_id in paper_ids if str(arxiv_id).strip()]

    def _build_disliked_paper_examples(
        self,
        disliked_records: List[Dict[str, Any]],
        negative_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """保存少量点踩样本及其向量，供样本不足或聚类失败时做实例级惩罚。

        最多保存 max_disliked_examples 条（取输入顺序的前 N 条），可用样本少于
        min_disliked_for_instance_feedback 时不保存，避免一两次误点踩造成大范围误伤。
        """
        if not bool(negative_config["enabled"]) or not bool(negative_config["store_disliked_examples"]):
            return []
        if len(disliked_records) < int(negative_config["min_disliked_for_instance_feedback"]):
            return []
        max_examples = int(negative_config["max_disliked_examples"])
        examples: List[Dict[str, Any]] = []
        for record in disliked_records[:max_examples]:
            arxiv_id = str(record.get("arxiv_id", "") or "").strip()
            raw_vector = record.get("vector") or []
            if not arxiv_id or not raw_vector:
                continue
            paper = self.paper_catalog_store.get_paper(arxiv_id) or {}
            examples.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": str(paper.get("title", "") or ""),
                    "categories": self._split_categories(paper.get("categories")),
                    "vector": self._normalize_vector([float(value) for value in raw_vector]),
                    "vector_source": record.get("source"),
                }
            )
        return examples

    def _cluster_negative_feedback_vectors(
        self,
        disliked_records: List[Dict[str, Any]],
        negative_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """把足量点踩论文聚成多个负向方向，避免单一负向均值掩盖不同拒绝原因。

        HDBSCAN 判为噪声的点踩论文直接丢弃（它们仍可通过 examples 产生实例级惩罚）。
        结果按簇大小降序截断到 max_negative_clusters 个，cluster_id 重新编号为 negative_cluster_0..N。
        """
        min_disliked_for_clustering = int(negative_config["min_disliked_for_clustering"])
        valid_records = [record for record in disliked_records if record.get("vector")]
        # HDBSCAN 至少需要两个样本才有意义。
        if len(valid_records) < max(2, min_disliked_for_clustering):
            return []

        vectors = [self._normalize_vector([float(value) for value in record["vector"]]) for record in valid_records]
        paper_ids = [str(record.get("arxiv_id", "") or "").strip() for record in valid_records]
        labels = self._run_hdbscan(vectors, negative_config["clustering"])
        cluster_members, _ = self._group_cluster_labels(labels)
        return self._build_ranked_clusters(
            cluster_members=cluster_members,
            vectors=vectors,
            paper_ids=paper_ids,
            id_prefix="negative_cluster",
            max_clusters=int(negative_config["max_negative_clusters"]),
            keep_label=True,
        )

    def _build_negative_feedback_stats(
        self,
        *,
        negative_enabled: bool,
        mode: str,
        disliked_ids: List[str],
        disliked_records: List[Dict[str, Any]],
        disliked_vector_data: Optional[List[float]],
        disliked_paper_examples: List[Dict[str, Any]],
        negative_clusters: List[Dict[str, Any]],
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """把负向反馈的建模方式显式化，便于排序解释和旧数据兼容判断。

        usable_disliked 会被排序阶段用来计算负向惩罚置信度（见 _calculate_negative_feedback_confidence）。
        participates_in_main_vector 恒为 False，明确负向信号不参与主兴趣向量。
        """
        stats = {
            "enabled": negative_enabled,
            "mode": mode,
            "total_disliked": len(disliked_ids),
            "usable_disliked": len(disliked_records),
            "unresolved_disliked": len(disliked_ids) - len(disliked_records) if negative_enabled else len(disliked_ids),
            "milvus_count": sum(1 for record in disliked_records if record.get("source") == "milvus"),
            "fallback_count": sum(1 for record in disliked_records if record.get("source") == "fallback"),
            "stored_examples": len(disliked_paper_examples),
            "negative_cluster_count": len(negative_clusters),
            "vector_available": disliked_vector_data is not None,
            "participates_in_main_vector": False,
        }
        if fallback_reason:
            stats["fallback_reason"] = fallback_reason
        return stats

    # ------------------------------------------------------------------
    # 通用向量工具
    # ------------------------------------------------------------------

    def _is_vector_stale(self, vector_updated_at: Optional[str], latest_preference_ts: Optional[str]) -> bool:
        """判断已存兴趣向量是否早于最新用户行为，以决定是否需要重建；任一时间缺失或无法解析时视为不过期。"""
        if not vector_updated_at or not latest_preference_ts:
            return False
        vector_dt = self._parse_datetime(vector_updated_at)
        preference_dt = self._parse_datetime(latest_preference_ts)
        if not vector_dt or not preference_dt:
            return False
        return vector_dt < preference_dt

    def _mean_vector(self, vectors: List[List[float]]) -> List[float]:
        """计算一组向量的逐维平均值，作为简单且稳定的兴趣中心表示（假设所有向量维度一致）。"""
        if not vectors:
            return []
        dimension = len(vectors[0])
        return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """L2 归一化；零向量原样返回。"""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    # ------------------------------------------------------------------
    # 向量补齐
    # ------------------------------------------------------------------

    def _hydrate_vectors_with_metadata(
        self,
        requested_ids: List[str],
        milvus_embeddings: List[Dict[str, Any]],
        config: Any,
        label: str,
    ) -> List[Dict[str, Any]]:
        """为论文 ID 补齐可用向量，并标记向量来源。

        该方法会优先使用向量库里的现成数据；缺失时先尝试从本地论文表或 arXiv 回填并写回向量库，
        最后再退化为使用数据库中的文本临时生成 embedding（不写库）。

        Returns:
            ``[{"arxiv_id", "vector", "source"}]``，按 requested_ids 顺序排列；
            source 为 "milvus"（向量库已有或回填成功）或 "fallback"（临时生成）。
            三级兜底都失败的 ID 不会出现在结果中。
        """
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
                "Missing paper embeddings for %s papers in collection %s, attempting backfill: %s",
                label,
                self.collection_name,
                ", ".join(sorted(missing_ids)),
            )

        # 第一优先级：尝试把缺失论文补齐并重新写入向量库，下次即可直接复用。
        recovered_vectors = self._backfill_missing_vectors_from_arxiv(missing_ids, label)
        for arxiv_id, vector in recovered_vectors.items():
            vector_records[arxiv_id] = {"arxiv_id": arxiv_id, "vector": vector, "source": "milvus"}

        remaining_missing = [arxiv_id for arxiv_id in missing_ids if arxiv_id not in recovered_vectors]
        if remaining_missing:
            # 第二优先级：如果本地数据库里已有论文文本，就直接临时补 embedding，避免整条链路失败。
            # 能走到这里的通常是“回填时写向量库 / 论文表失败”的论文；本地没有记录的论文这里也无能为力。
            fallback_papers = [paper for paper in (self.paper_catalog_store.get_paper(arxiv_id) for arxiv_id in remaining_missing) if paper]
            embedded_fallbacks = self._embed_papers(fallback_papers, config)
            fallback_map = {item["arxiv_id"]: item["vector"] for item in embedded_fallbacks if item.get("arxiv_id") and item.get("vector")}
            for arxiv_id in remaining_missing:
                if arxiv_id in fallback_map:
                    vector_records[arxiv_id] = {"arxiv_id": arxiv_id, "vector": fallback_map[arxiv_id], "source": "fallback"}

        return [vector_records[arxiv_id] for arxiv_id in requested_ids if arxiv_id in vector_records]

    def _backfill_missing_vectors_from_arxiv(
        self,
        missing_ids: List[str],
        label: str,
    ) -> Dict[str, List[float]]:
        """为缺失向量的论文执行回填：生成 embedding 并同时写入向量库与论文表。

        论文元数据优先取本地论文表，本地没有时才（限流地）请求 arXiv。

        Returns:
            回填成功的 {arxiv_id: 向量}；失败的论文只记日志，由调用方继续兜底。
        """
        recovered_vectors: Dict[str, List[float]] = {}

        for arxiv_id in missing_ids:
            source_paper = self.paper_catalog_store.get_paper(arxiv_id)
            if source_paper is None:
                try:
                    # 本地没有论文详情时，再访问 arXiv，避免不必要的外部请求。
                    source_paper = self._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to fetch arXiv paper %s for %s backfill: %s", arxiv_id, label, exc)
                    source_paper = None

            if not source_paper:
                continue

            try:
                normalized_paper = self._normalize_paper_record(source_paper, arxiv_id)
                # 成功回填时同时补写向量库和论文表，保证下次可以直接复用。
                embedding_id, embedding_vector = self._build_and_insert_paper_embedding(normalized_paper)
                stored = self._build_paper_store_payload(normalized_paper, embedding_id=str(embedding_id))
                if not self.paper_catalog_store.add_paper(stored):
                    raise HTTPException(status_code=500, detail=f"Failed to store paper {arxiv_id}")
                recovered_vectors[arxiv_id] = embedding_vector
                logger.debug("Backfilled missing paper %s into paper store and vector store", arxiv_id)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to materialize missing paper %s for %s backfill: %s", arxiv_id, label, exc)

        return recovered_vectors

    # ------------------------------------------------------------------
    # 兴趣聚类
    # ------------------------------------------------------------------

    def _cluster_interest_vectors(self, liked_records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """对点赞论文向量做 HDBSCAN 聚类，返回（兴趣簇列表, 弱兴趣池）。

        - 兴趣簇：按簇大小降序，最多保留 max_interest_clusters 个，cluster_id 重新编号为 cluster_0..N；
        - 弱兴趣池：被 HDBSCAN 判为噪声的点赞论文，单独求中心，避免它们被强行并入某个簇；
        - 样本不足或 HDBSCAN 失败时返回 ([], None)，由调用方退回均值向量。
        """
        positive_config = self._get_positive_profile_config()
        min_liked_for_clustering = int(positive_config["min_liked_for_clustering"])
        valid_records = [record for record in liked_records if record.get("vector")]
        # HDBSCAN 至少需要两个有效样本；业务阈值仍由 positive.min_liked_for_clustering 控制。
        if len(valid_records) < max(2, min_liked_for_clustering):
            return [], None

        vectors = [self._normalize_vector([float(value) for value in record["vector"]]) for record in valid_records]
        paper_ids = [str(record.get("arxiv_id", "") or "").strip() for record in valid_records]

        try:
            labels = self._run_hdbscan(vectors, self.get_clustering_config())
        except Exception as exc:  # pragma: no cover
            logger.debug("HDBSCAN interest clustering failed; using mean fallback: %s", exc)
            return [], None

        cluster_members, weak_member_indices = self._group_cluster_labels(labels)

        weak_interest_pool: Optional[Dict[str, Any]] = None
        if weak_member_indices:
            weak_paper_ids = [paper_ids[index] for index in weak_member_indices if paper_ids[index]]
            weak_interest_pool = {
                "pool_id": "weak_interest_pool",
                "paper_count": len(weak_paper_ids),
                "paper_ids": weak_paper_ids,
                "centroid_vector": self._normalize_vector(self._mean_vector([vectors[index] for index in weak_member_indices])),
            }

        if not cluster_members:
            if weak_interest_pool:
                logger.debug("HDBSCAN produced only weak-interest noise points; keeping weak pool and using mean fallback")
            else:
                logger.debug("HDBSCAN produced no stable interest clusters; using mean fallback")
            return [], weak_interest_pool

        max_interest_clusters = int(positive_config["max_interest_clusters"])
        if len(cluster_members) > max_interest_clusters:
            logger.info(
                "HDBSCAN produced %s interest clusters; keeping top %s by size",
                len(cluster_members),
                max_interest_clusters,
            )
        clusters = self._build_ranked_clusters(
            cluster_members=cluster_members,
            vectors=vectors,
            paper_ids=paper_ids,
            id_prefix="cluster",
            max_clusters=max_interest_clusters,
        )
        return clusters, weak_interest_pool

    @staticmethod
    def _run_hdbscan(vectors: List[List[float]], clustering_config: Dict[str, Any]) -> List[int]:
        """按配置运行 HDBSCAN，返回每个向量的簇标签（-1 表示噪声点）。"""
        clusterer = HDBSCAN(
            min_cluster_size=int(clustering_config["min_cluster_size"]),
            min_samples=int(clustering_config["min_samples"]),
            metric=str(clustering_config["metric"]).strip(),
            cluster_selection_method=str(clustering_config["cluster_selection_method"]).strip(),
            allow_single_cluster=bool(clustering_config["allow_single_cluster"]),
        )
        return list(clusterer.fit_predict(vectors))

    @staticmethod
    def _group_cluster_labels(labels: List[int]) -> tuple[Dict[int, List[int]], List[int]]:
        """把 HDBSCAN 标签分组为 {簇标签: [向量下标]} 与噪声点下标列表。"""
        cluster_members: Dict[int, List[int]] = {}
        noise_indices: List[int] = []
        for vector_index, cluster_label in enumerate(labels):
            if cluster_label < 0:
                noise_indices.append(vector_index)
                continue
            cluster_members.setdefault(int(cluster_label), []).append(vector_index)
        return cluster_members, noise_indices

    def _build_ranked_clusters(
        self,
        *,
        cluster_members: Dict[int, List[int]],
        vectors: List[List[float]],
        paper_ids: List[str],
        id_prefix: str,
        max_clusters: int,
        keep_label: bool = False,
    ) -> List[Dict[str, Any]]:
        """由簇成员构建簇对象：计算归一化中心，按簇大小降序（同大小按首篇论文 ID）排序、截断并重新编号。

        Args:
            keep_label: 是否在结果中保留 HDBSCAN 原始标签（cluster_label），负向簇需要用于诊断。
        """
        clusters: List[Dict[str, Any]] = []
        for cluster_label, members in cluster_members.items():
            member_paper_ids = [paper_ids[index] for index in members if paper_ids[index]]
            cluster: Dict[str, Any] = {
                "cluster_id": f"{id_prefix}_{cluster_label}",
                "centroid_vector": self._normalize_vector(self._mean_vector([vectors[index] for index in members])),
                "paper_count": len(member_paper_ids),
                "paper_ids": member_paper_ids,
            }
            if keep_label:
                cluster["cluster_label"] = int(cluster_label)
            clusters.append(cluster)

        clusters.sort(key=lambda item: (-int(item.get("paper_count", 0) or 0), (item.get("paper_ids") or [""])[0]))
        clusters = clusters[:max_clusters]
        # 截断后按新顺序重新编号，保证 ID 连续且与簇大小排名一致。
        for index, cluster in enumerate(clusters):
            cluster["cluster_id"] = f"{id_prefix}_{index}"
        return clusters
