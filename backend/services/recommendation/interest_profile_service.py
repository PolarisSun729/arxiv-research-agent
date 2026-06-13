from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)
NEGATIVE_FEEDBACK_PROFILE_VERSION = "negative_feedback_profile_v1"


class InterestProfileService:
    def generate_user_interest_vector(
        self,
        user_id: str,
        negative_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """根据用户正向反馈生成兴趣向量，并把负向反馈作为独立信号保存。

        该方法是推荐系统个性化画像的入口。主兴趣向量只表达“用户喜欢什么”，
        点踩论文只记录为“不要再推荐什么”的负向信号，供排序、过滤和解释阶段消费。
        """
        if negative_weight is not None:
            logger.debug(
                "negative_weight is deprecated and ignored when generating vector_data for user %s",
                user_id,
            )
        positive_config = self._get_positive_profile_config()
        negative_config = self._get_negative_profile_config()
        min_liked_for_vector = int(positive_config["min_liked_for_vector"])
        negative_enabled = bool(negative_config["enabled"])
        liked_ids = self.db_service.get_liked_papers(user_id=user_id)
        disliked_ids = self.db_service.get_disliked_papers(user_id=user_id)

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
        disliked_embeddings = []
        if negative_enabled and disliked_ids:
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
        disliked_records = []
        if negative_enabled and disliked_ids:
            disliked_records = self._hydrate_vectors_with_metadata(
                requested_ids=disliked_ids,
                milvus_embeddings=disliked_embeddings,
                config=config,
                label="disliked",
            )

        if len(liked_records) < min_liked_for_vector:
            raise HTTPException(status_code=500, detail="No reusable embeddings found for liked papers")

        # 主兴趣方向只来自 liked papers，避免 dislike 原因多样时把正向兴趣中心拉偏。
        liked_mean = self._mean_vector([record["vector"] for record in liked_records])
        fallback_interest_vector = self._normalize_vector(liked_mean)
        disliked_mean = self._mean_vector([record["vector"] for record in disliked_records]) if disliked_records else []
        disliked_vector_data = self._normalize_vector(disliked_mean) if negative_enabled and disliked_mean else None
        weak_interest_pool: Optional[Dict[str, Any]] = None
        interest_clusters: List[Dict[str, Any]] = []
        profile_mode = "mean"
        min_liked_for_clustering = int(positive_config["min_liked_for_clustering"])
        if len(liked_records) >= min_liked_for_clustering:
            try:
                # 当用户正反馈足够多时，进一步拆分成多个兴趣簇，提升召回覆盖面和解释性。
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
                    logger.debug(
                        "User %s liked papers clustered into %s interest clusters: %s",
                        user_id,
                        len(interest_clusters),
                        cluster_summary,
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
        disliked_milvus_count = sum(1 for record in disliked_records if record.get("source") == "milvus")
        disliked_fallback_count = sum(1 for record in disliked_records if record.get("source") == "fallback")
        disliked_unresolved_count = len(disliked_ids) - len(disliked_records) if negative_enabled else len(disliked_ids)
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
            "disliked_milvus_count": disliked_milvus_count,
            "liked_fallback_count": liked_fallback_count,
            "disliked_fallback_count": disliked_fallback_count,
            "liked_unresolved_count": liked_unresolved_count,
            "disliked_unresolved_count": disliked_unresolved_count,
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
            "cluster_summary": [
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "paper_count": cluster.get("paper_count", 0),
                    "paper_ids": cluster.get("paper_ids", []),
                }
                for cluster in interest_clusters
            ],
        }

    def _get_positive_profile_config(self) -> Dict[str, Any]:
        """读取正向兴趣画像配置，作为主向量和兴趣簇阈值的唯一来源。"""
        return dict(self.RECOMMENDATION_CONFIG["profile"]["positive"])

    def _get_negative_profile_config(self) -> Dict[str, Any]:
        """读取负向反馈画像配置，避免负向信号保存策略散落在业务代码中。"""
        return dict(self.RECOMMENDATION_CONFIG["profile"]["negative"])

    def _build_negative_feedback_profile(
        self,
        *,
        disliked_ids: List[str],
        disliked_records: List[Dict[str, Any]],
        disliked_vector_data: Optional[List[float]],
        negative_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """为点踩论文构建独立负向画像，失败时按配置退回实例级反馈。"""
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
            "hard_exclude_ids": [str(arxiv_id).strip() for arxiv_id in disliked_ids if str(arxiv_id).strip()],
            "examples": examples,
            "clusters": clusters,
            "stats": stats,
        }

    def _empty_negative_feedback_profile(self, disliked_ids: List[str], *, enabled: bool) -> Dict[str, Any]:
        """生成稳定的空负向画像，保证没有 disliked papers 时调用方不需要判空分支。"""
        stats = {
            "enabled": enabled,
            "mode": "none",
            "total_disliked": len(disliked_ids),
            "usable_disliked": 0,
            "unresolved_disliked": len(disliked_ids),
            "milvus_count": 0,
            "fallback_count": 0,
            "stored_examples": 0,
            "negative_cluster_count": 0,
            "vector_available": False,
            "participates_in_main_vector": False,
        }
        return {
            "version": NEGATIVE_FEEDBACK_PROFILE_VERSION,
            "enabled": enabled,
            "mode": "none",
            "hard_exclude_ids": [str(arxiv_id).strip() for arxiv_id in disliked_ids if str(arxiv_id).strip()],
            "examples": [],
            "clusters": [],
            "stats": stats,
        }

    def _build_disliked_paper_examples(
        self,
        disliked_records: List[Dict[str, Any]],
        negative_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """保存少量点踩样本及其向量，供样本不足或聚类失败时做实例级惩罚。"""
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
            paper = self.db_service.get_paper(arxiv_id) or {}
            examples.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": str(paper.get("title", "") or ""),
                    "categories": self._coerce_category_list(paper.get("categories")),
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
        """把足量点踩论文聚成多个负向方向，避免单一负向均值掩盖不同拒绝原因。"""
        min_disliked_for_clustering = int(negative_config["min_disliked_for_clustering"])
        valid_records = [record for record in disliked_records if record.get("vector")]
        if len(valid_records) < min_disliked_for_clustering or len(valid_records) < 2:
            return []

        vectors = [self._normalize_vector([float(value) for value in record["vector"]]) for record in valid_records]
        paper_ids = [str(record.get("arxiv_id", "") or "").strip() for record in valid_records]
        clustering_config = negative_config["clustering"]
        clusterer = HDBSCAN(
            min_cluster_size=int(clustering_config["min_cluster_size"]),
            min_samples=int(clustering_config["min_samples"]),
            metric=str(clustering_config["metric"]).strip(),
            cluster_selection_method=str(clustering_config["cluster_selection_method"]).strip(),
            allow_single_cluster=bool(clustering_config["allow_single_cluster"]),
        )
        labels = clusterer.fit_predict(vectors)

        cluster_members: Dict[int, List[int]] = {}
        for vector_index, cluster_label in enumerate(labels):
            if cluster_label < 0:
                continue
            cluster_members.setdefault(int(cluster_label), []).append(vector_index)

        clusters: List[Dict[str, Any]] = []
        for cluster_label, members in cluster_members.items():
            member_paper_ids = [paper_ids[index] for index in members if paper_ids[index]]
            cluster_vectors = [vectors[index] for index in members]
            clusters.append(
                {
                    "cluster_id": f"negative_cluster_{cluster_label}",
                    "cluster_label": int(cluster_label),
                    "centroid_vector": self._normalize_vector(self._mean_vector(cluster_vectors)),
                    "paper_count": len(member_paper_ids),
                    "paper_ids": member_paper_ids,
                }
            )

        clusters.sort(key=lambda item: (-int(item.get("paper_count", 0) or 0), (item.get("paper_ids") or [""])[0]))
        max_negative_clusters = int(negative_config["max_negative_clusters"])
        clusters = clusters[:max_negative_clusters]
        for index, cluster in enumerate(clusters):
            cluster["cluster_id"] = f"negative_cluster_{index}"
        return clusters

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
        """把负向反馈的建模方式显式化，便于排序解释和旧数据兼容判断。"""
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

    @staticmethod
    def _coerce_category_list(categories: Any) -> List[str]:
        """将论文分类整理成列表，保证负向样本字段对前端和调试输出稳定。"""
        if isinstance(categories, list):
            source = categories
        elif isinstance(categories, tuple):
            source = list(categories)
        elif isinstance(categories, str):
            source = categories.replace(";", ",").replace("|", ",").split(",")
        else:
            source = []
        return [str(item).strip() for item in source if str(item).strip()]

    def _is_vector_stale(self, vector_updated_at: Optional[str], latest_preference_ts: Optional[str]) -> bool:
        """判断已存兴趣向量是否早于最新用户行为，以决定是否需要重建。"""
        if not vector_updated_at or not latest_preference_ts:
            return False
        vector_dt = self._parse_datetime(vector_updated_at)
        preference_dt = self._parse_datetime(latest_preference_ts)
        if not vector_dt or not preference_dt:
            return False
        return vector_dt < preference_dt

    def _mean_vector(self, vectors: List[List[float]]) -> List[float]:
        """计算一组向量的逐维平均值，作为简单且稳定的兴趣中心表示。"""
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
        """为论文 ID 补齐可用向量，并标记向量来源。

        该方法会优先使用向量库里的现成数据；缺失时先尝试从 arXiv/OAI 回填并写回向量库，
        最后再退化为使用数据库中的文本重新生成 embedding。
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
                "Missing paper embeddings for %s papers in collection %s, attempting OAI DB backfill: %s",
                label,
                self.collection_name,
                ", ".join(sorted(missing_ids)),
            )

        # 第一优先级：尝试把缺失论文从外部源补齐并重新写入向量库。
        recovered_vectors, _, _ = self._backfill_missing_vectors_from_arxiv(missing_ids, label)
        for arxiv_id, vector in recovered_vectors.items():
            vector_records[arxiv_id] = {"arxiv_id": arxiv_id, "vector": vector, "source": "milvus"}

        remaining_missing = [arxiv_id for arxiv_id in missing_ids if arxiv_id not in recovered_vectors]
        if remaining_missing:
            # 第二优先级：如果本地数据库里已有论文文本，就直接临时补 embedding，避免整条链路失败。
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
        """为缺失向量的论文执行回填，并返回恢复成功与失败的 ID 列表。"""
        recovered_vectors: Dict[str, List[float]] = {}
        recovered_ids: List[str] = []
        unresolved_ids: List[str] = []

        for arxiv_id in missing_ids:
            existing_paper = self.db_service.get_paper(arxiv_id)
            source_paper = existing_paper
            if source_paper is None:
                try:
                    # 本地没有论文详情时，再访问 arXiv，避免不必要的外部请求。
                    source_paper = self._fetch_paper_from_arxiv_with_rate_limit(arxiv_id)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to fetch arXiv paper %s for %s backfill: %s", arxiv_id, label, exc)
                    source_paper = None

            if not source_paper:
                unresolved_ids.append(arxiv_id)
                continue

            try:
                normalized_paper = self._normalize_paper_record(source_paper, arxiv_id)
                # 成功回填时同时补写向量库和论文表，保证下次可以直接复用。
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
                logger.debug("Backfilled missing paper %s into paper store and vector store", arxiv_id)
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
        positive_config = self._get_positive_profile_config()
        min_liked_for_clustering = int(positive_config["min_liked_for_clustering"])
        if len(liked_records) < min_liked_for_clustering:
            return [], None

        valid_records = [record for record in liked_records if record.get("vector")]
        vectors = [self._normalize_vector([float(value) for value in record["vector"]]) for record in valid_records]
        paper_ids = [str(record.get("arxiv_id", "") or "").strip() for record in valid_records]
        # HDBSCAN 至少需要两个有效样本；业务阈值仍由 positive.min_liked_for_clustering 控制。
        if len(vectors) < min_liked_for_clustering or len(vectors) < 2:
            return [], None

        clustering_config = self.get_clustering_config()
        min_cluster_size = int(clustering_config["min_cluster_size"])
        min_samples = int(clustering_config["min_samples"])
        metric = str(clustering_config["metric"]).strip()
        cluster_selection_method = str(clustering_config["cluster_selection_method"]).strip()
        allow_single_cluster = bool(clustering_config["allow_single_cluster"])

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
            logger.debug("HDBSCAN interest clustering failed; using mean fallback: %s", exc)
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
                logger.debug("HDBSCAN produced only weak-interest noise points; keeping weak pool and using mean fallback")
                return [], weak_interest_pool
            logger.debug("HDBSCAN produced no stable interest clusters; using mean fallback")
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
        max_interest_clusters = int(positive_config["max_interest_clusters"])
        if len(clusters) > max_interest_clusters:
            logger.info(
                "HDBSCAN produced %s interest clusters; keeping top %s by size",
                len(clusters),
                max_interest_clusters,
            )
            clusters = clusters[:max_interest_clusters]
        for index, cluster in enumerate(clusters):
            cluster["cluster_id"] = f"cluster_{index}"
        return clusters, weak_interest_pool
