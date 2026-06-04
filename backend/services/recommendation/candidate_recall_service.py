from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CandidateRecallService:
    def _build_liked_category_frequency(self, liked_papers: List[Dict[str, Any]]) -> Counter:
        """统计用户点赞论文中的分类频次，作为后续召回与打分的偏好信号。

        该方法不会直接决定最终排序，而是为候选池构建提供一个轻量的兴趣侧信号，
        帮助系统判断用户更常接触哪些 arXiv 分类。
        """
        counter: Counter = Counter()
        for paper in liked_papers:
            # 一篇论文可能挂多个分类，因此这里逐个拆分累计，避免遗漏交叉领域兴趣。
            for category in self._split_categories(paper.get("categories")):
                counter[category] += 1
        return counter

    def _deduplicate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        excluded_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """按 arXiv ID 对候选论文去重，并过滤用户明确不应再次看到的论文。"""
        seen = set(excluded_ids)
        deduped = []
        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            # 空 ID 无法参与后续物化、向量查询和打分，因此直接跳过。
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            deduped.append(candidate)
        return deduped

    def _fetch_recent_db_candidates(
        self,
        liked_category_freq: Counter,
        max_age_months: int,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """从 OAI 数据库读取近期论文，作为无兴趣簇召回时的基础候选池。

        该方法优先使用系统预设的推荐分类范围，而不是动态拼接用户分类，
        以保证召回池稳定可控，并避免因用户历史过少导致召回范围过窄。
        """
        categories = self.RECOMMEND_CANDIDATE_CATEGORIES[:]
        logger.debug(
            "Fetching recent OAI DB candidates with categories=%s, max_age_months=%s, max_results=%s",
            categories,
            max_age_months,
            max_results,
        )
        papers = self.oai_db_service.get_recent_papers(
            categories=categories,
            max_age_months=max_age_months,
            max_results=max_results,
        )
        return [
            {
                "arxiv_id": str(paper.get("arxiv_id", "") or "").strip(),
                "title": str(paper.get("title", "") or "").strip(),
                "authors": paper.get("authors", []),
                "abstract": str(paper.get("abstract", "") or "").strip(),
                "categories": paper.get("categories", []),
                "published_date": str(paper.get("created") or paper.get("updated") or paper.get("oai_datestamp") or "").strip(),
                "url": str(paper.get("abs_url") or paper.get("pdf_url") or "").strip(),
                "score": 0.0,
                "abs_url": paper.get("abs_url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "primary_category": paper.get("primary_category", ""),
                "oai_datestamp": paper.get("oai_datestamp", ""),
                "fetched_at": paper.get("fetched_at", ""),
            }
            for paper in papers
        ]

    def _build_category_query(
        self,
        liked_category_freq: Counter,
        max_categories: Optional[int] = None,
    ) -> str:
        """构造分类检索表达式，便于文本检索或接口查询复用统一分类范围。"""
        if max_categories is None:
            max_categories = self.RECOMMENDATION_CONFIG["category_query_max_categories"]
        categories = self.RECOMMEND_CANDIDATE_CATEGORIES[:max_categories] if max_categories > 0 else self.RECOMMEND_CANDIDATE_CATEGORIES
        return " OR ".join(f"cat:{category}" for category in categories)

    def _fetch_cluster_recall_candidates(
        self,
        interest_clusters: List[Dict[str, Any]],
        excluded_ids: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """基于兴趣簇中心向量执行相似检索，生成个性化候选集。

        该方法会让每个兴趣簇独立召回一批论文，再按 arXiv ID 合并结果。
        这样既能保留不同兴趣子方向的覆盖度，也能记录论文同时命中多个兴趣簇的轨迹。
        """
        normalized_excluded_ids = [str(arxiv_id).strip() for arxiv_id in excluded_ids if str(arxiv_id).strip()]
        candidates_by_id: Dict[str, Dict[str, Any]] = {}

        for cluster_index, cluster in enumerate(interest_clusters):
            centroid_vector = cluster.get("centroid_vector") or []
            # 没有中心向量的簇无法执行向量召回，直接跳过该簇。
            if not centroid_vector:
                continue

            cluster_id = str(cluster.get("cluster_id") or f"cluster_{cluster_index}").strip() or f"cluster_{cluster_index}"
            try:
                recalled_papers = self.vector_store_service.search_similar_papers(
                    collection_name=self.collection_name,
                    query_vector=[float(value) for value in centroid_vector],
                    top_k=max(1, int(top_k)),
                    filter_arxiv_ids=normalized_excluded_ids,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed cluster recall for user cluster %s: %s", cluster_id, exc)
                continue

            for rank, paper in enumerate(recalled_papers, start=1):
                arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
                if not arxiv_id:
                    continue
                similarity = float(paper.get("similarity_score", paper.get("score", 0.0)) or 0.0)
                # 保留每个簇的命中明细，方便后续解释推荐理由和调试召回行为。
                recall_hit = {"cluster_id": cluster_id, "similarity": similarity, "rank": rank}
                candidate = {
                    **paper,
                    "recall_source": "cluster_recall",
                    "recall_cluster_id": cluster_id,
                    "recall_cluster_similarity": similarity,
                    "recall_cluster_rank": rank,
                    "recall_cluster_hits": [recall_hit],
                }
                existing_candidate = candidates_by_id.get(arxiv_id)
                if existing_candidate is None:
                    candidates_by_id[arxiv_id] = candidate
                    continue
                # 当同一论文命中多个兴趣簇时，既要汇总所有命中记录，也要挑选最优主命中簇。
                existing_hits = list(existing_candidate.get("recall_cluster_hits", []))
                combined_hits = self._sort_recall_hits(existing_hits + [recall_hit])
                existing_candidate["recall_cluster_hits"] = combined_hits
                if self._is_better_cluster_recall_candidate(candidate, existing_candidate):
                    candidate["recall_cluster_hits"] = combined_hits
                    candidates_by_id[arxiv_id] = candidate

        candidates = list(candidates_by_id.values())
        candidates.sort(
            key=lambda item: (
                -float(item.get("recall_cluster_similarity", 0.0) or 0.0),
                int(item.get("recall_cluster_rank", 0) or 0),
                str(item.get("arxiv_id", "") or ""),
            )
        )
        return candidates

    def _sort_recall_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对同一论文的多簇命中记录去重并排序，保留最有价值的召回轨迹。"""
        unique_hits: List[Dict[str, Any]] = []
        seen = set()
        for hit in hits:
            cluster_id = str(hit.get("cluster_id", "") or "").strip()
            rank = int(hit.get("rank", 0) or 0)
            similarity = float(hit.get("similarity", 0.0) or 0.0)
            key = (cluster_id, rank, similarity)
            if key in seen:
                continue
            seen.add(key)
            unique_hits.append({"cluster_id": cluster_id or None, "similarity": similarity, "rank": rank})
        unique_hits.sort(key=lambda item: (-float(item.get("similarity", 0.0) or 0.0), int(item.get("rank", 0) or 0), str(item.get("cluster_id", "") or "")))
        return unique_hits

    def _is_better_cluster_recall_candidate(self, candidate: Dict[str, Any], existing_candidate: Dict[str, Any]) -> bool:
        """比较两个同 ID 候选的主召回质量，并优先保留更强的簇命中结果。"""
        candidate_key = (float(candidate.get("recall_cluster_similarity", 0.0) or 0.0), -int(candidate.get("recall_cluster_rank", 0) or 0))
        existing_key = (float(existing_candidate.get("recall_cluster_similarity", 0.0) or 0.0), -int(existing_candidate.get("recall_cluster_rank", 0) or 0))
        return candidate_key > existing_key
