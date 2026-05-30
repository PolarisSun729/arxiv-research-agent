from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CandidateRecallService:
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

    def _fetch_recent_db_candidates(
        self,
        liked_category_freq: Counter,
        max_age_months: int,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        categories = self.RECOMMEND_CANDIDATE_CATEGORIES[:]
        logger.info(
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
        normalized_excluded_ids = [str(arxiv_id).strip() for arxiv_id in excluded_ids if str(arxiv_id).strip()]
        candidates_by_id: Dict[str, Dict[str, Any]] = {}

        for cluster_index, cluster in enumerate(interest_clusters):
            centroid_vector = cluster.get("centroid_vector") or []
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
        candidate_key = (float(candidate.get("recall_cluster_similarity", 0.0) or 0.0), -int(candidate.get("recall_cluster_rank", 0) or 0))
        existing_key = (float(existing_candidate.get("recall_cluster_similarity", 0.0) or 0.0), -int(existing_candidate.get("recall_cluster_rank", 0) or 0))
        return candidate_key > existing_key
