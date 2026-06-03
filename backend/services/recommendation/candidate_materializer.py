from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class CandidateMaterializer:
    def record_user_paper_action(
        self,
        user_id: str,
        arxiv_id: str,
        action_type: str,
        paper_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_arxiv_id = str(arxiv_id or "").strip()
        normalized_action = str(action_type or "").strip()
        if not normalized_arxiv_id:
            raise HTTPException(status_code=400, detail="arxiv_id is required")
        if not normalized_action:
            raise HTTPException(status_code=400, detail="action_type is required")

        paper = self._ensure_paper_materialized(normalized_arxiv_id, paper_payload=paper_payload)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {normalized_arxiv_id} could not be materialized")

        success = self.db_service.record_user_paper_action(
            user_id=user_id,
            arxiv_id=normalized_arxiv_id,
            action_type=normalized_action,
            metadata=metadata,
        )
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to record paper action {normalized_action}")

        try:
            self.memory_service.update_profile_from_paper_action(
                user_id=user_id,
                arxiv_id=normalized_arxiv_id,
                action_type=normalized_action,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to update profile from paper action: user_id=%s arxiv_id=%s action=%s error=%s", user_id, normalized_arxiv_id, normalized_action, exc)

        return {
            "status": "success",
            "message": f"Paper action '{normalized_action}' saved successfully",
            "arxiv_id": normalized_arxiv_id,
            "action_type": normalized_action,
            "paper": paper,
            "metadata": metadata or {},
        }

    def record_user_paper_preference(
        self,
        user_id: str,
        arxiv_id: str,
        liked: bool,
        paper_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_arxiv_id = str(arxiv_id or "").strip()
        if not normalized_arxiv_id:
            raise HTTPException(status_code=400, detail="arxiv_id is required")

        paper = self._ensure_paper_materialized(normalized_arxiv_id, paper_payload=paper_payload)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {normalized_arxiv_id} could not be materialized")

        if liked:
            success = self.db_service.add_liked_paper(user_id=user_id, arxiv_id=normalized_arxiv_id)
            action = "liked"
        else:
            success = self.db_service.add_disliked_paper(user_id=user_id, arxiv_id=normalized_arxiv_id)
            action = "disliked"

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to add paper to {action} list")

        try:
            self.memory_service.update_profile_from_preference(
                user_id=user_id,
                arxiv_id=normalized_arxiv_id,
                action_type="like" if liked else "dislike",
                paper_payload=paper,
            )
        except Exception as exc:
            logger.warning("Failed to update profile from preference: user_id=%s arxiv_id=%s liked=%s error=%s", user_id, normalized_arxiv_id, liked, exc)

        return {
            "status": "success",
            "message": f"Paper added to {action} list and materialized into paper store",
            "arxiv_id": normalized_arxiv_id,
            "paper": paper,
        }

    def _materialize_candidate_papers_for_recommendation(
        self,
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        stats = {"total": len(candidates), "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0}
        materialized_by_id: Dict[str, Dict[str, Any]] = {}
        batch_jobs: List[Dict[str, Any]] = []
        embedding_config = self.get_embedding_config()
        candidate_ids = [str(candidate.get("arxiv_id", "") or "").strip() for candidate in candidates if str(candidate.get("arxiv_id", "") or "").strip()]
        candidate_embedding_rows = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(collection_name=self.collection_name, arxiv_ids=candidate_ids)
        candidate_embedding_map = {str(item.get("arxiv_id", "") or "").strip(): item for item in candidate_embedding_rows if item.get("arxiv_id")}

        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if not arxiv_id:
                stats["unresolved"] += 1
                continue
            try:
                existing_paper = self.db_service.get_paper(arxiv_id)
                existing_embedding = candidate_embedding_map.get(arxiv_id)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to inspect candidate paper %s before materialization: %s", arxiv_id, exc)
                existing_paper = None
                existing_embedding = None

            if existing_paper and existing_paper.get("embedding_id") and existing_embedding:
                materialized_by_id[arxiv_id] = existing_paper
                stats["reused_existing"] += 1
                continue

            if existing_paper and existing_embedding and not existing_paper.get("embedding_id"):
                updated = self.db_service.update_paper_embedding(
                    arxiv_id=arxiv_id,
                    embedding_id=int(existing_embedding.get("embedding_id") or 0),
                    embedding_model=str(existing_embedding.get("embedding_model") or existing_paper.get("embedding_model") or embedding_config.model_name),
                )
                refreshed = self.db_service.get_paper(arxiv_id) if updated else None
                materialized_by_id[arxiv_id] = refreshed or {**existing_paper, "embedding_id": existing_embedding.get("embedding_id")}
                stats["db_only"] += 1
                continue

            if existing_embedding and not existing_paper:
                stored = {
                    "arxiv_id": str(candidate.get("arxiv_id", "") or "").strip(),
                    "title": str(candidate.get("title", "") or "").strip(),
                    "authors": candidate.get("authors", []),
                    "abstract": str(candidate.get("abstract", "") or "").strip(),
                    "categories": candidate.get("categories", []),
                    "published_date": str(candidate.get("published_date", "") or "").strip(),
                    "url": str(candidate.get("url", "") or candidate.get("abs_url", "") or candidate.get("pdf_url", "") or "").strip(),
                    "embedding_id": str(existing_embedding.get("embedding_id") or ""),
                    "embedding_model": str(existing_embedding.get("embedding_model") or embedding_config.model_name),
                }
                if self.db_service.add_paper(stored):
                    materialized_by_id[arxiv_id] = self.db_service.get_paper(arxiv_id) or stored
                    stats["db_only"] += 1
                else:
                    stats["unresolved"] += 1
                continue

            normalized_paper = self._normalize_paper_record(candidate, arxiv_id)
            text_to_embed = self.embedding_service.build_paper_embedding_text(normalized_paper["title"], normalized_paper["abstract"])
            if not text_to_embed:
                stats["unresolved"] += 1
                continue

            try:
                embedding = self.embedding_service.create_single_embedding(
                    text_to_embed,
                    provider=embedding_config.provider,
                    model=embedding_config.model_name,
                    api_key=embedding_config.api_key,
                    base_url=embedding_config.base_url,
                    dimension=embedding_config.dimension,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to embed candidate paper %s for recommendation: %s", arxiv_id, exc)
                stats["unresolved"] += 1
                continue

            batch_jobs.append({"arxiv_id": arxiv_id, "normalized_paper": normalized_paper, "embedding": [float(value) for value in embedding]})

        if batch_jobs:
            try:
                batch_insert_payload = [
                    {
                        "embedding": job["embedding"],
                        "metadata": {
                            "content": job["normalized_paper"]["abstract"],
                            "arxiv_id": job["normalized_paper"]["arxiv_id"],
                            "title": job["normalized_paper"]["title"],
                            "authors": job["normalized_paper"]["authors"],
                            "categories": job["normalized_paper"]["categories"],
                            "published_date": job["normalized_paper"]["published_date"],
                            "url": job["normalized_paper"]["url"],
                            "embedding_model": job["normalized_paper"]["embedding_model"],
                        },
                    }
                    for job in batch_jobs
                ]
                embedding_count = self.vector_store_service.insert_embeddings(self.collection_name, batch_insert_payload)
                stats["batch_embedded"] = len(batch_jobs)
                stats["batch_inserted"] = int(embedding_count)
                for job in batch_jobs:
                    stored = {
                        "arxiv_id": job["normalized_paper"]["arxiv_id"],
                        "title": job["normalized_paper"]["title"],
                        "authors": job["normalized_paper"]["authors"],
                        "abstract": job["normalized_paper"]["abstract"],
                        "categories": job["normalized_paper"]["categories"],
                        "published_date": job["normalized_paper"]["published_date"],
                        "url": job["normalized_paper"]["url"],
                        "embedding_id": "",
                        "embedding_model": job["normalized_paper"]["embedding_model"],
                    }
                    if self.db_service.add_paper(stored):
                        materialized_by_id[job["arxiv_id"]] = self.db_service.get_paper(job["arxiv_id"]) or stored
                    else:
                        stats["unresolved"] += 1
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed batch materialization for recommendation candidates: %s", exc)
                stats["unresolved"] += len(batch_jobs)

        ordered_candidates = []
        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if not arxiv_id or arxiv_id not in materialized_by_id:
                continue
            merged_candidate = {**candidate, **materialized_by_id[arxiv_id]}
            for key, value in candidate.items():
                if key.startswith("recall_") or key in {"similarity_score", "score", "distance", "metadata"}:
                    merged_candidate[key] = value
            ordered_candidates.append(merged_candidate)
        return ordered_candidates, stats

    def _ensure_paper_materialized(self, arxiv_id: str, paper_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        existing_paper = self.db_service.get_paper(arxiv_id)
        existing_embedding = self._get_existing_paper_embedding(arxiv_id)

        if existing_paper and existing_paper.get("embedding_id") and existing_embedding:
            return existing_paper

        source_paper = paper_payload or existing_paper or existing_embedding
        if not source_paper:
            raise HTTPException(status_code=400, detail=f"Paper {arxiv_id} metadata is required from the client to materialize the record")

        normalized_paper = self._normalize_paper_record(source_paper, arxiv_id)
        paper_payload = {
            "arxiv_id": normalized_paper["arxiv_id"],
            "title": normalized_paper["title"],
            "authors": normalized_paper["authors"],
            "abstract": normalized_paper["abstract"],
            "categories": normalized_paper["categories"],
            "published_date": normalized_paper["published_date"],
            "url": normalized_paper["url"],
            "embedding_model": normalized_paper["embedding_model"],
        }

        if existing_paper and existing_embedding and not existing_paper.get("embedding_id"):
            updated = self.db_service.update_paper_embedding(
                arxiv_id=arxiv_id,
                embedding_id=int(existing_embedding.get("embedding_id") or 0),
                embedding_model=str(existing_embedding.get("embedding_model") or normalized_paper["embedding_model"]),
            )
            if updated:
                refreshed = self.db_service.get_paper(arxiv_id)
                if refreshed:
                    return refreshed
            return {**existing_paper, "embedding_id": existing_embedding.get("embedding_id")}

        if existing_paper and existing_paper.get("embedding_id") and not existing_embedding:
            embedding_id = self._insert_paper_embedding(normalized_paper)
            self.db_service.update_paper_embedding(arxiv_id=arxiv_id, embedding_id=embedding_id, embedding_model=normalized_paper["embedding_model"])
            refreshed = self.db_service.get_paper(arxiv_id)
            return refreshed or {**existing_paper, "embedding_id": embedding_id}

        if existing_embedding and not existing_paper:
            stored = dict(paper_payload)
            stored["embedding_id"] = str(existing_embedding.get("embedding_id") or "")
            success = self.db_service.add_paper(stored)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to store paper {arxiv_id}")
            refreshed = self.db_service.get_paper(arxiv_id)
            return refreshed or stored

        embedding_id = self._insert_paper_embedding(normalized_paper)
        stored = dict(paper_payload)
        stored["embedding_id"] = str(embedding_id)
        success = self.db_service.add_paper(stored)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to store paper {arxiv_id}")

        refreshed = self.db_service.get_paper(arxiv_id)
        return refreshed or stored

    def _materialize_paper_from_source(self, source_paper: Dict[str, Any], fallback_arxiv_id: str) -> Dict[str, Any]:
        normalized_paper = self._normalize_paper_record(source_paper, fallback_arxiv_id)
        embedding_id = self._insert_paper_embedding(normalized_paper)
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
        success = self.db_service.add_paper(stored)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to store paper {normalized_paper['arxiv_id']}")
        refreshed = self.db_service.get_paper(normalized_paper["arxiv_id"])
        return refreshed or stored

    def _fetch_paper_from_arxiv_with_rate_limit(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        wait_seconds = 0.0
        with self._arxiv_backfill_lock:
            now = time.monotonic()
            if now < self._arxiv_backfill_next_allowed_time:
                wait_seconds = self._arxiv_backfill_next_allowed_time - now
            self._arxiv_backfill_next_allowed_time = max(self._arxiv_backfill_next_allowed_time, now) + self.ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS

        if wait_seconds > 0:
            logger.info("Waiting %.2f seconds before fetching arXiv paper %s", wait_seconds, arxiv_id)
            time.sleep(wait_seconds)

        arxiv_service = self.arxiv_service_factory()
        search_result = arxiv_service.search_papers(id_list=[arxiv_id], max_results=1, submitted_days_ago=None)
        papers = search_result.get("papers", []) if isinstance(search_result, dict) else []
        return papers[0] if papers else None

    def _normalize_paper_record(self, paper: Dict[str, Any], fallback_arxiv_id: str) -> Dict[str, Any]:
        authors = paper.get("authors", "")
        categories = paper.get("categories", "")
        if isinstance(authors, (list, tuple)):
            authors_value = ", ".join([str(item).strip() for item in authors if str(item).strip()])
        else:
            authors_value = str(authors or "").strip()
        if isinstance(categories, (list, tuple)):
            categories_value = ", ".join([str(item).strip() for item in categories if str(item).strip()])
        else:
            categories_value = str(categories or "").strip()

        title = str(paper.get("title", "") or "").strip()
        abstract = str(paper.get("abstract", "") or paper.get("summary", "") or "").strip()
        published_date = str(paper.get("published_date") or paper.get("published") or paper.get("updated") or paper.get("update_date") or paper.get("publishedAt") or "").strip()
        url = str(paper.get("url") or paper.get("abs_url") or paper.get("absUrl") or paper.get("pdf_url") or paper.get("pdfUrl") or "").strip()
        arxiv_identifier = str(paper.get("arxiv_id") or paper.get("id") or fallback_arxiv_id or "").strip()
        if arxiv_identifier.startswith("http"):
            arxiv_identifier = arxiv_identifier.rsplit("/", 1)[-1]

        embedding_config = self.get_embedding_config()
        return {
            "arxiv_id": arxiv_identifier,
            "title": title,
            "authors": authors_value,
            "abstract": abstract,
            "categories": categories_value,
            "published_date": published_date,
            "url": url,
            "embedding_model": embedding_config.model_name,
        }

    def _insert_paper_embedding(self, normalized_paper: Dict[str, Any]) -> int:
        embedding_id, _ = self._build_and_insert_paper_embedding(normalized_paper)
        return embedding_id

    def _build_and_insert_paper_embedding(self, normalized_paper: Dict[str, Any]) -> tuple[int, List[float]]:
        embedding_config = self.get_embedding_config()
        text_to_embed = self.embedding_service.build_paper_embedding_text(normalized_paper["title"], normalized_paper["abstract"])
        if not text_to_embed:
            raise HTTPException(status_code=400, detail=f"Paper {normalized_paper['arxiv_id']} has no text to embed")

        embedding = self.embedding_service.create_single_embedding(
            text_to_embed,
            provider=embedding_config.provider,
            model=embedding_config.model_name,
            api_key=embedding_config.api_key,
            base_url=embedding_config.base_url,
            dimension=embedding_config.dimension,
        )
        embedding_vector = [float(value) for value in embedding]
        metadata = {
            "content": normalized_paper["abstract"],
            "arxiv_id": normalized_paper["arxiv_id"],
            "title": normalized_paper["title"],
            "authors": normalized_paper["authors"],
            "categories": normalized_paper["categories"],
            "published_date": normalized_paper["published_date"],
            "url": normalized_paper["url"],
            "embedding_model": embedding_config.model_name,
        }
        embedding_id = self.vector_store_service.insert_single_embedding(self.collection_name, embedding_vector, metadata)
        return embedding_id, embedding_vector

    def _get_existing_paper_embedding(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(collection_name=self.collection_name, arxiv_ids=[arxiv_id])
        return embeddings[0] if embeddings else None

    def _embed_papers(self, papers: List[Dict[str, Any]], config: Any) -> List[Dict[str, Any]]:
        embedded: List[Dict[str, Any]] = []
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
            title = str(paper.get("title", "") or "").strip()
            abstract = str(paper.get("abstract", "") or "").strip()
            text_to_embed = self.embedding_service.build_paper_embedding_text(title, abstract)
            if not arxiv_id or not text_to_embed:
                continue
            embedding = self.embedding_service.create_single_embedding(
                text_to_embed,
                provider=config.provider,
                model=config.model_name,
                api_key=config.api_key,
                base_url=config.base_url,
                dimension=config.dimension,
            )
            embedded.append({"arxiv_id": arxiv_id, "vector": [float(value) for value in embedding]})
        return embedded
