from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from services.storage.paper_embedding_writes import insert_missing_paper_embeddings
from services.user_behavior_policy import validate_weak_paper_action_type

logger = logging.getLogger(__name__)


class CandidateMaterializer:
    """论文“物化”能力（mixin）：保证论文同时存在于论文表（SQLite）与向量库（Milvus），且两者互相关联。

    “物化”后的论文满足：论文表中有完整展示元数据；向量库中有该论文的 embedding；
    论文表的 embedding_id 指向该向量。后续画像、召回、排序都以 arXiv ID 为主键依赖这一状态。

    本类依赖宿主 RecommendationService 注入的属性：
    ``paper_catalog_store``、``user_preference_store``、``embedding_service``、
    ``vector_store_service``、``collection_name``、``get_embedding_config``、
    ``arxiv_service_factory``，以及限流相关的 ``_arxiv_backfill_lock`` /
    ``_arxiv_backfill_next_allowed_time`` / ``ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS``。
    """

    # ------------------------------------------------------------------
    # 对外入口：记录用户行为
    # ------------------------------------------------------------------

    def record_user_paper_action(
        self,
        user_id: str,
        arxiv_id: str,
        action_type: str,
        paper_payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """记录用户对论文的弱行为事件，并确保论文已被物化到本地存储链路中。

        这里的“物化”包含论文表记录与向量信息准备，目的是让后续画像更新、
        推荐召回与排序都能基于一致的数据主键工作。

        Raises:
            HTTPException(400): arxiv_id 为空、行为类型不支持，或传入了 like/dislike（应走偏好接口）。
            HTTPException(404/400/500): 论文无法物化或行为写入失败。
        """
        normalized_arxiv_id = str(arxiv_id or "").strip()
        normalized_action = str(action_type or "").strip()
        if not normalized_arxiv_id:
            raise HTTPException(status_code=400, detail="arxiv_id is required")
        try:
            normalized_action = validate_weak_paper_action_type(normalized_action)
        except ValueError as exc:
            if str(exc) == "explicit_preference_action_not_allowed":
                # 服务层也要挡住 like/dislike，避免 Agent 或测试绕过 router 后重新写出双套偏好状态。
                raise HTTPException(status_code=400, detail="paper-action does not accept like/dislike; use like-paper or dislike-paper") from exc
            raise HTTPException(status_code=400, detail="unsupported paper action type") from exc

        # 先确保论文已被标准化入库，避免只记录了行为却缺少可追踪的论文实体。
        paper = self._ensure_paper_materialized(normalized_arxiv_id, paper_payload=paper_payload)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {normalized_arxiv_id} could not be materialized")

        success = self.user_preference_store.record_user_paper_action(
            user_id=user_id,
            arxiv_id=normalized_arxiv_id,
            action_type=normalized_action,
            metadata=metadata,
        )
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to record paper action {normalized_action}")

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
        """记录用户喜欢或不喜欢的强偏好。

        画像重建由事件驱动异步完成，这里只保证论文物化与偏好状态落库，不同步重建兴趣向量。
        """
        normalized_arxiv_id = str(arxiv_id or "").strip()
        if not normalized_arxiv_id:
            raise HTTPException(status_code=400, detail="arxiv_id is required")

        paper = self._ensure_paper_materialized(normalized_arxiv_id, paper_payload=paper_payload)
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {normalized_arxiv_id} could not be materialized")

        if liked:
            success = self.user_preference_store.add_liked_paper(user_id=user_id, arxiv_id=normalized_arxiv_id)
            action = "liked"
        else:
            success = self.user_preference_store.add_disliked_paper(user_id=user_id, arxiv_id=normalized_arxiv_id)
            action = "disliked"

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to add paper to {action} list")

        return {
            "status": "success",
            # 画像重建改为事件驱动，点击链路只保证业务状态和事件流落库，避免慢速画像生成阻塞主流程。
            "message": f"Paper added to {action} list and materialized into paper store",
            "arxiv_id": normalized_arxiv_id,
            "paper": paper,
        }

    # ------------------------------------------------------------------
    # 批量物化：推荐候选
    # ------------------------------------------------------------------

    def _materialize_candidate_papers_for_recommendation(
        self,
        candidates: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """批量物化推荐候选论文，并尽可能复用已有论文记录与向量数据。

        对每个候选按以下情况处理（stats 中对应计数）：
        - reused_existing：论文表与向量库都有且已关联，直接复用；
        - db_only：两边只差关联，或只有向量缺论文记录，只写论文表；
        - batch_embedded / batch_inserted：缺向量，批量生成 embedding 后批量写入向量库；
        - unresolved：没有可嵌入文本、embedding 失败或写库失败，该候选会被丢弃。

        Returns:
            (按输入顺序排列的已物化候选, 统计信息)。
            已物化候选以论文表记录为准覆盖元数据，但保留召回阶段的 recall_* / score 等排序字段。
        """
        stats = {"total": len(candidates), "reused_existing": 0, "db_only": 0, "batch_embedded": 0, "batch_inserted": 0, "unresolved": 0}
        materialized_by_id: Dict[str, Dict[str, Any]] = {}
        batch_jobs: List[Dict[str, Any]] = []
        embedding_config = self.get_embedding_config()
        candidate_ids = [str(candidate.get("arxiv_id", "") or "").strip() for candidate in candidates if str(candidate.get("arxiv_id", "") or "").strip()]
        # 一次性批量查询向量库，避免每个候选单独请求 Milvus。
        candidate_embedding_rows = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(collection_name=self.collection_name, arxiv_ids=candidate_ids)
        candidate_embedding_map = {str(item.get("arxiv_id", "") or "").strip(): item for item in candidate_embedding_rows if item.get("arxiv_id")}

        for candidate in candidates:
            arxiv_id = str(candidate.get("arxiv_id", "") or "").strip()
            if not arxiv_id:
                stats["unresolved"] += 1
                continue
            try:
                existing_paper = self.paper_catalog_store.get_paper(arxiv_id)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to inspect candidate paper %s before materialization: %s", arxiv_id, exc)
                existing_paper = None
            existing_embedding = candidate_embedding_map.get(arxiv_id)

            # 数据库和向量库都已有时，直接复用，避免重复写入和重复 embedding。
            if existing_paper and existing_paper.get("embedding_id") and existing_embedding:
                materialized_by_id[arxiv_id] = existing_paper
                stats["reused_existing"] += 1
                continue

            # 数据库有论文、向量库也有 embedding，但二者关联缺失时，只修复映射关系。
            if existing_paper and existing_embedding:
                updated = self.paper_catalog_store.update_paper_embedding(
                    arxiv_id=arxiv_id,
                    embedding_id=int(existing_embedding.get("embedding_id") or 0),
                    embedding_model=str(existing_embedding.get("embedding_model") or existing_paper.get("embedding_model") or embedding_config.model_name),
                )
                refreshed = self.paper_catalog_store.get_paper(arxiv_id) if updated else None
                materialized_by_id[arxiv_id] = refreshed or {**existing_paper, "embedding_id": existing_embedding.get("embedding_id")}
                stats["db_only"] += 1
                continue

            # 只有向量没有论文时，补一份论文表记录，让向量结果能被业务层稳定引用。
            if existing_embedding:
                stored = self._build_paper_store_payload(
                    self._normalize_paper_record(candidate, arxiv_id, embedding_model=embedding_config.model_name),
                    embedding_id=str(existing_embedding.get("embedding_id") or ""),
                    embedding_model=str(existing_embedding.get("embedding_model") or embedding_config.model_name),
                )
                if self.paper_catalog_store.add_paper(stored):
                    materialized_by_id[arxiv_id] = self.paper_catalog_store.get_paper(arxiv_id) or stored
                    stats["db_only"] += 1
                else:
                    stats["unresolved"] += 1
                continue

            # 剩下的都是“缺向量”的情况（可能连论文记录也没有）。
            # 已有论文记录时与候选字段合并，避免候选里缺失的字段把论文表中已有的值覆盖为空。
            normalized_paper = self._normalize_paper_record(
                self._merge_non_empty_paper_fields(candidate, existing_paper),
                arxiv_id,
                embedding_model=embedding_config.model_name,
            )
            text_to_embed = self.embedding_service.build_paper_embedding_text(normalized_paper["title"], normalized_paper["abstract"])
            if not text_to_embed:
                stats["unresolved"] += 1
                continue
            batch_jobs.append({"arxiv_id": arxiv_id, "normalized_paper": normalized_paper, "text": text_to_embed})

        if batch_jobs:
            # 只有在完全没有现成向量可复用时，才调用 embedding 服务；所有缺向量的候选合并成一次批量请求。
            embeddings = self._create_embeddings_batch([job["text"] for job in batch_jobs], embedding_config)
            embedded_jobs: List[Dict[str, Any]] = []
            for job, embedding in zip(batch_jobs, embeddings):
                if embedding:
                    embedded_jobs.append({**job, "embedding": embedding})
                else:
                    stats["unresolved"] += 1
            batch_jobs = embedded_jobs

        if batch_jobs:
            try:
                # 先批量写向量库，再逐条补论文表，兼顾吞吐量和业务侧主键可追踪性。
                batch_insert_payload = [
                    {"embedding": job["embedding"], "metadata": self._build_vector_metadata(job["normalized_paper"])}
                    for job in batch_jobs
                ]
                # 生成 embedding 期间维护任务可能已经写了同一篇，写入前在锁内再查一次，避免重复向量。
                embedding_count = insert_missing_paper_embeddings(self.vector_store_service, self.collection_name, batch_insert_payload)
                stats["batch_embedded"] = len(batch_jobs)
                stats["batch_inserted"] = int(embedding_count)
                for job in batch_jobs:
                    # insert_embeddings 只返回写入条数、拿不到每条向量的主键，这里只能先写空 embedding_id。
                    # add_paper 不会用空值覆盖已有关联；下次该论文进入候选时会命中上面的“只修复映射”分支补齐关联。
                    stored = self._build_paper_store_payload(job["normalized_paper"], embedding_id="")
                    if self.paper_catalog_store.add_paper(stored):
                        materialized_by_id[job["arxiv_id"]] = self.paper_catalog_store.get_paper(job["arxiv_id"]) or stored
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
            # 论文表记录会覆盖同名字段，这里把召回阶段产生的排序相关字段还原回来。
            for key, value in candidate.items():
                if key.startswith("recall_") or key in {"similarity_score", "score", "distance", "metadata"}:
                    merged_candidate[key] = value
            ordered_candidates.append(merged_candidate)
        return ordered_candidates, stats

    # ------------------------------------------------------------------
    # 单篇物化
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_non_empty_paper_fields(
        primary: Optional[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """合并论文元数据，避免一次不完整请求把已有的可展示字段覆盖为空。

        以 fallback 为底，primary 中的非空字段（None、空白字符串、全空列表除外）覆盖上去。
        """
        merged = dict(fallback or {})
        for key, value in (primary or {}).items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, set)) and not any(str(item).strip() for item in value):
                continue
            merged[key] = value
        return merged

    def _ensure_paper_materialized(self, arxiv_id: str, paper_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """确保单篇论文已经具备论文表记录和可用 embedding 的最小落库状态。

        元数据来源：请求携带的 paper_payload 优先，缺失字段用已有论文记录（或向量库 metadata）补齐。
        向量来源：向量库已有则直接复用并修复关联，没有才调用 embedding 服务新建。

        Raises:
            HTTPException(400): 本地没有任何该论文的信息，且请求也没有携带元数据。
            HTTPException(500): 写论文表失败。
        """
        existing_paper = self.paper_catalog_store.get_paper(arxiv_id)
        existing_embedding = self._get_existing_paper_embedding(arxiv_id)
        paper_linked = bool(existing_paper and existing_paper.get("embedding_id"))

        # 已完整物化且没有新元数据需要回填时，直接返回，不产生任何写入。
        if paper_linked and existing_embedding and not paper_payload:
            return existing_paper

        source_paper = self._merge_non_empty_paper_fields(paper_payload, existing_paper or existing_embedding)
        if not source_paper:
            raise HTTPException(status_code=400, detail=f"Paper {arxiv_id} metadata is required from the client to materialize the record")
        normalized_paper = self._normalize_paper_record(source_paper, arxiv_id)

        if existing_embedding:
            # 已有向量只代表检索资产存在，不代表论文展示元数据完整；复用向量，只刷新元数据并修复关联。
            # 论文表已有关联时以论文表为准，否则采用向量库中的主键与模型名。
            linked_paper = existing_paper if paper_linked else {}
            embedding_id = str(linked_paper.get("embedding_id") or existing_embedding.get("embedding_id") or "")
            embedding_model = str(
                linked_paper.get("embedding_model")
                or existing_embedding.get("embedding_model")
                or normalized_paper["embedding_model"]
            )
        else:
            # 向量库中没有（包括论文表记着 embedding_id 但向量已丢失的情况），重新生成并写入。
            embedding_id = str(self._insert_paper_embedding(normalized_paper))
            embedding_model = normalized_paper["embedding_model"]

        stored = self._build_paper_store_payload(normalized_paper, embedding_id=embedding_id, embedding_model=embedding_model)
        if not self.paper_catalog_store.add_paper(stored):
            raise HTTPException(status_code=500, detail=f"Failed to store paper {arxiv_id}")
        return self.paper_catalog_store.get_paper(arxiv_id) or {**(existing_paper or {}), **stored}

    def _materialize_paper_from_source(self, source_paper: Dict[str, Any], fallback_arxiv_id: str) -> Dict[str, Any]:
        """把外部来源（如 arXiv API）拿到的论文物化到本地。

        直接复用 _ensure_paper_materialized：向量库里已有该论文时会复用向量，
        避免重复插入导致同一论文在 Milvus 中出现多条向量。
        """
        return self._ensure_paper_materialized(fallback_arxiv_id, paper_payload=source_paper)

    def _fetch_paper_from_arxiv_with_rate_limit(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """按限流策略从 arXiv 拉取单篇论文，避免外部接口在短时间内被频繁调用。

        限流方式：在锁内为本次请求“预约”一个时间槽（下一个可用时间 += 间隔），
        锁外再 sleep 到预约时间。这样多线程并发调用时请求会被均匀错开，且不会持锁 sleep。
        """
        wait_seconds = 0.0
        with self._arxiv_backfill_lock:
            now = time.monotonic()
            if now < self._arxiv_backfill_next_allowed_time:
                wait_seconds = self._arxiv_backfill_next_allowed_time - now
            self._arxiv_backfill_next_allowed_time = max(self._arxiv_backfill_next_allowed_time, now) + self.ARXIV_BACKFILL_REQUEST_INTERVAL_SECONDS

        if wait_seconds > 0:
            logger.debug("Waiting %.2f seconds before fetching arXiv paper %s", wait_seconds, arxiv_id)
            time.sleep(wait_seconds)

        arxiv_service = self.arxiv_service_factory()
        # 补拉单篇论文只需要后端协议的精确 ID 查询；日期过滤属于输入层，不再进入搜索后端。
        search_result = arxiv_service.search(id_list=[arxiv_id], max_results=1)
        papers = search_result.get("papers", []) if isinstance(search_result, dict) else []
        return papers[0] if papers else None

    # ------------------------------------------------------------------
    # 字段标准化与 payload 构造
    # ------------------------------------------------------------------

    def _normalize_paper_record(
        self,
        paper: Dict[str, Any],
        fallback_arxiv_id: str,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """把不同来源的论文字段统一整理为系统内部使用的标准结构。

        - authors / categories 统一转为逗号分隔字符串；
        - 兼容 arXiv API（summary / published / id 为 URL）、前端（publishedAt / absUrl）等多种字段名；
        - arxiv_id 优先取论文自身的 ID，URL 形式会截取最后一段；
        - embedding_model 取调用方传入的模型名，未传入时读取当前 embedding 配置。
          批量处理时应由调用方读取一次配置后传入，避免每篇论文都重新构造一次配置。
        """
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

        if embedding_model is None:
            embedding_model = self.get_embedding_config().model_name
        return {
            "arxiv_id": arxiv_identifier,
            "title": title,
            "authors": authors_value,
            "abstract": abstract,
            "categories": categories_value,
            "published_date": published_date,
            "url": url,
            "embedding_model": embedding_model,
        }

    @staticmethod
    def _build_paper_store_payload(
        normalized_paper: Dict[str, Any],
        *,
        embedding_id: str,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """由标准化论文构造写入论文表（paper_catalog_store.add_paper）的 payload。

        embedding_model 缺省时使用标准化记录里的当前配置模型名。
        """
        return {
            "arxiv_id": normalized_paper["arxiv_id"],
            "title": normalized_paper["title"],
            "authors": normalized_paper["authors"],
            "abstract": normalized_paper["abstract"],
            "categories": normalized_paper["categories"],
            "published_date": normalized_paper["published_date"],
            "url": normalized_paper["url"],
            "embedding_id": embedding_id,
            "embedding_model": embedding_model or normalized_paper["embedding_model"],
        }

    @staticmethod
    def _build_vector_metadata(normalized_paper: Dict[str, Any]) -> Dict[str, Any]:
        """由标准化论文构造写入向量库的 metadata；content 字段存摘要。"""
        return {
            "content": normalized_paper["abstract"],
            "arxiv_id": normalized_paper["arxiv_id"],
            "title": normalized_paper["title"],
            "authors": normalized_paper["authors"],
            "categories": normalized_paper["categories"],
            "published_date": normalized_paper["published_date"],
            "url": normalized_paper["url"],
            "embedding_model": normalized_paper["embedding_model"],
        }

    # ------------------------------------------------------------------
    # embedding 生成与写入
    # ------------------------------------------------------------------

    def _create_embedding(self, text: str, config: Any) -> List[float]:
        """按给定 embedding 配置为文本生成向量，统一转为 float 列表。"""
        embedding = self.embedding_service.create_single_embedding(
            text,
            provider=config.provider,
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            dimension=config.dimension,
        )
        return [float(value) for value in embedding]

    def _create_embeddings_batch(self, texts: List[str], config: Any) -> List[Optional[List[float]]]:
        """批量生成 embedding，返回与 texts 一一对应的向量列表，失败的位置为 None。

        优先走 embedding 服务的批量接口（一次请求处理多条文本，并复用缓存）；
        批量接口不可用或整体失败时，退回逐条生成，单条失败只影响该条。
        """
        if not texts:
            return []
        batch_create = getattr(self.embedding_service, "create_text_embeddings_with_usage", None)
        if callable(batch_create):
            try:
                embeddings, _usage = batch_create(
                    list(texts),
                    provider=config.provider,
                    model=config.model_name,
                    api_key=config.api_key,
                    base_url=config.base_url,
                    dimension=config.dimension,
                    batch_size=getattr(config, "batch_size", None),
                )
                if len(embeddings) == len(texts):
                    return [[float(value) for value in embedding] if embedding else None for embedding in embeddings]
                logger.warning("Batch embedding returned %s vectors for %s texts; falling back to per-text embedding", len(embeddings), len(texts))
            except Exception as exc:
                logger.warning("Batch embedding failed for %s texts, falling back to per-text embedding: %s", len(texts), exc)

        results: List[Optional[List[float]]] = []
        for text in texts:
            try:
                results.append(self._create_embedding(text, config))
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to embed text for recommendation: %s", exc)
                results.append(None)
        return results

    def _insert_paper_embedding(self, normalized_paper: Dict[str, Any]) -> int:
        """为标准化论文生成并插入 embedding，并仅返回向量库主键。"""
        embedding_id, _ = self._build_and_insert_paper_embedding(normalized_paper)
        return embedding_id

    def _build_and_insert_paper_embedding(self, normalized_paper: Dict[str, Any]) -> tuple[int, List[float]]:
        """构造论文 embedding 并写入向量库，同时返回向量内容与其主键。

        Raises:
            HTTPException(400): 标题和摘要都为空，无法生成 embedding。
        """
        embedding_config = self.get_embedding_config()
        text_to_embed = self.embedding_service.build_paper_embedding_text(normalized_paper["title"], normalized_paper["abstract"])
        if not text_to_embed:
            raise HTTPException(status_code=400, detail=f"Paper {normalized_paper['arxiv_id']} has no text to embed")

        embedding_vector = self._create_embedding(text_to_embed, embedding_config)
        embedding_id = self.vector_store_service.insert_single_embedding(
            self.collection_name,
            embedding_vector,
            self._build_vector_metadata(normalized_paper),
        )
        return embedding_id, embedding_vector

    def _get_existing_paper_embedding(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """按 arXiv ID 查询现有 embedding，供物化流程判断是否可以复用。"""
        embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(collection_name=self.collection_name, arxiv_ids=[arxiv_id])
        return embeddings[0] if embeddings else None

    def _embed_papers(self, papers: List[Dict[str, Any]], config: Any) -> List[Dict[str, Any]]:
        """仅基于论文文本临时生成向量（不写入向量库），用于回退场景下的画像计算。

        所有论文合并为一次批量 embedding 请求；单篇失败只跳过该篇：这是兜底中的兜底，
        不能因为某一篇 embedding 失败让整个画像重建中断。
        """
        jobs: List[tuple[str, str]] = []
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
            title = str(paper.get("title", "") or "").strip()
            abstract = str(paper.get("abstract", "") or "").strip()
            text_to_embed = self.embedding_service.build_paper_embedding_text(title, abstract)
            if arxiv_id and text_to_embed:
                jobs.append((arxiv_id, text_to_embed))
        vectors = self._create_embeddings_batch([text for _, text in jobs], config)
        embedded: List[Dict[str, Any]] = []
        for (arxiv_id, _), vector in zip(jobs, vectors):
            if vector:
                embedded.append({"arxiv_id": arxiv_id, "vector": vector})
            else:
                logger.warning("Failed to embed fallback paper %s", arxiv_id)
        return embedded
