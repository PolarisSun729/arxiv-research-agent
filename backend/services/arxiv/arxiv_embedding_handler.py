"""arXiv 论文 Embedding 处理模块。

该模块负责为 arXiv 论文生成向量表示（embedding），并将其写入向量库。
支持批量和单篇两种处理模式，提供成本统计和错误处理。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Embedding 生成统计信息。"""
    embeddings_attempted: int = 0
    embeddings_written: int = 0
    embeddings_skipped_existing: int = 0
    embedding_errors: int = 0
    embedding_input_tokens: int = 0
    embedding_output_tokens: int = 0
    embedding_total_tokens: int = 0
    embedding_cost_yuan: float = 0.0


class ArxivEmbeddingHandler:
    """arXiv 论文 Embedding 处理器。"""

    def __init__(
        self,
        embedding_service,
        vector_store_service,
        embedding_collection_name: str,
        embedding_batch_size: int = 25,
        vector_query_batch_size: int = 100,
        token_price_per_1k: float = 0.0007,
    ):
        """初始化 Embedding 处理器。

        参数:
            embedding_service: Embedding 生成服务
            vector_store_service: 向量存储服务
            embedding_collection_name: 向量集合名称
            embedding_batch_size: 批量生成 embedding 的大小
            vector_query_batch_size: 批量查询向量的大小
            token_price_per_1k: 每千 token 的价格（元）
        """
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.embedding_collection_name = embedding_collection_name
        self.embedding_batch_size = embedding_batch_size
        self.vector_query_batch_size = vector_query_batch_size
        self.token_price_per_1k = token_price_per_1k
        self.embedding_config = embedding_service.get_default_embedding_config()

    def store_paper_embeddings_batch(
        self,
        papers: Sequence[Dict[str, Any]],
        stats: Optional[EmbeddingStats] = None,
    ) -> EmbeddingStats:
        """批量为已落库论文生成 embedding，并写入向量库。

        参数:
            papers: 论文列表
            stats: 可选的统计对象，用于累计统计信息

        返回:
            EmbeddingStats: 统计信息
        """
        if stats is None:
            stats = EmbeddingStats()

        eligible_papers = []
        eligible_ids = []
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
            title = str(paper.get("title", "") or "").strip()
            abstract = str(paper.get("abstract", "") or "").strip()
            if not arxiv_id or not title or not abstract:
                # 没有标题或摘要时 embedding 质量不可控，直接跳过比写入噪声向量更安全。
                logger.debug(
                    "Skipping embedding for OAI paper %s because title or abstract is missing",
                    arxiv_id or "<unknown>",
                )
                continue
            eligible_papers.append(paper)
            eligible_ids.append(arxiv_id)

        if not eligible_papers:
            return stats

        existing_ids = set()
        try:
            for start in range(0, len(eligible_ids), self.vector_query_batch_size):
                batch_ids = eligible_ids[start : start + self.vector_query_batch_size]
                # 先批量探测已存在向量，避免重复生成 embedding 造成额外成本。
                existing_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
                    collection_name=self.embedding_collection_name,
                    arxiv_ids=batch_ids,
                )
                for item in existing_embeddings:
                    arxiv_id = str(item.get("arxiv_id", "") or "").strip()
                    if arxiv_id:
                        existing_ids.add(arxiv_id)
        except Exception as exc:  # pragma: no cover - vector store runtime dependent
            logger.warning("Failed to inspect existing embeddings for OAI papers: %s", exc)
        if existing_ids:
            stats.embeddings_skipped_existing += len(existing_ids)

        missing_papers = [
            paper for paper in eligible_papers if str(paper.get("arxiv_id", "") or "").strip() not in existing_ids
        ]
        if not missing_papers:
            return stats

        stats.embeddings_attempted += len(missing_papers)
        texts = [
            self.embedding_service.build_paper_embedding_text(
                str(paper.get("title", "") or "").strip(),
                str(paper.get("abstract", "") or "").strip(),
            )
            for paper in missing_papers
        ]

        try:
            # 批量 embedding 优先追求成本和吞吐；失败时再回退到单篇粒度定位问题。
            embeddings, usage = self.embedding_service.create_text_embeddings_with_usage(
                texts,
                provider=self.embedding_config.provider,
                model=self.embedding_config.model_name,
                api_key=self.embedding_config.api_key,
                base_url=self.embedding_config.base_url,
                dimension=self.embedding_config.dimension,
                batch_size=self.embedding_batch_size,
            )
            self._accumulate_embedding_usage(stats, usage, len(missing_papers))
            if len(embeddings) != len(missing_papers):
                raise ValueError(
                    f"Embedding batch returned {len(embeddings)} vectors for {len(missing_papers)} papers"
                )

            items = [
                {
                    "embedding": embedding,
                    "metadata": self._build_embedding_metadata(paper),
                }
                for paper, embedding in zip(missing_papers, embeddings)
                if embedding
            ]
            if not items:
                return stats

            inserted_count = self.vector_store_service.insert_embeddings(self.embedding_collection_name, items)
            stats.embeddings_written += int(inserted_count)
            # 批量写入后逐条记日志，便于后续按 arxiv_id 追踪 embedding 生命周期。
            for paper in missing_papers:
                logger.info("Embedded OAI paper into vector store: %s", paper.get("arxiv_id"))
        except Exception as exc:  # pragma: no cover - vector store/runtime dependent
            logger.warning("Batch embedding for OAI papers failed, falling back to single-item processing: %s", exc)
            for paper in missing_papers:
                self._store_single_paper_embedding(paper, stats)

        return stats

    def _store_single_paper_embedding(self, paper: Dict[str, Any], stats: EmbeddingStats) -> None:
        """为单篇论文生成并写入 embedding，作为批量失败时的兜底路径。"""
        arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
        title = str(paper.get("title", "") or "").strip()
        abstract = str(paper.get("abstract", "") or "").strip()
        if not arxiv_id or not title or not abstract:
            logger.debug(
                "Skipping embedding for OAI paper %s because title or abstract is missing", arxiv_id or "<unknown>"
            )
            return

        try:
            # 单篇兜底前先检查是否已有向量，避免批量失败后重复写入。
            existing_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
                collection_name=self.embedding_collection_name,
                arxiv_ids=[arxiv_id],
            )
        except Exception as exc:  # pragma: no cover - vector store runtime dependent
            logger.warning("Failed to inspect existing embedding for OAI paper %s: %s", arxiv_id, exc)
            existing_embeddings = []

        if existing_embeddings:
            stats.embeddings_skipped_existing += 1
            return

        stats.embeddings_attempted += 1
        text_to_embed = self.embedding_service.build_paper_embedding_text(title, abstract)
        if not text_to_embed:
            stats.embedding_errors += 1
            logger.warning("Skipping OAI embedding for %s because embedding text is empty", arxiv_id)
            return

        try:
            embedding, usage = self.embedding_service.create_single_embedding_with_usage(
                text_to_embed,
                provider=self.embedding_config.provider,
                model=self.embedding_config.model_name,
                api_key=self.embedding_config.api_key,
                base_url=self.embedding_config.base_url,
                dimension=self.embedding_config.dimension,
            )
            self._accumulate_embedding_usage(stats, usage, 1)
            # 单篇兜底沿用和批量路径一致的 metadata 结构，方便后续统一检索与排查。
            metadata = self._build_embedding_metadata(paper)
            self.vector_store_service.insert_single_embedding(
                self.embedding_collection_name,
                embedding,
                metadata,
            )
            stats.embeddings_written += 1
            logger.info("Embedded OAI paper into vector store: %s", arxiv_id)
        except Exception as exc:  # pragma: no cover - embedding/vector store runtime dependent
            stats.embedding_errors += 1
            logger.warning("Failed to embed OAI paper %s into vector store: %s", arxiv_id, exc)

    def _build_embedding_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """构造写入向量库的 OAI embedding 元数据。"""
        # metadata 保持与检索侧常用字段一致，方便后续按 arxiv_id / title / date 回查。
        return {
            "content": str(paper.get("abstract", "") or "").strip(),
            "arxiv_id": str(paper.get("arxiv_id", "") or "").strip(),
            "title": str(paper.get("title", "") or "").strip(),
            "authors": paper.get("authors", ""),
            "categories": paper.get("categories", ""),
            "published_date": str(
                paper.get("created") or paper.get("updated") or paper.get("oai_datestamp") or ""
            ).strip(),
            "url": str(paper.get("abs_url") or paper.get("pdf_url") or "").strip(),
            "embedding_model": self.embedding_config.model_name,
        }

    def _accumulate_embedding_usage(self, stats: EmbeddingStats, usage: Dict[str, Any], item_count: int) -> None:
        """把 embedding 调用的 token 用量和估算成本累计到统计中。"""
        if not usage:
            return

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(
            usage.get("total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens)
        )

        # 统计统一在这里收口，避免批量和单篇写入两条路径各自维护计费逻辑。
        stats.embedding_input_tokens += input_tokens
        stats.embedding_output_tokens += output_tokens
        stats.embedding_total_tokens += total_tokens

        if input_tokens > 0:
            # 当前成本估算只按输入 token 计价，和现有 DashScope 文本 embedding 计费规则保持一致。
            stats.embedding_cost_yuan += (input_tokens / 1000.0) * self.token_price_per_1k

        logger.info(
            "OAI embedding batch usage: items=%s input_tokens=%s output_tokens=%s total_tokens=%s estimated_cost=%.6f yuan",
            item_count,
            input_tokens,
            output_tokens,
            total_tokens,
            (input_tokens / 1000.0) * self.token_price_per_1k if input_tokens > 0 else 0.0,
        )
