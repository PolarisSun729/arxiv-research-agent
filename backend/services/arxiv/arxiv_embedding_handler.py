"""arXiv 论文 Embedding 处理模块。

该模块负责为 arXiv 论文生成向量表示（embedding），并将其写入向量库。
支持批量和单篇两种处理模式，提供成本统计和错误处理。
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from services.storage.paper_embedding_writes import insert_missing_paper_embeddings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Embedding 生成统计信息。"""
    embeddings_attempted: int = 0
    embeddings_written: int = 0
    embeddings_skipped_existing: int = 0
    embeddings_skipped_out_of_window: int = 0
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
        min_paper_date: Optional[str] = None,
    ):
        """初始化 Embedding 处理器。

        参数:
            embedding_service: Embedding 生成服务
            vector_store_service: 向量存储服务
            embedding_collection_name: 向量集合名称
            embedding_batch_size: 批量生成 embedding 的大小
            vector_query_batch_size: 批量查询向量的大小
            token_price_per_1k: 每千 token 的价格（元）
            min_paper_date: 向量保留窗口起点（YYYY-MM-DD），更早的论文不生成向量；None 表示不限
        """
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.embedding_collection_name = embedding_collection_name
        self.embedding_batch_size = embedding_batch_size
        self.vector_query_batch_size = vector_query_batch_size
        self.token_price_per_1k = token_price_per_1k
        self.min_paper_date = min_paper_date
        self.embedding_config = embedding_service.get_default_embedding_config()

    def store_paper_embeddings_batch(
        self,
        papers: Sequence[Dict[str, Any]],
        stats: Optional[EmbeddingStats] = None,
    ) -> EmbeddingStats:
        """批量为已落库论文生成 embedding，并写入向量库。

        已有向量的论文直接跳过，所以重复调用是幂等的；失败的论文只计入 embedding_errors，
        不抛异常，由论文向量维护任务下一轮重试。

        参数:
            papers: 论文列表
            stats: 可选的统计对象，用于累计统计信息

        返回:
            EmbeddingStats: 统计信息
        """
        if stats is None:
            stats = EmbeddingStats()

        eligible_papers = self._eligible_papers(papers)
        in_window_papers = self._papers_in_window(eligible_papers)
        stats.embeddings_skipped_out_of_window += len(eligible_papers) - len(in_window_papers)
        eligible_papers = in_window_papers
        if not eligible_papers:
            return stats

        try:
            missing_papers = self._exclude_embedded_papers(eligible_papers)
        except Exception as exc:  # pragma: no cover - vector store runtime dependent
            # 向量库不可用时整批放弃：查不到已有向量就去生成，写库又必然失败，只会白花 embedding 费用。
            stats.embedding_errors += len(eligible_papers)
            logger.warning("Failed to inspect existing embeddings for %s OAI papers, skipping batch: %s", len(eligible_papers), exc)
            return stats
        stats.embeddings_skipped_existing += len(eligible_papers) - len(missing_papers)
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
        except Exception as exc:  # pragma: no cover - embedding runtime dependent
            logger.warning("Batch embedding for OAI papers failed, falling back to single-item processing: %s", exc)
            for paper, text in zip(missing_papers, texts):
                self._store_single_paper_embedding(paper, text, stats)
            return stats

        items = []
        for paper, embedding in zip(missing_papers, embeddings):
            if embedding:
                items.append({"embedding": embedding, "metadata": self._build_embedding_metadata(paper)})
            else:
                stats.embedding_errors += 1
                logger.warning("Embedding service returned an empty vector for OAI paper %s", paper.get("arxiv_id"))
        if not items:
            return stats

        try:
            # 生成 embedding 期间推荐物化可能已经写了同一篇，写入前在锁内再查一次，避免重复向量。
            inserted_count = insert_missing_paper_embeddings(self.vector_store_service, self.embedding_collection_name, items)
        except Exception as exc:  # pragma: no cover - vector store runtime dependent
            # 向量已经算好，写库失败时不再逐篇重算，免得同一批付两次钱；漏掉的论文由维护任务下一轮补上。
            stats.embedding_errors += len(items)
            logger.warning("Failed to insert %s OAI paper embeddings into vector store: %s", len(items), exc)
            return stats
        stats.embeddings_written += inserted_count
        stats.embeddings_skipped_existing += len(items) - inserted_count
        for item in items:
            logger.debug("Embedded OAI paper into vector store: %s", item["metadata"]["arxiv_id"])
        return stats

    def find_papers_missing_embeddings(
        self,
        papers: Sequence[Dict[str, Any]],
        embedded_arxiv_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """返回在保留窗口内、有标题和摘要、但向量库里还没有向量的论文；向量库查询失败时直接抛出。

        传入 embedded_arxiv_ids（list_embedded_arxiv_ids 的结果）时在内存里比对，不再查向量库。
        """
        candidates = self._papers_in_window(self._eligible_papers(papers))
        if embedded_arxiv_ids is None:
            return self._exclude_embedded_papers(candidates)
        return [paper for paper in candidates if str(paper.get("arxiv_id", "") or "").strip() not in embedded_arxiv_ids]

    def list_embedded_arxiv_ids(self) -> Set[str]:
        """返回向量库里所有已有论文向量的 arxiv_id，供一次扫描大量论文时在内存里判断谁缺向量。"""
        return self.vector_store_service.list_paper_arxiv_ids(self.embedding_collection_name)

    def prune_out_of_window(self, keep_arxiv_ids: Set[str], dry_run: bool = False) -> int:
        """删除首次提交早于保留窗口的论文向量，keep_arxiv_ids（用户点赞/点踩过的论文）除外；返回删除条数。"""
        if not self.min_paper_date:
            return 0
        return self.vector_store_service.delete_paper_embeddings_published_before(
            collection_name=self.embedding_collection_name,
            cutoff_date=self.min_paper_date,
            keep_arxiv_ids=keep_arxiv_ids,
            dry_run=dry_run,
        )

    def _papers_in_window(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """只保留首次提交不早于窗口起点的论文；日期缺失的无法判断新旧，按窗口内处理。"""
        if not self.min_paper_date:
            return papers
        return [paper for paper in papers if not self._paper_date(paper) or self._paper_date(paper) >= self.min_paper_date]

    @staticmethod
    def _paper_date(paper: Dict[str, Any]) -> str:
        """论文日期：首次提交日期优先，和写入向量库的 published_date、推荐近期论文池的日期口径一致。"""
        return str(paper.get("created") or paper.get("updated") or paper.get("oai_datestamp") or "").strip()

    @staticmethod
    def _eligible_papers(papers: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤掉缺 arxiv_id、标题或摘要的论文。"""
        eligible_papers = []
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
        return eligible_papers

    def _exclude_embedded_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分批探测向量库，去掉已有向量的论文，避免重复生成 embedding 造成额外成本。"""
        ids = [str(paper.get("arxiv_id", "") or "").strip() for paper in papers]
        existing_ids = set()
        for start in range(0, len(ids), self.vector_query_batch_size):
            # 只取 arxiv_id 不读向量：维护任务每轮要探测整个窗口（数万篇），带上向量会白白搬运几百 MB。
            existing_ids |= self.vector_store_service.get_existing_paper_arxiv_ids(
                self.embedding_collection_name,
                ids[start : start + self.vector_query_batch_size],
            )
        return [paper for paper, arxiv_id in zip(papers, ids) if arxiv_id not in existing_ids]

    def _store_single_paper_embedding(self, paper: Dict[str, Any], text_to_embed: str, stats: EmbeddingStats) -> None:
        """为单篇论文生成并写入 embedding，作为批量 embedding 失败时的兜底路径。"""
        arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
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
            inserted_count = insert_missing_paper_embeddings(
                self.vector_store_service,
                self.embedding_collection_name,
                [{"embedding": embedding, "metadata": metadata}],
            )
            stats.embeddings_written += inserted_count
            stats.embeddings_skipped_existing += 1 - inserted_count
            logger.debug("Embedded OAI paper into vector store: %s", arxiv_id)
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
            "authors": self._join_list_field(paper.get("authors", "")),
            "categories": self._join_list_field(paper.get("categories", "")),
            "published_date": self._paper_date(paper),
            "url": str(paper.get("abs_url") or paper.get("pdf_url") or "").strip(),
            "embedding_model": self.embedding_config.model_name,
        }

    @staticmethod
    def _join_list_field(value: Any) -> str:
        """把作者、分类统一成逗号分隔字符串，和推荐侧物化论文时写入向量库的格式一致。

        OAI 库里这两个字段存的是 JSON 数组字符串，直接 str() 会把方括号和引号写进向量库。
        """
        if isinstance(value, str):
            text = value.strip()
            if not text.startswith("["):
                return text
            try:
                value = json.loads(text)
            except ValueError:
                return text
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value or "").strip()

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
