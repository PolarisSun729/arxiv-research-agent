"""基于 bm25s 库的关键词召回后端。

对整个 collection 的 retrieval index 一次性建立 bm25s 索引（缓存在 retrieval_index 对象上），
每条 query view 用 bm25s 批量检索候选；之后的 RRF 融合、chunk 聚合、结果回填与 debug
全部复用 ``KeywordBackend`` 基类，保证与内置 BM25 的输出契约一致。
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from services.retrieval.keyword_backend import KeywordBackend, KeywordBackendResult

logger = logging.getLogger(__name__)


class BM25sBackend(KeywordBackend):
    """使用 bm25s 库的 BM25 后端；bm25s 未安装或执行失败时通过 fallback_reason 通知调用方降级。"""

    def __init__(
        self,
        *,
        query_tools: Any,
        route_confidence_builder: Any,
        structural_bonus_builder: Any,
        fusion_service: Any,
    ) -> None:
        super().__init__(
            query_tools=query_tools,
            route_confidence_builder=route_confidence_builder,
            structural_bonus_builder=structural_bonus_builder,
            fusion_service=fusion_service,
        )
        # route_retriever 初始化时读取这两个属性，决定是否直接改用内置 BM25。
        self._bm25s_available = False
        self._import_error: Optional[str] = None
        self._try_import_bm25s()

    def _try_import_bm25s(self) -> None:
        """尝试导入 bm25s（可选依赖），记录是否可用及失败原因。"""
        try:
            import bm25s
            self._bm25s_available = True
            self._bm25s = bm25s
        except ImportError as exc:
            self._bm25s_available = False
            self._import_error = str(exc)
            logger.warning("bm25s not available: %s", exc)

    def backend_name(self) -> str:
        """返回后端标识。"""
        return "bm25s"

    def retrieve(
        self,
        *,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> KeywordBackendResult:
        """用 bm25s 执行关键词检索。

        单条 view 查询失败只记录到该 view 的 debug 并跳过；整体失败则返回带 fallback_reason 的空结果，
        由 route_retriever 降级到内置 BM25。
        """
        if not self._bm25s_available:
            return KeywordBackendResult(
                results=[],
                debug={"bm25s_available": False, "import_error": self._import_error},
                backend_name=self.backend_name(),
                fallback_reason=f"bm25s_not_available: {self._import_error}",
            )

        if not retrieval_index.documents:
            return KeywordBackendResult(
                results=[],
                debug={
                    **retrieval_index.to_keyword_debug(0, full_scan_used=False),
                    "bm25s_available": True,
                    "bm25s_index_built": False,
                    "bm25s_corpus_size": 0,
                },
                backend_name=self.backend_name(),
                fallback_reason="empty_collection",
            )

        try:
            bm25s_index, corpus_mapping = self._build_bm25s_index(retrieval_index)
            state = self._new_run_state()
            for query_view in query_views:
                view = self._prepare_query_view(query_view)
                if view is None:
                    continue
                try:
                    rows = self._query_bm25s(
                        bm25s_index=bm25s_index,
                        query_tokens=view.expanded_tokens,
                        top_k=top_k,
                        corpus_mapping=corpus_mapping,
                        retrieval_index=retrieval_index,
                    )
                except Exception as query_exc:
                    logger.warning("bm25s query failed for view %s: %s", view.view_id, query_exc)
                    state.record_view(view, [], top_k=top_k, retrieval_index=retrieval_index, error=str(query_exc))
                    continue
                state.record_view(view, rows, top_k=top_k, retrieval_index=retrieval_index)

            # bm25s 的 corpus 仍是一条 index 一个 document；基类会在输出前聚合回 chunk，避免候选重复。
            results, debug = self._finalize_keyword_retrieval(
                state,
                query_views=query_views,
                top_k=top_k,
                query_profile=query_profile,
                retrieval_index=retrieval_index,
                backend_debug={
                    "bm25s_available": True,
                    "bm25s_index_built": True,
                    "bm25s_corpus_size": len(corpus_mapping),
                },
            )
            return KeywordBackendResult(results=results, debug=debug, backend_name=self.backend_name())

        except Exception as exc:
            logger.warning("bm25s backend failed: %s", exc, exc_info=True)
            return KeywordBackendResult(
                results=[],
                debug={
                    **retrieval_index.to_keyword_debug(0, full_scan_used=False),
                    "bm25s_available": True,
                    "bm25s_error": str(exc),
                },
                backend_name=self.backend_name(),
                fallback_reason=f"bm25s_execution_failed: {exc}",
            )

    def _build_bm25s_index(self, retrieval_index: Any) -> tuple:
        """为 retrieval_index 的全部文档建立 bm25s 索引，并缓存在 retrieval_index 对象上。

        Returns:
            (bm25s_index, corpus_mapping)，其中 corpus_mapping[corpus_idx] = doc_id。
        """
        # bm25s 索引挂在 retrieval_index 对象上：同一个索引对象只建一次，对象被替换时缓存自然随之失效。
        if hasattr(retrieval_index, "_bm25s_index") and hasattr(retrieval_index, "_bm25s_corpus_mapping"):
            return retrieval_index._bm25s_index, retrieval_index._bm25s_corpus_mapping

        corpus_texts: List[List[str]] = []
        corpus_mapping: List[int] = []
        for doc_id, document in enumerate(retrieval_index.documents):
            corpus_texts.append(self._build_document_tokens(document))
            corpus_mapping.append(doc_id)

        bm25_index = self._bm25s.BM25()
        # benchmark/接口会频繁构建小集合索引，关闭 bm25s 自带进度条，避免诊断输出被 tqdm 打散。
        bm25_index.index(corpus_texts, show_progress=False)

        retrieval_index._bm25s_index = bm25_index
        retrieval_index._bm25s_corpus_mapping = corpus_mapping

        logger.info(
            "Built bm25s index: collection=%s corpus_size=%d",
            retrieval_index.collection_name,
            len(corpus_mapping),
        )

        return bm25_index, corpus_mapping

    def _build_document_tokens(self, document: Any) -> List[str]:
        """把文档的字段级 token 计数展开成 bm25s 可索引的 token 序列。

        bm25s 不支持字段权重，这里用“按 字段权重 × index 权重 重复 token”近似实现加权。
        """
        # 延迟导入：测试会重新导入 retrieval_index，这里需要读取当前生效的权重表。
        from services.retrieval.retrieval_index import KEYWORD_FIELD_WEIGHTS

        tokens: List[str] = []
        field_token_counts = document.field_token_counts or {"body": document.token_counts}
        try:
            index_weight = float(getattr(document, "retrieval_index_weight", 1.0) or 1.0)
        except (TypeError, ValueError):
            index_weight = 1.0

        for field_name, field_counts in field_token_counts.items():
            weight = KEYWORD_FIELD_WEIGHTS.get(field_name, 1.0)
            if weight <= 0:
                continue

            # bm25s 只能接收离散 token，低权重 index 仍保留至少一次匹配，避免 section_anchor 被取整吞掉。
            for token, count in field_counts.items():
                weighted_count = max(1, int(round(count * weight * index_weight)))
                tokens.extend([token] * weighted_count)

        # bm25s 不接受空文档，用一个空字符串 token 占位（不会被任何真实 query token 命中）。
        return tokens if tokens else [""]

    def _query_bm25s(
        self,
        *,
        bm25s_index: Any,
        query_tokens: List[str],
        top_k: int,
        corpus_mapping: List[int],
        retrieval_index: Any,
    ) -> List[Dict[str, Any]]:
        """用 bm25s 检索一条 query view，返回基类约定格式的候选行（按 bm25s 分数降序）。

        每行为 ``{"doc_id", "raw_score", "matched_fields": Counter, "term_details": [...]}``；
        bm25s 只返回总分，命中字段与逐词明细由本类根据文档 token 计数补算。
        """
        if not query_tokens:
            return []

        effective_top_k = min(max(int(top_k or 0), 0), len(corpus_mapping))
        if effective_top_k <= 0:
            return []

        # bm25s 要求 k 不能超过 corpus size；benchmark 小 fixture 会传较大的 recall_candidate_limit，
        # 因此这里按真实索引规模截断，只影响候选上限，不改变 BM25 排序语义。
        results, scores = bm25s_index.retrieve(
            [query_tokens],
            k=effective_top_k,
            show_progress=False,
        )

        query_results: List[Dict[str, Any]] = []
        # bm25s 返回批量结果，这里只查了一条 query，取第 0 行。
        for corpus_idx, score in zip(results[0], scores[0]):
            if corpus_idx < 0 or corpus_idx >= len(corpus_mapping):
                continue

            doc_id = corpus_mapping[corpus_idx]
            if doc_id >= len(retrieval_index.documents):
                continue
            if float(score) <= 0:
                # bm25s 会用 0 分候选补齐 top_k；这些不是关键词命中，不能进入后续 RRF 投票。
                continue

            document = retrieval_index.documents[doc_id]
            query_results.append(
                {
                    "doc_id": doc_id,
                    "raw_score": float(score),
                    # bm25s 拿不到字段级得分，每个命中字段按 1 计数（多个 view 命中时累加）。
                    "matched_fields": Counter(dict.fromkeys(self._extract_matched_fields(document, query_tokens), 1.0)),
                    "term_details": self._extract_matched_terms(document, query_tokens, retrieval_index),
                }
            )

        return query_results

    @staticmethod
    def _extract_matched_fields(document: Any, query_tokens: List[str]) -> List[str]:
        """返回包含任一 query token 的字段名（按文档字段顺序）。"""
        field_token_counts = document.field_token_counts or {"body": document.token_counts}
        query_token_set = set(query_tokens)
        return [
            field_name
            for field_name, field_counts in field_token_counts.items()
            if any(token in field_counts for token in query_token_set)
        ]

    @staticmethod
    def _extract_matched_terms(document: Any, query_tokens: List[str], retrieval_index: Any) -> List[Dict[str, Any]]:
        """补算命中词明细：idf、近似得分和所在字段，按得分降序。

        idf 与内置 BM25 一致使用 Lucene 公式 ``log(1 + (N - df + 0.5) / (df + 0.5))``；
        term_score 只是 ``idf × 词频`` 的近似值，仅用于排序与解释，不参与召回打分。
        """
        doc_counts = document.token_counts
        field_token_counts = document.field_token_counts or {"body": document.token_counts}
        total_docs = len(retrieval_index.documents)

        matched_terms: List[Dict[str, Any]] = []
        for token in set(query_tokens):
            if token not in doc_counts:
                continue

            df = retrieval_index.document_frequency.get(token, 1)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            token_fields = [field_name for field_name, field_counts in field_token_counts.items() if token in field_counts]

            matched_terms.append(
                {
                    "token": token,
                    "idf": float(idf),
                    "term_score": float(idf * doc_counts[token]),
                    "fields": token_fields,
                }
            )

        matched_terms.sort(key=lambda t: t["term_score"], reverse=True)
        return matched_terms
