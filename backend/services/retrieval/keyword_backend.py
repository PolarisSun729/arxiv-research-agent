"""关键词（BM25）召回后端。

route_retriever 的 keyword 路由通过 ``KeywordBackend`` 抽象在两种实现之间切换：

- ``InternalBM25Backend``：项目内置 BM25，逐候选文档按字段加权打分，能给出逐词得分明细；
  同时是 bm25s 不可用或运行失败时的兜底实现。
- ``BM25sBackend``（见 bm25s_backend.py）：基于 bm25s 库的 BM25，整库建索引后批量检索，速度更快。

两种实现只在“单条 query view 怎么召回候选文档”上不同。token 预处理、跨 view 的加权 RRF 融合、
把 index 级命中聚合回 chunk、结果字段回填和 debug 结构都放在基类 ``KeywordBackend`` 中共用，
保证切换后端时下游（融合、rerank、trace）看到的输出契约完全一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from services.retrieval.numeric_utils import min_max_normalize
from services.retrieval.retrieval_rules import RetrievalRules

# 图表 caption / 图表辅助文本字段；命中这些字段说明证据来自图表而不是正文，下游会据此降权或加标记。
_ASSET_FIELDS = frozenset({"asset_caption", "asset_aux"})


def _route_retriever() -> Any:
    """延迟导入 RouteRetriever，复用其中的 BM25 打分、噪声检测等静态方法。

    不在模块顶层导入：一是避免 route_retriever ↔ keyword_backend 形成循环依赖；
    二是测试会从 sys.modules 移除并重新导入 route_retriever，延迟导入能保证拿到的是当前生效的类。
    """
    from services.retrieval.route_retriever import RouteRetriever

    return RouteRetriever


def _append_unique(items: List[Any], value: Any) -> None:
    """保持插入顺序地追加去重元素（列表很短，线性查找足够）。"""
    if value not in items:
        items.append(value)


def _top_fields(matched_fields: Counter, limit: int = 4) -> List[str]:
    """按命中权重取前 limit 个字段名；权重相同的字段保持首次出现顺序（most_common 是稳定排序）。"""
    return [field for field, _ in matched_fields.most_common(limit)]


def _sort_view_contributions(contributions: Any) -> List[Dict[str, Any]]:
    """按 RRF 投票、原始 BM25 分降序排列各 query view 的贡献，让 debug 先展示主导命中的 view。"""
    return sorted(
        list(contributions or []),
        key=lambda item: (
            float(item.get("rrf_vote", 0.0) or 0.0),
            float(item.get("raw_score", 0.0) or 0.0),
        ),
        reverse=True,
    )


def build_keyword_matched_terms(term_traces: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 term trace 字典展开成命中词列表，按单词得分降序、idf 次之，便于解释“哪些词贡献最大”。"""
    matched = [
        {
            "token": str(trace.get("token", "")),
            "idf": float(trace.get("idf", 0.0) or 0.0),
            "term_score": float(trace.get("best_term_score", 0.0) or 0.0),
            "fields": list(trace.get("fields", []) or []),
            "query_sources": list(trace.get("query_sources", []) or []),
        }
        for trace in term_traces.values()
    ]
    matched.sort(key=lambda item: (item["term_score"], item["idf"]), reverse=True)
    return matched


def aggregate_keyword_index_hits_to_chunks(ranked: List[tuple], retrieval_index: Any) -> List[tuple]:
    """把 retrieval-index 级 BM25 命中聚合回 chunk，保留每个 chunk 的 top index 命中供 debug/rerank 使用。

    BM25 的“文档”是 retrieval index（同一 chunk 可能有正文、标题、摘要等多个 index），
    但 route 对外输出的是 chunk 候选；不聚合的话同一 chunk 会重复占用多个候选名额。
    聚合规则：RRF 融合分相加、原始 BM25 分取最大、命中字段与 term trace 合并。
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for doc_id, payload in ranked:
        if doc_id < 0 or doc_id >= len(retrieval_index.documents):
            continue
        document = retrieval_index.documents[doc_id]
        chunk_key = keyword_chunk_key_for_document(document, doc_id)
        group = grouped.setdefault(
            chunk_key,
            {
                # 以该 chunk 第一个（即融合分最高的）index 命中作为代表文档，后续回填 chunk 字段时使用。
                "representative_doc_id": doc_id,
                "fusion_score": 0.0,
                "best_raw_score": 0.0,
                "matched_fields": Counter(),
                "view_contributions": [],
                "term_traces": {},
                "query_sources": [],
                "matched_doc_ids": [],
                "top_matched_indexes": [],
            },
        )
        group["fusion_score"] += float(payload.get("fusion_score", 0.0) or 0.0)
        group["best_raw_score"] = max(
            float(group.get("best_raw_score", 0.0) or 0.0),
            float(payload.get("best_raw_score", 0.0) or 0.0),
        )
        group["matched_fields"].update(payload.get("matched_fields", Counter()))
        _append_unique(group["matched_doc_ids"], doc_id)
        for source in payload.get("query_sources", []) or []:
            _append_unique(group["query_sources"], source)

        for contribution in payload.get("view_contributions", []) or []:
            contribution_row = dict(contribution)
            contribution_row.setdefault("retrieval_index_id", document.retrieval_index_id)
            contribution_row.setdefault("retrieval_index_type", document.retrieval_index_type)
            group["view_contributions"].append(contribution_row)

        _merge_keyword_term_traces(group["term_traces"], payload.get("term_traces", {}) or {})
        group["top_matched_indexes"].append(build_keyword_matched_index_entry(document, payload))

    chunk_ranked: List[tuple] = []
    for group in grouped.values():
        group["top_matched_indexes"].sort(
            key=lambda item: (
                float(item.get("bm25_fusion_score", 0.0) or 0.0),
                float(item.get("best_raw_bm25_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        # 只透传最强的 3 个 index 命中，避免 debug/rerank 载荷随 index 数量膨胀。
        group["top_matched_indexes"] = group["top_matched_indexes"][:3]
        group["matched_index_count"] = len(group.get("matched_doc_ids", []) or [])
        representative_doc_id = int(group.pop("representative_doc_id"))
        chunk_ranked.append((representative_doc_id, group))

    chunk_ranked.sort(
        key=lambda item: (
            float(item[1].get("fusion_score", 0.0) or 0.0),
            float(item[1].get("best_raw_score", 0.0) or 0.0),
            int(item[1].get("matched_index_count", 0) or 0),
        ),
        reverse=True,
    )
    return chunk_ranked


def keyword_chunk_key_for_document(document: Any, fallback_doc_id: int) -> str:
    """生成 chunk 聚合键；优先使用稳定 chunk 标识，缺失时才退回内容前缀防止误合并。"""
    chunk = dict(getattr(document, "chunk", {}) or {})
    metadata = dict(chunk.get("metadata", {}) or {})
    for key in ("chunk_id", "parent_chunk_id", "original_chunk_id"):
        value = chunk.get(key) if key in chunk else metadata.get(key)
        normalized = str(value or "").strip()
        if normalized:
            return f"{key}:{normalized}"
    # 没有任何稳定 ID 时把 doc_id 也拼进键里：宁可不合并，也不能把不同 chunk 误合并成一个。
    content = str(chunk.get("content") or metadata.get("content") or "")[:120]
    return f"doc:{fallback_doc_id}:{content}"


def build_keyword_matched_index_entry(document: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """把单个 index 命中的 BM25 证据压成可透传的 debug/rerank 明细。

    ``index_*`` 与 ``matched_index_*`` 两组键取值相同：前者是 chunk 契约字段，
    后者是 rerank / trace 读取的命中字段，两边都保留以兼容各自的读取方。
    """
    terms = build_keyword_matched_terms(payload.get("term_traces", {}) or {})
    contributions = _sort_view_contributions(payload.get("view_contributions"))[:4]
    fusion_score = float(payload.get("fusion_score", 0.0) or 0.0)
    return {
        "index_id": str(getattr(document, "retrieval_index_id", "") or ""),
        "index_type": str(getattr(document, "retrieval_index_type", "") or "body"),
        "index_text": str(getattr(document, "retrieval_index_text", "") or ""),
        "index_weight": float(getattr(document, "retrieval_index_weight", 1.0) or 1.0),
        "matched_index_id": str(getattr(document, "retrieval_index_id", "") or ""),
        "matched_index_type": str(getattr(document, "retrieval_index_type", "") or "body"),
        "matched_index_text": str(getattr(document, "retrieval_index_text", "") or ""),
        "matched_index_score": fusion_score,
        "bm25_fusion_score": fusion_score,
        "best_raw_bm25_score": float(payload.get("best_raw_score", 0.0) or 0.0),
        "matched_fields": _top_fields(payload.get("matched_fields", Counter())),
        "matched_terms": terms[:8],
        "query_sources": list(payload.get("query_sources", []) or []),
        "view_contributions": contributions,
    }


def _merge_keyword_term_traces(target: Dict[str, Dict[str, Any]], source: Dict[str, Dict[str, Any]]) -> None:
    """合并多个 index 的 term trace，保留最高 term score 和所有来源字段。"""
    for token, term in source.items():
        trace = target.setdefault(
            token,
            {
                "token": token,
                "idf": 0.0,
                "best_term_score": 0.0,
                "fields": [],
                "query_sources": [],
            },
        )
        trace["idf"] = max(float(trace.get("idf", 0.0) or 0.0), float(term.get("idf", 0.0) or 0.0))
        trace["best_term_score"] = max(
            float(trace.get("best_term_score", 0.0) or 0.0),
            float(term.get("best_term_score", 0.0) or 0.0),
        )
        for field_name in term.get("fields", []) or []:
            _append_unique(trace["fields"], field_name)
        for source_name in term.get("query_sources", []) or []:
            _append_unique(trace["query_sources"], source_name)


def _record_term_trace(term_traces: Dict[str, Dict[str, Any]], term: Dict[str, Any], source: str) -> None:
    """把单条 query view 的一个命中词记入 index 级 term trace（同词跨 view 取最大得分）。

    term 来自后端的逐词明细：``{"token", "idf", "term_score", "fields"}``。
    idf 恒为非负数，因此初值用 0.0 再取 max 即可得到真实 idf。
    """
    token = term["token"]
    trace = term_traces.setdefault(
        token,
        {
            "token": token,
            "idf": 0.0,
            "best_term_score": 0.0,
            "fields": [],
            "query_sources": [],
        },
    )
    trace["idf"] = max(float(trace["idf"]), float(term.get("idf", 0.0)))
    trace["best_term_score"] = max(float(trace["best_term_score"]), float(term.get("term_score", 0.0)))
    for field in term.get("fields", []):
        _append_unique(trace["fields"], field)
    _append_unique(trace["query_sources"], source)


class _PreparedQueryView(NamedTuple):
    """预处理完成、确定要参与检索的 query view。"""

    view_id: Any
    source: Any
    query: str
    weight: float
    # 过滤停用词后的 token，直接用于 BM25 检索。
    tokens: List[str]


class _KeywordRunState:
    """一次关键词检索中跨 query view 累积的中间状态。

    每条 query view 先各自召回候选，再用加权 RRF 投票融合：
    ``rrf_vote = view_weight / (rrf_k + rank)``，同一 index 文档的投票累加为 fusion_score。
    这样多条 view 都命中的文档排名靠前，而单条 view 的原始 BM25 分量纲不会互相干扰。
    """

    def __init__(self, *, rrf_k: int) -> None:
        self.rrf_k = rrf_k
        # doc_id（retrieval index 文档下标）-> 融合中的命中条目。
        self.aggregated: Dict[int, Dict[str, Any]] = {}
        self.per_query_debug: List[Dict[str, Any]] = []
        self.original_query_tokens: List[str] = []

    def record_view(
        self,
        view: _PreparedQueryView,
        rows: List[Dict[str, Any]],
        *,
        top_k: int,
        retrieval_index: Any,
        error: Optional[str] = None,
    ) -> None:
        """记录一条 query view 的召回结果，并把前 top_k 个候选投票进融合池。

        rows 是后端统一格式的候选（按原始分降序）：
        ``{"doc_id", "raw_score", "matched_fields": Counter, "term_details": [...]}``。
        归一化分数基于该 view 的全部候选计算，只有前 top_k 个参与 RRF 投票。
        """
        for token in view.tokens:
            _append_unique(self.original_query_tokens, token)

        if not rows:
            self.per_query_debug.append(self._view_debug_row(view, candidate_count=0, top_hits=[], error=error))
            return

        normalized_scores = min_max_normalize([float(row["raw_score"]) for row in rows])
        top_hits: List[Dict[str, Any]] = []
        for rank, (row, normalized_score) in enumerate(zip(rows[:top_k], normalized_scores), start=1):
            doc_id = row["doc_id"]
            raw_score = float(row["raw_score"])
            normalized_score = float(normalized_score)
            rrf_vote = view.weight * (1.0 / (self.rrf_k + rank))
            term_details = row.get("term_details", []) or []

            entry = self.aggregated.setdefault(
                doc_id,
                {
                    "fusion_score": 0.0,
                    "best_raw_score": 0.0,
                    "matched_fields": Counter(),
                    "view_contributions": [],
                    "term_traces": {},
                    "query_sources": [],
                },
            )
            entry["fusion_score"] += rrf_vote
            entry["best_raw_score"] = max(float(entry["best_raw_score"]), raw_score)
            entry["matched_fields"].update(row["matched_fields"])
            _append_unique(entry["query_sources"], view.source)
            for term in term_details:
                _record_term_trace(entry["term_traces"], term, view.source)

            entry["view_contributions"].append(
                {
                    "view_id": view.view_id,
                    "source": view.source,
                    "query": view.query,
                    "rank": rank,
                    "weight": view.weight,
                    "raw_score": raw_score,
                    "normalized_score": normalized_score,
                    "rrf_vote": float(rrf_vote),
                    # contribution_score 只用于 debug 对比“按归一化分加权”的效果，排序仍以 RRF 投票为准。
                    "contribution_score": float(view.weight * normalized_score),
                    "matched_fields": _top_fields(row["matched_fields"]),
                    "matched_terms": [term["token"] for term in term_details[:6]],
                }
            )

            document = retrieval_index.documents[doc_id]
            top_hits.append(
                {
                    "chunk_id": str(document.chunk.get("chunk_id", "") or ""),
                    "retrieval_index_id": document.retrieval_index_id,
                    "retrieval_index_type": document.retrieval_index_type,
                    "rank": rank,
                    "raw_score": raw_score,
                    "normalized_score": normalized_score,
                    "rrf_vote": float(rrf_vote),
                    "matched_fields": _top_fields(row["matched_fields"]),
                }
            )

        self.per_query_debug.append(self._view_debug_row(view, candidate_count=len(rows), top_hits=top_hits))

    @staticmethod
    def _view_debug_row(
        view: _PreparedQueryView,
        *,
        candidate_count: int,
        top_hits: List[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成单条 query view 的 debug 行；后端查询报错时额外带上 error。"""
        row = {
            "view_id": view.view_id,
            "query": view.query,
            "source": view.source,
            "weight": view.weight,
            "query_tokens": view.tokens,
            "candidate_count": candidate_count,
            "top_hits": top_hits,
        }
        if error is not None:
            row["error"] = error
        return row

    def ranked_chunks(self, retrieval_index: Any) -> List[tuple]:
        """按融合分（并列看最佳原始分）排序 index 命中，再聚合成 chunk 级候选。"""
        ranked = [
            (doc_id, payload)
            for doc_id, payload in self.aggregated.items()
            if float(payload.get("fusion_score", 0.0) or 0.0) > 0
        ]
        ranked.sort(
            key=lambda item: (
                float(item[1].get("fusion_score", 0.0) or 0.0),
                float(item[1].get("best_raw_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return aggregate_keyword_index_hits_to_chunks(ranked, retrieval_index)

    def token_debug(self) -> Dict[str, Any]:
        """汇总所有 view 的 query token。"""
        return {"query_tokens": self.original_query_tokens}


class KeywordBackendResult:
    """关键词后端的标准化返回值。

    fallback_reason 非空表示该后端无法给出可信结果（如 bm25s 未安装、执行异常），
    route_retriever 会据此降级到 InternalBM25Backend 重新检索。
    """

    def __init__(
        self,
        *,
        results: List[Dict[str, Any]],
        debug: Dict[str, Any],
        backend_name: str,
        fallback_reason: str = "",
    ) -> None:
        self.results = results
        self.debug = debug
        self.backend_name = backend_name
        self.fallback_reason = fallback_reason


class KeywordBackend(ABC):
    """关键词召回后端基类。

    子类只负责“单条 query view → 候选文档行”的召回，并在 ``retrieve`` 中按以下顺序调用共用流程：
    ``_new_run_state`` → 逐条 ``_prepare_query_view`` 后召回并 ``record_view`` → ``_finalize_keyword_retrieval``。
    """

    def __init__(
        self,
        *,
        query_tools: Any,
        structural_bonus_builder: Any,
        fusion_service: Any,
    ) -> None:
        # 提供分词、停用词过滤、同义词扩展、乱码 token 判断（通常是 QueryPlanner）。
        self.query_tools = query_tools
        # 计算 chunk 的结构加分（通常是 RetrievalRules.compute_structural_bonus）。
        self.structural_bonus_builder = structural_bonus_builder
        # 两个后端目前都不直接使用 fusion_service（跨路由融合在 route_retriever 之后完成），
        # 保留该参数是为了与 RouteRetriever 的构造方式及现有调用方保持一致。
        self.fusion_service = fusion_service

    @abstractmethod
    def retrieve(
        self,
        *,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> KeywordBackendResult:
        """对一组 query view 执行关键词检索。

        Args:
            query_views: query view 列表，每项包含 query / source / weight / view_id 等字段。
            top_k: 每条 view 参与融合的候选数，同时也是最终返回的 chunk 数上限。
            query_profile: QueryProfile，提供意图、原始查询等信息。
            retrieval_index: CollectionRetrievalIndex，包含文档、倒排表和统计量。

        Returns:
            KeywordBackendResult：回填好路由字段的 chunk 列表和 debug 信息。
        """

    @abstractmethod
    def backend_name(self) -> str:
        """返回后端标识（写入 debug，并用于降级原因说明）。"""

    def _new_run_state(self) -> _KeywordRunState:
        """创建一次检索的累积状态；keyword 路由使用较小的 RRF 常数放大少量高质量 view 的区分度。"""
        return _KeywordRunState(rrf_k=_route_retriever().keyword_rrf_k())

    def _prepare_query_view(self, query_view: Dict[str, Any]) -> Optional[_PreparedQueryView]:
        """分词并过滤一条 query view；没有有效 token 或权重不为正时返回 None（跳过该 view）。"""
        query = str(query_view.get("query", "") or "")
        raw_tokens = self.query_tools.tokenize_for_keyword_search(query)
        tokens = self._filter_keyword_query_tokens(raw_tokens)
        if not tokens:
            return None
        query_weight = float(query_view.get("weight", 0.0) or 0.0)
        if query_weight <= 0:
            return None
        return _PreparedQueryView(
            view_id=query_view["view_id"],
            source=query_view["source"],
            query=query,
            weight=query_weight,
            tokens=tokens,
        )

    def _finalize_keyword_retrieval(
        self,
        state: _KeywordRunState,
        *,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
        backend_debug: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """把融合池聚合成 chunk 候选，回填结果字段并组装 debug。

        backend_debug 是后端特有的诊断字段（如 bm25s 的索引规模、内置 BM25 的字段权重），
        紧跟在索引统计信息之后写入 debug。
        """
        ranked = state.ranked_chunks(retrieval_index)
        debug: Dict[str, Any] = {
            **retrieval_index.to_keyword_debug(len(ranked), full_scan_used=False),
            **(backend_debug or {}),
            **state.token_debug(),
            "query_contributions": state.per_query_debug,
            "keyword_fusion_strategy": "weighted_rrf",
        }
        if not ranked:
            debug["matched_chunks"] = []
            return [], debug

        results = self._build_keyword_results(
            ranked=ranked,
            top_k=top_k,
            query_profile=query_profile,
            retrieval_index=retrieval_index,
        )
        debug["keyword_rrf_k"] = state.rrf_k
        debug["matched_chunks"] = self._build_matched_chunks_debug(ranked[:top_k], retrieval_index, query_profile)
        return results, debug

    def _build_keyword_results(
        self,
        *,
        ranked: List[tuple],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> List[Dict[str, Any]]:
        """把前 top_k 个 chunk 候选回填成 keyword 路由的标准 chunk 结果。

        归一化分数基于全部候选计算再截断，保证 top_k 变化时同一 chunk 的归一化分不变。
        """
        normalized_scores = min_max_normalize(
            [float(payload.get("fusion_score", 0.0) or 0.0) for _, payload in ranked]
        )

        results: List[Dict[str, Any]] = []
        for rank, ((doc_id, payload), normalized_score) in enumerate(
            zip(ranked[:top_k], normalized_scores[:top_k]), start=1
        ):
            document = retrieval_index.documents[doc_id]
            chunk = dict(document.chunk)
            fusion_score = float(payload.get("fusion_score", 0.0) or 0.0)
            best_raw_score = float(payload.get("best_raw_score", 0.0) or 0.0)
            top_matched_indexes = list(payload.get("top_matched_indexes", []) or [])
            top_index = top_matched_indexes[0] if top_matched_indexes else build_keyword_matched_index_entry(document, payload)
            query_contributions = _sort_view_contributions(payload.get("view_contributions"))
            main_fields = _top_fields(payload.get("matched_fields", Counter()))
            matched_terms = build_keyword_matched_terms(payload.get("term_traces", {}))
            query_sources = list(payload.get("query_sources", []))

            noise_flags = self._detect_keyword_noise_flags(
                matched_terms=matched_terms,
                matched_fields=main_fields,
                query_profile=query_profile,
                chunk=chunk,
            )

            chunk["retrieval_route"] = "keyword"
            chunk["source_query"] = " | ".join([item["query"] for item in query_contributions[:3]])
            chunk["route_rank"] = rank
            chunk["route_score"] = fusion_score
            chunk["normalized_route_score"] = float(normalized_score)
            chunk["structural_bonus"] = float(self.structural_bonus_builder(chunk, query_profile))
            chunk["keyword_match_fields"] = main_fields
            chunk["keyword_hit_asset_field"] = any(field in _ASSET_FIELDS for field in main_fields)
            chunk["keyword_query_contributions"] = query_contributions
            chunk["keyword_best_raw_bm25_score"] = best_raw_score
            chunk["bm25_raw_score"] = best_raw_score
            chunk["bm25_fused_score"] = fusion_score
            chunk["keyword_matched_terms"] = matched_terms
            chunk["keyword_query_sources"] = query_sources
            chunk["keyword_noise_flags"] = noise_flags
            # 对外仍是 chunk 候选；index 字段只说明本次 BM25 最强的命中入口，top 列表留给 rerank/debug。
            chunk["retrieval_index_id"] = top_index.get("index_id", document.retrieval_index_id)
            chunk["retrieval_index_type"] = top_index.get("index_type", document.retrieval_index_type)
            chunk["retrieval_index_text"] = top_index.get("index_text", document.retrieval_index_text)
            chunk["retrieval_index_weight"] = top_index.get("index_weight", document.retrieval_index_weight)
            chunk["retrieval_index_enabled_routes"] = list(document.retrieval_index_enabled_routes)
            chunk["index_id"] = chunk["retrieval_index_id"]
            chunk["index_type"] = chunk["retrieval_index_type"]
            chunk["index_text"] = chunk["retrieval_index_text"]
            chunk["index_weight"] = chunk["retrieval_index_weight"]
            chunk["matched_index_id"] = top_index.get("matched_index_id", chunk["retrieval_index_id"])
            chunk["matched_index_type"] = top_index.get("matched_index_type", chunk["retrieval_index_type"])
            chunk["matched_index_text"] = top_index.get("matched_index_text", chunk["retrieval_index_text"])
            chunk["matched_index_score"] = top_index.get("matched_index_score", fusion_score)
            chunk["matched_indexes"] = top_matched_indexes
            chunk["keyword_top_matched_indexes"] = top_matched_indexes
            chunk["keyword_matched_index_count"] = int(payload.get("matched_index_count", len(top_matched_indexes)) or 0)
            results.append(chunk)

        return results

    def _build_matched_chunks_debug(
        self, ranked: List[tuple], retrieval_index: Any, query_profile: Any
    ) -> List[Dict[str, Any]]:
        """生成 debug.matched_chunks：逐个 chunk 说明融合分、命中词/字段、噪声标记和主导 view。

        这里重新构造命中词等对象而不复用结果 chunk 中的同名字段，
        避免下游修改结果 chunk 时连带改掉 debug 快照。
        """
        matched_chunks = []
        for doc_id, payload in ranked:
            document = retrieval_index.documents[doc_id]
            matched_terms = build_keyword_matched_terms(payload.get("term_traces", {}))
            main_fields = _top_fields(payload.get("matched_fields", Counter()))
            top_index = (payload.get("top_matched_indexes") or [{}])[0]

            matched_chunks.append(
                {
                    "chunk_id": str(document.chunk.get("chunk_id", "") or ""),
                    "retrieval_index_id": top_index.get("matched_index_id", document.retrieval_index_id),
                    "retrieval_index_type": top_index.get("matched_index_type", document.retrieval_index_type),
                    "keyword_fusion_score": float(payload.get("fusion_score", 0.0) or 0.0),
                    "best_raw_bm25_score": float(payload.get("best_raw_score", 0.0) or 0.0),
                    "matched_fields": main_fields,
                    "matched_terms": matched_terms,
                    "query_sources": list(payload.get("query_sources", [])),
                    "matched_index_count": int(payload.get("matched_index_count", 0) or 0),
                    "top_matched_indexes": list(payload.get("top_matched_indexes", []) or []),
                    "noise_flags": self._detect_keyword_noise_flags(
                        matched_terms=matched_terms,
                        matched_fields=main_fields,
                        query_profile=query_profile,
                        chunk=document.chunk,
                    ),
                    "hit_asset_field": any(field in _ASSET_FIELDS for field in main_fields),
                    "view_contributions": _sort_view_contributions(payload.get("view_contributions"))[:4],
                }
            )
        return matched_chunks

    @staticmethod
    def _build_matched_terms(term_traces: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """兼容旧调用名，等价于模块函数 build_keyword_matched_terms。"""
        return build_keyword_matched_terms(term_traces)

    def _filter_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        """调用 query_tools 过滤停用词、保留关键词（limit 设为 token 数，即只过滤不截断）。"""
        extractor = getattr(self.query_tools, "extract_query_keywords", None)
        if callable(extractor):
            return extractor(tokens, limit=max(len(tokens), 1))
        return tokens

    def _detect_keyword_noise_flags(
        self,
        *,
        matched_terms: List[Dict[str, Any]],
        matched_fields: List[str],
        query_profile: Any,
        chunk: Dict[str, Any],
    ) -> List[str]:
        """检测关键词命中噪声（无有效命中词、只命中低 IDF 泛词、乱码词、只命中图表字段等），规则复用 RouteRetriever。"""
        return _route_retriever().detect_keyword_noise_flags(
            matched_terms=matched_terms,
            matched_fields=matched_fields,
            query_profile=query_profile,
            chunk=chunk,
            is_informative_token=self._is_informative_token,
        )

    def _is_informative_token(self, token: str) -> bool:
        """判断 token 是否有信息量；优先使用注入的 query_tools 规则，缺失时直接用 RetrievalRules 的同一套规则。"""
        checker = getattr(self.query_tools, "is_informative_keyword_token", None)
        if not callable(checker):
            checker = RetrievalRules.is_informative_keyword_token
        return bool(checker(token))


class InternalBM25Backend(KeywordBackend):
    """内置 BM25：用倒排表找候选，逐文档按字段加权计算 BM25 并给出逐词得分明细。

    也是 bm25s 不可用或执行失败时 route_retriever 的兜底后端。
    """

    def backend_name(self) -> str:
        """返回后端标识。"""
        return "internal_bm25"

    def retrieve(
        self,
        *,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> KeywordBackendResult:
        """执行内置 BM25 检索并包装成统一的后端结果。"""
        payload = self._internal_keyword_retrieve(
            query_views=query_views,
            top_k=top_k,
            query_profile=query_profile,
            retrieval_index=retrieval_index,
        )
        return KeywordBackendResult(
            results=payload["results"],
            debug=payload["debug"],
            backend_name=self.backend_name(),
        )

    def _internal_keyword_retrieve(
        self,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> Dict[str, Any]:
        """内置 BM25 主流程：逐 view 打分召回 → 加权 RRF 融合 → 聚合回 chunk → 回填结果字段。

        返回 ``{"results": [...], "debug": {...}}``（测试会直接调用本方法检查中间结构）。
        """
        if not retrieval_index.documents:
            return {"results": [], "debug": retrieval_index.to_keyword_debug(0, full_scan_used=False)}

        field_weights = self._keyword_field_weights_for_query(query_profile)
        state = self._new_run_state()
        for query_view in query_views:
            view = self._prepare_query_view(query_view)
            if view is None:
                continue
            rows = self._score_candidates(
                view.tokens,
                field_weights=field_weights,
                query_profile=query_profile,
                retrieval_index=retrieval_index,
            )
            state.record_view(view, rows, top_k=top_k, retrieval_index=retrieval_index)

        results, debug = self._finalize_keyword_retrieval(
            state,
            query_views=query_views,
            top_k=top_k,
            query_profile=query_profile,
            retrieval_index=retrieval_index,
            backend_debug={"field_weights": field_weights},
        )
        return {"results": results, "debug": debug}

    def _score_candidates(
        self,
        query_tokens: List[str],
        *,
        field_weights: Dict[str, float],
        query_profile: Any,
        retrieval_index: Any,
    ) -> List[Dict[str, Any]]:
        """用倒排表找出含任一 query token 的文档，逐个按字段加权计算 BM25，返回按原始分降序的候选。

        只对倒排表命中的文档打分，不做全量扫描；BM25 分不为正的文档直接丢弃。
        """
        route_retriever = _route_retriever()
        token_counter = Counter(query_tokens)
        candidate_doc_ids = set()
        for token in token_counter:
            candidate_doc_ids.update(retrieval_index.postings.get(token, set()))

        rows: List[Dict[str, Any]] = []
        for doc_id in candidate_doc_ids:
            document = retrieval_index.documents[doc_id]
            doc_field_weights = self._keyword_field_weights_for_document(field_weights, document.chunk, query_profile)
            doc_counts, doc_length, match_fields, token_field_hits = route_retriever.weight_keyword_document_fields(
                document.field_token_counts or {"body": document.token_counts},
                token_counter,
                doc_field_weights,
            )
            if not doc_counts:
                continue

            bm25_detail = route_retriever.bm25_score_detailed(
                token_counter,
                doc_counts,
                retrieval_index.document_frequency,
                len(retrieval_index.documents),
                retrieval_index.avgdl,
                document.content_prefix,
                doc_length=doc_length or document.doc_length,
                token_field_hits=token_field_hits,
            )
            raw_score = bm25_detail["score"]
            if raw_score <= 0:
                continue

            rows.append(
                {
                    "doc_id": doc_id,
                    "raw_score": float(raw_score),
                    "matched_fields": Counter(match_fields),
                    "term_details": bm25_detail["term_details"],
                }
            )

        rows.sort(key=lambda item: item["raw_score"], reverse=True)
        return rows

    def _keyword_field_weights_for_query(self, query_profile: Any) -> Dict[str, float]:
        """按查询意图调整字段权重：图表类问题放大图表 caption/辅助文本，其他问题压低以免图表噪声抢排名。"""
        # 延迟导入：测试会重新导入 retrieval_index，这里需要读取当前生效的权重表。
        from services.retrieval.retrieval_index import KEYWORD_FIELD_WEIGHTS

        weights = dict(KEYWORD_FIELD_WEIGHTS)
        main_intent = self._query_main_intent(query_profile)

        if main_intent == "figure_table":
            weights["asset_caption"] = 1.15
            weights["asset_aux"] = 0.72
        else:
            weights["asset_caption"] = min(weights.get("asset_caption", 0.0), 0.12)
            weights["asset_aux"] = min(weights.get("asset_aux", 0.0), 0.05)
        return weights

    def _keyword_field_weights_for_document(
        self,
        base_weights: Dict[str, float],
        chunk: Dict[str, Any],
        query_profile: Any,
    ) -> Dict[str, float]:
        """在查询级权重基础上做文档级调整。

        - 非图表问题命中图表 chunk 时，整体压低各字段权重，避免图表 chunk 靠正文碰词挤进前排；
        - 再乘以 retrieval index 自身的权重（如摘要 index 权重低于正文）。
        """
        weights = dict(base_weights)
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        is_asset_chunk = chunk_type in {"figure", "table"}
        is_figure_query = self._query_main_intent(query_profile) == "figure_table"

        if is_asset_chunk and not is_figure_query:
            weights["body"] = min(weights.get("body", 1.0), 0.38)
            weights["section_title"] = min(weights.get("section_title", 1.0), 0.35)
            weights["section_path"] = min(weights.get("section_path", 1.0), 0.25)
            weights["asset_caption"] = min(weights.get("asset_caption", 0.0), 0.08)
            weights["asset_aux"] = min(weights.get("asset_aux", 0.0), 0.03)
        try:
            index_weight = float(chunk.get("retrieval_index_weight", chunk.get("index_weight", 1.0)) or 1.0)
        except (TypeError, ValueError):
            index_weight = 1.0
        if index_weight != 1.0:
            # index_type 权重只调节该 index 文档的稀疏匹配强度，不改变 chunk 回填字段。
            weights = {field_name: float(value) * index_weight for field_name, value in weights.items()}
        return weights

    @staticmethod
    def _query_main_intent(query_profile: Any) -> str:
        """返回意图服务已经确认的正式 intent（小写，缺失为 other），与 RouteRetriever 共用同一实现。"""
        return _route_retriever().query_main_intent(query_profile)
