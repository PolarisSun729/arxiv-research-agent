"""按证据需求编排的检索适配器：把研究图的语义检索动作翻译成一次检索管线调用。

四条职责边界：

1. **只翻译，不决策**：检索轮数上限、目标需求是否仍然开放，都由图骨架的 Action Gate
   在调用之前裁决。适配器只把 ``SearchPaperAction`` 变成 (query, collection, options)，
   再把召回 chunk 映射回证据候选，永远不是预算或安全性边界。
2. **末轮降级**：预算最后一轮跳过 query rewrite 与 LLM rerank（PRD D5）。判定依赖图的
   调用时序——``graph._search_node`` 先自增 ``retrieval_count`` 再调用适配器，因此"本轮
   之后没有剩余额度"等价于 ``retrieval_count >= max_retrievals``。
3. **候选池去重**：已在池中且已归属本需求的 chunk 不再回传；同一 chunk 被另一个需求命中
   时仍然回传，让 ``evidence_pool.merge_candidates`` 补记 matched_need_ids——跨需求的证据
   归属是覆盖投影与证据包选择的输入，不能被去重顺手吞掉。
4. **失败不上抛**：研究是有界循环，单轮检索故障回传空候选加失败状态，由图的 no_progress
   计数收口成有界终止，并把 status 留在研究轨迹里；把它升级成系统异常会让已经攒到证据的
   运行整体作废。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from services.retrieval.contracts import RetrievalOptions
from utils.logging_utils import info_event

from ..evidence_pool import candidate_identity

logger = logging.getLogger(__name__)

_MAX_QUERY_CHARS = 300
_MAX_SECTION_HINTS = 3
_DEFAULT_TOP_K = 8


@dataclass(frozen=True)
class PaperRetrievalTarget:
    """一次研究运行的检索目标：论文 QA 索引 collection 及其画像上下文。"""

    collection_name: str
    paper_context: dict[str, Any] = field(default_factory=dict)


def _coerce_target(raw: Any) -> PaperRetrievalTarget | None:
    """解析器可以返回 dataclass 或普通字典；collection 缺失即视为目标不可用。"""

    if raw is None:
        return None
    if isinstance(raw, PaperRetrievalTarget):
        return raw if raw.collection_name.strip() else None
    if isinstance(raw, Mapping):
        collection_name = str(raw.get("collection_name") or "").strip()
        paper_context = raw.get("paper_context")
        return (
            PaperRetrievalTarget(
                collection_name=collection_name,
                paper_context=dict(paper_context or {}),
            )
            if collection_name
            else None
        )
    collection_name = str(getattr(raw, "collection_name", "") or "").strip()
    if not collection_name:
        return None
    return PaperRetrievalTarget(
        collection_name=collection_name,
        paper_context=dict(getattr(raw, "paper_context", None) or {}),
    )


def _identity_text(value: Any) -> str:
    """索引 ID 归一：``chunking_service`` 的 chunk 编号从 1 开始，0 是归一化层的未知哨兵值。"""

    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"", "0"} else text


def _coerce_page_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _find_need(state: Any, need_id: str) -> Any:
    for need in getattr(state, "evidence_needs", []) or []:
        if getattr(need, "need_id", "") == need_id:
            return need
    return None


def _build_query(action: Any, state: Any) -> str:
    """检索查询 = 动作提出的中立取证任务 + 需求描述 + 章节线索。

    动作 query 是决策器针对本轮取证目标提出的检索词，优先级最高；需求描述定义了这一轮
    到底要证实什么，动作 query 没覆盖时补上；研究问题只在前两者都为空时兜底——它是整个
    问题的范围，直接拼进去会把针对单一需求的查询重新泛化。
    """

    parts: list[str] = []
    base = str(getattr(action, "query", "") or "").strip()
    if base:
        parts.append(base)
    need = _find_need(state, str(getattr(action, "target_need_id", "") or ""))
    description = str(getattr(need, "description", "") or "").strip()
    if description and description not in base:
        parts.append(description)
    if not parts:
        parts.append(str(getattr(state, "research_question", "") or "").strip())
    hints = [
        hint
        for hint in (str(item or "").strip() for item in getattr(action, "section_hints", []) or [])
        if hint
    ][:_MAX_SECTION_HINTS]
    joined = " ".join(part for part in parts if part)
    for hint in hints:
        if hint.lower() not in joined.lower():
            joined = f"{joined} {hint}"
    return joined[:_MAX_QUERY_CHARS].strip()


class NeedOrchestratedRetriever:
    """把语义检索动作接到生产检索管线上的适配器。"""

    def __init__(
        self,
        *,
        retrieval_pipeline: Any,
        target_resolver: Any,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._target_resolver = target_resolver
        self._top_k = max(1, int(top_k))

    def retrieve(self, action: Any, state: Any) -> dict[str, Any]:
        arxiv_id = str(getattr(getattr(state, "request", None), "arxiv_id", "") or "")
        query = _build_query(action, state)
        target = self._resolve_target(arxiv_id)
        if target is None:
            return {"status": "target_unavailable", "candidates": [], "query": query}

        lightweight = self._is_final_round(state)
        try:
            result = self._retrieval_pipeline.retrieve(
                user_query=query,
                collection_name=target.collection_name,
                paper_context=target.paper_context,
                options=self._build_options(lightweight=lightweight),
            )
        except Exception as exc:
            logger.exception(
                "research retrieval failed: arxiv_id=%s collection_name=%s need_id=%s",
                arxiv_id,
                target.collection_name,
                getattr(action, "target_need_id", ""),
            )
            return {
                "status": "retrieval_failed",
                "candidates": [],
                "query": query,
                "error": str(exc)[:200],
            }

        chunks = list((result or {}).get("chunks") or [])
        mapped = [
            payload
            for payload in (self._map_chunk(chunk, rank) for rank, chunk in enumerate(chunks, start=1))
            if payload is not None
        ]
        target_need_id = str(getattr(action, "target_need_id", "") or "")
        candidates = [
            payload for payload in mapped if not self._already_bound(payload, state, target_need_id)
        ]
        status = self._resolve_status(chunk_count=len(chunks), candidate_count=len(candidates))
        info_event(
            logger,
            "research.retrieval_round",
            run_id=getattr(getattr(state, "request", None), "research_run_id", ""),
            arxiv_id=arxiv_id,
            collection_name=target.collection_name,
            need_id=target_need_id,
            retrieval_round=getattr(state, "retrieval_count", 0),
            lightweight=lightweight,
            input=query,
            retrieved_count=len(chunks),
            candidate_count=len(candidates),
            status=status,
        )
        return {
            "status": status,
            "candidates": candidates,
            "query": query,
            "retrieved_count": len(chunks),
            "pool_duplicate_count": len(mapped) - len(candidates),
            "lightweight": lightweight,
        }

    def _resolve_target(self, arxiv_id: str) -> PaperRetrievalTarget | None:
        try:
            target = _coerce_target(self._target_resolver(arxiv_id))
        except Exception as exc:
            logger.warning("research retrieval target resolution failed: arxiv_id=%s error=%s", arxiv_id, exc)
            return None
        if target is None:
            logger.warning("research retrieval target unavailable: arxiv_id=%s", arxiv_id)
        return target

    def _is_final_round(self, state: Any) -> bool:
        limits = getattr(getattr(state, "request", None), "limits", None)
        max_retrievals = int(getattr(limits, "max_retrievals", 0) or 0)
        # 图在调用适配器之前已经自增计数，因此相等即代表本轮用尽最后一次检索额度。
        return max_retrievals > 0 and int(getattr(state, "retrieval_count", 0) or 0) >= max_retrievals

    def _build_options(self, *, lightweight: bool) -> RetrievalOptions:
        # None 表示沿用全局配置；只有末轮显式关掉两个 LLM 阶段，其余开关不在这里改写。
        return RetrievalOptions(
            top_k=self._top_k,
            enable_query_rewrite=False if lightweight else None,
            enable_llm_rerank=False if lightweight else None,
        )

    def _map_chunk(self, chunk: Any, rank: int) -> dict[str, Any] | None:
        payload = dict(chunk or {})
        content = str(payload.get("content") or payload.get("text") or "").strip()
        if not content:
            # 空正文进不了证据池，也会退化成内容哈希 ID，直接丢弃比留一条不可引用的候选好。
            return None
        return {
            "content": content,
            "chunk_type": str(payload.get("chunk_type") or "text").strip() or "text",
            "chunk_id": _identity_text(payload.get("chunk_id")),
            "parent_chunk_id": _identity_text(payload.get("parent_chunk_id")),
            "original_chunk_id": _identity_text(payload.get("original_chunk_id")),
            "section_path": str(payload.get("section_path") or "").strip(),
            "page_number": _coerce_page_number(payload.get("page_number")),
            "table_id": _identity_text(payload.get("table_id")),
            # rank/score 不进 EvidenceCandidate，只供研究轨迹记录本轮召回排名。
            "rank": rank,
            "score": _coerce_score(payload.get("score")),
        }

    def _already_bound(self, payload: dict[str, Any], state: Any, target_need_id: str) -> bool:
        existing = (getattr(state, "evidence_candidates", None) or {}).get(candidate_identity(payload))
        if existing is None:
            return False
        return target_need_id in (getattr(existing, "matched_need_ids", None) or [])

    def _resolve_status(self, *, chunk_count: int, candidate_count: int) -> str:
        if candidate_count > 0:
            return "completed"
        return "duplicate_only" if chunk_count > 0 else "empty_recall"
