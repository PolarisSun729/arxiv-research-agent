"""按证据需求编排的检索适配器单测：查询拼装、末轮降级、chunk→候选映射、去重与失败降级。"""

from __future__ import annotations

from typing import Any

from services.paper_evidence_research.actions import SearchPaperAction
from services.paper_evidence_research.contracts import PaperEvidenceResearchRequest, ResearchLimits
from services.paper_evidence_research.dependencies import NeedOrchestratedRetriever, PaperRetrievalTarget
from services.paper_evidence_research.evidence_pool import merge_candidates
from services.paper_evidence_research.state import (
    EvidenceCandidate,
    EvidenceNeed,
    PaperEvidenceResearchState,
)

_METHOD_NEED = {
    "need_id": "need-method",
    "description": "反思 token 触发检索的判定机制",
    "importance": "core",
    "status": "open",
}
_ABLATION_NEED = {
    "need_id": "need-ablation",
    "description": "消融实验是否支持反思模块有效",
    "importance": "core",
    "status": "open",
}


class RecordingPipeline:
    """记录每次检索调用的入参，返回预设 chunk 列表。"""

    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self.chunks = list(chunks or [])
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        *,
        user_query: str,
        collection_name: str,
        paper_context: Any = None,
        options: Any = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "user_query": user_query,
                "collection_name": collection_name,
                "paper_context": paper_context,
                "options": options,
            }
        )
        return {"chunks": [dict(chunk) for chunk in self.chunks]}


class ExplodingPipeline:
    def retrieve(self, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("vector store unavailable")


def _resolve_target(_arxiv_id: str) -> PaperRetrievalTarget:
    return PaperRetrievalTarget(
        collection_name="paper_qa_2401_00001",
        paper_context={"arxiv_id": "2401.00001", "collection_name": "paper_qa_2401_00001"},
    )


def _state(
    *,
    needs: list[dict[str, Any]] | None = None,
    candidates: dict[str, EvidenceCandidate] | None = None,
    retrieval_count: int = 1,
    limits: ResearchLimits | None = None,
    research_question: str = "反思 token 如何决定是否检索，消融实验是否支持它有效？",
) -> PaperEvidenceResearchState:
    return PaperEvidenceResearchState(
        request=PaperEvidenceResearchRequest(
            arxiv_id="2401.00001",
            original_question=research_question,
            user_id="user-1",
            session_id="session-1",
            research_run_id="research-run-retriever",
            limits=limits or ResearchLimits(),
        ),
        research_question=research_question,
        evidence_needs=[EvidenceNeed.model_validate(item) for item in (needs or [_METHOD_NEED])],
        evidence_candidates=dict(candidates or {}),
        retrieval_count=retrieval_count,
    )


def _action(**overrides: Any) -> SearchPaperAction:
    payload: dict[str, Any] = {
        "action": "search_paper",
        "target_need_id": "need-method",
        "objective": "discover",
        "retrieval_mode": "method",
        "query": "reflection token retrieval decision",
        "reason_code": "OPEN_EVIDENCE_NEED",
    }
    payload.update(overrides)
    return SearchPaperAction.model_validate(payload)


def _retriever(pipeline: Any, *, target_resolver: Any = _resolve_target, top_k: int = 8) -> NeedOrchestratedRetriever:
    return NeedOrchestratedRetriever(
        retrieval_pipeline=pipeline,
        target_resolver=target_resolver,
        top_k=top_k,
    )


def test_query_combines_action_query_need_description_and_section_hints() -> None:
    pipeline = RecordingPipeline()
    _retriever(pipeline).retrieve(_action(section_hints=["Method", "Method"]), _state())

    assert pipeline.calls[0]["user_query"] == (
        "reflection token retrieval decision 反思 token 触发检索的判定机制 Method"
    )
    assert pipeline.calls[0]["collection_name"] == "paper_qa_2401_00001"
    assert pipeline.calls[0]["paper_context"]["arxiv_id"] == "2401.00001"


def test_research_question_is_only_a_query_fallback() -> None:
    """研究问题是整问题范围，只有动作查询和需求描述都为空时才兜底，否则会把单需求查询重新泛化。"""

    pipeline = RecordingPipeline()
    state = _state(needs=[{"need_id": "need-method", "description": ""}])
    _retriever(pipeline).retrieve(_action(query=""), state)

    assert pipeline.calls[0]["user_query"] == state.research_question


def test_final_retrieval_round_drops_query_rewrite_and_llm_rerank() -> None:
    """图先自增 retrieval_count 再调用适配器，因此计数等于上限即代表这是最后一轮检索。"""

    pipeline = RecordingPipeline()
    limits = ResearchLimits(max_retrievals=2)

    result = _retriever(pipeline, top_k=5).retrieve(_action(), _state(retrieval_count=1, limits=limits))
    assert result["lightweight"] is False
    assert pipeline.calls[0]["options"].top_k == 5
    assert pipeline.calls[0]["options"].enable_query_rewrite is None
    assert pipeline.calls[0]["options"].enable_llm_rerank is None

    result = _retriever(pipeline, top_k=5).retrieve(_action(), _state(retrieval_count=2, limits=limits))
    assert result["lightweight"] is True
    assert pipeline.calls[1]["options"].enable_query_rewrite is False
    assert pipeline.calls[1]["options"].enable_llm_rerank is False


def test_chunks_are_mapped_into_evidence_candidate_payloads() -> None:
    pipeline = RecordingPipeline(
        [
            {
                "content": "  The reflection token decides whether retrieval is required.  ",
                # 归一化层用 0 表示"未知 chunk 编号"，真实编号从 1 开始，不能当成 ID 参与去重。
                "chunk_id": 0,
                "parent_chunk_id": "parent-method",
                "original_chunk_id": "original-method",
                "chunk_type": "text",
                "section_path": "2 Method",
                "page_number": "4",
                "score": 0.42,
            },
            {"content": "   ", "chunk_id": "chunk-blank"},
            {
                "content": "Without reflection tokens the score drops from 54.1 to 50.3.",
                "chunk_id": "chunk-ablation",
                "chunk_type": "table",
                "table_id": "table-3",
                "page_number": "",
                "score": None,
            },
        ]
    )

    result = _retriever(pipeline).retrieve(_action(), _state())

    assert result["status"] == "completed"
    assert result["retrieved_count"] == 3
    first, second = result["candidates"]
    assert first["content"] == "The reflection token decides whether retrieval is required."
    assert first["chunk_id"] == ""
    assert first["parent_chunk_id"] == "parent-method"
    assert first["page_number"] == 4
    assert first["rank"] == 1
    assert first["score"] == 0.42
    # 空正文候选被丢弃，但它确实占用了本轮召回的第 2 名；重排名会让轨迹里的 rank 失真。
    assert second["chunk_id"] == "chunk-ablation"
    assert second["table_id"] == "table-3"
    assert second["page_number"] is None
    assert second["rank"] == 3
    assert second["score"] is None


def _pooled_state(*, matched_need_ids: list[str]) -> PaperEvidenceResearchState:
    pooled = EvidenceCandidate(
        candidate_id="chunk-method",
        content="The reflection token decides whether retrieval is required.",
        chunk_id="chunk-method",
        section_path="2 Method",
        matched_need_ids=list(matched_need_ids),
    )
    return _state(
        needs=[_METHOD_NEED, _ABLATION_NEED],
        candidates={pooled.candidate_id: pooled},
    )


def test_candidate_already_bound_to_the_current_need_is_filtered() -> None:
    pipeline = RecordingPipeline(
        [{"content": "The reflection token decides whether retrieval is required.", "chunk_id": "chunk-method"}]
    )
    state = _pooled_state(matched_need_ids=["need-method"])

    result = _retriever(pipeline).retrieve(_action(target_need_id="need-method"), state)

    assert result["candidates"] == []
    assert result["pool_duplicate_count"] == 1
    assert result["status"] == "duplicate_only"


def test_candidate_matched_by_another_need_still_reaches_the_pool() -> None:
    """跨需求命中必须回传，让 merge_candidates 补记 matched_need_ids——覆盖投影与证据包选择都靠它。"""

    pipeline = RecordingPipeline(
        [{"content": "The reflection token decides whether retrieval is required.", "chunk_id": "chunk-method"}]
    )
    state = _pooled_state(matched_need_ids=["need-method"])

    result = _retriever(pipeline).retrieve(_action(target_need_id="need-ablation"), state)

    assert result["status"] == "completed"
    assert [item["chunk_id"] for item in result["candidates"]] == ["chunk-method"]

    merged, new_count, duplicate_count = merge_candidates(
        state.evidence_candidates,
        result["candidates"],
        target_need_id="need-ablation",
        max_items=20,
        max_total_context_chars=30_000,
    )
    assert (new_count, duplicate_count) == (0, 1)
    assert merged["chunk-method"].matched_need_ids == ["need-method", "need-ablation"]


def test_candidate_payload_enters_the_pool_without_leaking_rank_or_score() -> None:
    """rank/score 只服务研究轨迹；它们必须止步于证据池模型，不能变成候选的可引用字段。"""

    pipeline = RecordingPipeline(
        [
            {
                "content": "Without reflection tokens the score drops from 54.1 to 50.3.",
                "chunk_id": "chunk-ablation",
                "chunk_type": "table",
                "section_path": "5 Experiments/Ablation",
                "page_number": 5,
                "score": 0.87,
            }
        ]
    )
    result = _retriever(pipeline).retrieve(_action(), _state())

    merged, new_count, duplicate_count = merge_candidates(
        {},
        result["candidates"],
        target_need_id="need-method",
        max_items=20,
        max_total_context_chars=30_000,
    )

    assert (new_count, duplicate_count) == (1, 0)
    candidate = merged["chunk-ablation"]
    assert candidate.chunk_type == "table"
    assert candidate.section_path == "5 Experiments/Ablation"
    assert candidate.page_number == 5
    assert candidate.matched_need_ids == ["need-method"]
    assert not hasattr(candidate, "rank")
    assert not hasattr(candidate, "score")


def test_empty_recall_is_reported_as_its_own_status() -> None:
    result = _retriever(RecordingPipeline([])).retrieve(_action(), _state())

    assert result["status"] == "empty_recall"
    assert result["candidates"] == []
    assert result["retrieved_count"] == 0


def test_missing_retrieval_target_does_not_touch_the_pipeline() -> None:
    pipeline = RecordingPipeline([{"content": "should never be retrieved", "chunk_id": "chunk-x"}])

    result = _retriever(pipeline, target_resolver=lambda _arxiv_id: None).retrieve(_action(), _state())

    assert result["status"] == "target_unavailable"
    assert result["candidates"] == []
    # 目标不可用时仍然回传 query，轨迹才能说明这一轮本来要查什么。
    assert result["query"].startswith("reflection token retrieval decision")
    assert pipeline.calls == []


def test_target_resolver_failure_degrades_to_target_unavailable() -> None:
    def _explode(_arxiv_id: str) -> Any:
        raise RuntimeError("qa index lookup failed")

    result = _retriever(RecordingPipeline(), target_resolver=_explode).retrieve(_action(), _state())

    assert result["status"] == "target_unavailable"


def test_blank_collection_name_counts_as_unavailable_target() -> None:
    def _blank(_arxiv_id: str) -> dict[str, Any]:
        return {"collection_name": "   ", "paper_context": {}}

    result = _retriever(RecordingPipeline(), target_resolver=_blank).retrieve(_action(), _state())

    assert result["status"] == "target_unavailable"


def test_mapping_target_is_accepted_by_the_adapter() -> None:
    pipeline = RecordingPipeline()

    def _mapping(_arxiv_id: str) -> dict[str, Any]:
        return {"collection_name": "paper_qa_mapping", "paper_context": {"arxiv_id": "2401.00001"}}

    _retriever(pipeline, target_resolver=_mapping).retrieve(_action(), _state())

    assert pipeline.calls[0]["collection_name"] == "paper_qa_mapping"
    assert pipeline.calls[0]["paper_context"] == {"arxiv_id": "2401.00001"}


def test_retrieval_failure_is_a_status_not_an_exception() -> None:
    """单轮检索故障必须由图的 no_progress 计数收口；上抛会让已经攒到证据的运行整体作废。"""

    result = _retriever(ExplodingPipeline()).retrieve(_action(), _state())

    assert result["status"] == "retrieval_failed"
    assert result["candidates"] == []
    assert "vector store unavailable" in result["error"]
