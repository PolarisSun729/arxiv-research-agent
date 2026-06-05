from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .schemas import ObservationResult, PlanRuntime, PlanStep
from .state import AgentState


def _tokenize(text: Any) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", str(text or "").lower()) if len(token) >= 2]


def _extract_papers(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, Mapping):
        papers = value.get("papers")
        if isinstance(papers, list):
            return [dict(item) for item in papers if isinstance(item, Mapping)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _extract_chunks(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in ("chunks", "paper_chunks", "reranked_chunks", "retrieved_chunks"):
            chunks = value.get(key)
            if isinstance(chunks, list):
                return [dict(item) for item in chunks if isinstance(item, Mapping)]
    return []


class Observer:
    """区分“工具执行成功”和“结果质量达标”，为后续重规划提供统一语义。"""

    def observe(
        self,
        *,
        step: PlanStep,
        resolved_input: Mapping[str, Any],
        raw_output: Any,
        normalized_output: Any,
        runtime: PlanRuntime,
        state: AgentState,
        error: Optional[str] = None,
    ) -> ObservationResult:
        if error:
            return ObservationResult(status="tool_error", reason=error, confidence=0.0, suggested_action="fallback")

        handler = getattr(self, f"_observe_{step.tool_name}", None)
        if callable(handler):
            return handler(resolved_input=resolved_input, raw_output=raw_output, normalized_output=normalized_output, runtime=runtime, state=state)
        return ObservationResult(status="success", reason="no_special_rule", confidence=1.0)

    def _observe_search_arxiv(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del normalized_output, runtime, state
        papers = _extract_papers(raw_output)
        if not papers:
            return ObservationResult(status="empty_result", reason="search_returned_no_papers", confidence=0.1, details={"paper_count": 0}, suggested_action="rewrite_arxiv_query")
        query_tokens = _tokenize((resolved_input.get("search_spec") or {}).get("query") if isinstance(resolved_input.get("search_spec"), Mapping) else None)
        if query_tokens:
            matched_titles = 0
            for paper in papers[:5]:
                title = str(paper.get("title") or "")
                abstract = str(paper.get("summary") or paper.get("abstract") or "")
                if any(token in f"{title} {abstract}".lower() for token in query_tokens):
                    matched_titles += 1
            if matched_titles == 0:
                return ObservationResult(status="low_confidence", reason="search_results_not_related_to_query", confidence=0.25, details={"paper_count": len(papers)}, suggested_action="rewrite_arxiv_query")
        return ObservationResult(status="success", reason="search_results_available", confidence=0.85, details={"paper_count": len(papers)})

    def _observe_validate_arxiv_results(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del runtime, state
        quality = normalized_output if isinstance(normalized_output, Mapping) else raw_output if isinstance(raw_output, Mapping) else {}
        result_count = int(quality.get("result_count") or 0)
        if result_count <= 0:
            return ObservationResult(status="empty_result", reason="validated_arxiv_results_empty", confidence=0.1, details={"result_count": result_count}, suggested_action="rewrite_arxiv_query")
        warnings = list(quality.get("warnings") or [])
        if "tool_failed" in warnings or "no_results" in warnings:
            return ObservationResult(status="low_confidence", reason="validated_arxiv_results_low_quality", confidence=0.35, details={"warnings": warnings}, suggested_action="rewrite_arxiv_query")
        papers = _extract_papers(resolved_input.get("arxiv_results"))
        titles = [str(paper.get("title") or "").strip().lower() for paper in papers if str(paper.get("title") or "").strip()]
        if len(set(titles)) < max(1, len(titles) // 2):
            return ObservationResult(status="low_confidence", reason="validated_arxiv_results_too_many_duplicates", confidence=0.45, details={"paper_count": len(papers)}, suggested_action="rewrite_arxiv_query")
        return ObservationResult(status="success", reason="validated_arxiv_results_ok", confidence=0.85, details={"result_count": result_count})

    def _observe_check_paper_index(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        status = str(payload.get("status") or "").lower()
        has_index = bool(payload.get("has_index"))
        if status in {"available", "indexed"} or has_index:
            return ObservationResult(status="success", reason="paper_index_available", confidence=0.95)
        if status in {"missing", "not_found"} or not has_index:
            return ObservationResult(status="need_confirmation", reason="paper_index_missing", confidence=0.9, suggested_action="request_confirmation")
        if status == "corrupted":
            return ObservationResult(status="tool_error", reason="paper_index_corrupted", confidence=0.1, suggested_action="rebuild_index")
        return ObservationResult(status="partial_success", reason="paper_index_status_unknown", confidence=0.5)

    def _observe_request_confirmation(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, normalized_output, runtime, state
        # request_confirmation 在新的执行模型里只负责准备确认上下文；
        # 真正的暂停点统一放在有副作用 step 的工具调用前，避免提前在“准备步骤”上打断并劫持恢复顺序。
        return ObservationResult(status="success", reason="confirmation_already_available", confidence=1.0)

    def _observe_retrieve_paper_chunks(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del raw_output, runtime
        chunks = _extract_chunks(normalized_output)
        if not chunks:
            return ObservationResult(status="empty_result", reason="paper_chunk_recall_empty", confidence=0.1, suggested_action="rewrite_paper_query")
        if len(chunks) < 2:
            return ObservationResult(status="partial_success", reason="paper_chunk_recall_too_small", confidence=0.45, details={"chunk_count": len(chunks)})
        query_tokens = _tokenize(resolved_input.get("message") or state.message)
        if query_tokens:
            matched = 0
            for chunk in chunks[:3]:
                text = str(chunk.get("text") or chunk.get("content") or "")
                if any(token in text.lower() for token in query_tokens):
                    matched += 1
            if matched == 0:
                return ObservationResult(status="low_confidence", reason="paper_chunk_recall_not_relevant", confidence=0.25, details={"chunk_count": len(chunks)}, suggested_action="rewrite_paper_query")
        return ObservationResult(status="success", reason="paper_chunk_recall_ok", confidence=0.8, details={"chunk_count": len(chunks)})

    def _observe_rerank_paper_chunks(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        chunks = _extract_chunks(normalized_output)
        if not chunks:
            return ObservationResult(status="empty_result", reason="reranked_chunks_empty", confidence=0.1, suggested_action="rewrite_paper_query")
        top_score = float((chunks[0].get("score") or chunks[0].get("rerank_score") or 0.0)) if chunks else 0.0
        if top_score < 0.15:
            return ObservationResult(status="low_confidence", reason="reranked_chunk_score_too_low", confidence=0.2, details={"top_score": top_score}, suggested_action="rewrite_paper_query")
        return ObservationResult(status="success", reason="reranked_chunks_usable", confidence=0.8, details={"top_score": top_score})

    def _observe_validate_qa_evidence(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        if not bool(payload.get("ok")):
            return ObservationResult(status="low_confidence", reason="qa_evidence_insufficient", confidence=0.2, details=dict(payload), suggested_action="rewrite_paper_query")
        if int(payload.get("chunk_count") or 0) < 1:
            return ObservationResult(status="low_confidence", reason="qa_evidence_missing_chunks", confidence=0.2, details=dict(payload), suggested_action="rewrite_paper_query")
        return ObservationResult(status="success", reason="qa_evidence_ok", confidence=0.85, details=dict(payload))

    def _observe_verify_answer_grounding(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del runtime, state
        answer = str(normalized_output or raw_output or "").strip()
        draft_answer = resolved_input.get("draft_answer") if isinstance(resolved_input.get("draft_answer"), Mapping) else {}
        sources = draft_answer.get("sources") if isinstance(draft_answer, Mapping) else None
        if not answer:
            return ObservationResult(status="low_confidence", reason="grounded_answer_empty", confidence=0.1, suggested_action="regenerate_with_stricter_grounding")
        if not sources:
            return ObservationResult(status="low_confidence", reason="grounded_answer_without_sources", confidence=0.3, details={"answer_preview": answer[:120]}, suggested_action="regenerate_with_stricter_grounding")
        return ObservationResult(status="success", reason="grounded_answer_ok", confidence=0.8)

    def _observe_load_user_profile(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        profile = payload.get("research_profile") if isinstance(payload.get("research_profile"), Mapping) else {}
        summary = payload.get("user_memory_summary")
        if not profile and not summary:
            return ObservationResult(status="empty_result", reason="user_profile_empty", confidence=0.2, suggested_action="fallback_to_message_recommendation")
        if not profile or not summary:
            return ObservationResult(status="partial_success", reason="user_profile_partial", confidence=0.55)
        return ObservationResult(status="success", reason="user_profile_available", confidence=0.85)

    def _observe_generate_recommendations(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        recommendations = list(payload.get("recommendations") or [])
        if not recommendations:
            return ObservationResult(status="empty_result", reason="recommendations_empty", confidence=0.1, suggested_action="fallback")
        titles = [str((item or {}).get("title") or "").strip().lower() for item in recommendations if isinstance(item, Mapping)]
        if len(set([title for title in titles if title])) < len([title for title in titles if title]):
            return ObservationResult(status="low_confidence", reason="recommendations_duplicated", confidence=0.35, details={"count": len(recommendations)}, suggested_action="fallback")
        return ObservationResult(status="success", reason="recommendations_available", confidence=0.8, details={"count": len(recommendations)})

    def _observe_update_preference_store(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del raw_output, runtime, state
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        if not str(payload.get("arxiv_id") or "").strip():
            return ObservationResult(status="need_clarification", reason="preference_target_missing", confidence=0.2, suggested_action="clarify_target")
        tool_result = payload.get("tool_result") if isinstance(payload.get("tool_result"), Mapping) else {}
        if tool_result and not bool(tool_result.get("ok", False)):
            return ObservationResult(status="tool_error", reason="preference_store_write_failed", confidence=0.1, details={"tool_error": tool_result.get("error")}, suggested_action="fallback")
        return ObservationResult(status="success", reason="preference_store_updated", confidence=0.9)

    def _observe_update_reading_list_store(self, *, resolved_input: Mapping[str, Any], raw_output: Any, normalized_output: Any, runtime: PlanRuntime, state: AgentState) -> ObservationResult:
        del resolved_input, raw_output, runtime, state
        payload = normalized_output if isinstance(normalized_output, Mapping) else {}
        if not bool(payload.get("ok")):
            return ObservationResult(status="need_clarification", reason="reading_list_action_invalid", confidence=0.2, suggested_action="clarify_target")
        return ObservationResult(status="success", reason="reading_list_updated", confidence=0.9)


__all__ = ["Observer"]
