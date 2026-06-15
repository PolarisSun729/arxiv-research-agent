from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from services.paper_qa.repair_actions import (
    ASK_CLARIFICATION,
    ASK_USER_TO_REBUILD_INDEX,
    RETRY_WITH_EXPANDED_CONTEXT,
    RETRY_WITH_KEYWORD_EMPHASIS,
    RETRY_WITH_QUERY_REWRITE,
    RETRY_WITH_SECTION_FOCUS,
    RETRY_WITHOUT_HYDE,
    describe_repair_actions,
    normalize_repair_actions,
)

QA_OBSERVATION_SCHEMA_VERSION = "paper_qa_observation_v1"

RETRIEVAL_STAGE_NAMES = (
    "query_planning",
    "rewrite",
    "hyde",
    "keyword",
    "vector",
    "memory",
    "fusion",
    "rerank",
    "context_expansion",
    "generation",
    "verification",
)

ROUTE_STAGE_NAMES = (
    "vector_original",
    "vector_rewrite",
    "vector_hyde",
    "keyword",
    "table_structured",
    "memory_context",
)

MIN_RETRIEVAL_SOURCE_COUNT = 2
MIN_FUSED_CHUNK_COUNT = 3

MISSING_EVIDENCE_BY_QUESTION_TYPE = {
    "method": "method_flow",
    "method_flow": "method_flow",
    "experiment": "experiment_setup",
    "experiment_setup": "experiment_setup",
    "dataset": "experiment_setup",
    "results_analysis": "result_table",
    "comparison": "result_table",
    "metric": "formula_derivation",
    "definition": "definition",
    "limitation": "limitation",
    "figure_table": "figure_explanation",
}


def build_qa_observation(
    *,
    retrieval_debug: Optional[Mapping[str, Any]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    verification_result: Optional[Mapping[str, Any]] = None,
    generation_result: Optional[Mapping[str, Any]] = None,
    error_code: Optional[str] = None,
    error_stage: Optional[str] = None,
    error_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """把 Paper QA 内部 debug 压缩成 Agent 可稳定消费的质量观察契约。"""
    debug = retrieval_debug if isinstance(retrieval_debug, Mapping) else {}
    normalized_sources = [dict(item) for item in (sources or []) if isinstance(item, Mapping)]
    verification = verification_result if isinstance(verification_result, Mapping) else {}
    generation = generation_result if isinstance(generation_result, Mapping) else {}

    stage_status = _build_stage_status(
        debug,
        sources=normalized_sources,
        verification_result=verification,
        generation_result=generation,
        error_stage=error_stage,
        error_reason=error_reason,
    )
    route_status = _build_route_status(debug)
    answer_insufficient = _answer_insufficient_evidence(verification, generation, error_code=error_code)
    weak_source_reason = _weak_source_reason(debug, normalized_sources, verification)
    missing_evidence_type = _missing_evidence_type(debug, answer_insufficient, weak_source_reason)
    rerank_failed_reason = _rerank_failed_reason(stage_status)
    answer_quality = _answer_quality(stage_status, answer_insufficient, error_code=error_code)
    answer_quality_reason = _answer_quality_reason(stage_status, answer_insufficient, error_code=error_code)
    retrieval_quality = _retrieval_quality(
        stage_status=stage_status,
        route_status=route_status,
        sources=normalized_sources,
        answer_insufficient=answer_insufficient,
        weak_source_reason=weak_source_reason,
        error_code=error_code,
    )
    retrieval_quality_reason = _retrieval_quality_reason(
        stage_status=stage_status,
        route_status=route_status,
        sources=normalized_sources,
        answer_insufficient=answer_insufficient,
        weak_source_reason=weak_source_reason,
        error_code=error_code,
    )
    answer_insufficient_reason = _answer_insufficient_evidence_reason(
        verification,
        generation,
        error_code=error_code,
    )
    repair_actions = _recommended_repair_actions(
        retrieval_quality=retrieval_quality,
        stage_status=stage_status,
        missing_evidence_type=missing_evidence_type,
        weak_source_reason=weak_source_reason,
        rerank_failed_reason=rerank_failed_reason,
        error_code=error_code,
    )

    return {
        "schema_version": QA_OBSERVATION_SCHEMA_VERSION,
        "retrieval_quality": retrieval_quality,
        "retrieval_quality_reason": retrieval_quality_reason,
        "answer_quality": answer_quality,
        "answer_quality_reason": answer_quality_reason,
        # quality_signals 是给 Agent 的紧凑入口；完整阶段细节仍保留在 retrieval_stage_status / route_status。
        "quality_signals": _build_quality_signals(
            stage_status=stage_status,
            route_status=route_status,
            retrieval_quality=retrieval_quality,
            retrieval_quality_reason=retrieval_quality_reason,
            answer_quality=answer_quality,
            answer_quality_reason=answer_quality_reason,
        ),
        "retrieval_stage_status": stage_status,
        "route_status": route_status,
        "degraded_stages": _degraded_stages(stage_status),
        "missing_evidence_type": missing_evidence_type,
        "weak_source_reason": weak_source_reason,
        "rerank_failed_reason": rerank_failed_reason,
        "answer_insufficient_evidence": answer_insufficient,
        "answer_insufficient_evidence_reason": answer_insufficient_reason,
        "recommended_repair_actions": repair_actions,
        "repair_action_details": describe_repair_actions(repair_actions),
        "source_count": len(normalized_sources),
        "error_code": error_code or "not_available",
        "error_stage": error_stage or "not_available",
        "observation_reason": error_reason or _observation_reason(retrieval_quality, weak_source_reason, answer_quality_reason),
    }


def build_error_qa_observation(
    *,
    error_code: str,
    error_stage: str,
    error_reason: Optional[str] = None,
    retrieval_debug: Optional[Mapping[str, Any]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """失败分支没有完整生成/校验结果时，仍返回同一形状的观察结构。"""
    return build_qa_observation(
        retrieval_debug=retrieval_debug,
        sources=sources,
        error_code=error_code,
        error_stage=error_stage,
        error_reason=error_reason,
    )


def _stage(
    enabled: Any = "unknown",
    status: str = "unknown",
    *,
    fallback: Any = "unknown",
    reason: str = "not_available",
    details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result = {
        "enabled": enabled,
        "status": status,
        "fallback": fallback,
        "reason": reason or "not_available",
    }
    if details is not None:
        result["details"] = dict(details)
    return result


def _build_stage_status(
    debug: Mapping[str, Any],
    *,
    sources: List[Dict[str, Any]],
    verification_result: Mapping[str, Any],
    generation_result: Mapping[str, Any],
    error_stage: Optional[str],
    error_reason: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    stages = {name: _stage() for name in RETRIEVAL_STAGE_NAMES}
    if not debug:
        stages["generation"] = _generation_stage(generation_result)
        stages["verification"] = _verification_stage(verification_result)
        if error_stage:
            _mark_error_stage(stages, error_stage, error_reason)
        return stages

    stages["query_planning"] = _query_planning_stage(debug)
    stages["rewrite"] = _rewrite_stage(debug.get("query_rewrite"))
    stages["hyde"] = _hyde_stage(debug.get("hyde"))
    stages["keyword"] = _metric_stage(_route_metric(debug, "keyword"))
    stages["vector"] = _vector_stage(debug)
    stages["memory"] = _memory_stage(debug)
    stages["fusion"] = _fusion_stage(debug)
    stages["rerank"] = _rerank_stage(debug.get("llm_rerank"))
    stages["context_expansion"] = _context_expansion_stage(debug)
    stages["generation"] = _generation_stage(generation_result)
    stages["verification"] = _verification_stage(verification_result)
    stages["fusion"] = _enrich_fusion_with_sources(stages["fusion"], debug, sources)
    if error_stage:
        _mark_error_stage(stages, error_stage, error_reason)
    return stages


def _mark_error_stage(stages: Dict[str, Dict[str, Any]], error_stage: str, error_reason: Optional[str]) -> None:
    normalized = str(error_stage or "").strip()
    stage_name = {
        "enhanced_retrieve": "vector",
        "vector_store": "vector",
        "build_qa_context": "vector",
        "paper_qa_final_answer": "generation",
        "qa_stream": "generation",
    }.get(normalized, normalized)
    if stage_name in stages:
        stages[stage_name] = _stage(True, "failed", fallback=False, reason=error_reason or normalized)


def _query_planning_stage(debug: Mapping[str, Any]) -> Dict[str, Any]:
    details = {
        "question_type": _question_type(debug) or "unknown",
        "preferred_sections": _preferred_sections(debug),
        "rewrite_query_count": _rewrite_query_count(debug),
        "contextualized_question_changed": _question_was_contextualized(debug),
    }
    if debug.get("query_profile") or debug.get("query_plan") or debug.get("intent_profile"):
        return _stage(True, "success", fallback=False, reason="query_profile_available", details=details)
    if debug.get("original_question") or debug.get("contextualized_question"):
        return _stage(True, "partial", fallback=False, reason="question_available_without_query_profile", details=details)
    return _stage(details=details)


def _rewrite_stage(value: Any) -> Dict[str, Any]:
    selected = _selected_rewrite_queries(value)
    details = {"rewrite_query_count": len(selected), "selected_queries": selected[:5]}
    if not isinstance(value, Mapping):
        if selected:
            return _stage(True, "success", fallback=False, reason="selected_queries_available", details=details)
        return _stage(details=details)
    enabled = bool(value.get("enabled", False))
    if not enabled:
        return _stage(False, "disabled", fallback=False, reason="disabled", details=details)
    if value.get("llm_error"):
        return _stage(True, "fallback", fallback=True, reason=str(value.get("llm_error")), details=details)
    if selected:
        return _stage(True, "success", fallback=False, reason="selected_queries_available", details=details)
    return _stage(True, "fallback", fallback=True, reason="selected_queries_empty", details=details)


def _hyde_stage(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return _stage()
    enabled = bool(value.get("enabled", False))
    details = {
        "hyde_text_generated": bool(str(value.get("text") or "").strip()),
        "source_query_count": len(value.get("source_queries") or []) if isinstance(value.get("source_queries"), list) else 0,
    }
    if not enabled:
        return _stage(False, "disabled", fallback=False, reason="disabled", details=details)
    if str(value.get("text") or "").strip():
        return _stage(True, "success", fallback=False, reason="hyde_text_available", details=details)
    return _stage(True, "fallback", fallback=True, reason="hyde_text_empty", details=details)


def _route_metric(debug: Mapping[str, Any], key: str) -> Any:
    route_metrics = debug.get("route_metrics")
    if isinstance(route_metrics, Mapping):
        return route_metrics.get(key)
    return None


def _metric_stage(metric: Any) -> Dict[str, Any]:
    if not isinstance(metric, Mapping):
        return _stage()
    enabled = bool(metric.get("enabled", False))
    status = str(metric.get("status") or "").strip().lower()
    candidate_count = _safe_int(metric.get("candidate_count"), default=0)
    applied = bool(metric.get("applied", False))
    reason = str(metric.get("fallback_reason") or metric.get("error") or status or "not_available")
    details = {
        "candidate_count": candidate_count,
        "applied": applied,
        "latency_ms": metric.get("latency_ms"),
        "cache_hit": metric.get("cache_hit"),
    }
    if not enabled or status == "disabled":
        return _stage(False, "disabled", fallback=False, reason=reason or "disabled", details=details)
    if (status == "ok" or applied) and candidate_count > 0:
        return _stage(True, "success", fallback=False, reason=f"candidate_count:{candidate_count}", details=details)
    if status == "ok" or applied:
        return _stage(True, "partial", fallback=False, reason="candidate_empty", details=details)
    if status in {"timeout", "partial"} or bool(metric.get("timeout", False)):
        return _stage(True, "fallback", fallback=True, reason=reason, details=details)
    if status in {"error", "failed"}:
        return _stage(True, "failed", fallback=False, reason=reason, details=details)
    return _stage(True, "unknown", fallback="unknown", reason=reason, details=details)


def _vector_stage(debug: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = [_route_metric(debug, name) for name in ("vector_original", "vector_rewrite", "vector_hyde")]
    known = [metric for metric in metrics if isinstance(metric, Mapping)]
    details = {
        "route_count": len(known),
        "candidate_count": sum(_safe_int(metric.get("candidate_count"), default=0) for metric in known),
        "successful_routes": [
            name
            for name in ("vector_original", "vector_rewrite", "vector_hyde")
            if _route_has_candidates(_route_metric(debug, name))
        ],
    }
    if not known:
        return _stage(details=details)
    if any(_route_has_candidates(metric) for metric in known):
        fallback = any(str(metric.get("status") or "").lower() in {"partial", "timeout", "error", "failed"} for metric in known)
        return _stage(True, "partial" if fallback else "success", fallback=fallback, reason="vector_candidates_available", details=details)
    if all(str(metric.get("status") or "").lower() == "disabled" for metric in known):
        return _stage(False, "disabled", fallback=False, reason="disabled", details=details)
    return _stage(True, "failed", fallback=False, reason="vector_candidates_empty", details=details)


def _memory_stage(debug: Mapping[str, Any]) -> Dict[str, Any]:
    metric = _route_metric(debug, "memory_context")
    if isinstance(metric, Mapping):
        return _metric_stage(metric)

    memory_debug = debug.get("memory")
    if not isinstance(memory_debug, Mapping):
        return _stage()
    enabled = bool(memory_debug.get("enabled", False))
    candidate_count = _safe_int(memory_debug.get("candidate_count"), default=0)
    details = {
        "candidate_count": candidate_count,
        "final_context_hit_count": len(memory_debug.get("final_context_hits") or []) if isinstance(memory_debug.get("final_context_hits"), list) else 0,
        "lookup_mode": memory_debug.get("memory_lookup_mode"),
    }
    if not enabled:
        return _stage(False, "disabled", fallback=False, reason=str(memory_debug.get("fallback_reason") or "disabled"), details=details)
    if bool(memory_debug.get("applied", False)) or candidate_count > 0:
        return _stage(True, "success", fallback=False, reason=f"candidate_count:{candidate_count}", details=details)
    return _stage(True, "partial", fallback=False, reason=str(memory_debug.get("fallback_reason") or "candidate_empty"), details=details)


def _fusion_stage(debug: Mapping[str, Any]) -> Dict[str, Any]:
    fusion = debug.get("fusion")
    stages = debug.get("stages") if isinstance(debug.get("stages"), Mapping) else {}
    fused = stages.get("fused_top30") if isinstance(stages, Mapping) else None
    final_chunks = debug.get("final_chunks")
    fused_count = len(fused) if isinstance(fused, list) else 0
    final_count = len(final_chunks) if isinstance(final_chunks, list) else 0
    details = {
        "fused_count": fused_count,
        "final_chunk_count": final_count,
    }
    if isinstance(fused, list) and fused:
        if fused_count < MIN_FUSED_CHUNK_COUNT:
            return _stage(True, "partial", fallback=False, reason=f"fused_candidates_too_few:{fused_count}", details=details)
        return _stage(True, "success", fallback=False, reason="fused_candidates_available", details=details)
    if isinstance(fusion, Mapping):
        return _stage(True, "partial", fallback=False, reason="fusion_debug_available_without_fused_candidates", details=details)
    return _stage(details=details)


def _enrich_fusion_with_sources(stage: Dict[str, Any], debug: Mapping[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    details = dict(stage.get("details") or {})
    details.update(_section_distribution_details(debug, sources))
    if not sources:
        return _stage(True, "failed", fallback=False, reason="final_chunks_empty", details=details)
    if len(sources) < MIN_RETRIEVAL_SOURCE_COUNT:
        return _stage(True, "partial", fallback=False, reason=f"top_chunks_too_few:{len(sources)}", details=details)
    preferred_sections = details.get("preferred_sections") or []
    if preferred_sections and not details.get("matches_preferred_section"):
        return _stage(True, "partial", fallback=False, reason="preferred_section_mismatch", details=details)
    if float(details.get("noisy_section_ratio") or 0.0) >= 0.5:
        return _stage(True, "partial", fallback=False, reason="context_noise_contamination", details=details)
    return _stage(stage.get("enabled", True), stage.get("status", "success"), fallback=stage.get("fallback", False), reason=stage.get("reason", "ok"), details=details)


def _rerank_stage(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return _stage()
    enabled = bool(value.get("enabled", False))
    reason = str(value.get("reason") or value.get("fallback_reason") or "not_available")
    details = {
        "executed": bool(value.get("applied", False)),
        "mode": value.get("mode"),
        "input_chunks": _safe_int(value.get("input_chunks"), default=0),
        "output_chunks": _safe_int(value.get("output_chunks"), default=0),
        "candidate_limit": value.get("candidate_limit"),
    }
    if not enabled:
        return _stage(False, "disabled", fallback=False, reason=reason or "disabled", details=details)
    if bool(value.get("applied", False)):
        return _stage(True, "success", fallback=False, reason="ok", details=details)
    mode = str(value.get("mode") or "").strip().lower()
    if mode in {"fallback_fused", "passthrough", "remote_failed"} or reason not in {"", "disabled", "not_available"}:
        return _stage(True, "fallback", fallback=True, reason=reason, details=details)
    return _stage(True, "unknown", fallback="unknown", reason=reason, details=details)


def _context_expansion_stage(debug: Mapping[str, Any]) -> Dict[str, Any]:
    context_expansion = debug.get("context_expansion") if isinstance(debug.get("context_expansion"), Mapping) else {}
    context_budget = debug.get("context_budget")
    if not isinstance(context_budget, Mapping):
        context_pack = debug.get("context_pack") if isinstance(debug.get("context_pack"), Mapping) else {}
        context_budget = context_pack.get("context_budget_debug") if isinstance(context_pack, Mapping) else None
    details = {
        "anchor_count": len(context_expansion.get("anchors") or []) if isinstance(context_expansion, Mapping) else 0,
        "expansion_candidate_count": len(context_expansion.get("candidate_pool") or []) if isinstance(context_expansion, Mapping) else 0,
    }
    if not isinstance(context_budget, Mapping):
        return _stage(details=details)
    # 兼容两类 debug：检索层 context_budget 有 enabled/applied，PaperQA context_pack 只有最终预算统计。
    has_expansion_budget_flags = "enabled" in context_budget or "applied" in context_budget or "fallback_reason" in context_budget
    enabled = bool(context_budget.get("enabled", False)) if has_expansion_budget_flags else bool(context_budget)
    used_chars = _safe_int(context_budget.get("used_context_chars") or context_budget.get("text_context_chars"), default=0)
    max_chars = _safe_int(context_budget.get("max_context_chars"), default=0)
    details.update(
        {
            "applied": bool(context_budget.get("applied", False)),
            "used_context_chars": used_chars,
            "max_context_chars": max_chars,
            "within_budget": True if not max_chars else used_chars <= max_chars,
            "final_context_count": len(context_budget.get("final_context_after_budget") or []) if isinstance(context_budget.get("final_context_after_budget"), list) else context_budget.get("source_count"),
        }
    )
    if not has_expansion_budget_flags:
        if details["within_budget"]:
            return _stage(True, "success", fallback=False, reason="context_pack_within_budget", details=details)
        return _stage(True, "partial", fallback=False, reason="context_pack_budget_exceeded", details=details)
    if not enabled:
        return _stage(False, "disabled", fallback=False, reason=str(context_budget.get("fallback_reason") or "disabled"), details=details)
    if bool(context_budget.get("applied", False)):
        if details["within_budget"]:
            return _stage(True, "success", fallback=False, reason="context_budget_applied", details=details)
        return _stage(True, "partial", fallback=False, reason="context_budget_exceeded", details=details)
    return _stage(True, "fallback", fallback=True, reason=str(context_budget.get("fallback_reason") or "context_expansion_not_applied"), details=details)


def _generation_stage(generation: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(generation, Mapping) or not generation:
        return _stage()
    answer = str(generation.get("answer") or "").strip()
    generation_debug = generation.get("generation_debug") if isinstance(generation.get("generation_debug"), Mapping) else {}
    details = {
        "answer_generated": bool(answer),
        "answer_chars": len(answer),
        "insufficient_evidence": bool(generation.get("insufficient_evidence", False)),
        "search_result_count": generation_debug.get("search_result_count"),
        "image_input_count": generation_debug.get("image_input_count"),
    }
    if not answer:
        return _stage(True, "failed", fallback=False, reason="answer_empty", details=details)
    if bool(generation.get("insufficient_evidence", False)):
        return _stage(True, "insufficient_evidence", fallback=False, reason="generation_declared_insufficient_evidence", details=details)
    return _stage(True, "success", fallback=False, reason="answer_generated", details=details)


def _verification_stage(verification: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(verification, Mapping) or not verification:
        return _stage()
    status = str(verification.get("status") or "unknown").strip() or "unknown"
    details = {
        "status": status,
        "source_count": _safe_int(verification.get("source_count"), default=0),
        "citation_count": len(verification.get("cited_source_ids") or []) if isinstance(verification.get("cited_source_ids"), list) else 0,
        "missing_citation_count": len(verification.get("missing_cited_source_ids") or []) if isinstance(verification.get("missing_cited_source_ids"), list) else 0,
        "unsupported_claim_count": len(verification.get("unsupported_claims") or []) if isinstance(verification.get("unsupported_claims"), list) else 0,
        "guardrail_triggered": bool(verification.get("insufficient_evidence", False)),
        "warnings": list(verification.get("warnings") or []) if isinstance(verification.get("warnings"), list) else [],
    }
    if bool(verification.get("insufficient_evidence", False)):
        return _stage(True, "insufficient_evidence", fallback=False, reason="verification_insufficient_evidence", details=details)
    if status == "passed":
        return _stage(True, "success", fallback=False, reason="verification_passed", details=details)
    if status == "warning":
        return _stage(True, "partial", fallback=False, reason="verification_warning", details=details)
    return _stage(True, status, fallback=False, reason=status or "verification_unknown", details=details)


def _answer_insufficient_evidence(verification: Mapping[str, Any], generation: Mapping[str, Any], *, error_code: Optional[str]) -> str:
    if bool(generation.get("insufficient_evidence", False)) or bool(verification.get("insufficient_evidence", False)):
        return "yes"
    if verification:
        return "no"
    if error_code:
        return "unknown"
    return "unknown"


def _answer_quality(stage_status: Mapping[str, Mapping[str, Any]], answer_insufficient: str, *, error_code: Optional[str]) -> str:
    if error_code == "llm_generation_failed":
        return "generation_failed"
    generation = stage_status.get("generation") if isinstance(stage_status, Mapping) else {}
    verification = stage_status.get("verification") if isinstance(stage_status, Mapping) else {}
    if str(generation.get("status") or "") == "failed":
        return "generation_failed"
    if answer_insufficient == "yes" or str(verification.get("status") or "") == "insufficient_evidence":
        return "insufficient_evidence"
    if str(verification.get("status") or "") in {"success", "partial"}:
        return "grounded" if str(verification.get("status")) == "success" else "warning"
    return "unknown"


def _answer_quality_reason(stage_status: Mapping[str, Mapping[str, Any]], answer_insufficient: str, *, error_code: Optional[str]) -> str:
    """给顶层 answer_quality 补充稳定 reason，避免 Agent 反查阶段 map。"""
    if error_code == "llm_generation_failed":
        return "llm_generation_failed"
    generation = stage_status.get("generation") if isinstance(stage_status, Mapping) else {}
    verification = stage_status.get("verification") if isinstance(stage_status, Mapping) else {}
    if str(generation.get("status") or "") == "failed":
        return str(generation.get("reason") or "answer_empty")
    if answer_insufficient == "yes":
        return str(verification.get("reason") or generation.get("reason") or "insufficient_evidence")
    verification_status = str(verification.get("status") or "")
    if verification_status in {"success", "partial"}:
        return str(verification.get("reason") or verification_status)
    return "verification_not_available"


def _answer_insufficient_evidence_reason(verification: Mapping[str, Any], generation: Mapping[str, Any], *, error_code: Optional[str]) -> str:
    if bool(generation.get("insufficient_evidence", False)):
        return "generation_declared_insufficient_evidence"
    if bool(verification.get("insufficient_evidence", False)):
        return "verification_guardrail_triggered"
    if verification:
        return "verification_checked_no_insufficient_evidence"
    if error_code:
        return f"unknown_due_to_error:{error_code}"
    return "verification_not_available"


def _build_quality_signals(
    *,
    stage_status: Mapping[str, Mapping[str, Any]],
    route_status: Mapping[str, Mapping[str, Any]],
    retrieval_quality: str,
    retrieval_quality_reason: str,
    answer_quality: str,
    answer_quality_reason: str,
) -> Dict[str, Dict[str, Any]]:
    """把散落阶段状态收束为 Agent 易消费的七段质量信号。"""
    route_candidate_counts = {
        route_name: (route.get("details") or {}).get("candidate_count")
        for route_name, route in (route_status or {}).items()
        if isinstance(route, Mapping)
    }
    route_successes = [
        route_name
        for route_name, route in (route_status or {}).items()
        if isinstance(route, Mapping) and str(route.get("status") or "") == "success"
    ]
    signals = {
        "query_planning": _quality_signal_from_stage(stage_status.get("query_planning")),
        "retrieval": _stage(
            True,
            retrieval_quality,
            fallback=False,
            reason=retrieval_quality_reason,
            details={
                "successful_routes": route_successes,
                "route_candidate_counts": route_candidate_counts,
            },
        ),
        "fusion": _quality_signal_from_stage(stage_status.get("fusion")),
        "rerank": _quality_signal_from_stage(stage_status.get("rerank")),
        "context_expansion": _quality_signal_from_stage(stage_status.get("context_expansion")),
        "generation": _quality_signal_from_stage(stage_status.get("generation")),
        "verification": _quality_signal_from_stage(stage_status.get("verification")),
        "answer": _stage(True, answer_quality, fallback=False, reason=answer_quality_reason),
    }
    return signals


def _quality_signal_from_stage(stage: Any) -> Dict[str, Any]:
    if isinstance(stage, Mapping):
        # 复制一份，避免调用方误改 retrieval_stage_status 的完整阶段对象。
        return dict(stage)
    return _stage()


def _build_route_status(debug: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """按 route 记录候选是否真正产出，避免 Agent 只看到整体 retrieval_quality。"""
    routes = debug.get("routes") if isinstance(debug.get("routes"), Mapping) else {}
    route_status: Dict[str, Dict[str, Any]] = {}
    for route_name in ROUTE_STAGE_NAMES:
        metric = _route_metric(debug, route_name)
        stage = _metric_stage(metric)
        details = dict(stage.get("details") or {})
        route_results = routes.get(route_name) if isinstance(routes, Mapping) else None
        if isinstance(route_results, list):
            details["debug_candidate_count"] = len(route_results)
            if stage.get("status") == "unknown" and route_results:
                stage = _stage(True, "success", fallback=False, reason=f"debug_candidates:{len(route_results)}", details=details)
            else:
                stage["details"] = details
        route_status[route_name] = stage
    return route_status


def _degraded_stages(stage_status: Mapping[str, Mapping[str, Any]]) -> List[str]:
    degraded: List[str] = []
    for stage_name, stage in (stage_status or {}).items():
        status = str(stage.get("status") or "")
        if status in {"partial", "weak", "fallback", "failed", "insufficient_evidence"}:
            degraded.append(stage_name)
    return degraded


def _weak_source_reason(debug: Mapping[str, Any], sources: List[Dict[str, Any]], verification: Mapping[str, Any]) -> str:
    if not sources:
        return "citation_source_insufficient"
    if len(sources) < 2:
        return "citation_source_insufficient"
    if bool(verification.get("missing_cited_source_ids")):
        return "citation_source_insufficient"
    if _average_source_text_length(sources) < 80:
        return "evidence_too_short"
    preferred_sections = _preferred_sections(debug)
    if preferred_sections and not _sources_match_sections(sources, preferred_sections):
        return "section_mismatch"
    if _context_noise_ratio(sources) >= 0.5:
        return "context_noise_contamination"
    return "not_available"


def _average_source_text_length(sources: List[Dict[str, Any]]) -> float:
    lengths = []
    for source in sources:
        text = str(source.get("content") or source.get("text") or source.get("asset_summary") or "").strip()
        if text:
            lengths.append(len(text))
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def _preferred_sections(debug: Mapping[str, Any]) -> List[str]:
    query_profile = debug.get("query_profile")
    if isinstance(query_profile, Mapping):
        sections = query_profile.get("section_preferences")
        if isinstance(sections, list):
            return [str(item).strip().lower() for item in sections if str(item).strip()]
    intent = debug.get("intent_profile") or debug.get("intent")
    if isinstance(intent, Mapping):
        sections = intent.get("preferred_sections")
        if isinstance(sections, list):
            return [str(item).strip().lower() for item in sections if str(item).strip()]
    return []


def _rewrite_query_count(debug: Mapping[str, Any]) -> int:
    return len(_selected_rewrite_queries(debug.get("query_rewrite") or debug.get("rewritten_queries")))


def _selected_rewrite_queries(value: Any) -> List[str]:
    if isinstance(value, Mapping):
        selected = value.get("selected_queries")
        if not selected:
            selected = value.get("model_queries")
    else:
        selected = value
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        return []
    return [str(item).strip() for item in selected if str(item).strip()]


def _question_was_contextualized(debug: Mapping[str, Any]) -> bool:
    original = str(debug.get("original_question") or debug.get("original_query") or "").strip()
    contextualized = str(debug.get("contextualized_question") or "").strip()
    return bool(original and contextualized and original != contextualized)


def _section_distribution_details(debug: Mapping[str, Any], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总 top chunks 的 section 分布，给 fusion 质量判断提供可解释依据。"""
    preferred_sections = _preferred_sections(debug)
    section_labels = [_source_section_label(source) for source in sources]
    section_counts = Counter(label for label in section_labels if label)
    source_count = len(sources)
    dominant_section, dominant_count = section_counts.most_common(1)[0] if section_counts else ("not_available", 0)
    noisy_count = sum(
        count
        for section, count in section_counts.items()
        if any(marker in section for marker in ("references", "bibliography", "appendix", "prompt"))
    )
    matched_sections = [
        section
        for section in section_counts
        if any(preferred in section for preferred in preferred_sections)
    ]
    return {
        "preferred_sections": preferred_sections,
        "section_counts": dict(section_counts),
        "section_count": len(section_counts),
        "dominant_section": dominant_section,
        "dominant_section_ratio": round(dominant_count / max(1, source_count), 4),
        "noisy_section_ratio": round(noisy_count / max(1, source_count), 4),
        "matches_preferred_section": True if not preferred_sections else bool(matched_sections),
        "matched_preferred_sections": matched_sections,
    }


def _source_section_label(source: Mapping[str, Any]) -> str:
    section = str(source.get("section_path") or source.get("section_title") or "").strip().lower()
    if section:
        return section
    chunk_type = str(source.get("chunk_type") or source.get("asset_kind") or "").strip().lower()
    return chunk_type or "unknown"


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _route_has_candidates(metric: Any) -> bool:
    if not isinstance(metric, Mapping):
        return False
    if _safe_int(metric.get("candidate_count"), default=0) > 0:
        return True
    return bool(metric.get("applied", False)) and str(metric.get("status") or "").lower() == "ok"


def _sources_match_sections(sources: List[Dict[str, Any]], preferred_sections: List[str]) -> bool:
    for source in sources:
        haystack = " ".join(
            str(source.get(key) or "").lower()
            for key in ("section_path", "section_title", "chunk_type", "asset_kind")
        )
        if any(section in haystack for section in preferred_sections):
            return True
    return False


def _context_noise_ratio(sources: List[Dict[str, Any]]) -> float:
    noisy = 0
    for source in sources:
        haystack = " ".join(str(source.get(key) or "").lower() for key in ("section_path", "section_title", "chunk_type"))
        if any(marker in haystack for marker in ("references", "bibliography", "appendix", "prompt")):
            noisy += 1
    return noisy / max(1, len(sources))


def _missing_evidence_type(debug: Mapping[str, Any], answer_insufficient: str, weak_source_reason: str) -> str:
    if answer_insufficient != "yes" and weak_source_reason == "not_available":
        return "not_available"
    question_type = _question_type(debug)
    if question_type in MISSING_EVIDENCE_BY_QUESTION_TYPE:
        return MISSING_EVIDENCE_BY_QUESTION_TYPE[question_type]
    if weak_source_reason == "section_mismatch":
        return "cross_paragraph_evidence"
    return "unknown"


def _question_type(debug: Mapping[str, Any]) -> str:
    query_profile = debug.get("query_profile")
    if isinstance(query_profile, Mapping):
        value = str(query_profile.get("question_type") or "").strip().lower()
        if value:
            return value
    intent = debug.get("intent_profile") or debug.get("intent")
    if isinstance(intent, Mapping):
        value = str(intent.get("main_intent") or intent.get("intent") or "").strip().lower()
        if value:
            return value
    return ""


def _rerank_failed_reason(stage_status: Mapping[str, Mapping[str, Any]]) -> str:
    rerank = stage_status.get("rerank") if isinstance(stage_status, Mapping) else None
    if not isinstance(rerank, Mapping):
        return "unknown"
    status = str(rerank.get("status") or "").strip()
    if status in {"failed", "fallback", "disabled"}:
        return str(rerank.get("reason") or status or "unknown")
    return "not_available"


def _retrieval_quality(
    *,
    stage_status: Mapping[str, Mapping[str, Any]],
    route_status: Mapping[str, Mapping[str, Any]],
    sources: List[Dict[str, Any]],
    answer_insufficient: str,
    weak_source_reason: str,
    error_code: Optional[str],
) -> str:
    if error_code:
        return "failed"
    if not sources:
        return "failed"
    if len(sources) < MIN_RETRIEVAL_SOURCE_COUNT:
        return "weak"
    if answer_insufficient == "yes":
        return "weak"
    if weak_source_reason != "not_available":
        return "partial"
    if any(str(stage.get("status") or "") == "failed" for stage in stage_status.values()):
        return "partial"
    if any(str(route.get("status") or "") == "failed" for route in route_status.values()):
        return "partial"
    if any(str(stage.get("status") or "") == "fallback" for stage in stage_status.values()):
        return "partial"
    if any(str(route.get("status") or "") == "fallback" for route in route_status.values()):
        return "partial"
    return "good"


def _retrieval_quality_reason(
    *,
    stage_status: Mapping[str, Mapping[str, Any]],
    route_status: Mapping[str, Mapping[str, Any]],
    sources: List[Dict[str, Any]],
    answer_insufficient: str,
    weak_source_reason: str,
    error_code: Optional[str],
) -> str:
    """和 retrieval_quality 使用同一组规则，返回可复盘的首要原因。"""
    if error_code:
        return f"error_code:{error_code}"
    if not sources:
        return "source_chunks_empty"
    if len(sources) < MIN_RETRIEVAL_SOURCE_COUNT:
        return f"source_chunks_too_few:{len(sources)}"
    if answer_insufficient == "yes":
        return "answer_insufficient_evidence"
    if weak_source_reason != "not_available":
        return weak_source_reason
    failed_stage = _first_stage_with_status(stage_status, "failed")
    if failed_stage:
        return f"stage_failed:{failed_stage}"
    failed_route = _first_stage_with_status(route_status, "failed")
    if failed_route:
        return f"route_failed:{failed_route}"
    fallback_stage = _first_stage_with_status(stage_status, "fallback")
    if fallback_stage:
        return f"stage_fallback:{fallback_stage}"
    fallback_route = _first_stage_with_status(route_status, "fallback")
    if fallback_route:
        return f"route_fallback:{fallback_route}"
    return "retrieval_and_evidence_passed"


def _first_stage_with_status(items: Mapping[str, Mapping[str, Any]], status: str) -> str:
    for name, value in (items or {}).items():
        if isinstance(value, Mapping) and str(value.get("status") or "") == status:
            return str(name)
    return ""


def _recommended_repair_actions(
    *,
    retrieval_quality: str,
    stage_status: Mapping[str, Mapping[str, Any]],
    missing_evidence_type: str,
    weak_source_reason: str,
    rerank_failed_reason: str,
    error_code: Optional[str],
) -> List[str]:
    actions: List[str] = []
    if error_code == "qa_index_not_found":
        actions.append(ASK_USER_TO_REBUILD_INDEX)
    if retrieval_quality in {"failed", "weak", "partial"}:
        actions.append(RETRY_WITH_QUERY_REWRITE)
    if missing_evidence_type in {"method_flow", "experiment_setup", "result_table", "cross_paragraph_evidence"} or weak_source_reason in {"citation_source_insufficient", "evidence_too_short"}:
        actions.append(RETRY_WITH_EXPANDED_CONTEXT)
    if missing_evidence_type == "method_flow" or weak_source_reason == "section_mismatch":
        actions.append(RETRY_WITH_SECTION_FOCUS)
    if missing_evidence_type in {"formula_derivation", "result_table"} or _keyword_emphasis_needed(stage_status):
        actions.append(RETRY_WITH_KEYWORD_EMPHASIS)
    if stage_status.get("hyde", {}).get("status") == "fallback" or (stage_status.get("hyde", {}).get("status") == "success" and retrieval_quality in {"weak", "partial"}):
        actions.append(RETRY_WITHOUT_HYDE)
    if rerank_failed_reason not in {"not_available", "disabled", "unknown"}:
        actions.append(RETRY_WITH_QUERY_REWRITE)
    if missing_evidence_type == "unknown" and retrieval_quality in {"failed", "weak"}:
        actions.append(ASK_CLARIFICATION)
    if not actions and retrieval_quality == "good":
        return []
    return normalize_repair_actions(_dedupe(actions))


def _keyword_emphasis_needed(stage_status: Mapping[str, Mapping[str, Any]]) -> bool:
    query_details = (stage_status.get("query_planning") or {}).get("details") if isinstance(stage_status, Mapping) else {}
    question_type = str((query_details or {}).get("question_type") or "").strip().lower() if isinstance(query_details, Mapping) else ""
    if question_type in {"metric", "formula", "figure_table", "comparison", "results_analysis"}:
        return True
    preferred_sections = (query_details or {}).get("preferred_sections") if isinstance(query_details, Mapping) else []
    return any(str(section).lower() in {"results", "experiments", "tables", "figures"} for section in (preferred_sections or []))


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _observation_reason(retrieval_quality: str, weak_source_reason: str, answer_quality_reason: str) -> str:
    if retrieval_quality == "good":
        return answer_quality_reason or "retrieval_and_evidence_passed"
    if weak_source_reason != "not_available":
        return weak_source_reason
    return f"retrieval_quality_{retrieval_quality}"
