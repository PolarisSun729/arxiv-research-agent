"""Paper QA 质量观察契约（qa_observation）。

把一次问答的内部状态压缩成形状固定的字典，供 Agent 恢复策略、前端与评估记录稳定消费：
retrieval_quality / answer_quality / 各阶段状态 / 推荐修复动作等。

生产调用方：
- build_research_qa_observation：研究图（PaperEvidenceResearchService）成功收尾时，按 outcome 投影；
- build_error_qa_observation：paper_qa_service、routers/qa_router.py、tools/paper_qa_tools.py 的失败分支；
- research_outcome_from_result：Agent 侧读取结果里的研究终态。

历史上这里还会解析旧检索链路的 retrieval_debug（rewrite / hyde / vector / memory / fusion / rerank /
context_expansion 等阶段），但研究图上线后已没有调用方传入该 debug，相关解析逻辑已移除。
为保证前端与 Agent 读取的字段形状不变，retrieval_stage_status / route_status 仍保留全部阶段键，
没有观测数据的阶段统一为 unknown。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from services.paper_qa.repair_actions import (
    ASK_CLARIFICATION,
    ASK_USER_TO_REBUILD_INDEX,
    RETRY_WITH_EXPANDED_CONTEXT,
    RETRY_WITH_QUERY_REWRITE,
    describe_repair_actions,
    normalize_repair_actions,
)

QA_OBSERVATION_SCHEMA_VERSION = "paper_qa_observation_v1"

# retrieval_stage_status 中固定存在的阶段键（保证输出形状稳定，缺数据时为 unknown）。
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

# route_status 中固定存在的召回路由键。
ROUTE_STAGE_NAMES = (
    "vector_original",
    "vector_rewrite",
    "vector_hyde",
    "keyword",
    "table_structured",
    "memory_context",
)

# 最终来源少于该数量视为证据偏弱。
MIN_RETRIEVAL_SOURCE_COUNT = 2
# 来源平均文本长度低于该值视为证据过短。
MIN_AVERAGE_SOURCE_TEXT_LENGTH = 80
# 引用来源中参考文献 / 附录 / prompt 等噪声章节占比达到该值，视为上下文被污染。
NOISY_SOURCE_RATIO_THRESHOLD = 0.5
NOISY_SECTION_MARKERS = ("references", "bibliography", "appendix", "prompt")

# 调用方错误阶段名 → 观察契约阶段键。
# 前四项是旧检索链路的阶段名（仍可能出现在历史调用里）；后三项是研究图结构化异常实际使用的阶段名。
# prepare_research / research / persist_completed_turn 等没有对应检索阶段，只体现在顶层 error_stage。
ERROR_STAGE_ALIASES = {
    "enhanced_retrieve": "vector",
    "vector_store": "vector",
    "paper_qa_final_answer": "generation",
    "qa_stream": "generation",
    "draft_generation": "generation",
    "claim_extraction": "verification",
    "claim_verification": "verification",
}


def build_qa_observation(
    *,
    sources: Optional[List[Dict[str, Any]]] = None,
    verification_result: Optional[Mapping[str, Any]] = None,
    generation_result: Optional[Mapping[str, Any]] = None,
    error_code: Optional[str] = None,
    error_stage: Optional[str] = None,
    error_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """把 Paper QA 的来源、生成、校验与错误信息压缩成 Agent 可稳定消费的质量观察契约。

    所有入参都可缺省；缺失的部分对应字段为 unknown / not_available，输出键集合始终不变。
    error_reason 若提供会直接作为 observation_reason 对外暴露，调用方必须传稳定的错误码类字符串，
    不能传 str(exc)（可能包含 provider 响应、服务器路径等敏感信息）。
    """
    normalized_sources = [dict(item) for item in (sources or []) if isinstance(item, Mapping)]
    verification = verification_result if isinstance(verification_result, Mapping) else {}
    generation = generation_result if isinstance(generation_result, Mapping) else {}

    stage_status = _build_stage_status(
        verification_result=verification,
        generation_result=generation,
        error_stage=error_stage,
        error_reason=error_reason,
    )
    route_status = {route_name: _stage() for route_name in ROUTE_STAGE_NAMES}
    answer_insufficient = _answer_insufficient_evidence(verification, generation)
    weak_source_reason = _weak_source_reason(normalized_sources, verification)
    missing_evidence_type = _missing_evidence_type(answer_insufficient, weak_source_reason)
    rerank_failed_reason = _rerank_failed_reason(stage_status)
    answer_quality = _answer_quality(stage_status, answer_insufficient, error_code=error_code)
    answer_quality_reason = _answer_quality_reason(stage_status, answer_insufficient, error_code=error_code)
    retrieval_quality, retrieval_quality_reason = _retrieval_quality(
        stage_status=stage_status,
        route_status=route_status,
        sources=normalized_sources,
        answer_insufficient=answer_insufficient,
        weak_source_reason=weak_source_reason,
        error_code=error_code,
    )
    repair_actions = _recommended_repair_actions(
        retrieval_quality=retrieval_quality,
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
        "answer_insufficient_evidence_reason": _answer_insufficient_evidence_reason(
            verification,
            generation,
            error_code=error_code,
        ),
        "recommended_repair_actions": repair_actions,
        "repair_action_details": describe_repair_actions(repair_actions),
        "source_count": len(normalized_sources),
        "error_code": error_code or "not_available",
        "error_stage": error_stage or "not_available",
        "observation_reason": error_reason or _observation_reason(retrieval_quality, weak_source_reason, answer_quality_reason),
    }


def research_outcome_from_result(result: Mapping[str, Any]) -> str | None:
    """新图的业务终态优先；执行失败或旧结果仍走原质量判断，不能信任残留 outcome。"""
    if result.get("status") in {"failed", "error"} or result.get("run_status") == "error":
        return None
    observation = result.get("qa_observation")
    outcome = result.get("outcome") or (observation.get("outcome") if isinstance(observation, Mapping) else None)
    return outcome if outcome in {"completed", "partial", "abstained"} else None


def build_research_qa_observation(
    *, outcome: str, research_summary: Mapping[str, Any], sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """把研究终态投影到原观察契约；有界研究已自行修复，外层不能再按来源数量重试。"""
    qualities = {"completed": "grounded", "partial": "warning", "abstained": "insufficient_evidence"}
    if outcome not in qualities:
        raise ValueError("未知研究终态")
    reason = f"research_{outcome}:{research_summary.get('termination_reason', 'unknown')}"
    observation = build_qa_observation(sources=sources)
    stages = observation["retrieval_stage_status"]
    # 未逐轮观测的旧检索阶段保留 unknown，不把新图的通过状态伪装成每条旧路由都成功。
    stages["generation"] = _stage(True, "completed" if outcome != "abstained" else "skipped", fallback=False, reason=reason)
    # 与 _verification_stage 保持同一套取值：通过统一用 success。
    stages["verification"] = _stage(True, "success" if sources else "not_available", fallback=False, reason=reason)
    stages["evidence_coverage"] = _stage(True, outcome, fallback=False, reason=reason, details=research_summary)
    observation.update({
        "outcome": outcome, "termination_reason": research_summary.get("termination_reason"),
        "research_summary": dict(research_summary), "answer_quality": qualities[outcome],
        "answer_quality_reason": reason,
        "retrieval_quality": {"completed": "sufficient", "partial": "partial", "abstained": "weak"}[outcome],
        "retrieval_quality_reason": reason,
        "answer_insufficient_evidence": "yes" if outcome == "abstained" else "no",
        "answer_insufficient_evidence_reason": reason if outcome == "abstained" else "not_available",
        "degraded_stages": ["evidence_coverage"] if outcome == "partial" else [],
        "weak_source_reason": "unresolved_core_needs" if outcome != "completed" else "not_available",
        "missing_evidence_type": "unresolved_core_needs" if outcome != "completed" else "not_available",
        "recommended_repair_actions": [], "repair_action_details": [], "observation_reason": reason,
    })
    observation["quality_signals"] = _build_quality_signals(
        stage_status=stages, route_status=observation["route_status"],
        retrieval_quality=observation["retrieval_quality"], retrieval_quality_reason=reason,
        answer_quality=qualities[outcome], answer_quality_reason=reason,
    )
    return observation


def build_error_qa_observation(
    *,
    error_code: str,
    error_stage: str,
    error_reason: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """失败分支没有完整生成/校验结果时，仍返回同一形状的观察结构。

    error_stage 会经 ERROR_STAGE_ALIASES 映射到具体阶段并标记为 failed；
    映射不到的阶段名只体现在顶层 error_stage 字段中。
    """
    return build_qa_observation(
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
    """构造统一形状的阶段状态：enabled / status / fallback / reason（+ 可选 details）。"""
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
    *,
    verification_result: Mapping[str, Any],
    generation_result: Mapping[str, Any],
    error_stage: Optional[str],
    error_reason: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """生成全部阶段的状态表：只有生成与校验两段有输入可解析，其余检索阶段保持 unknown。"""
    stages = {name: _stage() for name in RETRIEVAL_STAGE_NAMES}
    stages["generation"] = _generation_stage(generation_result)
    stages["verification"] = _verification_stage(verification_result)
    if error_stage:
        _mark_error_stage(stages, error_stage, error_reason)
    return stages


def _mark_error_stage(stages: Dict[str, Dict[str, Any]], error_stage: str, error_reason: Optional[str]) -> None:
    """把调用方的错误阶段名映射到观察契约中的阶段键，并将该阶段标记为 failed。"""
    normalized = str(error_stage or "").strip()
    stage_name = ERROR_STAGE_ALIASES.get(normalized, normalized)
    if stage_name in stages:
        stages[stage_name] = _stage(True, "failed", fallback=False, reason=error_reason or normalized)


def _generation_stage(generation: Mapping[str, Any]) -> Dict[str, Any]:
    """根据生成结果判断生成阶段：空答案 failed，模型声明证据不足 insufficient_evidence，否则 success。"""
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
    """把引用校验结果映射为阶段状态：passed→success，warning→partial，其余原样透传。"""
    if not isinstance(verification, Mapping) or not verification:
        return _stage()
    status = str(verification.get("status") or "unknown").strip() or "unknown"
    details = {
        "status": status,
        "source_count": _safe_int(verification.get("source_count"), default=0),
        "citation_count": _list_length(verification.get("cited_source_ids")),
        "missing_citation_count": _list_length(verification.get("missing_cited_source_ids")),
        "unsupported_claim_count": _list_length(verification.get("unsupported_claims")),
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


def _answer_insufficient_evidence(verification: Mapping[str, Any], generation: Mapping[str, Any]) -> str:
    """答案是否证据不足：yes / no / unknown（没有校验结果时无法判断）。"""
    if bool(generation.get("insufficient_evidence", False)) or bool(verification.get("insufficient_evidence", False)):
        return "yes"
    if verification:
        return "no"
    return "unknown"


def _answer_insufficient_evidence_reason(verification: Mapping[str, Any], generation: Mapping[str, Any], *, error_code: Optional[str]) -> str:
    """与 _answer_insufficient_evidence 一一对应的原因字符串。"""
    if bool(generation.get("insufficient_evidence", False)):
        return "generation_declared_insufficient_evidence"
    if bool(verification.get("insufficient_evidence", False)):
        return "verification_guardrail_triggered"
    if verification:
        return "verification_checked_no_insufficient_evidence"
    if error_code:
        return f"unknown_due_to_error:{error_code}"
    return "verification_not_available"


def _answer_quality(stage_status: Mapping[str, Mapping[str, Any]], answer_insufficient: str, *, error_code: Optional[str]) -> str:
    """答案质量分级：generation_failed > insufficient_evidence > grounded / warning > unknown。"""
    if error_code == "llm_generation_failed":
        return "generation_failed"
    generation = stage_status.get("generation") or {}
    verification = stage_status.get("verification") or {}
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
    generation = stage_status.get("generation") or {}
    verification = stage_status.get("verification") or {}
    if str(generation.get("status") or "") == "failed":
        return str(generation.get("reason") or "answer_empty")
    if answer_insufficient == "yes":
        return str(verification.get("reason") or generation.get("reason") or "insufficient_evidence")
    verification_status = str(verification.get("status") or "")
    if verification_status in {"success", "partial"}:
        return str(verification.get("reason") or verification_status)
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
    """把散落阶段状态收束为 Agent 易消费的质量信号。"""
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
    return {
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


def _quality_signal_from_stage(stage: Any) -> Dict[str, Any]:
    """从阶段状态表取一项作为质量信号（浅拷贝）；缺失时返回 unknown 阶段。"""
    if isinstance(stage, Mapping):
        # 复制一份，避免调用方误改 retrieval_stage_status 的完整阶段对象。
        return dict(stage)
    return _stage()


def _degraded_stages(stage_status: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """列出状态为降级类（partial / weak / fallback / failed / insufficient_evidence）的阶段名。"""
    return [
        stage_name
        for stage_name, stage in (stage_status or {}).items()
        if str(stage.get("status") or "") in {"partial", "weak", "fallback", "failed", "insufficient_evidence"}
    ]


def _weak_source_reason(sources: List[Dict[str, Any]], verification: Mapping[str, Any]) -> str:
    """只依据最终来源本身判断证据是否偏弱，返回首要原因；证据正常时返回 not_available。"""
    if len(sources) < MIN_RETRIEVAL_SOURCE_COUNT:
        return "citation_source_insufficient"
    if bool(verification.get("missing_cited_source_ids")):
        return "citation_source_insufficient"
    if _average_source_text_length(sources) < MIN_AVERAGE_SOURCE_TEXT_LENGTH:
        return "evidence_too_short"
    if _context_noise_ratio(sources) >= NOISY_SOURCE_RATIO_THRESHOLD:
        return "context_noise_contamination"
    return "not_available"


def _average_source_text_length(sources: List[Dict[str, Any]]) -> float:
    """来源正文（或资产摘要）的平均字符数；全部为空时返回 0。"""
    lengths = [
        len(text)
        for text in (str(source.get("content") or source.get("text") or source.get("asset_summary") or "").strip() for source in sources)
        if text
    ]
    return sum(lengths) / len(lengths) if lengths else 0.0


def _context_noise_ratio(sources: List[Dict[str, Any]]) -> float:
    """来源中落在参考文献 / 附录 / prompt 等噪声章节的比例。"""
    noisy = 0
    for source in sources:
        haystack = " ".join(str(source.get(key) or "").lower() for key in ("section_path", "section_title", "chunk_type"))
        if any(marker in haystack for marker in NOISY_SECTION_MARKERS):
            noisy += 1
    return noisy / max(1, len(sources))


def _missing_evidence_type(answer_insufficient: str, weak_source_reason: str) -> str:
    """缺失证据类型：证据正常时 not_available；否则在没有问题意图信息的情况下只能给出 unknown。"""
    if answer_insufficient != "yes" and weak_source_reason == "not_available":
        return "not_available"
    return "unknown"


def _rerank_failed_reason(stage_status: Mapping[str, Mapping[str, Any]]) -> str:
    """rerank 阶段失败 / 降级 / 关闭时返回原因，否则 not_available。"""
    rerank = stage_status.get("rerank")
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
) -> tuple[str, str]:
    """检索质量分级及其首要原因（同一组规则按优先级短路，保证等级与原因永远一致）。

    有 error_code 或无来源 → failed；来源过少或答案证据不足 → weak；
    存在弱来源原因、任一阶段/路由 failed 或 fallback → partial；否则 good。
    """
    if error_code:
        return "failed", f"error_code:{error_code}"
    if not sources:
        return "failed", "source_chunks_empty"
    if len(sources) < MIN_RETRIEVAL_SOURCE_COUNT:
        return "weak", f"source_chunks_too_few:{len(sources)}"
    if answer_insufficient == "yes":
        return "weak", "answer_insufficient_evidence"
    if weak_source_reason != "not_available":
        return "partial", weak_source_reason
    for status in ("failed", "fallback"):
        stage_name = _first_stage_with_status(stage_status, status)
        if stage_name:
            return "partial", f"stage_{status}:{stage_name}"
        route_name = _first_stage_with_status(route_status, status)
        if route_name:
            return "partial", f"route_{status}:{route_name}"
    return "good", "retrieval_and_evidence_passed"


def _first_stage_with_status(items: Mapping[str, Mapping[str, Any]], status: str) -> str:
    """返回第一个处于指定状态的阶段 / 路由名，没有则返回空串。"""
    for name, value in (items or {}).items():
        if isinstance(value, Mapping) and str(value.get("status") or "") == status:
            return str(name)
    return ""


def _recommended_repair_actions(
    *,
    retrieval_quality: str,
    missing_evidence_type: str,
    weak_source_reason: str,
    rerank_failed_reason: str,
    error_code: Optional[str],
) -> List[str]:
    """根据质量信号推荐 Agent 可执行的修复动作（顺序、去重和别名归一由 normalize_repair_actions 统一处理）。"""
    actions: List[str] = []
    if error_code == "qa_index_not_found":
        actions.append(ASK_USER_TO_REBUILD_INDEX)
    if retrieval_quality in {"failed", "weak", "partial"}:
        actions.append(RETRY_WITH_QUERY_REWRITE)
    if weak_source_reason in {"citation_source_insufficient", "evidence_too_short"}:
        actions.append(RETRY_WITH_EXPANDED_CONTEXT)
    if rerank_failed_reason not in {"not_available", "disabled", "unknown"}:
        actions.append(RETRY_WITH_QUERY_REWRITE)
    if missing_evidence_type == "unknown" and retrieval_quality in {"failed", "weak"}:
        actions.append(ASK_CLARIFICATION)
    return normalize_repair_actions(actions)


def _observation_reason(retrieval_quality: str, weak_source_reason: str, answer_quality_reason: str) -> str:
    """没有显式 error_reason 时的总体说明。"""
    if retrieval_quality == "good":
        return answer_quality_reason or "retrieval_and_evidence_passed"
    if weak_source_reason != "not_available":
        return weak_source_reason
    return f"retrieval_quality_{retrieval_quality}"


def _safe_int(value: Any, *, default: int = 0) -> int:
    """宽松整数转换，失败返回 default。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _list_length(value: Any) -> int:
    """value 为 list 时返回长度，否则 0。"""
    return len(value) if isinstance(value, list) else 0
