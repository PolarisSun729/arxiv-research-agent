from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from .action_gate import validate_action
from .actions import AbstainAction, DraftAnswerAction, FinalizeAnswerAction, SearchPaperAction, parse_research_action
from .completion_gate import can_finalize, has_verifiable_support
from .contracts import PaperEvidenceResearchResult, ResearchSummary, VerifiedCitation
from .evidence_pool import merge_candidates
from .state import (
    AnswerClaim,
    ClaimAssessment,
    ClaimExtractionRequest,
    ClaimVerificationRequest,
    DraftAnswer,
    DraftGenerationRequest,
    EvidenceNeed,
    PaperEvidenceResearchState,
    ResearchDecisionContext,
)


@dataclass(frozen=True)
class ResearchGraphDependencies:
    question_analyzer: Any
    decision_policy: Any
    retriever: Any
    draft_generator: Any
    claim_extractor: Any
    claim_verifier: Any


def _coerce_state(state: Any) -> PaperEvidenceResearchState:
    if isinstance(state, PaperEvidenceResearchState):
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        return PaperEvidenceResearchState.model_validate(dict(state))
    return PaperEvidenceResearchState.model_validate(state)


def _trace(state: PaperEvidenceResearchState, event_type: str, **payload: Any) -> None:
    state.trace_events.append(
        {
            "sequence": len(state.trace_events) + 1,
            "event_type": event_type,
            **payload,
        }
    )


def _prepare_node(state: Any, dependencies: ResearchGraphDependencies) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    analysis = dict(dependencies.question_analyzer.analyze(next_state.request) or {})
    next_state.research_question = str(analysis.get("research_question") or next_state.request.original_question).strip()
    next_state.evidence_needs = [EvidenceNeed.model_validate(item) for item in analysis.get("evidence_needs") or []]
    if not next_state.evidence_needs:
        raise ValueError("question analyzer did not produce evidence needs")
    _trace(next_state, "research_started", research_question=next_state.research_question)
    return next_state


def _decide_node(state: Any, dependencies: ResearchGraphDependencies) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    context = ResearchDecisionContext(
        research_question=next_state.research_question,
        evidence_needs=next_state.evidence_needs,
        candidate_count=len(next_state.evidence_candidates),
        current_draft_version=next_state.current_draft.version if next_state.current_draft else None,
        claim_assessments=next_state.claim_assessments,
        retrievals_remaining=max(0, next_state.request.limits.max_retrievals - next_state.retrieval_count),
        drafts_remaining=max(0, next_state.request.limits.max_draft_attempts - next_state.draft_attempt_count),
    )
    next_state.pending_action = parse_research_action(dependencies.decision_policy.decide(context))
    next_state.action_accepted = False
    next_state.action_rejection_code = None
    _trace(next_state, "action_proposed", action=next_state.pending_action.action)
    return next_state


def _validate_node(state: Any) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    if next_state.pending_action is None:
        next_state.invalid_action_count += 1
        next_state.action_rejection_code = "ACTION_MISSING"
        return next_state
    validation = validate_action(next_state, next_state.pending_action)
    next_state.action_accepted = validation.accepted
    next_state.action_rejection_code = validation.rejection_code
    if validation.accepted:
        _trace(next_state, "action_accepted", action=next_state.pending_action.action)
    else:
        next_state.invalid_action_count += 1
        _trace(
            next_state,
            "action_rejected",
            action=next_state.pending_action.action,
            rejection_code=validation.rejection_code,
        )
    return next_state


def _route_after_validation(state: Any) -> str:
    current = _coerce_state(state)
    if not current.action_accepted:
        if current.action_rejection_code in {"RETRIEVAL_BUDGET_EXHAUSTED", "DRAFT_BUDGET_EXHAUSTED"}:
            # 预算耗尽是有界研究的正常终止信号，不应继续消耗无效动作重试额度。
            return "terminal"
        if current.invalid_action_count >= current.request.limits.max_invalid_actions:
            return "terminal"
        return "decide"
    if isinstance(current.pending_action, SearchPaperAction):
        return "search"
    if isinstance(current.pending_action, DraftAnswerAction):
        return "draft"
    if isinstance(current.pending_action, FinalizeAnswerAction):
        return "finalize"
    if isinstance(current.pending_action, AbstainAction):
        return "abstain"
    return "terminal"


def _search_node(state: Any, dependencies: ResearchGraphDependencies) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    action = next_state.pending_action
    if not isinstance(action, SearchPaperAction):
        raise ValueError("search node received non-search action")
    next_state.retrieval_count += 1
    retrieval = dict(dependencies.retriever.retrieve(action, next_state) or {})
    candidates = retrieval.get("candidates") or []
    merged, new_count, duplicate_count = merge_candidates(
        next_state.evidence_candidates,
        candidates,
        target_need_id=action.target_need_id,
        max_items=next_state.request.limits.max_evidence_items,
        max_total_context_chars=next_state.request.limits.max_total_context_chars,
    )
    next_state.evidence_candidates = merged
    # 空召回或纯重复都没有改善当前研究状态，连续发生时由图进入有界终止审查。
    if new_count == 0:
        next_state.no_progress_count += 1
        if (
            next_state.request.limits.max_no_progress > 0
            and next_state.no_progress_count >= next_state.request.limits.max_no_progress
        ):
            # 连续检索没有扩充证据池时，将有界停止原因显式写入状态，供终态结果和审计信息复用。
            next_state.action_rejection_code = "NO_PROGRESS_LIMIT_REACHED"
    else:
        next_state.no_progress_count = 0
    _trace(
        next_state,
        "retrieval_completed",
        target_need_id=action.target_need_id,
        status=str(retrieval.get("status") or "completed"),
        new_candidate_count=new_count,
        duplicate_candidate_count=duplicate_count,
    )
    return next_state


def _route_after_search(state: Any) -> str:
    current = _coerce_state(state)
    if current.action_rejection_code == "NO_PROGRESS_LIMIT_REACHED":
        return "terminal"
    return "decide"


def _draft_node(state: Any, dependencies: ResearchGraphDependencies) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    action = next_state.pending_action
    if not isinstance(action, DraftAnswerAction):
        raise ValueError("draft node received non-draft action")
    version = next_state.draft_attempt_count + 1
    generated = dict(
        dependencies.draft_generator.generate(
            DraftGenerationRequest(
                version=version,
                research_question=next_state.research_question,
                addressed_need_ids=list(action.addressed_need_ids),
                candidates=list(next_state.evidence_candidates.values()),
            )
        )
        or {}
    )
    next_state.draft_attempt_count = version
    next_state.current_draft = DraftAnswer(
        draft_id=str(generated.get("draft_id") or f"draft-{version}"),
        version=version,
        answer=str(generated.get("answer") or "").strip(),
        declared_claims=list(generated.get("declared_claims") or []),
        used_candidate_ids=[str(item) for item in generated.get("used_candidate_ids") or []],
    )
    _trace(next_state, "draft_created", draft_id=next_state.current_draft.draft_id, version=version)
    return next_state


def _extract_claims_node(state: Any, dependencies: ResearchGraphDependencies) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    draft = next_state.current_draft
    if draft is None:
        raise ValueError("claim extraction requires a draft")
    extracted = dict(
        dependencies.claim_extractor.extract(
            ClaimExtractionRequest(
                research_question=next_state.research_question,
                answer=draft.answer,
                draft_version=draft.version,
            )
        )
        or {}
    )
    next_state.claims = [AnswerClaim.model_validate(item) for item in extracted.get("claims") or []]
    _trace(next_state, "claims_extracted", draft_version=draft.version, claim_count=len(next_state.claims))
    return next_state


def _verify_claims_node(state: Any, dependencies: ResearchGraphDependencies) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    draft = next_state.current_draft
    if draft is None:
        raise ValueError("claim verification requires a draft")
    verified = dict(
        dependencies.claim_verifier.verify(
            ClaimVerificationRequest(
                research_question=next_state.research_question,
                draft_version=draft.version,
                claims=next_state.claims,
                candidates=list(next_state.evidence_candidates.values()),
            )
        )
        or {}
    )
    next_state.claim_assessments = [
        ClaimAssessment.model_validate(item) for item in verified.get("assessments") or []
    ]
    next_state.verification_count += 1
    _trace(
        next_state,
        "claim_verification_completed",
        draft_version=draft.version,
        assessment_count=len(next_state.claim_assessments),
    )
    return next_state


def _project_coverage_node(state: Any) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    assessment_by_claim = {item.claim_id: item for item in next_state.claim_assessments}
    claims_by_need: dict[str, list[AnswerClaim]] = {}
    for claim in next_state.claims:
        for need_id in claim.addressed_need_ids:
            claims_by_need.setdefault(need_id, []).append(claim)

    projected_needs: list[EvidenceNeed] = []
    for need in next_state.evidence_needs:
        projected = need.model_copy(deep=True)
        related_claims = claims_by_need.get(need.need_id, [])
        related_assessments = [assessment_by_claim.get(claim.claim_id) for claim in related_claims]
        if related_claims and all(has_verifiable_support(next_state, item) for item in related_assessments):
            # 需求满足权只来自已校验主张，检索候选本身永远不能直接修改覆盖状态。
            projected.status = "satisfied"
            projected.supporting_claim_ids = [claim.claim_id for claim in related_claims]
            projected.verified_evidence_ids = sorted(
                {
                    evidence_id
                    for assessment in related_assessments
                    if assessment is not None
                    for evidence_id in assessment.supporting_evidence_ids
                }
            )
        elif related_claims:
            projected.status = "open"
            projected.supporting_claim_ids = []
            projected.verified_evidence_ids = []
        projected_needs.append(projected)
    next_state.evidence_needs = projected_needs
    _trace(
        next_state,
        "evidence_coverage_projected",
        satisfied_need_ids=[need.need_id for need in projected_needs if need.status == "satisfied"],
    )
    return next_state


def _build_citations(state: PaperEvidenceResearchState, supported_claim_ids: set[str]) -> list[VerifiedCitation]:
    claim_ids_by_source: dict[str, list[str]] = {}
    for assessment in state.claim_assessments:
        if assessment.claim_id not in supported_claim_ids or assessment.verdict != "supported":
            continue
        for source_id in assessment.supporting_evidence_ids:
            claim_ids_by_source.setdefault(source_id, []).append(assessment.claim_id)
    citations: list[VerifiedCitation] = []
    for source_id, claim_ids in claim_ids_by_source.items():
        candidate = state.evidence_candidates.get(source_id)
        if candidate is None:
            continue
        citations.append(
            VerifiedCitation(
                source_id=source_id,
                content=candidate.content,
                claim_ids=sorted(set(claim_ids)),
                chunk_type=candidate.chunk_type,
                section_path=candidate.section_path,
                page_number=candidate.page_number,
            )
        )
    return citations


def _build_result(
    state: PaperEvidenceResearchState,
    *,
    outcome: str,
    answer: str,
    termination_reason: str,
    supported_claim_ids: set[str],
) -> PaperEvidenceResearchResult:
    unresolved = [
        need.description
        for need in state.evidence_needs
        if need.importance == "core" and need.status != "satisfied"
    ]
    return PaperEvidenceResearchResult(
        outcome=outcome,
        answer=answer,
        citations=_build_citations(state, supported_claim_ids),
        research_summary=ResearchSummary(
            research_run_id=state.request.research_run_id,
            retrieval_count=state.retrieval_count,
            draft_attempt_count=state.draft_attempt_count,
            verification_count=state.verification_count,
            confirmed_need_count=len(
                [need for need in state.evidence_needs if need.status not in {"provisional", "abandoned", "superseded"}]
            ),
            satisfied_need_count=len([need for need in state.evidence_needs if need.status == "satisfied"]),
            blocked_need_count=len([need for need in state.evidence_needs if need.status == "blocked"]),
            supported_claim_count=len(supported_claim_ids),
            outcome=outcome,
            termination_reason=termination_reason,
            unresolved_topics=unresolved,
        ),
        research_trace_id=state.request.research_run_id,
    )


def _finalize_node(state: Any) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    allowed, reason = can_finalize(next_state)
    if not allowed or next_state.current_draft is None:
        raise ValueError(f"completion gate rejected finalization: {reason}")
    assessment_by_claim = {item.claim_id: item for item in next_state.claim_assessments}
    supported_claim_ids = {
        claim.claim_id
        for claim in next_state.claims
        if has_verifiable_support(next_state, assessment_by_claim.get(claim.claim_id))
    }
    next_state.result = _build_result(
        next_state,
        outcome="completed",
        answer=next_state.current_draft.answer,
        termination_reason="core_needs_satisfied",
        supported_claim_ids=supported_claim_ids,
    )
    _trace(next_state, "research_completed", outcome="completed")
    return next_state


def _terminal_node(state: Any, *, requested_abstention: bool = False) -> PaperEvidenceResearchState:
    next_state = _coerce_state(state)
    abstention_reason = (
        next_state.pending_action.reason_code
        if requested_abstention and isinstance(next_state.pending_action, AbstainAction)
        else None
    )
    assessment_by_claim = {item.claim_id: item for item in next_state.claim_assessments}
    supported_claims = [
        claim
        for claim in next_state.claims
        if has_verifiable_support(next_state, assessment_by_claim.get(claim.claim_id))
    ]
    supported_claim_ids = {claim.claim_id for claim in supported_claims}
    for need in next_state.evidence_needs:
        if need.importance == "core" and need.status == "open":
            need.status = "blocked"
            need.blocking_reason = next_state.action_rejection_code or "research_stopped_without_full_coverage"
    if supported_claims and not requested_abstention:
        # 有界终止只保留已验证主张，避免把初稿中的无支持内容带入有限回答。
        answer = "；".join(claim.text.rstrip("。；") for claim in supported_claims) + "。"
        outcome = "partial"
        termination_reason = next_state.action_rejection_code or "partial_answer_after_budget_exhaustion"
    else:
        answer = "当前论文证据不足以支持问题中的核心结论。"
        outcome = "abstained"
        termination_reason = abstention_reason or next_state.action_rejection_code or "no_supported_core_claim"
    next_state.result = _build_result(
        next_state,
        outcome=outcome,
        answer=answer,
        termination_reason=termination_reason,
        supported_claim_ids=supported_claim_ids,
    )
    _trace(next_state, "research_completed", outcome=outcome)
    return next_state


def build_paper_evidence_research_graph(
    dependencies: ResearchGraphDependencies,
    *,
    checkpointer: Any = None,
) -> Any:
    graph = StateGraph(PaperEvidenceResearchState)
    graph.add_node("prepare", lambda state: _prepare_node(state, dependencies))
    graph.add_node("decide", lambda state: _decide_node(state, dependencies))
    graph.add_node("validate", _validate_node)
    graph.add_node("search", lambda state: _search_node(state, dependencies))
    graph.add_node("draft", lambda state: _draft_node(state, dependencies))
    graph.add_node("extract_claims", lambda state: _extract_claims_node(state, dependencies))
    graph.add_node("verify_claims", lambda state: _verify_claims_node(state, dependencies))
    graph.add_node("project_coverage", _project_coverage_node)
    graph.add_node("finalize", _finalize_node)
    graph.add_node("terminal", _terminal_node)
    graph.add_node("abstain", lambda state: _terminal_node(state, requested_abstention=True))

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "decide")
    graph.add_edge("decide", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validation,
        {
            "search": "search",
            "draft": "draft",
            "finalize": "finalize",
            "abstain": "abstain",
            "decide": "decide",
            "terminal": "terminal",
        },
    )
    graph.add_conditional_edges("search", _route_after_search, {"decide": "decide", "terminal": "terminal"})
    graph.add_edge("draft", "extract_claims")
    graph.add_edge("extract_claims", "verify_claims")
    graph.add_edge("verify_claims", "project_coverage")
    graph.add_edge("project_coverage", "decide")
    graph.add_edge("finalize", END)
    graph.add_edge("terminal", END)
    graph.add_edge("abstain", END)
    return graph.compile(checkpointer=checkpointer)
