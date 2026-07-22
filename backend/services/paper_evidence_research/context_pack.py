from __future__ import annotations

from collections.abc import Iterable

from .state import EvidenceCandidate, EvidenceContextPack, PaperEvidenceResearchState


def build_context_pack(
    state: PaperEvidenceResearchState,
    *,
    target_need_ids: Iterable[str],
    pinned_candidate_ids: Iterable[str] = (),
) -> EvidenceContextPack:
    selected_ids: list[str] = []
    seen: set[str] = set()

    def select(candidate_id: str) -> None:
        if candidate_id in state.evidence_candidates and candidate_id not in seen:
            seen.add(candidate_id)
            selected_ids.append(candidate_id)

    # 校验必须先保留答案实际引用；已验证证据随后锁定，避免后续相关性排序挤掉可靠依据。
    for candidate_id in pinned_candidate_ids:
        select(candidate_id)
    for need in state.evidence_needs:
        if need.status == "satisfied":
            for candidate_id in need.verified_evidence_ids:
                select(candidate_id)

    requested_need_ids = list(dict.fromkeys(target_need_ids))
    core_need_ids = [need.need_id for need in state.evidence_needs if need.importance == "core"]
    need_priority = list(dict.fromkeys([*requested_need_ids, *core_need_ids]))
    candidates_by_need = {
        need_id: [
            candidate.candidate_id
            for candidate in state.evidence_candidates.values()
            if need_id in candidate.matched_need_ids
        ]
        for need_id in need_priority
    }
    # 每轮从各需求各取一个候选，避免某一章节的多个片段先占满整个上下文包。
    while True:
        progress = False
        for need_id in need_priority:
            for candidate_id in candidates_by_need[need_id]:
                if candidate_id not in seen:
                    select(candidate_id)
                    progress = True
                    break
        if not progress:
            break

    # 余量候选只在覆盖优先项之后补齐；当前池本身已受条数和字符预算约束。
    for candidate_id in state.evidence_candidates:
        select(candidate_id)

    candidates: list[EvidenceCandidate] = [state.evidence_candidates[candidate_id] for candidate_id in selected_ids]
    return EvidenceContextPack(
        selected_need_ids=need_priority,
        candidates=candidates,
        total_context_chars=sum(len(candidate.content) for candidate in candidates),
    )
