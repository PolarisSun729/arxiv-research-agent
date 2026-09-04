"""确定性查表决策策略。

LLM 决策失败或被禁用时的兜底，也是行为可预期的最小基线。它只负责在预算内提出
"合理的贪婪动作"；预算、目标合法性与终态放行仍由图骨架的 Action Gate / Completion
Gate 裁决——策略永远不是安全性边界。
"""

from __future__ import annotations

from typing import Any

_RETRIEVAL_MODE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("method", ("method", "approach", "architecture", "pipeline", "module", "方法", "流程", "架构", "模块")),
    ("experiment", ("experiment", "dataset", "benchmark", "metric", "ablation", "实验", "数据集", "指标", "消融")),
    ("comparison", ("compare", "comparison", "baseline", "versus", "outperform", "比较", "对比", "基线", "优于")),
    ("definition", ("definition", "define", "what is", "概念", "定义", "是什么")),
)

_MAX_QUERY_CHARS = 200


def _pick_retrieval_mode(text: str) -> str:
    lowered = text.lower()
    for mode, keywords in _RETRIEVAL_MODE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return mode
    return "broad"


def _build_search_query(context: Any, need: Any) -> str:
    question = str(getattr(context, "research_question", "") or "").strip()
    description = str(getattr(need, "description", "") or "").strip()
    if question and question in description:
        query = description
    else:
        query = " ".join(part for part in (question, description) if part)
    return query[:_MAX_QUERY_CHARS].strip()


def _need_id(need: Any) -> str:
    return str(getattr(need, "need_id", "") or "")


class RuleDecisionPolicy:
    """按固定优先级查表：先补证据、再起草、验证后收口；无路可走时交给门禁有界终止。"""

    def decide(self, context: Any) -> dict[str, Any]:
        needs = list(getattr(context, "evidence_needs", []) or [])
        core_needs = [need for need in needs if getattr(need, "importance", "") == "core"]
        open_core = [
            need for need in core_needs if getattr(need, "status", "") in {"provisional", "open"}
        ]
        has_draft = getattr(context, "current_draft_version", None) is not None
        assessments = list(getattr(context, "claim_assessments", []) or [])
        all_supported = bool(assessments) and all(
            getattr(assessment, "verdict", "") == "supported" for assessment in assessments
        )
        candidate_need_ids = set(getattr(context, "candidate_need_ids", []) or [])
        retrievals_remaining = int(getattr(context, "retrievals_remaining", 0) or 0)
        drafts_remaining = int(getattr(context, "drafts_remaining", 0) or 0)
        retrievals_used = int(getattr(context, "retrievals_used", 0) or 0)
        candidate_count = int(getattr(context, "candidate_count", 0) or 0)

        if has_draft:
            return self._decide_with_draft(
                context=context,
                core_needs=core_needs,
                open_core=open_core,
                all_supported=all_supported,
                has_assessments=bool(assessments),
                candidate_need_ids=candidate_need_ids,
                retrievals_remaining=retrievals_remaining,
                drafts_remaining=drafts_remaining,
            )
        return self._decide_without_draft(
            context=context,
            core_needs=core_needs,
            open_core=open_core,
            candidate_need_ids=candidate_need_ids,
            candidate_count=candidate_count,
            retrievals_remaining=retrievals_remaining,
            retrievals_used=retrievals_used,
        )

    def _decide_with_draft(
        self,
        *,
        context: Any,
        core_needs: list[Any],
        open_core: list[Any],
        all_supported: bool,
        has_assessments: bool,
        candidate_need_ids: set[str],
        retrievals_remaining: int,
        drafts_remaining: int,
    ) -> dict[str, Any]:
        if all_supported and has_assessments and not open_core:
            return self._finalize("CORE_CLAIMS_SUPPORTED")
        if drafts_remaining <= 0:
            # 没有草稿预算修订时申请终局；门禁拒绝会让图以 partial 有界收口，比空转更快。
            return self._finalize("DRAFT_BUDGET_EXHAUSTED_REQUEST_FINALIZE")
        uncovered_open = [need for need in open_core if _need_id(need) not in candidate_need_ids]
        if uncovered_open and retrievals_remaining > 0:
            objective = "verify_claim" if has_assessments else "discover"
            return self._search(
                context,
                uncovered_open[0],
                objective=objective,
                reason_code="UNSUPPORTED_OR_OPEN_CORE_NEED",
            )
        return self._draft(core_needs, "REVISE_WITH_CURRENT_EVIDENCE")

    def _decide_without_draft(
        self,
        *,
        context: Any,
        core_needs: list[Any],
        open_core: list[Any],
        candidate_need_ids: set[str],
        candidate_count: int,
        retrievals_remaining: int,
        retrievals_used: int,
    ) -> dict[str, Any]:
        if candidate_count <= 0:
            if retrievals_remaining <= 0 or not core_needs:
                return self._abstain("NO_EVIDENCE_BEFORE_DRAFT")
            target = open_core[0] if open_core else core_needs[0]
            return self._search(
                context,
                target,
                objective="discover",
                reason_code="OPEN_CORE_NEED_NO_CANDIDATES",
            )
        if not open_core:
            return self._draft(core_needs, "CORE_NEEDS_SATISFIED_READY_TO_DRAFT")
        uncovered_open = [need for need in open_core if _need_id(need) not in candidate_need_ids]
        # 起草前给每个核心需求一轮检索机会；全部覆盖或轮数用尽即带现有证据起草。
        if retrievals_remaining > 0 and uncovered_open and retrievals_used < max(len(core_needs), 1):
            return self._search(
                context,
                uncovered_open[0],
                objective="discover",
                reason_code="GATHER_EVIDENCE_FOR_OPEN_CORE_NEED",
            )
        return self._draft(core_needs, "EVIDENCE_GATHERED_DRAFT_NOW")

    def _search(self, context: Any, need: Any, *, objective: str, reason_code: str) -> dict[str, Any]:
        return {
            "action": "search_paper",
            "target_need_id": _need_id(need),
            "objective": objective,
            "retrieval_mode": _pick_retrieval_mode(
                f"{getattr(context, 'research_question', '')} {getattr(need, 'description', '')}"
            ),
            "query": _build_search_query(context, need),
            "section_hints": [],
            "reason_code": reason_code,
        }

    def _draft(self, core_needs: list[Any], reason_code: str) -> dict[str, Any]:
        return {
            "action": "draft_answer",
            "addressed_need_ids": [_need_id(need) for need in core_needs],
            "reason_code": reason_code,
        }

    def _finalize(self, reason_code: str) -> dict[str, Any]:
        return {"action": "finalize_answer", "reason_code": reason_code}

    def _abstain(self, reason_code: str) -> dict[str, Any]:
        return {"action": "abstain", "reason_code": reason_code}
