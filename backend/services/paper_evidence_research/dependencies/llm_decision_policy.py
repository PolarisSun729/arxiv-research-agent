"""LLM 优先的决策策略：在预算内提议一个语义动作，失败即降级到规则查表。

两条边界必须分清：

- 这里只提议动作的结构化语义；动作的状态合法性（预算、目标需求是否 open、终态放行）
  仍由图的 Action Gate 裁决，策略层不做也不能做安全判断。
- LLM 连续失败达到阈值后熔断为纯规则，避免同一 run 内反复为失效的 LLM 付费并烧掉
  无效动作重试额度。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..actions import parse_research_action
from .degradation import DependencyDegradation, DegradationListener, REASON_LLM_CALL_FAILED, REASON_LLM_OUTPUT_INVALID
from .llm_question_analyzer import _extract_json_payload
from .rule_decision_policy import RuleDecisionPolicy

logger = logging.getLogger(__name__)

_DECISION_PROMPT = """You are the decision policy of a bounded evidence-research engine for single-paper QA.
Given the research state, choose exactly ONE next action. The gate will enforce budgets and validity; you should still respect the remaining budgets shown in the state.
Rules:
1. Return JSON only, no extra text before or after.
2. Choose "finalize_answer" only when every claim assessment is "supported" and no core need is open.
3. Choose "abstain" only when the paper clearly cannot answer the question and no further action would help.
4. Use "search_paper" to gather evidence for an open core need; write a concrete retrieval query.
5. Use "draft_answer" to (re)write the answer from current evidence; list the need ids it addresses.
6. The action JSON schemas are:
   {{"action": "search_paper", "target_need_id": "...", "objective": "discover|verify_claim|resolve_conflict|check_absence", "retrieval_mode": "broad|definition|method|experiment|comparison|numeric", "query": "...", "section_hints": [], "reason_code": "..."}}
   {{"action": "draft_answer", "addressed_need_ids": ["..."], "reason_code": "..."}}
   {{"action": "finalize_answer", "reason_code": "..."}}
   {{"action": "abstain", "reason_code": "..."}}

Research state:
{state_json}"""


class LlmDecisionPolicy:
    """LLM 决策器：读决策上下文 -> 提议动作；解析失败降级规则，连续失败熔断。"""

    def __init__(
        self,
        generation_service: Any,
        *,
        fallback: Any = None,
        degradation_listener: DegradationListener | None = None,
        max_consecutive_llm_failures: int = 2,
    ) -> None:
        self._generation_service = generation_service
        self._fallback = fallback or RuleDecisionPolicy()
        self._degradation_listener = degradation_listener
        self._max_consecutive_failures = max(1, int(max_consecutive_llm_failures))
        self._consecutive_failures = 0
        self._llm_disabled = False

    def decide(self, context: Any) -> dict[str, Any]:
        if self._llm_disabled:
            return self._fallback.decide(context)
        prompt = _DECISION_PROMPT.format(state_json=json.dumps(self._compact_state(context), ensure_ascii=False))
        try:
            response = self._generation_service.complete_with_qwen(
                prompt,
                task_type="research_decision",
            )
        except Exception as exc:
            return self._degrade(context, REASON_LLM_CALL_FAILED, {"error": str(exc)[:200]})
        try:
            payload = _extract_json_payload(response)
            action = parse_research_action(payload)
        except Exception as exc:
            return self._degrade(context, REASON_LLM_OUTPUT_INVALID, {"error": str(exc)[:200]})
        self._consecutive_failures = 0
        return action.model_dump()

    def _compact_state(self, context: Any) -> dict[str, Any]:
        needs = list(getattr(context, "evidence_needs", []) or [])
        assessments = list(getattr(context, "claim_assessments", []) or [])
        return {
            "research_question": getattr(context, "research_question", ""),
            "evidence_needs": [
                {
                    "need_id": getattr(need, "need_id", ""),
                    "description": getattr(need, "description", ""),
                    "importance": getattr(need, "importance", ""),
                    "status": getattr(need, "status", ""),
                }
                for need in needs
            ],
            "candidate_count": getattr(context, "candidate_count", 0),
            "candidate_need_ids": list(getattr(context, "candidate_need_ids", []) or []),
            "current_draft_version": getattr(context, "current_draft_version", None),
            "claim_assessments": [
                {"claim_id": getattr(item, "claim_id", ""), "verdict": getattr(item, "verdict", "")}
                for item in assessments
            ],
            "retrievals_remaining": getattr(context, "retrievals_remaining", 0),
            "drafts_remaining": getattr(context, "drafts_remaining", 0),
        }

    def _degrade(self, context: Any, reason: str, detail: dict[str, Any]) -> dict[str, Any]:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            self._llm_disabled = True
        logger.warning("research decision policy degraded: reason=%s detail=%s", reason, detail)
        if self._degradation_listener is not None:
            try:
                self._degradation_listener(
                    DependencyDegradation(organ="decision_policy", reason=reason, detail=detail)
                )
            except Exception:  # pragma: no cover - 监听器异常不能影响兜底路径
                logger.exception("degradation listener failed")
        return self._fallback.decide(context)
