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

from services.paper_qa.context_pack_builder import ContextPackBuilder

from ..actions import parse_research_action
from .degradation import DependencyDegradation, DegradationListener, REASON_LLM_CALL_FAILED, REASON_LLM_OUTPUT_INVALID
from .llm_question_analyzer import _extract_json_payload
from .rule_decision_policy import RuleDecisionPolicy

logger = logging.getLogger(__name__)

_DECISION_PROMPT = """You are the decision policy of a bounded evidence-research engine that answers one question about one paper.
The goal is an answer that fully answers the research question, with every claim supported by the paper. Each turn you choose the ONE next action that brings the answer closest to that goal.

What the research state means:
- evidence_needs: the parts a complete answer has to cover ("core" ones are required). Status "provisional" = no verified draft has addressed it yet; "satisfied" = the latest verified draft has at least one supported claim for it; "open" = the latest draft addressed it but none of those claims was supported. One supported sentence is enough for "satisfied", so a satisfied need can still be covered only superficially; judge depth yourself against what the question asks.
- need_evidence: which candidates were retrieved while targeting each need. This records the retrieval target only; a candidate may also contain content for other needs, so judge by reading the evidence.
- latest_draft_claims: what the latest draft actually says, claim by claim, with the needs each claim addresses, the verifier's verdict, and the facets the verifier found missing.
- The evidence below is only what has been retrieved so far, not the whole paper. If the question asks for something the evidence only touches on (for example a method whose details sit in sections not retrieved yet), the paper most likely contains more.
- When the engine stops (budgets used up or no valid action), it delivers the best verified draft trimmed to its supported claims, so the last draft should be written from the best evidence you could gather.

Actions:
- search_paper: retrieve more passages from the paper for one core or supporting need whose status is "provisional" or "open". It adds new evidence and costs one retrieval. Write the query for the specific content still missing after reading the evidence, different from earlier queries. Not possible when retrievals_remaining is 0 or evidence_pool_full is true.
- draft_answer: (re)write the answer from the current evidence, listing the need ids it addresses. The draft is then split into claims and each claim is verified against the evidence, which shows what is supported and what is missing. It costs one draft. Rewriting from unchanged evidence only helps when the latest verdicts point to something the drafter can fix with that evidence (a wrong citation, or content the evidence has but the draft left out); otherwise gather the missing evidence first. Not possible when drafts_remaining is 0.
- finalize_answer: deliver the latest draft. Accepted only when every claim is supported and every core need is satisfied.
- abstain: give up on the question. Only when, after reading the evidence, the paper clearly does not contain what the question asks and further search could not find it. Not while the evidence has on-topic content for a core need.

Evidence filtering: put the source_id of every retrieved candidate that is irrelevant to the question AND to all evidence needs into "discard_evidence_ids". Discarded candidates leave the pool and are never shown to the drafter again. Keep anything you are unsure about, use only source_ids that appear in the evidence below, and use [] when nothing is irrelevant.

Return JSON only, no extra text before or after, matching one of these schemas:
   {{"action": "search_paper", "target_need_id": "...", "objective": "discover|verify_claim|resolve_conflict|check_absence", "retrieval_mode": "broad|definition|method|experiment|comparison|numeric", "query": "...", "section_hints": [], "discard_evidence_ids": [], "reason_code": "..."}}
   {{"action": "draft_answer", "addressed_need_ids": ["..."], "discard_evidence_ids": [], "reason_code": "..."}}
   {{"action": "finalize_answer", "discard_evidence_ids": [], "reason_code": "..."}}
   {{"action": "abstain", "discard_evidence_ids": [], "reason_code": "..."}}

Research state:
{state_json}

Retrieved evidence ({evidence_count} candidates, full text in retrieval rank order):
{evidence_text}"""


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
        evidence_pack = self._evidence_pack(context)
        prompt = _DECISION_PROMPT.format(
            state_json=json.dumps(self._compact_state(context), ensure_ascii=False),
            evidence_count=len(getattr(context, "evidence", []) or []),
            evidence_text=evidence_pack["text_context"] or "(none retrieved yet)",
        )
        image_inputs = evidence_pack["image_inputs"]
        try:
            if image_inputs:
                # 图表证据附原图走多模态输入；没有图时保持纯文本调用。
                response = self._generation_service.complete_with_qwen(
                    prompt,
                    task_type="research_decision",
                    image_inputs=image_inputs,
                )
            else:
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

    @staticmethod
    def _evidence_pack(context: Any) -> dict[str, Any]:
        # 与起草使用同一个打包器：文本块带 source_id、页码、章节，图片候选产出原图输入。
        chunks = [
            dict(candidate.model_dump(mode="json"), source_id=candidate.candidate_id)
            for candidate in getattr(context, "evidence", []) or []
        ]
        if not chunks:
            return {"text_context": "", "image_inputs": []}
        pack = ContextPackBuilder().build(chunks)
        return {"text_context": pack.get("text_context") or "", "image_inputs": pack.get("image_inputs") or []}

    def _compact_state(self, context: Any) -> dict[str, Any]:
        needs = list(getattr(context, "evidence_needs", []) or [])
        assessment_by_claim = {
            getattr(item, "claim_id", ""): item for item in getattr(context, "claim_assessments", []) or []
        }
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
            "evidence_pool_capacity": getattr(context, "evidence_pool_capacity", 0),
            "evidence_pool_full": bool(getattr(context, "evidence_pool_full", False)),
            "need_evidence": [
                {
                    "need_id": getattr(item, "need_id", ""),
                    "candidate_count": getattr(item, "candidate_count", 0),
                    "candidate_ids": list(getattr(item, "candidate_ids", []) or []),
                }
                for item in getattr(context, "need_evidence", []) or []
            ],
            "current_draft_version": getattr(context, "current_draft_version", None),
            "latest_draft_claims": [
                {
                    "claim_id": getattr(claim, "claim_id", ""),
                    "text": getattr(claim, "text", ""),
                    "addressed_need_ids": list(getattr(claim, "addressed_need_ids", []) or []),
                    "verdict": getattr(assessment_by_claim.get(getattr(claim, "claim_id", "")), "verdict", "unverified"),
                    "missing_facets": list(
                        getattr(assessment_by_claim.get(getattr(claim, "claim_id", "")), "missing_facets", []) or []
                    ),
                }
                for claim in getattr(context, "claims", []) or []
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
