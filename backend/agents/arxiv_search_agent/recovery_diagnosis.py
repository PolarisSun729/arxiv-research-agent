from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .schemas import ExecutablePlan, Goal, LLMRecoveryDiagnosis, ObservationResult, PlanRuntime, PlanStep, RecoveryCandidate
from .state import AgentState


def _get_recovery_diagnosis_config() -> Dict[str, Any]:
    try:
        from utils.config import get_agent_planner_runtime_config

        return dict(get_agent_planner_runtime_config())
    except Exception:
        return {"enable_llm_recovery_diagnosis": False, "llm_recovery_timeout": 6}


class LLMRecoveryDiagnoser:
    """可选 LLM 诊断层，只返回语义判断和候选排序建议。"""

    def __init__(self, generation_service: Optional[Any] = None, *, enabled: Optional[bool] = None, timeout_seconds: Optional[int] = None) -> None:
        config = _get_recovery_diagnosis_config()
        self.enabled = bool(config.get("enable_llm_recovery_diagnosis", False)) if enabled is None else bool(enabled)
        self.timeout_seconds = int(timeout_seconds or config.get("llm_recovery_timeout") or 6)
        self.generation_service = generation_service

    def diagnose(
        self,
        *,
        goal: Goal,
        current_plan: ExecutablePlan,
        runtime: PlanRuntime,
        failed_step: PlanStep,
        observation: ObservationResult,
        recovery_candidates: Sequence[RecoveryCandidate],
        state: Optional[AgentState] = None,
    ) -> Optional[LLMRecoveryDiagnosis]:
        del current_plan, runtime
        if not self.enabled:
            return None
        if not recovery_candidates:
            return LLMRecoveryDiagnosis(error="no_candidates")
        service = self.generation_service or self._resolve_generation_service()
        if service is None or not hasattr(service, "complete_with_qwen"):
            return LLMRecoveryDiagnosis(error="llm_service_unavailable")
        try:
            prompt = self._build_prompt(goal=goal, failed_step=failed_step, observation=observation, recovery_candidates=recovery_candidates, state=state)
            response = service.complete_with_qwen(prompt, task_type="agent_recovery_diagnosis", timeout=self.timeout_seconds)
            payload = self._parse_json_payload(response)
            diagnosis = LLMRecoveryDiagnosis(**payload)
        except Exception as exc:
            # LLM 诊断失败不能影响规则恢复，只把失败原因写入 trace。
            return LLMRecoveryDiagnosis(error=f"llm_diagnosis_failed:{exc}")
        valid_candidate_ids = {candidate.candidate_id for candidate in list(recovery_candidates or [])}
        valid_ranked: List[str] = []
        ignored: List[str] = []
        for candidate_id in list(diagnosis.ranked_candidate_ids or []):
            normalized = str(candidate_id or "").strip()
            if normalized in valid_candidate_ids:
                valid_ranked.append(normalized)
            elif normalized:
                ignored.append(normalized)
        return diagnosis.model_copy(update={"ranked_candidate_ids": valid_ranked, "ignored_candidate_ids": ignored})

    def _resolve_generation_service(self) -> Optional[Any]:
        try:
            from services.llm.generation_service import GenerationService

            return GenerationService()
        except Exception:
            return None

    def _build_prompt(
        self,
        *,
        goal: Goal,
        failed_step: PlanStep,
        observation: ObservationResult,
        recovery_candidates: Sequence[RecoveryCandidate],
        state: Optional[AgentState],
    ) -> str:
        candidates = [
            {
                "candidate_id": candidate.candidate_id,
                "action_type": candidate.action_type,
                "reason": candidate.reason,
                "risk_level": candidate.risk_level,
                "requires_confirmation": candidate.requires_confirmation,
            }
            for candidate in list(recovery_candidates or [])
        ]
        payload = {
            "user_request": getattr(state, "message", None),
            "goal_type": goal.goal_type,
            "failed_step": {"step_id": failed_step.step_id, "tool_name": failed_step.tool_name},
            "observation": observation.model_dump(),
            "recovery_candidates": candidates,
        }
        return (
            "你是 Agent recovery 诊断器，只能输出 JSON。不要创建新动作，不要输出 plan patch。\n"
            "输出字段：diagnosis, recommended_recovery_type, ranked_candidate_ids, clarification_question, "
            "user_facing_reason, confidence。\n"
            f"输入：{json.dumps(payload, ensure_ascii=False)}"
        )

    def _parse_json_payload(self, response: Any) -> Mapping[str, Any]:
        text = str(response or "").strip()
        try:
            payload = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("llm_output_not_json")
            payload = json.loads(match.group(0))
        if not isinstance(payload, Mapping):
            raise ValueError("llm_output_not_object")
        return payload


__all__ = ["LLMRecoveryDiagnoser"]
