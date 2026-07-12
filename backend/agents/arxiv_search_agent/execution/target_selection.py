from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping
from uuid import uuid4

from ..schemas import PlanRuntime, PlanStep
from .interactions import AgentInteraction, TargetSelectionPayload


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "arxiv_id", "paper_id", "id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return value
    return ""


class TargetSelectionCoordinator:
    """构造论文目标选择交互，并把已校验选择投影为 resolver 的成功输出。"""

    def build_interaction(self, *, step: PlanStep, runtime: PlanRuntime, payload: Mapping[str, Any]) -> AgentInteraction:
        resolution = dict(payload.get("target_resolution") or {})
        raw_candidates = payload.get("candidates") or resolution.get("candidates") or []
        candidates = [dict(candidate) for candidate in raw_candidates if isinstance(candidate, Mapping)]
        recommended = payload.get("recommended_candidate") or resolution.get("recommended_candidate")
        recommended_id = _candidate_id(recommended) if isinstance(recommended, Mapping) else _candidate_id(candidates[0]) if candidates else None
        now = datetime.now(timezone.utc)
        return AgentInteraction(
            interaction_id=str(uuid4()),
            kind="target_selection",
            plan_id=str(runtime.plan.plan_id if runtime.plan else ""),
            step_id=step.step_id,
            payload=TargetSelectionPayload(
                candidates=candidates,
                recommended_candidate_id=recommended_id or None,
                reference_hint=dict(payload.get("reference_hint") or resolution.get("reference_hint") or {}),
            ),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=10)).isoformat(),
        )

    def materialize(self, *, selected_candidate: Mapping[str, Any], payload: Mapping[str, Any]) -> Dict[str, Any]:
        candidate = dict(selected_candidate)
        arxiv_id = str(candidate.get("arxiv_id") or candidate.get("paper_id") or candidate.get("id") or "").strip()
        return {
            **dict(payload),
            # resolver 的 output_key 本身通常就是 paper_ref，顶层必须直接暴露论文身份，
            # 否则记录输出时会形成 paper_ref.paper_ref 的双层结构并破坏后续输入绑定。
            **candidate,
            "paper": candidate,
            "target_paper": candidate,
            "confirmed_paper_id": _candidate_id(candidate),
            "confirmed_arxiv_id": arxiv_id or None,
            "confirmed_by_user": True,
            "final_target_resolved": True,
            "target_resolution": {
                **dict(payload.get("target_resolution") or {}),
                "status": "resolved",
                "selected_candidate": candidate,
                "resolution_reason": "user_confirmed_target",
                "final_target_resolved": True,
            },
        }
