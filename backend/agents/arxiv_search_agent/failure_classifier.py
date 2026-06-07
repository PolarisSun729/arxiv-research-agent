from __future__ import annotations

from typing import Optional

from .schemas import FailureCategory, ObservationResult, PlanStep


class FailureClassifier:
    """把 observation 归一化成稳定失败类别，避免恢复链路绑定自然语言 reason。"""

    def classify(self, *, step: PlanStep, observation: ObservationResult) -> ObservationResult:
        if observation.failure_category:
            return observation
        category = self._infer_category(step=step, observation=observation)
        if category is None:
            return observation
        return observation.model_copy(update={"failure_category": category})

    def _infer_category(self, *, step: PlanStep, observation: ObservationResult) -> Optional[FailureCategory]:
        status = str(observation.status or "").strip()
        tool_name = str(step.tool_name or "").strip()

        # 这里仅为旧 observation 做保守兜底；新增分类仍应优先由 Observer 显式写入。
        if tool_name == "search_arxiv" and status == "empty_result":
            return "search_empty"
        if tool_name == "validate_arxiv_results" and status == "low_confidence":
            return "search_low_confidence"
        if tool_name == "check_paper_index" and status == "need_confirmation":
            return "paper_index_missing"
        if tool_name == "check_paper_index" and status == "low_confidence":
            return "paper_index_stale"
        if tool_name == "load_user_profile" and status == "empty_result":
            return "empty_user_profile"
        if tool_name == "answer_paper_question" and status == "low_confidence":
            return "qa_no_answer"
        if tool_name in {"update_preference_store", "resolve_preference_target"} and status == "need_clarification":
            return "preference_target_missing"
        if status == "invalid_output":
            return "tool_invalid_output"
        if status == "tool_error":
            reason = str(observation.reason or "").lower()
            if "timeout" in reason or "timed out" in reason or "超时" in str(observation.reason or ""):
                return "tool_timeout"
            return "tool_runtime_error"
        if status == "need_clarification":
            return "ambiguous_user_request"
        return None


__all__ = ["FailureClassifier"]
