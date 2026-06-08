from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base import BaseToolAdapter


class AnalyzeAmbiguityInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""


class MissingInformationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = ""
    missing_fields: List[str] = Field(default_factory=list)
    reason: str = "request_is_ambiguous"


class GenerateClarificationInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    missing_information: Dict[str, Any] = Field(default_factory=dict)


class GenerateFallbackInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""
    fallback_reason: Optional[str] = None


class FinalAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str


class AnalyzeAmbiguityAdapter(BaseToolAdapter[AnalyzeAmbiguityInput, MissingInformationOutput]):
    tool_name = "analyze_ambiguity"
    input_model = AnalyzeAmbiguityInput
    output_model = MissingInformationOutput

    def _run(self, tool_input: AnalyzeAmbiguityInput) -> MissingInformationOutput:
        return MissingInformationOutput(message=str(tool_input.message or "").strip(), missing_fields=["topic"])


class GenerateClarificationAdapter(BaseToolAdapter[GenerateClarificationInput, FinalAnswerOutput]):
    tool_name = "generate_clarification"
    input_model = GenerateClarificationInput
    output_model = FinalAnswerOutput

    def _run(self, tool_input: GenerateClarificationInput) -> FinalAnswerOutput:
        missing_fields = list((tool_input.missing_information or {}).get("missing_fields") or [])
        if missing_fields:
            return FinalAnswerOutput(final_answer=f"我还缺少关键信息：{', '.join(str(item) for item in missing_fields)}。请补充后我再继续。")
        return FinalAnswerOutput(final_answer="当前请求还不够明确，请补充目标主题、论文或操作对象。")


class GenerateFallbackResponseAdapter(BaseToolAdapter[GenerateFallbackInput, FinalAnswerOutput]):
    tool_name = "generate_fallback_response"
    input_model = GenerateFallbackInput
    output_model = FinalAnswerOutput

    def _run(self, tool_input: GenerateFallbackInput) -> FinalAnswerOutput:
        message = str(tool_input.message or "").strip()
        if message:
            return FinalAnswerOutput(final_answer=f"当前请求暂不在该 agent 的支持范围内：{message}。建议改成 arXiv 搜索、论文问答、推荐或偏好更新。")
        return FinalAnswerOutput(final_answer="当前请求暂不在该 agent 的支持范围内。")


class RequestConfirmationInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    pending_action: Optional[Dict[str, Any]] = None
    pending_state: Dict[str, Any] = Field(default_factory=dict)


class ConfirmationStatusOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    pending_action: Optional[Dict[str, Any]] = None


class RequestConfirmationAdapter(BaseToolAdapter[RequestConfirmationInput, ConfirmationStatusOutput]):
    tool_name = "request_confirmation"
    input_model = RequestConfirmationInput
    output_model = ConfirmationStatusOutput

    def _run(self, tool_input: RequestConfirmationInput) -> ConfirmationStatusOutput:
        status = "approved" if tool_input.pending_state.get("status") == "approved" else "waiting_confirmation"
        return ConfirmationStatusOutput(status=status, pending_action=tool_input.pending_action)
