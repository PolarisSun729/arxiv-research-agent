from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..clarification_analysis import build_clarification_diagnostic
from .base import BaseToolAdapter


class AnalyzeAmbiguityInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    goal: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    search_spec: Dict[str, Any] = Field(default_factory=dict)


class MissingInformationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = ""
    needs_clarification: bool = True
    missing_fields: List[str] = Field(default_factory=list)
    missing_field_details: List[Dict[str, Any]] = Field(default_factory=list)
    reason: str = "request_is_ambiguous"
    inferred_intent: str = "unclear"
    confidence: float = 0.0
    suggested_questions: List[str] = Field(default_factory=list)
    allow_default_continuation: bool = False
    default_values: Dict[str, Any] = Field(default_factory=dict)
    minimum_required_fields: List[str] = Field(default_factory=list)
    used_context_fields: List[str] = Field(default_factory=list)
    support_status: str = "supported"
    resolution_strategy: str = "ask_clarification"
    analysis_source: str = "rule_based_missing_information_analysis"
    analysis_mode: str = "deterministic_request_diagnostic"


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

    def execute(self, tool_input: AnalyzeAmbiguityInput) -> Any:
        result = super().execute(tool_input)
        metadata = dict(result.metadata or {})
        # 这里明确标记为规则诊断，避免 trace 读者误以为澄清问题来自自由生成。
        metadata.update(
            {
                "clarification_stage": "missing_information_analysis",
                "analysis_source": "rule_based_missing_information_analysis",
                "analysis_mode": "deterministic_request_diagnostic",
                "is_llm_backed": False,
            }
        )
        return result.model_copy(update={"metadata": metadata})

    def _run(self, tool_input: AnalyzeAmbiguityInput) -> MissingInformationOutput:
        # 澄清诊断需要同时看消息、上下文、身份和当前任务方向，不能只做关键词占位判断。
        diagnostic = build_clarification_diagnostic(
            message=tool_input.message,
            context=tool_input.context,
            goal=tool_input.goal,
            user_id=tool_input.user_id,
            search_spec=tool_input.search_spec,
        )
        return MissingInformationOutput.model_validate(diagnostic)


class GenerateClarificationAdapter(BaseToolAdapter[GenerateClarificationInput, FinalAnswerOutput]):
    tool_name = "generate_clarification"
    input_model = GenerateClarificationInput
    output_model = FinalAnswerOutput

    def execute(self, tool_input: GenerateClarificationInput) -> Any:
        result = super().execute(tool_input)
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "clarification_stage": "clarification_question_generation",
                "question_source": "template_clarification_response",
                "is_llm_backed": False,
            }
        )
        return result.model_copy(update={"metadata": metadata})

    def _run(self, tool_input: GenerateClarificationInput) -> FinalAnswerOutput:
        missing_information = dict(tool_input.missing_information or {})
        support_status = str(missing_information.get("support_status") or "supported")
        suggested_questions = list(missing_information.get("suggested_questions") or [])
        missing_fields = list(missing_information.get("missing_fields") or [])
        reason = str(missing_information.get("reason") or "request_is_ambiguous")
        allow_default_continuation = bool(missing_information.get("allow_default_continuation"))
        default_values = dict(missing_information.get("default_values") or {})
        inferred_intent = str(missing_information.get("inferred_intent") or "unclear")

        if support_status == "unsupported":
            return FinalAnswerOutput(
                final_answer="当前请求不在这个 agent 的支持范围内。你可以改成 arXiv 搜索、单篇论文问答、论文推荐、偏好更新，或处理待确认操作。"
            )

        if suggested_questions:
            primary_question = str(suggested_questions[0]).strip()
            if allow_default_continuation and default_values:
                default_sources = "、".join(
                    f"{key} 来自 {value.get('source')}"
                    for key, value in default_values.items()
                    if isinstance(value, dict) and value.get("source")
                )
                if default_sources:
                    return FinalAnswerOutput(final_answer=f"{primary_question} 如果你不补充，我也可以先按默认值继续。默认值来源：{default_sources}。")
            return FinalAnswerOutput(final_answer=primary_question)

        if missing_fields:
            joined = "、".join(str(item) for item in missing_fields)
            return FinalAnswerOutput(final_answer=f"当前请求还缺少关键信息：{joined}。请补充后我再继续。")

        if reason == "request_too_broad":
            return FinalAnswerOutput(final_answer="当前请求范围还太宽，直接执行会得到很发散的结果。请补充更具体的主题、论文对象或筛选条件。")

        return FinalAnswerOutput(final_answer=f"当前请求还不够明确，我推断你可能想做“{inferred_intent}”，但还需要你补充关键条件。")


class GenerateFallbackResponseAdapter(BaseToolAdapter[GenerateFallbackInput, FinalAnswerOutput]):
    tool_name = "generate_fallback_response"
    input_model = GenerateFallbackInput
    output_model = FinalAnswerOutput

    def _run(self, tool_input: GenerateFallbackInput) -> FinalAnswerOutput:
        message = str(tool_input.message or "").strip()
        fallback_reason = str(tool_input.fallback_reason or "").strip()
        # legacy/template fallback 代表主 planner 不可用，而不是用户请求本身一定不支持；
        # 回复文案必须区分这两类原因，避免把内部降级误报成用户意图错误。
        if fallback_reason and fallback_reason not in {"unsupported_goal", "unsupported_request"}:
            return FinalAnswerOutput(
                final_answer="当前规划链路临时降级，无法安全执行完整工具流程。请稍后重试，或把请求缩小为 arXiv 搜索、论文问答、推荐或偏好更新中的一种。"
            )
        if message:
            return FinalAnswerOutput(final_answer=f"当前请求暂不在该 agent 的支持范围内：{message}。建议改成 arXiv 搜索、论文问答、推荐或偏好更新。")
        return FinalAnswerOutput(final_answer="当前请求暂不在该 agent 的支持范围内。")
