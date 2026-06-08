from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from .base import BaseToolAdapter, backend_tool_error
from .models import ToolExecutionResult


class LoadUserProfileInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    context: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    message: Optional[str] = None


class UserProfileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = None
    research_profile: Dict[str, Any] = Field(default_factory=dict)
    user_memory_summary: Any = None
    message: Optional[str] = None
    request_context: Dict[str, Any] = Field(default_factory=dict)


class LoadCandidatePapersInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    context: Dict[str, Any] = Field(default_factory=dict)


class CandidatePapersOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_papers: List[Dict[str, Any]] = Field(default_factory=list)


class GenerateRecommendationsInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    recommendation_profile: Dict[str, Any] = Field(default_factory=dict)
    candidate_papers: Optional[List[Dict[str, Any]]] = None
    user_id: Optional[str] = None
    message: Optional[str] = None


class RecommendationResultOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_result: Dict[str, Any] = Field(default_factory=dict)
    candidate_papers: Optional[List[Dict[str, Any]]] = None


class ValidateRecommendationsInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    recommendation_result: Dict[str, Any] = Field(default_factory=dict)


class ValidatedRecommendationsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)


class ExplainRecommendationsInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    validated_recommendations: Dict[str, Any] = Field(default_factory=dict)


class RecommendationAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_answer: str


class LoadUserProfileAdapter(BaseToolAdapter[LoadUserProfileInput, UserProfileOutput]):
    tool_name = "load_user_profile"
    input_model = LoadUserProfileInput
    output_model = UserProfileOutput

    def _run(self, tool_input: LoadUserProfileInput) -> UserProfileOutput:
        context = tool_input.context if isinstance(tool_input.context, Mapping) else {}
        profile = context.get("research_profile") if isinstance(context.get("research_profile"), Mapping) else {}
        return UserProfileOutput(
            user_id=tool_input.user_id,
            research_profile=dict(profile),
            user_memory_summary=context.get("user_memory_summary"),
            message=tool_input.message,
            request_context=dict(context),
        )


class LoadCandidatePapersAdapter(BaseToolAdapter[LoadCandidatePapersInput, CandidatePapersOutput]):
    tool_name = "load_candidate_papers"
    input_model = LoadCandidatePapersInput
    output_model = CandidatePapersOutput

    def _run(self, tool_input: LoadCandidatePapersInput) -> CandidatePapersOutput:
        context = tool_input.context if isinstance(tool_input.context, Mapping) else {}
        for key in ("papers", "last_papers", "candidate_papers"):
            papers = context.get(key)
            if isinstance(papers, list):
                return CandidatePapersOutput(candidate_papers=[dict(item) for item in papers if isinstance(item, Mapping)])
        return CandidatePapersOutput()


class GenerateRecommendationsAdapter(BaseToolAdapter[GenerateRecommendationsInput, RecommendationResultOutput]):
    tool_name = "generate_recommendations"
    input_model = GenerateRecommendationsInput
    output_model = RecommendationResultOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def execute(self, tool_input: GenerateRecommendationsInput) -> ToolExecutionResult:
        profile = tool_input.recommendation_profile or {}
        if not profile.get("research_profile") and not profile.get("user_memory_summary"):
            # 推荐工具允许基于 message 降级，但必须把降级原因放进 metadata，供 observer/replanner 判断。
            result = super().execute(tool_input)
            metadata = dict(result.metadata or {})
            metadata["profile_fallback"] = "message_driven_recommendation"
            return result.model_copy(update={"metadata": metadata})
        result = super().execute(tool_input)
        if result.ok and isinstance(result.data, RecommendationResultOutput) and result.data.tool_result and not bool(result.data.tool_result.get("ok", False)):
            backend_result = result.data.tool_result
            return result.model_copy(
                update={
                    "ok": False,
                    "error": backend_tool_error(
                        error_code=((backend_result.get("error") or {}).get("code") if isinstance(backend_result.get("error"), Mapping) else None) or "recommendation_failed",
                        message=str((backend_result.get("error") or {}).get("message") if isinstance(backend_result.get("error"), Mapping) else backend_result.get("summary") or "推荐工具失败"),
                        detail={"backend_error": backend_result.get("error")},
                        suggested_recovery="fallback_answer",
                        safe_debug={"backend_tool_name": "recommend_papers"},
                    ),
                }
            )
        return result

    def _run(self, tool_input: GenerateRecommendationsInput) -> RecommendationResultOutput:
        profile = tool_input.recommendation_profile or {}
        message = profile.get("message") or tool_input.message
        tool_result = self.invoke_backend_tool(
            "recommend_papers",
            user_id=tool_input.user_id or profile.get("user_id") or "",
            message=message,
            user_memory_summary=profile.get("user_memory_summary"),
            research_profile=profile.get("research_profile"),
            request_context=profile.get("request_context"),
        )
        return RecommendationResultOutput(
            recommendations=list((((tool_result or {}).get("data") or {}).get("recommendations") or [])),
            tool_result=dict(tool_result or {}),
            candidate_papers=tool_input.candidate_papers,
        )


class ValidateRecommendationsAdapter(BaseToolAdapter[ValidateRecommendationsInput, ValidatedRecommendationsOutput]):
    tool_name = "validate_recommendations"
    input_model = ValidateRecommendationsInput
    output_model = ValidatedRecommendationsOutput

    def _run(self, tool_input: ValidateRecommendationsInput) -> ValidatedRecommendationsOutput:
        recommendations = tool_input.recommendation_result.get("recommendations")
        return ValidatedRecommendationsOutput(ok=isinstance(recommendations, list) and len(recommendations) > 0, recommendations=list(recommendations or []))


class ExplainRecommendationsAdapter(BaseToolAdapter[ExplainRecommendationsInput, RecommendationAnswerOutput]):
    tool_name = "explain_recommendations"
    input_model = ExplainRecommendationsInput
    output_model = RecommendationAnswerOutput

    def _run(self, tool_input: ExplainRecommendationsInput) -> RecommendationAnswerOutput:
        recommendations = list((tool_input.validated_recommendations or {}).get("recommendations") or [])
        if not recommendations:
            return RecommendationAnswerOutput(final_answer="当前没有生成可用的推荐结果，建议先补充偏好或改成明确主题搜索。")
        titles = [str((item or {}).get("title") or "").strip() for item in recommendations[:3] if isinstance(item, Mapping)]
        return RecommendationAnswerOutput(final_answer=f"已生成推荐结果，可优先阅读：{'；'.join([title for title in titles if title]) or '候选论文列表'}。")
