from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..fallbacks import build_fallback_record
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
        normalized_message = _normalize_text(tool_input.message)
        topic_hint = _resolve_topic_hint(normalized_message, context)
        # Agent 侧负责把“当前轮的显式约束”整理成结构化上下文，真正的召回/排序仍交给 RecommendationService。
        request_context = {
            **dict(context),
            "query": normalized_message,
            "topic_hint": topic_hint,
            "positive_topics": _merge_text_terms(
                context.get("positive_topics"),
                context.get("recommendation_positive_topics"),
                [topic_hint] if topic_hint else [],
            ),
            "negative_topics": _merge_text_terms(
                context.get("negative_topics"),
                context.get("recommendation_negative_topics"),
                _extract_negative_preferences(normalized_message),
            ),
            "category_constraints": _merge_text_terms(
                context.get("category_constraints"),
                context.get("preferred_categories"),
                context.get("recommendation_categories"),
            ),
            "temporary_requirements": _merge_text_terms(
                context.get("temporary_requirements"),
                context.get("recommendation_constraints"),
            ),
            "recent_papers": _collect_recent_papers(context),
            # 缺少明确 user_id 且没有可复用画像时，后端要走冷启动路径而不是误用默认账号画像。
            "force_cold_start": not bool(tool_input.user_id) and not bool(profile) and not bool(context.get("user_memory_summary")),
            "user_id_provided": bool(tool_input.user_id),
        }
        return UserProfileOutput(
            user_id=tool_input.user_id,
            research_profile=dict(profile),
            user_memory_summary=context.get("user_memory_summary"),
            message=normalized_message,
            request_context=request_context,
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
        def _annotate_metadata(result: ToolExecutionResult, *, fallback_mode: Optional[str] = None) -> ToolExecutionResult:
            metadata = dict(result.metadata or {})
            # 推荐算法核心仍放在 RecommendationService，Adapter 只记录本轮是否走了画像降级或消息驱动兜底。
            metadata.update(
                {
                    "recommendation_stage": "agent_orchestration",
                    "agent_adapter": self.__class__.__name__,
                    "backend_tool_name": "recommend_papers",
                    "core_service": "RecommendationService.recommend_papers",
                    "core_execution_mode": "service_passthrough",
                    "is_algorithm_core_in_agent": False,
                }
            )
            if fallback_mode:
                metadata["fallback_mode"] = fallback_mode
                metadata["fallback_record"] = build_fallback_record(
                    fallback_mode,
                    stage="recommendation",
                    source=self.__class__.__name__,
                )
            return result.model_copy(update={"metadata": metadata})

        profile = tool_input.recommendation_profile or {}
        if not profile.get("research_profile") and not profile.get("user_memory_summary"):
            # 允许退化成“只靠当前请求”的冷启动推荐，但要显式留下 trace，方便 observer/replanner 判断质量。
            result = super().execute(tool_input)
            result = _annotate_metadata(result, fallback_mode="message_driven_recommendation")
            metadata = dict(result.metadata or {})
            metadata["profile_fallback"] = "message_driven_recommendation"
            return result.model_copy(update={"metadata": metadata})

        result = super().execute(tool_input)
        if result.ok and isinstance(result.data, RecommendationResultOutput) and result.data.tool_result and not bool(result.data.tool_result.get("ok", False)):
            backend_result = result.data.tool_result
            result = result.model_copy(
                update={
                    "ok": False,
                    "error": backend_tool_error(
                        error_code=((backend_result.get("error") or {}).get("code") if isinstance(backend_result.get("error"), Mapping) else None) or "recommendation_failed",
                        message=str((backend_result.get("error") or {}).get("message") if isinstance(backend_result.get("error"), Mapping) else backend_result.get("summary") or "推荐工具执行失败"),
                        detail={"backend_error": backend_result.get("error")},
                        suggested_recovery="fallback_answer",
                        safe_debug={"backend_tool_name": "recommend_papers"},
                    ),
                }
            )
        return _annotate_metadata(result)

    def _run(self, tool_input: GenerateRecommendationsInput) -> RecommendationResultOutput:
        profile = tool_input.recommendation_profile or {}
        message = profile.get("message") or tool_input.message
        request_context = dict(profile.get("request_context") or {})
        tool_result = self.invoke_backend_tool(
            "recommend_papers",
            user_id=tool_input.user_id or profile.get("user_id") or "",
            message=message,
            topic_hint=request_context.get("topic_hint"),
            user_memory_summary=profile.get("user_memory_summary"),
            research_profile=profile.get("research_profile"),
            request_context=request_context,
            candidate_papers=tool_input.candidate_papers,
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
        return ValidatedRecommendationsOutput(
            ok=isinstance(recommendations, list) and len(recommendations) > 0,
            recommendations=list(recommendations or []),
        )


class ExplainRecommendationsAdapter(BaseToolAdapter[ExplainRecommendationsInput, RecommendationAnswerOutput]):
    tool_name = "explain_recommendations"
    input_model = ExplainRecommendationsInput
    output_model = RecommendationAnswerOutput

    def _run(self, tool_input: ExplainRecommendationsInput) -> RecommendationAnswerOutput:
        recommendations = list((tool_input.validated_recommendations or {}).get("recommendations") or [])
        if not recommendations:
            return RecommendationAnswerOutput(final_answer="当前没有生成可用的推荐结果，建议先补充偏好或改成明确主题搜索。")

        lines: List[str] = []
        for item in recommendations[:3]:
            if not isinstance(item, Mapping):
                continue
            title = str(item.get("title") or "").strip() or "未命名论文"
            reason = str(
                item.get("recommendation_explanation")
                or item.get("match_reason")
                or item.get("personalized_reason")
                or ""
            ).strip()
            lines.append(f"- {title}" + (f"：{reason}" if reason else ""))

        if not lines:
            return RecommendationAnswerOutput(final_answer="已生成推荐结果，但当前缺少可展示的推荐解释。")
        return RecommendationAnswerOutput(final_answer="我结合你的长期兴趣、当前请求和最近交互整理了这些推荐：\n" + "\n".join(lines))


def _normalize_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _normalize_term(value: Any) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text or None


def _merge_text_terms(*sources: Any) -> List[str]:
    merged: List[str] = []
    seen = set()
    for source in sources:
        values = source if isinstance(source, list) else [source]
        for value in values:
            term = _normalize_term(value)
            if not term:
                continue
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(term)
    return merged


def _resolve_topic_hint(message: Optional[str], context: Mapping[str, Any]) -> Optional[str]:
    explicit = _normalize_text(context.get("recommendation_topic") or context.get("topic_hint"))
    if explicit:
        return explicit
    text = str(message or "").strip()
    if not text:
        return None
    normalized = re.sub(r"^(请|帮我|麻烦|想要|给我)?(推荐|找|搜|看看|来点|整点)(一下|一些|几篇|几篇论文|论文)?", "", text).strip()
    normalized = re.sub(r"(方向|领域|相关|方面)(的)?论文.*$", "", normalized).strip("：:，,。！？!? ")
    return normalized or text


def _extract_negative_preferences(message: Optional[str]) -> List[str]:
    text = str(message or "").strip()
    if not text:
        return []
    matches = re.findall(r"(?:不要|别|排除|避开|not about|without)\s*([^，。,;；!?！？]{1,24})", text, flags=re.IGNORECASE)
    return _merge_text_terms(matches)


def _collect_recent_papers(context: Mapping[str, Any]) -> List[Dict[str, Any]]:
    recent_papers: List[Dict[str, Any]] = []
    for key in ("last_papers", "papers", "candidate_papers"):
        papers = context.get(key)
        if isinstance(papers, list):
            for item in papers:
                if isinstance(item, Mapping):
                    recent_papers.append(dict(item))
            if recent_papers:
                break
    selected_paper = context.get("selected_paper")
    if isinstance(selected_paper, Mapping):
        selected_id = str(selected_paper.get("arxiv_id") or selected_paper.get("id") or "").strip().lower()
        if selected_id and all(str(item.get("arxiv_id") or item.get("id") or "").strip().lower() != selected_id for item in recent_papers):
            recent_papers.insert(0, dict(selected_paper))
    return recent_papers[:8]
