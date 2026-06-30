"""Profile-aware planner：把科研任务画像编译为现有执行器可消费的 PlanDraft。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .artifact_refinement import refine_artifact_evidence_plan
from .artifact_templates import (
    artifact_plan_needs_evidence,
    artifact_plan_projection,
    build_artifact_evidence_plan,
    evidence_quality_reservations,
    evidence_tool_mapping,
)
from .schemas import (
    ArtifactEvidencePlan,
    Goal,
    PlanDraft,
    PlanDraftStep,
    PlannerContext,
    StepPolicy,
    ToolCandidate,
    ToolSpec,
)
from .state import AgentState
from .tool_registry import ToolRegistry
from .tool_aware_planner import (
    RuleBasedToolAwarePlanBuilder,
    _binding_dict,
    _context_mapping_from_planner_context,
    _dedupe_tool_names,
    _get_selected_paper_hint,
    _has_candidate_paper_context,
    _message_has_paper_hint,
    _minimal_planner_context,
    _planner_context_used_fields,
    _planner_has_paper_target,
    _planner_has_profile_context,
)


PROFILE_AWARE_TASK_PLANNER_SOURCE = "profile_aware_task_planner"


class ProfileAwareResearchTaskPlanBuilder(RuleBasedToolAwarePlanBuilder):
    """把 Research Task Profile 编译成当前执行器仍可消费的 PlanDraft。

    该 builder 只接管“科研任务语义明确且本地可执行”的复杂请求；简单搜索、低置信分类和澄清型
    Profile 会直接让位给原有 Tool-Aware 规则 planner，避免为了科研语义层引入额外成本或错误步骤。
    """

    _COMPLEX_ARXIV_TASKS = {
        "direction_exploration",
        "multi_paper_comparison",
        "reading_planning",
        "research_gap_analysis",
    }

    def __init__(
        self,
        *,
        generation_service: Optional[Any] = None,
        enable_artifact_refinement: bool = False,
        refinement_timeout_seconds: int = 6,
        max_refinement_patches: int = 12,
    ) -> None:
        super().__init__()
        self.generation_service = generation_service
        self.enable_artifact_refinement = bool(enable_artifact_refinement)
        self.refinement_timeout_seconds = max(1, int(refinement_timeout_seconds or 6))
        self.max_refinement_patches = max(1, int(max_refinement_patches or 12))

    def build(
        self,
        goal: Goal,
        state: AgentState,
        candidate_tools: Sequence[ToolCandidate],
        tool_registry: ToolRegistry,
        planner_context: Optional[PlannerContext] = None,
    ) -> Optional[PlanDraft]:
        goal_type = str(goal.goal_type or "unsupported").strip() or "unsupported"
        planner_context = planner_context or _minimal_planner_context(goal, state, tool_registry)
        context = _context_mapping_from_planner_context(planner_context, state)
        candidate_names = _dedupe_tool_names([tool.tool_name for tool in list(candidate_tools or [])])
        tools_by_name = {
            tool_name: tool_registry.get(tool_name)
            for tool_name in candidate_names
            if tool_registry.get(tool_name) is not None
        }
        profile = _profile_mapping(planner_context.research_task_profile)
        self.last_debug = {
            "enabled": False,
            "skip_reason": None,
            "selected_steps": [],
            "skipped_steps": [],
            "skipped_tools": [],
            "candidate_tool_names": candidate_names,
            "planner_context_refs": list(planner_context.context_refs or []),
            "planner_context_used_fields": _planner_context_used_fields(planner_context),
            "research_task_profile": _compact_profile_for_debug(profile),
            "artifact_evidence_plan": {},
            "artifact_plan": [],
            "evidence_tool_mapping": [],
            "unmet_evidence_requirements": [],
            "evidence_quality_reservations": {},
            "artifact_refinement": {
                "enabled": self.enable_artifact_refinement,
                "llm_available": self.generation_service is not None,
            },
        }
        if not profile:
            return self._skip_profile("research_task_profile_missing")

        skip_reason = self._route_skip_reason(goal_type, profile, planner_context, state, context)
        if skip_reason:
            return self._skip_profile(skip_reason)

        research_task_type = str(profile.get("research_task_type") or "").strip()
        if goal_type == "arxiv_search" and research_task_type in self._COMPLEX_ARXIV_TASKS:
            draft = self._build_profile_aware_arxiv_search(goal, planner_context, profile, tools_by_name)
        elif goal_type == "paper_qa" and research_task_type == "single_paper_deep_read":
            draft = self._build_profile_aware_paper_qa(goal, state, context, planner_context, profile, tools_by_name)
        else:
            return self._skip_profile(f"unsupported_profile_goal_pair:{goal_type}:{research_task_type}")

        self.last_debug["enabled"] = True
        self.last_debug["selected_tools"] = list(draft.selected_tools or [])
        self.last_debug["draft_fallback_reason"] = draft.fallback_reason
        return draft

    def _route_skip_reason(
        self,
        goal_type: str,
        profile: Mapping[str, Any],
        planner_context: PlannerContext,
        state: AgentState,
        context: Mapping[str, Any],
    ) -> Optional[str]:
        research_task_type = str(profile.get("research_task_type") or "").strip()
        source = str(profile.get("source") or "").strip()
        readiness = str(profile.get("execution_readiness") or "").strip()
        confidence = _coerce_float(profile.get("confidence"), default=0.0)
        if source == "fallback_to_intent_route" or readiness == "fallback_intent_route":
            return "profile_fallback_to_intent_route"
        if readiness == "needs_user_clarification" or bool(profile.get("needs_clarification")):
            return "profile_needs_user_clarification"
        if confidence and confidence < 0.55:
            return f"profile_confidence_too_low:{confidence:.2f}"
        if research_task_type == "single_paper_deep_read" and not _planner_has_paper_target(planner_context, state):
            return "single_paper_deep_read_missing_target_paper"
        if research_task_type == "direction_exploration" and confidence < 0.7 and not list(profile.get("secondary_task_types") or []):
            # 方向探索很容易被普通搜索误触发；低置信且没有次任务时保留原 intent 路线，避免把轻量搜索升级成复杂综述。
            return "direction_exploration_not_confident_enough_for_profile_planning"
        if research_task_type == "multi_paper_comparison" and goal_type != "arxiv_search" and not _has_candidate_paper_context(context):
            return "multi_paper_comparison_without_candidate_route"
        return None

    def _build_profile_aware_arxiv_search(
        self,
        goal: Goal,
        planner_context: PlannerContext,
        profile: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        normalize = self._require_tool(tools_by_name, "normalize_request", required_tags={"search"})
        build_spec = self._require_tool(tools_by_name, "build_arxiv_search_spec", required_tags={"search"})
        search = self._require_tool(tools_by_name, "search_arxiv", required_tags={"search"})
        validate = self._require_tool(tools_by_name, "validate_arxiv_results", required_tags={"validate"})
        synthesize = self._require_tool(tools_by_name, "synthesize_arxiv_response", required_tags={"answer"})
        task_type = str(profile.get("research_task_type") or "direction_exploration").strip()
        artifact_evidence_plan = build_artifact_evidence_plan(_profile_with_task_type(profile, task_type), planner_context)
        artifact_evidence_plan = self._refine_artifact_evidence_plan(artifact_evidence_plan, profile, planner_context)

        steps: List[PlanDraftStep] = [
            self._draft_step(
                "normalize_request",
                normalize,
                action_type="write_state",
                output_key="topic_terms",
                reason="先把用户请求归一成主题术语和约束，后续候选集、代表集和综述产物都以这个科研对象为锚点。",
                input_bindings=[
                    _binding_dict("intent", source_type="state", source_key="intent"),
                    _binding_dict("message", source_type="state", source_key="message"),
                    _binding_dict("search_spec", source_type="search_spec"),
                ],
            ),
            self._draft_step(
                "build_arxiv_search_spec",
                build_spec,
                action_type="search",
                output_key="candidate_search_spec",
                reason="把主题术语编译成可执行检索规格，承担“候选论文集合”这一中间产物的证据入口职责。",
                depends_on=["normalize_request"],
                input_bindings=[_binding_dict("normalized_request", source_type="step_output", step_id="normalize_request")],
            ),
            self._draft_step(
                "search_arxiv",
                search,
                action_type="search",
                output_key="candidate_paper_set",
                reason="复杂科研任务先产生候选论文集合，而不是直接生成回答；后续比较、gap 或阅读顺序都依赖这个候选集。",
                depends_on=["build_arxiv_search_spec"],
                input_bindings=[_binding_dict("search_spec", source_type="step_output", step_id="build_arxiv_search_spec")],
                retry_policy=StepPolicy(policy_type="retry", mode="allow_search_relaxation", max_attempts=3) if search.can_retry else None,
            ),
            self._draft_step(
                "validate_arxiv_results",
                validate,
                action_type="validate",
                output_key="candidate_evidence_quality",
                reason="先评估候选集数量和可用性，给 Observer/Replanner 预留“证据不足也可补检索”的质量入口。",
                depends_on=["search_arxiv"],
                input_bindings=[_binding_dict("arxiv_results", source_type="step_output", step_id="search_arxiv")],
            ),
        ]

        personalize = self._optional_tool(tools_by_name, preferred_name="personalize_paper_results", required_tags={"personalize", "rerank"})
        synthesize_depends_on = ["validate_arxiv_results"]
        synthesize_bindings = [
            _binding_dict("arxiv_result_quality", source_type="step_output", step_id="validate_arxiv_results"),
            _binding_dict("arxiv_results", source_type="step_output", step_id="search_arxiv", required=False),
        ]
        if personalize and _planner_has_profile_context(planner_context):
            steps.append(
                self._draft_step(
                    "personalize_paper_results",
                    personalize,
                    action_type="rerank",
                    output_key="representative_paper_set",
                    reason="存在用户画像时才把候选集重排为代表论文集合，避免无画像时伪造个性化阅读依据。",
                    depends_on=["search_arxiv"],
                    input_bindings=[
                        _binding_dict("arxiv_results", source_type="step_output", step_id="search_arxiv"),
                        _binding_dict("user_memory_summary", source_type="context", source_key="user_memory_summary", required=False),
                        _binding_dict("research_profile", source_type="context", source_key="research_profile", required=False),
                    ],
                )
            )
            synthesize_depends_on.append("personalize_paper_results")
            synthesize_bindings.insert(0, _binding_dict("ranked_papers", source_type="step_output", step_id="personalize_paper_results", required=False))
        else:
            reason = "no user profile context; use validated candidate_paper_set as generic representative evidence"
            self._skip_step("personalize_paper_results", reason)
            if artifact_plan_needs_evidence(artifact_evidence_plan, "user_profile"):
                artifact_evidence_plan.diagnostics.unmet_evidence_requirements.append(
                    {
                        "evidence_type": "user_profile",
                        "reason": reason,
                        "fallback_policy": "generic_without_profile",
                        "capability_status": "requires_user_profile",
                    }
                )

        steps.append(
            self._draft_step(
                "synthesize_arxiv_response",
                synthesize,
                action_type="answer",
                output_key="final_answer",
                reason="最终回答要解释中间产物和证据来源，把搜索结果组织成科研任务需要的分析产物。",
                depends_on=synthesize_depends_on,
                input_bindings=synthesize_bindings,
            )
        )
        draft = self._make_draft(goal, task_type, steps)
        return self._attach_profile_metadata(draft, profile, artifact_evidence_plan)

    def _build_profile_aware_paper_qa(
        self,
        goal: Goal,
        state: AgentState,
        context: Mapping[str, Any],
        planner_context: PlannerContext,
        profile: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        resolve = self._require_tool(tools_by_name, "resolve_paper", required_tags={"retrieve"})
        if not (_get_selected_paper_hint(context) or _has_candidate_paper_context(context) or _message_has_paper_hint(state.message)):
            return self._fallback_draft(goal, "single_paper_deep_read", "single_paper_deep_read_missing_target_paper")
        check_index = self._require_tool(tools_by_name, "check_paper_index", required_tags={"validate", "retrieve"})
        answer = self._require_tool(tools_by_name, "answer_paper_question", required_tags={"answer"})
        assess_quality = self._require_tool(tools_by_name, "assess_paper_qa_quality", required_tags={"validate", "answer"})
        artifact_evidence_plan = build_artifact_evidence_plan(_profile_with_task_type(profile, "single_paper_deep_read"), planner_context)
        artifact_evidence_plan = self._refine_artifact_evidence_plan(artifact_evidence_plan, profile, planner_context)

        steps = [
            self._draft_step(
                "resolve_paper",
                resolve,
                action_type="retrieve",
                output_key="target_paper_ref",
                reason="单篇深读必须先解析唯一目标论文，避免把没有落地的引用线索直接交给 Paper QA。",
                input_bindings=[
                    _binding_dict("message", source_type="state", source_key="message"),
                    _binding_dict("selected_paper", source_type="context", source_key="selected_paper", required=False),
                    _binding_dict("context", source_type="state", source_key="context", required=False),
                ],
            ),
            self._draft_step(
                "check_paper_index",
                check_index,
                action_type="validate",
                output_key="paper_index_status",
                reason="深读依赖正文 chunk 证据；索引状态先显式检查，缺索引时仍由既有 Observer/Replanner 接管补建链路。",
                depends_on=["resolve_paper"],
                input_bindings=[_binding_dict("paper_ref", source_type="step_output", step_id="resolve_paper")],
            ),
            self._draft_step(
                "answer_paper_question",
                answer,
                action_type="answer",
                output_key="paper_qa_result",
                reason="用 Paper QA 产出结构、方法、实验、贡献和局限的深读回答，保持 output_key 兼容现有执行器与响应汇总。",
                depends_on=["resolve_paper", "check_paper_index"],
                input_bindings=[
                    _binding_dict("paper_ref", source_type="step_output", step_id="resolve_paper"),
                    _binding_dict("message", source_type="state", source_key="message"),
                    _binding_dict("qa_mode", source_type="literal", value="deep_read", required=False),
                ],
            ),
            self._draft_step(
                "assess_paper_qa_quality",
                assess_quality,
                action_type="validate",
                output_key="deep_read_evidence_quality",
                reason="深读回答不能只看工具是否成功，还要把检索证据质量写入后续可观测入口。",
                depends_on=["answer_paper_question"],
                input_bindings=[_binding_dict("paper_qa_result", source_type="step_output", step_id="answer_paper_question")],
            ),
        ]
        draft = self._make_draft(goal, "single_paper_deep_read", steps)
        return self._attach_profile_metadata(draft, profile, artifact_evidence_plan)

    def _refine_artifact_evidence_plan(
        self,
        artifact_evidence_plan: ArtifactEvidencePlan,
        profile: Mapping[str, Any],
        planner_context: PlannerContext,
    ) -> ArtifactEvidencePlan:
        # refinement 是模板 skeleton 之后的增强层；即使 LLM 未启用，也会通过本地 binder 生成最终 evidence requirements。
        return refine_artifact_evidence_plan(
            artifact_evidence_plan,
            profile=profile,
            planner_context=planner_context,
            generation_service=self.generation_service,
            enabled=self.enable_artifact_refinement,
            timeout_seconds=self.refinement_timeout_seconds,
            max_patches=self.max_refinement_patches,
        )

    def _attach_profile_metadata(
        self,
        draft: PlanDraft,
        profile: Mapping[str, Any],
        artifact_evidence_plan: ArtifactEvidencePlan,
    ) -> PlanDraft:
        artifact_evidence_payload = artifact_evidence_plan.model_dump(mode="json")
        artifact_plan = artifact_plan_projection(artifact_evidence_plan)
        evidence_mapping = evidence_tool_mapping(artifact_evidence_plan)
        unmet_evidence = list(artifact_evidence_plan.diagnostics.unmet_evidence_requirements or [])
        quality_reservations = evidence_quality_reservations(artifact_evidence_plan)
        planning_diagnostics = artifact_evidence_payload.get("diagnostics") or {}
        metadata = dict(draft.metadata or {})
        metadata.update(
            {
                "source": PROFILE_AWARE_TASK_PLANNER_SOURCE,
                "research_task_profile": _compact_profile_for_debug(profile),
                "artifact_evidence_plan": artifact_evidence_payload,
                "profile_aware_plan": {
                    "research_task_type": profile.get("research_task_type"),
                    "profile_source": profile.get("source"),
                    "classification_basis": profile.get("classification_basis"),
                    "artifact_evidence_plan": artifact_evidence_payload,
                    "artifact_plan": list(artifact_plan or []),
                    "evidence_tool_mapping": list(evidence_mapping or []),
                    "unmet_evidence_requirements": list(unmet_evidence or []),
                    "planning_diagnostics": planning_diagnostics,
                    # 第一版只预留质量字段，执行期仍由现有 Observer/Replanner 按工具结果推进。
                    "evidence_quality_reservations": quality_reservations,
                },
            }
        )
        self.last_debug.update(
            {
                "artifact_evidence_plan": artifact_evidence_payload,
                "artifact_plan": list(artifact_plan or []),
                "evidence_tool_mapping": list(evidence_mapping or []),
                "unmet_evidence_requirements": list(unmet_evidence or []),
                "evidence_quality_reservations": quality_reservations,
                "planning_diagnostics": planning_diagnostics,
                "artifact_refinement": {
                    "enabled": self.enable_artifact_refinement,
                    "llm_available": self.generation_service is not None,
                    "attempted": bool(planning_diagnostics.get("llm_refinement_attempted")),
                    "accepted_patch_count": len(planning_diagnostics.get("accepted_refinement_patches") or []),
                    "rejected_patch_count": len(planning_diagnostics.get("rejected_refinement_patches") or []),
                    "error": planning_diagnostics.get("llm_refinement_error"),
                },
            }
        )
        return draft.model_copy(update={"metadata": metadata})

    def _skip_profile(self, reason: str) -> None:
        self.last_debug["enabled"] = False
        self.last_debug["skip_reason"] = reason
        return None


def attach_profile_aware_metadata(plan: Any, draft: PlanDraft, builder_debug: Mapping[str, Any]) -> Any:
    """把科研产物规划挂到 ExecutablePlan.metadata，保持 PlanStep 结构不变。"""
    metadata = dict(getattr(plan, "metadata", None) or {})
    draft_metadata = dict(draft.metadata or {})
    for key in ("source", "research_task_profile", "artifact_evidence_plan", "profile_aware_plan"):
        if key in draft_metadata:
            metadata[key] = draft_metadata[key]
    metadata["profile_aware_used"] = bool(builder_debug.get("enabled"))
    return plan.model_copy(update={"metadata": metadata})


def _profile_mapping(profile: Any) -> Dict[str, Any]:
    if profile is None:
        return {}
    model_dump = getattr(profile, "model_dump", None)
    if callable(model_dump):
        profile = model_dump()
    return dict(profile) if isinstance(profile, Mapping) else {}


def _compact_profile_for_debug(profile: Mapping[str, Any]) -> Dict[str, Any]:
    if not profile:
        return {}
    return {
        "research_task_type": profile.get("research_task_type"),
        "secondary_task_types": list(profile.get("secondary_task_types") or []),
        "source": profile.get("source"),
        "confidence": profile.get("confidence"),
        "execution_readiness": profile.get("execution_readiness"),
        "needs_clarification": profile.get("needs_clarification"),
        "classification_basis": profile.get("classification_basis"),
        "arbitration_notes": list(profile.get("arbitration_notes") or []),
    }


def _profile_with_task_type(profile: Mapping[str, Any], task_type: str) -> Dict[str, Any]:
    payload = dict(profile or {})
    payload["research_task_type"] = task_type
    return payload


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "PROFILE_AWARE_TASK_PLANNER_SOURCE",
    "ProfileAwareResearchTaskPlanBuilder",
    "attach_profile_aware_metadata",
]
