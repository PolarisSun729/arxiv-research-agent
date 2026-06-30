"""Profile-aware planner：把科研任务画像编译为现有执行器可消费的 PlanDraft。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .schemas import Goal, PlanDraft, PlanDraftStep, PlannerContext, StepPolicy, ToolCandidate, ToolSpec
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
            "artifact_plan": [],
            "evidence_tool_mapping": [],
            "unmet_evidence_requirements": [],
            "evidence_quality_reservations": {},
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
        artifact_plan = _default_artifact_plan_for_task(task_type)
        evidence_mapping = _map_artifacts_to_tools(artifact_plan)
        unmet_evidence = _unmet_evidence_requirements(artifact_plan, planner_context)

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
            if _artifact_plan_needs_evidence(artifact_plan, "user_profile_evidence"):
                unmet_evidence.append({"evidence_type": "user_profile_evidence", "reason": reason, "fallback": "generic_priority"})

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
        return self._attach_profile_metadata(draft, profile, artifact_plan, evidence_mapping, unmet_evidence)

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
        artifact_plan = _default_artifact_plan_for_task("single_paper_deep_read")
        evidence_mapping = _map_artifacts_to_tools(artifact_plan)
        unmet_evidence = _unmet_evidence_requirements(artifact_plan, planner_context)

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
        return self._attach_profile_metadata(draft, profile, artifact_plan, evidence_mapping, unmet_evidence)

    def _attach_profile_metadata(
        self,
        draft: PlanDraft,
        profile: Mapping[str, Any],
        artifact_plan: Sequence[Mapping[str, Any]],
        evidence_mapping: Sequence[Mapping[str, Any]],
        unmet_evidence: Sequence[Mapping[str, Any]],
    ) -> PlanDraft:
        quality_reservations = _evidence_quality_reservations(artifact_plan)
        metadata = dict(draft.metadata or {})
        metadata.update(
            {
                "source": PROFILE_AWARE_TASK_PLANNER_SOURCE,
                "research_task_profile": _compact_profile_for_debug(profile),
                "profile_aware_plan": {
                    "research_task_type": profile.get("research_task_type"),
                    "profile_source": profile.get("source"),
                    "classification_basis": profile.get("classification_basis"),
                    "artifact_plan": list(artifact_plan or []),
                    "evidence_tool_mapping": list(evidence_mapping or []),
                    "unmet_evidence_requirements": list(unmet_evidence or []),
                    # 第一版只预留质量字段，执行期仍由现有 Observer/Replanner 按工具结果推进。
                    "evidence_quality_reservations": quality_reservations,
                },
            }
        )
        self.last_debug.update(
            {
                "artifact_plan": list(artifact_plan or []),
                "evidence_tool_mapping": list(evidence_mapping or []),
                "unmet_evidence_requirements": list(unmet_evidence or []),
                "evidence_quality_reservations": quality_reservations,
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
    for key in ("source", "research_task_profile", "profile_aware_plan"):
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


def _default_artifact_plan_for_task(task_type: str) -> List[Dict[str, Any]]:
    """定义 task -> artifact -> evidence 的默认路线；这里只描述规划语义，不直接驱动执行。"""
    plans: Dict[str, List[Dict[str, Any]]] = {
        "direction_exploration": [
            {"artifact": "topic_terms", "evidence_types": ["metadata"], "purpose": "确定主题词、时间范围和检索约束"},
            {"artifact": "candidate_paper_set", "evidence_types": ["metadata", "abstract"], "purpose": "形成可筛选的候选论文集合"},
            {"artifact": "representative_paper_set", "evidence_types": ["metadata", "abstract"], "purpose": "挑选能代表方向分支的论文"},
            {"artifact": "direction_overview", "evidence_types": ["metadata", "abstract"], "purpose": "总结方向脉络和近期关注点"},
        ],
        "multi_paper_comparison": [
            {"artifact": "candidate_paper_set", "evidence_types": ["metadata", "abstract"], "purpose": "收集可比较候选论文"},
            {"artifact": "representative_paper_set", "evidence_types": ["metadata", "abstract"], "purpose": "筛出代表论文，避免比较对象过散"},
            {"artifact": "method_cards", "evidence_types": ["abstract", "method_chunk"], "purpose": "抽取每篇论文的方法要点"},
            {"artifact": "comparison_dimensions", "evidence_types": ["metadata", "method_chunk", "experiment_chunk"], "purpose": "确定统一比较维度"},
            {"artifact": "comparison_matrix", "evidence_types": ["metadata", "abstract", "method_chunk", "experiment_chunk", "table_evidence"], "purpose": "形成可解释的对比矩阵"},
        ],
        "single_paper_deep_read": [
            {"artifact": "paper_structure", "evidence_types": ["metadata", "full_text_chunk"], "purpose": "识别论文结构和章节边界"},
            {"artifact": "method_explanation", "evidence_types": ["method_chunk", "figure_evidence"], "purpose": "解释核心方法和模型设计"},
            {"artifact": "experiment_setup", "evidence_types": ["experiment_chunk", "table_evidence"], "purpose": "梳理实验设置、指标和数据集"},
            {"artifact": "contributions_and_limitations", "evidence_types": ["abstract", "method_chunk", "experiment_chunk"], "purpose": "归纳贡献、局限和适用边界"},
        ],
        "reading_planning": [
            {"artifact": "candidate_paper_set", "evidence_types": ["metadata", "abstract"], "purpose": "收集可读论文候选"},
            {"artifact": "profile_match", "evidence_types": ["metadata", "abstract", "user_profile_evidence"], "purpose": "评估候选与用户画像的匹配度"},
            {"artifact": "difficulty_priority", "evidence_types": ["metadata", "abstract"], "purpose": "判断阅读难度和优先级"},
            {"artifact": "reading_order", "evidence_types": ["metadata", "abstract", "user_profile_evidence"], "purpose": "生成循序渐进的阅读顺序"},
        ],
        "research_gap_analysis": [
            {"artifact": "candidate_paper_set", "evidence_types": ["metadata", "abstract"], "purpose": "收集已有工作"},
            {"artifact": "existing_method_categories", "evidence_types": ["abstract", "method_chunk"], "purpose": "归类现有方法路线"},
            {"artifact": "limitation_evidence", "evidence_types": ["abstract", "method_chunk", "experiment_chunk", "table_evidence"], "purpose": "定位局限和失败条件"},
            {"artifact": "uncovered_questions", "evidence_types": ["metadata", "abstract", "experiment_chunk"], "purpose": "识别尚未覆盖的问题"},
            {"artifact": "potential_directions", "evidence_types": ["metadata", "abstract", "method_chunk"], "purpose": "形成潜在研究方向"},
        ],
    }
    return [dict(item) for item in plans.get(task_type, [])]


def _map_artifacts_to_tools(artifact_plan: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    mapping_by_artifact = {
        "topic_terms": ["normalize_request", "build_arxiv_search_spec"],
        "candidate_paper_set": ["build_arxiv_search_spec", "search_arxiv", "validate_arxiv_results"],
        "representative_paper_set": ["validate_arxiv_results", "personalize_paper_results", "synthesize_arxiv_response"],
        "direction_overview": ["synthesize_arxiv_response"],
        "method_cards": ["search_arxiv", "validate_arxiv_results", "synthesize_arxiv_response"],
        "comparison_dimensions": ["validate_arxiv_results", "synthesize_arxiv_response"],
        "comparison_matrix": ["synthesize_arxiv_response"],
        "paper_structure": ["resolve_paper", "check_paper_index", "answer_paper_question"],
        "method_explanation": ["answer_paper_question", "assess_paper_qa_quality"],
        "experiment_setup": ["answer_paper_question", "assess_paper_qa_quality"],
        "contributions_and_limitations": ["answer_paper_question", "assess_paper_qa_quality"],
        "profile_match": ["personalize_paper_results", "synthesize_arxiv_response"],
        "difficulty_priority": ["validate_arxiv_results", "synthesize_arxiv_response"],
        "reading_order": ["personalize_paper_results", "synthesize_arxiv_response"],
        "existing_method_categories": ["search_arxiv", "validate_arxiv_results", "synthesize_arxiv_response"],
        "limitation_evidence": ["validate_arxiv_results", "synthesize_arxiv_response"],
        "uncovered_questions": ["validate_arxiv_results", "synthesize_arxiv_response"],
        "potential_directions": ["synthesize_arxiv_response"],
    }
    result: List[Dict[str, Any]] = []
    for item in list(artifact_plan or []):
        artifact = str(item.get("artifact") or "").strip()
        result.append(
            {
                "artifact": artifact,
                "evidence_types": list(item.get("evidence_types") or []),
                "tool_steps": list(mapping_by_artifact.get(artifact, ["synthesize_arxiv_response"])),
            }
        )
    return result


def _unmet_evidence_requirements(artifact_plan: Sequence[Mapping[str, Any]], planner_context: PlannerContext) -> List[Dict[str, Any]]:
    unmet: List[Dict[str, Any]] = []
    if _artifact_plan_needs_evidence(artifact_plan, "user_profile_evidence") and not _planner_has_profile_context(planner_context):
        unmet.append({"evidence_type": "user_profile_evidence", "reason": "user profile is unavailable in planner context", "fallback": "generic_direction_or_priority"})
    indexed_evidence = {"full_text_chunk", "method_chunk", "experiment_chunk", "table_evidence", "figure_evidence"}
    if any(evidence in indexed_evidence for item in artifact_plan for evidence in list(item.get("evidence_types") or [])):
        qa_result = planner_context.paper_qa_result if isinstance(planner_context.paper_qa_result, Mapping) else {}
        if not qa_result:
            # 正文、图表类证据需要 Paper QA 索引或后续回答工具确认；这里先记录缺口，避免 planner 假装证据已满足。
            unmet.append({"evidence_type": "indexed_paper_evidence", "reason": "paper index evidence is not observed at planning time", "fallback": "check_paper_index_then_observe_quality"})
    return unmet


def _evidence_quality_reservations(artifact_plan: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("artifact") or "unknown"): {
            "status": "not_observed_yet",
            "observer_reserved": True,
            "expected_evidence_types": list(item.get("evidence_types") or []),
        }
        for item in list(artifact_plan or [])
    }


def _artifact_plan_needs_evidence(artifact_plan: Sequence[Mapping[str, Any]], evidence_type: str) -> bool:
    return any(evidence_type in set(item.get("evidence_types") or []) for item in list(artifact_plan or []))


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