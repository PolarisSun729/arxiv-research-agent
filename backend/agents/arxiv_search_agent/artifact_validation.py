"""Artifact / Evidence Plan 的本地校验与能力对齐。

该模块位于 ArtifactEvidencePlan 与 Tool Planner 之间：它不调用 LLM，也不执行工具；
只负责确认科研中间产物契约合法、证据需求能否被当前系统能力覆盖，以及工具步骤应回写哪个 artifact。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .schemas import (
    Artifact,
    ArtifactEvidencePlan,
    ArtifactPlanValidationIssue,
    ArtifactPlanValidationReport,
    ArtifactProgressReservation,
    ArtifactStepMapping,
    CapabilityStatus,
    EvidenceCapabilityAlignment,
    EvidenceRequirement,
    PlannerContext,
)
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


_INDEX_EVIDENCE_TYPES = {
    "paper_chunk",
    "method_section",
    "experiment_section",
    "result_section",
    "limitation_section",
    "table",
    "figure",
}

_NO_FALLBACK_POLICIES = {"", "none", None}


class ArtifactPlanValidator:
    """对 ArtifactEvidencePlan 做本地、可复现的 guard 校验。"""

    def validate(
        self,
        plan: ArtifactEvidencePlan,
        *,
        planner_context: Optional[PlannerContext] = None,
        step_mapping: Optional[Sequence[ArtifactStepMapping]] = None,
    ) -> ArtifactPlanValidationReport:
        artifacts = list(plan.artifacts or [])
        artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
        issues: List[ArtifactPlanValidationIssue] = []

        self._check_artifact_graph(artifacts, issues)
        self._check_required_evidence(artifacts, issues)
        self._check_task_consistency(plan, planner_context, issues)
        self._check_budget(plan, issues)
        if step_mapping is not None:
            self._check_acquisition_mapping(artifacts, step_mapping, issues)

        final_artifact_ids = _final_artifact_ids(artifacts)
        if not final_artifact_ids:
            issues.append(
                _issue(
                    "artifact_plan_missing_final_artifact",
                    "error",
                    "Artifact plan 必须至少有一个没有下游依赖的 final artifact。",
                )
            )

        checked_evidence_ids = [
            requirement.requirement_id
            for artifact in artifacts
            for requirement in list(artifact.evidence_requirements or [])
        ]
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        status = "failed" if error_count else ("warning" if warning_count else "passed")
        return ArtifactPlanValidationReport(
            status=status,
            issue_count=len(issues),
            error_count=error_count,
            warning_count=warning_count,
            issues=issues,
            checked_artifact_ids=list(artifacts_by_id.keys()),
            checked_evidence_ids=checked_evidence_ids,
            final_artifact_ids=final_artifact_ids,
            validator_stack=[
                "ArtifactEvidencePlan.model_validate",
                "ArtifactPlanValidator.graph",
                "ArtifactPlanValidator.task_consistency",
                "ArtifactPlanValidator.budget",
                "ArtifactPlanValidator.acquisition_mapping",
            ],
        )

    def _check_artifact_graph(self, artifacts: Sequence[Artifact], issues: List[ArtifactPlanValidationIssue]) -> None:
        seen: Set[str] = set()
        artifact_ids = {artifact.artifact_id for artifact in artifacts}
        for artifact in artifacts:
            if artifact.artifact_id in seen:
                issues.append(_issue("duplicate_artifact_id", "error", f"重复 artifact_id: {artifact.artifact_id}", artifact))
            seen.add(artifact.artifact_id)
            for dependency in list(artifact.depends_on or []):
                if dependency not in artifact_ids:
                    issues.append(
                        _issue(
                            "artifact_dependency_missing",
                            "error",
                            f"{artifact.artifact_id} 依赖不存在的 artifact {dependency}",
                            artifact,
                        )
                    )
        if _has_cycle({artifact.artifact_id: list(artifact.depends_on or []) for artifact in artifacts}):
            issues.append(_issue("artifact_dependency_cycle", "error", "Artifact 依赖图不能包含环。"))

    def _check_required_evidence(self, artifacts: Sequence[Artifact], issues: List[ArtifactPlanValidationIssue]) -> None:
        for artifact in artifacts:
            requirements = list(artifact.evidence_requirements or [])
            if artifact.required and not requirements:
                issues.append(
                    _issue(
                        "required_artifact_missing_evidence",
                        "error",
                        f"required artifact {artifact.artifact_id} 必须声明 evidence_requirements。",
                        artifact,
                    )
                )
            for requirement in requirements:
                if requirement.target_artifact_id != artifact.artifact_id:
                    issues.append(
                        _issue(
                            "evidence_target_mismatch",
                            "error",
                            f"{requirement.requirement_id} 指向 {requirement.target_artifact_id}，但被挂在 {artifact.artifact_id} 下。",
                            artifact,
                            requirement,
                        )
                    )
                if requirement.required and not list(requirement.target_fields or []):
                    issues.append(
                        _issue(
                            "required_evidence_missing_target_fields",
                            "error",
                            f"{requirement.requirement_id} 必须绑定 artifact 的目标字段。",
                            artifact,
                            requirement,
                        )
                    )
                if requirement.required and not list(requirement.coverage_criteria or []):
                    issues.append(
                        _issue(
                            "required_evidence_missing_coverage",
                            "error",
                            f"{requirement.requirement_id} 必须声明覆盖标准。",
                            artifact,
                            requirement,
                        )
                    )
                if requirement.required and requirement.min_evidence_count <= 0:
                    issues.append(
                        _issue(
                            "required_evidence_zero_min_count",
                            "warning",
                            f"{requirement.requirement_id} 是 required evidence，但 min_evidence_count 为 0。",
                            artifact,
                            requirement,
                        )
                    )
                if requirement.required and requirement.capability_status in {"blocked", "unsupported"}:
                    severity = "error" if requirement.fallback_policy in _NO_FALLBACK_POLICIES else "warning"
                    issues.append(
                        _issue(
                            "required_evidence_not_acquirable",
                            severity,
                            f"{requirement.requirement_id} 当前能力状态为 {requirement.capability_status}。",
                            artifact,
                            requirement,
                        )
                    )

    def _check_task_consistency(
        self,
        plan: ArtifactEvidencePlan,
        planner_context: Optional[PlannerContext],
        issues: List[ArtifactPlanValidationIssue],
    ) -> None:
        task_type = str(plan.research_task_type or "").strip()
        artifact_types = {artifact.artifact_type for artifact in list(plan.artifacts or [])}
        artifact_ids = {artifact.artifact_id for artifact in list(plan.artifacts or [])}
        has_candidate_context = _has_candidate_context(planner_context)

        if task_type == "single_paper_deep_read" and not _has_target_paper_context(planner_context):
            issues.append(
                _issue(
                    "single_paper_deep_read_missing_target",
                    "error",
                    "单篇深读必须有目标论文、候选论文上下文或可解析的论文引用线索。",
                )
            )
        if task_type in {"multi_paper_comparison", "reading_planning", "research_gap_analysis"}:
            if "candidate_papers" not in artifact_ids and not has_candidate_context:
                issues.append(
                    _issue(
                        "paper_set_task_missing_candidate_source",
                        "error",
                        f"{task_type} 必须有 candidate_paper_set 或可复用候选论文上下文。",
                    )
                )
        if task_type == "research_gap_analysis" and "gap_hypothesis_set" in artifact_types:
            gap_artifacts = [artifact for artifact in plan.artifacts if artifact.artifact_type == "gap_hypothesis_set"]
            has_index_evidence = any(
                requirement.evidence_type in _INDEX_EVIDENCE_TYPES
                for artifact in gap_artifacts
                for requirement in list(artifact.evidence_requirements or [])
            )
            if not has_index_evidence:
                issues.append(
                    _issue(
                        "gap_analysis_lacks_deep_evidence",
                        "warning",
                        "研究空白分析不能只依赖摘要级证据生成高置信 gap，应保留正文/结果/局限证据需求。",
                    )
                )
        if task_type == "personalized_recommendation" and not _has_profile_context(planner_context):
            has_generic_fallback = any(
                requirement.evidence_type == "user_profile" and requirement.fallback_policy == "generic_without_profile"
                for artifact in list(plan.artifacts or [])
                for requirement in list(artifact.evidence_requirements or [])
            )
            if not has_generic_fallback:
                issues.append(
                    _issue(
                        "personalized_recommendation_missing_profile_fallback",
                        "warning",
                        "个性化推荐缺少用户画像时必须声明非个性化降级路径。",
                    )
                )

    def _check_budget(self, plan: ArtifactEvidencePlan, issues: List[ArtifactPlanValidationIssue]) -> None:
        budget = plan.diagnostics.budget
        totals: Dict[str, int] = {}
        for artifact in list(plan.artifacts or []):
            for key, value in dict(artifact.budget_cost or {}).items():
                try:
                    totals[key] = totals.get(key, 0) + int(value)
                except (TypeError, ValueError):
                    issues.append(
                        _issue(
                            "artifact_budget_cost_invalid",
                            "warning",
                            f"{artifact.artifact_id} 的预算成本 {key}={value} 不是整数。",
                            artifact,
                        )
                    )

        limit_map = {
            "candidate_papers": budget.max_candidate_papers,
            "selected_papers": budget.max_selected_papers,
            "deep_read_papers": budget.max_deep_read_papers,
            "qa_calls": budget.max_qa_calls,
            "refinement_rounds": budget.max_refinement_rounds,
        }
        for key, limit in limit_map.items():
            if limit is None:
                continue
            used = int(totals.get(key, 0) or 0)
            if used > int(limit):
                issues.append(
                    _issue(
                        "artifact_budget_exceeded",
                        "error",
                        f"Artifact plan 预算超限：{key} 使用 {used}，上限 {limit}。",
                    )
                )

    def _check_acquisition_mapping(
        self,
        artifacts: Sequence[Artifact],
        step_mapping: Sequence[ArtifactStepMapping],
        issues: List[ArtifactPlanValidationIssue],
    ) -> None:
        mapped_requirement_ids = {
            str(item.target_evidence_id or "").strip()
            for item in list(step_mapping or [])
            if str(item.target_evidence_id or "").strip()
        }
        for artifact in artifacts:
            for requirement in list(artifact.evidence_requirements or []):
                if not requirement.required:
                    continue
                if requirement.requirement_id in mapped_requirement_ids:
                    continue
                if requirement.capability_status == "supported" and requirement.fallback_policy in _NO_FALLBACK_POLICIES:
                    issues.append(
                        _issue(
                            "supported_evidence_missing_tool_mapping",
                            "error",
                            f"{requirement.requirement_id} 已标记 supported，但没有任何工具步骤贡献该证据。",
                            artifact,
                            requirement,
                        )
                    )
                    continue
                if requirement.capability_status in {"requires_index", "requires_user_profile", "requires_confirmation", "degraded"}:
                    issues.append(
                        _issue(
                            "evidence_uses_fallback_without_direct_mapping",
                            "warning",
                            f"{requirement.requirement_id} 当前没有直接工具映射，将依赖 fallback 或后续 Replanner。",
                            artifact,
                            requirement,
                        )
                    )


class CapabilityAlignmentChecker:
    """把 EvidenceRequirement 对齐到当前 ToolRegistry 与 planner context 的能力边界。"""

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry

    def align(
        self,
        plan: ArtifactEvidencePlan,
        *,
        planner_context: PlannerContext,
        available_tool_names: Optional[Iterable[str]] = None,
    ) -> ArtifactEvidencePlan:
        registry_tool_names = {tool.tool_name for tool in self.tool_registry.list_tools()}
        visible_tool_names = {str(name or "").strip() for name in list(available_tool_names or []) if str(name or "").strip()}
        tool_names = registry_tool_names.union(visible_tool_names)
        alignments: List[EvidenceCapabilityAlignment] = []
        updated_artifacts: List[Artifact] = []

        for artifact in list(plan.artifacts or []):
            updated_requirements: List[EvidenceRequirement] = []
            for requirement in list(artifact.evidence_requirements or []):
                alignment = self._alignment_for_requirement(
                    artifact=artifact,
                    requirement=requirement,
                    planner_context=planner_context,
                    tool_names=tool_names,
                )
                alignments.append(alignment)
                updated_requirements.append(
                    requirement.model_copy(
                        update={
                            "capability_status": alignment.capability_status,
                            "fallback_policy": alignment.fallback_policy,
                            "confidence_impact": alignment.confidence_impact,
                        }
                    )
                )
            updated_artifacts.append(artifact.model_copy(update={"evidence_requirements": updated_requirements}))

        diagnostics = plan.diagnostics.model_copy(
            update={
                "capability_alignment": alignments,
                "capability_summary": _capability_summary(updated_artifacts),
                "blocked_evidence_requirements": [
                    alignment.model_dump(mode="json")
                    for alignment in alignments
                    if alignment.capability_status in {"blocked", "unsupported"} or alignment.blocked
                ],
            }
        )
        return plan.model_copy(update={"artifacts": updated_artifacts, "diagnostics": diagnostics})

    def _alignment_for_requirement(
        self,
        *,
        artifact: Artifact,
        requirement: EvidenceRequirement,
        planner_context: PlannerContext,
        tool_names: Set[str],
    ) -> EvidenceCapabilityAlignment:
        evidence_type = str(requirement.evidence_type)
        fallback_policy = requirement.fallback_policy
        confidence_impact = requirement.confidence_impact
        status: CapabilityStatus = requirement.capability_status
        acquisition_tools: List[str] = []
        reason = "kept template capability status"

        if evidence_type in {"metadata", "abstract"}:
            if _has_candidate_context(planner_context):
                status = "supported"
                acquisition_tools = ["planner_context.last_papers", "validate_arxiv_results"]
                reason = "已有候选论文上下文可直接提供 metadata/abstract 级证据"
            elif "search_arxiv" in tool_names:
                status = "supported"
                acquisition_tools = ["build_arxiv_search_spec", "search_arxiv", "validate_arxiv_results"]
                reason = "arXiv search 工具链可提供 metadata/abstract 级证据"
            elif artifact.target_scope == "paper" and "resolve_paper" in tool_names:
                status = "supported"
                acquisition_tools = ["resolve_paper"]
                reason = "目标论文解析工具可提供单篇论文元数据入口"
            else:
                status = "blocked" if requirement.required else "degraded"
                reason = "缺少搜索结果、候选上下文或目标论文解析能力"

        elif evidence_type in _INDEX_EVIDENCE_TYPES:
            if _has_paper_qa_result(planner_context):
                status = "supported"
                acquisition_tools = ["answer_paper_question", "assess_paper_qa_quality"]
                reason = "planner context 中已有 Paper QA 结果，可视为正文证据已观测"
            elif {"check_paper_index", "answer_paper_question"}.issubset(tool_names):
                status = "requires_index"
                acquisition_tools = ["check_paper_index", "answer_paper_question"]
                if "parse_and_index_paper" in tool_names:
                    acquisition_tools.append("parse_and_index_paper")
                reason = "正文、图表和段落证据需要 Paper QA 索引/RAG 路径确认"
            else:
                status = "degraded" if requirement.fallback_policy not in _NO_FALLBACK_POLICIES else "blocked"
                reason = "当前工具集合没有 Paper QA 索引/RAG 能力，只能降级或阻断"

        elif evidence_type == "user_profile":
            if _has_profile_context(planner_context):
                status = "supported"
                acquisition_tools = [
                    tool for tool in ("load_user_profile", "personalize_paper_results", "generate_recommendations") if tool in tool_names
                ]
                reason = "planner context 中已有用户画像或记忆摘要"
            else:
                status = "blocked" if requirement.required and requirement.fallback_policy in _NO_FALLBACK_POLICIES else "degraded"
                fallback_policy = "generic_without_profile"
                confidence_impact = "low"
                reason = "缺少用户画像，个性化证据只能走通用降级"

        elif evidence_type == "previous_context":
            if artifact.depends_on or _has_reusable_context(planner_context):
                status = "degraded"
                acquisition_tools = ["upstream_artifact", "synthesize_arxiv_response"]
                reason = "previous_context 是上游 artifact 的派生证据，规划期只能作为降级信号"
            else:
                status = "blocked" if requirement.required else "degraded"
                reason = "没有上游 artifact 或可复用上下文支撑 previous_context"

        else:
            status = "blocked" if requirement.required else "unsupported"
            reason = f"证据类型 {evidence_type} 不在当前能力边界内"

        blocked = status in {"blocked", "unsupported"}
        return EvidenceCapabilityAlignment(
            target_artifact_id=artifact.artifact_id,
            requirement_id=requirement.requirement_id,
            evidence_type=requirement.evidence_type,
            target_fields=list(requirement.target_fields or []),
            capability_status=status,
            reason=reason,
            acquisition_tools=acquisition_tools,
            fallback_policy=fallback_policy,
            confidence_impact=confidence_impact,
            blocked=blocked,
        )


def build_artifact_step_mapping(
    plan: ArtifactEvidencePlan,
    steps: Sequence[Any],
) -> List[ArtifactStepMapping]:
    """根据最终草稿步骤生成 artifact-step 映射。

    这里只映射真实存在的步骤；需要索引但当前计划没有 Paper QA 步骤的 evidence 会保持 unmapped，
    由 diagnostics/unmet_evidence_requirements 暴露给后续 Observer/Replanner，而不是伪造工具贡献。
    """

    step_records = [
        {
            "step_id": str(getattr(step, "step_id", "") or "").strip(),
            "tool_name": str(getattr(step, "tool_name", "") or "").strip(),
        }
        for step in list(steps or [])
    ]
    step_tool_names = {record["tool_name"] for record in step_records if record["tool_name"]}
    has_search_step = "search_arxiv" in step_tool_names
    mappings: List[ArtifactStepMapping] = []
    seen: Set[tuple[str, str, str]] = set()

    for record in step_records:
        step_id = record["step_id"]
        tool_name = record["tool_name"]
        if not step_id or not tool_name:
            continue
        for artifact in list(plan.artifacts or []):
            for requirement in list(artifact.evidence_requirements or []):
                if requirement.capability_status in {"blocked", "unsupported"}:
                    continue
                allowed_tools = _tools_for_requirement(artifact, requirement, has_search_step=has_search_step)
                if tool_name not in allowed_tools:
                    continue
                key = (step_id, artifact.artifact_id, requirement.requirement_id)
                if key in seen:
                    continue
                seen.add(key)
                mappings.append(
                    ArtifactStepMapping(
                        step_id=step_id,
                        tool_name=tool_name,
                        target_artifact_id=artifact.artifact_id,
                        target_evidence_id=requirement.requirement_id,
                        artifact_type=artifact.artifact_type,
                        evidence_type=requirement.evidence_type,
                        contribution_type=_contribution_type(tool_name, requirement.capability_status, has_search_step),
                        expected_artifact_update=_expected_update(artifact, requirement, tool_name),
                        capability_status=requirement.capability_status,
                    )
                )
    return mappings


def build_artifact_progress_reservations(
    plan: ArtifactEvidencePlan,
    step_mapping: Sequence[ArtifactStepMapping],
) -> List[ArtifactProgressReservation]:
    """为 Observer/Replanner 预留 artifact 进度槽位。"""

    mapped_by_artifact: Dict[str, List[ArtifactStepMapping]] = {}
    mapped_evidence_ids = {
        str(item.target_evidence_id or "").strip()
        for item in list(step_mapping or [])
        if str(item.target_evidence_id or "").strip()
    }
    for item in list(step_mapping or []):
        mapped_by_artifact.setdefault(item.target_artifact_id, []).append(item)

    reservations: List[ArtifactProgressReservation] = []
    for artifact in list(plan.artifacts or []):
        blocked_ids: List[str] = []
        missing_ids: List[str] = []
        degraded = False
        for requirement in list(artifact.evidence_requirements or []):
            if requirement.capability_status in {"blocked", "unsupported"}:
                blocked_ids.append(requirement.requirement_id)
                continue
            if requirement.capability_status in {"degraded", "requires_index", "requires_user_profile", "requires_confirmation"}:
                degraded = True
            if requirement.required and requirement.requirement_id not in mapped_evidence_ids:
                missing_ids.append(requirement.requirement_id)
        status = "blocked" if blocked_ids else ("degraded" if degraded else ("partial" if missing_ids else "pending"))
        reservations.append(
            ArtifactProgressReservation(
                artifact_id=artifact.artifact_id,
                artifact_type=artifact.artifact_type,
                status=status,
                mapped_step_ids=_dedupe([item.step_id for item in mapped_by_artifact.get(artifact.artifact_id, [])]),
                missing_evidence_ids=missing_ids,
                blocked_evidence_ids=blocked_ids,
            )
        )
    return reservations


def attach_artifact_tooling_diagnostics(
    plan: ArtifactEvidencePlan,
    *,
    validation_report: ArtifactPlanValidationReport,
    step_mapping: Sequence[ArtifactStepMapping],
    progress_reservations: Sequence[ArtifactProgressReservation],
) -> ArtifactEvidencePlan:
    """把 validator/alignment/tool mapping 统一回写到 ArtifactEvidencePlan diagnostics。"""

    diagnostics = plan.diagnostics.model_copy(
        update={
            "validation_report": validation_report,
            "artifact_step_mapping": list(step_mapping or []),
            "artifact_progress_reservations": list(progress_reservations or []),
        }
    )
    return plan.model_copy(update={"diagnostics": diagnostics})


def _tools_for_requirement(
    artifact: Artifact,
    requirement: EvidenceRequirement,
    *,
    has_search_step: bool,
) -> Set[str]:
    evidence_type = str(requirement.evidence_type)
    artifact_type = str(artifact.artifact_type)
    if evidence_type in {"metadata", "abstract"}:
        if artifact_type == "topic_term_set":
            return {"normalize_request", "build_arxiv_search_spec"}
        if artifact_type == "candidate_paper_set":
            return {"search_arxiv", "validate_arxiv_results"} if has_search_step else {"validate_arxiv_results"}
        if artifact_type == "selected_paper_set":
            return {"resolve_paper", "validate_arxiv_results", "personalize_paper_results", "synthesize_arxiv_response"}
        if artifact_type == "user_profile_match":
            return {"personalize_paper_results", "generate_recommendations"}
        return {"synthesize_arxiv_response", "answer_paper_question", "explain_recommendations"}
    if evidence_type in _INDEX_EVIDENCE_TYPES:
        return {"check_paper_index", "answer_paper_question", "assess_paper_qa_quality"}
    if evidence_type == "user_profile":
        return {"load_user_profile", "personalize_paper_results", "generate_recommendations"}
    if evidence_type == "previous_context":
        return {"synthesize_arxiv_response", "answer_paper_question", "assess_paper_qa_quality", "explain_recommendations"}
    return set()


def _contribution_type(tool_name: str, status: CapabilityStatus, has_search_step: bool) -> str:
    if tool_name == "validate_arxiv_results" and not has_search_step:
        return "context_reuse"
    if tool_name.startswith("validate") or tool_name.startswith("assess") or tool_name == "check_paper_index":
        return "validate_evidence" if status == "supported" else "quality_gate"
    if tool_name in {"search_arxiv", "answer_paper_question", "load_user_profile", "generate_recommendations"}:
        return "acquire_evidence"
    if tool_name in {"synthesize_arxiv_response", "explain_recommendations"}:
        return "synthesize_artifact"
    return "derive_artifact"


def _expected_update(artifact: Artifact, requirement: EvidenceRequirement, tool_name: str) -> str:
    fields = ", ".join(list(requirement.target_fields or [])[:4])
    return f"{tool_name} updates {artifact.artifact_id}.{fields or requirement.requirement_id}"


def _final_artifact_ids(artifacts: Sequence[Artifact]) -> List[str]:
    depended = {dependency for artifact in artifacts for dependency in list(artifact.depends_on or [])}
    return [artifact.artifact_id for artifact in artifacts if artifact.artifact_id not in depended]


def _issue(
    code: str,
    severity: str,
    message: str,
    artifact: Optional[Artifact] = None,
    requirement: Optional[EvidenceRequirement] = None,
) -> ArtifactPlanValidationIssue:
    return ArtifactPlanValidationIssue(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        target_artifact_id=artifact.artifact_id if artifact is not None else None,
        target_evidence_id=requirement.requirement_id if requirement is not None else None,
        evidence_type=requirement.evidence_type if requirement is not None else None,
        capability_status=requirement.capability_status if requirement is not None else None,
    )


def _has_cycle(dependencies_by_step: Mapping[str, Sequence[str]]) -> bool:
    adjacency: Dict[str, List[str]] = {step_id: [] for step_id in dependencies_by_step.keys()}
    for step_id, dependencies in dependencies_by_step.items():
        for dependency in list(dependencies or []):
            adjacency.setdefault(str(dependency), []).append(str(step_id))

    visited: Set[str] = set()
    stack: Set[str] = set()

    def dfs(node: str) -> bool:
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for child in adjacency.get(node, []):
            if dfs(child):
                return True
        stack.remove(node)
        return False

    return any(dfs(node) for node in adjacency.keys() if node not in visited)


def _capability_summary(artifacts: Sequence[Artifact]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for artifact in artifacts:
        for requirement in list(artifact.evidence_requirements or []):
            status = str(requirement.capability_status)
            summary[status] = summary.get(status, 0) + 1
    return summary


def _has_candidate_context(planner_context: Optional[PlannerContext]) -> bool:
    if planner_context is None:
        return False
    return bool(
        list(planner_context.last_papers or [])
        or planner_context.intermediate_results.get("papers")
        or planner_context.intermediate_results.get("last_papers")
    )


def _has_target_paper_context(planner_context: Optional[PlannerContext]) -> bool:
    if planner_context is None:
        return False
    return bool(
        planner_context.selected_paper
        or _has_candidate_context(planner_context)
        or _message_has_paper_hint(planner_context.raw_user_request)
    )


def _has_profile_context(planner_context: Optional[PlannerContext]) -> bool:
    if planner_context is None:
        return False
    for value in (planner_context.user_memory_summary, planner_context.research_profile):
        if isinstance(value, Mapping) and value:
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def _has_paper_qa_result(planner_context: Optional[PlannerContext]) -> bool:
    if planner_context is None:
        return False
    return isinstance(planner_context.paper_qa_result, Mapping) and bool(planner_context.paper_qa_result)


def _has_reusable_context(planner_context: Optional[PlannerContext]) -> bool:
    if planner_context is None:
        return False
    return bool(planner_context.intermediate_results or planner_context.reusable_outputs)


def _message_has_paper_hint(message: Optional[str]) -> bool:
    text = str(message or "").strip().lower()
    return bool(text and ("arxiv" in text or "http" in text or "论文" in text or "paper" in text))


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


__all__ = [
    "ArtifactPlanValidator",
    "CapabilityAlignmentChecker",
    "attach_artifact_tooling_diagnostics",
    "build_artifact_progress_reservations",
    "build_artifact_step_mapping",
]
