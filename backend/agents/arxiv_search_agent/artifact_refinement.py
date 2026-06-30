"""Artifact / Evidence Plan 的受控 LLM refinement 与证据绑定。

该模块保持两条边界：
1. LLM 只能输出受限 patch，不能直接生成完整 ArtifactEvidencePlan；
2. EvidenceRequirement 始终由本地规则根据 artifact 字段生成，LLM 只能影响优先级、字段和约束。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, get_args

from pydantic import ValidationError

from .artifact_templates import refresh_artifact_evidence_diagnostics
from .schemas import (
    Artifact,
    ArtifactEvidencePlan,
    ArtifactPlanningBudget,
    ArtifactRefinementPatch,
    ArtifactRefinementPatchSet,
    ArtifactType,
    CapabilityStatus,
    ConfidenceImpact,
    EvidenceFallbackPolicy,
    EvidencePriority,
    EvidenceRequirement,
    EvidenceSection,
    EvidenceSourceScope,
    EvidenceType,
    PlannerContext,
    PlanningDiagnostics,
)


ARTIFACT_REFINEMENT_SERVICE_SOURCE = "artifact_refinement_service"
EVIDENCE_REQUIREMENT_BINDER_SOURCE = "evidence_requirement_binder"

_INDEX_EVIDENCE_TYPES = {
    "paper_chunk",
    "method_section",
    "experiment_section",
    "result_section",
    "limitation_section",
    "table",
    "figure",
}

_KNOWN_TOOL_NAMES = {
    "normalize_request",
    "build_arxiv_search_spec",
    "search_arxiv",
    "validate_arxiv_results",
    "personalize_paper_results",
    "synthesize_arxiv_response",
    "resolve_paper",
    "check_paper_index",
    "answer_paper_question",
    "assess_paper_qa_quality",
    "generate_recommendations",
    "update_preference_store",
}

_BUDGET_KEYS = set(ArtifactPlanningBudget.model_fields.keys())


class EvidenceRequirementBinder:
    """把 artifact 字段绑定成 Observer/RAG 可消费的 EvidenceRequirement。

    Binder 是本地规则层：它负责把“方法机制”“实验设置”“阅读理由”等字段映射到固定 evidence type，
    避免 LLM 直接发明证据结构或把证据需求停留在粗粒度 section 标签。
    """

    def bind(
        self,
        plan: ArtifactEvidencePlan,
        planner_context: PlannerContext,
        *,
        accepted_patches: Optional[Sequence[ArtifactRefinementPatch]] = None,
    ) -> ArtifactEvidencePlan:
        accepted_patches = list(accepted_patches or [])
        priority_effects = _priority_effects(accepted_patches)
        diagnostics = plan.diagnostics.model_copy(deep=True)
        artifacts: List[Artifact] = []

        for artifact in list(plan.artifacts or []):
            target_fields = self.effective_target_fields(artifact)
            requirements = self._requirements_for_artifact(
                artifact=artifact,
                target_fields=target_fields,
                planner_context=planner_context,
                priority_effects=priority_effects,
            )
            artifacts.append(
                artifact.model_copy(
                    update={
                        "target_fields": target_fields,
                        "evidence_requirements": requirements,
                    }
                )
            )

        diagnostics.local_corrections.append(
            {
                "code": "evidence_requirement_binder_applied",
                "source": EVIDENCE_REQUIREMENT_BINDER_SOURCE,
                "reason": "根据 artifact_type 与 target_fields 重新生成字段级 evidence requirements",
            }
        )
        diagnostics.final_evidence_requirements = _final_evidence_requirement_summary(artifacts)
        bound = _rebuild_plan(plan, artifacts=artifacts, diagnostics=diagnostics)
        return refresh_artifact_evidence_diagnostics(bound, planner_context)

    def effective_target_fields(self, artifact: Artifact) -> List[str]:
        fields = list(artifact.target_fields or [])
        if not fields:
            fields = list(_DEFAULT_TARGET_FIELDS.get(artifact.artifact_type, []))
        return _dedupe_texts(fields)

    def allowed_fields(self, artifact_type: str) -> Set[str]:
        fields = set(_DEFAULT_TARGET_FIELDS.get(artifact_type, []))
        fields.update(_OPTIONAL_TARGET_FIELDS.get(artifact_type, []))
        fields.update(_FIELD_EVIDENCE_RULES.get(artifact_type, {}).keys())
        return fields

    def required_fields(self, artifact_type: str) -> Set[str]:
        return set(_REQUIRED_TARGET_FIELDS.get(artifact_type, []))

    def _requirements_for_artifact(
        self,
        *,
        artifact: Artifact,
        target_fields: Sequence[str],
        planner_context: PlannerContext,
        priority_effects: Mapping[Tuple[str, str], EvidencePriority],
    ) -> List[EvidenceRequirement]:
        grouped: Dict[str, Dict[str, Any]] = {}
        rules = _FIELD_EVIDENCE_RULES.get(artifact.artifact_type, {})
        for field in list(target_fields or []):
            field_rules = list(rules.get(field) or _fallback_field_rules(field))
            for rule in field_rules:
                evidence_type = str(rule["evidence_type"])
                bucket = grouped.setdefault(
                    evidence_type,
                    {
                        "target_fields": [],
                        "coverage_criteria": [],
                        "required": False,
                    },
                )
                bucket["target_fields"].append(field)
                bucket["coverage_criteria"].append(str(rule["coverage"]))
                bucket["required"] = bool(bucket["required"] or rule.get("required", True))

        requirements: List[EvidenceRequirement] = []
        for evidence_type, bucket in grouped.items():
            fields = _dedupe_texts(bucket["target_fields"])
            priority = _priority_for_requirement(artifact.artifact_id, evidence_type, fields, priority_effects)
            required = bool(bucket["required"])
            min_count = _min_evidence_count(artifact, evidence_type, required)
            if priority == "high" and evidence_type in {"experiment_section", "result_section", "table", "figure"}:
                # 用户显式强调实验/图表时，把原本可选的证据提升为至少尝试获取一次，供 Observer 判断缺口。
                min_count = max(1, min_count)
                required = True
            capability_status = _capability_status_for_evidence(evidence_type, planner_context)
            fallback_policy = _fallback_policy_for_evidence(evidence_type, capability_status)
            if capability_status == "degraded" and evidence_type == "user_profile":
                required = False
                min_count = 0

            requirements.append(
                EvidenceRequirement(
                    requirement_id=f"{artifact.artifact_id}_{evidence_type}",
                    target_artifact_id=artifact.artifact_id,
                    required=required,
                    target_fields=fields,
                    evidence_type=evidence_type,
                    source_scope=_source_scope_for_evidence(evidence_type),
                    source_preference=_source_preference_for_evidence(evidence_type),
                    preferred_sections=_preferred_sections_for_evidence(evidence_type),
                    min_evidence_count=min_count,
                    coverage_criteria=_dedupe_texts(bucket["coverage_criteria"]),
                    fallback_policy=fallback_policy,
                    confidence_impact=_confidence_impact_for_priority(priority, required),
                    capability_status=capability_status,
                    priority=priority,
                    field_weights={field: _field_weight(priority) for field in fields},
                    scope_notes=list(artifact.scope_constraints or []),
                )
            )
        return requirements


class ArtifactRefinementService:
    """Patch-Based LLM refinement 服务。

    服务只负责“生成 patch、校验 patch、合并合法 patch”；一旦 LLM 不可用、JSON 非法或 patch 越权，
    就回退到模板 skeleton，并继续执行本地 EvidenceRequirementBinder，不能阻断 Agent 规划。
    """

    def __init__(
        self,
        generation_service: Optional[Any] = None,
        *,
        timeout_seconds: int = 6,
        max_patches: int = 12,
        binder: Optional[EvidenceRequirementBinder] = None,
    ) -> None:
        self.generation_service = generation_service
        self.timeout_seconds = max(1, int(timeout_seconds or 6))
        self.max_patches = max(1, int(max_patches or 12))
        self.binder = binder or EvidenceRequirementBinder()
        self.last_debug: Dict[str, Any] = {}

    def refine(
        self,
        plan: ArtifactEvidencePlan,
        *,
        profile: Mapping[str, Any],
        planner_context: PlannerContext,
    ) -> ArtifactEvidencePlan:
        diagnostics = plan.diagnostics.model_copy(deep=True)
        diagnostics.planner_source = diagnostics.planner_source or ARTIFACT_REFINEMENT_SERVICE_SOURCE
        diagnostics.template_defaults = diagnostics.template_defaults or _template_default_summary(plan)
        working_plan = _rebuild_plan(plan, diagnostics=diagnostics)
        accepted_patches: List[ArtifactRefinementPatch] = []

        if self.generation_service is None:
            # refinement 缺少 LLM 时只记录跳过原因；证据绑定仍继续，保证 metadata 有最终 evidence requirement。
            working_plan.diagnostics.llm_refinement_attempted = False
            working_plan.diagnostics.local_corrections.append(
                {
                    "code": "llm_refinement_skipped",
                    "source": ARTIFACT_REFINEMENT_SERVICE_SOURCE,
                    "reason": "generation_service_unavailable",
                }
            )
            refined = self.binder.bind(working_plan, planner_context, accepted_patches=accepted_patches)
            self.last_debug = refined.diagnostics.model_dump(mode="json")
            return refined

        working_plan.diagnostics.llm_refinement_attempted = True
        try:
            prompt = self._build_prompt(working_plan, profile, planner_context)
            raw_text = self._invoke_generation_service(prompt)
            working_plan.diagnostics.llm_refinement_raw_summary = _raw_patch_summary(raw_text)
            raw_patches = self._parse_patch_payload(raw_text)
            accepted_patches, rejected_patches = self._validate_patches(raw_patches, working_plan)
            working_plan.diagnostics.llm_refinement_patches = [_compact_raw_patch(item) for item in raw_patches]
            working_plan.diagnostics.accepted_refinement_patches = [
                patch.model_dump(mode="json") for patch in accepted_patches
            ]
            working_plan.diagnostics.rejected_refinement_patches = rejected_patches
            working_plan = self._merge_accepted_patches(working_plan, accepted_patches)
        except Exception as exc:
            # LLM refinement 是增强层；任何失败都降级为模板默认计划，避免污染主执行路径。
            working_plan.diagnostics.llm_refinement_error = str(exc)
            working_plan.diagnostics.local_corrections.append(
                {
                    "code": "llm_refinement_failed_fallback_to_template",
                    "source": ARTIFACT_REFINEMENT_SERVICE_SOURCE,
                    "reason": str(exc),
                }
            )

        refined = self.binder.bind(working_plan, planner_context, accepted_patches=accepted_patches)
        self.last_debug = refined.diagnostics.model_dump(mode="json")
        return refined

    def _build_prompt(
        self,
        plan: ArtifactEvidencePlan,
        profile: Mapping[str, Any],
        planner_context: PlannerContext,
    ) -> str:
        payload = {
            "user_request": planner_context.raw_user_request,
            "research_task_profile": dict(profile or {}),
            "artifact_skeleton": plan.model_dump(mode="json"),
            "allowed_artifact_types": list(get_args(ArtifactType)),
            "allowed_evidence_types": list(get_args(EvidenceType)),
            "allowed_patch_operations": list(get_args(ArtifactRefinementPatch.model_fields["operation"].annotation)),
            "allowed_target_fields_by_artifact_type": {
                artifact_type: sorted(self.binder.allowed_fields(artifact_type))
                for artifact_type in get_args(ArtifactType)
            },
            "system_capability_boundary": {
                "supported": ["metadata", "abstract"],
                "requires_index": sorted(_INDEX_EVIDENCE_TYPES),
                "requires_user_profile": ["user_profile"],
                "degraded": ["previous_context"],
                "unsupported_main_flow": ["citation_graph", "external_citation_count", "code_reproduction", "google_scholar_statistics"],
            },
            "forbidden": [
                "不要输出完整 ArtifactEvidencePlan",
                "不要新增未注册 artifact_type 或 evidence_type",
                "不要删除 required artifact",
                "不要修改 depends_on 或指定具体工具名",
                "不要扩大预算上限或扩大任务范围",
            ],
            "output_schema": ArtifactRefinementPatchSet.model_json_schema(),
        }
        return (
            "你是受控 Artifact Refinement Planner，只能输出 JSON-only patch set。\n"
            "你只能在模板 skeleton 内做细化：开启/关闭 optional artifact、调整证据优先级、补充目标字段、"
            "细化比较维度、设置 scope constraint、降低预算提示或补充 fallback note。\n"
            "禁止输出自然语言解释、Markdown、代码块、完整计划或任何具体工具调用。\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    def _invoke_generation_service(self, prompt: str) -> str:
        service = self.generation_service
        generate = getattr(service, "generate", None)
        if callable(generate):
            try:
                result = generate(prompt=prompt, task_type="artifact_refinement_patch", timeout=self.timeout_seconds)
            except TypeError:
                try:
                    result = generate("qwen", prompt, [], task_type="artifact_refinement_patch", show_reasoning=False)
                except TypeError:
                    result = generate(prompt)
            return _extract_text(result)
        complete = getattr(service, "complete_with_qwen", None)
        if callable(complete):
            return _extract_text(complete(prompt, task_type="artifact_refinement_patch", timeout=self.timeout_seconds))
        raise ValueError("generation service has no supported completion method")

    def _parse_patch_payload(self, raw_text: str) -> List[Mapping[str, Any]]:
        text = str(raw_text or "").strip()
        if not (text.startswith("{") and text.endswith("}")):
            raise ValueError("llm_refinement_output_not_json_only")
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("llm_refinement_output_not_object")
        patches = payload.get("patches")
        if not isinstance(patches, list):
            raise ValueError("llm_refinement_patches_must_be_list")
        return [item for item in patches[: self.max_patches] if isinstance(item, Mapping)]

    def _validate_patches(
        self,
        raw_patches: Sequence[Mapping[str, Any]],
        plan: ArtifactEvidencePlan,
    ) -> Tuple[List[ArtifactRefinementPatch], List[Dict[str, Any]]]:
        accepted: List[ArtifactRefinementPatch] = []
        rejected: List[Dict[str, Any]] = []
        for index, raw_patch in enumerate(list(raw_patches or [])):
            try:
                patch = ArtifactRefinementPatch.model_validate(dict(raw_patch))
            except ValidationError as exc:
                rejected.append({"index": index, "raw_patch": _compact_raw_patch(raw_patch), "reason": str(exc)})
                continue
            valid, reason = self._validate_patch_semantics(patch, plan)
            if valid:
                accepted.append(patch)
            else:
                rejected.append(
                    {
                        "index": index,
                        "patch": patch.model_dump(mode="json"),
                        "reason": reason,
                    }
                )
        return accepted, rejected

    def _validate_patch_semantics(self, patch: ArtifactRefinementPatch, plan: ArtifactEvidencePlan) -> Tuple[bool, str]:
        if _mentions_concrete_tool(patch.model_dump(mode="json")):
            return False, "patch must not specify concrete tool names"

        artifacts_by_id = {artifact.artifact_id: artifact for artifact in list(plan.artifacts or [])}
        if patch.operation == "set_budget_hint":
            return self._validate_budget_patch(patch, plan)

        artifact = artifacts_by_id.get(str(patch.target_artifact_id or ""))
        if artifact is None:
            return False, f"target artifact does not exist: {patch.target_artifact_id}"

        if patch.operation == "disable_optional_artifact":
            if artifact.required:
                return False, "required artifact cannot be disabled"
            dependents = [
                item.artifact_id
                for item in list(plan.artifacts or [])
                if item.required and artifact.artifact_id in list(item.depends_on or [])
            ]
            if dependents:
                return False, f"optional artifact is required by downstream artifacts: {dependents}"
        elif patch.operation == "enable_optional_artifact" and artifact.required:
            return False, "target artifact is already required"

        field_errors = self._validate_target_fields(patch, artifact)
        if field_errors:
            return False, "; ".join(field_errors)

        if patch.operation == "refine_comparison_dimension" and artifact.artifact_type not in {
            "comparison_dimension_set",
            "comparison_matrix",
        }:
            return False, "comparison dimension refinement only applies to comparison artifacts"
        return True, "accepted"

    def _validate_budget_patch(self, patch: ArtifactRefinementPatch, plan: ArtifactEvidencePlan) -> Tuple[bool, str]:
        current = plan.diagnostics.budget
        for key, value in dict(patch.budget_hint or {}).items():
            if key not in _BUDGET_KEYS:
                return False, f"unknown budget key: {key}"
            if int(value) < 0:
                return False, f"budget value must be non-negative: {key}"
            current_value = getattr(current, key, None)
            if current_value is not None and int(value) > int(current_value):
                return False, f"budget patch cannot expand {key}: {value} > {current_value}"
        return True, "accepted"

    def _validate_target_fields(self, patch: ArtifactRefinementPatch, artifact: Artifact) -> List[str]:
        errors: List[str] = []
        fields = list(patch.target_fields or [])
        if patch.comparison_dimension:
            fields.append(str(patch.comparison_dimension))
        if not fields:
            return errors

        allowed_fields = self.binder.allowed_fields(artifact.artifact_type)
        required_fields = self.binder.required_fields(artifact.artifact_type)
        for field in fields:
            if field not in allowed_fields:
                errors.append(f"target field is not registered for {artifact.artifact_type}: {field}")
            if patch.operation == "remove_optional_field" and field in required_fields:
                errors.append(f"required target field cannot be removed: {field}")
        return errors

    def _merge_accepted_patches(
        self,
        plan: ArtifactEvidencePlan,
        accepted_patches: Sequence[ArtifactRefinementPatch],
    ) -> ArtifactEvidencePlan:
        diagnostics = plan.diagnostics.model_copy(deep=True)
        artifacts_by_id = {artifact.artifact_id: artifact for artifact in list(plan.artifacts or [])}
        disabled_artifact_ids: Set[str] = set()

        for patch in list(accepted_patches or []):
            if patch.operation == "set_budget_hint":
                diagnostics.budget = _apply_budget_hint(diagnostics.budget, patch.budget_hint)
                diagnostics.local_corrections.append(
                    {"code": "budget_hint_applied", "patch": patch.model_dump(mode="json")}
                )
                continue

            artifact = artifacts_by_id.get(str(patch.target_artifact_id or ""))
            if artifact is None:
                continue
            if patch.operation == "disable_optional_artifact":
                disabled_artifact_ids.add(artifact.artifact_id)
                diagnostics.pruned_artifacts.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "artifact_type": artifact.artifact_type,
                        "reason": "disabled_by_accepted_llm_refinement_patch",
                    }
                )
            elif patch.operation == "enable_optional_artifact":
                diagnostics.retained_optional_artifacts.append(artifact.artifact_id)
            else:
                artifacts_by_id[artifact.artifact_id] = self._merge_patch_into_artifact(artifact, patch, diagnostics)

        artifacts = [
            artifact.model_copy(
                update={
                    "depends_on": [
                        dependency
                        for dependency in list(artifact.depends_on or [])
                        if dependency not in disabled_artifact_ids
                    ]
                }
            )
            for artifact in list(artifacts_by_id.values())
            if artifact.artifact_id not in disabled_artifact_ids
        ]
        return _rebuild_plan(plan, artifacts=artifacts, diagnostics=diagnostics)

    def _merge_patch_into_artifact(
        self,
        artifact: Artifact,
        patch: ArtifactRefinementPatch,
        diagnostics: PlanningDiagnostics,
    ) -> Artifact:
        target_fields = self.binder.effective_target_fields(artifact)
        quality_criteria = list(artifact.quality_criteria or [])
        scope_constraints = list(artifact.scope_constraints or [])
        fallback_notes = list(artifact.fallback_notes or [])

        if patch.operation == "add_target_field":
            target_fields = _dedupe_texts([*target_fields, *patch.target_fields])
        elif patch.operation == "remove_optional_field":
            target_fields = [field for field in target_fields if field not in set(patch.target_fields)]
        elif patch.operation == "refine_comparison_dimension":
            dimension_fields = list(patch.target_fields or [])
            if patch.comparison_dimension:
                dimension_fields.append(str(patch.comparison_dimension))
            target_fields = _dedupe_texts([*target_fields, *dimension_fields])
            quality_criteria.append("比较维度来自受控 refinement patch，后续证据必须按字段对齐")
        elif patch.operation == "set_scope_constraint":
            scope_constraints = _dedupe_texts([*scope_constraints, str(patch.scope_constraint or "").strip()])
        elif patch.operation == "add_fallback_note":
            fallback_notes = _dedupe_texts([*fallback_notes, str(patch.fallback_note or "").strip()])

        diagnostics.local_corrections.append(
            {
                "code": "accepted_patch_merged",
                "artifact_id": artifact.artifact_id,
                "operation": patch.operation,
            }
        )
        return artifact.model_copy(
            update={
                "target_fields": _dedupe_texts(target_fields),
                "quality_criteria": _dedupe_texts(quality_criteria),
                "scope_constraints": _dedupe_texts(scope_constraints),
                "fallback_notes": _dedupe_texts(fallback_notes),
            }
        )


def refine_artifact_evidence_plan(
    plan: ArtifactEvidencePlan,
    *,
    profile: Mapping[str, Any],
    planner_context: PlannerContext,
    generation_service: Optional[Any] = None,
    enabled: bool = False,
    timeout_seconds: int = 6,
    max_patches: int = 12,
) -> ArtifactEvidencePlan:
    """对 skeleton plan 做可选 LLM refinement，并始终执行 evidence binder。"""
    service = ArtifactRefinementService(
        generation_service if enabled else None,
        timeout_seconds=timeout_seconds,
        max_patches=max_patches,
    )
    return service.refine(plan, profile=profile, planner_context=planner_context)


def _priority_effects(patches: Sequence[ArtifactRefinementPatch]) -> Dict[Tuple[str, str], EvidencePriority]:
    effects: Dict[Tuple[str, str], EvidencePriority] = {}
    for patch in list(patches or []):
        if patch.operation not in {"raise_evidence_priority", "lower_evidence_priority"}:
            continue
        priority: EvidencePriority = "high" if patch.operation == "raise_evidence_priority" else "low"
        if patch.target_evidence_type:
            effects[(str(patch.target_artifact_id or ""), str(patch.target_evidence_type))] = priority
        for field in list(patch.target_fields or []):
            effects[(str(patch.target_artifact_id or ""), f"field:{field}")] = priority
    return effects


def _priority_for_requirement(
    artifact_id: str,
    evidence_type: str,
    fields: Sequence[str],
    priority_effects: Mapping[Tuple[str, str], EvidencePriority],
) -> EvidencePriority:
    if (artifact_id, evidence_type) in priority_effects:
        return priority_effects[(artifact_id, evidence_type)]
    field_priorities = [
        priority_effects[(artifact_id, f"field:{field}")]
        for field in list(fields or [])
        if (artifact_id, f"field:{field}") in priority_effects
    ]
    if "high" in field_priorities:
        return "high"
    if "low" in field_priorities:
        return "low"
    return "normal"


def _rebuild_plan(
    plan: ArtifactEvidencePlan,
    *,
    artifacts: Optional[Sequence[Artifact]] = None,
    diagnostics: Optional[PlanningDiagnostics] = None,
) -> ArtifactEvidencePlan:
    payload = plan.model_dump(mode="json")
    if artifacts is not None:
        payload["artifacts"] = [artifact.model_dump(mode="json") for artifact in artifacts]
    if diagnostics is not None:
        diagnostic_payload = diagnostics.model_dump(mode="json")
        diagnostic_payload["dependency_order"] = []
        diagnostic_payload["capability_summary"] = {}
        payload["diagnostics"] = diagnostic_payload
    return ArtifactEvidencePlan.model_validate(payload)


def _apply_budget_hint(budget: ArtifactPlanningBudget, hint: Mapping[str, int]) -> ArtifactPlanningBudget:
    payload = budget.model_dump(mode="json")
    for key, value in dict(hint or {}).items():
        if key in _BUDGET_KEYS:
            current = payload.get(key)
            payload[key] = int(value) if current is None else min(int(current), int(value))
    return ArtifactPlanningBudget.model_validate(payload)


def _template_default_summary(plan: ArtifactEvidencePlan) -> Dict[str, Any]:
    return {
        "artifact_count": len(list(plan.artifacts or [])),
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "required": artifact.required,
                "target_fields": list(artifact.target_fields or []),
                "evidence_requirement_ids": [
                    requirement.requirement_id for requirement in list(artifact.evidence_requirements or [])
                ],
            }
            for artifact in list(plan.artifacts or [])
        ],
    }


def _final_evidence_requirement_summary(artifacts: Sequence[Artifact]) -> List[Dict[str, Any]]:
    return [
        {
            "artifact_id": artifact.artifact_id,
            "requirement_id": requirement.requirement_id,
            "target_fields": list(requirement.target_fields or []),
            "evidence_type": requirement.evidence_type,
            "priority": requirement.priority,
            "min_evidence_count": requirement.min_evidence_count,
            "capability_status": requirement.capability_status,
            "fallback_policy": requirement.fallback_policy,
        }
        for artifact in list(artifacts or [])
        for requirement in list(artifact.evidence_requirements or [])
    ]


def _raw_patch_summary(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "")
    stripped = text.strip()
    summary = {
        "length": len(text),
        "is_json_object": stripped.startswith("{") and stripped.endswith("}"),
        "patch_count": 0,
    }
    if summary["is_json_object"]:
        try:
            payload = json.loads(stripped)
            patches = payload.get("patches") if isinstance(payload, Mapping) else None
            summary["patch_count"] = len(patches) if isinstance(patches, list) else 0
        except Exception:
            summary["parse_error"] = "invalid_json"
    return summary


def _compact_raw_patch(raw_patch: Mapping[str, Any]) -> Dict[str, Any]:
    compact = dict(raw_patch or {})
    for key, value in list(compact.items()):
        if isinstance(value, str) and len(value) > 200:
            compact[key] = value[:200] + "..."
    return compact


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, Mapping):
        for key in ("response", "result", "text", "answer"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("generation result does not contain text")


def _mentions_concrete_tool(value: Any) -> bool:
    for text in _iter_strings(value):
        normalized = text.lower()
        if any(tool_name.lower() in normalized for tool_name in _KNOWN_TOOL_NAMES):
            return True
    return False


def _iter_strings(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: List[str] = []
        for item in value.values():
            result.extend(_iter_strings(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_iter_strings(item))
        return result
    return []


def _dedupe_texts(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in list(values or []):
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _fallback_field_rules(field: str) -> List[Dict[str, Any]]:
    return [
        {
            "evidence_type": "abstract",
            "required": True,
            "coverage": f"{field} 字段至少需要摘要级证据支撑",
        }
    ]


def _min_evidence_count(artifact: Artifact, evidence_type: str, required: bool) -> int:
    if not required:
        return 0
    if artifact.target_scope == "paper_set" and evidence_type in {"metadata", "abstract"}:
        return 2
    return 1


def _capability_status_for_evidence(evidence_type: str, planner_context: PlannerContext) -> CapabilityStatus:
    if evidence_type in {"metadata", "abstract"}:
        return "supported"
    if evidence_type in _INDEX_EVIDENCE_TYPES:
        return "requires_index"
    if evidence_type == "user_profile":
        return "requires_user_profile" if _has_profile_context(planner_context) else "degraded"
    if evidence_type == "previous_context":
        return "degraded"
    return "unsupported"


def _fallback_policy_for_evidence(evidence_type: str, capability_status: str) -> EvidenceFallbackPolicy:
    if capability_status == "degraded" and evidence_type == "user_profile":
        return "generic_without_profile"
    if evidence_type in {"metadata", "abstract"}:
        return "none"
    if evidence_type in _INDEX_EVIDENCE_TYPES:
        return "use_metadata_abstract_surrogate"
    if evidence_type == "user_profile":
        return "generic_without_profile"
    return "use_previous_context"


def _source_scope_for_evidence(evidence_type: str) -> EvidenceSourceScope:
    if evidence_type in {"metadata", "abstract"}:
        return "search_results"
    if evidence_type in _INDEX_EVIDENCE_TYPES:
        return "paper_index"
    if evidence_type == "user_profile":
        return "user_profile"
    return "previous_context"


def _source_preference_for_evidence(evidence_type: str) -> List[EvidenceSourceScope]:
    primary = _source_scope_for_evidence(evidence_type)
    if evidence_type in _INDEX_EVIDENCE_TYPES:
        return [primary, "search_results"]
    return [primary]


def _preferred_sections_for_evidence(evidence_type: str) -> List[EvidenceSection]:
    mapping = {
        "metadata": ["title"],
        "abstract": ["abstract"],
        "paper_chunk": ["method", "experiment", "result", "limitation"],
        "method_section": ["method"],
        "experiment_section": ["experiment"],
        "result_section": ["result"],
        "limitation_section": ["limitation"],
        "table": ["table"],
        "figure": ["figure"],
        "user_profile": ["profile"],
        "previous_context": ["context"],
    }
    return list(mapping.get(evidence_type, []))


def _confidence_impact_for_priority(priority: EvidencePriority, required: bool) -> ConfidenceImpact:
    if priority == "high":
        return "high"
    if priority == "low":
        return "low"
    return "medium" if required else "low"


def _field_weight(priority: EvidencePriority) -> float:
    if priority == "high":
        return 1.5
    if priority == "low":
        return 0.7
    return 1.0


def _has_profile_context(planner_context: PlannerContext) -> bool:
    for value in (planner_context.user_memory_summary, planner_context.research_profile):
        if isinstance(value, Mapping) and value:
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def _rule(evidence_type: EvidenceType, coverage: str, *, required: bool = True) -> Dict[str, Any]:
    return {"evidence_type": evidence_type, "coverage": coverage, "required": required}


_DEFAULT_TARGET_FIELDS: Dict[str, List[str]] = {
    "topic_term_set": ["query_terms", "constraints"],
    "candidate_paper_set": ["title", "authors", "submitted_date", "abstract_summary"],
    "selected_paper_set": ["selection_reason", "topic_coverage", "representativeness"],
    "paper_feature_card_set": [
        "problem",
        "core_idea",
        "method_mechanism",
        "claimed_contribution",
        "evaluation_setting",
        "main_results",
        "limitation",
    ],
    "comparison_dimension_set": ["comparison_dimensions", "method_mechanism", "evaluation_setting", "main_results"],
    "comparison_matrix": ["comparison_dimensions", "method_mechanism", "evaluation_setting", "main_results", "limitation"],
    "reading_plan": ["paper_refs", "topic_coverage", "representativeness", "recommendation_reason", "user_profile_match"],
    "gap_hypothesis_set": ["scope_boundary", "limitation", "failure_cases", "unaddressed_question"],
    "grounded_summary": ["paper_refs", "main_findings", "grounded_points"],
    "user_profile_match": ["interest_match", "paper_topic", "ranking_reason"],
}

_OPTIONAL_TARGET_FIELDS: Dict[str, List[str]] = {
    "paper_feature_card_set": ["dataset", "metrics", "architecture", "difficulty"],
    "comparison_dimension_set": ["dataset", "metrics", "efficiency", "scope_boundary", "difficulty"],
    "comparison_matrix": ["dataset", "metrics", "efficiency", "scope_boundary", "difficulty"],
    "reading_plan": ["user_profile_match", "difficulty", "reading_order"],
    "gap_hypothesis_set": ["future_work", "dataset", "metrics"],
    "grounded_summary": ["method_mechanism", "evaluation_setting", "limitation"],
}

_REQUIRED_TARGET_FIELDS: Dict[str, List[str]] = {
    "topic_term_set": ["query_terms"],
    "candidate_paper_set": ["title", "abstract_summary"],
    "selected_paper_set": ["selection_reason", "topic_coverage"],
    "paper_feature_card_set": ["problem", "core_idea", "method_mechanism", "claimed_contribution"],
    "comparison_dimension_set": ["comparison_dimensions"],
    "comparison_matrix": ["comparison_dimensions", "method_mechanism", "main_results"],
    "reading_plan": ["paper_refs", "recommendation_reason"],
    "gap_hypothesis_set": ["scope_boundary", "unaddressed_question"],
    "grounded_summary": ["main_findings"],
    "user_profile_match": ["interest_match", "paper_topic"],
}

_FIELD_EVIDENCE_RULES: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "topic_term_set": {
        "query_terms": [_rule("metadata", "主题词必须能回溯到用户请求或规范化 search_spec")],
        "constraints": [_rule("metadata", "时间、数量和领域约束必须可回填到检索规格")],
    },
    "candidate_paper_set": {
        "title": [_rule("metadata", "候选论文标题来自 arXiv 元数据")],
        "authors": [_rule("metadata", "候选论文作者来自 arXiv 元数据")],
        "submitted_date": [_rule("metadata", "候选论文日期来自 arXiv 元数据")],
        "abstract_summary": [_rule("abstract", "候选论文摘要支撑主题相关性初筛")],
    },
    "selected_paper_set": {
        "selection_reason": [_rule("metadata", "代表论文选择理由至少引用题名、日期或类别")],
        "topic_coverage": [_rule("abstract", "代表性判断需要摘要级主题覆盖证据")],
        "representativeness": [_rule("abstract", "代表性说明应由候选论文摘要支撑")],
    },
    "paper_feature_card_set": {
        "problem": [_rule("abstract", "问题定义优先来自摘要中的任务描述")],
        "core_idea": [_rule("abstract", "核心思路至少需要摘要支撑"), _rule("paper_chunk", "正文 chunk 可补充核心机制", required=False)],
        "method_mechanism": [_rule("method_section", "方法机制必须优先来自方法段落")],
        "claimed_contribution": [_rule("abstract", "贡献声明优先来自摘要或引言摘要级信息")],
        "evaluation_setting": [_rule("experiment_section", "实验设置需要实验段落支撑")],
        "main_results": [_rule("result_section", "主要结果需要结果段落支撑"), _rule("table", "指标表格可补充定量结果", required=False)],
        "limitation": [_rule("limitation_section", "局限性优先来自 limitation 或 discussion 段落", required=False)],
        "dataset": [_rule("experiment_section", "数据集信息需要实验设置支撑", required=False)],
        "metrics": [_rule("table", "指标字段优先来自表格", required=False)],
        "architecture": [_rule("figure", "结构图可补充方法架构说明", required=False)],
        "difficulty": [_rule("abstract", "入门难度判断至少需要摘要级线索", required=False)],
    },
    "comparison_dimension_set": {
        "comparison_dimensions": [_rule("previous_context", "比较维度来自上游特征卡或候选摘要对齐结果")],
        "method_mechanism": [_rule("method_section", "方法维度需要方法段落对齐")],
        "evaluation_setting": [_rule("experiment_section", "实验维度需要实验设置对齐")],
        "main_results": [_rule("result_section", "结果维度需要结果段落对齐")],
        "limitation": [_rule("limitation_section", "局限维度需要 limitation 段落对齐", required=False)],
        "dataset": [_rule("experiment_section", "数据集维度需要实验段落支撑", required=False)],
        "metrics": [_rule("table", "指标维度优先使用表格证据", required=False)],
        "efficiency": [_rule("result_section", "效率维度需要结果或实验段落支撑", required=False)],
        "scope_boundary": [_rule("abstract", "适用范围维度至少需要摘要级边界说明", required=False)],
        "difficulty": [_rule("abstract", "阅读难度维度至少需要摘要级判断依据", required=False)],
    },
    "comparison_matrix": {
        "comparison_dimensions": [_rule("previous_context", "矩阵列必须来自 comparison_dimension_set")],
        "method_mechanism": [_rule("method_section", "方法列需要方法段落证据")],
        "evaluation_setting": [_rule("experiment_section", "实验列需要实验段落证据")],
        "main_results": [_rule("result_section", "结果列需要结果段落证据")],
        "limitation": [_rule("limitation_section", "局限列需要 limitation 段落证据", required=False)],
        "dataset": [_rule("experiment_section", "数据集列需要实验段落证据", required=False)],
        "metrics": [_rule("table", "指标列优先使用表格证据", required=False)],
        "efficiency": [_rule("result_section", "效率列需要结果段落或表格支撑", required=False)],
        "scope_boundary": [_rule("abstract", "范围列至少需要摘要级边界证据", required=False)],
        "difficulty": [_rule("abstract", "难度列至少需要摘要级理由", required=False)],
    },
    "reading_plan": {
        "paper_refs": [_rule("metadata", "阅读计划中的论文引用必须能回溯到元数据")],
        "topic_coverage": [_rule("abstract", "阅读顺序需要覆盖主题差异")],
        "representativeness": [_rule("abstract", "代表性说明需要摘要或候选选择依据")],
        "recommendation_reason": [_rule("abstract", "推荐理由需要摘要级主题证据")],
        "user_profile_match": [_rule("user_profile", "结合个人方向时需要用户画像匹配证据", required=False)],
        "difficulty": [_rule("abstract", "入门难度需要摘要级理由", required=False)],
        "reading_order": [_rule("previous_context", "阅读顺序可消费上游代表性排序", required=False)],
    },
    "gap_hypothesis_set": {
        "scope_boundary": [_rule("abstract", "研究范围边界至少需要摘要支撑")],
        "limitation": [_rule("limitation_section", "gap 假设优先由论文局限推出")],
        "failure_cases": [_rule("result_section", "失败案例或不足需要结果段落支撑")],
        "unaddressed_question": [_rule("abstract", "未覆盖问题需要摘要级任务边界支撑")],
        "future_work": [_rule("limitation_section", "future work 线索来自局限或讨论段落", required=False)],
        "dataset": [_rule("experiment_section", "数据覆盖不足需要实验段落支撑", required=False)],
        "metrics": [_rule("table", "指标覆盖不足可由表格支撑", required=False)],
    },
    "grounded_summary": {
        "paper_refs": [_rule("metadata", "总结中的论文引用必须可回溯到元数据")],
        "main_findings": [_rule("abstract", "主要结论至少需要摘要支撑")],
        "grounded_points": [_rule("previous_context", "总结要点应消费上游 artifact 字段")],
        "method_mechanism": [_rule("method_section", "方法总结优先来自方法段落", required=False)],
        "evaluation_setting": [_rule("experiment_section", "实验总结优先来自实验段落", required=False)],
        "limitation": [_rule("limitation_section", "局限总结优先来自局限段落", required=False)],
    },
    "user_profile_match": {
        "interest_match": [_rule("user_profile", "兴趣匹配必须来自用户画像")],
        "paper_topic": [_rule("abstract", "论文主题必须由摘要支撑")],
        "ranking_reason": [_rule("user_profile", "个性化排序理由需要用户画像证据", required=False)],
    },
}


__all__ = [
    "ARTIFACT_REFINEMENT_SERVICE_SOURCE",
    "EVIDENCE_REQUIREMENT_BINDER_SOURCE",
    "ArtifactRefinementService",
    "EvidenceRequirementBinder",
    "refine_artifact_evidence_plan",
]
