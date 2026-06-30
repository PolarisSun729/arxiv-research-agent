from __future__ import annotations

import importlib
import json

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]

artifact_templates = importlib.import_module("backend.agents.arxiv_search_agent.artifact_templates")
artifact_refinement = importlib.import_module("backend.agents.arxiv_search_agent.artifact_refinement")
planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")

AgentState = state_module.AgentState
ArtifactSkeletonBuilder = artifact_templates.ArtifactSkeletonBuilder
ArtifactRefinementService = artifact_refinement.ArtifactRefinementService
EvidenceRequirementBinder = artifact_refinement.EvidenceRequirementBinder


class _FakePatchService:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"response": self.payload}


def _profile(task_type: str) -> dict:
    return {
        "research_task_type": task_type,
        "source": "rule_high_confidence",
        "confidence": 0.92,
        "classification_basis": "test profile",
    }


def _patch_payload(patches: list[dict]) -> str:
    return json.dumps({"patches": patches}, ensure_ascii=False)


def test_refinement_accepts_valid_patch_and_binder_raises_experiment_priority() -> None:
    plan = ArtifactSkeletonBuilder().build(
        _profile("multi_paper_comparison"),
        schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search", raw_user_request="重点比较实验设置和结果表格"),
    )
    service = ArtifactRefinementService(
        _FakePatchService(
            _patch_payload(
                [
                    {
                        "operation": "add_target_field",
                        "target_artifact_id": "paper_feature_cards",
                        "target_fields": ["metrics"],
                        "rationale": "用户关注实验表格指标",
                    },
                    {
                        "operation": "raise_evidence_priority",
                        "target_artifact_id": "paper_feature_cards",
                        "target_evidence_type": "experiment_section",
                        "target_fields": ["evaluation_setting"],
                    },
                    {
                        "operation": "raise_evidence_priority",
                        "target_artifact_id": "paper_feature_cards",
                        "target_evidence_type": "result_section",
                        "target_fields": ["main_results"],
                    },
                    {
                        "operation": "raise_evidence_priority",
                        "target_artifact_id": "paper_feature_cards",
                        "target_evidence_type": "table",
                        "target_fields": ["metrics"],
                    },
                ]
            )
        )
    )

    refined = service.refine(
        plan,
        profile=_profile("multi_paper_comparison"),
        planner_context=schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search"),
    )
    feature_cards = next(artifact for artifact in refined.artifacts if artifact.artifact_id == "paper_feature_cards")
    by_evidence = {requirement.evidence_type: requirement for requirement in feature_cards.evidence_requirements}

    assert refined.diagnostics.llm_refinement_attempted is True
    assert len(refined.diagnostics.accepted_refinement_patches) == 4
    assert not refined.diagnostics.rejected_refinement_patches
    assert "metrics" in feature_cards.target_fields
    assert by_evidence["experiment_section"].priority == "high"
    assert by_evidence["result_section"].priority == "high"
    assert by_evidence["table"].priority == "high"
    assert by_evidence["table"].min_evidence_count == 1


def test_refinement_rejects_illegal_patches_and_keeps_template_skeleton() -> None:
    plan = ArtifactSkeletonBuilder().build(
        _profile("multi_paper_comparison"),
        schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search"),
    )
    service = ArtifactRefinementService(
        _FakePatchService(
            _patch_payload(
                [
                    {
                        "operation": "disable_optional_artifact",
                        "target_artifact_id": "selected_papers",
                    },
                    {
                        "operation": "add_target_field",
                        "target_artifact_id": "comparison_matrix",
                        "target_fields": ["citation_graph"],
                    },
                    {
                        "operation": "raise_evidence_priority",
                        "target_artifact_id": "comparison_matrix",
                        "target_evidence_type": "citation_graph",
                    },
                    {
                        "operation": "add_fallback_note",
                        "target_artifact_id": "comparison_matrix",
                        "fallback_note": "证据不足时显式标记为摘要级降级",
                    },
                ]
            )
        )
    )

    refined = service.refine(
        plan,
        profile=_profile("multi_paper_comparison"),
        planner_context=schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search"),
    )
    artifact_ids = {artifact.artifact_id for artifact in refined.artifacts}
    matrix = next(artifact for artifact in refined.artifacts if artifact.artifact_id == "comparison_matrix")

    assert "selected_papers" in artifact_ids
    assert len(refined.diagnostics.accepted_refinement_patches) == 1
    assert len(refined.diagnostics.rejected_refinement_patches) == 3
    assert "证据不足" in matrix.fallback_notes[0]


def test_refinement_invalid_json_falls_back_to_bound_template_plan() -> None:
    plan = ArtifactSkeletonBuilder().build(
        _profile("single_paper_deep_read"),
        schemas.PlannerContext(goal_type="paper_qa", intent="paper_qa"),
    )

    refined = ArtifactRefinementService(_FakePatchService("not json")).refine(
        plan,
        profile=_profile("single_paper_deep_read"),
        planner_context=schemas.PlannerContext(goal_type="paper_qa", intent="paper_qa"),
    )

    assert refined.diagnostics.llm_refinement_attempted is True
    assert refined.diagnostics.llm_refinement_error == "llm_refinement_output_not_json_only"
    assert refined.diagnostics.final_evidence_requirements
    assert any(item["code"] == "llm_refinement_failed_fallback_to_template" for item in refined.diagnostics.local_corrections)


def test_binder_marks_reading_plan_user_profile_evidence_degraded_without_profile() -> None:
    plan = ArtifactSkeletonBuilder().build(
        _profile("reading_planning"),
        schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search"),
    )

    bound = EvidenceRequirementBinder().bind(
        plan,
        schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search"),
    )
    reading_plan = next(artifact for artifact in bound.artifacts if artifact.artifact_type == "reading_plan")
    profile_requirement = next(
        requirement for requirement in reading_plan.evidence_requirements if requirement.evidence_type == "user_profile"
    )

    assert profile_requirement.required is False
    assert profile_requirement.min_evidence_count == 0
    assert profile_requirement.capability_status == "degraded"
    assert profile_requirement.fallback_policy == "generic_without_profile"


def test_profile_aware_planner_records_accepted_artifact_refinement_patch() -> None:
    state = AgentState(intent="arxiv_search", message="比较这些论文的方法机制和实验设置")
    state.research_task_profile = schemas.ResearchTaskProfile(
        research_task_type="multi_paper_comparison",
        intent="arxiv_search",
        goal_type="arxiv_search",
        task_object={"object_type": "paper_set", "topic": "retrieval augmented generation"},
        confidence=0.9,
        execution_readiness="ready",
        classification_basis="test profile",
        source="rule_high_confidence",
    )

    _, plan, debug = planner_module.build_executable_plan(
        state,
        enable_tool_aware_planner=True,
        enable_llm_plan_draft=False,
        enable_artifact_refinement=True,
        llm_generation_service=_FakePatchService(
            _patch_payload(
                [
                    {
                        "operation": "raise_evidence_priority",
                        "target_artifact_id": "paper_feature_cards",
                        "target_evidence_type": "method_section",
                        "target_fields": ["method_mechanism"],
                    }
                ]
            )
        ),
    )

    diagnostics = plan.metadata["profile_aware_plan"]["planning_diagnostics"]
    feature_cards = next(
        artifact
        for artifact in schemas.ArtifactEvidencePlan.model_validate(plan.metadata["artifact_evidence_plan"]).artifacts
        if artifact.artifact_id == "paper_feature_cards"
    )
    method_requirement = next(
        requirement for requirement in feature_cards.evidence_requirements if requirement.evidence_type == "method_section"
    )

    assert debug["selected_plan_source"] == "profile_aware_task_planner"
    assert debug["profile_aware_planning"]["artifact_refinement"]["attempted"] is True
    assert diagnostics["llm_refinement_attempted"] is True
    assert len(diagnostics["accepted_refinement_patches"]) == 1
    assert method_requirement.priority == "high"
