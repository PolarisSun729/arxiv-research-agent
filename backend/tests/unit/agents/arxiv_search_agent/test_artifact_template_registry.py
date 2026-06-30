from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]

artifact_templates = importlib.import_module("backend.agents.arxiv_search_agent.artifact_templates")

ArtifactSkeletonBuilder = artifact_templates.ArtifactSkeletonBuilder
DEFAULT_ARTIFACT_TEMPLATE_REGISTRY = artifact_templates.DEFAULT_ARTIFACT_TEMPLATE_REGISTRY


def _profile(task_type: str) -> dict:
    return {
        "research_task_type": task_type,
        "source": "rule_high_confidence",
        "classification_basis": "test profile",
    }


def test_default_registry_covers_all_research_task_types() -> None:
    assert set(DEFAULT_ARTIFACT_TEMPLATE_REGISTRY.task_types()) == {
        "direction_exploration",
        "multi_paper_comparison",
        "single_paper_deep_read",
        "reading_planning",
        "research_gap_analysis",
        "personalized_recommendation",
    }


def test_skeleton_builder_is_deterministic_and_exports_budget() -> None:
    context = schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search")
    builder = ArtifactSkeletonBuilder()

    first = builder.build(_profile("multi_paper_comparison"), context).model_dump(mode="json")
    second = builder.build(_profile("multi_paper_comparison"), context).model_dump(mode="json")

    assert first == second
    diagnostics = first["diagnostics"]
    assert diagnostics["template_id"] == "multi_paper_comparison.default.v1"
    assert diagnostics["template_version"] == "v1"
    assert diagnostics["budget"]["max_candidate_papers"] == 15
    assert diagnostics["budget"]["max_qa_calls"] == 6


def test_skeleton_builder_prunes_search_artifacts_when_candidate_context_exists() -> None:
    context = schemas.PlannerContext(
        goal_type="arxiv_search",
        intent="arxiv_search",
        last_papers=[{"title": "Paper A", "arxiv_id": "2401.00001"}, {"title": "Paper B", "arxiv_id": "2401.00002"}],
    )

    plan = ArtifactSkeletonBuilder().build(_profile("multi_paper_comparison"), context)
    artifact_ids = [artifact.artifact_id for artifact in plan.artifacts]
    selected_papers = next(artifact for artifact in plan.artifacts if artifact.artifact_id == "selected_papers")

    assert "topic_terms" not in artifact_ids
    assert "candidate_papers" not in artifact_ids
    assert selected_papers.depends_on == []
    assert {item["artifact_id"] for item in plan.diagnostics.pruned_artifacts} == {"topic_terms", "candidate_papers"}
    assert "detected_candidate_papers_context" in plan.diagnostics.context_adjustments


def test_skeleton_builder_degrades_user_profile_evidence_without_profile_context() -> None:
    context = schemas.PlannerContext(goal_type="arxiv_search", intent="arxiv_search")

    plan = ArtifactSkeletonBuilder().build(_profile("reading_planning"), context)
    profile_requirements = [
        requirement
        for artifact in plan.artifacts
        for requirement in artifact.evidence_requirements
        if requirement.evidence_type == "user_profile"
    ]

    assert profile_requirements
    assert all(requirement.required is False for requirement in profile_requirements)
    assert all(requirement.min_evidence_count == 0 for requirement in profile_requirements)
    assert all(requirement.capability_status == "degraded" for requirement in profile_requirements)
    assert plan.diagnostics.degraded_evidence_requirements
    assert any(item["evidence_type"] == "user_profile" for item in plan.diagnostics.unmet_evidence_requirements)
