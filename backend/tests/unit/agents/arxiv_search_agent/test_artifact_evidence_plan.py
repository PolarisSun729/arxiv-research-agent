from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]


def _evidence(
    requirement_id: str,
    target_artifact_id: str,
    *,
    evidence_type: str = "metadata",
    capability_status: str = "supported",
):
    return schemas.EvidenceRequirement(
        requirement_id=requirement_id,
        target_artifact_id=target_artifact_id,
        target_fields=["title"],
        evidence_type=evidence_type,
        source_scope="search_results",
        preferred_sections=["title"],
        min_evidence_count=1,
        coverage_criteria=["字段必须能回溯到候选论文元数据"],
        fallback_policy="none",
        confidence_impact="medium",
        capability_status=capability_status,
    )


def _artifact(artifact_id: str, *, depends_on: list[str] | None = None):
    return schemas.Artifact(
        artifact_id=artifact_id,
        artifact_type="candidate_paper_set",
        depends_on=depends_on or [],
        required=True,
        target_scope="paper_set",
        expected_output="候选论文集合",
        quality_criteria=["候选集必须包含可回溯的论文元数据"],
        fallback_strategy="none",
        consumer=["tool_planner", "observer"],
        evidence_requirements=[_evidence(f"{artifact_id}_metadata", artifact_id)],
    )


def test_artifact_evidence_plan_is_structured_and_serializable() -> None:
    plan = schemas.ArtifactEvidencePlan(
        plan_id="direction:artifact_evidence_plan",
        research_task_type="direction_exploration",
        artifacts=[_artifact("candidate_papers")],
    )

    payload = plan.model_dump(mode="json")

    assert payload["artifacts"][0]["artifact_type"] == "candidate_paper_set"
    assert payload["artifacts"][0]["evidence_requirements"][0]["evidence_type"] == "metadata"
    assert payload["diagnostics"]["dependency_order"] == ["candidate_papers"]
    assert payload["diagnostics"]["capability_summary"] == {"supported": 1}


def test_artifact_evidence_plan_rejects_freeform_artifact_evidence_and_status() -> None:
    with pytest.raises(ValidationError):
        schemas.Artifact(
            artifact_id="citation_graph",
            artifact_type="citation_graph",
            target_scope="paper_set",
            expected_output="引用图",
            quality_criteria=["不应进入第一版主流程"],
            consumer=["tool_planner"],
            evidence_requirements=[_evidence("citation_metadata", "citation_graph")],
        )

    with pytest.raises(ValidationError):
        _evidence("scholar_count", "candidate_papers", evidence_type="google_scholar_citation_count")

    with pytest.raises(ValidationError):
        _evidence("unknown_status", "candidate_papers", capability_status="maybe_supported")


def test_required_artifact_must_define_evidence_requirements() -> None:
    with pytest.raises(ValidationError, match="must define evidence_requirements"):
        schemas.Artifact(
            artifact_id="candidate_papers",
            artifact_type="candidate_paper_set",
            required=True,
            target_scope="paper_set",
            expected_output="候选论文集合",
            quality_criteria=["候选集必须有证据支撑"],
            consumer=["tool_planner"],
            evidence_requirements=[],
        )


def test_artifact_dependency_graph_rejects_cycles() -> None:
    first = _artifact("candidate_papers", depends_on=["selected_papers"])
    second = schemas.Artifact(
        artifact_id="selected_papers",
        artifact_type="selected_paper_set",
        depends_on=["candidate_papers"],
        required=True,
        target_scope="paper_set",
        expected_output="代表论文集合",
        quality_criteria=["代表论文必须来自候选集"],
        fallback_strategy="none",
        consumer=["tool_planner", "observer"],
        evidence_requirements=[_evidence("selected_metadata", "selected_papers")],
    )

    with pytest.raises(ValidationError, match="cycle"):
        schemas.ArtifactEvidencePlan(research_task_type="direction_exploration", artifacts=[first, second])


def test_required_artifact_cannot_depend_on_unsupported_evidence() -> None:
    unsupported = schemas.EvidenceRequirement(
        requirement_id="unsupported_metric",
        target_artifact_id="candidate_papers",
        target_fields=["citation_count"],
        evidence_type="metadata",
        source_scope="search_results",
        preferred_sections=["title"],
        min_evidence_count=0,
        coverage_criteria=["当前主流程不使用外部引用统计"],
        fallback_policy="skip_optional_artifact",
        confidence_impact="high",
        capability_status="unsupported",
    )

    with pytest.raises(ValidationError, match="cannot depend on unsupported evidence"):
        schemas.Artifact(
            artifact_id="candidate_papers",
            artifact_type="candidate_paper_set",
            required=True,
            target_scope="paper_set",
            expected_output="候选论文集合",
            quality_criteria=["候选集必须有可支持证据"],
            consumer=["tool_planner"],
            evidence_requirements=[unsupported],
        )
