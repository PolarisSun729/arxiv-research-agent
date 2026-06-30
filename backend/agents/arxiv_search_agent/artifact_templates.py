"""Artifact Template Registry 与本地 Skeleton Builder。

该模块把“系统当前稳定支持哪些科研中间产物”固化为模板注册表，而不是交给 LLM 从零生成。
Skeleton Builder 只做本地模板选择、上下文裁剪和证据能力初始化；具体工具执行仍由 PlanDraft /
ExecutablePlan 层负责。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .schemas import (
    Artifact,
    ArtifactEvidencePlan,
    ArtifactPlanningBudget,
    EvidenceRequirement,
    EvidenceType,
    PlannerContext,
    PlanningDiagnostic,
    PlanningDiagnostics,
    ResearchTaskType,
)


ARTIFACT_TEMPLATE_REGISTRY_SOURCE = "artifact_template_registry"
ARTIFACT_SKELETON_BUILDER_SOURCE = "artifact_skeleton_builder"


class ResearchTaskArtifactTemplate(BaseModel):
    """单个 research_task_type 的 artifact skeleton 模板。

    模板是系统能力边界的一部分：它固定 artifact 主干、依赖关系、默认证据、fallback 和预算。
    第一版保持 artifact-level planning，不展开 per-paper artifact graph，避免多论文任务过早膨胀。
    """

    model_config = ConfigDict(extra="forbid")

    template_id: str
    research_task_type: ResearchTaskType
    version: str = "v1"
    artifacts: List[Artifact] = Field(default_factory=list, min_length=1)
    budget: ArtifactPlanningBudget = Field(default_factory=ArtifactPlanningBudget)
    notes: List[str] = Field(default_factory=list)


class ArtifactTemplateRegistry:
    """Research Task -> Artifact Template 的轻量注册表。

    后续如果模板数量继续增长，可以把 register 阶段切换为 YAML/TOML 配置加载；
    核心 Skeleton Builder 只依赖 registry.get，不需要改成大段条件分支。
    """

    def __init__(self, templates: Optional[Sequence[ResearchTaskArtifactTemplate]] = None) -> None:
        self._templates: Dict[str, ResearchTaskArtifactTemplate] = {}
        for template in list(templates or []):
            self.register(template)

    def register(self, template: ResearchTaskArtifactTemplate) -> None:
        self._templates[template.research_task_type] = template

    def get(self, research_task_type: str) -> ResearchTaskArtifactTemplate:
        template = self._templates.get(str(research_task_type or "").strip())
        if template is None:
            raise KeyError(f"Artifact template not registered: {research_task_type}")
        return template.model_copy(deep=True)

    def task_types(self) -> List[str]:
        return sorted(self._templates.keys())

    @classmethod
    def build_default(cls) -> "ArtifactTemplateRegistry":
        return cls(
            [
                _direction_exploration_template(),
                _multi_paper_comparison_template(),
                _single_paper_deep_read_template(),
                _reading_planning_template(),
                _research_gap_analysis_template(),
                _personalized_recommendation_template(),
            ]
        )


class ArtifactSkeletonBuilder:
    """从模板生成本轮 ArtifactEvidencePlan 的本地 builder。"""

    def __init__(self, registry: Optional[ArtifactTemplateRegistry] = None) -> None:
        self.registry = registry or DEFAULT_ARTIFACT_TEMPLATE_REGISTRY

    def build(self, profile: Mapping[str, Any], planner_context: PlannerContext) -> ArtifactEvidencePlan:
        task_type = str(profile.get("research_task_type") or "").strip()
        template = self.registry.get(task_type)
        artifacts = [artifact.model_copy(deep=True) for artifact in template.artifacts]
        diagnostics = PlanningDiagnostics(
            planner_source=ARTIFACT_SKELETON_BUILDER_SOURCE,
            template_id=template.template_id,
            template_version=template.version,
            research_task_type=template.research_task_type,
            profile_source=str(profile.get("source") or "") or None,
            classification_basis=str(profile.get("classification_basis") or "") or None,
            budget=template.budget.model_copy(deep=True),
        )

        artifacts = self._apply_context_pruning(
            artifacts=artifacts,
            task_type=template.research_task_type,
            planner_context=planner_context,
            diagnostics=diagnostics,
        )
        artifacts = self._remove_pruned_dependencies(artifacts, diagnostics)
        artifacts = self._apply_evidence_context(
            artifacts=artifacts,
            planner_context=planner_context,
            diagnostics=diagnostics,
        )
        plan = ArtifactEvidencePlan(
            plan_id=f"{template.template_id}:skeleton",
            research_task_type=template.research_task_type,
            profile_source=str(profile.get("source") or "") or None,
            classification_basis=str(profile.get("classification_basis") or "") or None,
            artifacts=artifacts,
            diagnostics=diagnostics,
        )
        unmet = _unmet_evidence_requirements(plan, planner_context)
        plan.diagnostics.unmet_evidence_requirements = unmet
        plan.diagnostics.diagnostic_events = _diagnostic_events_for_plan(plan, unmet)
        return plan

    def _apply_context_pruning(
        self,
        *,
        artifacts: List[Artifact],
        task_type: str,
        planner_context: PlannerContext,
        diagnostics: PlanningDiagnostics,
    ) -> List[Artifact]:
        has_candidate_context = _has_candidate_papers(planner_context)
        has_selected_paper = _has_selected_paper(planner_context)
        pruned_ids: set[str] = set()

        if has_candidate_context and task_type in {
            "direction_exploration",
            "multi_paper_comparison",
            "reading_planning",
            "research_gap_analysis",
        }:
            # 已有候选论文时，skeleton 从候选集合之后继续，避免 Artifact 层无脑要求重复检索。
            pruned_ids.update({"topic_terms", "candidate_papers"})
            diagnostics.context_adjustments.append("detected_candidate_papers_context")

        if has_selected_paper and task_type == "single_paper_deep_read":
            # 单篇深读模板本身不含搜索型 artifact；这里记录上下文命中，供 debug 解释为什么不需要候选检索。
            diagnostics.context_adjustments.append("detected_selected_paper_context")

        result: List[Artifact] = []
        for artifact in artifacts:
            if artifact.artifact_id in pruned_ids and artifact.prunable:
                diagnostics.pruned_artifacts.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "artifact_type": artifact.artifact_type,
                        "reason": "context_already_provides_candidate_papers",
                    }
                )
                continue
            if not artifact.required:
                diagnostics.retained_optional_artifacts.append(artifact.artifact_id)
            result.append(artifact)
        return result

    def _remove_pruned_dependencies(self, artifacts: List[Artifact], diagnostics: PlanningDiagnostics) -> List[Artifact]:
        available_ids = {artifact.artifact_id for artifact in artifacts}
        removed_ids = {item.get("artifact_id") for item in diagnostics.pruned_artifacts}
        adjusted: List[Artifact] = []
        for artifact in artifacts:
            depends_on = [dependency for dependency in artifact.depends_on if dependency in available_ids]
            removed_dependencies = [dependency for dependency in artifact.depends_on if dependency in removed_ids]
            if removed_dependencies:
                diagnostics.context_adjustments.append(
                    f"removed_pruned_dependencies:{artifact.artifact_id}:{','.join(removed_dependencies)}"
                )
            adjusted.append(artifact.model_copy(update={"depends_on": depends_on}))
        return adjusted

    def _apply_evidence_context(
        self,
        *,
        artifacts: List[Artifact],
        planner_context: PlannerContext,
        diagnostics: PlanningDiagnostics,
    ) -> List[Artifact]:
        has_profile = _has_profile_context(planner_context)
        adjusted_artifacts: List[Artifact] = []
        for artifact in artifacts:
            adjusted_requirements: List[EvidenceRequirement] = []
            for requirement in artifact.evidence_requirements:
                adjusted = requirement
                if requirement.evidence_type == "user_profile" and not has_profile:
                    # 无用户画像时不能伪造个性化证据；可选画像证据降级为通用排序线索。
                    adjusted = requirement.model_copy(
                        update={
                            "required": False,
                            "min_evidence_count": 0,
                            "capability_status": "degraded",
                            "fallback_policy": "generic_without_profile",
                            "confidence_impact": "low",
                        }
                    )
                    diagnostics.degraded_evidence_requirements.append(
                        {
                            "artifact_id": artifact.artifact_id,
                            "requirement_id": requirement.requirement_id,
                            "evidence_type": requirement.evidence_type,
                            "reason": "user_profile_unavailable",
                        }
                    )
                adjusted_requirements.append(adjusted)
            adjusted_artifacts.append(artifact.model_copy(update={"evidence_requirements": adjusted_requirements}))
        return adjusted_artifacts


def build_artifact_evidence_plan(
    profile: Mapping[str, Any],
    planner_context: PlannerContext,
    *,
    registry: Optional[ArtifactTemplateRegistry] = None,
) -> ArtifactEvidencePlan:
    return ArtifactSkeletonBuilder(registry=registry).build(profile, planner_context)


def artifact_plan_projection(plan: ArtifactEvidencePlan) -> List[Dict[str, Any]]:
    """保留旧 debug 字段的轻量投影；主契约仍以 artifact_evidence_plan 为准。"""
    return [
        {
            "artifact_id": artifact.artifact_id,
            "artifact": artifact.artifact_type,
            "depends_on": list(artifact.depends_on or []),
            "required": artifact.required,
            "target_fields": list(artifact.target_fields or []),
            "evidence_types": [requirement.evidence_type for requirement in artifact.evidence_requirements],
            "purpose": artifact.expected_output,
            "quality_criteria": list(artifact.quality_criteria or []),
        }
        for artifact in list(plan.artifacts or [])
    ]


def evidence_tool_mapping(plan: ArtifactEvidencePlan) -> List[Dict[str, Any]]:
    mapping_by_artifact_type = {
        "topic_term_set": ["normalize_request", "build_arxiv_search_spec"],
        "candidate_paper_set": ["build_arxiv_search_spec", "search_arxiv", "validate_arxiv_results"],
        "selected_paper_set": ["validate_arxiv_results", "personalize_paper_results", "synthesize_arxiv_response"],
        "paper_feature_card_set": ["check_paper_index", "answer_paper_question", "assess_paper_qa_quality", "synthesize_arxiv_response"],
        "comparison_dimension_set": ["validate_arxiv_results", "synthesize_arxiv_response"],
        "comparison_matrix": ["synthesize_arxiv_response"],
        "reading_plan": ["personalize_paper_results", "synthesize_arxiv_response"],
        "gap_hypothesis_set": ["validate_arxiv_results", "synthesize_arxiv_response"],
        "grounded_summary": ["synthesize_arxiv_response", "answer_paper_question"],
        "user_profile_match": ["personalize_paper_results", "synthesize_arxiv_response"],
    }
    return [
        {
            "artifact_id": artifact.artifact_id,
            "artifact": artifact.artifact_type,
            "evidence_types": [requirement.evidence_type for requirement in artifact.evidence_requirements],
            "tool_steps": list(mapping_by_artifact_type.get(artifact.artifact_type, ["synthesize_arxiv_response"])),
        }
        for artifact in list(plan.artifacts or [])
    ]


def evidence_quality_reservations(plan: ArtifactEvidencePlan) -> Dict[str, Dict[str, Any]]:
    return {
        artifact.artifact_id: {
            "status": "not_observed_yet",
            "observer_reserved": True,
            "artifact_type": artifact.artifact_type,
            "expected_evidence_types": [requirement.evidence_type for requirement in artifact.evidence_requirements],
            "target_fields": sorted({field for requirement in artifact.evidence_requirements for field in list(requirement.target_fields or [])}),
        }
        for artifact in list(plan.artifacts or [])
    }


def artifact_plan_needs_evidence(plan: ArtifactEvidencePlan, evidence_type: EvidenceType) -> bool:
    return any(requirement.evidence_type == evidence_type for artifact in list(plan.artifacts or []) for requirement in list(artifact.evidence_requirements or []))


def refresh_artifact_evidence_diagnostics(plan: ArtifactEvidencePlan, planner_context: PlannerContext) -> ArtifactEvidencePlan:
    """重算 evidence 缺口和诊断事件。

    Skeleton、LLM refinement 和 Binder 都可能改变 evidence_requirements；统一在这里刷新 debug 字段，
    避免每个阶段各自拼一套不一致的 diagnostics。
    """
    refreshed = ArtifactEvidencePlan.model_validate(plan.model_dump(mode="json"))
    unmet = _unmet_evidence_requirements(refreshed, planner_context)
    refreshed.diagnostics.unmet_evidence_requirements = unmet
    refreshed.diagnostics.diagnostic_events = _diagnostic_events_for_plan(refreshed, unmet)
    return refreshed


def _direction_exploration_template() -> ResearchTaskArtifactTemplate:
    return ResearchTaskArtifactTemplate(
        template_id="direction_exploration.default.v1",
        research_task_type="direction_exploration",
        budget=ArtifactPlanningBudget(
            max_candidate_papers=12,
            max_selected_papers=5,
            max_deep_read_papers=0,
            max_qa_calls=0,
            max_refinement_rounds=1,
        ),
        artifacts=[
            _artifact(
                "topic_terms",
                "topic_term_set",
                prunable=True,
                target_scope="topic",
                expected_output="主题词、时间范围和检索约束",
                quality_criteria=["主题词能驱动 arXiv 检索", "约束可回填到 search_spec"],
                evidence=[_evidence("topic_terms_metadata", "topic_terms", "metadata", ["query_terms", "constraints"], ["主题词来自用户请求或规范化 search_spec"])],
                budget_cost={"refinement_rounds": 0},
            ),
            _artifact(
                "candidate_papers",
                "candidate_paper_set",
                depends_on=["topic_terms"],
                prunable=True,
                target_scope="paper_set",
                expected_output="可筛选的候选论文集合",
                quality_criteria=["候选集包含标题、作者、日期和摘要", "数量满足检索规格或记录降级原因"],
                evidence=[
                    _evidence("candidate_metadata", "candidate_papers", "metadata", ["title", "authors", "submitted_date"], ["候选论文元数据可用于排序和去重"], min_count=3),
                    _evidence("candidate_abstract", "candidate_papers", "abstract", ["abstract_summary"], ["候选摘要能支持初步主题相关性判断"], min_count=3),
                ],
                budget_cost={"candidate_papers": 12},
            ),
            _artifact(
                "selected_papers",
                "selected_paper_set",
                depends_on=["candidate_papers"],
                target_scope="paper_set",
                expected_output="代表性论文集合",
                quality_criteria=["代表论文覆盖候选集的主要方向", "选择依据可由 metadata/abstract 解释"],
                evidence=[
                    _evidence("selected_metadata", "selected_papers", "metadata", ["selection_reason"], ["选择理由至少引用标题、日期或类别"], min_count=2),
                    _evidence("selected_abstract", "selected_papers", "abstract", ["topic_coverage"], ["摘要能解释代表性覆盖范围"], min_count=2),
                ],
                budget_cost={"selected_papers": 5},
            ),
            _artifact(
                "grounded_summary",
                "grounded_summary",
                depends_on=["selected_papers"],
                target_scope="topic",
                expected_output="基于候选论文的方向概览",
                quality_criteria=["总结不超出候选论文可支撑的信息", "结论能回溯到论文元数据或摘要"],
                evidence=[
                    _evidence("summary_metadata", "grounded_summary", "metadata", ["paper_refs"], ["总结中的论文引用可回溯"], min_count=2),
                    _evidence("summary_abstract", "grounded_summary", "abstract", ["main_findings"], ["主要结论由摘要支撑"], min_count=2),
                ],
            ),
        ],
    )


def _multi_paper_comparison_template() -> ResearchTaskArtifactTemplate:
    base = _direction_exploration_template().artifacts[:3]
    return ResearchTaskArtifactTemplate(
        template_id="multi_paper_comparison.default.v1",
        research_task_type="multi_paper_comparison",
        budget=ArtifactPlanningBudget(
            max_candidate_papers=15,
            max_selected_papers=6,
            max_deep_read_papers=3,
            max_qa_calls=6,
            max_refinement_rounds=2,
        ),
        artifacts=[
            *base,
            _artifact(
                "paper_feature_cards",
                "paper_feature_card_set",
                depends_on=["selected_papers"],
                target_scope="paper_set",
                expected_output="每篇论文的方法、实验和结果特征卡片",
                quality_criteria=["每张卡片至少包含方法和实验/结果字段", "缺少正文索引时必须记录摘要级降级"],
                evidence=[
                    _evidence("feature_abstract", "paper_feature_cards", "abstract", ["problem", "method_hint"], ["摘要能提供问题和方法线索"], min_count=2),
                    _evidence("feature_method", "paper_feature_cards", "method_section", ["method"], ["方法字段优先来自正文方法段"], min_count=1),
                    _evidence("feature_experiment", "paper_feature_cards", "experiment_section", ["experiment_setup"], ["实验字段优先来自实验段"], min_count=1),
                    _evidence("feature_result", "paper_feature_cards", "result_section", ["main_results"], ["结果字段优先来自结果段或表格"], min_count=1),
                ],
                budget_cost={"deep_read_papers": 3, "qa_calls": 3},
            ),
            _artifact(
                "comparison_dimensions",
                "comparison_dimension_set",
                depends_on=["paper_feature_cards"],
                target_scope="paper_set",
                expected_output="统一比较维度集合",
                quality_criteria=["比较维度适用于至少两篇论文", "维度来源能回溯到特征卡片"],
                evidence=[
                    _evidence("dimension_method", "comparison_dimensions", "method_section", ["method_dimensions"], ["方法维度由方法证据支撑"], min_count=1),
                    _evidence("dimension_result", "comparison_dimensions", "result_section", ["evaluation_dimensions"], ["评测维度由实验或结果证据支撑"], min_count=1),
                ],
            ),
            _artifact(
                "comparison_matrix",
                "comparison_matrix",
                depends_on=["comparison_dimensions"],
                target_scope="paper_set",
                expected_output="论文对比矩阵",
                quality_criteria=["矩阵单元格能对应论文和比较维度", "正文证据不足时标注摘要级降级"],
                evidence=[
                    _evidence("matrix_metadata", "comparison_matrix", "metadata", ["paper_refs"], ["矩阵行可回溯到论文元数据"], min_count=2),
                    _evidence("matrix_table", "comparison_matrix", "table", ["quantitative_results"], ["表格证据优先用于量化结果"], min_count=0, confidence_impact="low"),
                    _evidence("matrix_result", "comparison_matrix", "result_section", ["result_cells"], ["结果单元格优先来自结果段"], min_count=1),
                ],
            ),
            _artifact(
                "grounded_summary",
                "grounded_summary",
                depends_on=["comparison_matrix"],
                target_scope="paper_set",
                expected_output="基于比较矩阵的归纳总结",
                quality_criteria=["总结引用比较维度和代表论文", "无法覆盖的维度进入诊断"],
                evidence=[
                    _evidence("comparison_summary_metadata", "grounded_summary", "metadata", ["paper_refs"], ["总结引用可回溯"], min_count=2),
                    _evidence("comparison_summary_previous", "grounded_summary", "previous_context", ["matrix_refs"], ["总结消费上游比较矩阵"], min_count=1, confidence_impact="low"),
                ],
            ),
        ],
    )


def _single_paper_deep_read_template() -> ResearchTaskArtifactTemplate:
    return ResearchTaskArtifactTemplate(
        template_id="single_paper_deep_read.default.v1",
        research_task_type="single_paper_deep_read",
        budget=ArtifactPlanningBudget(
            max_candidate_papers=0,
            max_selected_papers=1,
            max_deep_read_papers=1,
            max_qa_calls=3,
            max_refinement_rounds=1,
        ),
        artifacts=[
            _artifact(
                "selected_papers",
                "selected_paper_set",
                target_scope="paper",
                expected_output="唯一目标论文",
                quality_criteria=["目标论文可由 selected_paper、候选序号或 arXiv ID 解析", "无法解析时应转澄清计划"],
                evidence=[
                    _evidence("target_metadata", "selected_papers", "metadata", ["title", "arxiv_id"], ["目标论文元数据可确定唯一对象"], min_count=1),
                    _evidence("target_abstract", "selected_papers", "abstract", ["abstract_summary"], ["摘要可提供深读入口"], min_count=1),
                ],
                budget_cost={"selected_papers": 1},
            ),
            _artifact(
                "paper_feature_cards",
                "paper_feature_card_set",
                depends_on=["selected_papers"],
                target_scope="paper",
                expected_output="单篇论文结构、方法、实验、结果和局限卡片",
                quality_criteria=["核心字段优先来自正文 chunk", "正文缺失时记录需要索引或摘要级降级"],
                evidence=[
                    _evidence("deep_chunk", "paper_feature_cards", "paper_chunk", ["structure"], ["正文 chunk 支撑论文结构"], min_count=2),
                    _evidence("deep_method", "paper_feature_cards", "method_section", ["method"], ["方法解释来自方法段"], min_count=1),
                    _evidence("deep_experiment", "paper_feature_cards", "experiment_section", ["experiment_setup"], ["实验设置来自实验段"], min_count=1),
                    _evidence("deep_result", "paper_feature_cards", "result_section", ["results"], ["结果来自结果段或表格"], min_count=1),
                    _evidence("deep_limitation", "paper_feature_cards", "limitation_section", ["limitations"], ["局限来自局限/讨论段，缺失时标记未知"], min_count=0, confidence_impact="low"),
                    _evidence("deep_table", "paper_feature_cards", "table", ["metrics"], ["表格优先支撑指标和对比结果"], min_count=0, confidence_impact="low"),
                    _evidence("deep_figure", "paper_feature_cards", "figure", ["architecture"], ["图像优先支撑模型结构说明"], min_count=0, confidence_impact="low"),
                ],
                budget_cost={"deep_read_papers": 1, "qa_calls": 2},
            ),
            _artifact(
                "grounded_summary",
                "grounded_summary",
                depends_on=["paper_feature_cards"],
                target_scope="paper",
                expected_output="基于正文证据的单篇深读总结",
                quality_criteria=["总结覆盖方法、实验、贡献和局限", "每类结论能对应特征卡片字段"],
                evidence=[
                    _evidence("deep_summary_chunk", "grounded_summary", "paper_chunk", ["grounded_points"], ["总结要点来自正文或特征卡片"], min_count=2),
                    _evidence("deep_summary_previous", "grounded_summary", "previous_context", ["feature_card_refs"], ["总结消费上游特征卡片"], min_count=1, confidence_impact="low"),
                ],
                budget_cost={"qa_calls": 1},
            ),
        ],
    )


def _reading_planning_template() -> ResearchTaskArtifactTemplate:
    base = _direction_exploration_template().artifacts[:3]
    return ResearchTaskArtifactTemplate(
        template_id="reading_planning.default.v1",
        research_task_type="reading_planning",
        budget=ArtifactPlanningBudget(
            max_candidate_papers=12,
            max_selected_papers=6,
            max_deep_read_papers=0,
            max_qa_calls=0,
            max_refinement_rounds=1,
        ),
        artifacts=[
            *base,
            _artifact(
                "user_profile_match",
                "user_profile_match",
                depends_on=["selected_papers"],
                required=False,
                target_scope="user_profile",
                expected_output="候选论文与用户画像的匹配说明",
                quality_criteria=["有画像时解释兴趣匹配点", "无画像时允许降级为通用阅读优先级"],
                fallback_strategy="skip_optional",
                evidence=[
                    _evidence("profile_match", "user_profile_match", "user_profile", ["interest_match"], ["匹配说明来自用户画像"], min_count=1, confidence_impact="medium"),
                    _evidence("profile_match_metadata", "user_profile_match", "metadata", ["paper_refs"], ["匹配对象可回溯到论文元数据"], min_count=2),
                ],
            ),
            _artifact(
                "reading_plan",
                "reading_plan",
                depends_on=["selected_papers"],
                target_scope="paper_set",
                expected_output="循序渐进的阅读顺序和理由",
                quality_criteria=["阅读顺序覆盖代表论文", "排序理由可由摘要、元数据或用户画像支撑"],
                evidence=[
                    _evidence("reading_metadata", "reading_plan", "metadata", ["paper_refs", "recency"], ["阅读列表引用论文元数据"], min_count=2),
                    _evidence("reading_abstract", "reading_plan", "abstract", ["difficulty_reason"], ["摘要支持难度和主题判断"], min_count=2),
                    _evidence("reading_profile", "reading_plan", "user_profile", ["interest_fit"], ["有用户画像时补充兴趣匹配"], min_count=0, confidence_impact="low"),
                ],
            ),
        ],
    )


def _research_gap_analysis_template() -> ResearchTaskArtifactTemplate:
    base = _multi_paper_comparison_template().artifacts[:5]
    return ResearchTaskArtifactTemplate(
        template_id="research_gap_analysis.default.v1",
        research_task_type="research_gap_analysis",
        budget=ArtifactPlanningBudget(
            max_candidate_papers=15,
            max_selected_papers=6,
            max_deep_read_papers=3,
            max_qa_calls=6,
            max_refinement_rounds=2,
        ),
        artifacts=[
            *base,
            _artifact(
                "gap_hypotheses",
                "gap_hypothesis_set",
                depends_on=["comparison_dimensions"],
                target_scope="topic",
                expected_output="潜在研究空白和未覆盖问题集合",
                quality_criteria=["每个 gap 由已有方法/实验覆盖不足推出", "不能使用外部引用数或 Scholar 统计作为第一版依据"],
                evidence=[
                    _evidence("gap_limitation", "gap_hypotheses", "limitation_section", ["limitations"], ["优先引用局限/讨论段"], min_count=1),
                    _evidence("gap_result", "gap_hypotheses", "result_section", ["failure_cases"], ["结果或失败案例支持 gap 判断"], min_count=1),
                    _evidence("gap_abstract", "gap_hypotheses", "abstract", ["scope_boundary"], ["摘要支持问题范围和边界"], min_count=2),
                ],
            ),
            _artifact(
                "grounded_summary",
                "grounded_summary",
                depends_on=["gap_hypotheses"],
                target_scope="topic",
                expected_output="基于证据的研究空白总结",
                quality_criteria=["总结中的 gap 能回溯到 gap_hypothesis_set", "证据不足的 gap 必须降级或标注不确定"],
                evidence=[
                    _evidence("gap_summary_previous", "grounded_summary", "previous_context", ["gap_refs"], ["总结消费上游 gap 假设"], min_count=1, confidence_impact="low"),
                    _evidence("gap_summary_metadata", "grounded_summary", "metadata", ["paper_refs"], ["总结引用的论文可回溯"], min_count=2),
                ],
            ),
        ],
    )


def _personalized_recommendation_template() -> ResearchTaskArtifactTemplate:
    base = _direction_exploration_template().artifacts[:2]
    return ResearchTaskArtifactTemplate(
        template_id="personalized_recommendation.default.v1",
        research_task_type="personalized_recommendation",
        budget=ArtifactPlanningBudget(
            max_candidate_papers=12,
            max_selected_papers=5,
            max_deep_read_papers=0,
            max_qa_calls=0,
            max_refinement_rounds=1,
        ),
        artifacts=[
            *base,
            _artifact(
                "user_profile_match",
                "user_profile_match",
                depends_on=["candidate_papers"],
                target_scope="user_profile",
                expected_output="用户画像与候选论文的匹配结果",
                quality_criteria=["匹配理由来自画像和论文摘要", "无画像时不能伪造个性化依据"],
                fallback_strategy="fallback_to_generic_route",
                evidence=[
                    _evidence("recommend_profile", "user_profile_match", "user_profile", ["interest_match"], ["推荐依据必须包含用户画像"], min_count=1),
                    _evidence("recommend_abstract", "user_profile_match", "abstract", ["paper_topic"], ["论文主题来自摘要"], min_count=2),
                ],
            ),
            _artifact(
                "selected_papers",
                "selected_paper_set",
                depends_on=["user_profile_match"],
                target_scope="paper_set",
                expected_output="个性化排序后的论文集合",
                quality_criteria=["排序能解释画像匹配点", "无画像时应降级为普通方向探索"],
                evidence=[
                    _evidence("recommend_selected_metadata", "selected_papers", "metadata", ["paper_refs"], ["推荐项可回溯到元数据"], min_count=2),
                    _evidence("recommend_selected_profile", "selected_papers", "user_profile", ["ranking_reason"], ["排序理由引用画像"], min_count=1),
                ],
            ),
        ],
    )


def _artifact(
    artifact_id: str,
    artifact_type: str,
    *,
    depends_on: Optional[Sequence[str]] = None,
    required: bool = True,
    prunable: bool = False,
    target_scope: str,
    expected_output: str,
    quality_criteria: Sequence[str],
    evidence: Sequence[EvidenceRequirement],
    fallback_strategy: str = "none",
    budget_cost: Optional[Mapping[str, int]] = None,
) -> Artifact:
    return Artifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        depends_on=list(depends_on or []),
        required=required,
        prunable=prunable,
        target_scope=target_scope,
        expected_output=expected_output,
        quality_criteria=list(quality_criteria or []),
        fallback_strategy=fallback_strategy,
        consumer=_consumers_for_artifact_type(artifact_type),
        evidence_requirements=list(evidence or []),
        budget_cost=dict(budget_cost or {}),
    )


def _evidence(
    requirement_id: str,
    target_artifact_id: str,
    evidence_type: EvidenceType,
    target_fields: Sequence[str],
    coverage_criteria: Sequence[str],
    *,
    min_count: int = 1,
    confidence_impact: str = "medium",
) -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=requirement_id,
        target_artifact_id=target_artifact_id,
        required=min_count > 0,
        target_fields=list(target_fields or []),
        evidence_type=evidence_type,
        source_scope=_source_scope_for_evidence(evidence_type),
        source_preference=_source_preference_for_evidence(evidence_type),
        preferred_sections=_preferred_sections_for_evidence(evidence_type),
        min_evidence_count=min_count,
        coverage_criteria=list(coverage_criteria or []),
        fallback_policy=_fallback_policy_for_evidence(evidence_type),
        confidence_impact=confidence_impact,
        capability_status=_capability_status_for_evidence(evidence_type),
    )


def _capability_status_for_evidence(evidence_type: EvidenceType) -> str:
    if evidence_type in {"metadata", "abstract"}:
        return "supported"
    if evidence_type in {"paper_chunk", "method_section", "experiment_section", "result_section", "limitation_section", "table", "figure"}:
        return "requires_index"
    if evidence_type == "user_profile":
        return "requires_user_profile"
    if evidence_type == "previous_context":
        return "degraded"
    return "unsupported"


def _source_scope_for_evidence(evidence_type: EvidenceType) -> str:
    if evidence_type in {"metadata", "abstract"}:
        return "search_results"
    if evidence_type in {"paper_chunk", "method_section", "experiment_section", "result_section", "limitation_section", "table", "figure"}:
        return "paper_index"
    if evidence_type == "user_profile":
        return "user_profile"
    return "previous_context"


def _source_preference_for_evidence(evidence_type: EvidenceType) -> List[str]:
    primary = _source_scope_for_evidence(evidence_type)
    if evidence_type in {"method_section", "experiment_section", "result_section", "limitation_section", "table", "figure"}:
        return [primary, "search_results"]
    return [primary]


def _preferred_sections_for_evidence(evidence_type: EvidenceType) -> List[str]:
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


def _fallback_policy_for_evidence(evidence_type: EvidenceType) -> str:
    if evidence_type in {"metadata", "abstract"}:
        return "none"
    if evidence_type in {"paper_chunk", "method_section", "experiment_section", "result_section", "limitation_section", "table", "figure"}:
        return "use_metadata_abstract_surrogate"
    if evidence_type == "user_profile":
        return "generic_without_profile"
    return "use_previous_context"


def _consumers_for_artifact_type(artifact_type: str) -> List[str]:
    consumers = ["tool_planner", "observer", "replanner"]
    if artifact_type in {"comparison_matrix", "reading_plan", "gap_hypothesis_set", "grounded_summary", "user_profile_match"}:
        consumers.append("response_synthesizer")
    return consumers


def _unmet_evidence_requirements(plan: ArtifactEvidencePlan, planner_context: PlannerContext) -> List[Dict[str, Any]]:
    unmet: List[Dict[str, Any]] = []
    qa_result = planner_context.paper_qa_result if isinstance(planner_context.paper_qa_result, Mapping) else {}
    for artifact in list(plan.artifacts or []):
        for requirement in list(artifact.evidence_requirements or []):
            if requirement.capability_status == "requires_user_profile" and not _has_profile_context(planner_context):
                unmet.append(_unmet_payload(artifact, requirement, "user profile is unavailable in planner context"))
            elif requirement.capability_status == "requires_index" and not qa_result:
                # 正文、图表类证据需要 Paper QA 索引或后续回答工具确认；这里先记录缺口，避免 planner 假装证据已满足。
                unmet.append(_unmet_payload(artifact, requirement, "paper index evidence is not observed at planning time"))
            elif requirement.capability_status == "requires_confirmation":
                # 索引构建或高风险获取路径需要用户确认时，planner 只能记录等待确认边界，不能直接编译成无确认工具步骤。
                unmet.append(_unmet_payload(artifact, requirement, "evidence acquisition requires explicit confirmation"))
            elif requirement.capability_status in {"blocked", "unsupported"}:
                # blocked/unsupported 是能力边界，不是执行失败；后续 Tool Planner 不应为它生成虚假的工具步骤。
                unmet.append(_unmet_payload(artifact, requirement, "evidence is outside the currently executable capability boundary"))
            elif requirement.capability_status == "degraded":
                unmet.append(_unmet_payload(artifact, requirement, "evidence is available only as a degraded planning signal"))
    return unmet


def _unmet_payload(artifact: Artifact, requirement: EvidenceRequirement, reason: str) -> Dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "requirement_id": requirement.requirement_id,
        "target_fields": list(requirement.target_fields or []),
        "evidence_type": requirement.evidence_type,
        "capability_status": requirement.capability_status,
        "fallback_policy": requirement.fallback_policy,
        "confidence_impact": requirement.confidence_impact,
        "reason": reason,
    }


def _diagnostic_events_for_plan(plan: ArtifactEvidencePlan, unmet: Sequence[Mapping[str, Any]]) -> List[PlanningDiagnostic]:
    events = [
        PlanningDiagnostic(
            code="artifact_template_selected",
            severity="info",
            message=f"selected template {plan.diagnostics.template_id}",
        )
    ]
    for item in list(plan.diagnostics.pruned_artifacts or []):
        events.append(
            PlanningDiagnostic(
                code="artifact_pruned_by_context",
                severity="info",
                message=str(item.get("reason") or "artifact pruned by context"),
                target_artifact_id=str(item.get("artifact_id") or "") or None,
            )
        )
    for item in list(unmet or []):
        status = str(item.get("capability_status") or "")
        severity = "error" if status in {"blocked", "unsupported"} else ("info" if status == "degraded" else "warning")
        events.append(
            PlanningDiagnostic(
                code=f"evidence_{status or 'unknown'}",
                severity=severity,
                message=str(item.get("reason") or "evidence requirement needs follow-up"),
                target_artifact_id=str(item.get("artifact_id") or "") or None,
                evidence_type=item.get("evidence_type"),
                capability_status=item.get("capability_status"),
            )
        )
    return events


def _has_candidate_papers(planner_context: PlannerContext) -> bool:
    return bool(list(planner_context.last_papers or []))


def _has_selected_paper(planner_context: PlannerContext) -> bool:
    return bool(planner_context.selected_paper)


def _has_profile_context(planner_context: PlannerContext) -> bool:
    for value in (planner_context.user_memory_summary, planner_context.research_profile):
        if isinstance(value, Mapping) and value:
            return True
        if value not in (None, "", [], {}):
            return True
    return False


DEFAULT_ARTIFACT_TEMPLATE_REGISTRY = ArtifactTemplateRegistry.build_default()


__all__ = [
    "ARTIFACT_SKELETON_BUILDER_SOURCE",
    "ARTIFACT_TEMPLATE_REGISTRY_SOURCE",
    "ArtifactSkeletonBuilder",
    "ArtifactTemplateRegistry",
    "DEFAULT_ARTIFACT_TEMPLATE_REGISTRY",
    "ResearchTaskArtifactTemplate",
    "artifact_plan_needs_evidence",
    "artifact_plan_projection",
    "build_artifact_evidence_plan",
    "evidence_quality_reservations",
    "evidence_tool_mapping",
    "refresh_artifact_evidence_diagnostics",
]
