"""统一 Tool Contract 注册表。

这个模块是 planner、executor、observer/replanner 共同读取的工具事实来源：
- planner 使用 contract 中的能力标签、副作用等级和确认策略做工具选择；
- executor 使用同一份 contract 找到 adapter，并按声明的模型/协议校验输入输出；
- debug 直接暴露 contract 来源，便于确认“计划看到的工具”和“执行调用的工具”一致。

旧的 `PLANNER_TOOL_REGISTRY` 名称保留为兼容入口，但它只代理本模块中的统一注册表，
不再维护独立的 planner schema 或执行映射。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from tools.tool_registry import invoke_tool as invoke_backend_tool

from .schemas import ToolSpec
from .tool_adapters.clarification import (
    AnalyzeAmbiguityAdapter,
    AnalyzeAmbiguityInput,
    ConfirmationStatusOutput,
    FinalAnswerOutput as ClarificationAnswerOutput,
    GenerateClarificationAdapter,
    GenerateClarificationInput,
    GenerateFallbackInput,
    GenerateFallbackResponseAdapter,
    MissingInformationOutput,
    RequestConfirmationAdapter,
    RequestConfirmationInput,
)
from .tool_adapters.models import ToolError
from .tool_adapters.paper_qa import (
    AnswerPaperQuestionAdapter,
    AnswerPaperQuestionInput,
    CheckPaperIndexAdapter,
    CheckPaperIndexInput,
    IndexBuildOutput,
    PaperIndexStatusOutput,
    PaperQAAnswerOutput,
    PaperReferenceOutput,
    ParseAndIndexPaperAdapter,
    ParseAndIndexPaperInput,
    ResolvePaperAdapter,
    ResolvePaperInput,
)
from .tool_adapters.preference import (
    PreferenceActionOutput,
    PreferenceResponseOutput,
    PreferenceTargetOutput,
    ResolvePreferenceTargetAdapter,
    ResolvePreferenceTargetInput,
    SynthesizePreferenceResponseAdapter,
    SynthesizePreferenceResponseInput,
    UpdatePreferenceStoreAdapter,
    UpdatePreferenceStoreInput,
    VerifiedPreferenceOutput,
    VerifyPreferenceUpdateAdapter,
    VerifyPreferenceUpdateInput,
)
from .tool_adapters.recommendation import (
    CandidatePapersOutput,
    ExplainRecommendationsAdapter,
    ExplainRecommendationsInput,
    GenerateRecommendationsAdapter,
    GenerateRecommendationsInput,
    LoadCandidatePapersAdapter,
    LoadCandidatePapersInput,
    LoadUserProfileAdapter,
    LoadUserProfileInput,
    RecommendationAnswerOutput,
    RecommendationResultOutput,
    UserProfileOutput,
    ValidateRecommendationsAdapter,
    ValidateRecommendationsInput,
    ValidatedRecommendationsOutput,
)
from .tool_adapters.search import (
    ArxivSearchSpecOutput,
    BuildArxivSearchSpecAdapter,
    BuildArxivSearchSpecInput,
    FinalAnswerOutput as SearchAnswerOutput,
    NormalizeRequestAdapter,
    NormalizeRequestInput,
    NormalizedRequestOutput,
    PersonalizePaperResultsAdapter,
    PersonalizePaperResultsInput,
    RankedPapersOutput,
    RewriteArxivQueryAdapter,
    RewriteArxivQueryInput,
    RewriteArxivQueryOutput,
    SearchArxivAdapter,
    SearchArxivInput,
    SearchArxivOutput,
    SynthesizeArxivResponseAdapter,
    SynthesizeArxivResponseInput,
    ValidateArxivResultsAdapter,
    ValidateArxivResultsInput,
    ValidateArxivResultsOutput,
)

CONTRACT_SOURCE = "backend.agents.arxiv_search_agent.tool_registry.UNIFIED_TOOL_REGISTRY"


class ToolAdapter(Protocol):
    """统一执行入口。

    adapter 不决定工具是否可被 planner 选择，也不自行声明风险；这些静态事实都在
    ToolContract 中，adapter 只承担“把已校验输入转换为实际业务调用”的职责。
    """

    def execute(self, tool_input: Any) -> Any:
        ...


@dataclass(frozen=True)
class ToolContract:
    """单个工具的唯一可信契约。"""

    tool_name: str
    description: str
    capability_tags: List[str]
    side_effect_level: str = "none"
    requires_confirmation: bool = False
    can_retry: bool = False
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    input_model: Optional[type] = None
    output_model: Optional[type] = None
    error_model: Optional[type] = None
    adapter: Optional[ToolAdapter] = None
    backend_tool_name: Optional[str] = None
    implementation: Optional[str] = None
    failure_modes: List[str] = field(default_factory=list)
    recovery_policy: Dict[str, Any] = field(default_factory=dict)
    confirmation_policy: Dict[str, Any] = field(default_factory=dict)
    contract_source: str = CONTRACT_SOURCE

    def to_tool_spec(self) -> ToolSpec:
        """生成 planner/debug 消费的只读投影，避免旧调用点直接持有 adapter 实例。"""
        return ToolSpec(
            tool_name=self.tool_name,
            description=self.description,
            capability_tags=list(self.capability_tags or []),
            input_schema=dict(self.input_schema or {}),
            output_schema=dict(self.output_schema or {}),
            input_model=_model_name(self.input_model),
            output_model=_model_name(self.output_model),
            error_model=_model_name(self.error_model),
            side_effect_level=self.side_effect_level,  # type: ignore[arg-type]
            requires_confirmation=bool(self.requires_confirmation),
            can_retry=bool(self.can_retry),
            failure_modes=list(self.failure_modes or []),
            implementation=self.implementation,
            backend_tool_name=self.backend_tool_name,
            adapter=self.adapter.__class__.__name__ if self.adapter is not None else None,
            recovery_policy=dict(self.recovery_policy or {}),
            confirmation_policy=dict(self.confirmation_policy or {}),
            contract_source=self.contract_source,
        )

    def debug_summary(self) -> Dict[str, Any]:
        """压缩后的 contract 摘要，用于 trace/debug 证明 planner 与 executor 读取同源定义。"""
        return {
            "tool_name": self.tool_name,
            "contract_source": self.contract_source,
            "adapter": self.adapter.__class__.__name__ if self.adapter is not None else None,
            "backend_tool_name": self.backend_tool_name,
            "input_model": _model_name(self.input_model),
            "output_model": _model_name(self.output_model),
            "error_model": _model_name(self.error_model),
            "side_effect_level": self.side_effect_level,
            "requires_confirmation": self.requires_confirmation,
            "can_retry": self.can_retry,
            "recovery_policy": dict(self.recovery_policy or {}),
            "confirmation_policy": dict(self.confirmation_policy or {}),
        }


class ToolRegistry:
    """统一维护 Agent 可用工具 contract 的注册表。"""

    def __init__(self) -> None:
        self._contracts: Dict[str, ToolContract] = {}
        self._spec_cache: Dict[str, ToolSpec] = {}

    def register(self, contract: ToolContract) -> None:
        if contract.adapter is None:
            raise ValueError(f"Tool contract {contract.tool_name} must declare an adapter")
        self._contracts[contract.tool_name] = contract
        self._spec_cache.pop(contract.tool_name, None)

    def get_contract(self, tool_name: str) -> Optional[ToolContract]:
        return self._contracts.get(str(tool_name or "").strip())

    def get(self, tool_name: str) -> Optional[ToolSpec]:
        contract = self.get_contract(tool_name)
        if contract is None:
            return None
        if contract.tool_name not in self._spec_cache:
            self._spec_cache[contract.tool_name] = contract.to_tool_spec()
        return self._spec_cache[contract.tool_name]

    def exists(self, tool_name: str) -> bool:
        return self.get_contract(tool_name) is not None

    def validate_tool_name(self, tool_name: str) -> str:
        normalized_tool_name = str(tool_name or "").strip()
        if not normalized_tool_name or normalized_tool_name not in self._contracts:
            raise ValueError(f"Unknown planner tool: {tool_name}")
        return normalized_tool_name

    def list_contracts(self) -> List[ToolContract]:
        return list(self._contracts.values())

    def list_tools(self) -> List[ToolSpec]:
        return [contract.to_tool_spec() for contract in self._contracts.values()]

    def list_by_capability(self, tag: str) -> List[ToolSpec]:
        normalized_tag = str(tag or "").strip()
        if not normalized_tag:
            return []
        return [
            contract.to_tool_spec()
            for contract in self._contracts.values()
            if normalized_tag in list(contract.capability_tags or [])
        ]

    def describe_contract(self, tool_name: str) -> Dict[str, Any]:
        contract = self.get_contract(tool_name)
        return contract.debug_summary() if contract is not None else {"tool_name": tool_name, "contract_source": None}

    def tool_contract_matrix(self) -> List[Dict[str, Any]]:
        """输出 planner 工具、adapter 和后端工具的对应关系，供 debug/审计使用。"""
        rows: List[Dict[str, Any]] = []
        for contract in self._contracts.values():
            rows.append(
                {
                    "tool_name": contract.tool_name,
                    "planner_spec": contract.to_tool_spec().model_dump(),
                    "executor_adapter": contract.adapter.__class__.__name__ if contract.adapter is not None else None,
                    "backend_tool_name": contract.backend_tool_name,
                    "input_fields": list((contract.input_schema or {}).keys()),
                    "output_fields": list((contract.output_schema or {}).keys()),
                    "side_effect_level": contract.side_effect_level,
                    "requires_confirmation": contract.requires_confirmation,
                    "contract_source": contract.contract_source,
                }
            )
        return rows


def _model_name(model: Optional[type]) -> Optional[str]:
    return getattr(model, "__name__", None) if model is not None else None


def _contract(
    tool_name: str,
    *,
    description: str,
    capability_tags: List[str],
    adapter: ToolAdapter,
    input_schema: Optional[Dict[str, object]] = None,
    output_schema: Optional[Dict[str, object]] = None,
    side_effect_level: str = "none",
    requires_confirmation: bool = False,
    can_retry: bool = False,
    failure_modes: Optional[List[str]] = None,
    implementation: Optional[str] = None,
    backend_tool_name: Optional[str] = None,
    input_model: Optional[type] = None,
    output_model: Optional[type] = None,
    error_model: Optional[type] = None,
    recovery_policy: Optional[Dict[str, Any]] = None,
    confirmation_policy: Optional[Dict[str, Any]] = None,
) -> ToolContract:
    return ToolContract(
        tool_name=tool_name,
        description=description,
        capability_tags=list(capability_tags or []),
        input_schema=dict(input_schema or {}),
        output_schema=dict(output_schema or {}),
        input_model=input_model,
        output_model=output_model,
        error_model=error_model,
        side_effect_level=side_effect_level,
        requires_confirmation=requires_confirmation,
        can_retry=can_retry,
        failure_modes=list(failure_modes or []),
        implementation=implementation,
        backend_tool_name=backend_tool_name,
        adapter=adapter,
        recovery_policy=dict(recovery_policy or {}),
        confirmation_policy=dict(confirmation_policy or {}),
    )


def _default_recovery(*modes: str) -> Dict[str, Any]:
    return {"modes": [mode for mode in modes if mode]}


UNIFIED_TOOL_REGISTRY = ToolRegistry()

for contract in [
    _contract("normalize_request", description="归一化当前 Agent 请求，供后续计划步骤稳定读取。", capability_tags=["search", "clarify"], input_schema={"message": "str"}, output_schema={"normalized_request": "dict"}, side_effect_level="session_write", implementation="search.NormalizeRequestAdapter", input_model=NormalizeRequestInput, output_model=NormalizedRequestOutput, error_model=ToolError, adapter=NormalizeRequestAdapter()),
    _contract("build_arxiv_search_spec", description="把归一化请求转换为 arXiv 结构化检索参数。", capability_tags=["search"], input_schema={"normalized_request": "dict", "search_spec": "ArxivSearchSpec"}, output_schema={"arxiv_search_spec": "dict"}, side_effect_level="session_write", implementation="search.BuildArxivSearchSpecAdapter", input_model=BuildArxivSearchSpecInput, output_model=ArxivSearchSpecOutput, error_model=ToolError, adapter=BuildArxivSearchSpecAdapter()),
    _contract("search_arxiv", description="调用后端结构化 arXiv 检索工具并返回论文列表。", capability_tags=["search"], input_schema={"search_spec": "ArxivSearchSpec"}, output_schema={"papers": "list", "tool_result": "dict"}, side_effect_level="external_call", can_retry=True, failure_modes=["tool_argument_validation_failed", "tool_execution_failed", "empty_results"], implementation="tools.search_arxiv_structured", backend_tool_name="search_arxiv_structured", input_model=SearchArxivInput, output_model=SearchArxivOutput, error_model=ToolError, recovery_policy=_default_recovery("retry_step", "patch_plan"), adapter=SearchArxivAdapter(invoke_backend_tool)),
    _contract("validate_arxiv_results", description="检查 arXiv 检索结果是否为空或工具失败。", capability_tags=["search", "validate"], input_schema={"arxiv_results": "dict"}, output_schema={"ok": "bool", "result_count": "int", "warnings": "list"}, implementation="search.ValidateArxivResultsAdapter", input_model=ValidateArxivResultsInput, output_model=ValidateArxivResultsOutput, error_model=ToolError, recovery_policy=_default_recovery("patch_plan", "fallback_answer"), adapter=ValidateArxivResultsAdapter()),
    _contract("rewrite_arxiv_query", description="在检索结果为空或质量低时放宽 arXiv 查询条件。", capability_tags=["search", "rewrite"], input_schema={"search_spec": "ArxivSearchSpec"}, output_schema={"rewritten_search_spec": "dict"}, side_effect_level="session_write", can_retry=True, implementation="search.RewriteArxivQueryAdapter", input_model=RewriteArxivQueryInput, output_model=RewriteArxivQueryOutput, error_model=ToolError, recovery_policy=_default_recovery("retry_step"), adapter=RewriteArxivQueryAdapter()),
    _contract("personalize_paper_results", description="根据用户上下文对检索结果做轻量排序标注。", capability_tags=["rerank", "personalize"], input_schema={"arxiv_results": "dict", "user_memory_summary": "dict"}, output_schema={"ranked_papers": "list"}, implementation="search.PersonalizePaperResultsAdapter", input_model=PersonalizePaperResultsInput, output_model=RankedPapersOutput, error_model=ToolError, adapter=PersonalizePaperResultsAdapter()),
    _contract("synthesize_arxiv_response", description="汇总 arXiv 检索结果并生成最终回复。", capability_tags=["answer"], input_schema={"ranked_papers": "list", "warnings": "list"}, output_schema={"final_answer": "str"}, implementation="search.SynthesizeArxivResponseAdapter", input_model=SynthesizeArxivResponseInput, output_model=SearchAnswerOutput, error_model=ToolError, adapter=SynthesizeArxivResponseAdapter()),
    _contract("resolve_paper", description="从消息、选中论文或上下文中解析目标论文。", capability_tags=["retrieve"], input_schema={"message": "str", "context": "dict"}, output_schema={"paper_reference": "dict"}, implementation="paper_qa.ResolvePaperAdapter", input_model=ResolvePaperInput, output_model=PaperReferenceOutput, error_model=ToolError, recovery_policy=_default_recovery("ask_clarification"), adapter=ResolvePaperAdapter()),
    _contract("check_paper_index", description="检查目标论文是否已有 Paper QA 索引。", capability_tags=["retrieve", "validate"], input_schema={"paper_ref": "dict"}, output_schema={"status": "str", "has_index": "bool"}, implementation="paper_qa.CheckPaperIndexAdapter", backend_tool_name="check_paper_qa_index", input_model=CheckPaperIndexInput, output_model=PaperIndexStatusOutput, error_model=ToolError, recovery_policy=_default_recovery("request_confirmation", "patch_plan"), adapter=CheckPaperIndexAdapter(invoke_backend_tool)),
    _contract("request_confirmation", description="生成等待用户确认的结构化状态。", capability_tags=["clarify", "confirm"], input_schema={"pending_action": "dict"}, output_schema={"status": "str", "pending_action": "dict"}, side_effect_level="session_write", implementation="clarification.RequestConfirmationAdapter", input_model=RequestConfirmationInput, output_model=ConfirmationStatusOutput, error_model=ToolError, confirmation_policy={"mode": "explicit_user_confirmation_required"}, adapter=RequestConfirmationAdapter()),
    _contract("parse_and_index_paper", description="调用 Paper QA 索引构建工具解析并索引目标论文。", capability_tags=["retrieve", "index"], input_schema={"paper_reference": "dict"}, output_schema={"status": "str", "tool_result": "dict"}, side_effect_level="external_call", requires_confirmation=True, failure_modes=["paper_not_found", "index_build_failed"], implementation="paper_qa.ParseAndIndexPaperAdapter", backend_tool_name="build_paper_qa_index", input_model=ParseAndIndexPaperInput, output_model=IndexBuildOutput, error_model=ToolError, recovery_policy=_default_recovery("request_confirmation", "retry_step"), confirmation_policy={"mode": "explicit_user_confirmation_required", "reason": "external_index_build"}, adapter=ParseAndIndexPaperAdapter(invoke_backend_tool)),
    _contract("answer_paper_question", description="调用真实 PaperQAService 回答目标论文问题。", capability_tags=["answer"], input_schema={"paper_ref": "dict", "message": "str"}, output_schema={"paper_qa_result": "dict"}, side_effect_level="external_call", implementation="paper_qa.AnswerPaperQuestionAdapter", backend_tool_name="answer_paper_question", input_model=AnswerPaperQuestionInput, output_model=PaperQAAnswerOutput, error_model=ToolError, recovery_policy=_default_recovery("patch_plan", "fallback_answer"), adapter=AnswerPaperQuestionAdapter(invoke_backend_tool)),
    _contract("load_user_profile", description="从上下文读取用户画像与记忆摘要。", capability_tags=["retrieve", "profile"], input_schema={"context": "dict"}, output_schema={"recommendation_profile": "dict"}, implementation="recommendation.LoadUserProfileAdapter", input_model=LoadUserProfileInput, output_model=UserProfileOutput, error_model=ToolError, recovery_policy=_default_recovery("patch_plan"), adapter=LoadUserProfileAdapter()),
    _contract("load_candidate_papers", description="从上下文读取已有候选论文列表。", capability_tags=["retrieve", "recommendation"], input_schema={"context": "dict"}, output_schema={"candidate_papers": "list"}, implementation="recommendation.LoadCandidatePapersAdapter", input_model=LoadCandidatePapersInput, output_model=CandidatePapersOutput, error_model=ToolError, adapter=LoadCandidatePapersAdapter()),
    _contract("generate_recommendations", description="调用后端推荐工具生成论文推荐。", capability_tags=["recommendation"], input_schema={"recommendation_profile": "dict", "candidate_papers": "list"}, output_schema={"recommendations": "list", "tool_result": "dict"}, side_effect_level="external_call", implementation="recommendation.GenerateRecommendationsAdapter", backend_tool_name="recommend_papers", input_model=GenerateRecommendationsInput, output_model=RecommendationResultOutput, error_model=ToolError, recovery_policy=_default_recovery("patch_plan", "fallback_answer"), adapter=GenerateRecommendationsAdapter(invoke_backend_tool)),
    _contract("validate_recommendations", description="检查推荐结果是否包含可展示候选项。", capability_tags=["validate", "recommendation"], input_schema={"recommendation_result": "dict"}, output_schema={"ok": "bool", "recommendations": "list"}, implementation="recommendation.ValidateRecommendationsAdapter", input_model=ValidateRecommendationsInput, output_model=ValidatedRecommendationsOutput, error_model=ToolError, recovery_policy=_default_recovery("patch_plan"), adapter=ValidateRecommendationsAdapter()),
    _contract("explain_recommendations", description="把推荐结果整理成用户可读回复。", capability_tags=["answer", "recommendation"], input_schema={"validated_recommendations": "dict"}, output_schema={"final_answer": "str"}, implementation="recommendation.ExplainRecommendationsAdapter", input_model=ExplainRecommendationsInput, output_model=RecommendationAnswerOutput, error_model=ToolError, adapter=ExplainRecommendationsAdapter()),
    _contract("resolve_preference_target", description="解析偏好更新要作用到哪篇论文。", capability_tags=["retrieve", "preference"], input_schema={"message": "str", "context": "dict"}, output_schema={"paper_reference": "dict"}, implementation="preference.ResolvePreferenceTargetAdapter", input_model=ResolvePreferenceTargetInput, output_model=PreferenceTargetOutput, error_model=ToolError, recovery_policy=_default_recovery("ask_clarification"), adapter=ResolvePreferenceTargetAdapter(ResolvePaperAdapter())),
    _contract("update_preference_store", description="写入或删除用户论文偏好。", capability_tags=["memory_write", "preference"], input_schema={"paper_reference": "dict", "message": "str"}, output_schema={"preference_action_result": "dict"}, side_effect_level="persistent_write", implementation="preference.UpdatePreferenceStoreAdapter", backend_tool_name="record_paper_preference/remove_paper_preference", input_model=UpdatePreferenceStoreInput, output_model=PreferenceActionOutput, error_model=ToolError, recovery_policy=_default_recovery("retry_step", "ask_clarification"), adapter=UpdatePreferenceStoreAdapter(invoke_backend_tool)),
    _contract("verify_preference_update", description="校验偏好写入工具是否返回成功状态。", capability_tags=["validate", "preference"], input_schema={"preference_action_result": "dict"}, output_schema={"ok": "bool", "detail": "dict"}, implementation="preference.VerifyPreferenceUpdateAdapter", input_model=VerifyPreferenceUpdateInput, output_model=VerifiedPreferenceOutput, error_model=ToolError, recovery_policy=_default_recovery("retry_step"), adapter=VerifyPreferenceUpdateAdapter()),
    _contract("synthesize_preference_response", description="把偏好更新结果整理成最终回复。", capability_tags=["answer", "preference"], input_schema={"verified_preference_update": "dict"}, output_schema={"final_answer": "str"}, implementation="preference.SynthesizePreferenceResponseAdapter", input_model=SynthesizePreferenceResponseInput, output_model=PreferenceResponseOutput, error_model=ToolError, adapter=SynthesizePreferenceResponseAdapter()),
    _contract("analyze_ambiguity", description="分析模糊请求缺少的关键信息。", capability_tags=["clarify"], input_schema={"message": "str"}, output_schema={"missing_information": "dict"}, implementation="clarification.AnalyzeAmbiguityAdapter", input_model=AnalyzeAmbiguityInput, output_model=MissingInformationOutput, error_model=ToolError, adapter=AnalyzeAmbiguityAdapter()),
    _contract("generate_clarification", description="根据缺失信息生成澄清问题。", capability_tags=["clarify", "answer"], input_schema={"missing_information": "dict"}, output_schema={"final_answer": "str"}, implementation="clarification.GenerateClarificationAdapter", input_model=GenerateClarificationInput, output_model=ClarificationAnswerOutput, error_model=ToolError, adapter=GenerateClarificationAdapter()),
    _contract("generate_fallback_response", description="为不支持的请求生成兜底回复。", capability_tags=["fallback", "answer"], input_schema={"message": "str"}, output_schema={"final_answer": "str"}, implementation="clarification.GenerateFallbackResponseAdapter", input_model=GenerateFallbackInput, output_model=ClarificationAnswerOutput, error_model=ToolError, adapter=GenerateFallbackResponseAdapter()),
]:
    UNIFIED_TOOL_REGISTRY.register(contract)


# 兼容旧调用点：名称保留，但对象本身就是统一 registry，不再是独立 planner 事实来源。
PLANNER_TOOL_REGISTRY = UNIFIED_TOOL_REGISTRY


__all__ = [
    "CONTRACT_SOURCE",
    "PLANNER_TOOL_REGISTRY",
    "ToolAdapter",
    "ToolContract",
    "ToolRegistry",
    "UNIFIED_TOOL_REGISTRY",
]
