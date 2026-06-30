from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


@lru_cache(maxsize=1)
def get_valid_arxiv_categories() -> Set[str]:
    """加载当前项目认可的 arXiv 学科分类集合。

    这个函数的目标不是“随便给一份分类表”，而是尽量与仓库里已经存在的
    arXiv taxonomy 保持一致。这样 Agent 在做分类校验、默认范围裁剪时，
    就不会凭空发明一套和项目其他模块不一致的类别体系。

    整体策略是：
    1. 优先从 `ArxivSearchService.get_subject_categories()` 读取项目已有分类源；
    2. 如果服务导入失败或运行异常，则退回一份保守的默认分类集合；
    3. 用 `lru_cache` 缓存结果，避免每次校验字段都重复读取分类表。
    """
    try:
        from services.arxiv.arxiv_search_service import ArxivSearchService

        categories: Set[str] = set()
        for item in ArxivSearchService.get_subject_categories():
            code = str((item or {}).get("code", "")).strip()
            if code:
                categories.add(code)
        if categories:
            return categories
    except Exception:
        pass

    # 保守兜底：即使分类源不可用，也至少保证 Agent 还能在核心 AI 相关类别中工作。
    return {"cs.AI", "cs.CL", "cs.IR", "cs.LG"}


@lru_cache(maxsize=1)
def get_default_agent_arxiv_categories() -> List[str]:
    """返回当前 arXiv Agent 默认使用的分类搜索范围。

    这里体现的是一个重要设计原则：
    Agent 的分类范围应当由项目配置和后端规则统一控制，而不是每次交给 LLM 自行猜测。
    这样做的好处包括：
    - 检索范围稳定；
    - 与项目运行配置一致；
    - 避免模型输出不合法或过于发散的分类代码。

    整体流程是：
    1. 读取项目配置里的 `target_categories`；
    2. 与合法分类集合做交集校验；
    3. 去重并保留顺序；
    4. 若配置缺失，则退回默认的 AI/NLP/IR/LG 分类组合。
    """
    valid_categories = get_valid_arxiv_categories()

    try:
        from utils.config import get_arxiv_oai_runtime_config

        configured_categories = [
            str(item).strip()
            for item in get_arxiv_oai_runtime_config().get("target_categories", [])
            if str(item).strip()
        ]
    except Exception:
        configured_categories = []

    normalized: List[str] = []
    seen = set()
    for category in configured_categories:
        if category in valid_categories and category not in seen:
            seen.add(category)
            normalized.append(category)

    if normalized:
        return normalized

    return [category for category in ("cs.CL", "cs.LG", "cs.IR", "cs.AI") if category in valid_categories]


class ArxivSearchRequest(BaseModel):
    """定义搜索 Agent 接口接收的一次原始请求。

    这是用户请求进入 Agent 之前最外层的输入模型。
    它只关心“请求长什么样”，而不关心“请求最终会被识别成什么意图”。

    包含的主要信息有：
    - user_id / session_id：用于跨轮上下文、个性化推荐与追踪；
    - message：用户原始输入；
    - context：前端或上游链路补充的上下文信息。
    """
    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    resume: Optional["ResumeRequest"] = None

    @field_validator("user_id", "session_id", "message", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        # 在模型入参阶段统一做基础文本清洗，避免后续节点反复 strip。
        if value is None:
            return None
        text = str(value).strip()
        return text

    @field_validator("resume", mode="before")
    @classmethod
    def _coerce_resume(cls, value: Any) -> Any:
        # 新前端应直接传结构化 resume；这里顺手兼容对象/字典两种输入形态，
        # 避免 service 层还要到处判断具体载荷类型。
        if value in (None, "", {}):
            return None
        model_dump = getattr(value, "model_dump", None)
        return model_dump() if callable(model_dump) else value

    @model_validator(mode="after")
    def _validate_message(self) -> "ArxivSearchRequest":
        # message 是整个 Agent 的最核心输入字段，必须保证非空。
        if not self.message or not str(self.message).strip():
            raise ValueError("message cannot be empty")
        self.message = str(self.message).strip()
        return self


class ArxivSearchSpec(BaseModel):
    """定义一次 arXiv 结构化搜索的标准规格。

    这个模型是“自然语言请求”和“底层检索工具参数”之间的中间层契约。
    parse_search_request 节点会尝试把用户输入解析成这个结构，再由后续节点把它转成
    真正的工具入参。

    它主要表达的是：
    - 搜索主题词在哪个字段检索（query / title_query / abstract_query）；
    - 搜索范围在哪些分类中执行；
    - 时间范围、返回条数、排序方式是什么；
    - 以及一份面向调试/解释的 reasoning_summary。
    """
    model_config = ConfigDict(extra="forbid")

    intent: Literal["arxiv_search"]
    query: Optional[str] = None
    title_query: Optional[str] = None
    abstract_query: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    submitted_days_ago: Optional[int] = None
    max_results: int = 10
    sort_by: str = "submittedDate"
    sort_order: str = "descending"
    field_operator: str = "AND"
    category_operator: str = "OR"
    reasoning_summary: Optional[str] = None

    @field_validator("query", "title_query", "abstract_query", "reasoning_summary", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> Any:
        # 可选文本字段统一裁剪空白，并把空字符串转成 None，便于后续判断“是否提供了值”。
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("categories", mode="before")
    @classmethod
    def _normalize_categories(cls, value: Any) -> List[str]:
        # categories 既要支持 list/tuple/set，也要支持单个字符串；
        # 同时必须严格校验是否属于项目认可的 arXiv 分类代码。
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raise ValueError("categories must be a list of category codes")

        valid_categories = get_valid_arxiv_categories()
        canonical_map = {category.lower(): category for category in valid_categories}
        normalized: List[str] = []
        invalid: List[str] = []
        seen = set()
        for item in raw_items:
            code = str(item).strip()
            if not code:
                continue
            canonical_code = canonical_map.get(code.lower())
            if canonical_code is None:
                invalid.append(code)
                continue
            if canonical_code not in seen:
                seen.add(canonical_code)
                normalized.append(canonical_code)

        if invalid:
            raise ValueError(f"invalid arxiv categories: {', '.join(invalid)}")
        return normalized

    @field_validator("submitted_days_ago", mode="before")
    @classmethod
    def _normalize_submitted_days_ago(cls, value: Any) -> Any:
        # 最近 N 天是一个常用的自然语言约束，这里统一归一化成非负整数。
        if value is None or value == "":
            return None
        days = int(value)
        if days < 0:
            raise ValueError("submitted_days_ago must be greater than or equal to 0")
        return days

    @field_validator("max_results", mode="before")
    @classmethod
    def _normalize_max_results(cls, value: Any) -> Any:
        # 控制返回条数上限，避免用户一次性请求过多结果，也避免 LLM 输出异常大值。
        if value is None or value == "":
            return 10
        results = int(value)
        if not (1 <= results <= 20):
            raise ValueError("max_results must be between 1 and 20")
        return results

    @field_validator("sort_by", mode="before")
    @classmethod
    def _normalize_sort_by(cls, value: Any) -> str:
        # 仅允许后端检索工具真正支持的排序字段。
        text = str(value or "").strip()
        if not text:
            return "submittedDate"
        if text not in {"relevance", "lastUpdatedDate", "submittedDate"}:
            raise ValueError("sort_by must be one of relevance, lastUpdatedDate, submittedDate")
        return text

    @field_validator("sort_order", mode="before")
    @classmethod
    def _normalize_sort_order(cls, value: Any) -> str:
        # 排序方向统一收敛为 ascending / descending 两种。
        text = str(value or "").strip()
        if not text:
            return "descending"
        if text not in {"ascending", "descending"}:
            raise ValueError("sort_order must be one of ascending, descending")
        return text

    @field_validator("field_operator", mode="before")
    @classmethod
    def _normalize_field_operator(cls, value: Any) -> str:
        # 多字段组合检索时使用的布尔操作符。
        text = str(value or "").strip().upper()
        if not text:
            return "AND"
        if text not in {"AND", "OR", "ANDNOT"}:
            raise ValueError("field_operator must be one of AND, OR, ANDNOT")
        return text

    @field_validator("category_operator", mode="before")
    @classmethod
    def _normalize_category_operator(cls, value: Any) -> str:
        # 多类别组合时使用的布尔操作符。
        text = str(value or "").strip().upper()
        if not text:
            return "OR"
        if text not in {"AND", "OR"}:
            raise ValueError("category_operator must be one of AND, OR")
        return text

    @model_validator(mode="after")
    def _validate_search_fields(self) -> "ArxivSearchSpec":
        # 至少要有一个真实可用的检索入口，不能既没有 query，也没有 title/abstract/category 约束。
        if not any([self.query, self.title_query, self.abstract_query, self.categories]):
            raise ValueError("at least one search field or category must be provided")
        return self


class AgentToolCall(BaseModel):
    """记录一次工具调用的结构化轨迹。

    这个模型不会直接参与业务判断，但对调试和可观测性非常重要。
    它描述的是：
    - 调了哪个工具；
    - 传了什么参数；
    - 成功还是失败；
    - 摘要说明是什么；
    - trace / error 里有哪些进一步的细节。
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: str
    summary: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class ToolCallRequest(BaseModel):
    """定义 Agent 准备发起的一次内部工具调用请求。

    这个模型表达的不是“工具已经执行了什么”，而是“Agent 接下来打算怎么行动”。
    阶段 2 中它主要作为计划与行动之间的桥梁数据结构存在，便于后续把
    execution_plan 中的某一步自然映射成标准化工具调用协议。
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None
    expected_result: Optional[str] = None
    plan_step_id: Optional[str] = None
    fallback_tools: List[str] = Field(default_factory=list)


class ToolObservation(BaseModel):
    """定义一次工具执行后的标准化观察结果。

    与 `ToolCallRequest` 配对使用：前者描述“准备调用什么”，后者描述
    “执行之后观察到了什么、是否足够支持下一步决策”。

    这里刻意把语义收敛到一组稳定字段，供 graph 路由和 response_node 直接消费：
    1. ok：工具是否执行成功；
    2. status：success / failed / skipped 等稳定状态；
    3. result_summary：简短结果摘要；
    4. result_ref：轻量结果引用，不承载巨大原始数据；
    5. error：结构化错误；
    6. is_sufficient：当前结果是否足够支持下一步；
    7. next_action_hint：结果失败或不足时建议的下一步。
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: Optional[str] = None
    ok: Optional[bool] = None
    status: Optional[Literal["success", "failed", "skipped"]] = None
    result_summary: Optional[str] = None
    result_ref: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    is_sufficient: Optional[bool] = None
    next_action_hint: Optional[
        Literal[
            "retry_with_relaxed_query",
            "ask_user_confirmation",
            "ask_user_for_target_paper",
            "inspect_tool_request_or_choose_alternative",
            "answer_with_available_context",
            "fallback_to_normal_search",
            "request_user_confirmation_before_building_index",
            "inspect_build_result_or_retry_index_build",
            "inspect_answer_result_or_retry_with_adjusted_question",
            "adjust_recommendation_constraints_or_collect_more_preferences",
            "check_existing_preference_or_retry_mutation",
        ]
    ] = None
    raw_trace: Optional[Dict[str, Any]] = None


class AgentStep(BaseModel):
    """记录 LangGraph 中某一个节点步骤的执行摘要。

    与 `AgentToolCall` 相比，这个模型关注的是“节点级”执行过程，而不是单次工具调用。
    它通常会被写入 `AgentState.steps`，用于：
    - 前端展示执行进度；
    - 调试时回放节点执行轨迹；
    - 失败后快速定位是哪个阶段出了问题。
    """
    model_config = ConfigDict(extra="forbid")

    step: str
    status: Literal["success", "failed", "skipped"]
    action: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class Goal(BaseModel):
    """定义 Agent 对用户本轮目标的结构化理解。

    `intent` 主要回答“这是什么类型的任务”，而 `Goal` 更关注：
    - 用户真正想达成什么；
    - 本轮任务范围有多大；
    - 是否存在约束、成功标准；
    - 是否需要读取记忆或等待用户确认。

    阶段 1 中它主要作为状态表达、调试和后续规划能力的基础数据结构。
    """
    model_config = ConfigDict(extra="forbid")

    goal_id: Optional[str] = None
    goal_type: Optional[str] = None
    user_request: Optional[str] = None
    intent: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    context_refs: List[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    # 兼容旧响应读取路径，后续新逻辑不再依赖这些字段。
    user_goal: Optional[str] = None
    task_scope: Optional[str] = None


# =============================================================================
# Research Task Profile 语义层（intent 与工具计划之间的科研任务语义）
# -----------------------------------------------------------------------------
# 设计动机见 docs/项目说明：intent 只说明“系统进入哪条能力链路”，而 research_task_type
# 说明“用户正处于哪一类科研任务场景”。同一个 intent=arxiv_search 可能对应方向探索、
# 多论文比较或阅读规划等不同科研任务，因此需要一个与 intent 并行存在的结构化 Profile，
# 显式表达任务类型、任务对象、中间产物、证据需求和置信度，供后续 planner 从
# “科研任务 → 中间产物 → 证据需求 → 工具计划”而不是“intent → 工具”进行规划。
#
# 枚举刻意固定，避免后续规则或 LLM 自由创造标签导致语义漂移。
# =============================================================================

# 固定的科研任务类型枚举：方向探索 / 多论文比较 / 单篇深读 / 阅读规划 / 研究空白分析 / 个性化推荐。
ResearchTaskType = Literal[
    "direction_exploration",
    "multi_paper_comparison",
    "single_paper_deep_read",
    "reading_planning",
    "research_gap_analysis",
    "personalized_recommendation",
]

# 任务对象类型：主题词、单篇论文、论文集合或用户画像。
ResearchTaskObjectType = Literal["topic", "paper", "paper_set", "user_profile"]

# 后续 planner 需要消费的“中间产物”枚举（工作产物，而非工具本身）。
ResearchTaskArtifact = Literal[
    "candidate_paper_set",
    "representative_paper_set",
    "method_cards",
    "experiment_info",
    "comparison_matrix",
    "reading_order",
]

# 中间产物需要的“证据需求”枚举（这些产物要由什么证据支撑）。
ResearchTaskEvidence = Literal[
    "metadata",
    "abstract",
    "method_chunk",
    "experiment_chunk",
    "table_evidence",
    "user_profile_evidence",
]


# Profile 的本地可执行性状态：ready 表示可按当前上下文规划；needs_retrieval 表示需先检索候选；
# needs_user_clarification 表示缺少目标论文/主题等关键条件；fallback_intent_route 表示语义层放弃接管。
ResearchTaskExecutionReadiness = Literal[
    "ready",
    "needs_retrieval",
    "needs_user_clarification",
    "fallback_intent_route",
]

class ResearchTaskObject(BaseModel):
    """描述本轮科研任务真正作用的对象。

    object_type 决定后续 planner 应该围绕主题检索、围绕单篇深读，还是围绕论文集合做比较；
    topic / paper_refs / description 则是更细的对象描述，便于 debug 解释“任务对象是什么”。
    """
    model_config = ConfigDict(extra="forbid")

    object_type: ResearchTaskObjectType
    topic: Optional[str] = None
    paper_refs: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class ResearchTaskConstraints(BaseModel):
    """承载用户对科研任务显式提出的约束。

    这些约束来自请求和 search_spec，例如时间范围、数量上限、研究领域，以及是否要结合个人兴趣。
    它们不替代 search_spec，而是把“科研任务层面关心的限制”单独抽出来，供 planner 解释和复用。
    """
    model_config = ConfigDict(extra="forbid")

    time_range: Optional[str] = None
    max_count: Optional[int] = None
    research_fields: List[str] = Field(default_factory=list)
    combine_with_interest: bool = False


class ResearchTaskProfile(BaseModel):
    """科研任务语义层的统一数据契约。

    它与 intent 并行存在：intent 表示系统能力入口，research_task_type 表示用户处于哪一类科研任务。
    第一版不引入复杂 ResearchTaskGraph，只用结构化 Profile 明确：
    - research_task_type：科研任务类型（固定枚举）；
    - task_object：任务对象（主题/论文/论文集合/用户画像）；
    - constraints：显式约束（时间范围/数量/领域/是否结合兴趣）；
    - intermediate_artifacts：后续 planner 需要产出的中间产物；
    - evidence_requirements：这些产物需要的证据支撑；
    - confidence / classification_basis / needs_clarification：分类置信度、分类依据与是否需要澄清。

    Profile 不替代 Goal、intent 或 ExecutablePlan：Goal 表示用户目标，intent 表示系统能力入口，
    ExecutablePlan 仍表示最终可执行工具步骤；Profile 只补齐二者之间缺失的科研任务语义。
    """
    model_config = ConfigDict(extra="forbid")

    profile_id: Optional[str] = None
    research_task_type: ResearchTaskType
    secondary_task_types: List[ResearchTaskType] = Field(default_factory=list)
    intent: Optional[str] = None
    goal_type: Optional[str] = None
    task_object: ResearchTaskObject
    constraints: ResearchTaskConstraints = Field(default_factory=ResearchTaskConstraints)
    intermediate_artifacts: List[ResearchTaskArtifact] = Field(default_factory=list)
    evidence_requirements: List[ResearchTaskEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    classification_basis: Optional[str] = None
    needs_clarification: bool = False
    execution_readiness: ResearchTaskExecutionReadiness = "ready"
    arbitration_notes: List[str] = Field(default_factory=list)
    classification_trace: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None


# =============================================================================
# Artifact / Evidence Plan 契约层（科研任务语义与工具计划之间的证据规划）
# -----------------------------------------------------------------------------
# ResearchTaskProfile 只回答“用户正在做哪类科研任务”；ArtifactEvidencePlan 继续回答：
# 为了完成该任务，需要哪些科研中间产物、每个产物需要哪些证据、这些证据是否被当前系统能力支持。
#
# 这里的类型集合全部固定，目的是把 LLM/规则产出的自然语言判断收敛到系统可识别的能力边界内。
# 该层只描述科研产物和证据需求，不直接选择工具，也不替代 ExecutablePlan / PlanRuntime。
# =============================================================================

# 当前系统允许进入主流程的科研中间产物类型。暂不稳定的外部统计、引用图或复现实验不进入第一版集合。
ArtifactType = Literal[
    "topic_term_set",
    "candidate_paper_set",
    "selected_paper_set",
    "paper_feature_card_set",
    "comparison_dimension_set",
    "comparison_matrix",
    "reading_plan",
    "gap_hypothesis_set",
    "grounded_summary",
    "user_profile_match",
]

# 当前 RAG / 画像 / 上下文层能直接获取或能明确降级处理的证据类型。
EvidenceType = Literal[
    "metadata",
    "abstract",
    "paper_chunk",
    "method_section",
    "experiment_section",
    "result_section",
    "limitation_section",
    "table",
    "figure",
    "user_profile",
    "previous_context",
]

# 能力状态用于向 Tool Planner / Observer / Replanner 显式暴露边界，而不是让下游猜测证据是否可取。
CapabilityStatus = Literal[
    "supported",
    "degraded",
    "requires_index",
    "requires_user_profile",
    "requires_confirmation",
    "blocked",
    "unsupported",
]

ArtifactTargetScope = Literal["topic", "paper", "paper_set", "user_profile", "conversation"]
ArtifactConsumer = Literal["tool_planner", "observer", "replanner", "response_synthesizer"]
EvidenceSourceScope = Literal["arxiv_metadata", "search_results", "paper_index", "user_profile", "previous_context"]
EvidenceSection = Literal["title", "abstract", "method", "experiment", "result", "limitation", "table", "figure", "profile", "context"]
ArtifactFallbackStrategy = Literal[
    "none",
    "skip_optional",
    "generate_partial_summary",
    "ask_user_clarification",
    "fallback_to_generic_route",
    "defer_until_evidence_ready",
]
EvidenceFallbackPolicy = Literal[
    "none",
    "use_metadata_abstract_surrogate",
    "use_previous_context",
    "ask_user_clarification",
    "skip_optional_artifact",
    "generic_without_profile",
    "defer_until_index_ready",
]
ConfidenceImpact = Literal["none", "low", "medium", "high"]
EvidencePriority = Literal["low", "normal", "high"]
PlanningDiagnosticSeverity = Literal["info", "warning", "error"]
ArtifactPlanValidationIssueSeverity = Literal["info", "warning", "error"]
ArtifactContributionType = Literal[
    "acquire_evidence",
    "validate_evidence",
    "derive_artifact",
    "synthesize_artifact",
    "quality_gate",
    "context_reuse",
]
ArtifactProgressStatus = Literal["pending", "partial", "completed", "degraded", "blocked"]

ArtifactRefinementPatchOperation = Literal[
    "enable_optional_artifact",
    "disable_optional_artifact",
    "raise_evidence_priority",
    "lower_evidence_priority",
    "add_target_field",
    "remove_optional_field",
    "refine_comparison_dimension",
    "set_scope_constraint",
    "set_budget_hint",
    "add_fallback_note",
]


class EvidenceRequirement(BaseModel):
    """描述某个 artifact 的字段需要什么证据来支撑。

    EvidenceRequirement 绑定 target_artifact_id 与 target_fields，避免后续 Observer 只看到
    “需要摘要/方法”这种粗标签却无法判断哪个产物、哪个字段已经被证据覆盖。
    """
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    target_artifact_id: str
    required: bool = True
    target_fields: List[str] = Field(default_factory=list, min_length=1)
    evidence_type: EvidenceType
    source_scope: EvidenceSourceScope
    source_preference: List[EvidenceSourceScope] = Field(default_factory=list)
    preferred_sections: List[EvidenceSection] = Field(default_factory=list)
    min_evidence_count: int = Field(default=1, ge=0)
    coverage_criteria: List[str] = Field(default_factory=list, min_length=1)
    fallback_policy: EvidenceFallbackPolicy = "none"
    confidence_impact: ConfidenceImpact = "medium"
    capability_status: CapabilityStatus = "supported"
    priority: EvidencePriority = "normal"
    field_weights: Dict[str, float] = Field(default_factory=dict)
    scope_notes: List[str] = Field(default_factory=list)

    @field_validator("requirement_id", "target_artifact_id")
    @classmethod
    def _require_non_empty_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("EvidenceRequirement id fields cannot be empty")
        return text

    @model_validator(mode="after")
    def _normalize_source_preference(self) -> "EvidenceRequirement":
        # source_preference 是给检索/Observer 的偏好顺序；未显式设置时至少保留主 source_scope。
        if not self.source_preference:
            self.source_preference = [self.source_scope]
        return self


class Artifact(BaseModel):
    """科研中间产物的结构化定义。

    Artifact 只表达“要产出什么”和“证据完成标准是什么”，不表达具体调用哪个工具；
    工具选择仍由 Tool Planner / ExecutablePlan 负责。
    """
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: ArtifactType
    depends_on: List[str] = Field(default_factory=list)
    required: bool = True
    prunable: bool = False
    target_scope: ArtifactTargetScope
    expected_output: str
    target_fields: List[str] = Field(default_factory=list)
    quality_criteria: List[str] = Field(default_factory=list, min_length=1)
    scope_constraints: List[str] = Field(default_factory=list)
    fallback_strategy: ArtifactFallbackStrategy = "none"
    fallback_notes: List[str] = Field(default_factory=list)
    consumer: List[ArtifactConsumer] = Field(default_factory=list, min_length=1)
    evidence_requirements: List[EvidenceRequirement] = Field(default_factory=list)
    budget_cost: Dict[str, int] = Field(default_factory=dict)

    @field_validator("artifact_id")
    @classmethod
    def _require_non_empty_artifact_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Artifact artifact_id cannot be empty")
        return text

    @model_validator(mode="after")
    def _validate_artifact_contract(self) -> "Artifact":
        if self.required and not self.evidence_requirements:
            raise ValueError(f"Required artifact {self.artifact_id} must define evidence_requirements")
        for requirement in self.evidence_requirements:
            if requirement.target_artifact_id != self.artifact_id:
                raise ValueError(
                    f"EvidenceRequirement {requirement.requirement_id} targets {requirement.target_artifact_id}, "
                    f"but is nested under artifact {self.artifact_id}"
                )
            if self.required and requirement.capability_status == "unsupported":
                # unsupported 只能作为边界诊断存在，不能进入 required 主流程，避免后续 planner 规划不可兑现的证据。
                raise ValueError(f"Required artifact {self.artifact_id} cannot depend on unsupported evidence")
        return self


class PlanningDiagnostic(BaseModel):
    """Artifact / Evidence 规划过程中的可观测诊断事件。"""
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: PlanningDiagnosticSeverity = "info"
    message: str
    target_artifact_id: Optional[str] = None
    evidence_type: Optional[EvidenceType] = None
    capability_status: Optional[CapabilityStatus] = None


class ArtifactPlanValidationIssue(BaseModel):
    """ArtifactPlanValidator 输出的单条结构化问题。"""
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: ArtifactPlanValidationIssueSeverity = "error"
    message: str
    target_artifact_id: Optional[str] = None
    target_evidence_id: Optional[str] = None
    evidence_type: Optional[EvidenceType] = None
    capability_status: Optional[CapabilityStatus] = None


class ArtifactPlanValidationReport(BaseModel):
    """Validator-guarded planning 的结构化报告。

    report 进入 plan metadata/debug，便于后续确认是 schema、任务一致性、预算还是工具映射问题。
    """
    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "warning", "failed"] = "passed"
    issue_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    issues: List[ArtifactPlanValidationIssue] = Field(default_factory=list)
    checked_artifact_ids: List[str] = Field(default_factory=list)
    checked_evidence_ids: List[str] = Field(default_factory=list)
    final_artifact_ids: List[str] = Field(default_factory=list)
    validator_stack: List[str] = Field(default_factory=list)


class EvidenceCapabilityAlignment(BaseModel):
    """单条 EvidenceRequirement 与当前系统能力的对齐结果。"""
    model_config = ConfigDict(extra="forbid")

    target_artifact_id: str
    requirement_id: str
    evidence_type: EvidenceType
    target_fields: List[str] = Field(default_factory=list)
    capability_status: CapabilityStatus
    reason: str
    acquisition_tools: List[str] = Field(default_factory=list)
    fallback_policy: EvidenceFallbackPolicy = "none"
    confidence_impact: ConfidenceImpact = "medium"
    blocked: bool = False


class ArtifactStepMapping(BaseModel):
    """ExecutablePlan step 与 artifact/evidence requirement 的映射。

    PlanStep 暂不扩展 metadata 字段，因此映射集中保存在 ExecutablePlan.metadata；
    Observer/Replanner 后续可以按 step_id 反查该步骤预计推进哪个科研产物。
    """
    model_config = ConfigDict(extra="forbid")

    step_id: str
    tool_name: str
    target_artifact_id: str
    target_evidence_id: Optional[str] = None
    artifact_type: ArtifactType
    evidence_type: Optional[EvidenceType] = None
    contribution_type: ArtifactContributionType
    expected_artifact_update: str
    capability_status: Optional[CapabilityStatus] = None


class ArtifactProgressReservation(BaseModel):
    """执行前预留的 artifact 进度槽位。

    这里只记录 planner 对未来观察点的预期，不把执行结果提前写死；真实完成度仍由 Observer 根据工具输出推进。
    """
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_type: ArtifactType
    status: ArtifactProgressStatus = "pending"
    mapped_step_ids: List[str] = Field(default_factory=list)
    missing_evidence_ids: List[str] = Field(default_factory=list)
    blocked_evidence_ids: List[str] = Field(default_factory=list)


class ArtifactRefinementPatch(BaseModel):
    """LLM refinement 的唯一允许输出单元。

    LLM 只能提交这些受控 patch，不能直接生成 ArtifactEvidencePlan，也不能改写工具步骤；
    后续 ArtifactRefinementService 会逐条校验、部分接受，并把拒绝原因写入 diagnostics。
    """
    model_config = ConfigDict(extra="forbid")

    patch_id: Optional[str] = None
    operation: ArtifactRefinementPatchOperation
    target_artifact_id: Optional[str] = None
    target_evidence_type: Optional[EvidenceType] = None
    target_fields: List[str] = Field(default_factory=list)
    comparison_dimension: Optional[str] = None
    scope_constraint: Optional[str] = None
    budget_hint: Dict[str, int] = Field(default_factory=dict)
    fallback_note: Optional[str] = None
    priority: Optional[EvidencePriority] = None
    rationale: Optional[str] = None

    @model_validator(mode="after")
    def _validate_patch_shape(self) -> "ArtifactRefinementPatch":
        operation = self.operation
        if operation != "set_budget_hint" and not str(self.target_artifact_id or "").strip():
            raise ValueError(f"{operation} requires target_artifact_id")
        if operation in {"raise_evidence_priority", "lower_evidence_priority"} and not (
            self.target_evidence_type or self.target_fields
        ):
            raise ValueError(f"{operation} requires target_evidence_type or target_fields")
        if operation in {"add_target_field", "remove_optional_field"} and not self.target_fields:
            raise ValueError(f"{operation} requires target_fields")
        if operation == "refine_comparison_dimension" and not (self.comparison_dimension or self.target_fields):
            raise ValueError("refine_comparison_dimension requires comparison_dimension or target_fields")
        if operation == "set_scope_constraint" and not str(self.scope_constraint or "").strip():
            raise ValueError("set_scope_constraint requires scope_constraint")
        if operation == "set_budget_hint" and not self.budget_hint:
            raise ValueError("set_budget_hint requires budget_hint")
        if operation == "add_fallback_note" and not str(self.fallback_note or "").strip():
            raise ValueError("add_fallback_note requires fallback_note")
        return self


class ArtifactRefinementPatchSet(BaseModel):
    """LLM refinement 的 JSON-only 顶层协议。"""
    model_config = ConfigDict(extra="forbid")

    patches: List[ArtifactRefinementPatch] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ArtifactPlanningBudget(BaseModel):
    """Artifact skeleton 的成本边界。

    预算属于模板能力边界的一部分：Skeleton Builder 只把预算写入 plan metadata，
    后续 Tool Planner / Replanner 再据此控制检索、全文深读和 refinement 成本。
    """
    model_config = ConfigDict(extra="forbid")

    max_candidate_papers: Optional[int] = Field(default=None, ge=0)
    max_selected_papers: Optional[int] = Field(default=None, ge=0)
    max_deep_read_papers: Optional[int] = Field(default=None, ge=0)
    max_qa_calls: Optional[int] = Field(default=None, ge=0)
    max_refinement_rounds: Optional[int] = Field(default=None, ge=0)


class PlanningDiagnostics(BaseModel):
    """给 debug / metadata 使用的规划摘要。

    diagnostics 只描述契约构造和能力边界，不记录执行结果；执行期质量仍由 Observer / PlanRuntime 负责。
    """
    model_config = ConfigDict(extra="forbid")

    planner_source: Optional[str] = None
    template_id: Optional[str] = None
    template_version: Optional[str] = None
    research_task_type: Optional[ResearchTaskType] = None
    profile_source: Optional[str] = None
    classification_basis: Optional[str] = None
    budget: ArtifactPlanningBudget = Field(default_factory=ArtifactPlanningBudget)
    dependency_order: List[str] = Field(default_factory=list)
    capability_summary: Dict[str, int] = Field(default_factory=dict)
    unmet_evidence_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    pruned_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    retained_optional_artifacts: List[str] = Field(default_factory=list)
    degraded_evidence_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    context_adjustments: List[str] = Field(default_factory=list)
    template_defaults: Dict[str, Any] = Field(default_factory=dict)
    llm_refinement_attempted: bool = False
    llm_refinement_raw_summary: Dict[str, Any] = Field(default_factory=dict)
    llm_refinement_error: Optional[str] = None
    llm_refinement_patches: List[Dict[str, Any]] = Field(default_factory=list)
    accepted_refinement_patches: List[Dict[str, Any]] = Field(default_factory=list)
    rejected_refinement_patches: List[Dict[str, Any]] = Field(default_factory=list)
    local_corrections: List[Dict[str, Any]] = Field(default_factory=list)
    final_evidence_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    validation_report: ArtifactPlanValidationReport = Field(default_factory=ArtifactPlanValidationReport)
    capability_alignment: List[EvidenceCapabilityAlignment] = Field(default_factory=list)
    artifact_step_mapping: List[ArtifactStepMapping] = Field(default_factory=list)
    artifact_progress_reservations: List[ArtifactProgressReservation] = Field(default_factory=list)
    blocked_evidence_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostic_events: List[PlanningDiagnostic] = Field(default_factory=list)


class ArtifactEvidencePlan(BaseModel):
    """ResearchTaskProfile 与 ExecutablePlan 之间的统一 Artifact / Evidence 契约。

    这个对象可以直接序列化进 debug 和 plan metadata；下游只需要消费固定字段，
    不再依赖临时 dict 或自然语言说明来理解科研中间产物与证据需求。
    """
    model_config = ConfigDict(extra="forbid")

    plan_id: Optional[str] = None
    research_task_type: ResearchTaskType
    profile_source: Optional[str] = None
    classification_basis: Optional[str] = None
    artifacts: List[Artifact] = Field(default_factory=list, min_length=1)
    diagnostics: PlanningDiagnostics = Field(default_factory=PlanningDiagnostics)

    @model_validator(mode="after")
    def _validate_plan_contract(self) -> "ArtifactEvidencePlan":
        artifacts_by_id: Dict[str, Artifact] = {}
        for artifact in self.artifacts:
            if artifact.artifact_id in artifacts_by_id:
                raise ValueError(f"Duplicate artifact_id: {artifact.artifact_id}")
            artifacts_by_id[artifact.artifact_id] = artifact

        for artifact in self.artifacts:
            for dependency in artifact.depends_on:
                if dependency not in artifacts_by_id:
                    raise ValueError(f"Artifact {artifact.artifact_id} depends on unknown artifact {dependency}")
            for requirement in artifact.evidence_requirements:
                if requirement.target_artifact_id not in artifacts_by_id:
                    raise ValueError(
                        f"EvidenceRequirement {requirement.requirement_id} targets unknown artifact {requirement.target_artifact_id}"
                    )

        dependency_order = _topological_sort_artifact_ids(artifacts_by_id)
        self.diagnostics.dependency_order = self.diagnostics.dependency_order or dependency_order
        self.diagnostics.research_task_type = self.diagnostics.research_task_type or self.research_task_type
        self.diagnostics.profile_source = self.diagnostics.profile_source or self.profile_source
        self.diagnostics.classification_basis = self.diagnostics.classification_basis or self.classification_basis
        self.diagnostics.capability_summary = self.diagnostics.capability_summary or _capability_summary_from_artifacts(self.artifacts)
        return self


def _topological_sort_artifact_ids(artifacts_by_id: Dict[str, Artifact]) -> List[str]:
    """校验 artifact 依赖并返回拓扑顺序。

    当前依赖图只是少量 artifact_id 的 DAG，本地 DFS 足够且更轻；如果后续引入跨任务复杂图，
    再切换到 networkx 这类成熟图工具，避免在这里重复实现复杂图算法。
    """
    visited: Set[str] = set()
    visiting: Set[str] = set()
    ordered: List[str] = []

    def visit(artifact_id: str) -> None:
        if artifact_id in visited:
            return
        if artifact_id in visiting:
            raise ValueError(f"Artifact dependency graph contains cycle at {artifact_id}")
        visiting.add(artifact_id)
        for dependency in artifacts_by_id[artifact_id].depends_on:
            visit(dependency)
        visiting.remove(artifact_id)
        visited.add(artifact_id)
        ordered.append(artifact_id)

    for artifact_id in artifacts_by_id:
        visit(artifact_id)
    return ordered


def _capability_summary_from_artifacts(artifacts: List[Artifact]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for artifact in artifacts:
        for requirement in artifact.evidence_requirements:
            status = requirement.capability_status
            summary[status] = summary.get(status, 0) + 1
    return summary


class PlanDraftStep(BaseModel):
    """表示尚未被信任的单个规划草稿步骤。

    PlanDraft 只描述 planner 的意图和工具选择，不能直接交给 Executor；
    后续必须经过 ToolRegistry 解析、依赖校验和 PlanValidator 才能成为 ExecutablePlan。
    """
    model_config = ConfigDict(extra="forbid")

    step_id: str
    action_type: str
    tool_name: str
    step_reason: Optional[str] = None
    why_this_step: Optional[str] = None
    input_bindings: List["StepInputBinding"] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    expected_output_key: Optional[str] = None
    expected_output: Dict[str, Any] = Field(default_factory=dict)
    retry_policy: Optional["StepPolicy"] = None
    risk_level: Literal["low", "medium", "high"] = "low"
    risk_notes: Optional[str] = None
    requires_confirmation: bool = False
    failure_recovery_hint: Optional[str] = None
    fallback_reason: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_llm_protocol_aliases(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        # LLM 协议使用更面向解释的字段名；这里统一映射到既有内部字段，避免破坏旧 planner。
        if not normalized.get("step_reason") and normalized.get("why_this_step"):
            normalized["step_reason"] = normalized.get("why_this_step")
        expected_output = normalized.get("expected_output")
        if not normalized.get("expected_output_key") and expected_output is not None:
            if isinstance(expected_output, str):
                normalized["expected_output_key"] = expected_output
                normalized["expected_output"] = {"output_key": expected_output}
            elif isinstance(expected_output, dict):
                for key in ("output_key", "key", "name", "field"):
                    output_key = str(expected_output.get(key) or "").strip()
                    if output_key:
                        normalized["expected_output_key"] = output_key
                        break
        if not normalized.get("fallback_reason") and normalized.get("failure_recovery_hint"):
            normalized["fallback_reason"] = normalized.get("failure_recovery_hint")
        retry_policy = normalized.get("retry_policy")
        model_dump = getattr(retry_policy, "model_dump", None)
        if callable(model_dump):
            # planner 单测会通过不同模块路径重复加载 schema；同形 StepPolicy 先转 dict，避免类身份不一致。
            normalized["retry_policy"] = model_dump()
        return normalized


class PlanDraft(BaseModel):
    """Tool-Aware Planner 产出的不可信计划草稿。

    草稿中的 tool_name、依赖和输出键都只是候选声明；只有转换器校验通过并生成
    ExecutablePlan 后，执行器才允许消费对应计划。
    """
    model_config = ConfigDict(extra="forbid")

    draft_id: str
    plan_intent: Optional[str] = None
    selected_tools: List[str] = Field(default_factory=list)
    steps: List[PlanDraftStep] = Field(default_factory=list)
    fallback_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlannerToolContext(BaseModel):
    """Planner 输入层看到的工具能力快照。

    它只保留规划所需的静态 contract 信息，不携带 adapter 实例，避免 planner 输入
    与 executor 运行对象耦合。
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    description: Optional[str] = None
    capability_tags: List[str] = Field(default_factory=list)
    side_effect_level: Literal["none", "session_write", "persistent_write", "external_call"] = "none"
    requires_confirmation: bool = False
    can_retry: bool = False
    failure_modes: List[str] = Field(default_factory=list)
    recovery_policy: Dict[str, Any] = Field(default_factory=dict)
    confirmation_policy: Dict[str, Any] = Field(default_factory=dict)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


class PlannerContext(BaseModel):
    """Planner 本轮决策使用的统一上下文。

    后续 planner 只应消费这个稳定对象里的摘要和引用，避免继续从 AgentState、
    context、ToolRegistry 等位置临时拼装判断条件。
    """
    model_config = ConfigDict(extra="forbid")

    raw_user_request: Optional[str] = None
    normalized_goal: Optional[Goal] = None
    goal_type: Optional[str] = None
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    selected_paper: Optional[Dict[str, Any]] = None
    last_papers: List[Dict[str, Any]] = Field(default_factory=list)
    paper_qa_result: Optional[Dict[str, Any]] = None
    pending_action: Optional[Dict[str, Any]] = None
    user_memory_summary: Any = None
    research_profile: Any = None
    # research_task_profile 是 intent 与工具计划之间的科研任务语义层；planner 可据此从
    # “科研任务 → 中间产物 → 证据需求”推导计划，而不是直接 intent → 工具。缺失时不影响既有规划。
    research_task_profile: Any = None
    available_tools: List[PlannerToolContext] = Field(default_factory=list)
    available_tool_names: List[str] = Field(default_factory=list)
    context_refs: List[str] = Field(default_factory=list)
    context_field_summary: Dict[str, Any] = Field(default_factory=dict)
    session_state: Dict[str, Any] = Field(default_factory=dict)
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    reusable_outputs: Dict[str, Any] = Field(default_factory=dict)
    high_risk_tools: List[str] = Field(default_factory=list)

    @field_validator("normalized_goal", mode="before")
    @classmethod
    def _coerce_goal_model(cls, value: Any) -> Any:
        # 单测和运行时可能通过不同包路径加载 schema；同形 Goal 先转 dict，避免 planner context 构造失败。
        model_dump = getattr(value, "model_dump", None)
        return model_dump() if callable(model_dump) else value


class ToolCandidate(BaseModel):
    """候选工具筛选结果中的单个工具说明。"""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    capability_tags: List[str] = Field(default_factory=list)
    side_effect_level: str = "none"
    requires_confirmation: bool = False
    failure_modes: List[str] = Field(default_factory=list)
    recovery_policy: Dict[str, Any] = Field(default_factory=dict)
    confirmation_policy: Dict[str, Any] = Field(default_factory=dict)
    selection_reason: Optional[str] = None


class ExcludedToolCandidate(BaseModel):
    """记录被排除的工具及原因，便于 planner debug 判断候选边界是否过宽。"""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    capability_tags: List[str] = Field(default_factory=list)
    side_effect_level: str = "none"
    requires_confirmation: bool = False
    failure_modes: List[str] = Field(default_factory=list)
    recovery_policy: Dict[str, Any] = Field(default_factory=dict)
    confirmation_policy: Dict[str, Any] = Field(default_factory=dict)
    exclusion_reason: Optional[str] = None


class ToolCandidateSelection(BaseModel):
    """Tool Candidate Selector 的结构化输出。"""
    model_config = ConfigDict(extra="forbid")

    goal_type: Optional[str] = None
    candidate_tools: List[ToolCandidate] = Field(default_factory=list)
    excluded_tools: List[ExcludedToolCandidate] = Field(default_factory=list)
    selection_reason: Optional[str] = None
    risk_summary: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# LEGACY 计划模型区（兼容专用，新 planner 开发不要在这里扩展）
# -----------------------------------------------------------------------------
# 下面的 ExecutionPlanStep 是历史轻量计划步骤模型，仅供：
#   1) state.py 的 _coerce_legacy_execution_plan 把旧 step 列表转成 ExecutablePlan；
#   2) 历史测试构造只含 step_type/description 的步骤。
# 当前主路径计划模型是 PlanStep / ExecutablePlan / PlanRuntime / AgentRuntimeState，
# 新增计划能力请改这些模型，不要再扩展 ExecutionPlanStep。
# =============================================================================
class ExecutionPlanStep(BaseModel):
    """[LEGACY] 兼容旧版节点流使用的轻量计划步骤。

    新执行器使用 PlanStep/ExecutablePlan 承载工具契约；旧 LangGraph 节点和历史测试仍会
    构造只包含 step_type/description 的步骤，因此这里保留窄模型避免破坏旧入口。
    新开发不要依赖本模型，详见上方 LEGACY 计划模型区说明。
    """
    model_config = ConfigDict(extra="forbid")

    step_id: str
    step_type: str
    description: str
    expected_input: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "in_progress", "completed", "success", "failed", "skipped", "waiting_confirmation"] = "pending"
    depends_on: List[str] = Field(default_factory=list)


PlanStepStatus = Literal["pending", "running", "success", "failed", "skipped", "waiting_confirmation"]
AgentTurnStatus = Literal["success", "waiting_confirmation", "need_clarification", "failed", "fallback"]
SideEffectLevel = Literal["none", "low", "high"]
ConfirmationRequestType = Literal["tool_approval", "paper_target_confirmation"]
ConfirmationDecision = Literal["approve", "reject"]


class ToolSpec(BaseModel):
    """定义计划步骤要调用的真实工具及其静态约束。

    这是统一 ToolContract 投影给 planner/debug 的只读视图；真正的执行入口、
    输入输出模型和恢复/确认策略都来自同一份 contract，避免 planner 与 executor
    各自维护一套看似相同但可能漂移的工具描述。
    """
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    description: Optional[str] = None
    capability_tags: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    input_model: Optional[str] = None
    output_model: Optional[str] = None
    error_model: Optional[str] = None
    side_effect_level: Literal["none", "session_write", "persistent_write", "external_call"] = "none"
    requires_confirmation: bool = False
    can_retry: bool = False
    failure_modes: List[str] = Field(default_factory=list)
    implementation: Optional[str] = None
    backend_tool_name: Optional[str] = None
    adapter: Optional[str] = None
    recovery_policy: Dict[str, Any] = Field(default_factory=dict)
    confirmation_policy: Dict[str, Any] = Field(default_factory=dict)
    contract_source: Optional[str] = None


class StepInputBinding(BaseModel):
    """声明步骤输入的绑定来源，避免执行期再猜每个参数从哪里取。"""
    model_config = ConfigDict(extra="forbid")

    input_key: str
    source_type: Literal["state", "context", "goal", "search_spec", "step_output", "literal"]
    source_key: Optional[str] = None
    step_id: Optional[str] = None
    required: bool = True
    value: Any = None


class StepCondition(BaseModel):
    """定义步骤是否应该执行的结构化条件。"""
    model_config = ConfigDict(extra="forbid")

    condition_type: Literal["always", "field_exists", "field_equals", "step_output_exists", "step_status_is"]
    field_path: Optional[str] = None
    expected_value: Any = None
    step_id: Optional[str] = None
    negate: bool = False


class StepPolicy(BaseModel):
    """统一表达重试、失败兜底和用户确认策略。"""
    model_config = ConfigDict(extra="forbid")

    policy_type: Literal["retry", "failure", "confirmation"]
    mode: str
    max_attempts: int = 0
    fallback_step_id: Optional[str] = None
    requires_confirmation: bool = False
    note: Optional[str] = None


class StepResult(BaseModel):
    """记录单个计划步骤的结构化执行结果。"""
    model_config = ConfigDict(extra="forbid")

    step_id: str
    status: PlanStepStatus
    output_key: Optional[str] = None
    output: Any = None
    error: Optional[str] = None

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        normalized = str(value or "").strip().lower()
        alias_map = {"in_progress": "running", "completed": "success"}
        return alias_map.get(normalized, normalized or "pending")


class ExecutionTrace(BaseModel):
    """保存计划执行轨迹，后续执行器和调试视图都消费这里。"""
    model_config = ConfigDict(extra="forbid")

    step_id: str
    event: str
    status: Optional[PlanStepStatus] = None
    detail: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        if value in (None, ""):
            return None
        return StepResult._normalize_status(value)


class ConfirmationDecisionOption(BaseModel):
    """定义一次确认请求允许的稳定决策枚举。"""

    model_config = ConfigDict(extra="forbid")

    code: ConfirmationDecision
    label: Optional[str] = None
    description: Optional[str] = None


class ConfirmationDecisionPayload(BaseModel):
    """定义用户恢复执行时提交的标准化决策。

    edited_arguments 用于“确认目标论文”这类显式改参场景；后端仍只接受 pending confirmation
    中声明和校验过的字段，不能把它当作重新解析自然语言的入口。
    """

    model_config = ConfigDict(extra="forbid")

    decision: ConfirmationDecision
    note: Optional[str] = None
    edited_arguments: Optional[Dict[str, Any]] = None


class ResumeRequest(BaseModel):
    """定义前端发起 interrupt 恢复时使用的结构化请求。

    step_id / interrupt_id / tool_name / pending_action_id 共同标识本次要消费的确认任务；
    edited_arguments 只承载确认框中用户明确选择的 paper_id/arxiv_id 等字段，后端必须用
    pending confirmation 里的候选集合再次校验。
    """

    model_config = ConfigDict(extra="forbid")

    decision: ConfirmationDecision
    note: Optional[str] = None
    step_id: Optional[str] = None
    interrupt_id: Optional[str] = None
    tool_name: Optional[str] = None
    pending_action_id: Optional[str] = None
    edited_arguments: Optional[Dict[str, Any]] = None

    @field_validator("note", "step_id", "interrupt_id", "tool_name", "pending_action_id", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @model_validator(mode="after")
    def _validate_locator_fields(self) -> "ResumeRequest":
        # resume 是结构化确认动作，不能再退回“只靠 message 文本猜当前待确认任务”。
        # 至少要携带 pending_action_id 或 step_id/tool_name 这组稳定定位字段，后端才能可靠校验并发、旧 checkpoint 和重复点击。
        has_pending_action_id = bool(self.pending_action_id)
        has_step_and_tool = bool(self.step_id and self.tool_name)
        if not has_pending_action_id and not has_step_and_tool:
            raise ValueError("resume request must include pending_action_id or step_id plus tool_name")
        return self


class ConfirmationRequest(BaseModel):
    """定义一次标准化的确认请求 payload。

    这个结构会被用于 interrupt payload 和前后端交互，因此只保留可序列化、可展示的轻量字段，
    不应塞入完整 runtime/state、原始 PDF、论文 chunk 或工具返回大对象。
    """

    model_config = ConfigDict(extra="forbid")

    request_type: ConfirmationRequestType = "tool_approval"
    pending_action_id: Optional[str] = None
    step_id: str
    tool_name: str
    action_type: str
    side_effect_level: str
    reason: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    arguments_summary: Dict[str, Any] = Field(default_factory=dict)
    original_question: Optional[str] = None
    original_message: Optional[str] = None
    target_paper: Optional[Dict[str, Any]] = None
    # 目标论文确认需要把候选论文随 checkpoint 一起保存；resume 时只允许从这里匹配，不能重新解析自然语言。
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_candidate: Optional[Dict[str, Any]] = None
    default_candidate_id: Optional[str] = None
    reference_hint: Dict[str, Any] = Field(default_factory=dict)
    target_resolution: Dict[str, Any] = Field(default_factory=dict)
    confirmation_fields: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    allowed_decisions: List[ConfirmationDecisionOption] = Field(default_factory=list)
    allow_argument_edit: bool = False
    allow_reject: bool = True
    allow_note: bool = True
    trace_id: Optional[str] = None
    plan_id: Optional[str] = None
    session_id: Optional[str] = None
    thread_id: Optional[str] = None


class PlanStep(BaseModel):
    """定义执行计划中的单个步骤。

    execution_plan 在阶段 1 先不直接驱动工具执行，而是表达：
    - 当前任务预计要分几步完成；
    - 每步输入输出预期是什么；
    - 步骤之间是否存在依赖；
    - 当前规划/执行状态如何。
    """
    model_config = ConfigDict(extra="forbid")

    step_id: str
    action_type: str
    tool_name: str
    tool: "ToolSpec"
    input_bindings: List["StepInputBinding"] = Field(default_factory=list)
    output_key: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    condition: Optional["StepCondition"] = None
    preconditions: List["StepCondition"] = Field(default_factory=list)
    postconditions: List["StepCondition"] = Field(default_factory=list)
    retry_policy: Optional["StepPolicy"] = None
    failure_policy: Optional["StepPolicy"] = None
    confirmation_policy: Optional["StepPolicy"] = None
    side_effect_level: Literal["none", "session_write", "persistent_write", "external_call"] = "none"
    status: Literal["pending", "running", "success", "failed", "skipped", "waiting_confirmation"] = "pending"

    @field_validator("tool", mode="before")
    @classmethod
    def _coerce_tool_spec(cls, value: Any) -> Any:
        # 测试和轻量运行时可能用不同模块名加载同一份 schema，先转成 dict 避免类身份不一致。
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return value

    @field_validator("retry_policy", "failure_policy", "confirmation_policy", mode="before")
    @classmethod
    def _coerce_step_policy(cls, value: Any) -> Any:
        # planner/converter 可能跨模块实例传入同形 StepPolicy；统一转 dict 后再由当前 schema 校验。
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return value

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> Any:
        return StepResult._normalize_status(value)

    @property
    def step_type(self) -> str:
        """兼容旧调用方按 step_type 读取动作类型。"""
        return self.action_type


class ExecutablePlan(BaseModel):
    """承载完整计划拓扑，供规划器、执行器和调试链共享。"""
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    goal: Goal
    steps: List[PlanStep] = Field(default_factory=list)
    entry_step_ids: List[str] = Field(default_factory=list)
    final_step_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __iter__(self):
        return iter(self.steps or [])

    def __len__(self) -> int:
        return len(self.steps or [])

    def __getitem__(self, index: int) -> PlanStep:
        return list(self.steps or [])[index]


class PlanRuntime(BaseModel):
    """保存计划执行过程中的运行态，而不是把状态混进计划定义。

    这是执行器内部继续使用的可变运行容器；对 LangGraph 节点和跨请求恢复更友好的
    一等执行现场由 AgentRuntimeState 表达，避免上层只能理解 executor 内部循环变量。
    """
    model_config = ConfigDict(extra="forbid")

    state: Dict[str, Any] = Field(default_factory=dict)
    goal: Optional[Goal] = None
    plan: Optional[ExecutablePlan] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    step_status: Dict[str, PlanStepStatus] = Field(default_factory=dict)
    trace: List[ExecutionTrace] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    replan_counts: Dict[str, int] = Field(default_factory=dict)
    step_replan_counts: Dict[str, int] = Field(default_factory=dict)
    approved_step_ids: List[str] = Field(default_factory=list)
    current_step_id: Optional[str] = None
    current_step_index: Optional[int] = None
    last_observation: Optional[Dict[str, Any]] = None
    last_step_output: Optional[Dict[str, Any]] = None
    needs_replan: bool = False
    is_finished: bool = False
    recovery_strategy: Optional[Dict[str, Any]] = None
    pending_confirmation: Optional[ConfirmationRequest] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    turn_status: Optional[AgentTurnStatus] = None


class AgentRuntimeState(BaseModel):
    """统一表达一次 Agent 计划执行现场。

    这个结构是后续拆 LangGraph 节点时跨节点传递的稳定契约，字段只保留可序列化数据：
    计划、当前 step、step 状态、工具输出、最近 observation、确认态、重规划计数和失败恢复建议。
    它不持有函数、工具实例或闭包，避免 checkpoint 恢复时依赖内存对象。
    """
    model_config = ConfigDict(extra="forbid")

    request_state: Dict[str, Any] = Field(default_factory=dict)
    goal: Optional[Goal] = None
    plan: Optional[ExecutablePlan] = None
    current_step_id: Optional[str] = None
    current_step_index: Optional[int] = None
    step_status: Dict[str, PlanStepStatus] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    last_observation: Optional[Dict[str, Any]] = None
    last_step_output: Optional[Dict[str, Any]] = None
    trace: List[ExecutionTrace] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    replan_counts: Dict[str, int] = Field(default_factory=dict)
    step_replan_counts: Dict[str, int] = Field(default_factory=dict)
    approved_step_ids: List[str] = Field(default_factory=list)
    pending_confirmation: Optional[ConfirmationRequest] = None
    needs_replan: bool = False
    is_finished: bool = False
    failure_reason: Optional[str] = None
    recovery_strategy: Optional[Dict[str, Any]] = None
    turn_status: Optional[AgentTurnStatus] = None
    final_answer: Optional[str] = None

    @field_validator("plan", "goal", "pending_confirmation", mode="before")
    @classmethod
    def _coerce_nested_models(cls, value: Any) -> Any:
        # 不同测试加载路径可能产生同形不同类的 Pydantic 对象，统一转 dict 再按当前 schema 校验。
        model_dump = getattr(value, "model_dump", None)
        return model_dump() if callable(model_dump) else value

    @field_validator("trace", mode="before")
    @classmethod
    def _coerce_trace_models(cls, value: Any) -> Any:
        if isinstance(value, list):
            normalized = []
            for item in value:
                model_dump = getattr(item, "model_dump", None)
                normalized.append(model_dump() if callable(model_dump) else item)
            return normalized
        return value


class StepExecutionResult(BaseModel):
    """单步执行器返回给上层节点的结构化结果。

    Executor 的新边界只负责“执行一个待执行 step”，因此必须把本步状态、输出、错误、
    observation、下一步动作和 runtime_patch 一次性说清楚，不能再让调用方去解析内部 trace。
    """
    model_config = ConfigDict(extra="forbid")

    step_id: Optional[str] = None
    step_status: Optional[PlanStepStatus] = None
    output_key: Optional[str] = None
    output: Any = None
    error: Optional[str] = None
    observation: Optional[Dict[str, Any]] = None
    next_action: Literal[
        "continue",
        "wait_for_confirmation",
        "replan",
        "finish",
        "fail",
        "noop",
    ] = "continue"
    pending_confirmation: Optional[ConfirmationRequest] = None
    runtime_patch: Dict[str, Any] = Field(default_factory=dict)
    turn_result: Optional[Any] = None

    @field_validator("pending_confirmation", mode="before")
    @classmethod
    def _coerce_result_models(cls, value: Any) -> Any:
        # 确认请求需要进入 checkpoint；单轮结果保留模型实例，避免兼容执行入口丢失属性访问语义。
        model_dump = getattr(value, "model_dump", None)
        return model_dump() if callable(model_dump) else value


class AgentTurnResult(BaseModel):
    """统一承载单轮计划执行结果，避免执行器把结果散落在多个临时结构中。"""
    model_config = ConfigDict(extra="forbid")

    status: AgentTurnStatus
    final_answer: Optional[str] = None
    plan: Optional[ExecutablePlan] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    trace: List[ExecutionTrace] = Field(default_factory=list)
    pending_confirmation: Optional[ConfirmationRequest] = None
    error: Optional[str] = None
    runtime: Optional[PlanRuntime] = None

    @field_validator("plan", "runtime", "pending_confirmation", mode="before")
    @classmethod
    def _coerce_runtime_models(cls, value: Any) -> Any:
        # 混合测试会重复导入 schema；这里仅把同形 Pydantic 对象转回原始 dict 重新校验。
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return value

    @field_validator("trace", mode="before")
    @classmethod
    def _coerce_trace_models(cls, value: Any) -> Any:
        # replan 模块可能来自另一份 schema 导入；trace 逐项转 dict 后再按当前模型校验。
        if isinstance(value, list):
            normalized = []
            for item in value:
                model_dump = getattr(item, "model_dump", None)
                normalized.append(model_dump() if callable(model_dump) else item)
            return normalized
        return value


FailureCategory = Literal[
    "search_empty",
    "search_low_confidence",
    "search_too_broad",
    "search_too_narrow",
    "paper_index_missing",
    "paper_index_stale",
    "paper_index_corrupted",
    "qa_no_answer",
    "qa_no_sources",
    "qa_low_grounding",
    "preference_target_missing",
    "preference_write_failed",
    "tool_timeout",
    "tool_invalid_output",
    "tool_runtime_error",
    "insufficient_context",
    "ambiguous_user_request",
    "empty_user_profile",
]

RecoveryActionType = Literal[
    "patch_plan",
    "retry_step",
    "ask_clarification",
    "request_confirmation",
    "skip_step",
    "fallback_answer",
    "abort_with_error",
]

RecoverySeverity = Literal["info", "warning", "error", "critical"]
RecoveryRiskLevel = Literal["low", "medium", "high"]
ObservationSignal = Literal[
    "success_with_sufficient_result",
    "success_but_empty_result",
    "success_but_low_quality",
    "missing_required_context",
    "target_not_resolved",
    "index_not_found",
    "confirmation_required",
    "user_rejected",
    "external_tool_failed",
    "validation_failed",
    "persistent_write_succeeded",
    "persistent_write_uncertain",
    "unrecoverable_error",
]
RecoveryActionSemantic = Literal[
    "retry_same_step",
    "patch_current_step_inputs",
    "insert_step_before_current",
    "append_step_after_current",
    "replace_remaining_plan",
    "ask_clarification",
    "request_confirmation",
    "fallback_answer",
    "terminate_success",
    "terminate_failed",
    "skip_step",
]


class RecoveryCandidate(BaseModel):
    """描述一个可选恢复动作，供后续 chooser/patcher 使用，本身不直接改写执行计划。"""
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    action_type: RecoveryActionType
    action_semantic: Optional[RecoveryActionSemantic] = None
    failure_category: FailureCategory
    priority: int = 100
    confidence: float = 1.0
    reason: str
    target_step_id: str
    required_tools: List[str] = Field(default_factory=list)
    policy_source: Optional[str] = None
    tool_recovery_policy: Dict[str, Any] = Field(default_factory=dict)
    patch_strategy: Optional[str] = None
    strategy_payload: Dict[str, Any] = Field(default_factory=dict)
    risk_level: RecoveryRiskLevel = "low"
    requires_confirmation: bool = False
    max_attempts: Optional[int] = None
    expected_effect: Optional[str] = None
    fallback_if_failed: Optional[str] = None


class RecoveryAction(BaseModel):
    """RecoveryChooser 选出的最终恢复动作，是 Replanner 与 PlanPatcher 之间的稳定接口。"""
    model_config = ConfigDict(extra="forbid")

    action_type: RecoveryActionType
    action_semantic: Optional[RecoveryActionSemantic] = None
    target_step_id: str
    selected_candidate_id: Optional[str] = None
    patch_strategy: Optional[str] = None
    patch_payload: Dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    user_message: Optional[str] = None
    fallback_reason: Optional[str] = None
    debug_reason: Optional[str] = None


class LLMRecoveryDiagnosis(BaseModel):
    """LLM 仅提供恢复诊断和候选排序建议，不能创建动作或直接修改计划。"""
    model_config = ConfigDict(extra="forbid")

    diagnosis: Optional[str] = None
    recommended_recovery_type: Optional[RecoveryActionType] = None
    ranked_candidate_ids: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    user_facing_reason: Optional[str] = None
    confidence: float = 0.0
    ignored_candidate_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class RecoverySafetyCheckResult(BaseModel):
    """SafetyGuard 的结构化结果，确保 LLM 和 policy 都不能绕过副作用边界。"""
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reasons: List[str] = Field(default_factory=list)
    fallback_action: Optional[RecoveryAction] = None
    checked_action: Dict[str, Any] = Field(default_factory=dict)


class ObservationResult(BaseModel):
    """统一承载 Observer 对单步结果质量的判断，避免把质量语义混进工具异常分支。"""
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "success",
        "partial_success",
        "empty_result",
        "low_confidence",
        "invalid_output",
        "tool_error",
        "need_confirmation",
        "need_clarification",
    ]
    observation_signal: Optional[ObservationSignal] = None
    reason: Optional[str] = None
    confidence: float = 1.0
    details: Dict[str, Any] = Field(default_factory=dict)
    suggested_action: Optional[str] = None
    failure_category: Optional[FailureCategory] = None
    severity: Optional[RecoverySeverity] = None
    recoverable: Optional[bool] = None
    suggested_recovery_types: List[RecoveryActionType] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    retryable: Optional[bool] = None
    requires_user_input: Optional[bool] = None


class AgentStreamEvent(BaseModel):
    """定义流式输出模式下的单条事件结构。

    当 Agent 需要边执行边向前端推送状态时，就会使用这类事件。
    通过 event_type + sequence + data，可以让前端按顺序还原：
    - run 开始；
    - step 开始/结束；
    - tool 调用开始/结束；
    - 最终回复；
    - 异常与流结束。
    """
    model_config = ConfigDict(extra="forbid")

    event_type: Literal[
        "run_start",
        "step_start",
        "step_end",
        "tool_call_start",
        "tool_call_end",
        "final_response",
        "exception",
        "stream_end",
    ]
    sequence: int
    run_id: str
    timestamp: str
    data: Dict[str, Any] = Field(default_factory=dict)


class ArxivSearchResponse(BaseModel):
    """定义 arXiv Agent 单次执行后的完整响应结构。

    这是面向调用方/前端的“最终聚合响应”，基本上是 `AgentState` 中用户和调试方关心的
    那部分字段的外显版本。它会把：
    - 意图识别结果；
    - 搜索规格；
    - 论文结果；
    - 工具调用轨迹；
    - 最终 answer 与 next_actions；
    - 以及 pending_action / paper_qa_result / preference_action_result
    一并返回给上层。
    """
    model_config = ConfigDict(extra="forbid")

    session_id: Optional[str] = None
    intent: Literal[
        "arxiv_search",
        "paper_detail",
        "paper_summary",
        "paper_qa",
        "recommendation",
        "preference_action",
        "unclear",
        "unsupported",
    ]
    intent_source: Optional[str] = None
    fallback_reason: Optional[str] = None
    llm_confidence: Optional[float] = None
    answer: str
    search_spec: Optional[ArxivSearchSpec] = None
    goal: Optional[Goal] = None
    # research_task_profile 与 intent 并行存在，描述本轮科研任务语义（任务类型/对象/中间产物/证据需求）；
    # 它是可选语义层，缺失时不影响其余响应字段。
    research_task_profile: Optional[ResearchTaskProfile] = None
    execution_plan: Optional[ExecutablePlan] = None
    plan_runtime: Optional[PlanRuntime] = None
    runtime_state: Optional[AgentRuntimeState] = None
    pending_action: Optional[Dict[str, Any]] = None
    paper_qa_result: Optional[Dict[str, Any]] = None
    preference_action_result: Optional[Dict[str, Any]] = None
    plan: List[str] = Field(default_factory=list)
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    papers: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


class ArxivSearchGraphResponse(BaseModel):
    """定义图结构导出接口的响应模型。

    主要用于调试页或前端可视化直接渲染当前 Agent 的 LangGraph 拓扑结构。
    """
    model_config = ConfigDict(extra="forbid")

    graph_name: str
    render_source: str
    node_names: List[str] = Field(default_factory=list)
    mermaid: str
    supports_png: bool = False
