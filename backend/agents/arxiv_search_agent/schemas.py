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
    input_bindings: List["StepInputBinding"] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    expected_output_key: Optional[str] = None
    retry_policy: Optional["StepPolicy"] = None
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_confirmation: bool = False
    fallback_reason: Optional[str] = None


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


class ToolCandidate(BaseModel):
    """候选工具筛选结果中的单个工具说明。"""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    capability_tags: List[str] = Field(default_factory=list)
    side_effect_level: str = "none"
    requires_confirmation: bool = False
    selection_reason: Optional[str] = None


class ExcludedToolCandidate(BaseModel):
    """记录被排除的工具及原因，便于 planner debug 判断候选边界是否过宽。"""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    capability_tags: List[str] = Field(default_factory=list)
    side_effect_level: str = "none"
    requires_confirmation: bool = False
    exclusion_reason: Optional[str] = None


class ToolCandidateSelection(BaseModel):
    """Tool Candidate Selector 的结构化输出。"""
    model_config = ConfigDict(extra="forbid")

    goal_type: Optional[str] = None
    candidate_tools: List[ToolCandidate] = Field(default_factory=list)
    excluded_tools: List[ExcludedToolCandidate] = Field(default_factory=list)
    selection_reason: Optional[str] = None
    risk_summary: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlanStep(BaseModel):
    """兼容旧版节点流使用的轻量计划步骤。

    新执行器使用 PlanStep/ExecutablePlan 承载工具契约；旧 LangGraph 节点和历史测试仍会
    构造只包含 step_type/description 的步骤，因此这里保留窄模型避免破坏旧入口。
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
ConfirmationRequestType = Literal["tool_approval"]
ConfirmationDecision = Literal["approve", "reject"]


class ToolSpec(BaseModel):
    """定义计划步骤要调用的真实工具及其静态约束。"""
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    capability_tags: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    side_effect_level: Literal["none", "session_write", "persistent_write", "external_call"] = "none"
    requires_confirmation: bool = False
    can_retry: bool = False
    failure_modes: List[str] = Field(default_factory=list)
    implementation: Optional[str] = None


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

    第一版只支持 approve / reject。
    edited_arguments 仅作为未来扩展预留字段，当前版本不启用参数修改能力。
    """

    model_config = ConfigDict(extra="forbid")

    decision: ConfirmationDecision
    note: Optional[str] = None
    edited_arguments: Optional[Dict[str, Any]] = None


class ResumeRequest(BaseModel):
    """定义前端发起 interrupt 恢复时使用的结构化请求。

    第一版只把 approve / reject 做成稳定契约。
    step_id / interrupt_id 先作为幂等校验与后续扩展预留字段，不在当前版本里驱动参数编辑。
    """

    model_config = ConfigDict(extra="forbid")

    decision: ConfirmationDecision
    note: Optional[str] = None
    step_id: Optional[str] = None
    interrupt_id: Optional[str] = None
    edited_arguments: Optional[Dict[str, Any]] = None


class ConfirmationRequest(BaseModel):
    """定义一次标准化的确认请求 payload。

    这个结构会被用于 interrupt payload 和前后端交互，因此只保留可序列化、可展示的轻量字段，
    不应塞入完整 runtime/state、原始 PDF、论文 chunk 或工具返回大对象。
    """

    model_config = ConfigDict(extra="forbid")

    request_type: ConfirmationRequestType = "tool_approval"
    step_id: str
    tool_name: str
    action_type: str
    side_effect_level: str
    reason: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    arguments_summary: Dict[str, Any] = Field(default_factory=dict)
    original_question: Optional[str] = None
    target_paper: Optional[Dict[str, Any]] = None
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
    """保存计划执行过程中的运行态，而不是把状态混进计划定义。"""
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
    pending_confirmation: Optional[ConfirmationRequest] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
    turn_status: Optional[AgentTurnStatus] = None


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


class RecoveryCandidate(BaseModel):
    """描述一个可选恢复动作，供后续 chooser/patcher 使用，本身不直接改写执行计划。"""
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    action_type: RecoveryActionType
    failure_category: FailureCategory
    priority: int = 100
    confidence: float = 1.0
    reason: str
    target_step_id: str
    required_tools: List[str] = Field(default_factory=list)
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
    execution_plan: Optional[ExecutablePlan] = None
    plan_runtime: Optional[PlanRuntime] = None
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
