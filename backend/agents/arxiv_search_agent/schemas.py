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

    @field_validator("user_id", "session_id", "message", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        # 在模型入参阶段统一做基础文本清洗，避免后续节点反复 strip。
        if value is None:
            return None
        text = str(value).strip()
        return text

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
        "reading_list_action",
        "unclear",
        "unsupported",
    ]
    intent_source: Optional[str] = None
    fallback_reason: Optional[str] = None
    llm_confidence: Optional[float] = None
    answer: str
    search_spec: Optional[ArxivSearchSpec] = None
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
