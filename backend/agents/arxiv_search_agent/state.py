from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .schemas import (
    AgentStep,
    AgentToolCall,
    ArxivSearchSpec,
    ExecutionPlanStep,
    ExecutablePlan,
    Goal,
    PlanStep,
    PlanRuntime,
    ToolCallRequest,
    ToolObservation,
    ToolSpec,
)


_LEGACY_STEP_TOOL_NAMES = {
    "goal_interpretation": "normalize_request",
    "search_execution": "search_arxiv",
    "result_validation": "validate_arxiv_results",
    "personalization": "personalize_paper_results",
    "response_synthesis": "synthesize_arxiv_response",
    "paper_resolution": "resolve_paper",
    "qa_index_check": "check_paper_index",
    "confirmation_gate": "request_confirmation",
    "paper_response": "answer_paper_question",
    "preference_update": "update_preference_store",
    "profile_loading": "load_user_profile",
    "recommendation_generation": "generate_recommendations",
    "recommendation_explanation": "explain_recommendations",
    "ambiguity_analysis": "analyze_ambiguity",
    "clarification_response": "generate_clarification",
    "capability_check": "generate_fallback_response",
    "fallback_response": "generate_fallback_response",
}


def _coerce_legacy_execution_plan(value: Any) -> Any:
    """把历史 ExecutionPlanStep 列表规范化成当前 ExecutablePlan。

    项目里仍有少量兼容节点和旧测试会传入轻量 step 列表；这里在状态边界统一转换，
    避免业务节点继续同时处理两套计划容器。
    """
    if not isinstance(value, list):
        return value
    steps: List[PlanStep] = []
    for index, item in enumerate(value):
        payload = item.model_dump() if hasattr(item, "model_dump") else dict(item or {})
        if "action_type" in payload and "tool_name" in payload and "tool" in payload:
            steps.append(PlanStep.model_validate(payload))
            continue
        legacy_step = ExecutionPlanStep.model_validate(payload)
        action_type = str(legacy_step.step_type or "").strip()
        tool_name = _LEGACY_STEP_TOOL_NAMES.get(action_type, action_type or f"legacy_step_{index}")
        steps.append(
            PlanStep(
                step_id=legacy_step.step_id,
                action_type=action_type,
                tool_name=tool_name,
                tool=ToolSpec(tool_name=tool_name),
                depends_on=list(legacy_step.depends_on or []),
                status=legacy_step.status,
            )
        )
    depended_ids = {dependency for step in steps for dependency in list(step.depends_on or [])}
    return ExecutablePlan(
        plan_id="legacy_execution_plan",
        goal=Goal(goal_type="legacy_execution_plan"),
        steps=steps,
        entry_step_ids=[step.step_id for step in steps if not step.depends_on],
        final_step_ids=[step.step_id for step in steps if step.step_id not in depended_ids],
        metadata={"source": "legacy_execution_plan"},
    )


class AgentState(BaseModel):
    """定义 arXiv Agent 在 LangGraph 中流转的统一状态对象。

    可以把这个模型理解为整张工作流的“共享上下文容器”。
    graph.py 里的每个节点函数都会读取它的一部分字段，并在处理完成后返回一个新的
    `AgentState` 副本。这样无论当前走的是：
    - 搜索链路，
    - 论文阅读链路，
    - 待确认任务链路，
    - 偏好更新链路，
    最终都能通过同一个状态模型进行衔接。

    字段大致可以分成几组：
    1. 请求上下文：user_id、session_id、message、context；
    2. 意图识别结果：intent、intent_source、fallback_reason、llm_confidence、search_spec；
    3. 工具与数据结果：tool_name、tool_args、tool_result、tool_calls、papers；
    4. 阅读/偏好相关中间态：待确认任务（pending_action）、paper_qa_result、preference_action_result；
    5. 面向用户与调试的信息：plan、warnings、answer、next_actions、steps、errors、debug；
    6. 运行时控制字段：personalized_rerank_applied、search_retry_count、fallback_specs。

    之所以把这些状态集中到一个模型里，而不是拆成多个零散 dict，主要是为了：
    - 保持节点之间的数据契约稳定；
    - 让调试和日志更容易理解；
    - 利用 Pydantic 的默认值和类型约束，减少状态缺字段的问题。
    """
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: Optional[str] = None
    # context 用来承接跨轮对话信息、前端回传信息和上游服务补充信息，
    # 例如 selected_paper、last_papers、research_profile、loading_method 等。
    context: Dict[str, Any] = Field(default_factory=dict)

    # pending_action 是对外兼容字段：新确认链路里它主要镜像 ConfirmationRequest 供前端展示。
    # 真正的 interrupt/resume 执行现场依赖 LangGraph checkpointer，不能只靠这个业务摘要恢复。
    pending_action: Optional[Dict[str, Any]] = None

    # paper_qa_result 保存论文阅读链路的结构化结果：可能是成功答案、失败信息，
    # 也可能是 waiting_confirmation 状态。
    paper_qa_result: Optional[Dict[str, Any]] = None

    # intent 系列字段由 parse_search_request 节点产出，决定图中后续路由方向。
    intent: Optional[str] = None
    intent_source: Optional[str] = None
    fallback_reason: Optional[str] = None
    llm_confidence: Optional[float] = None
    search_spec: Optional[ArxivSearchSpec] = None

    # goal 表达用户本轮真实想完成的目标；execution_plan 表达结构化步骤规划。
    # 阶段 1 中它们主要用于状态表达、调试和后续能力扩展，不直接替代现有 plan。
    goal: Optional[Goal] = None
    execution_plan: Optional[ExecutablePlan] = None
    plan_runtime: Optional[PlanRuntime] = None

    @field_validator("execution_plan", mode="before")
    @classmethod
    def _normalize_execution_plan(cls, value: Any) -> Any:
        return _coerce_legacy_execution_plan(value)

    # 偏好动作执行后的结果，例如喜欢/不喜欢/取消标记的处理结果。
    preference_action_result: Optional[Dict[str, Any]] = None

    # debug 保存面向开发排查的中间态与详细错误上下文，不直接面向终端用户。
    debug: Dict[str, Any] = Field(default_factory=dict)

    # plan 表示系统理解到的处理计划；next_actions 表示建议用户下一步可以做什么。
    plan: List[str] = Field(default_factory=list)

    # tool_* 字段记录当前节点最近一次工具调用的名称、参数和原始结果；
    # tool_calls 则保留完整工具调用轨迹列表。
    # 阶段 2 新增的 tool_call_request / tool_observations 先只作为统一协议字段，
    # 暂不替代已有工具执行流，避免影响现有搜索链路和调试展示逻辑。
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    tool_call_request: Optional[ToolCallRequest] = None
    tool_observations: List[ToolObservation] = Field(default_factory=list)

    # papers 用于保存搜索结果或某些阅读链路回传的论文列表。
    papers: List[Dict[str, Any]] = Field(default_factory=list)

    # warnings 只放用户或前端可理解的提示；answer / next_actions 是最终回复层直接消费的字段。
    warnings: List[str] = Field(default_factory=list)
    answer: Optional[str] = None
    next_actions: List[str] = Field(default_factory=list)

    # steps 记录节点级执行轨迹；errors 存放结构化错误信息，供后端或调试工具排查。
    steps: List[AgentStep] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # 个性化排序是否生效，以及搜索是否进入过自动放宽关键词重试。
    personalized_rerank_applied: bool = False
    search_retry_count: int = 0

    # fallback_specs 记录每轮放宽关键词后的搜索规格历史，便于排查召回策略。
    fallback_specs: List[Dict[str, Any]] = Field(default_factory=list)
