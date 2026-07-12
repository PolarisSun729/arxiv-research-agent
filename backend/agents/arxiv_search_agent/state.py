from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .schemas import (
    AgentStep,
    AgentToolCall,
    ArxivSearchSpec,
    AgentRuntimeState,
    ExecutionPlanStep,
    ExecutablePlan,
    Goal,
    PlanStep,
    PlanRuntime,
    ResearchTaskProfile,
    ToolCallRequest,
    ToolObservation,
    ToolSpec,
)
from .execution.interactions import AgentInteraction


_LEGACY_STEP_TOOL_NAMES = {
    "goal_interpretation": "normalize_request",
    "search_execution": "search_arxiv",
    "result_validation": "validate_arxiv_results",
    "personalization": "personalize_paper_results",
    "response_synthesis": "synthesize_arxiv_response",
    "paper_resolution": "resolve_paper",
    "qa_index_check": "check_paper_index",
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

    ====================================================================
    字段分层（务必先读这一段，再读下面各字段）：本模型刻意区分三类字段，
    维护者改动前必须分清自己在动哪一类，避免再次出现“同一件事多处副本”。

    A. 执行真源（唯一可写的业务状态，所有判断都应只读这里）：
       - goal / execution_plan：本轮目标与结构化计划；
       - plan_runtime（PlanRuntime）：运行中执行现场，承载当前 step、工具输出
         （runtime.outputs）、最近 observation、interaction、final_answer
         与失败原因。用户是否等待确认、工具结果、最终回答都以它为准。
       - runtime_state（AgentRuntimeState）：plan_runtime 的可序列化 checkpoint
         投影，专供 resume 恢复；它是 plan_runtime 的快照镜像，业务逻辑不得
         把它当成与 plan_runtime 并列的第二份真源同时读写。

    B. 出站展示 / 响应投影（出站时从执行真源生成，输入侧不可信、不得反向驱动执行）：
       - interaction：等待用户输入时的唯一结构化业务交互；
       - answer / papers / paper_qa_result / preference_action_result：最终响应
         适配字段，由 graph._apply_turn_result 从 runtime.outputs 投影得到。

    C. 旧式平行节点的兼容字段（不在 graph 主路径上，仅历史 node/* 与其测试使用）：
       - tool_name / tool_args / tool_result：旧 node 路径单步工具中间态，
         主路径（graph → PlanExecutor → tool_adapters）不读写它们；
       - tool_calls：工具调用轨迹列表，仅供展示与调试。
       新协议字段 tool_call_request / tool_observations 属于工具协议层，
       同样不替代主路径的 runtime.outputs。

    其余请求上下文（user_id/session_id/message/context）、意图识别结果
    （intent/search_spec 等）、调试信息（debug/steps/errors/warnings）和运行时
    控制位（personalized_rerank_applied/search_retry_count/fallback_specs）按字段
    注释理解即可。

    之所以把这些状态集中到一个模型里，而不是拆成多个零散 dict，主要是为了：
    - 保持节点之间的数据契约稳定；
    - 让调试和日志更容易理解；
    - 利用 Pydantic 的默认值和类型约束，减少状态缺字段的问题。
    ====================================================================
    """
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: Optional[str] = None
    # context 用来承接跨轮对话信息、前端回传信息和上游服务补充信息，
    # 例如 selected_paper、last_papers、research_profile、loading_method 等。
    context: Dict[str, Any] = Field(default_factory=dict)

    interaction: Optional[AgentInteraction] = None

    # [B 出站响应投影] paper_qa_result 由 _apply_turn_result 从 runtime.outputs 投影得到，
    # 可能是成功答案或失败信息；等待用户输入统一由 interaction 表达。
    paper_qa_result: Optional[Dict[str, Any]] = None

    # intent 系列字段由 parse_search_request 节点产出，决定图中后续路由方向。
    intent: Optional[str] = None
    intent_source: Optional[str] = None
    fallback_reason: Optional[str] = None
    llm_confidence: Optional[float] = None
    search_spec: Optional[ArxivSearchSpec] = None

    # [A 执行真源] goal 表达用户本轮真实想完成的目标；execution_plan 表达结构化步骤规划。
    # plan_runtime（PlanRuntime）是运行中执行现场，承载当前 step、工具输出、observation
    # 与 interaction，所有业务判断只读它。
    # runtime_state（AgentRuntimeState）是 plan_runtime 的可序列化 checkpoint 投影，
    # 仅供 resume 恢复，不得与 plan_runtime 并列当成第二份真源同时读写。
    goal: Optional[Goal] = None
    execution_plan: Optional[ExecutablePlan] = None
    plan_runtime: Optional[PlanRuntime] = None
    runtime_state: Optional[AgentRuntimeState] = None

    # [A 执行真源-语义层] research_task_profile 表达本轮“科研任务语义”，与 intent 并行存在：
    # intent 说明系统进入哪条能力链路，research_task_profile.research_task_type 说明用户处于
    # 哪一类科研任务（方向探索 / 多论文比较 / 单篇深读 / 阅读规划 / 研究空白分析 / 个性化推荐）。
    # 它由 build_goal 阶段在 Goal 之后、ExecutablePlan 之前推断；缺失时不影响既有 intent/计划流程，
    # 后续 planner 可选择消费其中的中间产物（intermediate_artifacts）和证据需求（evidence_requirements）。
    research_task_profile: Optional[ResearchTaskProfile] = None

    @field_validator("execution_plan", mode="before")
    @classmethod
    def _normalize_execution_plan(cls, value: Any) -> Any:
        return _coerce_legacy_execution_plan(value)

    # [B 出站投影] 偏好动作执行后的结果，例如喜欢/不喜欢/取消标记的处理结果；
    # 由 _apply_turn_result 从 runtime.outputs 投影，不作为执行过程真源。
    preference_action_result: Optional[Dict[str, Any]] = None

    # debug 保存面向开发排查的中间态与详细错误上下文，不直接面向终端用户。
    debug: Dict[str, Any] = Field(default_factory=dict)

    # plan 表示系统理解到的处理计划；next_actions 表示建议用户下一步可以做什么。
    plan: List[str] = Field(default_factory=list)

    # [C 兼容字段] tool_name / tool_args / tool_result 是旧式平行 node 路径的单步工具
    # 中间态：当前节点最近一次工具调用的名称、参数和原始结果；tool_calls 保留调用轨迹。
    # graph 主路径（PlanExecutor + tool_adapters）不读写它们，工具结果以 runtime.outputs 为真源。
    # tool_call_request / tool_observations 属于工具协议层，同样不替代主路径的 runtime.outputs。
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    tool_call_request: Optional[ToolCallRequest] = None
    tool_observations: List[ToolObservation] = Field(default_factory=list)

    # [B 出站投影] papers 保存搜索结果或阅读链路回传的论文列表，
    # 由 _apply_turn_result 从 runtime.outputs（ranked_papers/arxiv_results 等）投影。
    papers: List[Dict[str, Any]] = Field(default_factory=list)

    # [B 出站投影] warnings 只放用户或前端可理解的提示；answer / next_actions 是最终回复层
    # 直接消费的字段，answer 由 runtime.final_answer / outputs 投影，不作为执行过程真源。
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
