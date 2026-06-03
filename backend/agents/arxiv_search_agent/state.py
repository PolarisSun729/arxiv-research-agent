from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schemas import AgentStep, AgentToolCall, ArxivSearchSpec


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
    5. 面向用户与调试的信息：plan、warnings、answer、next_actions、steps、debug；
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

    # pending_action 用来表示“还不能立刻执行，需要先征求用户确认”的待确认任务，
    # 典型场景是论文尚未建立 QA 索引，需要先问用户是否解析 PDF。
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

    # 偏好动作执行后的结果，例如喜欢/不喜欢/取消标记的处理结果。
    preference_action_result: Optional[Dict[str, Any]] = None

    # debug 保存面向开发排查的中间态，不直接面向终端用户。
    debug: Dict[str, Any] = Field(default_factory=dict)

    # plan 表示系统理解到的处理计划；next_actions 表示建议用户下一步可以做什么。
    plan: List[str] = Field(default_factory=list)

    # tool_* 字段记录当前节点最近一次工具调用的名称、参数和原始结果；
    # tool_calls 则保留完整工具调用轨迹列表。
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None
    tool_calls: List[AgentToolCall] = Field(default_factory=list)

    # papers 用于保存搜索结果或某些阅读链路回传的论文列表。
    papers: List[Dict[str, Any]] = Field(default_factory=list)

    # warnings / answer / next_actions 是最终给回复节点使用的重要字段。
    warnings: List[str] = Field(default_factory=list)
    answer: Optional[str] = None
    next_actions: List[str] = Field(default_factory=list)

    # steps 会把每个节点的执行摘要以结构化形式记录下来，便于回放与调试。
    steps: List[AgentStep] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)

    # 个性化排序是否生效，以及搜索是否进入过自动放宽关键词重试。
    personalized_rerank_applied: bool = False
    search_retry_count: int = 0

    # fallback_specs 记录每轮放宽关键词后的搜索规格历史，便于排查召回策略。
    fallback_specs: List[Dict[str, Any]] = Field(default_factory=list)
