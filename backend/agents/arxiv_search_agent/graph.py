from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from langgraph.graph import END, START, StateGraph

from .node import (
    _coerce_state,
    apply_preference_action,
    build_search_tool_args,
    check_search_result,
    classify_pending_action_confirmation,
    handle_paper_reading_request,
    handle_pending_action_confirmation,
    invoke_search_tool,
    parse_search_request,
    personalized_rank_and_annotate_papers,
    relax_search_for_retry,
    synthesize_response,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# 这个文件负责定义 arXiv Agent 的 LangGraph 主流程结构。
#
# 可以把它理解为“流程编排层”：
# - node/ 下的模块定义每个节点具体做什么；
# - state.py 定义节点之间共享什么状态；
# - graph.py 决定这些节点按什么顺序、在什么条件下被执行。
#
# 因此这里最重要的不是业务细节本身，而是：
# 1. 整张图有哪些节点；
# 2. 每个分叉点如何路由；
# 3. 搜索链路、论文阅读链路、偏好更新链路、待确认任务链路如何汇合。

# 图导出接口会把这些节点名暴露给调试页/前端，便于渲染流程图、展示节点列表。
# 这里维护的是“图中会实际注册的节点名清单”，主要用于：
# 1. 前端直接展示当前工作流包含哪些节点；
# 2. 调试时快速比对 Mermaid 图和运行时图是否一致；
# 3. 避免节点名散落在多个位置，后续增删节点时更容易维护。
_ARXIV_GRAPH_NODE_NAMES = (
    "parse_search_request",
    "build_search_tool_args",
    "invoke_search_tool",
    "check_search_result",
    "relax_search_for_retry",
    "personalized_rank_and_annotate_papers",
    "apply_preference_action",
    "classify_pending_action_confirmation",
    "handle_pending_action_confirmation",
    "handle_paper_reading_request",
    "synthesize_response",
)


def route_after_parse(state: Any) -> str:
    """根据解析后的状态决定 parse 节点之后应该走向哪条分支。

    这个函数是整张 LangGraph 图里最核心的“第一次分流路由器”。
    `parse_search_request` 节点负责把用户输入解析成结构化状态，例如：
    - 用户当前的意图是什么（搜索、问答、推荐、偏好设置等）；
    - 当前是否还挂着上一轮尚未确认的待确认任务（pending_action）；
    - 是否存在等待用户确认的论文问答任务。

    本函数不负责修改业务数据，而是只根据当前 state 做路由判断，返回下一跳节点名。
    判断优先级大致如下：
    1. 先看是否存在“等待确认”的历史待确认任务；
    2. 再看是否存在“等待确认”的论文问答结果；
    3. 最后才根据 parse 阶段识别出的 intent 进入对应处理链路；
    4. 若 intent 无法识别，则兜底走 unsupported。

    之所以把“待确认任务状态”的判断放在意图判断之前，是为了避免用户回复“是/确认/继续”
    这类短消息时，被错误地重新当成一个全新的查询请求处理。
    """
    current_state = _coerce_state(state)

    # 第 1 步：优先恢复等待确认的待确认任务（pending_action）。
    #
    # 在某些链路里，待确认任务会暂存在 state.pending_action；但在流式请求、前端回填、
    # 状态序列化再反序列化之后，这个字段也可能只保存在 context.pending_action 中。
    # 因此这里要同时检查两个位置，尽量把“上一轮做到一半的任务”恢复出来。
    # 待确认的论文解析任务可能同时挂在 state.pending_action 和 context.pending_action 上，
    # 这里两处都检查，避免经过序列化/回填后丢掉待执行状态，导致确认消息又被当成新请求处理。
    pending_action = current_state.pending_action
    if pending_action is None and isinstance(current_state.context, dict):
        pending_action = current_state.context.get("pending_action")

    # 第 2 步：检查是否有“等待确认”的论文问答结果。
    #
    # paper_qa_result 与 pending_action 类似，也可能因为前端流式回传而被放到 context 中。
    # 这里把“等待确认”的 QA 结果视为同一种待确认任务状态：只要还在等待用户确认，
    # 当前消息就应优先进入确认分支，而不是重新发起新的论文处理流程。
    pending_qa_result = current_state.paper_qa_result
    if pending_qa_result is None and isinstance(current_state.context, dict):
        pending_qa_result = current_state.context.get("paper_qa_result")

    # 第 3 步：如果存在“先解析论文再问答”的待确认任务，并且它仍处于等待确认状态，
    # 则直接进入确认分类节点。这里不再看 intent，因为此时最重要的是续接未完成流程。
    if isinstance(pending_action, dict) and str(pending_action.get("type") or "").strip() == "parse_then_qa" and str(
        pending_action.get("status") or ""
    ).strip() == "waiting_confirmation":
        return "classify_pending_action_confirmation"

    # 第 4 步：如果没有 pending_action，但有等待确认的 paper_qa_result，
    # 同样进入确认分类节点。日志里额外记录 intent/pending 状态，方便排查路由问题。
    if isinstance(pending_qa_result, dict) and str(pending_qa_result.get("status") or "").strip() == "waiting_confirmation":
        logger.debug(
            "arxiv_agent route_after_parse -> classify_pending_action_confirmation: intent=%s pending_action_status=%s paper_qa_status=%s",
            current_state.intent or "none",
            str((pending_action or {}).get("status") or "none") if isinstance(pending_action, dict) else "none",
            str((pending_qa_result or {}).get("status") or "none"),
        )
        return "classify_pending_action_confirmation"

    # 第 5 步：如果不存在待确认任务，则按 parse 节点给出的 intent 做正常分流。
    # 这里列出的 intent 基本覆盖了当前 arXiv Agent 已支持的主要能力。
    intent = str(current_state.intent or "").strip()
    if intent in {
        "arxiv_search",
        "paper_detail",
        "paper_summary",
        "paper_qa",
        "recommendation",
        "preference_action",
        "reading_list_action",
        "unclear",
        "unsupported",
    }:
        logger.debug(
            "arxiv_agent route_after_parse -> %s: intent=%s pending_action_status=%s paper_qa_status=%s",
            intent,
            intent or "none",
            str((pending_action or {}).get("status") or "none") if isinstance(pending_action, dict) else "none",
            str((pending_qa_result or {}).get("status") or "none"),
        )
        return intent

    # 第 6 步：任何未识别或未登记的 intent 都统一降级为 unsupported。
    # 这样做的好处是图结构始终有明确出口，不会因为新增/异常 intent 导致路由失控。
    logger.warning(
        "arxiv_agent route_after_parse -> unsupported: intent=%s pending_action_status=%s paper_qa_status=%s",
        intent or "none",
        str((pending_action or {}).get("status") or "none") if isinstance(pending_action, dict) else "none",
        str((pending_qa_result or {}).get("status") or "none"),
    )
    return "unsupported"


def route_after_pending_confirmation(state: Any) -> str:
    """根据确认分类结果，决定是否真正执行待确认任务。

    这个函数对应的是“用户回复确认语句之后”的二次分流：
    - 如果识别结果是 confirm，说明用户同意继续执行上一轮待确认任务；
    - 否则统一走 `synthesize_response`，由回复节点给出解释、澄清或结束语。

    这里的判断依据来自 `classify_pending_action_confirmation` 节点写入到
    `state.debug.pending_action_decision` 的分类结果。
    """
    current_state = _coerce_state(state)

    # 从 debug 区域中读取确认分类结果。
    # 这里使用 lower() 做一次归一化，避免上游返回 Confirm/CONFIRM 等大小写差异。
    decision = str((current_state.debug or {}).get("pending_action_decision") or "").strip().lower()

    # 只有明确识别为 confirm 才执行待确认任务；其余情况一律保守处理，直接进入回复节点。
    if decision == "confirm":
        return "handle_pending_action_confirmation"
    return "synthesize_response"


def route_after_check(state: Any) -> str:
    """根据检索结果质量决定是重试搜索还是进入排序汇总阶段。

    这个函数位于“调用搜索工具之后”的关键分叉点，核心目标是处理一种常见情况：
    搜索工具本身调用成功了，但返回的论文列表为空。此时不应该立刻给用户一个
    “没有结果”的生硬响应，而是优先尝试放宽检索条件，提升召回率。

    当前策略是：
    - 工具调用成功；
    - 但 papers 为空；
    - 且重试次数还没有超过上限；
    满足以上 3 个条件时，进入 `relax_search_for_retry` 节点。

    否则，无论是已经搜到结果、工具失败、还是重试次数耗尽，都统一进入
    `personalized_rank_and_annotate_papers`。这样后续节点只需要处理“最终结果态”，
    不必关心前面是否发生过重试。
    """
    current_state = _coerce_state(state)

    # 提取当前搜索相关的关键状态：论文列表、已重试次数、工具原始结果。
    papers = list(current_state.papers or [])
    retry_count = int(current_state.search_retry_count or 0)
    tool_result = current_state.tool_result

    # `tool_ok` 表示“工具调用是否成功”，而不是“是否搜到了论文”。
    # 这两个概念需要分开：调用成功但无结果时才适合放宽条件重试。
    tool_ok = bool(tool_result and isinstance(tool_result, dict) and tool_result.get("ok"))

    # 只有搜索成功但结果为空，且未超过最大重试次数时才触发 fallback
    if tool_ok and not papers and retry_count < 3:
        return "relax_search_for_retry"

    # 其余情况统一收敛到排序与标注节点：
    # - 搜到了结果：进入排序与个性化增强；
    # - 工具失败：交给后续汇总节点基于错误态构造回复；
    # - 重试已达上限：避免死循环，直接结束检索阶段。
    return "personalized_rank_and_annotate_papers"


def build_arxiv_search_graph(generation_service: Optional[Any] = None) -> Any:
    """构建 arXiv 检索 Agent 的完整 LangGraph 工作流。

    这个函数的职责不是执行业务逻辑，而是把各个节点函数按照既定顺序和条件分支
    组装成一张可执行的状态图。编译后的图大致覆盖以下几类能力：
    - arXiv 论文检索；
    - 论文详情/总结/问答等阅读链路；
    - 用户偏好更新；
    - 待确认动作的识别与继续执行；
    - 最终统一回复生成。

    可以把这个函数理解为 Agent 的“主流程编排器”：
    - `add_node(...)` 定义每个处理步骤做什么；
    - `add_edge(...)` 定义固定顺序流转；
    - `add_conditional_edges(...)` 定义需要根据状态动态分支的地方。

    参数 `generation_service` 会透传给依赖大模型推理的节点，方便在运行时注入
    不同的模型服务实现。
    """
    graph = StateGraph(AgentState)

    # 先注册所有节点，再统一声明边，便于从上到下阅读整张图的结构。
    #
    # 其中有些节点是纯函数节点，有些节点需要额外注入 generation_service，
    # 因此这里通过 lambda 做一层轻量包装。
    graph.add_node("parse_search_request", lambda state: parse_search_request(state, generation_service=generation_service))
    graph.add_node("build_search_tool_args", build_search_tool_args)
    graph.add_node("invoke_search_tool", invoke_search_tool)
    graph.add_node("check_search_result", check_search_result)
    graph.add_node("relax_search_for_retry", relax_search_for_retry)
    graph.add_node("personalized_rank_and_annotate_papers", personalized_rank_and_annotate_papers)
    graph.add_node("apply_preference_action", apply_preference_action)
    graph.add_node(
        "classify_pending_action_confirmation",
        lambda state: classify_pending_action_confirmation(state, generation_service=generation_service),
    )
    graph.add_node("handle_pending_action_confirmation", handle_pending_action_confirmation)
    graph.add_node("handle_paper_reading_request", handle_paper_reading_request)
    graph.add_node("synthesize_response", synthesize_response)

    # 所有请求统一从解析节点进入；解析完成后再按意图分流。
    #
    # START -> parse_search_request 是整个图的唯一入口。这样做有两个好处：
    # 1. 所有用户请求都先被标准化解析，避免各分支重复做意图识别；
    # 2. 任何新能力只要挂在 parse 之后即可扩展，整体结构更清晰。
    graph.add_edge(START, "parse_search_request")
    graph.add_conditional_edges(
        "parse_search_request",
        route_after_parse,
        {
            # 标准 arXiv 搜索链路：构造搜索参数 -> 调工具 -> 检查结果 -> 排序标注 -> 生成回复。
            "arxiv_search": "build_search_tool_args",
            # 论文详情/总结/问答都依赖读取具体论文内容，因此走同一处理节点。
            "paper_detail": "handle_paper_reading_request",
            "paper_summary": "handle_paper_reading_request",
            "paper_qa": "handle_paper_reading_request",
            # 推荐、阅读清单操作不需要检索流程，直接汇总回复即可。
            "recommendation": "synthesize_response",
            "preference_action": "apply_preference_action",
            # 若当前消息是在确认上一轮待办动作，则进入确认分类节点。
            "classify_pending_action_confirmation": "classify_pending_action_confirmation",
            "reading_list_action": "synthesize_response",
            "unclear": "synthesize_response",
            "unsupported": "synthesize_response",
        },
    )

    # 针对“当前消息是对上一个挂起动作的确认/拒绝”的场景，再做一次专门分流：
    # - confirm -> 执行挂起动作；
    # - 其他 -> 不继续执行，直接组织回复。
    graph.add_conditional_edges(
        "classify_pending_action_confirmation",
        route_after_pending_confirmation,
        {
            "handle_pending_action_confirmation": "handle_pending_action_confirmation",
            "synthesize_response": "synthesize_response",
        },
    )

    # 这些边表示若干“非搜索型流程”的固定收敛关系：
    # 它们各自完成动作后，不再进入搜索链路，而是统一进入回复生成节点。
    graph.add_edge("apply_preference_action", "synthesize_response")
    graph.add_edge("handle_pending_action_confirmation", "synthesize_response")
    graph.add_edge("handle_paper_reading_request", "synthesize_response")

    # 下面是标准搜索链路：
    # 1. 根据解析结果构造搜索参数；
    # 2. 调用外部搜索工具；
    # 3. 检查工具返回，决定是否需要放宽条件重试。
    graph.add_edge("build_search_tool_args", "invoke_search_tool")
    graph.add_edge("invoke_search_tool", "check_search_result")
    graph.add_conditional_edges(
        "check_search_result",
        route_after_check,
        {
            # 命中为空时先放宽条件重试，否则进入个性化排序与标注。
            "relax_search_for_retry": "relax_search_for_retry",
            "personalized_rank_and_annotate_papers": "personalized_rank_and_annotate_papers",
        },
    )

    # 放宽检索条件后会重新构造工具参数，形成有限次重试闭环。
    graph.add_edge("relax_search_for_retry", "build_search_tool_args")

    # 一旦拿到最终论文结果，就进入个性化排序和标注，再统一由回复节点产出用户可读文本。
    graph.add_edge("personalized_rank_and_annotate_papers", "synthesize_response")

    # 整个图的唯一出口。无论走哪条业务分支，最终都应收敛到 synthesize_response 后结束。
    graph.add_edge("synthesize_response", END)

    return graph.compile()


def export_arxiv_search_graph_mermaid(generation_service: Optional[Any] = None) -> dict[str, Any]:
    """导出 arXiv Agent 的图结构，方便在调试页或前端直接渲染。

    优先使用 LangGraph 自带的 `get_graph().draw_mermaid()`，这样导出的内容
    会严格跟随运行时图结构；如果当前环境缺少对应绘图能力，就回退到手工
    生成的 Mermaid，保证调试入口仍然可用。

    返回值除了 Mermaid 文本本身，还包含：
    - graph_name：图名称，便于前端标识；
    - render_source：图内容来自 langgraph 还是 fallback；
    - node_names：节点名列表，便于调试页展示；
    - supports_png：当前运行环境是否支持直接导出 PNG。

    这个接口主要服务于“可观测性”和“开发调试”，而不是核心业务链路。
    """

    # 先按当前配置构建并编译实际运行时图，确保导出的结构与真实执行流程一致。
    compiled_graph = build_arxiv_search_graph(generation_service=generation_service)
    drawable_graph = None
    mermaid = ""
    render_source = "langgraph"

    try:
        # 优先走 LangGraph 官方提供的可视化能力，这样最不容易和真实图结构脱节。
        drawable_graph = compiled_graph.get_graph()
        mermaid = drawable_graph.draw_mermaid()
    except Exception as exc:  # pragma: no cover - 依赖版本差异时走兜底
        # 某些环境中可能没有安装完整绘图依赖，或者不同版本的 LangGraph API 有差异。
        # 这里降级为手工 Mermaid，至少保证调试页“有图可看”。
        render_source = "fallback"
        logger.warning("arxiv_agent graph mermaid export fallback: error=%s", exc)
        mermaid = _build_fallback_mermaid()

    # 统一返回给上层调用方，便于前端/调试接口直接消费。
    return {
        "graph_name": "arxiv_search_agent",
        "render_source": render_source,
        "node_names": list(_ARXIV_GRAPH_NODE_NAMES),
        "mermaid": mermaid,
        "supports_png": bool(drawable_graph and hasattr(drawable_graph, "draw_mermaid_png")),
    }


def _coerce_state(state: Any) -> AgentState:
    """把运行时可能出现的多种 state 形态统一规整为 AgentState。

    在 LangGraph 运行过程中，节点和路由函数接收到的 state 并不总是完全一致的类型：
    - 可能已经是 `AgentState` 实例；
    - 可能是普通 `dict` / `Mapping`；
    - 也可能是其他可被 Pydantic 校验的对象。

    为了让后续代码始终可以用统一字段访问方式（如 `current_state.intent`），
    这里集中做一次“状态归一化”。这样上层路由函数就不需要到处写类型分支。
    """
    if isinstance(state, AgentState):
        # 返回深拷贝，避免路由函数误改原始状态对象。
        return state.model_copy(deep=True)
    if isinstance(state, Mapping):
        # Mapping 先转成普通 dict，再交给 Pydantic 校验，避免某些惰性映射对象行为不一致。
        return AgentState.model_validate(dict(state))

    # 兜底分支：只要对象结构能被 AgentState 接受，就尝试直接校验转换。
    return AgentState.model_validate(state)


def _build_fallback_mermaid() -> str:
    """生成一份静态 Mermaid 图，作为运行时绘图失败时的兜底结果。

    这份 Mermaid 不是动态从运行时对象反射出来的，而是手工维护的一份静态结构描述。
    它的主要作用不是 100% 替代 LangGraph 原生绘图，而是在以下场景下提供最小可用能力：
    - 本地环境缺少 Mermaid/PNG 渲染依赖；
    - LangGraph 版本差异导致 `draw_mermaid()` 不可用；
    - 调试页只需要快速看到主流程拓扑，不要求完整元数据。

    因为这是一份手工兜底图，所以后续如果主流程节点有增删改，最好同步更新这里，
    否则调试页看到的 fallback 图可能与真实执行图存在偏差。
    """
    return "\n".join(
        [
            # 图整体结构遵循与 build_arxiv_search_graph 相同的主流程顺序：
            # parse -> 分流 -> 搜索/阅读/偏好处理 -> synthesize -> END。
            "graph TD;",
            "    START([START]) --> parse_search_request;",
            "    parse_search_request --> build_search_tool_args;",
            "    parse_search_request --> handle_paper_reading_request;",
            "    parse_search_request --> apply_preference_action;",
            "    parse_search_request --> classify_pending_action_confirmation;",
            "    parse_search_request -->|recommendation| synthesize_response;",
            "    parse_search_request -->|reading_list_action| synthesize_response;",
            "    parse_search_request -->|unclear| synthesize_response;",
            "    parse_search_request -->|unsupported| synthesize_response;",
            "    build_search_tool_args --> invoke_search_tool;",
            "    invoke_search_tool --> check_search_result;",
            "    check_search_result --> relax_search_for_retry;",
            "    check_search_result --> personalized_rank_and_annotate_papers;",
            "    relax_search_for_retry --> build_search_tool_args;",
            "    personalized_rank_and_annotate_papers --> synthesize_response;",
            "    apply_preference_action --> synthesize_response;",
            "    classify_pending_action_confirmation --> handle_pending_action_confirmation;",
            "    classify_pending_action_confirmation --> synthesize_response;",
            "    handle_pending_action_confirmation --> synthesize_response;",
            "    handle_paper_reading_request --> synthesize_response;",
            "    synthesize_response --> END([END]);",
        ]
    )


__all__ = [
    "build_arxiv_search_graph",
    "export_arxiv_search_graph_mermaid",
    "route_after_parse",
    "START",
    "END",
]
