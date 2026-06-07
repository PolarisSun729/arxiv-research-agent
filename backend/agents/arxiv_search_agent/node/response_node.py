"""把中间状态整理成最终返回给用户的自然语言答复。

这个节点是工作流的最后一站，不执行搜索或问答本身，而是：
1. 按优先级检查论文阅读、待确认任务、偏好操作和搜索结果；
2. 选择对应的话术模板、后续建议和错误提示；
3. 生成最终 answer，并把 next_actions 保持在可直接展示的结构上。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..state import AgentState
from ..utils.state_utils import _append_step, _coerce_state, _update_execution_plan_step
from .search_node import _collect_priority_titles, _summarize_search_spec


def _latest_tool_observation(state: AgentState, tool_name: str) -> Optional[Dict[str, Any]]:
    for observation in reversed(list(state.tool_observations or [])):
        if str(getattr(observation, "tool_name", "") or "").strip() == tool_name:
            return observation.model_dump()
    return None


def _build_paper_qa_result_from_observation(state: AgentState) -> Optional[Dict[str, Any]]:
    observation = _latest_tool_observation(state, "answer_paper_question")
    if not observation or not observation.get("ok"):
        return None

    tool_result = dict(state.tool_result or {})
    if str(tool_result.get("tool_name") or "").strip() != "answer_paper_question":
        return None
    data = tool_result.get("data") if isinstance(tool_result.get("data"), Mapping) else {}
    answer = str((data or {}).get("answer") or "").strip()
    if not answer:
        return None

    context = dict(state.context or {})
    selected_paper = context.get("selected_paper") if isinstance(context.get("selected_paper"), Mapping) else {}
    return {
        "status": "success",
        "arxiv_id": context.get("arxiv_id") or selected_paper.get("arxiv_id"),
        "title": selected_paper.get("title"),
        "question": (state.debug or {}).get("qa_question"),
        "answer": answer,
        "sources": (data or {}).get("sources", []),
        "retrieval_debug": (data or {}).get("retrieval_debug"),
        "qa_index_status": (state.debug or {}).get("qa_index_status"),
        "index_created": bool((state.debug or {}).get("index_created")),
        "error": None,
    }


def _has_preference_insufficiency_signals(state: AgentState) -> bool:
    warnings = [str(item).strip() for item in list(state.warnings or []) if str(item).strip()]
    keywords = ("缺少", "偏好", "画像", "喜欢", "不喜欢", "user_id")
    return any(any(keyword in warning for keyword in keywords) for warning in warnings)


def _build_non_search_answer(intent: str) -> Tuple[str, List[str]]:
    """为未真正进入搜索执行链路的意图生成兜底回复。
    
    主要用途：
    1. 给 paper_summary / paper_detail / recommendation 等非搜索分支提供解释性文案；
    2. 在当前轮缺少可执行上下文时，明确告诉用户还缺什么；
    3. 统一返回 answer 与 next_actions，避免各分支自行拼接不一致的话术。
    """
    if intent == "paper_summary":
        return (
            "我已经识别到你想总结某篇论文，但当前这次请求还没有拿到可执行的目标论文上下文。你可以直接给我论文标题、arXiv ID，或先搜索后再让我总结。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你要总结某篇论文，请提供标题或 arXiv ID",
            ],
        )

    if intent == "paper_detail":
        return (
            "我已经识别到你想解释某篇论文的方法或细节，但当前这次请求还没有拿到可执行的目标论文上下文。你可以直接给我论文标题、arXiv ID，或先搜索后再让我展开解释。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你已经有论文标题或 arXiv ID，请把它发给我",
            ],
        )

    if intent == "paper_qa":
        return (
            "我已经识别到你想围绕某篇论文提问，但当前这次请求还没有拿到可执行的目标论文上下文。你可以直接给我目标论文标题、arXiv ID，或先搜索后再继续提问。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你已经有论文标题或 arXiv ID，请把它发给我",
            ],
        )

    if intent == "recommendation":
        return (
            "我已经识别到你想做论文推荐，但当前这次请求还没有拿到可用的推荐结果。你可以补充研究方向、关键词，或者先标记几篇喜欢 / 不喜欢的论文让我建立更稳定的推荐依据。",
            [
                "补充研究方向、关键词或时间范围",
                "先标记几篇喜欢 / 不喜欢的论文，再让我做推荐",
            ],
        )

    if intent == "preference_action":
        return (
            "我已经识别到你想做偏好操作，但当前这次请求还没有拿到足够明确的目标论文。你可以直接提供 arXiv ID，或者先搜索后再说“喜欢第一篇 / 不喜欢这篇”。",
            [
                "如果你要的是搜索，请直接描述论文主题或关键词",
                "如果你要标记某篇论文，请直接提供 arXiv ID 或引用搜索结果中的论文",
            ],
        )

    if intent == "unclear":
        return (
            "我能确定你是在找论文，但主题还不够明确。",
            [
                "补充研究方向、关键词或时间范围",
                "例如：RAG、LLM、Agent、NLP、推荐系统",
            ],
        )

    return (
        "当前请求超出 arXiv 搜索 Agent 的处理范围，请改写成明确的论文检索需求。",
        [
            "改写成 arXiv 论文搜索问题",
            "如果需要论文总结或问答，请先提供目标论文标题或 arXiv ID",
        ],
    )


def synthesize_response(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """根据 AgentState 中的执行结果生成最终用户答复。
    
    优先级从高到低依次处理：
    1. 论文阅读成功或失败；
    2. 挂起任务的确认、取消与等待提示；
    3. 偏好动作结果；
    4. arXiv 搜索结果；
    5. 非搜索兜底文案。
    
    输出：返回 answer、next_actions 已就绪的 AgentState，并追加 final_answer_generation trace。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    def _finalize_response_plan(status: str) -> AgentState:
        return _update_execution_plan_step(next_state, step_type="response_synthesis", status=status)

    pending_action = dict(next_state.pending_action or (next_state.context or {}).get("pending_action") or {})
    paper_qa_result = dict(next_state.paper_qa_result or {})
    if next_state.intent in {"paper_summary", "paper_detail", "paper_qa"} and not paper_qa_result:
        inferred_paper_qa_result = _build_paper_qa_result_from_observation(next_state)
        if inferred_paper_qa_result:
            paper_qa_result = inferred_paper_qa_result
            next_state.paper_qa_result = inferred_paper_qa_result

    # 已经拿到论文 QA 的最终答案时，直接复用该答案，不再重新包装过多说明。
    if next_state.intent in {"paper_summary", "paper_detail", "paper_qa"} and paper_qa_result.get("status") == "success":
        next_state.answer = str(paper_qa_result.get("answer") or next_state.answer or "").strip()
        if not next_state.next_actions:
            next_state.next_actions = [
                "继续追问这篇论文的其他细节",
                "切换到其他论文继续阅读",
            ]
        response_state = _finalize_response_plan("completed")
        return _append_step(
            response_state,
            step="final_answer_generation",
            status="success",
            action="生成论文阅读回复",
            inputs={"intent": next_state.intent, "paper_qa_result": dict(paper_qa_result)},
            outputs={
                "answer": response_state.answer,
                "next_actions": list(response_state.next_actions),
                "qa_index_status": paper_qa_result.get("qa_index_status"),
            },
        )

    # 失败状态来自实际工具执行或兼容节点的安全停止，直接返回可读错误，避免再依赖旧的自然语言确认分类结果。
    if next_state.intent in {"paper_summary", "paper_detail", "paper_qa"} and paper_qa_result.get("status") == "failed":
        error_message = str(paper_qa_result.get("error") or paper_qa_result.get("answer") or "论文解析或问答执行失败").strip()
        next_state.answer = error_message
        next_state.next_actions = [
            "通过 Agent 主流程重新发起论文问答",
            "或稍后重试这篇论文",
        ]
        response_state = _finalize_response_plan("failed")
        return _append_step(
            response_state,
            step="final_answer_generation",
            status="failed",
            action="生成论文阅读回复",
            inputs={"intent": next_state.intent, "paper_qa_result": dict(paper_qa_result)},
            outputs={"answer": response_state.answer, "next_actions": list(response_state.next_actions)},
            error=error_message,
        )

    # 新确认链路中的 pending_action 只是 ConfirmationRequest 的展示镜像；
    # 用户操作必须走结构化 resume，不能再通过“解析/取消”这类普通聊天消息驱动状态机。
    if isinstance(pending_action, dict) and str(pending_action.get("type") or "").strip() == "tool_approval":
        title = str(pending_action.get("title") or "").strip()
        arxiv_id = str(pending_action.get("arxiv_id") or "").strip()
        qa_question = str(pending_action.get("qa_question") or "").strip()
        base_prompt = f"《{title}》" if title else "这篇论文"
        pending_status = str(pending_action.get("status") or "").strip().lower()
        if pending_status == "cancelled" or str(pending_action.get("decision") or "").strip().lower() == "reject":
            next_state.answer = f"已取消解析{base_prompt}。暂时无法基于全文回答。"
            next_state.paper_qa_result = {
                **paper_qa_result,
                "status": "failed",
                "arxiv_id": arxiv_id or paper_qa_result.get("arxiv_id"),
                "title": title or paper_qa_result.get("title"),
                "question": qa_question or paper_qa_result.get("question"),
                "answer": "",
                "error": "user cancelled pending action",
            }
            next_state.pending_action = None
            next_state.context = dict(next_state.context or {})
            next_state.context.pop("pending_action", None)
            next_state.next_actions = [
                "继续搜索其他论文",
                "重新发起论文阅读请求",
            ]
        else:
            pending_message = str(paper_qa_result.get("answer") or "").strip()
            if not pending_message:
                pending_message = f"{base_prompt} 还没有建立问答索引，需要先确认是否执行 {pending_action.get('tool_name') or '解析索引'}。"
            next_state.answer = pending_message
            next_state.next_actions = ["使用确认卡片继续", "拒绝后重新选择论文"]

        response_state = _finalize_response_plan("completed")
        return _append_step(
            response_state,
            step="final_answer_generation",
            status="success",
            action="生成论文阅读回复",
            inputs={
                "intent": next_state.intent,
                "pending_action": pending_action,
                "paper_qa_result": dict(paper_qa_result),
            },
            outputs={"answer": response_state.answer, "next_actions": list(response_state.next_actions)},
        )

    # 偏好动作以“动作是否成功”为主来组织答复，同时补充标题和 arXiv ID 便于用户确认对象。
    if next_state.intent == "preference_action":
        result = next_state.preference_action_result or {}
        title = str(result.get("title") or "").strip()
        arxiv_id = str(result.get("arxiv_id") or "").strip()
        action = str(result.get("action") or "remove")
        label = str(result.get("label") or "none")
        message = str(result.get("message") or "").strip()
        error = str(result.get("error") or "").strip()

        if result.get("status") == "success":
            if action == "like":
                next_state.answer = "已将该论文标记为感兴趣。"
            elif action == "dislike":
                next_state.answer = "已将该论文标记为不感兴趣。"
            else:
                next_state.answer = "已取消该论文的偏好标记。"
            if title:
                next_state.answer += f"\n\n《{title}》"
            if arxiv_id:
                next_state.answer += f"\narXiv ID: {arxiv_id}"
            if action == "like":
                next_state.answer += "\n\n后续推荐会参考这个偏好。"
            elif action == "dislike":
                next_state.answer += "\n\n后续推荐会尽量降低类似论文的权重。"
            else:
                next_state.answer += "\n\n后续推荐会恢复对这篇论文的中性处理。"
            next_state.next_actions = [
                "继续对其他论文执行喜欢、不喜欢或取消偏好动作",
                "也可以继续搜索、查看推荐或打开论文详情",
            ]
        else:
            next_state.answer = message or error or "偏好动作执行失败。"
            next_state.next_actions = [
                "先搜索论文，再使用“第一篇 / 第二篇”来标记",
                "也可以直接提供 arXiv ID 后重试",
            ]

        response_state = _finalize_response_plan("completed")
        return _append_step(
            response_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={"intent": next_state.intent, "preference_action_result": dict(result)},
            outputs={"answer": response_state.answer, "next_actions": list(response_state.next_actions), "label": label},
        )

    if next_state.intent == "recommendation":
        papers = list(next_state.papers or [])
        tool_result = dict(next_state.tool_result or {})
        recommendation_debug = dict((next_state.debug or {}).get("recommendation_result") or {})
        personalization_signals = dict(recommendation_debug.get("personalization_signals") or {})
        paper_count = len(papers)
        priority_titles = _collect_priority_titles(papers, limit=3)
        if paper_count > 0:
            next_state.answer = f"已生成 {paper_count} 篇个性化论文推荐。"
            if priority_titles:
                next_state.answer += f" 建议优先阅读：{', '.join(priority_titles)}。"
            rationale_parts = []
            if personalization_signals.get("used_user_memory_summary") or personalization_signals.get("used_research_profile"):
                rationale_parts.append("已参考用户画像")
            if personalization_signals.get("has_behavior_history"):
                rationale_parts.append("已参考历史偏好")
            if personalization_signals.get("interest_cluster_count"):
                rationale_parts.append(f"兴趣簇数量={personalization_signals.get('interest_cluster_count')}")
            if personalization_signals.get("recall_mode"):
                rationale_parts.append(f"召回模式={personalization_signals.get('recall_mode')}")
            if rationale_parts:
                next_state.answer += "\n\n推荐依据：" + "，".join(rationale_parts) + "。"
            next_state.next_actions = [
                "继续查看其中某篇论文的详情或总结",
                "也可以继续标记喜欢 / 不喜欢来优化后续推荐",
            ]
            response_state = _finalize_response_plan("completed")
            return _append_step(
                response_state,
                step="final_answer_generation",
                status="success",
                action="生成推荐结果回复",
                inputs={"intent": next_state.intent, "paper_count": paper_count},
                outputs={
                    "answer": response_state.answer,
                    "next_actions": list(response_state.next_actions),
                    "personalization_signals": personalization_signals,
                },
            )

        if tool_result:
            if bool(tool_result.get("ok")):
                if _has_preference_insufficiency_signals(next_state):
                    next_state.answer = "本次已执行推荐工具，但当前可用的偏好或用户画像还不足，暂时无法生成稳定的个性化推荐结果。"
                else:
                    next_state.answer = "本次已执行推荐工具，但暂时没有生成可展示的推荐结果。"
                if next_state.warnings:
                    next_state.answer += "\n\n提示：" + "；".join(str(item) for item in next_state.warnings if str(item).strip())
                next_state.next_actions = [
                    "补充更明确的研究方向或关键词后重试",
                    "先标记几篇喜欢 / 不喜欢的论文，再让我重新推荐",
                ]
                response_state = _finalize_response_plan("completed")
                return _append_step(
                    response_state,
                    step="final_answer_generation",
                    status="success",
                    action="生成推荐结果回复",
                    inputs={"intent": next_state.intent, "paper_count": paper_count},
                    outputs={
                        "answer": response_state.answer,
                        "next_actions": list(response_state.next_actions),
                        "warnings": list(next_state.warnings or []),
                    },
                )

            error_message = str(((tool_result.get("error") or {}).get("message")) or tool_result.get("summary") or "推荐论文失败").strip()
            next_state.answer = error_message
            next_state.next_actions = [
                "稍后重试推荐",
                "也可以先搜索某个明确主题的论文",
            ]
            response_state = _finalize_response_plan("failed")
            return _append_step(
                response_state,
                step="final_answer_generation",
                status="failed",
                action="生成推荐结果回复",
                inputs={"intent": next_state.intent, "paper_count": paper_count},
                outputs={"answer": response_state.answer, "next_actions": list(response_state.next_actions)},
                error=error_message,
            )

    # 搜索回复会结合是否做过个性化重排，给出不同的解释和后续建议。
    if next_state.intent == "arxiv_search":
        spec = next_state.search_spec
        papers = list(next_state.papers or [])
        paper_count = len(papers)
        max_results = spec.max_results if spec is not None else 10
        summary = _summarize_search_spec(spec)
        priority_titles = _collect_priority_titles(papers, limit=3)
        personalized_applied = bool(next_state.personalized_rerank_applied)

        if paper_count > 0:
            next_state.answer = f"已按“{summary}”搜索 arXiv，当前返回 {paper_count} 篇论文。"
            if personalized_applied:
                if priority_titles:
                    next_state.answer += f" 本次结果已根据用户偏好重新排序，建议优先阅读：{', '.join(priority_titles)}。"
                else:
                    next_state.answer += " 本次结果已根据用户偏好重新排序，建议优先阅读排序靠前的论文。"
            else:
                next_state.answer += " 本次结果未使用用户偏好向量，保持普通搜索排序。"
            next_state.next_actions = [
                "继续缩小到某个子方向搜索",
                "选择一篇论文查看详情",
                "后续可以接论文总结或 QA 功能",
            ]
        else:
            next_state.answer = f"已按“{summary}”搜索 arXiv，但当前没有找到结果。"
            next_state.next_actions = [
                "放宽关键词或扩大时间范围后重试",
                "只保留核心主题词再搜索",
                "后续可以接论文总结或 QA 功能",
            ]

        if max_results and paper_count < max_results:
            next_state.answer += f" 本次最多期望返回 {max_results} 篇。"

        response_state = _finalize_response_plan("completed")
        return _append_step(
            response_state,
            step="final_answer_generation",
            status="success",
            action="生成最终答复并给出后续动作",
            inputs={
                "intent": next_state.intent,
                "paper_count": paper_count,
                "personalized_rerank_applied": personalized_applied,
            },
            outputs={
                "answer": response_state.answer,
                "next_actions": list(response_state.next_actions),
                "top_papers": _collect_priority_titles(papers, limit=3),
            },
        )

    answer, next_actions = _build_non_search_answer(next_state.intent or "unsupported")
    next_state.answer = answer
    next_state.next_actions = next_actions
    response_state = _finalize_response_plan("completed")
    return _append_step(
        response_state,
        step="final_answer_generation",
        status="success",
        action="生成最终答复并给出后续动作",
        inputs={"intent": next_state.intent, "paper_count": len(next_state.papers or [])},
        outputs={"answer": response_state.answer, "next_actions": list(response_state.next_actions)},
    )


__all__ = ["synthesize_response"]
