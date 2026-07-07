"""处理论文总结、细节解释和论文问答请求。

这个节点负责把“围绕某篇论文继续阅读”的自然语言请求转成可执行流程：
1. 解析用户引用的目标论文；
2. 把原始表达改写成更适合全文 QA 的标准问题；
3. 判断论文是否已有问答索引；
4. 已有索引时直接回答，缺索引时提示调用方改走 PlanExecutor 的统一确认链路。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

_BACKEND_DIR = str(Path(__file__).resolve().parents[3])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ..schemas import ToolCallRequest
from ..state import AgentState
from ..utils.result_utils import _extract_error_message, _result_mapping, _result_ok
from ..utils.paper_reference_resolver import _normalize_context_paper, _resolve_paper_reference
from ..utils.paper_question_normalizer import normalize_single_paper_qa_question
from ..utils.state_utils import _append_step, _coerce_state, _get_execution_plan_step
from ..utils.text_utils import _matches_any, _normalize_text
from .tool_node import execute_tool


def _build_qa_question_for_paper(intent: str, message: str, paper: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    """把用户原始表达清洗成更适合论文全文问答服务的标准问题。
    
    主要步骤：
    1. 去掉“问一下”“第一篇”“这篇论文”“arXiv ID”这类引用性噪声；
    2. 在 paper_summary 分支下生成覆盖研究问题、方法、实验和局限性的总结模板；
    3. 在 paper_detail 分支下把过短的“解释/方法”请求扩展成更完整的问题；
    4. paper_qa 分支则尽量保留用户原问题，只在空文本时给默认问法。
    """
    original_question = _normalize_text(message)
    title = _normalize_text(str(paper.get("title") or reference.get("title") or ""))

    cleaned_question = normalize_single_paper_qa_question(original_question, reference)
    # 依次剥离“问一下”等动作噪声；目标论文引用已在已解析目标的前提下归一化，避免误删论文内部的“第二节”。
    cleaned_question = re.sub(r"^(问一下|请问一下|请问|问|帮我问一下|帮我问|想问一下)\s*", "", cleaned_question).strip()
    cleaned_question = re.sub(
        r"^(?:(?:最后一|最后1|最后|末一|末)\s*篇(?:论文|paper)?|第\s*[一二三四五六七八九十两0-9]+\s*篇(?:论文|paper)?|[1-9]|1[0-9]|20)\s*[:：,，]?\s*",
        "",
        cleaned_question,
    ).strip()
    cleaned_question = re.sub(r"^arxiv\s*id\s*[:：]?\s*\d{4}\.\d{4,5}(?:v\d+)?\s*", "", cleaned_question, flags=re.IGNORECASE).strip()
    cleaned_question = re.sub(r"^\d{4}\.\d{4,5}(?:v\d+)?\s*", "", cleaned_question).strip()
    cleaned_question = cleaned_question.lstrip("，,:：.。;； ")

    if intent == "paper_summary":
        # 总结类请求统一扩展成完整的总结模板，保证输出覆盖研究问题、方法和结果等关键维度。
        return (
            f"请基于论文全文总结这篇论文，包含研究问题、核心贡献、方法流程、实验设置、主要结果和局限性。"
            f"{f' 论文标题：{title}' if title else ''}"
        ).strip()

    if intent == "paper_detail":
        detail_question = cleaned_question or original_question
        # 对“解释 / 方法 / 介绍一下”这类过短请求，主动补成更完整的细节解释指令。
        if _matches_any(detail_question, (r"^解释$", r"^讲讲$", r"^介绍一下$", r"^方法$", r"^讲讲方法$", r"^解释方法$")):
            detail_question = "请详细解释这篇论文的方法设计、关键模块、输入输出流程，以及这样设计的原因。"
        if not detail_question:
            detail_question = "请详细解释这篇论文的主要方法、实验设计和贡献。"
        return detail_question

    if not cleaned_question:
        cleaned_question = "请基于论文全文回答这个问题。"
    return cleaned_question


def _get_plan_step_id(state: AgentState, step_type: str) -> Optional[str]:
    step = _get_execution_plan_step(state, step_type=step_type)
    if step is not None:
        step_id = str(getattr(step, "step_id", "") or "").strip()
        return step_id or None
    return None


def _has_ready_qa_index(qa_status: Mapping[str, Any]) -> bool:
    normalized_status = str(qa_status.get("status") or "").strip().lower()
    return bool(qa_status.get("has_index")) or normalized_status == "indexed"


def handle_paper_reading_request(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """处理论文阅读类请求，并在兼容调用场景下尽量直接回答。
    
    主要分支包括：
    1. 不是论文阅读 intent，直接跳过；
    2. 无法解析目标论文，返回失败提示；
    3. 已有 QA 索引，直接调用问答服务并返回答案；
    4. 尚无 QA 索引，停止在本节点内处理，提示调用方改走 PlanExecutor 的 interrupt/resume 确认机制。
    
    输出：返回更新后的 AgentState，包含 paper_qa_result、answer、next_actions、debug 和上下文中的 selected_paper。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    intent = str(next_state.intent or "").strip()
    if intent not in {"paper_summary", "paper_detail", "paper_qa"}:
        return _append_step(
            next_state,
            step="handle_paper_reading_request",
            status="skipped",
            action="处理论文阅读请求",
            inputs={"intent": intent},
            outputs={"reason": "当前 intent 不是论文阅读类请求"},
        )

    # 先提取“这篇论文 / 第一篇论文”这类引用线索；最终论文绑定必须交给后续解析层。
    message = _normalize_text(next_state.message or "")
    context = dict(next_state.context or {})
    resolution = _resolve_paper_reference(message, context)
    # 兼容节点不能把 hint extractor 的结果当作最终论文；只有统一 Target Resolver 标记
    # final_target_resolved 后，才允许进入后续业务工具。
    if resolution.get("status") != "success" or not resolution.get("final_target_resolved"):
        hint_only_reason = (
            "已提取论文引用线索，但尚未结合上下文解析成最终论文。"
            if resolution.get("status") == "hint_extracted"
            else "无法解析目标论文"
        )
        result = {
            "status": "failed",
            "arxiv_id": None,
            "title": None,
            "question": message,
            "answer": str(resolution.get("reason") or hint_only_reason),
            "sources": [],
            "retrieval_debug": None,
            "qa_index_status": None,
            "index_created": False,
            "error": str(resolution.get("reason") or hint_only_reason),
        }
        next_state.paper_qa_result = result
        next_state.answer = result["answer"]
        next_state.next_actions = [
            "请先搜索论文，或直接提供 arXiv ID",
            "也可以用“第一篇 / 第二篇”再次指定目标论文",
        ]
        next_state.debug = dict(next_state.debug or {})
        next_state.debug["qa_question"] = None
        next_state.debug["paper_resolution"] = resolution
        return _append_step(
            next_state,
            step="handle_paper_reading_request",
            status="failed",
            action="处理论文阅读请求",
            inputs={"message": message, "intent": intent, "context": context},
            outputs={"paper_qa_result": result, "resolution": resolution},
            error=result["error"],
        )

    # 解析成功后把论文基础信息拆出来，后续无论是直接问答还是挂起确认都会复用这些字段。
    paper = resolution.get("paper") or {}
    arxiv_id = str(resolution.get("arxiv_id") or "").strip()
    title = str(resolution.get("title") or paper.get("title") or "").strip()
    # 论文定位成功后，把用户问题重写成更适合全文 QA 的标准化问题。
    qa_question = _build_qa_question_for_paper(intent, message, paper, resolution)
    loading_method = str(context.get("loading_method") or "docling").strip() or "docling"
    debug = dict(next_state.debug or {})
    debug["qa_question"] = qa_question
    debug["qa_index_status"] = None
    debug["paper_resolution"] = resolution
    debug["paper_reading_intent"] = intent
    debug["paper_qa_target"] = {"arxiv_id": arxiv_id, "title": title}
    debug["confirmation_state"] = "none"
    debug["paper_qa_answer_status"] = "pending"
    next_state.debug = debug
    next_state.context = dict(next_state.context or {})
    # 把本轮解析出的目标论文写回上下文，支持后续“继续问这篇论文”这类省略表达。
    next_state.context["selected_paper"] = _normalize_context_paper(
        {
            **({} if not isinstance(paper, Mapping) else dict(paper)),
            "arxiv_id": arxiv_id,
            "title": title,
        }
    )
    next_state.context["arxiv_id"] = arxiv_id
    next_state.tool_name = "check_paper_qa_index"
    next_state.tool_args = {
        "arxiv_id": arxiv_id,
        "intent": intent,
        "question": qa_question,
        "loading_method": loading_method,
    }
    next_state.tool_call_request = ToolCallRequest(
        tool_name="check_paper_qa_index",
        arguments={"arxiv_id": arxiv_id},
        reason="先确认目标论文是否已有可复用的 QA 索引",
        expected_result="返回论文 QA 索引状态",
        plan_step_id=_get_plan_step_id(next_state, "qa_index_check"),
    )
    next_state = execute_tool(next_state)
    qa_status = dict(_result_mapping(next_state.tool_result or {}, "data") or {})
    next_state.debug = dict(next_state.debug or {})
    next_state.debug["qa_index_status"] = qa_status
    next_state.debug["qa_index_checked"] = True

    if not _result_ok(next_state.tool_result or {}):
        error_message = _extract_error_message(next_state.tool_result or {}) or "检查论文 QA 索引失败"
        result = {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "title": title,
            "question": qa_question,
            "answer": error_message,
            "sources": [],
            "retrieval_debug": None,
            "qa_index_status": qa_status or None,
            "index_created": False,
            "error": error_message,
        }
        next_state.paper_qa_result = result
        next_state.answer = error_message
        next_state.debug["paper_qa_answer_status"] = "check_failed"
        next_state.next_actions = [
            "稍后重试该论文问答",
            "或者改用 arXiv ID 重新指定目标论文",
        ]
        return _append_step(
            next_state,
            step="handle_paper_reading_request",
            status="failed",
            action="处理论文阅读请求",
            inputs={"message": message, "intent": intent, "context": context},
            outputs={"paper_qa_result": result, "qa_index_status": qa_status},
            error=error_message,
        )

    # 统一把索引状态标准化成 dict，避免后续多处分支同时处理 None / Mapping 两种形态。
    qa_status = qa_status or {}

    # 已有索引时走快速路径：无需再让用户确认，直接问答即可。
    if _has_ready_qa_index(qa_status):
        try:
            # 已有索引时直接进入问答，不再要求用户重复确认，尽量缩短阅读链路延迟。
            next_state.tool_call_request = ToolCallRequest(
                tool_name="answer_paper_question",
                arguments={"arxiv_id": arxiv_id, "question": qa_question},
                reason="复用已有论文 QA 索引直接回答阅读问题",
                expected_result="返回论文问题答案、引用片段和检索调试信息",
                plan_step_id=_get_plan_step_id(next_state, "paper_response"),
            )
            next_state = execute_tool(next_state)
            if not _result_ok(next_state.tool_result or {}):
                raise RuntimeError(_extract_error_message(next_state.tool_result or {}) or "论文问答执行失败")

            answer_result = dict(_result_mapping(next_state.tool_result or {}, "data") or {})
            result = {
                "status": "success",
                "arxiv_id": arxiv_id,
                "title": title,
                "question": qa_question,
                "answer": answer_result.get("answer", ""),
                "sources": answer_result.get("sources", []),
                "retrieval_debug": answer_result.get("retrieval_debug"),
                "qa_index_status": qa_status,
                "index_created": False,
                "error": None,
            }
            next_state.paper_qa_result = result
            next_state.answer = result["answer"]
            next_state.papers = list(answer_result.get("papers", [])) if isinstance(answer_result, Mapping) and answer_result.get("papers") else list(next_state.papers or [])
            next_state.next_actions = [
                "继续追问这篇论文的其他细节",
                "切换到其他论文继续阅读",
            ]
            next_state.pending_action = None
            next_state.context.pop("pending_action", None)
            next_state.debug["confirmation_state"] = "none"
            next_state.debug["paper_qa_answer_status"] = "answered"
            return _append_step(
                next_state,
                step="handle_paper_reading_request",
                status="success",
                action="处理论文阅读请求",
                inputs={"message": message, "intent": intent, "context": context},
                outputs={
                    "paper_qa_result": result,
                    "qa_index_status": qa_status,
                    "qa_question": qa_question,
                },
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "arxiv_id": arxiv_id,
                "title": title,
                "question": qa_question,
                "answer": "",
                "sources": [],
                "retrieval_debug": None,
                "qa_index_status": qa_status,
                "index_created": False,
                "error": str(exc),
            }
            next_state.paper_qa_result = result
            next_state.answer = f"论文问答执行失败: {exc}"
            next_state.next_actions = [
                "稍后重试该论文问答",
                "或者先确认是否要重新解析 PDF",
            ]
            return _append_step(
                next_state,
                step="handle_paper_reading_request",
                status="failed",
                action="处理论文阅读请求",
                inputs={"message": message, "intent": intent, "context": context},
                outputs={"paper_qa_result": result, "qa_index_status": qa_status},
                error=str(exc),
            )

    # 兼容节点已经不再自建确认状态机，否则会生成没有 LangGraph checkpoint 的“假确认”。
    # 缺索引场景应交给 PlanExecutor 重规划出 parse_and_index_paper，并在副作用工具前触发 interrupt。
    next_state.pending_action = None
    next_state.context = dict(next_state.context or {})
    next_state.context.pop("pending_action", None)
    message_for_user = (
        "这篇论文还没有建立问答索引。请通过 Agent 主流程重新发起论文问答，"
        "系统会展示结构化确认卡片，并通过 resume 恢复解析与回答。"
    )
    result = {
        "status": "failed",
        "arxiv_id": arxiv_id,
        "title": title,
        "original_question": message,
        "qa_question": qa_question,
        "question": qa_question,
        "answer": message_for_user,
        "sources": [],
        "retrieval_debug": None,
        "qa_index_status": qa_status,
        "index_created": False,
        "error": "paper_index_missing_requires_plan_executor_confirmation",
    }
    next_state.paper_qa_result = result
    next_state.answer = message_for_user
    next_state.next_actions = [
        "重新通过 Agent 主流程发起论文问答",
        "也可以直接提供 arXiv ID 后重试",
    ]
    next_state.debug["confirmation_state"] = "delegated_to_plan_executor"
    next_state.debug["paper_qa_answer_status"] = "index_missing"
    return _append_step(
        next_state,
        step="handle_paper_reading_request",
        status="failed",
        action="处理论文阅读请求",
        inputs={"message": message, "intent": intent, "context": context},
        outputs={
            "paper_qa_result": result,
            "qa_index_status": qa_status,
            "qa_question": qa_question,
        },
        error=result["error"],
    )


__all__ = ["handle_paper_reading_request"]
