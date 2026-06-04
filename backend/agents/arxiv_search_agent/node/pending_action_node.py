"""处理论文解析确认类的挂起动作。

当前模块专门负责“是否要现在解析 PDF 并建立 QA 索引”这类二次确认流程：
1. 从 state 中恢复待确认任务；
2. 判断用户是在确认、取消、提出新请求还是语义不清；
3. 在确认后继续执行建索引与论文问答；
4. 把执行结果、失败阶段和调试信息统一写回 AgentState。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

_BACKEND_DIR = str(Path(__file__).resolve().parents[3])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

try:  # pragma: no cover
    from dependencies import get_generation_service
except ModuleNotFoundError:  # pragma: no cover
    from backend.dependencies import get_generation_service

from ..schemas import ToolCallRequest
from ..state import AgentState
from ..utils.result_utils import _extract_exception_detail, _extract_exception_stage
from ..utils.state_utils import _append_step, _coerce_state
from ..utils.text_utils import _extract_json_object, _matches_any, _normalize_text
from .tool_node import execute_tool

logger = logging.getLogger(__name__)


def _has_ready_qa_index(payload: Mapping[str, Any]) -> bool:
    normalized_status = str(payload.get("status") or "").strip().lower()
    return bool(payload.get("has_index")) or normalized_status in {"indexed", "success"}


def _get_plan_step_id(state: AgentState, step_type: str) -> Optional[str]:
    for step in list(state.execution_plan or []):
        if str(getattr(step, "step_type", "") or "").strip() == step_type:
            step_id = str(getattr(step, "step_id", "") or "").strip()
            return step_id or None
    return None


def _resolve_generation_service_instance(generation_service: Optional[Any] = None) -> Optional[Any]:
    """优先复用外部传入的生成服务，否则尝试从依赖注入容器获取实例。
    
    分支行为：
    - 传入 generation_service 时直接复用，避免重复初始化；
    - 未传入时尝试从容器获取；
    - 获取失败时返回 None，让上层决定是否退化为规则判断。
    """
    if generation_service is not None:
        return generation_service
    try:
        return get_generation_service()
    except Exception:
        return None


def _normalize_confirmation_decision(value: Any) -> str:
    """把不同来源的确认标签归一化为统一决策枚举。
    
    这个函数把规则、LLM 或外部调用方可能产出的 yes/no/accept/reject/new_request 等标签，
    统一折叠成 confirm、reject、unrelated、unclear 四类，方便后续流程只处理固定分支。
    """
    decision = _normalize_text(str(value or "")).lower()
    if decision in {"confirm", "confirmed", "accept", "yes", "ok", "continue"}:
        return "confirm"
    if decision in {"reject", "rejected", "cancel", "cancelled", "no", "rejecting"}:
        return "reject"
    if decision in {"unrelated", "other", "new_request"}:
        return "unrelated"
    return "unclear"


def _fast_path_pending_action_decision(message: str) -> Optional[str]:
    """用显式词表和正则快速识别确认/取消回复。
    
    主要步骤：
    1. 先处理空文本；
    2. 再用 confirm_tokens / reject_tokens 覆盖最常见的短句回复；
    3. 对更自然的长句补充正则匹配；
    4. 若仍无法判断则返回 None，交给更昂贵的 LLM 分类。
    
    输出：返回 confirm / reject / None。
    """
    text = _normalize_text(message)
    if not text:
        return None

    confirm_tokens = {
        "确认",
        "是",
        "好",
        "好的",
        "可以",
        "继续",
        "解析",
        "开始解析",
        "需要解析",
        "帮我解析",
        "ok",
        "yes",
        "continue",
        "go ahead",
        "继续解析",
    }
    reject_tokens = {
        "不用",
        "取消",
        "不解析",
        "先不",
        "暂时不用",
        "算了",
        "不要",
        "先不看",
        "no",
        "cancel",
        "别解析",
    }

    # 先做集合命中判断，覆盖最常见的单词/短句确认。
    if text in confirm_tokens:
        return "confirm"
    if text in reject_tokens:
        return "reject"

    lowered = text.lower()
    if lowered in confirm_tokens:
        return "confirm"
    if lowered in reject_tokens:
        return "reject"

    reject_patterns = (
        r"先不解析",
        r"不要解析",
        r"不需要解析",
        r"别解析",
        r"别建索引",
        r"不要建索引",
        r"不用建索引",
        r"取消.*(?:任务|解析|索引)",
    )
    confirm_patterns = (
        r"解析\s*pdf",
        r"开始解析",
        r"继续解析",
        r"继续处理",
        r"开始处理",
        r"解析并回答",
        r"创建.*全文检索索引",
        r"创建.*索引",
        r"建立.*索引",
        r"构建.*索引",
        r"可以解析",
        r"确认解析",
    )

    # 对更长的自然语言表达，再补一层正则模式判断。
    if _matches_any(text, reject_patterns) or _matches_any(lowered, reject_patterns):
        return "reject"
    if _matches_any(text, confirm_patterns) or _matches_any(lowered, confirm_patterns):
        return "confirm"
    return None


def _recover_pending_action(next_state: AgentState) -> Dict[str, Any]:
    """从 state 的多个可能来源中恢复待确认动作。
    
    主要分支：
    1. 优先读取显式保存的 pending_action；
    2. 若缺失，则尝试从 waiting_confirmation 状态的 paper_qa_result 反推出等价任务；
    3. 恢复成功后补齐 arxiv_id、title、original_question、qa_question 等关键字段。
    """
    # 优先使用显式挂在 state/context 上的 pending_action；只有缺失时才尝试从结果反推。
    pending_action = dict(next_state.pending_action or (next_state.context or {}).get("pending_action") or {})
    if not pending_action and isinstance(next_state.paper_qa_result, Mapping):
        paper_qa_result = dict(next_state.paper_qa_result or {})
        if str(paper_qa_result.get("status") or "").strip() == "waiting_confirmation":
            pending_action = {
                "type": "parse_then_qa",
                "status": "waiting_confirmation",
                "arxiv_id": paper_qa_result.get("arxiv_id"),
                "title": paper_qa_result.get("title"),
                "original_question": paper_qa_result.get("original_question") or paper_qa_result.get("question"),
                "qa_question": paper_qa_result.get("qa_question") or paper_qa_result.get("question"),
                "loading_method": (next_state.context or {}).get("loading_method") if isinstance(next_state.context, dict) else None,
            }
            logger.info(
                "arxiv_agent recovered pending_action from paper_qa_result: arxiv_id=%s title=%s",
                pending_action.get("arxiv_id") or "none",
                pending_action.get("title") or "none",
            )
    return pending_action


def classify_pending_action_confirmation(
    state: Union[AgentState, Mapping[str, Any]],
    generation_service: Optional[Any] = None,
) -> AgentState:
    """判断用户当前输入是确认、取消、无关新请求还是无法判断。
    
    主流程顺序固定为：
    1. 恢复 pending_action；
    2. 用规则 fast path 识别明确确认/取消；
    3. 对“解析 / 索引”等关键词做低成本兜底；
    4. 仅在前面都无法判断时才调用 LLM 细分 unrelated / unclear。
    
    输出：返回写入 confirmation_decision、confidence、reason 和 step trace 的 AgentState。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    pending_action = _recover_pending_action(next_state)
    message = _normalize_text(next_state.message or "")
    debug = dict(next_state.debug or {})

    debug["pending_action"] = pending_action
    debug["pending_action_confirmation_model"] = "qwen3.6-flash"

    # 没有待确认任务时，不应该把普通消息误判成确认结果，直接以 skipped 返回。
    if not pending_action or str(pending_action.get("type") or "").strip() != "parse_then_qa":
        debug["pending_action_decision"] = "unclear"
        debug["confirmation_confidence"] = 0.0
        debug["confirmation_reason"] = "当前没有待确认的论文解析任务"
        next_state.debug = debug
        return _append_step(
            next_state,
            step="classify_pending_action_confirmation",
            status="skipped",
            action="判断用户是否确认解析任务",
            inputs={"message": message, "pending_action": pending_action},
            outputs={"decision": "unclear", "reason": debug["confirmation_reason"]},
        )

    # 先走规则层；只有规则无法确定时，才考虑更昂贵的 LLM 分类。
    decision = _fast_path_pending_action_decision(message)
    confidence = 0.99 if decision in {"confirm", "reject"} else 0.0
    reason = ""
    # 用户经常会回复“建索引吧”“继续解析”，这里用关键词兜底覆盖这类半结构化输入。
    if decision is None and any(keyword in message for keyword in ("解析", "索引", "全文检索", "问答索引")):
        decision = "confirm"
        confidence = 0.95
        reason = "关键词兜底命中明确的继续执行意图"
    reason = reason or ("fast path 命中明确确认/取消表达" if decision in {"confirm", "reject"} else "")

    # 只有在规则和兜底都无法判断时，才把上下文交给 LLM 细分 unrelated / unclear 等状态。
    # 只有低成本规则无法判断时，才值得为确认语义付出一次 LLM 调用成本。
    if decision is None:
        service = _resolve_generation_service_instance(generation_service)
        if service is not None:
            # 只有简单规则无法判断时，才把完整上下文交给 LLM，尽量把高成本推断放在最后一层。
            prompt = (
                "你正在判断用户是否要继续执行一个待确认的论文解析任务。\n"
                "请只输出严格 JSON，不要输出解释性文本。\n\n"
                "JSON schema:\n"
                '{\n'
                '  "decision": "confirm | reject | unrelated | unclear",\n'
                '  "confidence": 0.0,\n'
                '  "reason": "简短原因"\n'
                '}\n\n'
                "待确认任务信息：\n"
                f"- 论文标题: {pending_action.get('title') or ''}\n"
                f"- arXiv ID: {pending_action.get('arxiv_id') or ''}\n"
                f"- 原始问题: {pending_action.get('original_question') or ''}\n"
                f"- qa_question: {pending_action.get('qa_question') or ''}\n"
                f"- 当前用户输入: {message}\n\n"
                "判定规则：\n"
                "- confirm: 用户明确表示继续解析或同意执行\n"
                "- reject: 用户明确取消或拒绝执行\n"
                "- unrelated: 用户提出了新的独立请求，而不是确认/取消\n"
                "- unclear: 无法判断用户是否确认\n"
            )
            try:
                raw_output = service.complete_with_qwen(
                    prompt,
                    task_type="pending_action_confirmation",
                    enable_thinking=False,
                )
                parsed = _extract_json_object(raw_output) or {}
                decision = _normalize_confirmation_decision(parsed.get("decision"))
                confidence = float(parsed.get("confidence") or 0.0)
                reason = _normalize_text(str(parsed.get("reason") or "")) or "LLM 分类结果未返回原因"
            except Exception as exc:
                decision = "unclear"
                confidence = 0.0
                reason = f"确认判断失败: {exc}"
        else:
            decision = "unclear"
            confidence = 0.0
            reason = "缺少生成服务，无法进行确认判断"

    debug["pending_action_decision"] = decision
    debug["confirmation_decision"] = decision
    debug["pending_action_confidence"] = confidence
    debug["confirmation_confidence"] = confidence
    debug["pending_action_reason"] = reason
    debug["confirmation_reason"] = reason
    next_state.debug = debug
    if not next_state.plan:
        next_state.plan = [
            "判断用户是否确认解析任务",
            "必要时创建论文 QA 索引",
            "继续回答原始问题",
        ]

    return _append_step(
        next_state,
        step="classify_pending_action_confirmation",
        status="success",
        action="判断用户是否确认解析任务",
        inputs={"message": message, "pending_action": pending_action},
        outputs={
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
            "pending_action": pending_action,
        },
    )


def handle_pending_action_confirmation(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """在用户确认后继续执行建索引与论文问答流程。
    
    主要分支包括：
    1. 缺少 arXiv ID 等关键字段时直接失败并标记失败阶段；
    2. 成功创建或复用索引后执行论文问答；
    3. 任一步异常都写入 error_stage / error_type / error_detail，便于排查。
    
    输出：返回更新后的 AgentState，包含 paper_qa_result、answer、debug 和 tool_calls。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)
    pending_action = _recover_pending_action(next_state)

    arxiv_id = str(pending_action.get("arxiv_id") or "").strip()
    title = str(pending_action.get("title") or "").strip()
    original_question = _normalize_text(str(pending_action.get("original_question") or next_state.message or ""))
    qa_question = _normalize_text(str(pending_action.get("qa_question") or original_question))
    loading_method = str(pending_action.get("loading_method") or "docling").strip() or "docling"
    debug = dict(next_state.debug or {})
    debug["paper_qa_target"] = {"arxiv_id": arxiv_id, "title": title}
    debug["qa_question"] = qa_question
    debug["waiting_user_confirmation"] = False
    debug["pending_action"] = pending_action

    # arXiv ID 是后续建索引和问答的最小必要条件，缺失时立即终止并返回明确错误。
    if not arxiv_id:
        result = {
            "status": "failed",
            "arxiv_id": None,
            "title": title or None,
            "original_question": original_question,
            "qa_question": qa_question or None,
            "question": qa_question or None,
            "answer": "",
            "sources": [],
            "retrieval_debug": None,
            "qa_index_status": None,
            "index_created": False,
            "error": "待确认任务缺少 arXiv ID",
            "error_stage": "pending_action_validation",
            "failed_stage": "pending_action_validation",
            "error_type": "ValueError",
        }
        next_state.paper_qa_result = result
        next_state.answer = result["error"]
        next_state.debug = {
            **debug,
            "qa_index_status": None,
            "index_created": False,
            "waiting_user_confirmation": False,
            "paper_qa_answer_status": "pending_action_validation_failed",
            "error_stage": result["error_stage"],
            "error_type": result["error_type"],
            "error_detail": result["error"],
        }
        return _append_step(
            next_state,
            step="handle_pending_action_confirmation",
            status="failed",
            action="创建并执行论文问答任务",
            inputs={"pending_action": pending_action},
            outputs={"paper_qa_result": result, "qa_index_status": None, "index_created": False},
            error=result["error"],
        )

    try:
        # 成功路径固定为：建索引 -> 读取最新索引状态 -> 基于新索引回答原问题。
        next_state.tool_call_request = ToolCallRequest(
            tool_name="build_paper_qa_index",
            arguments={"arxiv_id": arxiv_id, "loading_method": loading_method},
            reason="用户已确认解析论文，需要先构建 QA 索引",
            expected_result="返回索引构建结果和可用状态",
            plan_step_id=_get_plan_step_id(next_state, "confirmation_gate"),
        )
        next_state = execute_tool(next_state)
        build_tool_result = dict(next_state.tool_result or {})
        index_result = dict(build_tool_result.get("data") or {}) if isinstance(build_tool_result.get("data"), Mapping) else {}
        if not bool(build_tool_result.get("ok")):
            raise RuntimeError(str(((build_tool_result.get("error") or {}).get("message")) or build_tool_result.get("summary") or "构建 QA 索引失败"))

        qa_status = dict(index_result or {})
        index_created = _has_ready_qa_index(qa_status)
        debug["qa_index_status"] = qa_status
        debug["index_created"] = index_created
        debug["paper_qa_answer_status"] = "index_built"

        next_state.tool_call_request = ToolCallRequest(
            tool_name="answer_paper_question",
            arguments={"arxiv_id": arxiv_id, "question": qa_question},
            reason="论文索引已就绪，继续回答用户原始问题",
            expected_result="返回论文问题答案、来源片段和检索调试信息",
            plan_step_id=_get_plan_step_id(next_state, "paper_response"),
        )
        next_state = execute_tool(next_state)
        answer_tool_result = dict(next_state.tool_result or {})
        answer_result = dict(answer_tool_result.get("data") or {}) if isinstance(answer_tool_result.get("data"), Mapping) else {}
        if not bool(answer_tool_result.get("ok")):
            raise RuntimeError(str(((answer_tool_result.get("error") or {}).get("message")) or answer_tool_result.get("summary") or "论文问答执行失败"))

        result = {
            "status": "success",
            "arxiv_id": arxiv_id,
            "title": title,
            "original_question": original_question,
            "qa_question": qa_question,
            "question": qa_question,
            "answer": answer_result.get("answer", ""),
            "sources": answer_result.get("sources", []),
            "retrieval_debug": answer_result.get("retrieval_debug"),
            "qa_index_status": qa_status,
            "index_created": index_created,
            "error": None,
        }
        next_state.paper_qa_result = result
        next_state.answer = result["answer"]
        next_state.pending_action = None
        next_state.context = dict(next_state.context or {})
        next_state.context.pop("pending_action", None)
        next_state.next_actions = [
            "继续追问这篇论文的其他细节",
            "切换到其他论文继续阅读",
        ]
        if isinstance(answer_result, Mapping) and answer_result.get("papers"):
            next_state.papers = list(answer_result.get("papers") or [])
        next_state.debug = {
            **debug,
            "qa_index_status": qa_status,
            "index_created": index_created,
            "qa_question": qa_question,
            "waiting_user_confirmation": False,
            "paper_qa_answer_status": "answered",
        }
        return _append_step(
            next_state,
            step="handle_pending_action_confirmation",
            status="success",
            action="创建并执行论文问答任务",
            inputs={"pending_action": pending_action},
            outputs={"paper_qa_result": result, "qa_index_status": qa_status, "index_created": index_created},
        )
    except Exception as exc:
        # 执行失败时除了错误文案，还会把 stage/type/detail 记录到 debug 和 tool_calls。
        error_stage = _extract_exception_stage(exc, "handle_pending_action_confirmation")
        error_detail = _extract_exception_detail(exc)
        result = {
            "status": "failed",
            "arxiv_id": arxiv_id,
            "title": title or None,
            "original_question": original_question,
            "qa_question": qa_question or None,
            "question": qa_question or None,
            "answer": "",
            "sources": [],
            "retrieval_debug": None,
            "qa_index_status": None,
            "index_created": False,
            "error": str(error_detail or exc),
            "error_stage": error_stage,
            "failed_stage": error_stage,
            "error_type": type(exc).__name__,
        }
        next_state.paper_qa_result = result
        next_state.answer = result["error"]
        next_state.pending_action = None
        next_state.context = dict(next_state.context or {})
        next_state.context.pop("pending_action", None)
        next_state.debug = {
            **debug,
            "qa_index_status": None,
            "index_created": False,
            "waiting_user_confirmation": False,
            "paper_qa_answer_status": "failed",
            "error_stage": error_stage,
            "error_type": type(exc).__name__,
            "error_detail": result["error"],
        }
        return _append_step(
            next_state,
            step="handle_pending_action_confirmation",
            status="failed",
            action="创建并执行论文问答任务",
            inputs={"pending_action": pending_action},
            outputs={"paper_qa_result": result},
            error=result["error"],
        )


__all__ = [
    "classify_pending_action_confirmation",
    "handle_pending_action_confirmation",
]
