"""处理用户对论文的显式偏好动作。

这个节点负责把自然语言里的“喜欢 / 不喜欢 / 取消偏好”落到推荐系统：
1. 解析动作类型与撤销范围；
2. 结合上下文定位目标论文；
3. 调用推荐服务写入或删除偏好记录；
4. 把结果同步回 AgentState，供推荐与最终回复节点使用。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

_BACKEND_DIR = str(Path(__file__).resolve().parents[3])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from .intent_support import _dedupe_preserve_order
from ..schemas import ToolCallRequest
from ..state import AgentState
from ..utils.paper_reference_resolver import _resolve_paper_reference
from ..utils.result_utils import _extract_error_message, _result_mapping, _result_ok
from ..utils.state_utils import _append_step, _coerce_state, _get_execution_plan_step
from ..utils.text_utils import _matches_any, _normalize_text
from .tool_node import execute_tool


def _parse_preference_action(message: str) -> Optional[Dict[str, str]]:
    """从用户输入中提取偏好动作及其移除范围。
    
    主要步骤：
    1. 先归一化文本并处理空输入；
    2. 识别 remove 场景时进一步区分只撤销 liked、只撤销 disliked，还是两边都删；
    3. 判断顺序固定为 remove -> dislike -> like，避免“取消喜欢”误命中“喜欢”。
    
    输出：返回包含 action 和 remove_scope 的字典；无法识别时返回 None。
    """
    text = _normalize_text(message)
    lowered = text.lower()
    if not text:
        return None

    remove_patterns = (
        r"取消.*喜欢",
        r"取消.*不喜欢",
        r"取消.*标记",
        r"移除.*标记",
        r"撤销.*喜欢",
        r"撤销.*不喜欢",
        r"撤销.*标记",
        r"\bunlike\b",
        r"remove\s+like",
        r"remove\s+dislike",
    )
    dislike_patterns = (
        r"不喜欢",
        r"不感兴趣",
        r"标记.*不喜欢",
        r"标记.*不感兴趣",
        r"对.*不感兴趣",
    )
    like_patterns = (
        r"喜欢",
        r"感兴趣",
        r"标记.*喜欢",
        r"标记.*感兴趣",
        r"对.*感兴趣",
    )

    # 移除偏好时需要知道用户想撤销哪一类标记，默认保守地尝试两边都删除。
    remove_scope = "both"
    if _matches_any(text, (r"取消.*喜欢", r"撤销.*喜欢")):
        remove_scope = "liked"
    elif _matches_any(text, (r"取消.*不喜欢", r"撤销.*不喜欢")):
        remove_scope = "disliked"

    # 先判断 remove，再判断 dislike / like，避免“取消喜欢”被误识别成“喜欢”。
    if _matches_any(text, remove_patterns):
        return {"action": "remove", "remove_scope": remove_scope}
    if _matches_any(text, dislike_patterns) or "dislike" in lowered:
        return {"action": "dislike", "remove_scope": "none"}
    if _matches_any(text, like_patterns) or "like" in lowered:
        return {"action": "like", "remove_scope": "none"}
    return None


def _get_plan_step_id(state: AgentState, step_type: str) -> Optional[str]:
    accepted_step_types = [str(step_type or "").strip()]
    if step_type == "preference_mutation":
        accepted_step_types.append("preference_update")
    for accepted_step_type in accepted_step_types:
        step = _get_execution_plan_step(state, step_type=accepted_step_type)
        if step is not None:
            step_id = str(getattr(step, "step_id", "") or "").strip()
            return step_id or None
    return None


def apply_preference_action(state: Union[AgentState, Mapping[str, Any]]) -> AgentState:
    """执行论文偏好更新，并把结果写回 AgentState。
    
    主流程分为四步：
    1. 校验当前 intent 是否为 preference_action；
    2. 解析动作并定位目标论文；
    3. 根据 like / dislike / remove 分支调用不同服务逻辑；
    4. 统一记录 preference_action_result、tool_calls、warnings 和 step trace。
    
    输出：无论成功还是失败，都会返回包含可复盘执行痕迹的 AgentState。
    """
    current_state = _coerce_state(state)
    next_state = current_state.model_copy(deep=True)

    if next_state.intent != "preference_action":
        return _append_step(
            next_state,
            step="preference_action_execution",
            status="skipped",
            action="执行论文偏好动作",
            inputs={"intent": next_state.intent},
            outputs={"reason": "当前请求不是 preference_action"},
        )

    # 先把偏好动作和论文引用线索拆开；线索未落地到最终论文前不能写入持久化偏好。
    message = _normalize_text(next_state.message or "")
    parsed_action = _parse_preference_action(message)
    resolution = _resolve_paper_reference(message, next_state.context or {})
    user_id = str(next_state.user_id or "default").strip() or "default"
    tool_args: Dict[str, Any] = {"user_id": user_id, "message": message}
    tool_name = "apply_preference_action"
    preference_result: Dict[str, Any]
    paper_payload = resolution.get("paper")
    arxiv_id = str(resolution.get("arxiv_id") or "").strip()
    paper_title = str(resolution.get("title") or (paper_payload or {}).get("title") or "").strip()

    if not next_state.plan:
        next_state.plan = [
            "解析偏好动作",
            "解析目标论文",
            "更新用户偏好",
        ]
    debug = dict(next_state.debug or {})
    debug["preference_action"] = parsed_action
    debug["preference_target"] = {
        "arxiv_id": arxiv_id or None,
        "title": paper_title or None,
        "resolution_status": resolution.get("status"),
        "final_target_resolved": bool(resolution.get("final_target_resolved")),
    }
    debug["preference_action_status"] = "pending"
    next_state.debug = debug

    # 第一类失败：动作本身就无法识别，此时不再继续做论文解析和服务调用。
    if parsed_action is None:
        preference_result = {
            "status": "failed",
            "action": "remove",
            "label": "none",
            "arxiv_id": None,
            "title": None,
            "message": "我没有识别到明确的偏好动作，请使用“喜欢 / 不喜欢 / 取消标记”",
            "paper": None,
            "error": "unsupported preference action",
        }
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [preference_result["message"]])
    # 第二类失败：动作明确，但目标论文不明确，避免把偏好写到错误对象上。
    # 偏好写入有持久化副作用；兼容节点必须拒绝只有引用线索、没有 Target Resolver 最终判定的结果。
    elif resolution.get("status") != "success" or not resolution.get("final_target_resolved") or not arxiv_id:
        hint_only_reason = (
            "已提取论文引用线索，但尚未结合上下文解析成最终论文。"
            if resolution.get("status") == "hint_extracted"
            else "无法解析目标论文，请先搜索论文或直接提供 arXiv ID"
        )
        preference_result = {
            "status": "failed",
            "action": parsed_action["action"],
            "label": "none",
            "arxiv_id": resolution.get("arxiv_id"),
            "title": resolution.get("title"),
            "message": str(resolution.get("reason") or hint_only_reason),
            "paper": paper_payload,
            "error": str(resolution.get("reason") or hint_only_reason),
        }
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [preference_result["message"]])
    else:
        # 只有动作和目标论文都明确时，才真正写入或删除偏好记录。
        try:
            # like / dislike 都走统一的偏好记录接口，只是 liked 标志不同。
            if parsed_action["action"] in {"like", "dislike"}:
                tool_name = "record_paper_preference"
                liked = parsed_action["action"] == "like"
                tool_args = {"user_id": user_id, "arxiv_id": arxiv_id, "liked": liked, "paper": paper_payload}
                next_state.tool_call_request = ToolCallRequest(
                    tool_name=tool_name,
                    arguments=dict(tool_args),
                    reason="把用户对目标论文的显式偏好写入推荐系统",
                    expected_result="返回偏好写入结果以及归一化后的论文信息",
                    plan_step_id=_get_plan_step_id(next_state, "preference_mutation"),
                )
                next_state = execute_tool(next_state)
                if not _result_ok(next_state.tool_result or {}):
                    raise RuntimeError(_extract_error_message(next_state.tool_result or {}) or "偏好记录失败")
                service_result = dict(_result_mapping(next_state.tool_result or {}, "data") or {})
                if next_state.tool_observations:
                    next_state.tool_observations[-1].is_sufficient = True
                    next_state.tool_observations[-1].next_action_hint = None
                preference_result = {
                    "status": "success",
                    "action": parsed_action["action"],
                    "label": "liked" if liked else "disliked",
                    "arxiv_id": arxiv_id,
                    "title": paper_title,
                    "message": str(service_result.get("message") or "偏好已更新"),
                    "paper": service_result.get("paper") or paper_payload,
                    "error": None,
                }
            else:
                tool_name = "remove_paper_preference"
                remove_scope = str(parsed_action.get("remove_scope") or "both")
                tool_args = {"user_id": user_id, "arxiv_id": arxiv_id, "remove_scope": remove_scope}
                next_state.tool_call_request = ToolCallRequest(
                    tool_name=tool_name,
                    arguments=dict(tool_args),
                    reason="移除用户对目标论文的既有偏好标记",
                    expected_result="返回偏好移除结果以及移除范围",
                    plan_step_id=_get_plan_step_id(next_state, "preference_mutation"),
                )
                next_state = execute_tool(next_state)
                service_result = dict(_result_mapping(next_state.tool_result or {}, "data") or {})
                removed_liked = bool(service_result.get("removed_liked"))
                removed_disliked = bool(service_result.get("removed_disliked"))
                if next_state.tool_observations:
                    next_state.tool_observations[-1].is_sufficient = bool(_result_ok(next_state.tool_result or {}))
                    next_state.tool_observations[-1].next_action_hint = None if _result_ok(next_state.tool_result or {}) else "check_existing_preference_or_retry_mutation"
                preference_result = {
                    "status": "success" if _result_ok(next_state.tool_result or {}) else "failed",
                    "action": "remove",
                    "label": "none",
                    "arxiv_id": arxiv_id,
                    "title": paper_title,
                    "message": str(service_result.get("message") or ("已取消偏好标记" if (removed_liked or removed_disliked) else "未找到可取消的偏好标记")),
                    "paper": paper_payload,
                    "error": None if _result_ok(next_state.tool_result or {}) else _extract_error_message(next_state.tool_result or {}),
                }
                if not _result_ok(next_state.tool_result or {}):
                    next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [preference_result["message"]])
        except Exception as exc:
            preference_result = {
                "status": "failed",
                "action": parsed_action["action"],
                "label": "none",
                "arxiv_id": arxiv_id,
                "title": paper_title,
                "message": f"偏好动作执行失败：{exc}",
                "paper": paper_payload,
                "error": str(exc),
            }
            next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [str(exc)])

    next_state.preference_action_result = preference_result
    next_state.debug = {
        **dict(next_state.debug or {}),
        "preference_action_result": dict(preference_result),
        "preference_action_status": preference_result.get("status"),
    }
    if not next_state.next_actions:
        next_state.next_actions = [
            "继续对其他论文执行喜欢、不喜欢或取消偏好动作",
            "也可以继续搜索、查看推荐或打开论文详情",
        ]

    # 即使当前动作失败，也保留统一的工具调用记录，方便前端或日志层复盘原因。
    next_state.tool_name = tool_name
    next_state.tool_args = tool_args
    if next_state.tool_result is None:
        next_state.tool_result = dict(preference_result)

    return _append_step(
        next_state,
        step="preference_action_execution",
        status="success" if preference_result["status"] == "success" else "failed",
        action="执行论文偏好动作",
        inputs={"intent": next_state.intent, "message": next_state.message, "context": next_state.context},
        outputs={
            "preference_action_result": dict(preference_result),
            "plan": list(next_state.plan),
            "next_actions": list(next_state.next_actions),
        },
        error=None if preference_result["status"] == "success" else str(preference_result.get("error") or preference_result.get("message") or "preference action failed"),
    )


__all__ = ["apply_preference_action"]
