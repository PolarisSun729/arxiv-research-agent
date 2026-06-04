"""处理用户对论文的显式偏好动作。

这个节点负责把自然语言里的“喜欢 / 不喜欢 / 取消标记”落到推荐系统：
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

try:  # pragma: no cover
    from dependencies import get_recommendation_service
except ModuleNotFoundError:  # pragma: no cover
    from backend.dependencies import get_recommendation_service

from .intent_support import _dedupe_preserve_order
from ..schemas import AgentToolCall
from ..state import AgentState
from ..utils.paper_reference_resolver import _resolve_paper_reference
from ..utils.state_utils import _append_step, _coerce_state
from ..utils.text_utils import _matches_any, _normalize_text


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
        r"收藏",
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

    # 先把输入和引用论文解析成结构化对象，后续所有分支都复用这些结果。
    message = _normalize_text(next_state.message or "")
    parsed_action = _parse_preference_action(message)
    resolution = _resolve_paper_reference(message, next_state.context or {})
    user_id = str(next_state.user_id or "default").strip() or "default"
    preference_service = get_recommendation_service()
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
    elif resolution.get("status") != "success" or not arxiv_id:
        preference_result = {
            "status": "failed",
            "action": parsed_action["action"],
            "label": "none",
            "arxiv_id": resolution.get("arxiv_id"),
            "title": resolution.get("title"),
            "message": str(resolution.get("reason") or "无法解析目标论文，请先搜索论文或直接提供 arXiv ID"),
            "paper": paper_payload,
            "error": str(resolution.get("reason") or "paper reference resolution failed"),
        }
        next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + [preference_result["message"]])
    else:
        # 只有动作和目标论文都明确时，才真正写入或删除偏好记录。
        try:
            # like / dislike 都走统一的偏好记录接口，只是 liked 标志不同。
            if parsed_action["action"] in {"like", "dislike"}:
                tool_name = "record_user_paper_preference"
                liked = parsed_action["action"] == "like"
                service_result = preference_service.record_user_paper_preference(
                    user_id=user_id,
                    arxiv_id=arxiv_id,
                    liked=liked,
                    paper_payload=paper_payload,
                )
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
                tool_name = "remove_user_paper_preference"
                remove_scope = str(parsed_action.get("remove_scope") or "both")
                removed_liked = False
                removed_disliked = False
                remove_errors: List[str] = []
                # 移除逻辑允许分别删除 liked / disliked，兼容“只取消喜欢”这类表达。
                if remove_scope in {"both", "liked"}:
                    removed_liked = bool(preference_service.db_service.remove_liked_paper(user_id=user_id, arxiv_id=arxiv_id))
                if remove_scope in {"both", "disliked"}:
                    removed_disliked = bool(preference_service.db_service.remove_disliked_paper(user_id=user_id, arxiv_id=arxiv_id))
                if not removed_liked and not removed_disliked:
                    remove_errors.append("未找到可移除的喜欢/不喜欢标记")
                preference_result = {
                    "status": "success" if (removed_liked or removed_disliked) else "failed",
                    "action": "remove",
                    "label": "none",
                    "arxiv_id": arxiv_id,
                    "title": paper_title,
                    "message": "已取消偏好标记" if (removed_liked or removed_disliked) else "未找到可取消的偏好标记",
                    "paper": paper_payload,
                    "error": None if (removed_liked or removed_disliked) else "; ".join(remove_errors),
                }
                if not (removed_liked or removed_disliked):
                    next_state.warnings = _dedupe_preserve_order(list(next_state.warnings) + remove_errors)
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
    if not next_state.next_actions:
        next_state.next_actions = [
            "继续对其他论文执行喜欢、不喜欢或收藏动作",
            "也可以继续搜索、查看推荐或打开论文详情",
        ]

    # 即使当前动作失败，也保留统一的工具调用记录，方便前端或日志层复盘原因。
    next_state.tool_name = tool_name
    next_state.tool_args = tool_args
    next_state.tool_result = dict(preference_result)
    next_state.tool_calls = list(next_state.tool_calls or []) + [
        AgentToolCall(
            tool_name=tool_name,
            arguments=tool_args,
            status=preference_result["status"],
            summary=str(preference_result["message"]),
            trace={
                "action": parsed_action["action"] if parsed_action else None,
                "target": resolution.get("target"),
                "arxiv_id": preference_result.get("arxiv_id"),
            },
            error={"message": preference_result["error"]} if preference_result.get("error") else None,
        )
    ]

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
