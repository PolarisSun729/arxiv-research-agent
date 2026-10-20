"""论文偏好动作（标记感兴趣 / 不感兴趣 / 取消标记）的结构与归一化。

方向由意图识别的 LLM 读整句话后一并给出（preference_operation），这里只校验取值，不再从原话里猜。
偏好写入是持久化副作用，方向判反比不写更糟；取值不合法时返回 None，由调用方报错或澄清，
不能默认当成"感兴趣"。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

PREFERENCE_ACTIONS = ("like", "dislike", "remove")
PREFERENCE_REMOVE_SCOPES = ("liked", "disliked", "both")


def normalize_preference_operation(raw: Any) -> Optional[Dict[str, str]]:
    """返回 {"action": like|dislike|remove, "remove_scope": liked|disliked|both|none}；取值不合法返回 None。"""
    if not isinstance(raw, Mapping):
        return None
    action = str(raw.get("action") or "").strip().lower()
    if action not in PREFERENCE_ACTIONS:
        return None
    if action != "remove":
        return {"action": action, "remove_scope": "none"}
    remove_scope = str(raw.get("remove_scope") or "").strip().lower()
    # 只说"取消标记"时不知道撤哪一类，两边都删最保守。
    return {"action": "remove", "remove_scope": remove_scope if remove_scope in PREFERENCE_REMOVE_SCOPES else "both"}


__all__ = ["PREFERENCE_ACTIONS", "PREFERENCE_REMOVE_SCOPES", "normalize_preference_operation"]
