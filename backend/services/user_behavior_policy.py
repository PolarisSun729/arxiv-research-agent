from __future__ import annotations

from typing import Any

# 显式偏好必须走 like/dislike 专用接口；这里列出明显等价的表达，避免通用行为接口重新承载强偏好语义。
EXPLICIT_PREFERENCE_ACTION_TYPES = {
    "like",
    "liked",
    "dislike",
    "disliked",
    "thumb_up",
    "thumbs_up",
    "thumb_down",
    "thumbs_down",
    "upvote",
    "downvote",
}

# paper-action 只保留弱行为或非权威行为状态，推荐和画像可以按低权重消费这些信号。
WEAK_PAPER_ACTION_TYPES = {
    "favorite",
    "read",
    "later",
    "archived",
    "note_saved",
    "not_interested",
}

PAPER_ACTION_ALIASES = {
    "bookmark": "favorite",
    "bookmarked": "favorite",
    "saved": "later",
    "save_for_later": "later",
    "uninterested": "not_interested",
    "note": "note_saved",
}


def normalize_paper_action_type(action_type: Any) -> str:
    """统一 action_type 的拼写归一化，让路由、服务和数据库共享同一个行为边界。"""
    normalized = str(action_type or "").strip().lower().replace("-", "_")
    return PAPER_ACTION_ALIASES.get(normalized, normalized)


def is_explicit_preference_action(action_type: Any) -> bool:
    return normalize_paper_action_type(action_type) in EXPLICIT_PREFERENCE_ACTION_TYPES


def validate_weak_paper_action_type(action_type: Any) -> str:
    """校验 paper-action 只能表达弱行为；强偏好必须使用专用接口，避免双写成两套权威状态。"""
    normalized = normalize_paper_action_type(action_type)
    if not normalized:
        raise ValueError("action_type is required")
    if normalized in EXPLICIT_PREFERENCE_ACTION_TYPES:
        raise ValueError("explicit_preference_action_not_allowed")
    if normalized not in WEAK_PAPER_ACTION_TYPES:
        raise ValueError("unsupported_paper_action_type")
    return normalized
