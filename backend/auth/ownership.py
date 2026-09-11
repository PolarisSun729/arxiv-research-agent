"""需要查询业务对象的所有权校验，在模型、检索和状态恢复之前完成。"""

from auth.context import bind_user_id, current_auth
from auth.errors import AuthError
from auth.permissions import RESEARCH_ROLES


def prepare_agent_request(payload, session_store) -> None:
    context = current_auth.get()
    if context is None:
        return
    context.require_roles(RESEARCH_ROLES)
    payload.user_id = bind_user_id(payload.user_id)
    if payload.session_id:
        session = session_store.get_agent_session(payload.session_id, user_id=context.user_id)
        if not session:
            # JWT 新会话只能由服务器分配，不能认领客户端猜中的旧 checkpoint/thread_id。
            raise AuthError("private_resource_not_found")
    else:
        session = session_store.create_or_get_agent_session(user_id=context.user_id)
        if not session:
            raise AuthError("security_storage_unavailable", retry_after=5)
        payload.session_id = session["session_id"]


def require_paper_session(arxiv_id: str, session_id: str | None, *, user_id: str | None = None, session_store=None) -> dict | None:
    context = current_auth.get()
    if context is None or not session_id:
        return None
    actor_id = bind_user_id(user_id)
    if session_store is None:
        from dependencies import get_paper_chat_session_store
        session_store = get_paper_chat_session_store()
    session = session_store.get_paper_chat_session(session_id, user_id=actor_id)
    if not session or session.get("arxiv_id") != arxiv_id:
        raise AuthError("private_resource_not_found")
    return session


def prepare_paper_qa(arxiv_id: str, payload) -> None:
    if current_auth.get() is None:
        return
    payload.user_id = bind_user_id(payload.user_id)
    require_paper_session(arxiv_id, payload.session_id, user_id=payload.user_id)


def validate_note_source(arxiv_id: str, payload, message_store) -> str | None:
    """来源必须同属当前用户及论文，不能创建指向他人消息/turn 的悬挂关联。"""
    if current_auth.get() is None:
        return payload.source_message_id
    user_id = bind_user_id(payload.user_id)
    require_paper_session(arxiv_id, payload.session_id, user_id=user_id)
    linked = None
    if payload.source_message_id:
        linked = message_store.get_paper_chat_message(payload.source_message_id, user_id=user_id)
    elif payload.source_turn_id:
        if not payload.session_id:
            raise AuthError("private_resource_not_found")
        linked = message_store.get_paper_chat_message_by_turn(payload.session_id, payload.source_turn_id, role="assistant", user_id=user_id)
    if payload.source_message_id or payload.source_turn_id:
        if (not linked or (payload.session_id and linked.get("session_id") != payload.session_id)
                or (payload.source_turn_id and linked.get("turn_id") != payload.source_turn_id)):
            raise AuthError("private_resource_not_found")
        require_paper_session(arxiv_id, linked["session_id"], user_id=user_id)
    return linked["message_id"] if linked else None
