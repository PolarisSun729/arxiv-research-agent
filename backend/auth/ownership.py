"""需要查询业务对象的所有权校验，在模型、检索和状态恢复之前完成。

中间件只能校验请求中显式出现的 user_id；像 session_id、message_id 这类业务对象 ID 是否属于当前用户，
必须查询业务存储才能确定，因此放在路由/服务层调用。

约定：
- 没有 JWT 上下文（API Key 模式或可信本地调用）时，各函数不做所有权限制；
- 对象不存在与对象属于他人统一返回 private_resource_not_found，不向调用方透露他人对象是否存在。
"""

from auth.context import bind_user_id, current_auth
from auth.errors import AuthError
from auth.permissions import RESEARCH_ROLES


def prepare_agent_request(payload, session_store) -> None:
    """Agent 对话请求的前置校验，并原地改写 payload 的 user_id 与 session_id。

    - 重新确认当前用户仍拥有研究员及以上角色；
    - payload.user_id 绑定为当前用户；
    - 指定了 session_id：该会话必须属于当前用户；
    - 未指定 session_id：由服务器为当前用户创建（或取得）会话，并回填到 payload。
    """
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
    """确认论文对话会话 session_id 属于当前用户，且确实是论文 arxiv_id 的会话。

    校验通过返回会话记录；没有 JWT 上下文或未提供 session_id 时返回 None（无需校验）。
    session_store 可由调用方注入（便于测试），未注入时使用全局依赖。
    """
    context = current_auth.get()
    if context is None or not session_id:
        return None
    actor_id = bind_user_id(user_id)
    if session_store is None:
        # 延迟导入：只有调用方未注入 store 时才加载全局依赖。
        from dependencies import get_paper_chat_session_store
        session_store = get_paper_chat_session_store()
    session = session_store.get_paper_chat_session(session_id, user_id=actor_id)
    # 会话属于本人还不够，还必须属于同一篇论文，防止把 A 论文的会话挂到 B 论文的问答上。
    if not session or session.get("arxiv_id") != arxiv_id:
        raise AuthError("private_resource_not_found")
    return session


def prepare_paper_qa(arxiv_id: str, payload) -> None:
    """论文问答请求的前置校验：绑定 user_id，并确认所用会话（如有）属于本人和该论文。"""
    if current_auth.get() is None:
        return
    payload.user_id = bind_user_id(payload.user_id)
    require_paper_session(arxiv_id, payload.session_id, user_id=payload.user_id)


def validate_note_source(arxiv_id: str, payload, message_store) -> str | None:
    """来源必须同属当前用户及论文，不能创建指向他人消息/turn 的悬挂关联。

    笔记可以通过 source_message_id 或 source_turn_id（二选一，前者优先）关联到某条问答消息。
    返回经过校验、应当保存的来源 message_id；没有来源时返回 None。
    无 JWT 上下文时原样返回 payload.source_message_id，不做校验。
    """
    if current_auth.get() is None:
        return payload.source_message_id
    user_id = bind_user_id(payload.user_id)
    # 笔记本身所在的会话（如有）必须属于本人和该论文。
    require_paper_session(arxiv_id, payload.session_id, user_id=user_id)
    linked = None
    if payload.source_message_id:
        linked = message_store.get_paper_chat_message(payload.source_message_id, user_id=user_id)
    elif payload.source_turn_id:
        # turn_id 只在会话内唯一，必须结合 session_id 才能定位到消息。
        if not payload.session_id:
            raise AuthError("private_resource_not_found")
        linked = message_store.get_paper_chat_message_by_turn(payload.session_id, payload.source_turn_id, role="assistant", user_id=user_id)
    if payload.source_message_id or payload.source_turn_id:
        # 找到的消息必须与请求中声明的会话、turn 一致，且其所在会话同样属于本人和该论文。
        if (not linked or (payload.session_id and linked.get("session_id") != payload.session_id)
                or (payload.source_turn_id and linked.get("turn_id") != payload.source_turn_id)):
            raise AuthError("private_resource_not_found")
        require_paper_session(arxiv_id, linked["session_id"], user_id=user_id)
    return linked["message_id"] if linked else None
