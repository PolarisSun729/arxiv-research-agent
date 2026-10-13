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
