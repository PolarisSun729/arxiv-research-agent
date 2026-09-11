"""HTTP 之外的 Agent 工具同样校验身份、实时权限和用量，不能成为权限旁路。"""

from auth.context import bind_identity_fields, bind_user_id, current_auth
from auth.errors import AuthError
from auth.permissions import RESEARCH_ROLES, TOOL_POLICIES


def bind_graph_node(node):
    """把可信身份封闭在节点 callable 中，不写入可持久化/可由 LLM 生成的 graph state。"""
    context = current_auth.get()
    if context is None:
        return node

    def guarded(state):
        token = current_auth.set(context)
        try:
            user_id = state.get("user_id") if isinstance(state, dict) else getattr(state, "user_id", None)
            authorize_agent_step(user_id, {})
            return node(state)
        finally:
            current_auth.reset(token)

    return guarded


def authorize_agent_step(state_user_id, arguments: dict) -> None:
    context = current_auth.get()
    if context is None:
        return
    context.require_roles(RESEARCH_ROLES)
    bind_user_id(state_user_id)
    bind_identity_fields(arguments)


def authorize_backend_tool(name: str, arguments: dict) -> None:
    context = current_auth.get()
    if context is None:
        return
    policy = TOOL_POLICIES.get(name)
    if policy is None:
        raise AuthError("insufficient_permissions")
    context.require_roles(policy.roles)
    bind_identity_fields(arguments)
    if policy.quota:
        context.consume(policy.quota)
