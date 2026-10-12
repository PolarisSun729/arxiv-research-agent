"""HTTP 之外的 Agent 工具同样校验身份、实时权限和用量，不能成为权限旁路。

Agent 在一次请求中可能执行很久（甚至在后台续跑），其间用户可能被降级、停用或登出；
工具参数又可能由 LLM 生成。因此每个图节点、每一步、每次工具调用都要重新校验，
而不是只依赖请求入口时做过的那一次检查。
"""

from auth.context import bind_identity_fields, bind_user_id, current_auth
from auth.errors import AuthError
from auth.permissions import RESEARCH_ROLES, TOOL_POLICIES


def bind_graph_node(node):
    """把可信身份封闭在节点 callable 中，不写入可持久化/可由 LLM 生成的 graph state。

    在构建 Agent 图时（处于已认证的请求上下文中）调用：捕获当时的 AuthContext，
    返回一个包装后的节点。图框架可能在其他线程或任务中执行节点，ContextVar 不一定会传递过去，
    所以包装函数在执行时重新设置 current_auth，并在每次执行节点前校验身份和角色。
    没有认证上下文时原样返回 node。
    """
    context = current_auth.get()
    if context is None:
        return node

    def guarded(state):
        token = current_auth.set(context)
        try:
            # state 可能是 dict，也可能是带 user_id 属性的对象。
            user_id = state.get("user_id") if isinstance(state, dict) else getattr(state, "user_id", None)
            authorize_agent_step(user_id, {})
            return node(state)
        finally:
            current_auth.reset(token)

    return guarded


def authorize_agent_step(state_user_id, arguments: dict) -> None:
    """Agent 每一步执行前的校验。

    - 当前用户在数据库中仍须是研究员及以上角色，会话仍须有效；
    - graph state 中记录的 user_id 必须是本人；
    - arguments 中的 user_id 字段原地改写为本人（与他人不一致时抛出 identity_mismatch）。
    """
    context = current_auth.get()
    if context is None:
        return
    context.require_roles(RESEARCH_ROLES)
    bind_user_id(state_user_id)
    bind_identity_fields(arguments)


def authorize_backend_tool(name: str, arguments: dict) -> None:
    """Agent 调用后端工具前的校验与计费。

    name 必须是已归一化的工具名（别名已解析）。未在 TOOL_POLICIES 中登记的工具一律拒绝；
    校验角色、绑定参数中的身份后，若该工具标记了配额，则同步扣减一次。
    """
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
