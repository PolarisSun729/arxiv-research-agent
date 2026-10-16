"""真实路由与工具的权限清单；新接口默认拒绝，评审后才能加入对应权限组。

quota 标记的是会产生外部成本（LLM、embedding、arXiv 回源）的操作：账号模式扣对应类型的配额，
密钥模式计入该密钥的 daily_quota。纯本地读取不标记配额，只受入口速率限制。

使用方：
- JwtAuthMiddleware：按 ROUTE_POLICIES 判断角色是否有权访问，未登记的路由一律拒绝；
- UserQuotaMiddleware / middleware.rate_limit：按 quota 字段决定是否扣减配额；
- middleware.rate_limit：按 rate_category() 决定认证后使用 read / write / expensive 哪一类速率预算；
- tool_access.authorize_backend_tool：按 TOOL_POLICIES 校验 Agent 工具调用。
"""

from dataclasses import dataclass, replace


# 常用的角色组合。
ALL_ROLES = frozenset({"admin", "researcher", "viewer", "guest"})
RESEARCH_ROLES = frozenset({"admin", "researcher"})
ADMIN_ROLES = frozenset({"admin"})


@dataclass(frozen=True)
class AccessPolicy:
    """一个路由或工具的访问策略。

    - roles：允许访问的角色集合；
    - quota：每次调用扣减的配额类型（见 auth.models.QUOTA_TYPES），None 表示不扣配额；
    - rate_category：显式指定的速率类别（read / write / expensive），None 表示按 rate_category() 的默认规则推导。
    """

    roles: frozenset[str]
    quota: str | None = None
    rate_category: str | None = None


# 无需令牌即可访问的认证入口：查询认证模式、登录、注册（注册还受 ALLOW_PUBLIC_REGISTRATION 控制）。
PUBLIC_AUTH_ROUTES = frozenset({("GET", "/api/auth/config"), ("POST", "/api/auth/login"), ("POST", "/api/auth/register")})
# (HTTP 方法, 路由模板) → 访问策略。键里的路径是 FastAPI 的路由模板（带 {参数} 占位符），
# 与 RequestContextMiddleware 写入的 scope["security_route"] 对应，而不是请求的原始路径。
ROUTE_POLICIES: dict[tuple[str, str], AccessPolicy] = {}


def _register(roles, quota, routes) -> None:
    """把一组路由登记为相同的访问策略。"""
    for method, path in routes:
        ROUTE_POLICIES[(method, path)] = AccessPolicy(roles, quota)


# ---- 所有登录用户可用、不扣配额 ----
# viewer/guest 可维护自己的阅读记录和笔记，也可在额度内 QA；共享资料的维护另授权限。
_register(ALL_ROLES, None, [
    # 账号自身
    ("GET", "/api/auth/check"), ("GET", "/api/auth/me"), ("POST", "/api/auth/logout"), ("POST", "/api/auth/password"),
    # 公共元数据与统计
    ("GET", "/api/arxiv/fields"), ("GET", "/api/arxiv/categories"), ("GET", "/api/stats"),
    # 搜索结果偏少时前端自动请求的放宽建议：本地只做 COUNT、远程后端不发请求，不应重复扣检索配额。
    ("POST", "/api/arxiv/search/suggestions"),
    # 个人偏好、阅读记录与研究画像
    ("GET", "/api/user/preferences/{user_id}"), ("GET", "/api/user/paper-actions/{user_id}"),
    ("GET", "/api/user/research-profile/{user_id}"), ("GET", "/api/user/research-profile/{user_id}/detail"),
    ("GET", "/api/user/research-profile/{user_id}/topic-evidence"),
    ("GET", "/api/user/research-profile/build-jobs/{job_id}"), ("GET", "/api/user/research-profile/{user_id}/build-jobs"),
    ("GET", "/api/user/interest-vector"), ("DELETE", "/api/user/like-paper"),
    ("DELETE", "/api/user/dislike-paper"), ("DELETE", "/api/user/paper-action"),
    ("PUT", "/api/user/research-profile"), ("PATCH", "/api/user/research-profile"),
    ("POST", "/api/user/research-profile/snapshots/activate"),
    # 论文问答的状态查询与对话会话（问答本身在下方单独计费）
    ("GET", "/api/paper/{arxiv_id}/qa-status"), ("GET", "/api/paper/{arxiv_id}/qa-index-jobs/latest"),
    ("GET", "/api/paper/{arxiv_id}/qa-index-jobs/{job_id}"), ("GET", "/api/paper/{arxiv_id}/evidence-assets/{source_id}"),
    ("GET", "/api/paper/{arxiv_id}/chat-sessions"), ("GET", "/api/paper/{arxiv_id}/chat-sessions/recent"),
    ("POST", "/api/paper/{arxiv_id}/chat-sessions"), ("GET", "/api/paper/{arxiv_id}/chat-sessions/{session_id}"),
    ("GET", "/api/paper/{arxiv_id}/chat-sessions/{session_id}/messages"),
    ("POST", "/api/paper/{arxiv_id}/chat-sessions/{session_id}/clear"),
    ("DELETE", "/api/paper/{arxiv_id}/chat-sessions/{session_id}"),
    # 本地目录读取不产生外部成本；论文详情只在回源 arXiv 时由路由内部扣减 papers 配额。
    ("GET", "/api/papers"), ("GET", "/api/papers/category/{category}"), ("GET", "/api/paper/{arxiv_id}"),
    ("GET", "/api/user/labeled-papers/{user_id}"),
])
# ---- 所有登录用户可用、扣 papers 配额：检索、标记偏好、推荐 ----
_register(ALL_ROLES, "papers", [
    ("POST", "/api/arxiv/search"), ("POST", "/api/user/like-paper"), ("POST", "/api/user/dislike-paper"),
    ("POST", "/api/user/paper-action"), ("POST", "/api/user/recommend-papers"),
])
# ---- 所有登录用户可用、扣 qa_queries 配额：论文问答（普通与流式） ----
_register(ALL_ROLES, "qa_queries", [("POST", "/api/paper/{arxiv_id}/qa"), ("POST", "/api/paper/{arxiv_id}/qa/stream")])
# ---- 研究员及以上、扣 papers 配额：入库、下载、建立问答索引、重建画像与兴趣向量 ----
_register(RESEARCH_ROLES, "papers", [
    ("POST", "/api/paper"), ("POST", "/api/arxiv/download"), ("POST", "/api/paper/{arxiv_id}/create-qa-index"),
    ("POST", "/api/user/research-profile/rebuild"), ("POST", "/api/user/generate-interest-vector"),
])
# ---- 研究员及以上、扣 agent_runs 配额：启动或续跑 Agent ----
_register(RESEARCH_ROLES, "agent_runs", [
    ("POST", "/api/agent/chat"), ("POST", "/api/agent/chat/stream"),
    ("POST", "/api/agent/work-continuations/{continuation_id}/resume/stream"),
])
# ---- 研究员及以上、不扣配额：Agent 会话与续跑任务的管理和查询 ----
_register(RESEARCH_ROLES, None, [
    ("GET", "/api/agent/sessions"), ("GET", "/api/agent/sessions/{session_id}/messages"),
    ("POST", "/api/agent/sessions/{session_id}/clear"), ("GET", "/api/agent/work-continuations/active"),
    ("GET", "/api/agent/work-continuations/{continuation_id}"),
    ("POST", "/api/agent/work-continuations/{continuation_id}/cancel"), ("GET", "/api/agent/resume-runs/{resume_run_id}"),
])
# ---- 仅管理员：账号管理、删除论文、系统诊断 ----
# 全局 trace 按论文落盘，可能包含别人的问题；即使知道文件名也只允许管理员诊断。
_register(ADMIN_ROLES, None, [
    ("GET", "/api/auth/users"), ("POST", "/api/auth/users"), ("GET", "/api/auth/users/{target_user_id}"),
    ("PATCH", "/api/auth/users/{target_user_id}"), ("PUT", "/api/auth/users/{target_user_id}/quotas"),
    ("DELETE", "/api/paper/{arxiv_id}"), ("GET", "/api/sync-status"), ("GET", "/api/agent/graph"),
    ("GET", "/api/paper/{arxiv_id}/qa-diagnose"), ("GET", "/api/paper/{arxiv_id}/qa-trace/latest"),
    ("GET", "/api/debug/chunks/files"), ("GET", "/api/debug/chunks/file/{filename}"),
])



def _set_rate_category(category: str, routes) -> None:
    """为已登记的路由显式指定速率类别；路由未登记时直接 KeyError，防止拼写错误让覆盖静默失效。"""
    for key in routes:
        ROUTE_POLICIES[key] = replace(ROUTE_POLICIES[key], rate_category=category)


# ---- 速率类别的例外 ----
# 检索虽然计 papers 配额，语义上是查询，与其他读取共用 read 预算。
_set_rate_category("read", [("POST", "/api/arxiv/search"), ("POST", "/api/arxiv/search/suggestions")])
# 详情在本地缺失时会回源 arXiv、物化论文并生成 embedding（配额在路由内部按需扣减），
# 不能因为是 GET 就降级为 read 预算。
_set_rate_category("expensive", [("GET", "/api/paper/{arxiv_id}")])


def rate_category(method: str, route: str) -> str:
    """返回 (方法, 路由模板) 在认证后使用的速率类别：read、write 或 expensive。

    规则：显式指定的 rate_category 优先；否则标记了 quota 的路由（会产生外部成本）归为 expensive；
    其余按方法归类，GET / HEAD / OPTIONS 为 read，其他为 write。
    新增付费接口只需在上面标记 quota，就会自动使用 expensive 预算，不需要再维护另一份清单。
    """
    policy = ROUTE_POLICIES.get((method, route))
    if policy is not None and policy.rate_category:
        return policy.rate_category
    if policy is not None and policy.quota:
        return "expensive"
    return "read" if method in {"GET", "HEAD", "OPTIONS"} else "write"


# Agent 可调用的后端工具：工具名 → 访问策略。所有工具都要求研究员及以上角色，
# 每次调用按各自的 quota 单独扣减（Agent 对话本身的 agent_runs 配额在路由层扣减）。
# 未登记的工具名一律拒绝。
TOOL_POLICIES = {
    name: AccessPolicy(RESEARCH_ROLES, quota) for name, quota in {
        "search_arxiv_raw": "papers", "search_arxiv_structured": "papers", "get_paper_metadata": "papers",
        "recommend_papers": "papers", "record_paper_preference": "papers", "remove_paper_preference": None,
        "check_paper_qa_index": None, "build_paper_qa_index": "papers", "answer_paper_question": "qa_queries",
    }.items()
}
