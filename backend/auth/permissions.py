"""真实路由与工具的权限清单；新接口默认拒绝，评审后才能加入对应权限组。"""

from dataclasses import dataclass


ALL_ROLES = frozenset({"admin", "researcher", "viewer", "guest"})
RESEARCH_ROLES = frozenset({"admin", "researcher"})
ADMIN_ROLES = frozenset({"admin"})


@dataclass(frozen=True)
class AccessPolicy:
    roles: frozenset[str]
    quota: str | None = None


PUBLIC_AUTH_ROUTES = frozenset({("GET", "/api/auth/config"), ("POST", "/api/auth/login"), ("POST", "/api/auth/register")})
ROUTE_POLICIES: dict[tuple[str, str], AccessPolicy] = {}


def _register(roles, quota, routes) -> None:
    for method, path in routes:
        ROUTE_POLICIES[(method, path)] = AccessPolicy(roles, quota)


# viewer/guest 可维护自己的阅读记录和笔记，也可在额度内 QA；共享资料的维护另授权限。
_register(ALL_ROLES, None, [
    ("GET", "/api/auth/check"), ("GET", "/api/auth/me"), ("POST", "/api/auth/logout"), ("POST", "/api/auth/password"),
    ("GET", "/api/arxiv/fields"), ("GET", "/api/arxiv/categories"), ("GET", "/api/stats"),
    ("GET", "/api/user/preferences/{user_id}"), ("GET", "/api/user/paper-actions/{user_id}"),
    ("GET", "/api/user/research-profile/{user_id}"), ("GET", "/api/user/research-profile/{user_id}/detail"),
    ("GET", "/api/user/research-profile/{user_id}/topic-evidence"),
    ("GET", "/api/user/research-profile/build-jobs/{job_id}"), ("GET", "/api/user/research-profile/{user_id}/build-jobs"),
    ("GET", "/api/user/interest-vector"), ("DELETE", "/api/user/like-paper"),
    ("DELETE", "/api/user/dislike-paper"), ("DELETE", "/api/user/paper-action"),
    ("PUT", "/api/user/research-profile"), ("PATCH", "/api/user/research-profile"),
    ("POST", "/api/user/research-profile/snapshots/activate"),
    ("GET", "/api/paper/{arxiv_id}/qa-status"), ("GET", "/api/paper/{arxiv_id}/qa-index-jobs/latest"),
    ("GET", "/api/paper/{arxiv_id}/qa-index-jobs/{job_id}"), ("GET", "/api/paper/{arxiv_id}/evidence-assets/{source_id}"),
    ("GET", "/api/paper/{arxiv_id}/chat-sessions"), ("GET", "/api/paper/{arxiv_id}/chat-sessions/recent"),
    ("POST", "/api/paper/{arxiv_id}/chat-sessions"), ("GET", "/api/paper/{arxiv_id}/chat-sessions/{session_id}"),
    ("GET", "/api/paper/{arxiv_id}/chat-sessions/{session_id}/messages"),
    ("POST", "/api/paper/{arxiv_id}/chat-sessions/{session_id}/clear"),
    ("DELETE", "/api/paper/{arxiv_id}/chat-sessions/{session_id}"),
    ("GET", "/api/paper/{arxiv_id}/notes"), ("POST", "/api/paper/{arxiv_id}/notes"),
    ("PATCH", "/api/paper/{arxiv_id}/notes/{note_id}"), ("DELETE", "/api/paper/{arxiv_id}/notes/{note_id}"),
    ("GET", "/api/paper/{arxiv_id}/notes/export"),
])
_register(ALL_ROLES, "papers", [
    ("POST", "/api/arxiv/search"), ("GET", "/api/papers"), ("GET", "/api/papers/category/{category}"),
    ("GET", "/api/paper/{arxiv_id}"), ("POST", "/api/user/like-paper"), ("POST", "/api/user/dislike-paper"),
    ("POST", "/api/user/paper-action"), ("POST", "/api/user/recommend-papers"),
])
_register(ALL_ROLES, "qa_queries", [("POST", "/api/paper/{arxiv_id}/qa"), ("POST", "/api/paper/{arxiv_id}/qa/stream")])
_register(RESEARCH_ROLES, "papers", [
    ("POST", "/api/paper"), ("POST", "/api/arxiv/download"), ("POST", "/api/paper/{arxiv_id}/create-qa-index"),
    ("POST", "/api/user/research-profile/rebuild"), ("POST", "/api/user/generate-interest-vector"),
])
_register(RESEARCH_ROLES, "agent_runs", [
    ("POST", "/api/agent/chat"), ("POST", "/api/agent/chat/stream"),
    ("POST", "/api/agent/work-continuations/{continuation_id}/resume/stream"),
])
_register(RESEARCH_ROLES, None, [
    ("POST", "/api/agent/sessions/{session_id}/clear"), ("GET", "/api/agent/work-continuations/active"),
    ("GET", "/api/agent/work-continuations/{continuation_id}"),
    ("POST", "/api/agent/work-continuations/{continuation_id}/cancel"), ("GET", "/api/agent/resume-runs/{resume_run_id}"),
])
# 全局 trace 按论文落盘，可能包含别人的问题；即使知道文件名也只允许管理员诊断。
_register(ADMIN_ROLES, None, [
    ("GET", "/api/auth/users"), ("POST", "/api/auth/users"), ("GET", "/api/auth/users/{target_user_id}"),
    ("PATCH", "/api/auth/users/{target_user_id}"), ("PUT", "/api/auth/users/{target_user_id}/quotas"),
    ("DELETE", "/api/paper/{arxiv_id}"), ("GET", "/api/sync-status"), ("GET", "/api/agent/graph"),
    ("GET", "/api/paper/{arxiv_id}/qa-diagnose"), ("GET", "/api/paper/{arxiv_id}/qa-trace/latest"),
    ("GET", "/api/debug/chunks/files"), ("GET", "/api/debug/chunks/file/{filename}"),
])

TOOL_POLICIES = {
    name: AccessPolicy(RESEARCH_ROLES, quota) for name, quota in {
        "search_arxiv_raw": "papers", "search_arxiv_structured": "papers", "get_paper_metadata": "papers",
        "recommend_papers": "papers", "record_paper_preference": "papers", "remove_paper_preference": None,
        "check_paper_qa_index": None, "build_paper_qa_index": "papers", "answer_paper_question": "qa_queries",
    }.items()
}
