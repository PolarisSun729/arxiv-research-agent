"""本地 OAI 搜索共享契约。

这个模块只承载本地 OAI 搜索边界的稳定常量、能力描述和结构化异常，
供 query parser / query compiler / database service 共同复用，避免同名异常
在不同模块里各自实现后出现状态码和错误载荷不一致。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from services.arxiv.contracts import ArxivSearchError

# 本地 OAI 只承诺这组可下推子集；前端和工具层都应围绕这份契约做提示。
LOCAL_OAI_SUPPORTED_QUERY_SUBSET = [
    "id",
    "cat",
    "submittedDate",
    "ti",
    "abs",
    "au",
    "all",
    "AND",
    "OR",
    "ANDNOT",
    "phrase",
]


def build_local_oai_query_capability(
    *,
    mode: str = "local_oai_sqlite_fts",
    unsupported_reason: Optional[str] = None,
    suggested_action: Optional[str] = None,
    fts5_available: Optional[bool] = None,
    search_index_status: Optional[str] = None,
) -> Dict[str, Any]:
    """构造后端和前端共享的本地检索能力契约。"""
    capability = {
        "source": "local_oai",
        "mode": mode,
        "supported_subset": list(LOCAL_OAI_SUPPORTED_QUERY_SUBSET),
        "precision_policy": "strict_token_or_phrase",
        "full_arxiv_syntax_supported": False,
        "unsupported_reason": unsupported_reason,
        "suggested_action": suggested_action,
    }
    if fts5_available is not None:
        capability["fts5_available"] = bool(fts5_available)
    if search_index_status:
        capability["search_index_status"] = search_index_status
    return capability


class LocalArxivSearchError(ArxivSearchError):
    """本地 OAI 搜索的稳定错误基类。"""

    code = "local_arxiv_search_error"
    status_code = 400

    def __init__(
        self,
        message: str,
        *,
        query: str = "",
        reason: str = "",
        query_capability: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None,
        fts5_available: Optional[bool] = None,
        search_index_status: Optional[str] = None,
    ) -> None:
        normalized_reason = reason or message
        capability = query_capability or build_local_oai_query_capability(
            mode="unsupported",
            unsupported_reason=normalized_reason,
            suggested_action=(
                suggested_action
                or "切换远程 arXiv API 后重试，或改写为本地 OAI 镜像支持的高精度查询子集。"
            ),
            fts5_available=fts5_available,
            search_index_status=search_index_status,
        )
        super().__init__(
            message,
            query=query,
            reason=normalized_reason,
            query_capability=capability,
        )

    def to_error_detail(self) -> Dict[str, Any]:
        """输出包含本地 OAI 来源信息的稳定错误详情。"""
        return {
            "code": self.code,
            "message": str(self),
            "details": {
                "query": self.query,
                "source": "local_oai",
                "reason": self.reason,
                "query_capability": self.query_capability,
                "suggested_action": (
                    self.query_capability.get("suggested_action")
                    if isinstance(self.query_capability, dict)
                    else None
                ),
            },
        }


class UnsupportedLocalArxivQuery(LocalArxivSearchError):
    """查询语法超出本地 OAI 支持子集时抛出。"""

    code = "unsupported_local_arxiv_query"


class LocalArxivSearchIndexUnavailable(LocalArxivSearchError):
    """本地 OAI 搜索索引不可用时抛出。"""

    code = "local_search_index_unavailable"
    status_code = 503
