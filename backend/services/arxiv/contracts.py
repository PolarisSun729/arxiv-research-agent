"""arXiv 搜索边界契约。

该模块只描述共享搜索边界的数据形态和能力要求，不包含任何远程请求、
SQLite 查询或输入拼装逻辑。这样可以把“谁能执行搜索”和“如何构造 query”
分开，避免重新长出兼容包装层。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypedDict


class ArxivSearchResult(TypedDict, total=False):
    """共享搜索后端的标准返回结构。"""

    query: Optional[str]
    id_list: List[str]
    total_results: int
    start_index: int
    items_per_page: int
    papers: List[Dict[str, Any]]
    timestamp: str
    source: str
    warnings: List[str]
    query_capability: Optional[Dict[str, Any]]


class PreparedArxivSearchRequest(TypedDict, total=False):
    """输入层归一化后的搜索请求。

    这个结构是 router、tools 和 remote-only workflow 之间传递搜索参数的唯一
    中间形态；底层后端只接收 final_search_query/id_list/paging/sorting。
    """

    mode: str
    raw_inputs: Dict[str, Any]
    normalized_inputs: Dict[str, Any]
    final_search_query: Optional[str]
    id_list: List[str]
    max_results: int
    start: int
    sort_by: str
    sort_order: str
    submitted_date_query: Optional[str]
    submitted_days_ago_applied: bool


class ArxivSearchBackend(Protocol):
    """本地和远程 arXiv 搜索实现共同遵守的最小能力协议。"""

    def search(
        self,
        search_query: Optional[str] = None,
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> ArxivSearchResult:
        """执行已经归一化好的搜索请求。"""
        ...

    def get_available_fields(self) -> List[Dict[str, str]]:
        """返回当前搜索实现支持展示的字段列表。"""
        ...

    def get_subject_categories(self) -> List[Dict[str, str]]:
        """返回当前搜索实现支持展示的分类列表。"""
        ...


class ArxivSearchError(ValueError):
    """共享 arXiv 搜索边界的稳定错误基类。"""

    code = "arxiv_search_error"
    status_code = 400

    def __init__(
        self,
        message: str,
        *,
        query: str = "",
        reason: str = "",
        query_capability: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.query = query
        self.reason = reason or message
        self.query_capability = query_capability

    def to_error_detail(self) -> Dict[str, Any]:
        """输出 router 和 tools 可直接透传的结构化错误详情。"""
        details: Dict[str, Any] = {
            "query": self.query,
            "reason": self.reason,
        }
        if isinstance(self.query_capability, dict):
            details["query_capability"] = self.query_capability
            details["suggested_action"] = self.query_capability.get("suggested_action")
        return {
            "code": self.code,
            "message": str(self),
            "details": details,
        }


class ArxivSearchRequestError(ArxivSearchError):
    """输入归一化或参数校验失败时抛出。"""

    code = "arxiv_invalid_query"


class ArxivRemoteSearchError(ArxivSearchError):
    """远程 arXiv API 搜索失败时抛出。"""

    code = "arxiv_remote_search_failed"
    status_code = 502
