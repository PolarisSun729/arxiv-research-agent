"""arXiv 查询构造工具模块。

该模块把 arXiv 搜索请求中的文本归一化、字段拼装、参数校验以及结构化
查询组合逻辑集中到一个位置，避免搜索服务和数据库服务各自维护一套查询
构造细节，从而提升可复用性和一致性。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from utils.config import get_arxiv_search_runtime_config

ARXIV_SEARCH_CONFIG = get_arxiv_search_runtime_config()
VALID_SORT_BY = {"relevance", "lastUpdatedDate", "submittedDate"}
VALID_SORT_ORDER = {"ascending", "descending"}
VALID_FIELD_OPERATORS = {"AND", "OR", "ANDNOT"}
VALID_CATEGORY_OPERATORS = {"AND", "OR"}
MAX_ALLOWED_RESULTS = ARXIV_SEARCH_CONFIG["max_allowed_results"]


class ArxivSearchValidationError(ValueError):
    """arXiv 搜索请求校验失败异常。"""
    pass


def normalize_text_value(value: Optional[str]) -> str:
    """规范化输入文本，折叠多余空白字符。

    参数:
        value (Optional[str]): 原始输入值。

    返回:
        str: 去首尾空白并压缩内部连续空白后的字符串。
    """
    return " ".join(str(value or "").strip().split())


def quote_arxiv_text(value: str) -> str:
    """按 arXiv 查询语法需要对文本进行转义和包裹。

    参数:
        value (str): 原始查询文本。

    返回:
        str: 可安全用于 arXiv 查询语法的文本片段。
    """
    normalized = normalize_text_value(value)
    if not normalized:
        return ""
    if normalized.startswith('"') and normalized.endswith('"') and len(normalized) >= 2:
        return normalized
    escaped = normalized.replace("\\", "\\\\").replace('"', '\\"')
    if any(char.isspace() for char in normalized) or any(char in normalized for char in (":", "(", ")", '"')):
        return f'"{escaped}"'
    return escaped


def build_arxiv_field_clause(prefix: str, value: str) -> str:
    """构造单个字段查询子句。

    参数:
        prefix (str): arXiv 字段前缀，例如 ti、au、abs、cat。
        value (str): 字段对应的查询文本。

    返回:
        str: 格式化后的字段子句；若 value 为空则返回空字符串。
    """
    normalized = normalize_text_value(value)
    if not normalized:
        return ""
    if prefix == "cat":
        return f"{prefix}:{normalized}"
    return f"{prefix}:{quote_arxiv_text(normalized)}"


def combine_arxiv_clauses(clauses: List[str], operator: str = "AND") -> str:
    """使用逻辑操作符组合多个 arXiv 查询子句。

    参数:
        clauses (List[str]): 子句列表。
        operator (str): 逻辑操作符。

    返回:
        str: 组合后的查询字符串。
    """
    filtered = [clause for clause in clauses if clause]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    return f" {operator} ".join(filtered)


def normalize_id_list(id_list: Optional[List[str]]) -> List[str]:
    """清洗并规范化 arXiv 论文 ID 列表。"""
    if not id_list:
        return []
    return [str(item).strip() for item in id_list if str(item).strip()]


def validate_arxiv_search_request(
    *,
    search_query: Optional[str],
    id_list: Optional[List[str]],
    max_results: int,
    start: int,
    sort_by: str,
    sort_order: str,
    require_query: bool = True,
) -> None:
    """校验 arXiv 搜索请求的通用分页、排序和查询参数。

    参数:
        search_query (Optional[str]): 原始查询字符串。
        id_list (Optional[List[str]]): 精确匹配 ID 列表。
        max_results (int): 最大返回条数。
        start (int): 分页起始偏移量。
        sort_by (str): 排序字段。
        sort_order (str): 排序方向。
        require_query (bool): 是否强制要求 query 或 id_list 至少提供一个。

    返回:
        None

    异常:
        ArxivSearchValidationError: 当任一参数不满足约束时抛出。
    """
    normalized_search_query = normalize_text_value(search_query)
    normalized_id_list = normalize_id_list(id_list)

    if require_query and not normalized_search_query and not normalized_id_list:
        raise ArxivSearchValidationError("arxiv_invalid_query: search_query and id_list cannot both be empty")
    if not (1 <= int(max_results) <= MAX_ALLOWED_RESULTS):
        raise ArxivSearchValidationError(f"arxiv_invalid_query: max_results must be between 1 and {MAX_ALLOWED_RESULTS}")
    if int(start) < 0:
        raise ArxivSearchValidationError("arxiv_invalid_query: start must be greater than or equal to 0")
    if sort_by not in VALID_SORT_BY:
        raise ArxivSearchValidationError(
            f"arxiv_invalid_query: sort_by must be one of {sorted(VALID_SORT_BY)}"
        )
    if sort_order not in VALID_SORT_ORDER:
        raise ArxivSearchValidationError(
            f"arxiv_invalid_query: sort_order must be one of {sorted(VALID_SORT_ORDER)}"
        )


def build_arxiv_submitted_date_query(days_ago: Optional[int]) -> str:
    """构建 arXiv 提交日期范围子句。"""
    end_date = datetime.now(timezone.utc)
    start_days = days_ago if days_ago is not None else 30
    start_date = end_date - timedelta(days=int(start_days))
    return f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO {end_date.strftime('%Y%m%d%H%M')}]"


def build_arxiv_raw_query(
    *,
    search_query: Optional[str] = None,
    id_list: Optional[List[str]] = None,
    submitted_days_ago: Optional[int] = None,
    append_date_when_query_missing: bool = False,
    strict_submitted_days_ago: bool = False,
) -> Dict[str, Any]:
    """从原始搜索参数构建最终 arXiv 查询表达式。

    参数:
        search_query (Optional[str]): 原始 search_query 字符串。
        id_list (Optional[List[str]]): ID 列表。
        submitted_days_ago (Optional[int]): 最近提交天数限制。
        append_date_when_query_missing (bool): 无查询词时是否仅使用日期条件。
        strict_submitted_days_ago (bool): 是否对负数天数严格报错。

    返回:
        Dict[str, Any]: 包含原始输入、归一化输入和最终查询串的结果字典。
    """
    normalized_search_query = normalize_text_value(search_query)
    normalized_id_list = normalize_id_list(id_list)

    final_search_query = normalized_search_query or None
    submitted_date_query = None
    submitted_days_ago_applied = False
    submitted_days_ago_value = submitted_days_ago
    if submitted_days_ago_value is not None:
        if submitted_days_ago_value < 0:
            if strict_submitted_days_ago:
                raise ArxivSearchValidationError(
                    "arxiv_invalid_query: submitted_days_ago must be greater than or equal to 0"
                )
        else:
            submitted_days_ago_applied = True
            submitted_date_query = build_arxiv_submitted_date_query(submitted_days_ago_value)
            if final_search_query:
                final_search_query = f"({final_search_query}) AND {submitted_date_query}"
            elif append_date_when_query_missing:
                final_search_query = submitted_date_query

    return {
        "raw_inputs": {
            "search_query": search_query,
            "id_list": id_list,
            "submitted_days_ago": submitted_days_ago_value,
        },
        "normalized_inputs": {
            "search_query": normalized_search_query,
            "id_list": normalized_id_list,
            "submitted_days_ago": submitted_days_ago_value,
            "submitted_days_ago_applied": submitted_days_ago_applied,
        },
        "final_search_query": final_search_query,
        "id_list": normalized_id_list,
        "submitted_date_query": submitted_date_query,
        "submitted_days_ago_applied": submitted_days_ago_applied,
    }


def build_arxiv_query_from_structured_params(
    *,
    query: Optional[str] = None,
    title_query: Optional[str] = None,
    author_query: Optional[str] = None,
    abstract_query: Optional[str] = None,
    categories: Optional[List[str]] = None,
    comment_query: Optional[str] = None,
    journal_ref_query: Optional[str] = None,
    report_number_query: Optional[str] = None,
    id_list: Optional[List[str]] = None,
    field_operator: str = "AND",
    category_operator: str = "OR",
    submitted_days_ago: Optional[int] = None,
) -> Dict[str, Any]:
    """从结构化字段参数构建 arXiv 查询表达式。

    参数:
        query (Optional[str]): 全字段查询词。
        title_query (Optional[str]): 标题查询词。
        author_query (Optional[str]): 作者查询词。
        abstract_query (Optional[str]): 摘要查询词。
        categories (Optional[List[str]]): 分类列表。
        comment_query (Optional[str]): comment 字段查询词。
        journal_ref_query (Optional[str]): journal_ref 字段查询词。
        report_number_query (Optional[str]): report number 查询词。
        id_list (Optional[List[str]]): 精确匹配 ID 列表。
        field_operator (str): 多字段之间的逻辑操作符。
        category_operator (str): 多分类之间的逻辑操作符。
        submitted_days_ago (Optional[int]): 最近提交天数限制。

    返回:
        Dict[str, Any]: 包含原始输入、归一化输入和最终查询串的结果字典。

    异常:
        ArxivSearchValidationError: 当结构化字段组合不合法时抛出。
    """
    normalized_field_operator = (field_operator or "AND").strip().upper()
    normalized_category_operator = (category_operator or "OR").strip().upper()
    if normalized_field_operator not in VALID_FIELD_OPERATORS:
        raise ArxivSearchValidationError(
            f"arxiv_invalid_query: field_operator must be one of {sorted(VALID_FIELD_OPERATORS)}"
        )
    if normalized_category_operator not in VALID_CATEGORY_OPERATORS:
        raise ArxivSearchValidationError(
            f"arxiv_invalid_query: category_operator must be one of {sorted(VALID_CATEGORY_OPERATORS)}"
        )

    normalized_id_list = normalize_id_list(id_list)
    normalized_categories = [str(item).strip() for item in (categories or []) if str(item).strip()]
    normalized_query = normalize_text_value(query)
    normalized_title = normalize_text_value(title_query)
    normalized_author = normalize_text_value(author_query)
    normalized_abstract = normalize_text_value(abstract_query)
    normalized_comment = normalize_text_value(comment_query)
    normalized_journal_ref = normalize_text_value(journal_ref_query)
    normalized_report_number = normalize_text_value(report_number_query)

    clauses: List[str] = []
    if normalized_query:
        clauses.append(build_arxiv_field_clause("all", normalized_query))
    if normalized_title:
        clauses.append(build_arxiv_field_clause("ti", normalized_title))
    if normalized_author:
        clauses.append(build_arxiv_field_clause("au", normalized_author))
    if normalized_abstract:
        clauses.append(build_arxiv_field_clause("abs", normalized_abstract))
    if normalized_categories:
        category_clauses = [f"cat:{category}" for category in normalized_categories]
        category_clause = combine_arxiv_clauses(category_clauses, normalized_category_operator)
        if len(category_clauses) > 1:
            category_clause = f"({category_clause})"
        clauses.append(category_clause)
    if normalized_comment:
        clauses.append(build_arxiv_field_clause("co", normalized_comment))
    if normalized_journal_ref:
        clauses.append(build_arxiv_field_clause("jr", normalized_journal_ref))
    if normalized_report_number:
        clauses.append(build_arxiv_field_clause("rn", normalized_report_number))

    if not clauses and not normalized_id_list:
        raise ArxivSearchValidationError(
            "arxiv_invalid_query: at least one structured search field or id_list must be provided"
        )

    final_search_query = combine_arxiv_clauses(clauses, normalized_field_operator)
    submitted_date_query = None
    if submitted_days_ago is not None and submitted_days_ago >= 0:
        submitted_date_query = build_arxiv_submitted_date_query(submitted_days_ago)
        if final_search_query:
            final_search_query = f"{final_search_query} AND {submitted_date_query}"

    return {
        "raw_inputs": {
            "query": query,
            "title_query": title_query,
            "author_query": author_query,
            "abstract_query": abstract_query,
            "categories": categories,
            "comment_query": comment_query,
            "journal_ref_query": journal_ref_query,
            "report_number_query": report_number_query,
            "id_list": id_list,
            "field_operator": field_operator,
            "category_operator": category_operator,
            "submitted_days_ago": submitted_days_ago,
        },
        "normalized_inputs": {
            "query": normalized_query,
            "title_query": normalized_title,
            "author_query": normalized_author,
            "abstract_query": normalized_abstract,
            "categories": normalized_categories,
            "comment_query": normalized_comment,
            "journal_ref_query": normalized_journal_ref,
            "report_number_query": normalized_report_number,
            "id_list": normalized_id_list,
            "field_operator": normalized_field_operator,
            "category_operator": normalized_category_operator,
            "submitted_days_ago": submitted_days_ago,
            "submitted_date_query": submitted_date_query,
        },
        "final_search_query": final_search_query,
        "id_list": normalized_id_list,
        "submitted_date_query": submitted_date_query,
    }
