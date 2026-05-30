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
    pass


def normalize_text_value(value: Optional[str]) -> str:
    return " ".join(str(value or "").strip().split())


def quote_arxiv_text(value: str) -> str:
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
    normalized = normalize_text_value(value)
    if not normalized:
        return ""
    if prefix == "cat":
        return f"{prefix}:{normalized}"
    return f"{prefix}:{quote_arxiv_text(normalized)}"


def combine_arxiv_clauses(clauses: List[str], operator: str = "AND") -> str:
    filtered = [clause for clause in clauses if clause]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    return f" {operator} ".join(filtered)


def normalize_id_list(id_list: Optional[List[str]]) -> List[str]:
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
