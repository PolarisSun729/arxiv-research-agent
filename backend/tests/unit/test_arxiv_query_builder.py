import pytest

from services.arxiv.arxiv_query_builder import (
    MAX_ALLOWED_RESULTS,
    ArxivSearchValidationError,
    build_arxiv_field_clause,
    build_arxiv_query_from_structured_params,
    build_arxiv_raw_query,
    combine_arxiv_clauses,
    normalize_id_list,
    normalize_text_value,
    prepare_arxiv_search_request,
    quote_arxiv_text,
    validate_arxiv_search_request,
)


def test_normalize_text_value_collapses_whitespace_and_handles_none() -> None:
    assert normalize_text_value("  rag\n systems \t today  ") == "rag systems today"
    assert normalize_text_value(None) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("rag", "rag"),
        ("rag systems", '"rag systems"'),
        ('"exact phrase"', '"exact phrase"'),
        (r"a:b", '"a:b"'),
        (r'quote " me', '"quote \\\" me"'),
        (r"path\\name", r"path\\\\name"),
    ],
)
def test_quote_arxiv_text_handles_plain_quoted_and_special_values(value: str, expected: str) -> None:
    assert quote_arxiv_text(value) == expected


def test_build_arxiv_field_clause_handles_text_category_and_empty_values() -> None:
    assert build_arxiv_field_clause("ti", "graph rag") == 'ti:"graph rag"'
    assert build_arxiv_field_clause("cat", "cs.CL") == "cat:cs.CL"
    assert build_arxiv_field_clause("ti", "   ") == ""


def test_combine_arxiv_clauses_handles_empty_single_and_multiple_clauses() -> None:
    assert combine_arxiv_clauses([], "AND") == ""
    assert combine_arxiv_clauses(["", "ti:rag", ""], "AND") == "ti:rag"
    assert combine_arxiv_clauses(["ti:rag", "au:lewis"], "OR") == "ti:rag OR au:lewis"


def test_normalize_id_list_strips_and_filters_blank_values() -> None:
    assert normalize_id_list([" 2401.00001 ", "", "  ", 2401]) == ["2401.00001", "2401"]
    assert normalize_id_list(None) == []


def test_validate_arxiv_search_request_accepts_valid_query() -> None:
    validate_arxiv_search_request(
        search_query="rag",
        id_list=None,
        max_results=10,
        start=0,
        sort_by="relevance",
        sort_order="descending",
    )


def test_validate_arxiv_search_request_allows_empty_query_when_require_query_disabled() -> None:
    validate_arxiv_search_request(
        search_query=None,
        id_list=None,
        max_results=1,
        start=0,
        sort_by="submittedDate",
        sort_order="ascending",
        require_query=False,
    )


@pytest.mark.parametrize(
    ("kwargs", "message_fragment"),
    [
        (
            {
                "search_query": None,
                "id_list": None,
                "max_results": 10,
                "start": 0,
                "sort_by": "relevance",
                "sort_order": "descending",
            },
            "search_query and id_list cannot both be empty",
        ),
        (
            {
                "search_query": "rag",
                "id_list": None,
                "max_results": 0,
                "start": 0,
                "sort_by": "relevance",
                "sort_order": "descending",
            },
            f"max_results must be between 1 and {MAX_ALLOWED_RESULTS}",
        ),
        (
            {
                "search_query": "rag",
                "id_list": None,
                "max_results": 1,
                "start": -1,
                "sort_by": "relevance",
                "sort_order": "descending",
            },
            "start must be greater than or equal to 0",
        ),
        (
            {
                "search_query": "rag",
                "id_list": None,
                "max_results": 1,
                "start": 0,
                "sort_by": "score",
                "sort_order": "descending",
            },
            "sort_by must be one of",
        ),
        (
            {
                "search_query": "rag",
                "id_list": None,
                "max_results": 1,
                "start": 0,
                "sort_by": "relevance",
                "sort_order": "upward",
            },
            "sort_order must be one of",
        ),
    ],
)
def test_validate_arxiv_search_request_rejects_invalid_inputs(kwargs: dict, message_fragment: str) -> None:
    with pytest.raises(ArxivSearchValidationError, match=message_fragment):
        validate_arxiv_search_request(**kwargs)


def test_build_arxiv_raw_query_appends_submitted_date_filter() -> None:
    payload = build_arxiv_raw_query(search_query="rag", id_list=[" 2401.00001 "], submitted_days_ago=7)

    assert payload["final_search_query"].startswith("(rag) AND submittedDate:[")
    assert payload["submitted_days_ago_applied"] is True
    assert payload["normalized_inputs"]["search_query"] == "rag"
    assert payload["id_list"] == ["2401.00001"]


def test_build_arxiv_raw_query_can_use_date_only_when_query_missing() -> None:
    payload = build_arxiv_raw_query(search_query=None, submitted_days_ago=3, append_date_when_query_missing=True)

    assert payload["final_search_query"] is not None
    assert payload["final_search_query"].startswith("submittedDate:[")


def test_build_arxiv_raw_query_ignores_negative_submitted_days_when_not_strict() -> None:
    payload = build_arxiv_raw_query(search_query="rag", submitted_days_ago=-3)

    assert payload["final_search_query"] == "rag"
    assert payload["submitted_date_query"] is None
    assert payload["submitted_days_ago_applied"] is False


def test_build_arxiv_raw_query_rejects_negative_submitted_days_when_strict() -> None:
    with pytest.raises(ArxivSearchValidationError, match="submitted_days_ago must be greater than or equal to 0"):
        build_arxiv_raw_query(search_query="rag", submitted_days_ago=-3, strict_submitted_days_ago=True)


def test_prepare_arxiv_search_request_normalizes_raw_query_once() -> None:
    payload = prepare_arxiv_search_request(
        search_query="  rag systems  ",
        id_list=[" 2401.00001 "],
        submitted_days_ago=7,
        max_results=5,
        start=2,
        sort_by="submittedDate",
        sort_order="ascending",
        strict_submitted_days_ago=True,
    )

    assert payload["mode"] == "raw"
    assert payload["final_search_query"].startswith("(rag systems) AND submittedDate:[")
    assert payload["id_list"] == ["2401.00001"]
    assert payload["max_results"] == 5
    assert payload["start"] == 2
    assert payload["normalized_inputs"]["mode"] == "raw"
    assert payload["submitted_days_ago_applied"] is True


def test_prepare_arxiv_search_request_normalizes_structured_query_once() -> None:
    payload = prepare_arxiv_search_request(
        search_query="retrieval",
        title_query="graph rag",
        categories=[" cs.IR "],
        category="cs.CL",
        field_operator="or",
        category_operator="and",
        max_results=3,
    )

    assert payload["mode"] == "structured"
    assert "all:retrieval" in payload["final_search_query"]
    assert 'ti:"graph rag"' in payload["final_search_query"]
    assert "(cat:cs.IR AND cat:cs.CL)" in payload["final_search_query"]
    assert payload["normalized_inputs"]["field_operator"] == "OR"
    assert payload["normalized_inputs"]["category_operator"] == "AND"
    assert payload["normalized_inputs"]["mode"] == "structured"


def test_prepare_arxiv_search_request_rejects_negative_submitted_days_for_structured_query() -> None:
    with pytest.raises(ArxivSearchValidationError, match="submitted_days_ago must be greater than or equal to 0"):
        prepare_arxiv_search_request(
            title_query="rag",
            submitted_days_ago=-1,
            strict_submitted_days_ago=True,
        )


def test_build_arxiv_query_from_structured_params_combines_core_fields() -> None:
    payload = build_arxiv_query_from_structured_params(
        query="rag",
        title_query="graph retrieval",
        author_query="Lewis",
        abstract_query="dense search",
        categories=["cs.CL", " cs.IR "],
        comment_query="accepted at ACL",
        journal_ref_query="TACL",
        report_number_query="TR-001",
        id_list=[" 2401.00001 "],
        field_operator="andnot",
        category_operator="or",
        submitted_days_ago=14,
    )

    final_query = payload["final_search_query"]
    assert "all:rag" in final_query
    assert 'ti:"graph retrieval"' in final_query
    assert "au:Lewis" in final_query
    assert 'abs:"dense search"' in final_query
    assert "(cat:cs.CL OR cat:cs.IR)" in final_query
    assert 'co:"accepted at ACL"' in final_query
    assert "jr:TACL" in final_query
    assert "rn:TR-001" in final_query
    assert " AND submittedDate:[" in final_query
    assert payload["normalized_inputs"]["field_operator"] == "ANDNOT"
    assert payload["normalized_inputs"]["category_operator"] == "OR"
    assert payload["id_list"] == ["2401.00001"]


@pytest.mark.parametrize(
    ("kwargs", "message_fragment"),
    [
        ({"query": "rag", "field_operator": "XOR"}, "field_operator must be one of"),
        ({"query": "rag", "category_operator": "ANDNOT"}, "category_operator must be one of"),
        ({}, "at least one structured search field or id_list must be provided"),
    ],
)
def test_build_arxiv_query_from_structured_params_rejects_invalid_inputs(
    kwargs: dict,
    message_fragment: str,
) -> None:
    with pytest.raises(ArxivSearchValidationError, match=message_fragment):
        build_arxiv_query_from_structured_params(**kwargs)
