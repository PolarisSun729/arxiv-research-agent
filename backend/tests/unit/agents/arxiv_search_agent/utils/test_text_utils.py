from __future__ import annotations

import pytest

from backend.tests.lightweight_imports import load_arxiv_utils_module


text_utils = load_arxiv_utils_module("text_utils")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GraphRAG", False),
        ("中文 query", True),
        ("mix English 和中文", True),
    ],
)
def test_contains_chinese_detects_chinese_characters(text: str, expected: bool) -> None:
    assert text_utils._contains_chinese(text) is expected


def test_normalize_text_collapses_whitespace_and_none() -> None:
    assert text_utils._normalize_text("  rag\n systems\t today  ") == "rag systems today"
    assert text_utils._normalize_text(None) == ""


def test_normalize_optional_str_returns_none_for_blank_values() -> None:
    assert text_utils._normalize_optional_str("  ") is None
    assert text_utils._normalize_optional_str(None) is None
    assert text_utils._normalize_optional_str(123) == "123"


def test_matches_any_is_case_insensitive() -> None:
    assert text_utils._matches_any("GraphRAG for Search", [r"graphrag", r"bm25"]) is True
    assert text_utils._matches_any("GraphRAG for Search", [r"dense retriever", r"bm25"]) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("7", 7),
        ("三", 3),
        ("十", 10),
        ("十二", 12),
        ("二十", 20),
        ("二十三", 23),
        ("百十", None),
    ],
)
def test_parse_small_chinese_number_handles_common_inputs(value: object, expected: int | None) -> None:
    assert text_utils._parse_small_chinese_number(value) == expected


def test_extract_json_block_prefers_fenced_json_then_raw_object() -> None:
    fenced = 'before ```json\n{"answer": 1}\n``` after'
    raw = 'prefix {"answer": 2, "ok": true} suffix'

    assert text_utils._extract_json_block(fenced) == '{"answer": 1}'
    assert text_utils._extract_json_block(raw) == '{"answer": 2, "ok": true}'
    assert text_utils._extract_json_block("no json here") == "no json here"


def test_extract_json_object_parses_fenced_and_embedded_objects() -> None:
    assert text_utils._extract_json_object('```json\n{"answer": 1}\n```') == {"answer": 1}
    assert text_utils._extract_json_object('analysis: {"answer": 2, "items": [1, 2]} done') == {
        "answer": 2,
        "items": [1, 2],
    }


@pytest.mark.parametrize("text", ["", "not json", "[1, 2, 3]", '{"answer": }'])
def test_extract_json_object_returns_none_for_invalid_or_non_object_json(text: str) -> None:
    assert text_utils._extract_json_object(text) is None
