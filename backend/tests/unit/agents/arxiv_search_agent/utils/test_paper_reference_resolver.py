from __future__ import annotations

import pytest

from backend.tests.lightweight_imports import load_arxiv_utils_module


resolver = load_arxiv_utils_module("paper_reference_resolver")


def _paper(arxiv_id: str | None, title: str) -> dict[str, object]:
    payload = {"title": title}
    if arxiv_id is not None:
        payload["arxiv_id"] = arxiv_id
    return payload


def test_extracts_explicit_arxiv_id_hint_without_binding_recent_paper() -> None:
    context = {"last_papers": [_paper("2401.00001", "Paper A"), _paper("2401.00002", "Paper B")]}

    result = resolver._resolve_paper_reference("请打开 arXiv 2401.00002", context)

    assert result["status"] == "hint_extracted"
    assert result["reference_type"] == "arxiv_id"
    assert result["value"] == "2401.00002"
    assert result["source"] == "explicit_arxiv_id"
    assert result["requires_context"] is False
    assert result["confidence"] >= 0.95
    # 引用提取层不能补全 title/paper，否则下游会继续把它当最终目标执行。
    assert result["arxiv_id"] is None
    assert result["paper"] is None
    assert result["title"] is None
    assert result["final_target_resolved"] is False


def test_extracts_url_style_arxiv_id_hint() -> None:
    result = resolver._resolve_paper_reference("看看 https://arxiv.org/abs/2401.12345v2 这篇", context={})

    assert result["status"] == "hint_extracted"
    assert result["reference_type"] == "arxiv_id"
    assert result["value"] == "2401.12345v2"
    assert result["requires_context"] is False


@pytest.mark.parametrize(
    ("message", "expected_type", "expected_value"),
    [
        ("第一篇论文", "ordinal", 1),
        ("第二篇论文", "ordinal", 2),
        ("最后一篇论文", "last_item", "last"),
        ("刚才那篇", "context_paper", "current"),
        ("不喜欢这个", "context_paper", "current"),
    ],
)
def test_extracts_position_hints_without_resolving_last_papers(message: str, expected_type: str, expected_value: object) -> None:
    context = {
        "last_papers": [
            _paper("2401.00001", "Paper A"),
            _paper("2401.00002", "Paper B"),
            _paper("2401.00003", "Paper C"),
        ]
    }

    result = resolver._resolve_paper_reference(message, context)

    assert result["status"] == "hint_extracted"
    assert result["reference_type"] == expected_type
    assert result["value"] == expected_value
    assert result["requires_context"] is True
    assert result["arxiv_id"] is None
    assert result["paper"] is None


def test_context_reference_is_hint_not_selected_paper_resolution() -> None:
    context = {
        "last_papers": [_paper("2401.00001", "Paper A"), _paper("2401.00002", "Paper B")],
        "selected_paper": _paper("2401.00002", "Paper B"),
    }

    result = resolver._resolve_paper_reference("这篇论文讲了什么？", context)

    assert result["status"] == "hint_extracted"
    assert result["reference_type"] == "context_paper"
    assert result["value"] == "current"
    assert result["source"] == "contextual_reference"
    assert result["requires_context"] is True
    assert result["arxiv_id"] is None
    assert result["paper"] is None


def test_ordinal_hint_wins_when_message_also_has_context_cue() -> None:
    context = {
        "selected_paper": _paper("2401.00001", "First Paper"),
        "last_papers": [_paper("2401.00001", "First Paper"), _paper("2401.00002", "Second Paper")],
    }

    result = resolver._resolve_paper_reference("这第2篇论文的方法是什么？", context)

    assert result["status"] == "hint_extracted"
    assert result["reference_type"] == "ordinal"
    assert result["value"] == 2
    assert result["source"] == "ordinal_expression"


def test_bare_number_hint_is_low_confidence_and_context_dependent() -> None:
    result = resolver._resolve_paper_reference("喜欢 2", context={"last_papers": [_paper("2401.00001", "Paper A")]})

    assert result["status"] == "hint_extracted"
    assert result["reference_type"] == "bare_number"
    assert result["value"] == 2
    assert result["source"] == "bare_number"
    assert result["requires_context"] is True
    assert 0.0 < result["confidence"] < 0.7


def test_no_reference_does_not_fallback_to_selected_paper() -> None:
    context = {"selected_paper": _paper("2401.00009", "Selected Paper")}

    result = resolver._resolve_paper_reference("请做个总结", context)

    assert result["status"] == "unknown"
    assert result["reference_type"] == "unknown"
    assert result["value"] is None
    assert result["source"] == "no_reference"
    assert result["arxiv_id"] is None
    assert result["paper"] is None


def test_no_reference_does_not_fallback_to_single_recent_paper() -> None:
    context = {"last_papers": [_paper("2401.00010", "Only Paper")]}

    result = resolver._resolve_paper_reference("请做个总结", context)

    assert result["status"] == "unknown"
    assert result["reference_type"] == "unknown"
    assert result["arxiv_id"] is None
    assert result["paper"] is None
