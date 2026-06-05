from __future__ import annotations

import pytest

from backend.tests.lightweight_imports import load_arxiv_utils_module


resolver = load_arxiv_utils_module("paper_reference_resolver")


def _paper(arxiv_id: str | None, title: str) -> dict[str, object]:
    payload = {"title": title}
    if arxiv_id is not None:
        payload["arxiv_id"] = arxiv_id
    return payload


def test_resolves_explicit_arxiv_id_from_recent_papers() -> None:
    context = {"last_papers": [_paper("2401.00001", "Paper A"), _paper("2401.00002", "Paper B")]}

    result = resolver._resolve_paper_reference("请打开 arXiv 2401.00002", context)

    assert result["status"] == "success"
    assert result["matched_from"] == "explicit_arxiv_id"
    assert result["arxiv_id"] == "2401.00002"
    assert result["title"] == "Paper B"


def test_resolves_url_style_arxiv_id() -> None:
    result = resolver._resolve_paper_reference("看看 https://arxiv.org/abs/2401.12345v2 这篇", context={})

    assert result["status"] == "success"
    assert result["matched_from"] == "explicit_arxiv_id"
    assert result["arxiv_id"] == "2401.12345v2"
    assert result["paper"] is None


@pytest.mark.parametrize(
    ("message", "expected_id"),
    [
        ("第一篇论文", "2401.00001"),
        ("第二篇论文", "2401.00002"),
        ("最后一篇论文", "2401.00003"),
    ],
)
def test_resolves_ordinal_references_from_last_papers(message: str, expected_id: str) -> None:
    context = {
        "last_papers": [
            _paper("2401.00001", "Paper A"),
            _paper("2401.00002", "Paper B"),
            _paper("2401.00003", "Paper C"),
        ]
    }

    result = resolver._resolve_paper_reference(message, context)

    assert result["status"] == "success"
    assert result["matched_from"] == "last_papers"
    assert result["arxiv_id"] == expected_id


def test_context_reference_prefers_selected_paper() -> None:
    context = {
        "last_papers": [_paper("2401.00001", "Paper A"), _paper("2401.00002", "Paper B")],
        "selected_paper": _paper("2401.00002", "Paper B"),
    }

    result = resolver._resolve_paper_reference("这篇论文讲了什么？", context)

    assert result["status"] == "success"
    assert result["matched_from"] == "selected_paper"
    assert result["target"]["target_type"] == "context_paper"
    assert result["arxiv_id"] == "2401.00002"


def test_no_reference_falls_back_to_selected_paper() -> None:
    context = {"selected_paper": _paper("2401.00009", "Selected Paper")}

    result = resolver._resolve_paper_reference("请做个总结", context)

    assert result["status"] == "success"
    assert result["matched_from"] == "selected_paper"
    assert result["target"] is None
    assert result["arxiv_id"] == "2401.00009"


def test_no_reference_falls_back_to_single_recent_paper() -> None:
    context = {"last_papers": [_paper("2401.00010", "Only Paper")]}

    result = resolver._resolve_paper_reference("请做个总结", context)

    assert result["status"] == "success"
    assert result["matched_from"] == "single_recent_paper"
    assert result["arxiv_id"] == "2401.00010"


def test_context_reference_falls_back_to_single_recent_paper() -> None:
    context = {"last_papers": [_paper("2401.00011", "Only Recent Paper")]}

    result = resolver._resolve_paper_reference("这篇论文值得读吗？", context)

    assert result["status"] == "success"
    assert result["matched_from"] == "single_recent_paper"
    assert result["arxiv_id"] == "2401.00011"


def test_ordinal_reference_rejects_out_of_range_selection() -> None:
    context = {"last_papers": [_paper("2401.00001", "Paper A"), _paper("2401.00002", "Paper B")]}

    result = resolver._resolve_paper_reference("第三篇论文", context)

    assert result["status"] == "failed"
    assert "只有 2 篇" in result["reason"]
    assert result["target"]["target_type"] == "ordinal"


def test_ordinal_reference_rejects_paper_without_arxiv_id() -> None:
    context = {"last_papers": [_paper(None, "Paper Without ID")]}

    result = resolver._resolve_paper_reference("第一篇论文", context)

    assert result["status"] == "failed"
    assert result["paper"]["title"] == "Paper Without ID"
    assert "缺少 arXiv ID" in result["reason"]
