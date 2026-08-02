from __future__ import annotations

import pytest

from services.context_merge import merge_backend_authoritative_context


def test_frontend_visible_paper_is_the_only_frontend_paper_candidate() -> None:
    result = merge_backend_authoritative_context(
        backend_context={
            "selected_paper": {"arxiv_id": "2401.00001", "title": "Backend focus"},
            "active_arxiv_id": "2401.00001",
        },
        frontend_context={
            "frontend_visible_paper": {"arxiv_id": "2401.00002", "title": "Visible paper"},
        },
    )

    assert result.merged_context["selected_paper"]["arxiv_id"] == "2401.00001"
    assert result.merged_context["frontend_visible_paper"]["arxiv_id"] == "2401.00002"
    assert result.debug["frontend_accepted_fields"] == {"frontend_visible_paper": "frontend_visible_paper"}


def test_legacy_frontend_paper_fields_are_not_reinterpreted_as_visible_paper() -> None:
    result = merge_backend_authoritative_context(
        backend_context={},
        frontend_context={
            "selected_paper": {"arxiv_id": "2401.00001"},
            "arxiv_id": "2401.00001",
        },
    )

    assert "frontend_visible_paper" not in result.merged_context
    assert result.debug["frontend_ignored_fields"] == {
        "selected_paper": "backend_authoritative",
        "arxiv_id": "backend_authoritative",
    }


@pytest.mark.parametrize(
    "value",
    [None, "2401.00001", {"title": "Missing ID"}],
)
def test_frontend_visible_paper_requires_a_structured_identity(value: object) -> None:
    with pytest.raises(ValueError):
        merge_backend_authoritative_context(
            backend_context={},
            frontend_context={"frontend_visible_paper": value},
        )
