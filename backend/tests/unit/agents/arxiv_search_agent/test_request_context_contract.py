from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.helpers.agent_runtime import load_agent_test_modules


ArxivSearchRequest = load_agent_test_modules()["schemas"].ArxivSearchRequest


def test_request_accepts_only_structured_frontend_visible_paper() -> None:
    request = ArxivSearchRequest(
        message="这篇论文讲了什么",
        context={
            "frontend_visible_paper": {
                "arxiv_id": "2401.00001",
                "title": "Visible paper",
            }
        },
    )

    assert request.context["frontend_visible_paper"]["arxiv_id"] == "2401.00001"


@pytest.mark.parametrize(
    "value",
    [None, "2401.00001", {"title": "Missing ID"}, {"arxiv_id": "2401.00001", "title": 3}],
)
def test_request_rejects_invalid_frontend_visible_paper(value: object) -> None:
    with pytest.raises(ValidationError):
        ArxivSearchRequest(
            message="这篇论文讲了什么",
            context={"frontend_visible_paper": value},
        )
