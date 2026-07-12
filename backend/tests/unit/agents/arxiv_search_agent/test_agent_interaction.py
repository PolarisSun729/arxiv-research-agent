import pytest
from pydantic import ValidationError

from agents.arxiv_search_agent.execution.interactions import InteractionResumeRequest


def test_resume_request_uses_only_business_interaction_identity() -> None:
    request = InteractionResumeRequest(
        interaction_id="interaction-1",
        decision="approve",
        response={},
    )

    assert request.model_dump() == {
        "interaction_id": "interaction-1",
        "decision": "approve",
        "response": {},
        "note": None,
    }


def test_resume_request_rejects_internal_step_and_tool_locators() -> None:
    with pytest.raises(ValidationError):
        InteractionResumeRequest.model_validate(
            {
                "interaction_id": "interaction-1",
                "decision": "approve",
                "step_id": "step-1",
                "tool_name": "parse_and_index_paper",
                "interrupt_id": "langgraph-internal-id",
            }
        )
