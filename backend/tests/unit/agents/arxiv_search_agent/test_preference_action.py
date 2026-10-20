from __future__ import annotations

import importlib
import json
import sys

import pytest

from tests.helpers.agent_runtime import load_agent_test_modules

load_agent_test_modules()
preference_action = importlib.import_module("backend.agents.arxiv_search_agent.utils.preference_action")
preference_adapters = importlib.import_module("backend.agents.arxiv_search_agent.tool_adapters.preference")
tool_execution = importlib.import_module("backend.agents.arxiv_search_agent.execution.tool_execution")
parse_module = sys.modules["backend.agents.arxiv_search_agent.node.parse_node"]
AgentState = importlib.import_module("backend.agents.arxiv_search_agent.state").AgentState


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"action": "dislike"}, {"action": "dislike", "remove_scope": "none"}),
        ({"action": "LIKE", "remove_scope": "liked"}, {"action": "like", "remove_scope": "none"}),
        ({"action": "remove", "remove_scope": "disliked"}, {"action": "remove", "remove_scope": "disliked"}),
        # 只说"取消标记"时 LLM 可能不给范围，两边都删最保守。
        ({"action": "remove"}, {"action": "remove", "remove_scope": "both"}),
        ({"action": "remove", "remove_scope": "everything"}, {"action": "remove", "remove_scope": "both"}),
        ({"action": "favorite"}, None),
        (None, None),
        ("dislike", None),
    ],
)
def test_normalize_preference_operation(raw, expected) -> None:
    assert preference_action.normalize_preference_operation(raw) == expected


class _ScriptedLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)

    def complete_with_qwen(self, prompt: str, task_type: str | None = None, **_: object) -> str:
        return self.responses.pop(0)


def test_intent_recognition_carries_preference_direction_into_state() -> None:
    llm = _ScriptedLLM(json.dumps({"intent": "preference_action", "confidence": 0.95, "preference_operation": {"action": "dislike", "remove_scope": None}}))

    state = parse_module.parse_search_request({"message": "把这篇给我标记为不感兴趣"}, generation_service=llm)

    assert state.intent == "preference_action"
    assert state.preference_operation == {"action": "dislike", "remove_scope": "none"}


def test_preference_direction_is_dropped_for_other_intents() -> None:
    llm = _ScriptedLLM(json.dumps({"intent": "paper_qa", "confidence": 0.9, "preference_operation": {"action": "like"}}))

    state = parse_module.parse_search_request({"message": "这篇论文为什么这么设计"}, generation_service=llm)

    assert state.preference_operation is None


def test_executor_injects_preference_direction_from_state() -> None:
    service = tool_execution.ToolExecutionService(registry=None, backend_invoker=lambda *_args, **_kwargs: {})
    state = AgentState(message="把这篇给我标记为不感兴趣", user_id="u1", intent="preference_action", preference_operation={"action": "dislike", "remove_scope": "none"})

    payload = service._augment_input("update_preference_store", {"paper_reference": {"arxiv_id": "2610.22297"}}, state)

    assert payload["preference_operation"] == {"action": "dislike", "remove_scope": "none"}
    assert payload["user_id"] == "u1"


def _recording_backend(calls: list):
    def backend(tool_name: str, **kwargs):
        calls.append((tool_name, kwargs))
        return {"ok": True, "tool_name": tool_name, "data": {"message": "ok"}}

    return backend


def test_update_preference_store_writes_dislike_from_llm_direction() -> None:
    calls: list = []
    adapter = preference_adapters.UpdatePreferenceStoreAdapter(_recording_backend(calls))
    result = adapter.execute(
        preference_adapters.UpdatePreferenceStoreInput(
            paper_reference={"arxiv_id": "2610.22297"},
            message="把这篇给我标记为不感兴趣",
            preference_operation={"action": "dislike", "remove_scope": "none"},
        )
    )

    assert result.ok is True
    assert result.data.label == "disliked"
    assert calls == [("record_paper_preference", {"user_id": "", "arxiv_id": "2610.22297", "liked": False, "paper": {"arxiv_id": "2610.22297"}})]


def test_update_preference_store_removes_with_llm_scope() -> None:
    calls: list = []
    adapter = preference_adapters.UpdatePreferenceStoreAdapter(_recording_backend(calls))
    result = adapter.execute(
        preference_adapters.UpdatePreferenceStoreInput(
            paper_reference={"arxiv_id": "2610.22297"},
            preference_operation={"action": "remove", "remove_scope": "disliked"},
        )
    )

    assert result.ok is True
    assert calls == [("remove_paper_preference", {"user_id": "", "arxiv_id": "2610.22297", "remove_scope": "disliked"})]


def test_update_preference_store_refuses_without_llm_direction() -> None:
    calls: list = []
    adapter = preference_adapters.UpdatePreferenceStoreAdapter(_recording_backend(calls))
    # 原话写得再明确也不从里面猜方向；意图识别没给出方向（如 LLM 不可用走了规则兜底）就拒绝写入。
    result = adapter.execute(
        preference_adapters.UpdatePreferenceStoreInput(paper_reference={"arxiv_id": "2610.22297"}, message="把这篇给我标记为不感兴趣")
    )

    assert result.ok is False
    assert result.error.error_code == "preference_action_unrecognized"
    assert calls == []
