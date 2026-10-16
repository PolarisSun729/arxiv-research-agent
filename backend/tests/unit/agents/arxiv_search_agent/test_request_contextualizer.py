from __future__ import annotations

import importlib

from tests.helpers.agent_runtime import load_agent_test_modules

_MODULES = load_agent_test_modules()

contextualizer = importlib.import_module("backend.agents.arxiv_search_agent.request_contextualizer")
service_module = _MODULES["service_module"]


def test_recent_turns_keep_latest_messages_with_search_summary() -> None:
    messages = [{"role": "user", "content": f"问题 {index}"} for index in range(10)] + [
        {"role": "user", "content": "帮我找 RAG 的论文"},
        {
            "role": "assistant",
            "content": "答" * 1000,
            "response": {
                "search_spec": {
                    "concepts": [{"term": "RAG", "aliases": ["retrieval-augmented generation"], "source": "user"}],
                    "submitted_days_ago": 30,
                    "constraint_sources": {"submitted_days_ago": "user"},
                },
                "papers": [{"arxiv_id": "1"}, {"arxiv_id": "2"}],
            },
        },
        {"role": "assistant", "content": "   "},
    ]

    turns = contextualizer.build_recent_turns(messages)

    assert len(turns) == contextualizer.MAX_RECENT_MESSAGES - 1
    assert turns[-2] == {"role": "user", "content": "帮我找 RAG 的论文"}
    assistant = turns[-1]
    assert len(assistant["content"]) == contextualizer.ASSISTANT_EXCERPT_CHARS
    assert assistant["search"] == {"concepts": ["RAG"], "submitted_days_ago": 30}
    assert assistant["paper_count"] == 2


def test_contextualize_skips_without_history_and_rejects_empty_output() -> None:
    class _LLM:
        def complete_with_qwen(self, prompt, task_type=None, **_):
            return '{"standalone_request": ""}'

    skipped = contextualizer.contextualize_request("找 RAG 论文", [], _LLM())
    assert skipped["status"] == "skipped" and skipped["message"] == "找 RAG 论文"

    invalid = contextualizer.contextualize_request("再多找点", [{"role": "user", "content": "找 RAG 论文"}], _LLM())
    assert invalid["status"] == "invalid_output"
    assert invalid["message"] == "再多找点"
    assert invalid["rewritten"] is False


def test_attach_recent_turns_reads_session_history_and_tolerates_store_errors() -> None:
    class _Store:
        def __init__(self, fail: bool = False) -> None:
            self.fail = fail
            self.calls: list[tuple] = []

        def list_agent_chat_messages(self, session_id, user_id=None):
            self.calls.append((session_id, user_id))
            if self.fail:
                raise RuntimeError("db locked")
            return [{"role": "user", "content": "帮我找 RAG 的论文"}]

    class _Storage:
        def __init__(self, store) -> None:
            self.agent_chat_messages = store

    store = _Store()
    context = service_module._attach_recent_turns({"k": 1}, _Storage(store), session_id="s1", user_id="u1")
    assert store.calls == [("s1", "u1")]
    assert context == {"k": 1, "recent_turns": [{"role": "user", "content": "帮我找 RAG 的论文"}]}

    # 历史读不出来只是少了追问改写，本轮照常执行。
    failing = service_module._attach_recent_turns({"k": 1}, _Storage(_Store(fail=True)), session_id="s1", user_id="u1")
    assert failing == {"k": 1}
    assert service_module._attach_recent_turns({"k": 1}, _Storage(store), session_id=None, user_id="u1") == {"k": 1}
