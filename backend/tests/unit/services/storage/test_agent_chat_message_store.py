import json
import sqlite3

from tests.helpers.sqlite import build_storage_container


def test_agent_chat_messages_replay_in_order(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "agent-chat.sqlite"))
    store = storage.agent_chat_messages

    store.append_agent_chat_message("session-1", "user", "找 RAG 论文", user_id="user-1")
    store.append_agent_chat_message(
        "session-1",
        "assistant",
        "找到 1 篇",
        user_id="user-1",
        response={"answer": "找到 1 篇", "papers": [{"arxiv_id": "2401.00001"}]},
    )
    store.append_agent_chat_message(
        "session-1",
        "assistant",
        "索引完成",
        user_id="user-1",
        response={"answer": "索引完成"},
        resume_run_id="resume-1",
    )

    messages = store.list_agent_chat_messages("session-1", user_id="user-1")

    assert [item["role"] for item in messages] == ["user", "assistant", "assistant"]
    assert messages[0]["response"] is None
    assert messages[1]["response"]["papers"] == [{"arxiv_id": "2401.00001"}]
    assert messages[2]["resume_run_id"] == "resume-1"


def _full_response() -> dict:
    return {
        "session_id": "session-1",
        "intent": "arxiv_search",
        "answer": "找到 1 篇",
        "search_spec": {"query": "RAG"},
        "papers": [{"arxiv_id": "2401.00001"}],
        "runtime_state": {"trace": ["x" * 1000]},
        "plan_runtime": {"trace": ["x" * 1000]},
        "execution_plan": {"steps": []},
        "tool_calls": [{"tool_name": "search_arxiv"}],
        "steps": [{"name": "search"}],
        "debug": {"planner": {"raw": "x" * 1000}, "runtime_error": {"code": "agent_runtime_error"}},
    }


def test_agent_chat_messages_store_only_replay_fields(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "agent-chat.sqlite"))
    store = storage.agent_chat_messages
    store.append_agent_chat_message("session-1", "assistant", "找到 1 篇", user_id="user-1", response=_full_response())

    response = store.list_agent_chat_messages("session-1", user_id="user-1")[0]["response"]

    assert response == {
        "session_id": "session-1",
        "intent": "arxiv_search",
        "answer": "找到 1 篇",
        "search_spec": {"query": "RAG"},
        "papers": [{"arxiv_id": "2401.00001"}],
        "debug": {"runtime_error": {"code": "agent_runtime_error"}},
    }


def test_schema_slims_legacy_full_responses(tmp_path) -> None:
    db_path = str(tmp_path / "agent-chat.sqlite")
    build_storage_container(db_path=db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO agent_chat_messages (message_id, session_id, user_id, role, content, response_json) "
            "VALUES ('m1', 'session-1', 'user-1', 'assistant', '找到 1 篇', ?)",
            (json.dumps(_full_response(), ensure_ascii=False),),
        )

    store = build_storage_container(db_path=db_path).agent_chat_messages
    response = store.list_agent_chat_messages("session-1", user_id="user-1")[0]["response"]

    assert "runtime_state" not in response
    assert "tool_calls" not in response
    assert response["papers"] == [{"arxiv_id": "2401.00001"}]
    assert response["debug"] == {"runtime_error": {"code": "agent_runtime_error"}}


def test_agent_chat_messages_are_isolated_by_user(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "agent-chat.sqlite"))
    store = storage.agent_chat_messages
    store.append_agent_chat_message("session-1", "user", "alice 的问题", user_id="alice")

    assert store.list_agent_chat_messages("session-1", user_id="bob") == []
    assert store.list_agent_chat_sessions("bob") == []


def test_agent_chat_sessions_use_first_user_message_as_title_and_latest_first(tmp_path) -> None:
    storage = build_storage_container(db_path=str(tmp_path / "agent-chat.sqlite"))
    store = storage.agent_chat_messages
    store.append_agent_chat_message("session-old", "user", "第一个问题", user_id="user-1")
    store.append_agent_chat_message("session-old", "assistant", "回答", user_id="user-1")
    store.append_agent_chat_message("session-old", "user", "追问", user_id="user-1")
    store.append_agent_chat_message("session-new", "user", "新会话", user_id="user-1")

    sessions = store.list_agent_chat_sessions("user-1")

    assert [item["session_id"] for item in sessions] == ["session-new", "session-old"]
    assert sessions[1]["title"] == "第一个问题"
    assert sessions[1]["message_count"] == 3
