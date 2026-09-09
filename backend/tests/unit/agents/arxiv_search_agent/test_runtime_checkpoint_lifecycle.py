"""无显式节点名的保存与终态补写不能读取不存在的提取函数或展示 debug。"""

from types import SimpleNamespace

import pytest

from agents.arxiv_search_agent.runtime_checkpoint import AgentRuntimeCheckpointManager


@pytest.mark.parametrize("operation", ["persist", "terminal"])
def test_checkpoint_without_node_can_be_written(operation):
    writes = []
    store = SimpleNamespace(
        upsert_agent_runtime_checkpoint=lambda **payload: writes.append(payload),
        mark_agent_runtime_checkpoint_status=lambda **payload: False,
    )
    manager = AgentRuntimeCheckpointManager(store)
    state = {"user_id": "offline", "session_id": "session", "plan_runtime": {"turn_status": "success"},
             "debug": {"current_node": "untrusted_display"}}
    if operation == "persist":
        manager.persist_state(state)
    else:
        manager.mark_terminal(state, status="completed")
    assert writes[0]["status"] == "completed"
    assert writes[0]["current_node"] is None


def test_explicit_graph_node_is_preserved():
    writes = []
    manager = AgentRuntimeCheckpointManager(SimpleNamespace(
        upsert_agent_runtime_checkpoint=lambda **payload: writes.append(payload),
    ))
    manager.persist_state({"user_id": "offline", "session_id": "session"}, current_node="execute_step")
    assert writes[0]["current_node"] == "execute_step"
