from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_agent_runtime_helper():
    helper_path = Path(__file__).resolve().parents[1] / "helpers" / "agent_runtime.py"
    module_name = "backend.tests.helpers.agent_runtime"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


load_agent_test_modules = _load_agent_runtime_helper().load_agent_test_modules


_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
service_module = _MODULES["service_module"]
AgentState = _MODULES["state_module"].AgentState
ArxivSearchRequest = schemas.ArxivSearchRequest
AgentStep = schemas.AgentStep
AgentToolCall = schemas.AgentToolCall


class _FakeCompiledGraph:
    def __init__(self, *, final_state=None, updates=None):
        self.final_state = final_state
        self.updates = list(updates or [])

    def invoke(self, _state):
        return self.final_state

    def stream(self, _state, stream_mode="updates"):
        del stream_mode
        for update in self.updates:
            yield update


def _step(step_name: str) -> AgentStep:
    return AgentStep(step=step_name, status="success", action=step_name, inputs={}, outputs={})


class AgentChatFlowIntegrationTests(unittest.TestCase):
    def test_run_arxiv_search_agent_returns_main_flow_response(self) -> None:
        final_state = AgentState(
            user_id="u1",
            session_id="s1",
            message="search rag",
            intent="arxiv_search",
            answer="agent final answer",
            papers=[{"arxiv_id": "2401.00001", "title": "RAG Paper"}],
            next_actions=["read the first paper"],
        )

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=_FakeCompiledGraph(final_state=final_state.model_dump())), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ):
            response = service_module.run_arxiv_search_agent(
                ArxivSearchRequest(user_id="u1", session_id="s1", message="search rag")
            )

        self.assertEqual(response.answer, "agent final answer")
        self.assertEqual(response.papers[0]["arxiv_id"], "2401.00001")
        self.assertEqual(response.next_actions, ["read the first paper"])

    def test_stream_arxiv_search_agent_emits_ordered_events(self) -> None:
        state_parse = AgentState(
            user_id="u1",
            session_id="s1",
            message="search rag",
            intent="arxiv_search",
            steps=[_step("parse_search_request")],
        )
        state_build = state_parse.model_copy(deep=True)
        state_build.tool_name = "search_arxiv_structured"
        state_build.tool_args = {"query": "rag", "max_results": 5}
        state_build.steps = [_step("build_search_tool_args")]

        state_invoke = state_build.model_copy(deep=True)
        state_invoke.tool_calls = [
            AgentToolCall(
                tool_name="search_arxiv_structured",
                arguments={"query": "rag", "max_results": 5},
                status="success",
                summary="search finished",
                trace={"returned_count": 1},
                error=None,
            )
        ]
        state_invoke.steps = [_step("invoke_search_tool")]
        state_invoke.papers = [{"arxiv_id": "2401.00001", "title": "RAG Paper"}]

        state_final = state_invoke.model_copy(deep=True)
        state_final.answer = "agent final answer"
        state_final.next_actions = ["read the first paper"]
        state_final.steps = [_step("synthesize_response")]

        fake_graph = _FakeCompiledGraph(
            updates=[
                {"parse_search_request": state_parse.model_dump()},
                {"build_search_tool_args": state_build.model_dump()},
                {"invoke_search_tool": state_invoke.model_dump()},
                {"synthesize_response": state_final.model_dump()},
            ]
        )

        app = FastAPI()

        @app.post("/stream")
        async def _stream_endpoint(request: ArxivSearchRequest):
            return service_module.stream_arxiv_search_agent(request)

        client = TestClient(app)

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=fake_graph), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ):
            response = client.post("/stream", json={"user_id": "u1", "session_id": "s1", "message": "search rag"})

        self.assertEqual(response.status_code, 200)
        payload_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        events = [json.loads(line) for line in payload_lines]
        event_types = [event["event_type"] for event in events]

        self.assertEqual(
            event_types,
            [
                "run_start",
                "step_start",
                "step_end",
                "step_start",
                "step_end",
                "step_start",
                "tool_call_start",
                "step_end",
                "tool_call_end",
                "step_start",
                "step_end",
                "final_response",
                "stream_end",
            ],
        )
        self.assertEqual(events[-2]["data"]["response"]["answer"], "agent final answer")


if __name__ == "__main__":
    unittest.main()
