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
Command = sys.modules["langgraph.types"].Command


class _FakeCompiledGraph:
    def __init__(self, *, final_state=None, updates=None, checkpoint_exists=True):
        self.final_state = final_state
        self.updates = list(updates or [])
        self.checkpoint_exists = checkpoint_exists
        self.last_invoke_config = None
        self.last_stream_config = None
        self.last_invoke_input = None
        self.last_stream_input = None

    def invoke(self, _state, config=None):
        self.last_invoke_config = config
        self.last_invoke_input = _state
        return self.final_state

    def stream(self, _state, config=None, stream_mode="updates"):
        self.last_stream_config = config
        self.last_stream_input = _state
        del stream_mode
        for update in self.updates:
            yield update

    def get_state(self, config=None):
        del config
        return {"checkpoint": "exists"} if self.checkpoint_exists else None


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
        fake_graph = _FakeCompiledGraph(final_state=final_state.model_dump())

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=fake_graph), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ):
            response = service_module.run_arxiv_search_agent(
                ArxivSearchRequest(user_id="u1", session_id="s1", message="search rag")
            )

        self.assertEqual(response.answer, "agent final answer")
        self.assertEqual(response.session_id, "s1")
        self.assertEqual(response.papers[0]["arxiv_id"], "2401.00001")
        self.assertEqual(response.next_actions, ["read the first paper"])
        self.assertEqual(fake_graph.last_invoke_config, {"configurable": {"thread_id": "s1"}})

    def test_run_arxiv_search_agent_generates_session_id_when_missing(self) -> None:
        final_state = AgentState(
            user_id="u1",
            session_id="generated-session",
            message="search rag",
            intent="arxiv_search",
            answer="agent final answer",
        )
        fake_graph = _FakeCompiledGraph(final_state=final_state.model_dump())

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=fake_graph), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ), mock.patch.object(service_module, "uuid4", return_value="generated-session"):
            response = service_module.run_arxiv_search_agent(
                ArxivSearchRequest(user_id="u1", session_id=None, message="search rag")
            )

        self.assertEqual(response.session_id, "generated-session")
        self.assertEqual(fake_graph.last_invoke_config, {"configurable": {"thread_id": "generated-session"}})

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
        self.assertEqual(events[0]["data"]["request"]["session_id"], "s1")
        self.assertEqual(fake_graph.last_stream_config, {"configurable": {"thread_id": "s1"}})

    def test_run_arxiv_search_agent_keeps_legacy_pending_action_compatible_for_legacy_ui(self) -> None:
        final_state = AgentState(
            user_id="u1",
            session_id="s1",
            message="search rag",
            intent="paper_qa",
            answer="",
            pending_action={
                "type": "tool_approval",
                "status": "waiting_confirmation",
                "step_id": "request_confirmation",
                "tool_name": "request_confirmation",
                "allowed_decisions": ["approve", "reject"],
                "confirmation_request": {
                    "request_type": "tool_approval",
                    "step_id": "request_confirmation",
                    "tool_name": "request_confirmation",
                    "action_type": "clarify",
                    "side_effect_level": "session_write",
                    "reason": "waiting_for_user_confirmation",
                    "title": "确认是否解析论文",
                    "description": "需要先确认是否创建 QA 索引。",
                    "arguments_summary": {"arxiv_id": "2401.00001", "qa_question": "what is the method?"},
                    "original_question": "what is the method?",
                    "target_paper": {"arxiv_id": "2401.00001", "title": "RAG Paper"},
                    "allowed_decisions": [
                        {"code": "approve", "label": "批准", "description": "继续执行当前工具操作"},
                        {"code": "reject", "label": "拒绝", "description": "取消当前工具操作"},
                    ],
                    "allow_argument_edit": False,
                    "allow_reject": True,
                    "allow_note": True,
                    "trace_id": "goal-1",
                    "plan_id": "plan-1",
                    "session_id": "s1",
                    "thread_id": "s1",
                },
            },
            paper_qa_result={"status": "waiting_confirmation"},
        )
        fake_graph = _FakeCompiledGraph(final_state=final_state.model_dump())

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=fake_graph), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ):
            response = service_module.run_arxiv_search_agent(
                ArxivSearchRequest(user_id="u1", session_id="s1", message="search rag")
            )

        assert response.pending_action is not None
        assert response.pending_action["status"] == "waiting_confirmation"
        assert response.pending_action["allowed_decisions"] == ["approve", "reject"]
        json.dumps(response.pending_action, ensure_ascii=False)

    def test_run_arxiv_search_agent_resume_uses_command_with_same_thread_id(self) -> None:
        final_state = AgentState(
            user_id="u1",
            session_id="s1",
            message="approve",
            intent="paper_qa",
            answer="resume finished",
        )
        fake_graph = _FakeCompiledGraph(final_state=final_state.model_dump(), checkpoint_exists=True)

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=fake_graph), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ):
            response = service_module.run_arxiv_search_agent(
                ArxivSearchRequest(
                    user_id="u1",
                    session_id="s1",
                    message="approve",
                    resume={"decision": "approve", "step_id": "parse_and_index_paper"},
                )
            )

        self.assertEqual(response.answer, "resume finished")
        self.assertIsInstance(fake_graph.last_invoke_input, Command)
        self.assertEqual(fake_graph.last_invoke_input.resume, {"decision": "approve", "step_id": "parse_and_index_paper"})
        self.assertEqual(fake_graph.last_invoke_config, {"configurable": {"thread_id": "s1"}})

    def test_run_arxiv_search_agent_resume_reject_uses_same_thread_id(self) -> None:
        final_state = AgentState(
            user_id="u1",
            session_id="s1",
            message="reject",
            intent="paper_qa",
            answer="cancelled",
        )
        fake_graph = _FakeCompiledGraph(final_state=final_state.model_dump(), checkpoint_exists=True)

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=fake_graph), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ):
            response = service_module.run_arxiv_search_agent(
                ArxivSearchRequest(
                    user_id="u1",
                    session_id="s1",
                    message="reject",
                    resume={"decision": "reject", "step_id": "parse_and_index_paper"},
                )
            )

        self.assertEqual(response.answer, "cancelled")
        self.assertIsInstance(fake_graph.last_invoke_input, Command)
        self.assertEqual(fake_graph.last_invoke_input.resume, {"decision": "reject", "step_id": "parse_and_index_paper"})
        self.assertEqual(fake_graph.last_invoke_config, {"configurable": {"thread_id": "s1"}})

    def test_run_arxiv_search_agent_resume_with_missing_checkpoint_returns_error(self) -> None:
        fake_graph = _FakeCompiledGraph(final_state=None, checkpoint_exists=False)

        with mock.patch.object(service_module, "build_arxiv_search_graph", return_value=fake_graph), mock.patch.object(
            service_module,
            "_persist_agent_session_memory",
            return_value=None,
        ):
            response = service_module.run_arxiv_search_agent(
                ArxivSearchRequest(
                    user_id="u1",
                    session_id="missing-session",
                    message="reject",
                    resume={"decision": "reject", "step_id": "parse_and_index_paper"},
                )
            )

        self.assertEqual(response.answer, "arXiv 搜索 Agent 运行失败")
        self.assertIn("未找到可恢复的执行现场", response.debug["runtime_error"]["detail"])

    def test_stream_arxiv_search_agent_resume_uses_command_with_same_thread_id(self) -> None:
        resumed_state = AgentState(
            user_id="u1",
            session_id="s1",
            message="approve",
            intent="paper_qa",
            answer="resume streamed",
            steps=[_step("run_agent_turn")],
        )
        fake_graph = _FakeCompiledGraph(
            updates=[{"run_agent_turn": resumed_state.model_dump()}],
            checkpoint_exists=True,
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
            response = client.post(
                "/stream",
                json={
                    "user_id": "u1",
                    "session_id": "s1",
                    "message": "approve",
                    "resume": {"decision": "approve", "step_id": "parse_and_index_paper"},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(fake_graph.last_stream_input, Command)
        self.assertEqual(fake_graph.last_stream_input.resume, {"decision": "approve", "step_id": "parse_and_index_paper"})
        self.assertEqual(fake_graph.last_stream_config, {"configurable": {"thread_id": "s1"}})


if __name__ == "__main__":
    unittest.main()
