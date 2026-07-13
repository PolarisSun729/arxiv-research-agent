from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse
from pydantic_core import PydanticUndefined


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
agent_router = _MODULES["router_module"]


def _build_model(model_cls, **overrides):
    payload = {}
    for field_name, field_info in model_cls.model_fields.items():
        if field_name in overrides:
            payload[field_name] = overrides[field_name]
        elif field_info.default_factory is not None:
            payload[field_name] = field_info.default_factory()
        elif field_info.default is not PydanticUndefined:
            payload[field_name] = field_info.default
        else:
            payload[field_name] = None
    return model_cls.model_construct(**payload)


class AgentRouterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(agent_router.router, prefix="/api")
        self.client = TestClient(app)

    def test_chat_endpoint_returns_agent_response(self) -> None:
        response_model = _build_model(
            agent_router.ArxivSearchResponse,
            user_id="u1",
            session_id="s1",
            answer="agent answer",
            plan=[],
            next_actions=["follow up"],
            papers=[],
            tool_calls=[],
            steps=[],
            warnings=[],
            errors=[],
            debug={},
        )

        with mock.patch.object(agent_router, "run_arxiv_search_agent", return_value=response_model) as mocked:
            response = self.client.post("/api/agent/chat", json={"user_id": "u1", "session_id": "s1", "message": "search rag"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "agent answer")
        mocked.assert_called_once()

    def test_stream_endpoint_returns_sse_response(self) -> None:
        stream_response = StreamingResponse(iter([b'data: {"event_type":"run_start"}\n\n']), media_type="text/event-stream")

        with mock.patch.object(agent_router, "stream_arxiv_search_agent", return_value=stream_response) as mocked:
            response = self.client.post("/api/agent/chat/stream", json={"user_id": "u1", "message": "search rag"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("run_start", response.text)
        mocked.assert_called_once()

    def test_graph_endpoint_returns_graph_payload(self) -> None:
        graph_model = _build_model(
            agent_router.ArxivSearchGraphResponse,
            mermaid="graph TD\n    parse_search_request --> synthesize_response",
            node_names=["parse_search_request", "synthesize_response"],
            edges=[["parse_search_request", "synthesize_response"]],
        )

        with mock.patch.object(agent_router, "export_arxiv_search_graph_mermaid", return_value=graph_model) as mocked:
            response = self.client.get("/api/agent/graph")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("mermaid", payload)
        self.assertIn("node_names", payload)
        mocked.assert_called_once()

    def test_active_work_continuations_are_scoped_to_request_actor(self) -> None:
        service = mock.Mock()
        service.list_active.return_value = [
            {
                "continuation_id": "continuation-1",
                "session_id": "s1",
                "status": "waiting_job",
                "job": {"status": "running", "progress": 45, "current_stage": "chunk_document"},
            }
        ]
        self.client.app.dependency_overrides[agent_router.get_agent_work_continuation_service] = lambda: service

        response = self.client.get(
            "/api/agent/work-continuations/active",
            params={"user_id": "u1", "session_id": "s1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["job"]["progress"], 45)
        service.list_active.assert_called_once_with(user_id="u1", session_id="s1")

    def test_resume_stream_returns_persisted_run_result(self) -> None:
        manager = mock.Mock()
        manager.claim_and_start.return_value = {
            "resume_run_id": "run-1",
            "status": "completed",
            "created": False,
        }
        manager.get.return_value = {
            "resume_run_id": "run-1",
            "status": "completed",
            "final_response": {"session_id": "s1", "intent": "paper_qa", "answer": "原问题答案"},
        }
        self.client.app.dependency_overrides[agent_router.get_agent_resume_run_manager] = lambda: manager

        response = self.client.post(
            "/api/agent/work-continuations/continuation-1/resume/stream",
            params={"user_id": "u1", "session_id": "s1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: final_response", response.text)
        self.assertIn("原问题答案", response.text)
        manager.claim_and_start.assert_called_once_with("continuation-1", user_id="u1", session_id="s1")
        manager.mark_result_retrieved.assert_called_once_with("run-1")

    def test_chat_endpoint_maps_service_http_exception(self) -> None:
        with mock.patch.object(agent_router, "run_arxiv_search_agent", side_effect=HTTPException(status_code=409, detail="confirmation required")):
            response = self.client.post("/api/agent/chat", json={"user_id": "u1", "message": "approve this"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "confirmation required")

    def test_chat_endpoint_rejects_missing_message(self) -> None:
        response = self.client.post("/api/agent/chat", json={"user_id": "u1"})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
