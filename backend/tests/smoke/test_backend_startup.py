from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock

from fastapi.testclient import TestClient


def _ensure_langgraph_stub() -> None:
    """为启动烟测提供最小 LangGraph 桩，避免 CI 因图运行时缺失而误触真实外部依赖。"""

    if "langgraph.graph" in sys.modules and "langgraph.types" in sys.modules:
        return

    langgraph_module = types.ModuleType("langgraph")
    graph_module = types.ModuleType("langgraph.graph")
    types_module = types.ModuleType("langgraph.types")
    checkpoint_module = types.ModuleType("langgraph.checkpoint")
    checkpoint_memory_module = types.ModuleType("langgraph.checkpoint.memory")

    class _MemorySaver:
        pass

    class _Command:
        def __init__(self, *, resume=None):
            self.resume = resume

    class _CompiledGraph:
        def get_graph(self):
            return types.SimpleNamespace(draw_mermaid=lambda: "graph TD\n    START --> END")

    class _StateGraph:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_node(self, *_args, **_kwargs):
            return None

        def add_edge(self, *_args, **_kwargs):
            return None

        def add_conditional_edges(self, *_args, **_kwargs):
            return None

        def compile(self, *_args, **_kwargs):
            return _CompiledGraph()

    graph_module.START = "START"
    graph_module.END = "END"
    graph_module.StateGraph = _StateGraph
    types_module.Command = _Command
    types_module.interrupt = lambda _payload: None
    checkpoint_memory_module.MemorySaver = _MemorySaver
    checkpoint_memory_module.InMemorySaver = _MemorySaver

    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module
    sys.modules["langgraph.types"] = types_module
    sys.modules["langgraph.checkpoint"] = checkpoint_module
    sys.modules["langgraph.checkpoint.memory"] = checkpoint_memory_module


class _FakeArxivService:
    def get_available_fields(self):
        return [{"field": "title", "prefix": "ti"}]

    def get_subject_categories(self):
        return [{"id": "cs.CL", "name": "Computation and Language"}]


class _FakeDatabaseService:
    def get_user_labeled_paper_count(self, user_id: str = "") -> int:
        return 0

    def get_paper(self, arxiv_id: str):
        return None


class _FakeOaiDatabaseService:
    def get_total_paper_count(self) -> int:
        return 0


class _FakePaperQAService:
    def get_qa_status(self, arxiv_id: str):
        return {"arxiv_id": arxiv_id, "status": "ready", "chunk_count": 0}


class _FakeRecommendationService:
    def _fetch_paper_from_arxiv_with_rate_limit(self, arxiv_id: str):
        return None

    def _materialize_paper_from_source(self, source_paper, arxiv_id: str):
        payload = dict(source_paper)
        payload.setdefault("arxiv_id", arxiv_id)
        return payload


class BackendStartupSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_langgraph_stub()
        cls.dependencies = importlib.import_module("dependencies")
        cls.main = importlib.import_module("main")

    def setUp(self) -> None:
        self.app = self.main.create_app(load_mode="lazy")
        self.app.dependency_overrides[self.dependencies.get_arxiv_service] = lambda: _FakeArxivService()
        self.app.dependency_overrides[self.dependencies.get_database_service] = lambda: _FakeDatabaseService()
        self.app.dependency_overrides[self.dependencies.get_oai_database_service] = lambda: _FakeOaiDatabaseService()
        self.app.dependency_overrides[self.dependencies.get_paper_qa_service] = lambda: _FakePaperQAService()
        self.app.dependency_overrides[self.dependencies.get_recommendation_service] = lambda: _FakeRecommendationService()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    def test_create_app_lazy_registers_expected_router_prefixes(self) -> None:
        routes = [getattr(route, "path", "") for route in self.app.routes]

        self.assertGreater(len(routes), 0)
        expected_paths = {
            "/api/arxiv/fields",
            "/api/agent/graph",
            "/api/stats",
            "/api/sync-status",
            "/api/paper/{arxiv_id}/qa-status",
            "/api/user/preferences/{user_id}",
            "/api/chunks/files",
        }
        self.assertTrue(expected_paths.issubset(set(routes)))

    def test_basic_api_requests_return_stable_payloads_without_external_services(self) -> None:
        fields_response = self.client.get("/api/arxiv/fields")
        stats_response = self.client.get("/api/stats")
        sync_response = self.client.get("/api/sync-status")
        qa_status_response = self.client.get("/api/paper/2401.00001/qa-status")

        self.assertEqual(fields_response.status_code, 200)
        self.assertEqual(fields_response.json()["fields"][0]["prefix"], "ti")
        self.assertEqual(stats_response.status_code, 200)
        self.assertIn("totalPapers", stats_response.json())
        self.assertEqual(sync_response.status_code, 200)
        self.assertIn("status", sync_response.json())
        self.assertEqual(qa_status_response.status_code, 200)
        self.assertEqual(qa_status_response.json()["status"], "ready")

    def test_global_http_exception_handler_returns_unified_error_payload(self) -> None:
        response = self.client.get("/api/paper/2401.404")

        self.assertEqual(response.status_code, 404)
        payload = response.json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["code"], "paper_not_found")
        self.assertIn("message", payload)
        self.assertIn("recoverable", payload)

    def test_request_validation_error_handler_returns_unified_error_payload(self) -> None:
        response = self.client.post("/api/arxiv/download", json={"arxiv_id": "2401.00001"})

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["code"], "request_validation_error")
        self.assertTrue(payload["recoverable"])

    def test_lazy_app_creation_does_not_warm_up_heavy_services(self) -> None:
        with mock.patch.object(self.dependencies, "iter_service_getters", side_effect=AssertionError("should stay lazy")):
            warmed = self.dependencies.warm_up_services(load_mode="lazy")

        self.assertEqual(warmed, [])

    def test_lazy_lifespan_does_not_preload_heavy_services(self) -> None:
        app = self.main.create_app(load_mode="lazy")
        app.dependency_overrides[self.dependencies.get_arxiv_service] = lambda: _FakeArxivService()

        # 进入 TestClient 上下文才会真正触发 FastAPI lifespan；这里验证 lazy 启动不会走预热分支。
        with mock.patch.object(self.main, "warm_up_services", side_effect=AssertionError("should stay lazy")):
            with TestClient(app) as client:
                response = client.get("/api/arxiv/fields")

        app.dependency_overrides.clear()
        self.assertEqual(response.status_code, 200)

    def test_dependency_getters_are_enumerable_without_initializing_them(self) -> None:
        getters = self.dependencies.iter_service_getters()

        self.assertGreaterEqual(len(getters), 10)
        self.assertTrue(all(isinstance(name, str) and callable(getter) for name, getter in getters))
        self.assertIn("paper_qa_service", {name for name, _getter in getters})


if __name__ == "__main__":
    unittest.main()
