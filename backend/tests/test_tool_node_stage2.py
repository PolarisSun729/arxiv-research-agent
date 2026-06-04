import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_tool_node_modules():
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root / "backend"
    agent_dir = backend_dir / "agents" / "arxiv_search_agent"
    utils_dir = agent_dir / "utils"
    node_dir = agent_dir / "node"

    packages = {
        "backend": backend_dir,
        "backend.agents": backend_dir / "agents",
        "backend.agents.arxiv_search_agent": agent_dir,
        "backend.agents.arxiv_search_agent.utils": utils_dir,
        "backend.agents.arxiv_search_agent.node": node_dir,
        "tools": backend_dir / "tools",
        "utils": backend_dir / "utils",
    }
    for package_name, package_path in packages.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]
        sys.modules[package_name] = module

    if "tools.tool_registry" not in sys.modules:
        tool_registry_module = types.ModuleType("tools.tool_registry")
        tool_registry_module.invoke_tool = lambda *args, **kwargs: None
        sys.modules["tools.tool_registry"] = tool_registry_module

    if "utils.config" not in sys.modules:
        config_module = types.ModuleType("utils.config")
        config_module.get_arxiv_oai_runtime_config = lambda: {"target_categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"]}
        sys.modules["utils.config"] = config_module

    def load(module_name: str, file_path: Path):
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    schemas = load("backend.agents.arxiv_search_agent.schemas", agent_dir / "schemas.py")
    state_module = load("backend.agents.arxiv_search_agent.state", agent_dir / "state.py")
    load("backend.agents.arxiv_search_agent.utils.result_utils", utils_dir / "result_utils.py")
    load("backend.agents.arxiv_search_agent.utils.state_utils", utils_dir / "state_utils.py")
    tool_node_module = load("backend.agents.arxiv_search_agent.node.tool_node", node_dir / "tool_node.py")
    return {
        "schemas": schemas,
        "state_module": state_module,
        "tool_node_module": tool_node_module,
    }


_MODULES = _load_tool_node_modules()
AgentState = _MODULES["state_module"].AgentState
ToolCallRequest = _MODULES["schemas"].ToolCallRequest
execute_tool = _MODULES["tool_node_module"].execute_tool
tool_node_module = _MODULES["tool_node_module"]


class ExecuteToolNodeStage2Tests(unittest.TestCase):
    def test_tool_call_request_supports_dump_and_validate(self) -> None:
        request = ToolCallRequest(
            tool_name="search_arxiv_structured",
            arguments={"query": "rag", "max_results": 3},
            reason="search candidate papers",
            expected_result="paper list",
            plan_step_id="step-2",
            fallback_tools=["search_arxiv_raw"],
        )

        payload = request.model_dump()
        restored = ToolCallRequest.model_validate(payload)

        self.assertEqual(payload["tool_name"], "search_arxiv_structured")
        self.assertEqual(restored.arguments["query"], "rag")
        self.assertEqual(restored.plan_step_id, "step-2")

    def test_execute_tool_skips_when_request_missing(self) -> None:
        result = execute_tool(AgentState())

        self.assertEqual(result.steps[-1].step, "execute_tool")
        self.assertEqual(result.steps[-1].status, "skipped")
        self.assertEqual(len(result.tool_observations), 1)
        self.assertEqual(result.tool_observations[-1].status, "skipped")
        self.assertEqual(result.tool_observations[-1].error["code"], "missing_tool_call_request")

    def test_execute_tool_generates_failed_observation_for_unknown_tool(self) -> None:
        state = AgentState(
            tool_call_request=ToolCallRequest(
                tool_name="unknown_tool",
                arguments={"query": "rag"},
                reason="search candidate papers",
                expected_result="paper list",
                plan_step_id="step-1",
            )
        )
        tool_result = {
            "ok": False,
            "tool_name": "unknown_tool",
            "summary": "Unknown tool",
            "data": None,
            "trace": {},
            "error": {
                "code": "tool_not_found",
                "message": "Unknown tool: unknown_tool",
            },
        }

        with patch.object(tool_node_module, "invoke_tool", return_value=tool_result) as mock_invoke:
            result = execute_tool(state)

        mock_invoke.assert_called_once_with("unknown_tool", query="rag")
        self.assertFalse(result.tool_observations[-1].ok)
        self.assertEqual(result.tool_observations[-1].status, "failed")
        self.assertEqual(result.tool_observations[-1].error["code"], "tool_not_found")
        self.assertEqual(result.tool_calls[-1].status, "failed")
        self.assertEqual(result.steps[-1].status, "failed")

    def test_execute_tool_generates_success_observation(self) -> None:
        state = AgentState(
            tool_call_request=ToolCallRequest(
                tool_name="search_arxiv_structured",
                arguments={"query": "rag", "max_results": 2},
                reason="search candidate papers",
                expected_result="paper list",
                plan_step_id="step-1",
                fallback_tools=["search_arxiv_raw"],
            )
        )
        tool_result = {
            "ok": True,
            "tool_name": "search_arxiv_structured",
            "summary": "found papers",
            "data": {"papers": [{"title": "RAG Paper"}]},
            "trace": {"tool_name": "search_arxiv_structured", "source": "mock", "returned_count": 1},
            "error": None,
        }

        with patch.object(tool_node_module, "invoke_tool", return_value=tool_result) as mock_invoke:
            result = execute_tool(state)

        mock_invoke.assert_called_once_with("search_arxiv_structured", query="rag", max_results=2)
        self.assertTrue(result.tool_observations[-1].ok)
        self.assertEqual(result.tool_observations[-1].status, "success")
        self.assertEqual(result.tool_calls[-1].status, "success")
        self.assertEqual(result.tool_result, tool_result)
        self.assertEqual(result.steps[-1].status, "success")

    def test_execute_tool_generates_failed_observation(self) -> None:
        state = AgentState(
            tool_call_request=ToolCallRequest(
                tool_name="search_arxiv_structured",
                arguments={"max_results": 999},
                reason="search candidate papers",
                expected_result="paper list",
                plan_step_id="step-1",
            )
        )
        tool_result = {
            "ok": False,
            "tool_name": "search_arxiv_structured",
            "summary": "Tool argument validation failed",
            "data": None,
            "trace": {"tool_name": "search_arxiv_structured", "validated": False},
            "error": {
                "code": "tool_argument_validation_failed",
                "message": "Tool arguments failed schema validation",
            },
        }

        with patch.object(tool_node_module, "invoke_tool", return_value=tool_result) as mock_invoke:
            result = execute_tool(state)

        mock_invoke.assert_called_once_with("search_arxiv_structured", max_results=999)
        self.assertFalse(result.tool_observations[-1].ok)
        self.assertEqual(result.tool_observations[-1].status, "failed")
        self.assertEqual(result.tool_observations[-1].error["code"], "tool_argument_validation_failed")
        self.assertEqual(result.tool_calls[-1].status, "failed")
        self.assertEqual(result.steps[-1].status, "failed")


if __name__ == "__main__":
    unittest.main()
