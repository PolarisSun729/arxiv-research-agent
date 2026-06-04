import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_modules():
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
        "services": backend_dir / "services",
        "services.memory": backend_dir / "services" / "memory",
        "services.storage": backend_dir / "services" / "storage",
        "tools": backend_dir / "tools",
        "utils": backend_dir / "utils",
    }
    for package_name, package_path in packages.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]
        sys.modules[package_name] = module

    # 如果前一个测试已经导入过图模块，这里先清掉再按当前 stub 重新装载，避免复用旧的 LangGraph 适配对象。
    sys.modules.pop("backend.agents.arxiv_search_agent.graph", None)

    # 这里不假设前一个测试留下的 dependencies 模块一定干净，直接补齐当前阶段要用的接口。
    dependencies_module = sys.modules.get("dependencies", types.ModuleType("dependencies"))
    dependencies_module.get_generation_service = lambda: None
    dependencies_module.get_recommendation_service = lambda: object()
    dependencies_module.get_paper_qa_service = lambda: object()
    sys.modules["dependencies"] = dependencies_module

    if "tools.tool_registry" not in sys.modules:
        tool_registry_module = types.ModuleType("tools.tool_registry")
        tool_registry_module.invoke_tool = lambda *args, **kwargs: None
        sys.modules["tools.tool_registry"] = tool_registry_module

    if "services.memory" not in sys.modules or not hasattr(sys.modules["services.memory"], "MemoryService"):
        memory_module = sys.modules.get("services.memory", types.ModuleType("services.memory"))

        class _MemoryService:
            def build_user_memory_summary(self, *_args, **_kwargs):
                return {}

            def load_agent_memory(self, _user_id, session_id, frontend_context=None):
                return {
                    "merged_context": dict(frontend_context or {}),
                    "session_id": session_id,
                }

            def save_agent_memory(self, **_kwargs):
                return None

        memory_module.MemoryService = _MemoryService
        sys.modules["services.memory"] = memory_module

    if "services.storage.database_service" not in sys.modules:
        database_service_module = types.ModuleType("services.storage.database_service")

        class _DatabaseService:
            def get_user_research_profile(self, **_kwargs):
                return {}

        database_service_module.DatabaseService = _DatabaseService
        sys.modules["services.storage.database_service"] = database_service_module

    config_module = sys.modules.get("utils.config", types.ModuleType("utils.config"))
    config_module.get_memory_runtime_config = lambda: {"enable_user_research_profile": False}
    config_module.get_arxiv_oai_runtime_config = lambda: {"target_categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"]}
    sys.modules["utils.config"] = config_module

    langgraph_module = types.ModuleType("langgraph")
    graph_module = types.ModuleType("langgraph.graph")

    class _CompiledGraph:
        def __init__(self, nodes=None, edges=None, conditional_edges=None):
            self._nodes = nodes or {}
            self._edges = list(edges or [])
            self._conditional_edges = dict(conditional_edges or {})

        def get_graph(self):
            class _GraphView:
                def draw_mermaid(self_inner):
                    return "graph TD\n    plan_task"

            return _GraphView()

        def invoke(self, state):
            return state

    class _StateGraph:
        def __init__(self, *_args, **_kwargs):
            self.nodes = {}
            self.edges = []
            self.conditional_edges = {}

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def add_edge(self, start, end):
            self.edges.append((start, end))

        def add_conditional_edges(self, source, router, mapping):
            self.conditional_edges[source] = (router, mapping)

        def compile(self):
            return _CompiledGraph(self.nodes, self.edges, self.conditional_edges)

    graph_module.END = "END"
    graph_module.START = "START"
    graph_module.StateGraph = _StateGraph
    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module

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
    load("backend.agents.arxiv_search_agent.utils.search_spec_builder", utils_dir / "search_spec_builder.py")
    load("backend.agents.arxiv_search_agent.utils.state_utils", utils_dir / "state_utils.py")
    load("backend.agents.arxiv_search_agent.utils.text_utils", utils_dir / "text_utils.py")
    load("backend.agents.arxiv_search_agent.utils.paper_reference_resolver", utils_dir / "paper_reference_resolver.py")
    load("backend.agents.arxiv_search_agent.node.tool_node", node_dir / "tool_node.py")
    search_node_module = load("backend.agents.arxiv_search_agent.node.search_node", node_dir / "search_node.py")
    load("backend.agents.arxiv_search_agent.node.intent_support", node_dir / "intent_support.py")
    load("backend.agents.arxiv_search_agent.node.parse_node", node_dir / "parse_node.py")
    load("backend.agents.arxiv_search_agent.node.plan_node", node_dir / "plan_node.py")
    load("backend.agents.arxiv_search_agent.node.preference_node", node_dir / "preference_node.py")
    load("backend.agents.arxiv_search_agent.node.pending_action_node", node_dir / "pending_action_node.py")
    load("backend.agents.arxiv_search_agent.node.paper_reading_node", node_dir / "paper_reading_node.py")
    load("backend.agents.arxiv_search_agent.node.response_node", node_dir / "response_node.py")

    node_package = sys.modules["backend.agents.arxiv_search_agent.node"]
    node_package._coerce_state = sys.modules["backend.agents.arxiv_search_agent.utils.state_utils"]._coerce_state
    node_package.adapt_search_tool_result = search_node_module.adapt_search_tool_result
    node_package.apply_preference_action = sys.modules["backend.agents.arxiv_search_agent.node.preference_node"].apply_preference_action
    node_package.build_search_tool_args = search_node_module.build_search_tool_args
    node_package.check_search_result = search_node_module.check_search_result
    node_package.classify_pending_action_confirmation = sys.modules[
        "backend.agents.arxiv_search_agent.node.pending_action_node"
    ].classify_pending_action_confirmation
    node_package.execute_tool = sys.modules["backend.agents.arxiv_search_agent.node.tool_node"].execute_tool
    node_package.handle_paper_reading_request = sys.modules[
        "backend.agents.arxiv_search_agent.node.paper_reading_node"
    ].handle_paper_reading_request
    node_package.handle_pending_action_confirmation = sys.modules[
        "backend.agents.arxiv_search_agent.node.pending_action_node"
    ].handle_pending_action_confirmation
    node_package.invoke_search_tool = search_node_module.invoke_search_tool
    node_package.parse_search_request = sys.modules["backend.agents.arxiv_search_agent.node.parse_node"].parse_search_request
    node_package.plan_task = sys.modules["backend.agents.arxiv_search_agent.node.plan_node"].plan_task
    node_package.personalized_rank_and_annotate_papers = search_node_module.personalized_rank_and_annotate_papers
    node_package.relax_search_for_retry = search_node_module.relax_search_for_retry
    node_package.synthesize_response = sys.modules["backend.agents.arxiv_search_agent.node.response_node"].synthesize_response

    graph_module = load("backend.agents.arxiv_search_agent.graph", agent_dir / "graph.py")

    return {
        "schemas": schemas,
        "state_module": state_module,
        "search_node_module": search_node_module,
        "tool_node_module": sys.modules["backend.agents.arxiv_search_agent.node.tool_node"],
        "graph_module": graph_module,
    }


_MODULES = _load_modules()
AgentState = _MODULES["state_module"].AgentState
ArxivSearchSpec = _MODULES["schemas"].ArxivSearchSpec
ExecutionPlanStep = _MODULES["schemas"].ExecutionPlanStep
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
build_search_tool_args = _MODULES["search_node_module"].build_search_tool_args
invoke_search_tool = _MODULES["search_node_module"].invoke_search_tool
adapt_search_tool_result = _MODULES["search_node_module"].adapt_search_tool_result
check_search_result = _MODULES["search_node_module"].check_search_result
tool_node_module = _MODULES["tool_node_module"]


class SearchToolProtocolStage2Tests(unittest.TestCase):
    def test_build_search_tool_args_populates_tool_call_request(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag", categories=["cs.CL"], max_results=5),
            execution_plan=[
                ExecutionPlanStep(
                    step_id="step_2",
                    step_type="search_execution",
                    description="Search arXiv papers.",
                )
            ],
        )

        result = build_search_tool_args(state)

        self.assertIsNotNone(result.tool_call_request)
        self.assertEqual(result.tool_call_request.tool_name, "search_arxiv_structured")
        self.assertEqual(result.tool_call_request.plan_step_id, "step_2")
        self.assertEqual(result.tool_call_request.arguments["query"], "rag")

    def test_search_chain_uses_tool_request_and_preserves_papers_for_downstream_nodes(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag", categories=["cs.CL"], max_results=2),
            execution_plan=[
                ExecutionPlanStep(
                    step_id="step_2",
                    step_type="search_execution",
                    description="Search arXiv papers.",
                )
            ],
        )
        tool_result = {
            "ok": True,
            "tool_name": "search_arxiv_structured",
            "summary": "found papers",
            "data": {"papers": [{"title": "RAG Paper", "arxiv_id": "2401.00001"}]},
            "trace": {"tool_name": "search_arxiv_structured", "source": "mock", "returned_count": 1},
            "error": None,
        }

        built = build_search_tool_args(state)
        with patch.object(tool_node_module, "invoke_tool", return_value=tool_result) as mock_invoke:
            invoked = invoke_search_tool(built)

        adapted = adapt_search_tool_result(invoked)
        checked = check_search_result(adapted)

        mock_invoke.assert_called_once_with("search_arxiv_structured", **built.tool_args)
        self.assertEqual(invoked.steps[-1].step, "search_tool_call")
        self.assertTrue(adapted.tool_result["ok"])
        self.assertEqual(len(adapted.tool_observations), 1)
        self.assertTrue(adapted.tool_observations[-1].ok)
        self.assertEqual(len(adapted.tool_calls), 1)
        self.assertEqual(len(adapted.papers), 1)
        self.assertEqual(adapted.papers[0]["title"], "RAG Paper")
        self.assertEqual(adapted.tool_observations[-1].result_ref["paper_count"], 1)
        self.assertEqual(checked.steps[-1].step, "search_result_check")
        self.assertEqual(checked.steps[-1].status, "success")

    def test_search_chain_preserves_structured_failure_without_runtime_error(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag", categories=["cs.CL"], max_results=2),
            execution_plan=[
                ExecutionPlanStep(
                    step_id="step_2",
                    step_type="search_execution",
                    description="Search arXiv papers.",
                )
            ],
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

        built = build_search_tool_args(state)
        with patch.object(tool_node_module, "invoke_tool", return_value=tool_result):
            invoked = invoke_search_tool(built)

        adapted = adapt_search_tool_result(invoked)
        checked = check_search_result(adapted)

        self.assertFalse(adapted.tool_observations[-1].ok)
        self.assertEqual(adapted.tool_observations[-1].status, "failed")
        self.assertEqual(adapted.tool_calls[-1].status, "failed")
        self.assertEqual(adapted.papers, [])
        self.assertTrue(any("工具调用失败" in warning for warning in checked.warnings))
        self.assertFalse(any(error.get("code") == "agent_runtime_error" for error in checked.errors))
        self.assertEqual(checked.steps[-1].step, "search_result_check")
        self.assertEqual(checked.steps[-1].status, "failed")

    def test_graph_compiles_with_adaptation_node(self) -> None:
        graph = build_arxiv_search_graph(generation_service=None)
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
