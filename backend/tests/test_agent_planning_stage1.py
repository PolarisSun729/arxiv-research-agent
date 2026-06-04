import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_stage1_modules():
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

    if "dependencies" not in sys.modules:
        dependencies_module = types.ModuleType("dependencies")
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

    if "utils.config" not in sys.modules:
        config_module = types.ModuleType("utils.config")
        config_module.get_memory_runtime_config = lambda: {"enable_user_research_profile": False}
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
    load("backend.agents.arxiv_search_agent.utils.search_spec_builder", utils_dir / "search_spec_builder.py")
    load("backend.agents.arxiv_search_agent.utils.state_utils", utils_dir / "state_utils.py")
    load("backend.agents.arxiv_search_agent.utils.text_utils", utils_dir / "text_utils.py")
    load("backend.agents.arxiv_search_agent.utils.paper_reference_resolver", utils_dir / "paper_reference_resolver.py")
    load("backend.agents.arxiv_search_agent.node.intent_support", node_dir / "intent_support.py")
    load("backend.agents.arxiv_search_agent.node.parse_node", node_dir / "parse_node.py")
    load("backend.agents.arxiv_search_agent.node.plan_node", node_dir / "plan_node.py")
    load("backend.agents.arxiv_search_agent.node.search_node", node_dir / "search_node.py")
    load("backend.agents.arxiv_search_agent.node.preference_node", node_dir / "preference_node.py")
    load("backend.agents.arxiv_search_agent.node.pending_action_node", node_dir / "pending_action_node.py")
    load("backend.agents.arxiv_search_agent.node.paper_reading_node", node_dir / "paper_reading_node.py")
    load("backend.agents.arxiv_search_agent.node.response_node", node_dir / "response_node.py")

    node_package = sys.modules["backend.agents.arxiv_search_agent.node"]
    node_package._coerce_state = sys.modules["backend.agents.arxiv_search_agent.utils.state_utils"]._coerce_state
    node_package.apply_preference_action = sys.modules["backend.agents.arxiv_search_agent.node.preference_node"].apply_preference_action
    node_package.build_search_tool_args = sys.modules["backend.agents.arxiv_search_agent.node.search_node"].build_search_tool_args
    node_package.check_search_result = sys.modules["backend.agents.arxiv_search_agent.node.search_node"].check_search_result
    node_package.classify_pending_action_confirmation = sys.modules[
        "backend.agents.arxiv_search_agent.node.pending_action_node"
    ].classify_pending_action_confirmation
    node_package.handle_paper_reading_request = sys.modules[
        "backend.agents.arxiv_search_agent.node.paper_reading_node"
    ].handle_paper_reading_request
    node_package.handle_pending_action_confirmation = sys.modules[
        "backend.agents.arxiv_search_agent.node.pending_action_node"
    ].handle_pending_action_confirmation
    node_package.invoke_search_tool = sys.modules["backend.agents.arxiv_search_agent.node.search_node"].invoke_search_tool
    node_package.parse_search_request = sys.modules["backend.agents.arxiv_search_agent.node.parse_node"].parse_search_request
    node_package.plan_task = sys.modules["backend.agents.arxiv_search_agent.node.plan_node"].plan_task
    node_package.personalized_rank_and_annotate_papers = sys.modules[
        "backend.agents.arxiv_search_agent.node.search_node"
    ].personalized_rank_and_annotate_papers
    node_package.relax_search_for_retry = sys.modules["backend.agents.arxiv_search_agent.node.search_node"].relax_search_for_retry
    node_package.synthesize_response = sys.modules["backend.agents.arxiv_search_agent.node.response_node"].synthesize_response

    graph_module = load("backend.agents.arxiv_search_agent.graph", agent_dir / "graph.py")
    service_module = load("backend.agents.arxiv_search_agent.service", agent_dir / "service.py")
    plan_module = sys.modules["backend.agents.arxiv_search_agent.node.plan_node"]
    return {
        "schemas": schemas,
        "state_module": state_module,
        "graph_module": graph_module,
        "service_module": service_module,
        "plan_module": plan_module,
    }


_MODULES = _load_stage1_modules()
AgentState = _MODULES["state_module"].AgentState
ArxivSearchSpec = _MODULES["schemas"].ArxivSearchSpec
ExecutionPlanStep = _MODULES["schemas"].ExecutionPlanStep
Goal = _MODULES["schemas"].Goal
plan_task = _MODULES["plan_module"].plan_task
build_arxiv_search_graph = _MODULES["graph_module"].build_arxiv_search_graph
export_arxiv_search_graph_mermaid = _MODULES["graph_module"].export_arxiv_search_graph_mermaid
_state_to_response = _MODULES["service_module"]._state_to_response
_compact_state = _MODULES["service_module"]._compact_state


class Stage1PlanningTests(unittest.TestCase):
    def test_arxiv_search_generates_search_goal_and_plan(self) -> None:
        state = AgentState(
            intent="arxiv_search",
            message="找最近的RAG论文",
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
            context={"user_memory_summary": {"profile": {"topics": ["rag"]}}},
        )

        result = plan_task(state)

        self.assertIsNotNone(result.goal)
        self.assertEqual(result.goal.goal_type, "arxiv_search")
        self.assertTrue(any(step.step_type == "search_execution" for step in result.execution_plan))
        self.assertTrue(any(step.step_type == "personalization" for step in result.execution_plan))
        self.assertEqual(result.steps[-1].step, "plan_task")
        self.assertEqual(result.steps[-1].status, "success")

    def test_paper_qa_generates_reading_plan(self) -> None:
        state = AgentState(
            intent="paper_qa",
            message="这篇论文的核心方法是什么？",
            context={"selected_paper": {"title": "Attention Is All You Need"}},
            pending_action={"type": "parse_then_qa", "status": "waiting_confirmation"},
        )

        result = plan_task(state)

        self.assertEqual(result.goal.goal_type, "paper_qa")
        self.assertTrue(result.goal.requires_user_confirmation)
        self.assertTrue(any(step.step_type == "qa_index_check" for step in result.execution_plan))
        self.assertTrue(any(step.step_type == "paper_response" for step in result.execution_plan))

    def test_preference_action_generates_preference_plan(self) -> None:
        state = AgentState(
            intent="preference_action",
            message="把这篇加入喜欢",
            context={"selected_paper": {"arxiv_id": "2401.12345"}, "user_memory_summary": {"profile": {}}},
        )

        result = plan_task(state)

        self.assertEqual(result.goal.goal_type, "preference_action")
        self.assertTrue(result.goal.requires_memory)
        self.assertTrue(any(step.step_type == "paper_resolution" for step in result.execution_plan))
        self.assertTrue(any(step.step_type == "preference_update" for step in result.execution_plan))

    def test_recommendation_generates_profile_aware_plan(self) -> None:
        state = AgentState(
            intent="recommendation",
            message="给我推荐几篇论文",
            context={"user_memory_summary": {"profile": {"topics": ["agents", "memory"]}}},
        )

        result = plan_task(state)

        self.assertEqual(result.goal.goal_type, "recommendation")
        self.assertTrue(result.goal.requires_memory)
        self.assertTrue(any(step.step_type == "profile_loading" for step in result.execution_plan))
        self.assertTrue(any(step.step_type == "recommendation_generation" for step in result.execution_plan))

    def test_unclear_and_unsupported_generate_safe_non_tool_plans(self) -> None:
        unclear_result = plan_task(AgentState(intent="unclear", message="帮我找那个东西"))
        unsupported_result = plan_task(AgentState(intent="unsupported", message="帮我画一张海报"))

        self.assertEqual(unclear_result.goal.goal_type, "unclear")
        self.assertEqual(unsupported_result.goal.goal_type, "unsupported")
        self.assertFalse(any(step.step_type == "search_execution" for step in unclear_result.execution_plan))
        self.assertFalse(any(step.step_type == "search_execution" for step in unsupported_result.execution_plan))
        self.assertFalse(any(step.step_type == "recommendation_generation" for step in unsupported_result.execution_plan))

    def test_pending_confirmation_route_is_preserved_after_plan_task(self) -> None:
        graph = build_arxiv_search_graph(generation_service=None)
        result = graph.invoke(
            AgentState(
                message="确认，继续处理",
                context={"pending_action": {"type": "parse_then_qa", "status": "waiting_confirmation"}},
                pending_action={"type": "parse_then_qa", "status": "waiting_confirmation"},
            ).model_dump()
        )

        steps = [item.step for item in result["steps"]]
        self.assertIn("intent_recognition", steps)
        self.assertIn("plan_task", steps)
        self.assertIn("classify_pending_action_confirmation", steps)

    def test_graph_mermaid_contains_plan_task(self) -> None:
        payload = export_arxiv_search_graph_mermaid(generation_service=None)

        self.assertIn("plan_task", payload["node_names"])
        self.assertIn("plan_task", payload["mermaid"])

    def test_response_and_compact_state_include_goal_and_execution_plan(self) -> None:
        state = AgentState(
            intent="unclear",
            answer="need clarification",
            goal=Goal(goal_type="unclear", user_goal="clarify request"),
            execution_plan=[
                ExecutionPlanStep(step_id="step_1", step_type="ambiguity_analysis", description="Analyze ambiguity"),
                ExecutionPlanStep(
                    step_id="step_2",
                    step_type="clarification_response",
                    description="Ask clarification question",
                    depends_on=["step_1"],
                ),
            ],
        )

        response = _state_to_response(state)
        compact = _compact_state(state)

        self.assertIsNotNone(response.goal)
        self.assertEqual(len(response.execution_plan), 2)
        self.assertEqual(compact["goal"]["goal_type"], "unclear")
        self.assertEqual(compact["execution_plan_summary"]["step_count"], 2)

    def test_response_defaults_when_goal_and_execution_plan_missing(self) -> None:
        response = _state_to_response(AgentState(intent="unsupported", answer="not supported"))

        self.assertIsNone(response.goal)
        self.assertEqual(response.execution_plan, [])


if __name__ == "__main__":
    unittest.main()
