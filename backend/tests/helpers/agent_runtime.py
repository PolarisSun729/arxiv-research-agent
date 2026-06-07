from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


class _DependencyBag:
    def __init__(self) -> None:
        self.generation_service: Any = None
        self.recommendation_service: Any = None
        self.paper_qa_service: Any = None
        self.arxiv_service: Any = None


DEPENDENCY_BAG = _DependencyBag()

PUBLIC_STUB_MODULES = [
    "dependencies",
    "services.memory",
    "services.storage.database_service",
    "tools.arxiv_tools",
    "tools.paper_qa_tools",
    "tools.recommendation_tools",
]


class FakeMemoryService:
    def __init__(self) -> None:
        self.saved_payloads = []

    def build_user_memory_summary(self, _user_id: str) -> Dict[str, Any]:
        return {
            "profile": {},
            "memory_status": {
                "preference_memory_available": False,
                "interest_vector_available": False,
            },
        }

    def load_agent_memory(self, _user_id: Optional[str], session_id: Optional[str], frontend_context: Optional[Mapping[str, Any]] = None):
        return {
            "merged_context": dict(frontend_context or {}),
            "session_id": session_id,
        }

    def save_agent_memory(self, **kwargs: Any) -> None:
        self.saved_payloads.append(dict(kwargs))

    def load_user_profile(self, user_id: str) -> Dict[str, Any]:
        return {"user_id": user_id}

    def patch_user_profile(self, user_id: str, patch: Mapping[str, Any], source: str) -> Dict[str, Any]:
        payload = dict(patch)
        payload["user_id"] = user_id
        payload["source"] = source
        return payload


class FakeDatabaseService:
    def get_user_research_profile(self, **_kwargs: Any) -> Dict[str, Any]:
        return {}


def _ensure_backend_packages() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    backend_dir = repo_root
    packages = {
        "backend": backend_dir,
        "backend.agents": backend_dir / "agents",
        "backend.agents.arxiv_search_agent": backend_dir / "agents" / "arxiv_search_agent",
        "backend.agents.arxiv_search_agent.utils": backend_dir / "agents" / "arxiv_search_agent" / "utils",
        "backend.agents.arxiv_search_agent.node": backend_dir / "agents" / "arxiv_search_agent" / "node",
        "backend.routers": backend_dir / "routers",
        "services": backend_dir / "services",
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

    backend_str = str(backend_dir)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
    return backend_dir


def _ensure_dependency_stubs() -> None:
    dependencies_module = sys.modules.get("dependencies", types.ModuleType("dependencies"))
    dependencies_module.get_database_service = getattr(dependencies_module, "get_database_service", lambda: None)
    dependencies_module.get_oai_database_service = getattr(dependencies_module, "get_oai_database_service", lambda: None)
    dependencies_module.get_embedding_service = getattr(dependencies_module, "get_embedding_service", lambda: None)
    dependencies_module.get_vector_store_service = getattr(dependencies_module, "get_vector_store_service", lambda: None)
    dependencies_module.get_current_embedding_config = getattr(
        dependencies_module,
        "get_current_embedding_config",
        lambda: None,
    )
    dependencies_module.get_enhanced_retrieval_service = getattr(
        dependencies_module,
        "get_enhanced_retrieval_service",
        lambda: None,
    )
    dependencies_module.get_index_job_manager = getattr(dependencies_module, "get_index_job_manager", lambda: None)
    dependencies_module.get_generation_service = lambda: DEPENDENCY_BAG.generation_service
    dependencies_module.get_recommendation_service = lambda: DEPENDENCY_BAG.recommendation_service
    dependencies_module.get_paper_qa_service = lambda: DEPENDENCY_BAG.paper_qa_service
    dependencies_module.get_arxiv_service = lambda: DEPENDENCY_BAG.arxiv_service
    dependencies_module.get_database_service = getattr(dependencies_module, "get_database_service", lambda: None)
    dependencies_module.get_memory_service = getattr(dependencies_module, "get_memory_service", lambda: None)
    sys.modules["dependencies"] = dependencies_module

    if "services.memory" not in sys.modules or not hasattr(sys.modules["services.memory"], "MemoryService"):
        memory_module = sys.modules.get("services.memory", types.ModuleType("services.memory"))
        memory_module.MemoryService = FakeMemoryService
        sys.modules["services.memory"] = memory_module

    if "services.storage.database_service" not in sys.modules:
        database_service_module = types.ModuleType("services.storage.database_service")

        class _PaperQATurnPersistenceError(RuntimeError):
            pass

        # Agent 测试只需要轻量数据库桩，但导出形状必须跟真实模块一致，避免影响同进程里的 QA 测试收集。
        database_service_module.PaperQATurnPersistenceError = _PaperQATurnPersistenceError
        database_service_module.DatabaseService = FakeDatabaseService
        sys.modules["services.storage.database_service"] = database_service_module

    config_module = sys.modules.get("utils.config", types.ModuleType("utils.config"))
    if not hasattr(config_module, "get_memory_runtime_config"):
        config_module.get_memory_runtime_config = lambda: {"enable_user_research_profile": False}
    if not hasattr(config_module, "get_arxiv_oai_runtime_config"):
        config_module.get_arxiv_oai_runtime_config = lambda: {"target_categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"]}
    if not hasattr(config_module, "get_default_user_id"):
        config_module.get_default_user_id = lambda: "default"
    # 共享测试环境里会混跑 intent / retrieval 相关单测，这里补足常见配置入口，避免后续导入被前一个 stub 卡住。
    if not hasattr(config_module, "RETRIEVAL_CONFIG"):
        config_module.RETRIEVAL_CONFIG = {
            "route_weights": {"search": 1.0, "summary": 0.8, "method": 0.9, "experiment": 0.9, "other": 0.3},
            "rrf_k": 60,
            "candidate_multiplier": 3,
            "enable_query_rewrite": True,
            "enable_hyde": False,
            "enable_keyword_search": True,
            "debug": False,
        }
    if not hasattr(config_module, "get_arxiv_search_runtime_config"):
        config_module.get_arxiv_search_runtime_config = lambda: {
            "target_categories": ["cs.CL", "cs.LG", "cs.IR", "cs.AI"],
        }
    if not hasattr(config_module, "get_enhanced_retrieval_runtime_config"):
        config_module.get_enhanced_retrieval_runtime_config = lambda: {
            "query_view_limit": 5,
            "query_plan_limit": 5,
            "final_context_top_k": 5,
            "max_final_context_top_k": 10,
            "recall_candidate_limit": 20,
            "rrf_candidate_limit": 20,
            "rerank_candidate_limit": 20,
            "rerank_document_preview_limit": 512,
            "preview_text_limit": 512,
            "candidate_debug_limit": 10,
            "short_text_preview_limit": 128,
            "source_sample_limit": 3,
            "source_sample_primary_limit": 128,
            "source_sample_secondary_limit": 128,
            "merge_candidate_terms_limit": 10,
            "extract_paper_terms_limit": 10,
            "paper_terms_preview_limit": 10,
            "compact_terms_limit": 10,
            "keyword_parts_limit": 8,
            "extract_query_keywords_limit": 8,
            "build_query_keywords_limit": 8,
            "sanitize_trace_slug_max_length": 64,
            "memory_source_boost_weight": 0.12,
            "query_weight_base_summary_other": 0.3,
            "query_weight_base_default": 0.5,
            "query_weight_base_ambiguous_keyword": 0.45,
            "query_weight_base_ambiguous_other": 0.35,
            "query_weight_base_keyword": 0.55,
            "query_weight_base_fallback": 0.25,
            "route_focus_bonus": 0.1,
            "route_summary_bonus": 0.1,
            "route_keyword_bonus": 0.1,
            "route_default_floor": 0.1,
            "route_confidence_multiplier": 0.8,
            "route_confidence_similarity_weight": 0.2,
            "section_bonus_weight": 0.1,
            "figure_table_bonus_weight": 0.1,
            "noisy_section_penalty_weight": 0.05,
            "preferred_section_bonus_weight": 0.1,
            "section_path_bonus_weight": 0.05,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "bm25_token_boost": 0.1,
            "specificity_content_weight": 0.4,
            "specificity_intent_weight": 0.3,
            "specificity_length_weight": 0.3,
            "specificity_floor": 0.2,
            "language_weight_zh_mixed": 0.1,
        }
    if not hasattr(config_module, "get_intent_routing_runtime_config"):
        config_module.get_intent_routing_runtime_config = lambda: {
            # 这里返回的是“主意图 -> 子权重字典”的结构，和 IntentService._build_route_weights 的读取方式一致。
            "intent_route_weights": {
                "paper_overview": {"vector_original": 1.05, "vector_rewrite": 1.03},
                "method_flow": {"vector_rewrite": 1.06, "keyword": 1.04},
                "experiment_setup": {"vector_rewrite": 1.07, "keyword": 1.05},
                "result_analysis": {"vector_rewrite": 1.08, "keyword": 1.06},
                "comparison": {"vector_rewrite": 1.05, "keyword": 1.04},
                "dataset": {"vector_rewrite": 1.04, "keyword": 1.03},
                "limitation": {"vector_rewrite": 1.03, "keyword": 1.02},
                "definition": {"vector_rewrite": 1.03, "keyword": 1.02},
                "implementation_detail": {"vector_rewrite": 1.04, "keyword": 1.02},
                "figure_table": {"keyword": 1.08, "vector_rewrite": 1.02},
                "other": {"vector_original": 0.95, "vector_rewrite": 0.95, "keyword": 0.95},
            }
        }
    if not hasattr(config_module, "get_agent_planner_runtime_config"):
        config_module.get_agent_planner_runtime_config = lambda: {
            "enable_tool_aware_planner": False,
            "enable_llm_plan_draft": False,
            "enable_llm_recovery_diagnosis": False,
            "llm_recovery_timeout": 6,
            "llm_plan_timeout": 8,
            "llm_plan_max_steps": 8,
            "llm_plan_fallback_to_rule": True,
            "llm_plan_fallback_to_template": True,
            "expose_planner_debug": True,
        }
    sys.modules["utils.config"] = config_module


def _ensure_tool_stubs() -> None:
    for module_name in ("tools.arxiv_tools", "tools.paper_qa_tools", "tools.recommendation_tools"):
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        module.search_arxiv_raw = lambda **_kwargs: {}
        module.search_arxiv_structured = lambda **_kwargs: {}
        module.get_paper_metadata = lambda **_kwargs: {}
        module.recommend_papers = lambda **_kwargs: {}
        module.record_paper_preference = lambda **_kwargs: {}
        module.record_user_paper_preference = lambda **_kwargs: {}
        module.remove_user_paper_preference = lambda **_kwargs: {}
        module.check_paper_qa_index = lambda **_kwargs: {}
        module.build_paper_qa_index = lambda **_kwargs: {}
        module.answer_paper_question = lambda **_kwargs: {}
        sys.modules[module_name] = module


def _ensure_langgraph_stub() -> None:
    langgraph_module = types.ModuleType("langgraph")
    graph_module = types.ModuleType("langgraph.graph")
    types_module = types.ModuleType("langgraph.types")
    checkpoint_module = types.ModuleType("langgraph.checkpoint")
    checkpoint_memory_module = types.ModuleType("langgraph.checkpoint.memory")

    class _MemorySaver:
        """测试桩里的内存 checkpoint。

        这里只需要保留“存在一个可共享的 checkpointer 实例”这一契约，
        让图编译和 config 透传逻辑可以在单测里被验证，而不引入真实持久化行为。
        """

        def __init__(self):
            self.snapshots = {}

    class _Command:
        def __init__(self, *, resume=None):
            self.resume = resume

    def _interrupt(payload):
        return None

    class _GraphView:
        def __init__(self, nodes, edges):
            self._nodes = nodes
            self._edges = edges

        def draw_mermaid(self):
            lines = ["graph TD"]
            for start, end in self._edges:
                lines.append(f"    {start} --> {end}")
            return "\n".join(lines)

    class _CompiledGraph:
        def __init__(self, nodes, edges, conditional_edges, checkpointer=None):
            self._nodes = nodes
            self._edges = list(edges)
            self._conditional_edges = dict(conditional_edges)
            self._checkpointer = checkpointer
            self.last_invoke_config = None
            self.last_stream_config = None
            self.last_invoke_input = None
            self.last_stream_input = None

        @staticmethod
        def _finalize_state(value):
            if hasattr(value, "model_dump"):
                payload = value.model_dump()
                for key in ("steps", "execution_plan", "tool_calls", "tool_observations"):
                    if hasattr(value, key):
                        payload[key] = getattr(value, key)
                return payload
            if isinstance(value, dict):
                return dict(value)
            return value

        def _next_node(self, current: str, current_state: Any) -> str:
            if current in self._conditional_edges:
                route_fn, mapping = self._conditional_edges[current]
                route_key = route_fn(current_state)
                return mapping[route_key]
            return next(end for start, end in self._edges if start == current)

        def invoke(self, state, config=None):
            self.last_invoke_config = config
            self.last_invoke_input = state
            current_state = state
            current = next(end for start, end in self._edges if start == "START")
            while current != "END":
                node_fn = self._nodes[current]
                current_state = node_fn(current_state)
                current = self._next_node(current, current_state)
            return self._finalize_state(current_state)

        def stream(self, state, config=None, stream_mode: str = "updates"):
            self.last_stream_config = config
            self.last_stream_input = state
            del stream_mode
            current_state = state
            current = next(end for start, end in self._edges if start == "START")
            while current != "END":
                node_fn = self._nodes[current]
                current_state = node_fn(current_state)
                yield {current: self._finalize_state(current_state)}
                current = self._next_node(current, current_state)

        def get_graph(self):
            return _GraphView(self._nodes, self._edges)

        def get_state(self, config=None):
            if not self._checkpointer:
                return None
            thread_id = str((((config or {}).get("configurable") or {}).get("thread_id")) or "").strip()
            if not thread_id:
                return None
            return self._checkpointer.snapshots.get(thread_id)

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

        def compile(self, checkpointer=None):
            return _CompiledGraph(self.nodes, self.edges, self.conditional_edges, checkpointer=checkpointer)

    graph_module.END = "END"
    graph_module.START = "START"
    graph_module.StateGraph = _StateGraph
    types_module.Command = _Command
    types_module.interrupt = _interrupt
    checkpoint_memory_module.MemorySaver = _MemorySaver
    checkpoint_memory_module.InMemorySaver = _MemorySaver
    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module
    sys.modules["langgraph.types"] = types_module
    sys.modules["langgraph.checkpoint"] = checkpoint_module
    sys.modules["langgraph.checkpoint.memory"] = checkpoint_memory_module


def _load_module(module_name: str, file_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _restore_public_stub_modules(saved_modules: Dict[str, Any]) -> None:
    # Agent 测试桩只服务当前加载过程；恢复公共模块可避免 pytest 混跑时污染后续 service/router 测试。
    for module_name in PUBLIC_STUB_MODULES:
        original = saved_modules.get(module_name)
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original


def load_agent_test_modules() -> Dict[str, Any]:
    saved_public_modules = {module_name: sys.modules.get(module_name) for module_name in PUBLIC_STUB_MODULES}
    # 测试进程里可能已经残留过上一轮导入的同名模块；先清理再按当前源码重建，
    # 可以避免拿到旧版对象而出现“属性存在但实际实现已变”的隐蔽问题。
    for module_name in [
        "dependencies",
        "langgraph",
        "langgraph.graph",
        "services.memory",
        "services.storage.database_service",
        "tools.arxiv_tools",
        "tools.paper_qa_tools",
        "tools.recommendation_tools",
        "tools.tool_result",
        "tools.schemas",
        "tools.tool_registry",
        "backend.agents.arxiv_search_agent.schemas",
        "backend.agents.arxiv_search_agent.state",
        "backend.agents.arxiv_search_agent.utils.result_utils",
        "backend.agents.arxiv_search_agent.utils.search_spec_builder",
        "backend.agents.arxiv_search_agent.utils.state_utils",
        "backend.agents.arxiv_search_agent.utils.text_utils",
        "backend.agents.arxiv_search_agent.utils.paper_reference_resolver",
        "backend.agents.arxiv_search_agent.observer",
        "backend.agents.arxiv_search_agent.failure_classifier",
        "backend.agents.arxiv_search_agent.recovery_policy",
        "backend.agents.arxiv_search_agent.recovery_diagnosis",
        "backend.agents.arxiv_search_agent.recovery_chooser",
        "backend.agents.arxiv_search_agent.recovery_safety",
        "backend.agents.arxiv_search_agent.plan_patcher",
        "backend.agents.arxiv_search_agent.tool_aware_planner",
        "backend.agents.arxiv_search_agent.planner",
        "backend.agents.arxiv_search_agent.plan_validator",
        "backend.agents.arxiv_search_agent.replanner",
        "backend.agents.arxiv_search_agent.plan_executor",
        "backend.agents.arxiv_search_agent.tool_registry",
        "backend.agents.arxiv_search_agent.node.tool_node",
        "backend.agents.arxiv_search_agent.node.intent_support",
        "backend.agents.arxiv_search_agent.node.parse_node",
        "backend.agents.arxiv_search_agent.node.plan_node",
        "backend.agents.arxiv_search_agent.node.search_node",
        "backend.agents.arxiv_search_agent.node.recommendation_node",
        "backend.agents.arxiv_search_agent.node.preference_node",
        "backend.agents.arxiv_search_agent.node.paper_reading_node",
        "backend.agents.arxiv_search_agent.node.response_node",
        "backend.agents.arxiv_search_agent.node",
        "backend.agents.arxiv_search_agent.graph",
        "backend.agents.arxiv_search_agent.service",
        "backend.routers.agent_router",
    ]:
        sys.modules.pop(module_name, None)

    backend_dir = _ensure_backend_packages()
    _ensure_dependency_stubs()
    _ensure_tool_stubs()
    _ensure_langgraph_stub()

    agent_dir = backend_dir / "agents" / "arxiv_search_agent"
    utils_dir = agent_dir / "utils"
    node_dir = agent_dir / "node"
    routers_dir = backend_dir / "routers"
    tools_dir = backend_dir / "tools"

    _load_module("tools.tool_result", tools_dir / "tool_result.py")
    _load_module("tools.schemas", tools_dir / "schemas.py")
    _load_module("tools.tool_registry", tools_dir / "tool_registry.py")

    schemas = _load_module("backend.agents.arxiv_search_agent.schemas", agent_dir / "schemas.py")
    state_module = _load_module("backend.agents.arxiv_search_agent.state", agent_dir / "state.py")
    _load_module("backend.agents.arxiv_search_agent.utils.result_utils", utils_dir / "result_utils.py")
    _load_module("backend.agents.arxiv_search_agent.utils.search_spec_builder", utils_dir / "search_spec_builder.py")
    _load_module("backend.agents.arxiv_search_agent.utils.state_utils", utils_dir / "state_utils.py")
    _load_module("backend.agents.arxiv_search_agent.utils.text_utils", utils_dir / "text_utils.py")
    _load_module("backend.agents.arxiv_search_agent.utils.paper_reference_resolver", utils_dir / "paper_reference_resolver.py")
    _load_module("backend.agents.arxiv_search_agent.node.tool_node", node_dir / "tool_node.py")
    _load_module("backend.agents.arxiv_search_agent.node.intent_support", node_dir / "intent_support.py")
    _load_module("backend.agents.arxiv_search_agent.node.parse_node", node_dir / "parse_node.py")
    _load_module("backend.agents.arxiv_search_agent.node.plan_node", node_dir / "plan_node.py")
    search_node_module = _load_module("backend.agents.arxiv_search_agent.node.search_node", node_dir / "search_node.py")
    _load_module("backend.agents.arxiv_search_agent.node.recommendation_node", node_dir / "recommendation_node.py")
    _load_module("backend.agents.arxiv_search_agent.node.preference_node", node_dir / "preference_node.py")
    _load_module("backend.agents.arxiv_search_agent.node.paper_reading_node", node_dir / "paper_reading_node.py")
    _load_module("backend.agents.arxiv_search_agent.node.response_node", node_dir / "response_node.py")

    node_package = sys.modules["backend.agents.arxiv_search_agent.node"]
    node_package._coerce_state = sys.modules["backend.agents.arxiv_search_agent.utils.state_utils"]._coerce_state
    node_package.adapt_search_tool_result = search_node_module.adapt_search_tool_result
    node_package.apply_preference_action = sys.modules["backend.agents.arxiv_search_agent.node.preference_node"].apply_preference_action
    node_package.build_search_tool_args = search_node_module.build_search_tool_args
    node_package.check_search_result = search_node_module.check_search_result
    node_package.execute_tool = sys.modules["backend.agents.arxiv_search_agent.node.tool_node"].execute_tool
    node_package.handle_paper_reading_request = sys.modules[
        "backend.agents.arxiv_search_agent.node.paper_reading_node"
    ].handle_paper_reading_request
    node_package.invoke_search_tool = search_node_module.invoke_search_tool
    node_package.parse_search_request = sys.modules["backend.agents.arxiv_search_agent.node.parse_node"].parse_search_request
    node_package.plan_task = sys.modules["backend.agents.arxiv_search_agent.node.plan_node"].plan_task
    node_package.personalized_rank_and_annotate_papers = search_node_module.personalized_rank_and_annotate_papers
    node_package.relax_search_for_retry = search_node_module.relax_search_for_retry
    node_package.synthesize_response = sys.modules["backend.agents.arxiv_search_agent.node.response_node"].synthesize_response

    graph_module = _load_module("backend.agents.arxiv_search_agent.graph", agent_dir / "graph.py")
    service_module = _load_module("backend.agents.arxiv_search_agent.service", agent_dir / "service.py")
    router_module = _load_module("backend.routers.agent_router", routers_dir / "agent_router.py")

    package_module = sys.modules["backend.agents.arxiv_search_agent"]
    package_module.ArxivSearchGraphResponse = schemas.ArxivSearchGraphResponse
    package_module.ArxivSearchRequest = schemas.ArxivSearchRequest
    package_module.ArxivSearchResponse = schemas.ArxivSearchResponse
    package_module.export_arxiv_search_graph_mermaid = graph_module.export_arxiv_search_graph_mermaid
    package_module.run_arxiv_search_agent = service_module.run_arxiv_search_agent
    package_module.stream_arxiv_search_agent = service_module.stream_arxiv_search_agent

    modules = {
        "schemas": schemas,
        "state_module": state_module,
        "graph_module": graph_module,
        "service_module": service_module,
        "router_module": router_module,
        "tool_node_module": sys.modules["backend.agents.arxiv_search_agent.node.tool_node"],
        "tool_registry_module": sys.modules["tools.tool_registry"],
    }
    _restore_public_stub_modules(saved_public_modules)
    return modules


__all__ = [
    "DEPENDENCY_BAG",
    "FakeDatabaseService",
    "FakeMemoryService",
    "load_agent_test_modules",
]
