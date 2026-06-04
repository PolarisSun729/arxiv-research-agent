import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def _load_search_node_module():
    repo_root = Path(__file__).resolve().parents[2]
    agent_dir = repo_root / "backend" / "agents" / "arxiv_search_agent"
    utils_dir = agent_dir / "utils"
    node_dir = agent_dir / "node"

    packages = {
        "backend": repo_root / "backend",
        "backend.agents": repo_root / "backend" / "agents",
        "backend.agents.arxiv_search_agent": agent_dir,
        "backend.agents.arxiv_search_agent.utils": utils_dir,
        "backend.agents.arxiv_search_agent.node": node_dir,
        "tools": repo_root / "backend" / "tools",
    }
    for package_name, package_path in packages.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]
        sys.modules[package_name] = module

    if "dependencies" not in sys.modules:
        dependencies_module = types.ModuleType("dependencies")
        dependencies_module.get_recommendation_service = lambda: object()
        sys.modules["dependencies"] = dependencies_module

    if "tools.tool_registry" not in sys.modules:
        tool_registry_module = types.ModuleType("tools.tool_registry")
        tool_registry_module.invoke_tool = lambda *args, **kwargs: None
        sys.modules["tools.tool_registry"] = tool_registry_module

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
    search_node = load("backend.agents.arxiv_search_agent.node.search_node", node_dir / "search_node.py")
    return search_node, state_module.AgentState, schemas.ArxivSearchSpec


search_node, AgentState, ArxivSearchSpec = _load_search_node_module()


class _FailingRecommendationService:
    def rerank_search_results_for_user(self, **kwargs):
        raise RuntimeError("rerank service boom")


class PersonalizedRerankNodeTests(unittest.TestCase):
    def _make_state(self, *, user_id="user-1", papers=None):
        paper_list = papers if papers is not None else [
            {"title": "Paper A", "arxiv_id": "1234.5678"},
            {"title": "Paper B", "arxiv_id": "2345.6789"},
        ]
        return AgentState(
            intent="arxiv_search",
            user_id=user_id,
            papers=list(paper_list),
            search_spec=ArxivSearchSpec(intent="arxiv_search", query="rag"),
        )

    def test_skips_when_user_id_missing(self) -> None:
        result = search_node.personalized_rank_and_annotate_papers(self._make_state(user_id=None))

        self.assertFalse(result.personalized_rerank_applied)
        self.assertTrue(result.steps)
        self.assertEqual(result.steps[-1].step, "personalized_rerank")
        self.assertEqual(result.steps[-1].status, "skipped")

    def test_skips_when_papers_missing(self) -> None:
        result = search_node.personalized_rank_and_annotate_papers(self._make_state(papers=[]))

        self.assertFalse(result.personalized_rerank_applied)
        self.assertTrue(result.steps)
        self.assertEqual(result.steps[-1].step, "personalized_rerank")
        self.assertEqual(result.steps[-1].status, "skipped")

    def test_keeps_papers_when_recommendation_service_init_fails(self) -> None:
        original_papers = [
            {"title": "Paper A", "arxiv_id": "1234.5678"},
            {"title": "Paper B", "arxiv_id": "2345.6789"},
        ]
        state = self._make_state(papers=original_papers)

        with mock.patch.object(
            search_node,
            "get_recommendation_service",
            side_effect=RuntimeError("init service boom"),
        ):
            result = search_node.personalized_rank_and_annotate_papers(state)

        self.assertEqual(result.papers, original_papers)
        self.assertFalse(result.personalized_rerank_applied)
        self.assertTrue(any("无法初始化推荐服务" in item for item in result.warnings))
        self.assertEqual(result.steps[-1].step, "personalized_rerank")
        self.assertEqual(result.steps[-1].status, "failed")

    def test_keeps_papers_when_rerank_call_fails(self) -> None:
        original_papers = [
            {"title": "Paper A", "arxiv_id": "1234.5678"},
            {"title": "Paper B", "arxiv_id": "2345.6789"},
        ]
        state = self._make_state(papers=original_papers)

        with mock.patch.object(
            search_node,
            "get_recommendation_service",
            return_value=_FailingRecommendationService(),
        ):
            result = search_node.personalized_rank_and_annotate_papers(state)

        self.assertEqual(result.papers, original_papers)
        self.assertFalse(result.personalized_rerank_applied)
        self.assertTrue(any("个性化重排失败" in item for item in result.warnings))
        self.assertEqual(result.steps[-1].step, "personalized_rerank")
        self.assertEqual(result.steps[-1].status, "failed")

    def test_failed_step_contains_personalized_rerank_metadata(self) -> None:
        state = self._make_state()

        with mock.patch.object(
            search_node,
            "get_recommendation_service",
            return_value=_FailingRecommendationService(),
        ):
            result = search_node.personalized_rank_and_annotate_papers(state)

        step = result.steps[-1]
        self.assertEqual(step.step, "personalized_rerank")
        self.assertEqual(step.status, "failed")
        self.assertEqual(step.outputs.get("personalized_rerank_applied"), False)
        self.assertEqual(step.outputs.get("papers_preserved"), True)


if __name__ == "__main__":
    unittest.main()
