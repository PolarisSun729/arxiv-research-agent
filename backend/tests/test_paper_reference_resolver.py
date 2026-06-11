import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_paper_reference_resolver():
    repo_root = Path(__file__).resolve().parents[2]
    utils_dir = repo_root / "backend" / "agents" / "arxiv_search_agent" / "utils"

    packages = {
        "backend": repo_root / "backend",
        "backend.agents": repo_root / "backend" / "agents",
        "backend.agents.arxiv_search_agent": repo_root / "backend" / "agents" / "arxiv_search_agent",
        "backend.agents.arxiv_search_agent.utils": utils_dir,
    }
    for package_name, package_path in packages.items():
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__path__ = [str(package_path)]
        sys.modules[package_name] = module

    for module_name, file_name in (
        ("backend.agents.arxiv_search_agent.utils.text_utils", "text_utils.py"),
        ("backend.agents.arxiv_search_agent.utils.paper_reference_resolver", "paper_reference_resolver.py"),
    ):
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, utils_dir / file_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec and spec.loader
        spec.loader.exec_module(module)

    return sys.modules["backend.agents.arxiv_search_agent.utils.paper_reference_resolver"]


_resolve_paper_reference = _load_paper_reference_resolver()._resolve_paper_reference


class PaperReferenceResolverTests(unittest.TestCase):
    def test_last_paper_reference_is_only_a_context_dependent_hint(self) -> None:
        context = {
            "last_papers": [
                {"arxiv_id": "2505.00001", "title": "Paper 1"},
                {"arxiv_id": "2505.00002", "title": "Paper 2"},
                {"arxiv_id": "2505.00003", "title": "Paper 3"},
            ],
            "selected_paper": {"arxiv_id": "2505.00001", "title": "Paper 1"},
        }

        result = _resolve_paper_reference("给我讲一下最后一篇论文的方法", context)

        self.assertEqual(result["status"], "hint_extracted")
        self.assertEqual(result["reference_type"], "last_item")
        self.assertEqual(result["value"], "last")
        self.assertTrue(result["requires_context"])
        self.assertIsNone(result["arxiv_id"])
        self.assertIsNone(result["paper"])

    def test_last_reference_does_not_fall_back_to_selected_paper(self) -> None:
        context = {
            "last_papers": [
                {"arxiv_id": "2505.00011", "title": "Autofocus Retrieval"},
                {"arxiv_id": "2505.00012", "title": "AdaptR1"},
            ],
            "selected_paper": {"arxiv_id": "2505.00011", "title": "Autofocus Retrieval"},
        }

        result = _resolve_paper_reference("最后一篇", context)

        self.assertEqual(result["status"], "hint_extracted")
        self.assertEqual(result["reference_type"], "last_item")
        self.assertEqual(result["value"], "last")
        self.assertIsNone(result["arxiv_id"])


if __name__ == "__main__":
    unittest.main()
