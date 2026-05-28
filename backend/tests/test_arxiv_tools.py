from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.arxiv_search_service import build_arxiv_query_from_structured_params
from tools import arxiv_tools


class FakeArxivService:
    def __init__(self) -> None:
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "query": kwargs.get("search_query"),
            "id_list": kwargs.get("id_list"),
            "papers": [{"id": "dummy"}],
            "total_results": 1,
            "start_index": kwargs.get("start", 0),
            "items_per_page": 1,
        }


class ArxivToolsTest(TestCase):
    def setUp(self) -> None:
        self.fake_service = FakeArxivService()
        patcher = patch("tools.arxiv_tools.get_arxiv_service", return_value=self.fake_service)
        self.addCleanup(patcher.stop)
        self.mock_get_service = patcher.start()

    def test_raw_query_is_preserved(self):
        result = arxiv_tools.search_arxiv_raw(search_query='all:"large language model" AND cat:cs.CL')
        self.assertTrue(result["ok"])
        self.assertEqual(self.fake_service.calls[0]["search_query"], 'all:"large language model" AND cat:cs.CL')

    def test_raw_id_list_only(self):
        result = arxiv_tools.search_arxiv_raw(id_list=["2404.03868"])
        self.assertTrue(result["ok"])
        self.assertIsNone(self.fake_service.calls[0]["search_query"])
        self.assertEqual(self.fake_service.calls[0]["id_list"], ["2404.03868"])

    def test_raw_query_and_id_list_are_both_passed(self):
        result = arxiv_tools.search_arxiv_raw(search_query="cat:cs.CL", id_list=["2404.03868"])
        self.assertTrue(result["ok"])
        self.assertEqual(self.fake_service.calls[0]["search_query"], "cat:cs.CL")
        self.assertEqual(self.fake_service.calls[0]["id_list"], ["2404.03868"])

    def test_raw_submitted_days_ago_without_search_query_builds_date_only_query(self):
        result = arxiv_tools.search_arxiv_raw(id_list=["2404.03868"], submitted_days_ago=7)
        self.assertTrue(result["ok"])
        self.assertEqual(self.fake_service.calls[0]["id_list"], ["2404.03868"])
        self.assertRegex(self.fake_service.calls[0]["search_query"], r"submittedDate:\[\d{12} TO \d{12}\]")
        self.assertTrue(result["trace"]["submitted_days_ago_applied"])

    def test_raw_negative_submitted_days_ago_is_rejected(self):
        result = arxiv_tools.search_arxiv_raw(search_query="LLM", submitted_days_ago=-1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "arxiv_invalid_query")
        self.assertEqual(self.fake_service.calls, [])

    def test_structured_query_and_categories(self):
        result = arxiv_tools.search_arxiv_structured(
            query="large language model agent",
            categories=["cs.CL", "cs.AI"],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            self.fake_service.calls[0]["search_query"],
            'all:"large language model agent" AND (cat:cs.CL OR cat:cs.AI)',
        )

    def test_structured_title_and_abstract(self):
        result = arxiv_tools.search_arxiv_structured(
            title_query="agent",
            abstract_query="retrieval augmented generation",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            self.fake_service.calls[0]["search_query"],
            'ti:agent AND abs:"retrieval augmented generation"',
        )

    def test_structured_submitted_days_ago_appends_date_query(self):
        result = arxiv_tools.search_arxiv_structured(query="LLM", submitted_days_ago=7)
        self.assertTrue(result["ok"])
        self.assertRegex(
            self.fake_service.calls[0]["search_query"],
            r'all:(?:"LLM"|LLM) AND submittedDate:\[\d{12} TO \d{12}\]',
        )

    def test_invalid_sort_by_is_rejected(self):
        result = arxiv_tools.search_arxiv_raw(search_query="LLM", sort_by="date")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "arxiv_invalid_query")
        self.assertEqual(self.fake_service.calls, [])

    def test_invalid_max_results_is_rejected(self):
        result = arxiv_tools.search_arxiv_raw(search_query="LLM", max_results=10000)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "arxiv_invalid_query")
        self.assertEqual(self.fake_service.calls, [])

    def test_structured_legacy_args_are_rewritten(self):
        result = arxiv_tools.search_arxiv_structured(
            query="LLM agent",
            categories=["cs.CL"],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            self.fake_service.calls[0]["search_query"],
            'all:"LLM agent" AND cat:cs.CL',
        )

    def test_builder_uses_or_for_categories(self):
        built = build_arxiv_query_from_structured_params(
            query="LLM agent",
            categories=["cs.CL", "cs.AI"],
        )
        self.assertEqual(built["final_search_query"], 'all:"LLM agent" AND (cat:cs.CL OR cat:cs.AI)')
