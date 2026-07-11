from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping

from tests.helpers.retrieval import build_sample_chunks


DATA_PATH = Path(__file__).resolve().parent / "data" / "smoke_golden_set.jsonl"


class _FakeGoldenSmokeRunner:
    def __init__(self) -> None:
        self._chunks = build_sample_chunks()
        self._chunk_map = {
            str(chunk.get("metadata", {}).get("chunk_id")): self._normalize_chunk(chunk)
            for chunk in self._chunks
        }
        self._routing = {
            "method_flow": ["chunk-method", "chunk-results", "chunk-experiment"],
            "experiment_setup": ["chunk-experiment", "chunk-dataset", "chunk-results"],
            "comparison": ["chunk-results", "chunk-figure-table", "chunk-experiment"],
            "limitation": ["chunk-limitation", "chunk-method", "chunk-results"],
            "dataset": ["chunk-dataset", "chunk-experiment", "chunk-results"],
            "figure_table": ["chunk-figure-table", "chunk-results", "chunk-experiment"],
        }

    @staticmethod
    def _normalize_chunk(chunk: Mapping[str, Any]) -> Dict[str, Any]:
        metadata = dict(chunk.get("metadata", {}) or {})
        return {
            "chunk_id": metadata.get("chunk_id"),
            "parent_chunk_id": metadata.get("parent_chunk_id"),
            "page_number": metadata.get("page_number"),
            "section_path": metadata.get("section_path"),
            "section_title": metadata.get("section_title"),
            "chunk_type": metadata.get("chunk_type"),
            "content": str(chunk.get("content", "") or metadata.get("asset_summary", "")),
            "metadata": metadata,
        }

    def retrieve(self, main_intent: str, top_k: int = 3) -> List[Dict[str, Any]]:
        chunk_ids = self._routing[main_intent][: max(1, int(top_k or 3))]
        return [dict(self._chunk_map[chunk_id]) for chunk_id in chunk_ids]

    def answer(self, main_intent: str, language: str, sources: List[Mapping[str, Any]]) -> str:
        source_text = " ".join(str(source.get("content", "")) for source in sources)
        summaries = {
            "method_flow": {
                "en": "The method uses a retrieval pipeline with two encoder stages.",
                "zh": "方法流程基于 retrieval pipeline，并且包含 two encoder stages。",
            },
            "experiment_setup": {
                "en": "The experiments use the LongBench dataset and exact match metrics.",
                "zh": "实验部分使用 LongBench dataset，并采用 exact match metrics 作为评估指标。",
            },
            "comparison": {
                "en": "The main results show better performance and comparison against baselines.",
                "zh": "主要结果显示模型有更好的性能，并且完成了 comparison against baselines。",
            },
            "limitation": {
                "en": "The paper struggles on noisy prompts and highlights future work.",
                "zh": "论文提到的局限性包括 noisy prompts 下表现较差，并指出 future work。",
            },
            "dataset": {
                "en": "The dataset includes training, dev, and test splits.",
                "zh": "数据集包含训练集、dev 和 test splits。",
            },
            "figure_table": {
                "en": "Figure 2 summarizes the experimental trend and Table 3 reports the best score.",
                "zh": "图2总结了实验趋势，表3给出了 best score。",
            },
        }
        lang_key = "zh" if language == "zh" else "en"
        return f"{summaries[main_intent][lang_key]} Evidence: {source_text}"


def _load_golden_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _matches_answer_points(answer: str, expected_points: List[str]) -> Dict[str, Any]:
    lowered_answer = str(answer or "").lower()
    point_results = []
    for point in expected_points:
        point_text = str(point or "")
        aliases = {
            "方法流程": ["方法流程", "retrieval pipeline"],
            "评估指标": ["评估指标", "metrics"],
            "主要结果": ["主要结果", "better performance"],
            "局限性": ["局限性", "struggles"],
            "训练集": ["训练集", "training"],
            "趋势": ["趋势", "trend"],
        }.get(point_text, [point_text])
        matched = any(alias.lower() in lowered_answer for alias in aliases)
        point_results.append({"point": point_text, "matched": matched})
    matched_count = sum(1 for item in point_results if item["matched"])
    return {
        "matched": matched_count == len(point_results),
        "matched_count": matched_count,
        "total": len(point_results),
        "details": point_results,
    }


def _evaluate_source_constraints(sources: List[Mapping[str, Any]], constraints: Mapping[str, Any], top_k: int) -> Dict[str, Any]:
    required_pages = set(int(page) for page in constraints.get("required_page_numbers", []) or [])
    required_sections = [str(item) for item in constraints.get("required_section_substrings", []) or []]
    required_keywords = [str(item) for item in constraints.get("required_chunk_keywords", []) or []]

    source_pages = {int(source.get("page_number")) for source in sources if source.get("page_number") is not None}
    page_hit = required_pages.issubset(source_pages)

    section_hit = True
    for section in required_sections:
        if not any(section.lower() in str(source.get("section_path") or "").lower() for source in sources):
            section_hit = False
            break

    keyword_hit = True
    for keyword in required_keywords:
        if not any(keyword.lower() in str(source.get("content") or "").lower() for source in sources):
            keyword_hit = False
            break

    unique_chunk_ids = {str(source.get("chunk_id")) for source in sources}
    return {
        "hit_at_k": bool(page_hit and section_hit and keyword_hit),
        "top_k": top_k,
        "page_hit": page_hit,
        "section_hit": section_hit,
        "keyword_hit": keyword_hit,
        "source_count": len(sources),
        "unique_source_count": len(unique_chunk_ids),
        "sources_deduped": len(unique_chunk_ids) == len(sources),
        "required_pages": sorted(required_pages),
        "observed_pages": sorted(source_pages),
    }


class RagGoldenSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = _FakeGoldenSmokeRunner()
        self.cases = _load_golden_cases()

    def test_smoke_golden_file_has_expected_size_and_fields(self) -> None:
        self.assertGreaterEqual(len(self.cases), 10)
        self.assertLessEqual(len(self.cases), 20)
        required_fields = {
            "case_id",
            "arxiv_id",
            "question",
            "main_intent",
            "expected_answer_points",
            "expected_source_constraints",
            "difficulty",
            "language",
        }
        for case in self.cases:
            with self.subTest(case_id=case.get("case_id")):
                self.assertTrue(required_fields.issubset(case.keys()))

    def test_rag_golden_smoke_framework_runs_and_reports_metrics(self) -> None:
        top_k = 3
        case_reports = []
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                sources = self.runner.retrieve(case["main_intent"], top_k=top_k)
                answer = self.runner.answer(case["main_intent"], case["language"], sources)
                source_eval = _evaluate_source_constraints(sources, case["expected_source_constraints"], top_k=top_k)
                answer_eval = _matches_answer_points(answer, list(case["expected_answer_points"] or []))

                report = {
                    "case_id": case["case_id"],
                    "main_intent": case["main_intent"],
                    "hit_at_k": source_eval["hit_at_k"],
                    "page_hit": source_eval["page_hit"],
                    "section_hit": source_eval["section_hit"],
                    "source_count": source_eval["source_count"],
                    "unique_source_count": source_eval["unique_source_count"],
                    "sources_deduped": source_eval["sources_deduped"],
                    "answer_points": answer_eval,
                }
                case_reports.append(report)

                self.assertTrue(source_eval["hit_at_k"], msg=report)
                self.assertTrue(source_eval["page_hit"], msg=report)
                self.assertTrue(source_eval["section_hit"], msg=report)
                self.assertGreaterEqual(source_eval["source_count"], 1, msg=report)
                self.assertTrue(source_eval["sources_deduped"], msg=report)
                self.assertTrue(answer_eval["matched"], msg=report)

        total = len(case_reports)
        summary = {
            "total_cases": total,
            "hit_at_k": {
                "passed": sum(1 for item in case_reports if item["hit_at_k"]),
                "rate": round(sum(1 for item in case_reports if item["hit_at_k"]) / total, 4),
            },
            "expected_page_or_section_hit": {
                "passed": sum(1 for item in case_reports if item["page_hit"] and item["section_hit"]),
                "rate": round(sum(1 for item in case_reports if item["page_hit"] and item["section_hit"]) / total, 4),
            },
            "source_count": {
                "min": min(item["source_count"] for item in case_reports),
                "max": max(item["source_count"] for item in case_reports),
                "avg": round(sum(item["source_count"] for item in case_reports) / total, 2),
            },
            "source_dedup": {
                "all_deduped": all(item["sources_deduped"] for item in case_reports),
                "unique_source_counts": sorted({item["unique_source_count"] for item in case_reports}),
            },
            "answer_points": {
                "passed": sum(1 for item in case_reports if item["answer_points"]["matched"]),
                "rate": round(sum(1 for item in case_reports if item["answer_points"]["matched"]) / total, 4),
            },
        }
        print("\nRAG golden smoke summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
