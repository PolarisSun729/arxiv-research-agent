"""检索适配器 × 真实检索管线的集成冒烟。

单测用脚本化管线锁适配器的翻译契约，这里换成 ``build_retrieval_service`` 的真实
``RetrievalPipeline``（query planning → route → fusion → rerank → normalize），验证真实召回结果
能变成可引用的证据候选，并且一次 ``research()`` 的研究轨迹里能按轮次读出 query 与带 rank 的候选。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.helpers import build_retrieval_service

from services.paper_evidence_research import (
    PaperEvidenceResearchRequest,
    PaperEvidenceResearchService,
    ResearchLimits,
)
from services.paper_evidence_research.actions import SearchPaperAction
from services.paper_evidence_research.dependencies import (
    NeedOrchestratedRetriever,
    PaperRetrievalTarget,
    ResearchTraceRecorder,
)
from services.paper_evidence_research.evidence_pool import merge_candidates
from services.paper_evidence_research.state import EvidenceNeed, PaperEvidenceResearchState

_NEEDS = [
    {
        "need_id": "need-method",
        "description": "论文方法由哪些阶段组成",
        "importance": "core",
        "status": "open",
        "directly_required_by_question": True,
    },
    {
        "need_id": "need-experiment",
        "description": "实验用哪些数据集和指标评估",
        "importance": "core",
        "status": "open",
        "directly_required_by_question": True,
    },
]


class StaticQuestionAnalyzer:
    def analyze(self, request: Any) -> dict[str, Any]:
        return {"research_question": request.original_question, "evidence_needs": [dict(need) for need in _NEEDS]}


class ScriptedPolicy:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._actions = iter(actions)

    def decide(self, _context: Any) -> dict[str, Any]:
        return next(self._actions)


class UnusedOrgan:
    """冒烟只驱动检索轮；草稿/主张环节被触发说明动作脚本走偏了。"""

    def generate(self, _request: Any) -> dict[str, Any]:
        raise AssertionError("检索冒烟不应进入草稿生成")

    def extract(self, _request: Any) -> dict[str, Any]:
        raise AssertionError("检索冒烟不应进入主张提取")

    def verify(self, _request: Any) -> dict[str, Any]:
        raise AssertionError("检索冒烟不应进入主张校验")


_SEARCH_ACTIONS = [
    {
        "action": "search_paper",
        "target_need_id": "need-method",
        "objective": "discover",
        "retrieval_mode": "method",
        "query": "What is the method of the paper?",
        "section_hints": ["Method"],
        "reason_code": "OPEN_EVIDENCE_NEED",
    },
    {
        "action": "search_paper",
        "target_need_id": "need-experiment",
        "objective": "discover",
        "retrieval_mode": "experiment",
        "query": "Which dataset and metrics are used for evaluation?",
        "section_hints": ["Experiments"],
        "reason_code": "OPEN_EVIDENCE_NEED",
    },
    {"action": "abstain", "reason_code": "PAPER_DOES_NOT_REPORT_ANSWER"},
]


class NeedOrchestratedRetrieverPipelineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service, self.collection_name, *_ = build_retrieval_service()
        self.retriever = NeedOrchestratedRetriever(
            retrieval_pipeline=self.service.retrieval_pipeline,
            target_resolver=self._resolve_target,
            top_k=3,
        )

    def _resolve_target(self, arxiv_id: str) -> PaperRetrievalTarget:
        return PaperRetrievalTarget(
            collection_name=self.collection_name,
            paper_context={"arxiv_id": arxiv_id, "collection_name": self.collection_name},
        )

    def _request(self, run_id: str) -> PaperEvidenceResearchRequest:
        return PaperEvidenceResearchRequest(
            arxiv_id="2401.00001",
            original_question="论文的方法由哪些阶段组成，实验用什么数据集和指标评估？",
            user_id="user-1",
            session_id="session-1",
            research_run_id=run_id,
            limits=ResearchLimits(max_retrievals=2, max_no_progress=2),
        )

    def _state(self, *, retrieval_count: int = 1) -> PaperEvidenceResearchState:
        return PaperEvidenceResearchState(
            request=self._request("research-run-smoke"),
            research_question="论文的方法由哪些阶段组成？",
            evidence_needs=[EvidenceNeed.model_validate(need) for need in _NEEDS],
            retrieval_count=retrieval_count,
        )

    def _action(self, index: int) -> SearchPaperAction:
        return SearchPaperAction.model_validate(_SEARCH_ACTIONS[index])

    def test_real_recall_becomes_citable_evidence_candidates(self) -> None:
        result = self.retriever.retrieve(self._action(0), self._state())

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["candidates"])
        self.assertEqual([item["rank"] for item in result["candidates"]], sorted(item["rank"] for item in result["candidates"]))

        merged, new_count, duplicate_count = merge_candidates(
            {},
            result["candidates"],
            target_need_id="need-method",
            max_items=20,
            max_total_context_chars=30_000,
        )
        self.assertEqual((new_count, duplicate_count), (len(result["candidates"]), 0))
        # 真实召回必须给出可引用的证据身份：0 是归一化层的未知哨兵值，不能变成候选 ID。
        self.assertNotIn("0", set(merged))
        for candidate in merged.values():
            self.assertTrue(candidate.candidate_id)
            self.assertTrue(candidate.content.strip())
            self.assertEqual(candidate.matched_need_ids, ["need-method"])
        self.assertIn("chunk-method", set(merged))

    def test_final_round_recalls_without_query_rewrite_or_llm_rerank(self) -> None:
        result = self.retriever.retrieve(self._action(0), self._state(retrieval_count=2))

        self.assertTrue(result["lightweight"])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["candidates"])

    def test_research_run_traces_each_round_query_and_ranked_candidates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="research-trace-smoke-") as trace_dir:
            recorder = ResearchTraceRecorder(trace_dir=trace_dir)
            service = PaperEvidenceResearchService(
                question_analyzer=StaticQuestionAnalyzer(),
                decision_policy=ScriptedPolicy(list(_SEARCH_ACTIONS)),
                retriever=self.retriever,
                draft_generator=UnusedOrgan(),
                claim_extractor=UnusedOrgan(),
                claim_verifier=UnusedOrgan(),
                trace_sink=recorder,
            )

            result = service.research(self._request("research-run-pipeline-smoke"))

            trace_path = Path(recorder.trace_path_for(result.research_trace_id))
            self.assertTrue(trace_path.exists())
            payload = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(result.research_summary.retrieval_count, 2)
        rounds = [event for event in payload["events"] if event["event_type"] == "retrieval_completed"]
        self.assertEqual([event["target_need_id"] for event in rounds], ["need-method", "need-experiment"])
        for round_event, action in zip(rounds, _SEARCH_ACTIONS):
            self.assertIn(action["query"], round_event["query"])
            self.assertTrue(round_event["candidates"])
            self.assertEqual(
                [item["rank"] for item in round_event["candidates"]],
                list(range(1, len(round_event["candidates"]) + 1)),
            )
            self.assertTrue(all(item["candidate_id"] for item in round_event["candidates"]))

        # 两轮召回会命中重叠 chunk，但同一 chunk 只能有一轮把它算成新增证据。
        traced_ids = [item["candidate_id"] for event in rounds for item in event["candidates"]]
        self.assertEqual(
            sum(event["new_candidate_count"] for event in rounds),
            len(set(traced_ids)),
        )


if __name__ == "__main__":
    unittest.main()
