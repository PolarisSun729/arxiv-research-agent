"""研究轨迹落盘与检索候选事件的契约测试。

轨迹是量化评测（Recall@k / MRR / 引用一致性）唯一的数据源，而它不在 ``PaperEvidenceResearchResult``
契约里；因此这里锁两件事：一次 ``research()`` 调用能按 run_id 找到轨迹文件，文件里能看到每一轮
检索的 query 与带 rank 的候选 chunk_id。
"""

from __future__ import annotations

import json
from typing import Any

from services.paper_evidence_research import (
    PaperEvidenceResearchRequest,
    PaperEvidenceResearchService,
    ResearchLimits,
)
from services.paper_evidence_research.dependencies import (
    DEFAULT_RESEARCH_TRACE_DIR,
    ResearchTraceRecorder,
)


class TwoNeedQuestionAnalyzer:
    def analyze(self, request: Any) -> dict[str, Any]:
        return {
            "research_question": request.original_question,
            "evidence_needs": [
                {
                    "need_id": "need-method",
                    "description": "反思 token 触发检索的判定机制",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                },
                {
                    "need_id": "need-ablation",
                    "description": "消融实验是否支持反思模块有效",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                },
            ],
        }


class RankedRetriever:
    """模拟适配器输出：候选自带 rank/score，且第二轮会重复回传第一轮已入池的 chunk。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, action: Any, _state: Any) -> dict[str, Any]:
        query = f"{action.query} | {action.target_need_id}"
        self.queries.append(query)
        if action.target_need_id == "need-method":
            candidates = [
                {
                    "content": "The reflection token decides whether retrieval is required.",
                    "chunk_id": "chunk-method",
                    "section_path": "2 Method",
                    "rank": 1,
                    "score": 0.9,
                },
                {
                    "content": "Both sections describe the shared reflection module.",
                    "chunk_id": "chunk-shared",
                    "section_path": "2 Method",
                    "rank": 2,
                    "score": 0.7,
                },
            ]
        else:
            candidates = [
                {
                    "content": "Both sections describe the shared reflection module.",
                    "chunk_id": "chunk-shared",
                    "section_path": "2 Method",
                    "rank": 1,
                    "score": 0.8,
                },
                {
                    "content": "Without reflection tokens the score drops from 54.1 to 50.3.",
                    "chunk_id": "chunk-ablation",
                    "chunk_type": "table",
                    "section_path": "5 Experiments/Ablation",
                    "rank": 2,
                    "score": 0.6,
                },
            ]
        return {"status": "completed", "candidates": candidates, "query": query}


class ScriptedPolicy:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._actions = iter(actions)

    def decide(self, _context: Any) -> dict[str, Any]:
        return next(self._actions)


class UnusedDraftGenerator:
    def generate(self, _request: Any) -> dict[str, Any]:
        raise AssertionError("轨迹用例只驱动检索轮，不应触发草稿生成")


class UnusedClaimExtractor:
    def extract(self, _request: Any) -> dict[str, Any]:
        raise AssertionError("轨迹用例只驱动检索轮，不应触发主张提取")


class UnusedClaimVerifier:
    def verify(self, _request: Any) -> dict[str, Any]:
        raise AssertionError("轨迹用例只驱动检索轮，不应触发主张校验")


_SEARCH_ACTIONS = [
    {
        "action": "search_paper",
        "target_need_id": "need-method",
        "objective": "discover",
        "retrieval_mode": "method",
        "query": "reflection token retrieval decision",
        "reason_code": "OPEN_EVIDENCE_NEED",
    },
    {
        "action": "search_paper",
        "target_need_id": "need-ablation",
        "objective": "discover",
        "retrieval_mode": "experiment",
        "query": "reflection token ablation results",
        "reason_code": "OPEN_EVIDENCE_NEED",
    },
    {"action": "abstain", "reason_code": "PAPER_DOES_NOT_REPORT_ANSWER"},
]


def _service(*, retriever: Any, trace_sink: Any = None) -> PaperEvidenceResearchService:
    return PaperEvidenceResearchService(
        question_analyzer=TwoNeedQuestionAnalyzer(),
        decision_policy=ScriptedPolicy(list(_SEARCH_ACTIONS)),
        retriever=retriever,
        draft_generator=UnusedDraftGenerator(),
        claim_extractor=UnusedClaimExtractor(),
        claim_verifier=UnusedClaimVerifier(),
        trace_sink=trace_sink,
    )


def _request(run_id: str) -> PaperEvidenceResearchRequest:
    return PaperEvidenceResearchRequest(
        arxiv_id="2401.00001",
        original_question="反思 token 如何决定是否检索，消融实验是否支持它有效？",
        user_id="user-1",
        session_id="session-1",
        research_run_id=run_id,
        # 两轮检索都要真正跑到，纯重复轮不能提前触发有界终止。
        limits=ResearchLimits(max_retrievals=2, max_no_progress=2),
    )


def _retrieval_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [event for event in payload["events"] if event["event_type"] == "retrieval_completed"]


def test_one_research_run_traces_each_round_query_and_ranked_candidates(tmp_path: Any) -> None:
    recorder = ResearchTraceRecorder(trace_dir=tmp_path)
    retriever = RankedRetriever()

    result = _service(retriever=retriever, trace_sink=recorder).research(_request("research-run-trace"))

    # 评测记录只持有 research_trace_id，必须能凭它单独复算出轨迹文件路径。
    trace_path = recorder.trace_path_for(result.research_trace_id)
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "research-run-trace"
    assert payload["event_count"] == len(payload["events"])

    rounds = _retrieval_events(payload)
    assert [event["query"] for event in rounds] == retriever.queries
    assert rounds[0]["candidates"] == [
        {"candidate_id": "chunk-method", "rank": 1, "score": 0.9},
        {"candidate_id": "chunk-shared", "rank": 2, "score": 0.7},
    ]
    assert rounds[1]["candidates"] == [
        {"candidate_id": "chunk-shared", "rank": 1, "score": 0.8},
        {"candidate_id": "chunk-ablation", "rank": 2, "score": 0.6},
    ]


def test_a_chunk_recalled_twice_enters_the_evidence_pool_once(tmp_path: Any) -> None:
    """chunk-shared 被两轮都召回，但只在第一轮入池；第二轮只记一次重复。"""

    recorder = ResearchTraceRecorder(trace_dir=tmp_path)

    _service(retriever=RankedRetriever(), trace_sink=recorder).research(_request("research-run-dedup"))

    payload = json.loads(recorder.trace_path_for("research-run-dedup").read_text(encoding="utf-8"))
    rounds = _retrieval_events(payload)
    assert [event["new_candidate_count"] for event in rounds] == [2, 1]
    assert [event["duplicate_candidate_count"] for event in rounds] == [0, 1]
    # 三个不同 chunk 被召回四次，入池的只有三个。
    assert sum(event["new_candidate_count"] for event in rounds) == 3


def test_run_id_is_the_trace_file_name_and_unsafe_characters_are_normalized() -> None:
    recorder = ResearchTraceRecorder(trace_dir="temp/research-traces")

    assert recorder.trace_path_for("research-run-1").name == "research-run-1.json"
    assert recorder.trace_path_for("run/../id 1").name == "run_.._id_1.json"
    assert recorder.trace_path_for("").name == "research.json"
    assert ResearchTraceRecorder().trace_dir == DEFAULT_RESEARCH_TRACE_DIR


def test_research_result_survives_a_failing_trace_sink() -> None:
    """轨迹是审计产物，写盘故障不能吞掉一次已经跑完的研究。"""

    def _explode(_run_id: str, _events: list[dict[str, Any]]) -> None:
        raise OSError("disk full")

    result = _service(retriever=RankedRetriever(), trace_sink=_explode).research(_request("research-run-sink-failure"))

    assert result.outcome == "abstained"
    assert result.research_summary.retrieval_count == 2


def test_no_trace_file_is_written_without_a_sink(tmp_path: Any) -> None:
    _service(retriever=RankedRetriever()).research(_request("research-run-no-sink"))

    assert list(tmp_path.iterdir()) == []
