"""冒烟验证研究引擎流式事件路径。

不走完整 PaperQAService → qa_router → SSE，只测 PaperEvidenceResearchService.research_stream()
能产出预期的渐进事件，并以经过校验的答案正常结束。
"""

from __future__ import annotations

from typing import Any

from services.paper_evidence_research import (
    PaperEvidenceResearchRequest,
    PaperEvidenceResearchService,
    ResearchLimits,
)


class StaticQuestionAnalyzer:
    def analyze(self, request: Any) -> dict[str, Any]:
        return {
            "research_question": request.original_question,
            "evidence_needs": [
                {
                    "need_id": "need-method",
                    "description": "论文方法由哪些阶段组成",
                    "importance": "core",
                    "status": "open",
                    "directly_required_by_question": True,
                }
            ],
        }


class ScriptedPolicy:
    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self._actions = actions
        self._index = 0

    def decide(self, _context: Any) -> dict[str, Any]:
        if self._index >= len(self._actions):
            return {"action": "abstain", "reason_code": "PAPER_DOES_NOT_REPORT_ANSWER"}
        action = self._actions[self._index]
        self._index += 1
        return action


class NoopRetriever:
    def retrieve(self, _action: Any, _state: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "candidates": [
                {
                    "chunk_id": "chunk-1",
                    "content": "The method has three stages.",
                    "rank": 1,
                    "paper_context": {"arxiv_id": "2401.00001"},
                }
            ],
        }


class NoopDraftGenerator:
    def generate(self, _request: Any) -> dict[str, Any]:
        return {
            "answer": "The method has three stages [source:chunk-1].",
            "used_candidate_ids": ["chunk-1"],
        }


class NoopClaimExtractor:
    def extract(self, _request: Any) -> dict[str, Any]:
        return {
            "claims": [
                {
                    "claim_id": "claim-1",
                    "text": "The method has three stages.",
                    "citation_ids": ["chunk-1"],
                    "addressed_need_ids": ["need-method"],
                    "importance": "core",
                }
            ]
        }


class NoopClaimVerifier:
    def verify(self, _request: Any) -> dict[str, Any]:
        return {
            "assessments": [{
                "claim_id": "claim-1", "verdict": "supported",
                "supporting_evidence_ids": ["chunk-1"],
            }],
        }


def test_research_stream_yields_progress_events_and_verified_result() -> None:
    service = PaperEvidenceResearchService(
        question_analyzer=StaticQuestionAnalyzer(),
        decision_policy=ScriptedPolicy(
            [
                {
                    "action": "search_paper",
                    "target_need_id": "need-method",
                    "objective": "discover",
                    "retrieval_mode": "method",
                    "query": "What is the method?",
                    "section_hints": [],
                    "reason_code": "OPEN_EVIDENCE_NEED",
                },
                {
                    "action": "draft_answer",
                    "addressed_need_ids": ["need-method"],
                    "reason_code": "SUFFICIENT_EVIDENCE_ACCUMULATED",
                },
                {
                    "action": "finalize_answer",
                    "reason_code": "ALL_CORE_CLAIMS_SUPPORTED",
                },
            ]
        ),
        retriever=NoopRetriever(),
        draft_generator=NoopDraftGenerator(),
        claim_extractor=NoopClaimExtractor(),
        claim_verifier=NoopClaimVerifier(),
    )

    request = PaperEvidenceResearchRequest(
        arxiv_id="2401.00001",
        original_question="论文的方法由哪些阶段组成？",
        user_id="user-1",
        session_id="session-1",
        research_run_id="smoke-run-1",
        limits=ResearchLimits(max_retrievals=1, max_no_progress=1),
    )

    events = []
    result = None
    gen = service.research_stream(request)
    try:
        while True:
            event = next(gen)
            events.append(event)
    except StopIteration as stop:
        result = stop.value

    expected_events = {"research_started", "retrieval_completed", "draft_created", "claim_verification_completed", "research_completed"}
    observed_events = {event["event"] for event in events}
    assert expected_events <= observed_events, f"Missing events: {expected_events - observed_events}"

    assert result is not None, "Generator did not return result"
    # 只收到结束事件仍可能是协议错误触发的拒答，必须验证证据支持与最终答案一起完成。
    assert result.outcome == "completed"
    assert result.answer == "The method has three stages [source:chunk-1]."
    assert [citation.source_id for citation in result.citations] == ["chunk-1"]

