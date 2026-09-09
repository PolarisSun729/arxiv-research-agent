"""评测用真实研究结果和人工可核对的合成证据，避免自造不存在的结果字段。"""

import pytest

from services.evaluation.eval_record import build_eval_record
from tests.unit.services.paper_evidence_research.test_research_service import ScriptedDecisionPolicy, _request, _service


@pytest.fixture
def golden_case():
    return {
        "case_id": "contract", "arxiv_id": "2310.11511", "question": "How do reflection tokens work?",
        "difficulty": "medium", "main_intent": "method_flow",
        "expected_answer_points": ["reflection token"],
        "expected_chunk_ids": ["chunk-method", "chunk-ablation"], "answerable": True,
        "gold_answer": "Reflection tokens decide whether retrieval is required.",
    }


@pytest.fixture
def research_run():
    request, events = _request(run_id="evaluation-unit"), []
    result = _service(ScriptedDecisionPolicy()).research(request, trace_listener=lambda _, trace: events.extend(trace))
    return request, result, events


@pytest.fixture
def completed_record(research_run):
    request, result, events = research_run
    return build_eval_record(request=request, result=result, trace_events=events)
