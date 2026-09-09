from __future__ import annotations

from services.paper_qa.citation_boundary import sanitize_agent_response_citations


def test_agent_final_boundary_strips_mixed_citations_without_llm_repair() -> None:
    result = sanitize_agent_response_citations(
        answer="结论 [source:2, source:4, source:32] source:58",
        paper_qa_result={
            "answer": "结论 [source:2, source:4, source:32] source:58",
            "sources": [{"source_id": "source-1-text-p1", "content": "evidence"}],
        },
    )

    assert result["answer"] == "结论"
    assert result["paper_qa_result"]["answer"] == "结论"
    assert result["citation_warning"]
    assert result["citation_debug"]["invalid_citations"]


def test_agent_final_boundary_keeps_separate_known_citations() -> None:
    result = sanitize_agent_response_citations(
        answer="结论 [source:s1] [source:s2]",
        paper_qa_result={
            "answer": "结论 [source:s1] [source:s2]",
            "sources": [{"source_id": "s1"}, {"source_id": "s2"}],
        },
    )

    assert result["answer"] == "结论 [source:s1] [source:s2]"
    assert result["cited_source_ids"] == ["s1", "s2"]
    assert result["citation_warning"] is None
