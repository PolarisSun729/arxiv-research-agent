from __future__ import annotations

from backend.tests.lightweight_imports import load_arxiv_utils_module


normalizer = load_arxiv_utils_module("paper_question_normalizer")


def _resolved_ref(reference_type: str = "ordinal") -> dict:
    return {
        "final_target_resolved": True,
        "reference_type": reference_type,
        "arxiv_id": "2401.00002",
    }


def test_ordinal_paper_reference_is_rewritten_to_single_paper_context() -> None:
    question = normalizer.normalize_single_paper_qa_question(
        "给我讲一下第二篇论文的核心内容",
        _resolved_ref("ordinal"),
    )

    assert question == "给我讲一下这篇论文的核心内容"


def test_inline_numbered_paper_reference_is_rewritten() -> None:
    question = normalizer.normalize_single_paper_qa_question(
        "这第2篇论文的方法是什么？",
        _resolved_ref("ordinal"),
    )

    assert question == "这篇论文的方法是什么？"


def test_paper_internal_section_reference_is_preserved() -> None:
    question = normalizer.normalize_single_paper_qa_question(
        "讲一下这篇论文第二节的方法",
        _resolved_ref("context_paper"),
    )

    assert question == "讲一下这篇论文第二节的方法"


def test_unresolved_target_does_not_rewrite_ordinal_text() -> None:
    question = normalizer.normalize_single_paper_qa_question(
        "给我讲一下第二篇论文的核心内容",
        {"reference_type": "ordinal", "final_target_resolved": False},
    )

    assert question == "给我讲一下第二篇论文的核心内容"
