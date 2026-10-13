from __future__ import annotations

import pytest

from services.paper_qa.question_contextualizer import QuestionContextualizer


@pytest.mark.parametrize(
    "question",
    [
        "它的训练数据是什么？",
        "他们用了什么数据集",
        "那实验部分呢？",
        "第二步具体做了什么",
        "为什么？",
        "然后呢",
        "How does it compare with BERT?",
        "What datasets are used with this approach",
    ],
)
def test_follow_up_questions_are_detected(question: str) -> None:
    assert QuestionContextualizer.looks_like_follow_up(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "与其他工作相比，本文的主要创新点是什么？",
        "尤其在长文本上表现怎样？",
        "这篇论文用了哪些 baseline？",
        "Transformer 的多头注意力机制是如何计算的？",
        "介绍一下论文的方法部分",
        "What is the main contribution with respect to prior work?",
        "",
    ],
)
def test_standalone_questions_are_not_treated_as_follow_up(question: str) -> None:
    assert QuestionContextualizer.looks_like_follow_up(question) is False


def test_extract_json_object_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        QuestionContextualizer.extract_json_object('["not", "an", "object"]')
