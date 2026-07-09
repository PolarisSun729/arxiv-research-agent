from __future__ import annotations

from services.prompt_context import PromptContextBuilder, TokenCounter


def test_paper_qa_context_assembly_keeps_stable_order_and_budget_debug() -> None:
    builder = PromptContextBuilder(
        prompt_config={"prompt_max_input_tokens": 2200, "prompt_safety_margin_tokens": 0},
        token_counter=TokenCounter(fallback_chars_per_token=1.0),
    )

    assembly = builder.build_paper_qa_final_answer_context(
        question="What is the experimental finding?",
        contextualized_question="What experimental finding is supported by this paper?",
        context_pack={
            "text_context": "RAG evidence " * 80,
            "generation_search_results": [{"source_id": "source-1", "text": "RAG evidence"}],
        },
        user_memory_summary={"profile": {"positive_topics": ["retrieval"], "negative_topics": ["background"]}},
        session_summary={"topic": "experiments", "user_preferences": ["skip background"]},
        recent_turns=[
            {
                "turn_id": "turn-1",
                "question": "Earlier question",
                "answer_summary": "Earlier answer",
                "sources": [{"source_id": "source-1"}],
            }
        ],
        preferred_answer_style="concise",
    )

    debug = assembly["debug"]
    assert debug["prompt_context_mode"] == "block_token_budget"
    assert debug["rendered_section_order"] == [
        "system_instruction",
        "user_memory",
        "session_summary",
        "recent_turns",
        "rag_evidence",
        "current_question",
    ]
    assert debug["total_input_budget_tokens"] == 2200
    assert debug["used_input_tokens"] <= 2200
    assert debug["rendered_tokens"] > 0
    assert all("content" not in section for section in assembly["sections"])
    assert assembly["text"].index("## user_memory") < assembly["text"].index("## session_summary")
    assert assembly["text"].index("## rag_evidence") < assembly["text"].index("## current_question")


def test_generation_search_results_use_single_assembled_context_result() -> None:
    builder = PromptContextBuilder()
    assembly = builder.build_paper_qa_final_answer_context(
        question="Q",
        contextualized_question="Q",
        context_pack={
            "text_context": "Evidence once",
            "generation_search_results": [
                {"source_id": "source-1", "text": "Evidence once"},
                {"source_id": "source-2", "text": "Evidence twice"},
            ],
        },
    )

    search_results = builder.generation_search_results_from_assembly(
        assembly,
        {
            "generation_search_results": [
                {"source_id": "source-1", "text": "Evidence once"},
                {"source_id": "source-2", "text": "Evidence twice"},
            ]
        },
    )

    assert len(search_results) == 1
    assert search_results[0]["source_id"] == "__prompt_context__"
    assert search_results[0]["text"].count("Evidence once") == 1
    assert search_results[0]["metadata"]["original_generation_search_result_count"] == 2
