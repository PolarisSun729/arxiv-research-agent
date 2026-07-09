from __future__ import annotations

from services.retrieval.context_expansion import ContextBudgetSelector


class FakeTraceBuilder:
    def mark_final_context_chunks(self, chunks):
        return [{**chunk, "is_final_context_chunk": True} for chunk in chunks]

    def debug_chunk_item(self, chunk):
        return {"chunk_id": chunk.get("chunk_id"), "context_role": chunk.get("context_role", "")}

    def count_chunk_types(self, chunks):
        counts = {}
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "text")
            counts[chunk_type] = counts.get(chunk_type, 0) + 1
        return counts


def test_context_budget_selector_uses_candidate_max_blocks_for_fallback() -> None:
    selector = ContextBudgetSelector(trace_builder=FakeTraceBuilder())
    chunks = [
        {"chunk_id": f"c-{index}", "content": f"content {index}", "chunk_type": "text"}
        for index in range(5)
    ]

    selected, debug = selector.select(
        reranked_chunks=chunks,
        context_expansion={"candidate_pool": []},
        retrieval_index=None,
        final_context_top_k=2,
        max_context_chars=10,
        enabled=False,
        candidate_max_blocks=4,
        candidate_max_tokens_soft=100,
    )

    assert [chunk["chunk_id"] for chunk in selected] == ["c-0", "c-1", "c-2", "c-3"]
    assert debug["mode"] == "candidate_preselector"
    assert debug["candidate_max_blocks"] == 4
    assert debug["candidate_max_tokens_soft"] == 100


def test_context_budget_selector_limits_candidate_pool_by_blocks() -> None:
    selector = ContextBudgetSelector(trace_builder=FakeTraceBuilder())
    chunks = [
        {"chunk_id": f"c-{index}", "content": "x" * 10, "chunk_type": "text"}
        for index in range(5)
    ]
    context_expansion = {
        "policy": {"name": "default"},
        "anchors": [{"chunk_id": chunk["chunk_id"]} for chunk in chunks],
        "candidate_pool": [
            {
                "candidate_chunk_id": chunk["chunk_id"],
                "relationship_types": ["self"],
                "expansion_source_anchor_ids": [chunk["chunk_id"]],
                "expansion_score": 1.0 - index * 0.1,
                "is_original_retrieval_hit": True,
                "matched_routes": ["vector"],
            }
            for index, chunk in enumerate(chunks)
        ],
    }

    selected, debug = selector.select(
        reranked_chunks=chunks,
        context_expansion=context_expansion,
        retrieval_index=None,
        final_context_top_k=2,
        max_context_chars=10,
        enabled=True,
        candidate_max_blocks=3,
        candidate_max_tokens_soft=100,
    )

    assert len(selected) == 3
    assert debug["mode"] == "candidate_preselector"
    assert debug["candidate_max_blocks"] == 3
    assert debug["max_context_chars"] >= 300
