from __future__ import annotations

from types import SimpleNamespace

from services.retrieval.context_expansion import ContextExpansionPreparer, ExpansionPolicy


def _index_with_parent(chunks):
    """构造只含 documents/by_parent_chunk_id 的最小索引；每个 chunk 有 body 与 summary 两条 index 文档。"""
    documents = []
    for chunk in chunks:
        documents.append(SimpleNamespace(chunk=chunk))
        documents.append(SimpleNamespace(chunk=chunk))
    return SimpleNamespace(documents=documents, by_parent_chunk_id={"p-1": list(range(len(documents)))})


def _siblings(chunks, anchor_chunk):
    preparer = ContextExpansionPreparer(trace_builder=None)
    anchor = preparer._build_anchor(anchor_chunk, original_rank=1, final_keys=set(), query_profile=None)
    candidate_map = {}
    preparer._add_parent_siblings(
        candidate_map,
        anchor=anchor,
        retrieval_index=_index_with_parent(chunks),
        final_keys=set(),
        original_hit_map={},
        policy=ExpansionPolicy(name="test", actions=("sibling",), sibling_window=1),
    )
    return {(item.expansion_type, item.candidate_chunk_id, item.distance) for item in candidate_map.values()}


def test_siblings_without_subchunk_index_use_relative_reading_order() -> None:
    chunks = [
        {"chunk_id": f"c-{index}", "parent_chunk_id": "p-1", "order_index": index, "content": f"part {index}"}
        for index in range(4)
    ]

    found = _siblings(chunks, chunks[1])

    assert found == {("sibling", "c-0", -1), ("self", "c-1", 0), ("sibling", "c-2", 1)}


def test_siblings_with_subchunk_index_keep_index_difference() -> None:
    chunks = [
        {"chunk_id": f"c-{index}", "parent_chunk_id": "p-1", "subchunk_index": index + 1, "content": f"part {index}"}
        for index in range(4)
    ]

    found = _siblings(chunks, chunks[2])

    assert found == {("sibling", "c-1", -1), ("self", "c-2", 0), ("sibling", "c-3", 1)}
