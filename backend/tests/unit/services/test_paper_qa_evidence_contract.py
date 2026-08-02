from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from services.paper_qa.answer_generator import AnswerGenerator
from services.paper_qa.evidence_contract import build_public_source_payload, resolve_evidence_asset_path


def test_citation_protocol_accepts_only_stable_source_markers() -> None:
    source_ids = ["chunk-1", "figure-2"]

    assert AnswerGenerator.extract_cited_source_ids("结论 [source:chunk-1]", source_ids) == ["chunk-1"]
    assert AnswerGenerator.extract_cited_source_ids("结论 [Source 1] [Image 1]", source_ids) == []

    debug = AnswerGenerator.validate_citations("结论 [source:unknown] [Source 1]", source_ids)
    assert debug["valid"] is False
    assert "unknown" in debug["invalid_citations"]
    assert "[Source 1]" in debug["invalid_citations"]
    assert AnswerGenerator.strip_invalid_citations("结论 [source:unknown] [Source 1]", source_ids) == "结论"


def test_answer_generator_repairs_invalid_citations_once() -> None:
    class Generation:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            return {"response": "答案 [source:unknown]" if self.calls == 1 else "答案 [source:s1]"}

    generation = Generation()
    result = AnswerGenerator(generation_service=generation).generate(
        generation_question="question",
        context_pack={
            "source_payload": [{"source_id": "s1", "content": "evidence"}],
            "prompt_blocks": [],
            "generation_search_results": [],
            "image_inputs": [],
            "asset_metadata": [],
        },
    )

    assert generation.calls == 2
    assert result["cited_source_ids"] == ["s1"]
    assert result["citation_debug"]["repair_succeeded"] is True


def test_public_source_payload_does_not_expose_local_asset_paths() -> None:
    payload = build_public_source_payload(
        [
            {
                "source_id": "figure-2",
                "chunk_type": "figure",
                "content": "figure caption",
                "asset_path": "03-docling-assets/paper/figure-2.png",
                "asset_abs_path": "C:/private/figure-2.png",
            }
        ],
        arxiv_id="2401.00001",
    )

    assert payload[0]["source_id"] == "figure-2"
    assert payload[0]["asset_url"] == "/api/paper/2401.00001/evidence-assets/figure-2"
    assert "asset_path" not in payload[0]
    assert "asset_abs_path" not in payload[0]


def test_evidence_asset_lookup_is_index_bound_and_rejects_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    asset_root = tmp_path / "03-docling-assets" / "2401.00001"
    asset_root.mkdir(parents=True)
    asset_path = asset_root / "figure-2.png"
    asset_path.write_bytes(b"png")

    class Store:
        def get_paper_qa_index(self, arxiv_id: str):
            return {"status": "indexed", "collection_name": "paper_2401"} if arxiv_id == "2401.00001" else None

    class Provider:
        def get_index(self, collection_name: str, *, index_record):
            assert collection_name == "paper_2401"
            return SimpleNamespace(
                documents=[
                    SimpleNamespace(
                        chunk={
                            "chunk_id": "figure-2",
                            "chunk_type": "figure",
                            "asset_abs_path": str(asset_path),
                        }
                    )
                ]
            )

    assert resolve_evidence_asset_path(
        arxiv_id="2401.00001",
        source_id="figure-2",
        paper_qa_index_store=Store(),
        enhanced_retrieval_service=SimpleNamespace(collection_retrieval_index_provider=Provider()),
    ) == str(asset_path.resolve())
    with pytest.raises(FileNotFoundError):
        resolve_evidence_asset_path(
            arxiv_id="2401.00001",
            source_id="../figure-2",
            paper_qa_index_store=Store(),
            enhanced_retrieval_service=SimpleNamespace(collection_retrieval_index_provider=Provider()),
        )
    with pytest.raises(FileNotFoundError):
        resolve_evidence_asset_path(
            arxiv_id="2401.00001",
            source_id="figure-3",
            paper_qa_index_store=Store(),
            enhanced_retrieval_service=SimpleNamespace(collection_retrieval_index_provider=Provider()),
        )


@pytest.mark.parametrize(
    "answer",
    [
        "结论 [source:2, source:4, source:32]",
        "结论 [source:s1, source:s2]",
        "结论 source:58",
        "结论 source:source-58",
        "结论 [Source 1]",
        "结论 [Image 1]",
    ],
)
def test_citation_protocol_rejects_mixed_or_legacy_shapes(answer: str) -> None:
    debug = AnswerGenerator.validate_citations(answer, ["s1", "s2"])

    assert debug["valid"] is False
    assert debug["invalid_citations"]


def test_strip_invalid_citations_removes_bare_markers_but_preserves_body() -> None:
    answer = "结论 [source:2, source:4] source:58，数据源是论文正文。"

    assert AnswerGenerator.strip_invalid_citations(answer, ["s1"]) == "结论，数据源是论文正文。"


def test_strip_invalid_citations_does_not_remove_ordinary_source_phrase() -> None:
    answer = "The data source is the paper registry."

    assert AnswerGenerator.strip_invalid_citations(answer, ["s1"]) == answer


def test_answer_generator_repairs_grouped_citations_once() -> None:
    class Generation:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            return {
                "response": (
                    "结论 [source:2, source:4, source:32]"
                    if self.calls == 1
                    else "结论 [source:s1] [source:s2]"
                )
            }

    generation = Generation()
    result = AnswerGenerator(generation_service=generation).generate(
        generation_question="question",
        context_pack={
            "source_payload": [
                {"source_id": "s1", "content": "evidence 1"},
                {"source_id": "s2", "content": "evidence 2"},
            ],
            "prompt_blocks": [],
            "generation_search_results": [],
            "image_inputs": [],
            "asset_metadata": [],
        },
    )

    assert generation.calls == 2
    assert result["answer"] == "结论 [source:s1] [source:s2]"
    assert result["citation_debug"]["repair_succeeded"] is True


def test_answer_generator_strips_invalid_repair_without_losing_answer_body() -> None:
    class Generation:
        def generate(self, **_kwargs):
            return {"response": "结论 [source:2, source:4] source:58"}

    result = AnswerGenerator(generation_service=Generation()).generate(
        generation_question="question",
        context_pack={
            "source_payload": [{"source_id": "s1", "content": "evidence"}],
            "prompt_blocks": [],
            "generation_search_results": [],
            "image_inputs": [],
            "asset_metadata": [],
        },
    )

    assert result["answer"] == "结论"
    assert result["citation_warning"]
    assert result["citation_debug"]["repair_succeeded"] is False


def test_answer_generator_requires_chinese_final_answer() -> None:
    class Generation:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, **kwargs):
            self.requests.append(kwargs)
            return {"response": "中文答案 [source:s1]"}

    generation = Generation()
    AnswerGenerator(generation_service=generation).generate(
        generation_question="请解释论文方法",
        context_pack={
            "source_payload": [{"source_id": "s1", "content": "evidence"}],
            "prompt_blocks": [],
            "generation_search_results": [],
            "image_inputs": [],
            "asset_metadata": [],
        },
    )

    query = generation.requests[0]["query"]
    assert "必须使用中文回复" in query
    assert "不要用英文整段回答" in query
