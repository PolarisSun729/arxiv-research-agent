from __future__ import annotations

from services.paper_qa.answer_generator import AnswerGenerator
from services.paper_qa.context_pack_builder import ContextPackBuilder
from services.paper_qa.session_service import PaperQASessionService
from tests.helpers import FakeGenerationService


def test_context_pack_builder_covers_text_assets_and_source_payload_fields() -> None:
    long_content = "x" * 400
    search_results = [
        {
            "content": long_content,
            "chunk_type": "text",
            "page_number": 1,
            "source": "body",
            "subchunk_label": "1.1",
            "section_path": "Intro",
            "parent_chunk_id": "p1",
        },
        {
            "chunk_type": "figure",
            "asset_kind": "image",
            "asset_abs_path": "/tmp/figure.png",
            "asset_path": "figure.png",
            "asset_summary": "Figure summary",
            "page_number": 2,
            "section_path": "Method/Figure",
            "source": "figure-source",
        },
        {
            "chunk_type": "table",
            "asset_kind": "table",
            "asset_summary": "Table summary",
            "asset_preview_text": "cell a | cell b",
            "page_number": 3,
            "section_path": "Results/Table",
            "source": "table-source",
        },
    ]

    context_pack_builder = ContextPackBuilder()
    context_pack = context_pack_builder.build(search_results)
    text_context = context_pack["text_context"]
    image_inputs = context_pack["image_inputs"]
    asset_metadata = context_pack["asset_metadata"]
    source_payload = context_pack_builder.build_source_payload(search_results)

    assert long_content in text_context
    assert "[Table 3]" in text_context
    assert image_inputs[0]["image_path"] == "/tmp/figure.png"
    assert len(asset_metadata) == 2
    assert source_payload[0]["source_id"] == "p1"
    assert image_inputs[0]["source_id"] == source_payload[1]["source_id"]
    assert source_payload[0]["parent_chunk_id"] == "p1"
    assert source_payload[1]["asset_summary"] == "Figure summary"
    assert "chunk_type" in source_payload[2]


def test_paper_qa_session_service_truncates_context_text() -> None:
    # 截断规则属于会话上下文职责，避免通过 PaperQAService 重新引入测试专用透传入口。
    truncated = PaperQASessionService.truncate_text("x" * 400, 32)

    assert truncated.endswith("...")
    assert len(truncated) == 32


def test_structured_table_evidence_enters_prompt_and_sources() -> None:
    table_result = {
        "chunk_id": "chunk-table-results",
        "original_chunk_id": "orig-table-results",
        "chunk_type": "table",
        "asset_kind": "table",
        "asset_summary": "Table 2 Main results on the benchmark.",
        "asset_preview_text": "Full model 87.5 | w/o memory 84.1",
        "table_id": "paper-table-2",
        "page_number": 5,
        "section_path": "Results/Ablation",
        "source": "table-source",
        "table_evidence": {
            "schema_version": "table_evidence_v2",
            "table": {
                "table_id": "paper-table-2",
                "caption": "Table 2 Main results on the benchmark.",
                "page_number": 5,
                "section_path": "Results/Ablation",
                "section_title": "Ablation",
                "source_chunk_id": "chunk-table-results",
                "original_chunk_id": "orig-table-results",
            },
            "decision": "compute",
            "operation_hint": "difference",
            "confidence": 0.91,
            "final_evidence": {
                "operation": "difference",
                "rows": ["Full model", "w/o memory"],
                "columns": ["Accuracy"],
                "cells": [
                    {
                        "row_index": 0,
                        "row_label": "Full model",
                        "col_name": "Accuracy",
                        "raw_value": "87.5",
                        "normalized_value": 87.5,
                        "unit": "%",
                        "confidence": 0.94,
                    },
                    {
                        "row_index": 1,
                        "row_label": "w/o memory",
                        "col_name": "Accuracy",
                        "raw_value": "84.1",
                        "normalized_value": 84.1,
                        "unit": "%",
                        "confidence": 0.92,
                    },
                ],
                "calculation": {
                    "operation": "difference",
                    "value": 3.4,
                    "display_value": "3.4",
                    "unit": "percentage_points",
                    "expression": "Full model / Accuracy - w/o memory / Accuracy",
                },
                "reason": "difference_between_reference_and_focus_row",
            },
            "candidate_evidence": None,
            "table_context": {
                "columns": ["Model", "Accuracy"],
                "rows": [],
                "truncated": False,
                "row_count": 2,
                "column_count": 2,
            },
            "reasons": {"matched": ["explicit_metric_column_match"], "ambiguity": [], "fallback": []},
            "debug": {},
        },
    }

    context_pack = ContextPackBuilder().build([table_result])
    source_payload = context_pack["source_payload"]
    text_context = context_pack["text_context"]

    assert source_payload[0]["source_id"].startswith("table-evidence-paper-table-2-chunk-table-results")
    assert "Table Evidence:" in text_context
    assert "table_id: paper-table-2" in text_context
    assert "matched row: Full model, w/o memory" in text_context
    assert "matched column: Accuracy" in text_context
    assert "Full model / Accuracy = 87.5" in text_context
    assert "Full model / Accuracy - w/o memory / Accuracy = 3.4" in text_context
    assert "source chunk: chunk-table-results" in text_context
    assert "Table Supplemental Context:" in text_context
    assert source_payload[0]["table_cell_citations"][1]["row_label"] == "w/o memory"
    assert context_pack["context_budget_debug"]["table_evidence_count"] == 1
    assert context_pack["context_budget_debug"]["table_cell_evidence_count"] == 2
    assert context_pack["context_budget_debug"]["table_numeric_calculation_used"]

    answer_generator = AnswerGenerator(generation_service=FakeGenerationService())
    prompt_assembly = answer_generator.prompt_context_builder.build_paper_qa_final_answer_context(
        question="Ablation 里去掉 memory 后下降多少？",
        contextualized_question="Ablation 里去掉 memory 后下降多少？",
        context_pack=context_pack,
    )
    assert "prefer structured Table Evidence blocks" in prompt_assembly["text"]
    assert prompt_assembly["debug"]["table_evidence_count"] == 1
    assert prompt_assembly["debug"]["table_cell_evidence_count"] == 2
