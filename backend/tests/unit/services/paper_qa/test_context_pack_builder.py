from __future__ import annotations

from services.paper_qa.context_pack_builder import ContextPackBuilder
from services.paper_qa.session_service import PaperQASessionService


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
    text_context, image_inputs, asset_metadata = context_pack_builder.build_generation_context(search_results)
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
