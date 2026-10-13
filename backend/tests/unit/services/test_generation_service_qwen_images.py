from __future__ import annotations

import os
from pathlib import Path

import pytest

from services.llm import generation_service as generation_module
from services.llm.generation_service import GenerationService, QWEN_RESPONSES_IMAGE_TARGET_BYTES


def test_qwen_input_without_images_keeps_plain_prompt_shape() -> None:
    service = GenerationService()

    qwen_input, debug = service._build_qwen_input_with_debug(
        query="What is the finding?",
        context="Evidence text",
    )

    assert isinstance(qwen_input, str)
    assert "证据上下文：\nEvidence text" in qwen_input
    assert "必须使用中文回复" in qwen_input
    assert debug["qwen_multimodal_enabled"] is False
    assert debug["sent_image_count"] == 0


def test_qwen_image_input_compresses_large_png_before_data_url(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "large-random.png"
    image = Image.frombytes("RGB", (1200, 1200), os.urandom(1200 * 1200 * 3))
    image.save(image_path, format="PNG")
    assert image_path.stat().st_size > QWEN_RESPONSES_IMAGE_TARGET_BYTES

    service = GenerationService()
    qwen_input, debug = service._build_qwen_input_with_debug(
        query="Explain Figure 2",
        context="Figure evidence text",
        image_inputs=[
            {
                "source_id": "figure-2",
                "image_path": str(image_path),
                "page_number": "3",
                "asset_summary": "Workflow figure",
                "section_path": "Method",
            }
        ],
        asset_metadata=[
            {
                "page_number": "3",
                "asset_summary": "Workflow figure",
                "section_path": "Method",
            }
        ],
    )

    content = qwen_input[0]["content"]
    assert "必须使用中文回复" in content[0]["text"]
    assert "[source:image-1]" in content[0]["text"]
    image_urls = [item["image_url"] for item in content if item["type"] == "input_image"]
    assert len(image_urls) == 1
    assert image_urls[0].startswith("data:image/jpeg;base64,")
    assert debug["sent_image_count"] == 1
    assert debug["images"][0]["compressed"] is True
    assert debug["images"][0]["compressed_bytes"] <= QWEN_RESPONSES_IMAGE_TARGET_BYTES


def test_qwen_image_input_skips_image_when_request_budget_would_be_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GenerationService()

    def fake_prepare(image_path: str, *, target_bytes: int) -> dict[str, object]:
        return {
            "data_url": "data:image/jpeg;base64," + ("A" * 5000),
            "debug": {
                "original_bytes": 4_000_000,
                "compressed": True,
                "compressed_bytes": 2_000_000,
                "output_mime_type": "image/jpeg",
            },
        }

    monkeypatch.setattr(service, "_prepare_qwen_image_data_url", fake_prepare)
    monkeypatch.setattr(generation_module, "QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES", 1000)

    qwen_input, debug = service._build_qwen_input_with_debug(
        query="Explain the figure",
        context="Text evidence remains available",
        image_inputs=[{"source_id": "figure-1", "image_path": "fake.png", "asset_summary": "Large figure"}],
        asset_metadata=[{"asset_summary": "Large figure"}],
    )

    assert qwen_input[0]["content"][0]["type"] == "input_text"
    assert all(item["type"] != "input_image" for item in qwen_input[0]["content"])
    assert debug["sent_image_count"] == 0
    assert debug["dropped_image_count"] == 1
    assert debug["images"][0]["skip_reason"] == "request_body_soft_limit_exceeded"
    assert debug["images"][0]["fallback_to_asset_summary"] is True


def test_qwen_image_input_respects_max_image_count(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GenerationService()
    monkeypatch.setattr(generation_module, "QWEN_RESPONSES_MAX_IMAGE_COUNT", 2)

    def fake_prepare(image_path: str, *, target_bytes: int) -> dict[str, object]:
        return {
            "data_url": "data:image/jpeg;base64,AA",
            "debug": {
                "original_bytes": 20,
                "compressed": False,
                "compressed_bytes": 20,
                "output_mime_type": "image/jpeg",
            },
        }

    monkeypatch.setattr(service, "_prepare_qwen_image_data_url", fake_prepare)
    qwen_input, debug = service._build_qwen_input_with_debug(
        query="Compare the figures",
        context="Text evidence",
        image_inputs=[
            {"source_id": "figure-1", "image_path": "one.png"},
            {"source_id": "figure-2", "image_path": "two.png"},
            {"source_id": "figure-3", "image_path": "three.png"},
        ],
        asset_metadata=[],
    )

    image_parts = [item for item in qwen_input[0]["content"] if item["type"] == "input_image"]
    assert len(image_parts) == 2
    assert debug["sent_image_count"] == 2
    assert debug["dropped_image_count"] == 1
    assert debug["images"][2]["skip_reason"] == "max_image_count_exceeded"


def test_image_limits_come_from_generation_config() -> None:
    from utils.config import GENERATION_CONFIG

    assert generation_module.QWEN_RESPONSES_MAX_IMAGE_COUNT == GENERATION_CONFIG["qwen_responses_max_image_count"]
    assert generation_module.QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES == GENERATION_CONFIG[
        "qwen_responses_request_body_soft_limit_bytes"
    ]
    # 图多时单图预算变小，压缩档位要有足够小的尺寸兜底。
    assert generation_module.QWEN_RESPONSES_IMAGE_COMPRESSION_STEPS[-1] == (640, 55)


def test_image_target_bytes_adapt_to_the_number_of_images(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GenerationService()
    targets: list[int] = []

    def fake_prepare(image_path: str, *, target_bytes: int) -> dict[str, object]:
        targets.append(target_bytes)
        return {"data_url": "data:image/jpeg;base64,AA", "debug": {}}

    monkeypatch.setattr(service, "_prepare_qwen_image_data_url", fake_prepare)
    monkeypatch.setattr(generation_module, "QWEN_RESPONSES_MAX_IMAGE_COUNT", 30)
    monkeypatch.setattr(generation_module, "QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES", 5_500_000)

    service._build_qwen_input_with_debug(
        query="q", context="c", image_inputs=[{"source_id": "only", "image_path": "one.png"}]
    )
    assert targets == [QWEN_RESPONSES_IMAGE_TARGET_BYTES]

    targets.clear()
    service._build_qwen_input_with_debug(
        query="q",
        context="c",
        image_inputs=[{"source_id": f"figure-{index}", "image_path": f"{index}.png"} for index in range(20)],
    )
    # 第一张图按 20 张平分约 5.5MB 请求体，base64 膨胀后单图原始字节约 200KB，远小于单图上限；
    # 前面的图实际用得少，省下的预算顺延给后面的图，单图目标只增不减、且不超过单图上限。
    assert len(targets) == 20
    assert 150_000 < targets[0] < 250_000
    assert targets == sorted(targets)
    assert targets[-1] <= QWEN_RESPONSES_IMAGE_TARGET_BYTES


def test_complete_with_qwen_attaches_images_after_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return type("Response", (), {"output_text": "{}", "usage": None})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.responses = FakeResponses()

    service = GenerationService()
    monkeypatch.setattr(generation_module, "OpenAI", FakeClient)
    monkeypatch.setattr(
        service,
        "_prepare_qwen_image_data_url",
        lambda image_path, *, target_bytes: {"data_url": "data:image/jpeg;base64,AA", "debug": {}},
    )
    debug: dict[str, object] = {}

    service.complete_with_qwen(
        "DECISION PROMPT",
        api_key="test-key",
        task_type="research_decision",
        image_inputs=[{"source_id": "54", "image_path": "figure.png"}],
        request_debug=debug,
    )

    content = captured["input"][0]["content"]
    # prompt 原样放在最前面，不套问答模板；图片带 source_id 标签附在后面。
    assert content[0] == {"type": "input_text", "text": "DECISION PROMPT"}
    assert "source_id=54" in content[1]["text"]
    assert content[2]["type"] == "input_image"
    assert debug["sent_image_count"] == 1


def test_complete_with_qwen_without_images_keeps_plain_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return type("Response", (), {"output_text": "ok", "usage": None})()

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(generation_module, "OpenAI", FakeClient)
    GenerationService().complete_with_qwen("PLAIN", api_key="test-key", task_type="research_decision")

    assert captured["input"] == "PLAIN"
