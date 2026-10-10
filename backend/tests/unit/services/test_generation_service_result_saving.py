from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.llm import generation_service as generation_module
from services.llm.generation_service import GenerationService


def _service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, enabled: bool) -> GenerationService:
    # 开关在构造时读取配置；结果目录指向临时目录，避免测试写入真实的 05-generation-results。
    monkeypatch.setitem(generation_module.GENERATION_CONFIG, "save_generation_results", enabled)
    service = GenerationService()
    service.generation_results_dir = tmp_path / "05-generation-results"
    monkeypatch.setattr(service, "_generate_with_qwen_responses", lambda *args, **kwargs: "生成的回答")
    return service


def test_generate_skips_result_file_when_saving_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = _service(monkeypatch, tmp_path, enabled=False)

    result = service.generate("qwen", "问题", [{"source_id": "s1", "text": "证据"}])

    assert result["response"] == "生成的回答"
    assert result["saved_filepath"] is None
    assert not service.generation_results_dir.exists()


def test_generate_writes_result_file_when_saving_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service = _service(monkeypatch, tmp_path, enabled=True)

    result = service.generate("qwen", "问题", [{"source_id": "s1", "text": "证据"}])

    saved = Path(result["saved_filepath"])
    assert saved.parent == service.generation_results_dir
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["query"] == "问题"
    assert payload["response"] == "生成的回答"
    assert payload["context"] == [{"source_id": "s1", "text": "证据"}]
