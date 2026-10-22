from __future__ import annotations

import pytest

from services.llm.generation_service import QWEN_TASK_MODEL_ROLES, GenerationService


def test_translate_query_uses_small_model_task_and_strips_output(monkeypatch: pytest.MonkeyPatch) -> None:
    service = GenerationService()
    calls: list[dict] = []

    def fake_complete(prompt: str, **kwargs) -> str:
        calls.append({"prompt": prompt, **kwargs})
        return "  What distribution do the entries of R follow?\n"

    monkeypatch.setattr(service, "complete_with_qwen", fake_complete)

    translated = service.translate_query_for_retrieval("随机投影矩阵 R 的元素服从什么分布？")

    assert translated == "What distribution do the entries of R follow?"
    assert calls[0]["task_type"] == "query_translation"
    assert "Question: 随机投影矩阵 R 的元素服从什么分布？" in calls[0]["prompt"]
    assert QWEN_TASK_MODEL_ROLES["query_translation"] == "small"
