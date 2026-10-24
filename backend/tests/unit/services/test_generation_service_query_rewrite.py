from __future__ import annotations

import json

import pytest

from services.llm.generation_service import QWEN_TASK_MODEL_ROLES, GenerationService


def _service_returning(monkeypatch: pytest.MonkeyPatch, response: str) -> tuple[GenerationService, list[dict]]:
    service = GenerationService()
    calls: list[dict] = []

    def fake_complete(prompt: str, **kwargs) -> str:
        calls.append({"prompt": prompt, **kwargs})
        return response

    monkeypatch.setattr(service, "complete_with_qwen", fake_complete)
    return service, calls


def test_rewrite_query_returns_cleaned_unique_queries_within_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"queries": ["  Which datasets  are used? ", "Which datasets are used?", "How large is the gain?", "Extra?"]}
    service, calls = _service_returning(monkeypatch, f"```json\n{json.dumps(payload)}\n```")

    queries = service.rewrite_query_for_retrieval(
        "用了什么数据集、提升多少？",
        max_queries=2,
        paper_context={"title": "Sparse Projection", "abstract": "We study ...", "section_titles": ["Method"]},
    )

    assert queries == ["Which datasets are used?", "How large is the gain?"]
    assert calls[0]["task_type"] == "query_rewrite"
    assert "Question: 用了什么数据集、提升多少？" in calls[0]["prompt"]
    assert "Paper title: Sparse Projection" in calls[0]["prompt"]
    assert QWEN_TASK_MODEL_ROLES["query_rewrite"] == "large"
    # 低温但不为 0：改写贴着原问，同时保留一点随机性。
    assert 0 < calls[0]["temperature"] < 1


def test_rewrite_query_accepts_bare_list(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _ = _service_returning(monkeypatch, '["What distribution do the entries of R follow?"]')

    assert service.rewrite_query_for_retrieval("R 的元素服从什么分布？") == ["What distribution do the entries of R follow?"]


@pytest.mark.parametrize("response", ['{"queries": []}', "[]", '{"rewrite_queries": ["x"]}'])
def test_rewrite_query_rejects_response_without_queries(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    service, _ = _service_returning(monkeypatch, response)

    with pytest.raises(ValueError):
        service.rewrite_query_for_retrieval("这个方法怎么做的？")
