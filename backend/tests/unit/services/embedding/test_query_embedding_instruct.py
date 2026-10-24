from __future__ import annotations

from types import SimpleNamespace

import services.embedding.embedding_service as embedding_service_module
from services.embedding.embedding_service import EmbeddingService


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"output": {"embeddings": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}, "usage": {}}


def _capture_posts(monkeypatch) -> list[dict]:
    payloads: list[dict] = []

    def fake_post(*_args, **kwargs):
        payloads.append(kwargs["json"])
        return _Response()

    monkeypatch.setattr(EmbeddingService, "_dashscope_http_session", lambda _self: SimpleNamespace(post=fake_post))
    return payloads


def _embed(service: EmbeddingService, provider: str = "dashscope", instruct: str | None = None):
    return service.create_text_embeddings_with_usage(
        ["what is the method"],
        provider=provider,
        model="qwen3-vl-embedding",
        api_key="test-key",
        base_url="https://example.test/embeddings",
        dimension=3,
        instruct=instruct,
    )


def test_dashscope_payload_carries_instruct_only_when_given(monkeypatch) -> None:
    payloads = _capture_posts(monkeypatch)
    service = EmbeddingService()

    _embed(service, instruct="Retrieve passages that answer the question.")
    _embed(service, instruct=None)

    assert payloads[0]["parameters"]["instruct"] == "Retrieve passages that answer the question."
    assert "instruct" not in payloads[1]["parameters"]


def test_same_text_with_different_instruct_is_cached_separately(monkeypatch) -> None:
    payloads = _capture_posts(monkeypatch)
    service = EmbeddingService()

    _embed(service, instruct="A")
    _embed(service, instruct="A")
    _embed(service, instruct="B")

    # 第二次命中缓存，第三次说明不同必须重新请求。
    assert [payload["parameters"].get("instruct") for payload in payloads] == ["A", "B"]


def test_instruct_ignored_for_non_dashscope_provider(monkeypatch) -> None:
    captured: list = []

    def fake_openai(_self, texts, config):
        captured.append(config)
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(EmbeddingService, "_create_openai_embeddings", fake_openai)

    _embed(EmbeddingService(), provider="openai", instruct="A")

    assert captured[0].instruct is None
