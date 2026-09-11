from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import logging
from types import SimpleNamespace

import requests
import pytest

import services.embedding.embedding_service as embedding_service_module
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService


_RETRY_TEST_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _DisabledBuildCache:
    def get_embedding(self, **_kwargs):
        return None

    def set_embedding(self, *_args, **_kwargs) -> None:
        return None


class _SuccessfulResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "output": {
                "embeddings": [
                    {
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3],
                    }
                ]
            },
            "usage": {"input_tokens": 3},
        }


class _ErrorResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})

    def raise_for_status(self) -> None:
        raise requests.exceptions.HTTPError(f"status={self.status_code}", response=self)


def _build_public_embedding_request() -> tuple[dict, EmbeddingConfig]:
    chunk = {
        "content": "Evidence text",
        "metadata": {
            "chunk_id": 1,
            "chunk_type": "text",
            "page_number": 1,
            "source": "paper.pdf",
        },
    }
    retrieval_index = {
        "index_id": "1:body:1",
        "chunk_id": "1",
        "index_type": "body",
        "index_text": "Text sent to DashScope",
        "index_weight": 1.0,
        "enabled_routes": ["vector_original"],
    }
    config = EmbeddingConfig(
        provider="dashscope",
        model_name="qwen3-vl-embedding",
        api_key="test-key",
        base_url="https://example.test/embeddings",
        dimension=3,
        batch_size=8,
        request_max_retries=3,
        backoff_base_seconds=1.0,
        backoff_max_seconds=30.0,
        jitter_ratio=0.0,
    )
    return {
        "chunks": [chunk],
        "retrieval_indexes": [retrieval_index],
        "metadata": {"filename": "paper.pdf"},
    }, config


@pytest.mark.parametrize(
    "network_error",
    [
        requests.exceptions.SSLError("unexpected eof"),
        requests.exceptions.ConnectionError("connection reset"),
        requests.exceptions.ConnectTimeout("connect timeout"),
        requests.exceptions.ReadTimeout("read timeout"),
    ],
)
def test_create_embeddings_retries_network_error_then_returns_success(monkeypatch, network_error) -> None:
    input_data, config = _build_public_embedding_request()
    post_results = [
        network_error,
        _SuccessfulResponse(),
    ]
    post_calls: list[dict] = []
    sleep_calls: list[float] = []

    def fake_post(*_args, **kwargs):
        post_calls.append(kwargs)
        result = post_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
        raising=False,
    )

    embeddings, usage = EmbeddingService().create_embeddings(input_data, config)

    assert len(post_calls) == 2
    assert sleep_calls == [1.0]
    assert embeddings[0]["embedding"] == [0.1, 0.2, 0.3]
    assert usage == {}


def test_create_embeddings_honors_retry_after_for_retryable_http_status(monkeypatch) -> None:
    input_data, config = _build_public_embedding_request()
    post_results = [
        _ErrorResponse(503, {"Retry-After": "5"}),
        _SuccessfulResponse(),
    ]
    post_calls: list[dict] = []
    sleep_calls: list[float] = []

    def fake_post(*_args, **kwargs):
        post_calls.append(kwargs)
        return post_results.pop(0)

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
    )

    embeddings, _usage = EmbeddingService().create_embeddings(input_data, config)

    assert len(post_calls) == 2
    assert sleep_calls == [5.0]
    assert embeddings[0]["embedding"] == [0.1, 0.2, 0.3]


def test_create_embeddings_caps_retry_after_at_thirty_seconds(monkeypatch) -> None:
    input_data, config = _build_public_embedding_request()
    config.backoff_max_seconds = 60.0
    post_results = [
        _ErrorResponse(503, {"Retry-After": "120"}),
        _SuccessfulResponse(),
    ]
    sleep_calls: list[float] = []

    def fake_post(*_args, **_kwargs):
        return post_results.pop(0)

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
    )

    EmbeddingService().create_embeddings(input_data, config)

    assert sleep_calls == [30.0]


@pytest.mark.parametrize(
    ("retry_after", "expected_delay"),
    [
        (format_datetime(_RETRY_TEST_NOW + timedelta(minutes=5), usegmt=True), 30.0),
        ("invalid-retry-after", 1.0),
    ],
)
def test_create_embeddings_handles_retry_after_variants(monkeypatch, retry_after: str, expected_delay: float) -> None:
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return _RETRY_TEST_NOW.astimezone(tz) if tz is not None else _RETRY_TEST_NOW.replace(tzinfo=None)

    # 参数在收集阶段生成；固定服务时钟，避免全量测试耗时缩短 Retry-After 的剩余等待时间。
    monkeypatch.setattr(embedding_service_module, "datetime", FixedDatetime)
    input_data, config = _build_public_embedding_request()
    post_results = [
        _ErrorResponse(503, {"Retry-After": retry_after}),
        _SuccessfulResponse(),
    ]
    sleep_calls: list[float] = []

    def fake_post(*_args, **_kwargs):
        return post_results.pop(0)

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
    )

    EmbeddingService().create_embeddings(input_data, config)

    assert sleep_calls == [expected_delay]


def test_create_embeddings_applies_configured_jitter(monkeypatch) -> None:
    input_data, config = _build_public_embedding_request()
    config.jitter_ratio = 0.2
    post_results = [requests.exceptions.SSLError("unexpected eof"), _SuccessfulResponse()]
    sleep_calls: list[float] = []

    def fake_post(*_args, **_kwargs):
        result = post_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(embedding_service_module.random, "uniform", lambda _lower, _upper: 0.2)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
    )

    EmbeddingService().create_embeddings(input_data, config)

    assert sleep_calls == [1.2]


def test_default_dashscope_config_enables_bounded_request_retries() -> None:
    config = EmbeddingConfig.from_env(provider="dashscope", model_name="qwen3-vl-embedding")

    assert config.request_max_retries == 3
    assert config.backoff_base_seconds == 1.0
    assert config.backoff_max_seconds == 30.0
    assert config.jitter_ratio == 0.2


def test_embedding_config_clamps_retry_policy_to_safety_limits() -> None:
    config = EmbeddingConfig(
        provider="dashscope",
        model_name="qwen3-vl-embedding",
        request_max_retries=99,
        backoff_base_seconds=60.0,
        backoff_max_seconds=120.0,
        jitter_ratio=2.0,
    )

    assert config.request_max_retries == 3
    assert config.backoff_base_seconds == 30.0
    assert config.backoff_max_seconds == 30.0
    assert config.jitter_ratio == 1.0


@pytest.mark.parametrize("status_code", [408, 425, 429, 500, 502, 503, 504])
def test_create_embeddings_retries_only_temporary_http_statuses(monkeypatch, status_code: int) -> None:
    input_data, config = _build_public_embedding_request()
    config.request_max_retries = 1
    post_results = [_ErrorResponse(status_code), _SuccessfulResponse()]
    sleep_calls: list[float] = []

    def fake_post(*_args, **_kwargs):
        return post_results.pop(0)

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
    )

    embeddings, _usage = EmbeddingService().create_embeddings(input_data, config)

    assert post_results == []
    assert sleep_calls == [1.0]
    assert embeddings[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 413, 422])
def test_create_embeddings_does_not_retry_permanent_http_errors(monkeypatch, status_code: int) -> None:
    input_data, config = _build_public_embedding_request()
    post_calls = 0
    sleep_calls: list[float] = []

    def fake_post(*_args, **_kwargs):
        nonlocal post_calls
        post_calls += 1
        return _ErrorResponse(status_code)

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
    )

    with pytest.raises(requests.exceptions.HTTPError):
        EmbeddingService().create_embeddings(input_data, config)

    assert post_calls == 1
    assert sleep_calls == []


def test_create_embeddings_raises_last_network_error_after_retry_exhaustion(monkeypatch, caplog) -> None:
    input_data, config = _build_public_embedding_request()
    errors = [requests.exceptions.SSLError(f"failure-{index}") for index in range(1, 5)]
    expected_last_error = errors[-1]
    sleep_calls: list[float] = []

    def fake_post(*_args, **_kwargs):
        raise errors.pop(0)

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(
        embedding_service_module,
        "time",
        SimpleNamespace(sleep=lambda seconds: sleep_calls.append(seconds)),
    )

    with caplog.at_level(logging.WARNING, logger="services.embedding.embedding_service"):
        with pytest.raises(requests.exceptions.SSLError) as exc_info:
            EmbeddingService().create_embeddings(input_data, config)

    assert exc_info.value is expected_last_error
    assert errors == []
    assert sleep_calls == [1.0, 2.0, 4.0]
    assert len(caplog.records) == 3
    assert all(record.exc_info is None for record in caplog.records)


def test_retry_warning_contains_only_sanitized_request_metadata(monkeypatch, caplog) -> None:
    input_data, config = _build_public_embedding_request()
    post_results = [requests.exceptions.SSLError("unexpected eof"), _SuccessfulResponse()]

    def fake_post(*_args, **_kwargs):
        result = post_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(embedding_service_module, "get_paper_qa_build_cache", lambda: _DisabledBuildCache())
    monkeypatch.setattr(embedding_service_module.requests, "post", fake_post)
    monkeypatch.setattr(embedding_service_module, "time", SimpleNamespace(sleep=lambda _seconds: None))

    with caplog.at_level(logging.WARNING, logger="services.embedding.embedding_service"):
        EmbeddingService().create_embeddings(input_data, config)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "model=qwen3-vl-embedding" in message
    assert "attempt=2/4" in message
    assert "reason=SSLError" in message
    assert "input_mode=text" in message
    assert "test-key" not in message
    assert "Authorization" not in message
    assert "Bearer" not in message
    assert "Text sent to DashScope" not in message
    assert "[0.1, 0.2, 0.3]" not in message
    assert caplog.records[0].exc_info is None
