"""在真实 GenerationService 的 SDK 边界验证成本统计，测试不发网络请求。"""

from importlib import import_module
from types import SimpleNamespace

import pytest

from services.llm.call_metrics import LLMCallStats, use_call_stats
from services.llm.generation_service import GenerationService
from services.retrieval.execution import RouteExecutionSupport


@pytest.fixture
def provider(monkeypatch):
    response = SimpleNamespace(
        output_text="离线答案", choices=[SimpleNamespace(message=SimpleNamespace(content="离线答案"))],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3),
    )

    def create(**kwargs):
        if kwargs.get("stream"):
            return iter([
                SimpleNamespace(type="response.output_text.delta", delta="离线答案"),
                SimpleNamespace(type="response.completed", response=response),
            ])
        return response

    sdk = SimpleNamespace(responses=SimpleNamespace(create=create),
                          chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(import_module("services.llm.generation_service"), "OpenAI", lambda **kwargs: sdk)
    # 该用例只访问 SDK 方法，不加载模型或创建生成产物目录。
    service = GenerationService.__new__(GenerationService)
    service.models = {"openai": {"test": "test-openai"}, "deepseek": {"test": "test-deepseek"}}
    return service, sdk


@pytest.mark.parametrize("method", ["complete", "qwen", "stream", "openai", "deepseek"])
def test_each_sdk_call_is_counted_once_with_usage(provider, method):
    service, _ = provider
    stats = LLMCallStats()
    with use_call_stats(stats):
        if method == "complete":
            service.complete_with_qwen("问题", api_key="offline", task_type="verification")
        elif method == "qwen":
            service._generate_with_qwen_responses("问题", "证据", api_key="offline")
        elif method == "stream":
            assert list(service.stream_qwen_responses("问题", "证据", api_key="offline"))[-1]["type"] == "completed"
        elif method == "openai":
            service._generate_with_openai("test", "问题", "证据", api_key="offline")
        else:
            service._generate_with_deepseek("test", "问题", "证据", api_key="offline")
    assert stats.to_dict()["llm_calls"] == 1
    assert stats.to_dict()["total_tokens"] == 15


def test_failed_call_keeps_cost_but_does_not_invent_usage(provider):
    service, sdk = provider

    def fail(**kwargs):
        raise TimeoutError("offline timeout")

    sdk.responses.create = fail
    stats = LLMCallStats()
    with use_call_stats(stats), pytest.raises(TimeoutError):
        service.complete_with_qwen("问题", api_key="offline")
    assert stats.to_dict()["llm_calls"] == 1
    assert stats.to_dict()["total_tokens"] is None
    assert stats.to_dict()["usage_reported_calls"] == 0


def test_retrieval_worker_inherits_request_stats_without_leaking(provider):
    service, _ = provider
    stats = LLMCallStats()
    support = RouteExecutionSupport(timeouts={"default": 2})
    try:
        with use_call_stats(stats):
            result = support.run("rerank", lambda: [{"answer": service.complete_with_qwen("问题", api_key="offline")}])
        assert result.status == "ok"
        assert stats.to_dict()["llm_calls"] == 1
        # 同一个线程池下一次无统计上下文的请求不能继续写入上一轮的计数器。
        support.run("rerank", lambda: [{"answer": service.complete_with_qwen("问题", api_key="offline")}])
        assert stats.to_dict()["llm_calls"] == 1
    finally:
        support.shutdown(wait=True)


@pytest.mark.parametrize("failure", [False, True])
def test_remote_rerank_provider_is_included_in_cost(monkeypatch, failure):
    from tests.helpers import build_retrieval_service

    owner, _, *_ = build_retrieval_service()
    owner.llm_rerank_api_key = "offline"
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        if failure:
            raise TimeoutError("offline rerank timeout")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "output": {"results": [{"index": 0, "relevance_score": 1.0}]},
            "usage": {"total_tokens": 17},
        })

    monkeypatch.setattr(import_module("services.retrieval.rerank_service").requests, "post", post)
    stats = LLMCallStats()
    chunk = {"chunk_id": "c1", "content": "论文方法"}
    with use_call_stats(stats):
        owner.rerank_service.rerank_with_dashscope("方法", [chunk], [chunk], ["论文方法"], 1, 1)
    assert len(calls) == 1
    assert stats.to_dict()["llm_calls"] == 1
    assert stats.to_dict()["task_calls"] == {"rerank": 1}
    assert stats.to_dict()["total_tokens"] == (None if failure else 17)
