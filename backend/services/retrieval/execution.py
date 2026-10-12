"""检索 route 的执行支撑：统一超时/异常降级与指标记录，以及查询 embedding 的批量生成。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _elapsed_ms(started: float) -> float:
    """从 perf_counter 起点到现在经过的毫秒数。"""
    return (perf_counter() - started) * 1000


@dataclass
class RouteExecutionResult:
    """统一的 route 执行结果；增强 route 失败时用 metrics 解释降级原因。"""

    route_name: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    latency_ms: float = 0.0
    error: str = ""
    fallback_reason: str = ""
    timeout: bool = False
    enabled: bool = True
    applied: bool = False
    cache_hit: Optional[bool] = None

    def to_metric(self) -> Dict[str, Any]:
        """导出到 route_metrics 的统一结构（不含 route_name，调用方以它为键）。"""
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "candidate_count": len(self.results),
            "error": self.error,
            "fallback_reason": self.fallback_reason,
            "cache_hit": self.cache_hit,
            "timeout": self.timeout,
        }


class RouteExecutionSupport:
    """为 route 提供统一 timeout、异常捕获和指标记录，避免增强 route 拖垮主链路。"""

    def __init__(self, *, timeouts: Dict[str, float], max_workers: int = 4) -> None:
        self.timeouts = dict(timeouts or {})
        self.max_workers = max(1, int(max_workers or 1))
        # 使用共享、有上限的 executor 承载 route 任务；超时后当前检索会立即降级，
        # 底层慢 IO 最多占用受控 worker，避免每个 route 临时创建线程导致并发失控。
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="retrieval-route")

    def run(
        self,
        route_name: str,
        callback: Callable[[], List[Dict[str, Any]]],
        *,
        required: bool = False,
        enabled: bool = True,
        cache_hit: Optional[bool] = None,
    ) -> RouteExecutionResult:
        """在 worker 线程中执行 callback 并包装结果。

        超时时间按 route_name 查 timeouts，没有则用 "default"，再没有用 8 秒。
        required=True 的 route 超时或出错时抛 RuntimeError；其余 route 只返回 timeout/error 状态，由调用方降级。
        """
        if not enabled:
            return RouteExecutionResult(
                route_name=route_name,
                status="disabled",
                enabled=False,
                cache_hit=cache_hit,
                fallback_reason="disabled",
            )

        started = perf_counter()
        timeout_seconds = float(self.timeouts.get(route_name, self.timeouts.get("default", 8.0)) or 8.0)
        try:
            # future.result(timeout=...) 只限制当前 route 等待时间；超时后不进入 executor
            # shutdown 等待路径，确保增强 route 的慢查询不会反向阻塞主检索链路。
            # 检索增强阶段可能调用模型；把请求上下文复制到 worker，避免漏计 rerank/HyDE 成本。
            future = self._executor.submit(copy_context().run, callback)
            results = future.result(timeout=timeout_seconds)
            return RouteExecutionResult(
                route_name=route_name,
                results=list(results or []),
                status="ok",
                latency_ms=_elapsed_ms(started),
                applied=bool(results),
                cache_hit=cache_hit,
            )
        except TimeoutError as exc:
            latency_ms = _elapsed_ms(started)
            if required:
                raise RuntimeError(f"Required route {route_name} timed out after {timeout_seconds:.2f}s") from exc
            # 已在运行的任务无法被取消，cancel 只对仍在排队的任务生效。
            future.cancel()
            return RouteExecutionResult(
                route_name=route_name,
                status="timeout",
                latency_ms=latency_ms,
                error=str(exc),
                fallback_reason=f"{route_name} timed out after {timeout_seconds:.2f}s",
                timeout=True,
                cache_hit=cache_hit,
            )
        except Exception as exc:
            latency_ms = _elapsed_ms(started)
            if required:
                raise RuntimeError(f"Required route {route_name} failed: {exc}") from exc
            logger.warning("Route failed and will be skipped: route=%s error=%s", route_name, exc)
            return RouteExecutionResult(
                route_name=route_name,
                status="error",
                latency_ms=latency_ms,
                error=str(exc),
                fallback_reason=f"{route_name} failed: {exc}",
                cache_hit=cache_hit,
            )

    def shutdown(self, *, wait: bool = False) -> None:
        """测试或进程退出时显式释放 route worker；主流程不依赖它做降级控制。"""
        self._executor.shutdown(wait=wait, cancel_futures=True)


class QueryEmbeddingBatcher:
    """收集本轮向量 route 查询并批量生成 embedding，避免 rewritten query 重复单条请求。"""

    def __init__(self, *, embedding_service: Any) -> None:
        self.embedding_service = embedding_service

    def build(
        self,
        queries: List[str],
        *,
        provider: str,
        model: str,
        dimension: Optional[int],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """对去重后的查询一次性生成 embedding。

        返回 {"embeddings": {归一化查询: 向量}, "debug": {...}}；embedding 服务出错或返回数量不符时
        embeddings 为空、debug.status 为 error，此时向量 route 查不到向量会各自报错并按 route 失败降级。
        """
        started = perf_counter()
        # 按空白归一化后保序去重，同一句话的不同空白写法只请求一次。
        normalized_queries = list(dict.fromkeys(filter(None, map(self.normalize_query, queries))))
        requested_count = len([query for query in queries if str(query or "").strip()])

        debug = {
            "requested_count": requested_count,
            "batch_size": len(normalized_queries),
            "deduped_count": len(normalized_queries),
            "duplicate_removed_count": max(0, requested_count - len(normalized_queries)),
            "latency_ms": 0.0,
            "status": "ok",
            "error": "",
        }
        if not normalized_queries:
            debug["status"] = "empty"
            return {"embeddings": {}, "debug": debug}

        try:
            embeddings, usage = self.embedding_service.create_text_embeddings_with_usage(
                normalized_queries,
                provider=provider,
                model=model,
                dimension=dimension,
                batch_size=batch_size,
            )
            if len(embeddings) != len(normalized_queries):
                raise ValueError(f"Embedding batch returned {len(embeddings)} vectors for {len(normalized_queries)} queries")
            debug["usage"] = usage
            return {
                "embeddings": {
                    query: [float(value) for value in embedding]
                    for query, embedding in zip(normalized_queries, embeddings)
                },
                "debug": {**debug, "latency_ms": _elapsed_ms(started)},
            }
        except Exception as exc:
            debug.update({"status": "error", "error": str(exc), "latency_ms": _elapsed_ms(started)})
            return {"embeddings": {}, "debug": debug}

    @staticmethod
    def normalize_query(query: Any) -> str:
        """把连续空白压成单个空格并去掉首尾空白；向量 route 用同一规则查 embeddings 字典。"""
        return " ".join(str(query or "").split())
