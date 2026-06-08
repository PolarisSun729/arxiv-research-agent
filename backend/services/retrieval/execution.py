from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


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
            future = self._executor.submit(callback)
            results = future.result(timeout=timeout_seconds)
            latency_ms = (perf_counter() - started) * 1000
            return RouteExecutionResult(
                route_name=route_name,
                results=list(results or []),
                status="ok",
                latency_ms=latency_ms,
                enabled=True,
                applied=bool(results),
                cache_hit=cache_hit,
            )
        except TimeoutError as exc:
            latency_ms = (perf_counter() - started) * 1000
            if required:
                raise RuntimeError(f"Required route {route_name} timed out after {timeout_seconds:.2f}s") from exc
            future.cancel()
            return RouteExecutionResult(
                route_name=route_name,
                status="timeout",
                latency_ms=latency_ms,
                error=str(exc),
                fallback_reason=f"{route_name} timed out after {timeout_seconds:.2f}s",
                timeout=True,
                enabled=True,
                applied=False,
                cache_hit=cache_hit,
            )
        except Exception as exc:
            latency_ms = (perf_counter() - started) * 1000
            if required:
                raise RuntimeError(f"Required route {route_name} failed: {exc}") from exc
            logger.warning("Route failed and will be skipped: route=%s error=%s", route_name, exc)
            return RouteExecutionResult(
                route_name=route_name,
                status="error",
                latency_ms=latency_ms,
                error=str(exc),
                fallback_reason=f"{route_name} failed: {exc}",
                enabled=True,
                applied=False,
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
        started = perf_counter()
        normalized_queries: List[str] = []
        for query in queries:
            normalized = self.normalize_query(query)
            if normalized and normalized not in normalized_queries:
                normalized_queries.append(normalized)

        debug = {
            "requested_count": len([query for query in queries if str(query or "").strip()]),
            "batch_size": len(normalized_queries),
            "deduped_count": len(normalized_queries),
            "duplicate_removed_count": max(0, len([query for query in queries if str(query or "").strip()]) - len(normalized_queries)),
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
                "debug": {**debug, "latency_ms": (perf_counter() - started) * 1000},
            }
        except Exception as exc:
            debug.update({"status": "error", "error": str(exc), "latency_ms": (perf_counter() - started) * 1000})
            return {"embeddings": {}, "debug": debug}

    @staticmethod
    def normalize_query(query: Any) -> str:
        return " ".join(str(query or "").split()).strip()
