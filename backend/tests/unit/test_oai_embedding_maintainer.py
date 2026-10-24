from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, Dict, List

from services.arxiv.arxiv_embedding_handler import ArxivEmbeddingHandler
from services.arxiv.oai_embedding_maintainer import OaiEmbeddingMaintainer


def _paper(arxiv_id: str) -> Dict[str, Any]:
    return {"arxiv_id": arxiv_id, "title": f"Title {arxiv_id}", "abstract": f"Abstract {arxiv_id}", "created": "2026-10-20"}


class FakeEmbeddingService:
    """标题里带 BAD 的论文永远生成失败（批量和单篇都失败），模拟一篇坏数据；down=True 模拟接口整体不可用。"""

    def __init__(self) -> None:
        self.batch_calls: List[List[str]] = []
        self.down = False

    def get_default_embedding_config(self) -> Any:
        return SimpleNamespace(provider="dashscope", model_name="test-model", api_key="k", base_url=None, dimension=2)

    @staticmethod
    def build_paper_embedding_text(title: str, abstract: str) -> str:
        return f"{title}\n{abstract}"

    def create_text_embeddings_with_usage(self, texts: List[str], **kwargs: Any):
        self.batch_calls.append(list(texts))
        if self.down or any("BAD" in text for text in texts):
            raise RuntimeError("embedding rejected")
        return [[0.1, 0.2] for _ in texts], {"input_tokens": 10 * len(texts)}

    def create_single_embedding_with_usage(self, text: str, **kwargs: Any):
        if self.down or "BAD" in text:
            raise RuntimeError("embedding rejected")
        return [0.1, 0.2], {"input_tokens": 10}


class FakeVectorStore:
    def __init__(self, existing_ids: set | None = None) -> None:
        self.existing_ids = set(existing_ids or ())
        self.probed_ids: List[str] = []
        self.list_calls = 0

    def get_existing_paper_arxiv_ids(self, collection_name: str, arxiv_ids: List[str]) -> set:
        self.probed_ids.extend(arxiv_ids)
        return {arxiv_id for arxiv_id in arxiv_ids if arxiv_id in self.existing_ids}

    def list_paper_arxiv_ids(self, collection_name: str) -> set:
        self.list_calls += 1
        return set(self.existing_ids)

    def insert_embeddings(self, collection_name: str, items: List[Dict[str, Any]]) -> int:
        self.existing_ids.update(item["metadata"]["arxiv_id"] for item in items)
        return len(items)


class FakeDatabase:
    """按入库顺序从新到旧分页返回论文。"""

    def __init__(self, arxiv_ids: List[str], page_size: int = 3) -> None:
        self.papers = [_paper(arxiv_id) for arxiv_id in arxiv_ids]
        self.page_size = page_size
        self.since_calls: List[Any] = []

    def iter_papers_newest_first(self, *, since=None, page_size: int = 500):
        self.since_calls.append(since)
        for start in range(0, len(self.papers), self.page_size):
            yield self.papers[start : start + self.page_size]


class Clock:
    def __init__(self) -> None:
        self.today = date(2026, 10, 24)

    def __call__(self) -> date:
        return self.today


def _maintainer(database: FakeDatabase, embedding_service: FakeEmbeddingService, vector_store: FakeVectorStore, **overrides: Any):
    prune_calls: List[str] = []

    def handler_factory() -> ArxivEmbeddingHandler:
        return ArxivEmbeddingHandler(
            embedding_service=embedding_service,
            vector_store_service=vector_store,
            embedding_collection_name="arxiv_paper_embeddings",
            embedding_batch_size=2,
            min_paper_date="2026-04-27",
        )

    def prune(handler: ArxivEmbeddingHandler) -> int:
        prune_calls.append(handler.min_paper_date)
        return 5

    clock = overrides.pop("today", Clock())
    maintainer = OaiEmbeddingMaintainer(
        handler_factory=handler_factory,
        database_service_factory=lambda: database,
        prune_out_of_window=prune,
        startup_delay_seconds=3600,
        today=clock,
        **overrides,
    )
    return maintainer, prune_calls, clock


def test_round_fills_missing_papers_batch_by_batch_then_prunes_once_a_day() -> None:
    database = FakeDatabase(["5", "4", "3", "2", "1"])
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(existing_ids={"4"})
    maintainer, prune_calls, clock = _maintainer(database, embedding_service, vector_store)

    result = maintainer.run_round()

    assert result["status"] == "covered"
    assert database.since_calls == ["2026-04-27"]
    # 最新的先补；每次请求不超过 embedding 批大小，已有向量的不重复生成。
    assert [len(call) for call in embedding_service.batch_calls] == [2, 2]
    assert embedding_service.batch_calls[0][0].startswith("Title 5")
    assert vector_store.existing_ids == {"1", "2", "3", "4", "5"}
    assert (result["scanned"], result["missing_found"], result["embeddings_written"]) == (5, 4, 4)
    # 已有向量一轮只全量取一次；逐个按 id 探测的只有真正要写的论文（写入前防并发的复查），已有的 4 不会被探测。
    assert vector_store.list_calls == 1 and "4" not in vector_store.probed_ids
    assert result["pruned_out_of_window"] == 5 and prune_calls == ["2026-04-27"]
    assert maintainer.get_status()["last_full_scan"]["missing_after_round"] == 0

    # 同一天再跑只查不写、不再清理；换天后才清理下一次。
    again = maintainer.run_round()
    assert again["embeddings_written"] == 0 and len(embedding_service.batch_calls) == 2
    assert prune_calls == ["2026-04-27"]
    clock.today = date(2026, 10, 25)
    maintainer.run_round()
    assert len(prune_calls) == 2


def test_round_stops_at_max_per_round_and_next_round_continues() -> None:
    database = FakeDatabase(["5", "4", "3", "2", "1"])
    vector_store = FakeVectorStore()
    maintainer, prune_calls, _ = _maintainer(database, FakeEmbeddingService(), vector_store, max_per_round=3)

    first = maintainer.run_round()

    assert first["status"] == "limit_reached"
    assert first["embeddings_attempted"] == 3 and vector_store.existing_ids == {"5", "4", "3"}
    # 没扫完整个窗口不清理，也不更新全窗口口径的缺失数。
    assert prune_calls == [] and maintainer.get_status()["last_full_scan"] is None
    assert maintainer._next_delay(first["status"]) < maintainer.interval_seconds

    second = maintainer.run_round()
    assert second["status"] == "covered"
    assert vector_store.existing_ids == {"1", "2", "3", "4", "5"}


def test_whole_batch_failure_backs_off_and_bad_paper_is_skipped_for_the_day() -> None:
    database = FakeDatabase(["BAD", "2", "1"])
    database.papers[0]["title"] = "BAD paper"
    vector_store = FakeVectorStore(existing_ids={"2", "1"})
    maintainer, _, clock = _maintainer(database, FakeEmbeddingService(), vector_store)

    statuses = [maintainer.run_round()["status"] for _ in range(3)]

    assert statuses == ["backoff"] * 3
    assert maintainer._backoff_seconds() == 240
    # 连续 3 次失败后当天跳过，不再卡住后面的论文。
    fourth = maintainer.run_round()
    assert fourth["status"] == "covered" and fourth["missing_found"] == 0
    assert maintainer.get_status()["skipped_papers"] == 1
    assert maintainer.get_status()["last_full_scan"]["missing_after_round"] == 1

    clock.today = date(2026, 10, 25)
    assert maintainer.run_round()["status"] == "backoff"


def test_partial_batch_failure_keeps_going_and_counts_only_the_failed_paper() -> None:
    database = FakeDatabase(["BAD", "3", "2", "1"])
    database.papers[0]["title"] = "BAD paper"
    vector_store = FakeVectorStore()
    maintainer, _, _ = _maintainer(database, FakeEmbeddingService(), vector_store)

    result = maintainer.run_round()

    assert result["status"] == "covered"
    assert vector_store.existing_ids == {"1", "2", "3"}
    assert (result["embeddings_written"], result["embedding_errors"]) == (3, 1)
    assert maintainer._paper_failures == {"BAD": 1}
    assert maintainer._consecutive_failed_batches == 0


def test_round_stops_promptly_when_stop_is_requested() -> None:
    database = FakeDatabase(["2", "1"])
    embedding_service = FakeEmbeddingService()
    maintainer, _, _ = _maintainer(database, embedding_service, FakeVectorStore())
    maintainer._stop_event.set()

    result = maintainer.run_round()

    assert result["status"] == "stopped" and embedding_service.batch_calls == []


def test_worker_thread_starts_and_stops_during_startup_delay() -> None:
    database = FakeDatabase(["1"])
    embedding_service = FakeEmbeddingService()
    maintainer, _, _ = _maintainer(database, embedding_service, FakeVectorStore())

    maintainer.start()
    assert maintainer.request_run() is True
    assert maintainer.get_status()["running"] is True
    maintainer.stop(join_timeout_seconds=2)

    assert maintainer.get_status()["running"] is False
    # 启动延迟期间停止，不会跑任何一轮。
    assert database.since_calls == [] and embedding_service.batch_calls == []
