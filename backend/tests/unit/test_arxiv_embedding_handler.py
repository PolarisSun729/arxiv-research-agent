from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from services.arxiv.arxiv_embedding_handler import ArxivEmbeddingHandler


def _paper(arxiv_id: str, **overrides: Any) -> Dict[str, Any]:
    paper = {
        "arxiv_id": arxiv_id,
        "title": f"Title {arxiv_id}",
        "abstract": f"Abstract {arxiv_id}",
        # OAI 库里作者、分类存的是 JSON 数组字符串。
        "authors": '["Alice", "Bob"]',
        "categories": '["cs.CL", "cs.LG"]',
        "created": "2026-10-20",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }
    paper.update(overrides)
    return paper


class FakeEmbeddingService:
    def __init__(self, *, batch_error: Optional[Exception] = None) -> None:
        self.batch_error = batch_error
        self.batch_calls: List[List[str]] = []
        self.single_calls: List[str] = []

    def get_default_embedding_config(self) -> Any:
        return SimpleNamespace(provider="dashscope", model_name="test-model", api_key="k", base_url=None, dimension=4)

    @staticmethod
    def build_paper_embedding_text(title: str, abstract: str) -> str:
        return f"{title}\n{abstract}".strip()

    def create_text_embeddings_with_usage(self, texts: List[str], **kwargs: Any):
        self.batch_calls.append(list(texts))
        if self.batch_error is not None:
            raise self.batch_error
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts], {"input_tokens": 10 * len(texts)}

    def create_single_embedding_with_usage(self, text: str, **kwargs: Any):
        self.single_calls.append(text)
        return [0.1, 0.2, 0.3, 0.4], {"input_tokens": 10}


class FakeVectorStore:
    def __init__(self, *, existing_ids: Optional[set] = None, query_error: Optional[Exception] = None, insert_error: Optional[Exception] = None) -> None:
        self.existing_ids = set(existing_ids or ())
        self.query_error = query_error
        self.insert_error = insert_error
        self.inserted: List[Dict[str, Any]] = []
        self.prune_calls: List[Dict[str, Any]] = []

    def get_paper_embeddings_by_arxiv_ids(self, collection_name: str, arxiv_ids: List[str]) -> List[Dict[str, Any]]:
        if self.query_error is not None:
            raise self.query_error
        return [{"arxiv_id": arxiv_id} for arxiv_id in arxiv_ids if arxiv_id in self.existing_ids]

    def insert_embeddings(self, collection_name: str, items: List[Dict[str, Any]]) -> int:
        if self.insert_error is not None:
            raise self.insert_error
        self.inserted.extend(items)
        return len(items)

    def insert_single_embedding(self, collection_name: str, embedding: List[float], metadata: Dict[str, Any]) -> int:
        self.inserted.append({"embedding": embedding, "metadata": metadata})
        return len(self.inserted)

    def delete_paper_embeddings_published_before(self, **kwargs: Any) -> int:
        self.prune_calls.append(kwargs)
        return 3


def _handler(
    embedding_service: FakeEmbeddingService,
    vector_store: FakeVectorStore,
    min_paper_date: Optional[str] = None,
) -> ArxivEmbeddingHandler:
    return ArxivEmbeddingHandler(
        embedding_service=embedding_service,
        vector_store_service=vector_store,
        embedding_collection_name="arxiv_paper_embeddings",
        vector_query_batch_size=2,
        token_price_per_1k=1.0,
        min_paper_date=min_paper_date,
    )


def test_store_batch_skips_existing_and_writes_comma_joined_metadata() -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(existing_ids={"2610.00002"})
    papers = [_paper("2610.00001"), _paper("2610.00002"), _paper("2610.00003"), _paper("2610.00004", abstract="")]

    stats = _handler(embedding_service, vector_store).store_paper_embeddings_batch(papers)

    assert embedding_service.batch_calls == [["Title 2610.00001\nAbstract 2610.00001", "Title 2610.00003\nAbstract 2610.00003"]]
    assert [item["metadata"]["arxiv_id"] for item in vector_store.inserted] == ["2610.00001", "2610.00003"]
    metadata = vector_store.inserted[0]["metadata"]
    # 与推荐侧物化写入的格式一致：逗号分隔，不能把 JSON 方括号和引号写进向量库。
    assert metadata["authors"] == "Alice, Bob"
    assert metadata["categories"] == "cs.CL, cs.LG"
    assert metadata["published_date"] == "2026-10-20"
    assert metadata["embedding_model"] == "test-model"
    assert (stats.embeddings_attempted, stats.embeddings_written, stats.embeddings_skipped_existing, stats.embedding_errors) == (2, 2, 1, 0)
    assert stats.embedding_cost_yuan == 0.02


def test_store_batch_gives_up_without_embedding_when_vector_store_is_unreachable() -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(query_error=ConnectionError("milvus down"))

    stats = _handler(embedding_service, vector_store).store_paper_embeddings_batch([_paper("2610.00001"), _paper("2610.00002")])

    assert embedding_service.batch_calls == []
    assert embedding_service.single_calls == []
    assert (stats.embeddings_attempted, stats.embedding_errors) == (0, 2)


def test_store_batch_does_not_re_embed_when_insert_fails() -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore(insert_error=RuntimeError("insert failed"))

    stats = _handler(embedding_service, vector_store).store_paper_embeddings_batch([_paper("2610.00001"), _paper("2610.00002")])

    assert len(embedding_service.batch_calls) == 1
    assert embedding_service.single_calls == []
    assert (stats.embeddings_attempted, stats.embeddings_written, stats.embedding_errors) == (2, 0, 2)


def test_store_batch_falls_back_to_single_items_when_batch_embedding_fails() -> None:
    embedding_service = FakeEmbeddingService(batch_error=RuntimeError("batch rejected"))
    vector_store = FakeVectorStore()

    stats = _handler(embedding_service, vector_store).store_paper_embeddings_batch([_paper("2610.00001"), _paper("2610.00002")])

    assert len(embedding_service.single_calls) == 2
    assert [item["metadata"]["arxiv_id"] for item in vector_store.inserted] == ["2610.00001", "2610.00002"]
    # 回退到单篇时不能把同一批论文的 attempted 再记一遍。
    assert (stats.embeddings_attempted, stats.embeddings_written, stats.embedding_errors) == (2, 2, 0)


def test_find_papers_missing_embeddings_checks_vector_store_in_batches() -> None:
    vector_store = FakeVectorStore(existing_ids={"2610.00003"})
    papers = [_paper(f"2610.0000{index}") for index in range(1, 6)] + [_paper("2610.00009", title="")]

    missing = _handler(FakeEmbeddingService(), vector_store).find_papers_missing_embeddings(papers)

    assert [paper["arxiv_id"] for paper in missing] == ["2610.00001", "2610.00002", "2610.00004", "2610.00005"]


def test_papers_first_submitted_before_window_are_not_embedded() -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    papers = [
        _paper("2610.00001", created="2026-10-20"),
        # 旧论文最近改过版本：updated / datestamp 很新，但按首次提交日期仍在窗口外。
        _paper("2401.00001", created="2024-01-05", updated="2026-10-20", oai_datestamp="2026-10-21"),
        # 日期全缺的无法判断新旧，按窗口内处理。
        _paper("2610.00002", created="", updated="", oai_datestamp=""),
    ]

    handler = _handler(embedding_service, vector_store, min_paper_date="2026-06-25")
    stats = handler.store_paper_embeddings_batch(papers)

    assert [item["metadata"]["arxiv_id"] for item in vector_store.inserted] == ["2610.00001", "2610.00002"]
    assert (stats.embeddings_attempted, stats.embeddings_skipped_out_of_window) == (2, 1)
    assert [paper["arxiv_id"] for paper in handler.find_papers_missing_embeddings(papers)] == ["2610.00001", "2610.00002"]


def test_prune_out_of_window_passes_cutoff_and_kept_ids_to_vector_store() -> None:
    vector_store = FakeVectorStore()

    deleted = _handler(FakeEmbeddingService(), vector_store, min_paper_date="2026-06-25").prune_out_of_window({"2401.00001"}, dry_run=True)

    assert deleted == 3
    assert vector_store.prune_calls == [
        {
            "collection_name": "arxiv_paper_embeddings",
            "cutoff_date": "2026-06-25",
            "keep_arxiv_ids": {"2401.00001"},
            "dry_run": True,
        }
    ]


def test_prune_out_of_window_is_noop_without_window() -> None:
    vector_store = FakeVectorStore()

    assert _handler(FakeEmbeddingService(), vector_store).prune_out_of_window(set()) == 0
    assert vector_store.prune_calls == []
