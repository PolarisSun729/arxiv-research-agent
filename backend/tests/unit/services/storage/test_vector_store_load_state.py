from __future__ import annotations

from unittest import mock

from pymilvus.client.types import LoadState
from pymilvus.exceptions import MilvusException

from services.storage.vector_store_service import VectorStoreService


class _ReleasedCollectionClient:
    """模拟 Milvus Lite 进程重启后的状态：collection 存在但处于 released，未 load 前拒绝读取。"""

    def __init__(self) -> None:
        self.loaded = False
        self.load_calls = 0

    def has_collection(self, collection_name: str) -> bool:
        return True

    def describe_collection(self, collection_name: str) -> dict:
        return {"fields": [{"name": name} for name in ("id", "arxiv_id", "title", "vector")]}

    def get_load_state(self, collection_name: str) -> dict:
        return {"state": LoadState.Loaded if self.loaded else LoadState.NotLoad}

    def load_collection(self, collection_name: str) -> None:
        self.load_calls += 1
        self.loaded = True

    def query(self, collection_name: str, **_kwargs) -> list:
        if not self.loaded:
            raise MilvusException(code=101, message=f"Collection '{collection_name}' is in state 'released'")
        return [{"id": 1, "arxiv_id": "2610.11314", "title": "Paper", "vector": [0.1, 0.2]}]


def test_reading_released_collection_loads_it_first() -> None:
    client = _ReleasedCollectionClient()
    service = VectorStoreService()

    with mock.patch.object(service, "_get_client", return_value=client):
        rows = service.get_paper_embeddings_by_arxiv_ids("arxiv_paper_embeddings", ["2610.11314"])

    assert [row["arxiv_id"] for row in rows] == ["2610.11314"]
    assert client.load_calls == 1


def test_reading_loaded_collection_does_not_reload() -> None:
    client = _ReleasedCollectionClient()
    client.loaded = True
    service = VectorStoreService()

    with mock.patch.object(service, "_get_client", return_value=client):
        service.get_paper_embeddings_by_arxiv_ids("arxiv_paper_embeddings", ["2610.11314"])

    assert client.load_calls == 0
