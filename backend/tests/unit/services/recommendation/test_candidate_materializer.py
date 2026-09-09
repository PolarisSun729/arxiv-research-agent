from __future__ import annotations

from types import SimpleNamespace

from services.recommendation.candidate_materializer import CandidateMaterializer


class _FakePaperCatalogStore:
    def __init__(self, paper: dict) -> None:
        self.paper = dict(paper)

    def get_paper(self, arxiv_id: str):
        return dict(self.paper) if self.paper.get("arxiv_id") == arxiv_id else None

    def add_paper(self, paper: dict) -> bool:
        self.paper.update(paper)
        return True


class _FakeVectorStoreService:
    def get_paper_embeddings_by_arxiv_ids(self, *, collection_name: str, arxiv_ids: list[str]):
        _ = collection_name, arxiv_ids
        return [{"embedding_id": "7", "embedding_model": "test-model"}]


def test_existing_embedding_is_enriched_by_full_preference_payload() -> None:
    """点赞时若本地论文记录元数据为空，不能因为已有 embedding 就继续返回空卡片。"""
    catalog = _FakePaperCatalogStore(
        {
            "arxiv_id": "2401.00001",
            "title": "",
            "abstract": "",
            "authors": "",
            "categories": "",
            "published_date": "",
            "url": "",
            "embedding_id": "7",
        }
    )
    materializer = CandidateMaterializer()
    materializer.paper_catalog_store = catalog
    materializer.vector_store_service = _FakeVectorStoreService()
    materializer.collection_name = "arxiv_paper_embeddings"
    materializer.get_embedding_config = lambda: SimpleNamespace(model_name="test-model")

    result = materializer._ensure_paper_materialized(
        "2401.00001",
        paper_payload={
            "arxiv_id": "2401.00001",
            "title": "A Complete Paper Title",
            "authors": ["Alice"],
            "abstract": "A complete abstract for the marked paper.",
            "categories": ["cs.CL"],
            "published_date": "2024-01-01",
            "abs_url": "https://arxiv.org/abs/2401.00001",
        },
    )

    assert result["title"] == "A Complete Paper Title"
    assert result["abstract"] == "A complete abstract for the marked paper."
    assert result["embedding_id"] == "7"
