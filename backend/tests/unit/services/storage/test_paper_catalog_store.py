import pytest

from tests.helpers.sqlite import build_storage_container
from services.storage.sqlite.stores.paper_catalog import paper_has_display_metadata


def test_get_papers_by_ids_returns_local_hits_keyed_by_id(tmp_path) -> None:
    catalog = build_storage_container(db_path=str(tmp_path / "catalog.sqlite")).paper_catalog
    catalog.add_paper({"arxiv_id": "2401.00001", "title": "First", "abstract": "A", "authors": ["Ada"], "categories": ["cs.AI"]})
    catalog.add_paper({"arxiv_id": "2401.00002", "title": "Second", "abstract": "B"})

    papers = catalog.get_papers_by_ids(["2401.00002", "missing", "2401.00001", "2401.00002", ""])

    # 本地缺失的 ID 不出现在结果里，重复 ID 只查一次；字段反序列化与单篇读取一致。
    assert set(papers) == {"2401.00001", "2401.00002"}
    assert papers["2401.00001"] == catalog.get_paper("2401.00001")
    assert papers["2401.00001"]["authors"] == ["Ada"]
    assert catalog.get_papers_by_ids([]) == {}


def test_get_papers_by_ids_batches_beyond_sqlite_parameter_limit(tmp_path) -> None:
    catalog = build_storage_container(db_path=str(tmp_path / "catalog.sqlite")).paper_catalog
    ids = [f"2401.{index:05d}" for index in range(1200)]
    for arxiv_id in ids[::100]:
        catalog.add_paper({"arxiv_id": arxiv_id, "title": arxiv_id, "abstract": "x"})

    assert set(catalog.get_papers_by_ids(ids)) == set(ids[::100])


@pytest.mark.parametrize("paper,expected", [
    ({"title": "T", "abstract": "A"}, True),
    ({"title": "T", "summary": "A"}, True),
    ({"title": " ", "abstract": "A"}, False),
    ({"title": "T"}, False),
    ({"arxiv_id": "only-id"}, False),
])
def test_paper_has_display_metadata(paper, expected) -> None:
    assert paper_has_display_metadata(paper) is expected
