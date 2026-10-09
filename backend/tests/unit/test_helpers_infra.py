import unittest

from tests.helpers import (
    FakeArxivService,
    FakeEmbeddingService,
    FakeGenerationService,
    FakePaperQAService,
    FakeVectorStoreService,
    TemporarySqliteDatabase,
    build_storage_container,
)


class HelperInfrastructureTests(unittest.TestCase):
    def test_temporary_sqlite_database_supports_isolated_writes(self) -> None:
        with TemporarySqliteDatabase() as temp_db:
            with temp_db.connect() as conn:
                conn.execute("CREATE TABLE sample (value TEXT)")
                conn.execute("INSERT INTO sample(value) VALUES (?)", ("ok",))
                row = conn.execute("SELECT value FROM sample").fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "ok")

    def test_build_storage_container_bootstraps_schema_on_temp_db(self) -> None:
        storage = build_storage_container()
        try:
            with storage.connection_provider.connect() as conn:
                table_names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
        finally:
            if getattr(storage, "_test_temp_db", None) is not None:
                storage._test_temp_db.cleanup()

        self.assertIn("arxiv_papers", table_names)
        self.assertIn("paper_qa_index", table_names)

    def test_fake_embedding_service_is_deterministic(self) -> None:
        service = FakeEmbeddingService(dimension=4)

        first = service.create_single_embedding("rag")
        second = service.create_single_embedding("rag")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_fake_generation_service_returns_canned_response(self) -> None:
        service = FakeGenerationService(response_text="hello world")

        response = service.generate("prompt")
        streamed = list(service.stream_qwen_responses("prompt"))

        self.assertEqual(response["answer"], "hello world")
        self.assertEqual(streamed, ["hello", "world"])

    def test_fake_vector_store_supports_basic_insert_and_search(self) -> None:
        service = FakeVectorStoreService()
        service.insert_single_embedding("papers", [0.0, 0.0], {"arxiv_id": "1"})
        service.insert_single_embedding("papers", [1.0, 1.0], {"arxiv_id": "2"})

        results = service.search_similar_vectors("papers", [0.1, 0.1], top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["arxiv_id"], "1")

    def test_fake_arxiv_service_filters_by_id(self) -> None:
        service = FakeArxivService(
            papers=[
                {"arxiv_id": "1", "title": "Paper 1"},
                {"arxiv_id": "2", "title": "Paper 2"},
            ]
        )

        result = service.search(id_list=["2"], max_results=5)

        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["papers"][0]["title"], "Paper 2")

    def test_fake_paper_qa_service_returns_stubbed_answer(self) -> None:
        service = FakePaperQAService(answer_text="stub answer", search_results=[{"title": "Paper A"}])

        result = service.answer_question("1234.5678", {"question": "what?"})

        self.assertEqual(result["answer"], "stub answer")
        self.assertEqual(result["sources"][0]["title"], "Paper A")


if __name__ == "__main__":
    unittest.main()
