import unittest

from tests.helpers import (
    FakeArxivService,
    FakeEmbeddingService,
    FakeGenerationService,
    FakePaperQAService,
    FakeVectorStoreService,
    TemporarySqliteDatabase,
    build_database_service,
)


class _DummyDatabaseService:
    def _ensure_database_directory(self) -> None:
        return None

    def _initialize_database(self) -> None:
        conn = self._get_connection()
        try:
            conn.execute("CREATE TABLE sample_table (id INTEGER PRIMARY KEY, value TEXT)")
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self):
        import sqlite3

        return sqlite3.connect(self.db_path, check_same_thread=self.check_same_thread)


class HelperInfrastructureTests(unittest.TestCase):
    def test_temporary_sqlite_database_supports_isolated_writes(self) -> None:
        with TemporarySqliteDatabase() as temp_db:
            with temp_db.connect() as conn:
                conn.execute("CREATE TABLE sample (value TEXT)")
                conn.execute("INSERT INTO sample(value) VALUES (?)", ("ok",))
                row = conn.execute("SELECT value FROM sample").fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "ok")

    def test_build_database_service_bootstraps_schema_on_temp_db(self) -> None:
        service = build_database_service(_DummyDatabaseService)
        try:
            conn = service._get_connection()
            try:
                table_names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
            finally:
                conn.close()
        finally:
            if getattr(service, "_test_temp_db", None) is not None:
                service._test_temp_db.cleanup()

        self.assertIn("sample_table", table_names)

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

        result = service.search_papers(id_list=["2"], max_results=5)

        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(result["papers"][0]["title"], "Paper 2")

    def test_fake_paper_qa_service_returns_stubbed_answer(self) -> None:
        service = FakePaperQAService(answer_text="stub answer", search_results=[{"title": "Paper A"}])

        result = service.answer_question("1234.5678", {"question": "what?"})

        self.assertEqual(result["answer"], "stub answer")
        self.assertEqual(result["sources"][0]["title"], "Paper A")


if __name__ == "__main__":
    unittest.main()
