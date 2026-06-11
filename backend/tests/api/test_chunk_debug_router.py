from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import chunk_router


class ChunkDebugRouterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_chunk_docs_dir = chunk_router.CHUNK_DOCS_DIR
        chunk_router.CHUNK_DOCS_DIR = Path(self.temp_dir.name)

        sample_payload = {
            "filename": r"C:\private\paper.pdf",
            "pdf_path": r"D:\internal\papers\paper.pdf",
            "pages": [{"page": 1, "raw_text": "secret full page text"}],
            "chunks": [
                {
                    "content": "x" * (chunk_router.CONTENT_PREVIEW_CHARS + 20),
                    "metadata": {"chunk_id": 1, "source": r"D:\internal\papers\paper.pdf"},
                }
            ],
            "embeddings": [
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "metadata": {"chunk_id": 1, "filename": r"D:\internal\chunks\sample.json"},
                }
            ],
        }
        (Path(self.temp_dir.name) / "sample.json").write_text(
            json.dumps(sample_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        (Path(self.temp_dir.name) / "ignored.txt").write_text("ignore", encoding="utf-8")

        app = FastAPI()
        app.include_router(chunk_router.router, prefix="/api")
        self.client = TestClient(app)

    def tearDown(self) -> None:
        chunk_router.CHUNK_DOCS_DIR = self.original_chunk_docs_dir
        self.temp_dir.cleanup()

    def test_list_chunk_files_returns_only_debug_json_metadata(self) -> None:
        response = self.client.get("/api/debug/chunks/files")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["debug"])
        self.assertEqual([item["filename"] for item in payload["files"]], ["sample.json"])
        self.assertTrue(payload["files"][0]["debug_only"])
        self.assertNotIn("path", payload["files"][0])

    def test_get_chunk_file_sanitizes_paths_raw_pages_and_vectors(self) -> None:
        response = self.client.get("/api/debug/chunks/file/sample.json")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        data = payload["data"]

        self.assertTrue(payload["debug"])
        self.assertTrue(payload["sanitized"])
        self.assertEqual(payload["filename"], "sample.json")
        self.assertEqual(data["filename"], "paper.pdf")
        self.assertEqual(data["pdf_path"], "paper.pdf")
        self.assertEqual(data["pages"], "[redacted: raw pages omitted from debug API]")
        self.assertEqual(data["chunks"][0]["metadata"]["source"], "paper.pdf")
        self.assertLess(len(data["chunks"][0]["content"]), chunk_router.CONTENT_PREVIEW_CHARS + 50)
        self.assertEqual(data["embeddings"][0]["embedding"], {"redacted": True, "vector_length": 3})
        self.assertEqual(data["embeddings"][0]["metadata"]["filename"], "sample.json")

    def test_get_chunk_file_rejects_invalid_filename(self) -> None:
        invalid_suffix = self.client.get("/api/debug/chunks/file/sample.txt")
        traversal = self.client.get("/api/debug/chunks/file/..%5Csample.json")

        self.assertEqual(invalid_suffix.status_code, 400)
        self.assertEqual(traversal.status_code, 400)


if __name__ == "__main__":
    unittest.main()
