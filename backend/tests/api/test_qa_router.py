import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import dependencies
from routers import qa_router
from core.errors import AppError, ErrorCode


class _FakePaperQAService:
    def __init__(self) -> None:
        self.raise_error = False
        self.answer_error_code = None
        self.context_error_code = None

    def get_qa_status(self, arxiv_id: str):
        if self.raise_error:
            raise RuntimeError("qa status failed")
        return {"arxiv_id": arxiv_id, "status": "ready", "chunk_count": 2}

    def build_qa_index(self, arxiv_id: str, loading_method: str = "docling"):
        return {"status": "indexed", "arxiv_id": arxiv_id, "loading_method": loading_method}

    def answer_question(self, arxiv_id: str, payload):
        if self.answer_error_code:
            raise AppError(self.answer_error_code, detail="answer failed in fake service")
        return {
            "arxiv_id": arxiv_id,
            "answer": f"answer:{payload.question}",
            "sources": [{"source_id": "s1"}],
        }

    def build_qa_context(self, arxiv_id: str, payload):
        if self.context_error_code:
            raise AppError(self.context_error_code, detail="context failed in fake service")
        return (
            {"status": "indexed", "collection_name": "paper_2401"},
            [{"content": "chunk", "page_number": "1", "source": "paper.pdf"}],
            {
                "text_context": "chunk",
                "image_inputs": [],
                "asset_metadata": [],
                "generation_question": payload.question,
                "question_contextualization": {},
                "chat_session": {"session_id": "session-1", "user_id": "u1", "arxiv_id": arxiv_id},
            },
            {"trace": "ok"},
        )

    def build_source_payload(self, search_results):
        return search_results

    def persist_completed_turn(self, **kwargs):
        return {"turn_id": "turn-1", "chat_session": kwargs.get("chat_session")}


class _FakeIndexJobManager:
    def __init__(self) -> None:
        self.raise_database_error = False

    def submit_job(self, arxiv_id: str, loading_method: str):
        if self.raise_database_error:
            raise AppError(ErrorCode.DATABASE_WRITE_FAILED, detail="job write failed")
        return {
            "job_id": "job-1",
            "arxiv_id": arxiv_id,
            "status": "pending",
            "current_stage": "pending",
            "progress": 0,
        }


class _FakeMemoryService:
    def __init__(self) -> None:
        self.updated_notes = []

    def update_profile_from_note(self, user_id: str, note):
        self.updated_notes.append((user_id, dict(note)))


class _FakeEnhancedRetrievalService:
    def __init__(self, trace_export_dir: str) -> None:
        self.trace_export_dir = trace_export_dir


class _FakeVectorStoreService:
    def list_collections(self, _provider: str):
        return ["paper_2401"]

    def collection_exists(self, _provider: str, collection_name: str) -> bool:
        return collection_name == "paper_2401"

    def get_collection_info(self, _provider: str, collection_name: str):
        return {"num_entities": 1, "collection_name": collection_name}

    def get_all_chunks(self, collection_name: str, limit: int = 1):
        return [{"chunk_id": "c1", "content": "chunk"}][:limit]


class _FakeGenerationService:
    def __init__(self) -> None:
        self.raise_error = False

    def stream_qwen_responses(self, **kwargs):
        if self.raise_error:
            raise RuntimeError("llm stream failed")
        yield {"type": "delta", "delta": "hello"}
        yield {"type": "completed", "answer": "hello", "usage": None}


class _FakeDatabaseService:
    def __init__(self) -> None:
        self.sessions = {}
        self.notes = {}
        self.latest_job = {
            "job_id": "job-1",
            "arxiv_id": "2401.00001",
            "status": "completed",
            "current_stage": "done",
            "progress": 100,
            "error_message": None,
            "loading_method": "docling",
            "created_at": "2026-06-04T10:00:00",
            "updated_at": "2026-06-04T10:05:00",
        }
        self.qa_index = {
            "arxiv_id": "2401.00001",
            "collection_name": "paper_2401",
            "status": "indexed",
            "chunk_count": 1,
        }
        self.chat_message = {
            "message_id": "msg-1",
            "turn_id": "turn-1",
            "sources": [{"chunk_id": "c1"}],
        }

    def get_paper_qa_index(self, arxiv_id: str):
        return self.qa_index if arxiv_id == "2401.00001" else None

    def get_latest_paper_index_job(self, arxiv_id: str):
        if arxiv_id != "2401.00001":
            return None
        return dict(self.latest_job)

    def list_paper_chat_sessions(self, arxiv_id: str, user_id: str, limit: int = 20):
        items = [item for item in self.sessions.values() if item["arxiv_id"] == arxiv_id and item["user_id"] == user_id]
        return items[:limit]

    def create_paper_chat_session(self, arxiv_id: str, user_id: str, title=None):
        session_id = f"session-{len(self.sessions) + 1}"
        item = {
            "session_id": session_id,
            "user_id": user_id,
            "arxiv_id": arxiv_id,
            "title": title or "",
            "created_at": "2026-06-04T11:00:00",
            "updated_at": "2026-06-04T11:00:00",
            "message_count": 0,
            "status": "active",
        }
        self.sessions[session_id] = item
        return dict(item)

    def list_paper_notes(self, arxiv_id: str, user_id: str, note_type=None):
        items = [item for item in self.notes.values() if item["arxiv_id"] == arxiv_id and item["user_id"] == user_id]
        if note_type:
            items = [item for item in items if item.get("note_type") == note_type]
        return items

    def create_paper_note(self, **kwargs):
        note_id = f"note-{len(self.notes) + 1}"
        item = {
            "note_id": note_id,
            "user_id": kwargs["user_id"],
            "arxiv_id": kwargs["arxiv_id"],
            "session_id": kwargs.get("session_id"),
            "source_message_id": kwargs.get("source_message_id"),
            "title": kwargs.get("title") or "",
            "content": kwargs.get("content") or "",
            "note_type": kwargs.get("note_type") or "custom",
            "source_chunk_ids": list(kwargs.get("source_chunk_ids") or []),
            "tags": list(kwargs.get("tags") or []),
            "include_in_profile": bool(kwargs.get("include_in_profile", False)),
            "created_at": "2026-06-04T11:05:00",
            "updated_at": "2026-06-04T11:05:00",
        }
        self.notes[note_id] = item
        return dict(item)

    def get_paper_chat_message(self, message_id: str, user_id: str = ""):
        if message_id == "msg-1":
            return {
                "message_id": "msg-1",
                "turn_id": "turn-1",
                "sources": [{"chunk_id": "c1"}],
            }
        return None

    def get_paper_chat_message_by_turn(self, session_id: str, turn_id: str, role: str = "assistant", user_id: str = ""):
        if turn_id == "turn-1":
            return {"message_id": "msg-1", "turn_id": "turn-1", "sources": [{"chunk_id": "c1"}]}
        return None


class QaRouterApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paper_qa_service = _FakePaperQAService()
        self.db_service = _FakeDatabaseService()
        self.vector_store_service = _FakeVectorStoreService()
        self.memory_service = _FakeMemoryService()
        self.index_job_manager = _FakeIndexJobManager()
        self.generation_service = _FakeGenerationService()
        self.enhanced_retrieval_service = _FakeEnhancedRetrievalService(self.temp_dir.name)

        app = FastAPI()
        app.include_router(qa_router.router, prefix="/api")
        app.dependency_overrides[dependencies.get_paper_qa_service] = lambda: self.paper_qa_service
        app.dependency_overrides[dependencies.get_database_service] = lambda: self.db_service
        app.dependency_overrides[dependencies.get_vector_store_service] = lambda: self.vector_store_service
        app.dependency_overrides[dependencies.get_memory_service] = lambda: self.memory_service
        app.dependency_overrides[dependencies.get_index_job_manager] = lambda: self.index_job_manager
        app.dependency_overrides[dependencies.get_enhanced_retrieval_service] = lambda: self.enhanced_retrieval_service
        app.dependency_overrides[dependencies.get_generation_service] = lambda: self.generation_service
        self.client = TestClient(app)
        self.addCleanup(self.temp_dir.cleanup)

    def test_get_qa_status_returns_payload(self) -> None:
        response = self.client.get("/api/paper/2401.00001/qa-status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_get_qa_status_maps_service_error_to_500(self) -> None:
        self.paper_qa_service.raise_error = True

        response = self.client.get("/api/paper/2401.00001/qa-status")

        self.assertEqual(response.status_code, 500)
        self.assertIn("qa status failed", response.json()["detail"])

    def test_get_qa_diagnose_returns_diagnostic_fields(self) -> None:
        response = self.client.get("/api/paper/2401.00001/qa-diagnose", params={"sample_limit": 2})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["checks"]["collection_exists"])

    def test_get_qa_trace_latest_supports_success_and_invalid_format(self) -> None:
        trace_dir = Path(self.temp_dir.name) / qa_router.sanitize_trace_slug("2401.00001")
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "trace.md").write_text("# trace", encoding="utf-8")

        success = self.client.get("/api/paper/2401.00001/qa-trace/latest")
        invalid = self.client.get("/api/paper/2401.00001/qa-trace/latest", params={"format": "txt"})

        self.assertEqual(success.status_code, 200)
        self.assertIn("text/markdown", success.headers["content-type"])
        self.assertEqual(invalid.status_code, 400)

    def test_create_qa_index_returns_submitted_job(self) -> None:
        response = self.client.post("/api/paper/2401.00001/create-qa-index")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "submitted")
        self.assertEqual(payload["job_id"], "job-1")

    def test_get_latest_qa_index_job_returns_404_when_missing(self) -> None:
        response = self.client.get("/api/paper/missing/qa-index-jobs/latest")

        self.assertEqual(response.status_code, 404)

    def test_chat_sessions_list_and_create_return_stable_items(self) -> None:
        created = self.client.post("/api/paper/2401.00001/chat-sessions", json={"user_id": "u1", "title": "Session A"})
        listed = self.client.get("/api/paper/2401.00001/chat-sessions", params={"user_id": "u1"})

        self.assertEqual(created.status_code, 200)
        self.assertIn("item", created.json())
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["items"]), 1)

    def test_notes_list_and_create_return_stable_items(self) -> None:
        created = self.client.post(
            "/api/paper/2401.00001/notes",
            json={
                "user_id": "u1",
                "source_message_id": "msg-1",
                "title": "My note",
                "content": "Useful detail",
                "note_type": "method",
                "tags": ["rag"],
            },
        )
        listed = self.client.get("/api/paper/2401.00001/notes", params={"user_id": "u1"})

        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["item"]["source_turn_id"], "turn-1")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["items"]), 1)

    def test_create_note_returns_422_when_content_missing(self) -> None:
        response = self.client.post(
            "/api/paper/2401.00001/notes",
            json={"user_id": "u1", "title": "No content"},
        )

        self.assertEqual(response.status_code, 422)

    def test_qa_returns_answer_payload(self) -> None:
        response = self.client.post(
            "/api/paper/2401.00001/qa",
            json={"question": "What is the contribution?", "user_id": "u1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", response.json())
        self.assertEqual(response.json()["arxiv_id"], "2401.00001")

    def test_qa_returns_qa_index_not_found_code(self) -> None:
        self.paper_qa_service.answer_error_code = ErrorCode.QA_INDEX_NOT_FOUND

        response = self.client.post(
            "/api/paper/2401.00001/qa",
            json={"question": "What is the contribution?", "user_id": "u1"},
        )

        self.assertEqual(response.status_code, 404)
        payload = response.json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["code"], ErrorCode.QA_INDEX_NOT_FOUND)
        self.assertTrue(payload["recoverable"])

    def test_qa_returns_vector_store_error_code(self) -> None:
        self.paper_qa_service.answer_error_code = ErrorCode.VECTOR_STORE_ERROR

        response = self.client.post(
            "/api/paper/2401.00001/qa",
            json={"question": "What is the contribution?", "user_id": "u1"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], ErrorCode.VECTOR_STORE_ERROR)

    def test_qa_returns_llm_generation_failed_code(self) -> None:
        self.paper_qa_service.answer_error_code = ErrorCode.LLM_GENERATION_FAILED

        response = self.client.post(
            "/api/paper/2401.00001/qa",
            json={"question": "What is the contribution?", "user_id": "u1"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["code"], ErrorCode.LLM_GENERATION_FAILED)

    def test_create_qa_index_database_write_failure_returns_code(self) -> None:
        self.index_job_manager.raise_database_error = True

        response = self.client.post("/api/paper/2401.00001/create-qa-index")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["code"], ErrorCode.DATABASE_WRITE_FAILED)

    def test_qa_stream_error_event_contains_code(self) -> None:
        self.paper_qa_service.context_error_code = ErrorCode.QA_INDEX_NOT_FOUND

        response = self.client.post(
            "/api/paper/2401.00001/qa/stream",
            json={"question": "What is the contribution?", "user_id": "u1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertIn(f'"code": "{ErrorCode.QA_INDEX_NOT_FOUND}"', response.text)

    def test_qa_stream_llm_exception_maps_to_generation_code(self) -> None:
        self.generation_service.raise_error = True

        response = self.client.post(
            "/api/paper/2401.00001/qa/stream",
            json={"question": "What is the contribution?", "user_id": "u1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertIn(f'"code": "{ErrorCode.LLM_GENERATION_FAILED}"', response.text)


if __name__ == "__main__":
    unittest.main()
