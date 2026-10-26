import io

import httpx
import pytest
from openai import BadRequestError
from PIL import Image

from core.errors import AppError, ErrorCode
from services.paper_fast_qa import fast_answer_service as fast_module
from services.paper_fast_qa.fast_answer_service import PaperFastQAService
from services.paper_fast_qa.fulltext_service import FulltextParseError, PaperFulltext


def _jpeg_bytes(size=(64, 48)):
    output = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(output, format="JPEG")
    return output.getvalue()


@pytest.fixture
def fulltext(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "fig-1.jpg").write_bytes(_jpeg_bytes())
    return PaperFulltext(arxiv_id="2401.00001", directory=tmp_path, manifest={
        "page_count": 2, "char_count": 30, "image_count": 1, "parse_seconds": 1.0,
        "blocks": [
            {"id": "p1-1", "page": 1, "kind": "heading", "text": "## 2 **Method**"},
            {"id": "p1-2", "page": 1, "kind": "text", "text": "We propose X."},
            {"id": "fig-1", "page": 2, "kind": "figure", "text": "Figure 1: Overview.", "image": "fig-1.jpg"},
            {"id": "p2-1", "page": 2, "kind": "table", "text": "|a|b|"},
        ],
    })


class FakeFulltextService:
    def __init__(self, fulltext, *, ready_after_waits=0, error=None):
        self.fulltext = fulltext
        self.ready_after_waits = ready_after_waits
        self.error = error
        self.waits = 0

    def load_cached(self, arxiv_id):
        return self.fulltext if self.ready_after_waits == 0 and self.error is None else None

    def get_status(self, arxiv_id):
        return {"status": "parsing"}

    def wait_for_fulltext(self, arxiv_id, timeout_seconds):
        self.waits += 1
        if self.error is not None:
            raise self.error
        return self.fulltext if self.waits >= self.ready_after_waits else None


class FakeGeneration:
    def __init__(self, deltas=(), error=None):
        self.deltas = list(deltas)
        self.error = error
        self.calls = []

    def stream_with_qwen(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.error is not None:
            raise self.error
        kwargs["usage_sink"].update({"input_tokens": 100, "input_tokens_details": {"cached_tokens": 80}})
        yield from self.deltas


class FakeSession:
    def __init__(self, history=()):
        self.history = list(history)
        self.persisted = []

    def resolve_chat_session(self, arxiv_id, payload, *, create_if_missing=True):
        return {"session_id": "s-1"} if create_if_missing else {}

    def load_conversation_state(self, *, arxiv_id, payload, chat_session):
        return {"recent_conversation_context": self.history}

    def persist_completed_turn(self, **kwargs):
        self.persisted.append(kwargs)
        return {"turn_id": "t-1", "chat_session": {"session_id": "s-1"}}


@pytest.fixture(autouse=True)
def no_trace_files(monkeypatch):
    monkeypatch.setattr(fast_module, "_write_trace", lambda trace, *, reason: None)


def _service(fulltext_service, generation, session=None, *, history_turns=3):
    return PaperFastQAService(
        fulltext_service=fulltext_service, generation_service=generation,
        session_service=session or FakeSession(), ask_wait_seconds=60, history_turns=history_turns,
    )


def _run(service, question="方法是什么？"):
    events = list(service.answer_stream("2401.00001", {"question": question, "user_id": "u-1"}))
    return events


def test_streams_answer_and_keeps_only_known_citations(fulltext):
    generation = FakeGeneration(["论文提出 X [source:p1-2]", "，结构见图 1 [source:fig-1][source:p9-9]。"])
    session = FakeSession()
    events = _run(_service(FakeFulltextService(fulltext), generation, session))

    assert [event["event"] for event in events] == ["meta", "progress", "delta", "delta", "done"]
    done = events[-1]["data"]
    assert done["answer"] == "论文提出 X [source:p1-2]，结构见图 1 [source:fig-1]。"
    assert done["cited_source_ids"] == ["p1-2", "fig-1"]
    assert done["retrieval_debug"]["invalid_citation_ids"] == ["p9-9"]
    assert done["retrieval_debug"]["usage"]["input_tokens_details"]["cached_tokens"] == 80
    text_source, figure_source = done["sources"]
    assert text_source["chunk_type"] == "text"
    assert text_source["section_title"] == "2 Method"
    assert "asset_url" not in text_source
    assert figure_source["chunk_type"] == "figure"
    assert figure_source["asset_url"] == "/api/paper/2401.00001/evidence-assets/fig-1"
    assert session.persisted[0]["answer"] == done["answer"]
    assert generation.calls[0]["task_type"] == "paper_fast_qa"
    assert generation.calls[0]["enable_thinking"] is False


def test_input_puts_paper_before_recent_history_and_question(fulltext):
    history = [{"question": f"q{index}", "answer_summary": f"a{index}"} for index in range(5)]
    generation = FakeGeneration(["好"])
    _run(_service(FakeFulltextService(fulltext), generation, FakeSession(history)), question="本轮问题")

    messages = generation.calls[0]["prompt"]
    assert messages[0]["role"] == "system"
    paper_parts = messages[1]["content"]
    assert [part["type"] for part in paper_parts] == ["input_text", "input_image", "input_text"]
    assert "[p1-2] We propose X." in paper_parts[0]["text"]
    assert "[fig-1]（第 2 页，图片）图注：Figure 1: Overview." in paper_parts[0]["text"]
    assert paper_parts[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "[p2-1] |a|b|" in paper_parts[2]["text"]
    # 只带最近 3 轮，按时间顺序排在论文之后、本轮问题之前。
    assert [(m["role"], m["content"]) for m in messages[2:]] == [
        ("user", "q2"), ("assistant", "a2"), ("user", "q3"), ("assistant", "a3"),
        ("user", "q4"), ("assistant", "a4"), ("user", "本轮问题"),
    ]


def test_waits_for_parse_with_progress_events(fulltext):
    fulltext_service = FakeFulltextService(fulltext, ready_after_waits=2)
    events = _run(_service(fulltext_service, FakeGeneration(["好"])))

    stages = [event["data"].get("stage") for event in events if event["event"] == "progress"]
    assert stages == ["parsing", "parsing", "generating"]
    assert events[-1]["event"] == "done"


def test_parse_failure_becomes_fulltext_unavailable(fulltext):
    fulltext_service = FakeFulltextService(fulltext, error=FulltextParseError("empty", "no text"))
    generation = FakeGeneration(["不该调用"])

    with pytest.raises(AppError) as caught:
        _run(_service(fulltext_service, generation))
    assert caught.value.code == ErrorCode.PAPER_FULLTEXT_UNAVAILABLE
    assert "扫描版" in caught.value.message
    assert generation.calls == []


def test_model_input_length_error_suggests_research_mode(fulltext):
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/responses"))
    error = BadRequestError("Range of input length should be [1, 258048]", response=response, body=None)
    session = FakeSession()

    with pytest.raises(AppError) as caught:
        _run(_service(FakeFulltextService(fulltext), FakeGeneration(error=error), session))
    assert caught.value.code == ErrorCode.PAPER_TOO_LONG_FOR_FAST_QA
    assert session.persisted == []


def test_oversized_request_downscales_images_once(fulltext, monkeypatch):
    full_size = fast_module._estimate_bytes(fast_module._assemble_input(fulltext, [], "q", image_scale=1.0))
    monkeypatch.setattr(fast_module, "QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES", full_size - 1)
    generation = FakeGeneration(["好"])

    done = _run(_service(FakeFulltextService(fulltext), generation), question="q")[-1]["data"]
    assert done["retrieval_debug"]["input"]["images_downscaled"] is True
    assert done["retrieval_debug"]["input"]["request_bytes"] < full_size

    monkeypatch.setattr(fast_module, "QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES", 100)
    with pytest.raises(AppError) as caught:
        _run(_service(FakeFulltextService(fulltext), FakeGeneration(["好"])), question="q")
    assert caught.value.code == ErrorCode.PAPER_TOO_LONG_FOR_FAST_QA
