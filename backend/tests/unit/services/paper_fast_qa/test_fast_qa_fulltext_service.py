import threading
from pathlib import Path

import pytest

from services.paper_fast_qa import fulltext_service as fulltext_module
from services.paper_fast_qa.fulltext_service import FulltextParseError, PaperFulltextService


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeParser:
    """直接在当前进程里"解析"：记录调用顺序，可按论文阻塞或返回指定结果。"""

    def __init__(self):
        self.calls = []
        self.gates = {}
        self.results = {}
        self.errors = {}

    def __call__(self, pdf_path, image_dir, image_dpi):
        arxiv_id = Path(pdf_path).stem
        self.calls.append(arxiv_id)
        gate = self.gates.get(arxiv_id)
        if gate is not None:
            assert gate.wait(10), "测试闸门没有放行"
        if arxiv_id in self.errors:
            raise self.errors[arxiv_id]
        if arxiv_id in self.results:
            return self.results[arxiv_id]
        Path(image_dir).mkdir(parents=True, exist_ok=True)
        (Path(image_dir) / "fig-1.jpg").write_bytes(b"jpeg")
        return {
            "schema_version": 1, "page_count": 1, "char_count": 5, "image_count": 1,
            "blocks": [
                {"id": "p1-1", "page": 1, "kind": "text", "text": "hello"},
                {"id": "fig-1", "page": 1, "kind": "figure", "text": "Figure 1", "image": "fig-1.jpg"},
            ],
        }


@pytest.fixture
def parser(monkeypatch):
    fake = FakeParser()

    # 子进程隔离由 isolated_subprocess 自己的测试覆盖；这里直接调用，便于控制解析时序。
    def run_inline(target, *args, **kwargs):
        return target(*args)

    monkeypatch.setattr(fulltext_module, "run_in_isolated_subprocess", run_inline)
    return fake


def _make_service(tmp_path, parser, *, concurrency=2, queue_limit=20, clock=None):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    def provide_pdf(arxiv_id):
        path = pdf_dir / f"{arxiv_id}.pdf"
        if not path.exists():
            path.write_bytes(f"%PDF {arxiv_id}".encode())
        return str(path)

    return PaperFulltextService(
        cache_dir=tmp_path / "cache", pdf_provider=provide_pdf, parse_concurrency=concurrency,
        parse_queue_limit=queue_limit, parse_timeout_seconds=30, failure_cooldown_seconds=300,
        image_dpi=144, parse_function=parser, clock=clock or FakeClock(),
    )


def _wait_until_ready(service, arxiv_id):
    for _ in range(500):
        if service.get_status(arxiv_id)["status"] == "ready":
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"{arxiv_id} 没有解析完成")


def test_parse_result_is_cached_and_reused(tmp_path, parser):
    service = _make_service(tmp_path, parser)

    fulltext = service.wait_for_fulltext("2401.00001", 10)

    assert [block["id"] for block in fulltext.blocks] == ["p1-1", "fig-1"]
    assert fulltext.manifest["pdf_sha256"]
    assert service.get_status("2401.00001")["status"] == "ready"
    assert service.request_parse("2401.00001", priority=False)["status"] == "ready"
    assert service.wait_for_fulltext("2401.00001", 10).directory == fulltext.directory
    assert parser.calls == ["2401.00001"]
    # 写入走临时目录再 rename，完成后不留临时目录。
    assert [path.name for path in (tmp_path / "cache" / "2401.00001").iterdir()] == [fulltext.directory.name]


def test_image_path_only_resolves_blocks_in_manifest(tmp_path, parser):
    service = _make_service(tmp_path, parser)
    service.wait_for_fulltext("2401.00001", 10)

    assert service.resolve_image_path("2401.00001", "fig-1").endswith("fig-1.jpg")
    # 不是全文块 ID 的交给详细模式的证据查找。
    assert service.resolve_image_path("2401.00001", "12") is None
    with pytest.raises(FileNotFoundError):
        service.resolve_image_path("2401.00001", "fig-9")
    with pytest.raises(FileNotFoundError):
        service.resolve_image_path("2401.00001", "p1-1")


def test_concurrent_requests_for_same_paper_share_one_parse(tmp_path, parser):
    parser.gates["2401.00001"] = threading.Event()
    service = _make_service(tmp_path, parser)

    assert service.request_parse("2401.00001", priority=False)["status"] in {"queued", "parsing"}
    assert service.request_parse("2401.00001", priority=False)["status"] in {"queued", "parsing"}
    parser.gates["2401.00001"].set()
    assert service.wait_for_fulltext("2401.00001", 10) is not None
    assert parser.calls == ["2401.00001"]


def test_question_triggered_parse_jumps_ahead_of_prefetch(tmp_path, parser):
    blocker = threading.Event()
    parser.gates["2401.00000"] = blocker
    service = _make_service(tmp_path, parser, concurrency=1)
    service.request_parse("2401.00000", priority=False)
    service.request_parse("2401.00001", priority=False)
    service.request_parse("2401.00002", priority=False)

    # 唯一的工作线程被第一篇占住时，提问触发的第三篇应排到已在排队的预解析前面。
    assert service.wait_for_fulltext("2401.00003", 0) is None
    blocker.set()
    # 只能轮询状态：wait_for_fulltext 本身按提问优先级排队，会改变要验证的顺序。
    _wait_until_ready(service, "2401.00002")
    assert parser.calls == ["2401.00000", "2401.00003", "2401.00001", "2401.00002"]


def test_prefetch_queue_limit_does_not_block_questions(tmp_path, parser):
    blocker = threading.Event()
    parser.gates["2401.00000"] = blocker
    service = _make_service(tmp_path, parser, concurrency=1, queue_limit=1)
    service.request_parse("2401.00000", priority=False)
    for _ in range(100):
        if parser.calls:
            break
        threading.Event().wait(0.01)

    assert service.request_parse("2401.00001", priority=False)["status"] == "queued"
    assert service.request_parse("2401.00002", priority=False) == {"status": "busy", "reason": "queue_full"}
    assert service.wait_for_fulltext("2401.00003", 0) is None
    blocker.set()
    assert service.wait_for_fulltext("2401.00003", 10) is not None


def test_failed_parse_cools_down_before_retry(tmp_path, parser):
    clock = FakeClock()
    parser.errors["2401.00001"] = RuntimeError("bad pdf")
    service = _make_service(tmp_path, parser, clock=clock)

    with pytest.raises(FulltextParseError) as caught:
        service.wait_for_fulltext("2401.00001", 10)
    assert caught.value.reason == "failed"
    assert service.get_status("2401.00001") == {"status": "failed", "reason": "cooling_down"}
    with pytest.raises(FulltextParseError) as caught:
        service.wait_for_fulltext("2401.00001", 10)
    assert caught.value.reason == "cooling_down"
    assert parser.calls == ["2401.00001"]

    clock.now += 301
    del parser.errors["2401.00001"]
    assert service.wait_for_fulltext("2401.00001", 10) is not None
    assert parser.calls == ["2401.00001", "2401.00001"]


def test_pdf_without_text_is_reported_as_empty(tmp_path, parser):
    parser.results["2401.00001"] = {"blocks": [], "page_count": 3}
    service = _make_service(tmp_path, parser)

    with pytest.raises(FulltextParseError) as caught:
        service.wait_for_fulltext("2401.00001", 10)
    assert caught.value.reason == "empty"


@pytest.mark.parametrize("bad_id", ["..", "a b", "x\\y", ""])
def test_unsafe_paper_id_is_rejected(tmp_path, parser, bad_id):
    service = _make_service(tmp_path, parser)
    with pytest.raises(ValueError):
        service.request_parse(bad_id, priority=False)
