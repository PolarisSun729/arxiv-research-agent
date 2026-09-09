"""运行器用真实研究图执行离线样本；不访问模型或索引服务。"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from services.evaluation import golden_runner, metrics_report
from services.evaluation.scoring import score_record
from tests.unit.services.paper_evidence_research.test_research_service import ScriptedDecisionPolicy, _service


class FreshResearch:
    def __init__(self, fail_at=()):
        self.calls = 0
        self.fail_at = fail_at

    def research(self, request, *, trace_listener=None):
        self.calls += 1
        if self.calls in self.fail_at:
            raise TimeoutError("offline provider timeout")
        # 每次使用新的脚本策略；图、请求、结果和轨迹都是生产模型。
        return _service(ScriptedDecisionPolicy(), configuration_provider=lambda: {
            "generation": {"provider": "offline", "models": {"default": "script-v1"}},
            "retrieval": {"top_k": 8, "provider": "scripted"},
        }).research(request, trace_listener=trace_listener)


def _write_cases(tmp_path, cases):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases), encoding="utf-8")
    return path


@pytest.mark.parametrize("invalid", [{"answerable": "false"}, {"question": "  "}, {"expected_chunk_ids": [""]}])
def test_invalid_labels_are_rejected_with_line_number(tmp_path, golden_case, invalid):
    path = _write_cases(tmp_path, [{**golden_case, **invalid}])
    with pytest.raises(ValueError, match=r"cases.jsonl:1:"):
        golden_runner.load_golden_cases(path)


def test_duplicate_cases_are_not_silently_skipped(tmp_path, golden_case):
    path = _write_cases(tmp_path, [golden_case, golden_case])
    with pytest.raises(ValueError, match="重复 case_id"):
        golden_runner.load_golden_cases(path)


def test_unknown_annotations_stop_before_initializing_provider(tmp_path, golden_case, monkeypatch):
    golden_case.pop("answerable")
    path = _write_cases(tmp_path, [golden_case])
    import dependencies

    def forbidden():
        raise AssertionError("不应初始化模型")

    monkeypatch.setattr(dependencies, "get_paper_evidence_research_service", forbidden)
    with pytest.raises(ValueError, match="缺少人工标注"):
        asyncio.run(golden_runner.run_golden_evaluation(path, 3, tmp_path / "reports"))


def test_repeated_failure_is_visible_despite_successful_median(golden_case):
    case_record = asyncio.run(golden_runner.run_case_with_repeats(golden_case, FreshResearch(fail_at={3}), 3))
    report = metrics_report.generate_report([case_record])
    assert case_record["metrics"]["three_state_accuracy"] == 1.0
    assert case_record["run_stats"]["outcomes"] == ["completed", "completed", "error"]
    assert report["overall_run_metrics"]["three_state_accuracy"] == pytest.approx(2 / 3)
    assert report["run_summary"]["run_failure_rate"] == pytest.approx(1 / 3)
    assert report["run_summary"]["unstable_cases"] == 1
    assert len({run["session_id"] for run in case_record["raw_runs"]}) == 3
    assert not report["baseline_ready"]
    # 落盘后重新评分必须保留与首次运行相同的失败分母及成本未知值。
    for raw, metrics in zip(case_record["raw_runs"], case_record["run_metrics"]):
        restored = json.loads(json.dumps(raw))
        assert score_record(golden_case, restored)["metrics"] == metrics


def test_cli_factory_wiring_and_report_are_offline(tmp_path, golden_case, monkeypatch):
    import dependencies

    engine = FreshResearch()
    monkeypatch.setattr(dependencies, "get_paper_evidence_research_service", lambda: engine)
    path, output_dir = _write_cases(tmp_path, [golden_case]), tmp_path / "reports"
    monkeypatch.setattr(sys, "argv", ["golden_runner", "--cases", str(path), "--repeat", "3", "--output-dir", str(output_dir)])
    assert golden_runner.main() == 0
    report = json.loads(next(output_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert engine.calls == 3
    assert report["baseline_ready"]
    assert report["run_summary"]["successful_runs"] == 3
    assert report["overall_metrics"]["recall@5"] == 1.0
    assert report["case_details"][0]["raw_runs"][0]["answer"]


def test_default_report_directory_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = Path(golden_runner.__file__).resolve().parents[2] / "06-evaluation-result" / "reports"
    assert golden_runner.DEFAULT_REPORTS_DIR == expected
    assert golden_runner.DEFAULT_REPORTS_DIR.is_absolute()


def test_validate_only_reports_missing_labels(tmp_path, golden_case, monkeypatch, capsys):
    for field in ("expected_chunk_ids", "answerable", "gold_answer"):
        golden_case.pop(field)
    path = _write_cases(tmp_path, [golden_case])
    monkeypatch.setattr(sys, "argv", ["golden_runner", "--cases", str(path), "--validate-only"])
    assert golden_runner.main() == 1
    missing = json.loads(capsys.readouterr().out)["missing_annotations"]["contract"]
    assert set(missing) == {"expected_chunk_ids", "answerable", "gold_answer"}
