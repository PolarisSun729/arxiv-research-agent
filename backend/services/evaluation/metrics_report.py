"""保留失败分母与缺测；数据、指标、索引、模型和预算一致时才比较基线。"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import METRICS_VERSION
from .configuration import configuration_metadata

RED_FLAG_METRICS = {"recall@5", "three_state_accuracy", "citation_fidelity"}
RED_FLAG_THRESHOLD = 0.05


def _metric_values(records, metric_name):
    values = []
    for rec in records:
        metrics = rec.get("metrics") or {}
        # 空指标是失败记录的旧占位形式；不能悄悄从分母删掉。明确的 None 则代表不可评。
        value = metrics.get(metric_name) if metrics else 0.0
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _compute_aggregate_metrics(records):
    names = set().union(*(set(rec.get("metrics", {})) for rec in records)) if records else set()
    result = {}
    for name in sorted(names):
        values = _metric_values(records, name)
        result[name] = statistics.mean(values) if values else None
    return result


def _metric_counts(records):
    metrics = _compute_aggregate_metrics(records)
    counts = {}
    for name in metrics:
        runs = [(run, score) for rec in records
                for run, score in zip(rec.get("raw_runs", []), rec.get("run_metrics", []))]
        scored = [(run, score) for run, score in runs if isinstance(score.get(name), (int, float))]
        counts[name] = {
            "scored_cases": len(_metric_values(records, name)),
            "unscored_cases": len(records) - len(_metric_values(records, name)),
            "scored_runs": len(scored), "unscored_runs": len(runs) - len(scored),
            "successful_scored_runs": sum(run.get("run_status") == "success" for run, _ in scored),
            "failed_scored_runs": sum(run.get("run_status") == "error" for run, _ in scored),
        }
    return counts


def bucket_by_field(case_records, field_name):
    grouped = {}
    for rec in case_records:
        grouped.setdefault(rec.get("golden_case", {}).get(field_name, "unknown"), []).append(rec)
    return {name: {"count": len(records), "metrics": _compute_aggregate_metrics(records),
                   "metric_counts": _metric_counts(records)} for name, records in grouped.items()}


def compute_diff(current_metrics, previous_metrics):
    diff, flags = {}, []
    for name in sorted(RED_FLAG_METRICS):
        current, previous = current_metrics.get(name), previous_metrics.get(name)
        if current is None or previous is None:
            continue
        delta = current - previous
        diff[name] = delta
        if delta < -RED_FLAG_THRESHOLD - 1e-12:
            flags.append(name)
    return diff, flags


def load_previous_report(reports_dir: Path):
    if not reports_dir.exists():
        return None
    for path in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 单份报告损坏不妨碍本次评测，继续寻找最近可读版本。
            continue
    return None


def compute_repeat_stats(repeat_metrics):
    names = set().union(*(set(metrics) for metrics in repeat_metrics)) if repeat_metrics else set()
    result = {}
    for name in sorted(names):
        values = [metrics.get(name) for metrics in repeat_metrics]
        valid = [value for value in values if value is not None]
        result[name] = {
            "median": statistics.median(valid) if valid else None,
            "std": statistics.stdev(valid) if len(valid) > 1 else 0.0 if valid else None,
            "min": min(valid) if valid else None, "max": max(valid) if valid else None,
            "values": values, "valid_count": len(valid), "missing_count": len(values) - len(valid),
        }
    return result


def generate_report(case_records, previous_report=None, *, metadata=None):
    metadata = {**(metadata or {}), **configuration_metadata(case_records)}
    overall = _compute_aggregate_metrics(case_records)
    diff, flags = {}, []
    comparison_status = "no_previous_report"
    if previous_report:
        previous_metadata = previous_report.get("metadata", {})
        comparable = (
            previous_report.get("metrics_version") == METRICS_VERSION
            and previous_metadata.get("dataset_hash") == metadata.get("dataset_hash")
            and bool(metadata.get("dataset_hash"))
        )
        if not comparable:
            comparison_status = "incompatible_dataset_or_metrics"
        elif not metadata["evaluation_config_hash"] or not previous_metadata.get("evaluation_config_hash"):
            comparison_status = "missing_configuration"
        elif (
            metadata["evaluation_config_hash"] != previous_metadata["evaluation_config_hash"]
            or metadata.get("repeat_count") != previous_metadata.get("repeat_count")
        ):
            comparison_status = "incompatible_configuration"
        elif not previous_report.get("baseline_ready"):
            comparison_status = "previous_not_baseline_ready"
        else:
            diff, flags = compute_diff(overall, previous_report.get("overall_metrics", {}))
            comparison_status = "comparable"

    raw_runs = [run for rec in case_records for run in rec.get("raw_runs", [])]
    failed = sum(run.get("run_status") == "error" for run in raw_runs)
    if failed or any(not rec.get("metrics") for rec in case_records):
        flags.append("run_failures")
    missing_annotations = {
        rec["golden_case"]["case_id"]: rec["missing_annotations"]
        for rec in case_records if rec.get("missing_annotations")
    }
    missing_observations = {
        rec["golden_case"]["case_id"]: rec["missing_observations"]
        for rec in case_records if rec.get("missing_observations")
    }
    if missing_observations:
        flags.append("missing_observations")
    matrix = {label: {outcome: 0 for outcome in ("completed", "partial", "abstained", "error")}
              for label in ("answerable", "unanswerable", "unlabeled")}
    for rec in case_records:
        answerable = rec.get("golden_case", {}).get("answerable")
        row = "unlabeled" if answerable is None else "answerable" if answerable else "unanswerable"
        for run in rec.get("raw_runs", []):
            outcome = "error" if run.get("run_status") == "error" else run.get("outcome")
            if outcome in matrix[row]:
                matrix[row][outcome] += 1
    successful = len(raw_runs) - failed
    return {
        "schema_version": "paper_evidence_report_v1", "metrics_version": METRICS_VERSION,
        "metadata": metadata, "overall_metrics": overall, "metric_counts": _metric_counts(case_records),
        # 中位数保留为基线口径，同时公开逐次均值；2 次通过、1 次失败不能只看到中位数 100%。
        "overall_run_metrics": _compute_aggregate_metrics([
            {"metrics": metrics} for rec in case_records for metrics in rec.get("run_metrics", [])
        ]),
        "run_summary": {"case_count": len(case_records), "run_count": len(raw_runs),
                        "successful_runs": successful, "failed_runs": failed,
                        "run_failure_rate": failed / len(raw_runs) if raw_runs else None,
                        "unstable_cases": sum(bool(rec.get("run_stats", {}).get("outcome_disagreement")) for rec in case_records)},
        "baseline_ready": bool(raw_runs) and not failed and not missing_annotations and not missing_observations
                          and bool(metadata["evaluation_config_hash"])
                          and all(len(rec.get("raw_runs", [])) >= 3 for rec in case_records),
        "missing_annotations": missing_annotations, "missing_observations": missing_observations,
        "outcome_matrix": matrix,
        "termination_reasons": dict(Counter(run["termination_reason"] for run in raw_runs if run.get("termination_reason"))),
        "buckets": {"difficulty": bucket_by_field(case_records, "difficulty"),
                    "main_intent": bucket_by_field(case_records, "main_intent")},
        "comparison_status": comparison_status, "diff": diff, "red_flags": flags,
        # 保存三次答案与原始终态，避免把 completed/partial 的波动压成同一个正确率。
        "case_details": case_records,
    }


def save_report(report: dict[str, Any], output_path: Path, git_commit: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"git_commit": git_commit, "created_at": datetime.now(timezone.utc).isoformat(), **report}
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
