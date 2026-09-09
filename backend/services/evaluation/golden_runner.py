"""运行单篇论文 golden 评测；失败、拒答和缺少人工标注分别记录。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from services.llm.call_metrics import LLMCallStats, use_call_stats
from services.paper_evidence_research.contracts import PaperEvidenceResearchRequest
from . import metrics_report
from .contracts import GoldenCase
from .eval_record import BACKEND_DIR, _get_git_commit_hash, build_success_eval_record, build_error_eval_record
from .scoring import score_record

logger = logging.getLogger(__name__)
DEFAULT_REPORTS_DIR = BACKEND_DIR / "06-evaluation-result" / "reports"


def load_golden_cases(jsonl_path: Path) -> list[dict[str, Any]]:
    cases, ids = [], set()
    # 格式错误直接指出行号，不能跳过困难样本后还声称完成了整个数据集。
    with jsonl_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                case = GoldenCase.model_validate(raw)
                if case.case_id in ids:
                    raise ValueError(f"重复 case_id: {case.case_id}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{jsonl_path}:{line_number}: {exc}") from exc
            ids.add(case.case_id)
            cases.append({**raw, **case.model_dump(mode="json")})
    if not cases:
        raise ValueError("评测集不能为空")
    return cases


async def run_single_case(golden_case: dict[str, Any], research_service: Any) -> dict[str, Any]:
    case = GoldenCase.model_validate(golden_case)
    run_id = f"eval-{uuid4().hex}"
    # 每次重复都用独立会话与 run_id，避免 checkpoint 或历史回答污染后续试验。
    request = PaperEvidenceResearchRequest(
        arxiv_id=case.arxiv_id, original_question=case.question, user_id="evaluation",
        session_id=f"evaluation-{run_id}", research_run_id=run_id,
    )
    events, stats = [], LLMCallStats()
    started = perf_counter()
    result, error = None, None
    try:
        with use_call_stats(stats):
            result = research_service.research(request, trace_listener=lambda run_id, trace: events.extend(trace))
    except Exception as exc:
        logger.warning("评测运行失败: case_id=%s error_type=%s", case.case_id, type(exc).__name__)
        error = {"code": getattr(exc, "code", "research_execution_failed"),
                 "stage": getattr(exc, "stage", "research"), "error_type": type(exc).__name__}

    # 根据是否有错误选择对应的构造函数
    if error is not None:
        return build_error_eval_record(
            request=request,
            error=error,
            trace_events=events,
            latency_ms=(perf_counter() - started) * 1000,
            llm_usage=stats.to_dict(),
        )
    else:
        return build_success_eval_record(
            request=request,
            result=result,
            trace_events=events,
            latency_ms=(perf_counter() - started) * 1000,
            llm_usage=stats.to_dict(),
        )


async def run_case_with_repeats(golden_case, research_service, repeat_count):
    if repeat_count < 1:
        raise ValueError("repeat 必须为正整数")
    case = GoldenCase.model_validate(golden_case)
    raw_runs = [await run_single_case(golden_case, research_service) for _ in range(repeat_count)]
    all_metrics, missing_observations = [], []
    for run in raw_runs:
        scored = score_record(golden_case, run)
        missing_observations.extend(scored["missing_observations"])
        all_metrics.append(scored["metrics"])
    stats = metrics_report.compute_repeat_stats(all_metrics)
    outcomes = ["error" if run["run_status"] == "error" else run["outcome"] for run in raw_runs]
    counts = dict(Counter(outcomes))
    return {
        "case_id": case.case_id, "golden_case": golden_case,
        "metrics": {name: item["median"] for name, item in stats.items()}, "repeat_stats": stats,
        "run_stats": {"repeat_count": repeat_count, "successful_runs": repeat_count - counts.get("error", 0),
                      "failed_runs": counts.get("error", 0), "outcomes": outcomes, "outcome_counts": counts,
                      "outcome_disagreement": len(counts) > 1},
        "missing_annotations": case.missing_annotations(), "missing_observations": missing_observations,
        "raw_runs": raw_runs,
        "run_metrics": all_metrics,
    }


async def run_golden_evaluation(
    cases_path: Path, repeat_count: int, output_dir: Path = DEFAULT_REPORTS_DIR,
    *, allow_unlabeled: bool = False, research_service: Any = None,
) -> dict[str, Any]:
    if repeat_count < 1:
        raise ValueError("repeat 必须为正整数")
    cases = load_golden_cases(cases_path)
    missing = {case["case_id"]: GoldenCase.model_validate(case).missing_annotations() for case in cases}
    missing = {case_id: fields for case_id, fields in missing.items() if fields}
    if missing and not allow_unlabeled:
        raise ValueError(f"golden 缺少人工标注: {missing}；仅诊断管线可显式使用 --allow-unlabeled")
    if research_service is None:
        # 与生产 QA 使用同一套适配器，导入延迟到真正执行，纯指标测试不会初始化外部依赖。
        from dependencies import get_paper_evidence_research_service
        research_service = get_paper_evidence_research_service()
    records = [await run_case_with_repeats(case, research_service, repeat_count) for case in cases]
    dataset_hash = hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    report = metrics_report.generate_report(
        records, metrics_report.load_previous_report(output_dir),
        # 预算、模型与索引取自逐次运行记录，不能用当前默认值替代真实执行配置。
        metadata={"dataset_hash": dataset_hash, "repeat_count": repeat_count},
    )
    commit = _get_git_commit_hash()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"{timestamp}_{commit}.json"
    metrics_report.save_report(report, output_path, commit)
    logger.info("评测报告: %s；运行统计: %s；基线可用: %s", output_path, report["run_summary"], report["baseline_ready"])
    if report["red_flags"]:
        logger.warning("评测告警: %s", report["red_flags"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="单篇论文生成效果评测")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--allow-unlabeled", action="store_true", help="允许未标注 smoke，仅输出诊断报告")
    parser.add_argument("--validate-only", action="store_true", help="只检查数据和标注，不初始化模型")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.repeat < 1:
            raise ValueError("repeat 必须为正整数")
        if args.validate_only:
            cases = load_golden_cases(args.cases)
            missing = {case["case_id"]: GoldenCase.model_validate(case).missing_annotations() for case in cases}
            missing = {key: value for key, value in missing.items() if value}
            print(json.dumps({"case_count": len(cases), "missing_annotations": missing}, ensure_ascii=False, indent=2))
            return 1 if missing and not args.allow_unlabeled else 0
        report = asyncio.run(run_golden_evaluation(
            args.cases, args.repeat, args.output_dir, allow_unlabeled=args.allow_unlabeled,
        ))
        # 指标回退只告警；技术执行失败返回非零，防止运行脚本误报成功。
        return 1 if report["run_summary"]["failed_runs"] else 0
    except Exception as exc:
        logger.error("评测未完成: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
