"""统计 Agent 规划基线：读取 agent_planning_record 日志行，输出 LLM 计划通过率、失败类型、超时率和规划耗时。

记录由 backend/agents/arxiv_search_agent/planning_record.py 每轮打一行 INFO 日志：
    ... agent_planning_record {"plan_attempts": [...], "plan_source": "llm", ...}
行首的时间戳、级别、logger 名或 journald 前缀都会被忽略，只取前缀后面的 JSON。

用法（任意目录运行均可）：
    journalctl -u arxiv-agent --since "7 days ago" -o cat | python scripts/planning_baseline_stats.py
    python scripts/planning_baseline_stats.py saved.log another.log
    python scripts/planning_baseline_stats.py saved.log --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# 与 planning_record.PLANNING_RECORD_LOG_PREFIX 保持一致；脚本不导入后端，服务器上不装依赖也能跑。
LOG_PREFIX = "agent_planning_record "
FAILURE_KINDS = ("none", "timeout", "llm_error", "parse_error", "validation_failed")


def parse_records(lines: Iterable[str]) -> List[Dict[str, Any]]:
    """从日志行里取出规划记录；不含前缀或 JSON 坏掉的行直接跳过。"""
    records: List[Dict[str, Any]] = []
    for line in lines:
        index = line.find(LOG_PREFIX)
        if index < 0:
            continue
        try:
            record = json.loads(line[index + len(LOG_PREFIX):].strip())
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """最近秩法分位数：排序后取第 ceil(q·n) 个，样本少时也不插值出不存在的值。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _ratio(count: int, total: int) -> Optional[float]:
    return round(count / total, 4) if total else None


def _shares(counter: Counter, total: int, keys: Sequence[str] = ()) -> Dict[str, Dict[str, Any]]:
    ordered = list(keys) + sorted(key for key in counter if key not in keys)
    return {key: {"count": counter.get(key, 0), "ratio": _ratio(counter.get(key, 0), total)} for key in ordered}


def _llm_attempts(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [attempt for attempt in record.get("plan_attempts") or [] if isinstance(attempt, Mapping) and attempt.get("source") == "llm"]


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    llm_rounds = [record for record in records if _llm_attempts(record)]
    first_llm = [_llm_attempts(record)[0] for record in llm_rounds]
    all_llm = [attempt for record in llm_rounds for attempt in _llm_attempts(record)]
    failure_counter = Counter(str(attempt.get("failure_kind") or "none") for attempt in all_llm)
    source_counter = Counter(str(record.get("plan_source") or "none") for record in records)
    total_ms = [float(record["total_planning_ms"]) for record in records if isinstance(record.get("total_planning_ms"), (int, float))]
    llm_ms = [float(attempt["elapsed_ms"]) for attempt in all_llm if isinstance(attempt.get("elapsed_ms"), (int, float))]
    return {
        "rounds": len(records),
        "llm_rounds": len(llm_rounds),
        "llm_attempts": len(all_llm),
        # 首次通过率：尝试过 LLM 的轮次里，第一次 LLM 计划就通过的比例。
        "llm_first_pass_rate": _ratio(sum(1 for attempt in first_llm if attempt.get("ok")), len(first_llm)),
        # failure_kind 和超时率都按 LLM 尝试次数算；规则 / 模板路径没有超时这回事。
        "llm_failure_kinds": _shares(failure_counter, len(all_llm), FAILURE_KINDS),
        "llm_timeout_rate": _ratio(failure_counter.get("timeout", 0), len(all_llm)),
        # plan_source 为 none 表示这一轮规划直接抛错（llm_only_strict）。
        "plan_sources": _shares(source_counter, len(records), ("llm", "rule", "legacy_template", "unsupported")),
        "total_planning_ms": {"p50": percentile(total_ms, 0.5), "p95": percentile(total_ms, 0.95), "samples": len(total_ms)},
        "llm_attempt_ms": {"p50": percentile(llm_ms, 0.5), "p95": percentile(llm_ms, 0.95), "samples": len(llm_ms)},
    }


def _format_ratio(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _format_ms(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.0f} ms"


def format_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"规划轮次：{summary['rounds']}（尝试 LLM 的 {summary['llm_rounds']} 轮，共 {summary['llm_attempts']} 次 LLM 尝试）",
        f"LLM 计划首次通过率：{_format_ratio(summary['llm_first_pass_rate'])}",
        f"LLM 超时率：{_format_ratio(summary['llm_timeout_rate'])}",
        "LLM 失败类型占比：",
    ]
    for kind, item in summary["llm_failure_kinds"].items():
        lines.append(f"  {kind:<18}{item['count']:>6}  {_format_ratio(item['ratio'])}")
    lines.append("最终计划来源占比：")
    for source, item in summary["plan_sources"].items():
        lines.append(f"  {source:<18}{item['count']:>6}  {_format_ratio(item['ratio'])}")
    for key, title in (("total_planning_ms", "整轮规划耗时"), ("llm_attempt_ms", "单次 LLM 尝试耗时")):
        stats = summary[key]
        lines.append(f"{title}：p50 {_format_ms(stats['p50'])}，p95 {_format_ms(stats['p95'])}（{stats['samples']} 个样本）")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="统计 agent_planning_record 日志，输出规划基线指标")
    parser.add_argument("paths", nargs="*", help="日志文件；不给则从标准输入读")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是文字报告")
    args = parser.parse_args(argv)

    if args.paths:
        lines: List[str] = []
        for path in args.paths:
            lines.extend(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())
    else:
        lines = sys.stdin.read().splitlines()
    summary = summarize(parse_records(lines))
    # Windows 控制台默认 GBK，中文报告统一按 UTF-8 输出。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
