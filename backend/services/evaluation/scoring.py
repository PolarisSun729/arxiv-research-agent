"""新运行与落盘重评分共用入口，避免运行失败在两条路径中使用不同分母。"""

from typing import Any

from .contracts import EvaluationRecord, GoldenCase
from .metrics_generation import compute_generation_metrics
from .metrics_retrieval import compute_retrieval_metrics


def score_record(golden_case: dict[str, Any], eval_record: dict[str, Any]) -> dict[str, Any]:
    case = GoldenCase.model_validate(golden_case)
    record = EvaluationRecord.model_validate(eval_record)
    retrieval = compute_retrieval_metrics(case.expected_chunk_ids, record.trace_events)
    if record.run_status == "error":
        # 标注完备的失败样本占分母；不可评价的字段保留 null，而非伪造正确拒答或零成本。
        retrieval = {name: 0.0 if value is not None else None for name, value in retrieval.items()}
    generation = compute_generation_metrics(golden_case, record.model_dump(mode="json"))
    missing = []
    if record.run_status == "success" and not any(event.get("event_type") == "research_completed" for event in record.trace_events):
        missing.append(f"{record.record_id}:research_trace")
    if record.citations and generation["citation_fidelity"] is None:
        missing.append(f"{record.record_id}:claim_verification")
    return {
        "metrics": {**retrieval, **generation,
                    **{name: record.efficiency.get(name) for name in ("latency_ms", "llm_calls", "total_tokens")}},
        "missing_annotations": case.missing_annotations(), "missing_observations": missing,
    }
