"""生成指标使用实际主张校验和草稿覆盖快照；词面覆盖仅用于要点，不充当证据支持。"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from .contracts import EvaluationRecord, GoldenCase


@lru_cache(maxsize=1)
def _load_aliases() -> dict[str, list[str]]:
    return json.loads(Path(__file__).with_name("answer_point_aliases.json").read_text(encoding="utf-8"))


def get_point_aliases(point: str) -> list[str]:
    normalized = _normalize_text(point)
    for aliases in _load_aliases().values():
        if any(_normalize_text(alias) == normalized for alias in aliases):
            return aliases
    return [point]


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def _point_covered(point: str, answer_text: str) -> bool:
    answer = _normalize_text(answer_text)
    for alias in get_point_aliases(point):
        phrase = _normalize_text(alias)
        if not phrase:
            continue
        # 中文没有空格词界；英文使用词界，避免把 cat 在 concatenate 中误判为命中。
        if re.search(r"[\u3400-\u9fff]", phrase):
            if phrase in answer:
                return True
        elif re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", answer):
            return True
    return False


def calculate_key_coverage_rate(expected_answer_points: list[str], answer_text: str) -> float | None:
    points = [point for point in expected_answer_points if point.strip()]
    if not points:
        return None
    return sum(_point_covered(point, answer_text) for point in points) / len(points)


def _verification_view(trace_events: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any], dict[str, Any], bool]:
    known_ids: set[str] = set()
    latest = None
    for event in trace_events:
        if event.get("event_type") == "retrieval_completed":
            known_ids.update(c["candidate_id"] for c in event.get("candidates", []) if c.get("candidate_id"))
        elif event.get("event_type") == "claim_verification_completed":
            known_ids.update(event.get("candidate_ids") or [])
            if latest is None or event.get("draft_version", 0) >= latest.get("draft_version", 0):
                latest = event
    assessments = {a["claim_id"]: a for a in (latest or {}).get("assessments", []) if a.get("claim_id")}
    claims = {c["claim_id"]: c for c in (latest or {}).get("claims", []) if c.get("claim_id")}
    return known_ids, assessments, claims, latest is not None and "claims" in latest



def calculate_citation_fidelity(
    citations: list[dict[str, Any]], evidence_pool: set[str],
    claim_assessments: dict[str, Any] | None = None,
    claims: dict[str, Any] | None = None,
) -> float | None:
    if not citations or claim_assessments is None:
        return None
    faithful = 0
    for citation in citations:
        source_id = citation.get("source_id")
        claim_ids = citation.get("claim_ids") or []
        # 每条绑定主张都必须被该真实片段支持，不能只检查词面相似或任意一个主张。
        if source_id in evidence_pool and claim_ids and all(
            claim_assessments.get(claim_id, {}).get("verdict") == "supported"
            and source_id in claim_assessments[claim_id].get("supporting_evidence_ids", [])
            and set(claim_assessments[claim_id].get("supporting_evidence_ids", [])) <= evidence_pool
            and (claims is None or (
                claim_id in claims and not claims[claim_id].get("requires_visual_verification")
                and set(claims[claim_id].get("citation_ids", []))
                == set(claim_assessments[claim_id].get("supporting_evidence_ids", []))
            ))
            for claim_id in claim_ids
        ):
            faithful += 1
    return faithful / len(citations)


def classify_expected_state(answerable: bool | None) -> str:
    return "unknown" if answerable is None else "answered_or_partial" if answerable else "abstained"


def classify_actual_state(outcome: str | None) -> str:
    return {"completed": "answered", "partial": "partial", "abstained": "abstained"}.get(outcome, "error")


def calculate_three_state_accuracy(answerable: bool | None, outcome: str | None) -> float | None:
    if answerable is None:
        return None
    if outcome not in {"completed", "partial", "abstained"}:
        return 0.0
    return float(outcome in {"completed", "partial"} if answerable else outcome == "abstained")


def batch_three_state_accuracy(cases) -> float | None:
    scores = [calculate_three_state_accuracy(a, o) for a, o in cases]
    valid = [score for score in scores if score is not None]
    return sum(valid) / len(valid) if valid else None


def _draft_snapshots(trace_events) -> list[dict[str, Any]]:
    snapshots = {
        event["draft_version"]: event
        for event in trace_events
        if event.get("event_type") == "evidence_coverage_projected" and event.get("draft_version", 0) > 0
    }
    return [snapshots[version] for version in sorted(snapshots)]


def calculate_self_repair_gain(trace_events: list[dict[str, Any]]) -> float | None:
    snapshots = _draft_snapshots(trace_events)
    if not snapshots:
        return None
    # 比较可靠支持的前后差值；保留负数，才能观察有害修复。
    return float(snapshots[-1]["supported_claim_count"] - snapshots[0]["supported_claim_count"])


def _core_coverage_gain(trace_events: list[dict[str, Any]]) -> float | None:
    snapshots = _draft_snapshots(trace_events)
    if not snapshots or not snapshots[0].get("core_need_count") or not snapshots[-1].get("core_need_count"):
        return None
    def ratio(snapshot):
        return snapshot["satisfied_core_need_count"] / snapshot["core_need_count"]
    return ratio(snapshots[-1]) - ratio(snapshots[0])


def compute_generation_metrics(golden_case: dict[str, Any], eval_record: dict[str, Any]) -> dict[str, Any]:
    case = GoldenCase.model_validate(golden_case)
    record = EvaluationRecord.model_validate(eval_record)
    failed = record.run_status == "error"
    events = record.trace_events
    known_ids, assessments, claims, has_verification = _verification_view(events)
    fidelity = calculate_citation_fidelity(record.citations, known_ids, assessments if has_verification else None, claims)
    if (failed and (record.citations or case.answerable is not False)) or (record.outcome in {"completed", "partial"} and not record.citations):
        # 持久化失败仍可能保留研究引用，适用指标必须归零，不能因负例标签而给错误运行满分。
        fidelity = 0.0
    gain = None if failed else calculate_self_repair_gain(events)
    coverage_gain = None if failed else _core_coverage_gain(events)
    repaired = len(_draft_snapshots(events)) > 1
    return {
        "key_coverage_rate": calculate_key_coverage_rate(case.expected_answer_points, "" if failed else record.answer),
        "citation_fidelity": fidelity,
        "three_state_accuracy": calculate_three_state_accuracy(case.answerable, None if failed else record.outcome),
        "self_repair_gain": gain,
        "core_coverage_gain": coverage_gain,
        "harmful_repair": float((gain or 0) < 0 or (coverage_gain or 0) < 0) if repaired and not failed else None,
    }
