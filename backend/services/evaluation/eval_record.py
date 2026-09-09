"""构造可重算的评测记录，并按日期落盘；观测写入失败不影响问答。"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.paper_evidence_research.contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult
from .contracts import EvaluationRecord

logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parents[2]
EVAL_RECORDS_DIR = BACKEND_DIR / "06-evaluation-result" / "records"


def _get_git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BACKEND_DIR,
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        logger.debug("获取评测版本失败", exc_info=True)
    return "unknown"


def _extract_main_intent(retrieval_debug: dict[str, Any] | None) -> str:
    profile = (retrieval_debug or {}).get("intent_profile")
    return str(profile.get("main_intent") or "unknown").strip().lower() if isinstance(profile, dict) else "unknown"


def _build_citations_payload(result: PaperEvidenceResearchResult) -> list[dict[str, Any]]:
    # source_id 是图、答案标记和候选池共用的标识；不能读取不存在的 citation_id 属性。
    return [dict(c.model_dump(mode="json"), citation_id=c.source_id, chunk_id=c.source_id) for c in result.citations]


def _build_research_summary_payload(result: PaperEvidenceResearchResult) -> dict[str, Any]:
    return result.research_summary.model_dump(mode="json")


def build_eval_record(
    *, request: PaperEvidenceResearchRequest, result: PaperEvidenceResearchResult | None = None,
    turn_id: str | None = None, retrieval_debug: dict[str, Any] | None = None,
    trace_events: list[dict[str, Any]] | None = None, latency_ms: float = 0.0,
    llm_usage: dict[str, Any] | None = None, raw_question: str | None = None,
    main_intent: str | None = None, error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生产和评测共用构造器；缺少调用观测时写 null，不伪造零成本。"""
    events = list(trace_events or [])
    if main_intent is None:
        main_intent = _extract_main_intent(retrieval_debug)
        if main_intent == "unknown":
            main_intent = next((str(e["main_intent"]) for e in events if e.get("main_intent")), "unknown")
    summary = _build_research_summary_payload(result) if result else {}
    indexes = []
    for event in events:
        snapshot = event.get("index_snapshot")
        if isinstance(snapshot, dict) and snapshot not in indexes:
            indexes.append(snapshot)
    configuration = next((event["configuration"] for event in events
                          if event.get("event_type") == "research_started" and event.get("configuration")),
                         {"research_limits": request.limits.model_dump(mode="json"), "engine": {}})
    return EvaluationRecord(
        record_id=f"eval-{request.research_run_id}", timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=_get_git_commit_hash(), run_status="error" if error is not None else "success",
        user_id=request.user_id, session_id=request.session_id, turn_id=turn_id or "",
        paper_context={"arxiv_id": request.arxiv_id, "indexes": indexes}, configuration=configuration,
        query={"raw": request.original_question if raw_question is None else raw_question,
               "rewritten": request.original_question, "main_intent": main_intent},
        outcome=result.outcome if result and error is None else None,
        answer=result.answer if result else "",
        termination_reason=summary.get("termination_reason"),
        citations=_build_citations_payload(result) if result else [], research_summary=summary,
        efficiency={"latency_ms": round(latency_ms, 1), "llm_calls": None,
                    "input_tokens": None, "output_tokens": None, "total_tokens": None,
                    **(llm_usage or {}), "retrieval_count": summary.get("retrieval_count"),
                    "draft_attempt_count": summary.get("draft_attempt_count")},
        trace_ref=request.research_run_id, trace_events=events, error=error,
    ).model_dump(mode="json")


def write_eval_record(*, record: dict[str, Any] | None = None, **kwargs: Any) -> str | None:
    """构造和写盘均在容错边界内，避免评测异常破坏已经完成的问答。"""
    try:
        payload = EvaluationRecord.model_validate(record).model_dump(mode="json") if record is not None else build_eval_record(**kwargs)
        output_dir = EVAL_RECORDS_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_dir.mkdir(parents=True, exist_ok=True)
        # run_id 允许由调用方提供，只用于文件名，不能改变评测产物目录。
        safe_id = re.sub(r"[^0-9A-Za-z._-]+", "_", payload["trace_ref"]).strip(".") or "research"
        output_path = output_dir / f"{safe_id}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output_path)
    except Exception:
        logger.warning("评测记录写入失败，保留问答结果", exc_info=True)
        return None
