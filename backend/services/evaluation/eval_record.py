"""评测记录落盘：每轮问答自动记录评测数据到磁盘。

落盘路径：backend/06-evaluation-result/records/YYYY-MM-DD/{research_run_id}.json

记录字段（PRD Phase 3 D2）：
- record_id, timestamp, app_version (git commit hash)
- user_id, session_id, turn_id
- paper_context {arxiv_id}
- query {raw, rewritten, main_intent}
- outcome, termination_reason
- citations [{citation_id, chunk_id, claim_ids}]
- research_summary {...}
- efficiency {latency_ms, llm_calls, retrieval_count, draft_attempt_count}
- trace_ref (research_run_id for trace file lookup)

容错策略：写失败只告警，不影响问答主链路（与 _write_qa_trace 同策略）。
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 评测记录根目录（相对 backend/）
EVAL_RECORDS_DIR = Path(__file__).parent.parent.parent / "06-evaluation-result" / "records"


def _get_git_commit_hash() -> str:
    """获取当前 git commit hash 作为 app_version。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to get git commit hash: %s", exc)
    return "unknown"


def _extract_main_intent(retrieval_debug: Optional[Dict[str, Any]]) -> str:
    """从 retrieval_debug 提取 main_intent。"""
    if not retrieval_debug:
        return "other"
    intent_profile = retrieval_debug.get("intent_profile", {})
    if isinstance(intent_profile, dict):
        return str(intent_profile.get("main_intent", "other") or "other").strip().lower()
    return "other"


def _build_citations_payload(result: Any) -> List[Dict[str, Any]]:
    """从 PaperEvidenceResearchResult 提取 citations。"""
    citations = getattr(result, "citations", None) or []
    return [
        {
            "citation_id": str(getattr(citation, "citation_id", "")),
            "chunk_id": str(getattr(citation, "source_id", "")),
            "claim_ids": list(getattr(citation, "claim_ids", [])),
        }
        for citation in citations
    ]


def _build_research_summary_payload(result: Any) -> Dict[str, Any]:
    """从 PaperEvidenceResearchResult 提取 research_summary。"""
    summary = getattr(result, "research_summary", None)
    if not summary:
        return {}
    return {
        "retrieval_count": getattr(summary, "retrieval_count", 0),
        "draft_attempt_count": getattr(summary, "draft_attempt_count", 0),
        "evidence_pool_size": getattr(summary, "evidence_pool_size", 0),
        "supported_claim_count": getattr(summary, "supported_claim_count", 0),
        "citation_repair_count": getattr(summary, "citation_repair_count", 0),
        "unresolved_topics": list(getattr(summary, "unresolved_topics", [])),
    }


def write_eval_record(
    *,
    result: Any,  # PaperEvidenceResearchResult
    request: Any,  # PaperEvidenceResearchRequest
    turn_id: Optional[str] = None,
    retrieval_debug: Optional[Dict[str, Any]] = None,
    latency_ms: float = 0.0,
) -> Optional[str]:
    """记录一次问答的评测数据到磁盘。

    Args:
        result: PaperEvidenceResearchResult 研究结果
        request: PaperEvidenceResearchRequest 研究请求
        turn_id: QA turn ID（可选）
        retrieval_debug: 检索 debug 信息（用于提取 main_intent）
        latency_ms: 请求延迟（毫秒）

    Returns:
        写入的文件路径，失败返回 None
    """
    try:
        # 提取字段
        research_run_id = getattr(request, "research_run_id", "unknown")
        record_id = f"eval-{research_run_id}"
        timestamp = datetime.utcnow().isoformat() + "Z"
        app_version = _get_git_commit_hash()

        # 用户与会话
        user_id = getattr(request, "user_id", "unknown")
        session_id = getattr(request, "session_id", "unknown")

        # 论文上下文
        arxiv_id = getattr(request, "arxiv_id", "unknown")

        # 查询信息
        original_question = getattr(request, "original_question", "")
        main_intent = _extract_main_intent(retrieval_debug)

        # 结果状态
        outcome = getattr(result, "outcome", "unknown")
        termination_reason = getattr(result, "termination_reason", "UNKNOWN")

        # 引用与摘要
        citations = _build_citations_payload(result)
        research_summary = _build_research_summary_payload(result)

        # 效率指标（llm_calls 暂用占位符 0，Phase 4 补充）
        efficiency = {
            "latency_ms": round(latency_ms, 1),
            "llm_calls": 0,  # TODO: Phase 4 补充 GenerationService 计数器
            "retrieval_count": research_summary.get("retrieval_count", 0),
            "draft_attempt_count": research_summary.get("draft_attempt_count", 0),
        }

        # 构造记录
        record = {
            "record_id": record_id,
            "timestamp": timestamp,
            "app_version": app_version,
            "user_id": user_id,
            "session_id": session_id,
            "turn_id": turn_id or "",
            "paper_context": {"arxiv_id": arxiv_id},
            "query": {
                "raw": original_question,
                "rewritten": original_question,  # TODO: 暂无改写逻辑，后续补充
                "main_intent": main_intent,
            },
            "outcome": outcome,
            "termination_reason": termination_reason,
            "citations": citations,
            "research_summary": research_summary,
            "efficiency": efficiency,
            "trace_ref": research_run_id,
        }

        # 按日期分桶写入
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        output_dir = EVAL_RECORDS_DIR / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{research_run_id}.json"
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "Eval record written: record_id=%s arxiv_id=%s outcome=%s path=%s",
            record_id,
            arxiv_id,
            outcome,
            output_path,
        )
        return str(output_path)

    except Exception as exc:  # pragma: no cover
        # 写失败只告警，不影响主链路
        logger.warning(
            "Failed to write eval record: research_run_id=%s error=%s",
            getattr(request, "research_run_id", "unknown"),
            exc,
            exc_info=True,
        )
        return None
