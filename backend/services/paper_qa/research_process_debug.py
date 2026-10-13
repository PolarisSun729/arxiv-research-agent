"""把研究轨迹投影成问答面板可展示的执行过程。

研究引擎会多轮检索、起草和校验，单一的"改写→召回→融合→rerank"时间线描述不了它；
这里从轨迹事件中提取两份视图：

- ``research_rounds``：每轮检索一条，带触发它的决策理由和该轮的检索阶段摘要；
- ``research_steps``：按时间顺序的研究动作（检索、筛掉无关证据、兜底起草、起草、主张拆解、校验、被拒动作、退回最佳草稿、结束）；
- ``removed_claims``：部分回答里因缺少证据支持而从正文删掉的句子，带引用、校验结论与中文原因；
- ``answer_pruning``：部分回答的正文怎么来的（``in_place`` 原位删除 / ``joined_supported_claims`` 拼接受支持主张）。

两份视图都是浅层的字符串与计数结构，随回答落库时的 debug 快照裁剪不会把它们截成形状标记。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def _need_label(need_id: Any) -> str:
    return str(need_id or "").strip() or "-"


def _decision_reading(proposed: Mapping[str, Any]) -> str:
    """决策时读了多少条证据全文、几张图；旧轨迹没有这些字段时不显示。"""
    if "decision_evidence_count" not in proposed:
        return ""
    text = f"决策时阅读 {int(proposed.get('decision_evidence_count') or 0)} 条证据全文"
    image_count = int(proposed.get("decision_image_count") or 0)
    return text + (f"、{image_count} 张图" if image_count else "")


def _join_detail(*parts: str) -> str:
    return " · ".join(part for part in parts if part)


_REMOVED_CLAIM_FIELDS = (
    "claim_id", "text", "importance", "citation_ids", "verdict",
    "supporting_evidence_ids", "missing_facets", "reason_code", "reason",
)


def build_research_process_debug(trace_events: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rounds: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    removed_claims: List[Dict[str, Any]] = []
    answer_pruning: Any = None
    proposed: Mapping[str, Any] = {}
    for event in trace_events or []:
        kind = event.get("event_type")
        if kind == "action_proposed":
            # 决策理由要挂到它触发的那一步上，单独列出会让面板把"提议"和"执行"重复展示。
            proposed = event
        elif kind == "evidence_discarded":
            discarded = list(event.get("discarded_candidate_ids") or [])
            if discarded:
                steps.append({
                    "kind": "filter",
                    "title": f"筛掉 {len(discarded)} 条无关证据",
                    "detail": f"剩余 {int(event.get('remaining_candidate_count') or 0)} 条",
                })
        elif kind == "action_rejected":
            steps.append({
                "kind": "rejected",
                "title": f"动作被拒：{event.get('action') or 'unknown'}",
                "detail": str(event.get("rejection_code") or ""),
            })
        elif kind == "retrieval_completed":
            round_no = int(event.get("retrieval_count") or len(rounds) + 1)
            new_count = int(event.get("new_candidate_count") or 0)
            duplicate_count = int(event.get("duplicate_candidate_count") or 0)
            search_proposed = proposed if proposed.get("action") == "search_paper" else {}
            reason = str(search_proposed.get("reason_code") or "")
            rounds.append({
                "round": round_no,
                "target_need_id": _need_label(event.get("target_need_id")),
                "decision_reason": reason,
                "query": str(event.get("query") or ""),
                "status": str(event.get("status") or ""),
                "new_candidate_count": new_count,
                "duplicate_candidate_count": duplicate_count,
                **dict(event.get("stage_summary") or {}),
            })
            steps.append({
                "kind": "search",
                "title": f"第 {round_no} 轮检索（{_need_label(event.get('target_need_id'))}）",
                "detail": _join_detail(
                    f"新增 {new_count} 条候选，重复 {duplicate_count} 条",
                    f"决策理由 {reason}" if reason else "",
                    _decision_reading(search_proposed),
                ),
            })
        elif kind == "fallback_draft_scheduled":
            steps.append({
                "kind": "fallback",
                "title": "检索无进展，改为基于现有证据起草",
                "detail": str(event.get("reason_code") or ""),
            })
        elif kind == "draft_created":
            draft_proposed = proposed if proposed.get("action") == "draft_answer" else {}
            reason = str(draft_proposed.get("reason_code") or "")
            steps.append({
                "kind": "draft",
                "title": f"起草第 {event.get('version') or '-'} 版答案",
                "detail": _join_detail(f"决策理由 {reason}" if reason else "", _decision_reading(draft_proposed)),
            })
        elif kind == "claims_extracted":
            source = {"llm": "LLM 拆解", "rule": "规则兜底"}.get(str(event.get("extractor_source") or ""), "")
            steps.append({
                "kind": "extract",
                "title": f"拆解第 {event.get('draft_version') or '-'} 版草稿的主张",
                "detail": f"{int(event.get('claim_count') or 0)} 条主张" + (f" · {source}" if source else ""),
            })
        elif kind == "claim_verification_completed":
            steps.append({
                "kind": "verify",
                "title": f"校验第 {event.get('draft_version') or '-'} 版草稿",
                "detail": f"{int(event.get('supported_claim_count') or 0)}/{int(event.get('assessment_count') or 0)} 条主张有证据支持",
            })
        elif kind == "best_verified_draft_restored":
            steps.append({
                "kind": "restore",
                "title": f"退回校验结果最好的第 {event.get('draft_version') or '-'} 版草稿",
                "detail": f"{int(event.get('supported_claim_count') or 0)} 条主张有证据支持",
            })
        elif kind == "research_completed":
            removed_claims = [
                {field: item.get(field) for field in _REMOVED_CLAIM_FIELDS}
                for item in event.get("removed_claims") or []
                if isinstance(item, Mapping)
            ]
            answer_pruning = event.get("answer_pruning")
            detail = str(event.get("outcome") or "")
            if removed_claims:
                detail += f" · 移除 {len(removed_claims)} 条无证据支持的句子"
            steps.append({"kind": "done", "title": "研究结束", "detail": detail})
    return {
        "research_rounds": rounds,
        "research_steps": steps,
        "removed_claims": removed_claims,
        "answer_pruning": answer_pruning,
    }
