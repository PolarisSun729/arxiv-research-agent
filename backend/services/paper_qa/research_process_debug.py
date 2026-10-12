"""把研究轨迹投影成问答面板可展示的执行过程。

研究引擎会多轮检索、起草和校验，单一的"改写→召回→融合→rerank"时间线描述不了它；
这里从轨迹事件中提取两份视图：

- ``research_rounds``：每轮检索一条，带触发它的决策理由和该轮的检索阶段摘要；
- ``research_steps``：按时间顺序的研究动作（检索、兜底起草、起草、校验、被拒动作、退回最佳草稿、结束）。

两份视图都是浅层的字符串与计数结构，随回答落库时的 debug 快照裁剪不会把它们截成形状标记。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def _need_label(need_id: Any) -> str:
    return str(need_id or "").strip() or "-"


def build_research_process_debug(trace_events: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rounds: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    proposed: Mapping[str, Any] = {}
    for event in trace_events or []:
        kind = event.get("event_type")
        if kind == "action_proposed":
            # 决策理由要挂到它触发的那一步上，单独列出会让面板把"提议"和"执行"重复展示。
            proposed = event
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
            reason = str(proposed.get("reason_code") or "") if proposed.get("action") == "search_paper" else ""
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
                "detail": f"新增 {new_count} 条候选，重复 {duplicate_count} 条" + (f" · 决策理由 {reason}" if reason else ""),
            })
        elif kind == "fallback_draft_scheduled":
            steps.append({
                "kind": "fallback",
                "title": "检索无进展，改为基于现有证据起草",
                "detail": str(event.get("reason_code") or ""),
            })
        elif kind == "draft_created":
            reason = str(proposed.get("reason_code") or "") if proposed.get("action") == "draft_answer" else ""
            steps.append({
                "kind": "draft",
                "title": f"起草第 {event.get('version') or '-'} 版答案",
                "detail": f"决策理由 {reason}" if reason else "",
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
            steps.append({"kind": "done", "title": "研究结束", "detail": str(event.get("outcome") or "")})
    return {"research_rounds": rounds, "research_steps": steps}
