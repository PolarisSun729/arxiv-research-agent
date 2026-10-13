from __future__ import annotations

import math
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


TOPIC_SCORER_VERSION = "topic_scorer_v1"

# 用户显式表达的语义不需要再用跨时间重复出现来证明自己是长期兴趣。
EXPLICIT_TOPIC_SOURCES = {"manual_profile"}


@dataclass
class TopicScoringConfig:
    """画像打分与投影阈值；集中在这里，避免选择规则散落在聚合器和审查器两层。"""

    long_half_life_days: float = 120.0
    recent_half_life_days: float = 14.0
    # 正负一致性下限：同一概念在喜欢和不喜欢的论文里都常出现时一致性低，三个区都不进。
    # 0.5 相当于占优一方的证据权重至少是另一方的 3 倍。
    min_consistency: float = 0.5
    # 长期关注的持续性：正向证据最早与最晚时间至少相隔这么多天，单日集中点赞不会直接变成长期兴趣。
    focus_min_span_days: float = 7.0
    focus_min_papers: int = 1
    exclude_min_papers: int = 1
    recent_min_score: float = 0.6
    # 近期关注至少由窗口期内这么多篇论文支撑，避免一篇论文的专有概念占满近期区。
    recent_min_papers: int = 2
    recent_window_days: float = 30.0
    confidence_paper_scale: float = 3.0
    evidence_keep_limit: int = 40

    @classmethod
    def from_mapping(cls, values: Optional[Dict[str, Any]]) -> "TopicScoringConfig":
        config = cls()
        for field in fields(cls):
            raw = (values or {}).get(field.name)
            if raw is None:
                continue
            try:
                setattr(config, field.name, type(getattr(config, field.name))(raw))
            except (TypeError, ValueError):
                continue
        return config


class TopicScorer:
    """把 canonical topic 的成员信号合成一个带符号的分数，并投影成关注 / 近期 / 排除三个区。

    每个 topic 只有一行分数：长期区与排除区由 net 的符号决定，因此天然互斥；
    近期区是独立的短半衰期视角，可以与长期关注重叠，但不会与排除区重叠。
    """

    def __init__(self, config: Optional[TopicScoringConfig] = None):
        self.config = config or TopicScoringConfig()

    def score(self, topics: Iterable[Dict[str, Any]], *, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = now or datetime.now(timezone.utc)
        return [self._score_topic(topic, now) for topic in topics or [] if isinstance(topic, dict)]

    def project(self, rows: List[Dict[str, Any]], *, limits: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
        focus: List[Dict[str, Any]] = []
        exclude: List[Dict[str, Any]] = []
        recent: List[Dict[str, Any]] = []
        for row in rows:
            placement: List[str] = []
            reasons: List[str] = []
            if self._qualifies_focus(row, reasons):
                focus.append(row)
                placement.append("focus")
            elif self._qualifies_exclude(row, reasons):
                exclude.append(row)
                placement.append("exclude")
            if "exclude" not in placement and self._qualifies_recent(row, reasons):
                recent.append(row)
                placement.append("recent")
            row["placement"] = placement
            # 记录未进入其它区的原因（即使已进入近期区），用于解释「为什么没成为长期关注」。
            row["rejection_reasons"] = reasons

        focus.sort(key=lambda item: (-item["rank_score"], item["label"].lower()))
        exclude.sort(key=lambda item: (-item["rank_score"], item["label"].lower()))
        recent.sort(key=lambda item: (-item["recent_net"], item["label"].lower()))
        projected = {
            "focus": focus[: limits.get("positive_topics", 12)],
            "exclude": exclude[: limits.get("negative_topics", 10)],
            "recent": recent[: limits.get("recent_topics", 10)],
        }
        # 被截断的 topic 不再算作已投影，避免 topic_evidence 里的 placement 与最终列表不一致。
        kept = {
            "focus": {id(item) for item in projected["focus"]},
            "exclude": {id(item) for item in projected["exclude"]},
            "recent": {id(item) for item in projected["recent"]},
        }
        for row in rows:
            dropped = [zone for zone in row["placement"] if id(row) not in kept[zone]]
            if dropped:
                row["placement"] = [zone for zone in row["placement"] if zone not in dropped]
                row["rejection_reasons"] = [*row["rejection_reasons"], *[f"{zone}_limit_exceeded" for zone in dropped]]
        return projected

    def _score_topic(self, topic: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        long_pos = long_neg = recent_pos = recent_neg = 0.0
        positive_papers: set[str] = set()
        negative_papers: set[str] = set()
        positive_times: List[datetime] = []
        actions: List[str] = []
        sources: set[str] = set()
        events: List[str] = []
        questions: List[str] = []

        recent_papers: set[str] = set()
        members = [member for member in topic.get("members") or [] if isinstance(member, dict)]
        scored_ids = {id(member) for member in self._strongest_member_per_paper(members)}
        for member in members:
            polarity = 1 if float(member.get("polarity") or 0) >= 0 else -1
            # 同一篇论文的多个字段可能落进同一 topic；只让最强的一条计分，其余只贡献来源元数据。
            confidence = float(member.get("confidence") or 0.0) if id(member) in scored_ids else 0.0
            timestamp = member.get("timestamp") if isinstance(member.get("timestamp"), datetime) else None
            age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0) if timestamp else 0.0
            long_decay = self._decay(age_days, self.config.long_half_life_days)
            recent_decay = self._decay(age_days, self.config.recent_half_life_days)
            long_value = float(member.get("long_weight") or 0.0) * confidence * long_decay
            recent_value = float(member.get("recent_weight") or 0.0) * confidence * recent_decay
            paper_key = str(member.get("paper_id") or "").strip()
            eligible = bool(member.get("long_term_eligible", True))

            if polarity > 0:
                if eligible and long_value > 0:
                    long_pos += long_value
                    if paper_key:
                        positive_papers.add(paper_key)
                    if timestamp:
                        positive_times.append(timestamp)
                recent_pos += recent_value
                action_name = str(member.get("action") or "").strip().lower()
                if paper_key and recent_value > 0 and action_name != "read" and age_days <= self.config.recent_window_days:
                    recent_papers.add(paper_key)
            else:
                if eligible and long_value > 0:
                    long_neg += long_value
                    if paper_key:
                        negative_papers.add(paper_key)
                recent_neg += recent_value

            action = str(member.get("action") or "").strip().lower()
            if action and action not in actions:
                actions.append(action)
            source = str(member.get("source") or "").strip()
            if source:
                sources.add(source)
            for value, bucket in (
                (member.get("event_id"), events),
                (member.get("question_ref") if action == "qa_asked" else None, questions),
            ):
                text = str(value or "").strip()
                if text and text not in bucket:
                    bucket.append(text)

        total = long_pos + long_neg
        net = long_pos - long_neg
        consistency = abs(net) / total if total > 0 else 0.0
        dominant_papers = len(positive_papers) if net >= 0 else len(negative_papers)
        confidence = consistency * (1.0 - math.exp(-dominant_papers / max(self.config.confidence_paper_scale, 0.1)))
        span_days = (max(positive_times) - min(positive_times)).total_seconds() / 86400.0 if len(positive_times) >= 2 else 0.0
        return {
            "label": topic.get("label"),
            "topic": topic,
            "long_pos": round(long_pos, 4),
            "long_neg": round(long_neg, 4),
            "recent_pos": round(recent_pos, 4),
            "recent_neg": round(recent_neg, 4),
            "net_score": round(net, 4),
            "recent_net": round(recent_pos - recent_neg, 4),
            "consistency": round(consistency, 4),
            "confidence": round(confidence, 4),
            "rank_score": round(abs(net) * confidence, 4),
            "liked_papers": len(positive_papers),
            "disliked_papers": len(negative_papers),
            "positive_span_days": round(span_days, 2),
            "recent_papers": len(recent_papers),
            "explicit": bool(sources & EXPLICIT_TOPIC_SOURCES),
            "source_actions": actions,
            "source_events": events,
            "source_questions": questions,
        }

    def _qualifies_focus(self, row: Dict[str, Any], reasons: List[str]) -> bool:
        if row["net_score"] <= 0:
            reasons.append("net_not_positive")
            return False
        if row["consistency"] < self.config.min_consistency:
            reasons.append("mixed_positive_negative_evidence")
            return False
        if row["explicit"]:
            return True
        if row["liked_papers"] < self.config.focus_min_papers:
            reasons.append("insufficient_positive_papers")
            return False
        if row["positive_span_days"] < self.config.focus_min_span_days:
            reasons.append("not_persistent_enough")
            return False
        return True

    def _qualifies_exclude(self, row: Dict[str, Any], reasons: List[str]) -> bool:
        if row["net_score"] >= 0:
            return False
        if row["consistency"] < self.config.min_consistency:
            reasons.append("mixed_positive_negative_evidence")
            return False
        if row["disliked_papers"] < self.config.exclude_min_papers:
            reasons.append("insufficient_negative_papers")
            return False
        return True

    def _qualifies_recent(self, row: Dict[str, Any], reasons: List[str]) -> bool:
        if row["recent_net"] < self.config.recent_min_score:
            reasons.append("weak_recent_signal")
            return False
        if set(row["source_actions"]) <= {"read"}:
            reasons.append("read_only_recent_signal")
            return False
        if row["long_pos"] + row["long_neg"] > 0 and row["consistency"] < self.config.min_consistency:
            # 长期证据里正负混杂说明是过泛概念（如 large language models），近期再被问到也不代表新兴趣；
            # 没有长期证据的新话题不受这条限制。
            if "mixed_positive_negative_evidence" not in reasons:
                reasons.append("mixed_positive_negative_evidence")
            return False
        if not row["explicit"] and row["recent_papers"] < self.config.recent_min_papers:
            reasons.append("insufficient_recent_papers")
            return False
        return True

    @staticmethod
    def _strongest_member_per_paper(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 (论文, 行为, 极性) 去重，保留置信度最高的成员；没有论文标识的信号各自独立计分。"""
        best: Dict[Any, Dict[str, Any]] = {}
        for member in members:
            paper_key = str(member.get("paper_id") or "").strip()
            if not paper_key:
                best[id(member)] = member
                continue
            polarity = 1 if float(member.get("polarity") or 0) >= 0 else -1
            key = (paper_key, str(member.get("action") or "").strip().lower(), polarity)
            current = best.get(key)
            if current is None or float(member.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
                best[key] = member
        return list(best.values())

    @staticmethod
    def _decay(age_days: float, half_life_days: float) -> float:
        if half_life_days <= 0:
            return 1.0
        return 0.5 ** (age_days / half_life_days)
