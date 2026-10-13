from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from services.memory.concept_normalizer import ConceptNormalizer, PROFILE_NORMALIZER_VERSION
from services.memory.research_profile_generator import ResearchProfileGenerator
from services.memory.topic_scorer import TOPIC_SCORER_VERSION, TopicScorer, TopicScoringConfig


PROFILE_AGGREGATOR_VERSION = "profile_aggregator_v2"


class ProfileAggregator:
    """把事件流和 evidence card 聚合成带证据权重的 generated profile 草稿。

    流程：证据展平为带极性和时间戳的信号 → 所有信号统一聚类成 canonical topic
    → TopicScorer 给每个 topic 一行带符号的分数 → 投影成关注 / 近期 / 排除三个区。
    """

    FIELD_LIMITS = {
        "positive_topics": 12,
        "negative_topics": 10,
        "recent_topics": 10,
        "preferred_categories": 12,
        "common_question_types": 12,
        "representative_papers": 12,
    }

    # long 进入长期分（120 天半衰期），recent 进入近期分（14 天半衰期）。
    ACTION_WEIGHTS = {
        "liked": {"long": 3.0, "recent": 1.7},
        "favorite": {"long": 2.8, "recent": 1.5},
        "later": {"long": 1.3, "recent": 0.7},
        "note_saved": {"long": 3.2, "recent": 2.6},
        "qa_asked": {"long": 0.0, "recent": 2.2},
        "read": {"long": 0.0, "recent": 0.18},
        "disliked": {"long": 2.4, "recent": 1.2},
        "not_interested": {"long": 1.5, "recent": 0.8},
    }
    ACTION_ALIASES = {"like": "liked", "dislike": "disliked"}

    # 证据卡里的粗粒度字段：candidate_concepts 往往细到只出现在一篇论文里，跨论文复现靠这些字段。
    # 置信度按字段的抽象程度递减；同一论文的多个字段落进同一 topic 时，打分层只取最强的一条。
    COARSE_CARD_FIELDS = (
        ("main_research_area", "main_research_area", 0.85),
        ("tasks", "task", 0.75),
        ("research_objects", "research_object", 0.7),
        ("application_domains", "application_domain", 0.6),
    )

    PAPER_LEVEL_POSITIVE_ACTIONS = {"liked", "favorite", "later"}
    PAPER_LEVEL_NEGATIVE_ACTIONS = {"disliked", "not_interested"}
    RECENT_ONLY_ACTIONS = {"read", "qa_asked"}
    STRONG_POSITIVE_ACTIONS = {"liked", "favorite", "later", "note_saved", "qa_asked"}

    def __init__(self, concept_normalizer: Optional[ConceptNormalizer] = None):
        self.concept_normalizer = concept_normalizer or ConceptNormalizer()

    def aggregate(
        self,
        evidence: Dict[str, Any],
        *,
        now: Optional[datetime] = None,
        scoring_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        behavior_gate = self._build_behavior_gating_state(evidence)
        category_scores: Counter[str] = Counter()
        question_type_scores: Counter[str] = Counter()

        records = self._collect_paper_records(evidence)
        signals: List[Dict[str, Any]] = []
        signal_count_by_action: Counter[str] = Counter()
        for record in records:
            record_signals = self._paper_record_signals(record, behavior_gate)
            signals.extend(record_signals)
            signal_count_by_action[record["action"]] += len(record_signals)
            if record_signals and record["action"] in self.STRONG_POSITIVE_ACTIONS - self.RECENT_ONLY_ACTIONS:
                if any(signal["long_term_eligible"] for signal in record_signals):
                    self._add_categories(category_scores, record["paper"].get("categories"), self.ACTION_WEIGHTS[record["action"]]["long"])

        for note in evidence.get("notes") or []:
            if not isinstance(note, dict) or not note.get("include_in_profile"):
                continue
            note_signals = self._note_signals(note)
            signals.extend(note_signals)
            signal_count_by_action["note_saved"] += len(note_signals)
            self._add_categories(category_scores, note.get("categories"), 1.0)
            note_type = str(note.get("note_type") or "").strip().lower()
            if note_type:
                question_type_scores[note_type] += 2.0

        config = TopicScoringConfig.from_mapping(scoring_config)
        if behavior_gate.get("enabled"):
            config.focus_min_papers = max(config.focus_min_papers, int(behavior_gate["positive_min_topic_source_papers"]))
            config.exclude_min_papers = max(config.exclude_min_papers, int(behavior_gate["negative_min_topic_source_papers"]))
        scorer = TopicScorer(config)
        topics = self.concept_normalizer.cluster_signals(signals)
        rows = scorer.score(topics, now=now)
        projected = scorer.project(rows, limits=self.FIELD_LIMITS)

        canonical = {zone: [self._canonical_entry(row) for row in projected[zone]] for zone in ("focus", "exclude", "recent")}
        topic_evidence = self._build_topic_evidence(rows, config.evidence_keep_limit)
        representative_papers = self._select_representative_papers(projected["focus"], self._collect_representative_candidates(records, evidence, behavior_gate))

        return {
            "positive_topics": [item["label"] for item in canonical["focus"]],
            "negative_topics": [item["label"] for item in canonical["exclude"]],
            "recent_topics": [item["label"] for item in canonical["recent"]],
            "canonical_topics": canonical["focus"],
            "canonical_negative_topics": canonical["exclude"],
            "canonical_recent_topics": canonical["recent"],
            "normalizer_version": PROFILE_NORMALIZER_VERSION,
            "aggregator_version": PROFILE_AGGREGATOR_VERSION,
            "topic_evidence": topic_evidence,
            "aggregation_report": {
                "aggregator_version": PROFILE_AGGREGATOR_VERSION,
                "scorer_version": TOPIC_SCORER_VERSION,
                "scored_at": now.isoformat(timespec="seconds"),
                "signal_count": len(signals),
                "topic_count": len(rows),
                "signal_count_by_action": dict(signal_count_by_action),
                "action_weights": self.ACTION_WEIGHTS,
                "scoring_config": vars(config),
                "behavior_profile_gating": self._behavior_gating_report(behavior_gate, rows),
            },
            "preferred_categories": self._rank_counter_values(category_scores, self.FIELD_LIMITS["preferred_categories"]),
            "preferred_answer_style": "",
            "common_question_types": self._rank_counter_values(question_type_scores, self.FIELD_LIMITS["common_question_types"]),
            "representative_papers": representative_papers,
        }

    # ---- 证据展平 ----

    def _collect_paper_records(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """按 (论文, 行为) 去重：同一次点赞会同时出现在 liked_papers 和 recent_actions，不能计两次。"""
        records: Dict[tuple, Dict[str, Any]] = {}

        def _add(paper: Any, action_type: Any, event: Optional[Dict[str, Any]], fallback_time: Any) -> None:
            if not isinstance(paper, dict):
                return
            action = self._normalize_action(action_type)
            paper_id = self._paper_id(paper)
            if not paper_id or action not in self.ACTION_WEIGHTS:
                return
            event = event if isinstance(event, dict) else (paper.get("_profile_event") if isinstance(paper.get("_profile_event"), dict) else None)
            timestamp = self.parse_profile_time((event or {}).get("created_at")) or self.parse_profile_time(fallback_time)
            key = (paper_id, action)
            existing = records.get(key)
            if existing:
                # 保留最新一次行为时间；证据卡以任一引用上已经回填的为准。
                if timestamp and (not existing["timestamp"] or timestamp > existing["timestamp"]):
                    existing["timestamp"] = timestamp
                    existing["event"] = event or existing["event"]
                if not self._card(existing["paper"]) and self._card(paper):
                    existing["paper"] = paper
                return
            records[key] = {"paper": paper, "paper_id": paper_id, "action": action, "event": event, "timestamp": timestamp}

        for paper in evidence.get("liked_papers") or []:
            if isinstance(paper, dict):
                _add(paper, "liked", None, paper.get("liked_at"))
        for paper in evidence.get("disliked_papers") or []:
            if isinstance(paper, dict):
                _add(paper, "disliked", None, paper.get("disliked_at"))
        for action in evidence.get("recent_actions") or []:
            if not isinstance(action, dict):
                continue
            paper = action.get("paper") if isinstance(action.get("paper"), dict) else None
            if paper is not None and not self._paper_id(paper) and action.get("arxiv_id"):
                paper = {**paper, "arxiv_id": action.get("arxiv_id")}
            _add(paper, action.get("action_type"), action.get("_profile_event"), action.get("updated_at") or action.get("created_at"))
        return list(records.values())

    def _paper_record_signals(self, record: Dict[str, Any], behavior_gate: Dict[str, Any]) -> List[Dict[str, Any]]:
        action = record["action"]
        paper_id = record["paper_id"]
        polarity = -1 if action in self.PAPER_LEVEL_NEGATIVE_ACTIONS else 1
        long_term_eligible = True
        if action in self.PAPER_LEVEL_POSITIVE_ACTIONS and not self._allows_behavior_paper(behavior_gate, "positive", paper_id):
            # 不在稳定兴趣簇里的点赞只算近期信号，避免一次误点放大成长期方向。
            long_term_eligible = False
            self._record_behavior_gate_skip(behavior_gate, "positive", paper_id)
        if action in self.PAPER_LEVEL_NEGATIVE_ACTIONS and not self._allows_behavior_paper(behavior_gate, "negative", paper_id):
            self._record_behavior_gate_skip(behavior_gate, "negative", paper_id)
            return []
        if action in self.RECENT_ONLY_ACTIONS:
            long_term_eligible = False

        weights = self.ACTION_WEIGHTS[action]
        event = record.get("event") or {}
        signals = []
        for concept in self._extract_card_candidate_concepts(self._card(record["paper"])):
            signals.append(
                self._signal(
                    concept,
                    polarity=polarity,
                    action=action,
                    weights=weights,
                    long_term_eligible=long_term_eligible,
                    timestamp=record["timestamp"],
                    paper_id=paper_id,
                    event=event,
                )
            )
        return signals

    def _note_signals(self, note: Dict[str, Any]) -> List[Dict[str, Any]]:
        event = note.get("_profile_event") if isinstance(note.get("_profile_event"), dict) else {}
        timestamp = (
            self.parse_profile_time(event.get("created_at"))
            or self.parse_profile_time(note.get("updated_at"))
            or self.parse_profile_time(note.get("created_at"))
        )
        paper_id = str(note.get("arxiv_id") or "").strip()
        note_id = str(note.get("note_id") or note.get("id") or "").strip()
        weights = self.ACTION_WEIGHTS["note_saved"]
        common = {
            "polarity": 1,
            "action": "note_saved",
            "weights": weights,
            "long_term_eligible": True,
            "timestamp": timestamp,
            "paper_id": paper_id,
            "event": event,
            "note_id": note_id,
        }
        signals = []
        for topic in ResearchProfileGenerator.normalize_system_topics(note.get("tags"), limit=10):
            # 笔记标签是用户显式语义，置信度高，且在打分层豁免持续性条件。
            concept = {"label": topic, "type": "manual_topic", "confidence": 0.95, "source": "note_tag", "evidence_text": str(note.get("title") or "")[:200]}
            signals.append(self._signal(concept, **common))
        paper = note.get("paper") if isinstance(note.get("paper"), dict) else {}
        for concept in self._extract_card_candidate_concepts(self._card(paper)):
            signals.append(self._signal(concept, **common))
        return signals

    def _signal(
        self,
        concept: Dict[str, Any],
        *,
        polarity: int,
        action: str,
        weights: Dict[str, float],
        long_term_eligible: bool,
        timestamp: Optional[datetime],
        paper_id: str,
        event: Dict[str, Any],
        note_id: str = "",
    ) -> Dict[str, Any]:
        confidence = float(concept.get("confidence") or 0.0)
        event_id = str(event.get("event_id") or "").strip()
        return {
            "label": str(concept.get("label") or "").strip(),
            "type": str(concept.get("type") or "technical_concept").strip() or "technical_concept",
            "confidence": confidence,
            "score": round(max(weights["long"], weights["recent"]) * confidence, 4),
            "source": str(concept.get("source") or "paper_evidence_card").strip() or "paper_evidence_card",
            "evidence_text": str(concept.get("evidence_text") or "").strip(),
            "source_papers": [paper_id] if paper_id else [],
            "source_events": [event_id] if event_id else [],
            "polarity": polarity,
            "action": action,
            "long_weight": weights["long"],
            "recent_weight": weights["recent"],
            "long_term_eligible": long_term_eligible,
            "timestamp": timestamp,
            "paper_id": paper_id,
            "event_id": event_id,
            "note_id": note_id or str(event.get("note_id") or "").strip(),
            "question_ref": str(event.get("session_id") or event.get("source_id") or event_id).strip(),
        }

    def _extract_card_candidate_concepts(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(card, dict) or not card.get("schema_valid"):
            return []
        concepts: List[Dict[str, Any]] = []
        for item in card.get("candidate_concepts") or []:
            if not isinstance(item, dict) or not item.get("whether_generalizable", True):
                continue
            label = str(item.get("label") or "").strip()
            if not ResearchProfileGenerator.normalize_system_topics([label], limit=1):
                continue
            confidence = float(item.get("confidence") or card.get("extraction_confidence") or 0.5)
            if confidence <= 0:
                continue
            concepts.append(
                {
                    "label": label,
                    "type": str(item.get("type") or "technical_concept").strip() or "technical_concept",
                    "confidence": confidence,
                    "evidence_text": str(item.get("evidence_text") or "").strip(),
                    "source": str(item.get("source") or "paper_evidence_card").strip() or "paper_evidence_card",
                }
            )
        concepts = concepts[:20]
        seen = {label.lower() for label in (concept["label"] for concept in concepts)}
        for field, concept_type, confidence in self.COARSE_CARD_FIELDS:
            values = card.get(field)
            for value in values if isinstance(values, list) else [values]:
                label = str(value or "").strip()
                if not label or label.lower() in seen or not ResearchProfileGenerator.normalize_system_topics([label], limit=1):
                    continue
                seen.add(label.lower())
                concepts.append({"label": label, "type": concept_type, "confidence": confidence, "evidence_text": "", "source": f"card_{field}"})
        return concepts

    # ---- 输出投影 ----

    @staticmethod
    def _score_fields(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: row[key]
            for key in (
                "long_pos",
                "long_neg",
                "recent_pos",
                "recent_neg",
                "net_score",
                "recent_net",
                "consistency",
                "confidence",
                "rank_score",
                "liked_papers",
                "disliked_papers",
                "positive_span_days",
                "recent_papers",
            )
        }

    def _canonical_entry(self, row: Dict[str, Any]) -> Dict[str, Any]:
        topic = row["topic"]
        return {
            "label": row["label"],
            "aliases": list(topic.get("aliases") or []),
            "description": topic.get("description") or "",
            "topic_type": topic.get("topic_type") or "technical_concept",
            "merge_confidence": topic.get("merge_confidence", 0.0),
            "score": row["rank_score"],
            "source_concepts": self._public_source_concepts(topic),
            "source_papers": list(topic.get("source_papers") or []),
            "source_events": list(topic.get("source_events") or []),
            "source": topic.get("source") or "",
            "pinned": False,
            "normalizer_version": PROFILE_NORMALIZER_VERSION,
            **self._score_fields(row),
        }

    def _build_topic_evidence(self, rows: List[Dict[str, Any]], keep_limit: int) -> Dict[str, Dict[str, Any]]:
        # 已投影的 topic 全部保留；未入选的按证据量保留前若干个，方便解释「为什么没进画像」。
        placed = [row for row in rows if row.get("placement")]
        unplaced = sorted(
            (row for row in rows if not row.get("placement")),
            key=lambda row: (-(row["long_pos"] + row["long_neg"] + row["recent_pos"]), str(row["label"]).lower()),
        )
        selected = [*placed, *unplaced[: max(0, keep_limit - len(placed))]]
        evidence: Dict[str, Dict[str, Any]] = {}
        for row in selected:
            topic = row["topic"]
            evidence[row["label"]] = {
                "label": row["label"],
                "provenance": "generated",
                "placement": list(row.get("placement") or []),
                "rejection_reasons": list(row.get("rejection_reasons") or []),
                "aliases": list(topic.get("aliases") or []),
                **self._score_fields(row),
                "source_papers": list(topic.get("source_papers") or []),
                "source_events": list(row.get("source_events") or []),
                "source_actions": list(row.get("source_actions") or []),
                "source_notes": list(row.get("source_notes") or []),
                "source_questions": list(row.get("source_questions") or []),
                "source_concepts": self._public_source_concepts(topic),
            }
        return evidence

    @staticmethod
    def _public_source_concepts(topic: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        return [
            {key: concept.get(key) for key in ("label", "type", "confidence", "score", "source", "source_papers", "evidence_texts")}
            for concept in (topic.get("source_concepts") or [])[:limit]
            if isinstance(concept, dict)
        ]

    def _select_representative_papers(self, focus_rows: List[Dict[str, Any]], fallback_candidates: List[str]) -> List[str]:
        paper_scores: Counter[str] = Counter()
        for row in focus_rows:
            for paper_id in ResearchProfileGenerator.normalize_representative_papers(row["topic"].get("source_papers"), limit=20):
                paper_scores[paper_id] += row["rank_score"]
        ranked = [paper_id for paper_id, _score in sorted(paper_scores.items(), key=lambda item: (-item[1], item[0]))]
        for paper_id in fallback_candidates:
            if paper_id not in ranked:
                ranked.append(paper_id)
        return ranked[: self.FIELD_LIMITS["representative_papers"]]

    def _collect_representative_candidates(
        self,
        records: List[Dict[str, Any]],
        evidence: Dict[str, Any],
        behavior_gate: Dict[str, Any],
    ) -> List[str]:
        candidates: List[str] = []
        for record in sorted(records, key=lambda item: item["timestamp"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
            if record["action"] not in self.STRONG_POSITIVE_ACTIONS:
                continue
            # 不在稳定兴趣簇里的点赞论文同样不能作为代表论文，否则一次误点会出现在画像首页。
            if record["action"] in self.PAPER_LEVEL_POSITIVE_ACTIONS and not self._allows_behavior_paper(behavior_gate, "positive", record["paper_id"]):
                continue
            candidates.extend(ResearchProfileGenerator.normalize_representative_papers([record["paper_id"]], limit=1))
        for note in evidence.get("notes") or []:
            if isinstance(note, dict) and note.get("include_in_profile"):
                candidates.extend(ResearchProfileGenerator.normalize_representative_papers([note.get("arxiv_id")], limit=1))
        return self._merge_unique(candidates)

    # ---- 行为闸门 ----

    def _build_behavior_gating_state(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        interest_model = evidence.get("interest_model") if isinstance(evidence.get("interest_model"), dict) else None
        gating = interest_model.get("profile_gating") if isinstance((interest_model or {}).get("profile_gating"), dict) else {}

        def _id_set(values: Any) -> Set[str]:
            return {str(item or "").strip() for item in values or [] if str(item or "").strip()}

        return {
            "enabled": interest_model is not None,
            "status": str((interest_model or {}).get("status") or "not_provided").strip() or "not_provided",
            "stable_positive_paper_ids": _id_set((interest_model or {}).get("stable_positive_paper_ids")),
            "weak_positive_paper_ids": _id_set((interest_model or {}).get("weak_positive_paper_ids")),
            "stable_negative_paper_ids": _id_set((interest_model or {}).get("stable_negative_paper_ids")),
            "weak_negative_paper_ids": _id_set((interest_model or {}).get("weak_negative_paper_ids")),
            "positive_min_topic_source_papers": self._coerce_positive_int(gating.get("positive_min_topic_source_papers"), default=1),
            "negative_min_topic_source_papers": self._coerce_positive_int(gating.get("negative_min_topic_source_papers"), default=1),
            "skipped_positive_papers": [],
            "skipped_negative_papers": [],
        }

    @staticmethod
    def _allows_behavior_paper(behavior_gate: Dict[str, Any], polarity: str, paper_id: str) -> bool:
        if not behavior_gate.get("enabled"):
            return True
        return bool(paper_id and paper_id in behavior_gate.get(f"stable_{polarity}_paper_ids", set()))

    @staticmethod
    def _record_behavior_gate_skip(behavior_gate: Dict[str, Any], polarity: str, paper_id: str) -> None:
        if not behavior_gate.get("enabled") or not paper_id:
            return
        skipped = behavior_gate.setdefault(f"skipped_{polarity}_papers", [])
        if paper_id not in skipped:
            skipped.append(paper_id)

    @staticmethod
    def _behavior_gating_report(behavior_gate: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        filtered_topics = [
            {"topic": row["label"], "reasons": row.get("rejection_reasons") or []}
            for row in rows
            if set(row.get("rejection_reasons") or []) & {"insufficient_positive_papers", "insufficient_negative_papers"}
        ]
        return {
            "enabled": bool(behavior_gate.get("enabled")),
            "status": behavior_gate.get("status"),
            "stable_positive_paper_count": len(behavior_gate.get("stable_positive_paper_ids") or []),
            "weak_positive_paper_count": len(behavior_gate.get("weak_positive_paper_ids") or []),
            "stable_negative_paper_count": len(behavior_gate.get("stable_negative_paper_ids") or []),
            "weak_negative_paper_count": len(behavior_gate.get("weak_negative_paper_ids") or []),
            "positive_min_topic_source_papers": behavior_gate.get("positive_min_topic_source_papers"),
            "negative_min_topic_source_papers": behavior_gate.get("negative_min_topic_source_papers"),
            "skipped_positive_papers": list(behavior_gate.get("skipped_positive_papers") or []),
            "skipped_negative_papers": list(behavior_gate.get("skipped_negative_papers") or []),
            "filtered_topics": filtered_topics,
        }

    # ---- 工具函数 ----

    @staticmethod
    def parse_profile_time(value: Any) -> Optional[datetime]:
        """解析 SQLite CURRENT_TIMESTAMP（UTC、无时区）或 ISO 时间；无法解析时返回 None。"""
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @classmethod
    def _normalize_action(cls, value: Any) -> str:
        action = str(value or "").strip().lower()
        return cls.ACTION_ALIASES.get(action, action)

    @staticmethod
    def _card(paper: Any) -> Dict[str, Any]:
        card = (paper or {}).get("evidence_card") if isinstance(paper, dict) else None
        return card if isinstance(card, dict) else {}

    @staticmethod
    def _merge_unique(values: Iterable[Any]) -> List[str]:
        merged: List[str] = []
        for item in values or []:
            text = str(item or "").strip()
            if text and text not in merged:
                merged.append(text)
        return merged

    @staticmethod
    def _paper_id(paper: Dict[str, Any]) -> str:
        return str((paper or {}).get("arxiv_id") or (paper or {}).get("id") or "").strip()

    @staticmethod
    def _coerce_positive_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(1, parsed)

    @staticmethod
    def _add_categories(scores: Counter[str], values: Any, weight: float) -> None:
        if weight <= 0:
            return
        for category in ResearchProfileGenerator.normalize_preferred_categories(values, limit=20):
            scores[category] += weight

    @staticmethod
    def _rank_counter_values(scores: Counter[str], limit: int) -> List[str]:
        return [value for value, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))[:limit]]
