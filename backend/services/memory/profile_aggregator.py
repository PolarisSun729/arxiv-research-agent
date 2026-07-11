from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set

from services.memory.concept_normalizer import ConceptNormalizer, PROFILE_NORMALIZER_VERSION
from services.memory.research_profile_generator import ResearchProfileGenerator


PROFILE_AGGREGATOR_VERSION = "profile_aggregator_v1"


class ProfileAggregator:
    """把事件流和 evidence card 聚合成带证据权重的 generated profile 草稿。"""

    FIELD_LIMITS = {
        "positive_topics": 12,
        "negative_topics": 10,
        "recent_topics": 10,
        "preferred_categories": 12,
        "common_question_types": 12,
        "representative_papers": 12,
    }

    ACTION_WEIGHTS = {
        "liked": {"positive": 3.0, "recent": 1.7},
        "like": {"positive": 3.0, "recent": 1.7},
        "favorite": {"positive": 2.8, "recent": 1.5},
        "later": {"positive": 1.3, "recent": 0.7},
        "note_saved": {"positive": 3.2, "recent": 2.6},
        "qa_asked": {"positive": 1.8, "recent": 2.2},
        "read": {"positive": 0.25, "recent": 0.18},
        "disliked": {"negative": 2.4},
        "dislike": {"negative": 2.4},
        "not_interested": {"negative": 1.5},
    }

    STRONG_RECENT_ACTIONS = {"liked", "like", "favorite", "later", "note_saved", "qa_asked"}
    POSITIVE_CATEGORY_ACTIONS = {"liked", "like", "favorite", "later", "note_saved", "qa_asked"}
    PAPER_LEVEL_POSITIVE_ACTIONS = {"liked", "like", "favorite", "later"}
    PAPER_LEVEL_NEGATIVE_ACTIONS = {"disliked", "dislike", "not_interested"}
    EXPLICIT_PROFILE_SOURCES = {"manual_profile", "note_tag", "note_auxiliary_text"}

    def __init__(self, concept_normalizer: Optional[ConceptNormalizer] = None):
        self.concept_normalizer = concept_normalizer or ConceptNormalizer()

    def aggregate(
        self,
        evidence: Dict[str, Any],
        current_profile: Optional[Dict[str, Any]] = None,
        *,
        preserve_existing_topics: bool = True,
        preserve_existing_representative_papers: bool = True,
    ) -> Dict[str, Any]:
        current = dict(current_profile or {})
        positive_candidates: List[Dict[str, Any]] = []
        negative_candidates: List[Dict[str, Any]] = []
        recent_candidates: List[Dict[str, Any]] = []
        category_scores: Counter[str] = Counter()
        question_type_scores: Counter[str] = Counter()
        event_map = self._event_map(evidence.get("profile_events") or [])
        candidate_count_by_action: Counter[str] = Counter()
        behavior_gate = self._build_behavior_gating_state(evidence)

        if preserve_existing_topics:
            # 旧画像没有可靠来源，只能作为低权重 legacy 候选，正式重建默认会关闭这条路径。
            self._add_legacy_candidates(current, positive_candidates, negative_candidates, recent_candidates)
        self._merge_existing_preferences(
            current,
            category_scores,
            question_type_scores,
            preserve_representative_papers=preserve_existing_representative_papers,
        )

        for paper in evidence.get("liked_papers") or []:
            paper_id = self._paper_id(paper)
            allowed_signals = {"positive", "recent"} if self._allows_positive_behavior_paper(behavior_gate, paper_id) else {"recent"}
            if "positive" not in allowed_signals:
                self._record_behavior_gate_skip(behavior_gate, "positive", paper_id)
            self._add_paper_candidates(
                paper,
                action_type="liked",
                positive_candidates=positive_candidates,
                recent_candidates=recent_candidates,
                negative_candidates=negative_candidates,
                candidate_count_by_action=candidate_count_by_action,
                allowed_signals=allowed_signals,
            )
            if "positive" in allowed_signals:
                self._add_categories(category_scores, paper.get("categories"), self.ACTION_WEIGHTS["liked"]["positive"])

        for paper in evidence.get("disliked_papers") or []:
            paper_id = self._paper_id(paper)
            allowed_signals = {"negative"} if self._allows_negative_behavior_paper(behavior_gate, paper_id) else set()
            if "negative" not in allowed_signals:
                self._record_behavior_gate_skip(behavior_gate, "negative", paper_id)
            self._add_paper_candidates(
                paper,
                action_type="disliked",
                positive_candidates=positive_candidates,
                recent_candidates=recent_candidates,
                negative_candidates=negative_candidates,
                candidate_count_by_action=candidate_count_by_action,
                allowed_signals=allowed_signals,
            )

        for index, action in enumerate(evidence.get("recent_actions") or []):
            if not isinstance(action, dict):
                continue
            paper = action.get("paper") if isinstance(action.get("paper"), dict) else None
            if not paper:
                continue
            action_type = str(action.get("action_type") or "").strip().lower()
            paper_id = self._paper_id(paper) or str(action.get("arxiv_id") or "").strip()
            allowed_signals = self._allowed_signals_for_recent_action(behavior_gate, action_type, paper_id)
            if not allowed_signals:
                continue
            recency_boost = max(0.2, 1.0 - index * 0.08)
            self._add_paper_candidates(
                paper,
                action_type=action_type,
                positive_candidates=positive_candidates,
                recent_candidates=recent_candidates,
                negative_candidates=negative_candidates,
                candidate_count_by_action=candidate_count_by_action,
                recency_boost=recency_boost,
                source_event=action.get("_profile_event") if isinstance(action.get("_profile_event"), dict) else None,
                allowed_signals=allowed_signals,
            )
            if action_type in self.POSITIVE_CATEGORY_ACTIONS and "positive" in allowed_signals:
                self._add_categories(category_scores, paper.get("categories"), self.ACTION_WEIGHTS.get(action_type, {}).get("positive", 0.0))

        for note in evidence.get("notes") or []:
            if not isinstance(note, dict) or not note.get("include_in_profile"):
                continue
            self._add_note_candidates(note, positive_candidates, recent_candidates, candidate_count_by_action)
            self._add_categories(category_scores, note.get("categories"), 1.0)
            note_type = str(note.get("note_type") or "").strip().lower()
            if note_type:
                question_type_scores[note_type] += 2.0

        normalized_topics = self.concept_normalizer.normalize(
            positive_candidates=positive_candidates,
            negative_candidates=negative_candidates,
            recent_candidates=recent_candidates,
            positive_limit=self.FIELD_LIMITS["positive_topics"],
            negative_limit=self.FIELD_LIMITS["negative_topics"],
            recent_limit=self.FIELD_LIMITS["recent_topics"],
        )
        topic_evidence = self._build_topic_evidence(normalized_topics, event_map)
        self._enrich_canonical_topics(normalized_topics, topic_evidence)
        normalized_topics = self._apply_selection_rules(normalized_topics, topic_evidence)
        normalized_topics = self._apply_behavior_topic_gates(normalized_topics, topic_evidence, behavior_gate)
        representative_candidates = self._collect_representative_candidates(evidence, behavior_gate)
        representative_papers = self._select_representative_papers(normalized_topics, topic_evidence, current, representative_candidates)

        return {
            "positive_topics": normalized_topics["positive_topics"],
            "negative_topics": normalized_topics["negative_topics"],
            "recent_topics": normalized_topics["recent_topics"],
            "canonical_topics": normalized_topics["canonical_topics"],
            "canonical_negative_topics": normalized_topics["canonical_negative_topics"],
            "canonical_recent_topics": normalized_topics["canonical_recent_topics"],
            "normalizer_version": normalized_topics.get("normalizer_version", PROFILE_NORMALIZER_VERSION),
            "normalization_signature": normalized_topics.get("normalization_signature", ""),
            "aggregator_version": PROFILE_AGGREGATOR_VERSION,
            "topic_evidence": topic_evidence,
            "aggregation_report": {
                "aggregator_version": PROFILE_AGGREGATOR_VERSION,
                "positive_candidate_count": len(positive_candidates),
                "negative_candidate_count": len(negative_candidates),
                "recent_candidate_count": len(recent_candidates),
                "candidate_count_by_action": dict(candidate_count_by_action),
                "behavior_weight_schema": self.ACTION_WEIGHTS,
                "behavior_profile_gating": self._behavior_gating_report(behavior_gate),
            },
            "preferred_categories": self._rank_counter_values(category_scores, self.FIELD_LIMITS["preferred_categories"]),
            "preferred_answer_style": str(current.get("preferred_answer_style") or "").strip(),
            "common_question_types": self._rank_counter_values(question_type_scores, self.FIELD_LIMITS["common_question_types"]),
            "representative_papers": representative_papers,
        }

    def _add_legacy_candidates(
        self,
        current: Dict[str, Any],
        positive_candidates: List[Dict[str, Any]],
        negative_candidates: List[Dict[str, Any]],
        recent_candidates: List[Dict[str, Any]],
    ) -> None:
        for topic in ResearchProfileGenerator.normalize_system_topics(current.get("positive_topics"), limit=30):
            positive_candidates.append(self._text_candidate(topic, signal="positive", score=0.7, source="legacy_manual_candidate"))
        for topic in ResearchProfileGenerator.normalize_system_topics(current.get("negative_topics"), limit=30):
            negative_candidates.append(self._text_candidate(topic, signal="negative", score=0.7, source="legacy_negative_candidate"))
        for topic in ResearchProfileGenerator.normalize_system_topics(current.get("recent_topics"), limit=30):
            recent_candidates.append(self._text_candidate(topic, signal="recent", score=0.5, source="legacy_recent_candidate"))

    def _merge_existing_preferences(
        self,
        current: Dict[str, Any],
        category_scores: Counter[str],
        question_type_scores: Counter[str],
        *,
        preserve_representative_papers: bool,
    ) -> None:
        del preserve_representative_papers
        for category in ResearchProfileGenerator.normalize_preferred_categories(current.get("preferred_categories"), limit=30):
            category_scores[category] += 1.0
        for question_type in ResearchProfileGenerator.normalize_profile_list(current.get("common_question_types"), limit=30):
            question_type_scores[question_type] += 1.0

    def _add_paper_candidates(
        self,
        paper: Dict[str, Any],
        *,
        action_type: str,
        positive_candidates: List[Dict[str, Any]],
        negative_candidates: List[Dict[str, Any]],
        recent_candidates: List[Dict[str, Any]],
        candidate_count_by_action: Counter[str],
        recency_boost: float = 1.0,
        source_event: Optional[Dict[str, Any]] = None,
        allowed_signals: Optional[Set[str]] = None,
    ) -> None:
        weights = self.ACTION_WEIGHTS.get(action_type, {})
        allowed = allowed_signals if allowed_signals is not None else {"positive", "negative", "recent"}
        if not allowed:
            return
        concepts = self._extract_card_candidate_concepts(paper.get("evidence_card") if isinstance(paper.get("evidence_card"), dict) else {})
        for concept in concepts:
            confidence = float(concept.get("confidence") or 0.0)
            if confidence <= 0:
                continue
            if "positive" in allowed and weights.get("positive", 0.0) > 0:
                score = weights["positive"] * confidence
                positive_candidates.append(
                    self._concept_candidate(concept, paper, signal="positive", action_type=action_type, score=score, source_event=source_event)
                )
                candidate_count_by_action[action_type] += 1
            if "negative" in allowed and weights.get("negative", 0.0) > 0:
                score = weights["negative"] * confidence
                negative_candidates.append(
                    self._concept_candidate(concept, paper, signal="negative", action_type=action_type, score=score, source_event=source_event)
                )
                candidate_count_by_action[action_type] += 1
            if "recent" in allowed and weights.get("recent", 0.0) > 0 and action_type in self.STRONG_RECENT_ACTIONS | {"read"}:
                score = weights["recent"] * recency_boost * confidence
                recent_candidates.append(
                    self._concept_candidate(concept, paper, signal="recent", action_type=action_type, score=score, source_event=source_event)
                )

    def _add_note_candidates(
        self,
        note: Dict[str, Any],
        positive_candidates: List[Dict[str, Any]],
        recent_candidates: List[Dict[str, Any]],
        candidate_count_by_action: Counter[str],
    ) -> None:
        source_event = note.get("_profile_event") if isinstance(note.get("_profile_event"), dict) else None
        for topic in ResearchProfileGenerator.normalize_system_topics(note.get("tags"), limit=10):
            # 笔记标签是用户显式语义，权重高于标题/正文里的弱文本线索。
            positive_candidates.append(
                self._text_candidate(topic, signal="positive", score=3.2, source="note_tag", note=note, source_event=source_event, confidence=0.95)
            )
            recent_candidates.append(
                self._text_candidate(topic, signal="recent", score=2.6, source="note_tag", note=note, source_event=source_event, confidence=0.95)
            )
            candidate_count_by_action["note_saved"] += 1

        auxiliary_text = " ".join([str(note.get("title") or ""), str(note.get("content") or "")])
        for topic in ResearchProfileGenerator().extract_topics(auxiliary_text):
            positive_candidates.append(
                self._text_candidate(topic, signal="positive", score=0.8, source="note_auxiliary_text", note=note, source_event=source_event, confidence=0.55)
            )
            recent_candidates.append(
                self._text_candidate(topic, signal="recent", score=0.7, source="note_auxiliary_text", note=note, source_event=source_event, confidence=0.55)
            )

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
            concepts.append(
                {
                    "label": label,
                    "type": str(item.get("type") or "technical_concept").strip() or "technical_concept",
                    "confidence": float(item.get("confidence") or card.get("extraction_confidence") or 0.5),
                    "evidence_text": str(item.get("evidence_text") or "").strip(),
                    "source": str(item.get("source") or "paper_evidence_card").strip() or "paper_evidence_card",
                }
            )
        return concepts[:20]

    def _concept_candidate(
        self,
        concept: Dict[str, Any],
        paper: Dict[str, Any],
        *,
        signal: str,
        action_type: str,
        score: float,
        source_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = source_event or paper.get("_profile_event") if isinstance(paper, dict) else None
        event_id = str((event or {}).get("event_id") or "").strip()
        arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
        confidence = float(concept.get("confidence") or 0.0)
        return {
            "label": str(concept.get("label") or "").strip(),
            "type": str(concept.get("type") or "technical_concept").strip() or "technical_concept",
            "confidence": confidence,
            "weight": round(score / confidence, 4) if confidence else score,
            "score": round(score, 4),
            "source": str(concept.get("source") or "paper_evidence_card").strip() or "paper_evidence_card",
            "source_papers": [arxiv_id] if arxiv_id else [],
            "source_events": [event_id] if event_id else [],
            "evidence_text": str(concept.get("evidence_text") or "").strip(),
            "signal": signal,
            "source_action": action_type,
        }

    def _text_candidate(
        self,
        topic: str,
        *,
        signal: str,
        score: float,
        source: str,
        note: Optional[Dict[str, Any]] = None,
        source_event: Optional[Dict[str, Any]] = None,
        confidence: float = 0.65,
    ) -> Dict[str, Any]:
        event_id = str((source_event or {}).get("event_id") or "").strip()
        arxiv_id = str((note or {}).get("arxiv_id") or "").strip()
        return {
            "label": topic,
            "type": "manual_topic" if source == "note_tag" else "technical_concept",
            "confidence": confidence,
            "weight": round(score / confidence, 4) if confidence else score,
            "score": round(score, 4),
            "source": source,
            "source_papers": [arxiv_id] if arxiv_id else [],
            "source_events": [event_id] if event_id else [],
            "evidence_text": str((note or {}).get("title") or "").strip()[:200],
            "signal": signal,
        }

    def _build_topic_evidence(self, normalized_topics: Dict[str, Any], event_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        evidence: Dict[str, Dict[str, Any]] = {}
        bucket_map = {
            "canonical_topics": "positive_score",
            "canonical_negative_topics": "negative_score",
            "canonical_recent_topics": "recent_score",
        }
        for bucket_name, score_field in bucket_map.items():
            for topic in normalized_topics.get(bucket_name) or []:
                if not isinstance(topic, dict):
                    continue
                label = str(topic.get("label") or "").strip()
                if not label:
                    continue
                item = evidence.setdefault(label, self._empty_topic_evidence(label))
                item[score_field] = round(float(item.get(score_field) or 0.0) + float(topic.get("score") or 0.0), 4)
                item["source_papers"] = self._merge_unique([*item["source_papers"], *(topic.get("source_papers") or [])])
                item["source_events"] = self._merge_unique([*item["source_events"], *(topic.get("source_events") or [])])
                item["source_concepts"] = self._merge_source_concepts(item["source_concepts"], topic.get("source_concepts") or [])

        for item in evidence.values():
            for event_id in item["source_events"]:
                event = event_map.get(event_id) or {}
                action = str(event.get("event_type") or event.get("action_type") or "").strip().lower()
                if action and action not in item["source_actions"]:
                    item["source_actions"].append(action)
                note_id = str(event.get("note_id") or "").strip()
                if note_id and note_id not in item["source_notes"]:
                    item["source_notes"].append(note_id)
                if action == "qa_asked":
                    question_ref = str(event.get("session_id") or event.get("source_id") or event_id).strip()
                    if question_ref and question_ref not in item["source_questions"]:
                        item["source_questions"].append(question_ref)
            item["net_score"] = round(float(item["positive_score"]) - float(item["negative_score"]), 4)
            item["evidence_count"] = len(set(item["source_events"])) + len(set(item["source_papers"])) + len(item["source_concepts"])
            item["confidence"] = self._topic_confidence(item)
        return evidence

    def _enrich_canonical_topics(self, normalized_topics: Dict[str, Any], topic_evidence: Dict[str, Dict[str, Any]]) -> None:
        for bucket_name in ("canonical_topics", "canonical_negative_topics", "canonical_recent_topics"):
            for topic in normalized_topics.get(bucket_name) or []:
                label = str(topic.get("label") or "").strip()
                evidence = topic_evidence.get(label) or {}
                topic.update(
                    {
                        "positive_score": evidence.get("positive_score", 0.0),
                        "negative_score": evidence.get("negative_score", 0.0),
                        "recent_score": evidence.get("recent_score", 0.0),
                        "net_score": evidence.get("net_score", 0.0),
                        "evidence_count": evidence.get("evidence_count", 0),
                        "confidence": evidence.get("confidence", topic.get("merge_confidence", 0.0)),
                        "source_notes": evidence.get("source_notes", []),
                        "source_questions": evidence.get("source_questions", []),
                        "source_actions": evidence.get("source_actions", []),
                    }
                )

    def _apply_selection_rules(self, normalized_topics: Dict[str, Any], topic_evidence: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        # 选择规则集中在聚合器里，reviewer 只负责质量闸门，避免不同层重复定义业务权重。
        positive = [
            item
            for item in normalized_topics.get("canonical_topics") or []
            if bool(item.get("pinned"))
            or (topic_evidence.get(item.get("label"), {}).get("net_score", 0.0) >= 1.0 and topic_evidence.get(item.get("label"), {}).get("evidence_count", 0) >= 1)
            or topic_evidence.get(item.get("label"), {}).get("positive_score", 0.0) >= 2.5
        ]
        negative = [
            item
            for item in normalized_topics.get("canonical_negative_topics") or []
            if topic_evidence.get(item.get("label"), {}).get("negative_score", 0.0) >= 1.8
            and topic_evidence.get(item.get("label"), {}).get("negative_score", 0.0)
            > topic_evidence.get(item.get("label"), {}).get("positive_score", 0.0)
        ]
        recent = [
            item
            for item in normalized_topics.get("canonical_recent_topics") or []
            if topic_evidence.get(item.get("label"), {}).get("recent_score", 0.0) >= 0.6
            and topic_evidence.get(item.get("label"), {}).get("net_score", 0.0) >= 0.0
        ]
        normalized_topics["canonical_topics"] = positive[: self.FIELD_LIMITS["positive_topics"]]
        normalized_topics["canonical_negative_topics"] = negative[: self.FIELD_LIMITS["negative_topics"]]
        normalized_topics["canonical_recent_topics"] = recent[: self.FIELD_LIMITS["recent_topics"]]
        normalized_topics["positive_topics"] = [item["label"] for item in normalized_topics["canonical_topics"]]
        normalized_topics["negative_topics"] = [item["label"] for item in normalized_topics["canonical_negative_topics"]]
        normalized_topics["recent_topics"] = [item["label"] for item in normalized_topics["canonical_recent_topics"]]
        return normalized_topics

    def _apply_behavior_topic_gates(
        self,
        normalized_topics: Dict[str, Any],
        topic_evidence: Dict[str, Dict[str, Any]],
        behavior_gate: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not behavior_gate.get("enabled"):
            return normalized_topics

        def _keep_topic(topic: Dict[str, Any], field_name: str, min_source_papers: int) -> bool:
            if bool(topic.get("pinned")) or self._topic_has_explicit_profile_source(topic):
                return True
            label = str(topic.get("label") or "").strip()
            source_paper_count = len(set((topic_evidence.get(label) or {}).get("source_papers") or []))
            if source_paper_count >= min_source_papers:
                return True
            # 长期画像不是“有高分就收”，还要求跨论文重复出现；否则一篇误点论文会放大成方向偏好。
            behavior_gate["filtered_topics"].append(
                {
                    "topic": label,
                    "field": field_name,
                    "source_paper_count": source_paper_count,
                    "required_source_papers": min_source_papers,
                    "reason": "insufficient_stable_topic_sources",
                }
            )
            return False

        positive_min_sources = int(behavior_gate.get("positive_min_topic_source_papers") or 1)
        negative_min_sources = int(behavior_gate.get("negative_min_topic_source_papers") or 1)
        normalized_topics["canonical_topics"] = [
            topic
            for topic in normalized_topics.get("canonical_topics") or []
            if isinstance(topic, dict) and _keep_topic(topic, "positive_topics", positive_min_sources)
        ]
        normalized_topics["canonical_negative_topics"] = [
            topic
            for topic in normalized_topics.get("canonical_negative_topics") or []
            if isinstance(topic, dict) and _keep_topic(topic, "negative_topics", negative_min_sources)
        ]
        normalized_topics["positive_topics"] = [item["label"] for item in normalized_topics["canonical_topics"]]
        normalized_topics["negative_topics"] = [item["label"] for item in normalized_topics["canonical_negative_topics"]]
        return normalized_topics

    def _select_representative_papers(
        self,
        normalized_topics: Dict[str, Any],
        topic_evidence: Dict[str, Dict[str, Any]],
        current: Dict[str, Any],
        fallback_candidates: List[str],
    ) -> List[str]:
        paper_scores: Counter[str] = Counter()
        for topic in normalized_topics.get("canonical_topics") or []:
            label = str(topic.get("label") or "").strip()
            evidence = topic_evidence.get(label) or {}
            topic_score = max(float(evidence.get("positive_score") or 0.0), float(topic.get("score") or 0.0))
            for paper_id in ResearchProfileGenerator.normalize_representative_papers(evidence.get("source_papers"), limit=20):
                paper_scores[paper_id] += topic_score
        ranked = [paper_id for paper_id, _score in sorted(paper_scores.items(), key=lambda item: (-item[1], item[0]))]
        for paper_id in fallback_candidates:
            if paper_id not in ranked:
                ranked.append(paper_id)
        for paper_id in ResearchProfileGenerator.normalize_representative_papers(current.get("representative_papers"), limit=20):
            if paper_id not in ranked:
                ranked.append(paper_id)
        return ranked[: self.FIELD_LIMITS["representative_papers"]]

    def _collect_representative_candidates(self, evidence: Dict[str, Any], behavior_gate: Optional[Dict[str, Any]] = None) -> List[str]:
        candidates: List[str] = []
        for paper in evidence.get("liked_papers") or []:
            if behavior_gate and not self._allows_positive_behavior_paper(behavior_gate, self._paper_id(paper)):
                continue
            candidates.extend(ResearchProfileGenerator.normalize_representative_papers([paper.get("arxiv_id")], limit=1))
        for action in evidence.get("recent_actions") or []:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action_type") or "").strip().lower()
            if action_type not in self.STRONG_RECENT_ACTIONS:
                continue
            paper = action.get("paper") if isinstance(action.get("paper"), dict) else {}
            paper_id = self._paper_id(paper) or str(action.get("arxiv_id") or "").strip()
            if behavior_gate and action_type in self.PAPER_LEVEL_POSITIVE_ACTIONS and not self._allows_positive_behavior_paper(behavior_gate, paper_id):
                continue
            candidates.extend(ResearchProfileGenerator.normalize_representative_papers([paper.get("arxiv_id") or action.get("arxiv_id")], limit=1))
        for note in evidence.get("notes") or []:
            if isinstance(note, dict) and note.get("include_in_profile"):
                candidates.extend(ResearchProfileGenerator.normalize_representative_papers([note.get("arxiv_id")], limit=1))
        return self._merge_unique(candidates)

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
            "filtered_topics": [],
        }

    def _allowed_signals_for_recent_action(self, behavior_gate: Dict[str, Any], action_type: str, paper_id: str) -> Set[str]:
        if action_type in self.PAPER_LEVEL_POSITIVE_ACTIONS:
            if self._allows_positive_behavior_paper(behavior_gate, paper_id):
                return {"positive", "recent"}
            self._record_behavior_gate_skip(behavior_gate, "positive", paper_id)
            return {"recent"}
        if action_type in self.PAPER_LEVEL_NEGATIVE_ACTIONS:
            if self._allows_negative_behavior_paper(behavior_gate, paper_id):
                return {"negative"}
            self._record_behavior_gate_skip(behavior_gate, "negative", paper_id)
            return set()
        if action_type in {"read", "qa_asked"}:
            return {"recent"}
        if action_type == "note_saved":
            return {"positive", "recent"}
        return {"positive", "negative", "recent"}

    @staticmethod
    def _allows_positive_behavior_paper(behavior_gate: Dict[str, Any], paper_id: str) -> bool:
        if not behavior_gate.get("enabled"):
            return True
        return bool(paper_id and paper_id in behavior_gate.get("stable_positive_paper_ids", set()))

    @staticmethod
    def _allows_negative_behavior_paper(behavior_gate: Dict[str, Any], paper_id: str) -> bool:
        if not behavior_gate.get("enabled"):
            return True
        return bool(paper_id and paper_id in behavior_gate.get("stable_negative_paper_ids", set()))

    @staticmethod
    def _record_behavior_gate_skip(behavior_gate: Dict[str, Any], signal: str, paper_id: str) -> None:
        if not behavior_gate.get("enabled") or not paper_id:
            return
        field = "skipped_positive_papers" if signal == "positive" else "skipped_negative_papers"
        skipped = behavior_gate.setdefault(field, [])
        if paper_id not in skipped:
            skipped.append(paper_id)

    @classmethod
    def _topic_has_explicit_profile_source(cls, topic: Dict[str, Any]) -> bool:
        sources = {
            str(concept.get("source") or "").strip()
            for concept in topic.get("source_concepts") or []
            if isinstance(concept, dict)
        }
        return bool(sources & cls.EXPLICIT_PROFILE_SOURCES)

    @staticmethod
    def _behavior_gating_report(behavior_gate: Dict[str, Any]) -> Dict[str, Any]:
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
            "filtered_topics": list(behavior_gate.get("filtered_topics") or []),
        }

    @staticmethod
    def _empty_topic_evidence(label: str) -> Dict[str, Any]:
        return {
            "label": label,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "recent_score": 0.0,
            "net_score": 0.0,
            "evidence_count": 0,
            "confidence": 0.0,
            "source_papers": [],
            "source_notes": [],
            "source_questions": [],
            "source_actions": [],
            "source_events": [],
            "source_concepts": [],
        }

    @staticmethod
    def _topic_confidence(item: Dict[str, Any]) -> float:
        base = min(1.0, max(float(item.get("positive_score") or 0.0), float(item.get("negative_score") or 0.0)) / 4.0)
        evidence_bonus = min(0.25, 0.05 * int(item.get("evidence_count") or 0))
        recent_bonus = min(0.1, float(item.get("recent_score") or 0.0) / 20.0)
        return round(min(1.0, base + evidence_bonus + recent_bonus), 4)

    @staticmethod
    def _event_map(events: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {str(event.get("event_id") or ""): event for event in events if isinstance(event, dict) and event.get("event_id")}

    @staticmethod
    def _merge_source_concepts(current: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_label = {str(item.get("label") or "").strip().lower(): dict(item) for item in current if isinstance(item, dict)}
        for item in incoming:
            if not isinstance(item, dict):
                continue
            key = str(item.get("label") or "").strip().lower()
            if not key:
                continue
            if key not in by_label:
                by_label[key] = dict(item)
                continue
            by_label[key]["score"] = round(float(by_label[key].get("score") or 0.0) + float(item.get("score") or 0.0), 4)
        return sorted(by_label.values(), key=lambda item: (-float(item.get("score") or 0.0), str(item.get("label") or "").lower()))

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
