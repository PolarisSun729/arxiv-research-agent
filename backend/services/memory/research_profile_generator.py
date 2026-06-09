from __future__ import annotations

import re
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services.memory.concept_normalizer import ConceptNormalizer, PROFILE_NORMALIZER_VERSION


CANONICAL_TOPIC_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("RAG retrieval optimization", ("retrieval-augmented generation", "retrieval augmented generation", "rag", "hybrid retrieval", "reranking", "rerank")),
    ("agent memory", ("agent memory", "memory-augmented agent", "memory augmented agent", "long-term memory", "episodic memory")),
    ("long-context reasoning", ("long-context", "long context", "context window", "long-range reasoning")),
    ("LLM reasoning", ("llm reasoning", "language model reasoning", "chain-of-thought", "multi-step reasoning", "reasoning")),
    ("tool-using agents", ("tool use", "tool-using", "tool using", "function calling", "api calling")),
    ("knowledge graph construction", ("knowledge graph", "entity extraction", "relation extraction", "graph construction")),
    ("multi-agent systems", ("multi-agent", "multi agent", "agent collaboration", "agent coordination")),
    ("vision-language models", ("vision-language", "vision language", "multimodal", "image-text")),
    ("diffusion models", ("diffusion model", "diffusion models", "text-to-image", "image generation")),
    ("code generation", ("code generation", "program synthesis", "software engineering", "code repair")),
    ("benchmark evaluation", ("benchmark", "evaluation", "leaderboard", "ablation")),
    ("recommendation systems", ("recommendation", "recommender", "personalization", "ranking")),
    ("alignment and preference learning", ("alignment", "preference learning", "rlhf", "reward model")),
    ("question answering", ("question answering", "qa", "reading comprehension")),
]

ARXIV_ID_PATTERN = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$")
ARXIV_CATEGORY_PATTERN = re.compile(r"^[a-z-]+(?:\.[A-Z]{2})?$")
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+-]*")

LOW_INFORMATION_TOPIC_TERMS = {
    "analysis",
    "approach",
    "framework",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "system",
    "systems",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "over",
    "paper",
    "study",
    "the",
    "this",
    "to",
    "towards",
    "using",
    "via",
    "with",
}

ANCHOR_TERMS = {
    "agent",
    "agents",
    "alignment",
    "benchmark",
    "code",
    "context",
    "diffusion",
    "evaluation",
    "generation",
    "graph",
    "knowledge",
    "language",
    "llm",
    "memory",
    "multimodal",
    "qa",
    "question",
    "rag",
    "reasoning",
    "recommendation",
    "rerank",
    "retrieval",
    "search",
    "tool",
}


class ResearchProfileGenerator:
    """鎶婄敤鎴疯涓鸿瘉鎹綊绾虫垚绋冲畾鐨勭爺绌剁敾鍍忓瓧娈点€?
    璇ョ敓鎴愬櫒鍙帴鏀跺凡鏀堕泦濂界殑璇佹嵁锛屼笉鐩存帴璁块棶鏁版嵁搴擄紝鑱岃矗鏄妸璁烘枃鏍囬銆?    鎽樿銆佸垎绫汇€佽涓哄拰绗旇杞崲鎴愬彲灞曠ず銆佸彲妫€绱㈢殑鐭富棰樸€?    """

    FIELD_LIMITS = {
        "positive_topics": 12,
        "negative_topics": 10,
        "recent_topics": 10,
        "preferred_categories": 12,
        "common_question_types": 12,
        "representative_papers": 12,
    }

    def __init__(self, concept_normalizer: Optional[ConceptNormalizer] = None):
        self.concept_normalizer = concept_normalizer or ConceptNormalizer()

    def generate(
        self,
        evidence: Dict[str, Any],
        current_profile: Optional[Dict[str, Any]] = None,
        *,
        preserve_existing_topics: bool = True,
        preserve_existing_representative_papers: bool = True,
    ) -> Dict[str, Any]:
        """根据当前事件和 evidence card 重新生成系统画像，并输出旧字段兼容投影。"""
        from services.memory.profile_aggregator import ProfileAggregator
        from services.memory.profile_reviewer import ProfileReviewer

        # 兼容旧调用入口：真实画像生成已经拆到 aggregator/reviewer，避免继续在本类里维护 Counter 主路径。
        draft = ProfileAggregator(concept_normalizer=self.concept_normalizer).aggregate(
            evidence=evidence,
            current_profile=current_profile,
            preserve_existing_topics=preserve_existing_topics,
            preserve_existing_representative_papers=preserve_existing_representative_papers,
        )
        return ProfileReviewer().review(draft)["revised_profile"]
        current = dict(current_profile or {})
        positive_scores: Counter[str] = Counter()
        negative_scores: Counter[str] = Counter()
        recent_scores: Counter[str] = Counter()
        category_scores: Counter[str] = Counter()
        question_type_scores: Counter[str] = Counter()
        representative_papers: List[str] = []
        positive_candidates: List[Dict[str, Any]] = []
        negative_candidates: List[Dict[str, Any]] = []
        recent_candidates: List[Dict[str, Any]] = []

        self._merge_existing_profile(
            current,
            positive_scores,
            negative_scores,
            recent_scores,
            category_scores,
            question_type_scores,
            representative_papers,
            preserve_topics=preserve_existing_topics,
            preserve_representative_papers=preserve_existing_representative_papers,
        )

        for paper in evidence.get("liked_papers") or []:
            self._add_paper_topics(positive_scores, paper, weight=3.0, candidate_bucket=positive_candidates, signal="positive")
            self._add_categories(category_scores, paper.get("categories"), weight=2.0)
            self._append_representative_paper(representative_papers, paper)

        for paper in evidence.get("disliked_papers") or []:
            self._add_paper_topics(negative_scores, paper, weight=2.0, candidate_bucket=negative_candidates, signal="negative")
            # 负向主题只来自具体论文语义，不把 arXiv 大类当作用户不感兴趣方向。

        for index, action in enumerate(evidence.get("recent_actions") or []):
            paper = action.get("paper") if isinstance(action, dict) else None
            if not isinstance(paper, dict):
                continue
            action_type = str(action.get("action_type") or "").strip().lower()
            recency_boost = max(0.2, 1.0 - index * 0.08)
            source_event = action.get("_profile_event") if isinstance(action.get("_profile_event"), dict) else None
            if action_type in {"like", "liked"}:
                self._add_paper_topics(recent_scores, paper, weight=1.5 * recency_boost, candidate_bucket=recent_candidates, signal="recent", source_event=source_event)
                self._append_representative_paper(representative_papers, paper)
            elif action_type == "favorite":
                self._add_paper_topics(recent_scores, paper, weight=1.3 * recency_boost, candidate_bucket=recent_candidates, signal="recent", source_event=source_event)
                self._append_representative_paper(representative_papers, paper)
            elif action_type == "note_saved":
                self._add_paper_topics(recent_scores, paper, weight=1.4 * recency_boost, candidate_bucket=recent_candidates, signal="recent", source_event=source_event)
                self._append_representative_paper(representative_papers, paper)
            elif action_type == "later":
                self._add_paper_topics(recent_scores, paper, weight=0.5 * recency_boost, candidate_bucket=recent_candidates, signal="recent", source_event=source_event)
            elif action_type == "read":
                # 普通阅读只是弱兴趣，不能和点赞、收藏、笔记保存同等放大。
                self._add_paper_topics(recent_scores, paper, weight=0.2 * recency_boost, candidate_bucket=recent_candidates, signal="recent", source_event=source_event)
            elif action_type == "qa_asked":
                self._add_paper_topics(recent_scores, paper, weight=0.35 * recency_boost, candidate_bucket=recent_candidates, signal="recent", source_event=source_event)
            elif action_type in {"not_interested", "dislike", "disliked"}:
                # 负向点击只作为谨慎的负向候选，不直接用高权重否定整个大方向。
                penalty_weight = 0.7 if action_type in {"dislike", "disliked"} else 0.45
                self._add_paper_topics(negative_scores, paper, weight=penalty_weight * recency_boost, candidate_bucket=negative_candidates, signal="negative", source_event=source_event)

        for note in evidence.get("notes") or []:
            if not isinstance(note, dict) or not note.get("include_in_profile"):
                continue
            self._add_note_topics(positive_scores, recent_scores, note, positive_candidates, recent_candidates)
            note_type = str(note.get("note_type") or "").strip().lower()
            if note_type:
                question_type_scores[note_type] += 2.0
            self._append_representative_paper(representative_papers, note)

        self._resolve_topic_conflicts(positive_scores, negative_scores)

        # 旧 positive_topics 等字段只作为兼容投影；内部主模型保留 canonical topic 对象和证据来源。
        normalized_topics = self.concept_normalizer.normalize(
            positive_candidates=positive_candidates or self._counter_candidates(positive_scores, "positive"),
            negative_candidates=negative_candidates or self._counter_candidates(negative_scores, "negative"),
            recent_candidates=recent_candidates or self._counter_candidates(recent_scores, "recent"),
            positive_limit=self.FIELD_LIMITS["positive_topics"],
            negative_limit=self.FIELD_LIMITS["negative_topics"],
            recent_limit=self.FIELD_LIMITS["recent_topics"],
        )

        return {
            "positive_topics": normalized_topics["positive_topics"],
            "negative_topics": normalized_topics["negative_topics"],
            "recent_topics": normalized_topics["recent_topics"],
            "canonical_topics": normalized_topics["canonical_topics"],
            "canonical_negative_topics": normalized_topics["canonical_negative_topics"],
            "canonical_recent_topics": normalized_topics["canonical_recent_topics"],
            "normalizer_version": normalized_topics.get("normalizer_version", PROFILE_NORMALIZER_VERSION),
            "normalization_signature": normalized_topics.get("normalization_signature", ""),
            "preferred_categories": self._rank_counter_values(category_scores, self.FIELD_LIMITS["preferred_categories"]),
            "preferred_answer_style": str(current.get("preferred_answer_style") or "").strip(),
            "common_question_types": self._rank_counter_values(question_type_scores, self.FIELD_LIMITS["common_question_types"]),
            "representative_papers": representative_papers[: self.FIELD_LIMITS["representative_papers"]],
        }

    def _merge_existing_profile(
        self,
        current: Dict[str, Any],
        positive_scores: Counter[str],
        negative_scores: Counter[str],
        recent_scores: Counter[str],
        category_scores: Counter[str],
        question_type_scores: Counter[str],
        representative_papers: List[str],
        *,
        preserve_topics: bool = True,
        preserve_representative_papers: bool = True,
    ) -> None:
        """把已有画像作为弱证据保留；新模型的正式重建默认不会走这条路径。"""
        if preserve_topics:
            # 旧库没有来源标记，只能把存量 topic 当作低权重手动候选保留，并先经过清洗。
            for topic in self.normalize_system_topics(current.get("positive_topics"), limit=30):
                positive_scores[topic] += 1.0
            for topic in self.normalize_system_topics(current.get("negative_topics"), limit=30):
                negative_scores[topic] += 1.0
            for topic in self.normalize_system_topics(current.get("recent_topics"), limit=30):
                recent_scores[topic] += 0.8
        for category in self.normalize_preferred_categories(current.get("preferred_categories"), limit=30):
            category_scores[category] += 1.0
        for question_type in self.normalize_profile_list(current.get("common_question_types"), limit=30):
            question_type_scores[question_type] += 1.0
        if preserve_representative_papers:
            for paper_id in self.normalize_representative_papers(current.get("representative_papers"), limit=30):
                if paper_id not in representative_papers:
                    representative_papers.append(paper_id)

    def _add_paper_topics(
        self,
        scores: Counter[str],
        paper: Dict[str, Any],
        weight: float,
        *,
        candidate_bucket: Optional[List[Dict[str, Any]]] = None,
        signal: str = "positive",
        source_event: Optional[Dict[str, Any]] = None,
    ) -> None:
        """只从 evidence card 的可泛化候选概念聚合主题，不再从 title/abstract 直接切词。"""
        card = paper.get("evidence_card") if isinstance(paper.get("evidence_card"), dict) else {}
        for concept in self._extract_card_candidate_concepts(card):
            confidence = float(concept.get("confidence") or 0.0)
            if confidence <= 0:
                continue
            score = weight * confidence
            scores[str(concept["label"])] += score
            if candidate_bucket is not None:
                candidate_bucket.append(
                    self._build_concept_candidate(
                        concept,
                        paper=paper,
                        weight=weight,
                        score=score,
                        signal=signal,
                        source_event=source_event,
                    )
                )

    def _extract_card_candidate_concepts(self, card: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(card, dict) or not card.get("schema_valid"):
            return []
        concepts: List[Dict[str, Any]] = []
        for item in card.get("candidate_concepts") or []:
            if not isinstance(item, dict) or not item.get("whether_generalizable", True):
                continue
            label = str(item.get("label") or "").strip()
            if not self.normalize_system_topics([label], limit=1):
                continue
            concepts.append(
                {
                    "label": label,
                    "type": str(item.get("type") or "technical_concept"),
                    "confidence": float(item.get("confidence") or card.get("extraction_confidence") or 0.5),
                    "evidence_text": str(item.get("evidence_text") or "").strip(),
                    "source": str(item.get("source") or "llm").strip() or "llm",
                }
            )
        return concepts[:20]

    def _add_note_topics(
        self,
        positive_scores: Counter[str],
        recent_scores: Counter[str],
        note: Dict[str, Any],
        positive_candidates: Optional[List[Dict[str, Any]]] = None,
        recent_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """融合笔记信号；tags 是高置信度用户信号，标题/内容只作为辅助弱证据。"""
        source_event = note.get("_profile_event") if isinstance(note.get("_profile_event"), dict) else None
        tag_topics = self.normalize_system_topics(note.get("tags"), limit=10)
        for topic in tag_topics:
            positive_scores[topic] += 3.0
            recent_scores[topic] += 2.5
            candidate = self._build_manual_text_candidate(
                topic,
                note=note,
                confidence=0.95,
                weight=3.0,
                score=3.0,
                source="note_tag",
                source_event=source_event,
            )
            if positive_candidates is not None:
                positive_candidates.append(candidate)
            if recent_candidates is not None:
                recent_candidates.append({**candidate, "weight": 2.5, "score": 2.5, "signal": "recent"})

        auxiliary_text = " ".join([str(note.get("title") or ""), str(note.get("content") or "")])
        for topic in self.extract_topics(auxiliary_text):
            positive_scores[topic] += 0.8
            recent_scores[topic] += 0.8
            candidate = self._build_manual_text_candidate(
                topic,
                note=note,
                confidence=0.55,
                weight=0.8,
                score=0.8,
                source="note_auxiliary_text",
                source_event=source_event,
            )
            if positive_candidates is not None:
                positive_candidates.append(candidate)
            if recent_candidates is not None:
                recent_candidates.append({**candidate, "signal": "recent"})

    def _build_concept_candidate(
        self,
        concept: Dict[str, Any],
        *,
        paper: Dict[str, Any],
        weight: float,
        score: float,
        signal: str,
        source_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = source_event or paper.get("_profile_event")
        event_id = str((event or {}).get("event_id") or "").strip()
        arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
        return {
            "label": str(concept.get("label") or "").strip(),
            "type": str(concept.get("type") or "technical_concept").strip() or "technical_concept",
            "confidence": float(concept.get("confidence") or 0.0),
            "weight": weight,
            "score": score,
            "source": str(concept.get("source") or "paper_evidence_card").strip() or "paper_evidence_card",
            "source_papers": [arxiv_id] if arxiv_id else [],
            "source_events": [event_id] if event_id else [],
            "evidence_text": str(concept.get("evidence_text") or "").strip(),
            "signal": signal,
        }

    def _build_manual_text_candidate(
        self,
        topic: str,
        *,
        note: Dict[str, Any],
        confidence: float,
        weight: float,
        score: float,
        source: str,
        source_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event_id = str((source_event or {}).get("event_id") or "").strip()
        arxiv_id = str(note.get("arxiv_id") or "").strip()
        return {
            "label": topic,
            "type": "manual_topic" if source == "note_tag" else "technical_concept",
            "confidence": confidence,
            "weight": weight,
            "score": score,
            "source": source,
            "source_papers": [arxiv_id] if arxiv_id else [],
            "source_events": [event_id] if event_id else [],
            "evidence_text": str(note.get("title") or "").strip()[:200],
            "signal": "positive",
        }

    def _counter_candidates(self, scores: Counter[str], signal: str) -> List[Dict[str, Any]]:
        """兼容少量旧调用：只有 score 字符串时，也先转成低来源信息的候选对象再归一化。"""
        return [
            {
                "label": topic,
                "type": "technical_concept",
                "confidence": 0.65,
                "weight": 1.0,
                "score": float(score),
                "source": "legacy_score_counter",
                "source_papers": [],
                "source_events": [],
                "signal": signal,
            }
            for topic, score in scores.items()
        ]

    def _add_categories(self, scores: Counter[str], values: Any, weight: float) -> None:
        for category in self.normalize_preferred_categories(values, limit=20):
            scores[category] += weight

    def _append_representative_paper(self, representative_papers: List[str], payload: Dict[str, Any]) -> None:
        for paper_id in self.normalize_representative_papers([payload.get("arxiv_id")], limit=1):
            if paper_id not in representative_papers:
                representative_papers.append(paper_id)

    @staticmethod
    def _paper_text(paper: Dict[str, Any]) -> str:
        return " ".join([str(paper.get("title") or ""), str(paper.get("abstract") or ""), str(paper.get("summary") or "")])

    def _extract_explicit_topic_terms(self, payload: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for field_name in ("topics", "topic", "keywords", "keyword", "tags", "labels"):
            terms.extend(self.normalize_profile_list(payload.get(field_name), limit=20))
        return terms

    def extract_topics(self, text: str, explicit_terms: Optional[Iterable[str]] = None) -> List[str]:
        """从辅助文本中提取弱主题，主路径仍以 evidence card 和显式 tags 为准。"""
        candidates: List[str] = []
        candidates.extend(self.normalize_system_topics(explicit_terms, limit=20))

        normalized_text = self._normalize_text(text)
        for label, patterns in CANONICAL_TOPIC_PATTERNS:
            if any(pattern in normalized_text for pattern in patterns):
                candidates.append(label)

        candidates.extend(self._extract_keyword_phrases(normalized_text))
        return self.normalize_system_topics(candidates, limit=10)

    def _extract_keyword_phrases(self, text: str) -> List[str]:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
        phrases: List[str] = []
        for size in (3, 2):
            for index in range(0, max(len(tokens) - size + 1, 0)):
                window = tokens[index : index + size]
                if not any(token in ANCHOR_TERMS for token in window):
                    continue
                phrase = " ".join(window)
                if phrase not in phrases:
                    phrases.append(phrase)
                if len(phrases) >= 8:
                    return phrases
        return phrases

    def _resolve_topic_conflicts(self, positive_scores: Counter[str], negative_scores: Counter[str]) -> None:
        """同一主题同时出现在正负两侧时，只保留证据更强的一侧。"""
        for topic in set(positive_scores) & set(negative_scores):
            if positive_scores[topic] >= negative_scores[topic]:
                negative_scores.pop(topic, None)
            else:
                positive_scores.pop(topic, None)

    def _rank_topics(self, scores: Counter[str], limit: int) -> List[str]:
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))
        return self.normalize_system_topics([topic for topic, _score in ranked], limit=limit)

    @staticmethod
    def _rank_counter_values(scores: Counter[str], limit: int) -> List[str]:
        return [value for value, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0].lower()))[:limit]]

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").lower()
        text = text.replace("_", " ").replace("/", " ")
        text = re.sub(r"[-\u2010-\u2015]+", "-", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def normalize_profile_list(values: Any, limit: int = 30) -> List[str]:
        if values is None:
            return []
        source = values if isinstance(values, list) else [values]
        normalized: List[str] = []
        for item in source:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def normalize_categories(values: Any, limit: int = 12) -> List[str]:
        if isinstance(values, str):
            stripped = values.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return ResearchProfileGenerator.normalize_categories(parsed, limit=limit)
            source = [item.strip() for item in re.split(r"[,\s]+", values) if item.strip()]
        elif isinstance(values, list):
            source = [str(item or "").strip() for item in values]
        else:
            source = [str(values or "").strip()] if values is not None else []
        return [item for item in ResearchProfileGenerator.normalize_profile_list(source, limit=limit) if item]

    @staticmethod
    def looks_like_arxiv_category(value: str) -> bool:
        text = str(value or "").strip()
        return bool(text and ARXIV_CATEGORY_PATTERN.match(text) and ("." in text or text.startswith("cs.")))

    @staticmethod
    def normalize_preferred_categories(values: Any, limit: int = 20) -> List[str]:
        categories: List[str] = []
        for item in ResearchProfileGenerator.normalize_categories(values, limit=limit * 2):
            if not ResearchProfileGenerator.looks_like_arxiv_category(item):
                continue
            if item not in categories:
                categories.append(item)
            if len(categories) >= limit:
                break
        return categories

    @staticmethod
    def looks_like_arxiv_id(value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        if text.startswith(("http://", "https://")):
            text = text.rstrip("/").rsplit("/", 1)[-1]
        return bool(ARXIV_ID_PATTERN.match(text))

    @staticmethod
    def normalize_representative_papers(values: Any, limit: int = 20) -> List[str]:
        normalized: List[str] = []
        for item in ResearchProfileGenerator.normalize_profile_list(values, limit=limit * 2):
            text = item.rstrip("/").rsplit("/", 1)[-1] if item.startswith(("http://", "https://")) else item
            if not (ResearchProfileGenerator.looks_like_arxiv_id(text) or (len(text) <= 80 and not URL_PATTERN.search(text))):
                continue
            if text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def looks_like_full_paper_title(value: str) -> bool:
        text = str(value or "").strip()
        words = [part for part in re.split(r"\s+", text) if part]
        if len(text) > 60 or len(words) > 6:
            return True
        lower_words = {word.strip(".,:;!?()[]{}").lower() for word in words}
        title_joiners = {"for", "with", "of", "using", "via", "towards", "toward", "based"}
        if len(words) >= 4 and lower_words & title_joiners:
            return True
        return any(marker in text for marker in (":", "?", "!", " -- ", " - "))

    @staticmethod
    def normalize_system_topics(values: Any, limit: int = 30) -> List[str]:
        """统一规整 topic 输出，避免混入标题、分类、URL 或低信息泛词。"""
        normalized: List[str] = []
        for item in ResearchProfileGenerator.normalize_profile_list(values, limit=limit * 3):
            text = str(item or "").strip()
            if not text:
                continue
            if URL_PATTERN.search(text) or ResearchProfileGenerator.looks_like_arxiv_id(text):
                continue
            if ResearchProfileGenerator.looks_like_arxiv_category(text) or text.lower().startswith("cs."):
                continue
            if ResearchProfileGenerator.looks_like_full_paper_title(text):
                continue
            if text.lower() in LOW_INFORMATION_TOPIC_TERMS:
                continue
            if text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized
