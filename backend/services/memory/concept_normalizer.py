from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

PROFILE_NORMALIZER_VERSION = "profile_normalizer_v1"

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
    "toward",
    "using",
    "via",
    "with",
}

CANONICAL_TOPIC_RULES: List[Dict[str, Any]] = [
    {
        "label": "RAG retrieval optimization",
        "topic_type": "technical_concept",
        "description": "Retrieval-augmented generation, hybrid retrieval, reranking, and retrieval quality optimization.",
        "patterns": (
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "rag",
            "hybrid retrieval",
            "reranking",
            "rerank",
            "retrieval optimization",
        ),
    },
    {
        "label": "agent memory",
        "topic_type": "technical_concept",
        "description": "Long-term, episodic, session, or graph-structured memory for LLM agents.",
        "patterns": (
            "agent memory",
            "llm memory",
            "llm long-term memory",
            "memory-augmented agent",
            "memory augmented agent",
            "memory-augmented agents",
            "memory augmented agents",
            "long-term memory for agents",
            "long horizon agent memory",
            "long-horizon agent memory",
            "graph-structured session memory",
            "session memory",
            "episodic memory",
        ),
    },
    {
        "label": "long-context reasoning",
        "topic_type": "technical_concept",
        "description": "Reasoning over long contexts, extended context windows, and long-range dependencies.",
        "patterns": ("long-context", "long context", "context window", "long-range reasoning", "long horizon reasoning"),
    },
    {
        "label": "LLM reasoning",
        "topic_type": "technical_concept",
        "description": "Reasoning behavior and multi-step inference in large language models.",
        "patterns": ("llm reasoning", "language model reasoning", "chain-of-thought", "multi-step reasoning"),
    },
    {
        "label": "tool-using agents",
        "topic_type": "system_type",
        "description": "Agents that call tools, functions, APIs, or external systems during task execution.",
        "patterns": ("tool use", "tool-using", "tool using", "function calling", "api calling"),
    },
    {
        "label": "knowledge graph construction",
        "topic_type": "technical_concept",
        "description": "Entity extraction, relation extraction, and construction of knowledge graph evidence.",
        "patterns": ("knowledge graph", "entity extraction", "relation extraction", "kg construction", "graph construction"),
    },
    {
        "label": "multi-agent systems",
        "topic_type": "system_type",
        "description": "Collaboration, coordination, or communication among multiple agents.",
        "patterns": ("multi-agent", "multi agent", "agent collaboration", "agent coordination"),
    },
    {
        "label": "vision-language models",
        "topic_type": "technical_concept",
        "description": "Multimodal models that connect image, vision, and language representations.",
        "patterns": ("vision-language", "vision language", "multimodal", "image-text"),
    },
    {
        "label": "diffusion models",
        "topic_type": "technical_concept",
        "description": "Diffusion-based image, text-to-image, or visual generation models.",
        "patterns": ("diffusion model", "diffusion models", "text-to-image", "image generation"),
    },
    {
        "label": "code generation",
        "topic_type": "task",
        "description": "Program synthesis, code generation, code repair, and software-engineering LLM tasks.",
        "patterns": ("code generation", "program synthesis", "software engineering", "code repair"),
    },
    {
        "label": "benchmark evaluation",
        "topic_type": "evaluation_focus",
        "description": "Benchmarking, leaderboard comparison, ablation, and evaluation methodology.",
        "patterns": ("benchmark", "evaluation", "leaderboard", "ablation"),
    },
    {
        "label": "recommendation systems",
        "topic_type": "application_domain",
        "description": "Recommendation, ranking, personalization, and recommender systems.",
        "patterns": ("recommendation", "recommender", "personalization", "ranking"),
    },
    {
        "label": "alignment and preference learning",
        "topic_type": "technical_concept",
        "description": "Alignment, preference learning, reward modeling, and RLHF-style optimization.",
        "patterns": ("alignment", "preference learning", "rlhf", "reward model"),
    },
    {
        "label": "question answering",
        "topic_type": "task",
        "description": "Question answering, reading comprehension, and QA-oriented retrieval or generation.",
        "patterns": ("question answering", "qa", "reading comprehension"),
    },
]


class ConceptNormalizer:
    """将论文级概念归一化成用户画像可展示、可追溯的 canonical topic。

    归一化层只处理已经抽取好的候选概念，不直接读取 title/abstract；这样可以把证据理解、
    概念聚类、最终画像投影三个职责分开，后续更换 embedding 或 LLM 命名策略时不会影响事件层。
    """

    def __init__(self, generation_service: Optional[Any] = None, *, enable_llm_naming: bool = False):
        self.generation_service = generation_service
        self.enable_llm_naming = enable_llm_naming

    def cluster_signals(self, signals: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把正向、负向、近期信号放在同一次聚类里归并成 canonical topic。

        旧实现按正/负/近期三个桶分别聚类，同一概念在不同桶里可能得到不同 label，
        冲突只能事后修补；统一聚类后每个 topic 只有一份成员列表，立场由打分层根据成员极性决定。
        返回值里的 members 保留原始信号，打分层需要它们的极性、权重和时间戳。
        """
        candidates = [item for item in (self._normalize_signal(signal) for signal in signals or []) if item]
        topics: List[Dict[str, Any]] = []
        for cluster in self._cluster_candidates(candidates):
            topic = self._build_canonical_topic(cluster)
            if topic:
                topic["members"] = cluster
                topics.append(topic)
        return topics

    def _normalize_signal(self, signal: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(signal, dict):
            return None
        label = self.clean_label(signal.get("label"))
        if not label:
            return None
        confidence = self._coerce_float(signal.get("confidence"), default=0.55)
        return {
            **signal,
            "label": label,
            "type": str(signal.get("type") or "technical_concept").strip() or "technical_concept",
            "confidence": confidence,
            "score": self._coerce_float(signal.get("score"), default=confidence),
            "source": str(signal.get("source") or "unknown").strip() or "unknown",
            "source_papers": self._merge_unique(signal.get("source_papers") or []),
            "source_events": self._merge_unique(signal.get("source_events") or []),
            "evidence_text": str(signal.get("evidence_text") or "").strip()[:500],
        }

    LOOSE_CLUSTER_MIN_SIMILARITY = 0.55

    def _cluster_candidates(self, candidates: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        clusters_by_rule: Dict[str, List[Dict[str, Any]]] = {}
        groups_by_key: Dict[str, List[Dict[str, Any]]] = {}

        for candidate in candidates:
            rule = self._match_rule(candidate["label"])
            if rule:
                candidate = {**candidate, "canonical_label": rule["label"], "rule": rule}
                clusters_by_rule.setdefault(rule["label"], []).append(candidate)
                continue
            # 同一概念（大小写、单复数不同）先精确归组，保证它们一定落在同一个 topic。
            groups_by_key.setdefault(self._topic_key(candidate["label"]), []).append(candidate)

        # 支持论文多的概念先成为簇种子；后来的概念只和种子比较并归入最相似的簇。
        # 只比种子而不是比簇内任一成员，可以避免 A≈B、B≈C 把 A 与 C 串进同一簇。
        ordered_groups = sorted(
            groups_by_key.values(),
            key=lambda group: (-self._group_support(group), len(str(group[0].get("label") or "")), str(group[0].get("label") or "").lower()),
        )
        seeded_clusters: List[tuple[set[str], List[Dict[str, Any]]]] = []
        for group in ordered_groups:
            tokens = self._topic_tokens(group[0].get("label"))
            best_cluster: Optional[List[Dict[str, Any]]] = None
            best_similarity = 0.0
            for seed_tokens, cluster in seeded_clusters:
                similarity = self._token_similarity(tokens, seed_tokens)
                if similarity > best_similarity:
                    best_cluster, best_similarity = cluster, similarity
            if best_cluster is not None and best_similarity >= self.LOOSE_CLUSTER_MIN_SIMILARITY:
                best_cluster.extend(group)
            else:
                seeded_clusters.append((tokens, list(group)))

        return [*clusters_by_rule.values(), *[cluster for _, cluster in seeded_clusters]]

    @staticmethod
    def _group_support(group: List[Dict[str, Any]]) -> int:
        papers = {paper for item in group for paper in item.get("source_papers") or []}
        return len(papers) or len(group)

    def _build_canonical_topic(self, cluster: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cluster:
            return None
        max_confidence = max(float(item.get("confidence") or 0.0) for item in cluster)
        if len(cluster) == 1 and max_confidence < 0.5 and not cluster[0].get("pinned"):
            # 单篇、低置信度概念很容易是局部实验细节，先保守丢弃，避免画像碎片化。
            return None

        rule = next((item.get("rule") for item in cluster if isinstance(item.get("rule"), dict)), None)
        label = str((rule or {}).get("label") or "").strip()
        if not label:
            label = self._name_cluster_with_llm(cluster) or self._fallback_cluster_label(cluster)
        label = self.clean_label(label)
        if not label:
            return None

        source_concepts = self._source_concepts(cluster)
        source_papers = self._merge_unique(item for concept in source_concepts for item in concept.get("source_papers", []))
        source_events = self._merge_unique(item for concept in source_concepts for item in concept.get("source_events", []))
        aliases = self._merge_unique(
            [
                *[concept["label"] for concept in source_concepts if self._topic_key(concept["label"]) != self._topic_key(label)],
                *([label] if label and rule else []),
            ]
        )
        score = sum(float(item.get("score") or 0.0) for item in cluster)
        merge_confidence = min(1.0, max_confidence + (0.08 * max(0, len(cluster) - 1)))
        return {
            "label": label,
            "aliases": aliases,
            "description": str((rule or {}).get("description") or self._fallback_description(label, cluster)).strip(),
            "topic_type": str((rule or {}).get("topic_type") or self._dominant_type(cluster) or "technical_concept"),
            "merge_confidence": round(merge_confidence, 4),
            "score": round(score, 4),
            "source_concepts": source_concepts,
            "source_papers": source_papers,
            "source_events": source_events,
            "source": "rule" if rule else "semantic_cluster",
            "pinned": any(bool(item.get("pinned")) for item in cluster),
            "normalizer_version": PROFILE_NORMALIZER_VERSION,
        }

    def _name_cluster_with_llm(self, cluster: List[Dict[str, Any]]) -> str:
        if not self.enable_llm_naming or self.generation_service is None or len(cluster) < 2:
            return ""
        complete = getattr(self.generation_service, "complete_with_qwen", None)
        if not callable(complete):
            return ""
        labels = [str(item.get("label") or "").strip() for item in cluster if str(item.get("label") or "").strip()]
        prompt = (
            "你是研究画像 topic 命名器。请只输出严格 JSON，不要输出 Markdown。\n"
            "任务：为一组相近研究概念生成一个简短、自然、可展示、可用于推荐语义匹配的 canonical topic label。\n"
            "禁止：论文标题、完整句子、arXiv 分类、paper/method/model/framework 等过泛词。\n"
            'JSON schema: {"label": "string", "description": "string"}\n'
            f"concepts: {json.dumps(labels[:12], ensure_ascii=False)}\n"
        )
        try:
            raw = str(complete(prompt, task_type="profile_topic_normalization", enable_thinking=False) or "").strip()
            payload = self._parse_json_object(raw)
            return self.clean_label(payload.get("label"))
        except Exception as exc:
            logger.warning("Profile topic LLM naming failed, falling back to deterministic label: %s", exc)
            return ""

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"```$", "", raw).strip()
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return {}
            raw = match.group(0)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}

    def _match_rule(self, label: str) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_text(label)
        for rule in CANONICAL_TOPIC_RULES:
            for pattern in rule["patterns"]:
                # 必须按词边界匹配：子串匹配会让 "rag" 命中 coverage/storage/average 这类无关词。
                if self._rule_pattern_regex(pattern).search(normalized):
                    return rule
        return None

    @classmethod
    def _rule_pattern_regex(cls, pattern: str) -> "re.Pattern[str]":
        normalized_pattern = cls._normalize_text(pattern)
        return re.compile(rf"(?<![a-z0-9]){re.escape(normalized_pattern)}(?:s|es)?(?![a-z0-9])")

    @staticmethod
    def _token_similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    def _source_concepts(self, cluster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        concepts: Dict[str, Dict[str, Any]] = {}
        for item in cluster:
            key = self._topic_key(item.get("label"))
            if key not in concepts:
                concepts[key] = {
                    "label": item.get("label"),
                    "type": item.get("type"),
                    "confidence": float(item.get("confidence") or 0.0),
                    "weight": float(item.get("weight") or 0.0),
                    "score": float(item.get("score") or 0.0),
                    "source": item.get("source"),
                    "source_papers": [],
                    "source_events": [],
                    "evidence_texts": [],
                }
            concept = concepts[key]
            concept["confidence"] = max(float(concept.get("confidence") or 0.0), float(item.get("confidence") or 0.0))
            concept["score"] = round(float(concept.get("score") or 0.0) + float(item.get("score") or 0.0), 4)
            concept["source_papers"] = self._merge_unique([*concept.get("source_papers", []), *item.get("source_papers", [])])
            concept["source_events"] = self._merge_unique([*concept.get("source_events", []), *item.get("source_events", [])])
            evidence_text = str(item.get("evidence_text") or "").strip()
            if evidence_text and evidence_text not in concept["evidence_texts"]:
                concept["evidence_texts"].append(evidence_text)
        return sorted(concepts.values(), key=lambda item: (-float(item.get("score") or 0.0), item.get("label", "").lower()))

    @staticmethod
    def _dominant_type(cluster: List[Dict[str, Any]]) -> str:
        counter = Counter(str(item.get("type") or "technical_concept") for item in cluster)
        return counter.most_common(1)[0][0] if counter else "technical_concept"

    @staticmethod
    def _fallback_description(label: str, cluster: List[Dict[str, Any]]) -> str:
        aliases = [str(item.get("label") or "").strip() for item in cluster if str(item.get("label") or "").strip()]
        if len(aliases) <= 1:
            return label
        return f"Canonical topic merged from related concepts: {', '.join(aliases[:4])}."

    def _fallback_cluster_label(self, cluster: List[Dict[str, Any]]) -> str:
        """簇名取被最多论文支持的概念；单条信号置信度高不代表它能代表整个簇。"""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in cluster:
            groups.setdefault(self._topic_key(item.get("label")), []).append(item)
        ranked = sorted(
            groups.values(),
            key=lambda group: (-self._group_support(group), -max(float(item.get("score") or 0.0) for item in group), len(str(group[0].get("label") or ""))),
        )
        return str(ranked[0][0].get("label") or "").strip() if ranked else ""

    @classmethod
    def clean_label(cls, value: Any) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return ""
        if URL_PATTERN.search(text) or cls.looks_like_arxiv_id(text):
            return ""
        if cls.looks_like_arxiv_category(text) or text.lower().startswith("cs."):
            return ""
        if cls.looks_like_full_paper_title(text):
            return ""
        if text.lower() in LOW_INFORMATION_TOPIC_TERMS:
            return ""
        if cls._looks_like_unnatural_ngram(text):
            return ""
        return text

    @staticmethod
    def looks_like_arxiv_id(value: str) -> bool:
        text = str(value or "").strip()
        if text.startswith(("http://", "https://")):
            text = text.rstrip("/").rsplit("/", 1)[-1]
        return bool(text and ARXIV_ID_PATTERN.match(text))

    @staticmethod
    def looks_like_arxiv_category(value: str) -> bool:
        text = str(value or "").strip()
        return bool(text and ARXIV_CATEGORY_PATTERN.match(text) and ("." in text or text.startswith("cs.")))

    @staticmethod
    def looks_like_full_paper_title(value: str) -> bool:
        text = str(value or "").strip()
        words = [part for part in re.split(r"\s+", text) if part]
        if len(text) > 60 or len(words) > 6:
            return True
        title_joiners = {"for", "with", "of", "using", "via", "towards", "toward", "based"}
        lower_words = {word.strip(".,:;!?()[]{}").lower() for word in words}
        if len(words) >= 4 and lower_words & title_joiners:
            return True
        return any(marker in text for marker in (":", "?", "!", " -- ", " - "))

    @staticmethod
    def _looks_like_unnatural_ngram(value: str) -> bool:
        text = str(value or "").strip()
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        if len(tokens) < 2:
            return False
        if tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
            return True
        return len(tokens) >= 4 and sum(1 for token in tokens if token in STOPWORDS) >= 2

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").lower()
        text = text.replace("_", " ").replace("/", " ")
        text = re.sub(r"[-\u2010-\u2015]+", "-", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _topic_tokens(cls, value: Any) -> set[str]:
        tokens = {cls._singularize(token.lower()) for token in TOKEN_PATTERN.findall(str(value or ""))}
        return {token for token in tokens if token not in STOPWORDS and len(token) > 1}

    @staticmethod
    def _singularize(token: str) -> str:
        # 只做最保守的复数折叠，让 "large language models" 与 "large language model" 共用同一个 key。
        if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            return token[:-1]
        return token

    @classmethod
    def _topic_key(cls, value: Any) -> str:
        return " ".join(sorted(cls._topic_tokens(value))) or cls._normalize_text(value)

    @staticmethod
    def _coerce_float(value: Any, *, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(10.0, numeric))

    @staticmethod
    def _merge_unique(values: Iterable[Any]) -> List[str]:
        merged: List[str] = []
        for item in values or []:
            text = str(item or "").strip()
            if text and text not in merged:
                merged.append(text)
        return merged
