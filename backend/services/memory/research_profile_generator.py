from __future__ import annotations

import re
import json
from typing import Any, Iterable, List, Optional, Tuple

from services.memory.concept_normalizer import ConceptNormalizer


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
    """把用户行为证据归纳成稳定的研究画像字段。

    该生成器只接收已收集好的证据，不直接访问数据库，职责是把论文标题、
    摘要、分类、行为和笔记转换成可展示、可检索的短主题。
    """

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
