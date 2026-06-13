from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "please",
    "should",
    "summarize",
    "summarise",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
    "would",
    "you",
}

ZH_STOPWORDS = {
    "鐨?",
    "浜?",
    "鍚?",
    "鍛?",
    "璇?",
    "杩欑瘒",
    "璁烘枃",
    "鏈枃",
    "璇ユ枃",
    "杩欎釜",
    "閭ｄ釜",
    "濡備綍",
    "浠€涔?",
    "鎬荤粨",
    "璇存槑",
    "涓€涓?",
}

INTENT_RULES = {
    "summary": {
        "keywords": [
            "summary",
            "summarize",
            "summarise",
            "overview",
            "contribution",
            "contributions",
            "finding",
            "findings",
            "鏍稿績",
            "鎬荤粨",
            "璐＄尞",
            "姒傝堪",
            "瑕佺偣",
        ],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "method": {
        "keywords": [
            "method",
            "methods",
            "approach",
            "framework",
            "architecture",
            "model",
            "training",
            "implementation",
            "鏂规硶",
            "妯″瀷",
            "妗嗘灦",
            "鏋舵瀯",
            "璁粌",
            "瀹炵幇",
        ],
        "preferred_sections": ["method", "approach", "model", "architecture"],
    },
    "experiment": {
        "keywords": [
            "experiment",
            "experiments",
            "evaluation",
            "result",
            "results",
            "benchmark",
            "benchmarks",
            "metric",
            "metrics",
            "瀹為獙",
            "缁撴灉",
            "璇勪及",
            "鍩哄噯",
        ],
        "preferred_sections": ["experiment", "results", "evaluation", "ablation"],
    },
    "comparison": {
        "keywords": [
            "baseline",
            "baselines",
            "compare",
            "comparison",
            "ablation",
            "compared",
            "瀵规瘮",
            "姣旇緝",
            "鍩虹嚎",
            "娑堣瀺",
        ],
        "preferred_sections": ["experiment", "results", "ablation"],
    },
    "limitation": {
        "keywords": [
            "limitation",
            "limitations",
            "weakness",
            "future work",
            "failure",
            "灞€闄?",
            "闄愬埗",
            "涓嶈冻",
            "鏈潵宸ヤ綔",
        ],
        "preferred_sections": ["conclusion", "discussion", "limitations", "appendix"],
    },
    "definition": {
        "keywords": [
            "definition",
            "define",
            "what is",
            "problem setup",
            "formulation",
            "瀹氫箟",
            "姒傚康",
            "浠诲姟瀹氫箟",
            "闂璁惧畾",
        ],
        "preferred_sections": ["introduction", "background", "method"],
    },
    "dataset": {
        "keywords": [
            "dataset",
            "datasets",
            "corpus",
            "data",
            "training set",
            "娴嬭瘯闆?",
            "鏁版嵁闆?",
            "璇枡",
        ],
        "preferred_sections": ["experiment", "dataset", "data"],
    },
}

QUESTION_TYPE_RULES = {
    "method_flow": {
        "keywords": [
            "method",
            "methods",
            "approach",
            "framework",
            "workflow",
            "pipeline",
            "algorithm",
            "model",
            "architecture",
            "training",
            "inference",
            "娴佺▼",
            "鏂规硶",
            "妗嗘灦",
            "妯″瀷",
            "绠楁硶",
        ],
        "preferred_sections": ["method", "approach", "model", "architecture", "introduction"],
    },
    "experiment_setup": {
        "keywords": [
            "experiment",
            "experiments",
            "setup",
            "evaluation",
            "dataset",
            "benchmark",
            "baseline",
            "metric",
            "implementation",
            "瀹為獙",
            "璁剧疆",
            "鏁版嵁闆?",
            "鍩哄噯",
            "璇勪及",
        ],
        "preferred_sections": ["experiment", "evaluation", "dataset", "implementation"],
    },
    "results_analysis": {
        "keywords": [
            "result",
            "results",
            "performance",
            "ablation",
            "comparison",
            "baseline",
            "finding",
            "缁撴灉",
            "鎬ц兘",
            "娑堣瀺",
            "瀵规瘮",
        ],
        "preferred_sections": ["results", "experiment", "evaluation", "ablation"],
    },
    "contribution": {
        "keywords": [
            "contribution",
            "novelty",
            "main idea",
            "innovation",
            "key idea",
            "璐＄尞",
            "鍒涙柊",
            "鏍稿績鎬濇兂",
        ],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "limitation": {
        "keywords": [
            "limitation",
            "limitations",
            "weakness",
            "future work",
            "failure",
            "灞€闄?",
            "涓嶈冻",
            "鏈潵宸ヤ綔",
        ],
        "preferred_sections": ["discussion", "conclusion", "limitations", "appendix"],
    },
    "dataset": {
        "keywords": ["dataset", "datasets", "corpus", "benchmark", "data", "鏁版嵁闆?", "璇枡", "鍩哄噯"],
        "preferred_sections": ["dataset", "experiment", "data"],
    },
    "metric": {
        "keywords": ["metric", "metrics", "score", "formula", "objective", "鎸囨爣", "鍏紡", "璇勪环"],
        "preferred_sections": ["experiment", "method", "evaluation"],
    },
    "figure_table": {
        "keywords": ["figure", "fig.", "table", "chart", "diagram", "鍥?", "琛?", "鍥捐〃"],
        "preferred_sections": ["figure", "table", "results", "appendix"],
    },
    "summary": {
        "keywords": ["summary", "summarize", "overview", "main", "abstract", "鎬荤粨", "姒傝堪", "涓昏"],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "other": {
        "keywords": [],
        "preferred_sections": ["abstract", "introduction", "method", "results"],
    },
}

SECTION_TAG_RULES = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "method": ["method", "methods", "approach", "framework", "architecture", "model"],
    "experiment": ["experiment", "experiments", "evaluation", "results"],
    "ablation": ["ablation"],
    "conclusion": ["conclusion", "summary", "discussion", "future work"],
    "appendix": ["appendix"],
    "prompt": ["prompt", "instruction"],
    "figure": ["figure", "fig."],
    "table": ["table"],
    "references": ["references", "bibliography"],
}

NOISY_SECTION_TAGS = {"appendix", "prompt", "figure", "table", "references"}


class RetrievalRules:
    """集中维护检索规则与启发式判断，避免编排层继续持有大段常量和规则分支。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or ENHANCED_RETRIEVAL_CONFIG

    @staticmethod
    def normalize_query_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def tokenize_for_keyword_search(text: str) -> List[str]:
        lowered = text.lower()
        english_tokens = re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", lowered)
        chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
        return english_tokens + chinese_tokens

    def dedupe_terms(self, terms: List[str]) -> str:
        seen: List[str] = []
        for term in terms:
            normalized = self.normalize_query_text(str(term))
            if normalized and normalized not in seen:
                seen.append(normalized)
        return " ".join(seen)

    def dedupe_list(self, items: List[str]) -> List[str]:
        seen_normalized: List[str] = []
        unique: List[str] = []
        for item in items:
            raw = str(item).strip()
            normalized = self.normalize_query_text(raw)
            if normalized and normalized not in seen_normalized:
                seen_normalized.append(normalized)
                unique.append(raw)
        return unique

    def extract_query_keywords(self, tokens: List[str], limit: Optional[int] = None) -> List[str]:
        limit = int(limit or self.config["extract_query_keywords_limit"])
        filtered = [token for token in tokens if token not in EN_STOPWORDS and token not in ZH_STOPWORDS]
        if not filtered:
            filtered = [token for token in tokens if len(token) > 1]
        seen: List[str] = []
        for token in filtered:
            if token not in seen:
                seen.append(token)
        return seen[:limit]

    def extract_paper_terms_from_text(self, text: str, limit: Optional[int] = None) -> List[str]:
        limit = int(limit or self.config["extract_paper_terms_limit"])
        tokens = self.tokenize_for_keyword_search(text)
        filtered = [token for token in tokens if token not in EN_STOPWORDS and token not in ZH_STOPWORDS]
        seen: List[str] = []
        for token in filtered:
            if token not in seen:
                seen.append(token)
        return seen[:limit]

    def build_query_keywords(self, queries: List[str], limit: Optional[int] = None) -> List[str]:
        limit = int(limit or self.config["build_query_keywords_limit"])
        counts: Dict[str, int] = {}
        for query in queries:
            for token in self.tokenize_for_keyword_search(query):
                counts[token] = counts.get(token, 0) + 1
        return [token for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]]

    def build_query_term_details(self, queries: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "query": query,
                "keywords": list(dict.fromkeys(self.tokenize_for_keyword_search(query))),
                "keyword_count": len(list(dict.fromkeys(self.tokenize_for_keyword_search(query)))),
            }
            for query in queries
        ]

    @staticmethod
    def detect_language(user_query: str, tokens: List[str]) -> str:
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_query))
        has_latin = any(re.search(r"[a-zA-Z]", token) for token in tokens)
        if has_cjk and has_latin:
            return "mixed"
        if has_cjk:
            return "zh"
        if has_latin:
            return "en"
        return "unknown"

    def detect_intent_tags(self, normalized_query: str, tokens: List[str]) -> List[str]:
        query_text = " ".join(tokens + [normalized_query])
        detected: List[str] = []
        for intent, spec in INTENT_RULES.items():
            if any(keyword.lower() in query_text for keyword in spec["keywords"]):
                detected.append(intent)
        return detected

    def estimate_ambiguity(self, keywords: List[str], intent_tags: List[str], language: str, user_query: str) -> float:
        content_weight = min(1.0, len(keywords) / self.config["extract_query_keywords_limit"])
        intent_weight = min(1.0, len(intent_tags) / 3.0)
        length_weight = min(1.0, len(user_query.strip()) / 50.0)
        language_weight = self.config["language_weight_zh_mixed"] if language in {"zh", "mixed"} else 0.0
        specificity = min(
            1.0,
            self.config["specificity_content_weight"] * content_weight
            + self.config["specificity_intent_weight"] * intent_weight
            + self.config["specificity_length_weight"] * length_weight
            + language_weight,
        )
        return max(self.config["specificity_floor"], min(1.0, 1.0 - specificity))

    @staticmethod
    def legacy_intent_bucket(intent: str) -> str:
        aliases = {
            "contribution": "summary",
            "paper_overview": "summary",
            "method_flow": "method",
            "implementation_detail": "method",
            "definition": "method",
            "experiment_setup": "experiment",
            "result_analysis": "experiment",
            "results_analysis": "experiment",
            "comparison": "comparison",
            "dataset": "dataset",
            "limitation": "limitation",
            "figure_table": "figure_table",
            "other": "other",
            "summary": "summary",
            "method": "method",
            "experiment": "experiment",
        }
        normalized = str(intent or "other").strip().lower() or "other"
        return aliases.get(normalized, normalized)

    def preferred_section_tags(self, intent_tags: List[str]) -> List[str]:
        preferred: List[str] = []
        for intent in intent_tags:
            preferred.extend(INTENT_RULES.get(intent, {}).get("preferred_sections", []))
        return self.dedupe_terms(preferred).split()

    def preferred_sections_for_question_type(self, question_type: str, intent_tags: List[str]) -> List[str]:
        alias = {
            "method": "method_flow",
            "experiment": "experiment_setup",
            "result_analysis": "results_analysis",
            "comparison": "results_analysis",
            "definition": "summary",
            "implementation_detail": "method_flow",
            "paper_overview": "summary",
        }
        preferred = list(
            QUESTION_TYPE_RULES.get(alias.get(question_type, question_type), QUESTION_TYPE_RULES["other"]).get(
                "preferred_sections",
                [],
            )
        )
        preferred.extend(self.preferred_section_tags(intent_tags))
        return self.dedupe_list(preferred)[:6]

    def classify_question_type(self, normalized_query: str, intent_tags: List[str]) -> str:
        query_text = normalized_query.lower()
        for question_type, spec in QUESTION_TYPE_RULES.items():
            if question_type != "other" and any(keyword.lower() in query_text for keyword in spec.get("keywords", [])):
                return question_type
        if "summary" in intent_tags:
            return "summary"
        return "other"

    def query_type_terms(self, question_type: str) -> List[str]:
        alias = {
            "paper_overview": "summary",
            "method": "method_flow",
            "experiment": "experiment_setup",
            "result_analysis": "results_analysis",
            "comparison": "results_analysis",
            "definition": "summary",
            "implementation_detail": "method_flow",
        }
        spec = QUESTION_TYPE_RULES.get(alias.get(question_type, question_type), QUESTION_TYPE_RULES["other"])
        return self.dedupe_list([str(item).strip() for item in spec.get("keywords", []) if str(item).strip()])

    def preferred_section_tags_from_plan(self, query_plan: Dict[str, Any], intent_tags: List[str]) -> List[str]:
        preferred = [str(item).strip() for item in (query_plan.get("preferred_sections", []) or []) if str(item).strip()]
        question_type = self.legacy_intent_bucket(query_plan.get("question_type", "other"))
        preferred.extend(QUESTION_TYPE_RULES.get(question_type, QUESTION_TYPE_RULES["other"]).get("preferred_sections", []))
        preferred.extend(self.preferred_section_tags(intent_tags))
        return self.dedupe_list(preferred)[:6]

    @staticmethod
    def compact_terms(terms: List[str], limit: int) -> List[str]:
        compacted: List[str] = []
        for term in terms:
            normalized = str(term).strip()
            if normalized and normalized not in compacted:
                compacted.append(normalized)
            if len(compacted) >= limit:
                break
        return compacted

    def summarize_intent(self, question_type: str, intent_tags: List[str], paper_terms: List[str]) -> str:
        summaries = {
            "paper_overview": "understand the paper overview and key ideas",
            "contribution": "understand the paper's main contribution and novelty",
            "method_flow": "understand the method flow and paper-specific implementation details",
            "experiment_setup": "understand the experimental setup, datasets, baselines, and evaluation details",
            "results_analysis": "understand the results, comparison, and ablation analysis",
            "limitation": "understand the limitations and future work",
            "dataset": "understand the dataset or benchmark used in the paper",
            "metric": "understand the metric, formula, or evaluation protocol",
            "figure_table": "find the relevant figure or table and interpret it",
            "summary": "summarize the paper around its main ideas and findings",
            "definition": "understand the definition or concept being asked about",
            "implementation_detail": "understand the implementation details and training settings",
        }
        normalized = {"method": "method_flow", "experiment": "experiment_setup", "comparison": "results_analysis"}.get(question_type, question_type)
        if normalized in summaries:
            return summaries[normalized]
        if intent_tags:
            return f"understand the paper with focus on {', '.join(intent_tags[:3])}"
        if paper_terms:
            return f"retrieve evidence around {', '.join(paper_terms[:3])}"
        return "retrieve the most relevant paper evidence"

    def intent_to_terms(self, intent_tags: List[str]) -> List[str]:
        mapping = {
            "summary": ["main", "contributions", "key", "findings", "summary"],
            "method": ["proposed", "method", "approach", "architecture", "implementation"],
            "experiment": ["experimental", "results", "evaluation", "benchmarks"],
            "comparison": ["baseline", "comparison", "ablation", "competing"],
            "limitation": ["limitations", "future", "work", "failure", "cases"],
            "definition": ["definition", "formulation", "problem", "setup"],
            "dataset": ["datasets", "corpus", "data", "splits"],
        }
        terms: List[str] = []
        for intent in intent_tags:
            terms.extend(mapping.get(intent, []))
        return self.dedupe_terms(terms).split()

    def intent_to_evidence_terms(self, intent_tags: List[str]) -> List[str]:
        mapping = {
            "summary": ["abstract", "introduction", "conclusion"],
            "method": ["method", "approach", "architecture", "model"],
            "experiment": ["experiment", "results", "evaluation", "ablation"],
            "comparison": ["baseline", "comparison", "ablation", "results"],
            "limitation": ["limitations", "discussion", "future work"],
            "definition": ["background", "definition", "problem setup"],
            "dataset": ["dataset", "corpus", "data"],
        }
        terms: List[str] = []
        for intent in intent_tags:
            terms.extend(mapping.get(intent, []))
        return self.dedupe_terms(terms).split()

    def build_semantic_query(self, user_query: str, keywords: List[str], intent_tags: List[str], intent_profile: Optional[Any] = None) -> str:
        parts: List[str] = []
        if intent_profile and getattr(intent_profile, "rewrite_focus", None):
            parts.extend(intent_profile.rewrite_focus[:4])
        if intent_tags:
            parts.extend(self.intent_to_terms(intent_tags))
        parts.extend(keywords[:6])
        if not parts:
            parts.extend(self.tokenize_for_keyword_search(user_query)[:6])
        return self.dedupe_terms(parts) or self.normalize_query_text(user_query)

    def build_evidence_query(self, keywords: List[str], intent_tags: List[str], language: str, intent_profile: Optional[Any] = None) -> str:
        parts = self.intent_to_evidence_terms(intent_tags)
        if intent_profile and getattr(intent_profile, "rerank_focus", None):
            parts.extend(intent_profile.rerank_focus[:4])
        parts.extend(keywords[:4])
        parts.extend(["璁烘枃", "璇佹嵁", "娈佃惤"] if language == "zh" else ["paper", "evidence", "passage"])
        return self.dedupe_terms(parts)

    def build_keyword_query(self, keywords: List[str], intent_tags: List[str], intent_profile: Optional[Any] = None) -> str:
        parts = keywords[: self.config["keyword_parts_limit"]] + self.intent_to_terms(intent_tags)
        if intent_profile and getattr(intent_profile, "rewrite_focus", None):
            parts.extend(intent_profile.rewrite_focus[:4])
        if not parts:
            parts = keywords[: self.config["keyword_parts_limit"]]
        return self.dedupe_terms(parts)

    def detect_section_tags(self, content: str) -> List[str]:
        prefix = self.normalize_query_text(content[:900])
        tags: List[str] = []
        for tag, patterns in SECTION_TAG_RULES.items():
            if any(pattern in prefix for pattern in patterns):
                tags.append(tag)
        return tags

    def query_similarity(self, left: str, right: str) -> float:
        left_tokens = set(self.tokenize_for_keyword_search(left))
        right_tokens = set(self.tokenize_for_keyword_search(right))
        if not left_tokens or not right_tokens:
            return 0.0
        intersection = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        return intersection / union if union else 0.0

    def route_confidence(
        self,
        route_name: str,
        query_profile: Any,
        source_query: str,
        route_queries: Optional[List[str]] = None,
        intent_profile: Optional[Any] = None,
    ) -> float:
        route_queries = route_queries or [source_query]
        source_text = " ".join(route_queries) if route_queries else source_query
        similarity = self.query_similarity(query_profile.normalized_query, source_text)
        ambiguity = query_profile.ambiguity_score
        main_intent = self.legacy_intent_bucket(intent_profile.main_intent if intent_profile else query_profile.question_type)
        if route_name == "vector_original":
            base = self.config["query_weight_base_summary_other"] if main_intent in {"summary", "other"} else self.config["query_weight_base_default"]
        elif route_name == "vector_rewrite":
            base = self.config["query_weight_base_ambiguous_keyword"] + 0.18 * ambiguity
            if main_intent in {"method", "experiment", "comparison", "dataset"}:
                base += self.config["route_focus_bonus"]
        elif route_name == "vector_hyde":
            base = self.config["query_weight_base_ambiguous_other"] + 0.25 * ambiguity
            if main_intent == "summary":
                base += self.config["route_summary_bonus"]
        elif route_name == "keyword":
            base = self.config["query_weight_base_keyword"] + 0.18 * min(
                1.0,
                len(query_profile.keywords) / self.config["extract_query_keywords_limit"],
            )
            if main_intent in {"method", "experiment", "figure_table"}:
                base += self.config["route_keyword_bonus"]
        elif route_name == "memory_context":
            base = 0.42 + 0.2 * ambiguity
            if main_intent in {"method", "experiment", "comparison", "figure_table", "dataset"}:
                base += 0.08
        else:
            base = self.config["query_weight_base_fallback"]
        return max(
            self.config["route_default_floor"],
            min(
                1.0,
                base * (self.config["route_confidence_multiplier"] + self.config["route_confidence_similarity_weight"] * similarity),
            ),
        )

    def compute_structural_bonus(self, chunk: Dict[str, Any], query_profile: Any) -> float:
        section_tags = set(chunk.get("section_tags", []) or [])
        if not section_tags:
            section_tags = set(self.detect_section_tags(str(chunk.get("content", "") or "")))
        preferred = set(query_profile.section_preferences)
        bonus = 0.0
        if preferred:
            bonus += self.config["section_bonus_weight"] * len(section_tags & preferred)
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        noisy_tags = set(section_tags & NOISY_SECTION_TAGS)
        main_intent = self.legacy_intent_bucket(query_profile.intent_profile.main_intent if query_profile.intent_profile else query_profile.question_type)
        if chunk_type in {"figure", "table"} and (main_intent == "figure_table" or "figure_table" in query_profile.intent_tags):
            noisy_tags -= {"figure", "table"}
            bonus += self.config["figure_table_bonus_weight"]
        if noisy_tags:
            bonus -= self.config["noisy_section_penalty_weight"] * len(noisy_tags)
        if "abstract" in section_tags and (main_intent == "summary" or "paper_overview" in query_profile.intent_tags or "contribution" in query_profile.intent_tags):
            bonus += self.config["preferred_section_bonus_weight"]
        if "conclusion" in section_tags and (main_intent == "summary" or "paper_overview" in query_profile.intent_tags or "contribution" in query_profile.intent_tags):
            bonus += self.config["section_path_bonus_weight"]
        return max(-0.05, min(0.12, bonus))

    def normalize_chunk(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(item.get("metadata", {}) or {})
        chunk = dict(item)
        chunk["content"] = item.get("content") or item.get("text") or metadata.get("content") or metadata.get("text") or ""
        chunk["text"] = chunk["content"]
        chunk["source"] = item.get("source") or metadata.get("source", "")
        chunk["document_name"] = item.get("document_name") or metadata.get("document_name", "")
        chunk["chunk_id"] = item.get("chunk_id") or metadata.get("chunk_id", 0)
        chunk["chunk_index"] = item.get("chunk_index") or metadata.get("chunk_index", 0)
        chunk["parent_chunk_id"] = item.get("parent_chunk_id") or metadata.get("parent_chunk_id", chunk["chunk_id"])
        chunk["original_chunk_id"] = item.get("original_chunk_id") or metadata.get("original_chunk_id", chunk["parent_chunk_id"])
        chunk["page_number"] = item.get("page_number") or metadata.get("page_number", "")
        chunk["page_start"] = item.get("page_start") or metadata.get("page_start")
        chunk["page_end"] = item.get("page_end") or metadata.get("page_end")
        chunk["page_range"] = item.get("page_range") or metadata.get("page_range", "")
        chunk["subchunk_index"] = item.get("subchunk_index") or metadata.get("subchunk_index", 0)
        chunk["subchunk_count"] = item.get("subchunk_count") or metadata.get("subchunk_count", 0)
        chunk["subchunk_label"] = item.get("subchunk_label") or metadata.get("subchunk_label", "")
        chunk["content_part_label"] = item.get("content_part_label") or metadata.get("content_part_label", "")
        chunk["section_path"] = item.get("section_path") or metadata.get("section_path", "")
        chunk["section_title"] = item.get("section_title") or metadata.get("section_title", "")
        chunk["title"] = item.get("title") or metadata.get("title", "")
        chunk["authors"] = item.get("authors") or metadata.get("authors", "")
        chunk["categories"] = item.get("categories") or metadata.get("categories", "")
        chunk["published_date"] = item.get("published_date") or metadata.get("published_date", "")
        chunk["url"] = item.get("url") or metadata.get("url", "")
        chunk["chunk_type"] = item.get("chunk_type") or metadata.get("chunk_type", "text")
        chunk["asset_kind"] = item.get("asset_kind") or metadata.get("asset_kind", "")
        chunk["asset_path"] = item.get("asset_path") or metadata.get("asset_path", "")
        chunk["asset_abs_path"] = item.get("asset_abs_path") or metadata.get("asset_abs_path", "")
        chunk["asset_summary"] = item.get("asset_summary") or metadata.get("asset_summary", "")
        chunk["asset_preview_text"] = item.get("asset_preview_text") or metadata.get("asset_preview_text", "")
        chunk["asset_caption"] = item.get("asset_caption") or metadata.get("asset_caption", "")
        chunk["asset_rows"] = item.get("asset_rows") or metadata.get("asset_rows", 0)
        chunk["asset_columns"] = item.get("asset_columns") or metadata.get("asset_columns", 0)
        chunk["order_index"] = item.get("order_index") or metadata.get("order_index", 0)
        # section tags 是后续结构加权、context expansion 和 trace 观察的共同输入，统一在这里补齐。
        chunk["section_tags"] = self.detect_section_tags(str(chunk["content"] or ""))
        return chunk
