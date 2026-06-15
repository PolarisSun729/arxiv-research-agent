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
    "的",
    "了",
    "和",
    "呢",
    "请",
    "这篇",
    "论文",
    "本文",
    "该文",
    "这个",
    "那个",
    "如何",
    "什么",
    "总结",
    "说明",
    "一个",
    "一下",
    "怎样",
    "是什么",
    "有哪些",
}

ZH_QUERY_EXPANSION = {
    "方法": ["method", "methods", "approach"],
    "流程": ["pipeline", "workflow", "procedure"],
    "方法流程": ["method", "approach", "pipeline", "framework", "procedure"],
    "框架": ["framework", "architecture"],
    "架构": ["architecture", "framework"],
    "模型": ["model"],
    "算法": ["algorithm", "method"],
    "训练": ["training"],
    "推理": ["inference"],
    "实验": ["experiment", "experiments", "evaluation"],
    "实验设置": ["experiment", "experimental", "setup", "evaluation", "implementation"],
    "设置": ["setup", "setting", "configuration"],
    "评估": ["evaluation", "benchmark", "metric"],
    "指标": ["metric", "metrics", "score"],
    "数据": ["data", "dataset", "datasets"],
    "数据集": ["dataset", "datasets", "corpus", "benchmark"],
    "基准": ["benchmark", "baseline"],
    "基线": ["baseline", "baselines"],
    "对比": ["comparison", "compare", "baseline"],
    "比较": ["comparison", "compare", "baseline"],
    "消融": ["ablation"],
    "结果": ["result", "results", "performance"],
    "性能": ["performance", "result", "results"],
    "贡献": ["contribution", "contributions", "novelty"],
    "主要贡献": ["contribution", "contributions", "novelty", "key", "idea"],
    "创新": ["innovation", "novelty", "contribution"],
    "局限": ["limitation", "limitations", "weakness"],
    "不足": ["limitation", "limitations", "weakness"],
    "未来工作": ["future", "work", "limitations"],
    "图": ["figure", "fig", "diagram"],
    "表": ["table"],
    "图表": ["figure", "table", "chart"],
}

EN_QUERY_EXPANSION = {
    "dataset": ["datasets", "data", "corpus", "benchmark"],
    "datasets": ["dataset", "data", "corpus", "benchmark"],
    "baseline": ["baselines", "comparison", "experiment"],
    "baselines": ["baseline", "comparison", "experiment"],
    "method": ["methods", "approach", "pipeline", "framework"],
    "methods": ["method", "approach", "pipeline", "framework"],
    "limitation": ["limitations", "weakness", "future", "work"],
    "limitations": ["limitation", "weakness", "future", "work"],
    "contribution": ["contributions", "novelty", "innovation"],
    "contributions": ["contribution", "novelty", "innovation"],
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
            "核心",
            "总结",
            "贡献",
            "概述",
            "要点",
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
            "方法",
            "模型",
            "框架",
            "架构",
            "训练",
            "实现",
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
            "实验",
            "结果",
            "评估",
            "基准",
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
            "对比",
            "比较",
            "基线",
            "消融",
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
            "局限",
            "限制",
            "不足",
            "未来工作",
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
            "定义",
            "概念",
            "任务定义",
            "问题设定",
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
            "测试集",
            "数据集",
            "语料",
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
            "流程",
            "方法",
            "框架",
            "模型",
            "算法",
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
            "实验",
            "设置",
            "数据集",
            "基准",
            "评估",
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
            "结果",
            "性能",
            "消融",
            "对比",
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
            "贡献",
            "创新",
            "核心思想",
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
            "局限",
            "不足",
            "未来工作",
        ],
        "preferred_sections": ["discussion", "conclusion", "limitations", "appendix"],
    },
    "dataset": {
        "keywords": ["dataset", "datasets", "corpus", "benchmark", "data", "数据集", "语料", "基准"],
        "preferred_sections": ["dataset", "experiment", "data"],
    },
    "metric": {
        "keywords": ["metric", "metrics", "score", "formula", "objective", "指标", "公式", "评价"],
        "preferred_sections": ["experiment", "method", "evaluation"],
    },
    "figure_table": {
        "keywords": ["figure", "fig.", "table", "chart", "diagram", "图", "表", "图表"],
        "preferred_sections": ["figure", "table", "results", "appendix"],
    },
    "summary": {
        "keywords": ["summary", "summarize", "overview", "main", "abstract", "总结", "概述", "主要"],
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

    @classmethod
    def tokenize_for_keyword_search(cls, text: str) -> List[str]:
        lowered = text.lower()
        english_tokens = re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", lowered)
        chinese_tokens: List[str] = []
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", lowered):
            chinese_tokens.extend(cls.split_chinese_keyword_segment(segment))
        return [token for token in english_tokens + chinese_tokens if cls.is_informative_keyword_token(token)]

    @staticmethod
    def split_chinese_keyword_segment(segment: str) -> List[str]:
        """轻量抽取论文 QA 常用中文短语，避免整句中文只形成一个不可命中的长 token。"""
        normalized = str(segment or "").strip()
        if not normalized:
            return []
        tokens: List[str] = []
        for phrase in sorted(ZH_QUERY_EXPANSION, key=len, reverse=True):
            if phrase in normalized and phrase not in tokens:
                tokens.append(phrase)
        # 已命中领域短语时不保留过长整句，防止“这篇论文的方法流程是怎样的”这类问句污染关键词。
        if normalized not in tokens and not (tokens and len(normalized) > 8):
            tokens.append(normalized)
        return tokens

    @staticmethod
    def is_informative_keyword_token(token: str) -> bool:
        """过滤 OCR/乱码/低信息量 token，避免噪声在 BM25 倒排表里被放大。"""
        normalized = str(token or "").strip().lower()
        if not normalized:
            return False
        if re.search(r"[\ufffd�]", normalized):
            return False
        if re.fullmatch(r"[_\-\d.]+", normalized):
            return False
        if len(normalized) <= 1:
            return False
        # 连续重复字符和符号占比过高通常来自 OCR 或版面解析污染，不适合作为关键词。
        if re.search(r"(.)\1{4,}", normalized):
            return False
        symbol_count = len(re.findall(r"[^a-z0-9\u4e00-\u9fff_\-]", normalized))
        if symbol_count / max(len(normalized), 1) > 0.3:
            return False
        return True

    def expand_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        """把中文论文 QA 表达映射到英文论文术语，补齐中文问题到英文正文之间的词项桥接。"""
        expanded: List[str] = []
        token_set = set(tokens)
        compact_query = "".join(token for token in tokens if re.search(r"[\u4e00-\u9fff]", token))
        for token in tokens:
            if token not in expanded:
                expanded.append(token)
            for synonym in EN_QUERY_EXPANSION.get(token, []):
                if self.is_informative_keyword_token(synonym) and synonym not in expanded:
                    expanded.append(synonym)
            for synonym in ZH_QUERY_EXPANSION.get(token, []):
                if self.is_informative_keyword_token(synonym) and synonym not in expanded:
                    expanded.append(synonym)
        # 正则分词会把连续中文问句作为一个片段；这里额外按子串触发常见问法扩展。
        for phrase, synonyms in ZH_QUERY_EXPANSION.items():
            if phrase in token_set or (compact_query and phrase in compact_query):
                for synonym in synonyms:
                    if self.is_informative_keyword_token(synonym) and synonym not in expanded:
                        expanded.append(synonym)
        return expanded

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
        filtered = [
            token
            for token in tokens
            if self.is_informative_keyword_token(token) and token not in EN_STOPWORDS and token not in ZH_STOPWORDS
        ]
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
        filtered = [
            token
            for token in tokens
            if self.is_informative_keyword_token(token) and token not in EN_STOPWORDS and token not in ZH_STOPWORDS
        ]
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
        elif route_name == "table_structured":
            base = self.config["query_weight_base_keyword"] + 0.22 * min(
                1.0,
                len(query_profile.keywords) / self.config["extract_query_keywords_limit"],
            )
            if main_intent in {"experiment", "comparison", "figure_table", "dataset"}:
                base += self.config["route_keyword_bonus"] + 0.06
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

        def first_present(key: str, default: Any = "") -> Any:
            value = item.get(key) if key in item else metadata.get(key, default)
            return default if value is None else value

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
        chunk["table_id"] = item.get("table_id") or metadata.get("table_id", "")
        chunk["asset_kind"] = item.get("asset_kind") or metadata.get("asset_kind", "")
        chunk["asset_path"] = item.get("asset_path") or metadata.get("asset_path", "")
        chunk["asset_abs_path"] = item.get("asset_abs_path") or metadata.get("asset_abs_path", "")
        chunk["asset_summary"] = item.get("asset_summary") or metadata.get("asset_summary", "")
        chunk["asset_preview_text"] = item.get("asset_preview_text") or metadata.get("asset_preview_text", "")
        chunk["asset_caption"] = item.get("asset_caption") or metadata.get("asset_caption", "")
        # asset-section 匹配字段用于解释章节锚点来源，不应被 section_title/path 的旧字段语义吞掉。
        chunk["asset_section_match_type"] = first_present("asset_section_match_type", "")
        chunk["asset_section_match_confidence"] = first_present("asset_section_match_confidence", 0.0)
        chunk["asset_section_match_reason"] = first_present("asset_section_match_reason", "")
        chunk["asset_section_match_is_heuristic"] = first_present("asset_section_match_is_heuristic", False)
        chunk["asset_section_match_allow_embedding"] = first_present("asset_section_match_allow_embedding", False)
        chunk["table_structured_text"] = item.get("table_structured_text") or metadata.get("table_structured_text", "")
        chunk["table_structured_evidence"] = item.get("table_structured_evidence") or metadata.get("table_structured_evidence", {})
        chunk["asset_rows"] = item.get("asset_rows") or metadata.get("asset_rows", 0)
        chunk["asset_columns"] = item.get("asset_columns") or metadata.get("asset_columns", 0)
        chunk["order_index"] = item.get("order_index") or metadata.get("order_index", 0)
        # section tags 是后续结构加权、context expansion 和 trace 观察的共同输入，统一在这里补齐。
        chunk["section_tags"] = self.detect_section_tags(str(chunk["content"] or ""))
        return chunk
