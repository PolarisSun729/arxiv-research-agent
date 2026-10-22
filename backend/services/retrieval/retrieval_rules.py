from __future__ import annotations

import re
import threading
from typing import Any, Dict, Iterable, List, Optional

import Stemmer

from services.intent.intent_service import EXPERIMENT_INTENTS, METHOD_INTENTS, OVERVIEW_INTENTS
from services.retrieval.contracts import legacy_body_index_id
from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

# 关键词抽取时过滤的英文停用词；包含 paper/summarize 等论文问答里几乎每句都有的词。
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

# PyStemmer 的 Stemmer 对象不是线程安全的，检索链路会在线程池里分词，每个线程各持一个。
_STEMMER_LOCAL = threading.local()


def stem_english_tokens(tokens: Iterable[str]) -> List[str]:
    """Snowball 英文词干化（cats → cat、datasets → dataset、distributions → distribut）。"""
    stemmer = getattr(_STEMMER_LOCAL, "stemmer", None)
    if stemmer is None:
        stemmer = _STEMMER_LOCAL.stemmer = Stemmer.Stemmer("english")
    return stemmer.stemWords(list(tokens))


# 分词结果已经是词干，停用词也按词干比较（does → doe）。
EN_STOPWORD_STEMS = set(stem_english_tokens(sorted(EN_STOPWORDS)))

# 中文问句里的虚词与“这篇论文/总结一下”等无检索价值的套话。
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

# 根据 chunk 开头文本识别章节标签：标签 → 触发词（子串匹配）。
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

# 这些章节通常不是问题的直接证据（附录、提示词、参考文献等），命中时在结构加分里扣分。
NOISY_SECTION_TAGS = {"appendix", "prompt", "figure", "table", "references"}

class RetrievalRules:
    """集中维护检索规则与启发式判断，避免编排层继续持有大段常量和规则分支。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or ENHANCED_RETRIEVAL_CONFIG

    @staticmethod
    def normalize_query_text(text: str) -> str:
        """去首尾空白、转小写并把连续空白压成一个空格。"""
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def tokenize_for_keyword_search(cls, text: str) -> List[str]:
        """关键词检索分词：英文按字母数字串切分（至少 2 个字符）后词干化，中文连续片段整体作为一个 token，并过滤噪声 token。

        建索引和查询都走这个函数，单复数、时态等词形差异在两边落到同一个词干上，BM25 才能互相命中。
        关键词路由检索用的是英文译文和英文改写，中文问题到英文正文的桥接交给 LLM，这里不做中文短语切分和词表映射。
        """
        lowered = text.lower()
        english_tokens = [
            token for token in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", lowered) if cls.is_informative_keyword_token(token)
        ]
        chinese_tokens = [
            token for token in re.findall(r"[\u4e00-\u9fff]{2,}", lowered) if cls.is_informative_keyword_token(token)
        ]
        return stem_english_tokens(english_tokens) + chinese_tokens

    @staticmethod
    def is_informative_keyword_token(token: str) -> bool:
        """过滤 OCR/乱码/低信息量 token，避免噪声在 BM25 倒排表里被放大。"""
        normalized = str(token or "").strip().lower()
        if not normalized:
            return False
        # U+FFFD 是解码失败时的替换字符，出现即说明 token 来自乱码文本。
        if "\ufffd" in normalized:
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

    def dedupe_list(self, items: List[str]) -> List[str]:
        """按归一化结果去重，但返回原始写法（仅去首尾空白），保持首次出现顺序。"""
        seen_normalized = set()
        unique: List[str] = []
        for item in items:
            raw = str(item).strip()
            normalized = self.normalize_query_text(raw)
            if normalized and normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique.append(raw)
        return unique

    def _drop_stopwords(self, tokens: List[str]) -> List[str]:
        """去掉噪声 token 与中英文停用词。"""
        return [
            token
            for token in tokens
            if self.is_informative_keyword_token(token)
            and token not in EN_STOPWORDS
            and token not in EN_STOPWORD_STEMS
            and token not in ZH_STOPWORDS
        ]

    def extract_query_keywords(self, tokens: List[str], limit: Optional[int] = None) -> List[str]:
        """从分词结果中抽取查询关键词（去停用词、去重、截断）；全被过滤时退回所有长度大于 1 的 token。"""
        limit = int(limit or self.config["extract_query_keywords_limit"])
        filtered = self._drop_stopwords(tokens)
        if not filtered:
            filtered = [token for token in tokens if len(token) > 1]
        return list(dict.fromkeys(filtered))[:limit]

    def build_query_keywords(self, queries: List[str], limit: Optional[int] = None) -> List[str]:
        """统计多条 query 中 token 的出现次数，返回出现最多的前 limit 个（次数相同保持首次出现顺序）。"""
        limit = int(limit or self.config["build_query_keywords_limit"])
        counts: Dict[str, int] = {}
        for query in queries:
            for token in self.tokenize_for_keyword_search(query):
                counts[token] = counts.get(token, 0) + 1
        return [token for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]]

    def build_query_term_details(self, queries: List[str]) -> List[Dict[str, Any]]:
        """为每条 query 列出去重后的分词结果，供 debug 展示各 query view 实际贡献了哪些词。"""
        details: List[Dict[str, Any]] = []
        for query in queries:
            keywords = list(dict.fromkeys(self.tokenize_for_keyword_search(query)))
            details.append({"query": query, "keywords": keywords, "keyword_count": len(keywords)})
        return details

    @staticmethod
    def detect_language(user_query: str, tokens: List[str]) -> str:
        """粗略判断查询语言：同时含中文和拉丁字母为 mixed，否则 zh / en / unknown。"""
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_query))
        has_latin = any(re.search(r"[a-zA-Z]", token) for token in tokens)
        if has_cjk and has_latin:
            return "mixed"
        if has_cjk:
            return "zh"
        if has_latin:
            return "en"
        return "unknown"

    def detect_section_tags(self, content: str) -> List[str]:
        """只看 chunk 前 900 个字符（章节标题通常在开头），按 SECTION_TAG_RULES 打标签。"""
        prefix = self.normalize_query_text(content[:900])
        tags: List[str] = []
        for tag, patterns in SECTION_TAG_RULES.items():
            if any(pattern in prefix for pattern in patterns):
                tags.append(tag)
        return tags

    def query_similarity(self, left: str, right: str) -> float:
        """两段文本关键词集合的 Jaccard 相似度。"""
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
        """估算一条路由对当前问题的可信度，用作 RRF 投票的乘数。

        base 由路由类型、问题歧义度和主意图决定（例如方法/实验类问题更信任关键词路由，
        概述类问题更信任 HyDE）；再乘以“路由 query 与英文检索问句（原问译文）的相似度”修正，结果夹在 [floor, 1.0]。
        """
        route_queries = route_queries or [source_query]
        source_text = " ".join(route_queries)
        similarity = self.query_similarity(query_profile.retrieval_query, source_text)
        ambiguity = query_profile.ambiguity_score
        main_intent = str(intent_profile.main_intent if intent_profile else "other")
        # 关键词覆盖度：抽到的关键词越接近上限，关键词类路由越可信。
        keyword_coverage = min(1.0, len(query_profile.keywords) / self.config["extract_query_keywords_limit"])
        if route_name == "vector_original":
            base = self.config["query_weight_base_summary_other"] if main_intent in {*OVERVIEW_INTENTS, "other"} else self.config["query_weight_base_default"]
        elif route_name == "vector_rewrite":
            base = self.config["query_weight_base_ambiguous_keyword"] + 0.18 * ambiguity
            if main_intent in {*METHOD_INTENTS, *EXPERIMENT_INTENTS, "comparison", "dataset"}:
                base += self.config["route_focus_bonus"]
        elif route_name == "vector_hyde":
            base = self.config["query_weight_base_ambiguous_other"] + 0.25 * ambiguity
            if main_intent in OVERVIEW_INTENTS:
                base += self.config["route_summary_bonus"]
        elif route_name == "keyword":
            base = self.config["query_weight_base_keyword"] + 0.18 * keyword_coverage
            if main_intent in {*METHOD_INTENTS, *EXPERIMENT_INTENTS, "figure_table"}:
                base += self.config["route_keyword_bonus"]
        elif route_name == "table_structured":
            base = self.config["query_weight_base_keyword"] + 0.22 * keyword_coverage
            if main_intent in {*EXPERIMENT_INTENTS, "comparison", "figure_table", "dataset"}:
                base += self.config["route_keyword_bonus"] + 0.06
        elif route_name == "memory_context":
            base = 0.42 + 0.2 * ambiguity
            if main_intent in {*METHOD_INTENTS, *EXPERIMENT_INTENTS, "comparison", "figure_table", "dataset"}:
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
        """根据章节标签与问题意图给 chunk 一个小幅结构加/减分，结果夹在 [-0.05, 0.12]。

        命中偏好章节加分；附录/参考文献等噪声章节扣分（图表类问题命中图表时不扣、反而加分）；
        概述类问题额外偏好摘要和结论。
        """
        section_tags = set(chunk.get("section_tags", []) or [])
        if not section_tags:
            section_tags = set(self.detect_section_tags(str(chunk.get("content", "") or "")))
        preferred = set(query_profile.section_preferences)
        bonus = 0.0
        if preferred:
            bonus += self.config["section_bonus_weight"] * len(section_tags & preferred)
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        noisy_tags = set(section_tags & NOISY_SECTION_TAGS)
        main_intent = str(query_profile.intent_profile.main_intent if query_profile.intent_profile else "other")
        if chunk_type in {"figure", "table"} and (main_intent == "figure_table" or "figure_table" in query_profile.intent_tags):
            noisy_tags -= {"figure", "table"}
            bonus += self.config["figure_table_bonus_weight"]
        if noisy_tags:
            bonus -= self.config["noisy_section_penalty_weight"] * len(noisy_tags)
        if "abstract" in section_tags and main_intent in OVERVIEW_INTENTS:
            bonus += self.config["preferred_section_bonus_weight"]
        if "conclusion" in section_tags and main_intent in OVERVIEW_INTENTS:
            bonus += self.config["section_path_bonus_weight"]
        return max(-0.05, min(0.12, bonus))

    def normalize_chunk(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """把向量库/关键词索引返回的命中统一成检索层的 chunk 结构。

        字段优先取命中顶层，缺失时回落到 metadata；同时补齐 index_* / retrieval_index_* /
        matched_index_* 三套别名，并根据正文重新识别 section_tags。
        """
        metadata = dict(item.get("metadata", {}) or {})
        chunk = dict(item)

        def pick(key: str, default: Any = "") -> Any:
            """顶层值为真时用顶层，否则取 metadata（顶层的 0、""、False 也视为缺失）。"""
            return item.get(key) or metadata.get(key, default)

        def first_present(key: str, default: Any = "") -> Any:
            """只要顶层存在该键就用顶层（即使是 0/False），仅值为 None 时回落 default。

            用于布尔/置信度字段：False、0.0 是有意义的取值，不能被 pick 的真值判断吞掉。
            """
            value = item.get(key) if key in item else metadata.get(key, default)
            return default if value is None else value

        chunk["content"] = item.get("content") or item.get("text") or metadata.get("content") or metadata.get("text") or ""
        chunk["text"] = chunk["content"]
        # retrieval_index_* 描述本次召回命中的入口文本，必须和原 chunk 字段并存，不能覆盖最终证据内容。
        chunk["retrieval_index_id"] = item.get("retrieval_index_id") or item.get("index_id") or metadata.get("retrieval_index_id", metadata.get("index_id", ""))
        chunk["retrieval_index_type"] = item.get("retrieval_index_type") or item.get("index_type") or metadata.get("retrieval_index_type", metadata.get("index_type", ""))
        chunk["retrieval_index_text"] = item.get("retrieval_index_text") or item.get("index_text") or metadata.get("retrieval_index_text", metadata.get("index_text", ""))
        chunk["retrieval_index_weight"] = item.get("retrieval_index_weight") or item.get("index_weight") or metadata.get("retrieval_index_weight", metadata.get("index_weight", 1.0))
        chunk["retrieval_index_enabled_routes"] = item.get("retrieval_index_enabled_routes") or metadata.get("retrieval_index_enabled_routes", [])
        # matched_index_* 描述“本次命中的检索入口”，和 chunk.content 的最终证据职责分离，方便 trace 解释同一 chunk 的多入口命中。
        if not chunk["retrieval_index_id"]:
            legacy_chunk_ref = item.get("parent_chunk_id") or metadata.get("parent_chunk_id") or item.get("chunk_id") or metadata.get("chunk_id", 0)
            chunk["retrieval_index_id"] = legacy_body_index_id(legacy_chunk_ref)
            chunk["retrieval_index_type"] = "body"
            chunk["retrieval_index_text"] = chunk["content"]
            chunk["retrieval_index_weight"] = 1.0
        chunk["index_id"] = chunk["retrieval_index_id"]
        chunk["index_type"] = chunk["retrieval_index_type"]
        chunk["index_text"] = chunk["retrieval_index_text"]
        chunk["index_weight"] = chunk["retrieval_index_weight"]
        chunk["matched_index_id"] = item.get("matched_index_id") or metadata.get("matched_index_id") or chunk["retrieval_index_id"]
        chunk["matched_index_type"] = item.get("matched_index_type") or metadata.get("matched_index_type") or chunk["retrieval_index_type"]
        chunk["matched_index_text"] = item.get("matched_index_text") or metadata.get("matched_index_text") or chunk["retrieval_index_text"]
        chunk["matched_index_score"] = item.get("matched_index_score", metadata.get("matched_index_score", item.get("score")))
        chunk["source"] = pick("source")
        chunk["document_name"] = pick("document_name")
        chunk["chunk_id"] = pick("chunk_id", 0)
        chunk["chunk_index"] = pick("chunk_index", 0)
        chunk["parent_chunk_id"] = pick("parent_chunk_id", chunk["chunk_id"])
        chunk["original_chunk_id"] = pick("original_chunk_id", chunk["parent_chunk_id"])
        chunk["page_number"] = pick("page_number")
        chunk["page_start"] = pick("page_start", None)
        chunk["page_end"] = pick("page_end", None)
        chunk["page_range"] = pick("page_range")
        chunk["subchunk_index"] = pick("subchunk_index", 0)
        chunk["subchunk_count"] = pick("subchunk_count", 0)
        chunk["subchunk_label"] = pick("subchunk_label")
        # 与向量库入库规则一致：分片标签为空时取 subchunk_label（chunk 文件里只有后者）。
        chunk["content_part_label"] = pick("content_part_label") or chunk["subchunk_label"]
        chunk["section_path"] = pick("section_path")
        chunk["section_title"] = pick("section_title")
        chunk["title"] = pick("title")
        chunk["authors"] = pick("authors")
        chunk["categories"] = pick("categories")
        chunk["published_date"] = pick("published_date")
        chunk["url"] = pick("url")
        chunk["chunk_type"] = pick("chunk_type", "text")
        chunk["table_id"] = pick("table_id")
        chunk["asset_kind"] = pick("asset_kind")
        chunk["asset_path"] = pick("asset_path")
        chunk["asset_abs_path"] = pick("asset_abs_path")
        chunk["asset_summary"] = pick("asset_summary")
        chunk["asset_preview_text"] = pick("asset_preview_text")
        chunk["asset_caption"] = pick("asset_caption")
        # asset-section 匹配字段用于解释章节锚点来源，不应被 section_title/path 的旧字段语义吞掉。
        chunk["asset_section_match_type"] = first_present("asset_section_match_type", "")
        chunk["asset_section_match_confidence"] = first_present("asset_section_match_confidence", 0.0)
        chunk["asset_section_match_reason"] = first_present("asset_section_match_reason", "")
        chunk["asset_section_match_is_heuristic"] = first_present("asset_section_match_is_heuristic", False)
        chunk["asset_section_match_allow_embedding"] = first_present("asset_section_match_allow_embedding", False)
        # 结构化表格证据只接受 v2 单一入口；旧 table_structured_* 字段不再在归一化层续命。
        chunk["table_evidence"] = pick("table_evidence", {})
        chunk["asset_rows"] = pick("asset_rows", 0)
        chunk["asset_columns"] = pick("asset_columns", 0)
        chunk["order_index"] = pick("order_index", 0)
        # section tags 是后续结构加权、context expansion 和 trace 观察的共同输入，统一在这里补齐。
        chunk["section_tags"] = self.detect_section_tags(str(chunk["content"] or ""))
        return chunk
