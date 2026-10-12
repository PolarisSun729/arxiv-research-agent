from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.intent.intent_service import EXPERIMENT_INTENTS, MAIN_INTENTS, METHOD_INTENTS, OVERVIEW_INTENTS
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

# 中文问法 → 英文论文术语的桥接词表：论文正文多为英文，中文问题需要映射后才能命中 BM25 倒排。
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

# 英文单复数与近义术语扩展，弥补 BM25 不做词形还原的问题。
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

# evidence query 末尾追加的“证据导向”提示词，按查询语言选择。
# 注意：中文版本此前因编码错误被写成了乱码（"璁烘枃" 等，即 UTF-8 字节被按 GBK 解码），现已更正。
_EVIDENCE_HINT_TERMS = {
    "zh": ["论文", "证据", "段落"],
    "default": ["paper", "evidence", "passage"],
}


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
        """关键词检索分词：英文按字母数字串切分（至少 2 个字符），中文连续片段再拆出领域短语，最后过滤噪声 token。"""
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
        """归一化后去重（保持首次出现顺序），再用空格拼成一个 query 字符串。"""
        normalized_terms = (self.normalize_query_text(str(term)) for term in terms)
        return " ".join(dict.fromkeys(term for term in normalized_terms if term))

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
            if self.is_informative_keyword_token(token) and token not in EN_STOPWORDS and token not in ZH_STOPWORDS
        ]

    def extract_query_keywords(self, tokens: List[str], limit: Optional[int] = None) -> List[str]:
        """从分词结果中抽取查询关键词（去停用词、去重、截断）；全被过滤时退回所有长度大于 1 的 token。"""
        limit = int(limit or self.config["extract_query_keywords_limit"])
        filtered = self._drop_stopwords(tokens)
        if not filtered:
            filtered = [token for token in tokens if len(token) > 1]
        return list(dict.fromkeys(filtered))[:limit]

    def extract_paper_terms_from_text(self, text: str, limit: Optional[int] = None) -> List[str]:
        """从任意文本中抽取论文术语（去停用词、去重、截断）；与查询关键词不同，这里没有兜底。"""
        limit = int(limit or self.config["extract_paper_terms_limit"])
        filtered = self._drop_stopwords(self.tokenize_for_keyword_search(text))
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

    def preferred_sections_for_main_intent(self, main_intent: str) -> List[str]:
        """主意图对应的偏好章节；未知意图按 other 处理。"""
        spec = MAIN_INTENTS.get(main_intent, MAIN_INTENTS["other"])
        return list(spec["preferred_sections"])

    def main_intent_terms(self, main_intent: str) -> List[str]:
        """主意图对应的关键词（去重）；未知意图按 other 处理。"""
        spec = MAIN_INTENTS.get(main_intent, MAIN_INTENTS["other"])
        return self.dedupe_list([str(item).strip() for item in spec["keywords"] if str(item).strip()])

    def preferred_section_tags_from_plan(self, query_plan: Dict[str, Any], main_intent: str) -> List[str]:
        """合并 query plan 指定的章节与意图默认章节，去重后最多保留 6 个（plan 中的排在前面）。"""
        preferred = [str(item).strip() for item in (query_plan.get("preferred_sections", []) or []) if str(item).strip()]
        preferred.extend(self.preferred_sections_for_main_intent(main_intent))
        return self.dedupe_list(preferred)[:6]

    @staticmethod
    def compact_terms(terms: List[str], limit: int) -> List[str]:
        """去空白、去重后保留前 limit 个词（区分大小写）。"""
        compacted: List[str] = []
        for term in terms:
            normalized = str(term).strip()
            if normalized and normalized not in compacted:
                compacted.append(normalized)
            if len(compacted) >= limit:
                break
        return compacted

    def build_semantic_query(self, user_query: str, keywords: List[str], intent_tags: List[str], intent_profile: Optional[Any] = None) -> str:
        """语义 query：意图改写焦点 + 前 6 个关键词；都没有时用原问分词，最后兜底为归一化原问。"""
        parts: List[str] = []
        if intent_profile and getattr(intent_profile, "rewrite_focus", None):
            parts.extend(intent_profile.rewrite_focus[:4])
        parts.extend(keywords[:6])
        if not parts:
            parts.extend(self.tokenize_for_keyword_search(user_query)[:6])
        return self.dedupe_terms(parts) or self.normalize_query_text(user_query)

    def build_evidence_query(self, keywords: List[str], intent_tags: List[str], language: str, intent_profile: Optional[Any] = None) -> str:
        """证据 query：rerank 焦点 + 前 4 个关键词 + 按语言选择的“证据导向”提示词。"""
        parts: List[str] = []
        if intent_profile and getattr(intent_profile, "rerank_focus", None):
            parts.extend(intent_profile.rerank_focus[:4])
        parts.extend(keywords[:4])
        parts.extend(_EVIDENCE_HINT_TERMS["zh"] if language == "zh" else _EVIDENCE_HINT_TERMS["default"])
        return self.dedupe_terms(parts)

    def build_keyword_query(self, keywords: List[str], intent_tags: List[str], intent_profile: Optional[Any] = None) -> str:
        """关键词 query：前 keyword_parts_limit 个关键词 + 意图改写焦点。"""
        # 切片得到的是新列表，后续 extend 不会修改调用方传入的 keywords。
        # （原先这里还有一个 `if not parts: parts = keywords[:limit]` 分支：parts 为空时该切片必然也为空，属于死代码，已删除。）
        parts = keywords[: self.config["keyword_parts_limit"]]
        if intent_profile and getattr(intent_profile, "rewrite_focus", None):
            parts.extend(intent_profile.rewrite_focus[:4])
        return self.dedupe_terms(parts)

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
        概述类问题更信任 HyDE）；再乘以“路由 query 与原问的相似度”修正，结果夹在 [floor, 1.0]。
        """
        route_queries = route_queries or [source_query]
        source_text = " ".join(route_queries)
        similarity = self.query_similarity(query_profile.normalized_query, source_text)
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
        chunk["content_part_label"] = pick("content_part_label")
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
