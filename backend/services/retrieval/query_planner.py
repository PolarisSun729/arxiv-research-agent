"""查询规划：把用户问题加工成检索所需的各种查询形态。

产出依次为：
1. IntentProfile —— 意图识别结果；
2. QueryProfile —— 分词、关键词、语言、意图标签与偏好章节；
3. query views —— 原问英文译文（原问路由用）与 LLM 改写问句（改写路由用）；
4. rerank query —— 交给 rerank 服务的查询。
本模块只做规划，不执行任何召回。
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from services.intent.intent_service import IntentProfile, IntentService
from services.retrieval.contracts import QueryProfile
from services.retrieval.retrieval_rules import RetrievalRules

logger = logging.getLogger(__name__)
# 改写要求输出英文问句；含中文字符的改写视为不合格输出。
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def _clean_str_list(values: Optional[Iterable[Any]]) -> List[str]:
    """转成字符串、去首尾空白并丢弃空项（不去重）；LLM/外部输入的列表字段都先过一遍。"""
    return [str(item).strip() for item in (values or []) if str(item).strip()]


class QueryPlanner:
    """负责 query profile、query views 和 rerank query，不执行任何召回。

    文件末尾的一组薄封装方法（分词、关键词提取、同义词扩展、文本归一化等）委托给 RetrievalRules，
    它们构成其它检索组件依赖的 ``query_tools`` 接口（route_retriever、关键词后端、表格路由都通过它调用）。
    """

    def __init__(
        self,
        *,
        vector_store_service: Any,
        generation_service: Any,
        intent_service: IntentService,
        rerank_service: Any,
        enhanced_config: Dict[str, Any],
        retrieval_rules: Optional[RetrievalRules] = None,
    ) -> None:
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.intent_service = intent_service
        self.rerank_service = rerank_service
        self.config = enhanced_config
        # 规则模块集中提供 query type、stopwords 和 section 偏好，避免 planner 继续复制一份常量。
        self.retrieval_rules = retrieval_rules or RetrievalRules(config=enhanced_config)
        # 原问翻译、原问改写与意图识别并发执行，单独的小池子。
        self._llm_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query-llm")

    def translate_original_query(self, user_query: str) -> Dict[str, Any]:
        """把原问译成英文供原问向量路由检索；失败时 query 沿用原问，status/error 记下原因。"""
        started = perf_counter()
        result: Dict[str, Any] = {"query": user_query, "status": "ok", "error": "", "latency_ms": 0.0}
        try:
            if self.generation_service is None:
                raise RuntimeError("generation service unavailable")
            translated = " ".join(str(self.generation_service.translate_query_for_retrieval(user_query) or "").split())
            if not translated:
                raise ValueError("empty translation")
            result["query"] = translated
        except Exception as exc:
            logger.warning("Original query translation failed, using the original question: %s", exc)
            result.update({"status": "error", "error": str(exc)})
        result["latency_ms"] = round((perf_counter() - started) * 1000, 1)
        return result

    def rewrite_original_query(self, user_query: str, paper_context: Dict[str, Any]) -> Dict[str, Any]:
        """让 LLM 把原问改写成英文检索问句（具体化模糊指代、拆分复合问题），供改写路由检索。

        不做任何规则兜底：失败（含没给出任何问句）时 queries 为空、status=error，改写路由直接跳过。
        """
        started = perf_counter()
        result: Dict[str, Any] = {"queries": [], "status": "ok", "error": "", "latency_ms": 0.0}
        try:
            if self.generation_service is None:
                raise RuntimeError("generation service unavailable")
            result["queries"] = _clean_str_list(
                self.generation_service.rewrite_query_for_retrieval(user_query, paper_context=paper_context)
            )
        except Exception as exc:
            logger.warning("Query rewrite failed, skipping the rewrite route: %s", exc)
            result.update({"status": "error", "error": str(exc)})
        result["latency_ms"] = round((perf_counter() - started) * 1000, 1)
        return result

    def build_intent_profile(self, user_query: str, paper_context: Optional[Dict[str, Any]] = None) -> IntentProfile:
        """调用意图服务识别问题意图。"""
        return self.intent_service.build_intent_profile(user_query, paper_context=paper_context or {})

    def build_query_profile(self, user_query: str, intent_profile: IntentProfile, retrieval_query: str) -> QueryProfile:
        """构造 QueryProfile：分词与关键词、语言、意图标签和偏好章节（偏好章节只来自意图识别）。

        分词与关键词取自英文检索问句 retrieval_query，语言仍按原问判断。
        """
        tokens = self.tokenize_for_keyword_search(retrieval_query)
        return QueryProfile(
            original_query=user_query,
            normalized_query=self.normalize_query_text(user_query),
            retrieval_query=retrieval_query,
            language=self.detect_language(user_query, self.tokenize_for_keyword_search(user_query)),
            intent_profile=intent_profile,
            tokens=tokens,
            keywords=self.extract_query_keywords(tokens),
            intent_tags=list(intent_profile.sub_intents),
            ambiguity_score=intent_profile.ambiguity_score,
            section_preferences=self.dedupe_list(list(intent_profile.preferred_sections)),
        )

    def build_query_views(
        self,
        user_query: str,
        rewrite: Dict[str, Any],
        translation: Dict[str, Any],
        enable_query_rewrite: bool,
        intent_profile: Optional[IntentProfile] = None,
    ) -> Dict[str, Any]:
        """整理改写路由的查询列表（selected_queries），每条改写的去留原因记录在 candidates 中。

        与原问或原问译文相同的改写会被丢弃，因为原问路由已经召回过；含中文的改写不符合输出要求，也丢弃。
        改写关闭、失败或全部被丢弃时 selected_queries 为空，改写路由跳过，原因记在 rewrite_status。
        """
        covered = {
            self.normalize_query_text(user_query),
            self.normalize_query_text(str(translation.get("query") or "")),
        }
        candidate_rows: List[Dict[str, Any]] = []
        selected_queries: List[str] = []
        seen = set()
        for idx, query in enumerate(rewrite.get("queries") or []):
            normalized = self.normalize_query_text(query)
            row = {
                "query": query,
                "source": "rewrite",
                "source_index": idx,
                "normalized": normalized,
                "selected": False,
                "reason": "kept",
            }
            if _CJK_PATTERN.search(query):
                row["reason"] = "not_english"
            elif normalized in covered:
                row["reason"] = "same_as_original"
            elif normalized in seen:
                row["reason"] = "duplicate"
            else:
                seen.add(normalized)
                row["selected"] = True
                selected_queries.append(query)
            candidate_rows.append(row)
        rewrite_status = str(rewrite.get("status") or "")
        if rewrite_status == "ok" and not selected_queries:
            # 只要有一条和原问/译文重复，就说明这一路能召回的原问路由已经覆盖；否则全是不合格输出。
            covered_rows = [row for row in candidate_rows if row["reason"] == "same_as_original"]
            rewrite_status = "covered_by_original" if covered_rows else "invalid_output"

        # 返回值与 rewrite_debug 共用同一组字段（关键词与逐词明细只计算一次），rewrite_debug 额外带意图摘要。
        views = {
            "enabled": enable_query_rewrite,
            "original_query": user_query,
            "original_translation": translation,
            "rewrite_status": rewrite_status,
            "rewrite_latency_ms": float(rewrite.get("latency_ms") or 0.0),
            "llm_error": rewrite.get("error") or None,
            "model_queries": list(rewrite.get("queries") or []),
            "selected_queries": selected_queries,
            "selected_keywords": self.build_query_keywords(selected_queries),
            "selected_query_details": self.build_query_term_details(selected_queries),
            "candidates": candidate_rows,
        }
        rewrite_debug = {**views, "intent_profile": self.debug_intent_profile(intent_profile)}
        return {**views, "rewrite_debug": rewrite_debug}

    def build_rerank_query(self, user_query: str, query_profile: QueryProfile) -> str:
        """rerank 查询由 rerank 服务按其模型偏好构造。"""
        return self.rerank_service.build_rerank_query(user_query, query_profile)

    def build_query_bundle(
        self,
        *,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        enable_query_rewrite: bool,
    ) -> Dict[str, Any]:
        """一次性生成意图、QueryProfile、query views 和 rerank query（检索主流程的入口）。

        原问翻译、原问改写与意图识别三次 LLM 调用互不依赖，并发执行。
        """
        paper_context_payload = self.build_paper_context(collection_name, paper_context=paper_context)
        translation_future = self._llm_pool.submit(copy_context().run, self.translate_original_query, user_query)
        rewrite_future = None
        if enable_query_rewrite:
            rewrite_future = self._llm_pool.submit(
                copy_context().run, self.rewrite_original_query, user_query, paper_context_payload
            )
        intent_profile = self.build_intent_profile(user_query, paper_context=paper_context)
        translation = translation_future.result()
        query_profile = self.build_query_profile(user_query, intent_profile, str(translation["query"]))
        rewrite = (
            rewrite_future.result()
            if rewrite_future is not None
            else {"queries": [], "status": "disabled", "error": "", "latency_ms": 0.0}
        )
        query_views = self.build_query_views(
            user_query,
            rewrite,
            translation,
            enable_query_rewrite,
            intent_profile=intent_profile,
        )
        rerank_query = self.build_rerank_query(user_query, query_profile)
        return {
            "intent_profile": intent_profile,
            "query_profile": query_profile,
            "query_views": query_views,
            "rerank_query": rerank_query,
        }

    def build_paper_context(
        self,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        sample_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """合并调用方给的论文信息与 collection 抽样 chunk，得到改写所需的标题、摘要和章节标题。

        调用方提供的字段优先；缺失的标题/摘要从抽样 chunk 中补（摘要取最长的 abstract 段）。
        向量库不可用时只使用调用方信息。
        """
        sample_limit = int(sample_limit or self.config["sample_limit"])
        merged: Dict[str, Any] = {"title": "", "abstract": "", "section_titles": []}
        if paper_context:
            merged["title"] = str(paper_context.get("title", "") or "").strip()
            merged["abstract"] = str(paper_context.get("abstract", "") or "").strip()
            merged["section_titles"] = _clean_str_list(paper_context.get("section_titles"))
        try:
            sample_chunks = self.vector_store_service.get_all_chunks(collection_name, limit=sample_limit)
        except Exception:
            sample_chunks = []

        section_titles = list(merged["section_titles"])
        abstract_candidates: List[str] = []
        for chunk in sample_chunks:
            title = str(chunk.get("title", "") or "").strip()
            content = str(chunk.get("content", "") or "").strip()
            section_title = str(
                chunk.get("section_title")
                or chunk.get("content_part_label")
                or chunk.get("subchunk_label")
                or ""
            ).strip()
            if not merged["title"] and title:
                merged["title"] = title
            if section_title:
                section_titles.append(section_title)
            if content and any(tag == "abstract" for tag in chunk.get("section_tags", []) or []):
                abstract_candidates.append(content)

        if not merged["abstract"] and abstract_candidates:
            merged["abstract"] = max(abstract_candidates, key=len)[:1800]
        # dedupe_list 按归一化形式去重，章节标题写法不同（大小写、空白）也只保留一条。
        merged["section_titles"] = self.dedupe_list(section_titles)
        return merged

    # ---- 以下为委托给 RetrievalRules 的薄封装，同时构成其它检索组件使用的 query_tools 接口 ----

    @staticmethod
    def detect_language(user_query: str, tokens: List[str]) -> str:
        """粗略判断问题语言（zh / en / mixed / unknown）。"""
        return RetrievalRules.detect_language(user_query, tokens)

    def extract_query_keywords(self, tokens: List[str], limit: Optional[int] = None) -> List[str]:
        """去停用词、去重后提取关键词（limit 为空时用配置上限）。"""
        return self.retrieval_rules.extract_query_keywords(tokens, limit=limit)

    @staticmethod
    def tokenize_for_keyword_search(text: str) -> List[str]:
        """关键词检索使用的分词（中英文混合）。"""
        return RetrievalRules.tokenize_for_keyword_search(text)

    @staticmethod
    def is_informative_keyword_token(token: str) -> bool:
        """判断关键词 token 是否有信息量（过滤乱码、纯数字符号、重复字符等噪声）。"""
        return RetrievalRules.is_informative_keyword_token(token)

    def build_query_keywords(self, queries: List[str], limit: Optional[int] = None) -> List[str]:
        """统计多条查询中出现最多的关键词。"""
        return self.retrieval_rules.build_query_keywords(queries, limit=limit)

    def build_query_term_details(self, queries: List[str]) -> List[Dict[str, Any]]:
        """逐条查询列出去重后的分词结果（debug 用）。"""
        return self.retrieval_rules.build_query_term_details(queries)

    @staticmethod
    def normalize_query_text(text: str) -> str:
        """归一化查询文本（去首尾空白、小写、压缩连续空白）。"""
        return RetrievalRules.normalize_query_text(text)

    def dedupe_list(self, items: List[str]) -> List[str]:
        """按归一化结果保序去重，返回原始写法。"""
        return self.retrieval_rules.dedupe_list(items)

    @staticmethod
    def debug_intent_profile(intent_profile: Optional[IntentProfile]) -> Optional[Dict[str, Any]]:
        """意图结果的可序列化形式（用于 rewrite debug）。"""
        if intent_profile is None:
            return None
        return intent_profile.to_dict()
