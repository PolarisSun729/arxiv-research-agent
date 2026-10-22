"""查询规划：把用户问题加工成检索所需的各种查询形态。

产出依次为：
1. IntentProfile —— 意图识别结果；
2. QueryProfile —— 分词、关键词、语言、论文术语，以及 semantic / evidence / keyword 三种改写查询；
3. query views —— 参与多路召回的查询列表（原问题 + 改写 + LLM/启发式扩展，去重后截断）；
4. rerank query —— 交给 rerank 服务的查询。
本模块只做规划，不执行任何召回。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from services.intent.intent_service import METHOD_INTENTS, IntentProfile, IntentService
from services.retrieval.contracts import QueryProfile
from services.retrieval.retrieval_rules import RetrievalRules
from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()
# 参与召回的改写查询条数上限。
QUERY_VIEW_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_view_limit"]
# 请求 LLM 生成检索计划时的改写查询条数上限。
QUERY_PLAN_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_plan_limit"]

logger = logging.getLogger(__name__)


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
        # 原问翻译与意图识别并发执行，单独的小池子。
        self._translation_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query-translation")

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

    def build_intent_profile(self, user_query: str, paper_context: Optional[Dict[str, Any]] = None) -> IntentProfile:
        """调用意图服务识别问题意图。"""
        return self.intent_service.build_intent_profile(user_query, paper_context=paper_context or {})

    def build_query_profile(
        self,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        intent_profile: Optional[IntentProfile] = None,
    ) -> QueryProfile:
        """构造 QueryProfile：分词与关键词、论文上下文、检索计划、偏好章节和三种改写查询。

        未传入 intent_profile 时现场识别；论文上下文由传入信息与 collection 抽样 chunk 合并得到。
        """
        normalized_query = self.normalize_query_text(user_query)
        tokens = self.tokenize_for_keyword_search(user_query)
        keywords = self.extract_query_keywords(tokens)
        language = self.detect_language(user_query, tokens)
        paper_context_payload = self.build_paper_context(collection_name, paper_context=paper_context)
        intent_profile = intent_profile or self.build_intent_profile(user_query, paper_context=paper_context)
        query_plan = self.build_query_plan(user_query, paper_context_payload, intent_profile=intent_profile)
        intent_tags = list(intent_profile.sub_intents)
        paper_terms = _clean_str_list(query_plan.get("paper_terms"))
        section_preferences = self.preferred_section_tags_from_plan(query_plan, intent_profile.main_intent)
        section_preferences = self.dedupe_list([*intent_profile.preferred_sections, *section_preferences])
        semantic_query, evidence_query, keyword_query = self.build_query_views_from_plan(
            user_query=user_query,
            query_plan=query_plan,
            keywords=keywords,
            intent_tags=intent_tags,
            language=language,
            paper_context=paper_context_payload,
            intent_profile=intent_profile,
        )
        return QueryProfile(
            original_query=user_query,
            normalized_query=normalized_query,
            language=language,
            intent_profile=intent_profile,
            tokens=tokens,
            keywords=keywords,
            intent_tags=intent_tags,
            paper_terms=paper_terms,
            ambiguity_score=intent_profile.ambiguity_score,
            semantic_query=semantic_query,
            evidence_query=evidence_query,
            keyword_query=keyword_query,
            section_preferences=section_preferences,
            query_plan=query_plan,
        )

    def build_query_views(self, user_query: str, query_profile: QueryProfile, enable_query_rewrite: bool) -> Dict[str, Any]:
        """生成参与召回的查询列表（selected_queries）及其筛选记录。

        候选来源按顺序：原问题与三种改写（core）→ 检索计划或 LLM 改写（plan）→ 启发式改写（heuristic）；
        关闭改写时只用 core。归一化后为空、与原问题相同或重复的候选被丢弃，其余取前 QUERY_VIEW_LIMIT 条，
        每个候选的去留原因都记录在 candidates 中。
        """
        query_plan = query_profile.query_plan or {}
        plan_queries = self.extract_plan_queries(query_plan)
        fallback_rewrites = self.heuristic_query_rewrites(query_profile)
        llm_rewrites: List[str] = []
        llm_error: Optional[str] = None
        if plan_queries:
            llm_rewrites = plan_queries[:QUERY_VIEW_LIMIT]
        elif enable_query_rewrite and self.generation_service is not None:
            try:
                llm_rewrites = self.generation_service.rewrite_query_for_retrieval(
                    user_query,
                    max_queries=QUERY_VIEW_LIMIT,
                    paper_context={
                        "title": query_plan.get("paper_title", ""),
                        "abstract": query_plan.get("paper_abstract", ""),
                        "section_titles": query_plan.get("section_titles", []),
                        "candidate_terms": query_profile.paper_terms,
                    },
                    intent_profile=query_profile.intent_profile.to_dict(),
                )
            except Exception as exc:  # pragma: no cover
                llm_error = str(exc)

        candidate_sources = [
            ("core", query_profile.original_query),
            ("core", query_profile.semantic_query),
            ("core", query_profile.evidence_query),
            ("core", query_profile.keyword_query),
        ]
        if enable_query_rewrite:
            candidate_sources.extend(("plan", query) for query in llm_rewrites)
            candidate_sources.extend(("heuristic", query) for query in fallback_rewrites)

        candidate_rows: List[Dict[str, Any]] = []
        seen = set()
        selected_queries: List[str] = []
        for idx, (source, query) in enumerate(candidate_sources):
            stripped = str(query).strip()
            normalized = self.normalize_query_text(stripped)
            row = {
                "query": stripped,
                "source": source,
                "source_index": idx,
                "normalized": normalized,
                "selected": False,
                "reason": "kept",
            }
            if not normalized:
                row["reason"] = "empty"
                candidate_rows.append(row)
                continue
            if normalized == query_profile.normalized_query:
                row["reason"] = "same_as_original"
                candidate_rows.append(row)
                continue
            if normalized in seen:
                row["reason"] = "duplicate"
                candidate_rows.append(row)
                continue
            seen.add(normalized)
            candidate_rows.append(row)
            if len(selected_queries) < QUERY_VIEW_LIMIT:
                row["selected"] = True
                selected_queries.append(stripped)
            else:
                row["reason"] = "trimmed_to_top_k"
        if not selected_queries:
            # 所有候选都被过滤时至少保留 semantic 改写，保证下游召回有查询可用。
            selected_queries.append(query_profile.semantic_query)

        # 返回值与 rewrite_debug 共用同一组字段（关键词与逐词明细只计算一次），rewrite_debug 额外带意图摘要。
        views = {
            "enabled": enable_query_rewrite,
            "original_query": user_query,
            "query_plan": query_plan,
            "model_queries": llm_rewrites,
            "heuristic_queries": fallback_rewrites,
            "selected_queries": selected_queries,
            "selected_keywords": self.build_query_keywords(selected_queries),
            "selected_query_details": self.build_query_term_details(selected_queries),
            "candidates": candidate_rows,
            "llm_error": llm_error,
            "view_queries": {
                "original": query_profile.original_query,
                "semantic": query_profile.semantic_query,
                "evidence": query_profile.evidence_query,
                "keywords": query_profile.keyword_query,
            },
        }
        rewrite_debug = {**views, "intent_profile": self.debug_intent_profile(query_profile.intent_profile)}
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

        原问的英文翻译与这些步骤并发，结果放在 query_views["original_translation"]。
        """
        translation_future = self._translation_pool.submit(copy_context().run, self.translate_original_query, user_query)
        intent_profile = self.build_intent_profile(user_query, paper_context=paper_context)
        query_profile = self.build_query_profile(
            user_query,
            collection_name,
            paper_context=paper_context,
            intent_profile=intent_profile,
        )
        query_views = self.build_query_views(user_query, query_profile, enable_query_rewrite)
        query_views["original_translation"] = translation_future.result()
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
        """合并调用方给的论文信息与 collection 抽样 chunk，得到标题、摘要、章节标题、候选术语和正文样例。

        调用方提供的字段优先；缺失的标题/摘要从抽样 chunk 中补（摘要取最长的 abstract 段）。
        向量库不可用时只使用调用方信息。
        """
        sample_limit = int(sample_limit or self.config["sample_limit"])
        merged: Dict[str, Any] = {
            "title": "",
            "abstract": "",
            "section_titles": [],
            "candidate_terms": [],
            "source_samples": [],
        }
        if paper_context:
            merged["title"] = str(paper_context.get("title", "") or "").strip()
            merged["abstract"] = str(paper_context.get("abstract", "") or "").strip()
            merged["section_titles"] = _clean_str_list(paper_context.get("section_titles"))
            merged["candidate_terms"] = _clean_str_list(paper_context.get("candidate_terms"))
        try:
            sample_chunks = self.vector_store_service.get_all_chunks(collection_name, limit=sample_limit)
        except Exception:
            sample_chunks = []

        section_titles = list(merged["section_titles"])
        # 已收录章节标题的归一化形式，用于按“写法无关”的方式去重。
        seen_sections = {self.normalize_query_text(item) for item in section_titles}
        source_samples: List[str] = []
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
                normalized_section = self.normalize_query_text(section_title)
                if normalized_section and normalized_section not in seen_sections:
                    section_titles.append(section_title)
                    seen_sections.add(normalized_section)
            if content and len(source_samples) < self.config["source_sample_limit"]:
                source_samples.append(content[: self.config["source_sample_primary_limit"]])
            # 摘要段落留作摘要候选；引言/结论/方法/实验段落再补一份更短的样例，丰富候选术语来源。
            section_tags = chunk.get("section_tags", []) or []
            if any(tag == "abstract" for tag in section_tags) and content:
                abstract_candidates.append(content)
            elif any(tag in {"introduction", "conclusion", "method", "experiment"} for tag in section_tags) and content:
                source_samples.append(content[: self.config["source_sample_secondary_limit"]])

        if not merged["abstract"] and abstract_candidates:
            merged["abstract"] = max(abstract_candidates, key=len)[:1800]

        merged["section_titles"] = self.dedupe_list(section_titles)
        merged["candidate_terms"] = self.merge_candidate_terms(
            merged["candidate_terms"],
            [
                merged["title"],
                merged["abstract"][:900],
                " ".join(merged["section_titles"][:20]),
                " ".join(source_samples[: self.config["source_sample_limit"]]),
            ],
        )
        merged["source_samples"] = source_samples[: self.config["source_sample_limit"]]
        return merged

    def build_query_plan(
        self,
        user_query: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> Dict[str, Any]:
        """生成检索计划（论文术语、偏好章节、改写查询）。

        优先调用生成服务的 plan_queries_for_retrieval；旧版生成服务只有 rewrite_query_for_retrieval 时，
        把改写结果包装成计划；生成服务不可用、出错或返回空时退回启发式计划，错误信息记在 llm_error。
        """
        query_plan: Dict[str, Any] = {}
        llm_error: Optional[str] = None
        intent_payload = intent_profile.to_dict() if intent_profile is not None else None
        if self.generation_service is not None:
            try:
                if hasattr(self.generation_service, "plan_queries_for_retrieval"):
                    query_plan = self.generation_service.plan_queries_for_retrieval(
                        question=user_query,
                        max_queries=QUERY_PLAN_LIMIT,
                        paper_context=paper_context,
                        intent_profile=intent_payload,
                    )
                else:
                    rewrites = self.generation_service.rewrite_query_for_retrieval(
                        question=user_query,
                        max_queries=QUERY_PLAN_LIMIT,
                        paper_context=paper_context,
                    )
                    query_plan = {
                        "paper_terms": paper_context.get("candidate_terms", [])[: self.config["paper_terms_preview_limit"]],
                        "preferred_sections": intent_profile.preferred_sections if intent_profile else [],
                        "rewrite_queries": [
                            {"query": query, "focus": "retrieval", "channels": ["vector", "keyword"]}
                            for query in rewrites
                        ],
                    }
            except Exception as exc:  # pragma: no cover
                llm_error = str(exc)
        if not query_plan:
            query_plan = self.heuristic_query_plan(user_query, paper_context, intent_profile=intent_profile)
        else:
            if not isinstance(query_plan, dict):
                query_plan = {}
            query_plan = self.normalize_query_plan(query_plan, user_query, paper_context, intent_profile=intent_profile)
        if llm_error and not query_plan.get("llm_error"):
            query_plan["llm_error"] = llm_error
        return query_plan

    def normalize_query_plan(
        self,
        query_plan: Dict[str, Any],
        user_query: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> Dict[str, Any]:
        """清洗外部检索计划：只保留白名单字段，改写查询为空时改用启发式计划，并补齐论文上下文字段。"""
        source_plan = query_plan if isinstance(query_plan, dict) else {}
        # 外部 query planner 只能提供检索计划字段，IntentProfile 之外的数据不会进入运行时状态。
        normalized = {
            key: source_plan[key]
            for key in ("paper_terms", "preferred_sections", "rewrite_queries")
            if key in source_plan
        }
        rewrite_queries = self.extract_plan_queries(normalized)
        if not rewrite_queries:
            normalized = self.heuristic_query_plan(user_query, paper_context, intent_profile=intent_profile)
            rewrite_queries = self.extract_plan_queries(normalized)
        normalized["rewrite_queries"] = rewrite_queries
        normalized["paper_terms"] = self.dedupe_list(_clean_str_list(normalized.get("paper_terms")))
        normalized["preferred_sections"] = self.dedupe_list(_clean_str_list(normalized.get("preferred_sections")))
        normalized["paper_title"] = str(paper_context.get("title", "") or "").strip()
        normalized["paper_abstract"] = str(paper_context.get("abstract", "") or "").strip()
        normalized["section_titles"] = _clean_str_list(paper_context.get("section_titles"))
        if intent_profile is not None:
            # 意图服务给出的偏好章节排在前面。
            normalized["preferred_sections"] = self.dedupe_list(
                [*intent_profile.preferred_sections, *normalized.get("preferred_sections", [])]
            )
        return normalized

    def heuristic_query_plan(
        self,
        user_query: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> Dict[str, Any]:
        """不依赖 LLM 的检索计划：按主意图选模板，把论文术语、章节标题与意图关键词拼成改写查询。"""
        if intent_profile is None:
            intent_profile = self.build_intent_profile(user_query, paper_context=paper_context)
        main_intent = intent_profile.main_intent
        paper_terms = _clean_str_list(paper_context.get("candidate_terms"))
        section_titles = _clean_str_list(paper_context.get("section_titles"))
        preferred_sections = self.preferred_sections_for_main_intent(main_intent)
        term_focus = self.compact_terms(paper_terms, limit=5)
        section_focus = self.compact_terms(section_titles, limit=4)
        type_terms = self.main_intent_terms(main_intent)

        def join_parts(parts: List[str]) -> str:
            """拼接查询片段并去重；拼出来为空时退回原问题。"""
            return self.dedupe_terms([part for part in parts if part]).strip() or user_query.strip()

        # 每个模板是 (查询片段, focus 说明)；贡献类与概览类问题共用同一组模板。
        overview_templates = [
            ([*term_focus[:3], "summary", "overview", "contribution"], "overview"),
            ([*term_focus[:3], "abstract", "introduction", "conclusion"], "paper arc"),
        ]
        templates_by_intent = {
            "contribution": overview_templates,
            "paper_overview": overview_templates,
            "method_flow": [
                ([*term_focus[:3], "method", "framework", "architecture"], "method overview"),
                ([*term_focus[:3], "training", "inference", "implementation"], "technical details"),
                ([*section_focus[:2], "approach", "model", "pipeline"], "paper structure"),
            ],
            "experiment_setup": [
                ([*term_focus[:3], "experiment", "dataset", "baseline"], "setup"),
                ([*term_focus[:3], "evaluation", "metric", "implementation"], "evaluation details"),
                ([*section_focus[:2], "ablation", "results", "benchmark"], "experiment sections"),
            ],
            "result_analysis": [
                ([*term_focus[:3], "results", "performance", "comparison"], "results"),
                ([*term_focus[:3], "ablation", "analysis", "effect"], "analysis"),
                ([*section_focus[:2], "table", "figure", "result"], "result evidence"),
            ],
            "comparison": [
                ([*term_focus[:3], "results", "performance", "comparison"], "results"),
                ([*term_focus[:3], "baseline", "ablation", "effect"], "comparison evidence"),
                ([*section_focus[:2], "table", "figure", "result"], "tables and figures"),
            ],
            "dataset": [
                ([*term_focus[:3], "dataset", "corpus", "benchmark"], "data source"),
                ([*term_focus[:3], "data split", "training data", "evaluation"], "data splits"),
                ([*section_focus[:2], "dataset", "setup", "experiment"], "dataset section"),
            ],
            "limitation": [
                ([*term_focus[:3], "limitation", "future work", "constraint"], "limitations"),
                ([*term_focus[:3], "failure case", "assumption", "weakness"], "failure cases"),
                ([*section_focus[:2], "discussion", "appendix", "future work"], "discussion"),
            ],
            "figure_table": [
                ([*term_focus[:3], "figure", "table", "diagram"], "visuals"),
                ([*term_focus[:3], "figure", "table", "result"], "figure or table caption"),
                ([*section_focus[:2], "appendix", "results", "experiment"], "visual evidence"),
            ],
            "other": [
                ([*term_focus[:4], *type_terms[:2]], "semantic"),
                ([*term_focus[:3], *section_focus[:2], *type_terms[2:4]], "section-aware"),
                ([user_query, *term_focus[:2], *type_terms[:3]], "query expansion"),
            ],
        }
        # 新 intent 只替换 schema 名称；同属方法类的 intent 继续复用原 method 策略，避免清理时顺带调参。
        template_intent = "method_flow" if main_intent in METHOD_INTENTS else main_intent
        queries = [
            {"query": join_parts(parts), "focus": focus, "channels": ["vector", "keyword"]}
            for parts, focus in templates_by_intent.get(template_intent, templates_by_intent["other"])
        ]
        return {
            "paper_terms": term_focus,
            "preferred_sections": preferred_sections,
            # 改写条数由意图服务决定（intent_profile 在函数开头已确保非空）。
            "rewrite_queries": queries[: intent_profile.rewrite_count],
            "paper_title": str(paper_context.get("title", "") or "").strip(),
            "paper_abstract": str(paper_context.get("abstract", "") or "").strip(),
            "section_titles": section_titles,
        }

    def build_query_views_from_plan(
        self,
        user_query: str,
        query_plan: Dict[str, Any],
        keywords: List[str],
        intent_tags: List[str],
        language: str,
        paper_context: Dict[str, Any],
        intent_profile: Optional[IntentProfile] = None,
    ) -> tuple[str, str, str]:
        """生成 (semantic_query, evidence_query, keyword_query)。

        检索计划里有改写查询时依次取前三条（不足三条时 evidence 复用第一条、keyword 由术语拼接）；
        没有计划或某项为空时，再用“问题关键词 + 论文术语”按规则构造。
        """
        plan_queries = self.extract_plan_queries(query_plan)
        paper_terms = _clean_str_list(query_plan.get("paper_terms"))
        section_titles = _clean_str_list(query_plan.get("section_titles"))
        main_intent = str((intent_profile.main_intent if intent_profile else "other") or "other").strip() or "other"
        if not paper_terms:
            paper_terms = self.extract_paper_terms_from_text(
                " ".join(
                    [
                        str(paper_context.get("title", "") or ""),
                        str(paper_context.get("abstract", "") or ""),
                        " ".join(section_titles),
                    ]
                ),
                limit=6,
            )
        semantic_query = evidence_query = keyword_query = ""
        if plan_queries:
            semantic_query = plan_queries[0]
            evidence_query = plan_queries[1] if len(plan_queries) > 1 else plan_queries[0]
            keyword_query = plan_queries[2] if len(plan_queries) > 2 else " ".join(
                self.dedupe_list([*paper_terms[:4], *keywords[:4], main_intent])
            ).strip()
        # 没有计划、或计划给出的某项为空时，用规则补齐（规则构造是确定性的，只需调用一次）。
        query_terms = keywords + paper_terms
        if not semantic_query:
            semantic_query = self.build_semantic_query(user_query, query_terms, intent_tags, intent_profile=intent_profile)
        if not evidence_query:
            evidence_query = self.build_evidence_query(query_terms, intent_tags, language, intent_profile=intent_profile)
        if not keyword_query:
            keyword_query = self.build_keyword_query(query_terms, intent_tags, intent_profile=intent_profile)
        return semantic_query, evidence_query, keyword_query

    def heuristic_query_rewrites(self, query_profile: QueryProfile) -> List[str]:
        """启发式改写：三种改写查询 + “论文术语 + 意图模板词”，去重后取前 QUERY_VIEW_LIMIT 条。"""
        rewrites = [
            item
            for item in [query_profile.semantic_query, query_profile.evidence_query, query_profile.keyword_query]
            if item
        ]
        main_intent = query_profile.intent_profile.main_intent if query_profile.intent_profile else "other"
        paper_terms = query_profile.paper_terms[:4]
        type_templates = {
            "method_flow": ["method framework algorithm training inference", "architecture component pipeline implementation"],
            "experiment_setup": ["experiment dataset baseline metric implementation", "evaluation setup data split benchmark"],
            "result_analysis": ["results performance comparison ablation analysis", "result table figure effect improvement"],
            "contribution": ["main contribution novel proposed method", "key idea summary contribution overview"],
            "limitation": ["limitations future work failure cases", "discussion constraints assumptions weaknesses"],
            "dataset": ["dataset corpus benchmark data split", "training data evaluation dataset"],
            "figure_table": ["figure table diagram caption", "table figure result appendix"],
            "paper_overview": ["abstract introduction conclusion summary", "main findings key contribution overview"],
            "other": ["paper evidence section relevant passages", "retrieval relevant chunks academic paper"],
        }
        # 方法类正式 intent 共用原有 method rewrite，schema 拆分不改变启发式检索词。
        template_intent = "method_flow" if main_intent in METHOD_INTENTS else main_intent
        for template in type_templates.get(template_intent, type_templates["other"]):
            rewrites.append(" ".join(self.dedupe_list([*paper_terms, template])))
        return self.dedupe_list(rewrites)[:QUERY_VIEW_LIMIT]

    def extract_plan_queries(self, query_plan: Dict[str, Any]) -> List[str]:
        """从检索计划的 rewrite_queries 中取出查询文本（兼容 dict 与纯字符串两种元素），去重并截断。"""
        rewrites: List[str] = []
        raw_queries = query_plan.get("rewrite_queries", [])
        if not isinstance(raw_queries, list):
            return rewrites
        for item in raw_queries:
            if isinstance(item, dict):
                query = str(item.get("query", "")).strip()
                if query:
                    rewrites.append(query)
            elif isinstance(item, str) and item.strip():
                rewrites.append(item.strip())
        return self.dedupe_list(rewrites)[:QUERY_VIEW_LIMIT]

    def merge_candidate_terms(self, existing_terms: List[str], texts: List[str], limit: Optional[int] = None) -> List[str]:
        """在已有术语后追加从各段文本中抽取的新术语，总数达到 limit 即停止（texts 按优先级排列）。"""
        limit = int(limit or self.config["merge_candidate_terms_limit"])
        terms: List[str] = _clean_str_list(existing_terms)
        for text in texts:
            if not text:
                continue
            for token in self.extract_paper_terms_from_text(text):
                if token not in terms:
                    terms.append(token)
                if len(terms) >= limit:
                    return terms[:limit]
        return terms[:limit]

    # ---- 以下为委托给 RetrievalRules 的薄封装，同时构成其它检索组件使用的 query_tools 接口 ----

    def extract_paper_terms_from_text(self, text: str, limit: Optional[int] = None) -> List[str]:
        """从论文文本中抽取候选术语。"""
        return self.retrieval_rules.extract_paper_terms_from_text(text, limit=limit)

    def preferred_sections_for_main_intent(self, main_intent: str) -> List[str]:
        """主意图对应的偏好章节。"""
        return self.retrieval_rules.preferred_sections_for_main_intent(main_intent)

    def main_intent_terms(self, main_intent: str) -> List[str]:
        """主意图对应的检索提示词。"""
        return self.retrieval_rules.main_intent_terms(main_intent)

    def preferred_section_tags_from_plan(self, query_plan: Dict[str, Any], main_intent: str) -> List[str]:
        """从检索计划与主意图推导偏好章节标签。"""
        return self.retrieval_rules.preferred_section_tags_from_plan(query_plan, main_intent)

    @staticmethod
    def compact_terms(terms: List[str], limit: int) -> List[str]:
        """去重并截取前 limit 个术语。"""
        return RetrievalRules.compact_terms(terms, limit)

    def build_semantic_query(
        self,
        user_query: str,
        keywords: List[str],
        intent_tags: List[str],
        intent_profile: Optional[IntentProfile] = None,
    ) -> str:
        """构造面向向量检索的 semantic 改写查询。"""
        return self.retrieval_rules.build_semantic_query(user_query, keywords, intent_tags, intent_profile=intent_profile)

    def build_evidence_query(
        self,
        keywords: List[str],
        intent_tags: List[str],
        language: str,
        intent_profile: Optional[IntentProfile] = None,
    ) -> str:
        """构造强调证据段落的 evidence 改写查询。"""
        return self.retrieval_rules.build_evidence_query(keywords, intent_tags, language, intent_profile=intent_profile)

    def build_keyword_query(
        self,
        keywords: List[str],
        intent_tags: List[str],
        intent_profile: Optional[IntentProfile] = None,
    ) -> str:
        """构造面向 BM25 的 keyword 改写查询。"""
        return self.retrieval_rules.build_keyword_query(keywords, intent_tags, intent_profile=intent_profile)

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

    def expand_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        """同义词/缩写扩展关键词 token。"""
        return self.retrieval_rules.expand_keyword_query_tokens(tokens)

    def build_query_term_details(self, queries: List[str]) -> List[Dict[str, Any]]:
        """逐条查询列出去重后的分词结果（debug 用）。"""
        return self.retrieval_rules.build_query_term_details(queries)

    @staticmethod
    def normalize_query_text(text: str) -> str:
        """归一化查询文本（去首尾空白、小写、压缩连续空白）。"""
        return RetrievalRules.normalize_query_text(text)

    def dedupe_terms(self, terms: List[str]) -> str:
        """归一化去重后用空格拼成一个查询串。"""
        return self.retrieval_rules.dedupe_terms(terms)

    def dedupe_list(self, items: List[str]) -> List[str]:
        """按归一化结果保序去重，返回原始写法。"""
        return self.retrieval_rules.dedupe_list(items)

    @staticmethod
    def debug_intent_profile(intent_profile: Optional[IntentProfile]) -> Optional[Dict[str, Any]]:
        """意图结果的可序列化形式（用于 rewrite debug）。"""
        if intent_profile is None:
            return None
        return intent_profile.to_dict()
