from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from services.intent.intent_service import IntentProfile, IntentService
from services.retrieval.contracts import QueryProfile
from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()
QUERY_VIEW_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_view_limit"]
QUERY_PLAN_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_plan_limit"]

EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "in", "into", "is", "it", "of", "on", "or", "paper",
    "please", "should", "summarize", "summarise", "tell", "that", "the", "their", "this",
    "to", "was", "what", "which", "with", "would", "you",
}
ZH_STOPWORDS = {"的", "了", "吗", "呢", "请", "这篇", "论文", "本文", "该文", "这个", "那个", "如何", "什么", "总结", "说明", "一下"}

INTENT_RULES = {
    "summary": {
        "keywords": ["summary", "summarize", "summarise", "overview", "contribution", "contributions", "finding", "findings", "核心", "总结", "贡献", "概述", "要点"],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "method": {
        "keywords": ["method", "methods", "approach", "framework", "architecture", "model", "training", "implementation", "方法", "模型", "框架", "架构", "训练", "实现"],
        "preferred_sections": ["method", "approach", "model", "architecture"],
    },
    "experiment": {
        "keywords": ["experiment", "experiments", "evaluation", "result", "results", "benchmark", "benchmarks", "metric", "metrics", "实验", "结果", "评估", "基准"],
        "preferred_sections": ["experiment", "results", "evaluation", "ablation"],
    },
    "comparison": {
        "keywords": ["baseline", "baselines", "compare", "comparison", "ablation", "compared", "对比", "比较", "基线", "消融"],
        "preferred_sections": ["experiment", "results", "ablation"],
    },
    "limitation": {
        "keywords": ["limitation", "limitations", "weakness", "future work", "failure", "局限", "限制", "不足", "未来工作"],
        "preferred_sections": ["conclusion", "discussion", "limitations", "appendix"],
    },
    "definition": {
        "keywords": ["definition", "define", "what is", "problem setup", "formulation", "定义", "概念", "任务定义", "问题设定"],
        "preferred_sections": ["introduction", "background", "method"],
    },
    "dataset": {
        "keywords": ["dataset", "datasets", "corpus", "data", "training set", "测试集", "数据集", "语料"],
        "preferred_sections": ["experiment", "dataset", "data"],
    },
}

QUESTION_TYPE_RULES = {
    "method_flow": {
        "keywords": ["method", "methods", "approach", "framework", "workflow", "pipeline", "algorithm", "model", "architecture", "training", "inference", "流程", "方法", "框架", "模型", "算法"],
        "preferred_sections": ["method", "approach", "model", "architecture", "introduction"],
    },
    "experiment_setup": {
        "keywords": ["experiment", "experiments", "setup", "evaluation", "dataset", "benchmark", "baseline", "metric", "implementation", "实验", "设置", "数据集", "基准", "评估"],
        "preferred_sections": ["experiment", "evaluation", "dataset", "implementation"],
    },
    "results_analysis": {
        "keywords": ["result", "results", "performance", "ablation", "comparison", "baseline", "finding", "结果", "性能", "消融", "对比"],
        "preferred_sections": ["results", "experiment", "evaluation", "ablation"],
    },
    "contribution": {
        "keywords": ["contribution", "novelty", "main idea", "innovation", "key idea", "贡献", "创新", "核心思想"],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
    },
    "limitation": {
        "keywords": ["limitation", "limitations", "weakness", "future work", "failure", "局限", "不足", "未来工作"],
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
    "other": {"keywords": [], "preferred_sections": ["abstract", "introduction", "method", "results"]},
}


class QueryPlanner:
    """负责 query profile、query views 与 rerank query，不执行任何召回。"""

    def __init__(
        self,
        *,
        vector_store_service: Any,
        generation_service: Any,
        intent_service: IntentService,
        rerank_service: Any,
        route_weights: Dict[str, float],
        enhanced_config: Dict[str, Any],
    ) -> None:
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.intent_service = intent_service
        self.rerank_service = rerank_service
        self.route_weights = route_weights
        self.config = enhanced_config

    def build_intent_profile(self, user_query: str, paper_context: Optional[Dict[str, Any]] = None) -> IntentProfile:
        return self.intent_service.build_intent_profile(user_query, paper_context=paper_context or {})

    def build_query_profile(
        self,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        intent_profile: Optional[IntentProfile] = None,
    ) -> QueryProfile:
        normalized_query = self.normalize_query_text(user_query)
        tokens = self.tokenize_for_keyword_search(user_query)
        keywords = self.extract_query_keywords(tokens)
        language = self.detect_language(user_query, tokens)
        paper_context_payload = self.build_paper_context(collection_name, paper_context=paper_context)
        intent_profile = intent_profile or self.build_intent_profile(user_query, paper_context=paper_context)
        query_plan = self.build_query_plan(user_query, paper_context_payload, intent_profile=intent_profile)
        question_type = str(intent_profile.main_intent or query_plan.get("question_type", "other")).strip() or "other"
        intent_tags = list(intent_profile.sub_intents)
        intent_summary = str(intent_profile.intent_summary or query_plan.get("intent_summary", "")).strip()
        paper_terms = [str(item).strip() for item in query_plan.get("paper_terms", []) if str(item).strip()]
        section_preferences = self.preferred_section_tags_from_plan(query_plan, intent_tags)
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
            question_type=question_type,
            intent_summary=intent_summary,
            paper_terms=paper_terms,
            ambiguity_score=intent_profile.ambiguity_score,
            semantic_query=semantic_query,
            evidence_query=evidence_query,
            keyword_query=keyword_query,
            section_preferences=section_preferences,
            query_plan=query_plan,
        )

    def build_query_views(self, user_query: str, query_profile: QueryProfile, enable_query_rewrite: bool) -> Dict[str, Any]:
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
            row = {"query": stripped, "source": source, "source_index": idx, "normalized": normalized, "selected": False, "reason": "kept"}
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
            selected_queries.append(query_profile.semantic_query)

        rewrite_debug = {
            "enabled": enable_query_rewrite,
            "original_query": user_query,
            "intent_profile": self.debug_intent_profile(query_profile.intent_profile),
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
        return {
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
            "view_queries": rewrite_debug["view_queries"],
            "rewrite_debug": rewrite_debug,
        }

    def build_rerank_query(self, user_query: str, query_profile: QueryProfile) -> str:
        return self.rerank_service.build_rerank_query(user_query, query_profile)

    def build_query_bundle(
        self,
        *,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        enable_query_rewrite: bool,
    ) -> Dict[str, Any]:
        intent_profile = self.build_intent_profile(user_query, paper_context=paper_context)
        query_profile = self.build_query_profile(user_query, collection_name, paper_context=paper_context, intent_profile=intent_profile)
        query_views = self.build_query_views(user_query, query_profile, enable_query_rewrite)
        rerank_query = self.build_rerank_query(user_query, query_profile)
        return {"intent_profile": intent_profile, "query_profile": query_profile, "query_views": query_views, "rerank_query": rerank_query}

    def build_paper_context(self, collection_name: str, paper_context: Optional[Dict[str, Any]] = None, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        sample_limit = int(sample_limit or self.config["sample_limit"])
        merged: Dict[str, Any] = {"title": "", "abstract": "", "section_titles": [], "candidate_terms": [], "source_samples": []}
        if paper_context:
            merged["title"] = str(paper_context.get("title", "") or "").strip()
            merged["abstract"] = str(paper_context.get("abstract", "") or "").strip()
            merged["section_titles"] = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
            merged["candidate_terms"] = [str(item).strip() for item in (paper_context.get("candidate_terms", []) or []) if str(item).strip()]
        try:
            sample_chunks = self.vector_store_service.get_all_chunks(collection_name, limit=sample_limit)
        except Exception:
            sample_chunks = []
        section_titles = list(merged["section_titles"])
        source_samples: List[str] = []
        abstract_candidates: List[str] = []
        for chunk in sample_chunks:
            title = str(chunk.get("title", "") or "").strip()
            content = str(chunk.get("content", "") or "").strip()
            section_title = str(chunk.get("section_title") or chunk.get("content_part_label") or chunk.get("subchunk_label") or "").strip()
            if not merged["title"] and title:
                merged["title"] = title
            if section_title:
                normalized_section = self.normalize_query_text(section_title)
                if normalized_section and normalized_section not in {self.normalize_query_text(item) for item in section_titles}:
                    section_titles.append(section_title)
            if content and len(source_samples) < self.config["source_sample_limit"]:
                source_samples.append(content[: self.config["source_sample_primary_limit"]])
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
            [merged["title"], merged["abstract"][:900], " ".join(merged["section_titles"][:20]), " ".join(source_samples[: self.config["source_sample_limit"]])],
        )
        merged["source_samples"] = source_samples[: self.config["source_sample_limit"]]
        return merged

    def build_query_plan(self, user_query: str, paper_context: Dict[str, Any], intent_profile: Optional[IntentProfile] = None) -> Dict[str, Any]:
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
                        "question_type": "other",
                        "intent_summary": intent_profile.intent_summary if intent_profile else "",
                        "paper_terms": paper_context.get("candidate_terms", [])[: self.config["paper_terms_preview_limit"]],
                        "preferred_sections": intent_profile.preferred_sections if intent_profile else [],
                        "rewrite_queries": [{"query": query, "focus": "retrieval", "channels": ["vector", "keyword"]} for query in rewrites],
                        "main_intent": intent_profile.main_intent if intent_profile else "other",
                        "sub_intents": intent_profile.sub_intents if intent_profile else [],
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

    def normalize_query_plan(self, query_plan: Dict[str, Any], user_query: str, paper_context: Dict[str, Any], intent_profile: Optional[IntentProfile] = None) -> Dict[str, Any]:
        normalized = dict(query_plan or {})
        rewrite_queries = self.extract_plan_queries(normalized)
        if not rewrite_queries:
            normalized = self.heuristic_query_plan(user_query, paper_context, intent_profile=intent_profile)
            rewrite_queries = self.extract_plan_queries(normalized)
        normalized["rewrite_queries"] = rewrite_queries
        normalized["question_type"] = str(normalized.get("question_type", "other")).strip() or "other"
        normalized["intent_summary"] = str(normalized.get("intent_summary", "")).strip()
        normalized["paper_terms"] = self.dedupe_list([str(item).strip() for item in (normalized.get("paper_terms", []) or []) if str(item).strip()])
        normalized["preferred_sections"] = self.dedupe_list([str(item).strip() for item in (normalized.get("preferred_sections", []) or []) if str(item).strip()])
        normalized["paper_title"] = str(paper_context.get("title", "") or "").strip()
        normalized["paper_abstract"] = str(paper_context.get("abstract", "") or "").strip()
        normalized["section_titles"] = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
        if intent_profile is not None:
            normalized["question_type"] = intent_profile.main_intent
            normalized["main_intent"] = intent_profile.main_intent
            normalized["sub_intents"] = list(intent_profile.sub_intents)
            normalized["intent_confidence"] = intent_profile.confidence
            normalized["intent_fallback_reason"] = intent_profile.fallback_reason
            normalized["preferred_sections"] = self.dedupe_list([*intent_profile.preferred_sections, *normalized.get("preferred_sections", [])])
            normalized["route_weights"] = dict(intent_profile.route_weights)
            normalized["rewrite_count"] = intent_profile.rewrite_count
            normalized["use_keyword_search"] = intent_profile.use_keyword_search
            normalized["use_hyde"] = intent_profile.use_hyde
        return normalized

    def heuristic_query_plan(self, user_query: str, paper_context: Dict[str, Any], intent_profile: Optional[IntentProfile] = None) -> Dict[str, Any]:
        normalized_query = self.normalize_query_text(user_query)
        if intent_profile is None:
            intent_profile = self.build_intent_profile(user_query, paper_context=paper_context)
        intent_tags = list(intent_profile.sub_intents)
        question_type = self.legacy_intent_bucket(intent_profile.main_intent or self.classify_question_type(normalized_query, intent_tags))
        paper_terms = [str(item).strip() for item in (paper_context.get("candidate_terms", []) or []) if str(item).strip()]
        section_titles = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
        preferred_sections = self.preferred_sections_for_question_type(question_type, intent_tags)
        term_focus = self.compact_terms(paper_terms, limit=5)
        section_focus = self.compact_terms(section_titles, limit=4)
        type_terms = self.query_type_terms(question_type)

        def join_parts(parts: List[str]) -> str:
            return self.dedupe_terms([part for part in parts if part]).strip() or user_query.strip()

        templates_by_intent = {
            "summary": [([*term_focus[:3], "summary", "overview", "contribution"], "overview"), ([*term_focus[:3], "abstract", "introduction", "conclusion"], "paper arc")],
            "method": [([*term_focus[:3], "method", "framework", "architecture"], "method overview"), ([*term_focus[:3], "training", "inference", "implementation"], "technical details"), ([*section_focus[:2], "approach", "model", "pipeline"], "paper structure")],
            "experiment": [([*term_focus[:3], "experiment", "dataset", "baseline"], "setup"), ([*term_focus[:3], "evaluation", "metric", "implementation"], "evaluation details"), ([*section_focus[:2], "ablation", "results", "benchmark"], "experiment sections")],
            "comparison": [([*term_focus[:3], "results", "performance", "comparison"], "results"), ([*term_focus[:3], "baseline", "ablation", "effect"], "comparison evidence"), ([*section_focus[:2], "table", "figure", "result"], "tables and figures")],
            "dataset": [([*term_focus[:3], "dataset", "corpus", "benchmark"], "data source"), ([*term_focus[:3], "data split", "training data", "evaluation"], "data splits"), ([*section_focus[:2], "dataset", "setup", "experiment"], "dataset section")],
            "limitation": [([*term_focus[:3], "limitation", "future work", "constraint"], "limitations"), ([*term_focus[:3], "failure case", "assumption", "weakness"], "failure cases"), ([*section_focus[:2], "discussion", "appendix", "future work"], "discussion")],
            "figure_table": [([*term_focus[:3], "figure", "table", "diagram"], "visuals"), ([*term_focus[:3], "figure", "table", "result"], "figure or table caption"), ([*section_focus[:2], "appendix", "results", "experiment"], "visual evidence")],
            "other": [([*term_focus[:4], *type_terms[:2]], "semantic"), ([*term_focus[:3], *section_focus[:2], *type_terms[2:4]], "section-aware"), ([user_query, *term_focus[:2], *type_terms[:3]], "query expansion")],
        }
        queries = [{"query": join_parts(parts), "focus": focus, "channels": ["vector", "keyword"]} for parts, focus in templates_by_intent.get(question_type, templates_by_intent["other"])]
        rewrite_limit = intent_profile.rewrite_count if intent_profile else QUERY_VIEW_LIMIT
        return {
            "question_type": question_type,
            "intent_summary": intent_profile.intent_summary if intent_profile else self.summarize_intent(question_type, intent_tags, term_focus),
            "paper_terms": term_focus,
            "preferred_sections": preferred_sections,
            "rewrite_queries": queries[:rewrite_limit],
            "paper_title": str(paper_context.get("title", "") or "").strip(),
            "paper_abstract": str(paper_context.get("abstract", "") or "").strip(),
            "section_titles": section_titles,
            "main_intent": question_type,
            "sub_intents": intent_tags,
            "intent_confidence": intent_profile.confidence if intent_profile else None,
            "intent_fallback_reason": intent_profile.fallback_reason if intent_profile else "",
            "route_weights": intent_profile.route_weights if intent_profile else self.route_weights,
            "rewrite_count": intent_profile.rewrite_count if intent_profile else len(queries),
        }

    def build_query_views_from_plan(self, user_query: str, query_plan: Dict[str, Any], keywords: List[str], intent_tags: List[str], language: str, paper_context: Dict[str, Any], intent_profile: Optional[IntentProfile] = None) -> tuple[str, str, str]:
        plan_queries = self.extract_plan_queries(query_plan)
        paper_terms = [str(item).strip() for item in (query_plan.get("paper_terms", []) or []) if str(item).strip()]
        section_titles = [str(item).strip() for item in (query_plan.get("section_titles", []) or []) if str(item).strip()]
        question_type = str((intent_profile.main_intent if intent_profile else query_plan.get("question_type", "other")) or "other").strip() or "other"
        if not paper_terms:
            paper_terms = self.extract_paper_terms_from_text(" ".join([str(paper_context.get("title", "") or ""), str(paper_context.get("abstract", "") or ""), " ".join(section_titles)]), limit=6)
        if plan_queries:
            semantic_query = plan_queries[0]
            evidence_query = plan_queries[1] if len(plan_queries) > 1 else plan_queries[0]
            keyword_query = plan_queries[2] if len(plan_queries) > 2 else " ".join(self.dedupe_list([*paper_terms[:4], *keywords[:4], question_type])).strip()
        else:
            semantic_query = self.build_semantic_query(user_query, keywords + paper_terms, intent_tags, intent_profile=intent_profile)
            evidence_query = self.build_evidence_query(keywords + paper_terms, intent_tags, language, intent_profile=intent_profile)
            keyword_query = self.build_keyword_query(keywords + paper_terms, intent_tags, intent_profile=intent_profile)
        if not semantic_query:
            semantic_query = self.build_semantic_query(user_query, keywords + paper_terms, intent_tags, intent_profile=intent_profile)
        if not evidence_query:
            evidence_query = self.build_evidence_query(keywords + paper_terms, intent_tags, language, intent_profile=intent_profile)
        if not keyword_query:
            keyword_query = self.build_keyword_query(keywords + paper_terms, intent_tags, intent_profile=intent_profile)
        return semantic_query, evidence_query, keyword_query

    def heuristic_query_rewrites(self, query_profile: QueryProfile) -> List[str]:
        rewrites = [item for item in [query_profile.semantic_query, query_profile.evidence_query, query_profile.keyword_query] if item]
        question_type = query_profile.question_type or "other"
        paper_terms = query_profile.paper_terms[:4]
        type_templates = {
            "method_flow": ["method framework algorithm training inference", "architecture component pipeline implementation"],
            "experiment_setup": ["experiment dataset baseline metric implementation", "evaluation setup data split benchmark"],
            "results_analysis": ["results performance comparison ablation analysis", "result table figure effect improvement"],
            "contribution": ["main contribution novel proposed method", "key idea summary contribution overview"],
            "limitation": ["limitations future work failure cases", "discussion constraints assumptions weaknesses"],
            "dataset": ["dataset corpus benchmark data split", "training data evaluation dataset"],
            "metric": ["metric formula evaluation objective", "score measure evaluation protocol"],
            "figure_table": ["figure table diagram caption", "table figure result appendix"],
            "summary": ["abstract introduction conclusion summary", "main findings key contribution overview"],
            "other": ["paper evidence section relevant passages", "retrieval relevant chunks academic paper"],
        }
        for template in type_templates.get(question_type, type_templates["other"]):
            rewrites.append(" ".join(self.dedupe_list([*paper_terms, template])))
        return self.dedupe_list(rewrites)[:QUERY_VIEW_LIMIT]

    def extract_plan_queries(self, query_plan: Dict[str, Any]) -> List[str]:
        rewrites: List[str] = []
        raw_queries = query_plan.get("rewrite_queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = query_plan.get("queries", [])
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
        limit = int(limit or self.config["merge_candidate_terms_limit"])
        terms: List[str] = [str(item).strip() for item in existing_terms if str(item).strip()]
        for text in texts:
            if not text:
                continue
            for token in self.extract_paper_terms_from_text(text):
                if token not in terms:
                    terms.append(token)
                if len(terms) >= limit:
                    return terms[:limit]
        return terms[:limit]

    def extract_paper_terms_from_text(self, text: str, limit: Optional[int] = None) -> List[str]:
        limit = int(limit or self.config["extract_paper_terms_limit"])
        tokens = self.tokenize_for_keyword_search(text)
        filtered = [token for token in tokens if token not in EN_STOPWORDS and token not in ZH_STOPWORDS]
        seen: List[str] = []
        for token in filtered:
            if token not in seen:
                seen.append(token)
        return seen[:limit]

    def preferred_sections_for_question_type(self, question_type: str, intent_tags: List[str]) -> List[str]:
        alias = {"method": "method_flow", "experiment": "experiment_setup", "result_analysis": "results_analysis", "comparison": "results_analysis", "definition": "summary", "implementation_detail": "method_flow", "paper_overview": "summary"}
        preferred = list(QUESTION_TYPE_RULES.get(alias.get(question_type, question_type), QUESTION_TYPE_RULES["other"]).get("preferred_sections", []))
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
        alias = {"paper_overview": "summary", "method": "method_flow", "experiment": "experiment_setup", "result_analysis": "results_analysis", "comparison": "results_analysis", "definition": "summary", "implementation_detail": "method_flow"}
        spec = QUESTION_TYPE_RULES.get(alias.get(question_type, question_type), QUESTION_TYPE_RULES["other"])
        return self.dedupe_list([str(item).strip() for item in spec.get("keywords", []) if str(item).strip()])

    def preferred_section_tags_from_plan(self, query_plan: Dict[str, Any], intent_tags: List[str]) -> List[str]:
        preferred = [str(item).strip() for item in (query_plan.get("preferred_sections", []) or []) if str(item).strip()]
        question_type = self.legacy_intent_bucket(query_plan.get("question_type", "other"))
        preferred.extend(QUESTION_TYPE_RULES.get(question_type, QUESTION_TYPE_RULES["other"]).get("preferred_sections", []))
        preferred.extend(self.preferred_section_tags(intent_tags))
        return self.dedupe_list(preferred)[:6]

    @staticmethod
    def legacy_intent_bucket(intent: str) -> str:
        aliases = {"contribution": "summary", "paper_overview": "summary", "method_flow": "method", "implementation_detail": "method", "definition": "method", "experiment_setup": "experiment", "result_analysis": "experiment", "results_analysis": "experiment", "comparison": "comparison", "dataset": "dataset", "limitation": "limitation", "figure_table": "figure_table", "other": "other", "summary": "summary", "method": "method", "experiment": "experiment"}
        intent = str(intent or "other").strip().lower() or "other"
        return aliases.get(intent, intent)

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
        summaries = {"paper_overview": "understand the paper overview and key ideas", "contribution": "understand the paper's main contribution and novelty", "method_flow": "understand the method flow and paper-specific implementation details", "experiment_setup": "understand the experimental setup, datasets, baselines, and evaluation details", "results_analysis": "understand the results, comparison, and ablation analysis", "limitation": "understand the limitations and future work", "dataset": "understand the dataset or benchmark used in the paper", "metric": "understand the metric, formula, or evaluation protocol", "figure_table": "find the relevant figure or table and interpret it", "summary": "summarize the paper around its main ideas and findings", "definition": "understand the definition or concept being asked about", "implementation_detail": "understand the implementation details and training settings"}
        normalized = {"method": "method_flow", "experiment": "experiment_setup", "comparison": "results_analysis"}.get(question_type, question_type)
        if normalized in summaries:
            return summaries[normalized]
        if intent_tags:
            return f"understand the paper with focus on {', '.join(intent_tags[:3])}"
        if paper_terms:
            return f"retrieve evidence around {', '.join(paper_terms[:3])}"
        return "retrieve the most relevant paper evidence"

    def build_semantic_query(self, user_query: str, keywords: List[str], intent_tags: List[str], intent_profile: Optional[IntentProfile] = None) -> str:
        parts: List[str] = []
        if intent_profile and intent_profile.rewrite_focus:
            parts.extend(intent_profile.rewrite_focus[:4])
        if intent_tags:
            parts.extend(self.intent_to_terms(intent_tags))
        parts.extend(keywords[:6])
        if not parts:
            parts.extend(self.tokenize_for_keyword_search(user_query)[:6])
        return self.dedupe_terms(parts) or self.normalize_query_text(user_query)

    def build_evidence_query(self, keywords: List[str], intent_tags: List[str], language: str, intent_profile: Optional[IntentProfile] = None) -> str:
        parts = self.intent_to_evidence_terms(intent_tags)
        if intent_profile and intent_profile.rerank_focus:
            parts.extend(intent_profile.rerank_focus[:4])
        parts.extend(keywords[:4])
        parts.extend(["论文", "证据", "段落"] if language == "zh" else ["paper", "evidence", "passage"])
        return self.dedupe_terms(parts)

    def build_keyword_query(self, keywords: List[str], intent_tags: List[str], intent_profile: Optional[IntentProfile] = None) -> str:
        parts = keywords[: self.config["keyword_parts_limit"]] + self.intent_to_terms(intent_tags)
        if intent_profile and intent_profile.rewrite_focus:
            parts.extend(intent_profile.rewrite_focus[:4])
        return self.dedupe_terms(parts or keywords[: self.config["keyword_parts_limit"]])

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

    def preferred_section_tags(self, intent_tags: List[str]) -> List[str]:
        preferred: List[str] = []
        for intent in intent_tags:
            preferred.extend(INTENT_RULES.get(intent, {}).get("preferred_sections", []))
        return self.dedupe_terms(preferred).split()

    def intent_to_terms(self, intent_tags: List[str]) -> List[str]:
        mapping = {"summary": ["main", "contributions", "key", "findings", "summary"], "method": ["proposed", "method", "approach", "architecture", "implementation"], "experiment": ["experimental", "results", "evaluation", "benchmarks"], "comparison": ["baseline", "comparison", "ablation", "competing"], "limitation": ["limitations", "future", "work", "failure", "cases"], "definition": ["definition", "formulation", "problem", "setup"], "dataset": ["datasets", "corpus", "data", "splits"]}
        terms: List[str] = []
        for intent in intent_tags:
            terms.extend(mapping.get(intent, []))
        return self.dedupe_terms(terms).split()

    def intent_to_evidence_terms(self, intent_tags: List[str]) -> List[str]:
        mapping = {"summary": ["abstract", "introduction", "conclusion"], "method": ["method", "approach", "architecture", "model"], "experiment": ["experiment", "results", "evaluation", "ablation"], "comparison": ["baseline", "comparison", "ablation", "results"], "limitation": ["limitations", "discussion", "future work"], "definition": ["background", "definition", "problem setup"], "dataset": ["dataset", "corpus", "data"]}
        terms: List[str] = []
        for intent in intent_tags:
            terms.extend(mapping.get(intent, []))
        return self.dedupe_terms(terms).split()

    def extract_query_keywords(self, tokens: List[str], limit: Optional[int] = None) -> List[str]:
        limit = int(limit or self.config["extract_query_keywords_limit"])
        filtered = [token for token in tokens if token not in EN_STOPWORDS and token not in ZH_STOPWORDS] or [token for token in tokens if len(token) > 1]
        seen = []
        for token in filtered:
            if token not in seen:
                seen.append(token)
        return seen[:limit]

    @staticmethod
    def tokenize_for_keyword_search(text: str) -> List[str]:
        lowered = text.lower()
        return re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", lowered) + re.findall(r"[\u4e00-\u9fff]{2,}", lowered)

    def build_query_keywords(self, queries: List[str], limit: Optional[int] = None) -> List[str]:
        limit = int(limit or self.config["build_query_keywords_limit"])
        counts: Dict[str, int] = {}
        for query in queries:
            for token in self.tokenize_for_keyword_search(query):
                counts[token] = counts.get(token, 0) + 1
        return [token for token, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]]

    def build_query_term_details(self, queries: List[str]) -> List[Dict[str, Any]]:
        return [{"query": query, "keywords": list(dict.fromkeys(self.tokenize_for_keyword_search(query))), "keyword_count": len(list(dict.fromkeys(self.tokenize_for_keyword_search(query))))} for query in queries]

    @staticmethod
    def normalize_query_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def dedupe_terms(self, terms: List[str]) -> str:
        seen = []
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

    @staticmethod
    def debug_intent_profile(intent_profile: Optional[IntentProfile]) -> Optional[Dict[str, Any]]:
        if intent_profile is None:
            return None
        return intent_profile.to_dict()
