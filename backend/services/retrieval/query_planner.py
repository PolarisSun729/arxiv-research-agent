from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.intent.intent_service import METHOD_INTENTS, IntentProfile, IntentService
from services.retrieval.contracts import QueryProfile
from services.retrieval.retrieval_rules import RetrievalRules
from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()
QUERY_VIEW_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_view_limit"]
QUERY_PLAN_LIMIT = ENHANCED_RETRIEVAL_CONFIG["query_plan_limit"]


class QueryPlanner:
    """负责 query profile、query views 和 rerank query，不执行任何召回。"""

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
        intent_tags = list(intent_profile.sub_intents)
        paper_terms = [str(item).strip() for item in query_plan.get("paper_terms", []) if str(item).strip()]
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
        query_profile = self.build_query_profile(
            user_query,
            collection_name,
            paper_context=paper_context,
            intent_profile=intent_profile,
        )
        query_views = self.build_query_views(user_query, query_profile, enable_query_rewrite)
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
            merged["section_titles"] = [
                str(item).strip()
                for item in (paper_context.get("section_titles", []) or [])
                if str(item).strip()
            ]
            merged["candidate_terms"] = [
                str(item).strip()
                for item in (paper_context.get("candidate_terms", []) or [])
                if str(item).strip()
            ]
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
        normalized["paper_terms"] = self.dedupe_list(
            [str(item).strip() for item in (normalized.get("paper_terms", []) or []) if str(item).strip()]
        )
        normalized["preferred_sections"] = self.dedupe_list(
            [str(item).strip() for item in (normalized.get("preferred_sections", []) or []) if str(item).strip()]
        )
        normalized["paper_title"] = str(paper_context.get("title", "") or "").strip()
        normalized["paper_abstract"] = str(paper_context.get("abstract", "") or "").strip()
        normalized["section_titles"] = [
            str(item).strip()
            for item in (paper_context.get("section_titles", []) or [])
            if str(item).strip()
        ]
        if intent_profile is not None:
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
        if intent_profile is None:
            intent_profile = self.build_intent_profile(user_query, paper_context=paper_context)
        intent_tags = list(intent_profile.sub_intents)
        main_intent = intent_profile.main_intent
        paper_terms = [str(item).strip() for item in (paper_context.get("candidate_terms", []) or []) if str(item).strip()]
        section_titles = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
        preferred_sections = self.preferred_sections_for_main_intent(main_intent)
        term_focus = self.compact_terms(paper_terms, limit=5)
        section_focus = self.compact_terms(section_titles, limit=4)
        type_terms = self.main_intent_terms(main_intent)

        def join_parts(parts: List[str]) -> str:
            return self.dedupe_terms([part for part in parts if part]).strip() or user_query.strip()

        templates_by_intent = {
            "contribution": [
                ([*term_focus[:3], "summary", "overview", "contribution"], "overview"),
                ([*term_focus[:3], "abstract", "introduction", "conclusion"], "paper arc"),
            ],
            "paper_overview": [
                ([*term_focus[:3], "summary", "overview", "contribution"], "overview"),
                ([*term_focus[:3], "abstract", "introduction", "conclusion"], "paper arc"),
            ],
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
        rewrite_limit = intent_profile.rewrite_count if intent_profile else QUERY_VIEW_LIMIT
        return {
            "paper_terms": term_focus,
            "preferred_sections": preferred_sections,
            "rewrite_queries": queries[:rewrite_limit],
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
        plan_queries = self.extract_plan_queries(query_plan)
        paper_terms = [str(item).strip() for item in (query_plan.get("paper_terms", []) or []) if str(item).strip()]
        section_titles = [str(item).strip() for item in (query_plan.get("section_titles", []) or []) if str(item).strip()]
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
        if plan_queries:
            semantic_query = plan_queries[0]
            evidence_query = plan_queries[1] if len(plan_queries) > 1 else plan_queries[0]
            keyword_query = plan_queries[2] if len(plan_queries) > 2 else " ".join(
                self.dedupe_list([*paper_terms[:4], *keywords[:4], main_intent])
            ).strip()
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
        return self.retrieval_rules.extract_paper_terms_from_text(text, limit=limit)

    def preferred_sections_for_main_intent(self, main_intent: str) -> List[str]:
        return self.retrieval_rules.preferred_sections_for_main_intent(main_intent)

    def main_intent_terms(self, main_intent: str) -> List[str]:
        return self.retrieval_rules.main_intent_terms(main_intent)

    def preferred_section_tags_from_plan(self, query_plan: Dict[str, Any], main_intent: str) -> List[str]:
        return self.retrieval_rules.preferred_section_tags_from_plan(query_plan, main_intent)

    @staticmethod
    def compact_terms(terms: List[str], limit: int) -> List[str]:
        return RetrievalRules.compact_terms(terms, limit)

    def build_semantic_query(
        self,
        user_query: str,
        keywords: List[str],
        intent_tags: List[str],
        intent_profile: Optional[IntentProfile] = None,
    ) -> str:
        return self.retrieval_rules.build_semantic_query(user_query, keywords, intent_tags, intent_profile=intent_profile)

    def build_evidence_query(
        self,
        keywords: List[str],
        intent_tags: List[str],
        language: str,
        intent_profile: Optional[IntentProfile] = None,
    ) -> str:
        return self.retrieval_rules.build_evidence_query(keywords, intent_tags, language, intent_profile=intent_profile)

    def build_keyword_query(
        self,
        keywords: List[str],
        intent_tags: List[str],
        intent_profile: Optional[IntentProfile] = None,
    ) -> str:
        return self.retrieval_rules.build_keyword_query(keywords, intent_tags, intent_profile=intent_profile)

    @staticmethod
    def detect_language(user_query: str, tokens: List[str]) -> str:
        return RetrievalRules.detect_language(user_query, tokens)

    def extract_query_keywords(self, tokens: List[str], limit: Optional[int] = None) -> List[str]:
        return self.retrieval_rules.extract_query_keywords(tokens, limit=limit)

    @staticmethod
    def tokenize_for_keyword_search(text: str) -> List[str]:
        return RetrievalRules.tokenize_for_keyword_search(text)

    def build_query_keywords(self, queries: List[str], limit: Optional[int] = None) -> List[str]:
        return self.retrieval_rules.build_query_keywords(queries, limit=limit)

    def expand_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        return self.retrieval_rules.expand_keyword_query_tokens(tokens)

    def build_query_term_details(self, queries: List[str]) -> List[Dict[str, Any]]:
        return self.retrieval_rules.build_query_term_details(queries)

    @staticmethod
    def normalize_query_text(text: str) -> str:
        return RetrievalRules.normalize_query_text(text)

    def dedupe_terms(self, terms: List[str]) -> str:
        return self.retrieval_rules.dedupe_terms(terms)

    def dedupe_list(self, items: List[str]) -> List[str]:
        return self.retrieval_rules.dedupe_list(items)

    @staticmethod
    def debug_intent_profile(intent_profile: Optional[IntentProfile]) -> Optional[Dict[str, Any]]:
        if intent_profile is None:
            return None
        return intent_profile.to_dict()
