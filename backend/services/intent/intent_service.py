import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from utils.config import RETRIEVAL_CONFIG, get_intent_routing_runtime_config


logger = logging.getLogger(__name__)


MAIN_INTENTS = {
    "summary": {
        "keywords": ["summary", "summarize", "summarise", "overview", "contribution", "findings", "main idea", "总结", "概述", "贡献"],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
        "rewrite_count": 2,
    },
    "method": {
        "keywords": ["method", "methods", "approach", "framework", "architecture", "model", "training", "inference", "方法", "框架", "模型"],
        "preferred_sections": ["method", "approach", "model", "architecture"],
        "rewrite_count": 3,
    },
    "experiment": {
        "keywords": ["experiment", "experiments", "evaluation", "result", "results", "benchmark", "metric", "ablation", "实验", "结果", "评估"],
        "preferred_sections": ["experiment", "results", "evaluation", "ablation"],
        "rewrite_count": 3,
    },
    "comparison": {
        "keywords": ["compare", "comparison", "baseline", "ablation", "compared", "versus", "比较", "对比", "基线"],
        "preferred_sections": ["experiment", "results", "ablation"],
        "rewrite_count": 3,
    },
    "dataset": {
        "keywords": ["dataset", "datasets", "corpus", "data", "benchmark", "split", "数据集", "数据", "语料"],
        "preferred_sections": ["experiment", "dataset", "data", "setup"],
        "rewrite_count": 3,
    },
    "limitation": {
        "keywords": ["limitation", "limitations", "weakness", "future work", "failure", "constraint", "局限", "不足", "未来工作"],
        "preferred_sections": ["discussion", "conclusion", "limitations", "appendix"],
        "rewrite_count": 2,
    },
    "figure_table": {
        "keywords": ["figure", "fig.", "table", "chart", "diagram", "caption", "图", "表", "图表"],
        "preferred_sections": ["figure", "table", "appendix", "results"],
        "rewrite_count": 2,
    },
    "other": {
        "keywords": [],
        "preferred_sections": ["abstract", "introduction", "method", "experiment", "conclusion"],
        "rewrite_count": 3,
    },
}


SUB_INTENT_RULES = {
    "paper_overview": {
        "keywords": ["overview", "summary", "contribution", "main", "high level"],
    },
    "evidence_seeking": {
        "keywords": ["evidence", "supporting", "passage", "grounding", "support"],
    },
    "result_check": {
        "keywords": ["result", "results", "performance", "metric", "score", "结果", "性能", "指标"],
    },
    "table_lookup": {
        "keywords": ["table", "figure", "caption", "chart", "图", "表", "图表"],
    },
    "deep_method": {
        "keywords": ["pipeline", "architecture", "training", "inference", "implementation", "流程", "架构", "训练", "推理"],
    },
}


DEFAULT_ROUTE_WEIGHTS = dict(RETRIEVAL_CONFIG["route_weights"])


INTENT_ROUTE_WEIGHTS = get_intent_routing_runtime_config()["intent_route_weights"]

# New schema used by the LLM router prompt. This overrides the legacy table above
# while keeping the old definitions around for reference until all consumers are migrated.
MAIN_INTENTS = {
    "contribution": {
        "keywords": ["contribution", "contributions", "novelty", "novel", "innovation", "main contribution", "propose", "proposed", "创新", "贡献", "提出"],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
        "rewrite_count": 2,
    },
    "paper_overview": {
        "keywords": ["summary", "summarize", "summarise", "overview", "overall", "main idea", "what is this paper about", "paper about", "概述", "总结", "梗概"],
        "preferred_sections": ["abstract", "introduction", "conclusion"],
        "rewrite_count": 2,
    },
    "method_flow": {
        "keywords": ["method", "methods", "approach", "framework", "architecture", "algorithm", "pipeline", "workflow", "model", "training", "inference", "方法", "流程", "框架", "模型", "算法"],
        "preferred_sections": ["method", "approach", "model", "architecture"],
        "rewrite_count": 3,
    },
    "experiment_setup": {
        "keywords": ["experiment", "experiments", "evaluation", "dataset", "baseline", "benchmarks", "benchmark", "metric", "metrics", "setup", "setting", "split", "实验", "数据集", "评估", "基准"],
        "preferred_sections": ["experiment", "dataset", "data", "setup"],
        "rewrite_count": 3,
    },
    "result_analysis": {
        "keywords": ["result", "results", "performance", "ablation", "analysis", "effect", "improve", "findings", "结果", "性能", "分析", "比较"],
        "preferred_sections": ["results", "evaluation", "ablation", "discussion"],
        "rewrite_count": 3,
    },
    "comparison": {
        "keywords": ["compare", "comparison", "baseline", "ablation", "compared", "versus", "对比", "比较", "基线"],
        "preferred_sections": ["results", "evaluation", "ablation"],
        "rewrite_count": 3,
    },
    "dataset": {
        "keywords": ["dataset", "datasets", "corpus", "data", "benchmark", "split", "数据集", "数据", "语料"],
        "preferred_sections": ["experiment", "dataset", "data", "setup"],
        "rewrite_count": 3,
    },
    "limitation": {
        "keywords": ["limitation", "limitations", "weakness", "future work", "failure", "constraint", "局限", "不足", "未来工作"],
        "preferred_sections": ["discussion", "conclusion", "limitations", "appendix"],
        "rewrite_count": 2,
    },
    "definition": {
        "keywords": ["definition", "define", "what is", "meaning", "concept", "notation", "formulation", "problem setup", "定义", "概念", "是什么"],
        "preferred_sections": ["introduction", "background", "method"],
        "rewrite_count": 2,
    },
    "implementation_detail": {
        "keywords": ["implementation", "hyperparameter", "hyperparameters", "prompt", "training detail", "training details", "optimizer", "batch size", "learning rate", "code", "cost", "实现", "训练细节", "参数"],
        "preferred_sections": ["appendix", "method", "implementation details"],
        "rewrite_count": 3,
    },
    "figure_table": {
        "keywords": ["figure", "fig.", "table", "chart", "diagram", "caption", "图", "表", "图表"],
        "preferred_sections": ["figure", "table", "appendix", "results"],
        "rewrite_count": 2,
    },
    "other": {
        "keywords": [],
        "preferred_sections": ["abstract", "introduction", "method", "experiment", "conclusion"],
        "rewrite_count": 3,
    },
}

INTENT_ALIASES = {
    "summary": "paper_overview",
    "method": "method_flow",
    "experiment": "experiment_setup",
    "results_analysis": "result_analysis",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _tokenize(text: str) -> List[str]:
    lowered = (text or "").lower()
    return re.findall(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", lowered)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = _normalize_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(item)
    return result


@dataclass
class IntentProfile:
    original_query: str
    normalized_query: str
    language: str
    main_intent: str
    sub_intents: List[str]
    confidence: float
    ambiguity_score: float
    intent_summary: str
    preferred_sections: List[str]
    avoid_sections: List[str]
    route_weights: Dict[str, float]
    rewrite_count: int
    use_keyword_search: bool
    use_hyde: bool
    rewrite_focus: List[str]
    rerank_focus: List[str]
    final_context_policy: Dict[str, Any]
    fallback_reason: str
    source: str
    raw_model_output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IntentService:
    def __init__(self, generation_service: Any = None):
        self.generation_service = generation_service

    def build_intent_profile(
        self,
        question: str,
        paper_context: Optional[Dict[str, Any]] = None,
    ) -> IntentProfile:
        paper_context = paper_context or {}
        normalized_query = _normalize_text(question)
        tokens = _tokenize(question)
        language = self._detect_language(question, tokens)

        llm_profile: Optional[Dict[str, Any]] = None
        llm_error = None
        if self.generation_service is not None and hasattr(self.generation_service, "complete_with_qwen"):
            try:
                llm_profile = self._build_llm_profile(question, paper_context)
            except Exception as exc:  # pragma: no cover - depends on network/model availability
                llm_error = str(exc)
                logger.debug("Intent LLM profile failed, falling back to heuristics: %s", exc)

        if llm_profile:
            return self._normalize_profile(llm_profile, question, paper_context, language, source="llm", fallback_reason="")

        heuristic_profile = self._heuristic_profile(question, paper_context, language)
        heuristic_profile.fallback_reason = llm_error or "heuristic_fallback"
        heuristic_profile.source = "heuristic"
        return heuristic_profile

    def _build_llm_profile(self, question: str, paper_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        title = str(paper_context.get("title", "") or "").strip()
        abstract = str(paper_context.get("abstract", "") or "").strip()
        section_titles = [str(item).strip() for item in (paper_context.get("section_titles", []) or []) if str(item).strip()]
        candidate_terms = [str(item).strip() for item in (paper_context.get("candidate_terms", []) or []) if str(item).strip()]

        prompt = (
            "You are an academic paper QA retrieval router.\n"
            "Your task is not to answer the question, but to create a retrieval profile for a RAG system.\n"
            "Classify the user question into one main intent and optional sub intents.\n"
            "Return JSON only. Do not add commentary.\n\n"
            "Allowed main_intent values:\n"
            "contribution, paper_overview, method_flow, experiment_setup, result_analysis, "
            "comparison, dataset, limitation, definition, implementation_detail, figure_table, other\n\n"
            "Schema:\n"
            "{"
            '"main_intent": "contribution|paper_overview|method_flow|experiment_setup|result_analysis|comparison|dataset|limitation|definition|implementation_detail|figure_table|other",'
            '"sub_intents": ["paper_overview", "evidence_seeking", "result_check", "table_lookup", "deep_method"],'
            '"confidence": 0.0,'
            '"intent_summary": "short description of what the user wants",'
            '"preferred_sections": ["abstract", "introduction", "method", "experiments", "results", "conclusion", "limitation", "appendix"],'
            '"avoid_sections": ["references", "acknowledgements", "ethics", "appendix", "implementation_details"],'
            '"rewrite_focus": ["short English retrieval focuses"],'
            '"rerank_focus": ["what kind of chunks should be preferred by reranker"],'
            '"use_keyword_search": true,'
            '"use_hyde": false,'
            '"final_context_policy": {'
            '"allow_tables": false,'
            '"allow_figures": false,'
            '"allow_appendix": false,'
            '"max_table_chunks": 0,'
            '"max_appendix_chunks": 0'
            '}'
            "}\n\n"
            "Intent rules:\n"
            "- If the question asks about core contributions, main contributions, innovation, novelty, or 提出了什么, use main_intent=contribution.\n"
            "- If the question asks what the paper is about overall, use paper_overview.\n"
            "- If the question asks about method, pipeline, framework, algorithm, or 方法流程, use method_flow.\n"
            "- If the question asks about datasets, baselines, metrics, or experimental setup, use experiment_setup.\n"
            "- If the question asks about results, performance, ablation, or 实验结果说明什么, use result_analysis.\n"
            "- If the question asks about limitations, weaknesses, or future work, use limitation.\n"
            "- If the question asks about implementation, prompts, hyperparameters, training details, or cost, use implementation_detail.\n"
            "- If the question asks about a specific table, figure, chart, or number, use figure_table.\n\n"
            "Retrieval policy rules:\n"
            "- For contribution and paper_overview, prefer abstract, introduction, conclusion, and high-level method chunks. Avoid appendix, tables, implementation details, references, acknowledgements, and ethics unless explicitly requested.\n"
            "- For method_flow, prefer method, approach, framework, algorithm, and figure captions. Avoid pure experiment tables unless needed.\n"
            "- For result_analysis, prefer results, experiments, analysis, ablation, and tables.\n"
            "- For implementation_detail, prefer appendix and implementation details.\n"
            "- For figure_table, allow tables and figures and enable keyword search.\n"
            "- Keep rewrite_focus concise and retrieval-oriented.\n"
            "- Use paper context only as supporting evidence.\n\n"
            f"Question: {question}\n"
            f"Paper title: {title or 'N/A'}\n"
            f"Paper abstract: {abstract[:1600] or 'N/A'}\n"
            f"Section titles: {', '.join(section_titles[:24]) or 'N/A'}\n"
            f"Candidate terms: {', '.join(candidate_terms[:24]) or 'N/A'}"
        )
        # 意图画像只做路由和检索规划，走小模型即可满足稳定性与成本要求。
        response = self.generation_service.complete_with_qwen(
            prompt,
            task_type="intent_routing",
        )
        payload = json.loads(self._extract_json_block(response))
        if not isinstance(payload, dict):
            return None
        return payload

    def _normalize_profile(
        self,
        payload: Dict[str, Any],
        question: str,
        paper_context: Dict[str, Any],
        language: str,
        source: str,
        fallback_reason: str,
    ) -> IntentProfile:
        normalized_query = _normalize_text(question)
        main_intent = self._normalize_main_intent(payload.get("main_intent", "other"))

        confidence = self._safe_float(payload.get("confidence", 0.5), default=0.5)
        confidence = max(0.0, min(1.0, confidence))
        intent_summary = str(payload.get("intent_summary", "")).strip() or self._default_summary(main_intent)
        preferred_sections = self._normalize_str_list(payload.get("preferred_sections", []))
        if not preferred_sections:
            preferred_sections = list(MAIN_INTENTS[main_intent]["preferred_sections"])
        preferred_sections = _dedupe(preferred_sections)
        avoid_sections = self._normalize_str_list(payload.get("avoid_sections", []))
        if not avoid_sections:
            avoid_sections = self._default_avoid_sections(main_intent)
        avoid_sections = _dedupe(avoid_sections)

        sub_intents = self._normalize_str_list(payload.get("sub_intents", []))
        sub_intents = [item for item in sub_intents if item in SUB_INTENT_RULES]
        if not sub_intents:
            sub_intents = self._infer_sub_intents(normalized_query, question, main_intent)

        route_weights = self._build_route_weights(main_intent, sub_intents)
        rewrite_count = int(payload.get("rewrite_count", 0) or MAIN_INTENTS[main_intent]["rewrite_count"])
        rewrite_count = max(2, min(5, rewrite_count))
        use_keyword_search = self._safe_bool(payload.get("use_keyword_search"), default=True)
        use_hyde = self._safe_bool(payload.get("use_hyde"), default=False)
        rewrite_focus = self._normalize_str_list(payload.get("rewrite_focus", []))
        if not rewrite_focus:
            rewrite_focus = self._build_rewrite_focus(main_intent, sub_intents)
        rerank_focus = self._normalize_str_list(payload.get("rerank_focus", []))
        if not rerank_focus:
            rerank_focus = self._build_rerank_focus(main_intent, sub_intents)
        final_context_policy = self._normalize_final_context_policy(payload.get("final_context_policy"), main_intent)
        ambiguity_score = self._estimate_ambiguity(normalized_query, sub_intents, confidence)

        return IntentProfile(
            original_query=question,
            normalized_query=normalized_query,
            language=language,
            main_intent=main_intent,
            sub_intents=sub_intents,
            confidence=confidence,
            ambiguity_score=ambiguity_score,
            intent_summary=intent_summary,
            preferred_sections=preferred_sections,
            avoid_sections=avoid_sections,
            route_weights=route_weights,
            rewrite_count=rewrite_count,
            use_keyword_search=use_keyword_search,
            use_hyde=use_hyde,
            rewrite_focus=rewrite_focus,
            rerank_focus=rerank_focus,
            final_context_policy=final_context_policy,
            fallback_reason=fallback_reason,
            source=source,
            raw_model_output=payload,
        )

    def _heuristic_profile(self, question: str, paper_context: Dict[str, Any], language: str) -> IntentProfile:
        normalized_query = _normalize_text(question)
        tokens = _tokenize(question)
        main_intent = self._classify_main_intent(normalized_query, tokens)
        sub_intents = self._infer_sub_intents(normalized_query, question, main_intent)
        confidence = self._estimate_confidence(normalized_query, tokens, main_intent, sub_intents)
        intent_summary = self._default_summary(main_intent)
        preferred_sections = list(MAIN_INTENTS[main_intent]["preferred_sections"])
        avoid_sections = self._default_avoid_sections(main_intent)
        route_weights = self._build_route_weights(main_intent, sub_intents)
        rewrite_count = MAIN_INTENTS[main_intent]["rewrite_count"]
        rewrite_focus = self._build_rewrite_focus(main_intent, sub_intents)
        rerank_focus = self._build_rerank_focus(main_intent, sub_intents)
        final_context_policy = self._default_final_context_policy(main_intent)
        ambiguity_score = self._estimate_ambiguity(normalized_query, sub_intents, confidence)
        use_keyword_search = main_intent not in {"paper_overview", "contribution"} or confidence < 0.8
        use_hyde = main_intent in {"paper_overview", "contribution", "other"} and confidence < 0.6
        return IntentProfile(
            original_query=question,
            normalized_query=normalized_query,
            language=language,
            main_intent=main_intent,
            sub_intents=sub_intents,
            confidence=confidence,
            ambiguity_score=ambiguity_score,
            intent_summary=intent_summary,
            preferred_sections=preferred_sections,
            avoid_sections=avoid_sections,
            route_weights=route_weights,
            rewrite_count=rewrite_count,
            use_keyword_search=use_keyword_search,
            use_hyde=use_hyde,
            rewrite_focus=rewrite_focus,
            rerank_focus=rerank_focus,
            final_context_policy=final_context_policy,
            fallback_reason="heuristic_fallback",
            source="heuristic",
            raw_model_output=None,
        )

    def _classify_main_intent(self, normalized_query: str, tokens: List[str]) -> str:
        query_text = " ".join([normalized_query, " ".join(tokens)])
        for intent, spec in MAIN_INTENTS.items():
            if intent == "other":
                continue
            if any(keyword.lower() in query_text for keyword in spec["keywords"]):
                return intent
        return "other"

    def _infer_sub_intents(self, normalized_query: str, question: str, main_intent: str) -> List[str]:
        text = f"{normalized_query} {question}".lower()
        detected: List[str] = []
        for sub_intent, spec in SUB_INTENT_RULES.items():
            if any(keyword.lower() in text for keyword in spec["keywords"]):
                detected.append(sub_intent)
        if not detected:
            if main_intent in {"paper_overview", "contribution", "comparison", "result_analysis"}:
                detected.append("paper_overview")
            elif main_intent in {"method_flow", "experiment_setup", "dataset", "implementation_detail", "definition"}:
                detected.append("evidence_seeking")
            elif main_intent == "figure_table":
                detected.append("table_lookup")
            else:
                detected.append("evidence_seeking")
        return _dedupe(detected)[:3]

    def _build_route_weights(self, main_intent: str, sub_intents: List[str]) -> Dict[str, float]:
        weights = dict(DEFAULT_ROUTE_WEIGHTS)
        weights.update(INTENT_ROUTE_WEIGHTS.get(main_intent, {}))

        if "paper_overview" in sub_intents:
            weights["vector_original"] *= 1.08
            weights["vector_rewrite"] *= 1.05
        if "evidence_seeking" in sub_intents:
            weights["vector_rewrite"] *= 1.06
            weights["keyword"] *= 1.06
        if "result_check" in sub_intents:
            weights["vector_rewrite"] *= 1.07
            weights["keyword"] *= 1.08
        if "table_lookup" in sub_intents:
            weights["keyword"] *= 1.1
        if "deep_method" in sub_intents:
            weights["vector_rewrite"] *= 1.05
        return {key: round(value, 3) for key, value in weights.items()}

    def _build_rewrite_focus(self, main_intent: str, sub_intents: List[str]) -> List[str]:
        if main_intent in {"paper_overview", "contribution"}:
            focus = ["abstract", "introduction", "conclusion", "contribution"]
        elif main_intent in {"method_flow", "implementation_detail", "definition"}:
            focus = ["method", "architecture", "training", "inference"]
        elif main_intent == "experiment_setup":
            focus = ["experiment", "dataset", "baseline", "evaluation"]
        elif main_intent == "result_analysis":
            focus = ["results", "evaluation", "ablation", "analysis"]
        elif main_intent == "comparison":
            focus = ["baseline", "comparison", "ablation", "results"]
        elif main_intent == "dataset":
            focus = ["dataset", "corpus", "benchmark", "data"]
        elif main_intent == "limitation":
            focus = ["limitations", "future work", "failure cases"]
        elif main_intent == "figure_table":
            focus = ["figure", "table", "caption", "appendix"]
        else:
            focus = ["evidence", "passage", "paper", "section"]

        if "paper_overview" in sub_intents:
            focus.extend(["overview", "summary"])
        if "result_check" in sub_intents:
            focus.extend(["results", "metrics"])
        if "table_lookup" in sub_intents:
            focus.extend(["figure", "table", "caption"])
        return _dedupe(focus)[:6]

    def _build_rerank_focus(self, main_intent: str, sub_intents: List[str]) -> List[str]:
        focus = list(MAIN_INTENTS.get(main_intent, MAIN_INTENTS["other"])["preferred_sections"])
        if "paper_overview" in sub_intents:
            focus.extend(["abstract", "introduction", "conclusion"])
        if "evidence_seeking" in sub_intents:
            focus.extend(["evidence", "answerable", "facts"])
        if "result_check" in sub_intents:
            focus.extend(["results", "evaluation", "ablation"])
        if "table_lookup" in sub_intents:
            focus.extend(["figure", "table", "caption"])
        if "deep_method" in sub_intents:
            focus.extend(["method", "architecture", "training"])
        return _dedupe(focus)[:8]

    def _estimate_confidence(self, normalized_query: str, tokens: List[str], main_intent: str, sub_intents: List[str]) -> float:
        content_weight = min(1.0, len(tokens) / 8.0)
        intent_weight = 0.25 if main_intent != "other" else 0.0
        sub_weight = min(0.25, 0.08 * len(sub_intents))
        specificity = min(1.0, 0.25 + 0.35 * content_weight + intent_weight + sub_weight)
        return round(max(0.35, min(0.95, specificity)), 3)

    def _estimate_ambiguity(self, normalized_query: str, sub_intents: List[str], confidence: float) -> float:
        length_bonus = min(0.2, len(normalized_query) / 120.0)
        ambiguity = 1.0 - confidence + length_bonus
        if sub_intents:
            ambiguity -= min(0.15, 0.05 * len(sub_intents))
        return round(max(0.05, min(1.0, ambiguity)), 3)

    def _default_summary(self, main_intent: str) -> str:
        summaries = {
            "contribution": "understand the paper's main contribution and novelty",
            "paper_overview": "understand the paper overview and key ideas",
            "method_flow": "understand the method flow and implementation details",
            "experiment_setup": "understand the experiments, datasets, baselines, and setup",
            "result_analysis": "understand the results, evaluation, and analysis",
            "comparison": "understand the comparison and ablation evidence",
            "dataset": "understand the dataset or benchmark used in the paper",
            "limitation": "understand limitations and future work",
            "definition": "understand the definition or concept being asked about",
            "implementation_detail": "understand the implementation details and training settings",
            "figure_table": "find and interpret the relevant figure or table",
            "other": "retrieve the most relevant paper evidence",
        }
        return summaries.get(main_intent, summaries["other"])

    def _normalize_main_intent(self, value: Any) -> str:
        main_intent = str(value or "other").strip() or "other"
        main_intent = INTENT_ALIASES.get(main_intent, main_intent)
        if main_intent not in MAIN_INTENTS:
            return "other"
        return main_intent

    def _default_avoid_sections(self, main_intent: str) -> List[str]:
        if main_intent in {"contribution", "paper_overview"}:
            return ["references", "acknowledgements", "ethics", "appendix", "implementation_details"]
        if main_intent == "method_flow":
            return ["references", "acknowledgements", "ethics"]
        if main_intent == "experiment_setup":
            return ["references", "acknowledgements", "ethics", "appendix"]
        if main_intent == "result_analysis":
            return ["references", "acknowledgements", "ethics"]
        if main_intent == "comparison":
            return ["references", "acknowledgements", "ethics"]
        if main_intent == "dataset":
            return ["references", "acknowledgements", "ethics"]
        if main_intent == "limitation":
            return ["references", "acknowledgements", "ethics"]
        if main_intent == "definition":
            return ["references", "acknowledgements", "ethics"]
        if main_intent == "implementation_detail":
            return ["references", "acknowledgements", "ethics"]
        if main_intent == "figure_table":
            return ["references", "acknowledgements", "ethics"]
        return ["references", "acknowledgements", "ethics"]

    def _default_final_context_policy(self, main_intent: str) -> Dict[str, Any]:
        policy = {
            "allow_tables": False,
            "allow_figures": False,
            "allow_appendix": False,
            "max_table_chunks": 0,
            "max_appendix_chunks": 0,
        }
        if main_intent == "method_flow":
            policy["allow_figures"] = True
        elif main_intent == "implementation_detail":
            policy["allow_appendix"] = True
            policy["max_appendix_chunks"] = 1
        elif main_intent == "figure_table":
            policy["allow_tables"] = True
            policy["allow_figures"] = True
        elif main_intent == "result_analysis":
            policy["allow_tables"] = True
            policy["max_table_chunks"] = 1
        elif main_intent == "dataset":
            policy["allow_tables"] = True
        return policy

    def _detect_language(self, user_query: str, tokens: List[str]) -> str:
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_query))
        has_latin = any(re.search(r"[a-zA-Z]", token) for token in tokens)
        if has_cjk and has_latin:
            return "mixed"
        if has_cjk:
            return "zh"
        if has_latin:
            return "en"
        return "unknown"

    def _normalize_str_list(self, value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, (list, tuple, set)):
            return []
        items = [str(item).strip() for item in value if str(item).strip()]
        return _dedupe(items)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _safe_bool(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        return default

    def _normalize_final_context_policy(self, value: Any, main_intent: str) -> Dict[str, Any]:
        default_policy = self._default_final_context_policy(main_intent)
        if not isinstance(value, dict):
            return default_policy
        return {
            "allow_tables": self._safe_bool(value.get("allow_tables"), default=default_policy["allow_tables"]),
            "allow_figures": self._safe_bool(value.get("allow_figures"), default=default_policy["allow_figures"]),
            "allow_appendix": self._safe_bool(value.get("allow_appendix"), default=default_policy["allow_appendix"]),
            "max_table_chunks": max(0, int(self._safe_float(value.get("max_table_chunks"), default=default_policy["max_table_chunks"]))),
            "max_appendix_chunks": max(0, int(self._safe_float(value.get("max_appendix_chunks"), default=default_policy["max_appendix_chunks"]))),
        }

    def _extract_json_block(self, text: str) -> str:
        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)
        plain_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if plain_match:
            return plain_match.group(1)
        return text
