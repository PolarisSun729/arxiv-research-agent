from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService, QueryProfile


logger = logging.getLogger(__name__)


class RerankService:
    """Facade for rerank responsibilities split from EnhancedRetrievalService.

    This first extraction keeps the mature rerank implementation on the owning
    retrieval service so behavior and debug payloads stay compatible while the
    call sites move to a dedicated module.
    """

    def __init__(self, retrieval_service: "EnhancedRetrievalService"):
        self.retrieval_service = retrieval_service

    def build_rerank_query(self, user_query: str, query_profile: "QueryProfile") -> str:
        if self.retrieval_service.generation_service is not None and hasattr(self.retrieval_service.generation_service, "build_rerank_query"):
            try:
                rerank_query = self.retrieval_service.generation_service.build_rerank_query(
                    user_query,
                    intent_profile=query_profile.intent_profile.to_dict(),
                )
                if rerank_query and rerank_query.strip():
                    actual_rerank_query = rerank_query.strip()
                    logger.debug("original_question=%s", user_query)
                    logger.debug("actual_rerank_query=%s", actual_rerank_query)
                    return actual_rerank_query
            except Exception as exc:  # pragma: no cover
                logger.debug("Failed to rewrite rerank query with model: %s", exc)
        actual_rerank_query = self._fallback_rerank_query(user_query, query_profile=query_profile)
        logger.debug("original_question=%s", user_query)
        logger.debug("actual_rerank_query=%s", actual_rerank_query)
        return actual_rerank_query

    def build_rerank_document_text(self, chunk: Dict[str, object]) -> str:
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        if chunk_type in {"figure", "table"}:
            parts = [
                str(chunk.get("asset_summary", "") or "").strip(),
                str(chunk.get("asset_preview_text", "") or "").strip(),
                str(chunk.get("section_title", "") or "").strip(),
                str(chunk.get("section_path", "") or "").strip(),
                f"page {chunk.get('page_number') or chunk.get('page_range') or ''}".strip(),
            ]
            content = "\n".join(part for part in parts if part).strip()
            if content:
                return self.retrieval_service._limit_rerank_text(content, self.retrieval_service.llm_rerank_max_doc_chars)

        content = self.retrieval_service._limit_rerank_text(
            str(chunk.get("rerank_text", "") or ""),
            self.retrieval_service.llm_rerank_max_doc_chars,
        )
        if not content:
            content = self.retrieval_service._limit_rerank_text(
                str(chunk.get("content", "") or ""),
                self.retrieval_service.llm_rerank_max_doc_chars,
            )
        if not content:
            content = self.retrieval_service._limit_rerank_text(
                str(chunk.get("text", "") or ""),
                self.retrieval_service.llm_rerank_max_doc_chars,
            )

        section_title = str(chunk.get("section_title", "") or "").strip()
        if section_title:
            normalized_title = self.retrieval_service._normalize_query_text(section_title)
            normalized_content = self.retrieval_service._normalize_query_text(content)
            if normalized_title and normalized_title not in normalized_content[: max(len(normalized_title), 1) * 2]:
                content = f"{section_title}\n{content}".strip()

        return self.retrieval_service._limit_rerank_text(content, self.retrieval_service.llm_rerank_max_doc_chars)

    def llm_rerank(
        self,
        rerank_query: str,
        chunks: List[Dict[str, object]],
        top_k: int,
        query_profile: Optional["QueryProfile"] = None,
        original_question: Optional[str] = None,
        candidate_limit: Optional[int] = None,
    ) -> Dict[str, object]:
        return self.retrieval_service._llm_rerank_impl(
            rerank_query,
            chunks,
            top_k,
            query_profile=query_profile,
            original_question=original_question,
            candidate_limit=candidate_limit,
        )

    def _fallback_rerank_query(self, user_query: str, query_profile: "QueryProfile") -> str:
        normalized_question = re.sub(r"\s+", " ", (user_query or "")).strip()
        base_query = (
            "Select the passage that most directly supports an answer to the user's question. "
            "Prefer evidence-bearing chunks with explicit facts, definitions, steps, causes, results, comparisons, or other answerable statements; "
            "down-rank passages that are only loosely topic-related or background. "
        )
        intent_profile = query_profile.intent_profile
        if intent_profile is not None:
            main_intent = self.retrieval_service._legacy_intent_bucket(intent_profile.main_intent)
            intent_clauses = {
                "summary": "Prioritize abstract, introduction, and conclusion passages that state the paper's main contribution or findings. ",
                "method": "Prioritize method, architecture, training, inference, and implementation details. ",
                "experiment": "Prioritize experiment, evaluation, results, metric, baseline, and ablation evidence. ",
                "comparison": "Prioritize direct baseline comparisons and ablation evidence. ",
                "dataset": "Prioritize dataset, corpus, benchmark, split, and data description passages. ",
                "limitation": "Prioritize limitations, failure cases, discussion, and future work. ",
                "figure_table": "Prioritize figure captions, table captions, appendix references, and visual explanations. ",
            }
            base_query += intent_clauses.get(main_intent, "")
            if intent_profile.preferred_sections:
                base_query += f"Favor sections such as: {', '.join(intent_profile.preferred_sections[:4])}. "
            if intent_profile.sub_intents:
                if "paper_overview" in intent_profile.sub_intents:
                    base_query += "Prefer passages that summarize the paper at a high level. "
                if "evidence_seeking" in intent_profile.sub_intents:
                    base_query += "Prefer passages that provide direct answer-bearing evidence. "
                if "result_check" in intent_profile.sub_intents:
                    base_query += "Prefer passages with concrete numbers, metrics, and outcome descriptions. "
                if "table_lookup" in intent_profile.sub_intents:
                    base_query += "Prefer passages tied to figures, tables, captions, or appendix visual material. "
                if "deep_method" in intent_profile.sub_intents:
                    base_query += "Prefer passages that explain the technical pipeline and implementation. "
        if normalized_question:
            return f"{base_query}Original question: {normalized_question}"
        return base_query.rstrip()
