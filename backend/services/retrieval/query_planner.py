from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.intent.intent_service import IntentProfile
    from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService, QueryProfile


class QueryPlanner:
    """Thin planning facade extracted from EnhancedRetrievalService.

    The first step of the split keeps the underlying heuristics and debug
    payloads inside EnhancedRetrievalService, while moving the orchestration
    entry points behind a dedicated collaborator.
    """

    def __init__(self, retrieval_service: "EnhancedRetrievalService"):
        self.retrieval_service = retrieval_service

    def build_intent_profile(
        self,
        user_query: str,
        paper_context: Optional[Dict[str, Any]] = None,
    ) -> "IntentProfile":
        return self.retrieval_service._build_intent_profile(
            user_query,
            paper_context=paper_context,
        )

    def build_query_profile(
        self,
        user_query: str,
        collection_name: str,
        paper_context: Optional[Dict[str, Any]] = None,
        intent_profile: Optional["IntentProfile"] = None,
    ) -> "QueryProfile":
        return self.retrieval_service._build_query_profile(
            user_query,
            collection_name,
            paper_context=paper_context,
            intent_profile=intent_profile,
        )

    def build_query_views(
        self,
        user_query: str,
        query_profile: "QueryProfile",
        enable_query_rewrite: bool,
    ) -> Dict[str, Any]:
        return self.retrieval_service._build_query_views(
            user_query,
            query_profile,
            enable_query_rewrite,
        )

    def build_rerank_query(self, user_query: str, query_profile: "QueryProfile") -> str:
        return self.retrieval_service.rerank_service.build_rerank_query(user_query, query_profile)

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
