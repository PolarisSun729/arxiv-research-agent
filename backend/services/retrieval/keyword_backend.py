"""Keyword backend abstraction for BM25 retrieval.

Provides a lightweight interface to switch between internal_bm25 and bm25s implementations.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KeywordBackendResult:
    """Standardized result from keyword backend."""

    def __init__(
        self,
        *,
        results: List[Dict[str, Any]],
        debug: Dict[str, Any],
        backend_name: str,
        fallback_reason: str = "",
    ) -> None:
        self.results = results
        self.debug = debug
        self.backend_name = backend_name
        self.fallback_reason = fallback_reason


class KeywordBackend(ABC):
    """Abstract base for keyword retrieval backends."""

    @abstractmethod
    def retrieve(
        self,
        *,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> KeywordBackendResult:
        """Execute keyword retrieval for given query views.

        Args:
            query_views: List of query view dicts with 'query', 'source', 'weight', etc.
            top_k: Maximum number of results to return
            query_profile: QueryProfile containing intent and query metadata
            retrieval_index: CollectionRetrievalIndex with documents and postings

        Returns:
            KeywordBackendResult with retrieved chunks and debug info
        """
        pass

    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier."""
        pass


class InternalBM25Backend(KeywordBackend):
    """Internal BM25 implementation using project's existing tokenizer and scorer."""

    def __init__(
        self,
        *,
        query_tools: Any,
        route_confidence_builder: Any,
        structural_bonus_builder: Any,
        fusion_service: Any,
    ) -> None:
        self.query_tools = query_tools
        self.route_confidence_builder = route_confidence_builder
        self.structural_bonus_builder = structural_bonus_builder
        self.fusion_service = fusion_service

    def backend_name(self) -> str:
        return "internal_bm25"

    def retrieve(
        self,
        *,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> KeywordBackendResult:
        """Use existing internal BM25 implementation from route_retriever."""
        # Import locally to avoid circular dependency
        from services.retrieval.route_retriever import RouteRetriever

        # This is the existing keyword_retrieve logic extracted
        payload = self._internal_keyword_retrieve(
            query_views=query_views,
            top_k=top_k,
            query_profile=query_profile,
            retrieval_index=retrieval_index,
        )

        return KeywordBackendResult(
            results=payload["results"],
            debug=payload["debug"],
            backend_name=self.backend_name(),
        )

    def _internal_keyword_retrieve(
        self,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> Dict[str, Any]:
        """Internal BM25 retrieval logic (extracted from RouteRetriever.keyword_retrieve)."""
        from services.retrieval.retrieval_index import KEYWORD_FIELD_WEIGHTS
        from services.retrieval.route_retriever import RouteRetriever

        if not retrieval_index.documents:
            return {"results": [], "debug": retrieval_index.to_keyword_debug(0, full_scan_used=False)}

        aggregated: Dict[int, Dict[str, Any]] = {}
        per_query_debug: List[Dict[str, Any]] = []
        original_query_tokens: List[str] = []
        expanded_query_tokens: List[str] = []
        field_weights = self._keyword_field_weights_for_query(query_profile)

        for query_view in query_views:
            query = str(query_view.get("query", "") or "")
            raw_tokens = self.query_tools.tokenize_for_keyword_search(query)
            tokens = self._filter_keyword_query_tokens(raw_tokens)
            if not tokens:
                continue
            query_weight = float(query_view.get("weight", 0.0) or 0.0)
            if query_weight <= 0:
                continue

            expanded_tokens = self._expand_keyword_query_tokens(tokens)
            for token in tokens:
                if token not in original_query_tokens:
                    original_query_tokens.append(token)
            for token in expanded_tokens:
                if token not in expanded_query_tokens:
                    expanded_query_tokens.append(token)

            token_counter = Counter(expanded_tokens)
            candidate_doc_ids = set()
            for token in token_counter:
                candidate_doc_ids.update(retrieval_index.postings.get(token, set()))

            query_candidate_rows: List[Dict[str, Any]] = []
            for doc_id in candidate_doc_ids:
                document = retrieval_index.documents[doc_id]
                doc_field_weights = self._keyword_field_weights_for_document(
                    field_weights, document.chunk, query_profile
                )
                doc_counts, doc_length, match_fields, token_field_hits = RouteRetriever.weight_keyword_document_fields(
                    document.field_token_counts or {"body": document.token_counts},
                    token_counter,
                    doc_field_weights,
                )
                if not doc_counts:
                    continue

                bm25_detail = RouteRetriever.bm25_score_detailed(
                    token_counter,
                    doc_counts,
                    retrieval_index.document_frequency,
                    len(retrieval_index.documents),
                    retrieval_index.avgdl,
                    document.content_prefix,
                    doc_length=doc_length or document.doc_length,
                    token_field_hits=token_field_hits,
                )
                raw_score = bm25_detail["score"]
                if raw_score <= 0:
                    continue

                query_candidate_rows.append(
                    {
                        "doc_id": doc_id,
                        "raw_score": float(raw_score),
                        "matched_fields": Counter(match_fields),
                        "term_details": bm25_detail["term_details"],
                    }
                )

            query_candidate_rows.sort(key=lambda item: item["raw_score"], reverse=True)
            if not query_candidate_rows:
                per_query_debug.append(
                    {
                        "view_id": query_view["view_id"],
                        "query": query,
                        "source": query_view["source"],
                        "weight": query_weight,
                        "query_tokens": tokens,
                        "expanded_tokens": expanded_tokens,
                        "candidate_count": 0,
                        "top_hits": [],
                    }
                )
                continue

            normalized_scores = RouteRetriever.normalize_scores(
                [float(item["raw_score"]) for item in query_candidate_rows]
            )
            per_query_top_rows: List[Dict[str, Any]] = []
            keyword_rrf_k = RouteRetriever.keyword_rrf_k()

            for rank, row in enumerate(query_candidate_rows[:top_k], start=1):
                normalized_score = float(normalized_scores[rank - 1]) if rank - 1 < len(normalized_scores) else 0.0
                rrf_vote = query_weight * (1.0 / (keyword_rrf_k + rank))
                contribution_score = query_weight * normalized_score

                entry = aggregated.setdefault(
                    row["doc_id"],
                    {
                        "fusion_score": 0.0,
                        "best_raw_score": 0.0,
                        "matched_fields": Counter(),
                        "view_contributions": [],
                        "term_traces": {},
                        "query_sources": [],
                    },
                )
                entry["fusion_score"] += rrf_vote
                entry["best_raw_score"] = max(float(entry["best_raw_score"]), float(row["raw_score"]))
                entry["matched_fields"].update(row["matched_fields"])
                if query_view["source"] not in entry["query_sources"]:
                    entry["query_sources"].append(query_view["source"])

                for term in row.get("term_details", []):
                    token = term["token"]
                    trace = entry["term_traces"].setdefault(
                        token,
                        {
                            "token": token,
                            "idf": float(term["idf"]),
                            "best_term_score": 0.0,
                            "fields": [],
                            "query_sources": [],
                        },
                    )
                    trace["idf"] = max(float(trace["idf"]), float(term["idf"]))
                    trace["best_term_score"] = max(float(trace["best_term_score"]), float(term["term_score"]))
                    for field in term.get("fields", []):
                        if field not in trace["fields"]:
                            trace["fields"].append(field)
                    if query_view["source"] not in trace["query_sources"]:
                        trace["query_sources"].append(query_view["source"])

                entry["view_contributions"].append(
                    {
                        "view_id": query_view["view_id"],
                        "source": query_view["source"],
                        "query": query,
                        "rank": rank,
                        "weight": query_weight,
                        "raw_score": float(row["raw_score"]),
                        "normalized_score": normalized_score,
                        "rrf_vote": float(rrf_vote),
                        "contribution_score": float(contribution_score),
                        "matched_fields": [field for field, _ in row["matched_fields"].most_common(4)],
                        "matched_terms": [term["token"] for term in row.get("term_details", [])[:6]],
                    }
                )

                per_query_top_rows.append(
                    {
                        "chunk_id": str(retrieval_index.documents[row["doc_id"]].chunk.get("chunk_id", "") or ""),
                        "rank": rank,
                        "raw_score": float(row["raw_score"]),
                        "normalized_score": normalized_score,
                        "rrf_vote": float(rrf_vote),
                        "matched_fields": [field for field, _ in row["matched_fields"].most_common(4)],
                    }
                )

            per_query_debug.append(
                {
                    "view_id": query_view["view_id"],
                    "query": query,
                    "source": query_view["source"],
                    "weight": query_weight,
                    "query_tokens": tokens,
                    "expanded_tokens": expanded_tokens,
                    "candidate_count": len(query_candidate_rows),
                    "top_hits": per_query_top_rows,
                }
            )

        ranked = [
            (doc_id, payload)
            for doc_id, payload in aggregated.items()
            if float(payload.get("fusion_score", 0.0) or 0.0) > 0
        ]
        ranked.sort(
            key=lambda item: (
                float(item[1].get("fusion_score", 0.0) or 0.0),
                float(item[1].get("best_raw_score", 0.0) or 0.0),
            ),
            reverse=True,
        )

        if not ranked:
            return {
                "results": [],
                "debug": {
                    **retrieval_index.to_keyword_debug(0, full_scan_used=False),
                    "query_tokens": original_query_tokens,
                    "expanded_tokens": expanded_query_tokens,
                    "query_expansion_applied": bool(set(expanded_query_tokens) - set(original_query_tokens)),
                    "matched_chunks": [],
                    "field_weights": field_weights,
                    "query_contributions": per_query_debug,
                    "keyword_fusion_strategy": "weighted_rrf",
                },
            }

        normalized_scores = RouteRetriever.normalize_scores(
            [float(payload.get("fusion_score", 0.0) or 0.0) for _, payload in ranked]
        )
        base_route_confidence = self.route_confidence_builder(
            "keyword",
            query_profile,
            " | ".join([row["query"] for row in query_views]) if query_views else query_profile.original_query,
            route_queries=[row["query"] for row in query_views] or [query_profile.keyword_query],
            intent_profile=query_profile.intent_profile,
        )

        results: List[Dict[str, Any]] = []
        for rank, ((doc_id, payload), normalized_score) in enumerate(
            zip(ranked[:top_k], normalized_scores[:top_k]), start=1
        ):
            chunk = dict(retrieval_index.documents[doc_id].chunk)
            fusion_score = float(payload.get("fusion_score", 0.0) or 0.0)
            best_raw_score = float(payload.get("best_raw_score", 0.0) or 0.0)
            query_contributions = sorted(
                list(payload.get("view_contributions", [])),
                key=lambda item: (
                    float(item.get("rrf_vote", 0.0) or 0.0),
                    float(item.get("raw_score", 0.0) or 0.0),
                ),
                reverse=True,
            )
            match_fields = payload.get("matched_fields", Counter())
            main_fields = [field for field, _ in match_fields.most_common(4)]
            matched_terms = self._build_matched_terms(payload.get("term_traces", {}))
            query_sources = list(payload.get("query_sources", []))

            # Import locally to avoid circular dependency
            from services.retrieval.route_retriever import RouteRetriever

            noise_flags = self._detect_keyword_noise_flags(
                matched_terms=matched_terms,
                matched_fields=main_fields,
                query_profile=query_profile,
                chunk=chunk,
            )
            route_confidence = self._adjust_keyword_route_confidence(
                base_route_confidence,
                matched_terms=matched_terms,
                noise_flags=noise_flags,
                query_profile=query_profile,
            )

            chunk["retrieval_route"] = "keyword"
            chunk["source_query"] = " | ".join([item["query"] for item in query_contributions[:3]])
            chunk["route_rank"] = rank
            chunk["route_score"] = fusion_score
            chunk["normalized_route_score"] = float(normalized_score)
            chunk["route_confidence"] = float(route_confidence)
            chunk["structural_bonus"] = float(self.structural_bonus_builder(chunk, query_profile))
            chunk["keyword_match_fields"] = main_fields
            chunk["keyword_hit_asset_field"] = any(
                field in {"asset_caption", "asset_aux"} for field in main_fields
            )
            chunk["keyword_query_contributions"] = query_contributions
            chunk["keyword_best_raw_bm25_score"] = best_raw_score
            chunk["bm25_raw_score"] = best_raw_score
            chunk["bm25_fused_score"] = fusion_score
            chunk["keyword_matched_terms"] = matched_terms
            chunk["keyword_query_sources"] = query_sources
            chunk["keyword_noise_flags"] = noise_flags
            chunk["keyword_base_route_confidence"] = float(base_route_confidence)
            results.append(chunk)

        return {
            "results": results,
            "debug": {
                **retrieval_index.to_keyword_debug(len(ranked), full_scan_used=False),
                "query_tokens": original_query_tokens,
                "expanded_tokens": expanded_query_tokens,
                "query_expansion_applied": bool(set(expanded_query_tokens) - set(original_query_tokens)),
                "field_weights": field_weights,
                "query_contributions": per_query_debug,
                "keyword_fusion_strategy": "weighted_rrf",
                "keyword_rrf_k": RouteRetriever.keyword_rrf_k(),
                "base_route_confidence": float(base_route_confidence),
                "matched_chunks": [
                    {
                        "chunk_id": str(retrieval_index.documents[doc_id].chunk.get("chunk_id", "") or ""),
                        "keyword_fusion_score": float(payload.get("fusion_score", 0.0) or 0.0),
                        "best_raw_bm25_score": float(payload.get("best_raw_score", 0.0) or 0.0),
                        "matched_fields": [field for field, _ in payload.get("matched_fields", Counter()).most_common(4)],
                        "matched_terms": self._build_matched_terms(payload.get("term_traces", {})),
                        "query_sources": list(payload.get("query_sources", [])),
                        "noise_flags": self._detect_keyword_noise_flags(
                            matched_terms=self._build_matched_terms(payload.get("term_traces", {})),
                            matched_fields=[
                                field for field, _ in payload.get("matched_fields", Counter()).most_common(4)
                            ],
                            query_profile=query_profile,
                            chunk=retrieval_index.documents[doc_id].chunk,
                        ),
                        "hit_asset_field": any(
                            field in {"asset_caption", "asset_aux"}
                            for field, _ in payload.get("matched_fields", Counter()).most_common(4)
                        ),
                        "view_contributions": sorted(
                            list(payload.get("view_contributions", [])),
                            key=lambda item: (
                                float(item.get("rrf_vote", 0.0) or 0.0),
                                float(item.get("raw_score", 0.0) or 0.0),
                            ),
                            reverse=True,
                        )[:4],
                    }
                    for doc_id, payload in ranked[:top_k]
                ],
            },
        }

    def _expand_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        expander = getattr(self.query_tools, "expand_keyword_query_tokens", None)
        if callable(expander):
            return expander(tokens)
        return tokens

    def _filter_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        extractor = getattr(self.query_tools, "extract_query_keywords", None)
        if callable(extractor):
            return extractor(tokens, limit=max(len(tokens), 1))
        return tokens

    def _keyword_field_weights_for_query(self, query_profile: Any) -> Dict[str, float]:
        from services.retrieval.retrieval_index import KEYWORD_FIELD_WEIGHTS

        weights = dict(KEYWORD_FIELD_WEIGHTS)
        main_intent = "other"
        if query_profile.intent_profile is not None:
            main_intent = str(getattr(query_profile.intent_profile, "main_intent", "") or "other")
        main_intent = str(query_profile.question_type or main_intent or "other").strip().lower()

        if main_intent == "figure_table" or "figure_table" in (query_profile.intent_tags or []):
            weights["asset_caption"] = 1.15
            weights["asset_aux"] = 0.72
        else:
            weights["asset_caption"] = min(weights.get("asset_caption", 0.0), 0.12)
            weights["asset_aux"] = min(weights.get("asset_aux", 0.0), 0.05)
        return weights

    def _keyword_field_weights_for_document(
        self,
        base_weights: Dict[str, float],
        chunk: Dict[str, Any],
        query_profile: Any,
    ) -> Dict[str, float]:
        weights = dict(base_weights)
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        is_asset_chunk = chunk_type in {"figure", "table"}
        is_figure_query = query_profile.question_type == "figure_table" or "figure_table" in (
            query_profile.intent_tags or []
        )

        if is_asset_chunk and not is_figure_query:
            weights["body"] = min(weights.get("body", 1.0), 0.38)
            weights["section_title"] = min(weights.get("section_title", 1.0), 0.35)
            weights["section_path"] = min(weights.get("section_path", 1.0), 0.25)
            weights["asset_caption"] = min(weights.get("asset_caption", 0.0), 0.08)
            weights["asset_aux"] = min(weights.get("asset_aux", 0.0), 0.03)
        return weights

    @staticmethod
    def _build_matched_terms(term_traces: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        matched = [
            {
                "token": str(trace.get("token", "")),
                "idf": float(trace.get("idf", 0.0) or 0.0),
                "term_score": float(trace.get("best_term_score", 0.0) or 0.0),
                "fields": list(trace.get("fields", []) or []),
                "query_sources": list(trace.get("query_sources", []) or []),
            }
            for trace in term_traces.values()
        ]
        matched.sort(key=lambda item: (item["term_score"], item["idf"]), reverse=True)
        return matched

    def _detect_keyword_noise_flags(
        self,
        *,
        matched_terms: List[Dict[str, Any]],
        matched_fields: List[str],
        query_profile: Any,
        chunk: Dict[str, Any],
    ) -> List[str]:
        """Detect noise flags in keyword matches."""
        from services.retrieval.route_retriever import RouteRetriever

        # Use RouteRetriever's method with proper constants
        retriever_instance = type('TempRetriever', (), {
            'LOW_IDF_THRESHOLD': RouteRetriever.LOW_IDF_THRESHOLD,
            'ASSET_NOISE_FIELDS': RouteRetriever.ASSET_NOISE_FIELDS,
            '_normalize_field_text': staticmethod(RouteRetriever._normalize_field_text),
            '_is_informative_token': lambda self, token: self._is_informative_token_impl(token),
        })()
        retriever_instance._is_informative_token_impl = lambda token: self._is_informative_token(token)

        return RouteRetriever.detect_keyword_noise_flags(
            retriever_instance,
            matched_terms=matched_terms,
            matched_fields=matched_fields,
            query_profile=query_profile,
            chunk=chunk,
        )

    def _adjust_keyword_route_confidence(
        self,
        base_confidence: float,
        *,
        matched_terms: List[Dict[str, Any]],
        noise_flags: List[str],
        query_profile: Any,
    ) -> float:
        """Adjust keyword route confidence based on match quality."""
        from services.retrieval.route_retriever import RouteRetriever

        # Use RouteRetriever's method with proper constants
        retriever_instance = type('TempRetriever', (), {
            'LOW_IDF_THRESHOLD': RouteRetriever.LOW_IDF_THRESHOLD,
            'query_intent_bucket': lambda self, qp: self._query_intent_bucket(qp),
            'query_tools': self.query_tools,
        })()
        retriever_instance._query_intent_bucket = lambda qp: self._query_intent_bucket(qp)

        return RouteRetriever.adjust_keyword_route_confidence(
            retriever_instance,
            base_confidence,
            matched_terms=matched_terms,
            noise_flags=noise_flags,
            query_profile=query_profile,
        )

    def _query_intent_bucket(self, query_profile: Any) -> str:
        """Get intent bucket for query profile."""
        bucketizer = getattr(self.query_tools, "legacy_intent_bucket", None)
        raw_intent = ""
        if query_profile.intent_profile is not None:
            raw_intent = str(getattr(query_profile.intent_profile, "main_intent", "") or "")
        raw_intent = raw_intent or str(query_profile.question_type or "other")
        if callable(bucketizer):
            return bucketizer(raw_intent)
        return str(raw_intent or "other").strip().lower()

    def _is_informative_token(self, token: str) -> bool:
        """Check if token is informative (not garbled)."""
        checker = getattr(self.query_tools, "is_informative_keyword_token", None)
        if callable(checker):
            return bool(checker(token))
        normalized = str(token or "").strip().lower()
        if len(normalized) <= 1:
            return False
        if re.search(r"[��]", normalized):
            return False
        if re.search(r"(.)\1{4,}", normalized):
            return False
        return True
