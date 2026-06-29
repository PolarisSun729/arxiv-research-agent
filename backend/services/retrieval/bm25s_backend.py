"""BM25s-based keyword backend implementation.

Uses the bm25s library for BM25 retrieval with collection-level indexing.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from services.retrieval.keyword_backend import KeywordBackend, KeywordBackendResult

logger = logging.getLogger(__name__)


class BM25sBackend(KeywordBackend):
    """BM25 backend using the bm25s library."""

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
        self._bm25s_available = False
        self._import_error: Optional[str] = None
        self._try_import_bm25s()

    def _try_import_bm25s(self) -> None:
        """Try importing bm25s and record availability."""
        try:
            import bm25s
            self._bm25s_available = True
            self._bm25s = bm25s
        except ImportError as exc:
            self._bm25s_available = False
            self._import_error = str(exc)
            logger.warning("bm25s not available: %s", exc)

    def backend_name(self) -> str:
        return "bm25s"

    def retrieve(
        self,
        *,
        query_views: List[Dict[str, Any]],
        top_k: int,
        query_profile: Any,
        retrieval_index: Any,
    ) -> KeywordBackendResult:
        """Execute BM25 retrieval using bm25s library."""
        if not self._bm25s_available:
            return KeywordBackendResult(
                results=[],
                debug={"bm25s_available": False, "import_error": self._import_error},
                backend_name=self.backend_name(),
                fallback_reason=f"bm25s_not_available: {self._import_error}",
            )

        if not retrieval_index.documents:
            return KeywordBackendResult(
                results=[],
                debug={
                    **retrieval_index.to_keyword_debug(0, full_scan_used=False),
                    "bm25s_available": True,
                    "bm25s_index_built": False,
                    "bm25s_corpus_size": 0,
                },
                backend_name=self.backend_name(),
                fallback_reason="empty_collection",
            )

        try:
            # Build or retrieve bm25s index
            bm25s_index, corpus_mapping = self._build_bm25s_index(retrieval_index)

            # Execute query for each view
            aggregated: Dict[int, Dict[str, Any]] = {}
            per_query_debug: List[Dict[str, Any]] = []
            original_query_tokens: List[str] = []
            expanded_query_tokens: List[str] = []

            from services.retrieval.route_retriever import RouteRetriever

            keyword_rrf_k = RouteRetriever.keyword_rrf_k()

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

                # Query bm25s index
                try:
                    query_results = self._query_bm25s(
                        bm25s_index=bm25s_index,
                        query_tokens=expanded_tokens,
                        top_k=top_k,
                        corpus_mapping=corpus_mapping,
                        retrieval_index=retrieval_index,
                    )
                except Exception as query_exc:
                    logger.warning("bm25s query failed for view %s: %s", query_view.get("view_id"), query_exc)
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
                            "error": str(query_exc),
                        }
                    )
                    continue

                if not query_results:
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

                # Normalize scores for this query view
                raw_scores = [float(item["raw_score"]) for item in query_results]
                normalized_scores = RouteRetriever.normalize_scores(raw_scores)

                per_query_top_rows: List[Dict[str, Any]] = []
                for rank, (result, normalized_score) in enumerate(zip(query_results, normalized_scores), start=1):
                    doc_id = result["doc_id"]
                    raw_score = result["raw_score"]
                    rrf_vote = query_weight * (1.0 / (keyword_rrf_k + rank))
                    contribution_score = query_weight * normalized_score

                    entry = aggregated.setdefault(
                        doc_id,
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
                    entry["best_raw_score"] = max(float(entry["best_raw_score"]), float(raw_score))

                    # Extract matched fields and terms from bm25s result
                    matched_fields = result.get("matched_fields", [])
                    for field in matched_fields:
                        entry["matched_fields"][field] += 1.0

                    if query_view["source"] not in entry["query_sources"]:
                        entry["query_sources"].append(query_view["source"])

                    # Build term traces from bm25s matched terms
                    for term_info in result.get("matched_terms", []):
                        token = term_info["token"]
                        trace = entry["term_traces"].setdefault(
                            token,
                            {
                                "token": token,
                                "idf": 0.0,
                                "best_term_score": 0.0,
                                "fields": [],
                                "query_sources": [],
                            },
                        )
                        trace["idf"] = max(float(trace["idf"]), float(term_info.get("idf", 0.0)))
                        trace["best_term_score"] = max(
                            float(trace["best_term_score"]), float(term_info.get("term_score", 0.0))
                        )
                        for field in term_info.get("fields", []):
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
                            "raw_score": float(raw_score),
                            "normalized_score": normalized_score,
                            "rrf_vote": float(rrf_vote),
                            "contribution_score": float(contribution_score),
                            "matched_fields": matched_fields[:4],
                            "matched_terms": [t["token"] for t in result.get("matched_terms", [])[:6]],
                        }
                    )

                    per_query_top_rows.append(
                        {
                            "chunk_id": str(retrieval_index.documents[doc_id].chunk.get("chunk_id", "") or ""),
                            "rank": rank,
                            "raw_score": float(raw_score),
                            "normalized_score": normalized_score,
                            "rrf_vote": float(rrf_vote),
                            "matched_fields": matched_fields[:4],
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
                        "candidate_count": len(query_results),
                        "top_hits": per_query_top_rows,
                    }
                )

            # Rank aggregated results
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
                return KeywordBackendResult(
                    results=[],
                    debug={
                        **retrieval_index.to_keyword_debug(0, full_scan_used=False),
                        "bm25s_available": True,
                        "bm25s_index_built": True,
                        "bm25s_corpus_size": len(corpus_mapping),
                        "query_tokens": original_query_tokens,
                        "expanded_tokens": expanded_query_tokens,
                        "query_expansion_applied": bool(set(expanded_query_tokens) - set(original_query_tokens)),
                        "matched_chunks": [],
                        "query_contributions": per_query_debug,
                        "keyword_fusion_strategy": "weighted_rrf",
                    },
                    backend_name=self.backend_name(),
                )

            # Build final results
            results = self._build_keyword_results(
                ranked=ranked,
                top_k=top_k,
                query_views=query_views,
                query_profile=query_profile,
                retrieval_index=retrieval_index,
            )

            return KeywordBackendResult(
                results=results,
                debug={
                    **retrieval_index.to_keyword_debug(len(ranked), full_scan_used=False),
                    "bm25s_available": True,
                    "bm25s_index_built": True,
                    "bm25s_corpus_size": len(corpus_mapping),
                    "query_tokens": original_query_tokens,
                    "expanded_tokens": expanded_query_tokens,
                    "query_expansion_applied": bool(set(expanded_query_tokens) - set(original_query_tokens)),
                    "query_contributions": per_query_debug,
                    "keyword_fusion_strategy": "weighted_rrf",
                    "keyword_rrf_k": keyword_rrf_k,
                    "base_route_confidence": self._compute_base_confidence(query_views, query_profile),
                    "matched_chunks": self._build_matched_chunks_debug(ranked[:top_k], retrieval_index, query_profile),
                },
                backend_name=self.backend_name(),
            )

        except Exception as exc:
            logger.warning("bm25s backend failed: %s", exc, exc_info=True)
            return KeywordBackendResult(
                results=[],
                debug={
                    **retrieval_index.to_keyword_debug(0, full_scan_used=False),
                    "bm25s_available": True,
                    "bm25s_error": str(exc),
                },
                backend_name=self.backend_name(),
                fallback_reason=f"bm25s_execution_failed: {exc}",
            )

    def _build_bm25s_index(self, retrieval_index: Any) -> tuple:
        """Build bm25s index from retrieval_index documents.

        Returns:
            (bm25s_index, corpus_mapping) where corpus_mapping[corpus_idx] = doc_id
        """
        # Check if index is already cached in retrieval_index
        if hasattr(retrieval_index, "_bm25s_index") and hasattr(retrieval_index, "_bm25s_corpus_mapping"):
            return retrieval_index._bm25s_index, retrieval_index._bm25s_corpus_mapping

        # Build corpus from documents
        corpus_texts: List[List[str]] = []
        corpus_mapping: List[int] = []

        for doc_id, document in enumerate(retrieval_index.documents):
            # Build document text from fields with weights
            doc_tokens = self._build_document_tokens(document)
            corpus_texts.append(doc_tokens)
            corpus_mapping.append(doc_id)

        # Build bm25s index
        import bm25s

        # Create BM25 object
        bm25_index = bm25s.BM25()

        # Index the corpus (bm25s expects tokenized texts)
        bm25_index.index(corpus_texts)

        # Cache in retrieval_index
        retrieval_index._bm25s_index = bm25_index
        retrieval_index._bm25s_corpus_mapping = corpus_mapping

        logger.info(
            "Built bm25s index: collection=%s corpus_size=%d",
            retrieval_index.collection_name,
            len(corpus_mapping),
        )

        return bm25_index, corpus_mapping

    def _build_document_tokens(self, document: Any) -> List[str]:
        """Build tokenized representation of document for bm25s indexing."""
        from services.retrieval.retrieval_index import KEYWORD_FIELD_WEIGHTS

        tokens: List[str] = []
        field_token_counts = document.field_token_counts or {"body": document.token_counts}

        # Apply field weights by repeating tokens
        for field_name, field_counts in field_token_counts.items():
            weight = KEYWORD_FIELD_WEIGHTS.get(field_name, 1.0)
            if weight <= 0:
                continue

            # Repeat tokens based on weight (approximate field importance)
            for token, count in field_counts.items():
                weighted_count = int(round(count * weight))
                tokens.extend([token] * weighted_count)

        return tokens if tokens else [""]

    def _query_bm25s(
        self,
        *,
        bm25s_index: Any,
        query_tokens: List[str],
        top_k: int,
        corpus_mapping: List[int],
        retrieval_index: Any,
    ) -> List[Dict[str, Any]]:
        """Query bm25s index and return results."""
        if not query_tokens:
            return []

        # Query bm25s (expects list of tokenized queries, i.e., [[tokens]])
        # We only have one query, so wrap it in another list
        results, scores = bm25s_index.retrieve([query_tokens], k=top_k)

        # Convert to standard format
        query_results: List[Dict[str, Any]] = []
        for corpus_idx, score in zip(results[0], scores[0]):  # bm25s returns batched results
            if corpus_idx < 0 or corpus_idx >= len(corpus_mapping):
                continue

            doc_id = corpus_mapping[corpus_idx]
            if doc_id >= len(retrieval_index.documents):
                continue

            document = retrieval_index.documents[doc_id]

            # Extract matched fields and terms
            matched_fields = self._extract_matched_fields(document, query_tokens)
            matched_terms = self._extract_matched_terms(document, query_tokens, retrieval_index)

            query_results.append(
                {
                    "doc_id": doc_id,
                    "raw_score": float(score),
                    "matched_fields": matched_fields,
                    "matched_terms": matched_terms,
                }
            )

        return query_results

    def _extract_matched_fields(self, document: Any, query_tokens: List[str]) -> List[str]:
        """Extract which fields matched the query tokens."""
        matched_fields: List[str] = []
        field_token_counts = document.field_token_counts or {"body": document.token_counts}
        query_token_set = set(query_tokens)

        for field_name, field_counts in field_token_counts.items():
            if any(token in field_counts for token in query_token_set):
                if field_name not in matched_fields:
                    matched_fields.append(field_name)

        return matched_fields

    def _extract_matched_terms(
        self, document: Any, query_tokens: List[str], retrieval_index: Any
    ) -> List[Dict[str, Any]]:
        """Extract matched terms with IDF and field information."""
        matched_terms: List[Dict[str, Any]] = []
        doc_counts = document.token_counts
        query_token_set = set(query_tokens)

        for token in query_token_set:
            if token not in doc_counts:
                continue

            df = retrieval_index.document_frequency.get(token, 1)
            total_docs = len(retrieval_index.documents)

            # Compute IDF
            import math
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

            # Get fields where token appears
            token_fields = []
            field_token_counts = document.field_token_counts or {"body": document.token_counts}
            for field_name, field_counts in field_token_counts.items():
                if token in field_counts:
                    token_fields.append(field_name)

            matched_terms.append(
                {
                    "token": token,
                    "idf": float(idf),
                    "term_score": float(idf * doc_counts[token]),  # Simplified term score
                    "fields": token_fields,
                }
            )

        # Sort by term score
        matched_terms.sort(key=lambda t: t["term_score"], reverse=True)
        return matched_terms

    def _build_keyword_results(
        self,
        *,
        ranked: List[tuple],
        top_k: int,
        query_views: List[Dict[str, Any]],
        query_profile: Any,
        retrieval_index: Any,
    ) -> List[Dict[str, Any]]:
        """Build final keyword results with all metadata."""
        from services.retrieval.route_retriever import RouteRetriever

        normalized_scores = RouteRetriever.normalize_scores(
            [float(payload.get("fusion_score", 0.0) or 0.0) for _, payload in ranked]
        )

        base_route_confidence = self._compute_base_confidence(query_views, query_profile)

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

        return results

    def _compute_base_confidence(self, query_views: List[Dict[str, Any]], query_profile: Any) -> float:
        """Compute base route confidence for keyword route."""
        return self.route_confidence_builder(
            "keyword",
            query_profile,
            " | ".join([row["query"] for row in query_views]) if query_views else query_profile.original_query,
            route_queries=[row["query"] for row in query_views] or [query_profile.keyword_query],
            intent_profile=query_profile.intent_profile,
        )

    def _build_matched_chunks_debug(
        self, ranked: List[tuple], retrieval_index: Any, query_profile: Any
    ) -> List[Dict[str, Any]]:
        """Build matched_chunks debug info."""
        matched_chunks = []
        for doc_id, payload in ranked:
            matched_terms = self._build_matched_terms(payload.get("term_traces", {}))
            main_fields = [field for field, _ in payload.get("matched_fields", Counter()).most_common(4)]

            matched_chunks.append(
                {
                    "chunk_id": str(retrieval_index.documents[doc_id].chunk.get("chunk_id", "") or ""),
                    "keyword_fusion_score": float(payload.get("fusion_score", 0.0) or 0.0),
                    "best_raw_bm25_score": float(payload.get("best_raw_score", 0.0) or 0.0),
                    "matched_fields": main_fields,
                    "matched_terms": matched_terms,
                    "query_sources": list(payload.get("query_sources", [])),
                    "noise_flags": self._detect_keyword_noise_flags(
                        matched_terms=matched_terms,
                        matched_fields=main_fields,
                        query_profile=query_profile,
                        chunk=retrieval_index.documents[doc_id].chunk,
                    ),
                    "hit_asset_field": any(
                        field in {"asset_caption", "asset_aux"} for field in main_fields
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
            )
        return matched_chunks

    @staticmethod
    def _build_matched_terms(term_traces: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build matched terms list from term traces."""
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

    def _expand_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        """Expand query tokens using query_tools."""
        expander = getattr(self.query_tools, "expand_keyword_query_tokens", None)
        if callable(expander):
            return expander(tokens)
        return tokens

    def _filter_keyword_query_tokens(self, tokens: List[str]) -> List[str]:
        """Filter query tokens using query_tools."""
        extractor = getattr(self.query_tools, "extract_query_keywords", None)
        if callable(extractor):
            return extractor(tokens, limit=max(len(tokens), 1))
        return tokens

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
        from utils.config import get_enhanced_retrieval_runtime_config

        ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

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
