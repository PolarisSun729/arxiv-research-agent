from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService, QueryProfile, RetrievalOptions


logger = logging.getLogger(__name__)


class RouteRetriever:
    """Facade for multi-route retrieval orchestration.

    This extraction keeps the mature route implementations inside
    EnhancedRetrievalService for behavior compatibility, while moving the
    route-selection pipeline into a dedicated collaborator.
    """

    def __init__(self, retrieval_service: "EnhancedRetrievalService"):
        self.retrieval_service = retrieval_service

    def build_route_bundle(
        self,
        *,
        collection_name: str,
        user_query: str,
        query_profile: "QueryProfile",
        query_views: Dict[str, Any],
        options: "RetrievalOptions",
        enable_hyde: bool,
        enable_keyword_search: bool,
        recall_candidate_limit: int,
    ) -> Dict[str, Any]:
        intent_profile = query_profile.intent_profile

        hyde_text = ""
        hyde_debug: Dict[str, Any] = {
            "enabled": enable_hyde,
            "text": "",
            "source_queries": [],
            "focus_queries": [],
            "confidence": 0.0,
        }
        if enable_hyde:
            hyde_text = self._generate_hyde_document(
                user_query,
                query_profile,
                query_views["selected_queries"],
            )
            hyde_debug = {
                "enabled": True,
                "text": hyde_text,
                "source_queries": [user_query, *query_views["selected_queries"]],
                "focus_queries": query_views["selected_queries"][:2] if query_views["selected_queries"] else [user_query],
                "confidence": self.retrieval_service._route_confidence(
                    "vector_hyde",
                    query_profile,
                    hyde_text or user_query,
                    route_queries=query_views["selected_queries"] or [user_query],
                    intent_profile=intent_profile,
                ),
            }

        routes: Dict[str, List[Dict[str, Any]]] = {}
        routes["vector_original"] = self._vector_retrieve(
            collection_name=collection_name,
            query=user_query,
            top_k=recall_candidate_limit,
            route_name="vector_original",
            source_query=user_query,
            query_profile=query_profile,
            route_queries=[user_query],
        )

        if query_views["enabled"] and query_views["selected_queries"]:
            rewrite_hits: List[Dict[str, Any]] = []
            for query in query_views["selected_queries"]:
                rewrite_hits.extend(
                    self._vector_retrieve(
                        collection_name=collection_name,
                        query=query,
                        top_k=recall_candidate_limit,
                        route_name="vector_rewrite",
                        source_query=query,
                        query_profile=query_profile,
                        route_queries=query_views["selected_queries"],
                    )
                )
            routes["vector_rewrite"] = self.retrieval_service._dedupe_preserve_order(rewrite_hits)
        else:
            routes["vector_rewrite"] = []

        if hyde_text:
            routes["vector_hyde"] = self._vector_retrieve(
                collection_name=collection_name,
                query=hyde_text,
                top_k=recall_candidate_limit,
                route_name="vector_hyde",
                source_query="hyde",
                query_profile=query_profile,
                route_queries=[hyde_text],
            )
        else:
            routes["vector_hyde"] = []

        if enable_keyword_search:
            keyword_queries = [
                user_query,
                *query_views["selected_queries"],
                query_profile.semantic_query,
                query_profile.evidence_query,
            ]
            routes["keyword"] = self._keyword_retrieve(
                collection_name=collection_name,
                queries=keyword_queries,
                top_k=recall_candidate_limit,
                query_profile=query_profile,
            )
        else:
            routes["keyword"] = []
            keyword_queries = []

        memory_context = options.memory_context or {}
        memory_retrieval_enabled = bool(self.retrieval_service._memory_flag("enable_memory_aware_retrieval", True))
        if memory_retrieval_enabled:
            try:
                routes["memory_context"] = self._memory_retrieve(
                    collection_name=collection_name,
                    memory_context=memory_context,
                    top_k=recall_candidate_limit,
                    query_profile=query_profile,
                )
                memory_fallback_reason = None
            except Exception as exc:
                logger.warning("Memory-aware retrieval failed, skipping memory route: %s", exc)
                routes["memory_context"] = []
                memory_fallback_reason = str(exc)
        else:
            routes["memory_context"] = []
            memory_fallback_reason = "disabled by runtime config"

        keyword_debug = {
            "enabled": enable_keyword_search,
            "queries": keyword_queries,
            "selected_rewrite_queries": query_views["selected_queries"],
            "query_details": self.retrieval_service._build_query_term_details(keyword_queries),
            "keywords": self.retrieval_service._build_query_keywords(keyword_queries),
        }
        memory_debug = {
            "enabled": memory_retrieval_enabled,
            "applied": bool(memory_retrieval_enabled and memory_context.get("enabled", False) and routes["memory_context"]),
            "reason": str(memory_context.get("reason", "") or ""),
            "referenced_turn_ids": memory_context.get("referenced_turn_ids", []) or [],
            "referenced_source_ids": memory_context.get("referenced_source_ids", []) or [],
            "query_keywords": memory_context.get("query_keywords", []) or [],
            "candidates": memory_context.get("candidates", []) or [],
            "route_result_count": len(routes["memory_context"]),
            "fallback_reason": memory_fallback_reason or memory_context.get("fallback_reason"),
        }

        return {
            "routes": routes,
            "hyde_text": hyde_text,
            "hyde_debug": hyde_debug,
            "keyword_debug": keyword_debug,
            "memory_debug": memory_debug,
        }

    def _generate_hyde_document(
        self,
        user_query: str,
        query_profile: "QueryProfile",
        rewritten_queries: List[str],
    ) -> str:
        if self.retrieval_service.generation_service is not None:
            try:
                text = self.retrieval_service.generation_service.generate_hyde_document(user_query)
                if text and text.strip():
                    return text.strip()
            except Exception:
                pass
        return self.retrieval_service._heuristic_hyde_document(query_profile, rewritten_queries)

    def _vector_retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        route_name: str,
        source_query: str,
        query_profile: "QueryProfile",
        route_queries: List[str],
    ) -> List[Dict[str, Any]]:
        sample_chunks = self.retrieval_service.vector_store_service.get_all_chunks(collection_name, limit=1)
        sample_metadata = sample_chunks[0].get("metadata", {}) if sample_chunks else {}
        collection_info = self.retrieval_service.vector_store_service.get_collection_info("milvus", collection_name)
        vector_dimension = None
        schema = collection_info.get("schema", {}) if isinstance(collection_info, dict) else {}
        for field in schema.get("fields", []) if isinstance(schema, dict) else []:
            if field.get("name") == "vector":
                vector_dimension = field.get("dim")
                if vector_dimension is None:
                    params = field.get("params", {})
                    if isinstance(params, dict):
                        vector_dimension = params.get("dim")
                break
        default_embedding_config = self.retrieval_service.embedding_service.get_default_embedding_config()
        embedding_provider = sample_metadata.get("embedding_provider") or default_embedding_config.provider
        embedding_model = sample_metadata.get("embedding_model") or default_embedding_config.model_name
        embedding = self.retrieval_service.embedding_service.create_single_embedding(
            query,
            provider=str(embedding_provider),
            model=str(embedding_model),
            dimension=int(vector_dimension) if vector_dimension else None,
        )
        results = self.retrieval_service.vector_store_service.search_similar_vectors(
            collection_name=collection_name,
            query_vector=embedding,
            top_k=top_k,
        )
        route_confidence = self.retrieval_service._route_confidence(
            route_name,
            query_profile,
            source_query,
            route_queries=route_queries,
            intent_profile=query_profile.intent_profile,
        )
        return self.retrieval_service._normalize_route_results(
            results,
            route_name,
            source_query,
            route_confidence,
            query_profile,
        )

    def _keyword_retrieve(
        self,
        collection_name: str,
        queries: List[str],
        top_k: int,
        query_profile: "QueryProfile",
    ) -> List[Dict[str, Any]]:
        chunks = [
            self.retrieval_service._normalize_chunk(chunk)
            for chunk in self.retrieval_service.vector_store_service.get_all_chunks(collection_name)
        ]
        if not chunks:
            return []

        doc_tokens = [self.retrieval_service._tokenize_for_keyword_search(chunk.get("content", "")) for chunk in chunks]
        avgdl = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
        document_frequencies: Dict[str, int] = {}
        for tokens in doc_tokens:
            for token in set(tokens):
                document_frequencies[token] = document_frequencies.get(token, 0) + 1

        per_chunk_scores = [0.0 for _ in chunks]
        query_signatures: List[str] = []
        for query in queries:
            tokens = self.retrieval_service._tokenize_for_keyword_search(query)
            if not tokens:
                continue
            query_signatures.append(query)
            token_counter = Counter(tokens)
            for idx, chunk in enumerate(chunks):
                per_chunk_scores[idx] += self.retrieval_service._bm25_score(
                    token_counter=token_counter,
                    doc_tokens=doc_tokens[idx],
                    doc_freqs=document_frequencies,
                    total_docs=len(chunks),
                    avg_doc_length=avgdl,
                    content=chunk.get("content", ""),
                )

        ranked = [(idx, score) for idx, score in enumerate(per_chunk_scores) if score > 0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            return []

        raw_scores = [score for _, score in ranked]
        route_confidence = self.retrieval_service._route_confidence(
            "keyword",
            query_profile,
            " | ".join(query_signatures) if query_signatures else query_profile.original_query,
            route_queries=query_signatures or [query_profile.keyword_query],
            intent_profile=query_profile.intent_profile,
        )
        normalized_scores = self.retrieval_service._normalize_scores(raw_scores)

        results: List[Dict[str, Any]] = []
        for rank, ((idx, score), normalized_score) in enumerate(zip(ranked[:top_k], normalized_scores[:top_k])):
            chunk = dict(chunks[idx])
            chunk["retrieval_route"] = "keyword"
            chunk["source_query"] = " | ".join(query_signatures)
            chunk["route_rank"] = rank + 1
            chunk["route_score"] = float(score)
            chunk["normalized_route_score"] = float(normalized_score)
            chunk["route_confidence"] = float(route_confidence)
            chunk["structural_bonus"] = float(self.retrieval_service._compute_structural_bonus(chunk, query_profile))
            results.append(chunk)
        return results

    def _memory_retrieve(
        self,
        collection_name: str,
        memory_context: Dict[str, Any],
        top_k: int,
        query_profile: "QueryProfile",
    ) -> List[Dict[str, Any]]:
        if not memory_context or not bool(memory_context.get("enabled", False)):
            return []

        candidates = [item for item in (memory_context.get("candidates", []) or []) if isinstance(item, dict)]
        if not candidates:
            return []

        try:
            chunks = [
                self.retrieval_service._normalize_chunk(chunk)
                for chunk in self.retrieval_service.vector_store_service.get_all_chunks(collection_name)
            ]
        except Exception as exc:
            logger.warning("Memory retrieval skipped because chunk load failed: %s", exc)
            return []

        if not chunks:
            return []

        query_keywords = memory_context.get("query_keywords", []) or query_profile.keywords or []
        route_results: List[Dict[str, Any]] = []
        for candidate in candidates:
            matched_chunks = [chunk for chunk in chunks if self._memory_candidate_matches(candidate, chunk)]
            for chunk in matched_chunks:
                memory_score = self._memory_candidate_score(candidate, chunk, query_keywords, query_profile)
                route_confidence = min(
                    0.78,
                    self.retrieval_service._route_confidence(
                        "memory_context",
                        query_profile,
                        str(candidate.get("content_preview", "") or query_profile.original_query),
                        route_queries=[str(candidate.get("content_preview", "") or query_profile.original_query)],
                        intent_profile=query_profile.intent_profile,
                    ) + min(0.18, memory_score * 0.15),
                )
                memory_chunk = dict(chunk)
                memory_chunk["retrieval_route"] = "memory_context"
                memory_chunk["source_query"] = query_profile.original_query
                memory_chunk["route_score"] = float(memory_score)
                memory_chunk["normalized_route_score"] = float(min(1.0, memory_score / 1.4))
                memory_chunk["route_confidence"] = float(route_confidence)
                memory_chunk["structural_bonus"] = float(self.retrieval_service._compute_structural_bonus(memory_chunk, query_profile))
                memory_chunk["memory_score"] = float(memory_score)
                memory_chunk["memory_reason"] = str(candidate.get("memory_reason", "") or memory_context.get("reason", ""))
                memory_chunk["source_turn_id"] = str(candidate.get("source_turn_id", "") or "")
                memory_chunk["is_recent_turn"] = bool(candidate.get("is_recent_turn", False))
                memory_chunk["memory_match_type"] = str(candidate.get("match_type", "metadata") or "metadata")
                memory_chunk["memory_reference_strength"] = float(candidate.get("reference_strength", 0.0) or 0.0)
                route_results.append(memory_chunk)

        ranked = sorted(
            route_results,
            key=lambda item: (
                float(item.get("memory_score", 0.0) or 0.0),
                float(item.get("route_confidence", 0.0) or 0.0),
                float(item.get("structural_bonus", 0.0) or 0.0),
            ),
            reverse=True,
        )
        deduped = self.retrieval_service._dedupe_route_results(ranked)
        for rank, item in enumerate(deduped[:top_k], start=1):
            item["route_rank"] = rank
        return deduped[:top_k]

    @staticmethod
    def _memory_candidate_matches(candidate: Dict[str, Any], chunk: Dict[str, Any]) -> bool:
        candidate_ids = {
            str(candidate.get("chunk_id", "") or "").strip(),
            str(candidate.get("parent_chunk_id", "") or "").strip(),
            str(candidate.get("original_chunk_id", "") or "").strip(),
            str(candidate.get("source_id", "") or "").strip(),
        }
        candidate_ids.discard("")
        chunk_ids = {
            str(chunk.get("chunk_id", "") or "").strip(),
            str(chunk.get("parent_chunk_id", "") or "").strip(),
            str(chunk.get("original_chunk_id", "") or "").strip(),
        }
        chunk_ids.discard("")
        if candidate_ids and candidate_ids & chunk_ids:
            return True

        candidate_source = str(candidate.get("source", "") or "").strip().lower()
        candidate_section = str(candidate.get("section_path", "") or "").strip().lower()
        candidate_page = str(candidate.get("page_number", "") or "").strip()
        chunk_source = str(chunk.get("source", "") or "").strip().lower()
        chunk_section = str(chunk.get("section_path", "") or "").strip().lower()
        chunk_page = str(chunk.get("page_number", "") or "").strip()

        if candidate_source and candidate_section and candidate_source == chunk_source and candidate_section == chunk_section:
            return True
        if candidate_source and candidate_page and candidate_source == chunk_source and candidate_page == chunk_page:
            return True
        return False

    def _memory_candidate_score(
        self,
        candidate: Dict[str, Any],
        chunk: Dict[str, Any],
        query_keywords: List[str],
        query_profile: "QueryProfile",
    ) -> float:
        score = 0.25
        if bool(candidate.get("is_recent_turn", False)):
            score += 0.28
        score += min(0.24, float(candidate.get("reference_strength", 0.0) or 0.0) * 0.24)
        candidate_ids = {
            str(candidate.get("chunk_id", "") or "").strip(),
            str(candidate.get("parent_chunk_id", "") or "").strip(),
            str(candidate.get("original_chunk_id", "") or "").strip(),
            str(candidate.get("source_id", "") or "").strip(),
        }
        chunk_ids = {
            str(chunk.get("chunk_id", "") or "").strip(),
            str(chunk.get("parent_chunk_id", "") or "").strip(),
            str(chunk.get("original_chunk_id", "") or "").strip(),
        }
        if {item for item in candidate_ids if item} & {item for item in chunk_ids if item}:
            score += 0.34

        chunk_terms_text = " ".join(
            [
                str(chunk.get("content", "") or ""),
                str(chunk.get("section_path", "") or ""),
                str(chunk.get("asset_summary", "") or ""),
            ]
        ).lower()
        overlap = 0
        for keyword in query_keywords[:10]:
            token = str(keyword or "").strip().lower()
            if token and token in chunk_terms_text:
                overlap += 1
        if query_keywords:
            score += min(0.28, overlap / max(len(query_keywords[:10]), 1) * 0.28)

        score += max(0.0, float(self.retrieval_service._compute_structural_bonus(chunk, query_profile)))
        return min(1.4, score)
