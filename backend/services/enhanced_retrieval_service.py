import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.embedding_service import EmbeddingService
from services.generation_service import GenerationService
from services.vector_store_service import VectorStoreService
from utils.config import RETRIEVAL_CONFIG


@dataclass
class RetrievalOptions:
    top_k: int = RETRIEVAL_CONFIG["default_top_k"]
    enable_query_rewrite: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_keyword_search: Optional[bool] = None
    debug: Optional[bool] = None


class EnhancedRetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        generation_service: Optional[GenerationService] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.generation_service = generation_service
        self.rrf_k = RETRIEVAL_CONFIG["rrf_k"]
        self.route_weights = RETRIEVAL_CONFIG["route_weights"]
        self.candidate_multiplier = RETRIEVAL_CONFIG["candidate_multiplier"]

    def enhanced_retrieve(
        self,
        user_query: str,
        collection_name: str,
        options: Optional[RetrievalOptions] = None,
    ) -> Dict[str, Any]:
        options = options or RetrievalOptions()
        effective_top_k = max(1, options.top_k or RETRIEVAL_CONFIG["default_top_k"])
        enable_query_rewrite = self._resolve_option(options.enable_query_rewrite, RETRIEVAL_CONFIG["enable_query_rewrite"])
        enable_hyde = self._resolve_option(options.enable_hyde, RETRIEVAL_CONFIG["enable_hyde"])
        enable_keyword_search = self._resolve_option(options.enable_keyword_search, RETRIEVAL_CONFIG["enable_keyword_search"])
        debug_enabled = self._resolve_option(options.debug, RETRIEVAL_CONFIG["debug"])
        candidate_k = max(effective_top_k, effective_top_k * self.candidate_multiplier)

        rewritten_queries: List[str] = []
        if enable_query_rewrite:
            rewritten_queries = self._generate_query_rewrites(user_query)

        hyde_text = ""
        if enable_hyde:
            hyde_text = self._generate_hyde_document(user_query, rewritten_queries)

        routes: Dict[str, List[Dict[str, Any]]] = {}
        routes["vector_original"] = self._vector_retrieve(
            collection_name=collection_name,
            query=user_query,
            top_k=candidate_k,
            route_name="vector_original",
            source_query=user_query,
        )

        if rewritten_queries:
            rewrite_hits = []
            for query in rewritten_queries:
                rewrite_hits.extend(
                    self._vector_retrieve(
                        collection_name=collection_name,
                        query=query,
                        top_k=candidate_k,
                        route_name="vector_rewrite",
                        source_query=query,
                    )
                )
            routes["vector_rewrite"] = self._dedupe_preserve_order(rewrite_hits)
        else:
            routes["vector_rewrite"] = []

        if hyde_text:
            routes["vector_hyde"] = self._vector_retrieve(
                collection_name=collection_name,
                query=hyde_text,
                top_k=candidate_k,
                route_name="vector_hyde",
                source_query="hyde",
            )
        else:
            routes["vector_hyde"] = []

        if enable_keyword_search:
            keyword_queries = [user_query, *rewritten_queries]
            routes["keyword"] = self._keyword_retrieve(
                collection_name=collection_name,
                queries=keyword_queries,
                top_k=candidate_k,
            )
        else:
            routes["keyword"] = []

        fused_results = self._fuse_routes(routes, effective_top_k)
        result: Dict[str, Any] = {
            "chunks": fused_results,
        }

        if debug_enabled:
            result["debug"] = {
                "original_query": user_query,
                "rewritten_queries": rewritten_queries,
                "hyde_text": hyde_text,
                "routes": {
                    route_name: [self._debug_chunk_item(item) for item in route_results]
                    for route_name, route_results in routes.items()
                },
                "final_chunks": [self._debug_chunk_item(item) for item in fused_results],
                "config": {
                    "top_k": effective_top_k,
                    "candidate_k": candidate_k,
                    "enable_query_rewrite": enable_query_rewrite,
                    "enable_hyde": enable_hyde,
                    "enable_keyword_search": enable_keyword_search,
                },
                "fusion": {
                    "algorithm": "weighted_rrf",
                    "rrf_k": self.rrf_k,
                    "route_weights": self.route_weights,
                },
            }

        return result

    def _resolve_option(self, runtime_value: Optional[bool], default_value: bool) -> bool:
        return default_value if runtime_value is None else bool(runtime_value)

    def _vector_retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        route_name: str,
        source_query: str,
    ) -> List[Dict[str, Any]]:
        embedding = self.embedding_service.create_single_embedding_local(query)
        results = self.vector_store_service.search_similar_vectors(
            collection_name=collection_name,
            query_vector=embedding,
            top_k=top_k,
        )
        normalized = []
        for rank, item in enumerate(results):
            chunk = self._normalize_chunk(item)
            chunk["retrieval_route"] = route_name
            chunk["source_query"] = source_query
            chunk["route_rank"] = rank + 1
            chunk["route_score"] = float(item.get("score", 0.0) or 0.0)
            normalized.append(chunk)
        return normalized

    def _keyword_retrieve(
        self,
        collection_name: str,
        queries: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        chunks = [self._normalize_chunk(chunk) for chunk in self.vector_store_service.get_all_chunks(collection_name)]
        if not chunks:
            return []

        doc_tokens = [self._tokenize_for_keyword_search(chunk.get("content", "")) for chunk in chunks]
        avgdl = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
        document_frequencies = defaultdict(int)
        for tokens in doc_tokens:
            for token in set(tokens):
                document_frequencies[token] += 1

        per_chunk_scores = [0.0 for _ in chunks]
        query_signatures: List[str] = []
        for query in queries:
            tokens = self._tokenize_for_keyword_search(query)
            if not tokens:
                continue
            query_signatures.append(query)
            token_counter = Counter(tokens)
            for idx, chunk in enumerate(chunks):
                per_chunk_scores[idx] += self._bm25_score(
                    token_counter=token_counter,
                    doc_tokens=doc_tokens[idx],
                    doc_freqs=document_frequencies,
                    total_docs=len(chunks),
                    avg_doc_length=avgdl,
                    content=chunk.get("content", ""),
                )

        ranked: List[Tuple[int, float]] = [
            (idx, score)
            for idx, score in enumerate(per_chunk_scores)
            if score > 0
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked[:top_k]):
            chunk = dict(chunks[idx])
            chunk["retrieval_route"] = "keyword"
            chunk["source_query"] = " | ".join(query_signatures)
            chunk["route_rank"] = rank + 1
            chunk["route_score"] = float(score)
            results.append(chunk)
        return results

    def _fuse_routes(self, routes: Dict[str, List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        aggregated: Dict[str, Dict[str, Any]] = {}
        for route_name, route_results in routes.items():
            weight = self.route_weights.get(route_name, 1.0)
            for rank, item in enumerate(route_results):
                chunk_key = self._chunk_unique_key(item)
                entry = aggregated.setdefault(
                    chunk_key,
                    {
                        **item,
                        "score": 0.0,
                        "matched_routes": [],
                        "route_scores": {},
                        "source_queries": [],
                    },
                )
                entry["score"] += weight / (self.rrf_k + rank + 1)
                entry["matched_routes"].append(route_name)
                entry["route_scores"][route_name] = item.get("route_score")
                if item.get("source_query") and item["source_query"] not in entry["source_queries"]:
                    entry["source_queries"].append(item["source_query"])

        fused = sorted(
            aggregated.values(),
            key=lambda item: (
                float(item.get("score", 0.0)),
                max(item.get("route_scores", {}).values() or [0.0]),
            ),
            reverse=True,
        )
        return fused[:top_k]

    def _normalize_chunk(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = item.get("metadata", {})
        chunk = dict(item)
        chunk["content"] = item.get("content") or item.get("text") or ""
        chunk["text"] = chunk["content"]
        chunk["source"] = item.get("source") or metadata.get("source", "")
        chunk["document_name"] = item.get("document_name") or metadata.get("document_name", "")
        chunk["chunk_id"] = item.get("chunk_id") or metadata.get("chunk_id", 0)
        chunk["chunk_index"] = item.get("chunk_index") or metadata.get("chunk_index", 0)
        chunk["parent_chunk_id"] = item.get("parent_chunk_id") or metadata.get("parent_chunk_id", chunk["chunk_id"])
        chunk["original_chunk_id"] = item.get("original_chunk_id") or metadata.get("original_chunk_id", chunk["parent_chunk_id"])
        chunk["page_number"] = item.get("page_number") or metadata.get("page_number", "")
        chunk["page_start"] = item.get("page_start") or metadata.get("page_start")
        chunk["page_end"] = item.get("page_end") or metadata.get("page_end")
        chunk["page_range"] = item.get("page_range") or metadata.get("page_range", "")
        chunk["subchunk_label"] = item.get("subchunk_label") or metadata.get("subchunk_label", "")
        chunk["content_part_label"] = item.get("content_part_label") or metadata.get("content_part_label", "")
        return chunk

    def _generate_query_rewrites(self, user_query: str) -> List[str]:
        rewrites: List[str] = []
        if self.generation_service is not None:
            try:
                rewrites = self.generation_service.rewrite_query_for_retrieval(user_query, max_queries=3)
            except Exception:
                rewrites = []

        heuristic_rewrites = self._heuristic_query_rewrites(user_query)
        all_queries = []
        seen = set()
        for query in [*rewrites, *heuristic_rewrites]:
            normalized = self._normalize_query_text(query)
            if normalized and normalized not in seen and normalized != self._normalize_query_text(user_query):
                seen.add(normalized)
                all_queries.append(query.strip())
        return all_queries[:3]

    def _generate_hyde_document(self, user_query: str, rewritten_queries: List[str]) -> str:
        if self.generation_service is not None:
            try:
                return self.generation_service.generate_hyde_document(user_query)
            except Exception:
                pass
        return self._heuristic_hyde_document(user_query, rewritten_queries)

    def _heuristic_query_rewrites(self, user_query: str) -> List[str]:
        query_lower = user_query.lower()
        intents: List[str] = []
        intent_map = {
            "method": ["方法", "method", "approach", "methodology", "framework", "architecture"],
            "dataset": ["数据集", "dataset", "datasets", "benchmark", "benchmarks", "data"],
            "baseline": ["baseline", "baselines", "对比", "比较", "compared", "comparison"],
            "ablation": ["ablation", "消融", "component", "study"],
            "limitation": ["limitation", "limitations", "局限", "future work", "weakness"],
            "experiment": ["实验", "experiment", "evaluation", "results", "metrics"],
        }
        for intent, keywords in intent_map.items():
            if any(keyword in query_lower for keyword in keywords):
                intents.append(intent)

        rewrites = []
        if not intents:
            rewrites.extend(
                [
                    f"paper section answering: {user_query}",
                    f"research paper evidence about {user_query}",
                ]
            )

        if "method" in intents:
            rewrites.extend(
                [
                    "proposed method approach methodology framework model architecture",
                    "training objective algorithm design implementation details",
                ]
            )
        if "dataset" in intents:
            rewrites.extend(
                [
                    "experimental setup datasets benchmarks corpus evaluation data splits",
                    "which datasets benchmarks and evaluation settings are used",
                ]
            )
        if "baseline" in intents:
            rewrites.extend(
                [
                    "baseline methods comparison models compared against",
                    "competing methods baselines experimental comparison",
                ]
            )
        if "ablation" in intents:
            rewrites.extend(
                [
                    "ablation study component analysis effect of modules",
                    "appendix ablation analysis model variants",
                ]
            )
        if "limitation" in intents:
            rewrites.extend(
                [
                    "limitations discussion future work weaknesses",
                    "section limitations assumptions failure cases",
                ]
            )
        if "experiment" in intents:
            rewrites.extend(
                [
                    "experimental results evaluation metrics analysis",
                    "results discussion quantitative comparison",
                ]
            )

        return rewrites[:4]

    def _heuristic_hyde_document(self, user_query: str, rewritten_queries: List[str]) -> str:
        focus = ", ".join(rewritten_queries[:2]) if rewritten_queries else user_query
        return (
            "The paper likely contains a section directly relevant to the question. "
            "It describes the main evidence, terminology, experimental setup, or discussion needed to answer: "
            f"{user_query}. Relevant paper wording may include: {focus}. "
            "The most useful chunk is expected to come from a method, experiments, ablation, baseline comparison, or limitations section."
        )

    def _tokenize_for_keyword_search(self, text: str) -> List[str]:
        lowered = text.lower()
        english_tokens = re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", lowered)
        chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
        return english_tokens + chinese_tokens

    def _bm25_score(
        self,
        token_counter: Counter,
        doc_tokens: List[str],
        doc_freqs: Dict[str, int],
        total_docs: int,
        avg_doc_length: float,
        content: str,
    ) -> float:
        doc_counts = Counter(doc_tokens)
        doc_len = max(len(doc_tokens), 1)
        k1 = 1.5
        b = 0.75
        score = 0.0
        content_lower = content.lower()

        for token, qtf in token_counter.items():
            if token not in doc_counts:
                continue
            df = max(doc_freqs.get(token, 0), 1)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            tf = doc_counts[token]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_length, 1.0)))
            score += qtf * idf * (numerator / denominator)

            if token in {"method", "methods", "dataset", "datasets", "baseline", "ablation", "limitation", "limitations"}:
                if token in content_lower[:300]:
                    score += 0.2

        return score

    def _dedupe_preserve_order(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped = []
        seen = set()
        for item in items:
            key = self._chunk_unique_key(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _chunk_unique_key(self, item: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(item.get("source", "")),
                str(item.get("original_chunk_id", item.get("parent_chunk_id", item.get("chunk_id", 0)))),
                str(item.get("content_part_label", "")),
                str(item.get("page_range", "")),
            ]
        )

    def _normalize_query_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _debug_chunk_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        preview = item.get("content", "")[:220].replace("\n", " ").strip()
        return {
            "chunk_id": item.get("chunk_id"),
            "original_chunk_id": item.get("original_chunk_id"),
            "page_number": item.get("page_number"),
            "page_range": item.get("page_range"),
            "score": item.get("score"),
            "route_score": item.get("route_score"),
            "retrieval_route": item.get("retrieval_route"),
            "matched_routes": item.get("matched_routes", []),
            "source_query": item.get("source_query"),
            "source_queries": item.get("source_queries", []),
            "subchunk_label": item.get("subchunk_label"),
            "preview": preview,
        }
