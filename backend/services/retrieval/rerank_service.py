from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import requests
import torch

from services.retrieval.contracts import QueryProfile
from utils.config import get_enhanced_retrieval_runtime_config

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

logger = logging.getLogger(__name__)


class RerankService:
    """负责 rerank query、document 构造、reranker 调用、分数映射与 rerank debug。"""

    def __init__(
        self,
        *,
        generation_service: Any,
        config_owner: Any,
        reranker_loader: Any,
        dashscope_request_spec_builder: Any,
        dashscope_result_extractor: Any,
        query_normalizer: Any,
        intent_bucket: Any,
        query_profile_debugger: Any,
        trace_builder: Any,
    ) -> None:
        self.generation_service = generation_service
        self.config_owner = config_owner
        self.reranker_loader = reranker_loader
        self.dashscope_request_spec_builder = dashscope_request_spec_builder
        self.dashscope_result_extractor = dashscope_result_extractor
        self.query_normalizer = query_normalizer
        self.intent_bucket = intent_bucket
        self.query_profile_debugger = query_profile_debugger
        self.trace_builder = trace_builder

    def __getattr__(self, name: str) -> Any:
        # rerank 配置仍由兼容入口持有，避免运行时修改 service.llm_rerank_provider 等旧写法失效。
        if name.startswith("llm_rerank_") or name in {"_llm_reranker_error", "_llm_reranker_path", "_llm_reranker_device"}:
            return getattr(self.config_owner, name)
        raise AttributeError(name)

    def build_rerank_query(self, user_query: str, query_profile: QueryProfile) -> str:
        if self.generation_service is not None and hasattr(self.generation_service, "build_rerank_query"):
            try:
                rerank_query = self.generation_service.build_rerank_query(user_query, intent_profile=query_profile.intent_profile.to_dict())
                if rerank_query and rerank_query.strip():
                    actual_rerank_query = rerank_query.strip()
                    logger.debug("original_question=%s", user_query)
                    logger.debug("actual_rerank_query=%s", actual_rerank_query)
                    return actual_rerank_query
            except Exception as exc:  # pragma: no cover
                logger.debug("Failed to rewrite rerank query with model: %s", exc)
        actual_rerank_query = self.fallback_rerank_query(user_query, query_profile=query_profile)
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
                return self.limit_rerank_text(content, self.config_owner.llm_rerank_max_doc_chars)
        content = self.limit_rerank_text(str(chunk.get("rerank_text", "") or ""), self.config_owner.llm_rerank_max_doc_chars)
        if not content:
            content = self.limit_rerank_text(str(chunk.get("content", "") or ""), self.config_owner.llm_rerank_max_doc_chars)
        if not content:
            content = self.limit_rerank_text(str(chunk.get("text", "") or ""), self.config_owner.llm_rerank_max_doc_chars)
        section_title = str(chunk.get("section_title", "") or "").strip()
        if section_title:
            normalized_title = self.query_normalizer(section_title)
            normalized_content = self.query_normalizer(content)
            if normalized_title and normalized_title not in normalized_content[: max(len(normalized_title), 1) * 2]:
                content = f"{section_title}\n{content}".strip()
        return self.limit_rerank_text(content, self.config_owner.llm_rerank_max_doc_chars)

    def llm_rerank(
        self,
        rerank_query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        query_profile: Optional[QueryProfile] = None,
        original_question: Optional[str] = None,
        candidate_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        rerank_limit = min(len(chunks), max(1, int(candidate_limit or self.config_owner.llm_rerank_candidate_limit)))
        rerank_limit = max(0, rerank_limit)
        limited_chunks = chunks[: max(1, top_k)]
        original_question = original_question or rerank_query
        if not chunks:
            return self._fallback_result([], [], "no_candidates", rerank_query, original_question, query_profile, rerank_limit)

        candidate_chunks = [dict(chunk) for chunk in chunks[:rerank_limit]]
        rerank_documents = [self.build_rerank_document_text(chunk) for chunk in candidate_chunks]
        self.log_rerank_inputs(original_question, rerank_query, candidate_chunks, rerank_documents)

        if self.config_owner.llm_rerank_provider == "dashscope":
            remote_result = self.rerank_with_dashscope(rerank_query, chunks, candidate_chunks, rerank_documents, rerank_limit, top_k, query_profile)
            if remote_result is not None:
                return remote_result
            if not self.config_owner.llm_rerank_fallback_local:
                return self._fallback_result(limited_chunks, chunks, self.config_owner._llm_reranker_error or "remote_rerank_unavailable", rerank_query, original_question, query_profile, rerank_limit, mode="remote_failed", provider="dashscope")

        reranker = self.reranker_loader()
        if reranker is None:
            return self._fallback_result(limited_chunks, chunks, self.config_owner._llm_reranker_error or "llm_reranker_unavailable", rerank_query, original_question, query_profile, rerank_limit)

        pairs = [(rerank_query, document_text) for document_text in rerank_documents]
        try:
            raw_scores = reranker.predict(
                pairs,
                prompt=self.config_owner.llm_rerank_prompt,
                batch_size=self.config_owner.llm_rerank_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=self.config_owner._llm_reranker_device,
            )
            rerank_scores = torch.sigmoid(torch.as_tensor(raw_scores, dtype=torch.float32)).tolist()
            self.log_rerank_raw_scores("local", [{"returned_index": idx, "relevance_score": float(score)} for idx, score in enumerate(rerank_scores, start=1)])
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to rerank chunks with Qwen3-VL-Reranker")
            return self._fallback_result(limited_chunks, chunks, f"inference_error: {exc}", rerank_query, original_question, query_profile, rerank_limit)

        mapped_candidates, candidate_debug = self._map_local_scores(candidate_chunks, rerank_scores, rerank_documents)
        mapped_candidates.sort(key=lambda item: float(item.get("rerank_score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(mapped_candidates, start=1):
            item["rerank_rank"] = rank
            item["chunk"]["llm_rerank_rank"] = rank
        self.log_rerank_mapped_results("local", mapped_candidates)
        reranked_chunks = [item["chunk"] for item in mapped_candidates] + self._build_tail_chunks(chunks, rerank_limit, model_name=self.config_owner._llm_reranker_path)
        final_chunks = reranked_chunks[: max(1, top_k)]
        return {
            "chunks": final_chunks,
            "reranked_chunks": reranked_chunks,
            "debug": self._success_debug(
                mode="cross_encoder",
                model_path=self.config_owner._llm_reranker_path,
                device=self.config_owner._llm_reranker_device,
                chunks=chunks,
                final_chunks=final_chunks,
                rerank_limit=rerank_limit,
                rerank_query=rerank_query,
                original_question=original_question,
                query_profile=query_profile,
                candidate_debug=candidate_debug,
            ),
        }

    def rerank_with_dashscope(self, user_query: str, chunks: List[Dict[str, Any]], candidate_chunks: List[Dict[str, Any]], rerank_documents: List[str], rerank_limit: int, top_k: int, query_profile: Optional[QueryProfile] = None) -> Optional[Dict[str, Any]]:
        api_key = self.config_owner.llm_rerank_api_key
        if not api_key:
            self.config_owner._llm_reranker_error = "dashscope_api_key_not_set"
            return None
        documents = [doc[: self.config_owner.llm_rerank_max_doc_chars] for doc in rerank_documents]
        request_url, payload = self.dashscope_request_spec_builder(user_query=user_query, documents=documents, rerank_limit=rerank_limit)
        try:
            response = requests.post(request_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            self.config_owner._llm_reranker_error = f"dashscope_rerank_request_failed: {exc}"
            logger.warning("DashScope rerank failed: %s, will %s", exc, "fallback to local model" if self.config_owner.llm_rerank_fallback_local else "stop")
            return None
        results = self.dashscope_result_extractor(data)
        if not isinstance(results, list):
            self.config_owner._llm_reranker_error = f"dashscope_rerank_invalid_response: {data}"
            return None
        self.log_rerank_raw_scores("dashscope", results)
        score_by_index: Dict[int, float] = {}
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score"))
            try:
                if index is not None and score is not None:
                    score_by_index[int(index)] = float(score)
            except Exception:
                continue
        if not score_by_index:
            self.config_owner._llm_reranker_error = f"dashscope_rerank_empty_scores: {data}"
            return None
        self.config_owner._llm_reranker_path = self.config_owner.llm_rerank_model_name
        self.config_owner._llm_reranker_device = "remote"
        self.config_owner._llm_reranker_error = None
        ranked_candidates, candidate_debug, mapped_debug = self._map_dashscope_scores(candidate_chunks, rerank_documents, score_by_index)
        ranked_candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(ranked_candidates, start=1):
            item["llm_rerank_rank"] = rank
        self.log_rerank_mapped_results("dashscope", mapped_debug)
        reranked_chunks = ranked_candidates + self._build_tail_chunks(chunks, rerank_limit, model_name=self.config_owner.llm_rerank_model_name)
        final_chunks = reranked_chunks[: max(1, top_k)]
        return {
            "chunks": final_chunks,
            "reranked_chunks": reranked_chunks,
            "debug": self._success_debug(
                mode="dashscope",
                model_path=self.config_owner.llm_rerank_model_name,
                device="remote",
                chunks=chunks,
                final_chunks=final_chunks,
                rerank_limit=rerank_limit,
                rerank_query=user_query,
                original_question=user_query,
                query_profile=query_profile,
                candidate_debug=candidate_debug,
                provider="dashscope",
            ),
        }

    def fallback_rerank_query(self, user_query: str, query_profile: QueryProfile) -> str:
        normalized_question = re.sub(r"\s+", " ", (user_query or "")).strip()
        base_query = (
            "Select the passage that most directly supports an answer to the user's question. "
            "Prefer evidence-bearing chunks with explicit facts, definitions, steps, causes, results, comparisons, or other answerable statements; "
            "down-rank passages that are only loosely topic-related or background. "
        )
        intent_profile = query_profile.intent_profile
        if intent_profile is not None:
            main_intent = self.intent_bucket(intent_profile.main_intent)
            clauses = {
                "summary": "Prioritize abstract, introduction, and conclusion passages that state the paper's main contribution or findings. ",
                "method": "Prioritize method, architecture, training, inference, and implementation details. ",
                "experiment": "Prioritize experiment, evaluation, results, metric, baseline, and ablation evidence. ",
                "comparison": "Prioritize direct baseline comparisons and ablation evidence. ",
                "dataset": "Prioritize dataset, corpus, benchmark, split, and data description passages. ",
                "limitation": "Prioritize limitations, failure cases, discussion, and future work. ",
                "figure_table": "Prioritize figure captions, table captions, appendix references, and visual explanations. ",
            }
            base_query += clauses.get(main_intent, "")
            if intent_profile.preferred_sections:
                base_query += f"Favor sections such as: {', '.join(intent_profile.preferred_sections[:4])}. "
        return f"{base_query}Original question: {normalized_question}" if normalized_question else base_query.rstrip()

    def _map_local_scores(self, candidate_chunks: List[Dict[str, Any]], rerank_scores: List[float], rerank_documents: List[str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        mapped_candidates: List[Dict[str, Any]] = []
        candidate_debug: List[Dict[str, Any]] = []
        for idx, (chunk, score, document_text) in enumerate(zip(candidate_chunks, rerank_scores, rerank_documents), start=1):
            rerank_score = float(score)
            fused_score = float(chunk.get("score", 0.0) or 0.0)
            reranked_chunk = dict(chunk)
            reranked_chunk.update({
                "fusion_score": fused_score,
                "fusion_rank": int(chunk.get("fusion_rank", chunk.get("route_rank", idx)) or idx),
                "llm_rerank_score": rerank_score,
                "llm_rerank_model": self.config_owner._llm_reranker_path,
                "llm_rerank_prompt": self.config_owner.llm_rerank_prompt,
                "llm_rerank_input_rank": idx,
                "llm_rerank_returned_index": idx,
                "llm_rerank_document_preview": document_text[: ENHANCED_RETRIEVAL_CONFIG["rerank_document_preview_limit"]],
                "llm_rerank_used_compressed_text": bool(reranked_chunk.get("rerank_text")),
                "score": rerank_score,
            })
            mapped_candidates.append({"rerank_rank": None, "returned_index": idx, "chunk_id": reranked_chunk.get("chunk_id"), "page_number": reranked_chunk.get("page_number"), "fusion_rank": reranked_chunk.get("fusion_rank"), "fusion_score": fused_score, "rerank_score": rerank_score, "text_preview": document_text[: ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]], "chunk": reranked_chunk})
            candidate_debug.append(self._candidate_debug_row(idx, reranked_chunk, fused_score, rerank_score))
        return mapped_candidates, candidate_debug

    def _map_dashscope_scores(self, candidate_chunks: List[Dict[str, Any]], rerank_documents: List[str], score_by_index: Dict[int, float]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        ranked_candidates: List[Dict[str, Any]] = []
        candidate_debug: List[Dict[str, Any]] = []
        mapped_debug: List[Dict[str, Any]] = []
        for idx, (chunk, document_text) in enumerate(zip(candidate_chunks, rerank_documents), start=1):
            returned_index = idx if idx in score_by_index else (idx - 1 if (idx - 1) in score_by_index else None)
            rerank_score = float(score_by_index.get(idx - 1, score_by_index.get(idx, 0.0)))
            fused_score = float(chunk.get("score", 0.0) or 0.0)
            reranked_chunk = dict(chunk)
            reranked_chunk.update({"fusion_score": fused_score, "fusion_rank": int(chunk.get("fusion_rank", chunk.get("route_rank", idx)) or idx), "llm_rerank_score": rerank_score, "llm_rerank_model": self.config_owner.llm_rerank_model_name, "llm_rerank_prompt": self.config_owner.llm_rerank_prompt, "llm_rerank_input_rank": idx, "llm_rerank_returned_index": returned_index, "llm_rerank_document_preview": document_text[: ENHANCED_RETRIEVAL_CONFIG["rerank_document_preview_limit"]], "llm_rerank_used_compressed_text": bool(reranked_chunk.get("rerank_text")), "score": rerank_score})
            ranked_candidates.append(reranked_chunk)
            candidate_debug.append(self._candidate_debug_row(idx, reranked_chunk, fused_score, rerank_score))
            mapped_debug.append({"rerank_rank": idx, "returned_index": returned_index, "chunk_id": reranked_chunk.get("chunk_id"), "page_number": reranked_chunk.get("page_number"), "fusion_rank": reranked_chunk.get("fusion_rank"), "fusion_score": fused_score, "rerank_score": rerank_score, "text_preview": reranked_chunk.get("llm_rerank_document_preview", "")[: ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]]})
        return ranked_candidates, candidate_debug, mapped_debug

    def _candidate_debug_row(self, idx: int, chunk: Dict[str, Any], fused_score: float, rerank_score: float) -> Dict[str, Any]:
        return {"input_rank": idx, "chunk_id": chunk.get("chunk_id"), "page_number": chunk.get("page_number"), "fusion_score": fused_score, "rerank_score": rerank_score, "rerank_text_preview": self.trace_builder.short_text_preview(chunk.get("rerank_text", ""), ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]), "rerank_document_preview": chunk.get("llm_rerank_document_preview", "")[: ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]], "section_tags": chunk.get("section_tags", []), "source_query": chunk.get("source_query", "")}

    def _build_tail_chunks(self, chunks: List[Dict[str, Any]], rerank_limit: int, *, model_name: Optional[str]) -> List[Dict[str, Any]]:
        tail_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[rerank_limit:], start=rerank_limit + 1):
            tail_chunk = dict(chunk)
            tail_chunk.update({"fusion_score": float(tail_chunk.get("score", 0.0) or 0.0), "fusion_rank": int(tail_chunk.get("fusion_rank", tail_chunk.get("route_rank", idx)) or idx), "llm_rerank_score": None, "llm_rerank_model": model_name, "llm_rerank_prompt": self.config_owner.llm_rerank_prompt, "llm_rerank_input_rank": idx, "llm_rerank_rank": None})
            tail_chunks.append(tail_chunk)
        return tail_chunks

    def _fallback_result(self, chunks: List[Dict[str, Any]], all_chunks: List[Dict[str, Any]], reason: str, rerank_query: str, original_question: str, query_profile: Optional[QueryProfile], rerank_limit: int, *, mode: str = "passthrough", provider: Optional[str] = None) -> Dict[str, Any]:
        debug = {"enabled": True, "applied": False, "mode": mode, "reason": reason, "model_path": self.config_owner._llm_reranker_path, "device": self.config_owner._llm_reranker_device, "input_chunks": len(all_chunks), "output_chunks": len(chunks), "candidate_limit": rerank_limit, "batch_size": self.config_owner.llm_rerank_batch_size, "query": rerank_query, "original_question": original_question, "query_profile": self.query_profile_debugger(query_profile) if query_profile else None}
        if provider:
            debug["provider"] = provider
        return {"chunks": chunks, "reranked_chunks": chunks, "debug": debug}

    def _success_debug(self, *, mode: str, model_path: str, device: str, chunks: List[Dict[str, Any]], final_chunks: List[Dict[str, Any]], rerank_limit: int, rerank_query: str, original_question: str, query_profile: Optional[QueryProfile], candidate_debug: List[Dict[str, Any]], provider: Optional[str] = None) -> Dict[str, Any]:
        debug = {"enabled": True, "applied": True, "mode": mode, "reason": "ok", "model_path": model_path, "device": device, "batch_size": self.config_owner.llm_rerank_batch_size, "candidate_limit": rerank_limit, "input_chunks": len(chunks), "output_chunks": len(final_chunks), "query": rerank_query, "original_question": original_question, "query_profile": self.query_profile_debugger(query_profile) if query_profile else None, "candidate_scores": candidate_debug[: min(ENHANCED_RETRIEVAL_CONFIG["candidate_debug_limit"], len(candidate_debug))]}
        if provider:
            debug["provider"] = provider
        return debug

    @staticmethod
    def limit_rerank_text(text: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        return normalized if len(normalized) <= max_chars else normalized[:max_chars].rstrip()

    def log_rerank_inputs(self, original_question: str, rerank_query: str, candidate_chunks: List[Dict[str, Any]], rerank_documents: List[str]) -> None:
        logger.debug("original_question=%s", original_question)
        logger.debug("actual_rerank_query=%s", rerank_query)
        logger.debug("document_count=%d", len(rerank_documents))
        for idx, (chunk, document_text) in enumerate(zip(candidate_chunks, rerank_documents), start=1):
            logger.debug("rerank_input[%d] input_index=%s chunk_id=%s chunk_type=%s asset_kind=%s page=%s fusion_rank=%s fusion_score=%s clean_document_preview=%s", idx, idx, chunk.get("chunk_id"), chunk.get("chunk_type"), chunk.get("asset_kind"), chunk.get("page_number") or chunk.get("page_range"), chunk.get("fusion_rank"), chunk.get("fusion_score", chunk.get("score")), self.trace_builder.short_text_preview(document_text, ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]))

    def log_rerank_raw_scores(self, provider: str, raw_results: List[Any]) -> None:
        logger.debug("rerank_raw_results provider=%s count=%d", provider, len(raw_results))

    def log_rerank_mapped_results(self, provider: str, mapped_results: List[Dict[str, Any]]) -> None:
        logger.debug("rerank_mapped_results provider=%s count=%d", provider, len(mapped_results))
