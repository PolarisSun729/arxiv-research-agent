from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import requests

try:  # pragma: no cover - optional heavy dependency, environment dependent
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency import is environment dependent
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

from services.retrieval.contracts import QueryProfile
from services.llm.call_metrics import record_llm_call, record_llm_usage
from services.intent.intent_service import EXPERIMENT_INTENTS, METHOD_INTENTS, OVERVIEW_INTENTS
from services.retrieval.table_evidence_formatter import render_table_evidence_rerank_text
from utils.config import get_enhanced_retrieval_runtime_config
from utils.model_utils import get_huggingface_model_path

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

logger = logging.getLogger(__name__)


class RerankService:
    """负责 rerank query、document 构造、远程/本地 rerank、fallback 和 debug。"""

    def __init__(
        self,
        *,
        generation_service: Any,
        config_owner: Any,
        query_normalizer: Any,
        query_profile_debugger: Any,
        trace_builder: Any,
    ) -> None:
        self.generation_service = generation_service
        self.config_owner = config_owner
        self.query_normalizer = query_normalizer
        self.query_profile_debugger = query_profile_debugger
        self.trace_builder = trace_builder
        self._llm_reranker = None
        self._llm_reranker_lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        # 兼容历史调用：外部仍可通过 EnhancedRetrievalService 上的旧配置字段调试 rerank 行为。
        if name.startswith("llm_rerank_") or name in {"_llm_reranker_error", "_llm_reranker_path", "_llm_reranker_device"}:
            return getattr(self.config_owner, name)
        raise AttributeError(name)

    def build_rerank_query(self, user_query: str, query_profile: QueryProfile) -> str:
        if self.generation_service is not None and hasattr(self.generation_service, "build_rerank_query"):
            try:
                rerank_query = self.generation_service.build_rerank_query(
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
        actual_rerank_query = self.fallback_rerank_query(user_query, query_profile=query_profile)
        logger.debug("original_question=%s", user_query)
        logger.debug("actual_rerank_query=%s", actual_rerank_query)
        return actual_rerank_query

    def build_rerank_document_text(self, chunk: Dict[str, object]) -> str:
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        if chunk_type in {"figure", "table"}:
            match_type = str(chunk.get("asset_section_match_type", "") or "").strip()
            allow_section_anchor = bool(chunk.get("asset_section_match_allow_embedding")) if match_type else True
            # 新 asset 会显式标记弱章节锚点是否可入文本；旧数据缺字段时继续沿用历史 rerank 行为。
            table_evidence = chunk.get("table_evidence") if isinstance(chunk.get("table_evidence"), dict) else {}
            # rerank 只消费 v2 formatter 的文本结果，避免重新引入旧表格摘要字段的不透明语义。
            table_evidence_text = render_table_evidence_rerank_text(
                table_evidence,
                max_chars=self.config_owner.llm_rerank_max_doc_chars,
            ) if table_evidence else ""
            parts = [
                str(chunk.get("asset_summary", "") or "").strip(),
                table_evidence_text,
                str(chunk.get("asset_preview_text", "") or "").strip(),
                str(chunk.get("section_title", "") or "").strip() if allow_section_anchor else "",
                str(chunk.get("section_path", "") or "").strip() if allow_section_anchor else "",
                f"page {chunk.get('page_number') or chunk.get('page_range') or ''}".strip(),
            ]
            content = "\n".join(part for part in parts if part).strip()
            if content:
                return self._build_multi_index_rerank_document(chunk, content)
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
        return self._build_multi_index_rerank_document(chunk, content)

    def _build_multi_index_rerank_document(self, chunk: Dict[str, object], evidence_text: str) -> str:
        """把 index 命中摘要拼到 rerank 输入头部，同时保留 chunk 原始证据作为主体。"""
        max_chars = max(1, int(self.config_owner.llm_rerank_max_doc_chars or 4096))
        prefix_parts: List[str] = []
        section_path = str(chunk.get("section_path", "") or "").strip()
        section_title = str(chunk.get("section_title", "") or "").strip()
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        match_type = str(chunk.get("asset_section_match_type", "") or "").strip()
        allow_section_anchor = bool(chunk.get("asset_section_match_allow_embedding")) if match_type else True
        # 图表弱章节锚点已由上游判定不可入 embedding/rerank 时，摘要头也不能重新暴露该章节文本。
        if (section_path or section_title) and not (chunk_type in {"figure", "table"} and not allow_section_anchor):
            prefix_parts.append(f"Section: {section_path or section_title}")
        matched_index_summary = self._matched_indexes_rerank_summary(chunk)
        if matched_index_summary:
            # matched index 是召回解释，不是最终证据；rerank 仍必须看到完整 chunk evidence。
            prefix_parts.append(f"Matched retrieval indexes:\n{matched_index_summary}")
        if not prefix_parts:
            return self.limit_rerank_text(evidence_text, max_chars)

        prefix = "\n".join(prefix_parts).strip()
        evidence_header = "Chunk evidence:"
        evidence_budget = max(1, max_chars - len(prefix) - len(evidence_header) - 2)
        evidence = self.limit_rerank_text(evidence_text, evidence_budget)
        return self.limit_rerank_text(f"{prefix}\n{evidence_header}\n{evidence}", max_chars)

    def _matched_indexes_rerank_summary(self, chunk: Dict[str, object], *, limit: int = 3) -> str:
        rows = self._matched_index_rows_for_rerank(chunk)
        if not rows:
            return ""
        lines: List[str] = []
        for row in rows[: max(1, limit)]:
            index_type = str(row.get("matched_index_type") or row.get("index_type") or "body")
            index_id = str(row.get("matched_index_id") or row.get("index_id") or "")
            score = row.get("matched_index_score")
            score_text = ""
            try:
                if score is not None:
                    score_text = f" score={float(score):.4f}"
            except (TypeError, ValueError):
                score_text = ""
            text_preview = self.limit_rerank_text(
                str(row.get("matched_index_text") or row.get("index_text") or ""),
                240,
            )
            lines.append(f"- type={index_type} id={index_id}{score_text}: {text_preview}".strip())
        return "\n".join(line for line in lines if line)

    def _matched_index_rows_for_rerank(self, chunk: Dict[str, object]) -> List[Dict[str, Any]]:
        raw_rows = [row for row in (chunk.get("matched_indexes") or []) if isinstance(row, dict)]
        if not raw_rows and (chunk.get("matched_index_id") or chunk.get("retrieval_index_id") or chunk.get("index_id")):
            # 旧路径只有主 index 字段时合成一行，保证 rerank/debug 仍能展示命中入口。
            raw_rows = [
                {
                    "matched_index_id": chunk.get("matched_index_id") or chunk.get("retrieval_index_id") or chunk.get("index_id"),
                    "matched_index_type": chunk.get("matched_index_type") or chunk.get("retrieval_index_type") or chunk.get("index_type"),
                    "matched_index_text": chunk.get("matched_index_text") or chunk.get("retrieval_index_text") or chunk.get("index_text"),
                    "matched_index_score": chunk.get("matched_index_score") or chunk.get("route_score") or chunk.get("score"),
                }
            ]
        deduped: List[Dict[str, Any]] = []
        seen_ids = set()
        for row in raw_rows:
            index_id = str(row.get("matched_index_id") or row.get("index_id") or "")
            if index_id and index_id in seen_ids:
                continue
            if index_id:
                seen_ids.add(index_id)
            deduped.append(dict(row))
        deduped.sort(
            key=lambda row: self._score_value(
                row.get("weighted_index_score"),
                row.get("matched_index_score"),
                row.get("bm25_fusion_score"),
                row.get("best_raw_bm25_score"),
            ),
            reverse=True,
        )
        return deduped

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
            remote_result = self.rerank_with_dashscope(
                rerank_query,
                chunks,
                candidate_chunks,
                rerank_documents,
                rerank_limit,
                top_k,
                query_profile,
            )
            if remote_result is not None:
                return remote_result
            if not self.config_owner.llm_rerank_fallback_local:
                return self._fallback_result(
                    limited_chunks,
                    chunks,
                    self.config_owner._llm_reranker_error or "remote_rerank_unavailable",
                    rerank_query,
                    original_question,
                    query_profile,
                    rerank_limit,
                    mode="remote_failed",
                    provider="dashscope",
                )

        # reranker 的加载职责属于 RerankService；不再绕回 EnhancedRetrievalService 的历史私有 wrapper。
        reranker = self.load_llm_reranker()
        if reranker is None:
            return self._fallback_result(
                limited_chunks,
                chunks,
                self.config_owner._llm_reranker_error or "llm_reranker_unavailable",
                rerank_query,
                original_question,
                query_profile,
                rerank_limit,
            )

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
            self.log_rerank_raw_scores(
                "local",
                [{"returned_index": idx, "relevance_score": float(score)} for idx, score in enumerate(rerank_scores, start=1)],
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to rerank chunks with local cross encoder")
            return self._fallback_result(
                limited_chunks,
                chunks,
                f"inference_error: {exc}",
                rerank_query,
                original_question,
                query_profile,
                rerank_limit,
            )

        mapped_candidates, candidate_debug = self._map_local_scores(candidate_chunks, rerank_scores, rerank_documents)
        mapped_candidates.sort(key=lambda item: float(item.get("rerank_score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(mapped_candidates, start=1):
            item["rerank_rank"] = rank
            item["chunk"]["llm_rerank_rank"] = rank
        self.log_rerank_mapped_results("local", mapped_candidates)
        reranked_chunks = [item["chunk"] for item in mapped_candidates] + self._build_tail_chunks(
            chunks,
            rerank_limit,
            model_name=self.config_owner._llm_reranker_path,
        )
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

    def rerank_with_dashscope(
        self,
        user_query: str,
        chunks: List[Dict[str, Any]],
        candidate_chunks: List[Dict[str, Any]],
        rerank_documents: List[str],
        rerank_limit: int,
        top_k: int,
        query_profile: Optional[QueryProfile] = None,
    ) -> Optional[Dict[str, Any]]:
        api_key = self.config_owner.llm_rerank_api_key
        if not api_key:
            self.config_owner._llm_reranker_error = "dashscope_api_key_not_set"
            return None
        documents = [doc[: self.config_owner.llm_rerank_max_doc_chars] for doc in rerank_documents]
        request_url, payload = self.get_dashscope_rerank_request_spec(
            user_query=user_query,
            documents=documents,
            rerank_limit=rerank_limit,
        )
        try:
            # 远程 rerank 绕过 GenerationService，必须在实际 HTTP 边界单独计数，超时也计入。
            record_llm_call(model=self.config_owner.llm_rerank_model_name, task_type="rerank")
            response = requests.post(
                request_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            record_llm_usage(data.get("usage"))
        except Exception as exc:
            self.config_owner._llm_reranker_error = f"dashscope_rerank_request_failed: {exc}"
            logger.warning(
                "DashScope rerank failed: %s, will %s",
                exc,
                "fallback to local model" if self.config_owner.llm_rerank_fallback_local else "stop",
            )
            return None
        results = self.extract_dashscope_rerank_results(data)
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
        ranked_candidates, candidate_debug, mapped_debug = self._map_dashscope_scores(
            candidate_chunks,
            rerank_documents,
            score_by_index,
        )
        ranked_candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        for rank, item in enumerate(ranked_candidates, start=1):
            item["llm_rerank_rank"] = rank
        self.log_rerank_mapped_results("dashscope", mapped_debug)
        reranked_chunks = ranked_candidates + self._build_tail_chunks(
            chunks,
            rerank_limit,
            model_name=self.config_owner.llm_rerank_model_name,
        )
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

    def load_llm_reranker(self) -> Optional[Any]:
        model_path = self.resolve_reranker_model_path()
        if not model_path:
            self.config_owner._llm_reranker_error = "llm_rerank_model_not_configured"
            return None
        with self._llm_reranker_lock:
            if self._llm_reranker is not None and self.config_owner._llm_reranker_path == model_path:
                return self._llm_reranker
            if CrossEncoder is None or torch is None:
                self.config_owner._llm_reranker_error = "sentence_transformers_or_torch_unavailable"
                self._llm_reranker = None
                self.config_owner._llm_reranker_path = model_path
                self.config_owner._llm_reranker_device = None
                return None
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._llm_reranker = CrossEncoder(model_name=model_path, trust_remote_code=True, device=device)
                self.config_owner._llm_reranker_path = model_path
                self.config_owner._llm_reranker_device = device
                self.config_owner._llm_reranker_error = None
                return self._llm_reranker
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load local reranker: %s", exc)
                self._llm_reranker = None
                self.config_owner._llm_reranker_path = model_path
                self.config_owner._llm_reranker_device = None
                self.config_owner._llm_reranker_error = f"load_error: {exc}"
                return None

    def resolve_reranker_model_path(self) -> Optional[str]:
        configured_path = str(self.config_owner.llm_rerank_local_model_name_or_path or "").strip()
        if configured_path:
            resolved = get_huggingface_model_path(configured_path)
            if resolved:
                return resolved
        fallback_path = str(self.config_owner.llm_rerank_model_name_or_path or "").strip()
        if fallback_path:
            return get_huggingface_model_path(fallback_path) or fallback_path
        return None

    def is_dashscope_vl_rerank_model(self) -> bool:
        model_name = (self.config_owner.llm_rerank_model_name or "").strip().lower()
        return model_name in {"qwen3-vl-rerank", "gte-rerank-v2"}

    def get_dashscope_rerank_request_spec(
        self,
        *,
        user_query: str,
        documents: List[str],
        rerank_limit: int,
    ) -> Tuple[str, Dict[str, Any]]:
        top_n = min(len(documents), max(1, rerank_limit))
        model_name = self.config_owner.llm_rerank_model_name or "qwen3-rerank"
        if self.is_dashscope_vl_rerank_model():
            base_url = self.config_owner.llm_rerank_base_url or ""
            if not base_url or base_url.endswith("/compatible-api/v1/reranks"):
                base_url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
            payload: Dict[str, Any] = {
                "model": model_name,
                "input": {
                    "query": {"text": user_query},
                    "documents": [{"text": doc} for doc in documents],
                },
                "parameters": {
                    "return_documents": True,
                    "top_n": top_n,
                },
            }
            return base_url, payload
        base_url = self.config_owner.llm_rerank_base_url or "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        payload = {
            "model": model_name,
            "query": user_query,
            "documents": documents,
            "top_n": top_n,
        }
        if self.config_owner.llm_rerank_prompt:
            payload["instruct"] = self.config_owner.llm_rerank_prompt
        return base_url, payload

    @staticmethod
    def extract_dashscope_rerank_results(data: Any) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(data, dict):
            return None
        results = data.get("results")
        if isinstance(results, list):
            return results
        output = data.get("output")
        if isinstance(output, dict):
            output_results = output.get("results")
            if isinstance(output_results, list):
                return output_results
        return None

    def fallback_rerank_query(self, user_query: str, query_profile: QueryProfile) -> str:
        normalized_question = re.sub(r"\s+", " ", (user_query or "")).strip()
        base_query = (
            "Select the passage that most directly supports an answer to the user's question. "
            "Prefer evidence-bearing chunks with explicit facts, definitions, steps, causes, results, comparisons, or other answerable statements; "
            "down-rank passages that are only loosely topic-related or background. "
        )
        intent_profile = query_profile.intent_profile
        if intent_profile is not None:
            main_intent = intent_profile.main_intent
            if main_intent in OVERVIEW_INTENTS:
                base_query += "Prioritize abstract, introduction, and conclusion passages that state the paper's main contribution or findings. "
            elif main_intent in METHOD_INTENTS:
                base_query += "Prioritize method, architecture, training, inference, and implementation details. "
            elif main_intent in EXPERIMENT_INTENTS:
                base_query += "Prioritize experiment, evaluation, results, metric, baseline, and ablation evidence. "
            else:
                base_query += {
                    "comparison": "Prioritize direct baseline comparisons and ablation evidence. ",
                    "dataset": "Prioritize dataset, corpus, benchmark, split, and data description passages. ",
                    "limitation": "Prioritize limitations, failure cases, discussion, and future work. ",
                    "figure_table": "Prioritize figure captions, table captions, appendix references, and visual explanations. ",
                }.get(main_intent, "")
            if intent_profile.preferred_sections:
                base_query += f"Favor sections such as: {', '.join(intent_profile.preferred_sections[:4])}. "
        return f"{base_query}Original question: {normalized_question}" if normalized_question else base_query.rstrip()

    def _map_local_scores(
        self,
        candidate_chunks: List[Dict[str, Any]],
        rerank_scores: List[float],
        rerank_documents: List[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        mapped_candidates: List[Dict[str, Any]] = []
        candidate_debug: List[Dict[str, Any]] = []
        for idx, (chunk, score, document_text) in enumerate(zip(candidate_chunks, rerank_scores, rerank_documents), start=1):
            rerank_score = float(score)
            fused_score = float(chunk.get("score", 0.0) or 0.0)
            reranked_chunk = dict(chunk)
            reranked_chunk.update(
                {
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
                }
            )
            mapped_candidates.append(
                {
                    "rerank_rank": None,
                    "returned_index": idx,
                    "chunk_id": reranked_chunk.get("chunk_id"),
                    "page_number": reranked_chunk.get("page_number"),
                    "fusion_rank": reranked_chunk.get("fusion_rank"),
                    "fusion_score": fused_score,
                    "rerank_score": rerank_score,
                    "text_preview": document_text[: ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]],
                    "chunk": reranked_chunk,
                }
            )
            candidate_debug.append(self._candidate_debug_row(idx, reranked_chunk, fused_score, rerank_score))
        return mapped_candidates, candidate_debug

    def _map_dashscope_scores(
        self,
        candidate_chunks: List[Dict[str, Any]],
        rerank_documents: List[str],
        score_by_index: Dict[int, float],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        ranked_candidates: List[Dict[str, Any]] = []
        candidate_debug: List[Dict[str, Any]] = []
        mapped_debug: List[Dict[str, Any]] = []
        for idx, (chunk, document_text) in enumerate(zip(candidate_chunks, rerank_documents), start=1):
            returned_index = idx if idx in score_by_index else (idx - 1 if (idx - 1) in score_by_index else None)
            rerank_score = float(score_by_index.get(idx - 1, score_by_index.get(idx, 0.0)))
            fused_score = float(chunk.get("score", 0.0) or 0.0)
            reranked_chunk = dict(chunk)
            reranked_chunk.update(
                {
                    "fusion_score": fused_score,
                    "fusion_rank": int(chunk.get("fusion_rank", chunk.get("route_rank", idx)) or idx),
                    "llm_rerank_score": rerank_score,
                    "llm_rerank_model": self.config_owner.llm_rerank_model_name,
                    "llm_rerank_prompt": self.config_owner.llm_rerank_prompt,
                    "llm_rerank_input_rank": idx,
                    "llm_rerank_returned_index": returned_index,
                    "llm_rerank_document_preview": document_text[: ENHANCED_RETRIEVAL_CONFIG["rerank_document_preview_limit"]],
                    "llm_rerank_used_compressed_text": bool(reranked_chunk.get("rerank_text")),
                    "score": rerank_score,
                }
            )
            ranked_candidates.append(reranked_chunk)
            candidate_debug.append(self._candidate_debug_row(idx, reranked_chunk, fused_score, rerank_score))
            mapped_debug.append(
                {
                    "rerank_rank": idx,
                    "returned_index": returned_index,
                    "chunk_id": reranked_chunk.get("chunk_id"),
                    "page_number": reranked_chunk.get("page_number"),
                    "fusion_rank": reranked_chunk.get("fusion_rank"),
                    "fusion_score": fused_score,
                    "rerank_score": rerank_score,
                    "text_preview": reranked_chunk.get("llm_rerank_document_preview", "")[: ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]],
                }
            )
        return ranked_candidates, candidate_debug, mapped_debug

    def _candidate_debug_row(self, idx: int, chunk: Dict[str, Any], fused_score: float, rerank_score: float) -> Dict[str, Any]:
        return {
            "input_rank": idx,
            "chunk_id": chunk.get("chunk_id"),
            "page_number": chunk.get("page_number"),
            "fusion_score": fused_score,
            "rerank_score": rerank_score,
            "rerank_text_preview": self.trace_builder.short_text_preview(
                chunk.get("rerank_text", ""),
                ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"],
            ),
            "rerank_document_preview": chunk.get("llm_rerank_document_preview", "")[: ENHANCED_RETRIEVAL_CONFIG["preview_text_limit"]],
            "section_tags": chunk.get("section_tags", []),
            "source_query": chunk.get("source_query", ""),
            "matched_index_types": chunk.get("matched_index_types", []),
            "matched_indexes": self._matched_index_rows_for_rerank(chunk)[:3],
        }

    def _build_tail_chunks(self, chunks: List[Dict[str, Any]], rerank_limit: int, *, model_name: Optional[str]) -> List[Dict[str, Any]]:
        tail_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks[rerank_limit:], start=rerank_limit + 1):
            tail_chunk = dict(chunk)
            tail_chunk.update(
                {
                    "fusion_score": float(tail_chunk.get("score", 0.0) or 0.0),
                    "fusion_rank": int(tail_chunk.get("fusion_rank", tail_chunk.get("route_rank", idx)) or idx),
                    "llm_rerank_score": None,
                    "llm_rerank_model": model_name,
                    "llm_rerank_prompt": self.config_owner.llm_rerank_prompt,
                    "llm_rerank_input_rank": idx,
                    "llm_rerank_rank": None,
                }
            )
            tail_chunks.append(tail_chunk)
        return tail_chunks

    def _fallback_result(
        self,
        chunks: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        reason: str,
        rerank_query: str,
        original_question: str,
        query_profile: Optional[QueryProfile],
        rerank_limit: int,
        *,
        mode: str = "passthrough",
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        debug = {
            "enabled": True,
            "applied": False,
            "mode": mode,
            "reason": reason,
            "model_path": self.config_owner._llm_reranker_path,
            "device": self.config_owner._llm_reranker_device,
            "input_chunks": len(all_chunks),
            "output_chunks": len(chunks),
            "candidate_limit": rerank_limit,
            "batch_size": self.config_owner.llm_rerank_batch_size,
            "query": rerank_query,
            "original_question": original_question,
            "query_profile": self.query_profile_debugger(query_profile) if query_profile else None,
        }
        if provider:
            debug["provider"] = provider
        return {"chunks": chunks, "reranked_chunks": chunks, "debug": debug}

    def _success_debug(
        self,
        *,
        mode: str,
        model_path: str,
        device: str,
        chunks: List[Dict[str, Any]],
        final_chunks: List[Dict[str, Any]],
        rerank_limit: int,
        rerank_query: str,
        original_question: str,
        query_profile: Optional[QueryProfile],
        candidate_debug: List[Dict[str, Any]],
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        debug = {
            "enabled": True,
            "applied": True,
            "mode": mode,
            "reason": "ok",
            "model_path": model_path,
            "device": device,
            "batch_size": self.config_owner.llm_rerank_batch_size,
            "candidate_limit": rerank_limit,
            "input_chunks": len(chunks),
            "output_chunks": len(final_chunks),
            "query": rerank_query,
            "original_question": original_question,
            "query_profile": self.query_profile_debugger(query_profile) if query_profile else None,
            "candidate_scores": candidate_debug[: min(ENHANCED_RETRIEVAL_CONFIG["candidate_debug_limit"], len(candidate_debug))],
        }
        if provider:
            debug["provider"] = provider
        return debug

    @staticmethod
    def limit_rerank_text(text: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        return normalized if len(normalized) <= max_chars else normalized[:max_chars].rstrip()

    @staticmethod
    def _score_value(*values: Any) -> float:
        for value in values:
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def log_rerank_inputs(
        self,
        original_question: str,
        rerank_query: str,
        candidate_chunks: List[Dict[str, Any]],
        rerank_documents: List[str],
    ) -> None:
        logger.debug("original_question=%s", original_question)
        logger.debug("actual_rerank_query=%s", rerank_query)
        logger.debug("document_count=%d", len(rerank_documents))
        for idx, (chunk, document_text) in enumerate(zip(candidate_chunks, rerank_documents), start=1):
            logger.debug(
                "rerank_input[%d] input_index=%s chunk_id=%s chunk_type=%s asset_kind=%s page=%s fusion_rank=%s fusion_score=%s clean_document_preview=%s",
                idx,
                idx,
                chunk.get("chunk_id"),
                chunk.get("chunk_type"),
                chunk.get("asset_kind"),
                chunk.get("page_number") or chunk.get("page_range"),
                chunk.get("fusion_rank"),
                chunk.get("fusion_score", chunk.get("score")),
                self.trace_builder.short_text_preview(document_text, ENHANCED_RETRIEVAL_CONFIG["short_text_preview_limit"]),
            )

    def log_rerank_raw_scores(self, provider: str, raw_results: List[Any]) -> None:
        logger.debug("rerank_raw_results provider=%s count=%d", provider, len(raw_results))

    def log_rerank_mapped_results(self, provider: str, mapped_results: List[Dict[str, Any]]) -> None:
        logger.debug("rerank_mapped_results provider=%s count=%d", provider, len(mapped_results))
