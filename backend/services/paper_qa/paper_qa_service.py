from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

from services.arxiv.arxiv_search_service import ArxivSearchService
from services.document.chunking_service import ChunkingService
from services.storage.database_service import DatabaseService
from services.embedding.embedding_service import EmbeddingConfig, EmbeddingService
from services.retrieval.enhanced_retrieval_service import EnhancedRetrievalService, RetrievalOptions
from services.llm.generation_service import GenerationService
from services.document.loading_service import LoadingService
from services.paper_qa.paper_qa_index_builder import PaperQAIndexBuilder
from services.storage.vector_store_service import VectorStoreService
from utils.config import get_default_user_id, get_memory_runtime_config

logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 5
MAX_CONTEXT_ANSWER_CHARS = 280
MAX_CONTEXT_SOURCE_CHARS = 180
MAX_CONTEXT_SOURCES_PER_TURN = 3
MAX_REFERENCED_SOURCE_IDS = 8

FOLLOW_UP_PATTERN = re.compile(
    r"(它|他|她|这个|这个方法|这个模块|这种方法|该方法|该模块|上述|上面|前面|这里|其中|其|这些|那些|第二步|第一步|第三步|这一步|上一步|下一步|baseline|上一轮|上一条|上一个)",
    re.IGNORECASE,
)


class PaperQAService:
    def __init__(
        self,
        *,
        db_service: Optional[DatabaseService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        generation_service: Optional[GenerationService] = None,
        enhanced_retrieval_service: Optional[EnhancedRetrievalService] = None,
        arxiv_service_factory: Optional[Callable[[], Any]] = None,
        get_embedding_config: Optional[Callable[[], EmbeddingConfig]] = None,
        loading_service_factory: Optional[Callable[[], LoadingService]] = None,
        chunking_service_factory: Optional[Callable[[], ChunkingService]] = None,
        qa_index_builder: Optional[PaperQAIndexBuilder] = None,
    ):
        self.db_service = db_service or DatabaseService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.generation_service = generation_service or GenerationService()
        self.enhanced_retrieval_service = enhanced_retrieval_service or EnhancedRetrievalService(
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
        )
        self.arxiv_service_factory = arxiv_service_factory or (lambda: ArxivSearchService())
        self.get_embedding_config = get_embedding_config or self.embedding_service.get_default_embedding_config
        self.loading_service_factory = loading_service_factory or LoadingService
        self.chunking_service_factory = chunking_service_factory or ChunkingService
        self.qa_index_builder = qa_index_builder or PaperQAIndexBuilder(
            db_service=self.db_service,
            embedding_service=self.embedding_service,
            vector_store_service=self.vector_store_service,
            generation_service=self.generation_service,
            arxiv_service_factory=self.arxiv_service_factory,
            get_embedding_config=self.get_embedding_config,
            loading_service_factory=self.loading_service_factory,
            chunking_service_factory=self.chunking_service_factory,
        )
        self.memory_runtime_config = get_memory_runtime_config()

    def _memory_flag(self, key: str, default: Any = None) -> Any:
        return self.memory_runtime_config.get(key, default)

    def _build_memory_runtime_debug(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "config": {
                "enable_short_term_memory": bool(self._memory_flag("enable_short_term_memory", True)),
                "enable_paper_chat_session": bool(self._memory_flag("enable_paper_chat_session", True)),
                "enable_memory_aware_retrieval": bool(self._memory_flag("enable_memory_aware_retrieval", True)),
                "enable_user_research_profile": bool(self._memory_flag("enable_user_research_profile", False)),
                "short_term_memory_max_turns": int(self._memory_flag("short_term_memory_max_turns", MAX_CONVERSATION_TURNS)),
                "short_term_memory_max_chars": int(self._memory_flag("short_term_memory_max_chars", MAX_CONTEXT_ANSWER_CHARS)),
                "memory_source_boost_weight": float(self._memory_flag("memory_source_boost_weight", 0.12)),
                "memory_context_debug": bool(self._memory_flag("memory_context_debug", True)),
            },
        }

    @staticmethod
    def _payload_get(payload: Any, key: str, default: Any = None) -> Any:
        if payload is None:
            return default
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)

    @staticmethod
    def _resolve_user_id(value: Any = None) -> str:
        return str(value or get_default_user_id()).strip() or get_default_user_id()

    def _get_preferred_answer_style(self, payload: Any) -> str:
        if not bool(self._memory_flag("enable_user_research_profile", False)):
            return ""
        user_id = self._resolve_user_id(self._payload_get(payload, "user_id"))
        try:
            profile = self.db_service.get_user_research_profile(user_id=user_id)
        except Exception:
            profile = {}
        return str((profile or {}).get("preferred_answer_style") or "").strip()

    @staticmethod
    def _apply_answer_style_to_question(question: str, preferred_answer_style: str) -> str:
        style = str(preferred_answer_style or "").strip()
        normalized_question = str(question or "").strip()
        if not style:
            return normalized_question
        return f"请使用{style}风格回答，但事实必须严格基于本轮检索到的论文证据。问题：{normalized_question}"

    def _resolve_chat_session(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        if not bool(self._memory_flag("enable_paper_chat_session", True)):
            return {}
        user_id = self._resolve_user_id(self._payload_get(payload, "user_id"))
        requested_session_id = str(self._payload_get(payload, "session_id", "") or "").strip()
        try:
            if requested_session_id:
                existing_session = self.db_service.get_paper_chat_session(requested_session_id, user_id=user_id)
                if existing_session and existing_session.get("arxiv_id") == arxiv_id:
                    return existing_session

            session_title = self._truncate_text(str(self._payload_get(payload, "question", "") or "").strip(), 80)
            created_session = self.db_service.create_paper_chat_session(
                arxiv_id=arxiv_id,
                user_id=user_id,
                title=session_title,
            )
            if not created_session:
                raise HTTPException(status_code=500, detail="Failed to create paper chat session")
            return created_session
        except Exception as exc:
            logger.warning("Chat session resolution failed, fallback to stateless QA: %s", exc)
            return {}

    def persist_completed_turn(
        self,
        *,
        chat_session: Dict[str, Any],
        question: str,
        answer: str,
        source_payload: List[Dict[str, Any]],
        retrieval_debug: Optional[Dict[str, Any]],
        contextualized_question: str,
        question_contextualization: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        session_id = str(chat_session.get("session_id", "") or "").strip()
        user_id = self._resolve_user_id(chat_session.get("user_id"))
        if not bool(self._memory_flag("enable_paper_chat_session", True)) or not session_id:
            return {
                "turn_id": "",
                "user_message": None,
                "assistant_message": None,
                "chat_session": chat_session,
            }
        turn_id = str(uuid.uuid4())
        try:
            user_message = self.db_service.append_paper_chat_message(
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                role="user",
                content=question,
                contextualized_question=contextualized_question,
                question_contextualization=question_contextualization,
            )
            assistant_message = self.db_service.append_paper_chat_message(
                session_id=session_id,
                user_id=user_id,
                turn_id=turn_id,
                role="assistant",
                content=answer,
                sources=source_payload,
                retrieval_debug_snapshot=retrieval_debug,
                contextualized_question=contextualized_question,
                question_contextualization=question_contextualization,
            )
            refreshed_session = self.db_service.get_paper_chat_session(session_id, user_id=user_id) or chat_session
            return {
                "turn_id": turn_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "chat_session": refreshed_session,
            }
        except Exception as exc:
            logger.warning("Persisting QA turn failed, returning response without session write: %s", exc)
            return {
                "turn_id": turn_id,
                "user_message": None,
                "assistant_message": None,
                "chat_session": chat_session,
            }

    def get_qa_status(self, arxiv_id: str) -> Dict[str, Any]:
        qa_index = self.db_service.get_paper_qa_index(arxiv_id)
        if qa_index:
            return {
                "arxiv_id": arxiv_id,
                "has_index": qa_index["status"] == "indexed",
                "status": qa_index["status"],
                "collection_name": qa_index["collection_name"],
                "chunk_count": qa_index["chunk_count"],
                "embedding_model": qa_index["embedding_model"],
            }
        return {
            "arxiv_id": arxiv_id,
            "has_index": False,
            "status": "not_indexed",
        }

    def build_qa_index(self, arxiv_id: str, loading_method: str = "docling") -> Dict[str, Any]:
        return self.qa_index_builder.build_qa_index(arxiv_id, loading_method=loading_method)

    def build_generation_context(self, search_results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        text_parts: List[str] = []
        image_inputs: List[Dict[str, Any]] = []
        asset_metadata: List[Dict[str, Any]] = []

        for index, result in enumerate(search_results, start=1):
            chunk_type = str(result.get("chunk_type", "text") or "text").strip().lower()
            asset_info = {
                "chunk_type": chunk_type,
                "asset_kind": result.get("asset_kind", ""),
                "asset_path": result.get("asset_path", ""),
                "asset_abs_path": result.get("asset_abs_path", ""),
                "asset_summary": result.get("asset_summary", ""),
                "asset_preview_text": result.get("asset_preview_text", ""),
                "page_number": result.get("page_number", ""),
                "page_range": result.get("page_range", ""),
                "section_path": result.get("section_path", ""),
                "section_title": result.get("section_title", ""),
                "source": result.get("source", ""),
            }

            if chunk_type == "figure" and asset_info["asset_abs_path"]:
                image_inputs.append(
                    {
                        "image_path": asset_info["asset_abs_path"],
                        "page_number": asset_info["page_number"],
                        "asset_summary": asset_info["asset_summary"],
                        "section_path": asset_info["section_path"],
                    }
                )
                asset_metadata.append(asset_info)
                continue

            if chunk_type == "table":
                table_text = "\n".join(
                    part for part in [
                        f"[Table {index}]",
                        str(result.get("asset_summary", "") or "").strip(),
                        str(result.get("asset_preview_text", "") or "").strip(),
                    ] if part
                ).strip()
                if table_text:
                    text_parts.append(table_text)
                asset_metadata.append(asset_info)
                continue

            content = str(result.get("content", "") or "").strip()
            if content:
                text_parts.append(content)

        return "\n\n".join(text_parts), image_inputs, asset_metadata

    def build_source_payload(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "content": r.get("content", ""),
                "page_number": r.get("page_number", ""),
                "source": r.get("source", ""),
                "subchunk_label": r.get("subchunk_label", ""),
                "chunk_label": r.get("subchunk_label", ""),
                "section_path": r.get("section_path", ""),
                "parent_chunk_id": r.get("parent_chunk_id", r.get("chunk_id", 0)),
                "chunk_type": r.get("chunk_type", "text"),
                "asset_kind": r.get("asset_kind", ""),
                "asset_path": r.get("asset_path", ""),
                "asset_summary": r.get("asset_summary", ""),
                "asset_preview_text": r.get("asset_preview_text", ""),
            }
            for r in search_results
        ]

    @staticmethod
    def _truncate_text(value: Any, max_length: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        normalized = str(text or "").strip()
        if not normalized:
            return {}
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, flags=re.DOTALL)
        if fenced_match:
            normalized = fenced_match.group(1)
        else:
            object_match = re.search(r"(\{.*\})", normalized, flags=re.DOTALL)
            if object_match:
                normalized = object_match.group(1)
        return json.loads(normalized)

    def _normalize_context_source(self, source: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(source, dict):
            return None
        source_id = source.get("source_id", source.get("parent_chunk_id", source.get("chunk_id", source.get("id"))))
        normalized = {
            "source_id": str(source_id).strip() if source_id is not None else "",
            "page_number": str(source.get("page_number", "") or "").strip(),
            "section_path": str(source.get("section_path", "") or "").strip(),
            "chunk_type": str(source.get("chunk_type", "text") or "text").strip(),
            "content": self._truncate_text(
                source.get("content", source.get("asset_summary", source.get("asset_preview_text", ""))),
                MAX_CONTEXT_SOURCE_CHARS,
            ),
        }
        if not any(normalized.values()):
            return None
        return normalized

    def _normalize_conversation_context(self, raw_context: Any) -> List[Dict[str, Any]]:
        if not bool(self._memory_flag("enable_short_term_memory", True)):
            return []
        if not isinstance(raw_context, list):
            return []

        max_turns = max(1, int(self._memory_flag("short_term_memory_max_turns", MAX_CONVERSATION_TURNS)))
        max_chars = max(80, int(self._memory_flag("short_term_memory_max_chars", MAX_CONTEXT_ANSWER_CHARS)))
        normalized_turns: List[Dict[str, Any]] = []
        for raw_turn in raw_context[-max_turns:]:
            if not isinstance(raw_turn, dict):
                continue

            question = self._truncate_text(raw_turn.get("question", raw_turn.get("user_question", "")), 220)
            answer_summary = self._truncate_text(
                raw_turn.get("answer_summary", raw_turn.get("answer", raw_turn.get("assistant_summary", ""))),
                max_chars,
            )
            sources: List[Dict[str, Any]] = []
            for item in raw_turn.get("sources", [])[:MAX_CONTEXT_SOURCES_PER_TURN]:
                normalized_source = self._normalize_context_source(item)
                if normalized_source:
                    sources.append(normalized_source)

            if not question and not answer_summary and not sources:
                continue

            turn_id = raw_turn.get("turn_id", raw_turn.get("turnId", raw_turn.get("id", "")))
            normalized_turns.append(
                {
                    "turn_id": str(turn_id or "").strip(),
                    "created_at": str(raw_turn.get("created_at", raw_turn.get("createdAt", "")) or "").strip(),
                    "question": question,
                    "answer_summary": answer_summary,
                    "sources": sources,
                }
            )

        return normalized_turns

    @staticmethod
    def _looks_like_follow_up(question: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(question or "")).strip()
        if not normalized:
            return False
        if FOLLOW_UP_PATTERN.search(normalized):
            return True
        if len(normalized) <= 18:
            return True
        return normalized.endswith(("?", "？"))

    def _build_contextualization_fallback(
        self,
        question: str,
        conversation_context: List[Dict[str, Any]],
        referenced_source_ids: List[str],
    ) -> Dict[str, Any]:
        latest_turn = conversation_context[-1] if conversation_context else {}
        latest_question = str(latest_turn.get("question", "") or "").strip()
        latest_answer_summary = str(latest_turn.get("answer_summary", "") or "").strip()
        bridge_parts = []
        if latest_question:
            bridge_parts.append(f"Previous turn question: {latest_question}")
        if latest_answer_summary:
            bridge_parts.append(f"Previous turn answer summary: {latest_answer_summary}")
        if bridge_parts:
            contextualized_question = "\n".join(bridge_parts + [f"Current follow-up question: {question}"])
            memory_reason = "Used the most recent QA turn as a fallback to disambiguate a likely follow-up question."
        else:
            contextualized_question = question
            memory_reason = "No usable prior turn summary was available, so the original question was kept."
        return {
            "original_question": question,
            "contextualized_question": contextualized_question,
            "is_follow_up": bool(conversation_context),
            "referenced_turn_ids": [latest_turn.get("turn_id")] if latest_turn.get("turn_id") else [],
            "referenced_source_ids": referenced_source_ids[:MAX_REFERENCED_SOURCE_IDS],
            "memory_reason": memory_reason,
            "used_short_term_memory": bool(conversation_context),
            "status": "heuristic_fallback",
            "error": None,
        }

    def _contextualize_question(
        self,
        question: str,
        paper_context: Dict[str, Any],
        conversation_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not bool(self._memory_flag("enable_short_term_memory", True)):
            return {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "memory_reason": "Short-term memory is disabled by runtime config.",
                "used_short_term_memory": False,
                "status": "disabled",
                "error": None,
            }
        if not conversation_context:
            return {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "memory_reason": "No conversation context was provided, so retrieval used the original question.",
                "used_short_term_memory": False,
                "status": "not_provided",
                "error": None,
            }

        referenced_source_ids: List[str] = []
        referenced_turn_ids: List[str] = []
        source_lines: List[str] = []
        turn_lines: List[str] = []
        for index, turn in enumerate(conversation_context, start=1):
            turn_id = str(turn.get("turn_id", "") or "").strip()
            if turn_id:
                referenced_turn_ids.append(turn_id)
            question_text = str(turn.get("question", "") or "").strip()
            answer_summary = str(turn.get("answer_summary", "") or "").strip()
            turn_lines.append(
                f"Turn {index} (id={turn_id or f't{index}'}, created_at={turn.get('created_at', '') or 'unknown'}): "
                f"Q={question_text or 'N/A'} | A={answer_summary or 'N/A'}"
            )
            for source in turn.get("sources", []):
                source_id = str(source.get("source_id", "") or "").strip()
                if source_id and source_id not in referenced_source_ids:
                    referenced_source_ids.append(source_id)
                source_lines.append(
                    "- "
                    f"turn_id={turn_id or f't{index}'}, source_id={source_id or 'unknown'}, "
                    f"section={source.get('section_path', '') or 'N/A'}, "
                    f"page={source.get('page_number', '') or 'N/A'}, "
                    f"chunk_type={source.get('chunk_type', '') or 'text'}, "
                    f"summary={source.get('content', '') or 'N/A'}"
                )

        prompt = (
            "You are contextualizing a follow-up question for retrieval over a single academic paper.\n"
            "Rewrite the current user question into a fully self-contained retrieval question when the history indicates a follow-up.\n"
            "Rules:\n"
            "1. Return JSON only.\n"
            "2. Do not invent paper facts, method names, datasets, steps, or results not stated in the conversation context.\n"
            "3. Use short-term memory only to resolve references like it/this method/second step/above.\n"
            "4. If the reference is unclear, keep the original question and explain the uncertainty in memory_reason.\n"
            "5. referenced_turn_ids must only contain ids from the provided turns.\n"
            "6. referenced_source_ids must only contain ids from the provided sources.\n"
            "7. The JSON schema is exactly: "
            '{"contextualized_question":"...","is_follow_up":true,"referenced_turn_ids":["..."],"referenced_source_ids":["..."],"memory_reason":"..."}.\n\n'
            f"Current user question: {question}\n\n"
            f"Paper title: {paper_context.get('title', '') or 'N/A'}\n"
            f"Paper abstract: {self._truncate_text(paper_context.get('abstract', ''), 1200) or 'N/A'}\n\n"
            "Recent QA turns:\n"
            f"{chr(10).join(turn_lines) or 'N/A'}\n\n"
            "Recent source snippets used by those turns:\n"
            f"{chr(10).join(source_lines[:12]) or 'N/A'}"
        )

        heuristic_follow_up = self._looks_like_follow_up(question)
        try:
            response_text = self.generation_service.complete_with_qwen(
                prompt,
                task_type="paper_qa_question_contextualization",
            )
            parsed = self._extract_json_object(response_text)
            contextualized_question = self._truncate_text(parsed.get("contextualized_question", question), 800)
            candidate_turn_ids = [str(item).strip() for item in (parsed.get("referenced_turn_ids", []) or []) if str(item).strip()]
            candidate_source_ids = [
                str(item).strip() for item in (parsed.get("referenced_source_ids", []) or []) if str(item).strip()
            ]
            allowed_turn_ids = set(referenced_turn_ids)
            allowed_source_ids = set(referenced_source_ids)
            filtered_turn_ids = [item for item in candidate_turn_ids if item in allowed_turn_ids][:MAX_CONVERSATION_TURNS]
            filtered_source_ids = [item for item in candidate_source_ids if item in allowed_source_ids][:MAX_REFERENCED_SOURCE_IDS]
            is_follow_up = bool(parsed.get("is_follow_up", False) or heuristic_follow_up)
            if not contextualized_question:
                contextualized_question = question
            return {
                "original_question": question,
                "contextualized_question": contextualized_question,
                "is_follow_up": is_follow_up,
                "referenced_turn_ids": filtered_turn_ids,
                "referenced_source_ids": filtered_source_ids,
                "memory_reason": self._truncate_text(
                    parsed.get("memory_reason", "Short-term memory was used to resolve likely follow-up references."),
                    320,
                ),
                "used_short_term_memory": is_follow_up and contextualized_question != question,
                "status": "contextualized" if contextualized_question != question else "kept_original",
                "error": None,
            }
        except Exception as exc:
            logger.warning("Question contextualization failed, falling back: %s", exc)
            if heuristic_follow_up:
                fallback = self._build_contextualization_fallback(question, conversation_context, referenced_source_ids)
                fallback["error"] = str(exc)
                return fallback
            return {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": referenced_turn_ids[:MAX_CONVERSATION_TURNS],
                "referenced_source_ids": referenced_source_ids[:MAX_REFERENCED_SOURCE_IDS],
                "memory_reason": "Contextualization failed, so retrieval fell back to the original question.",
                "used_short_term_memory": False,
                "status": "fallback_original",
                "error": str(exc),
            }

    @staticmethod
    def _prioritize_results_by_source_ids(
        search_results: List[Dict[str, Any]], referenced_source_ids: List[str]
    ) -> List[Dict[str, Any]]:
        if not search_results or not referenced_source_ids:
            return search_results
        source_rank = {str(source_id): index for index, source_id in enumerate(referenced_source_ids)}

        def sort_key(item: Dict[str, Any]) -> Tuple[int, int]:
            candidate_ids = [
                str(item.get("parent_chunk_id", "") or "").strip(),
                str(item.get("chunk_id", "") or "").strip(),
            ]
            matched_ranks = [source_rank[candidate_id] for candidate_id in candidate_ids if candidate_id in source_rank]
            if matched_ranks:
                return (0, min(matched_ranks))
            return (1, len(source_rank))

        return sorted(search_results, key=sort_key)

    @staticmethod
    def _memory_query_keywords(question: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", str(question or "").lower())
        seen: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        return seen[:10]

    def _build_memory_context(
        self,
        conversation_context: List[Dict[str, Any]],
        question_contextualization: Dict[str, Any],
        retrieval_question: str,
    ) -> Dict[str, Any]:
        if not bool(self._memory_flag("enable_memory_aware_retrieval", True)):
            return {
                "enabled": False,
                "reason": "Memory-aware retrieval is disabled by runtime config.",
                "query_keywords": [],
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "candidates": [],
                "fallback_reason": "disabled",
            }
        referenced_turn_ids = [
            str(item).strip()
            for item in (question_contextualization.get("referenced_turn_ids", []) or [])
            if str(item).strip()
        ]
        referenced_source_ids = [
            str(item).strip()
            for item in (question_contextualization.get("referenced_source_ids", []) or [])
            if str(item).strip()
        ]
        enabled = bool(
            conversation_context
            and (
                question_contextualization.get("is_follow_up")
                or question_contextualization.get("used_short_term_memory")
                or referenced_turn_ids
                or referenced_source_ids
            )
        )
        query_keywords = self._memory_query_keywords(retrieval_question)
        candidates: List[Dict[str, Any]] = []
        total_turns = len(conversation_context)
        for index, turn in enumerate(conversation_context):
            turn_id = str(turn.get("turn_id", "") or "").strip()
            turn_distance = max(0, total_turns - index - 1)
            is_recent_turn = turn_distance == 0
            turn_is_referenced = turn_id in referenced_turn_ids if turn_id else False
            for source in turn.get("sources", []):
                source_id = str(source.get("source_id", "") or "").strip()
                source_text = str(source.get("content", "") or "").strip()
                overlap_count = sum(1 for token in query_keywords if token and token in source_text.lower())
                overlap_ratio = overlap_count / max(len(query_keywords), 1) if query_keywords else 0.0
                reference_strength = 0.2
                if is_recent_turn:
                    reference_strength += 0.35
                if turn_is_referenced:
                    reference_strength += 0.25
                if source_id and source_id in referenced_source_ids:
                    reference_strength += 0.25
                reference_strength += min(0.2, overlap_ratio)
                if not source_id and not source.get("section_path") and not source.get("page_number"):
                    continue
                candidates.append(
                    {
                        "source_turn_id": turn_id,
                        "source_id": source_id,
                        "chunk_id": source_id,
                        "parent_chunk_id": source_id,
                        "original_chunk_id": source_id,
                        "page_number": str(source.get("page_number", "") or "").strip(),
                        "section_path": str(source.get("section_path", "") or "").strip(),
                        "source": str(source.get("source", "") or "").strip(),
                        "chunk_type": str(source.get("chunk_type", "text") or "text").strip(),
                        "content_preview": self._truncate_text(source_text, 180),
                        "is_recent_turn": is_recent_turn,
                        "turn_distance": turn_distance,
                        "is_referenced_turn": turn_is_referenced,
                        "is_referenced_source": bool(source_id and source_id in referenced_source_ids),
                        "overlap_count": overlap_count,
                        "reference_strength": round(reference_strength, 4),
                        "match_type": "referenced_source" if source_id and source_id in referenced_source_ids else "recent_source",
                        "memory_reason": question_contextualization.get("memory_reason", ""),
                    }
                )

        candidates.sort(
            key=lambda item: (
                float(item.get("reference_strength", 0.0) or 0.0),
                1 if item.get("is_recent_turn") else 0,
                1 if item.get("is_referenced_source") else 0,
                -int(item.get("turn_distance", 0) or 0),
            ),
            reverse=True,
        )

        deduped_candidates: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in candidates:
            dedupe_key = "|".join(
                [
                    str(item.get("source_id", "") or ""),
                    str(item.get("section_path", "") or ""),
                    str(item.get("page_number", "") or ""),
                    str(item.get("source", "") or ""),
                ]
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped_candidates.append(item)
            if len(deduped_candidates) >= MAX_REFERENCED_SOURCE_IDS:
                break

        return {
            "enabled": enabled and bool(deduped_candidates),
            "reason": str(question_contextualization.get("memory_reason", "") or ""),
            "query_keywords": query_keywords,
            "referenced_turn_ids": referenced_turn_ids,
            "referenced_source_ids": referenced_source_ids,
            "candidates": deduped_candidates,
        }

    def build_qa_context(self, arxiv_id: str, payload: Any):
        qa_index = self.db_service.get_paper_qa_index(arxiv_id)
        if not qa_index or qa_index["status"] != "indexed":
            raise HTTPException(status_code=400, detail="Paper does not have QA index. Please create index first.")

        question = str(self._payload_get(payload, "question", "") or "").strip()
        chat_session = self._resolve_chat_session(arxiv_id, payload)
        memory_runtime = self._build_memory_runtime_debug()
        collection_name = qa_index["collection_name"]
        paper = self.db_service.get_paper(arxiv_id) or {}
        paper_context = {
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": paper.get("authors", ""),
            "categories": paper.get("categories", ""),
            "published_date": paper.get("published_date", ""),
            "url": paper.get("url", ""),
        }
        conversation_context: List[Dict[str, Any]] = []
        short_term_debug = {
            "enabled": bool(self._memory_flag("enable_short_term_memory", True)),
            "applied": False,
            "reason": "",
            "fallback_reason": None,
            "provided_turn_count": 0,
            "used_turn_count": 0,
        }
        raw_context = self._payload_get(payload, "conversation_context", None)
        if short_term_debug["enabled"]:
            try:
                short_term_debug["provided_turn_count"] = len(raw_context) if isinstance(raw_context, list) else 0
                conversation_context = self._normalize_conversation_context(raw_context)
                short_term_debug["used_turn_count"] = len(conversation_context)
            except Exception as exc:
                logger.warning("Conversation context normalization failed, fallback to single-turn QA: %s", exc)
                conversation_context = []
                short_term_debug["fallback_reason"] = str(exc)
        else:
            short_term_debug["reason"] = "disabled by runtime config"

        try:
            question_contextualization = self._contextualize_question(question, paper_context, conversation_context)
        except Exception as exc:
            logger.warning("Question contextualization raised unexpectedly, fallback to original question: %s", exc)
            question_contextualization = {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "memory_reason": "Contextualization raised unexpectedly, so the original question was used.",
                "used_short_term_memory": False,
                "status": "fallback_original",
                "error": str(exc),
            }

        retrieval_question = str(question_contextualization.get("contextualized_question", question) or question).strip() or question
        short_term_debug["applied"] = bool(question_contextualization.get("used_short_term_memory", False))
        short_term_debug["reason"] = str(question_contextualization.get("memory_reason", "") or short_term_debug["reason"])
        if not short_term_debug["fallback_reason"] and question_contextualization.get("error"):
            short_term_debug["fallback_reason"] = str(question_contextualization.get("error"))

        try:
            memory_context = self._build_memory_context(conversation_context, question_contextualization, retrieval_question)
        except Exception as exc:
            logger.warning("Memory context construction failed, skip memory-aware retrieval: %s", exc)
            memory_context = {
                "enabled": False,
                "reason": "Memory context construction failed.",
                "query_keywords": [],
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "candidates": [],
                "fallback_reason": str(exc),
            }

        session_debug = {
            "enabled": bool(self._memory_flag("enable_paper_chat_session", True)),
            "applied": bool(chat_session.get("session_id")),
            "reason": "session resolved" if chat_session.get("session_id") else "stateless fallback",
            "fallback_reason": None if chat_session.get("session_id") else "session unavailable or disabled",
            "session_id": chat_session.get("session_id"),
        }
        retrieval_result = self.enhanced_retrieval_service.enhanced_retrieve(
            collection_name=collection_name,
            user_query=retrieval_question,
            paper_context=paper_context,
            options=RetrievalOptions(
                top_k=self._payload_get(payload, "top_k", None) or 15,
                enable_query_rewrite=self._payload_get(payload, "enable_query_rewrite", None),
                enable_hyde=self._payload_get(payload, "enable_hyde", None),
                enable_keyword_search=self._payload_get(payload, "enable_keyword_search", None),
                enable_llm_rerank=self._payload_get(payload, "enable_llm_rerank", None),
                debug=self._payload_get(payload, "debug", None),
                memory_context=memory_context,
            ),
        )

        final_context_results = retrieval_result["chunks"]
        search_results = final_context_results
        if not search_results:
            raise HTTPException(status_code=400, detail="No relevant chunks found")

        text_context, image_inputs, asset_metadata = self.build_generation_context(search_results)
        retrieval_debug = retrieval_result.get("debug")
        if retrieval_debug is None:
            retrieval_debug = {}
        elif not isinstance(retrieval_debug, dict):
            retrieval_debug = {"raw_debug": retrieval_debug}
        retrieval_debug["original_question"] = question
        retrieval_debug["contextualized_question"] = retrieval_question
        retrieval_debug["question_contextualization"] = question_contextualization
        retrieval_debug["memory_context"] = memory_context
        retrieval_debug["memory_runtime"] = memory_runtime
        retrieval_debug["memory_modules"] = {
            "short_term_memory": short_term_debug,
            "session": session_debug,
            "memory_retrieval": {
                "enabled": bool(self._memory_flag("enable_memory_aware_retrieval", True)),
                "applied": bool(memory_context.get("enabled")),
                "reason": str(memory_context.get("reason", "") or ""),
                "fallback_reason": memory_context.get("fallback_reason"),
            },
            "user_profile": {
                "enabled": bool(self._memory_flag("enable_user_research_profile", False)),
                "applied": False,
                "reason": "answer style may use profile" if self._memory_flag("enable_user_research_profile", False) else "disabled",
                "fallback_reason": None,
            },
        }

        return qa_index, search_results, {
            "text_context": text_context,
            "image_inputs": image_inputs,
            "asset_metadata": asset_metadata,
            "generation_question": retrieval_question,
            "original_question": question,
            "question_contextualization": question_contextualization,
            "conversation_context": conversation_context,
            "memory_context": memory_context,
            "chat_session": chat_session,
            "memory_runtime": memory_runtime,
        }, retrieval_debug

    def answer_question(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        question = str(self._payload_get(payload, "question", "") or "").strip()
        logger.info("QA request for paper: %s, question: %s", arxiv_id, question)
        _, search_results, qa_context, retrieval_debug = self.build_qa_context(arxiv_id, payload)
        source_payload = self.build_source_payload(search_results)
        generation_question = str(qa_context.get("generation_question", question) or question).strip() or question
        question_contextualization = qa_context.get("question_contextualization", {}) or {}
        chat_session = qa_context.get("chat_session", {}) or {}
        preferred_answer_style = self._get_preferred_answer_style(payload)
        styled_generation_question = self._apply_answer_style_to_question(generation_question, preferred_answer_style)
        if isinstance(retrieval_debug, dict) and preferred_answer_style:
            retrieval_debug["preferred_answer_style"] = preferred_answer_style

        logger.info("Generating answer...")
        try:
            qwen_search_results = [
                {
                    "text": result.get("content", ""),
                    "page_number": result.get("page_number", ""),
                    "source": result.get("source", ""),
                    "subchunk_label": result.get("subchunk_label", ""),
                    "chunk_label": result.get("subchunk_label", ""),
                    "section_path": result.get("section_path", ""),
                    "chunk_type": result.get("chunk_type", "text"),
                    "asset_kind": result.get("asset_kind", ""),
                    "asset_summary": result.get("asset_summary", ""),
                    "asset_preview_text": result.get("asset_preview_text", ""),
                }
                for result in search_results
            ]
            generation_result = self.generation_service.generate(
                provider="qwen",
                # 最终答案生成保留大模型，以保证长上下文综合与表述质量。
                task_type="paper_qa_final_answer",
                query=styled_generation_question,
                search_results=qwen_search_results,
                image_inputs=qa_context["image_inputs"],
                asset_metadata=[item for item in qa_context["asset_metadata"] if item.get("chunk_type") == "figure"],
            )
            answer = generation_result["response"]
        except Exception as exc:
            logger.warning("Qwen generation failed, using fallback: %s", exc)
            answer = f'根据论文内容，关于您的问题 "{question}" 的相关信息如下：\n\n{qa_context["text_context"][:1000]}...'

        persisted_turn = self.persist_completed_turn(
            chat_session=chat_session,
            question=question,
            answer=answer,
            source_payload=source_payload,
            retrieval_debug=retrieval_debug if isinstance(retrieval_debug, dict) else None,
            contextualized_question=generation_question,
            question_contextualization=question_contextualization,
        )

        return {
            "status": "success",
            "arxiv_id": arxiv_id,
            "question": question,
            "session_id": chat_session.get("session_id"),
            "chat_session": persisted_turn.get("chat_session", chat_session),
            "turn_id": persisted_turn.get("turn_id"),
            "original_question": question,
            "contextualized_question": generation_question,
            "used_short_term_memory": bool(question_contextualization.get("used_short_term_memory", False)),
            "question_contextualization": question_contextualization,
            "answer": answer,
            "sources": source_payload,
            "image_inputs": qa_context["image_inputs"],
            "asset_metadata": qa_context["asset_metadata"],
            "retrieval_debug": retrieval_debug,
        }

    def _sanitize_trace_slug(self, text: str, max_length: int = 40) -> str:
        import re

        slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", (text or "").strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        if not slug:
            slug = "query"
        return slug[:max_length]

    def _get_latest_retrieval_trace(self, arxiv_id: str, format_name: str = "md") -> Optional[Path]:
        trace_root = Path(str(self.enhanced_retrieval_service.trace_export_dir))
        paper_dir = trace_root / self._sanitize_trace_slug(arxiv_id)
        if not paper_dir.exists() or not paper_dir.is_dir():
            return None

        suffix = ".json" if format_name == "json" else ".md"
        trace_files = sorted(
            paper_dir.glob(f"*{suffix}"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return trace_files[0] if trace_files else None
