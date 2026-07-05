from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from core.errors import AppError, ErrorCode
from services.context_lifecycle import ContextLifecycleService
from services.memory import MemoryService
from services.storage.sqlite.shared import PaperQATurnPersistenceError
from services.storage.sqlite.stores import (
    AgentRuntimeCheckpointStore,
    PaperChatSessionStore,
    PaperQATurnStore,
    ResearchProfileStore,
)
from utils.config import get_default_user_id

logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 5
MAX_CONTEXT_ANSWER_CHARS = 280
MAX_CONTEXT_SOURCE_CHARS = 180
MAX_CONTEXT_SOURCES_PER_TURN = 3
MAX_REFERENCED_SOURCE_IDS = 8
MAX_SESSION_SUMMARY_CHARS = 1600
MAX_SUMMARY_ITEMS = 8


class PaperQASessionService:
    """集中管理论文 QA 的会话、短期记忆上下文与 turn 持久化。"""

    def __init__(
        self,
        *,
        paper_chat_session_store: PaperChatSessionStore,
        paper_qa_turn_store: PaperQATurnStore,
        research_profile_store: ResearchProfileStore,
        agent_runtime_checkpoint_store: AgentRuntimeCheckpointStore,
        memory_service: MemoryService,
        memory_runtime_config: Dict[str, Any],
    ) -> None:
        self.paper_chat_session_store = paper_chat_session_store
        self.paper_qa_turn_store = paper_qa_turn_store
        self.research_profile_store = research_profile_store
        self.memory_service = memory_service
        self.memory_runtime_config = memory_runtime_config
        self.context_lifecycle_service = ContextLifecycleService(
            agent_runtime_checkpoint_store=agent_runtime_checkpoint_store,
        )

    def memory_flag(self, key: str, default: Any = None) -> Any:
        """读取记忆相关运行时开关，统一约束 session 与短期记忆模块的行为。"""
        return self.memory_runtime_config.get(key, default)

    def build_memory_runtime_debug(self) -> Dict[str, Any]:
        """构造当前记忆开关的调试快照，便于前端或日志观察实际生效配置。"""
        return {
            "enabled": True,
            "config": {
                "enable_short_term_memory": bool(self.memory_flag("enable_short_term_memory", True)),
                "enable_paper_chat_session": bool(self.memory_flag("enable_paper_chat_session", True)),
                "enable_memory_aware_retrieval": bool(self.memory_flag("enable_memory_aware_retrieval", True)),
                "enable_user_research_profile": bool(self.memory_flag("enable_user_research_profile", False)),
                "short_term_memory_max_turns": int(self.memory_flag("short_term_memory_max_turns", MAX_CONVERSATION_TURNS)),
                "short_term_memory_max_chars": int(self.memory_flag("short_term_memory_max_chars", MAX_CONTEXT_ANSWER_CHARS)),
                "memory_source_boost_weight": float(self.memory_flag("memory_source_boost_weight", 0.12)),
                "memory_context_debug": bool(self.memory_flag("memory_context_debug", True)),
            },
        }

    @staticmethod
    def payload_get(payload: Any, key: str, default: Any = None) -> Any:
        """兼容 dict 与对象两种 payload 访问方式，统一读取字段值。"""
        if payload is None:
            return default
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)

    @staticmethod
    def resolve_user_id(value: Any = None) -> str:
        """解析并兜底用户 ID，确保问答链路始终有稳定的用户标识。"""
        return str(value or get_default_user_id()).strip() or get_default_user_id()

    @staticmethod
    def truncate_text(value: Any, max_length: int) -> str:
        """压缩文本长度并清理多余空白，避免短期记忆与日志字段过长。"""
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    def get_preferred_answer_style(self, payload: Any) -> str:
        """解析用户偏好的回答风格，优先使用本轮上下文，其次回退到长期画像。"""
        user_memory_summary = self._get_user_memory_summary(payload)
        profile = dict(user_memory_summary.get("profile") or {})
        preferred_answer_style = str(profile.get("preferred_answer_style") or "").strip()
        if preferred_answer_style:
            return preferred_answer_style
        if not bool(self.memory_flag("enable_user_research_profile", False)):
            return ""
        user_id = self.resolve_user_id(self.payload_get(payload, "user_id"))
        try:
            profile = self.research_profile_store.get_user_research_profile(user_id=user_id)
        except Exception:
            profile = {}
        return str((profile or {}).get("preferred_answer_style") or "").strip()

    @staticmethod
    def apply_answer_style_to_question(question: str, preferred_answer_style: str) -> str:
        """把用户偏好的回答风格附加到问题提示中，但保持事实必须受证据约束。"""
        style = str(preferred_answer_style or "").strip()
        normalized_question = str(question or "").strip()
        if not style:
            return normalized_question
        return f"请使用{style}风格回答，但事实必须严格基于本轮检索到的论文证据。问题：{normalized_question}"

    def resolve_chat_session(self, arxiv_id: str, payload: Any) -> Dict[str, Any]:
        """解析或创建论文问答会话，优先复用当前论文下的活动会话。"""
        if not bool(self.memory_flag("enable_paper_chat_session", True)):
            return {}
        user_id = self.resolve_user_id(self.payload_get(payload, "user_id"))
        requested_session_id = str(self.payload_get(payload, "session_id", "") or "").strip()
        try:
            if requested_session_id:
                # 显式 session_id 必须属于当前用户和论文，避免跨论文串用历史上下文。
                existing_session = self.paper_chat_session_store.get_paper_chat_session(requested_session_id, user_id=user_id)
                if existing_session and existing_session.get("arxiv_id") == arxiv_id:
                    return existing_session

            recent_sessions = self.paper_chat_session_store.list_paper_chat_sessions(arxiv_id=arxiv_id, user_id=user_id, limit=5)
            for recent_session in recent_sessions:
                recent_status = str(recent_session.get("status") or "active").strip().lower()
                if recent_status == "active":
                    return recent_session
            if recent_sessions:
                return recent_sessions[0]

            # 没有可复用会话时，按当前问题摘要创建一个新的论文对话会话。
            session_title = self.truncate_text(str(self.payload_get(payload, "question", "") or "").strip(), 80)
            created_session = self.paper_chat_session_store.create_paper_chat_session(
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

    def load_conversation_state(self, *, arxiv_id: str, payload: Any, chat_session: Dict[str, Any]) -> Dict[str, Any]:
        """加载并合并 DB 与请求体里的短期记忆，失败时显式降级为单轮 QA。"""
        user_id = self.resolve_user_id(self.payload_get(payload, "user_id"))
        requested_session_id = str(self.payload_get(payload, "session_id", "") or "").strip() or None
        max_turns = max(1, int(self.memory_flag("short_term_memory_max_turns", MAX_CONVERSATION_TURNS)))
        conversation_context: List[Dict[str, Any]] = []
        raw_context = self.payload_get(payload, "conversation_context", None)
        short_term_debug = {
            "enabled": bool(self.memory_flag("enable_short_term_memory", True)),
            "applied": False,
            "reason": "",
            "fallback_reason": None,
            "merge_policy": "db_authoritative_payload_compat_supplement",
            "backend_loaded_fields": [],
            "payload_received_fields": ["conversation_context"] if isinstance(raw_context, list) else [],
            "payload_accepted_fields": [],
            "payload_ignored_fields": {},
            "provided_turn_count": 0,
            "db_turn_count": 0,
            "payload_turn_count": 0,
            "merged_turn_count": 0,
            "db_message_read_count": 0,
            "db_message_read_limit": 0,
            "total_message_count": 0,
            "selected_session_id": str(chat_session.get("session_id") or "").strip() or None,
            "source": "none",
            "used_turn_count": 0,
        }
        if short_term_debug["enabled"]:
            try:
                db_context_payload = self.memory_service.load_paper_conversation_context(
                    user_id=user_id,
                    arxiv_id=arxiv_id,
                    session_id=str(chat_session.get("session_id") or requested_session_id or "").strip() or None,
                    limit=max_turns,
                )
                db_turns = list(db_context_payload.get("turns") or [])
                payload_turns = list(raw_context or []) if isinstance(raw_context, list) else []
                merged_raw_context = self.memory_service.merge_conversation_context(
                    db_turns,
                    payload_turns,
                    limit=max_turns,
                )
                # 论文 QA 的短期记忆以数据库会话为主；payload 只补齐旧客户端尚未持久化的本轮可见上下文。
                if db_turns:
                    short_term_debug["backend_loaded_fields"] = ["conversation_context"]
                if payload_turns:
                    short_term_debug["payload_accepted_fields"] = ["conversation_context"]
                short_term_debug["selected_session_id"] = (
                    str(db_context_payload.get("selected_session_id") or chat_session.get("session_id") or "").strip() or None
                )
                short_term_debug["db_turn_count"] = len(db_turns)
                short_term_debug["payload_turn_count"] = len(self.normalize_conversation_context(payload_turns))
                short_term_debug["provided_turn_count"] = short_term_debug["payload_turn_count"]
                short_term_debug["db_message_read_count"] = int(db_context_payload.get("db_message_read_count") or db_context_payload.get("message_count") or 0)
                short_term_debug["db_message_read_limit"] = int(db_context_payload.get("db_message_read_limit") or 0)
                short_term_debug["total_message_count"] = int(db_context_payload.get("total_message_count") or 0)
                short_term_debug["filtered_incomplete_turn_count"] = int(db_context_payload.get("filtered_incomplete_turn_count") or 0)
                short_term_debug["invalid_turn_count"] = int(db_context_payload.get("invalid_turn_count") or 0)
                short_term_debug["filtered_turn_count"] = int(db_context_payload.get("filtered_turn_count") or 0)
                conversation_context = self.normalize_conversation_context(merged_raw_context)
                if short_term_debug["db_turn_count"] and short_term_debug["payload_turn_count"]:
                    short_term_debug["source"] = "db_plus_payload"
                elif short_term_debug["db_turn_count"]:
                    short_term_debug["source"] = "db_only"
                elif short_term_debug["payload_turn_count"]:
                    short_term_debug["source"] = "payload_only"
                short_term_debug["merged_turn_count"] = len(conversation_context)
                short_term_debug["used_turn_count"] = len(conversation_context)
            except Exception as exc:
                # 短期记忆只是增强信号，加载失败时保留本轮 QA 能力并把原因放进 debug。
                logger.warning("Conversation context normalization failed, fallback to single-turn QA: %s", exc)
                conversation_context = []
                short_term_debug["fallback_reason"] = str(exc)
        else:
            short_term_debug["reason"] = "disabled by runtime config"

        summary_state = self.build_session_summary_state(chat_session, recent_turns=conversation_context)
        combined_context = self.combine_summary_and_recent_turns(summary_state, conversation_context)
        short_term_debug["summary"] = summary_state["debug"]
        short_term_debug["used_turn_count"] = len(combined_context)

        return {
            "conversation_context": combined_context,
            "recent_conversation_context": conversation_context,
            "session_summary": summary_state["summary"],
            "short_term_debug": short_term_debug,
            "user_id": user_id,
        }

    def update_short_term_debug_with_contextualization(
        self,
        short_term_debug: Dict[str, Any],
        question_contextualization: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把追问改写结果回填到短期记忆 debug，保持会话模块 debug 自洽。"""
        debug = dict(short_term_debug or {})
        debug["applied"] = bool(question_contextualization.get("used_short_term_memory", False))
        debug["reason"] = str(question_contextualization.get("memory_reason", "") or debug.get("reason", ""))
        if not debug.get("fallback_reason") and question_contextualization.get("error"):
            debug["fallback_reason"] = str(question_contextualization.get("error"))
        return debug

    def build_session_summary_state(self, chat_session: Dict[str, Any], *, recent_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """读取会话摘要并生成运行时 debug，不读取完整历史。

        summary 是较早历史的压缩视图，只帮助问题改写和检索线索扩展；
        它不能替代当前轮 RAG 检索出来的证据。
        """
        summary = chat_session.get("summary") if isinstance(chat_session.get("summary"), dict) else None
        summary_turn_count = int(chat_session.get("summary_turn_count") or 0)
        recent_turn_ids = {str(turn.get("turn_id") or "").strip() for turn in recent_turns if str(turn.get("turn_id") or "").strip()}
        summary_last_turn_id = str(chat_session.get("summary_last_turn_id") or "").strip()
        overlaps_recent = bool(summary_last_turn_id and summary_last_turn_id in recent_turn_ids)
        applied = bool(summary and summary_turn_count > 0)
        return {
            "summary": summary if applied else None,
            "debug": {
                "enabled": True,
                "loaded": applied,
                "applied": applied,
                "summary_updated_at": chat_session.get("summary_updated_at"),
                "summary_turn_count": summary_turn_count,
                "summary_last_turn_id": summary_last_turn_id or None,
                "recent_turn_count": len(recent_turns),
                "overlaps_recent_turns": overlaps_recent,
                "fallback_reason": None if applied else "summary_not_available",
            },
        }

    def combine_summary_and_recent_turns(
        self,
        summary_state: Dict[str, Any],
        recent_turns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """把 session summary 放在最近 turn 前面，形成运行时短期上下文。"""
        summary = summary_state.get("summary") if isinstance(summary_state, dict) else None
        if not isinstance(summary, dict):
            return list(recent_turns or [])
        summary_text = self.session_summary_to_text(summary)
        if not summary_text:
            return list(recent_turns or [])
        summary_turn = {
            "turn_id": "__session_summary__",
            "created_at": str(summary_state.get("debug", {}).get("summary_updated_at") or ""),
            "question": "Session summary",
            "answer_summary": summary_text,
            "sources": self.summary_sources(summary),
            "context_type": "session_summary",
            "is_summary": True,
        }
        return [summary_turn] + list(recent_turns or [])

    def session_summary_to_text(self, summary: Dict[str, Any]) -> str:
        """把结构化摘要压缩成 prompt 可读文本，并限制总长度防止无限增长。"""
        sections = []
        field_labels = [
            ("topic", "会话主题"),
            ("confirmed_facts", "已确认事实"),
            ("user_preferences", "用户偏好"),
            ("task_progress", "当前进展"),
            ("source_clues", "来源线索"),
        ]
        for key, label in field_labels:
            value = summary.get(key)
            if isinstance(value, list):
                text = "; ".join(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, dict):
                text = "; ".join(f"{k}: {v}" for k, v in value.items() if str(v).strip())
            else:
                text = str(value or "").strip()
            if text:
                sections.append(f"{label}: {text}")
        return self.truncate_text("\n".join(sections), MAX_SESSION_SUMMARY_CHARS)

    def summary_sources(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """摘要中的来源线索只作为检索提示，不冒充最终回答证据。"""
        sources = []
        for clue in list(summary.get("source_clues") or [])[:MAX_REFERENCED_SOURCE_IDS]:
            if not isinstance(clue, dict):
                continue
            sources.append(
                {
                    "source_id": str(clue.get("source_id") or "").strip(),
                    "section_path": str(clue.get("section_path") or "").strip(),
                    "page_number": str(clue.get("page_number") or "").strip(),
                    "chunk_type": str(clue.get("chunk_type") or "summary_clue").strip(),
                    "content": self.truncate_text(clue.get("content") or clue.get("summary") or "", MAX_CONTEXT_SOURCE_CHARS),
                    "from_session_summary": True,
                }
            )
        return [item for item in sources if item.get("source_id") or item.get("section_path") or item.get("content")]

    def build_session_debug(self, chat_session: Dict[str, Any]) -> Dict[str, Any]:
        """构造会话解析 debug，用于区分有状态会话与无状态兜底。"""
        return {
            "enabled": bool(self.memory_flag("enable_paper_chat_session", True)),
            "applied": bool(chat_session.get("session_id")),
            "reason": "session resolved" if chat_session.get("session_id") else "stateless fallback",
            "fallback_reason": None if chat_session.get("session_id") else "session unavailable or disabled",
            "session_id": chat_session.get("session_id"),
        }

    def build_memory_context(
        self,
        conversation_context: List[Dict[str, Any]],
        question_contextualization: Dict[str, Any],
        retrieval_question: str,
    ) -> Dict[str, Any]:
        """把历史 turn 中的来源线索整理成 memory-aware retrieval 可消费的候选。"""
        if not bool(self.memory_flag("enable_memory_aware_retrieval", True)):
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
                        "content_preview": self.truncate_text(source_text, 180),
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

    def build_memory_modules_debug(
        self,
        *,
        short_term_debug: Dict[str, Any],
        session_debug: Dict[str, Any],
        memory_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """按模块聚合 memory debug，避免编排层手工拼接每个子模块字段。"""
        return {
            "short_term_memory": short_term_debug,
            "session": session_debug,
            "memory_retrieval": {
                "enabled": bool(self.memory_flag("enable_memory_aware_retrieval", True)),
                "applied": bool(memory_context.get("enabled")),
                "reason": str(memory_context.get("reason", "") or ""),
                "fallback_reason": memory_context.get("fallback_reason"),
            },
            "user_profile": {
                "enabled": bool(self.memory_flag("enable_user_research_profile", False)),
                "applied": False,
                "reason": "answer style may use profile" if self.memory_flag("enable_user_research_profile", False) else "disabled",
                "fallback_reason": None,
            },
        }

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
        """把一次完整问答轮次作为原子业务单元写入会话消息表。"""
        session_id = str(chat_session.get("session_id", "") or "").strip()
        user_id = self.resolve_user_id(chat_session.get("user_id"))
        if not bool(self.memory_flag("enable_paper_chat_session", True)) or not session_id:
            return {
                "turn_id": "",
                "user_message": None,
                "assistant_message": None,
                "chat_session": chat_session,
            }
        try:
            # 原始答案和 sources 长期保留；debug 快照只保存压缩视图，避免开发 trace/候选列表写入消息表后无限增长。
            debug_snapshot = self.context_lifecycle_service.prepare_debug_snapshot(retrieval_debug)
            # 数据库层一次性写入完整 turn，避免用户消息成功、助手消息失败后污染短期记忆。
            persisted_turn = self.paper_qa_turn_store.append_paper_qa_turn(
                session_id=session_id,
                user_id=user_id,
                question=question,
                answer=answer,
                sources=source_payload,
                retrieval_debug_snapshot=debug_snapshot,
                contextualized_question=contextualized_question,
                question_contextualization=question_contextualization,
            )
            if not persisted_turn.get("user_message") or not persisted_turn.get("assistant_message"):
                raise AppError(
                    ErrorCode.DATABASE_WRITE_FAILED,
                    detail="paper qa turn append returned empty result",
                    context={"session_id": session_id, "user_id": user_id, "stage": "persist_completed_turn"},
                )
            summary_update = self.update_session_summary_after_turn(
                chat_session=persisted_turn.get("refreshed_session") or persisted_turn.get("chat_session") or chat_session,
                turn_id=str(persisted_turn.get("turn_id") or ""),
                question=question,
                answer=answer,
                source_payload=source_payload,
            )
            return {
                "turn_id": persisted_turn.get("turn_id", ""),
                "user_message": persisted_turn.get("user_message"),
                "assistant_message": persisted_turn.get("assistant_message"),
                "chat_session": persisted_turn.get("refreshed_session") or persisted_turn.get("chat_session") or chat_session,
                "session_summary_update": summary_update,
            }
        except AppError:
            raise
        except PaperQATurnPersistenceError as exc:
            # 答案已经生成但持久化失败时，本轮不会出现在历史里，必须用明确错误码反馈给前端。
            logger.exception(
                "QA turn persistence failed: code=%s session_id=%s user_id=%s stage=%s",
                ErrorCode.DATABASE_WRITE_FAILED,
                session_id,
                user_id,
                "persist_completed_turn",
            )
            raise AppError(
                ErrorCode.DATABASE_WRITE_FAILED,
                detail=exc,
                context={"session_id": session_id, "user_id": user_id, "stage": "persist_completed_turn"},
            ) from exc
        except Exception as exc:
            logger.exception(
                "Unexpected QA turn persistence failure: code=%s session_id=%s user_id=%s stage=%s",
                ErrorCode.DATABASE_WRITE_FAILED,
                session_id,
                user_id,
                "persist_completed_turn",
            )
            raise AppError(
                ErrorCode.DATABASE_WRITE_FAILED,
                detail=exc,
                context={"session_id": session_id, "user_id": user_id, "stage": "persist_completed_turn"},
            ) from exc

    def update_session_summary_after_turn(
        self,
        *,
        chat_session: Dict[str, Any],
        turn_id: str,
        question: str,
        answer: str,
        source_payload: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """用旧 summary + 新 turn 增量更新会话摘要，失败时不影响 QA 主链路。"""
        session_id = str(chat_session.get("session_id") or "").strip()
        user_id = self.resolve_user_id(chat_session.get("user_id"))
        if not session_id:
            return {"updated": False, "fallback_reason": "session_unavailable"}
        try:
            previous_summary = chat_session.get("summary") if isinstance(chat_session.get("summary"), dict) else {}
            previous_count = int(chat_session.get("summary_turn_count") or 0)
            merged_summary = self.build_incremental_session_summary(
                previous_summary,
                chat_session=chat_session,
                turn_id=turn_id,
                question=question,
                answer=answer,
                source_payload=source_payload,
            )
            updated = self.paper_chat_session_store.update_paper_chat_session_summary(
                session_id,
                user_id=user_id,
                summary=merged_summary,
                summary_turn_count=previous_count + 1,
                summary_last_turn_id=turn_id,
            )
            return {
                "updated": bool(updated),
                "summary_turn_count": previous_count + 1 if updated else previous_count,
                "summary_last_turn_id": turn_id if updated else chat_session.get("summary_last_turn_id"),
                "fallback_reason": None if updated else "database_update_failed",
            }
        except Exception as exc:
            logger.warning("Session summary update failed, keep QA turn result: session_id=%s error=%s", session_id, exc)
            return {"updated": False, "fallback_reason": str(exc)}

    def build_incremental_session_summary(
        self,
        previous_summary: Dict[str, Any],
        *,
        chat_session: Dict[str, Any],
        turn_id: str,
        question: str,
        answer: str,
        source_payload: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """确定性合并旧摘要与新增 turn，避免摘要更新读取完整历史或额外阻塞 LLM。"""
        summary = dict(previous_summary or {})
        title = str(chat_session.get("title") or "").strip()
        arxiv_id = str(chat_session.get("arxiv_id") or "").strip()
        topic = str(summary.get("topic") or "").strip()
        if not topic:
            topic = f"围绕论文 {arxiv_id} 的问答" if arxiv_id else "论文问答会话"
        if title and title not in topic:
            topic = self.truncate_text(f"{topic}；当前主题：{title}", 240)
        summary["topic"] = topic

        fact = self.truncate_text(f"Q: {question} | A: {answer}", 360)
        summary["confirmed_facts"] = self._append_unique_summary_item(summary.get("confirmed_facts"), fact)
        preferences = self.extract_user_preferences(question)
        summary["user_preferences"] = self._merge_unique_summary_items(summary.get("user_preferences"), preferences)
        progress = self.truncate_text(f"已完成 turn {turn_id or 'unknown'}：{question}", 260)
        summary["task_progress"] = self._append_unique_summary_item(summary.get("task_progress"), progress)
        summary["source_clues"] = self.merge_summary_source_clues(summary.get("source_clues"), source_payload, turn_id=turn_id)
        return self.compact_session_summary(summary)

    def extract_user_preferences(self, question: str) -> List[str]:
        text = str(question or "").strip()
        lowered = text.lower()
        preferences: List[str] = []
        if any(token in text for token in ("不要", "不用", "不想看", "跳过")) or "skip" in lowered:
            preferences.append(self.truncate_text(f"用户提出排除或跳过要求：{text}", 220))
        if any(token in text for token in ("只看", "重点", "关注", "详细", "细节")) or any(token in lowered for token in ("focus", "detail", "only")):
            preferences.append(self.truncate_text(f"用户关注偏好：{text}", 220))
        if any(token in text for token in ("风格", "简洁", "先结论", "用中文")) or any(token in lowered for token in ("style", "concise")):
            preferences.append(self.truncate_text(f"回答风格偏好：{text}", 220))
        return preferences

    def merge_summary_source_clues(self, existing: Any, source_payload: List[Dict[str, Any]], *, turn_id: str) -> List[Dict[str, Any]]:
        clues: List[Dict[str, Any]] = [dict(item) for item in list(existing or []) if isinstance(item, dict)]
        seen = {str(item.get("source_id") or item.get("chunk_id") or "").strip() for item in clues}
        for source in list(source_payload or [])[:MAX_CONTEXT_SOURCES_PER_TURN]:
            if not isinstance(source, dict):
                continue
            source_id = str(source.get("source_id") or source.get("parent_chunk_id") or source.get("chunk_id") or "").strip()
            if source_id and source_id in seen:
                continue
            seen.add(source_id)
            clues.append(
                {
                    "turn_id": turn_id,
                    "source_id": source_id,
                    "section_path": str(source.get("section_path") or "").strip(),
                    "page_number": str(source.get("page_number") or "").strip(),
                    "chunk_type": str(source.get("chunk_type") or "text").strip(),
                    "content": self.truncate_text(source.get("content") or source.get("asset_summary") or "", MAX_CONTEXT_SOURCE_CHARS),
                }
            )
        return clues[-MAX_SUMMARY_ITEMS:]

    def compact_session_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        compacted = {
            "topic": self.truncate_text(summary.get("topic"), 240),
            "confirmed_facts": self._trim_summary_list(summary.get("confirmed_facts"), 360),
            "user_preferences": self._trim_summary_list(summary.get("user_preferences"), 220),
            "task_progress": self._trim_summary_list(summary.get("task_progress"), 260),
            "source_clues": list(summary.get("source_clues") or [])[-MAX_SUMMARY_ITEMS:],
        }
        # 最终再压一次整体文本规模，防止结构化字段组合后无限增长。
        if len(self.session_summary_to_text(compacted)) > MAX_SESSION_SUMMARY_CHARS:
            compacted["confirmed_facts"] = compacted["confirmed_facts"][-max(2, MAX_SUMMARY_ITEMS // 2):]
            compacted["task_progress"] = compacted["task_progress"][-max(2, MAX_SUMMARY_ITEMS // 2):]
        return compacted

    def _append_unique_summary_item(self, existing: Any, item: str) -> List[str]:
        return self._merge_unique_summary_items(existing, [item])

    def _merge_unique_summary_items(self, existing: Any, new_items: List[str]) -> List[str]:
        merged = [str(item).strip() for item in list(existing or []) if str(item).strip()]
        for item in new_items:
            normalized = str(item or "").strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged[-MAX_SUMMARY_ITEMS:]

    def _trim_summary_list(self, value: Any, max_chars: int) -> List[str]:
        return [self.truncate_text(item, max_chars) for item in list(value or []) if str(item).strip()][-MAX_SUMMARY_ITEMS:]

    def normalize_conversation_context(self, raw_context: Any) -> List[Dict[str, Any]]:
        """规范化历史对话上下文，裁剪轮次数量与文本长度以控制提示规模。"""
        if not bool(self.memory_flag("enable_short_term_memory", True)):
            return []
        if not isinstance(raw_context, list):
            return []

        max_turns = max(1, int(self.memory_flag("short_term_memory_max_turns", MAX_CONVERSATION_TURNS)))
        max_chars = max(80, int(self.memory_flag("short_term_memory_max_chars", MAX_CONTEXT_ANSWER_CHARS)))
        normalized_turns: List[Dict[str, Any]] = []
        for raw_turn in raw_context[-max_turns:]:
            if not isinstance(raw_turn, dict):
                continue

            question = self.truncate_text(raw_turn.get("question", raw_turn.get("user_question", "")), 220)
            answer_summary = self.truncate_text(
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

    def _get_user_memory_summary(self, payload: Any) -> Dict[str, Any]:
        """从请求负载中提取用户记忆摘要，兼容直接字段与嵌套 context 两种结构。"""
        direct_summary = self.payload_get(payload, "user_memory_summary", None)
        if isinstance(direct_summary, dict):
            return direct_summary
        nested_context = self.payload_get(payload, "context", None)
        if isinstance(nested_context, dict):
            nested_summary = nested_context.get("user_memory_summary")
            if isinstance(nested_summary, dict):
                return nested_summary
        return {}

    def _normalize_context_source(self, source: Any) -> Optional[Dict[str, Any]]:
        """把历史轮次里的来源信息压缩为短期记忆可消费的统一结构。"""
        if not isinstance(source, dict):
            return None
        source_id = source.get("source_id", source.get("parent_chunk_id", source.get("chunk_id", source.get("id"))))
        normalized = {
            "source_id": str(source_id).strip() if source_id is not None else "",
            "page_number": str(source.get("page_number", "") or "").strip(),
            "section_path": str(source.get("section_path", "") or "").strip(),
            "chunk_type": str(source.get("chunk_type", "text") or "text").strip(),
            "content": self.truncate_text(
                source.get("content", source.get("asset_summary", source.get("asset_preview_text", ""))),
                MAX_CONTEXT_SOURCE_CHARS,
            ),
        }
        if not any(normalized.values()):
            return None
        return normalized

    @staticmethod
    def _memory_query_keywords(question: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9][a-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", str(question or "").lower())
        seen: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        return seen[:10]
