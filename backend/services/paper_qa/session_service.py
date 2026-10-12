"""论文 QA 会话服务：会话解析、短期记忆加载、会话摘要维护与问答 turn 持久化。

设计要点：
- 短期记忆以数据库会话为准（db_authoritative），请求体里的 conversation_context 只作兼容补充；
- 会话摘要（summary）是较早历史的确定性压缩视图，只用于问题消歧和检索线索，不能当作回答证据；
- 记忆相关能力全部是「增强」：加载或更新失败时降级为单轮 QA，但 turn 持久化失败必须显式报错。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

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

# ---- 短期记忆与会话摘要的规模上限（运行时配置缺省时使用） ----
MAX_CONVERSATION_TURNS = 5  # 参与消歧的最近 turn 数
MAX_CONTEXT_ANSWER_CHARS = 280  # 每个历史 turn 的答案摘要长度
MAX_CONTEXT_SOURCE_CHARS = 180  # 每条历史来源的内容摘要长度
MAX_CONTEXT_SOURCES_PER_TURN = 3  # 每个历史 turn 最多保留的来源数
MAX_REFERENCED_SOURCE_IDS = 8  # 消歧结果中最多引用的 source_id 数
MAX_SESSION_SUMMARY_CHARS = 1600  # 会话摘要渲染成文本后的总长度上限
MAX_SUMMARY_ITEMS = 8  # 会话摘要中每个列表字段最多保留的条目数


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
        except Exception as exc:
            # 回答风格只是体验增强，读取画像失败时降级为无偏好，但要留下日志便于排查。
            logger.warning("Load research profile for answer style failed: user_id=%s error=%s", user_id, exc)
            profile = {}
        return str((profile or {}).get("preferred_answer_style") or "").strip()

    def resolve_chat_session(self, arxiv_id: str, payload: Any, *, create_if_missing: bool = True) -> Dict[str, Any]:
        """解析或创建论文问答会话，优先复用当前论文下的活动会话。

        解析顺序：显式 session_id（须属于当前用户且是同一篇论文）→ 最近 5 个会话中的 active 会话
        → 最近一个会话 → 新建会话。任何一步失败都返回 {}，表示本轮以无状态（stateless）方式问答。
        create_if_missing=False 时只复用已有会话、不新建（问答主流程用它避免失败请求留下空会话）。
        """
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
            if not create_if_missing:
                return {}

            # 没有可复用会话时，按当前问题摘要创建一个新的论文对话会话。
            session_title = self.truncate_text(str(self.payload_get(payload, "question", "") or "").strip(), 80)
            created_session = self.paper_chat_session_store.create_paper_chat_session(
                arxiv_id=arxiv_id,
                user_id=user_id,
                title=session_title,
            )
            if not created_session:
                # 原实现在这里抛 HTTPException 又被下方 except 立即吞掉，等价于直接降级；
                # 服务层也不应依赖 fastapi 的 HTTP 异常，因此改为显式记录并降级。
                logger.warning("Create paper chat session returned empty result, fallback to stateless QA: arxiv_id=%s", arxiv_id)
                return {}
            return created_session
        except Exception as exc:
            logger.warning("Chat session resolution failed, fallback to stateless QA: %s", exc)
            return {}

    def load_conversation_state(self, *, arxiv_id: str, payload: Any, chat_session: Dict[str, Any]) -> Dict[str, Any]:
        """加载并合并 DB 与请求体里的短期记忆，失败时显式降级为单轮 QA。

        返回值：
        - conversation_context：会话摘要伪 turn + 最近 turn，供问题消歧使用；
        - recent_conversation_context：仅最近 turn（不含摘要），供生成阶段的会话快照使用；
        - session_summary：结构化会话摘要（可能为 None）；
        - short_term_debug：记忆加载/合并过程的完整 debug，会随 retrieval_debug 一起落库（压缩后）。
        """
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

        # 摘要与短期记忆开关无关：即使短期记忆被禁用，只要会话里已有摘要仍会作为伪 turn 参与消歧。
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
        """把一次完整问答轮次作为原子业务单元写入会话消息表。

        - 会话功能关闭或本轮是无状态会话（没有 session_id）时直接跳过，返回空 turn_id；
        - turn 写入失败一律转换为 DATABASE_WRITE_FAILED 的 AppError 抛出，调用方据此告知前端「本轮未保存」；
        - 会话摘要更新在 turn 写入成功后进行，失败只记录在 session_summary_update 中，不影响本轮结果。
        """
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
            # 用写入后刷新的会话记录做摘要合并，确保拿到的是最新 summary / summary_turn_count。
            # 注意：摘要更新是「读-改-写」，同一会话并发提问时后写者会覆盖先写者的摘要增量。
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
        """用关键词规则从问题中识别用户偏好（排除/关注/风格三类），命中时把整句问题作为偏好记录。

        这是低成本启发式：例如英文 "only"、"detail" 在普通问题里也很常见，可能产生误判，
        因此结果只进入会话摘要的 user_preferences，不直接改变回答策略。
        """
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
        """把本轮前 N 条来源追加为摘要来源线索：按 source_id 去重，只保留最近 MAX_SUMMARY_ITEMS 条。"""
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
        """裁剪摘要各字段的条目数与单条长度；整体仍超长时再把事实和进展各砍到最近一半。"""
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
        """追加单条摘要条目（去重 + 保留最近 MAX_SUMMARY_ITEMS 条）。"""
        return self._merge_unique_summary_items(existing, [item])

    def _merge_unique_summary_items(self, existing: Any, new_items: List[str]) -> List[str]:
        """按出现顺序合并去重，超出上限时丢弃最旧的条目。"""
        merged = [str(item).strip() for item in list(existing or []) if str(item).strip()]
        for item in new_items:
            normalized = str(item or "").strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged[-MAX_SUMMARY_ITEMS:]

    def _trim_summary_list(self, value: Any, max_chars: int) -> List[str]:
        """截断每条文本长度并只保留最近 MAX_SUMMARY_ITEMS 条。"""
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
            # 旧客户端可能传 "sources": null 或非列表值；不能让一条脏数据导致整段短期记忆加载失败。
            raw_sources = raw_turn.get("sources")
            if not isinstance(raw_sources, list):
                raw_sources = []
            for item in raw_sources[:MAX_CONTEXT_SOURCES_PER_TURN]:
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
        # chunk_type \u603b\u6709\u9ed8\u8ba4\u503c "text"\uff0c\u4e0d\u80fd\u53c2\u4e0e\u300c\u662f\u5426\u4e3a\u7a7a\u6765\u6e90\u300d\u7684\u5224\u65ad\uff0c\u5426\u5219\u7a7a dict \u4e5f\u4f1a\u88ab\u5f53\u4f5c\u6709\u6548\u6765\u6e90\u3002
        if not any(normalized[key] for key in ("source_id", "page_number", "section_path", "content")):
            return None
        return normalized
