from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from services.memory.memory_debug import build_memory_debug_payload
from services.memory.memory_models import (
    AgentSessionMemory,
    BackendMemorySnapshot,
    PaperChatHistory,
    PreferenceSummary,
)
from services.storage.database_service import DatabaseService
from utils.config import get_default_user_id

logger = logging.getLogger(__name__)


class MemoryService:
    """封装记忆相关存储接口，并提供统一的画像、偏好与会话记忆能力。"""

    def __init__(self, db_service: Optional[DatabaseService] = None):
        """初始化记忆服务，并注入底层数据库访问依赖。"""
        self.db_service = db_service or DatabaseService()

    @staticmethod
    def _resolve_user_id(user_id: Optional[str] = None) -> str:
        """解析并兜底用户 ID，确保所有记忆查询都使用稳定主键。"""
        resolved = str(user_id or get_default_user_id()).strip()
        return resolved or get_default_user_id()

    @staticmethod
    def _coerce_limit(limit: int, default: int = 5) -> int:
        """把外部传入的数量限制规范化为正整数。"""
        try:
            normalized = int(limit)
        except (TypeError, ValueError):
            normalized = default
        return max(normalized, 1)

    @staticmethod
    def _truncate_text(value: Any, limit: int = 500) -> str:
        """截断过长文本，避免记忆摘要与调试载荷膨胀。"""
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(limit - 3, 0)] + "..."

    @staticmethod
    def _extract_arxiv_id(payload: Any) -> str:
        """从不同风格的论文载荷中提取 arXiv ID。"""
        if not isinstance(payload, dict):
            return ""
        return str(
            payload.get("arxiv_id") or payload.get("arxivId") or payload.get("id") or payload.get("paper_id") or ""
        ).strip()

    def _build_agent_context_from_session(self, agent_session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """从 Agent 会话态中提取对话上下文，供后续与后端记忆合并。"""
        session = dict(agent_session or {})
        backend_context: Dict[str, Any] = {}

        selected_paper = session.get("selected_paper")
        if isinstance(selected_paper, dict) and selected_paper:
            # 当前选中文章是最重要的会话锚点，后续检索和问答通常都依赖它。
            backend_context["selected_paper"] = selected_paper

        last_papers = session.get("last_papers")
        if isinstance(last_papers, list) and last_papers:
            # 保留最近论文列表，便于在多论文浏览场景下做上下文衔接。
            backend_context["last_papers"] = last_papers

        pending_action = session.get("pending_action")
        if isinstance(pending_action, dict) and pending_action:
            # 待执行动作可帮助 Agent 恢复中断状态，例如继续问答或继续推荐解释。
            backend_context["pending_action"] = pending_action

        paper_qa_result = session.get("paper_qa_result")
        if isinstance(paper_qa_result, dict) and paper_qa_result:
            # 最近一次论文问答结果可以作为短期显式记忆，方便下轮追问直接复用。
            backend_context["paper_qa_result"] = paper_qa_result

        active_arxiv_id = str(session.get("active_arxiv_id") or "").strip()
        if active_arxiv_id:
            backend_context["arxiv_id"] = active_arxiv_id

        active_paper_session_id = str(session.get("active_paper_session_id") or "").strip()
        if active_paper_session_id:
            backend_context["active_paper_session_id"] = active_paper_session_id

        last_intent = str(session.get("last_intent") or "").strip()
        if last_intent:
            backend_context["last_intent"] = last_intent

        last_tool_calls_summary = session.get("last_tool_calls_summary")
        if isinstance(last_tool_calls_summary, list) and last_tool_calls_summary:
            backend_context["last_tool_calls_summary"] = last_tool_calls_summary

        return backend_context

    @staticmethod
    def _merge_agent_context(backend_context: Optional[Dict[str, Any]], frontend_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """合并后端构造的上下文与前端传入的上下文，后者优先覆盖。"""
        merged_context = dict(backend_context or {})
        merged_context.update(dict(frontend_context or {}))
        return merged_context

    @staticmethod
    def _normalize_state_payload(final_state: Any) -> Dict[str, Any]:
        """把不同类型的最终状态对象统一规范化为字典。"""
        if final_state is None:
            return {}
        if isinstance(final_state, dict):
            return dict(final_state)
        if hasattr(final_state, "model_dump"):
            return dict(final_state.model_dump())
        return dict(final_state)

    def _summarize_tool_calls(self, tool_calls: Any, limit: int = 5) -> List[Dict[str, Any]]:
        """提炼最近工具调用摘要，避免把完整工具载荷直接写入会话记忆。"""
        summarized: List[Dict[str, Any]] = []
        for item in list(tool_calls or [])[-limit:]:
            if hasattr(item, "model_dump"):
                payload = item.model_dump()
            elif isinstance(item, dict):
                payload = dict(item)
            else:
                continue
            # 这里只保留最小可解释信息，避免把大参数、大结果写进数据库造成噪声。
            summarized.append(
                {
                    "tool_name": str(payload.get("tool_name") or "").strip(),
                    "status": str(payload.get("status") or "").strip(),
                    "summary": self._truncate_text(payload.get("summary"), limit=240),
                }
            )
        return summarized

    def _build_selected_paper_from_state(
        self,
        context: Dict[str, Any],
        paper_qa_result: Optional[Dict[str, Any]],
        active_arxiv_id: str,
    ) -> Optional[Dict[str, Any]]:
        """从最终状态中推断当前选中的论文信息，供 Agent 会话记忆复用。"""
        selected_paper = context.get("selected_paper")
        if isinstance(selected_paper, dict) and selected_paper:
            return selected_paper

        if isinstance(paper_qa_result, dict):
            inferred_id = self._extract_arxiv_id(paper_qa_result) or active_arxiv_id
            inferred_title = str(paper_qa_result.get("title") or "").strip()
            if inferred_id or inferred_title:
                return {
                    "arxiv_id": inferred_id,
                    "title": inferred_title,
                }
        return None

    def _extract_agent_memory_patch(self, final_state: Any) -> Dict[str, Any]:
        """从 Agent 最终状态中提取一份可增量写入的会话记忆补丁。"""
        state = self._normalize_state_payload(final_state)
        context = dict(state.get("context") or {})
        pending_action = state.get("pending_action") if "pending_action" in state else context.get("pending_action")
        paper_qa_result = state.get("paper_qa_result") if "paper_qa_result" in state else context.get("paper_qa_result")

        last_papers = context.get("last_papers") if "last_papers" in context else None
        if last_papers is None and isinstance(state.get("papers"), list) and state.get("papers"):
            # 某些状态对象不会把 last_papers 放在 context 中，这里做一次兼容回退。
            last_papers = list(state.get("papers") or [])

        active_arxiv_id = (
            self._extract_arxiv_id(context.get("selected_paper"))
            or self._extract_arxiv_id(paper_qa_result)
            or self._extract_arxiv_id(pending_action)
            or str(context.get("arxiv_id") or "").strip()
        )
        # active_arxiv_id 会作为当前会话聚焦论文的统一主键，后续恢复状态时优先依赖它。
        selected_paper = self._build_selected_paper_from_state(context, paper_qa_result, active_arxiv_id)
        active_paper_session_id = str(
            context.get("active_paper_session_id")
            or (paper_qa_result or {}).get("session_id")
            or ""
        ).strip()

        memory_patch: Dict[str, Any] = {
            "status": "active",
            "pending_action": pending_action,
            "paper_qa_result": paper_qa_result,
            "active_arxiv_id": active_arxiv_id or None,
            "active_paper_session_id": active_paper_session_id or None,
            "last_intent": state.get("intent"),
            "last_tool_calls_summary": self._summarize_tool_calls(state.get("tool_calls")),
            "last_response_summary": self._truncate_text(state.get("answer"), limit=500) or None,
        }

        if selected_paper is not None:
            memory_patch["selected_paper"] = selected_paper
        if last_papers is not None:
            memory_patch["last_papers"] = last_papers

        return memory_patch

    @staticmethod
    def _normalize_conversation_turn_payload(raw_turn: Any) -> Optional[Dict[str, Any]]:
        """把单轮对话结构规范化为统一字段格式。"""
        if not isinstance(raw_turn, dict):
            return None
        turn_id = str(raw_turn.get("turn_id", raw_turn.get("turnId", raw_turn.get("id", ""))) or "").strip()
        question = str(raw_turn.get("question", raw_turn.get("user_question", "")) or "").strip()
        answer_summary = str(
            raw_turn.get("answer_summary", raw_turn.get("answer", raw_turn.get("assistant_summary", ""))) or ""
        ).strip()
        created_at = str(raw_turn.get("created_at", raw_turn.get("createdAt", "")) or "").strip()
        sources = raw_turn.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        if not question and not answer_summary and not sources:
            return None
        return {
            "turn_id": turn_id,
            "created_at": created_at,
            "question": question,
            "answer_summary": answer_summary,
            "sources": sources,
        }

    @staticmethod
    def _conversation_turn_dedupe_key(turn: Dict[str, Any]) -> str:
        """为对话轮次生成去重键，优先使用 turn_id，其次使用问答内容。"""
        turn_id = str(turn.get("turn_id") or "").strip()
        if turn_id:
            return f"turn:{turn_id}"
        question = str(turn.get("question") or "").strip().lower()
        answer_summary = str(turn.get("answer_summary") or "").strip().lower()
        return f"qa:{question}|{answer_summary}"

    def _messages_to_conversation_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把消息级聊天记录重组为按轮次组织的对话上下文。"""
        turns_by_id: Dict[str, Dict[str, Any]] = {}
        ordered_turn_ids: List[str] = []
        derived_turns: List[Dict[str, Any]] = []
        current_unpaired_turn: Optional[Dict[str, Any]] = None

        def _new_turn(seed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            base = {
                "turn_id": "",
                "created_at": "",
                "question": "",
                "answer_summary": "",
                "sources": [],
            }
            if isinstance(seed, dict):
                base.update(seed)
            return base

        def _append_if_meaningful(candidate: Optional[Dict[str, Any]]) -> None:
            normalized_candidate = self._normalize_conversation_turn_payload(candidate)
            if normalized_candidate:
                derived_turns.append(normalized_candidate)

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            turn_id = str(message.get("turn_id") or "").strip()
            content = str(message.get("content") or "").strip()
            created_at = str(message.get("created_at") or "").strip()
            sources = message.get("sources", [])
            if not isinstance(sources, list):
                sources = []

            if turn_id:
                if turn_id not in turns_by_id:
                    # 显式 turn_id 表示上下游已经完成轮次配对，这里优先按 turn_id 聚合。
                    turns_by_id[turn_id] = _new_turn({"turn_id": turn_id, "created_at": created_at})
                    ordered_turn_ids.append(turn_id)
                current_turn = turns_by_id[turn_id]
                if created_at and not current_turn.get("created_at"):
                    current_turn["created_at"] = created_at
                if role == "user" and content and not current_turn.get("question"):
                    current_turn["question"] = content
                elif role == "assistant":
                    if content and not current_turn.get("answer_summary"):
                        current_turn["answer_summary"] = content
                    if sources:
                        current_turn["sources"] = sources
                continue

            if role == "user":
                # 没有 turn_id 时，遇到新的 user 消息就开启一轮临时配对。
                _append_if_meaningful(current_unpaired_turn)
                current_unpaired_turn = _new_turn({"created_at": created_at, "question": content})
                continue

            if role == "assistant":
                if current_unpaired_turn is None:
                    # 极端情况下先收到 assistant 消息，也要兜底生成一轮，避免信息丢失。
                    current_unpaired_turn = _new_turn({"created_at": created_at})
                if content and not current_unpaired_turn.get("answer_summary"):
                    current_unpaired_turn["answer_summary"] = content
                if sources:
                    current_unpaired_turn["sources"] = sources
                _append_if_meaningful(current_unpaired_turn)
                current_unpaired_turn = None

        _append_if_meaningful(current_unpaired_turn)

        ordered_turns = [turns_by_id[item] for item in ordered_turn_ids]
        normalized_turns = [
            normalized_turn
            for normalized_turn in (self._normalize_conversation_turn_payload(turn) for turn in ordered_turns + derived_turns)
            if normalized_turn
        ]
        return normalized_turns

    def load_paper_conversation_context(
        self,
        user_id: Optional[str],
        arxiv_id: str,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        resolved_user_id = self._resolve_user_id(user_id)
        message_limit = self._coerce_limit(limit)
        requested_session_id = str(session_id or "").strip() or None
        selected_session: Optional[Dict[str, Any]] = None

        if requested_session_id:
            candidate_session = self.db_service.get_paper_chat_session(requested_session_id, user_id=resolved_user_id)
            if candidate_session and candidate_session.get("arxiv_id") == arxiv_id:
                selected_session = candidate_session

        if selected_session is None:
            recent_sessions = self.db_service.list_paper_chat_sessions(
                arxiv_id=arxiv_id,
                user_id=resolved_user_id,
                limit=message_limit,
            )
            for candidate_session in recent_sessions:
                candidate_status = str(candidate_session.get("status") or "active").strip().lower()
                if candidate_status == "active":
                    selected_session = candidate_session
                    break
            if selected_session is None and recent_sessions:
                selected_session = recent_sessions[0]

        messages: List[Dict[str, Any]] = []
        turns: List[Dict[str, Any]] = []
        if selected_session:
            messages = self.db_service.list_paper_chat_messages(
                selected_session["session_id"],
                user_id=resolved_user_id,
            )
            turns = self._messages_to_conversation_context(messages)[-message_limit:]

        return {
            "user_id": resolved_user_id,
            "arxiv_id": arxiv_id,
            "requested_session_id": requested_session_id,
            "selected_session": selected_session,
            "selected_session_id": str((selected_session or {}).get("session_id") or "").strip() or None,
            "turns": turns,
            "turn_count": len(turns),
            "message_count": len(messages),
        }

    def merge_conversation_context(
        self,
        db_context: Any,
        payload_context: Any,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """合并数据库上下文与请求上下文，并按轮次去重后返回最近若干轮。"""
        normalized_limit = self._coerce_limit(limit)
        merged_turns: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        for candidate_turn in list(db_context or []) + list(payload_context or []):
            normalized_turn = self._normalize_conversation_turn_payload(candidate_turn)
            if not normalized_turn:
                continue
            dedupe_key = self._conversation_turn_dedupe_key(normalized_turn)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            merged_turns.append(normalized_turn)

        return merged_turns[-normalized_limit:]

    def load_user_profile(self, user_id: Optional[str]) -> Dict[str, Any]:
        """读取用户长期研究画像。"""
        resolved_user_id = self._resolve_user_id(user_id)
        return self.db_service.get_user_research_profile(user_id=resolved_user_id)

    @staticmethod
    def _normalize_profile_list(values: Any, limit: int = 30) -> List[str]:
        """把画像词项规范化为去重后的字符串列表。"""
        if values is None:
            return []
        source = values if isinstance(values, list) else [values]
        normalized: List[str] = []
        for item in source:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _merge_profile_list(existing: Any, incoming: Any, limit: int = 30, prepend: bool = False) -> List[str]:
        """合并画像词项列表并去重，支持控制新旧值的优先顺序。"""
        merged: List[str] = []
        ordered_values = [incoming, existing] if prepend else [existing, incoming]
        for bucket in ordered_values:
            for item in MemoryService._normalize_profile_list(bucket, limit=limit):
                if item not in merged:
                    merged.append(item)
                if len(merged) >= limit:
                    return merged
        return merged

    @staticmethod
    def _normalize_categories(values: Any, limit: int = 12) -> List[str]:
        """把分类字段规范化为去重后的分类列表。"""
        if isinstance(values, str):
            source = [item.strip() for item in values.split(",")]
        elif isinstance(values, list):
            source = [str(item or "").strip() for item in values]
        else:
            source = [str(values or "").strip()] if values is not None else []
        return [item for item in MemoryService._normalize_profile_list(source, limit=limit) if item]

    def _resolve_paper_payload(self, arxiv_id: str, paper_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """优先使用显式传入的论文载荷，缺失时再从数据库读取论文信息。"""
        if isinstance(paper_payload, dict) and paper_payload:
            return dict(paper_payload)
        paper = self.db_service.get_paper(arxiv_id)
        return dict(paper or {})

    def _extract_paper_profile_signals(
        self,
        arxiv_id: str,
        paper_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """从论文元数据中提取可用于更新用户画像的主题、分类与代表论文信号。"""
        paper = self._resolve_paper_payload(arxiv_id, paper_payload=paper_payload)
        categories = self._normalize_categories(paper.get("categories"), limit=12)
        title = str(paper.get("title") or "").strip()
        positive_topics = categories[:]
        representative_papers = [str(arxiv_id or "").strip()] if str(arxiv_id or "").strip() else []
        if title:
            positive_topics = self._merge_profile_list(positive_topics, [title], limit=12)
        return {
            "categories": categories,
            "positive_topics": positive_topics,
            "representative_papers": representative_papers,
        }

    def patch_user_profile(
        self,
        user_id: Optional[str],
        patch: Optional[Dict[str, Any]],
        source: str,
    ) -> Dict[str, Any]:
        """按来源策略更新用户研究画像，区分手动覆盖与系统增量合并。"""
        resolved_user_id = self._resolve_user_id(user_id)
        normalized_source = str(source or "unknown").strip().lower() or "unknown"
        current = self.load_user_profile(resolved_user_id)
        incoming = dict(patch or {})

        if normalized_source in {"manual_upsert", "manual_put"}:
            # 显式全量覆盖类来源直接交给 upsert，允许调用方完整重写画像。
            return self.db_service.upsert_user_research_profile(user_id=resolved_user_id, profile=incoming)

        if normalized_source in {"manual", "api", "user"}:
            # 人工/API 直接 patch 时，默认认为调用方已经自行控制字段粒度。
            return self.db_service.patch_user_research_profile(user_id=resolved_user_id, profile=incoming)

        merged_patch: Dict[str, Any] = {}
        list_limits = {
            "positive_topics": 30,
            "negative_topics": 30,
            "recent_topics": 30,
            "preferred_categories": 20,
            "common_question_types": 20,
            "representative_papers": 20,
        }
        for field_name, limit in list_limits.items():
            if field_name in incoming:
                # 系统自动写入画像时，统一采用列表合并而不是覆盖，避免历史偏好被瞬间抹掉。
                merged_patch[field_name] = self._merge_profile_list(current.get(field_name), incoming.get(field_name), limit=limit)

        if normalized_source in {"manual_answer_style", "manual_style"} and "preferred_answer_style" in incoming:
            merged_patch["preferred_answer_style"] = str(incoming.get("preferred_answer_style") or "").strip()

        if not merged_patch:
            return current
        return self.db_service.patch_user_research_profile(user_id=resolved_user_id, profile=merged_patch)

    def update_profile_from_note(self, user_id: Optional[str], note: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """根据用户保存并允许入画像的笔记内容更新长期研究画像。"""
        normalized_note = dict(note or {})
        if not normalized_note or not normalized_note.get("include_in_profile"):
            # 只有显式标记 include_in_profile 的笔记，才会参与长期画像学习。
            return self.load_user_profile(user_id)

        arxiv_id = str(normalized_note.get("arxiv_id") or "").strip()
        tags = self._normalize_profile_list(normalized_note.get("tags"), limit=20)
        note_title = str(normalized_note.get("title") or "").strip()
        note_topics = self._merge_profile_list(tags, [note_title] if note_title else [], limit=20)
        note_type = str(normalized_note.get("note_type") or "").strip()
        patch = {
            "positive_topics": note_topics,
            "recent_topics": note_topics,
            "common_question_types": [note_type] if note_type else [],
            "representative_papers": [arxiv_id] if arxiv_id else [],
        }
        return self.patch_user_profile(user_id, patch, source="note_include_in_profile")

    def update_profile_from_preference(
        self,
        user_id: Optional[str],
        arxiv_id: str,
        action_type: str,
        paper_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """根据喜欢/不喜欢等显式偏好动作更新用户长期画像。"""
        normalized_action = str(action_type or "").strip().lower()
        if normalized_action not in {"like", "liked", "dislike", "disliked", "not_interested"}:
            return self.load_user_profile(user_id)

        paper_signals = self._extract_paper_profile_signals(arxiv_id, paper_payload=paper_payload)
        if normalized_action in {"like", "liked"}:
            # 正反馈同时增强主题、近期兴趣、分类偏好与代表论文。
            patch = {
                "positive_topics": paper_signals.get("positive_topics", []),
                "recent_topics": paper_signals.get("positive_topics", []),
                "preferred_categories": paper_signals.get("categories", []),
                "representative_papers": paper_signals.get("representative_papers", []),
            }
            return self.patch_user_profile(user_id, patch, source="liked_paper")

        # 负反馈当前主要沉淀为 negative_topics，避免直接过度干预正向画像字段。
        patch = {
            "negative_topics": paper_signals.get("positive_topics", []),
        }
        return self.patch_user_profile(user_id, patch, source="disliked_paper")

    def update_profile_from_paper_action(
        self,
        user_id: Optional[str],
        arxiv_id: str,
        action_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """根据通用论文动作分发到对应画像更新逻辑。"""
        normalized_action = str(action_type or "").strip().lower()
        if normalized_action in {"like", "liked", "dislike", "disliked", "not_interested"}:
            return self.update_profile_from_preference(user_id, arxiv_id, normalized_action)

        if normalized_action == "note_saved" and isinstance(metadata, dict) and metadata.get("include_in_profile"):
            # note_saved 本身不是显式偏好，但如果笔记允许入画像，就按笔记信号处理。
            note_like_payload = {
                "arxiv_id": arxiv_id,
                "note_type": metadata.get("note_type"),
                "tags": metadata.get("tags") or [],
                "include_in_profile": True,
            }
            return self.update_profile_from_note(user_id, note_like_payload)

        return self.load_user_profile(user_id)

    def load_preference_summary(self, user_id: Optional[str]) -> Dict[str, Any]:
        """加载用户偏好摘要，包括点赞/点踩、动作映射与兴趣向量。"""
        resolved_user_id = self._resolve_user_id(user_id)
        liked_papers = self.db_service.get_liked_papers(user_id=resolved_user_id)
        disliked_papers = self.db_service.get_disliked_papers(user_id=resolved_user_id)
        paper_actions = self.db_service.get_user_paper_action_map(user_id=resolved_user_id)
        interest_vector = self.db_service.get_user_interest_vector(user_id=resolved_user_id)

        summary = PreferenceSummary(
            user_id=resolved_user_id,
            liked_papers=liked_papers,
            disliked_papers=disliked_papers,
            paper_actions=paper_actions,
            interest_vector=interest_vector,
            counts={
                "liked_papers": len(liked_papers),
                "disliked_papers": len(disliked_papers),
                "paper_action_types": len(paper_actions),
                "paper_action_entries": sum(len(values) for values in paper_actions.values()),
            },
        )
        return summary.to_dict()

    def build_user_memory_summary(self, user_id: Optional[str]) -> Dict[str, Any]:
        """构造面向前端与 Agent 的用户记忆摘要视图。"""
        resolved_user_id = self._resolve_user_id(user_id)
        profile = self.load_user_profile(resolved_user_id)
        preference_summary = self.load_preference_summary(resolved_user_id)
        interest_vector = preference_summary.get("interest_vector") or {}
        recent_actions = self.db_service.get_user_paper_actions(user_id=resolved_user_id)
        recent_action_types = {"read", "favorite", "later", "note_saved", "not_interested"}
        recent_actions_summary: Dict[str, List[str]] = {action_type: [] for action_type in sorted(recent_action_types)}
        for item in recent_actions:
            action_type = str(item.get("action_type") or "").strip()
            arxiv_id = str(item.get("arxiv_id") or "").strip()
            if action_type not in recent_action_types or not arxiv_id:
                continue
            if len(recent_actions_summary[action_type]) >= 5:
                continue
            if arxiv_id not in recent_actions_summary[action_type]:
                recent_actions_summary[action_type].append(arxiv_id)

        interest_clusters = interest_vector.get("interest_clusters") or []
        interest_clusters_summary: List[Dict[str, Any]] = []
        for cluster in interest_clusters[:3]:
            if not isinstance(cluster, dict):
                continue
            # 这里只提炼展示层真正关心的簇摘要，避免把完整向量等重数据暴露出去。
            interest_clusters_summary.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "paper_count": cluster.get("paper_count", 0),
                    "representative_papers": list(cluster.get("paper_ids") or [])[:3],
                    "labels": list(cluster.get("labels") or cluster.get("topics") or [])[:3],
                }
            )

        return {
            "profile": {
                "positive_topics": list(profile.get("positive_topics") or []),
                "negative_topics": list(profile.get("negative_topics") or []),
                "recent_topics": list(profile.get("recent_topics") or []),
                "preferred_categories": list(profile.get("preferred_categories") or []),
                "preferred_answer_style": str(profile.get("preferred_answer_style") or "").strip(),
                "common_question_types": list(profile.get("common_question_types") or []),
                "representative_papers": list(profile.get("representative_papers") or []),
            },
            "preference_summary": {
                "liked_count": int((preference_summary.get("counts") or {}).get("liked_papers", len(preference_summary.get("liked_papers") or []))),
                "disliked_count": int((preference_summary.get("counts") or {}).get("disliked_papers", len(preference_summary.get("disliked_papers") or []))),
                "recent_liked_papers": list(preference_summary.get("liked_papers") or [])[:5],
                "recent_disliked_papers": list(preference_summary.get("disliked_papers") or [])[:5],
                "has_interest_vector": bool(interest_vector),
                "profile_mode": str(interest_vector.get("profile_mode") or "").strip(),
                "cluster_count": int(interest_vector.get("cluster_count") or 0),
                "interest_clusters_summary": interest_clusters_summary,
            },
            "recent_actions_summary": recent_actions_summary,
            "memory_status": {
                "user_profile_enabled": True,
                "preference_memory_available": bool(
                    preference_summary.get("liked_papers")
                    or preference_summary.get("disliked_papers")
                    or preference_summary.get("paper_actions")
                ),
                "interest_vector_available": bool(interest_vector),
                "last_signal_timestamp": self.db_service.get_latest_user_signal_timestamp(user_id=resolved_user_id),
            },
        }

    def load_paper_notes(self, user_id: Optional[str], arxiv_id: str) -> List[Dict[str, Any]]:
        """读取指定用户在某篇论文下保存的笔记列表。"""
        resolved_user_id = self._resolve_user_id(user_id)
        return self.db_service.list_paper_notes(arxiv_id=arxiv_id, user_id=resolved_user_id)

    def load_paper_chat_history(
        self,
        user_id: Optional[str],
        arxiv_id: str,
        session_id: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """读取单篇论文的对话历史，并选择一个最合适的会话作为当前会话。"""
        resolved_user_id = self._resolve_user_id(user_id)
        message_limit = self._coerce_limit(limit)
        requested_session_id = str(session_id or "").strip() or None

        sessions: List[Dict[str, Any]] = []
        selected_session: Optional[Dict[str, Any]] = None

        if requested_session_id:
            candidate_session = self.db_service.get_paper_chat_session(requested_session_id, user_id=resolved_user_id)
            if candidate_session and candidate_session.get("arxiv_id") == arxiv_id:
                # 调用方显式指定 session_id 时，优先使用该会话，但前提是论文归属匹配。
                selected_session = candidate_session
                sessions = [candidate_session]
            else:
                logger.warning(
                    "Skipping paper chat session %s for user %s and arxiv_id %s due to mismatch or absence",
                    requested_session_id,
                    resolved_user_id,
                    arxiv_id,
                )

        if selected_session is None:
            # 未指定或指定失败时，退化为按论文读取最近会话，并默认取第一条作为当前会话。
            sessions = self.db_service.list_paper_chat_sessions(
                arxiv_id=arxiv_id,
                user_id=resolved_user_id,
                limit=message_limit,
            )
            selected_session = sessions[0] if sessions else None

        messages: List[Dict[str, Any]] = []
        total_messages = 0
        if selected_session:
            all_messages = self.db_service.list_paper_chat_messages(
                selected_session["session_id"],
                user_id=resolved_user_id,
            )
            total_messages = len(all_messages)
            # 返回给调用方的是最近若干条消息，但 total_messages 会保留完整规模信息。
            messages = all_messages[-message_limit:]

        history = PaperChatHistory(
            user_id=resolved_user_id,
            arxiv_id=arxiv_id,
            requested_session_id=requested_session_id,
            selected_session=selected_session,
            sessions=sessions,
            messages=messages,
            message_limit=message_limit,
            total_messages=total_messages,
        )
        return history.to_dict()

    def build_memory_snapshot(
        self,
        *,
        user_id: Optional[str],
        arxiv_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_profile: bool = True,
        include_preferences: bool = True,
        include_notes: bool = True,
        include_chat_history: bool = True,
        chat_limit: int = 5,
    ) -> Dict[str, Any]:
        """按需聚合用户画像、偏好、笔记与聊天历史，构造统一后端记忆快照。"""
        resolved_user_id = self._resolve_user_id(user_id)
        snapshot = BackendMemorySnapshot(
            user_profile=self.load_user_profile(resolved_user_id) if include_profile else None,
            preference_summary=self.load_preference_summary(resolved_user_id) if include_preferences else None,
            paper_notes=self.load_paper_notes(resolved_user_id, arxiv_id) if include_notes and arxiv_id else [],
            paper_chat_history=(
                self.load_paper_chat_history(
                    resolved_user_id,
                    arxiv_id,
                    session_id=session_id,
                    limit=chat_limit,
                )
                if include_chat_history and arxiv_id
                else None
            ),
        )
        return snapshot.to_dict()

    def load_agent_memory(
        self,
        user_id: Optional[str],
        session_id: Optional[str],
        frontend_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """加载 Agent 会话记忆，并把后端记忆上下文与前端上下文合并。"""
        resolved_user_id = self._resolve_user_id(user_id)
        agent_session = self.db_service.create_or_get_agent_session(
            user_id=resolved_user_id,
            session_id=session_id,
        )
        resolved_session_id = str((agent_session or {}).get("session_id") or session_id or "").strip() or None
        backend_memory = self._build_agent_context_from_session(agent_session)
        merged_context = self._merge_agent_context(backend_memory, frontend_context)

        payload = AgentSessionMemory(
            session_id=resolved_session_id,
            agent_session=agent_session,
            backend_memory=backend_memory,
            merged_context=merged_context,
        )
        return payload.to_dict()

    def save_agent_memory(
        self,
        user_id: Optional[str],
        session_id: Optional[str],
        final_state: Any,
    ) -> Optional[Dict[str, Any]]:
        """把 Agent 最终状态提炼成会话记忆补丁，并写回持久化会话记录。"""
        resolved_user_id = self._resolve_user_id(user_id)
        agent_session = self.db_service.create_or_get_agent_session(
            user_id=resolved_user_id,
            session_id=session_id,
        )
        if not agent_session:
            return None

        resolved_session_id = str(agent_session.get("session_id") or "").strip()
        if not resolved_session_id:
            return agent_session

        memory_patch = self._extract_agent_memory_patch(final_state)
        # update_agent_session 采用 patch 语义，避免每轮都覆盖整个已存会话记忆对象。
        self.db_service.update_agent_session(
            session_id=resolved_session_id,
            user_id=resolved_user_id,
            memory_patch=memory_patch,
        )
        return self.db_service.get_agent_session(resolved_session_id, user_id=resolved_user_id)

    def build_memory_debug(
        self,
        *,
        user_id: Optional[str],
        arxiv_id: Optional[str] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        preference_summary: Optional[Dict[str, Any]] = None,
        paper_notes: Optional[List[Dict[str, Any]]] = None,
        paper_chat_history: Optional[Dict[str, Any]] = None,
        frontend_context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构造统一的记忆调试结果，供前端观测或问题排查使用。"""
        resolved_user_id = self._resolve_user_id(user_id)
        return build_memory_debug_payload(
            user_id=resolved_user_id,
            arxiv_id=arxiv_id,
            user_profile=user_profile,
            preference_summary=preference_summary,
            paper_notes=paper_notes,
            paper_chat_history=paper_chat_history,
            frontend_context=frontend_context,
            extra=extra,
        )

    def merge_frontend_context_with_memory(
        self,
        frontend_context: Optional[Dict[str, Any]],
        backend_memory: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """把前端上下文与后端记忆块合并为一个可直接下发的上下文结构。"""
        merged_context = dict(frontend_context or {})
        existing_memory = merged_context.get("backend_memory")
        if isinstance(existing_memory, dict) and isinstance(backend_memory, dict):
            merged_context["backend_memory"] = {**existing_memory, **backend_memory}
            return merged_context

        merged_context["backend_memory"] = dict(backend_memory or {})
        return merged_context
