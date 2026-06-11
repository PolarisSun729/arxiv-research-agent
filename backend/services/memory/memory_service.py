from __future__ import annotations

import logging
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.context_merge import merge_backend_authoritative_context
from services.memory.memory_debug import build_memory_debug_payload
from services.memory.memory_models import (
    AgentSessionMemory,
    BackendMemorySnapshot,
    PaperChatHistory,
    PreferenceSummary,
)
from services.memory.concept_normalizer import ConceptNormalizer
from services.memory.paper_evidence_extractor import PAPER_EVIDENCE_EXTRACTOR_VERSION, PaperEvidenceExtractor
from services.memory.profile_aggregator import ProfileAggregator
from services.memory.profile_reviewer import ProfileReviewer
from services.memory.research_profile_generator import ResearchProfileGenerator
from services.storage.database_service import DatabaseService
from utils.config import PROFILE_EVIDENCE_CONFIG, get_default_user_id

logger = logging.getLogger(__name__)

ARXIV_ID_PATTERN = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$")
ARXIV_CATEGORY_PATTERN = re.compile(r"^[a-z-]+(?:\.[A-Z]{2})?$")
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
LOW_INFORMATION_TOPIC_TERMS = {
    "analysis",
    "approach",
    "framework",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "system",
    "systems",
}
PROFILE_BUILD_MODES = {"incremental", "full", "repair"}
PROFILE_BUILD_DEFAULT_PAPER_LIMITS = {
    "incremental": 40,
    "full": 1000,
    "repair": 80,
}
PROFILE_STRONG_EVENT_TYPES = {"liked", "disliked", "favorite", "later", "note_saved", "qa_asked", "not_interested"}
PROFILE_RECENT_CONTEXT_EVENT_TYPES = {"liked", "disliked", "favorite", "later", "note_saved", "qa_asked"}


class MemoryService:
    """封装记忆相关存储接口，并提供统一的画像、偏好与会话记忆能力。"""

    def __init__(self, db_service: Optional[DatabaseService] = None, generation_service: Optional[Any] = None):
        """初始化记忆服务，并注入底层数据库访问依赖。"""
        self.db_service = db_service or DatabaseService()
        self.profile_generator = ResearchProfileGenerator(
            concept_normalizer=ConceptNormalizer(generation_service=generation_service, enable_llm_naming=False)
        )
        self.profile_aggregator = ProfileAggregator(concept_normalizer=self.profile_generator.concept_normalizer)
        self.profile_reviewer = ProfileReviewer()
        self.paper_evidence_extractor = PaperEvidenceExtractor(generation_service=generation_service)

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
            # 这里只保存最近一次确认动作的业务摘要，方便前端展示、诊断和兼容旧会话。
            # 真正的 LangGraph interrupt 执行现场依赖 checkpointer，不能只靠数据库里的摘要恢复。
            backend_context["pending_action"] = pending_action

        paper_qa_result = session.get("paper_qa_result")
        if isinstance(paper_qa_result, dict) and paper_qa_result:
            # 最近一次论文问答结果可以作为短期显式记忆，方便下轮追问直接复用。
            backend_context["paper_qa_result"] = paper_qa_result

        active_arxiv_id = str(session.get("active_arxiv_id") or "").strip()
        if active_arxiv_id:
            backend_context["active_arxiv_id"] = active_arxiv_id
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
        """兼容旧调用点：后端会话状态为权威，前端 context 只补充白名单字段。"""
        return merge_backend_authoritative_context(
            backend_context=backend_context,
            frontend_context=frontend_context,
        ).merged_context

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
        if isinstance(paper_qa_result, dict):
            # 本轮 QA 的真实目标优先级高于旧 selected_paper；
            # 否则用户问“第二篇”后，会话记忆仍可能停留在前端默认选中的第一篇。
            inferred_id = self._extract_arxiv_id(paper_qa_result) or active_arxiv_id
            inferred_title = str(paper_qa_result.get("title") or "").strip()
            if inferred_id or inferred_title:
                return {
                    "arxiv_id": inferred_id,
                    "title": inferred_title,
                }

        selected_paper = context.get("selected_paper")
        if isinstance(selected_paper, dict) and selected_paper:
            return selected_paper
        return None

    def _extract_agent_memory_patch(self, final_state: Any) -> Dict[str, Any]:
        """从 Agent 最终状态中提取一份可增量写入的会话记忆补丁。"""
        state = self._normalize_state_payload(final_state)
        context = dict(state.get("context") or {})
        pending_action = state.get("pending_action") if "pending_action" in state else context.get("pending_action")
        paper_qa_result = state.get("paper_qa_result") if "paper_qa_result" in state else context.get("paper_qa_result")
        # pending_action 会随会话记忆保存，但它只是确认卡片的业务镜像；
        # resume 能否继续执行仍以 LangGraph checkpointer 中的现场为准。

        last_papers = context.get("last_papers") if "last_papers" in context else None
        if last_papers is None and isinstance(state.get("papers"), list) and state.get("papers"):
            # 某些状态对象不会把 last_papers 放在 context 中，这里做一次兼容回退。
            last_papers = list(state.get("papers") or [])

        active_arxiv_id = (
            # 当前轮次真实产物优先于旧 UI 选中态，防止“第 N 篇”解析成功后又被历史焦点覆盖。
            self._extract_arxiv_id(paper_qa_result)
            or self._extract_arxiv_id(pending_action)
            or self._extract_arxiv_id(context.get("selected_paper"))
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

    def _messages_to_conversation_context_with_debug(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """把消息级聊天记录重组为完整对话轮次，并统计被过滤的异常旧数据。"""
        turns_by_id: Dict[str, Dict[str, Any]] = {}
        turn_role_counts: Dict[str, Dict[str, int]] = {}
        ordered_turn_ids: List[str] = []
        derived_turns: List[Dict[str, Any]] = []
        current_unpaired_turn: Optional[Dict[str, Any]] = None
        debug = {
            "filtered_incomplete_turn_count": 0,
            "invalid_turn_count": 0,
            "filtered_turn_count": 0,
        }

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

        def _append_if_complete(candidate: Optional[Dict[str, Any]]) -> None:
            normalized_candidate = self._normalize_conversation_turn_payload(candidate)
            if normalized_candidate and normalized_candidate.get("question") and normalized_candidate.get("answer_summary"):
                derived_turns.append(normalized_candidate)
                return
            if normalized_candidate:
                # 没有 turn_id 的旧消息只能按相邻 user/assistant 配对；缺任一侧时不能进入问题改写上下文。
                debug["filtered_incomplete_turn_count"] += 1
                debug["filtered_turn_count"] += 1

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
                    # 显式 turn_id 是完整轮次的边界，聚合后再校验角色数量，兼容历史半轮脏数据。
                    turns_by_id[turn_id] = _new_turn({"turn_id": turn_id, "created_at": created_at})
                    turn_role_counts[turn_id] = {"user": 0, "assistant": 0}
                    ordered_turn_ids.append(turn_id)
                current_turn = turns_by_id[turn_id]
                if created_at and not current_turn.get("created_at"):
                    current_turn["created_at"] = created_at
                if role == "user":
                    turn_role_counts[turn_id]["user"] += 1
                    if content and not current_turn.get("question"):
                        current_turn["question"] = content
                elif role == "assistant":
                    turn_role_counts[turn_id]["assistant"] += 1
                    if content and not current_turn.get("answer_summary"):
                        current_turn["answer_summary"] = content
                    if sources and not current_turn.get("sources"):
                        current_turn["sources"] = sources
                continue

            if role == "user":
                # 没有 turn_id 时只能做相邻配对；前一轮没等到 assistant 就必须过滤，避免半轮污染追问。
                _append_if_complete(current_unpaired_turn)
                current_unpaired_turn = _new_turn({"created_at": created_at, "question": content})
                continue

            if role == "assistant":
                if current_unpaired_turn is None:
                    debug["filtered_incomplete_turn_count"] += 1
                    debug["filtered_turn_count"] += 1
                    continue
                if content and not current_unpaired_turn.get("answer_summary"):
                    current_unpaired_turn["answer_summary"] = content
                if sources:
                    current_unpaired_turn["sources"] = sources
                _append_if_complete(current_unpaired_turn)
                current_unpaired_turn = None

        _append_if_complete(current_unpaired_turn)

        ordered_turns: List[Dict[str, Any]] = []
        for item in ordered_turn_ids:
            role_counts = turn_role_counts.get(item, {})
            turn = turns_by_id[item]
            if role_counts.get("user", 0) != 1 or role_counts.get("assistant", 0) != 1:
                if role_counts.get("user", 0) > 1 or role_counts.get("assistant", 0) > 1:
                    debug["invalid_turn_count"] += 1
                else:
                    debug["filtered_incomplete_turn_count"] += 1
                debug["filtered_turn_count"] += 1
                continue
            if not turn.get("question") or not turn.get("answer_summary"):
                debug["filtered_incomplete_turn_count"] += 1
                debug["filtered_turn_count"] += 1
                continue
            ordered_turns.append(turn)
        normalized_turns = [
            normalized_turn
            for normalized_turn in (self._normalize_conversation_turn_payload(turn) for turn in ordered_turns + derived_turns)
            if normalized_turn
        ]
        return normalized_turns, debug

    def _messages_to_conversation_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把消息级聊天记录重组为按轮次组织的对话上下文。"""
        turns, _debug = self._messages_to_conversation_context_with_debug(messages)
        return turns

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
        total_message_count = 0
        db_message_read_limit = message_limit * 4 + 4
        filter_debug = {
            "filtered_incomplete_turn_count": 0,
            "invalid_turn_count": 0,
            "filtered_turn_count": 0,
        }
        if selected_session:
            # 模型上下文只需要最近 N 轮；这里从数据库层限制消息窗口，避免长会话全量加载后再裁剪。
            messages = self.db_service.list_recent_paper_chat_messages(
                session_id=selected_session["session_id"],
                user_id=resolved_user_id,
                limit=db_message_read_limit,
            )
            total_message_count = self.db_service.count_paper_chat_messages(
                session_id=selected_session["session_id"],
                user_id=resolved_user_id,
            )
            all_turns, filter_debug = self._messages_to_conversation_context_with_debug(messages)
            turns = all_turns[-message_limit:]

        return {
            "user_id": resolved_user_id,
            "arxiv_id": arxiv_id,
            "requested_session_id": requested_session_id,
            "selected_session": selected_session,
            "selected_session_id": str((selected_session or {}).get("session_id") or "").strip() or None,
            "turns": turns,
            "turn_count": len(turns),
            "message_count": len(messages),
            "db_message_read_count": len(messages),
            "db_message_read_limit": db_message_read_limit if selected_session else 0,
            "total_message_count": total_message_count,
            "filtered_incomplete_turn_count": filter_debug["filtered_incomplete_turn_count"],
            "invalid_turn_count": filter_debug["invalid_turn_count"],
            "filtered_turn_count": filter_debug["filtered_turn_count"],
            "memory_filter_debug": filter_debug,
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

    def load_user_profile_layers(self, user_id: Optional[str]) -> Dict[str, Any]:
        """读取画像分层视图，便于调试 manual/generated/effective 的真实边界。"""
        resolved_user_id = self._resolve_user_id(user_id)
        if hasattr(self.db_service, "get_user_profile_layers"):
            return self.db_service.get_user_profile_layers(user_id=resolved_user_id)
        return {
            "manual_profile": {},
            "generated_profile": {},
            "effective_profile": self.load_user_profile(resolved_user_id),
        }

    def load_user_profile_detail(self, user_id: Optional[str]) -> Dict[str, Any]:
        """返回画像详情页需要的三层画像、构建记录和快照摘要。"""
        resolved_user_id = self._resolve_user_id(user_id)
        layers = self.load_user_profile_layers(resolved_user_id)
        jobs = self.db_service.list_user_profile_build_jobs(resolved_user_id, limit=10) if hasattr(self.db_service, "list_user_profile_build_jobs") else []
        snapshots = self.db_service.list_user_profile_snapshots(resolved_user_id, limit=10) if hasattr(self.db_service, "list_user_profile_snapshots") else []
        generated = layers.get("generated_profile") if isinstance(layers.get("generated_profile"), dict) else {}
        effective = layers.get("effective_profile") if isinstance(layers.get("effective_profile"), dict) else {}
        return {
            "user_id": resolved_user_id,
            **layers,
            "evidence_summary": generated.get("evidence_summary") or {},
            "quality_report": generated.get("quality_report") or effective.get("quality_report") or {},
            "build_jobs": jobs,
            "snapshots": snapshots,
            "latest_build_job": jobs[0] if jobs else None,
        }

    def get_profile_topic_evidence(self, user_id: Optional[str], topic: str) -> Dict[str, Any]:
        """按 effective profile 查询单个 topic 的证据解释。"""
        resolved_user_id = self._resolve_user_id(user_id)
        profile = self.load_user_profile(resolved_user_id)
        topic_evidence = profile.get("topic_evidence") if isinstance(profile.get("topic_evidence"), dict) else {}
        normalized_topic = str(topic or "").strip()
        evidence = topic_evidence.get(normalized_topic)
        if evidence is None:
            topic_key = normalized_topic.lower()
            evidence = next((value for key, value in topic_evidence.items() if str(key).lower() == topic_key), None)
        return {"user_id": resolved_user_id, "topic": normalized_topic, "evidence": evidence or {}, "found": bool(evidence)}

    def create_profile_rebuild_job(self, user_id: Optional[str], build_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """创建画像重建任务记录；实际重建由 API 后台任务或调度器执行。"""
        resolved_user_id = self._resolve_user_id(user_id)
        incoming_config = dict(build_config or {})
        build_mode = self._normalize_profile_build_mode(incoming_config.get("build_mode"))
        paper_limit = self._resolve_profile_paper_limit(build_mode, incoming_config.get("max_papers"))
        config = {
            "legacy_output_projection": True,
            "async_requested": True,
            **incoming_config,
            "build_mode": build_mode,
            "max_papers": paper_limit,
        }
        job_id = self.db_service.create_user_profile_build_job(user_id=resolved_user_id, build_config=config) if hasattr(self.db_service, "create_user_profile_build_job") else None
        job = self.db_service.get_user_profile_build_job(job_id) if job_id and hasattr(self.db_service, "get_user_profile_build_job") else None
        return job or {"job_id": job_id, "user_id": resolved_user_id, "status": "running", "current_stage": "collect_evidence", "progress": 0}

    def run_profile_rebuild_job(
        self,
        user_id: Optional[str],
        job_id: Optional[str] = None,
        build_mode: Optional[str] = "incremental",
        max_papers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行画像重建任务；传入 job_id 时复用已创建任务，避免前端轮询丢失任务标识。"""
        return self.generate_user_research_profile(
            user_id=user_id,
            existing_job_id=job_id,
            build_mode=build_mode,
            max_papers=max_papers,
        )

    def get_profile_build_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.db_service.get_user_profile_build_job(job_id) if hasattr(self.db_service, "get_user_profile_build_job") else None

    def list_profile_build_jobs(self, user_id: Optional[str], limit: int = 20) -> List[Dict[str, Any]]:
        resolved_user_id = self._resolve_user_id(user_id)
        return self.db_service.list_user_profile_build_jobs(resolved_user_id, limit=limit) if hasattr(self.db_service, "list_user_profile_build_jobs") else []

    def activate_profile_snapshot(self, user_id: Optional[str], snapshot_id: str) -> Dict[str, Any]:
        resolved_user_id = self._resolve_user_id(user_id)
        return self.db_service.activate_user_profile_snapshot(resolved_user_id, snapshot_id)

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
            # arXiv 分类在不同入口可能是逗号分隔或空格分隔字符串，入库前统一拆成独立分类。
            stripped = values.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return MemoryService._normalize_categories(parsed, limit=limit)
            source = [item.strip() for item in re.split(r"[,\s]+", values) if item.strip()]
        elif isinstance(values, list):
            source = [str(item or "").strip() for item in values]
        else:
            source = [str(values or "").strip()] if values is not None else []
        return [item for item in MemoryService._normalize_profile_list(source, limit=limit) if item]

    @staticmethod
    def _looks_like_arxiv_category(value: str) -> bool:
        """判断字符串是否是 arXiv 分类；分类只能进入 preferred_categories。"""
        text = str(value or "").strip()
        return bool(text and ARXIV_CATEGORY_PATTERN.match(text) and ("." in text or text.startswith("cs.")))

    @staticmethod
    def _normalize_preferred_categories(values: Any, limit: int = 20) -> List[str]:
        """清洗系统自动写入的分类偏好，只保留真正的 arXiv 分类值。"""
        categories: List[str] = []
        for item in MemoryService._normalize_categories(values, limit=limit * 2):
            if not MemoryService._looks_like_arxiv_category(item):
                continue
            if item not in categories:
                categories.append(item)
            if len(categories) >= limit:
                break
        return categories

    @staticmethod
    def _looks_like_arxiv_id(value: str) -> bool:
        """判断字符串是否是 arXiv ID，避免代表论文标识污染主题字段。"""
        text = str(value or "").strip()
        if not text:
            return False
        if text.startswith(("http://", "https://")):
            text = text.rstrip("/").rsplit("/", 1)[-1]
        return bool(ARXIV_ID_PATTERN.match(text))

    @staticmethod
    def _normalize_representative_papers(values: Any, limit: int = 20) -> List[str]:
        """规范化代表论文字段，自动写入时只保留可追溯的论文 ID 或短引用。"""
        normalized: List[str] = []
        for item in MemoryService._normalize_profile_list(values, limit=limit * 2):
            text = item.rstrip("/").rsplit("/", 1)[-1] if item.startswith(("http://", "https://")) else item
            if not (MemoryService._looks_like_arxiv_id(text) or (len(text) <= 80 and not URL_PATTERN.search(text))):
                continue
            if text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _looks_like_full_paper_title(value: str) -> bool:
        """用保守启发式识别完整论文标题，避免自动画像退化成标题列表。"""
        text = str(value or "").strip()
        words = [part for part in re.split(r"\s+", text) if part]
        if len(text) > 60 or len(words) > 6:
            return True
        lower_words = {word.strip(".,:;!?()[]{}").lower() for word in words}
        title_joiners = {"for", "with", "of", "using", "via", "towards", "toward", "based"}
        if len(words) >= 4 and lower_words & title_joiners:
            return True
        # 标题常见副标题和句式标点通常比短研究主题更复杂，自动写入时直接拦截。
        return any(marker in text for marker in (":", "?", "!", " -- ", " - "))

    @staticmethod
    def _is_low_information_topic(value: str) -> bool:
        """过滤单独出现时没有区分度的泛词，保留真正能表达兴趣边界的短语。"""
        text = str(value or "").strip().lower()
        return text in LOW_INFORMATION_TOPIC_TERMS

    @staticmethod
    def _normalize_system_topics(values: Any, limit: int = 30) -> List[str]:
        """清洗系统自动生成的主题信号，只留下短、可区分的研究兴趣短语。"""
        normalized: List[str] = []
        for item in MemoryService._normalize_profile_list(values, limit=limit * 3):
            text = str(item or "").strip()
            if not text:
                continue
            if URL_PATTERN.search(text) or MemoryService._looks_like_arxiv_id(text):
                continue
            if MemoryService._looks_like_arxiv_category(text) or text.lower().startswith("cs."):
                continue
            if MemoryService._looks_like_full_paper_title(text) or MemoryService._is_low_information_topic(text):
                continue
            if text not in normalized:
                normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _extract_candidate_topic_values(payload: Dict[str, Any]) -> List[str]:
        """只从显式主题类字段抽取候选 topic，不从标题、摘要这类原始论文文本推断。"""
        topic_values: List[str] = []
        for field_name in ("topics", "topic", "keywords", "keyword", "tags", "labels"):
            if field_name not in payload:
                continue
            topic_values.extend(MemoryService._normalize_profile_list(payload.get(field_name), limit=20))
        return topic_values

    def _resolve_paper_payload(self, arxiv_id: str, paper_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """优先使用显式传入的论文载荷，缺失时再从数据库读取论文信息。"""
        if isinstance(paper_payload, dict) and paper_payload:
            paper = dict(paper_payload)
            paper.setdefault("arxiv_id", arxiv_id)
            return paper
        paper = self.db_service.get_paper(arxiv_id)
        resolved = dict(paper or {})
        if arxiv_id and resolved:
            resolved.setdefault("arxiv_id", arxiv_id)
        return resolved

    @staticmethod
    def _dedupe_papers(papers: Any) -> List[Dict[str, Any]]:
        """按 arXiv ID 去重论文证据，避免同一动作重复放大主题权重。"""
        deduped: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in papers or []:
            if not isinstance(item, dict):
                continue
            arxiv_id = str(item.get("arxiv_id") or item.get("id") or "").strip()
            dedupe_key = arxiv_id or f"paper-{len(deduped)}"
            if dedupe_key in seen_ids:
                continue
            seen_ids.add(dedupe_key)
            deduped.append(item)
        return deduped

    def _load_paper_details_for_ids(self, arxiv_ids: Any) -> List[Dict[str, Any]]:
        """把行为表里的论文 ID 扩展为带 title/abstract/categories 的画像证据。"""
        papers: List[Dict[str, Any]] = []
        for arxiv_id in self._normalize_profile_list(arxiv_ids, limit=50):
            paper = self._resolve_paper_payload(arxiv_id)
            if paper:
                papers.append(paper)
        return self._dedupe_papers(papers)

    @staticmethod
    def _normalize_profile_build_mode(build_mode: Optional[str]) -> str:
        """统一画像构建模式，避免 API、调度器和测试传入大小写或未知值后分叉。"""
        mode = str(build_mode or "incremental").strip().lower()
        return mode if mode in PROFILE_BUILD_MODES else "incremental"

    @staticmethod
    def _resolve_profile_paper_limit(build_mode: str, max_papers: Optional[int] = None) -> int:
        """不同模式使用不同论文上限；普通增量构建必须有硬上限，避免一次点击触发超长任务。"""
        default_limit = PROFILE_BUILD_DEFAULT_PAPER_LIMITS.get(build_mode, PROFILE_BUILD_DEFAULT_PAPER_LIMITS["incremental"])
        try:
            limit = int(max_papers) if max_papers is not None else default_limit
        except (TypeError, ValueError):
            limit = default_limit
        return max(1, min(limit, 1000))

    @staticmethod
    def _is_strong_profile_event(event_type: str) -> bool:
        """强行为可以触发证据卡生成；read 只作为弱上下文，不单独拉起大量 LLM 调用。"""
        return str(event_type or "").strip().lower() in PROFILE_STRONG_EVENT_TYPES

    def _select_profile_build_events(
        self,
        user_id: str,
        *,
        build_mode: str,
        event_limit: int,
    ) -> Dict[str, Any]:
        """按构建模式选择事件范围，并返回跳过统计供 job metrics 和 snapshot 回看。"""
        if not hasattr(self.db_service, "list_user_profile_events"):
            return {"events": [], "total_events": 0, "used_events": 0, "skipped_events": 0, "dirty_event_count": 0}

        if build_mode == "full":
            fetched_events = self.db_service.list_user_profile_events(
                user_id=user_id,
                include_consumed=True,
                include_in_profile_only=True,
                limit=event_limit + 1,
            )
            events = fetched_events[:event_limit]
            return {
                "events": events,
                "total_events": len(fetched_events),
                "used_events": len(events),
                "skipped_events": max(0, len(fetched_events) - len(events)),
                "dirty_event_count": 0,
            }

        if build_mode == "repair":
            # 修复模式扫描有限历史事件来发现缺失/失败卡，但真正调用 LLM 的范围仍由 paper_limit 控制。
            fetched_events = self.db_service.list_user_profile_events(
                user_id=user_id,
                include_consumed=True,
                include_in_profile_only=True,
                limit=event_limit + 1,
            )
            events = fetched_events[:event_limit]
            return {
                "events": events,
                "total_events": len(fetched_events),
                "used_events": len(events),
                "skipped_events": max(0, len(fetched_events) - len(events)),
                "dirty_event_count": len([event for event in events if event.get("profile_dirty")]),
            }

        dirty_events = self.db_service.list_user_profile_events(
            user_id=user_id,
            include_consumed=True,
            include_in_profile_only=True,
            dirty_only=True,
            limit=event_limit,
        )
        # 增量模式优先 dirty 事件，并用少量最近强信号补上下文；不把大量历史 read 拉进本次构建。
        recent_context = self.db_service.list_user_profile_events(
            user_id=user_id,
            event_types=sorted(PROFILE_RECENT_CONTEXT_EVENT_TYPES),
            include_consumed=True,
            include_in_profile_only=True,
            limit=max(10, event_limit // 3),
        )
        selected: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for event in [*dirty_events, *recent_context]:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "")
            if event_id and event_id in seen_ids:
                continue
            if event_id:
                seen_ids.add(event_id)
            selected.append(event)
            if len(selected) >= event_limit:
                break
        skipped_events = max(0, len(dirty_events) + len(recent_context) - len(selected))
        return {
            "events": selected,
            "total_events": len(dirty_events) + len(recent_context),
            "used_events": len(selected),
            "skipped_events": skipped_events,
            "dirty_event_count": len(dirty_events),
        }

    def _collect_research_profile_evidence(
        self,
        user_id: str,
        *,
        build_mode: str = "incremental",
        max_papers: Optional[int] = None,
        extra_liked_papers: Optional[List[Dict[str, Any]]] = None,
        extra_disliked_papers: Optional[List[Dict[str, Any]]] = None,
        extra_recent_actions: Optional[List[Dict[str, Any]]] = None,
        extra_notes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """从画像事件流和当前强偏好表收集构建证据，避免已消费事件导致权威偏好丢失。"""
        build_mode = self._normalize_profile_build_mode(build_mode)
        paper_limit = self._resolve_profile_paper_limit(build_mode, max_papers)
        if build_mode == "full":
            event_limit = 10000
        elif build_mode == "repair":
            event_limit = max(400, paper_limit * 5)
        else:
            event_limit = max(80, paper_limit * 4)
        event_selection = self._select_profile_build_events(
            user_id,
            build_mode=build_mode,
            event_limit=event_limit,
        )
        events = event_selection["events"]

        liked_papers: List[Dict[str, Any]] = []
        disliked_papers: List[Dict[str, Any]] = []
        recent_actions: List[Dict[str, Any]] = []
        notes: List[Dict[str, Any]] = []
        event_ids: List[str] = []

        for event in events:
            if not isinstance(event, dict):
                continue
            event_ids.append(str(event.get("event_id") or ""))
            event_type = str(event.get("event_type") or event.get("action_type") or "").strip().lower()
            arxiv_id = str(event.get("arxiv_id") or "").strip()
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            paper_from_event = metadata.get("paper") if isinstance(metadata.get("paper"), dict) else {}
            paper = self._resolve_paper_payload(arxiv_id, paper_payload=paper_from_event) if arxiv_id else {}
            enriched_paper = {
                **paper,
                "_profile_event": event,
                "_profile_action_types": [event_type],
                "_profile_dirty": bool(event.get("profile_dirty")),
            } if paper else {}
            if event_type == "liked" and paper:
                liked_papers.append(enriched_paper)
                recent_actions.append({"arxiv_id": arxiv_id, "action_type": "liked", "paper": enriched_paper, "_profile_event": event})
                continue
            if event_type == "disliked" and paper:
                # 负向事件只作为负向画像候选，聚合器会保持谨慎权重，避免一次点踩否定整个大方向。
                disliked_papers.append(enriched_paper)
                recent_actions.append({"arxiv_id": arxiv_id, "action_type": "disliked", "paper": enriched_paper, "_profile_event": event})
                continue
            if event_type in {"favorite", "later", "read", "not_interested", "qa_asked"}:
                if paper:
                    recent_actions.append({"arxiv_id": arxiv_id, "action_type": event_type, "paper": enriched_paper, "_profile_event": event})
                continue
            if event_type == "note_saved":
                note_id = str(event.get("note_id") or event.get("source_id") or "").strip()
                note = self.db_service.get_paper_note(note_id, user_id=user_id) if note_id and hasattr(self.db_service, "get_paper_note") else None
                if note:
                    notes.append({**note, "_profile_event": event})
                elif paper:
                    recent_actions.append({"arxiv_id": arxiv_id, "action_type": "note_saved", "paper": enriched_paper, "_profile_event": event})

        # 兼容单元测试或显式调用传入的即时证据，但正式重建仍以事件流为主证据。
        liked_papers = self._dedupe_papers([*liked_papers, *(extra_liked_papers or [])])
        disliked_papers = self._dedupe_papers([*disliked_papers, *(extra_disliked_papers or [])])
        recent_actions.extend(extra_recent_actions or [])
        notes.extend(extra_notes or [])

        # liked/disliked 表是当前强偏好的权威状态；事件可能已被上次构建消费，但推荐和画像仍应看到当前偏好。
        liked_papers = self._dedupe_papers([*liked_papers, *self.db_service.get_liked_papers_with_details(user_id=user_id)])
        disliked_papers = self._dedupe_papers([*disliked_papers, *self._load_paper_details_for_ids(self.db_service.get_disliked_papers(user_id=user_id))])

        if not events:
            # 仅作为旧库兜底：没有任何事件时回退弱行为和笔记表，避免升级后空库导致画像突然清零。
            if hasattr(self.db_service, "list_user_profile_notes"):
                notes.extend(self.db_service.list_user_profile_notes(user_id=user_id))
            for action in self.db_service.get_user_paper_actions(user_id=user_id):
                if not isinstance(action, dict):
                    continue
                arxiv_id = str(action.get("arxiv_id") or "").strip()
                paper = self._resolve_paper_payload(arxiv_id) if arxiv_id else {}
                if not paper:
                    continue
                action_type = str(action.get("action_type") or "").strip().lower()
                recent_actions.append({**action, "paper": {**paper, "_profile_action_types": [action_type]}})
        return {
            "liked_papers": liked_papers,
            "disliked_papers": disliked_papers,
            "recent_actions": recent_actions,
            "notes": notes,
            "profile_events": events,
            "profile_event_ids": [event_id for event_id in event_ids if event_id],
            "build_mode": build_mode,
            "paper_limit": paper_limit,
            "event_selection": event_selection,
        }

    def _summarize_profile_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """压缩画像证据摘要，snapshot 只保存可解释统计和代表 ID，避免写入过大的原始载荷。"""
        def _paper_ids(items: Any, limit: int = 10) -> List[str]:
            ids: List[str] = []
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                arxiv_id = str(item.get("arxiv_id") or item.get("id") or "").strip()
                if arxiv_id and arxiv_id not in ids:
                    ids.append(arxiv_id)
                if len(ids) >= limit:
                    break
            return ids

        action_counts: Dict[str, int] = {}
        for action in evidence.get("recent_actions") or []:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("action_type") or "").strip().lower() or "unknown"
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        event_counts: Dict[str, int] = {}
        unconsumed_event_count = 0
        for event in evidence.get("profile_events") or []:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type") or "unknown").strip().lower() or "unknown"
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            if not event.get("consumed_by_job_id"):
                unconsumed_event_count += 1

        return {
            "liked_paper_count": len(evidence.get("liked_papers") or []),
            "disliked_paper_count": len(evidence.get("disliked_papers") or []),
            "recent_action_count": len(evidence.get("recent_actions") or []),
            "profile_note_count": len(evidence.get("notes") or []),
            "liked_paper_ids": _paper_ids(evidence.get("liked_papers")),
            "disliked_paper_ids": _paper_ids(evidence.get("disliked_papers")),
            "action_counts": action_counts,
            "profile_event_count": len(evidence.get("profile_events") or []),
            "unconsumed_profile_event_count": unconsumed_event_count,
            "profile_event_counts": event_counts,
        }

    def _build_profile_quality_report(self, profile: Dict[str, Any], evidence_summary: Dict[str, Any]) -> Dict[str, Any]:
        """记录重建质量的轻量报告，后续排查空画像或脏值过滤时能看到原因边界。"""
        return {
            "positive_topic_count": len(profile.get("positive_topics") or []),
            "negative_topic_count": len(profile.get("negative_topics") or []),
            "recent_topic_count": len(profile.get("recent_topics") or []),
            "preferred_category_count": len(profile.get("preferred_categories") or []),
            "representative_paper_count": len(profile.get("representative_papers") or []),
            "has_behavior_evidence": bool(
                evidence_summary.get("liked_paper_count")
                or evidence_summary.get("disliked_paper_count")
                or evidence_summary.get("recent_action_count")
                or evidence_summary.get("profile_note_count")
            ),
        }

    @staticmethod
    def _profile_build_progress(stage: str, processed_papers: int = 0, total_papers: int = 0) -> int:
        """按阶段映射粗粒度百分比，避免长时间停在 collect_evidence 造成前端误判卡死。"""
        stage_ranges = {
            "collect_evidence": (0, 10),
            "prepare_papers": (10, 15),
            "extract_paper_evidence": (15, 75),
            "aggregate_profile": (75, 85),
            "normalize_topics": (85, 88),
            "review_profile": (88, 92),
            "save_snapshot": (92, 99),
            "completed": (100, 100),
            "needs_review": (100, 100),
            "failed": (0, 100),
        }
        start, end = stage_ranges.get(stage, (0, 100))
        if stage != "extract_paper_evidence" or total_papers <= 0:
            return end
        ratio = max(0.0, min(1.0, processed_papers / max(total_papers, 1)))
        return int(start + (end - start) * ratio)

    @staticmethod
    def _append_profile_job_log(metrics: Dict[str, Any], message: str) -> None:
        """recent_logs 只保留最近节点，防止长任务把 job 行写成过大的诊断载荷。"""
        logs = metrics.get("recent_logs") if isinstance(metrics.get("recent_logs"), list) else []
        logs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "message": message})
        metrics["recent_logs"] = logs[-12:]

    def _update_profile_build_job(
        self,
        job_id: Optional[str],
        metrics: Dict[str, Any],
        *,
        status: Optional[str] = "running",
        stage: Optional[str] = None,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
        stage_message: Optional[str] = None,
    ) -> None:
        """统一写入 job 阶段和 metrics，保证数据库轮询字段与调试统计同步更新。"""
        if not job_id or not hasattr(self.db_service, "update_user_profile_build_job"):
            return
        if stage:
            metrics["current_stage"] = stage
        if progress is not None:
            metrics["progress"] = max(0, min(100, int(progress)))
        if stage_message:
            metrics["stage_message"] = stage_message
            self._append_profile_job_log(metrics, stage_message)
        if error_message:
            metrics["error_message"] = str(error_message)[:500]
        self.db_service.update_user_profile_build_job(
            job_id,
            status=status,
            current_stage=metrics.get("current_stage"),
            progress=metrics.get("progress"),
            error_message=error_message,
            metrics=metrics,
        )

    @staticmethod
    def _is_systemic_evidence_error(error_message: Any) -> bool:
        """识别更像服务整体不可用的错误；缺摘要等单篇数据问题只计入 failed_papers。"""
        text = str(error_message or "").lower()
        systemic_markers = (
            "generation_service_unavailable",
            "generation_service_has_no_supported_method",
            "dashscope",
            "timeout",
            "connection",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
        return any(marker in text for marker in systemic_markers)

    def _iter_profile_evidence_paper_refs(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """展开所有论文证据引用，便于把缓存或新生成的 card 回填到每个消费者会读取的位置。"""
        return [
            *(evidence.get("liked_papers") or []),
            *(evidence.get("disliked_papers") or []),
            *[
                action.get("paper")
                for action in evidence.get("recent_actions") or []
                if isinstance(action, dict) and isinstance(action.get("paper"), dict)
            ],
            *[
                self._resolve_paper_payload(str(note.get("arxiv_id") or "").strip())
                for note in evidence.get("notes") or []
                if isinstance(note, dict) and str(note.get("arxiv_id") or "").strip()
            ],
        ]

    def _collect_profile_evidence_papers(self, evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集本次构建实际涉及的唯一论文，后续缓存统计和逐篇进度都基于这个边界。"""
        buckets = self._iter_profile_evidence_paper_refs(evidence)
        unique_papers: List[Dict[str, Any]] = []
        index_by_id: Dict[str, int] = {}
        for paper in buckets:
            if not isinstance(paper, dict):
                continue
            arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
            if not arxiv_id:
                continue
            action_types = [
                str(item or "").strip().lower()
                for item in paper.get("_profile_action_types", [])
                if str(item or "").strip()
            ]
            if arxiv_id in index_by_id:
                # 同一论文可能同时出现在 liked/recent/note 引用中，合并行为类型后再判断是否允许触发 LLM。
                existing = unique_papers[index_by_id[arxiv_id]]
                merged_actions = set(existing.get("_profile_action_types") or [])
                merged_actions.update(action_types)
                existing["_profile_action_types"] = sorted(merged_actions)
                existing["_profile_dirty"] = bool(existing.get("_profile_dirty") or paper.get("_profile_dirty"))
                continue
            index_by_id[arxiv_id] = len(unique_papers)
            if action_types:
                paper["_profile_action_types"] = sorted(set(action_types))
            unique_papers.append(paper)
        return unique_papers

    def _paper_has_strong_profile_signal(self, paper: Dict[str, Any]) -> bool:
        """判断论文是否来自显式强行为；read-only 论文默认不触发未缓存 evidence card 生成。"""
        for action_type in paper.get("_profile_action_types") or []:
            if self._is_strong_profile_event(str(action_type)):
                return True
        event = paper.get("_profile_event") if isinstance(paper.get("_profile_event"), dict) else {}
        return self._is_strong_profile_event(str(event.get("event_type") or event.get("action_type") or ""))

    def _assign_profile_evidence_cards(self, evidence: Dict[str, Any], cards_by_id: Dict[str, Dict[str, Any]]) -> None:
        """把唯一论文生成出的 card 回填到所有证据引用，保证聚合器读取到一致的论文语义证据。"""
        for paper in self._iter_profile_evidence_paper_refs(evidence):
            if not isinstance(paper, dict):
                continue
            arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
            if arxiv_id in cards_by_id:
                paper["evidence_card"] = cards_by_id[arxiv_id]

    @staticmethod
    def _resolve_profile_evidence_concurrency() -> int:
        """解析 evidence 生成并发度；这里硬性封顶，避免误配置把外部 LLM 和本地 SQLite 同时打满。"""
        try:
            configured = int(PROFILE_EVIDENCE_CONFIG.get("max_workers") or 2)
        except (TypeError, ValueError):
            configured = 2
        return max(1, min(configured, 4))

    @staticmethod
    def _profile_evidence_backoff_seconds() -> int:
        """解析限流退避时间；负值或异常配置都回到保守默认值。"""
        try:
            configured = int(PROFILE_EVIDENCE_CONFIG.get("rate_limit_backoff_seconds") or 3)
        except (TypeError, ValueError):
            configured = 3
        return max(0, configured)

    @staticmethod
    def _is_rate_limited_evidence_card(card: Dict[str, Any]) -> bool:
        """识别供应商限流/繁忙类失败，用于降低后续补充任务速度而不是继续压请求。"""
        if card.get("schema_valid"):
            return False
        if str(card.get("error_type") or "").lower() != "transient":
            return False
        text = str(card.get("error_message") or "").lower()
        return any(marker in text for marker in ("429", "rate limit", "too many", "throttle", "busy", "temporarily"))

    def _extract_and_store_profile_evidence_card(
        self,
        paper: Dict[str, Any],
        *,
        can_persist: bool,
    ) -> Dict[str, Any]:
        """单篇 evidence card 的并发工作单元：抽取、失败降级、立即落库，避免批量结束后才保存。"""
        arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
        started_at = time.perf_counter()
        try:
            # evidence card 是画像概念的唯一默认入口；单篇失败会落错误卡并继续，避免一篇论文拖垮整次构建。
            card = self.paper_evidence_extractor.extract(paper)
        except Exception as exc:
            # 防御 extractor 外层异常：转换成错误卡后继续处理，系统性失败在所有 worker 完成后统一判定。
            logger.exception("Profile rebuild paper evidence crashed arxiv_id=%s", arxiv_id)
            card = self.paper_evidence_extractor._error_card(
                self.paper_evidence_extractor.normalize_paper(paper),
                str(exc),
                extraction_confidence=0.0,
            )
        if can_persist:
            # DatabaseService 每次 upsert 自行取连接；worker 不共享 sqlite connection，降低并发写冲突风险。
            self.db_service.upsert_paper_profile_evidence(arxiv_id, card)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return {
            "arxiv_id": arxiv_id,
            "paper": paper,
            "card": card,
            "elapsed_ms": elapsed_ms,
        }

    def _upsert_paper_evidence_cards(
        self,
        evidence: Dict[str, Any],
        *,
        job_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """为参与画像的论文维护 LLM evidence card，并把缓存命中和逐篇生成进度写入 job。"""
        metrics = metrics if metrics is not None else {}
        build_mode = self._normalize_profile_build_mode(evidence.get("build_mode") or metrics.get("build_mode"))
        paper_limit = self._resolve_profile_paper_limit(build_mode, evidence.get("paper_limit") or metrics.get("paper_limit"))
        unique_papers = self._collect_profile_evidence_papers(evidence)
        cards_by_id: Dict[str, Dict[str, Any]] = {}
        generation_candidates: List[Dict[str, Any]] = []
        can_persist = hasattr(self.db_service, "upsert_paper_profile_evidence")
        skipped_weak_papers = 0
        skipped_failed_cache_papers = 0
        repair_candidate_count = 0

        for paper in unique_papers:
            arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
            cached = (
                self.db_service.get_paper_profile_evidence(arxiv_id, extractor_version=PAPER_EVIDENCE_EXTRACTOR_VERSION)
                if can_persist and hasattr(self.db_service, "get_paper_profile_evidence")
                else None
            )
            cached_is_valid = bool(cached and cached.get("schema_valid"))
            if cached_is_valid:
                cards_by_id[arxiv_id] = cached
                paper["evidence_card"] = cached
                continue
            needs_repair = bool(cached and not cached.get("schema_valid"))
            if needs_repair:
                repair_candidate_count += 1
            if build_mode == "incremental" and needs_repair and not paper.get("_profile_dirty"):
                # 失败卡本身就是缓存；普通增量构建不反复重试历史失败，repair/full 才负责集中修复。
                cards_by_id[arxiv_id] = cached
                paper["evidence_card"] = cached
                skipped_failed_cache_papers += 1
                continue
            if build_mode == "incremental" and not self._paper_has_strong_profile_signal(paper):
                if cached:
                    cards_by_id[arxiv_id] = cached
                    paper["evidence_card"] = cached
                skipped_weak_papers += 1
                continue
            if build_mode == "repair" and not (not cached or needs_repair):
                continue
            generation_candidates.append(paper)

        uncached_papers = generation_candidates[:paper_limit]
        skipped_limit_papers = max(0, len(generation_candidates) - len(uncached_papers))
        total_papers = len(cards_by_id) + len(uncached_papers)
        cached_papers = len(cards_by_id)
        uncached_count = len(uncached_papers)
        evidence_concurrency = min(self._resolve_profile_evidence_concurrency(), max(uncached_count, 1))
        metrics.update(
            {
                "build_mode": build_mode,
                "paper_limit": paper_limit,
                "candidate_papers": len(unique_papers),
                "total_papers": total_papers,
                "cached_papers": cached_papers,
                "uncached_papers": uncached_count,
                "processed_papers": cached_papers,
                "failed_papers": 0,
                "successful_papers": cached_papers,
                "repair_candidate_papers": repair_candidate_count,
                "skipped_paper_count": skipped_weak_papers + skipped_limit_papers + skipped_failed_cache_papers,
                "skipped_read_only_papers": skipped_weak_papers,
                "skipped_failed_cache_papers": skipped_failed_cache_papers,
                "skipped_limit_papers": skipped_limit_papers,
                "cache_hit_count": cached_papers,
                "generated_count": 0,
                "failed_count": 0,
                "skipped_count": skipped_weak_papers + skipped_limit_papers + skipped_failed_cache_papers,
                "evidence_concurrency": evidence_concurrency if uncached_count else 0,
                "rate_limit_backoff_count": 0,
                "total_evidence_extraction_seconds": 0,
                "average_seconds_per_paper": 0,
                "current_arxiv_id": "",
            }
        )
        logger.info(
            "Profile rebuild prepared papers mode=%s candidate=%s cached=%s uncached=%s skipped=%s concurrency=%s job_id=%s",
            build_mode,
            len(unique_papers),
            cached_papers,
            uncached_count,
            metrics["skipped_paper_count"],
            evidence_concurrency if uncached_count else 0,
            job_id,
        )
        logger.info(
            "Profile rebuild paper limit mode=%s limit=%s skipped_read_only=%s skipped_limit=%s repair_candidates=%s",
            build_mode,
            paper_limit,
            skipped_weak_papers,
            skipped_limit_papers,
            repair_candidate_count,
        )
        self._update_profile_build_job(
            job_id,
            metrics,
            stage="prepare_papers",
            progress=15,
            stage_message=(
                f"准备论文证据：模式 {build_mode}，候选 {len(unique_papers)} 篇，"
                f"缓存 {cached_papers} 篇，待生成 {uncached_count} 篇，并发 {evidence_concurrency if uncached_count else 0}，"
                f"跳过 {metrics['skipped_paper_count']} 篇"
            ),
        )

        if total_papers == 0 or uncached_count == 0:
            metrics["processed_papers"] = total_papers
            self._assign_profile_evidence_cards(evidence, cards_by_id)
            self._update_profile_build_job(
                job_id,
                metrics,
                stage="extract_paper_evidence",
                progress=75,
                stage_message="论文证据卡已全部命中缓存或被限流跳过" if total_papers else "本次构建没有需要处理的论文证据",
            )
            return metrics

        failed_errors: List[str] = []
        elapsed_values_ms: List[int] = []
        generated_count = 0
        next_submit_index = 0
        completed_generated = 0
        backoff_seconds = self._profile_evidence_backoff_seconds()
        inflight: Dict[Any, Tuple[int, Dict[str, Any]]] = {}
        extraction_wall_started_at = time.perf_counter()

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            nonlocal next_submit_index
            if next_submit_index >= uncached_count:
                return False
            paper = uncached_papers[next_submit_index]
            paper_index = next_submit_index + 1
            arxiv_id = str(paper.get("arxiv_id") or paper.get("id") or "").strip()
            metrics["current_arxiv_id"] = arxiv_id
            self._update_profile_build_job(
                job_id,
                metrics,
                stage="extract_paper_evidence",
                progress=self._profile_build_progress("extract_paper_evidence", metrics["processed_papers"], total_papers),
                stage_message=(
                    f"正在生成论文证据卡：{metrics['processed_papers']} / {total_papers}，"
                    f"提交 {paper_index} / {uncached_count}，当前 {arxiv_id}"
                ),
            )
            logger.info(
                "Profile rebuild extracting paper evidence submit=%s/%s arxiv_id=%s mode=%s concurrency=%s job_id=%s",
                paper_index,
                uncached_count,
                arxiv_id,
                build_mode,
                evidence_concurrency,
                job_id,
            )
            future = executor.submit(self._extract_and_store_profile_evidence_card, paper, can_persist=can_persist)
            inflight[future] = (paper_index, paper)
            next_submit_index += 1
            return True

        with ThreadPoolExecutor(max_workers=evidence_concurrency, thread_name_prefix="profile-evidence") as executor:
            for _ in range(evidence_concurrency):
                submit_next(executor)

            while inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    paper_index, original_paper = inflight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        # worker 内部已尽量降级为错误卡；这里再兜底，保证 future 异常不会中断整批并发队列。
                        logger.exception("Profile rebuild paper evidence worker failed job_id=%s", job_id)
                        arxiv_id = str(original_paper.get("arxiv_id") or original_paper.get("id") or "").strip()
                        result = {
                            "arxiv_id": arxiv_id,
                            "paper": original_paper,
                            "card": self.paper_evidence_extractor._error_card(
                                self.paper_evidence_extractor.normalize_paper(original_paper),
                                str(exc),
                                extraction_confidence=0.0,
                            ),
                            "elapsed_ms": 0,
                        }

                    arxiv_id = str(result.get("arxiv_id") or "").strip()
                    paper = result.get("paper") if isinstance(result.get("paper"), dict) else original_paper
                    card = result.get("card") if isinstance(result.get("card"), dict) else {}
                    elapsed_ms = int(result.get("elapsed_ms") or 0)
                    elapsed_values_ms.append(elapsed_ms)
                    completed_generated += 1
                    metrics["current_arxiv_id"] = arxiv_id
                    logger.info(
                        "Profile rebuild paper evidence finished %s/%s arxiv_id=%s elapsed_ms=%s job_id=%s",
                        paper_index,
                        uncached_count,
                        arxiv_id,
                        elapsed_ms,
                        job_id,
                    )
                    cards_by_id[arxiv_id] = card
                    paper["evidence_card"] = card
                    if not card.get("schema_valid"):
                        error_text = str(card.get("error_message") or "paper_evidence_failed")
                        failed_errors.append(error_text)
                        logger.warning(
                            "Profile rebuild paper evidence failed arxiv_id=%s elapsed_ms=%s error=%s",
                            arxiv_id,
                            elapsed_ms,
                            error_text,
                        )
                    else:
                        logger.info(
                            "Profile rebuild paper evidence completed arxiv_id=%s elapsed_ms=%s",
                            arxiv_id,
                            elapsed_ms,
                        )
                        generated_count += 1
                    metrics["processed_papers"] = cached_papers + completed_generated
                    metrics["failed_papers"] = len(failed_errors)
                    metrics["successful_papers"] = cached_papers + generated_count
                    metrics["generated_count"] = generated_count
                    metrics["failed_count"] = len(failed_errors)
                    metrics["paper_evidence_failures"] = failed_errors[-10:]
                    metrics["paper_evidence_failure_details"] = (
                        metrics.get("paper_evidence_failure_details", [])
                        + [
                            {
                                "arxiv_id": arxiv_id,
                                "error_type": card.get("error_type") or "unknown",
                                "error_message": card.get("error_message") or "paper_evidence_failed",
                            }
                        ]
                    )[-10:] if not card.get("schema_valid") else metrics.get("paper_evidence_failure_details", [])
                    total_worker_seconds = round(sum(elapsed_values_ms) / 1000, 3)
                    metrics["total_evidence_extraction_seconds"] = round(time.perf_counter() - extraction_wall_started_at, 3)
                    metrics["average_seconds_per_paper"] = round(
                        total_worker_seconds / max(completed_generated, 1),
                        3,
                    )
                    self._update_profile_build_job(
                        job_id,
                        metrics,
                        stage="extract_paper_evidence",
                        progress=self._profile_build_progress(
                            "extract_paper_evidence",
                            metrics["processed_papers"],
                            total_papers,
                        ),
                        stage_message=(
                            f"已完成论文证据卡：{metrics['processed_papers']} / {total_papers}，"
                            f"缓存 {cached_papers}，生成 {generated_count}，失败 {metrics['failed_papers']} 篇"
                        ),
                    )
                    if self._is_rate_limited_evidence_card(card) and backoff_seconds > 0 and next_submit_index < uncached_count:
                        metrics["rate_limit_backoff_count"] = int(metrics.get("rate_limit_backoff_count") or 0) + 1
                        logger.warning(
                            "Profile rebuild evidence rate limited arxiv_id=%s backoff_seconds=%s job_id=%s",
                            arxiv_id,
                            backoff_seconds,
                            job_id,
                        )
                        time.sleep(backoff_seconds)
                    submit_next(executor)

        self._assign_profile_evidence_cards(evidence, cards_by_id)
        metrics["current_arxiv_id"] = ""
        logger.info(
            "Profile rebuild evidence summary cache_hit=%s generated=%s failed=%s skipped=%s avg_seconds=%s total_seconds=%s job_id=%s",
            metrics.get("cache_hit_count"),
            metrics.get("generated_count"),
            metrics.get("failed_count"),
            metrics.get("skipped_count"),
            metrics.get("average_seconds_per_paper"),
            metrics.get("total_evidence_extraction_seconds"),
            job_id,
        )
        metrics["systemic_evidence_failure"] = bool(
            uncached_count > 0
            and len(failed_errors) == uncached_count
            and any(self._is_systemic_evidence_error(error) for error in failed_errors)
        )
        return metrics

    def generate_user_research_profile(
        self,
        user_id: Optional[str],
        *,
        extra_liked_papers: Optional[List[Dict[str, Any]]] = None,
        extra_disliked_papers: Optional[List[Dict[str, Any]]] = None,
        extra_recent_actions: Optional[List[Dict[str, Any]]] = None,
        extra_notes: Optional[List[Dict[str, Any]]] = None,
        preserve_existing_topics: bool = True,
        preserve_existing_representative_papers: bool = True,
        existing_job_id: Optional[str] = None,
        build_mode: Optional[str] = "incremental",
        max_papers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """从用户行为证据重新归纳系统画像，并生成快照后返回 effective 兼容结构。"""
        resolved_user_id = self._resolve_user_id(user_id)
        resolved_build_mode = self._normalize_profile_build_mode(build_mode)
        resolved_paper_limit = self._resolve_profile_paper_limit(resolved_build_mode, max_papers)
        build_config = {
            "preserve_existing_topics": False,
            "preserve_existing_representative_papers": False,
            "legacy_output_projection": True,
            "build_mode": resolved_build_mode,
            "max_papers": resolved_paper_limit,
        }
        metrics: Dict[str, Any] = {
            "build_mode": resolved_build_mode,
            "paper_limit": resolved_paper_limit,
            "total_papers": 0,
            "cached_papers": 0,
            "uncached_papers": 0,
            "processed_papers": 0,
            "failed_papers": 0,
            "current_arxiv_id": "",
            "current_stage": "collect_evidence",
            "progress": 0,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "recent_logs": [],
        }
        job_id = existing_job_id
        if hasattr(self.db_service, "create_user_profile_build_job"):
            if job_id:
                # 异步 API 先创建 job 再后台执行，这里复用同一个 job_id，避免前端轮询看到两个任务。
                self._update_profile_build_job(
                    job_id,
                    metrics,
                    stage="collect_evidence",
                    progress=5,
                    stage_message="开始收集画像事件、偏好动作和笔记证据",
                )
            else:
                job_id = self.db_service.create_user_profile_build_job(user_id=resolved_user_id, build_config=build_config)
                self._update_profile_build_job(
                    job_id,
                    metrics,
                    stage="collect_evidence",
                    progress=5,
                    stage_message="开始收集画像事件、偏好动作和笔记证据",
                )
        try:
            logger.info("Profile rebuild started user_id=%s job_id=%s", resolved_user_id, job_id)
            evidence = self._collect_research_profile_evidence(
                resolved_user_id,
                build_mode=resolved_build_mode,
                max_papers=resolved_paper_limit,
                extra_liked_papers=extra_liked_papers,
                extra_disliked_papers=extra_disliked_papers,
                extra_recent_actions=extra_recent_actions,
                extra_notes=extra_notes,
            )
            metrics["evidence_counts"] = {
                "total_events": int((evidence.get("event_selection") or {}).get("total_events") or 0),
                "used_events": int((evidence.get("event_selection") or {}).get("used_events") or 0),
                "skipped_events": int((evidence.get("event_selection") or {}).get("skipped_events") or 0),
                "dirty_events": int((evidence.get("event_selection") or {}).get("dirty_event_count") or 0),
                "liked_papers": len(evidence.get("liked_papers") or []),
                "disliked_papers": len(evidence.get("disliked_papers") or []),
                "recent_actions": len(evidence.get("recent_actions") or []),
                "notes": len(evidence.get("notes") or []),
                "profile_events": len(evidence.get("profile_events") or []),
            }
            self._update_profile_build_job(
                job_id,
                metrics,
                stage="collect_evidence",
                progress=10,
                stage_message=(
                    f"已收集画像证据：模式 {resolved_build_mode}，使用事件 {metrics['evidence_counts']['used_events']} 条，"
                    f"跳过 {metrics['evidence_counts']['skipped_events']} 条，行为 {metrics['evidence_counts']['recent_actions']} 条"
                ),
            )
            metrics = self._upsert_paper_evidence_cards(evidence, job_id=job_id, metrics=metrics)
            if metrics.get("systemic_evidence_failure"):
                error_message = "paper_evidence_systemic_failure: LLM 证据卡生成疑似整体不可用"
                self._update_profile_build_job(
                    job_id,
                    metrics,
                    status="failed",
                    stage="failed",
                    progress=metrics.get("progress", 75),
                    error_message=error_message,
                    stage_message=error_message,
                )
                raise RuntimeError(error_message)

            self._update_profile_build_job(
                job_id,
                metrics,
                stage="aggregate_profile",
                progress=75,
                stage_message="开始聚合正向、负向和近期画像候选",
            )
            generated_draft = self.profile_aggregator.aggregate(
                evidence=evidence,
                current_profile={},
                # 自动画像只从事件和 evidence card 重建；旧画像不再作为系统层证据，避免脏 topic 被反复带回。
                preserve_existing_topics=False,
                preserve_existing_representative_papers=False,
            )
            self._update_profile_build_job(
                job_id,
                metrics,
                stage="normalize_topics",
                progress=88,
                stage_message="已完成 canonical topics 归一化和画像字段清洗",
            )
            self._update_profile_build_job(
                job_id,
                metrics,
                stage="review_profile",
                progress=90,
                stage_message="开始执行画像质量审查",
            )
            review_result = self.profile_reviewer.review(generated_draft)
            generated = review_result["revised_profile"]
            evidence_summary = self._summarize_profile_evidence(evidence)
            # snapshot 中保留最终统计，便于构建完成后回看缓存命中、失败论文和阶段日志。
            evidence_summary["build_metrics"] = {
                key: value
                for key, value in metrics.items()
                if key
                in {
                    "total_papers",
                    "candidate_papers",
                    "cached_papers",
                    "uncached_papers",
                    "processed_papers",
                    "failed_papers",
                    "successful_papers",
                    "skipped_paper_count",
                    "skipped_read_only_papers",
                    "skipped_failed_cache_papers",
                    "skipped_limit_papers",
                    "repair_candidate_papers",
                    "cache_hit_count",
                    "generated_count",
                    "failed_count",
                    "skipped_count",
                    "average_seconds_per_paper",
                    "total_evidence_extraction_seconds",
                    "evidence_concurrency",
                    "rate_limit_backoff_count",
                    "build_mode",
                    "paper_limit",
                    "paper_evidence_failures",
                    "paper_evidence_failure_details",
                    "evidence_counts",
                    "recent_logs",
                }
            }
            quality_report = {
                **self._build_profile_quality_report(generated, evidence_summary),
                **(review_result.get("quality_report") or {}),
            }
            if any(str(action.get("action_type") or "").strip().lower() == "read" for action in evidence.get("recent_actions") or []):
                issues = quality_report.setdefault("issues", [])
                if not any(issue.get("code") == "recent_topic_pollution" for issue in issues if isinstance(issue, dict)):
                    # read-only 行为会被保留为弱证据，但不应提升为 recent topic；这里显式记录解释性审查结果。
                    issues.append(
                        {
                            "code": "recent_topic_pollution",
                            "severity": "info",
                            "message": "普通阅读信号已被降权，未单独提升为 recent topic。",
                        }
                    )
            attempted_papers = int(metrics.get("uncached_papers") or 0)
            failed_papers = int(metrics.get("failed_papers") or 0)
            successful_papers = int(metrics.get("successful_papers") or 0)
            failure_ratio = round(failed_papers / max(attempted_papers, 1), 4) if attempted_papers else 0.0
            if failed_papers:
                quality_report["evidence_failure_ratio"] = failure_ratio
                quality_report["evidence_warning"] = "paper_evidence_partial_failure"
            if attempted_papers and failure_ratio > 0.5:
                quality_report["evidence_warning"] = "paper_evidence_high_failure_ratio"
            has_structured_profile_signal = any(
                generated.get(field)
                for field in (
                    "positive_topics",
                    "negative_topics",
                    "recent_topics",
                    "preferred_categories",
                    "common_question_types",
                    "representative_papers",
                )
            )
            no_reliable_generated_signal = successful_papers <= 0 and not has_structured_profile_signal and not (evidence.get("notes") or [])
            if no_reliable_generated_signal:
                # 没有任何有效 evidence card 或笔记信号时只保留 snapshot 供排查，不移动 active profile。
                quality_report["evidence_warning"] = "no_valid_paper_evidence"
                quality_report["approved"] = False
            activate_snapshot = bool(review_result.get("approved", True)) and not no_reliable_generated_signal
            self._update_profile_build_job(
                job_id,
                metrics,
                stage="save_snapshot",
                progress=92,
                stage_message="正在保存画像 snapshot 并刷新 effective profile",
            )
            if hasattr(self.db_service, "save_generated_profile_snapshot"):
                snapshot = self.db_service.save_generated_profile_snapshot(
                    resolved_user_id,
                    generated_profile=generated,
                    evidence_summary=evidence_summary,
                    quality_report=quality_report,
                    build_config=build_config,
                    job_id=job_id,
                    activate=activate_snapshot,
                )
                if job_id and hasattr(self.db_service, "mark_user_profile_events_consumed"):
                    # 事件消费标记只在 snapshot 成功落库后更新，避免失败构建把证据错误标成已处理。
                    self.db_service.mark_user_profile_events_consumed(
                        resolved_user_id,
                        job_id,
                        evidence.get("profile_event_ids") or [],
                    )
                final_status = "completed" if activate_snapshot else "needs_review"
                metrics["snapshot_id"] = snapshot.get("snapshot_id")
                metrics["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._update_profile_build_job(
                    job_id,
                    metrics,
                    status=final_status,
                    stage=final_status,
                    progress=100,
                    stage_message=f"画像构建完成：{final_status}",
                )
                logger.info(
                    "Profile rebuild finished user_id=%s job_id=%s status=%s total=%s cached=%s failed=%s",
                    resolved_user_id,
                    job_id,
                    final_status,
                    metrics.get("total_papers"),
                    metrics.get("cached_papers"),
                    metrics.get("failed_papers"),
                )
                effective_profile = dict(snapshot.get("effective_profile") or {})
                # 即使 snapshot 因证据不足未激活，也把质量报告带回给调用方解释本次构建结果。
                effective_profile["quality_report"] = quality_report
                effective_profile["snapshot_id"] = snapshot.get("snapshot_id")
                effective_profile["profile_layers"] = {
                    "generated_snapshot_id": snapshot.get("snapshot_id"),
                    "manual_profile_available": bool((snapshot.get("manual_profile") or {}).get("updated_at")),
                    "generated_profile_available": True,
                }
                return effective_profile
            return self.db_service.upsert_user_research_profile(user_id=resolved_user_id, profile=generated)
        except Exception as exc:
            logger.exception("Profile rebuild failed user_id=%s job_id=%s", resolved_user_id, job_id)
            self._update_profile_build_job(
                job_id,
                metrics,
                status="failed",
                stage="failed",
                progress=metrics.get("progress", 0),
                error_message=str(exc),
                stage_message=f"画像构建失败：{str(exc)[:200]}",
            )
            raise

    def rebuild_user_research_profile(
        self,
        user_id: Optional[str],
        *,
        build_mode: Optional[str] = "incremental",
        max_papers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """用现有行为记录重建系统画像；manual profile 保留，effective 通过快照重新合并。"""
        # 新模型下重建只更新 generated/snapshot，不再读取旧 positive_topics 作为主证据。
        return self.generate_user_research_profile(
            user_id,
            preserve_existing_topics=False,
            preserve_existing_representative_papers=False,
            build_mode=build_mode,
            max_papers=max_papers,
        )

    def _extract_paper_profile_signals(
        self,
        arxiv_id: str,
        paper_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """从论文元数据中提取分类、代表论文和可清洗的显式主题信号。"""
        paper = self._resolve_paper_payload(arxiv_id, paper_payload=paper_payload)
        categories = self._normalize_preferred_categories(paper.get("categories"), limit=12)
        # 论文标题和 arXiv 分类都不是抽象研究主题：分类进入专属字段，标题只作为论文元数据保留。
        positive_topics = self._normalize_system_topics(self._extract_candidate_topic_values(paper), limit=12)
        representative_papers = self._normalize_representative_papers([arxiv_id], limit=1)
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
        """按来源策略更新用户研究画像；手动来源只写 manual，系统来源触发 generated 重建。"""
        resolved_user_id = self._resolve_user_id(user_id)
        normalized_source = str(source or "unknown").strip().lower() or "unknown"
        current_manual = self.db_service.get_user_manual_profile(resolved_user_id) if hasattr(self.db_service, "get_user_manual_profile") else self.load_user_profile(resolved_user_id)
        incoming = dict(patch or {})

        if normalized_source in {"manual_upsert", "manual_put"}:
            # 显式全量覆盖只代表用户手动画像的新状态，不能覆盖系统生成层。
            if hasattr(self.db_service, "upsert_user_manual_profile"):
                return self.db_service.upsert_user_manual_profile(user_id=resolved_user_id, profile=incoming, source=normalized_source)
            return self.db_service.upsert_user_research_profile(user_id=resolved_user_id, profile=incoming)

        if normalized_source in {"manual", "api", "user"}:
            # PATCH 语义只修改 manual profile；effective profile 会在数据库层重新合并。
            if hasattr(self.db_service, "patch_user_manual_profile"):
                return self.db_service.patch_user_manual_profile(user_id=resolved_user_id, profile=incoming, source=normalized_source)
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
                incoming_values = incoming.get(field_name)
                if field_name in {"positive_topics", "negative_topics", "recent_topics"}:
                    # topic 字段只接收抽象短主题；分类、论文 ID、标题和 URL 在这里统一止血。
                    incoming_values = self._normalize_system_topics(incoming_values, limit=limit)
                elif field_name == "preferred_categories":
                    incoming_values = self._normalize_preferred_categories(incoming_values, limit=limit)
                elif field_name == "representative_papers":
                    incoming_values = self._normalize_representative_papers(incoming_values, limit=limit)
                merged_patch[field_name] = self._merge_profile_list(current_manual.get(field_name), incoming_values, limit=limit)

        if normalized_source in {"manual_answer_style", "manual_style"} and "preferred_answer_style" in incoming:
            merged_patch["preferred_answer_style"] = str(incoming.get("preferred_answer_style") or "").strip()

        if not merged_patch:
            return self.load_user_profile(resolved_user_id)
        if hasattr(self.db_service, "patch_user_manual_profile"):
            return self.db_service.patch_user_manual_profile(user_id=resolved_user_id, profile=merged_patch, source=normalized_source)
        return self.db_service.patch_user_research_profile(user_id=resolved_user_id, profile=merged_patch)

    def update_profile_from_note(self, user_id: Optional[str], note: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """根据用户保存并允许入画像的笔记内容更新长期研究画像。"""
        normalized_note = dict(note or {})
        if not normalized_note or not normalized_note.get("include_in_profile"):
            # 只有显式标记 include_in_profile 的笔记，才会参与长期画像学习。
            return self.load_user_profile(user_id)

        resolved_user_id = self._resolve_user_id(user_id)
        note_id = str(normalized_note.get("note_id") or normalized_note.get("id") or "").strip()
        arxiv_id = str(normalized_note.get("arxiv_id") or "").strip()
        if hasattr(self.db_service, "record_user_profile_event"):
            # 兼容旧调用方：方法名保留，但只追加事件，不在请求链路同步跑完整画像生成。
            self.db_service.record_user_profile_event(
                user_id=resolved_user_id,
                event_type="note_saved",
                source_type="paper_note",
                source_id=note_id or arxiv_id,
                action_type="note_saved",
                arxiv_id=arxiv_id,
                note_id=note_id or None,
                metadata=normalized_note,
                include_in_profile=True,
            )
        return self.load_user_profile(resolved_user_id)

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

        paper = self._resolve_paper_payload(arxiv_id, paper_payload=paper_payload)
        resolved_user_id = self._resolve_user_id(user_id)
        if hasattr(self.db_service, "record_user_profile_event"):
            # 偏好动作只落事件流；生成器在构建任务中统一决定权重和正负向归因。
            self.db_service.record_user_profile_event(
                user_id=resolved_user_id,
                event_type="liked" if normalized_action in {"like", "liked"} else "disliked",
                source_type="paper_action",
                source_id=arxiv_id,
                action_type=normalized_action,
                arxiv_id=arxiv_id,
                metadata={"paper": paper},
                include_in_profile=True,
            )
        return self.load_user_profile(resolved_user_id)

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
            # note_saved 本身不是显式偏好，但如果笔记允许入画像，就作为笔记证据参与重生成。
            note_like_payload = {
                "arxiv_id": arxiv_id,
                "note_type": metadata.get("note_type"),
                "title": metadata.get("title"),
                "content": metadata.get("content"),
                "tags": metadata.get("tags") or [],
                "include_in_profile": True,
            }
            return self.update_profile_from_note(user_id, note_like_payload)

        if normalized_action in {"favorite", "later", "read"}:
            resolved_user_id = self._resolve_user_id(user_id)
            if hasattr(self.db_service, "record_user_profile_event"):
                self.db_service.record_user_profile_event(
                    user_id=resolved_user_id,
                    event_type=normalized_action,
                    source_type="paper_action",
                    source_id=arxiv_id,
                    action_type=normalized_action,
                    arxiv_id=arxiv_id,
                    metadata=metadata or {},
                    include_in_profile=True,
                )
            return self.load_user_profile(resolved_user_id)

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
        # Agent 会话恢复必须以后端持久化状态为准；前端 context 只作为 UI/本轮参数补充，并把拒绝覆盖的字段写入 debug。
        merge_result = merge_backend_authoritative_context(
            backend_context=backend_memory,
            frontend_context=frontend_context,
        )
        merged_context = merge_result.merged_context

        payload = AgentSessionMemory(
            session_id=resolved_session_id,
            agent_session=agent_session,
            backend_memory=backend_memory,
            merged_context=merged_context,
            context_merge_debug=merge_result.debug,
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
