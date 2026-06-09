from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


DEFAULT_BUDGETS = {
    "total": 28000,
    "system_instruction": 1800,
    "user_memory": 1800,
    "session_summary": 2200,
    "recent_turns": 4200,
    "agent_state": 2200,
    "rag_evidence": 18000,
    "current_question": 1400,
}

TOKEN_ESTIMATE_CHARS = 4

SECTION_ORDER = [
    "system_instruction",
    "user_memory",
    "session_summary",
    "recent_turns",
    "agent_state",
    "rag_evidence",
    "current_question",
]


@dataclass
class PromptSection:
    """统一描述进入模型上下文的单个 section。"""

    name: str
    source: str
    content: str
    priority: int
    budget_chars: int
    original_chars: int = 0
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PromptContextBuilder:
    """集中编排 Prompt 上下文顺序、预算和 debug。

    业务服务只负责加载记忆、检索证据或维护 Agent 状态；真正进入模型的顺序、
    裁剪和诊断统一由本类输出，避免各入口继续散落拼接 prompt。
    """

    def __init__(self, budgets: Optional[Mapping[str, int]] = None) -> None:
        merged = dict(DEFAULT_BUDGETS)
        for key, value in dict(budgets or {}).items():
            try:
                merged[key] = max(0, int(value))
            except (TypeError, ValueError):
                continue
        self.budgets = merged

    def build_paper_qa_final_answer_context(
        self,
        *,
        question: str,
        contextualized_question: str,
        context_pack: Mapping[str, Any],
        session_summary: Optional[Mapping[str, Any]] = None,
        recent_turns: Optional[Iterable[Mapping[str, Any]]] = None,
        user_memory_summary: Optional[Mapping[str, Any]] = None,
        preferred_answer_style: str = "",
    ) -> Dict[str, Any]:
        sections = [
            self._section(
                "system_instruction",
                "paper_qa_policy",
                (
                    "You are a strict academic QA assistant. Answer only from RAG evidence in the RAG evidence section. "
                    "Use user memory, session summary, and recent turns only to understand the question, preferences, and continuity. "
                    "If RAG evidence is insufficient, say you cannot determine it from the provided evidence."
                ),
                priority=1,
            ),
            self._section("user_memory", "memory_service", self._format_user_memory(user_memory_summary, preferred_answer_style), priority=2),
            self._section("session_summary", "paper_chat_sessions.summary", self._format_session_summary(session_summary), priority=3),
            self._section("recent_turns", "paper_chat_messages.recent_turns", self._format_recent_turns(recent_turns), priority=4),
            self._section("rag_evidence", "context_pack_builder", str(context_pack.get("text_context") or ""), priority=5),
            self._section(
                "current_question",
                "request",
                self._format_current_question(question=question, contextualized_question=contextualized_question),
                priority=6,
            ),
        ]
        return self._assemble(sections)

    def build_question_contextualization_context(
        self,
        *,
        question: str,
        paper_context: Mapping[str, Any],
        session_summary: Optional[Mapping[str, Any]] = None,
        recent_turns: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        sections = [
            self._section(
                "system_instruction",
                "question_contextualization_policy",
                (
                    "Rewrite follow-up questions into self-contained retrieval questions. "
                    "Session summary is compressed memory and may preserve earlier preferences or task focus; it is not paper evidence."
                ),
                priority=1,
            ),
            self._section("session_summary", "paper_chat_sessions.summary", self._format_session_summary(session_summary), priority=2),
            self._section("recent_turns", "paper_chat_messages.recent_turns", self._format_recent_turns(recent_turns), priority=3),
            self._section("current_question", "request", self._format_current_question(question=question), priority=4),
            self._section("agent_state", "paper_context", self._format_mapping("Paper context", paper_context), priority=5),
        ]
        return self._assemble(sections)

    def build_agent_planner_context(
        self,
        *,
        user_request: str,
        user_memory_summary: Any = None,
        research_profile: Any = None,
        agent_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        sections = [
            self._section("system_instruction", "agent_planner_policy", "Plan tool usage. Do not execute tools in the prompt.", priority=1),
            self._section("user_memory", "agent_memory", self._format_user_memory(user_memory_summary, ""), priority=2),
            self._section("agent_state", "agent_state", self._format_mapping("Agent state", {"research_profile": research_profile, **dict(agent_state or {})}), priority=3),
            self._section("current_question", "request", self._format_current_question(question=user_request), priority=4),
        ]
        return self._assemble(sections)

    def generation_search_results_from_assembly(
        self,
        assembly: Mapping[str, Any],
        context_pack: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """适配现有 GenerationService.generate(search_results=...) 接口。

        assembled context 已经包含 RAG evidence。这里不能再追加原始 chunk，
        否则同一证据会进入 prompt 两次，破坏统一预算和 section 顺序。
        """
        assembled_text = str(assembly.get("text") or "").strip()
        return [
            {
                "source_id": "__prompt_context__",
                "text": assembled_text,
                "chunk_type": "prompt_context",
                "section_path": "Prompt Context Assembly",
                "metadata": {
                    "assembled_section_order": list((assembly.get("debug") or {}).get("section_order") or []),
                    "original_generation_search_result_count": len(list(context_pack.get("generation_search_results") or [])),
                },
            }
        ]

    def _section(self, name: str, source: str, content: str, *, priority: int) -> PromptSection:
        original = str(content or "").strip()
        budget = int(self.budgets.get(name, self.budgets["total"]))
        trimmed, truncated = self._truncate(original, budget)
        return PromptSection(
            name=name,
            source=source,
            content=trimmed,
            priority=priority,
            budget_chars=budget,
            original_chars=len(original),
            truncated=truncated,
        )

    def _assemble(self, sections: List[PromptSection]) -> Dict[str, Any]:
        ordered = [section for section in sorted(sections, key=lambda item: SECTION_ORDER.index(item.name) if item.name in SECTION_ORDER else 99) if section.content]
        total_budget = int(self.budgets.get("total", DEFAULT_BUDGETS["total"]))
        rendered_parts: List[str] = []
        used_chars = 0
        for section in ordered:
            block = f"## {section.name}\n{section.content}".strip()
            remaining = max(0, total_budget - used_chars)
            if remaining <= 0:
                # 总预算耗尽时保留 section 诊断，但不把内容继续塞进 prompt。
                section.truncated = True
                section.content = ""
                section.metadata["dropped_by_total_budget"] = True
                continue
            block, total_truncated = self._truncate(block, remaining)
            section.truncated = bool(section.truncated or total_truncated)
            if total_truncated:
                # total budget 裁剪发生在渲染块级别，需要同步 section 内容，避免 debug 与实际 prompt 不一致。
                section.metadata["truncated_by_total_budget"] = True
                section.content = block.split("\n", 1)[1] if "\n" in block else ""
            used_chars += len(block)
            rendered_parts.append(block)
        text = "\n\n".join(rendered_parts)
        section_debug = [self._section_debug(section) for section in ordered]
        return {
            "text": text,
            "sections": [section.to_dict() for section in ordered],
            "debug": {
                "section_order": [section.name for section in ordered],
                "rendered_section_order": [
                    section.name for section in ordered if not section.metadata.get("dropped_by_total_budget")
                ],
                "total_budget_chars": total_budget,
                "used_chars": len(text),
                "estimated_tokens": self._estimate_tokens(text),
                "budget_by_section_chars": {
                    key: value for key, value in self.budgets.items() if key != "total"
                },
                "truncated_sections": [section.name for section in ordered if section.truncated],
                "sections": section_debug,
            },
        }

    @staticmethod
    def _truncate(value: str, limit: int) -> tuple[str, bool]:
        text = re.sub(r"\s+\n", "\n", str(value or "")).strip()
        if limit <= 0:
            return "", bool(text)
        if len(text) <= limit:
            return text, False
        return text[: max(0, limit - 3)].rstrip() + "...", True

    def _format_user_memory(self, user_memory_summary: Any, preferred_answer_style: str) -> str:
        payload = user_memory_summary if isinstance(user_memory_summary, Mapping) else {}
        profile = payload.get("profile") if isinstance(payload.get("profile"), Mapping) else payload
        parts = []
        for key in ("positive_topics", "negative_topics", "recent_topics", "preferred_categories", "common_question_types"):
            value = profile.get(key) if isinstance(profile, Mapping) else None
            if value:
                parts.append(f"{key}: {self._format_value(value)}")
        style = str(preferred_answer_style or (profile.get("preferred_answer_style") if isinstance(profile, Mapping) else "") or "").strip()
        if style:
            parts.append(f"preferred_answer_style: {style}")
        return "\n".join(parts)

    def _format_session_summary(self, summary: Optional[Mapping[str, Any]]) -> str:
        if not isinstance(summary, Mapping):
            return ""
        return self._format_mapping("Session summary", summary)

    def _format_recent_turns(self, turns: Optional[Iterable[Mapping[str, Any]]]) -> str:
        lines = []
        for index, turn in enumerate(list(turns or []), start=1):
            if not isinstance(turn, Mapping):
                continue
            if turn.get("is_summary") or turn.get("context_type") == "session_summary":
                continue
            source_ids = [
                str(source.get("source_id", "") or "").strip()
                for source in list(turn.get("sources") or [])
                if isinstance(source, Mapping) and str(source.get("source_id", "") or "").strip()
            ][:6]
            source_suffix = f" | sources={', '.join(source_ids)}" if source_ids else ""
            lines.append(
                f"Turn {index} id={turn.get('turn_id') or ''}: "
                f"Q={turn.get('question') or ''} | A={turn.get('answer_summary') or ''}{source_suffix}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_current_question(*, question: str, contextualized_question: str = "") -> str:
        parts = [f"original_question: {str(question or '').strip()}"]
        contextualized = str(contextualized_question or "").strip()
        if contextualized and contextualized != str(question or "").strip():
            parts.append(f"contextualized_question: {contextualized}")
        return "\n".join(parts)

    def _format_mapping(self, title: str, value: Mapping[str, Any]) -> str:
        payload = {key: item for key, item in dict(value or {}).items() if item not in (None, "", [], {})}
        if not payload:
            return ""
        return f"{title}:\n{json.dumps(payload, ensure_ascii=False, default=str)}"

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if str(item).strip())
        if isinstance(value, Mapping):
            return json.dumps(dict(value), ensure_ascii=False, default=str)
        return str(value)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text) / TOKEN_ESTIMATE_CHARS))

    def _section_debug(self, section: PromptSection) -> Dict[str, Any]:
        payload = section.to_dict()
        payload["content_chars"] = len(section.content or "")
        payload["estimated_tokens"] = self._estimate_tokens(section.content or "")
        # debug 只记录内容规模和来源，不回填完整 prompt 文本，避免诊断对象反向膨胀。
        payload.pop("content", None)
        return payload
