from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.budget_planner import PromptBudgetConfig, PromptBudgetPlanner
from services.prompt_context.compactor import RuleCompactionConfig, RuleCompactor
from services.prompt_context.llm_compactor import LLMCompactionConfig, LLMCompactor
from services.prompt_context.renderer import PromptRenderer
from services.prompt_context.token_counter import TokenCounter, build_token_counter
from utils.config import PROMPT_CONTEXT_CONFIG


logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        budgets: Optional[Mapping[str, int]] = None,
        *,
        prompt_config: Optional[Mapping[str, Any]] = None,
        token_counter: Optional[TokenCounter] = None,
        llm_compaction_client: Any = None,
    ) -> None:
        merged = dict(DEFAULT_BUDGETS)
        for key, value in dict(budgets or {}).items():
            try:
                merged[key] = max(0, int(value))
            except (TypeError, ValueError):
                continue
        self.budgets = merged
        self.prompt_config = {**dict(PROMPT_CONTEXT_CONFIG), **dict(prompt_config or {})}
        self.token_counter = token_counter or build_token_counter(self.prompt_config)
        self.llm_compaction_client = llm_compaction_client

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
        system_instruction = (
            "You are a strict academic QA assistant. Answer only from RAG evidence in the RAG evidence section. "
            "Use user memory, session summary, and recent turns only to understand the question, preferences, and continuity. "
            "If RAG evidence is insufficient, say you cannot determine it from the provided evidence. "
            "When the question asks about table values, rankings, increases, decreases, or differences, prefer structured Table Evidence blocks "
            "over table summaries or previews, and cite the table row, column, value, unit, operation, and source_id. "
            "If the block is Table Evidence Candidates, use it as candidate evidence only; do not present a candidate calculation as certain unless the evidence disambiguates it."
        )
        if self.prompt_config.get("enable_block_prompt_context", True):
            try:
                assembly = self._build_paper_qa_block_context(
                    system_instruction=system_instruction,
                    question=question,
                    contextualized_question=contextualized_question,
                    context_pack=context_pack,
                    session_summary=session_summary,
                    recent_turns=recent_turns,
                    user_memory_summary=user_memory_summary,
                    preferred_answer_style=preferred_answer_style,
                )
                self._attach_context_pack_debug(assembly, context_pack)
                return assembly
            except Exception as exc:  # pragma: no cover - 兜底只用于本地排查异常链路
                logger.exception("Block prompt context failed, fallback to legacy assembly: %s", exc)
        sections = [
            self._section(
                "system_instruction",
                "paper_qa_policy",
                system_instruction,
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
        assembly = self._assemble(sections)
        assembly["debug"]["prompt_context_mode"] = "legacy"
        if self.prompt_config.get("enable_block_prompt_context", True):
            assembly["debug"]["block_prompt_context_fallback"] = True
        self._attach_context_pack_debug(assembly, context_pack)
        return assembly

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
                    "table_evidence_count": (assembly.get("debug") or {}).get("table_evidence_count", 0),
                    "table_cell_evidence_count": (assembly.get("debug") or {}).get("table_cell_evidence_count", 0),
                    "table_numeric_calculation_used": (assembly.get("debug") or {}).get("table_numeric_calculation_used", False),
                },
            }
        ]

    def _build_paper_qa_block_context(
        self,
        *,
        system_instruction: str,
        question: str,
        contextualized_question: str,
        context_pack: Mapping[str, Any],
        session_summary: Optional[Mapping[str, Any]],
        recent_turns: Optional[Iterable[Mapping[str, Any]]],
        user_memory_summary: Optional[Mapping[str, Any]],
        preferred_answer_style: str,
    ) -> Dict[str, Any]:
        blocks = [
            self._prompt_block(
                block_id="system_instruction",
                section="system_instruction",
                block_type="system_instruction",
                source="paper_qa_policy",
                text=system_instruction,
                protected=True,
                droppable=False,
                compactable=False,
                priority=0,
            ),
            self._prompt_block(
                block_id="current_question",
                section="current_question",
                block_type="current_question",
                source="request",
                text=self._format_current_question(question=question, contextualized_question=contextualized_question),
                protected=True,
                droppable=False,
                compactable=False,
                priority=0,
            ),
            self._prompt_block(
                block_id="user_memory",
                section="user_memory",
                block_type="user_memory",
                source="memory_service",
                text=self._format_user_memory(user_memory_summary, preferred_answer_style),
                priority=5,
                metadata={"payload": self._memory_payload(user_memory_summary, preferred_answer_style)},
            ),
            self._prompt_block(
                block_id="session_summary",
                section="session_summary",
                block_type="session_summary",
                source="paper_chat_sessions.summary",
                text=self._format_session_summary(session_summary),
                priority=4,
                metadata={"payload": dict(session_summary or {})},
            ),
            *self._recent_turn_prompt_blocks(recent_turns),
            *self._rag_prompt_blocks(context_pack),
        ]
        blocks = [block for block in blocks if str(block.text or "").strip()]
        compactor = RuleCompactor(
            token_counter=self.token_counter,
            config=self._rule_compaction_config(),
        )
        planner = PromptBudgetPlanner(
            token_counter=self.token_counter,
            compactor=compactor,
            llm_compactor=LLMCompactor(
                token_counter=self.token_counter,
                config=self._llm_compaction_config(),
                client=self.llm_compaction_client,
            ),
            config=self._prompt_budget_config(),
        )
        plan = planner.plan(blocks)
        assembly = PromptRenderer(token_counter=self.token_counter).render(plan)
        assembly["debug"]["rule_compaction_enabled"] = bool(self.prompt_config.get("enable_rule_compaction", True))
        assembly["debug"]["llm_compaction_enabled"] = bool(self.prompt_config.get("enable_llm_compaction", False))
        return assembly

    def _prompt_block(
        self,
        *,
        block_id: str,
        section: str,
        block_type: str,
        source: str,
        text: str,
        priority: int,
        metadata: Optional[Dict[str, Any]] = None,
        protected: bool = False,
        droppable: bool = True,
        compactable: bool = True,
    ) -> PromptBlock:
        return PromptBlock(
            block_id=block_id,
            section=section,
            block_type=block_type,
            source=source,
            text=str(text or "").strip(),
            priority=priority,
            protected=protected,
            droppable=droppable,
            compactable=compactable,
            metadata=dict(metadata or {}),
        )

    def _recent_turn_prompt_blocks(self, turns: Optional[Iterable[Mapping[str, Any]]]) -> List[PromptBlock]:
        blocks: List[PromptBlock] = []
        for index, turn in enumerate(list(turns or []), start=1):
            if not isinstance(turn, Mapping):
                continue
            if turn.get("is_summary") or turn.get("context_type") == "session_summary":
                continue
            source_ids = [
                str(source.get("source_id", "") or "").strip()
                for source in list(turn.get("sources") or [])
                if isinstance(source, Mapping) and str(source.get("source_id", "") or "").strip()
            ]
            text = (
                f"turn_id: {turn.get('turn_id') or ''}\n"
                f"question: {turn.get('question') or ''}\n"
                f"answer_summary: {turn.get('answer_summary') or ''}"
            ).strip()
            blocks.append(
                PromptBlock(
                    block_id=f"recent_turn:{turn.get('turn_id') or index}",
                    section="recent_turns",
                    block_type="recent_turn",
                    source="paper_chat_messages.recent_turns",
                    text=text,
                    priority=6 + index,
                    original_rank=index,
                    metadata={
                        "turn_id": turn.get("turn_id") or "",
                        "question": turn.get("question") or "",
                        "answer_summary": turn.get("answer_summary") or "",
                        "cited_source_ids": source_ids,
                        "resolved_reference": turn.get("resolved_reference") or "",
                    },
                )
            )
        return blocks

    def _rag_prompt_blocks(self, context_pack: Mapping[str, Any]) -> List[PromptBlock]:
        raw_blocks = context_pack.get("prompt_blocks") if isinstance(context_pack.get("prompt_blocks"), list) else []
        blocks: List[PromptBlock] = []
        for item in raw_blocks:
            if isinstance(item, Mapping):
                blocks.append(PromptBlock.from_dict(dict(item)))
        if blocks:
            return blocks
        text_context = str(context_pack.get("text_context") or "").strip()
        if not text_context:
            return []
        # 旧 context_pack 没有 prompt_blocks 时退回单块 RAG，保证调用方迁移期间不丢证据。
        return [
            PromptBlock(
                block_id="rag:legacy_text_context",
                section="rag_evidence",
                block_type="anchor_evidence",
                source="context_pack_builder",
                text=text_context,
                priority=0,
                context_role="anchor_evidence",
                source_id="legacy_text_context",
                original_source_id="legacy_text_context",
                droppable=False,
            )
        ]

    def _prompt_budget_config(self) -> PromptBudgetConfig:
        return PromptBudgetConfig(
            max_input_tokens=int(self.prompt_config.get("prompt_max_input_tokens", 64000)),
            safety_margin_tokens=int(self.prompt_config.get("prompt_safety_margin_tokens", 2048)),
            rag_target_ratio=float(self.prompt_config.get("rag_target_ratio", 0.78)),
        )

    def _rule_compaction_config(self) -> RuleCompactionConfig:
        return RuleCompactionConfig(
            enabled=bool(self.prompt_config.get("enable_rule_compaction", True)),
            rag_max_block_tokens=int(self.prompt_config.get("rag_max_block_tokens", 4000)),
            rag_sibling_context_target_tokens=int(self.prompt_config.get("rag_sibling_context_target_tokens", 350)),
            rag_section_context_target_tokens=int(self.prompt_config.get("rag_section_context_target_tokens", 250)),
            rag_fallback_preview_tokens=int(self.prompt_config.get("rag_fallback_preview_tokens", 160)),
            table_candidate_cell_limit=int(self.prompt_config.get("table_candidate_cell_limit", 8)),
            figure_preview_tokens=int(self.prompt_config.get("figure_preview_tokens", 220)),
            recent_turns_recent_full_count=int(self.prompt_config.get("recent_turns_recent_full_count", 2)),
            recent_turns_source_id_limit=int(self.prompt_config.get("recent_turns_source_id_limit", 6)),
        )

    def _llm_compaction_config(self) -> LLMCompactionConfig:
        return LLMCompactionConfig(
            enabled=bool(self.prompt_config.get("enable_llm_compaction", False)),
            min_block_tokens=int(self.prompt_config.get("llm_compaction_min_block_tokens", 1200)),
            target_block_tokens=int(self.prompt_config.get("llm_compaction_target_block_tokens", 700)),
            max_blocks=int(self.prompt_config.get("llm_compaction_max_blocks", 8)),
            model_name=str(self.prompt_config.get("llm_compaction_model_name", "") or ""),
            task_type=str(self.prompt_config.get("llm_compaction_task_type", "prompt_context_compaction") or "prompt_context_compaction"),
            enable_thinking=bool(self.prompt_config.get("llm_compaction_enable_thinking", False)),
        )

    def _attach_context_pack_debug(self, assembly: Dict[str, Any], context_pack: Mapping[str, Any]) -> None:
        context_debug = context_pack.get("context_budget_debug") if isinstance(context_pack.get("context_budget_debug"), Mapping) else {}
        if isinstance(context_debug, Mapping):
            # prompt debug 只暴露结构化表格证据计数，不复制完整 cell，避免 debug 体积膨胀。
            assembly["debug"]["table_evidence_count"] = int(context_debug.get("table_evidence_count") or 0)
            assembly["debug"]["table_cell_evidence_count"] = int(context_debug.get("table_cell_evidence_count") or 0)
            assembly["debug"]["table_numeric_operations"] = list(context_debug.get("table_numeric_operations") or [])
            assembly["debug"]["table_numeric_calculation_used"] = bool(context_debug.get("table_numeric_calculation_used", False))

    def _memory_payload(self, user_memory_summary: Any, preferred_answer_style: str) -> Dict[str, Any]:
        payload = user_memory_summary if isinstance(user_memory_summary, Mapping) else {}
        profile = payload.get("profile") if isinstance(payload.get("profile"), Mapping) else payload
        result = dict(profile or {}) if isinstance(profile, Mapping) else {}
        if preferred_answer_style:
            result["preferred_answer_style"] = preferred_answer_style
        return result

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
