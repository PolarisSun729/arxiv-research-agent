from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.budget_planner import SECTION_ORDER, PromptBudgetConfig, PromptBudgetPlanner
from services.prompt_context.compactor import RuleCompactionConfig, RuleCompactor
from services.prompt_context.llm_compactor import LLMCompactionConfig, LLMCompactor
from services.prompt_context.renderer import PromptRenderer
from services.prompt_context.token_counter import TokenCounter, build_token_counter
from utils.config import PROMPT_CONTEXT_CONFIG


logger = logging.getLogger(__name__)


# legacy（按字符预算）组装路径使用的各 section 字符上限；"total" 是整段 prompt 的总字符上限。
# 新的 block + token 预算路径不使用这里的值，而是由 PromptBudgetPlanner 按 token 分配。
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

# 用户画像中会被写进 prompt 的字段（preferred_answer_style 单独处理，因为请求参数可以覆盖它）。
USER_MEMORY_PROMPT_KEYS = ("positive_topics", "negative_topics", "recent_topics", "preferred_categories", "common_question_types")


@dataclass
class PromptSection:
    """legacy 组装路径中的单个 section 及其按字符裁剪的诊断信息。"""

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

    目前有两条组装路径：
    - block 路径（论文问答最终回答使用）：把各类上下文拆成 PromptBlock，经过规则压缩、
      可选 LLM 压缩、token 预算规划后渲染；
    - legacy 路径（问题改写、Agent 规划，以及 block 路径异常时的兜底）：按 section 做字符级截断。
    两条路径输出的 dict 都包含 ``text`` / ``sections`` / ``debug`` 三个键。
    """

    def __init__(
        self,
        budgets: Optional[Mapping[str, int]] = None,
        *,
        prompt_config: Optional[Mapping[str, Any]] = None,
        token_counter: Optional[TokenCounter] = None,
        llm_compaction_client: Any = None,
    ) -> None:
        """
        Args:
            budgets: 覆盖 legacy 路径的字符预算，非法值会被忽略。
            prompt_config: 覆盖全局 PROMPT_CONTEXT_CONFIG 中的项。
            token_counter: 外部注入的计数器；缺省时按配置构造。
            llm_compaction_client: LLM 压缩使用的客户端，形态见 LLMCompactor。
        """
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

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

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
        """组装论文问答最终回答阶段的 prompt 上下文。

        默认走 block + token 预算路径；该路径抛异常时记录日志并退回 legacy 字符预算路径，
        并在 debug 中标记 block_prompt_context_fallback=True，保证问答主链路不因 prompt 组装失败而中断。
        """
        system_instruction = (
            "You are a strict academic QA assistant. Answer only from RAG evidence in the RAG evidence section. "
            "Use user memory, session summary, and recent turns only to understand the question, preferences, and continuity. "
            "If RAG evidence is insufficient, say you cannot determine it from the provided evidence. "
            "When the question asks about table values, rankings, increases, decreases, or differences, prefer structured Table Evidence blocks "
            "over table summaries or previews, and cite the table row, column, value, unit, operation, and source_id. "
            "If the block is Table Evidence Candidates, use it as candidate evidence only; do not present a candidate calculation as certain unless the evidence disambiguates it."
        )
        block_mode_enabled = bool(self.prompt_config.get("enable_block_prompt_context", True))
        if block_mode_enabled:
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
        if block_mode_enabled:
            # 走到这里说明 block 路径已启用但失败了，标记为兜底以便排查。
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
        """组装“追问改写为独立检索问题”阶段的上下文（legacy 路径）。"""
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
        """组装 Agent 工具规划阶段的上下文（legacy 路径）。"""
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
        assembly_debug = assembly.get("debug") or {}
        return [
            {
                "source_id": "__prompt_context__",
                "text": assembled_text,
                "chunk_type": "prompt_context",
                "section_path": "Prompt Context Assembly",
                "metadata": {
                    "assembled_section_order": list(assembly_debug.get("section_order") or []),
                    "original_generation_search_result_count": len(list(context_pack.get("generation_search_results") or [])),
                    "table_evidence_count": assembly_debug.get("table_evidence_count", 0),
                    "table_cell_evidence_count": assembly_debug.get("table_cell_evidence_count", 0),
                    "table_numeric_calculation_used": assembly_debug.get("table_numeric_calculation_used", False),
                },
            }
        ]

    # ------------------------------------------------------------------
    # block + token 预算路径
    # ------------------------------------------------------------------

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
        """把问答上下文拆成 PromptBlock，经过压缩与预算规划后渲染为最终 prompt。

        priority 约定（越小越优先）：系统指令与当前问题为 0 且受保护；
        会话摘要 4、用户画像 5；历史对话最新一轮为 7，越早的轮次数值越大、越容易被挤掉；
        RAG 证据的 priority 由 context_pack_builder 决定。
        """
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
                # payload 供 RuleCompactor 做字段白名单压缩。
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
        # 空内容的 block（如没有用户画像）不进入预算规划，避免出现空 section。
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
        """构造 PromptBlock 的便捷方法，统一 strip 文本并复制 metadata。"""
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
        """每轮历史对话生成一个 block；问题、回答摘要和引用 source_id 同时放进 metadata 供压缩器重建文本。

        输入轮次按时间正序排列（最后一个是最新一轮）。turn_recency 表示距最新一轮的距离（0 为最新），
        priority = 7 + turn_recency：越新的轮次 priority 越小、越不容易被预算挤掉；
        RuleCompactor 也据此决定最近几轮保留全文。original_rank 保持时间正序，保证渲染顺序不变。
        """
        dialog_turns = self._iter_dialog_turns(turns)
        blocks: List[PromptBlock] = []
        for position, (index, turn) in enumerate(dialog_turns):
            recency = len(dialog_turns) - 1 - position
            source_ids = self._turn_source_ids(turn)
            parts = [
                f"turn_id: {turn.get('turn_id') or ''}",
                f"question: {turn.get('question') or ''}",
                f"answer_summary: {turn.get('answer_summary') or ''}",
                f"resolved_reference: {turn.get('resolved_reference')}" if turn.get("resolved_reference") else "",
                f"cited_source_ids: {', '.join(source_ids)}" if source_ids else "",
            ]
            blocks.append(
                PromptBlock(
                    block_id=f"recent_turn:{turn.get('turn_id') or index}",
                    section="recent_turns",
                    block_type="recent_turn",
                    source="paper_chat_messages.recent_turns",
                    text="\n".join(part for part in parts if part).strip(),
                    priority=7 + recency,
                    original_rank=index,
                    metadata={
                        "turn_id": turn.get("turn_id") or "",
                        "turn_recency": recency,
                        "question": turn.get("question") or "",
                        "answer_summary": turn.get("answer_summary") or "",
                        "cited_source_ids": source_ids,
                        "resolved_reference": turn.get("resolved_reference") or "",
                    },
                )
            )
        return blocks

    def _rag_prompt_blocks(self, context_pack: Mapping[str, Any]) -> List[PromptBlock]:
        """读取 context_pack 中预先生成的 RAG prompt_blocks；没有时把 text_context 包装成单个不可丢弃的 block。"""
        raw_blocks = context_pack.get("prompt_blocks") if isinstance(context_pack.get("prompt_blocks"), list) else []
        blocks = [PromptBlock.from_dict(dict(item)) for item in raw_blocks if isinstance(item, Mapping)]
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
        """从 prompt_config 读取 token 预算参数，包括按 section 覆盖的软预算。"""
        return PromptBudgetConfig(
            max_input_tokens=int(self.prompt_config.get("prompt_max_input_tokens", 64000)),
            safety_margin_tokens=int(self.prompt_config.get("prompt_safety_margin_tokens", 2048)),
            rag_target_ratio=float(self.prompt_config.get("rag_target_ratio", 0.78)),
            section_targets=self._section_targets_config(),
        )

    def _section_targets_config(self) -> Dict[str, int]:
        """读取 section_targets 覆盖配置，只接受可参与软预算的 section 与非负整数，非法项忽略。

        system_instruction / current_question 是硬保护内容，不受软预算约束，因此不接受覆盖。
        """
        raw = self.prompt_config.get("section_targets")
        if not isinstance(raw, Mapping):
            return {}
        allowed = set(SECTION_ORDER) - {"system_instruction", "current_question"}
        targets: Dict[str, int] = {}
        for section, value in raw.items():
            if section not in allowed:
                logger.warning("Ignoring section_targets override for unsupported section: %s", section)
                continue
            try:
                targets[section] = max(0, int(value))
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid section_targets value for %s: %r", section, value)
        return targets

    def _rule_compaction_config(self) -> RuleCompactionConfig:
        """从 prompt_config 读取规则压缩参数。"""
        return RuleCompactionConfig(
            enabled=bool(self.prompt_config.get("enable_rule_compaction", True)),
            rag_max_block_tokens=int(self.prompt_config.get("rag_max_block_tokens", 4000)),
            rag_sibling_context_target_tokens=int(self.prompt_config.get("rag_sibling_context_target_tokens", 350)),
            rag_section_context_target_tokens=int(self.prompt_config.get("rag_section_context_target_tokens", 250)),
            rag_fallback_preview_tokens=int(self.prompt_config.get("rag_fallback_preview_tokens", 160)),
            table_candidate_cell_limit=int(self.prompt_config.get("table_candidate_cell_limit", 8)),
            figure_preview_tokens=int(self.prompt_config.get("figure_preview_tokens", 220)),
            recent_turns_recent_full_count=int(self.prompt_config.get("recent_turns_recent_full_count", 2)),
            recent_turns_older_answer_tokens=int(self.prompt_config.get("recent_turns_older_answer_tokens", 120)),
            recent_turns_source_id_limit=int(self.prompt_config.get("recent_turns_source_id_limit", 6)),
        )

    def _llm_compaction_config(self) -> LLMCompactionConfig:
        """从 prompt_config 读取 LLM 压缩参数。"""
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
        """把 context_pack 中的表格证据统计并入 prompt debug。"""
        context_debug = context_pack.get("context_budget_debug")
        if not isinstance(context_debug, Mapping):
            context_debug = {}
        # prompt debug 只暴露结构化表格证据计数，不复制完整 cell，避免 debug 体积膨胀。
        assembly["debug"]["table_evidence_count"] = int(context_debug.get("table_evidence_count") or 0)
        assembly["debug"]["table_cell_evidence_count"] = int(context_debug.get("table_cell_evidence_count") or 0)
        assembly["debug"]["table_numeric_operations"] = list(context_debug.get("table_numeric_operations") or [])
        assembly["debug"]["table_numeric_calculation_used"] = bool(context_debug.get("table_numeric_calculation_used", False))

    def _memory_payload(self, user_memory_summary: Any, preferred_answer_style: str) -> Dict[str, Any]:
        """取出用户画像 dict（请求指定的回答风格优先），作为 user_memory block 的结构化 payload。"""
        result = dict(self._resolve_memory_profile(user_memory_summary))
        if preferred_answer_style:
            result["preferred_answer_style"] = preferred_answer_style
        return result

    # ------------------------------------------------------------------
    # legacy 字符预算路径
    # ------------------------------------------------------------------

    def _section(self, name: str, source: str, content: str, *, priority: int) -> PromptSection:
        """按该 section 的字符预算截断内容，生成 PromptSection（未配置预算的 section 使用 total）。"""
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
        """按 SECTION_ORDER 拼接 section，并受 total 字符预算约束。

        先到先得：排在前面的 section 先占预算，后面的 section 可能被截断或整体丢弃，
        丢弃的 section 仍保留在 debug 中并打 dropped_by_total_budget 标记。
        """
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
                "estimated_tokens": self.token_counter.count(text),
                "budget_by_section_chars": {
                    key: value for key, value in self.budgets.items() if key != "total"
                },
                "truncated_sections": [section.name for section in ordered if section.truncated],
                "sections": section_debug,
            },
        }

    @staticmethod
    def _truncate(value: str, limit: int) -> tuple[str, bool]:
        """按字符数截断（超出时以 "..." 结尾），返回（截断后文本, 是否发生截断）。"""
        text = re.sub(r"\s+\n", "\n", str(value or "")).strip()
        if limit <= 0:
            return "", bool(text)
        if len(text) <= limit:
            return text, False
        return text[: max(0, limit - 3)].rstrip() + "...", True

    def _section_debug(self, section: PromptSection) -> Dict[str, Any]:
        """生成 section 级诊断。"""
        payload = section.to_dict()
        payload["content_chars"] = len(section.content or "")
        payload["estimated_tokens"] = self.token_counter.count(section.content or "")
        # debug 只记录内容规模和来源，不回填完整 prompt 文本，避免诊断对象反向膨胀。
        payload.pop("content", None)
        return payload

    # ------------------------------------------------------------------
    # 两条路径共用的格式化工具
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_memory_profile(user_memory_summary: Any) -> Mapping[str, Any]:
        """兼容两种画像结构：``{"profile": {...}}`` 包装形式，或直接就是画像 dict。"""
        payload = user_memory_summary if isinstance(user_memory_summary, Mapping) else {}
        profile = payload.get("profile")
        return profile if isinstance(profile, Mapping) else payload

    def _format_user_memory(self, user_memory_summary: Any, preferred_answer_style: str) -> str:
        """把用户画像格式化为 "key: value" 多行文本；请求指定的回答风格优先于画像中保存的风格。"""
        profile = self._resolve_memory_profile(user_memory_summary)
        parts = []
        for key in USER_MEMORY_PROMPT_KEYS:
            value = profile.get(key)
            if value:
                parts.append(f"{key}: {self._format_value(value)}")
        style = str(preferred_answer_style or profile.get("preferred_answer_style") or "").strip()
        if style:
            parts.append(f"preferred_answer_style: {style}")
        return "\n".join(parts)

    def _format_session_summary(self, summary: Optional[Mapping[str, Any]]) -> str:
        """把会话摘要格式化为带标题的 JSON 文本。"""
        if not isinstance(summary, Mapping):
            return ""
        return self._format_mapping("Session summary", summary)

    def _format_recent_turns(self, turns: Optional[Iterable[Mapping[str, Any]]]) -> str:
        """legacy 路径：把历史对话压成每轮一行 "Turn i id=..: Q=.. | A=.. | sources=..."。"""
        source_id_limit = int(self.prompt_config.get("recent_turns_source_id_limit", 6))
        lines = []
        for index, turn in self._iter_dialog_turns(turns):
            source_ids = self._turn_source_ids(turn)[:source_id_limit]
            source_suffix = f" | sources={', '.join(source_ids)}" if source_ids else ""
            lines.append(
                f"Turn {index} id={turn.get('turn_id') or ''}: "
                f"Q={turn.get('question') or ''} | A={turn.get('answer_summary') or ''}{source_suffix}"
            )
        return "\n".join(lines)

    @staticmethod
    def _iter_dialog_turns(turns: Optional[Iterable[Mapping[str, Any]]]) -> List[tuple[int, Mapping[str, Any]]]:
        """过滤出真实的问答轮次（跳过非 dict 项和摘要伪轮次），返回 (从 1 开始的原始序号, turn)。

        序号沿用原列表位置，因此被过滤掉的项会让序号出现空洞，这与历史行为保持一致。
        """
        result: List[tuple[int, Mapping[str, Any]]] = []
        for index, turn in enumerate(list(turns or []), start=1):
            if not isinstance(turn, Mapping):
                continue
            # 会话摘要已经作为独立 section 进入 prompt，这里跳过以免重复。
            if turn.get("is_summary") or turn.get("context_type") == "session_summary":
                continue
            result.append((index, turn))
        return result

    @staticmethod
    def _turn_source_ids(turn: Mapping[str, Any]) -> List[str]:
        """提取一轮对话引用过的非空 source_id 列表。"""
        return [
            str(source.get("source_id", "") or "").strip()
            for source in list(turn.get("sources") or [])
            if isinstance(source, Mapping) and str(source.get("source_id", "") or "").strip()
        ]

    @staticmethod
    def _format_current_question(*, question: str, contextualized_question: str = "") -> str:
        """输出原始问题；改写后的问题与原问题不同才额外附上。"""
        parts = [f"original_question: {str(question or '').strip()}"]
        contextualized = str(contextualized_question or "").strip()
        if contextualized and contextualized != str(question or "").strip():
            parts.append(f"contextualized_question: {contextualized}")
        return "\n".join(parts)

    def _format_mapping(self, title: str, value: Mapping[str, Any]) -> str:
        """去掉空值后序列化为 "title:\\n{json}"；全部为空时返回空串。"""
        payload = {key: item for key, item in dict(value or {}).items() if item not in (None, "", [], {})}
        if not payload:
            return ""
        return f"{title}:\n{json.dumps(payload, ensure_ascii=False, default=str)}"

    @staticmethod
    def _format_value(value: Any) -> str:
        """列表类型用逗号连接，dict 序列化为 JSON，其余转字符串。"""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value if str(item).strip())
        if isinstance(value, Mapping):
            return json.dumps(dict(value), ensure_ascii=False, default=str)
        return str(value)
