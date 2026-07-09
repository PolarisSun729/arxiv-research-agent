from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.compactor import RuleCompactor
from services.prompt_context.llm_compactor import LLMCompactor
from services.prompt_context.token_counter import TokenCounter


SECTION_ORDER = [
    "system_instruction",
    "user_memory",
    "session_summary",
    "recent_turns",
    "agent_state",
    "rag_evidence",
    "current_question",
]

SHARED_POOL_SECTION_PRIORITY = {
    "rag_evidence": 0,
    "session_summary": 1,
    "recent_turns": 2,
    "user_memory": 3,
    "agent_state": 4,
}


@dataclass
class PromptBudgetConfig:
    max_input_tokens: int = 64000
    safety_margin_tokens: int = 2048
    rag_target_ratio: float = 0.78
    section_targets: Dict[str, int] = field(default_factory=dict)


@dataclass
class PromptBudgetPlan:
    selected_blocks: List[PromptBlock]
    dropped_blocks: List[PromptBlock]
    compacted_blocks: List[PromptBlock]
    debug: Dict[str, Any]


class PromptBudgetPlanner:
    """用硬保护 + 软预算 + 预算回收选择最终 prompt blocks。"""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        compactor: RuleCompactor | None = None,
        llm_compactor: LLMCompactor | None = None,
        config: PromptBudgetConfig | None = None,
    ) -> None:
        self.token_counter = token_counter
        self.config = config or PromptBudgetConfig()
        self.compactor = compactor or RuleCompactor(token_counter=token_counter)
        self.llm_compactor = llm_compactor

    def plan(self, blocks: Iterable[PromptBlock]) -> PromptBudgetPlan:
        rule_prepared = [self._prepare_block(block) for block in blocks]
        if self.llm_compactor is not None:
            # LLM 压缩位于规则压缩之后，保证表格/数值等硬保护逻辑已经先执行并可被继续尊重。
            prepared, llm_compaction_debug = self.llm_compactor.compact_blocks(rule_prepared)
        else:
            prepared = rule_prepared
            llm_compaction_debug = {"enabled": False, "applied": False, "reason": "not_configured"}
        compacted_blocks = [
            block
            for block in prepared
            if block.evidence_origin in {"rule_compacted", "llm_compacted"}
        ]
        protected = [block for block in prepared if block.protected or block.section in {"system_instruction", "current_question"}]
        regular = [block for block in prepared if block not in protected]

        selected: List[PromptBlock] = []
        selected_ids: Set[str] = set()
        dropped: List[PromptBlock] = []
        used_tokens = 0

        for block in self._sort_blocks(protected):
            selected.append(block)
            selected_ids.add(block.block_id)
            used_tokens += block.token_count

        max_budget = max(1, int(self.config.max_input_tokens))
        safety_margin = max(0, int(self.config.safety_margin_tokens))
        usable_budget = max(0, max_budget - safety_margin)
        section_targets = self._section_targets(usable_budget=usable_budget, protected_tokens=used_tokens)
        section_used = {section: 0 for section in section_targets}

        # 第一轮只吃各 section 的软预算，避免非 RAG 上下文先把总池子占满。
        pending: List[PromptBlock] = []
        for section in SECTION_ORDER:
            section_blocks = [block for block in regular if block.section == section]
            target = section_targets.get(section, 0)
            for block in self._sort_blocks(section_blocks):
                if block.block_id in selected_ids:
                    continue
                if section_used.get(section, 0) + block.token_count <= target and used_tokens + block.token_count <= usable_budget:
                    selected.append(block)
                    selected_ids.add(block.block_id)
                    section_used[section] = section_used.get(section, 0) + block.token_count
                    used_tokens += block.token_count
                else:
                    pending.append(block)

        reclaimed_tokens = sum(max(0, section_targets.get(section, 0) - used) for section, used in section_used.items())

        # 第二轮把未用完的预算回收到共享池；共享池优先给 RAG，再给其他连续性上下文。
        for block in self._sort_blocks_for_shared_pool(pending):
            if block.block_id in selected_ids:
                continue
            if used_tokens + block.token_count <= usable_budget:
                selected.append(block)
                selected_ids.add(block.block_id)
                section_used[block.section] = section_used.get(block.section, 0) + block.token_count
                used_tokens += block.token_count
            elif block.droppable:
                dropped.append(block)
            else:
                # 不可丢块放不进预算时必须暴露诊断，不能静默吞掉关键上下文。
                dropped.append(
                    block.with_updates(
                        metadata={**dict(block.metadata or {}), "budget_error": "non_droppable_block_over_budget"}
                    )
                )

        selected = sorted(selected, key=lambda block: (SECTION_ORDER.index(block.section) if block.section in SECTION_ORDER else 99, block.original_rank))
        debug = {
            "total_input_budget_tokens": max_budget,
            "usable_input_budget_tokens": usable_budget,
            "safety_margin_tokens": safety_margin,
            "used_input_tokens": used_tokens,
            "protected_tokens": sum(block.token_count for block in protected),
            "section_targets": section_targets,
            "section_used": section_used,
            "reclaimed_tokens": reclaimed_tokens,
            "selected_blocks": [self._debug_block(block) for block in selected],
            "compacted_blocks": [self._debug_block(block) for block in compacted_blocks],
            "dropped_blocks": [self._debug_block(block) for block in dropped],
            "llm_compaction": llm_compaction_debug,
            **self.token_counter.debug(),
        }
        return PromptBudgetPlan(
            selected_blocks=selected,
            dropped_blocks=dropped,
            compacted_blocks=compacted_blocks,
            debug=debug,
        )

    def _prepare_block(self, block: PromptBlock) -> PromptBlock:
        compacted = self.compactor.compact(block)
        token_count = compacted.token_count or self.token_counter.count(compacted.text)
        original_token_count = compacted.original_token_count or self.token_counter.count(block.text)
        return compacted.with_updates(token_count=token_count, original_token_count=original_token_count)

    def _section_targets(self, *, usable_budget: int, protected_tokens: int) -> Dict[str, int]:
        available = max(0, usable_budget - protected_tokens)
        targets = {
            "rag_evidence": int(available * max(0.0, min(1.0, self.config.rag_target_ratio))),
            "session_summary": min(3000, int(available * 0.06)),
            "recent_turns": min(4000, int(available * 0.08)),
            "user_memory": min(2000, int(available * 0.04)),
            "agent_state": 0,
        }
        targets.update({key: max(0, int(value)) for key, value in self.config.section_targets.items()})
        return targets

    @staticmethod
    def _sort_blocks(blocks: Iterable[PromptBlock]) -> List[PromptBlock]:
        return sorted(
            list(blocks),
            key=lambda block: (
                block.priority,
                -float(block.score or 0.0),
                block.original_rank,
                block.block_id,
            ),
        )

    @staticmethod
    def _sort_blocks_for_shared_pool(blocks: Iterable[PromptBlock]) -> List[PromptBlock]:
        return sorted(
            list(blocks),
            key=lambda block: (
                block.priority,
                SHARED_POOL_SECTION_PRIORITY.get(block.section, 99),
                -float(block.score or 0.0),
                block.original_rank,
                block.block_id,
            ),
        )

    @staticmethod
    def _debug_block(block: PromptBlock) -> Dict[str, Any]:
        return {
            "block_id": block.block_id,
            "section": block.section,
            "block_type": block.block_type,
            "priority": block.priority,
            "score": block.score,
            "token_count": block.token_count,
            "original_token_count": block.original_token_count,
            "evidence_origin": block.evidence_origin,
            "source_id": block.source_id,
            "metadata": {
                key: value
                for key, value in dict(block.metadata or {}).items()
                if key
                in {
                    "compacted_by_rule",
                    "compaction_strategy",
                    "content_loss_level",
                    "budget_error",
                }
            },
        }
