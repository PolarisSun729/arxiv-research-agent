from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.compactor import RuleCompactor
from services.prompt_context.llm_compactor import LLMCompactor
from services.prompt_context.token_counter import TokenCounter


# prompt 中各 section 的固定渲染顺序；builder 的 legacy 路径与 renderer 都复用这里的定义。
# 不在此列表中的 section 无法被渲染，planner 会把它们作为 unknown_section 丢弃并暴露诊断。
SECTION_ORDER = [
    "system_instruction",
    "user_memory",
    "session_summary",
    "recent_turns",
    "agent_state",
    "rag_evidence",
    "current_question",
]

# 第二轮“共享池”回收时各 section 的抢占优先级（越小越优先）：RAG 证据最重要，其次是对话连续性。
SHARED_POOL_SECTION_PRIORITY = {
    "rag_evidence": 0,
    "session_summary": 1,
    "recent_turns": 2,
    "user_memory": 3,
    "agent_state": 4,
}

# 无论 block 是否标记 protected，这两个 section 都视为硬保护：没有指令或问题，prompt 就失去意义。
ALWAYS_PROTECTED_SECTIONS = {"system_instruction", "current_question"}


@dataclass
class PromptBudgetConfig:
    """预算规划参数。

    Attributes:
        max_input_tokens: 模型输入 token 上限。
        safety_margin_tokens: 预留给模板标题、分隔符以及 token 估算误差的安全余量。
        rag_target_ratio: 扣除硬保护内容后，RAG 证据在第一轮软预算中的占比。
        section_targets: 按 section 覆盖默认软预算（单位 token），优先级高于按比例计算的结果。
    """

    max_input_tokens: int = 245760
    safety_margin_tokens: int = 2048
    rag_target_ratio: float = 0.78
    section_targets: Dict[str, int] = field(default_factory=dict)


@dataclass
class PromptBudgetPlan:
    """预算规划结果。

    Attributes:
        selected_blocks: 最终入选的 block，已按 SECTION_ORDER + original_rank 排好序。
        dropped_blocks: 因预算不足或 section 未知而被丢弃的 block。
        compacted_blocks: 经过规则或 LLM 压缩的 block（无论最终是否入选）。
        debug: 预算分配、各 section 用量、压缩与丢弃明细等诊断信息。
    """

    selected_blocks: List[PromptBlock]
    dropped_blocks: List[PromptBlock]
    compacted_blocks: List[PromptBlock]
    debug: Dict[str, Any]


class PromptBudgetPlanner:
    """用硬保护 + 软预算 + 预算回收选择最终 prompt blocks。

    规划分四步：
    1. 预处理：每个 block 先走规则压缩并补齐 token 数；规则压缩后总量仍超出可用预算时，
       才可选地调用 LLM 压缩，且只压缩到补上缺口为止；
    2. 硬保护：protected block 与 system_instruction / current_question 无条件入选；
    3. 软预算（第一轮）：每个 section 只能在自己的 target 内按优先级选 block，
       防止记忆、历史对话等非证据内容先把总预算吃光；
    4. 共享池（第二轮）：第一轮没放进去的 block 按“priority → section 抢占优先级 → score”
       排序，只要总预算还有余量就继续放入，各 section 未用完的软预算由此被回收利用。
    """

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
        """对输入 blocks 执行压缩与预算选择，返回入选/丢弃/压缩结果及诊断。"""
        rule_prepared = [self._prepare_block(block) for block in blocks]
        max_budget = max(1, int(self.config.max_input_tokens))
        safety_margin = max(0, int(self.config.safety_margin_tokens))
        usable_budget = max(0, max_budget - safety_margin)
        if self.llm_compactor is not None:
            # LLM 压缩位于规则压缩之后，保证表格/数值等硬保护逻辑已经先执行并可被继续尊重。
            # 只有规则压缩后仍放不下时才需要 LLM 压缩，并且只压缩到补上缺口为止。
            # 全部 block 总量不超过可用预算时，两轮选择一定能全部放入，因此缺口即“总量 - 可用预算”。
            rule_total_tokens = sum(block.token_count for block in rule_prepared if block.section in SECTION_ORDER)
            prepared, llm_compaction_debug = self.llm_compactor.compact_blocks(
                rule_prepared,
                required_savings=rule_total_tokens - usable_budget,
            )
        else:
            prepared = rule_prepared
            llm_compaction_debug = {"enabled": False, "applied": False, "reason": "not_configured"}
        compacted_blocks = [
            block
            for block in prepared
            if block.evidence_origin in {"rule_compacted", "llm_compacted"}
        ]

        selected: List[PromptBlock] = []
        selected_ids: Set[str] = set()
        dropped: List[PromptBlock] = []
        used_tokens = 0

        # 按谓词一次性拆分，而不是用 `block not in protected`：后者依赖 dataclass 的逐字段相等比较，
        # 既是 O(n²)，又会把内容完全相同的两个 block 误判为同一个。
        protected: List[PromptBlock] = []
        regular: List[PromptBlock] = []
        for block in prepared:
            if block.section not in SECTION_ORDER:
                # renderer 只渲染 SECTION_ORDER 内的 section；未知 section 即使入选也不会出现在 prompt 中，
                # 这里显式丢弃并打标，避免“计入了预算却没渲染”的静默不一致。
                dropped.append(
                    block.with_updates(metadata={**dict(block.metadata or {}), "budget_error": "unknown_section"})
                )
            elif block.protected or block.section in ALWAYS_PROTECTED_SECTIONS:
                protected.append(block)
            else:
                regular.append(block)

        for block in self._sort_blocks(protected):
            selected.append(block)
            selected_ids.add(block.block_id)
            used_tokens += block.token_count
        protected_tokens = used_tokens

        section_targets = self._section_targets(usable_budget=usable_budget, protected_tokens=protected_tokens)
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

        # 第一轮结束时各 section 软预算的剩余量之和，即第二轮可被其他 section 回收的额度（仅用于诊断）。
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

        # 渲染顺序：先按 section 固定顺序，再按上游原始顺序，保证证据在 prompt 中的相对位置稳定。
        selected = sorted(selected, key=lambda block: (SECTION_ORDER.index(block.section), block.original_rank))
        debug = {
            "total_input_budget_tokens": max_budget,
            "usable_input_budget_tokens": usable_budget,
            "safety_margin_tokens": safety_margin,
            "used_input_tokens": used_tokens,
            "protected_tokens": protected_tokens,
            # 硬保护内容本身就超出可用预算时，后续 section 全部拿不到额度；单独标出便于排查超长问题。
            "protected_over_budget": protected_tokens > usable_budget,
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
        """规则压缩并确保 token_count / original_token_count 都已填好。"""
        compacted = self.compactor.compact(block)
        token_count = compacted.token_count or self.token_counter.count(compacted.text)
        original_token_count = compacted.original_token_count or self.token_counter.count(block.text)
        return compacted.with_updates(token_count=token_count, original_token_count=original_token_count)

    def _section_targets(self, *, usable_budget: int, protected_tokens: int) -> Dict[str, int]:
        """计算第一轮各 section 的软预算。

        以“可用预算 - 硬保护占用”为基数：RAG 按比例分配，记忆/摘要/历史对话按比例且设有上限，
        agent_state 默认 0（只能从第二轮共享池里拿额度）。config.section_targets 可以逐项覆盖。
        """
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
        """section 内排序：priority 升序 → score 降序 → 原始顺序 → block_id（保证结果确定）。"""
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
        """共享池排序：在 _sort_blocks 的基础上，同 priority 时按 section 抢占优先级决定先后。"""
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
        """block 的精简诊断视图：不带正文，只保留与预算和压缩相关的少量 metadata。"""
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
