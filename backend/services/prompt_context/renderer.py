from __future__ import annotations

from typing import Any, Dict, Iterable, List

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.budget_planner import SECTION_ORDER, PromptBudgetPlan
from services.prompt_context.token_counter import TokenCounter


class PromptRenderer:
    """把预算选择后的 blocks 渲染成单个最终 prompt，上游不再重复追加原始 chunks。

    渲染格式为按 SECTION_ORDER 排列的 Markdown 二级标题::

        ## system_instruction
        <block 文本>

        ## rag_evidence
        <block 1 文本>

        <block 2 文本>

    section 内 block 的先后沿用 planner 输出的顺序（即 original_rank）。
    """

    def __init__(self, *, token_counter: TokenCounter) -> None:
        self.token_counter = token_counter

    def render(self, plan: PromptBudgetPlan) -> Dict[str, Any]:
        """渲染最终 prompt，并在 planner 的 debug 基础上补充渲染阶段信息。

        Returns:
            ``text``: 最终 prompt 文本；
            ``blocks``: 入选 block 的完整序列化结果；
            ``sections``: 每个 section 的 block 数、token 数和是否经过压缩；
            ``debug``: planner debug + 渲染模式、实际渲染的 section 顺序与 token 数。
        """
        grouped = self._group_by_section(plan.selected_blocks)
        rendered_parts: List[str] = []
        rendered_sections: List[str] = []
        for section in SECTION_ORDER:
            blocks = grouped.get(section) or []
            content = "\n\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()
            if content:
                rendered_parts.append(f"## {section}\n{content}")
                rendered_sections.append(section)
        text = "\n\n".join(rendered_parts).strip()
        debug = {
            **plan.debug,
            "prompt_context_mode": "block_token_budget",
            # 只记录真正产出了内容的 section，避免“全是空文本的 section”出现在顺序里误导排查。
            "rendered_section_order": rendered_sections,
            # 渲染后再整体计一次 token：标题和分隔符也会占 token，因此会略大于 used_input_tokens。
            "rendered_tokens": self.token_counter.count(text),
        }
        return {
            "text": text,
            "blocks": [block.to_dict() for block in plan.selected_blocks],
            "sections": self._section_debug(grouped),
            "debug": debug,
        }

    @staticmethod
    def _group_by_section(blocks: Iterable[PromptBlock]) -> Dict[str, List[PromptBlock]]:
        """按 section 分组，保持组内原有顺序。"""
        grouped: Dict[str, List[PromptBlock]] = {}
        for block in blocks:
            grouped.setdefault(block.section, []).append(block)
        return grouped

    @staticmethod
    def _section_debug(grouped: Dict[str, List[PromptBlock]]) -> List[Dict[str, Any]]:
        """生成 section 级诊断；truncated 表示该 section 内至少有一个 block 被规则或 LLM 改写过。"""
        return [
            {
                "name": section,
                "block_count": len(blocks),
                "content_tokens": sum(block.token_count for block in blocks),
                "truncated": any(block.evidence_origin != "raw" for block in blocks),
            }
            for section, blocks in grouped.items()
        ]
