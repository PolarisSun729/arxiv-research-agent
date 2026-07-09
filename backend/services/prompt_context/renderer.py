from __future__ import annotations

from typing import Any, Dict, Iterable, List

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.budget_planner import SECTION_ORDER, PromptBudgetPlan
from services.prompt_context.token_counter import TokenCounter


class PromptRenderer:
    """把预算选择后的 blocks 渲染成单个最终 prompt，上游不再重复追加原始 chunks。"""

    def __init__(self, *, token_counter: TokenCounter) -> None:
        self.token_counter = token_counter

    def render(self, plan: PromptBudgetPlan) -> Dict[str, Any]:
        grouped = self._group_by_section(plan.selected_blocks)
        rendered_parts: List[str] = []
        for section in SECTION_ORDER:
            blocks = grouped.get(section) or []
            if not blocks:
                continue
            content = "\n\n".join(block.text.strip() for block in blocks if block.text.strip()).strip()
            if content:
                rendered_parts.append(f"## {section}\n{content}")
        text = "\n\n".join(rendered_parts).strip()
        debug = {
            **plan.debug,
            "prompt_context_mode": "block_token_budget",
            "rendered_section_order": [section for section in SECTION_ORDER if grouped.get(section)],
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
        grouped: Dict[str, List[PromptBlock]] = {}
        for block in blocks:
            grouped.setdefault(block.section, []).append(block)
        return grouped

    @staticmethod
    def _section_debug(grouped: Dict[str, List[PromptBlock]]) -> List[Dict[str, Any]]:
        return [
            {
                "name": section,
                "block_count": len(blocks),
                "content_tokens": sum(block.token_count for block in blocks),
                "truncated": any(block.evidence_origin != "raw" for block in blocks),
            }
            for section, blocks in grouped.items()
        ]
