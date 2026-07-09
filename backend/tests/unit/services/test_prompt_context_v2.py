from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from services.paper_qa.context_pack_builder import ContextPackBuilder
from services.prompt_context import (
    LLMCompactionConfig,
    LLMCompactor,
    PromptBlock,
    PromptBudgetConfig,
    PromptBudgetPlanner,
    RuleCompactor,
    TokenCounter,
    build_token_counter,
)


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return [item for item in str(text or "").split() if item]


def test_token_counter_uses_exact_adapter_and_fallback_debug() -> None:
    exact = TokenCounter(tokenizer=FakeTokenizer(), tokenizer_name="fake")
    assert exact.count("alpha beta gamma") == 3
    assert exact.debug()["token_counter_mode"] == "exact"
    assert exact.debug()["tokenizer_name"] == "fake"

    fallback = TokenCounter(fallback_chars_per_token=2.0)
    assert fallback.count("abcdef") == 3
    assert fallback.debug()["token_counter_mode"] == "estimated"
    assert fallback.debug()["fallback_reason"]


def test_build_token_counter_uses_configured_fallback_reasons(monkeypatch) -> None:
    missing_path = build_token_counter({"token_counter_provider": "auto", "tokenizer_name_or_path": ""})
    assert missing_path.debug()["token_counter_mode"] == "estimated"
    assert missing_path.debug()["fallback_reason"] == "tokenizer_name_or_path_not_configured"

    unsupported = build_token_counter({"token_counter_provider": "custom"})
    assert unsupported.debug()["token_counter_mode"] == "estimated"
    assert unsupported.debug()["fallback_reason"] == "unsupported_token_counter_provider: custom"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name_or_path: str, local_files_only: bool):
            assert name_or_path == "local-tokenizer"
            assert local_files_only is True
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))
    exact = build_token_counter(
        {
            "token_counter_provider": "transformers",
            "tokenizer_name_or_path": "local-tokenizer",
        }
    )

    assert exact.count("alpha beta") == 2
    assert exact.debug()["token_counter_mode"] == "exact"
    assert exact.debug()["tokenizer_name"] == "local-tokenizer"


def test_context_pack_builder_emits_prompt_blocks_without_changing_text_context() -> None:
    context_pack = ContextPackBuilder().build(
        [
            {
                "chunk_id": "chunk-1",
                "content": "Evidence text",
                "page_number": 3,
                "section_path": "Experiments",
                "context_role": "anchor_evidence",
                "context_budget_score": 0.9,
            }
        ]
    )

    assert "Evidence text" in context_pack["text_context"]
    assert len(context_pack["prompt_blocks"]) == 1
    block = context_pack["prompt_blocks"][0]
    assert block["section"] == "rag_evidence"
    assert block["block_type"] == "anchor_evidence"
    assert block["source_id"] == "chunk-1"
    assert block["metadata"]["page_number"] == 3


def test_budget_planner_keeps_protected_blocks_and_reclaims_unused_budget_for_rag() -> None:
    counter = TokenCounter(fallback_chars_per_token=1.0)
    planner = PromptBudgetPlanner(
        token_counter=counter,
        config=PromptBudgetConfig(
            max_input_tokens=80,
            safety_margin_tokens=0,
            rag_target_ratio=0.45,
            section_targets={"user_memory": 20, "rag_evidence": 20},
        ),
    )
    blocks = [
        PromptBlock(
            block_id="system",
            section="system_instruction",
            block_type="system_instruction",
            source="test",
            text="system",
            protected=True,
            droppable=False,
            priority=0,
        ),
        PromptBlock(
            block_id="question",
            section="current_question",
            block_type="current_question",
            source="test",
            text="question",
            protected=True,
            droppable=False,
            priority=0,
        ),
        PromptBlock(
            block_id="memory",
            section="user_memory",
            block_type="user_memory",
            source="test",
            text="m",
            priority=5,
        ),
        PromptBlock(
            block_id="rag-1",
            section="rag_evidence",
            block_type="anchor_evidence",
            source="test",
            text="r" * 20,
            priority=0,
            score=0.9,
        ),
        PromptBlock(
            block_id="rag-2",
            section="rag_evidence",
            block_type="anchor_evidence",
            source="test",
            text="r" * 20,
            priority=0,
            score=0.8,
        ),
    ]

    plan = planner.plan(blocks)
    selected_ids = [block.block_id for block in plan.selected_blocks]

    assert "system" in selected_ids
    assert "question" in selected_ids
    assert "rag-1" in selected_ids
    assert "rag-2" in selected_ids
    assert plan.debug["reclaimed_tokens"] > 0
    assert plan.debug["used_input_tokens"] <= 80


def test_rule_compactor_preserves_table_final_cells_and_drops_preview() -> None:
    counter = TokenCounter(fallback_chars_per_token=1.0)
    compactor = RuleCompactor(token_counter=counter)
    block = PromptBlock(
        block_id="table-1",
        section="rag_evidence",
        block_type="table_final_evidence",
        source="test",
        text="long preview should be dropped",
        metadata={
            "source_id": "table-source",
            "page_number": 4,
            "section_path": "Results",
            "table_evidence": {
                "operation_hint": "max",
                "decision": "final",
                "table": {"table_id": "t1"},
                "final_evidence": {
                    "cells": [
                        {
                            "row_index": 1,
                            "row_label": "Model A",
                            "col_name": "Accuracy",
                            "raw_value": "91.2",
                            "normalized_value": 91.2,
                            "unit": "%",
                            "confidence": 0.99,
                        }
                    ]
                },
            },
        },
    )

    compacted = compactor.compact(block)

    assert "Model A" in compacted.text
    assert "Accuracy" in compacted.text
    assert "91.2" in compacted.text
    assert "long preview" not in compacted.text
    assert compacted.evidence_origin == "rule_compacted"
    assert compacted.can_support_numeric_claim is True


def test_llm_compactor_disabled_by_default_returns_blocks_unchanged() -> None:
    counter = TokenCounter(fallback_chars_per_token=1.0)
    block = PromptBlock(
        block_id="rag-1",
        section="rag_evidence",
        block_type="anchor_evidence",
        source="test",
        text="x" * 2000,
        token_count=2000,
    )
    compacted, debug = LLMCompactor(token_counter=counter, client=lambda _: "{}").compact_blocks([block])

    assert compacted == [block]
    assert debug["enabled"] is False
    assert debug["reason"] == "disabled"


def test_llm_compactor_marks_derived_evidence_and_disables_quote_numeric_support() -> None:
    counter = TokenCounter(fallback_chars_per_token=1.0)
    block = PromptBlock(
        block_id="rag-1",
        section="rag_evidence",
        block_type="anchor_evidence",
        source="test",
        text="Long evidence " + "x" * 2000,
        token_count=2014,
        original_token_count=2014,
        can_support_direct_quote=True,
        source_id="source-1",
    )

    compacted, debug = LLMCompactor(
        token_counter=counter,
        config=LLMCompactionConfig(enabled=True, min_block_tokens=10, target_block_tokens=30),
        client=lambda _: json.dumps(
            {
                "compressed_text": "source_id: source-1\nKey finding is preserved.",
                "compression_notes": ["kept key finding"],
                "loss_level": "medium",
            }
        ),
    ).compact_blocks([block])

    assert debug["applied"] is True
    assert compacted[0].evidence_origin == "llm_compacted"
    assert compacted[0].can_support_numeric_claim is False
    assert compacted[0].can_support_direct_quote is False
    assert compacted[0].metadata["compacted_by_llm"] is True
    assert compacted[0].token_count < block.token_count


def test_llm_compactor_falls_back_to_rule_compacted_block_when_json_invalid() -> None:
    counter = TokenCounter(fallback_chars_per_token=1.0)
    block = PromptBlock(
        block_id="rag-1",
        section="rag_evidence",
        block_type="anchor_evidence",
        source="test",
        text="x" * 2000,
        token_count=2000,
        evidence_origin="rule_compacted",
    )

    compacted, debug = LLMCompactor(
        token_counter=counter,
        config=LLMCompactionConfig(enabled=True, min_block_tokens=10),
        client=lambda _: "not json",
    ).compact_blocks([block])

    assert compacted == [block]
    assert debug["applied"] is False
    assert debug["failed_blocks"][0]["reason"].startswith("llm_compaction_error:")


def test_llm_compactor_never_rewrites_table_final_evidence() -> None:
    counter = TokenCounter(fallback_chars_per_token=1.0)
    block = PromptBlock(
        block_id="table-1",
        section="rag_evidence",
        block_type="table_final_evidence",
        source="test",
        text="row_label=Model A; col_name=Accuracy; raw_value=91.2",
        token_count=2000,
        can_support_numeric_claim=True,
        source_id="table-source",
    )

    def fail_if_called(_: str) -> str:
        raise AssertionError("table final evidence must not be sent to LLM compaction")

    compacted, debug = LLMCompactor(
        token_counter=counter,
        config=LLMCompactionConfig(enabled=True, min_block_tokens=10),
        client=fail_if_called,
    ).compact_blocks([block])

    assert compacted == [block]
    assert debug["skipped_blocks"][0]["reason"] == "table_evidence_protected"


def test_budget_planner_applies_optional_llm_compactor_before_selection() -> None:
    counter = TokenCounter(fallback_chars_per_token=1.0)
    llm_compactor = LLMCompactor(
        token_counter=counter,
        config=LLMCompactionConfig(enabled=True, min_block_tokens=10, target_block_tokens=20),
        client=lambda _: {"compressed_text": "source_id: s1\ncompressed evidence"},
    )
    planner = PromptBudgetPlanner(
        token_counter=counter,
        llm_compactor=llm_compactor,
        config=PromptBudgetConfig(max_input_tokens=200, safety_margin_tokens=0),
    )
    plan = planner.plan(
        [
            PromptBlock(
                block_id="rag-1",
                section="rag_evidence",
                block_type="anchor_evidence",
                source="test",
                text="x" * 100,
                token_count=100,
                source_id="s1",
            )
        ]
    )

    assert plan.selected_blocks[0].evidence_origin == "llm_compacted"
    assert plan.debug["llm_compaction"]["applied"] is True
