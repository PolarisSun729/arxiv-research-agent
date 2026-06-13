from __future__ import annotations

import sys

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
intent_definitions = sys.modules["backend.agents.arxiv_search_agent.intent_definitions"]
intent_support = sys.modules["backend.agents.arxiv_search_agent.node.intent_support"]
clarification = sys.modules["backend.agents.arxiv_search_agent.clarification_analysis"]
search_spec_builder = sys.modules["backend.agents.arxiv_search_agent.utils.search_spec_builder"]


def test_parse_supported_excludes_runtime_confirmation() -> None:
    # confirmation 是运行时确认流专用意图，parse 阶段不应产出。
    assert "confirmation" not in intent_definitions.PARSE_SUPPORTED_INTENTS
    assert "confirmation" in intent_definitions.ALL_INTENTS


def test_all_intents_is_parse_plus_confirmation() -> None:
    assert intent_definitions.ALL_INTENTS == intent_definitions.PARSE_SUPPORTED_INTENTS | {"confirmation"}


def test_non_search_intents_subset_of_parse_supported() -> None:
    assert intent_definitions.NON_SEARCH_INTENTS <= intent_definitions.PARSE_SUPPORTED_INTENTS
    assert "arxiv_search" not in intent_definitions.NON_SEARCH_INTENTS


def test_intent_support_supported_intents_derived_from_single_source() -> None:
    # 三处消费方必须和单一事实来源保持一致，杜绝再次各自硬编码漂移。
    assert intent_support.SUPPORTED_INTENTS == set(intent_definitions.PARSE_SUPPORTED_INTENTS)
    assert set(intent_support.NON_SEARCH_INTENTS) == set(intent_definitions.NON_SEARCH_INTENTS)


def test_search_spec_builder_supported_intents_derived() -> None:
    assert search_spec_builder.SUPPORTED_INTENTS == set(intent_definitions.PARSE_SUPPORTED_INTENTS)


def test_clarification_supported_intents_include_confirmation() -> None:
    assert clarification.SUPPORTED_INTENTS == set(intent_definitions.ALL_INTENTS)
    assert "confirmation" in clarification.SUPPORTED_INTENTS


def test_minimum_execution_requirements_cover_all_intents() -> None:
    assert set(clarification._MINIMUM_EXECUTION_REQUIREMENTS) == set(intent_definitions.ALL_INTENTS)


def test_llm_intent_choices_match_parse_supported_and_ordered() -> None:
    # LLM prompt 的可选名单必须等于 parse 全集（顺序由配置决定）。
    assert set(intent_definitions.LLM_INTENT_CHOICES) == set(intent_definitions.PARSE_SUPPORTED_INTENTS)
    assert intent_definitions.LLM_INTENT_CHOICES[0] == "arxiv_search"


def test_llm_prompt_derives_intent_list_from_config() -> None:
    prompt = intent_support._build_llm_prompt_with_profile("找一些 RAG 论文")
    for intent in intent_definitions.LLM_INTENT_CHOICES:
        assert intent in prompt
    # 运行时确认意图不应出现在 parse prompt 里。
    assert "confirmation" not in prompt
