from __future__ import annotations

import re
import sys

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
intent_support = sys.modules["backend.agents.arxiv_search_agent.node.intent_support"]
parse_module = sys.modules["backend.agents.arxiv_search_agent.node.parse_node"]


def test_hard_rule_patterns_are_not_empty() -> None:
    # hard rule 为空会让 parse_node 的第一层精确匹配彻底失效，这里直接锁住配置回归。
    assert intent_support.HARD_RULE_PATTERNS
    assert intent_support.SEARCH_TRIGGER_PATTERNS


def test_hard_rule_patterns_are_valid_regex() -> None:
    # 这些模式最终都会交给 re.search，编译检查能尽早阻断拼错正则导致的线上静默失效。
    for patterns in intent_support.HARD_RULE_PATTERNS.values():
        for pattern in patterns:
            re.compile(pattern, flags=re.IGNORECASE)
    for pattern in intent_support.SEARCH_TRIGGER_PATTERNS:
        re.compile(pattern, flags=re.IGNORECASE)


def test_detect_non_search_rule_intent_hits_hard_rules() -> None:
    assert parse_module._detect_non_search_rule_intent("总结这篇论文") == "paper_summary"
    assert parse_module._detect_non_search_rule_intent("讲讲这篇论文的方法") == "paper_detail"
    assert parse_module._detect_non_search_rule_intent("这篇论文为什么这么设计？") == "paper_qa"
    assert parse_module._detect_non_search_rule_intent("喜欢第 2 篇") == "preference_action"
    assert parse_module._detect_non_search_rule_intent("打开我的收藏夹") is None


def test_save_for_later_requests_fall_back_to_unsupported_without_success_plan() -> None:
    state = parse_module.parse_search_request({"message": "把这篇论文加入阅读列表"}, generation_service=None)

    assert state.intent == "unsupported"
    assert not any("阅读列表" in item or "已加入" in item for item in state.plan)
    assert not any("阅读列表" in item or "已加入" in item for item in state.next_actions)


def test_search_trigger_patterns_mark_search_requests() -> None:
    assert intent_support._looks_search_like("帮我找最近的 RAG 论文") is True
    assert intent_support._looks_search_like("search recent papers about LLM agents") is True


def test_search_request_with_recommendation_topic_is_not_personalized_recommendation() -> None:
    assert parse_module._detect_non_search_rule_intent("找推荐系统论文") is None
    assert intent_support._looks_search_like("找推荐系统论文") is True


def test_summary_patterns_do_not_capture_abstract_filters_or_topic_search() -> None:
    assert parse_module._detect_non_search_rule_intent("找摘要里包含 RAG 的论文") != "paper_summary"
    assert parse_module._detect_non_search_rule_intent("找 paper summarization 相关论文") != "paper_summary"


def test_preference_action_requires_explicit_paper_target() -> None:
    assert parse_module._detect_non_search_rule_intent("我喜欢 RAG 方向，帮我找论文") != "preference_action"
    assert intent_support._looks_search_like("我喜欢 RAG 方向，帮我找论文") is True


def test_recommendation_topic_search_is_not_recommendation_intent() -> None:
    assert parse_module._detect_non_search_rule_intent("推荐系统相关论文") != "recommendation"
    assert intent_support._looks_search_like("推荐系统相关论文") is True


class _ScriptedLLM:
    """按调用顺序返回预设输出，并记录每次调用的 task_type 和 prompt。"""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_with_qwen(self, prompt: str, task_type: str | None = None, **_: object) -> str:
        self.calls.append({"task_type": task_type, "prompt": prompt})
        return self.responses.pop(0)


FOLLOW_UP = "我感觉目前论文数量还是不够，请你扩大一下搜索范围，然后重新搜索一下"
RECENT_TURNS = [
    {"role": "user", "content": "帮我找 RAG 的论文"},
    {"role": "assistant", "content": "找到 5 篇论文", "search": {"concepts": ["RAG"]}, "paper_count": 5},
]


def test_follow_up_is_rewritten_with_recent_turns_before_intent_recognition() -> None:
    import json

    llm = _ScriptedLLM(
        json.dumps({"standalone_request": "扩大搜索范围，再多找一些 RAG 的论文"}, ensure_ascii=False),
        json.dumps({"intent": "arxiv_search", "confidence": 0.9, "concepts": [{"term": "RAG", "evidence": "RAG"}]}),
    )

    state = parse_module.parse_search_request({"message": FOLLOW_UP, "context": {"recent_turns": RECENT_TURNS}}, generation_service=llm)

    assert [call["task_type"] for call in llm.calls] == ["agent_request_contextualization", "intent_recognition"]
    assert "帮我找 RAG 的论文" in llm.calls[0]["prompt"]
    assert "User message: 扩大搜索范围，再多找一些 RAG 的论文" in llm.calls[1]["prompt"]
    assert state.intent == "arxiv_search"
    assert [concept.term for concept in state.search_spec.concepts] == ["RAG"]
    # state.message 保留用户原话；改写结果只进 debug 供前端展示。
    assert state.message == FOLLOW_UP
    assert state.debug["contextualized_message"] == "扩大搜索范围，再多找一些 RAG 的论文"


def test_contextualization_failure_keeps_original_message() -> None:
    import json

    llm = _ScriptedLLM("not json", json.dumps({"intent": "arxiv_search", "confidence": 0.8, "concepts": []}))

    state = parse_module.parse_search_request({"message": FOLLOW_UP, "context": {"recent_turns": RECENT_TURNS}}, generation_service=llm)

    assert f"User message: {FOLLOW_UP}" in llm.calls[1]["prompt"]
    assert state.debug["contextualization"]["status"] == "failed"
    assert "contextualized_message" not in state.debug


def test_search_without_any_topic_asks_user_instead_of_using_the_sentence() -> None:
    import json

    # 没有历史就不改写；LLM 判定没主题、也没有画像可补时，降级为 unclear，而不是把整句话当关键词去搜。
    llm = _ScriptedLLM(json.dumps({"intent": "arxiv_search", "confidence": 0.8, "concepts": []}))

    state = parse_module.parse_search_request({"message": FOLLOW_UP}, generation_service=llm)

    assert [call["task_type"] for call in llm.calls] == ["intent_recognition"]
    assert state.intent == "unclear"
    assert state.search_spec is None
