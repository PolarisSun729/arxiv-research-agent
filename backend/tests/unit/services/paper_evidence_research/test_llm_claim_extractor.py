from __future__ import annotations

import json
from typing import Any

from services.paper_evidence_research.dependencies import (
    DegradationLedger,
    LlmClaimExtractor,
    REASON_LLM_CALL_FAILED,
    REASON_LLM_OUTPUT_INVALID,
)
from services.paper_evidence_research.state import ClaimExtractionRequest, EvidenceCandidate

# 复刻 6f26cbbf 的草稿形态：编号加粗小标题、引出句、项目符号里的加粗标签、句末引用。
_MARKDOWN_DRAFT = (
    "根据提供的证据，CogMem 的方法流程包含两个层面。\n\n"
    "具体步骤如下：\n\n"
    "**1. 记忆生成与演化层（顶层）**\n"
    "*   **记忆化（Memorize）：** 系统将原始对话转化为认知图中的节点。[source:54]\n"
    "*   **整合（Consolidate）：** 离线机制合并重复事实，并把冲突观点保留为主张。[source:5]\n\n"
    "**核心架构特点：**\n"
    "*   查询自适应控制器选择确定性的图算子，准确率提升 3.5%。[source:20]\n"
)

_NEEDS = [
    {"need_id": "need-1", "description": "记忆如何生成与演化", "importance": "core"},
    {"need_id": "need-2", "description": "推理阶段如何使用记忆", "importance": "core"},
    {"need_id": "need-bg", "description": "背景", "importance": "supporting"},
]


class FakeGenerationService:
    def __init__(self, replies: list[Any]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.task_types: list[str] = []

    def complete_with_qwen(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        self.task_types.append(kwargs.get("task_type"))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def _request(answer: str = _MARKDOWN_DRAFT, version: int = 2) -> ClaimExtractionRequest:
    return ClaimExtractionRequest(
        research_question="CogMem 的方法流程是什么？",
        answer=answer,
        draft_version=version,
        evidence_needs=_NEEDS,
        candidates=[EvidenceCandidate(candidate_id="54", content="memorize", matched_need_ids=["need-1"])],
    )


def _valid_reply() -> dict[str, Any]:
    return {
        "claims": [
            {
                "quote": "根据提供的证据，CogMem 的方法流程包含两个层面。",
                "citation_ids": [],
                "addressed_need_ids": ["need-1", "need-2"],
            },
            {
                # 带加粗与引用标记的摘录与不带的同样算原文。
                "quote": "**记忆化（Memorize）：** 系统将原始对话转化为认知图中的节点。[source:54]",
                "citation_ids": [],
                "addressed_need_ids": ["need-1", "need-unknown"],
            },
            {
                "quote": "整合（Consolidate）： 离线机制合并重复事实，并把冲突观点保留为主张。",
                "citation_ids": ["5", "999"],
                "addressed_need_ids": ["need-1"],
            },
            {
                "quote": "查询自适应控制器选择确定性的图算子，准确率提升 3.5%。",
                "citation_ids": ["20"],
                "addressed_need_ids": ["need-bg"],
            },
        ],
        "skipped": ["具体步骤如下：", "1. 记忆生成与演化层（顶层）", "核心架构特点："],
    }


def test_valid_output_becomes_claims_with_verbatim_text_and_draft_citations() -> None:
    service = FakeGenerationService([_valid_reply()])

    result = LlmClaimExtractor(service).extract(_request())

    assert result["extractor_source"] == "llm"
    assert service.task_types == ["research_claim_extraction"]
    claims = result["claims"]
    assert [claim["claim_id"] for claim in claims] == [f"claim-v2-{index}" for index in range(4)]
    # 标题、编号小标题、引出句都没有成为主张。
    assert all("具体步骤" not in claim["text"] and "顶层" not in claim["text"] for claim in claims)
    # 主张文本就是用户可见原文（只去掉了引用与加粗标记）。
    assert claims[1]["text"] == "记忆化（Memorize）： 系统将原始对话转化为认知图中的节点。"
    # 引用只认草稿中真实存在的标记；摘录自带的标记归属该主张。
    assert claims[0]["citation_ids"] == []
    assert claims[1]["citation_ids"] == ["54"]
    assert claims[2]["citation_ids"] == ["5"]
    # 未知需求被丢弃；importance 由绑定需求推导。
    assert claims[1]["addressed_need_ids"] == ["need-1"]
    assert claims[1]["importance"] == "core"
    assert claims[3]["importance"] == "supporting"


def test_paraphrased_quote_is_retried_with_feedback() -> None:
    paraphrased = _valid_reply()
    paraphrased["claims"][2]["quote"] = "整合阶段会去重事实"
    service = FakeGenerationService([paraphrased, _valid_reply()])

    result = LlmClaimExtractor(service).extract(_request())

    assert result["extractor_source"] == "llm"
    assert len(service.prompts) == 2
    assert "整合阶段会去重事实" in service.prompts[1]
    assert "not verbatim" in service.prompts[1]


def test_dropped_sentence_is_retried_with_the_missing_text() -> None:
    dropped = _valid_reply()
    del dropped["claims"][0]
    service = FakeGenerationService([dropped, _valid_reply()])

    result = LlmClaimExtractor(service).extract(_request())

    assert len(result["claims"]) == 4
    assert "CogMem 的方法流程包含两个层面" in service.prompts[1]
    assert "missing from every claim" in service.prompts[1]


def test_substantive_skipped_items_are_promoted_to_claims() -> None:
    reply = _valid_reply()
    numeric = reply["claims"].pop(3)["quote"]
    cited = reply["claims"].pop(1)["quote"]
    reply["skipped"] += [numeric, cited]
    service = FakeGenerationService([reply])

    result = LlmClaimExtractor(service).extract(_request())

    texts = {claim["text"]: claim for claim in result["claims"]}
    assert "查询自适应控制器选择确定性的图算子，准确率提升 3.5%。" in texts
    promoted = texts["记忆化（Memorize）： 系统将原始对话转化为认知图中的节点。"]
    assert promoted["citation_ids"] == ["54"]
    # 编号小标题里的序号不算实质数字，仍按排版跳过。
    assert all("顶层" not in text for text in texts)


def test_long_skipped_sentence_is_promoted_to_claim() -> None:
    # 无引用、无数字的概括长句被标成排版，也要送检：它陈述了事实，只是没写引用。
    long_intro = "根据提供的证据，CogMem 的方法流程包含记忆生成与演化、认知推理与应用这两个相互衔接的核心层面。"
    draft_with_long_intro = _MARKDOWN_DRAFT.replace("根据提供的证据，CogMem 的方法流程包含两个层面。", long_intro)
    reply = _valid_reply()
    reply["claims"].pop(0)
    reply["skipped"].append(long_intro)
    service = FakeGenerationService([reply])

    result = LlmClaimExtractor(service).extract(_request(answer=draft_with_long_intro))

    assert any(claim["text"].startswith("根据提供的证据") for claim in result["claims"])


def test_repeated_failures_degrade_to_rule_extractor() -> None:
    ledger = DegradationLedger()
    bad_quote = _valid_reply()
    bad_quote["claims"][0]["quote"] = "完全编造的句子"
    service = FakeGenerationService([RuntimeError("timeout"), "not json", bad_quote])

    result = LlmClaimExtractor(service, degradation_listener=ledger).extract(_request())

    assert len(service.prompts) == 3
    assert result["extractor_source"] == "rule"
    assert result["claims"], "规则兜底仍要给出主张"
    assert [(event.organ, event.reason) for event in ledger.events] == [
        ("claim_extractor", REASON_LLM_OUTPUT_INVALID)
    ]
    assert ledger.events[0].detail["attempts"] == 3


def test_call_failures_only_report_call_failed() -> None:
    ledger = DegradationLedger()
    service = FakeGenerationService([RuntimeError("down")] * 3)

    result = LlmClaimExtractor(service, degradation_listener=ledger).extract(_request())

    assert result["extractor_source"] == "rule"
    assert ledger.events[0].reason == REASON_LLM_CALL_FAILED
    # 调用失败不是输出问题，重试 prompt 不附带输出反馈。
    assert "previous output was rejected" not in service.prompts[2]


def test_empty_answer_skips_llm_call() -> None:
    service = FakeGenerationService([])

    result = LlmClaimExtractor(service).extract(_request(answer="  "))

    assert result == {"claims": [], "extractor_source": "llm"}
    assert service.prompts == []


def test_heading_with_model_name_digit_stays_skipped() -> None:
    # 696659ee：带模型名单个数字的小标题被当成主张，无引用必然校验失败，整轮只能部分回答。
    heading = "1. 核心架构：PEC 2 F 模式与认知图构建"
    draft = heading + "\n" + _MARKDOWN_DRAFT
    reply = _valid_reply()
    reply["skipped"].append(heading)
    service = FakeGenerationService([reply])

    result = LlmClaimExtractor(service).extract(_request(answer=draft))

    assert result["extractor_source"] == "llm"
    assert all("PEC 2 F" not in claim["text"] for claim in result["claims"])



def test_bilingual_heading_and_model_claimed_lead_ins_stay_as_layout() -> None:
    # bbd87754：中英双语标题按字符数超 40 被提升为主张；模型把"具体步骤如下："当成主张。
    # 二者都没引用，校验必然失败，部分回答会把标题整行删掉。
    heading = "### 1. 记忆生成与演化层 (Memory Generation & Evolution)"
    lead_in = "* a) 记忆化 (Memorize)："
    draft = heading + "\n" + lead_in + "\n" + _MARKDOWN_DRAFT
    reply = _valid_reply()
    reply["skipped"].append(heading)
    reply["skipped"].remove("具体步骤如下：")
    reply["claims"].append({"quote": lead_in, "citation_ids": [], "addressed_need_ids": ["need-1"]})
    reply["claims"].append({"quote": "具体步骤如下：", "citation_ids": [], "addressed_need_ids": []})

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    texts = [claim["text"] for claim in result["claims"]]
    assert result["extractor_source"] == "llm"
    assert not any("Memory Generation" in text or "Memorize)" in text or "具体步骤如下" in text for text in texts)


def test_cited_lead_in_is_still_verified() -> None:
    draft = "系统流程如下：[source:54]\n" + _MARKDOWN_DRAFT
    reply = _valid_reply()
    reply["claims"].append({"quote": "系统流程如下：", "citation_ids": ["54"], "addressed_need_ids": ["need-1"]})

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    assert "系统流程如下：" in [claim["text"] for claim in result["claims"]]


def test_claim_text_drops_leading_list_markers() -> None:
    draft = _MARKDOWN_DRAFT + "*   它将情节痕迹转化为图中的节点。[source:54]\n"
    reply = _valid_reply()
    reply["claims"].append(
        {"quote": "* 它将情节痕迹转化为图中的节点。", "citation_ids": ["54"], "addressed_need_ids": ["need-1"]}
    )

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    assert result["claims"][-1]["text"] == "它将情节痕迹转化为图中的节点。"
