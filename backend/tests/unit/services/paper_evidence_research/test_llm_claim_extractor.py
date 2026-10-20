from __future__ import annotations

import json
from typing import Any

from services.paper_evidence_research.dependencies import (
    DegradationLedger,
    LlmClaimExtractor,
    REASON_LLM_CALL_FAILED,
    REASON_LLM_OUTPUT_INVALID,
)
from services.paper_evidence_research.dependencies.draft_segments import split_draft_segments
from services.paper_evidence_research.state import ClaimExtractionRequest, EvidenceCandidate

# 复刻 6f26cbbf 的草稿形态：编号加粗小标题、引出句、项目符号里的加粗标签、句末引用。
# 切出的片段依次是：s1 概括句、s2 引出句、s3 加粗小标题、s4 记忆化、s5 整合、s6 加粗小标题、s7 带数字的结论。
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
            {"segments": ["s1"], "addressed_need_ids": ["need-1", "need-2"]},
            {"segments": ["s4"], "addressed_need_ids": ["need-1", "need-unknown"]},
            {"segments": ["s5"], "addressed_need_ids": ["need-1"]},
            {"segments": ["s7"], "addressed_need_ids": ["need-bg"]},
        ],
        "layout": ["s2", "s3", "s6"],
    }


def _texts(result: dict[str, Any]) -> list[str]:
    return [claim["text"] for claim in result["claims"]]


def test_valid_output_becomes_claims_with_draft_text_and_citations() -> None:
    service = FakeGenerationService([_valid_reply()])

    result = LlmClaimExtractor(service).extract(_request())

    assert result["extractor_source"] == "llm"
    assert service.task_types == ["research_claim_extraction"]
    claims = result["claims"]
    assert [claim["claim_id"] for claim in claims] == [f"claim-v2-{index}" for index in range(4)]
    # 标题、编号小标题、引出句都没有成为主张。
    assert all("具体步骤" not in text and "顶层" not in text for text in _texts(result))
    # 主张文本就是用户可见原文（只去掉了行首符号、引用与加粗标记）。
    assert claims[1]["text"] == "记忆化（Memorize）： 系统将原始对话转化为认知图中的节点。"
    # 引用由代码从片段原文解析，不依赖模型上报。
    assert [claim["citation_ids"] for claim in claims] == [[], ["54"], ["5"], ["20"]]
    # 未知需求被丢弃；importance 由绑定需求推导。
    assert claims[1]["addressed_need_ids"] == ["need-1"]
    assert claims[1]["importance"] == "core"
    assert claims[3]["importance"] == "supporting"


def test_prompt_marks_segments_in_place_instead_of_asking_for_quotes() -> None:
    service = FakeGenerationService([_valid_reply()])

    LlmClaimExtractor(service).extract(_request())

    prompt = service.prompts[0]
    # 编号插在原文里，标题、列表结构仍然可见。
    assert "[s3] **1. 记忆生成与演化层（顶层）**" in prompt
    assert "[s7] *   查询自适应控制器" in prompt
    assert '"quote"' not in prompt


def test_quoted_question_marks_do_not_split_a_sentence() -> None:
    # f7df103c：规则切句在引号里的问号处断开，一句话变成三个各自无法校验的碎片。
    sentence = "例如，系统会将复杂问题分解为如“谁持有威尔士亲王头衔？”和“他们的配偶是谁？”等子问题，并初始化其状态。[source:35]"

    segments = split_draft_segments("### 2. 示例\n" + sentence + "\n")

    assert [segment.text for segment in segments] == ["### 2. 示例", sentence]


def test_trailing_citations_stay_with_the_previous_sentence() -> None:
    segments = split_draft_segments("XoG 拆分子问题。[source:10][source:11] 随后更新状态 [source:15]。It improves F1. [source:3] Done.")

    assert [segment.text.strip() for segment in segments] == [
        "XoG 拆分子问题。[source:10][source:11]",
        "随后更新状态 [source:15]。",
        "It improves F1. [source:3]",
        "Done.",
    ]


def test_figure_links_are_one_segment_and_stay_layout() -> None:
    # ab-small-9024529c：草稿照抄证据里的图注链接，图注里的句点把它切成两半，核对模型看不到图只能判不支持。
    caption = "[Figure 7: Effect of entity context size K on hit rate. Average answer rank (AAR) is indicated.]() [source:47]"
    image = "![picture 3: Hierarchical bucketing.](fig3.png)[source:36]"
    draft = _MARKDOWN_DRAFT + caption + "\n" + image + "\n" + "效果很好![source:47]"
    assert [segment.text for segment in split_draft_segments(draft)][-3:] == [caption, image, "效果很好![source:47]"]

    reply = _valid_reply()
    reply["claims"] += [{"segments": ["s8"]}, {"segments": ["s9"]}, {"segments": ["s10"]}]
    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    texts = _texts(result)
    assert not any("Figure 7" in text or "picture 3" in text for text in texts)
    assert "效果很好!" in texts


def test_non_adjacent_segments_are_split_back_into_separate_claims() -> None:
    reply = _valid_reply()
    reply["claims"] = [{"segments": ["s4", "s7"], "addressed_need_ids": ["need-1"]}, {"segments": ["s1"]}, {"segments": ["s5"]}]

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request())

    texts = _texts(result)
    assert not any("记忆化" in text and "查询自适应" in text for text in texts)
    by_text = {claim["text"]: claim for claim in result["claims"]}
    assert by_text["查询自适应控制器选择确定性的图算子，准确率提升 3.5%。"]["addressed_need_ids"] == ["need-1"]


def test_adjacent_segments_on_one_line_merge_into_one_claim_with_both_citations() -> None:
    draft = _MARKDOWN_DRAFT + "XoG 先把问题拆成子问题 [source:54]。再逐个求解子问题 [source:5]。\n"
    reply = _valid_reply()
    reply["claims"].append({"segments": ["s9", "s8"], "addressed_need_ids": ["need-1"]})

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    merged = result["claims"][-1]
    assert merged["text"] == "XoG 先把问题拆成子问题 。再逐个求解子问题 。"
    assert merged["citation_ids"] == ["54", "5"]


def test_segments_on_different_lines_are_never_merged() -> None:
    # ab-small-4ea765c7：小模型把"### 1. 框架概述"连同下面几个列表项合成一条主张，整节只能一起通过或一起删掉。
    reply = _valid_reply()
    reply["claims"] = [{"segments": ["s3", "s4", "s5"], "addressed_need_ids": ["need-1"]}, {"segments": ["s1"]}, {"segments": ["s7"]}]
    reply["layout"] = ["s2", "s6"]

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request())

    texts = _texts(result)
    assert "记忆化（Memorize）： 系统将原始对话转化为认知图中的节点。" in texts
    assert "整合（Consolidate）： 离线机制合并重复事实，并把冲突观点保留为主张。" in texts
    # 拆开后单独成行的加粗小标题按排版处理，不再混进主张。
    assert not any("顶层" in text for text in texts)


def test_numbered_section_heading_stays_layout() -> None:
    draft = "#### 3.1 关系探索 (Relation Exploration)\n" + _MARKDOWN_DRAFT
    reply = {"claims": [{"segments": ["s1"]}, {"segments": ["s2"]}], "layout": []}

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    assert not any("关系探索" in text for text in _texts(result))


def test_segments_the_model_forgot_are_still_verified_unless_clearly_layout() -> None:
    reply = _valid_reply()
    del reply["claims"][0]  # 漏掉概括句 s1
    reply["layout"] = []  # 也漏掉了全部排版片段

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request())

    texts = _texts(result)
    assert result["extractor_source"] == "llm"
    assert texts[0] == "根据提供的证据，CogMem 的方法流程包含两个层面。"
    assert not any("具体步骤" in text or "顶层" in text or "核心架构特点" in text for text in texts)


def test_unknown_and_repeated_segment_ids_do_not_trigger_a_retry() -> None:
    reply = _valid_reply()
    reply["claims"][0]["segments"] = ["s1", "s99"]
    reply["claims"].append({"segments": ["s1"], "addressed_need_ids": ["need-bg"]})
    service = FakeGenerationService([reply])

    result = LlmClaimExtractor(service).extract(_request())

    assert len(service.prompts) == 1
    assert len(result["claims"]) == 4
    assert result["claims"][0]["addressed_need_ids"] == ["need-1", "need-2"]


def test_substantive_layout_items_are_promoted_to_claims() -> None:
    reply = _valid_reply()
    reply["claims"] = [claim for claim in reply["claims"] if claim["segments"] not in (["s4"], ["s7"])]
    reply["layout"] += ["s4", "s7"]

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request())

    by_text = {claim["text"]: claim for claim in result["claims"]}
    assert "查询自适应控制器选择确定性的图算子，准确率提升 3.5%。" in by_text
    promoted = by_text["记忆化（Memorize）： 系统将原始对话转化为认知图中的节点。"]
    assert promoted["citation_ids"] == ["54"]
    # 代码兜底的主张沿用引用继承：候选 54 是为 need-1 检索回来的。
    assert promoted["addressed_need_ids"] == ["need-1"]
    # 编号小标题里的序号不算实质数字，仍按排版跳过。
    assert all("顶层" not in text for text in by_text)


def test_long_layout_sentence_is_promoted_to_claim() -> None:
    # 无引用、无数字的概括长句被标成排版，也要送检：它陈述了事实，只是没写引用。
    long_intro = "根据提供的证据，CogMem 的方法流程包含记忆生成与演化、认知推理与应用这两个相互衔接的核心层面。"
    draft_with_long_intro = _MARKDOWN_DRAFT.replace("根据提供的证据，CogMem 的方法流程包含两个层面。", long_intro)
    reply = _valid_reply()
    reply["claims"].pop(0)
    reply["layout"].append("s1")

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft_with_long_intro))

    assert any(text.startswith("根据提供的证据") for text in _texts(result))


def test_unparseable_output_is_retried_with_feedback() -> None:
    service = FakeGenerationService(["not json", {"claims": "s1"}, _valid_reply()])

    result = LlmClaimExtractor(service).extract(_request())

    assert result["extractor_source"] == "llm"
    assert len(service.prompts) == 3
    assert "previous output was rejected" in service.prompts[2]
    assert "JSON arrays" in service.prompts[2]


def test_repeated_failures_degrade_to_rule_extractor_and_report_why() -> None:
    ledger = DegradationLedger()
    service = FakeGenerationService([RuntimeError("timeout"), "not json", {"claims": None}])

    result = LlmClaimExtractor(service, degradation_listener=ledger).extract(_request())

    assert len(service.prompts) == 3
    assert result["extractor_source"] == "rule"
    assert result["claims"], "规则兜底仍要给出主张"
    assert [(event.organ, event.reason) for event in ledger.events] == [
        ("claim_extractor", REASON_LLM_OUTPUT_INVALID)
    ]
    assert ledger.events[0].detail["attempts"] == 3
    # 降级原因随结果返回，由图写进本次研究轨迹。
    assert result["degradation"]["reason"] == REASON_LLM_OUTPUT_INVALID
    assert "JSON arrays" in result["degradation"]["error"]


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

    result = LlmClaimExtractor(service).extract(_request(answer="  \n**  **\n"))

    assert result == {"claims": [], "extractor_source": "llm"}
    assert service.prompts == []


def test_heading_with_model_name_digit_stays_layout() -> None:
    # 696659ee：带模型名单个数字的小标题被当成主张，无引用必然校验失败，整轮只能部分回答。
    draft = "1. 核心架构：PEC 2 F 模式与认知图构建\n" + _MARKDOWN_DRAFT
    reply = {
        "claims": [{"segments": [f"s{number + 1}"]} for number in (1, 4, 5, 7)],
        "layout": ["s1", "s3", "s4", "s7"],
    }

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    assert result["extractor_source"] == "llm"
    assert all("PEC 2 F" not in text for text in _texts(result))


def test_bilingual_heading_and_model_claimed_lead_ins_stay_as_layout() -> None:
    # bbd87754：中英双语标题按字符数超 40 被提升为主张；模型把"具体步骤如下："当成主张。
    # 二者都没引用，校验必然失败，部分回答会把标题整行删掉。
    draft = "### 1. 记忆生成与演化层 (Memory Generation & Evolution)\n* a) 记忆化 (Memorize)：\n" + _MARKDOWN_DRAFT
    # 草稿前插了两行，原 s1..s7 顺延为 s3..s9。
    reply = {
        "claims": [{"segments": [segment_id]} for segment_id in ("s2", "s3", "s4", "s6", "s7", "s9")],
        "layout": ["s1", "s5", "s8"],
    }

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    texts = _texts(result)
    assert result["extractor_source"] == "llm"
    assert not any("Memory Generation" in text or "Memorize)" in text or "具体步骤如下" in text for text in texts)


def test_cited_lead_in_is_still_verified() -> None:
    draft = "系统流程如下：[source:54]\n" + _MARKDOWN_DRAFT
    reply = {"claims": [{"segments": ["s1"], "addressed_need_ids": ["need-1"]}], "layout": []}

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    assert "系统流程如下：" in _texts(result)


def test_claim_text_drops_leading_list_markers() -> None:
    draft = _MARKDOWN_DRAFT + "*   它将情节痕迹转化为图中的节点。[source:54]\n"
    reply = _valid_reply()
    reply["claims"].append({"segments": ["s8"], "addressed_need_ids": ["need-1"]})

    result = LlmClaimExtractor(FakeGenerationService([reply])).extract(_request(answer=draft))

    assert result["claims"][-1]["text"] == "它将情节痕迹转化为图中的节点。"
