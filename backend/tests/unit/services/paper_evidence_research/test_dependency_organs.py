from __future__ import annotations

import json
from typing import Any

import pytest

from services.paper_evidence_research import PaperEvidenceResearchRequest
from services.paper_evidence_research.dependencies import (
    DegradationLedger,
    LlmDecisionPolicy,
    LlmQuestionAnalyzer,
    REASON_LLM_CALL_FAILED,
    REASON_LLM_OUTPUT_INVALID,
    RuleClaimExtractor,
    RuleDecisionPolicy,
    TemplateQuestionAnalyzer,
)
from services.paper_evidence_research.dependencies.rule_decision_policy import _pick_retrieval_mode


class FakeGenerationService:
    def __init__(self, replies: list[str] | None = None, *, raise_on_call: bool = False) -> None:
        self.replies = list(replies or [])
        self.raise_on_call = raise_on_call
        self.call_count = 0

    def complete_with_qwen(self, prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        if self.raise_on_call:
            raise RuntimeError("llm unavailable")
        if not self.replies:
            raise AssertionError("no scripted reply")
        return self.replies.pop(0)


class FakeRequest:
    def __init__(self, question: str = "这篇论文的方法流程是什么？") -> None:
        self.original_question = question


class TestTemplateQuestionAnalyzer:
    def test_produces_legal_ledger_with_core_need(self) -> None:
        result = TemplateQuestionAnalyzer().analyze(FakeRequest("实验用了什么数据集和指标？"))
        needs = result["evidence_needs"]
        assert result["analyzer_source"] == "template"
        assert any(need["importance"] == "core" for need in needs)
        assert len({need["need_id"] for need in needs}) == len(needs)

    @pytest.mark.parametrize(
        "question,intent",
        [
            ("这篇论文的方法流程是什么？", "method_flow"),
            ("实验部分用了什么数据集？", "experiment_setup"),
            ("主要结果提升了多少？", "result_analysis"),
            ("和基线方法比较如何？", "comparison"),
            ("论文的局限是什么？", "limitation"),
            ("这篇论文主要讲了什么？", "paper_overview"),
        ],
    )
    def test_intent_classification(self, question: str, intent: str) -> None:
        assert TemplateQuestionAnalyzer().analyze(FakeRequest(question))["intent"] == intent

    def test_unknown_intent_renders_question_into_need(self) -> None:
        result = TemplateQuestionAnalyzer().analyze(FakeRequest("随机无关字符串 xyz"))
        assert result["intent"] == "other"
        assert "随机无关字符串 xyz" in result["evidence_needs"][0]["description"]


class TestRuleDecisionPolicy:
    def _context(self, **overrides: Any) -> Any:
        from services.paper_evidence_research.state import ResearchDecisionContext

        base: dict[str, Any] = {
            "research_question": "问题",
            "evidence_needs": [
                {"need_id": "need-a", "description": "核心需求A", "importance": "core", "status": "open"},
                {"need_id": "need-b", "description": "核心需求B", "importance": "core", "status": "open"},
            ],
            "candidate_count": 0,
            "current_draft_version": None,
            "claim_assessments": [],
            "retrievals_remaining": 3,
            "drafts_remaining": 2,
            "retrievals_used": 0,
            "candidate_need_ids": [],
        }
        base.update(overrides)
        return ResearchDecisionContext.model_validate(base)

    def test_no_candidates_no_budget_abstains(self) -> None:
        action = RuleDecisionPolicy().decide(self._context(retrievals_remaining=0, candidate_count=0))
        assert action["action"] == "abstain"

    def test_no_candidates_with_budget_searches_open_core(self) -> None:
        action = RuleDecisionPolicy().decide(self._context())
        assert action["action"] == "search_paper"
        assert action["target_need_id"] == "need-a"
        assert action["query"].strip()

    def test_uncovered_need_gets_extra_search_before_draft(self) -> None:
        action = RuleDecisionPolicy().decide(
            self._context(candidate_count=2, candidate_need_ids=["need-a"], retrievals_used=1)
        )
        assert action["action"] == "search_paper"
        assert action["target_need_id"] == "need-b"

    def test_all_core_covered_drafts(self) -> None:
        action = RuleDecisionPolicy().decide(
            self._context(candidate_count=3, candidate_need_ids=["need-a", "need-b"], retrievals_used=2)
        )
        assert action["action"] == "draft_answer"
        assert set(action["addressed_need_ids"]) == {"need-a", "need-b"}

    def test_draft_all_supported_finalizes(self) -> None:
        from services.paper_evidence_research.state import ClaimAssessment

        # 真实流程里 project_coverage 会先把核心需求置为 satisfied 再回到 decide。
        action = RuleDecisionPolicy().decide(
            self._context(
                evidence_needs=[
                    {"need_id": "need-a", "description": "核心需求A", "importance": "core", "status": "satisfied"},
                    {"need_id": "need-b", "description": "核心需求B", "importance": "core", "status": "satisfied"},
                ],
                current_draft_version=1,
                candidate_count=3,
                candidate_need_ids=["need-a", "need-b"],
                claim_assessments=[ClaimAssessment(claim_id="c1", verdict="supported")],
            )
        )
        assert action["action"] == "finalize_answer"

    def test_draft_unsupported_no_retrieval_left_revises_then_requests_finalize(self) -> None:
        from services.paper_evidence_research.state import ClaimAssessment

        policy = RuleDecisionPolicy()
        revise = policy.decide(
            self._context(
                current_draft_version=1,
                candidate_count=3,
                candidate_need_ids=["need-a", "need-b"],
                claim_assessments=[ClaimAssessment(claim_id="c1", verdict="unsupported")],
                retrievals_remaining=0,
                drafts_remaining=1,
            )
        )
        assert revise["action"] == "draft_answer"
        finalize = policy.decide(
            self._context(
                current_draft_version=2,
                candidate_count=3,
                candidate_need_ids=["need-a", "need-b"],
                claim_assessments=[ClaimAssessment(claim_id="c1", verdict="unsupported")],
                retrievals_remaining=0,
                drafts_remaining=0,
            )
        )
        assert finalize["action"] == "finalize_answer"

    @pytest.mark.parametrize(
        "text,mode",
        [
            ("method and pipeline design", "method"),
            ("dataset and ablation experiment", "experiment"),
            ("compare with baseline", "comparison"),
            ("random words here", "broad"),
        ],
    )
    def test_retrieval_mode_mapping(self, text: str, mode: str) -> None:
        assert _pick_retrieval_mode(text) == mode


class TestRuleClaimExtractor:
    def _request(self, answer: str, needs: list[dict[str, Any]] | None = None, version: int = 1) -> Any:
        from services.paper_evidence_research.state import ClaimExtractionRequest

        return ClaimExtractionRequest(
            research_question="问题",
            answer=answer,
            draft_version=version,
            evidence_needs=needs or [],
        )

    def test_uncited_sentences_remain_visible_to_verification(self) -> None:
        result = RuleClaimExtractor().extract(
            self._request("第一句带引用。[source:chunk-a]第二句没有引用。又一句没有引用。")
        )
        claims = result["claims"]
        assert len(claims) == 3
        assert claims[0]["citation_ids"] == ["chunk-a"]
        assert claims[1]["citation_ids"] == []
        assert claims[2]["citation_ids"] == []
        assert "source:" not in claims[0]["text"]

    def test_binds_claim_to_needs_via_cited_candidates(self) -> None:
        from services.paper_evidence_research.state import EvidenceCandidate

        needs = [
            {"need_id": "need-method", "description": "反思 token 决定是否检索", "importance": "core"},
            {"need_id": "need-ablation", "description": "消融实验结果与分数变化", "importance": "core"},
        ]
        candidates = [
            EvidenceCandidate(
                candidate_id="chunk-method",
                content="method evidence",
                matched_need_ids=["need-method"],
            ),
            EvidenceCandidate(
                candidate_id="chunk-ablation",
                content="ablation evidence",
                matched_need_ids=["need-ablation"],
            ),
        ]
        from services.paper_evidence_research.state import ClaimExtractionRequest

        request = ClaimExtractionRequest(
            research_question="问题",
            answer="移除反思模块后分数从 54.1 降至 50.3 [source:chunk-ablation]。",
            draft_version=1,
            evidence_needs=needs,
            candidates=candidates,
        )
        result = RuleClaimExtractor().extract(request)
        assert result["claims"][0]["addressed_need_ids"] == ["need-ablation"]
        assert result["claims"][0]["importance"] == "core"

    def test_binds_claims_to_needs_by_lexical_overlap_fallback(self) -> None:
        needs = [
            {"need_id": "need-limit", "description": "移除模块后的消融分数结果", "importance": "core"},
        ]
        result = RuleClaimExtractor().extract(
            self._request("移除反思模块后分数显著下降 [source:chunk-x]。", needs=needs)
        )
        assert result["claims"][0]["addressed_need_ids"] == ["need-limit"]

    def test_unbound_claim_stays_supporting_and_unlinked(self) -> None:
        needs = [{"need_id": "need-x", "description": "完全无关的需求主题", "importance": "core"}]
        result = RuleClaimExtractor().extract(
            self._request("The transformer uses multi-head attention [source:chunk-1]。", needs=needs)
        )
        assert len(result["claims"]) == 1
        assert result["claims"][0]["addressed_need_ids"] == []
        assert result["claims"][0]["importance"] == "supporting"

    def test_strips_meta_discourse_and_dedups_citations(self) -> None:
        result = RuleClaimExtractor().extract(
            self._request("本文提出了检索决策机制 [source:chunk-a][source:chunk-a]。")
        )
        assert result["claims"][0]["text"].startswith("检索决策机制")
        assert result["claims"][0]["citation_ids"] == ["chunk-a"]

    def test_claim_ids_carry_draft_version(self) -> None:
        result = RuleClaimExtractor().extract(self._request("有效主张文本 [source:c]。", version=3))
        assert result["claims"][0]["claim_id"] == "claim-v3-0"


class TestLlmQuestionAnalyzer:
    def test_valid_llm_output_used_directly(self) -> None:
        reply = json.dumps(
            {
                "research_question": "方法与消融证据",
                "evidence_needs": [
                    {"need_id": "need-1", "description": "方法机制", "importance": "core"},
                    {"need_id": "need-2", "description": "消融数据", "importance": "supporting"},
                ],
            },
            ensure_ascii=False,
        )
        ledger = DegradationLedger()
        analyzer = LlmQuestionAnalyzer(FakeGenerationService([reply]), degradation_listener=ledger)
        result = analyzer.analyze(FakeRequest())
        assert result["analyzer_source"] == "llm"
        assert len(result["evidence_needs"]) == 2
        assert ledger.events == []

    def test_fenced_json_and_noise_tolerated(self) -> None:
        reply = "前置说明\n```json\n{\"research_question\": \"q\", \"evidence_needs\": [{\"need_id\": \"n1\", \"description\": \"d\", \"importance\": \"core\"}]}\n```\n后置说明"
        analyzer = LlmQuestionAnalyzer(FakeGenerationService([reply]))
        assert analyzer.analyze(FakeRequest())["analyzer_source"] == "llm"

    @pytest.mark.parametrize(
        "reply",
        [
            "不是 JSON",
            json.dumps({"research_question": "q", "evidence_needs": []}),
            json.dumps({"research_question": "q", "evidence_needs": [{"need_id": "n", "description": "d", "importance": "supporting"}]}),
            json.dumps({"research_question": "q", "evidence_needs": [{"need_id": "n", "description": "d"}, {"need_id": "n", "description": "d2"}]}),
        ],
    )
    def test_invalid_output_degrades_to_template(self, reply: str) -> None:
        ledger = DegradationLedger()
        analyzer = LlmQuestionAnalyzer(FakeGenerationService([reply]), degradation_listener=ledger)
        result = analyzer.analyze(FakeRequest("方法流程问题"))
        assert result["analyzer_source"] == "template"
        assert ledger.events and ledger.events[0].reason == REASON_LLM_OUTPUT_INVALID
        assert any(need["importance"] == "core" for need in result["evidence_needs"])

    def test_call_failure_degrades(self) -> None:
        ledger = DegradationLedger()
        analyzer = LlmQuestionAnalyzer(
            FakeGenerationService(raise_on_call=True), degradation_listener=ledger
        )
        result = analyzer.analyze(FakeRequest())
        assert result["analyzer_source"] == "template"
        assert ledger.events[0].reason == REASON_LLM_CALL_FAILED

    def test_overlong_need_list_truncated_core_first(self) -> None:
        needs = [{"need_id": f"n{i}", "description": f"支撑需求{i}", "importance": "supporting"} for i in range(8)]
        needs.append({"need_id": "n-core", "description": "核心需求", "importance": "core"})
        analyzer = LlmQuestionAnalyzer(
            FakeGenerationService([json.dumps({"research_question": "q", "evidence_needs": needs}, ensure_ascii=False)]),
            max_needs=3,
        )
        result = analyzer.analyze(FakeRequest())
        assert len(result["evidence_needs"]) == 3
        assert result["evidence_needs"][0]["importance"] == "core"


class TestLlmDecisionPolicy:
    def _context(self, **overrides: Any) -> Any:
        from services.paper_evidence_research.state import ResearchDecisionContext

        base: dict[str, Any] = {
            "research_question": "问题",
            "evidence_needs": [
                {"need_id": "need-a", "description": "核心需求", "importance": "core", "status": "open"}
            ],
            "candidate_count": 0,
            "current_draft_version": None,
            "claim_assessments": [],
            "retrievals_remaining": 2,
            "drafts_remaining": 2,
            "retrievals_used": 0,
            "candidate_need_ids": [],
        }
        base.update(overrides)
        return ResearchDecisionContext.model_validate(base)

    def test_valid_action_parsed(self) -> None:
        reply = json.dumps(
            {
                "action": "search_paper",
                "target_need_id": "need-a",
                "objective": "discover",
                "retrieval_mode": "method",
                "query": "核心需求检索词",
                "section_hints": ["Method"],
                "reason_code": "OPEN_NEED",
            },
            ensure_ascii=False,
        )
        policy = LlmDecisionPolicy(FakeGenerationService([reply]))
        action = policy.decide(self._context())
        assert action["action"] == "search_paper"
        assert action["query"] == "核心需求检索词"

    @pytest.mark.parametrize(
        "reply",
        [
            "不是 JSON",
            json.dumps({"action": "unknown_action"}),
            json.dumps({"action": "search_paper", "target_need_id": "need-a", "objective": "discover", "retrieval_mode": "broad", "query": "  "}),
        ],
    )
    def test_invalid_output_falls_back_to_rule_policy(self, reply: str) -> None:
        ledger = DegradationLedger()
        policy = LlmDecisionPolicy(FakeGenerationService([reply]), degradation_listener=ledger)
        action = policy.decide(self._context())
        assert action["action"] == "search_paper"  # 规则兜底给出合法动作
        assert ledger.events and ledger.events[0].reason == REASON_LLM_OUTPUT_INVALID

    def test_circuit_breaker_after_consecutive_failures(self) -> None:
        service = FakeGenerationService(["坏输出", "又坏", "再坏"])
        policy = LlmDecisionPolicy(service, max_consecutive_llm_failures=2)
        assert policy.decide(self._context())["action"] == "search_paper"
        assert policy.decide(self._context())["action"] == "search_paper"
        calls_after_breaker_attempt = service.call_count
        assert policy.decide(self._context())["action"] == "search_paper"
        assert service.call_count == calls_after_breaker_attempt  # 熔断后不再调 LLM


class TestRequestCompatibility:
    def test_organs_accept_real_request_contract(self) -> None:
        request = PaperEvidenceResearchRequest(
            arxiv_id="2401.00001",
            original_question="方法流程是什么",
            user_id="u",
            session_id="s",
            research_run_id="r",
        )
        result = TemplateQuestionAnalyzer().analyze(request)
        assert result["research_question"] == "方法流程是什么"
