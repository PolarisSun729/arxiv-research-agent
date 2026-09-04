"""研究引擎生产依赖的组装入口。

三器官（question_analyzer / decision_policy / claim_extractor）按"LLM 优先 + 规则兜底"
在此定型；retriever、draft_generator、claim_verifier 由调用方注入——retriever 在检索
适配阶段由 NeedOrchestratedRetriever 提供，draft_generator / claim_verifier 在生产切换
阶段包装 paper_qa 现有组件。
"""

from __future__ import annotations

from typing import Any

from ..module import PaperEvidenceResearchService
from .degradation import DegradationLedger, DegradationListener
from .llm_decision_policy import LlmDecisionPolicy
from .llm_question_analyzer import LlmQuestionAnalyzer
from .rule_claim_extractor import RuleClaimExtractor
from .rule_decision_policy import RuleDecisionPolicy
from .template_question_analyzer import TemplateQuestionAnalyzer


def build_paper_evidence_research_service(
    *,
    generation_service: Any,
    retriever: Any,
    draft_generator: Any,
    claim_verifier: Any,
    checkpointer: Any = None,
    degradation_listener: DegradationListener | None = None,
) -> PaperEvidenceResearchService:
    """组装完整的证据研究服务；默认把降级事件收进进程内台账。"""

    listener: DegradationListener = degradation_listener or DegradationLedger()
    return PaperEvidenceResearchService(
        question_analyzer=LlmQuestionAnalyzer(
            generation_service,
            fallback=TemplateQuestionAnalyzer(),
            degradation_listener=listener,
        ),
        decision_policy=LlmDecisionPolicy(
            generation_service,
            fallback=RuleDecisionPolicy(),
            degradation_listener=listener,
        ),
        retriever=retriever,
        draft_generator=draft_generator,
        claim_extractor=RuleClaimExtractor(),
        claim_verifier=claim_verifier,
        checkpointer=checkpointer,
    )
