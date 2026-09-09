"""研究引擎生产依赖的组装入口。

问题分析与决策采用 LLM 优先、规则兜底，主张从可见答案独立提取。
retriever、draft_generator、claim_verifier 由组合根注入生产适配器；校验器失败不能
由词面规则代替 supported 结论。
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
    trace_sink: Any = None,
    configuration_provider: Any = None,
) -> PaperEvidenceResearchService:
    """组装完整的证据研究服务；默认把降级事件收进进程内台账。

    ``trace_sink`` 保持显式注入：研究轨迹落盘是生产/评测侧的选择（见 ResearchTraceRecorder），
    默认不写盘，避免单测顺手往仓库里生成轨迹文件。
    """

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
        trace_sink=trace_sink,
        configuration_provider=configuration_provider,
    )
