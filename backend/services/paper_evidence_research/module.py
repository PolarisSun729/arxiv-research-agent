from __future__ import annotations

from typing import Any

from .contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult
from .errors import PaperEvidenceResearchError
from .graph import ResearchGraphDependencies, build_paper_evidence_research_graph
from .state import PaperEvidenceResearchState


class PaperEvidenceResearchService:
    """以单一 research 接口封装证据研究图及其内部依赖。"""

    def __init__(
        self,
        *,
        question_analyzer: Any,
        decision_policy: Any,
        retriever: Any,
        draft_generator: Any,
        claim_extractor: Any,
        claim_verifier: Any,
        checkpointer: Any = None,
    ) -> None:
        dependencies = ResearchGraphDependencies(
            question_analyzer=question_analyzer,
            decision_policy=decision_policy,
            retriever=retriever,
            draft_generator=draft_generator,
            claim_extractor=claim_extractor,
            claim_verifier=claim_verifier,
        )
        self._graph = build_paper_evidence_research_graph(dependencies, checkpointer=checkpointer)

    def research(self, request: PaperEvidenceResearchRequest) -> PaperEvidenceResearchResult:
        initial_state = PaperEvidenceResearchState(request=request)
        try:
            output = self._graph.invoke(
                initial_state,
                config={
                    "recursion_limit": 100,
                    "configurable": {"thread_id": request.research_run_id},
                },
            )
            final_state = PaperEvidenceResearchState.model_validate(output)
        except PaperEvidenceResearchError:
            raise
        except Exception as exc:
            # 模块只返回正常研究结果；依赖调用或状态流转异常统一转换成结构化系统错误。
            raise PaperEvidenceResearchError(
                code="paper_evidence_research_failed",
                stage="graph_invoke",
                message=str(exc),
                detail={"research_run_id": request.research_run_id, "error_type": type(exc).__name__},
            ) from exc
        if final_state.result is None:
            raise PaperEvidenceResearchError(
                code="research_result_missing",
                stage="graph_finalize",
                message="证据研究图已结束但没有生成研究结果",
                detail={"research_run_id": request.research_run_id},
            )
        return final_state.result
