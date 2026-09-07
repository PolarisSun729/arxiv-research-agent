from __future__ import annotations

import logging
from typing import Any

from .contracts import PaperEvidenceResearchRequest, PaperEvidenceResearchResult
from .errors import PaperEvidenceResearchError
from .graph import ResearchGraphDependencies, build_paper_evidence_research_graph
from .state import PaperEvidenceResearchState

logger = logging.getLogger(__name__)


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
        trace_sink: Any = None,
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
        # 研究轨迹按运行外抛给落盘器；它是审计与评测数据源，不进对外结果契约。
        self._trace_sink = trace_sink

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
        # 无论终态是完整回答、有限回答还是证据拒答，轨迹都要落盘；结果缺失同样需要可回放。
        self._emit_trace(request.research_run_id, final_state.trace_events)
        if final_state.result is None:
            raise PaperEvidenceResearchError(
                code="research_result_missing",
                stage="graph_finalize",
                message="证据研究图已结束但没有生成研究结果",
                detail={"research_run_id": request.research_run_id},
            )
        return final_state.result

    def _emit_trace(self, run_id: str, events: list[dict[str, Any]]) -> None:
        if self._trace_sink is None:
            return
        try:
            self._trace_sink(run_id, events)
        except Exception:  # pragma: no cover - 轨迹落盘永远不能影响研究结果
            logger.warning("research trace sink failed: run_id=%s", run_id, exc_info=True)
