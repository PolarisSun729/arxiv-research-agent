from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.paper_qa.repair_actions import (
    ASK_USER_TO_REBUILD_INDEX,
    build_repair_strategy_payload,
    describe_repair_actions,
    normalize_repair_actions,
)

from .base import BaseToolAdapter, backend_tool_error
from .models import ToolExecutionResult

try:
    from ..utils.paper_reference_resolver import _resolve_paper_reference
except Exception:  # pragma: no cover - 测试轻量导入场景下允许缺失。
    _resolve_paper_reference = None

from ..utils.paper_question_normalizer import normalize_single_paper_qa_question

try:
    from ..utils.paper_target_resolver import resolve_paper_target as _resolve_paper_target
except Exception:  # pragma: no cover - 测试轻量导入场景下允许缺失。
    _resolve_paper_target = None


logger = logging.getLogger(__name__)


class ResolvePaperInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""
    selected_paper: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    action_type: Optional[str] = None


class PaperReferenceOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    reference_type: str = "unknown"
    value: Optional[Any] = None
    confidence: float = 0.0
    requires_context: bool = False
    reason: Optional[str] = None
    final_target_resolved: bool = False
    reference_hint: Dict[str, Any] = Field(default_factory=dict)
    target_resolution: Dict[str, Any] = Field(default_factory=dict)
    target: Optional[Dict[str, Any]] = None
    paper: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_candidate: Optional[Dict[str, Any]] = None
    requires_confirmation: bool = False
    risk_level: Optional[str] = None
    action_type: Optional[str] = None
    resolution_reason: Optional[str] = None
    resolution_debug: Dict[str, Any] = Field(default_factory=dict)
    hint_confidence: float = 0.0
    paper_ref: Dict[str, Any] = Field(default_factory=dict)
    paper_reference: Dict[str, Any] = Field(default_factory=dict)
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    query: Optional[str] = None
    matched_by: Optional[str] = None
    source: Optional[str] = None


class CheckPaperIndexInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_ref: Optional[Dict[str, Any]] = None
    paper_reference: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None

    @property
    def arxiv_id(self) -> str:
        paper_ref = self.paper_ref or self.paper_reference or {}
        return str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""

    @property
    def reference_hint(self) -> Dict[str, Any]:
        paper_ref = self.paper_ref or self.paper_reference or {}
        if isinstance(paper_ref, Mapping) and (paper_ref.get("reference_type") or paper_ref.get("reference_hint")):
            nested_hint = paper_ref.get("reference_hint")
            return dict(nested_hint if isinstance(nested_hint, Mapping) else paper_ref)
        return {}


class PaperIndexStatusOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    has_index: bool = False
    tool_result: Optional[Dict[str, Any]] = None


class ParseAndIndexPaperInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_reference: Optional[Dict[str, Any]] = None
    paper_ref: Optional[Dict[str, Any]] = None
    run_id: Optional[str] = None

    @model_validator(mode="after")
    def _require_arxiv_id(self) -> "ParseAndIndexPaperInput":
        if not self.arxiv_id:
            raise ValueError("missing_arxiv_id")
        return self

    @property
    def arxiv_id(self) -> str:
        paper_ref = self.paper_reference or self.paper_ref or {}
        return str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""


class IndexBuildOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    has_index: Optional[bool] = None
    tool_result: Optional[Dict[str, Any]] = None


class AnswerPaperQuestionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_ref: Optional[Dict[str, Any]] = None
    paper_reference: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    question: Optional[str] = None
    run_id: Optional[str] = None
    qa_recovery_strategy: Optional[Dict[str, Any]] = None
    index_strategy: Optional[Dict[str, Any]] = None

    @property
    def arxiv_id(self) -> str:
        paper_ref = self.paper_ref or self.paper_reference or {}
        return str((paper_ref or {}).get("arxiv_id") or "").strip() if isinstance(paper_ref, Mapping) else ""

    @property
    def resolved_question(self) -> str:
        paper_ref = self.paper_ref or self.paper_reference or {}
        # 目标论文已由 resolve_paper 落地后，进入 QA 前把“第二篇论文”等列表引用收敛成单篇语境；
        # 原始 message 仍保留在 Agent trace 中，后端 QA 只消费归一化后的问题。
        return normalize_single_paper_qa_question(self.question or self.message or "", paper_ref)


class PaperQAAnswerOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "failed"
    answer: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_debug: Any = None
    qa_observation: Any = None
    error: Any = None
    arxiv_id: Optional[str] = None
    question: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


class AssessPaperQAQualityInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    paper_qa_result: Dict[str, Any] = Field(default_factory=dict)


class PaperQAQualityDecisionOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = "unknown"
    decision: str = "finalize"
    reason: str = "not_available"
    repair_required: bool = False
    repair_optional: bool = False
    repair_actions: List[str] = Field(default_factory=list)
    repair_action_details: List[Dict[str, Any]] = Field(default_factory=list)
    repair_strategy: Dict[str, Any] = Field(default_factory=dict)
    answer_available: bool = False
    source_count: int = 0
    retrieval_quality: str = "unknown"
    answer_quality: str = "unknown"
    answer_insufficient_evidence: str = "unknown"
    rerank_failed_reason: str = "not_available"
    degraded_stages: List[str] = Field(default_factory=list)
    qa_observation: Dict[str, Any] = Field(default_factory=dict)


def _with_reference_hint_aliases(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """把解析结果复制到旧 output_key 名称下，避免投影层丢失新增字段。

    resolved 状态下 paper_ref/paper_reference 才携带最终 arxiv_id；未解析状态只保留
    reference_hint、candidates 和 target_resolution，保证下游不能把线索误当最终目标。
    """
    normalized = dict(payload or {})
    normalized.setdefault("final_target_resolved", False)
    normalized.setdefault("arxiv_id", None)
    normalized.setdefault("title", None)
    hint = dict(normalized.get("reference_hint") if isinstance(normalized.get("reference_hint"), Mapping) else normalized)
    hint.setdefault("final_target_resolved", False)
    normalized.setdefault("reference_hint", hint)
    target_resolution = dict(normalized.get("target_resolution") if isinstance(normalized.get("target_resolution"), Mapping) else {})
    target = normalized.get("target") if isinstance(normalized.get("target"), Mapping) else None
    paper_payload = {
        "status": normalized.get("status"),
        "reference_type": normalized.get("reference_type"),
        "value": normalized.get("value"),
        "confidence": normalized.get("confidence"),
        "hint_confidence": normalized.get("hint_confidence", hint.get("confidence")),
        "source": normalized.get("source"),
        "requires_context": normalized.get("requires_context"),
        "reason": normalized.get("reason"),
        "resolution_reason": normalized.get("resolution_reason") or normalized.get("reason"),
        "reference_hint": hint,
        "target_resolution": target_resolution,
        "candidates": list(normalized.get("candidates") or []),
        "recommended_candidate": normalized.get("recommended_candidate"),
        "requires_confirmation": bool(normalized.get("requires_confirmation")),
        "risk_level": normalized.get("risk_level"),
        "action_type": normalized.get("action_type"),
        "final_target_resolved": bool(normalized.get("final_target_resolved")),
        "target": target,
        "paper": target,
        "arxiv_id": normalized.get("arxiv_id") if bool(normalized.get("final_target_resolved")) else None,
        "title": normalized.get("title") if bool(normalized.get("final_target_resolved")) else None,
        "matched_by": normalized.get("matched_by"),
    }
    normalized["paper_ref"] = paper_payload
    normalized["paper_reference"] = dict(paper_payload)
    return normalized


def _resolve_reference_then_target(message: str, context: Mapping[str, Any], *, action_type: str) -> Dict[str, Any]:
    """先抽取引用线索，再由 Target Resolver 结合上下文生成候选或最终目标。"""
    if callable(_resolve_paper_reference):
        reference_hint = _resolve_paper_reference(message, context)
    else:
        reference_hint = _resolve_paper_reference_fallback(message, context)
    if callable(_resolve_paper_target):
        return _resolve_paper_target(
            reference_hint=reference_hint if isinstance(reference_hint, Mapping) else {},
            message=message,
            context=context,
            action_type=action_type,
        )
    return dict(reference_hint or {})


def _resolve_paper_reference_fallback(message: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    del context
    arxiv_id_match = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", message or "")
    if arxiv_id_match:
        return _with_reference_hint_aliases(
            {
                "status": "hint_extracted",
                "reference_type": "arxiv_id",
                "value": arxiv_id_match.group(0),
                "confidence": 0.9,
                "source": "explicit_arxiv_id_fallback",
                "requires_context": False,
                "reason": None,
            }
        )
    return _with_reference_hint_aliases(
        {
            "status": "unknown",
            "reference_type": "unknown",
            "value": None,
            "confidence": 0.0,
            "source": "fallback_no_reference",
            "requires_context": False,
            "reason": "无法提取论文引用线索。",
            "query": message,
        }
    )


class ResolvePaperAdapter(BaseToolAdapter[ResolvePaperInput, PaperReferenceOutput]):
    tool_name = "resolve_paper"
    input_model = ResolvePaperInput
    output_model = PaperReferenceOutput

    def _run(self, tool_input: ResolvePaperInput) -> PaperReferenceOutput:
        message = str(tool_input.message or "").strip()
        context = dict(tool_input.context if isinstance(tool_input.context, Mapping) else {})
        selected_paper = tool_input.selected_paper
        if isinstance(selected_paper, Mapping) and "selected_paper" not in context:
            # selected_paper 是目标解析层的候选材料，不允许引用线索提取器直接消费。
            context["selected_paper"] = dict(selected_paper)
        resolution = _resolve_reference_then_target(
            message,
            context,
            action_type=str(tool_input.action_type or "paper_qa"),
        )
        return PaperReferenceOutput.model_validate(_with_reference_hint_aliases(resolution))


class CheckPaperIndexAdapter(BaseToolAdapter[CheckPaperIndexInput, PaperIndexStatusOutput]):
    tool_name = "check_paper_index"
    input_model = CheckPaperIndexInput
    output_model = PaperIndexStatusOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def _run(self, tool_input: CheckPaperIndexInput) -> PaperIndexStatusOutput:
        if not tool_input.arxiv_id:
            if tool_input.reference_hint:
                return PaperIndexStatusOutput(
                    status="target_unresolved",
                    has_index=False,
                    tool_result={
                        "reference_hint": tool_input.reference_hint,
                        "reason": "引用线索尚未解析成最终论文，不能检查或构建 QA 索引。",
                    },
                )
            return PaperIndexStatusOutput(status="missing", has_index=False)
        started = perf_counter()
        logger.debug("arxiv_agent paper qa index check started: arxiv_id=%s", tool_input.arxiv_id)
        tool_kwargs: Dict[str, Any] = {"arxiv_id": tool_input.arxiv_id}
        if tool_input.run_id:
            # run_id 只用于工具 trace 关联，索引状态检查本身不依赖它做业务判断。
            tool_kwargs["run_id"] = tool_input.run_id
        tool_result = self.invoke_backend_tool("check_paper_qa_index", **tool_kwargs)
        logger.debug(
            "arxiv_agent paper qa index check finished: arxiv_id=%s ok=%s elapsed_ms=%.1f",
            tool_input.arxiv_id,
            (tool_result or {}).get("ok") if isinstance(tool_result, Mapping) else None,
            (perf_counter() - started) * 1000,
        )
        data = (tool_result or {}).get("data") if isinstance(tool_result, Mapping) else {}
        return PaperIndexStatusOutput.model_validate(dict(data or {"status": "unknown", "has_index": False, "tool_result": tool_result}))


class ParseAndIndexPaperAdapter(BaseToolAdapter[ParseAndIndexPaperInput, IndexBuildOutput]):
    tool_name = "parse_and_index_paper"
    input_model = ParseAndIndexPaperInput
    output_model = IndexBuildOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def execute(self, tool_input: ParseAndIndexPaperInput) -> ToolExecutionResult:
        result = super().execute(tool_input)
        if result.ok and isinstance(result.data, IndexBuildOutput) and result.data.tool_result:
            backend_result = result.data.tool_result
            if not bool(backend_result.get("ok", False)):
                return result.model_copy(
                    update={
                        "ok": False,
                        "error": backend_tool_error(
                            error_code=((backend_result.get("error") or {}).get("code") if isinstance(backend_result.get("error"), Mapping) else None) or "index_build_failed",
                            message=str((backend_result.get("error") or {}).get("message") if isinstance(backend_result.get("error"), Mapping) else backend_result.get("summary") or "论文索引构建失败"),
                            detail={"backend_error": backend_result.get("error")},
                            retryable=False,
                            suggested_recovery="patch_plan",
                            safe_debug={"backend_tool_name": "build_paper_qa_index"},
                        ),
                    }
                )
        return result

    def _run(self, tool_input: ParseAndIndexPaperInput) -> IndexBuildOutput:
        # adapter 层只保留 DEBUG 边界日志；INFO 由 executor 和 index builder 统一输出，避免主时间线重复。
        started = perf_counter()
        logger.debug("arxiv_agent paper qa index build started: arxiv_id=%s", tool_input.arxiv_id)
        tool_kwargs: Dict[str, Any] = {"arxiv_id": tool_input.arxiv_id}
        if tool_input.run_id:
            # 构建链路耗时较长，run_id 需要继续传到 index builder 才能串起 Agent 和 QA 索引日志。
            tool_kwargs["run_id"] = tool_input.run_id
        tool_result = self.invoke_backend_tool("build_paper_qa_index", **tool_kwargs)
        logger.debug(
            "arxiv_agent paper qa index build finished: arxiv_id=%s ok=%s elapsed_ms=%.1f",
            tool_input.arxiv_id,
            (tool_result or {}).get("ok") if isinstance(tool_result, Mapping) else None,
            (perf_counter() - started) * 1000,
        )
        data = dict((tool_result or {}).get("data") or {})
        data.setdefault("tool_result", tool_result)
        return IndexBuildOutput.model_validate(data or {"tool_result": tool_result})


class AnswerPaperQuestionAdapter(BaseToolAdapter[AnswerPaperQuestionInput, PaperQAAnswerOutput]):
    tool_name = "answer_paper_question"
    input_model = AnswerPaperQuestionInput
    output_model = PaperQAAnswerOutput

    def __init__(self, invoke_backend_tool: Callable[..., Dict[str, Any]]) -> None:
        self.invoke_backend_tool = invoke_backend_tool

    def execute(self, tool_input: AnswerPaperQuestionInput) -> ToolExecutionResult:
        if not tool_input.arxiv_id:
            return self._error_result(
                started=__import__("time").perf_counter(),
                error_code="missing_arxiv_id",
                message="论文 QA 缺少 arxiv_id",
                detail={"paper_ref": tool_input.paper_ref or tool_input.paper_reference},
                failed_stage="input_validation",
                recoverable=True,
                retryable=False,
                suggested_recovery="ask_clarification",
            )
        return super().execute(tool_input)

    def _run(self, tool_input: AnswerPaperQuestionInput) -> PaperQAAnswerOutput:
        tool_kwargs: Dict[str, Any] = {"arxiv_id": tool_input.arxiv_id, "question": tool_input.resolved_question}
        if tool_input.run_id:
            # run_id 只用于日志/trace 关联，不参与 QA 业务判断。
            tool_kwargs["run_id"] = tool_input.run_id
        # recovery 策略只以结构化 payload 传递给 PaperQAService，避免在 adapter 里散落临时参数。
        if tool_input.qa_recovery_strategy:
            tool_kwargs["qa_recovery_strategy"] = dict(tool_input.qa_recovery_strategy)
        if tool_input.index_strategy:
            tool_kwargs["index_strategy"] = dict(tool_input.index_strategy)
        tool_result = self.invoke_backend_tool("answer_paper_question", **tool_kwargs)
        tool_data = dict((tool_result or {}).get("data") or {})
        tool_data.setdefault("status", "success" if bool((tool_result or {}).get("ok", False)) and str(tool_data.get("answer") or "").strip() else "failed")
        tool_data.setdefault("sources", [])
        tool_data.setdefault("retrieval_debug", None)
        tool_data.setdefault("qa_observation", None)
        if not bool((tool_result or {}).get("ok", False)):
            error_payload = (tool_result or {}).get("error") if isinstance((tool_result or {}).get("error"), Mapping) else {}
            tool_data.setdefault("error", error_payload.get("message") or (tool_result or {}).get("summary") or "answer_paper_question failed")
        else:
            tool_data.setdefault("error", None)
        tool_data.setdefault("arxiv_id", tool_input.arxiv_id)
        tool_data.setdefault("question", tool_input.resolved_question)
        tool_data["tool_result"] = tool_result
        return PaperQAAnswerOutput.model_validate(tool_data)


class AssessPaperQAQualityAdapter(BaseToolAdapter[AssessPaperQAQualityInput, PaperQAQualityDecisionOutput]):
    tool_name = "assess_paper_qa_quality"
    input_model = AssessPaperQAQualityInput
    output_model = PaperQAQualityDecisionOutput

    def _run(self, tool_input: AssessPaperQAQualityInput) -> PaperQAQualityDecisionOutput:
        result = dict(tool_input.paper_qa_result or {})
        qa_observation = result.get("qa_observation") if isinstance(result.get("qa_observation"), Mapping) else {}
        answer = str(result.get("answer") or "").strip()
        sources = result.get("sources") if isinstance(result.get("sources"), list) else []
        repair_actions = normalize_repair_actions(
            qa_observation.get("recommended_repair_actions") if isinstance(qa_observation.get("recommended_repair_actions"), list) else []
        )
        repair_action_details = describe_repair_actions(repair_actions)
        retrieval_quality = str(qa_observation.get("retrieval_quality") or "unknown").strip() or "unknown"
        answer_quality = str(qa_observation.get("answer_quality") or "unknown").strip() or "unknown"
        answer_insufficient = str(qa_observation.get("answer_insufficient_evidence") or "unknown").strip() or "unknown"
        rerank_failed_reason = str(qa_observation.get("rerank_failed_reason") or "not_available").strip() or "not_available"
        degraded_stages = [
            str(stage).strip()
            for stage in (qa_observation.get("degraded_stages") if isinstance(qa_observation.get("degraded_stages"), list) else [])
            if str(stage).strip()
        ]

        decision = "finalize"
        status = "passed"
        reason = str(qa_observation.get("observation_reason") or "paper_qa_quality_passed")
        repair_required = False
        repair_optional = False

        if not answer:
            decision, status, reason, repair_required = "repair_required", "low_quality", "paper_qa_answer_empty", True
        elif not sources:
            decision, status, reason, repair_required = "repair_required", "low_quality", "paper_qa_sources_empty", True
        elif answer_insufficient.lower() in {"yes", "true", "1"} or answer_quality in {"insufficient_evidence", "generation_failed"}:
            decision, status, reason, repair_required = (
                "repair_required",
                "low_quality",
                str(qa_observation.get("answer_quality_reason") or "paper_qa_insufficient_evidence"),
                True,
            )
        elif retrieval_quality in {"failed", "weak"}:
            decision, status, reason, repair_required = (
                "repair_required",
                "low_quality",
                str(qa_observation.get("retrieval_quality_reason") or f"paper_qa_retrieval_{retrieval_quality}"),
                True,
            )
        elif ASK_USER_TO_REBUILD_INDEX in set(repair_actions):
            decision, status, reason, repair_required = (
                "repair_required",
                "low_quality",
                str(qa_observation.get("retrieval_quality_reason") or "paper_qa_index_rebuild_required"),
                True,
            )
        elif (
            retrieval_quality == "partial"
            or answer_quality == "warning"
            or degraded_stages
            or rerank_failed_reason not in {"not_available", "disabled", "unknown"}
            or repair_actions
        ):
            # 降级不一定要强制重试，但必须进入决策 trace，避免把 fallback/rerank 退化伪装成纯成功。
            decision, status, repair_optional = "finalize_with_degradation", "degraded", True
            reason = str(qa_observation.get("observation_reason") or qa_observation.get("retrieval_quality_reason") or "paper_qa_degraded")

        return PaperQAQualityDecisionOutput(
            status=status,
            decision=decision,
            reason=reason,
            repair_required=repair_required,
            repair_optional=repair_optional,
            repair_actions=repair_actions,
            repair_action_details=repair_action_details,
            repair_strategy=build_repair_strategy_payload(actions=repair_actions, reason=reason, observation=qa_observation),
            answer_available=bool(answer),
            source_count=len(sources),
            retrieval_quality=retrieval_quality,
            answer_quality=answer_quality,
            answer_insufficient_evidence=answer_insufficient,
            rerank_failed_reason=rerank_failed_reason,
            degraded_stages=degraded_stages,
            qa_observation=dict(qa_observation),
        )
