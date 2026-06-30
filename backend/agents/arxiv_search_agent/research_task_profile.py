"""Research Task Classifier：补齐 intent 与工具计划之间的科研任务语义层。

这个模块运行在 Goal 构建之后、ExecutablePlan 生成之前，输出结构化 ResearchTaskProfile。
分类路线刻意拆成三段：
1. 规则高置信判断：只覆盖明确、低歧义表达，避免把分类器重新写成脆弱 if/else；
2. LLM 语义分类：规则不稳定时才调用，且只能在固定科研任务类型集合内输出 JSON 草稿；
3. 本地可执行性仲裁：检查目标论文、候选论文集合、用户画像等当前系统状态，必要时确认、降级或修正。

LLM 从不直接决定最终执行路线；最终 Profile 必须经过本地 schema 校验和仲裁层才能进入 state/debug。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schemas import (
    Goal,
    ResearchTaskArtifact,
    ResearchTaskConstraints,
    ResearchTaskEvidence,
    ResearchTaskExecutionReadiness,
    ResearchTaskObject,
    ResearchTaskProfile,
    ResearchTaskType,
)
from .state import AgentState

logger = logging.getLogger(__name__)


_ARTIFACTS_BY_TASK_TYPE: dict[str, List[ResearchTaskArtifact]] = {
    "direction_exploration": ["candidate_paper_set", "representative_paper_set"],
    "multi_paper_comparison": [
        "candidate_paper_set",
        "representative_paper_set",
        "method_cards",
        "experiment_info",
        "comparison_matrix",
    ],
    "single_paper_deep_read": ["method_cards", "experiment_info"],
    "reading_planning": ["candidate_paper_set", "representative_paper_set", "reading_order"],
    "research_gap_analysis": [
        "candidate_paper_set",
        "representative_paper_set",
        "method_cards",
        "comparison_matrix",
    ],
    "personalized_recommendation": ["candidate_paper_set", "representative_paper_set"],
}

_EVIDENCE_BY_TASK_TYPE: dict[str, List[ResearchTaskEvidence]] = {
    "direction_exploration": ["metadata", "abstract"],
    "multi_paper_comparison": [
        "metadata",
        "abstract",
        "method_chunk",
        "experiment_chunk",
        "table_evidence",
    ],
    "single_paper_deep_read": [
        "metadata",
        "abstract",
        "method_chunk",
        "experiment_chunk",
        "table_evidence",
    ],
    "reading_planning": ["metadata", "abstract"],
    "research_gap_analysis": ["metadata", "abstract", "method_chunk", "experiment_chunk"],
    "personalized_recommendation": ["metadata", "abstract", "user_profile_evidence"],
}

_TASK_TYPES: Sequence[ResearchTaskType] = (
    "direction_exploration",
    "multi_paper_comparison",
    "single_paper_deep_read",
    "reading_planning",
    "research_gap_analysis",
    "personalized_recommendation",
)

# 规则层只保留高精度信号；模糊表达交给 LLM，否则规则会膨胀成难维护的自然语言分类器。
_COMPARISON_PATTERNS: Sequence[str] = (
    r"比较|对比|差异|区别|异同|优劣|哪个更好|谁更好",
    r"\bcompar(?:e|ison|ing)\b|\bdifference\b|\bversus\b|\bvs\.?\b|\btrade[- ]?off",
)
_SINGLE_PAPER_PATTERNS: Sequence[str] = (
    r"这篇论文|这篇\s*paper|本文|该论文|第一篇|第二篇|第三篇|第[一二三四五六七八九十0-9]+篇|arxiv\s*id",
    r"\b\d{4}\.\d{4,5}(?:v\d+)?\b",
)
_READING_PLAN_PATTERNS: Sequence[str] = (
    r"阅读顺序|先读|从哪.*开始读|阅读路线|学习路线|阅读清单|阅读计划|入门顺序|怎么入门|学习路径|先看哪些",
    r"reading\s+order|reading\s+list|reading\s+plan|learning\s+path|where\s+to\s+start|how\s+to\s+get\s+started",
)
_RESEARCH_GAP_PATTERNS: Sequence[str] = (
    r"研究空白|空白点|尚未解决|还没有人|没有人做过|有哪些不足|局限性|开放问题|待解决|未解决|研究缺口|未来方向|future work",
    r"research\s+gap|open\s+problem|open\s+question|unsolved|limitation[s]?\b|future\s+direction|future\s+work",
)
_INTEREST_PATTERNS: Sequence[str] = (
    r"我的兴趣|我感兴趣|结合我的|根据我的|适合我|我的方向|我的研究|个性化|对我",
    r"my\s+interest|for\s+me|personali[sz]ed|based\s+on\s+my",
)


class _ClassificationDraft(BaseModel):
    """LLM/规则层的候选分类草稿；最终能否执行仍由本地仲裁决定。"""

    model_config = ConfigDict(extra="forbid")

    primary_task_type: ResearchTaskType
    secondary_task_types: List[ResearchTaskType] = Field(default_factory=list)
    task_object: ResearchTaskObject
    intermediate_artifacts: List[ResearchTaskArtifact] = Field(default_factory=list)
    evidence_requirements: List[ResearchTaskEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    classification_basis: Optional[str] = None
    missing_context: List[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


class ResearchTaskClassifier:
    """三段式科研任务分类器。"""

    def __init__(self, generation_service: Optional[Any] = None, *, timeout_seconds: int = 6) -> None:
        self.generation_service = generation_service
        self.timeout_seconds = max(1, int(timeout_seconds or 6))

    def classify(self, *, goal: Goal, state: AgentState) -> Optional[ResearchTaskProfile]:
        goal_type = _normalize_text(goal.goal_type or state.intent)
        if goal_type in {"unclear", "unsupported", "preference_action"}:
            return None

        trace: Dict[str, Any] = {
            "rule": {"matched": False},
            "llm": {"attempted": False},
            "arbitration": {"adjusted": False, "notes": []},
        }

        rule_draft = self._classify_by_rule(goal=goal, state=state)
        if rule_draft is not None:
            trace["rule"] = {
                "matched": True,
                "task_type": rule_draft.primary_task_type,
                "confidence": rule_draft.confidence,
                "basis": rule_draft.classification_basis,
            }
            return self._finalize_with_arbitration(
                draft=rule_draft,
                goal=goal,
                state=state,
                trace=trace,
                initial_source="rule_high_confidence",
            )

        llm_draft = self._classify_by_llm(goal=goal, state=state, trace=trace)
        if llm_draft is not None:
            return self._finalize_with_arbitration(
                draft=llm_draft,
                goal=goal,
                state=state,
                trace=trace,
                initial_source="llm_semantic_classifier",
            )

        # LLM 不可用或输出不可用时，语义层必须退回到原 intent 路线，不能强造复杂科研任务。
        fallback = self._fallback_to_intent_route(goal=goal, state=state, trace=trace)
        return fallback

    def _classify_by_rule(self, *, goal: Goal, state: AgentState) -> Optional[_ClassificationDraft]:
        goal_type = _normalize_text(goal.goal_type or state.intent)
        intent = _normalize_text(goal.intent or state.intent)
        message = str(state.message or "").strip()

        if goal_type == "paper_qa":
            return _draft_for(
                "single_paper_deep_read",
                state=state,
                task_object=_task_object_for("single_paper_deep_read", state=state, message=message),
                confidence=0.86,
                basis=f"intent={intent or goal_type} requires paper reading",
            )
        if goal_type == "recommendation":
            return _draft_for(
                "personalized_recommendation",
                state=state,
                task_object=ResearchTaskObject(object_type="user_profile", topic=_topic_from_state(state, message), description="recommendation centered on user profile"),
                confidence=0.84,
                basis="intent=recommendation maps to personalized_recommendation",
            )
        if goal_type != "arxiv_search":
            return None

        if _matches_any(message, _SINGLE_PAPER_PATTERNS):
            return _draft_for(
                "single_paper_deep_read",
                state=state,
                task_object=_task_object_for("single_paper_deep_read", state=state, message=message),
                confidence=0.82,
                basis="explicit paper reference signal",
            )
        if _matches_any(message, _COMPARISON_PATTERNS):
            return _draft_for(
                "multi_paper_comparison",
                state=state,
                task_object=_task_object_for("multi_paper_comparison", state=state, message=message),
                confidence=0.8,
                basis="explicit comparison signal",
            )
        if _matches_any(message, _READING_PLAN_PATTERNS):
            return _draft_for(
                "reading_planning",
                state=state,
                task_object=_task_object_for("reading_planning", state=state, message=message),
                confidence=0.78,
                basis="explicit reading-planning signal",
            )
        if _matches_any(message, _RESEARCH_GAP_PATTERNS):
            return _draft_for(
                "research_gap_analysis",
                state=state,
                task_object=_task_object_for("research_gap_analysis", state=state, message=message),
                confidence=0.78,
                basis="explicit research-gap signal",
            )
        if _matches_any(message, _INTEREST_PATTERNS):
            return _draft_for(
                "personalized_recommendation",
                state=state,
                task_object=ResearchTaskObject(object_type="user_profile", topic=_topic_from_state(state, message), description="interest-aware paper recommendation"),
                confidence=0.76,
                basis="explicit personal-interest signal",
            )
        return None

    def _classify_by_llm(self, *, goal: Goal, state: AgentState, trace: Dict[str, Any]) -> Optional[_ClassificationDraft]:
        service = self.generation_service
        if service is None:
            trace["llm"] = {"attempted": False, "error": "generation_service_unavailable"}
            return None
        prompt = self._build_llm_prompt(goal=goal, state=state)
        trace["llm"] = {"attempted": True}
        try:
            raw_text = self._invoke_generation_service(service, prompt)
            trace["llm"]["raw_summary"] = _safe_text_summary(raw_text)
            payload = self._parse_json_object(raw_text)
            draft = _ClassificationDraft.model_validate(payload)
        except Exception as exc:
            trace["llm"]["error"] = str(exc)
            return None
        trace["llm"].update(
            {
                "task_type": draft.primary_task_type,
                "secondary_task_types": list(draft.secondary_task_types or []),
                "confidence": draft.confidence,
                "basis": draft.classification_basis,
            }
        )
        return draft

    def _build_llm_prompt(self, *, goal: Goal, state: AgentState) -> str:
        context = state.context if isinstance(state.context, Mapping) else {}
        payload = {
            "user_request": state.message,
            "intent": state.intent,
            "goal": goal.model_dump(),
            "search_spec": state.search_spec.model_dump() if state.search_spec is not None else None,
            "context": {
                "selected_paper": _paper_debug(context.get("selected_paper")),
                "last_papers_count": len(_paper_list(context.get("last_papers") or context.get("papers"))),
                "last_paper_refs": _paper_refs_from_list(_paper_list(context.get("last_papers") or context.get("papers")))[:5],
                "has_user_profile": _has_profile_context(state),
                "context_keys": sorted(context.keys()),
            },
            "allowed_task_types": list(_TASK_TYPES),
            "allowed_task_object_types": ["topic", "paper", "paper_set", "user_profile"],
            "allowed_artifacts": sorted({item for values in _ARTIFACTS_BY_TASK_TYPE.values() for item in values}),
            "allowed_evidence": sorted({item for values in _EVIDENCE_BY_TASK_TYPE.values() for item in values}),
            "output_schema": {
                "primary_task_type": "one allowed_task_types value",
                "secondary_task_types": ["optional allowed_task_types values"],
                "task_object": {"object_type": "topic|paper|paper_set|user_profile", "topic": "string|null", "paper_refs": ["string"], "description": "string|null"},
                "intermediate_artifacts": ["allowed artifact values"],
                "evidence_requirements": ["allowed evidence values"],
                "confidence": "0.0-1.0",
                "classification_basis": "short reason grounded in the input/context",
                "missing_context": ["selected_paper|candidate_papers|user_profile|topic|other"],
            },
        }
        return (
            "你是受控 Research Task Classifier，只能输出 JSON-only 对象，禁止 Markdown 和自然语言解释。\n"
            "你的任务是在固定科研任务类型集合中选择主任务和可选次任务，并给出任务对象、中间产物、证据需求、置信度和分类依据。\n"
            "不要把简单搜索过度升级成综述；没有目标论文时不要判成可直接单篇深读；没有用户画像时不要判成可直接个性化推荐。\n"
            "最终是否可执行会由本地仲裁层决定，所以请诚实填写 missing_context。\n"
            f"输入：{json.dumps(payload, ensure_ascii=False)}"
        )

    def _invoke_generation_service(self, service: Any, prompt: str) -> str:
        generate = getattr(service, "generate", None)
        if callable(generate):
            try:
                result = generate(prompt=prompt, task_type="research_task_classifier", timeout=self.timeout_seconds)
            except TypeError:
                try:
                    result = generate("qwen", prompt, [], task_type="research_task_classifier", show_reasoning=False)
                except TypeError:
                    result = generate(prompt)
            return _extract_text(result)
        complete = getattr(service, "complete_with_qwen", None)
        if callable(complete):
            return _extract_text(complete(prompt, task_type="research_task_classifier", timeout=self.timeout_seconds))
        raise ValueError("generation service has no supported completion method")

    def _parse_json_object(self, raw_text: str) -> Mapping[str, Any]:
        text = str(raw_text or "").strip()
        if not (text.startswith("{") and text.endswith("}")):
            raise ValueError("llm_output_not_json_only")
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("llm_output_not_object")
        return payload

    def _finalize_with_arbitration(
        self,
        *,
        draft: _ClassificationDraft,
        goal: Goal,
        state: AgentState,
        trace: Dict[str, Any],
        initial_source: str,
    ) -> ResearchTaskProfile:
        if draft.confidence < 0.45:
            trace["arbitration"] = {
                "adjusted": True,
                "notes": ["分类置信度过低，退回原 intent 路线"],
            }
            fallback = self._fallback_to_intent_route(goal=goal, state=state, trace=trace)
            if fallback is not None:
                return fallback

        original_task_type = draft.primary_task_type
        task_type = draft.primary_task_type
        task_object = draft.task_object
        secondary = [item for item in list(draft.secondary_task_types or []) if item != task_type]
        readiness: ResearchTaskExecutionReadiness = "ready"
        needs_clarification = bool(draft.missing_context)
        notes: List[str] = []
        source = initial_source
        message = str(state.message or "").strip()
        goal_type = _normalize_text(goal.goal_type or state.intent)
        paper_refs = _merge_refs(list(task_object.paper_refs or []), _paper_refs_from_state(state))
        topic = task_object.topic or _topic_from_state(state, message)

        if task_type == "single_paper_deep_read":
            if paper_refs:
                task_object = ResearchTaskObject(object_type="paper", topic=topic, paper_refs=paper_refs, description=task_object.description or "single target paper deep read")
            elif goal_type == "arxiv_search":
                # 没有目标论文时，搜索类请求不能被 LLM 直接升级成单篇深读；降级为普通方向探索。
                task_type = "direction_exploration"
                task_object = ResearchTaskObject(object_type="topic", topic=topic, description="downgraded to topic exploration because no target paper is available")
                notes.append("缺少 selected_paper / arXiv ID，单篇深读降级为方向探索")
                source = "local_arbitration_adjusted"
            else:
                readiness = "needs_user_clarification"
                needs_clarification = True
                notes.append("单篇深读缺少目标论文，需要用户补充 selected_paper、序号或 arXiv ID")
                source = "local_arbitration_adjusted"
        elif task_type == "multi_paper_comparison":
            candidate_count = max(len(paper_refs), len(_candidate_papers_from_state(state)))
            if candidate_count >= 2:
                task_object = ResearchTaskObject(object_type="paper_set", topic=topic, paper_refs=paper_refs, description=task_object.description or "comparison over candidate papers")
            elif topic:
                readiness = "needs_retrieval"
                task_object = ResearchTaskObject(object_type="topic", topic=topic, paper_refs=paper_refs, description="comparison needs candidate retrieval first")
                notes.append("多论文比较缺少候选论文集合，planner 应先检索候选论文")
                source = "local_arbitration_adjusted"
            else:
                readiness = "needs_user_clarification"
                needs_clarification = True
                notes.append("多论文比较缺少候选论文和主题，需要用户补充比较对象")
                source = "local_arbitration_adjusted"
        elif task_type == "personalized_recommendation" and not _has_profile_context(state):
            if topic:
                # 没有画像时不假装能个性化；保留用户主题，回到普通方向探索。
                task_type = "direction_exploration"
                task_object = ResearchTaskObject(object_type="topic", topic=topic, description="downgraded because no user profile is available")
                notes.append("未加载到用户画像，个性化推荐降级为普通方向探索")
                source = "local_arbitration_adjusted"
            else:
                readiness = "needs_user_clarification"
                needs_clarification = True
                notes.append("个性化推荐缺少用户画像和明确主题，需要补充条件")
                source = "local_arbitration_adjusted"

        constraints = _build_constraints(task_type, state, message)
        # 仲裁层改写任务类型时，原 LLM/规则草稿的产物和证据可能已经不适用，必须按最终任务重建。
        draft_artifacts = list(draft.intermediate_artifacts or []) if task_type == original_task_type else []
        draft_evidence = list(draft.evidence_requirements or []) if task_type == original_task_type else []
        artifacts = _dedupe(draft_artifacts + list(_ARTIFACTS_BY_TASK_TYPE.get(task_type, [])))
        evidence = _dedupe(draft_evidence + list(_EVIDENCE_BY_TASK_TYPE.get(task_type, [])))
        if constraints.combine_with_interest and "user_profile_evidence" not in evidence and _has_profile_context(state):
            evidence.append("user_profile_evidence")

        trace["arbitration"] = {
            "adjusted": bool(notes) or source != initial_source,
            "notes": notes,
            "execution_readiness": readiness,
        }
        trace["final"] = {"source": source, "research_task_type": task_type, "execution_readiness": readiness}

        return ResearchTaskProfile(
            profile_id=f"{task_type}:{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            research_task_type=task_type,
            secondary_task_types=secondary,
            intent=_normalize_text(goal.intent or state.intent),
            goal_type=goal_type,
            task_object=task_object,
            constraints=constraints,
            intermediate_artifacts=artifacts,
            evidence_requirements=evidence,
            confidence=round(float(draft.confidence), 3),
            classification_basis=draft.classification_basis,
            needs_clarification=needs_clarification or readiness == "needs_user_clarification",
            execution_readiness=readiness,
            arbitration_notes=notes,
            classification_trace=trace,
            source=source,
        )

    def _fallback_to_intent_route(self, *, goal: Goal, state: AgentState, trace: Dict[str, Any]) -> Optional[ResearchTaskProfile]:
        goal_type = _normalize_text(goal.goal_type or state.intent)
        if goal_type in {"unclear", "unsupported", "preference_action"}:
            return None
        message = str(state.message or "").strip()
        task_type: ResearchTaskType = "single_paper_deep_read" if goal_type == "paper_qa" else "direction_exploration"
        task_object = _task_object_for(task_type, state=state, message=message)
        readiness: ResearchTaskExecutionReadiness = "ready"
        notes = ["规则层未高置信命中且 LLM 不可用或不可用，保留原 intent-based planner 路线"]
        if task_type == "single_paper_deep_read" and not task_object.paper_refs:
            readiness = "needs_user_clarification"
            notes.append("原 intent 是论文深读，但当前缺少目标论文")
        trace["arbitration"] = {"adjusted": True, "notes": notes, "execution_readiness": readiness}
        trace["final"] = {"source": "fallback_to_intent_route", "research_task_type": task_type, "execution_readiness": readiness}
        return ResearchTaskProfile(
            profile_id=f"{task_type}:{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            research_task_type=task_type,
            intent=_normalize_text(goal.intent or state.intent),
            goal_type=goal_type,
            task_object=task_object,
            constraints=_build_constraints(task_type, state, message),
            intermediate_artifacts=list(_ARTIFACTS_BY_TASK_TYPE.get(task_type, [])),
            evidence_requirements=list(_EVIDENCE_BY_TASK_TYPE.get(task_type, [])),
            confidence=0.4,
            classification_basis="fallback to original intent route",
            needs_clarification=readiness == "needs_user_clarification",
            execution_readiness=readiness,
            arbitration_notes=notes,
            classification_trace=trace,
            source="fallback_to_intent_route",
        )


def build_research_task_profile(
    goal: Goal,
    state: AgentState,
    generation_service: Optional[Any] = None,
) -> Optional[ResearchTaskProfile]:
    """根据 Goal 与 AgentState 推断本轮科研任务 Profile。

    失败时返回 None 或 fallback Profile，不向上抛异常；该语义层是增强能力，不能破坏既有 intent/计划流程。
    """
    try:
        return ResearchTaskClassifier(generation_service=generation_service).classify(goal=goal, state=state)
    except Exception as exc:  # pragma: no cover - 分类器失败必须降级，不能影响主链路
        logger.debug("research task classifier failed, skip profile: error=%s", exc)
        return None


def research_task_profile_debug(profile: Optional[ResearchTaskProfile]) -> Optional[dict]:
    """生成适合 debug/trace 暴露的 Profile 摘要。"""
    if profile is None:
        return None
    return {
        "profile_id": profile.profile_id,
        "research_task_type": profile.research_task_type,
        "secondary_task_types": list(profile.secondary_task_types or []),
        "intent": profile.intent,
        "goal_type": profile.goal_type,
        "task_object": profile.task_object.model_dump(),
        "constraints": profile.constraints.model_dump(),
        "intermediate_artifacts": list(profile.intermediate_artifacts or []),
        "evidence_requirements": list(profile.evidence_requirements or []),
        "confidence": profile.confidence,
        "classification_basis": profile.classification_basis,
        "needs_clarification": profile.needs_clarification,
        "execution_readiness": profile.execution_readiness,
        "arbitration_notes": list(profile.arbitration_notes or []),
        "source": profile.source,
        "classification_trace": dict(profile.classification_trace or {}),
    }


def _draft_for(
    task_type: ResearchTaskType,
    *,
    state: AgentState,
    task_object: ResearchTaskObject,
    confidence: float,
    basis: str,
) -> _ClassificationDraft:
    return _ClassificationDraft(
        primary_task_type=task_type,
        task_object=task_object,
        intermediate_artifacts=list(_ARTIFACTS_BY_TASK_TYPE.get(task_type, [])),
        evidence_requirements=list(_EVIDENCE_BY_TASK_TYPE.get(task_type, [])),
        confidence=confidence,
        classification_basis=basis,
        missing_context=[] if _has_minimum_context(task_type, state) else [],
    )


def _task_object_for(task_type: ResearchTaskType, *, state: AgentState, message: str) -> ResearchTaskObject:
    topic = _topic_from_state(state, message)
    paper_refs = _paper_refs_from_state(state)
    if task_type == "single_paper_deep_read":
        return ResearchTaskObject(object_type="paper", topic=topic, paper_refs=paper_refs, description="single target paper deep read")
    if task_type in {"multi_paper_comparison", "research_gap_analysis"} and paper_refs:
        return ResearchTaskObject(object_type="paper_set", topic=topic, paper_refs=paper_refs, description="paper-set research task")
    if task_type == "personalized_recommendation":
        return ResearchTaskObject(object_type="user_profile", topic=topic, description="profile-aware recommendation")
    return ResearchTaskObject(object_type="topic", topic=topic, paper_refs=paper_refs, description="topic-driven research task")


def _build_constraints(task_type: ResearchTaskType, state: AgentState, message: str) -> ResearchTaskConstraints:
    spec = state.search_spec
    time_range: Optional[str] = None
    max_count: Optional[int] = None
    research_fields: List[str] = []
    if spec is not None:
        if spec.submitted_days_ago is not None:
            time_range = f"submitted_days_ago<={spec.submitted_days_ago}"
        max_count = spec.max_results
        research_fields = list(spec.categories or [])
    combine_with_interest = task_type == "personalized_recommendation" or _matches_any(message, _INTEREST_PATTERNS) or _has_profile_context(state)
    return ResearchTaskConstraints(time_range=time_range, max_count=max_count, research_fields=research_fields, combine_with_interest=bool(combine_with_interest))


def _topic_from_state(state: AgentState, message: str) -> Optional[str]:
    spec = state.search_spec
    if spec is not None:
        for value in (spec.query, spec.title_query, spec.abstract_query):
            text = str(value or "").strip()
            if text:
                return text
    return message or None


def _paper_refs_from_state(state: AgentState) -> List[str]:
    refs: List[str] = []
    context = state.context if isinstance(state.context, Mapping) else {}
    selected = context.get("selected_paper")
    if isinstance(selected, Mapping):
        refs.extend(_paper_refs_from_mapping(selected))
    refs.extend(_paper_refs_from_list(_candidate_papers_from_state(state)))
    for match in re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", str(state.message or "")):
        refs.append(match)
    return _dedupe(refs)


def _candidate_papers_from_state(state: AgentState) -> List[Dict[str, Any]]:
    context = state.context if isinstance(state.context, Mapping) else {}
    for key in ("last_papers", "papers", "candidate_papers"):
        papers = _paper_list(context.get(key))
        if papers:
            return papers
    return []


def _has_minimum_context(task_type: ResearchTaskType, state: AgentState) -> bool:
    if task_type == "single_paper_deep_read":
        return bool(_paper_refs_from_state(state))
    if task_type == "multi_paper_comparison":
        return len(_candidate_papers_from_state(state)) >= 2 or len(_paper_refs_from_state(state)) >= 2
    if task_type == "personalized_recommendation":
        return _has_profile_context(state)
    return True


def _has_profile_context(state: AgentState) -> bool:
    context = state.context if isinstance(state.context, Mapping) else {}
    for key in ("user_memory_summary", "memory_summary", "research_profile"):
        value = context.get(key)
        if isinstance(value, Mapping) and value:
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def _paper_debug(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    return {"arxiv_id": value.get("arxiv_id") or value.get("paper_id"), "title": value.get("title")}


def _paper_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _paper_refs_from_list(papers: Sequence[Mapping[str, Any]]) -> List[str]:
    refs: List[str] = []
    for paper in list(papers or []):
        refs.extend(_paper_refs_from_mapping(paper))
    return _dedupe(refs)


def _paper_refs_from_mapping(paper: Mapping[str, Any]) -> List[str]:
    for key in ("arxiv_id", "paper_id", "title"):
        text = str(paper.get(key) or "").strip()
        if text:
            return [text]
    return []


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    normalized = str(text or "")
    if not normalized:
        return False
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _dedupe(items: Sequence[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for item in list(items or []):
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if item not in (None, "", [], {}) and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _merge_refs(*groups: Sequence[str]) -> List[str]:
    refs: List[str] = []
    for group in groups:
        refs.extend(str(item).strip() for item in list(group or []) if str(item or "").strip())
    return _dedupe(refs)


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, Mapping):
        for key in ("response", "result", "text", "answer"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("generation result does not contain text")


def _safe_text_summary(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    summary: Dict[str, Any] = {"chars": len(raw), "is_json_object": raw.strip().startswith("{") and raw.strip().endswith("}"), "preview": raw[:300]}
    try:
        payload = json.loads(raw)
    except Exception:
        return summary
    if isinstance(payload, Mapping):
        summary["keys"] = sorted(payload.keys())
        summary["primary_task_type"] = payload.get("primary_task_type")
        summary["confidence"] = payload.get("confidence")
    return summary


__all__ = ["ResearchTaskClassifier", "build_research_task_profile", "research_task_profile_debug"]
