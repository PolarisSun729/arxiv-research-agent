"""Tool-Aware Planner 的候选筛选与草稿转换边界。"""

from __future__ import annotations

import re
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .schemas import (
    ExcludedToolCandidate,
    ExecutablePlan,
    Goal,
    PlanDraft,
    PlanDraftStep,
    PlanStep,
    StepCondition,
    StepInputBinding,
    StepPolicy,
    ToolCandidate,
    ToolCandidateSelection,
    ToolSpec,
)
from .state import AgentState
from .tool_registry import PLANNER_TOOL_REGISTRY, ToolRegistry


class PlanDraftConversionError(ValueError):
    """PlanDraft 转换失败时使用的聚合错误类型。"""


class PlanDraftPlanningError(ValueError):
    """规则型 Tool-Aware planner 无法生成可信草稿时使用。"""


class LLMPlanDraftError(ValueError):
    """LLM PlanDraft 生成或规范化失败时使用。"""


class ToolCandidateSelector:
    """按 Goal 类型收窄 planner 可见工具，避免把全部工具无差别暴露给规划层。"""

    _GOAL_TAG_RULES: Dict[str, Set[str]] = {
        "arxiv_search": {"search", "validate", "rerank", "answer"},
        "paper_qa": {"retrieve", "validate", "answer", "index", "confirm"},
        "recommendation": {"profile", "recommendation", "validate", "answer"},
        "preference_action": {"preference", "memory_write", "clarify", "fallback"},
        "unclear": {"clarify", "fallback", "confirm"},
        "unsupported": {"clarify", "fallback"},
    }
    _GOAL_TOOL_RULES: Dict[str, Set[str]] = {
        "arxiv_search": {
            "normalize_request",
            "build_arxiv_search_spec",
            "search_arxiv",
            "validate_arxiv_results",
            "rewrite_arxiv_query",
            "personalize_paper_results",
            "synthesize_arxiv_response",
        },
        "paper_qa": {
            "resolve_paper",
            "check_paper_index",
            "answer_paper_question",
            "request_confirmation",
            "parse_and_index_paper",
        },
        "recommendation": {
            "load_user_profile",
            "load_candidate_papers",
            "generate_recommendations",
            "validate_recommendations",
            "explain_recommendations",
        },
        "preference_action": {
            "resolve_preference_target",
            "update_preference_store",
            "verify_preference_update",
            "synthesize_preference_response",
            "analyze_ambiguity",
            "generate_clarification",
            "generate_fallback_response",
        },
    }
    _BUSINESS_TAGS: Set[str] = {
        "search",
        "retrieve",
        "index",
        "recommendation",
        "preference",
        "memory_write",
        "rerank",
        "personalize",
    }

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry

    def select(self, goal: Goal, state: AgentState) -> ToolCandidateSelection:
        del state
        goal_type = str(goal.goal_type or "unsupported").strip() or "unsupported"
        allowed_tags = set(self._GOAL_TAG_RULES.get(goal_type, {"fallback"}))
        allowed_tool_names = set(self._GOAL_TOOL_RULES.get(goal_type, set()))
        candidate_tools: List[ToolCandidate] = []
        excluded_tools: List[ExcludedToolCandidate] = []

        for tool in self.tool_registry.list_tools():
            tags = set(tool.capability_tags or [])
            # 已知业务目标优先走显式白名单，避免仅凭 retrieve/answer 这类通用标签串到其他业务链路。
            selected = bool(tool.tool_name in allowed_tool_names) if allowed_tool_names else bool(tags.intersection(allowed_tags))
            if goal_type in {"unclear", "unsupported"} and tags.intersection(self._BUSINESS_TAGS):
                selected = False

            if selected:
                candidate_tools.append(
                    ToolCandidate(
                        tool_name=tool.tool_name,
                        capability_tags=list(tool.capability_tags or []),
                        side_effect_level=tool.side_effect_level,
                        requires_confirmation=tool.requires_confirmation,
                        selection_reason=self._selection_reason(goal_type, tool.tool_name, tags, allowed_tags),
                    )
                )
                continue

            excluded_tools.append(
                ExcludedToolCandidate(
                    tool_name=tool.tool_name,
                    capability_tags=list(tool.capability_tags or []),
                    side_effect_level=tool.side_effect_level,
                    requires_confirmation=tool.requires_confirmation,
                    exclusion_reason=self._exclusion_reason(goal_type, tags),
                )
            )

        risky_tools = [
            tool
            for tool in candidate_tools
            if tool.requires_confirmation or tool.side_effect_level in {"persistent_write", "external_call"}
        ]
        return ToolCandidateSelection(
            goal_type=goal_type,
            candidate_tools=candidate_tools,
            excluded_tools=excluded_tools,
            selection_reason=f"goal_type={goal_type} matched capability tags: {sorted(allowed_tags)}",
            risk_summary={
                "candidate_count": len(candidate_tools),
                "excluded_count": len(excluded_tools),
                "persistent_write_tools": [
                    tool.tool_name for tool in candidate_tools if tool.side_effect_level == "persistent_write"
                ],
                "external_call_tools": [
                    tool.tool_name for tool in candidate_tools if tool.side_effect_level == "external_call"
                ],
                "confirmation_required_tools": [
                    tool.tool_name for tool in candidate_tools if tool.requires_confirmation
                ],
                "risk_level": "high" if any(tool.side_effect_level == "persistent_write" for tool in risky_tools) else ("medium" if risky_tools else "low"),
            },
        )

    def _selection_reason(self, goal_type: str, tool_name: str, tags: Set[str], allowed_tags: Set[str]) -> str:
        matched_tags = sorted(tags.intersection(allowed_tags))
        if matched_tags:
            return f"{goal_type} allows tags {matched_tags}"
        return f"{goal_type} explicitly allows tool {tool_name}"

    def _exclusion_reason(self, goal_type: str, tags: Set[str]) -> str:
        if goal_type in {"unclear", "unsupported"} and tags.intersection(self._BUSINESS_TAGS):
            return "unclear/unsupported goals cannot expose business or write tools"
        return f"tool capabilities {sorted(tags)} do not match goal_type={goal_type}"


class LLMPlanDraftGenerator:
    """受控 LLM PlanDraft 生成器。

    LLM 只负责输出 JSON 草稿；这里会做 schema、候选工具、拓扑、风险和业务安全校验，
    不允许 LLM 直接产生 ExecutablePlan 或触达 Executor。
    """

    def __init__(self, generation_service: Optional[Any] = None, *, max_steps: int = 8, timeout_seconds: int = 8) -> None:
        self.generation_service = generation_service
        self.max_steps = max(1, int(max_steps or 8))
        self.timeout_seconds = max(1, int(timeout_seconds or 8))
        self.last_debug: Dict[str, Any] = {
            "raw_llm_plan": None,
            "invalid_reasons": [],
            "normalized_plan": None,
        }

    def generate(
        self,
        goal: Goal,
        state: AgentState,
        candidate_tools: Sequence[ToolCandidate],
        tool_registry: ToolRegistry,
    ) -> PlanDraft:
        if self.generation_service is None:
            raise LLMPlanDraftError("generation service unavailable")
        prompt = self._build_prompt(goal, state, candidate_tools, tool_registry)
        raw_text = self._invoke_generation_service(prompt)
        self.last_debug["raw_llm_plan"] = _safe_debug_text(raw_text)
        payload = self._parse_json_only(raw_text)
        draft = PlanDraft.model_validate(payload)
        normalized = self._normalize_draft(draft, goal, state, candidate_tools, tool_registry)
        self.last_debug["normalized_plan"] = normalized.model_dump()
        return normalized

    def _build_prompt(
        self,
        goal: Goal,
        state: AgentState,
        candidate_tools: Sequence[ToolCandidate],
        tool_registry: ToolRegistry,
    ) -> str:
        tool_payload = []
        for candidate in list(candidate_tools or []):
            tool = tool_registry.get(candidate.tool_name)
            if tool is None:
                continue
            tool_payload.append(
                {
                    "tool_name": tool.tool_name,
                    "capability_tags": list(tool.capability_tags or []),
                    "input_schema": dict(tool.input_schema or {}),
                    "output_schema": dict(tool.output_schema or {}),
                    "side_effect_level": tool.side_effect_level,
                    "requires_confirmation": tool.requires_confirmation,
                    "can_retry": tool.can_retry,
                }
            )
        prompt_payload = {
            "user_request": state.message,
            "goal": goal.model_dump(),
            "candidate_tools": tool_payload,
            "context_summary": _compact_mapping(state.context if isinstance(state.context, Mapping) else {}),
            "current_state_summary": {
                "intent": state.intent,
                "has_search_spec": state.search_spec is not None,
                "context_keys": sorted((state.context or {}).keys()) if isinstance(state.context, Mapping) else [],
                "has_selected_paper": bool(_get_selected_paper_hint(state.context if isinstance(state.context, Mapping) else {})),
            },
            "risk_policy": {
                "only_candidate_tools": True,
                "no_tool_execution": True,
                "persistent_write_requires_clear_target": True,
                "confirmation_required_for_tool_policy": True,
                "unsupported_or_unclear_must_use_clarification_or_fallback": True,
            },
            "allowed_output_schema": {
                "draft_id": "string",
                "plan_intent": "string",
                "selected_tools": ["tool_name"],
                "steps": [
                    {
                        "step_id": "string",
                        "action_type": "string",
                        "tool_name": "candidate tool only",
                        "step_reason": "why this step is needed",
                        "input_bindings": [
                            {
                                "input_key": "string",
                                "source_type": "state|context|goal|search_spec|step_output|literal",
                                "source_key": "optional string",
                                "step_id": "optional dependency step id",
                                "required": True,
                                "value": "optional literal",
                            }
                        ],
                        "depends_on": ["previous step_id"],
                        "expected_output_key": "unique output key",
                        "risk_level": "low|medium|high",
                        "requires_confirmation": False,
                        "fallback_reason": "optional string",
                    }
                ],
                "fallback_reason": "optional string",
                "metadata": {},
            },
        }
        return (
            "你是受控 Tool-Aware Planner，只能输出 JSON-only PlanDraft。\n"
            "禁止输出自然语言解释、Markdown、代码块或工具执行结果。\n"
            "只能选择 candidate_tools 中存在的工具，不要创造新工具。\n"
            "每个 step 必须有 step_reason、合法 depends_on、input_bindings 和 expected_output_key。\n"
            "高风险或 requires_confirmation 工具必须标记 requires_confirmation=true。\n"
            "persistent_write 工具只能在目标明确时使用；不确定时选择 clarification 工具。\n"
            "unsupported 请求只能选择 fallback 工具。\n"
            "输出 JSON 必须符合 allowed_output_schema。\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False)}"
        )

    def _invoke_generation_service(self, prompt: str) -> str:
        generate = getattr(self.generation_service, "generate", None)
        if not callable(generate):
            raise LLMPlanDraftError("generation service has no generate method")
        try:
            result = generate(prompt=prompt, task_type="agent_plan_draft", timeout=self.timeout_seconds)
        except TypeError:
            try:
                result = generate("qwen", prompt, [], task_type="agent_plan_draft", show_reasoning=False)
            except TypeError:
                result = generate(prompt)
        return self._extract_text(result)

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, Mapping):
            for key in ("response", "result", "text", "answer"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        raise LLMPlanDraftError("generation result does not contain text")

    def _parse_json_only(self, raw_text: str) -> Dict[str, Any]:
        text = str(raw_text or "").strip()
        if not (text.startswith("{") and text.endswith("}")):
            raise LLMPlanDraftError("LLM output is not JSON-only")
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise LLMPlanDraftError(f"LLM output is invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMPlanDraftError("LLM PlanDraft JSON must be an object")
        return payload

    def _normalize_draft(
        self,
        draft: PlanDraft,
        goal: Goal,
        state: AgentState,
        candidate_tools: Sequence[ToolCandidate],
        tool_registry: ToolRegistry,
    ) -> PlanDraft:
        invalid_reasons: List[str] = []
        candidate_names = {tool.tool_name for tool in list(candidate_tools or [])}
        steps = list(draft.steps or [])
        if not steps:
            invalid_reasons.append("LLM draft has no steps")
        if len(steps) > self.max_steps:
            invalid_reasons.append(f"LLM draft exceeds max steps: {len(steps)} > {self.max_steps}")

        seen_step_ids: Set[str] = set()
        seen_outputs: Set[str] = set()
        normalized_steps: List[PlanDraftStep] = []
        has_answer_step = False
        has_preference_write = False
        for step in steps:
            step_id = str(step.step_id or "").strip()
            if not step_id:
                invalid_reasons.append("step_id cannot be empty")
                continue
            if step_id in seen_step_ids:
                invalid_reasons.append(f"duplicate step_id: {step_id}")
            seen_step_ids.add(step_id)

            tool_name = str(step.tool_name or "").strip()
            tool = tool_registry.get(tool_name)
            if tool_name not in candidate_names:
                invalid_reasons.append(f"tool {tool_name} is outside candidate tools")
            if tool is None:
                invalid_reasons.append(f"tool {tool_name} is not registered")
                continue
            if not str(step.step_reason or "").strip():
                invalid_reasons.append(f"step {step_id} missing step_reason")
            output_key = str(step.expected_output_key or "").strip()
            if not output_key:
                invalid_reasons.append(f"step {step_id} missing expected_output_key")
            elif output_key in seen_outputs:
                invalid_reasons.append(f"duplicate output_key: {output_key}")
            else:
                seen_outputs.add(output_key)

            for dependency in list(step.depends_on or []):
                if dependency not in seen_step_ids:
                    invalid_reasons.append(f"step {step_id} depends on missing or later step {dependency}")

            if tool.input_schema and not list(step.input_bindings or []):
                invalid_reasons.append(f"step {step_id} missing input_bindings")
            for binding in list(step.input_bindings or []):
                if binding.source_type == "step_output" and str(binding.step_id or "").strip() not in seen_step_ids:
                    invalid_reasons.append(f"step {step_id} input {binding.input_key} references missing step {binding.step_id}")

            tags = set(tool.capability_tags or [])
            if "answer" in tags or "fallback" in tags or tool_name in {"generate_clarification", "answer_paper_question"}:
                has_answer_step = True
            if tool.side_effect_level == "persistent_write":
                has_preference_write = True
                if not step.requires_confirmation:
                    # LLM 对高风险工具漏标时统一补齐，避免草稿绕过确认策略。
                    step = step.model_copy(update={"requires_confirmation": True, "risk_level": "high"})
            if tool.requires_confirmation and not step.requires_confirmation:
                step = step.model_copy(update={"requires_confirmation": True})
            normalized_steps.append(step)

        if not has_answer_step:
            invalid_reasons.append("final answer step is missing")
        if has_preference_write and not _preference_target_is_clear(state):
            invalid_reasons.append("persistent_write preference plan requires a clear target")

        self.last_debug["invalid_reasons"] = invalid_reasons
        if invalid_reasons:
            raise LLMPlanDraftError("; ".join(invalid_reasons))
        return draft.model_copy(
            update={
                "selected_tools": _dedupe_tool_names([step.tool_name for step in normalized_steps]),
                "steps": normalized_steps,
                "metadata": dict(draft.metadata or {}, source="llm_plan_draft"),
            }
        )


class RuleBasedToolAwarePlanBuilder:
    """不用 LLM 的规则型 Tool-Aware PlanDraft 生成器。

    这个 builder 只负责从候选工具和上下文中组装“不可信草稿”；工具存在性、确认策略、
    依赖拓扑和副作用声明仍由转换器与 PlanValidator 兜底校验。
    """

    def __init__(self) -> None:
        self.last_debug: Dict[str, Any] = {}

    def build(
        self,
        goal: Goal,
        state: AgentState,
        candidate_tools: Sequence[ToolCandidate],
        tool_registry: ToolRegistry,
    ) -> PlanDraft:
        goal_type = str(goal.goal_type or "unsupported").strip() or "unsupported"
        context = state.context if isinstance(state.context, Mapping) else {}
        candidate_names = _dedupe_tool_names([tool.tool_name for tool in list(candidate_tools or [])])
        tools_by_name = {
            tool_name: tool_registry.get(tool_name)
            for tool_name in candidate_names
            if tool_registry.get(tool_name) is not None
        }
        self.last_debug = {
            "selected_steps": [],
            "skipped_steps": [],
            "skipped_tools": [],
            "fallback_reason": None,
            "candidate_tool_names": candidate_names,
        }

        builders = {
            "arxiv_search": self._build_arxiv_search,
            "paper_qa": self._build_paper_qa,
            "recommendation": self._build_recommendation,
            "preference_action": self._build_preference_action,
            "unclear": self._build_unclear,
            "unsupported": self._build_unsupported,
        }
        builder = builders.get(goal_type, self._build_unsupported)
        draft = builder(goal, state, context, tools_by_name)
        self.last_debug["selected_tools"] = list(draft.selected_tools or [])
        self.last_debug["draft_fallback_reason"] = draft.fallback_reason
        return draft

    def _build_arxiv_search(
        self,
        goal: Goal,
        state: AgentState,
        context: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        del context
        steps: List[PlanDraftStep] = []
        normalize = self._require_tool(tools_by_name, "normalize_request", required_tags={"search"})
        build_spec = self._require_tool(tools_by_name, "build_arxiv_search_spec", required_tags={"search"})
        search = self._require_tool(tools_by_name, "search_arxiv", required_tags={"search"})
        validate = self._require_tool(tools_by_name, "validate_arxiv_results", required_tags={"validate"})
        synthesize = self._require_tool(tools_by_name, "synthesize_arxiv_response", required_tags={"answer"})

        steps.append(
            self._draft_step(
                "normalize_request",
                normalize,
                action_type="write_state",
                output_key="normalized_request",
                reason="先把原始请求整理成后续搜索工具可复用的规范化输入。",
                input_bindings=[
                    _binding_dict("intent", source_type="state", source_key="intent"),
                    _binding_dict("message", source_type="state", source_key="message"),
                    _binding_dict("search_spec", source_type="search_spec"),
                ],
            )
        )
        steps.append(
            self._draft_step(
                "build_arxiv_search_spec",
                build_spec,
                action_type="search",
                output_key="search_spec",
                reason="候选工具具备 search 能力，用于把规范化请求转成结构化 arXiv 检索参数。",
                depends_on=["normalize_request"],
                input_bindings=[_binding_dict("normalized_request", source_type="step_output", step_id="normalize_request")],
            )
        )
        steps.append(
            self._draft_step(
                "search_arxiv",
                search,
                action_type="search",
                output_key="arxiv_results",
                reason="search 工具是 arxiv_search 目标的必要外部检索步骤。",
                depends_on=["build_arxiv_search_spec"],
                input_bindings=[_binding_dict("search_spec", source_type="step_output", step_id="build_arxiv_search_spec")],
                retry_policy=(
                    StepPolicy(policy_type="retry", mode="allow_search_relaxation", max_attempts=3)
                    if search.can_retry
                    else None
                ),
            )
        )
        steps.append(
            self._draft_step(
                "validate_arxiv_results",
                validate,
                action_type="validate",
                output_key="arxiv_result_quality",
                reason="用 validate 能力先检查检索质量，后续 response 和 replanner 都依赖这个判断。",
                depends_on=["search_arxiv"],
                input_bindings=[_binding_dict("arxiv_results", source_type="step_output", step_id="search_arxiv")],
            )
        )

        personalize = self._optional_tool(tools_by_name, preferred_name="personalize_paper_results", required_tags={"personalize", "rerank"})
        has_profile_context = _has_any_context_key(state, {"user_memory_summary", "memory_summary", "research_profile"})
        synthesize_depends_on = ["validate_arxiv_results"]
        synthesize_bindings = [_binding_dict("arxiv_result_quality", source_type="step_output", step_id="validate_arxiv_results")]
        if personalize and has_profile_context:
            steps.append(
                self._draft_step(
                    "personalize_paper_results",
                    personalize,
                    action_type="rerank",
                    output_key="ranked_papers",
                    reason="只有存在用户画像/记忆上下文时才启用个性化 rerank，避免无依据排序。",
                    depends_on=["search_arxiv"],
                    input_bindings=[
                        _binding_dict("arxiv_results", source_type="step_output", step_id="search_arxiv"),
                        _binding_dict("user_memory_summary", source_type="context", source_key="user_memory_summary", required=False),
                        _binding_dict("research_profile", source_type="context", source_key="research_profile", required=False),
                    ],
                )
            )
            synthesize_depends_on.append("personalize_paper_results")
            synthesize_bindings.insert(0, _binding_dict("ranked_papers", source_type="step_output", step_id="personalize_paper_results", required=False))
        else:
            self._skip_step(
                "personalize_paper_results",
                "candidate tool missing or no user memory/profile context; synthesize can use raw search results",
            )

        steps.append(
            self._draft_step(
                "synthesize_arxiv_response",
                synthesize,
                action_type="answer",
                output_key="final_answer",
                reason="answer 能力负责把检索质量和候选论文整理成用户可读回复。",
                depends_on=synthesize_depends_on,
                input_bindings=synthesize_bindings,
            )
        )
        return self._make_draft(goal, "arxiv_search", steps)

    def _build_paper_qa(
        self,
        goal: Goal,
        state: AgentState,
        context: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        resolve = self._require_tool(tools_by_name, "resolve_paper", required_tags={"retrieve"})
        check_index = self._require_tool(tools_by_name, "check_paper_index", required_tags={"validate", "retrieve"})
        answer = self._require_tool(tools_by_name, "answer_paper_question", required_tags={"answer"})
        has_selected_paper = bool(_get_selected_paper_hint(context))
        steps = [
            self._draft_step(
                "resolve_paper",
                resolve,
                action_type="retrieve",
                output_key="paper_ref",
                reason=(
                    "上下文已有 selected_paper，但仍保留 resolve 校验以统一后续 paper_ref 契约。"
                    if has_selected_paper
                    else "先解析目标论文，避免后续 QA 工具在缺少论文引用时误执行。"
                ),
                input_bindings=[
                    _binding_dict("message", source_type="state", source_key="message"),
                    _binding_dict("selected_paper", source_type="context", source_key="selected_paper", required=False),
                    # resolve_paper 需要完整上下文里的 last_papers，才能稳定处理“第二篇”等序号引用。
                    _binding_dict("context", source_type="state", source_key="context", required=False),
                ],
            ),
            self._draft_step(
                "check_paper_index",
                check_index,
                action_type="validate",
                output_key="paper_index_status",
                reason="索引检查不能跳过；缺索引时由 Observer/Replanner 追加确认和建索引链路。",
                depends_on=["resolve_paper"],
                input_bindings=[_binding_dict("paper_ref", source_type="step_output", step_id="resolve_paper")],
            ),
            self._draft_step(
                "answer_paper_question",
                answer,
                action_type="answer",
                output_key="paper_qa_result",
                reason="只调用真实 PaperQA answer 工具，检索、rerank 和 grounding 由服务内部处理。",
                depends_on=["resolve_paper", "check_paper_index"],
                input_bindings=[
                    _binding_dict("paper_ref", source_type="step_output", step_id="resolve_paper"),
                    _binding_dict("message", source_type="state", source_key="message"),
                ],
            ),
        ]
        if self._optional_tool(tools_by_name, preferred_name="parse_and_index_paper", required_tags={"index"}):
            self._skip_step("parse_and_index_paper", "高成本索引工具不默认进入主链路，缺索引时由确认机制重规划加入。")
        return self._make_draft(goal, "paper_qa", steps)

    def _build_recommendation(
        self,
        goal: Goal,
        state: AgentState,
        context: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        del state, context
        profile = self._require_tool(tools_by_name, "load_user_profile", required_tags={"profile"})
        candidate = self._optional_tool(tools_by_name, preferred_name="load_candidate_papers", required_tags={"recommendation", "retrieve"})
        recommend = self._require_tool(tools_by_name, "generate_recommendations", required_tags={"recommendation"})
        validate = self._require_tool(tools_by_name, "validate_recommendations", required_tags={"validate", "recommendation"})
        explain = self._require_tool(tools_by_name, "explain_recommendations", required_tags={"answer", "recommendation"})

        steps = [
            self._draft_step(
                "load_user_profile",
                profile,
                action_type="retrieve",
                output_key="recommendation_profile",
                reason="推荐先读取画像；画像为空时保留给 Observer/Replanner 做降级。",
                input_bindings=[_binding_dict("context", source_type="state", source_key="context", required=False)],
            )
        ]
        recommend_depends_on = ["load_user_profile"]
        recommend_bindings = [_binding_dict("recommendation_profile", source_type="step_output", step_id="load_user_profile")]
        if candidate:
            steps.append(
                self._draft_step(
                    "load_candidate_papers",
                    candidate,
                    action_type="retrieve",
                    output_key="candidate_papers",
                    reason="候选论文加载可选；缺失时推荐工具仍可按消息和画像降级生成。",
                    depends_on=["load_user_profile"],
                    input_bindings=[_binding_dict("recommendation_profile", source_type="step_output", step_id="load_user_profile")],
                    fallback_reason="candidate papers may be unavailable; downstream recommendation can degrade",
                )
            )
            recommend_depends_on.append("load_candidate_papers")
            recommend_bindings.append(_binding_dict("candidate_papers", source_type="step_output", step_id="load_candidate_papers", required=False))
        else:
            self._skip_step("load_candidate_papers", "candidate tool missing; generate_recommendations will rely on profile/message fallback")
            self.last_debug["fallback_reason"] = "candidate papers tool missing; optional recommendation step skipped"

        steps.extend(
            [
                self._draft_step(
                    "generate_recommendations",
                    recommend,
                    action_type="search",
                    output_key="recommendation_result",
                    reason="recommendation 能力负责生成候选推荐，外部调用风险来自 ToolSpec。",
                    depends_on=recommend_depends_on,
                    input_bindings=recommend_bindings,
                ),
                self._draft_step(
                    "validate_recommendations",
                    validate,
                    action_type="validate",
                    output_key="validated_recommendations",
                    reason="推荐结果先校验再解释，避免把空推荐直接交给用户。",
                    depends_on=["generate_recommendations"],
                    input_bindings=[_binding_dict("recommendation_result", source_type="step_output", step_id="generate_recommendations")],
                ),
                self._draft_step(
                    "explain_recommendations",
                    explain,
                    action_type="answer",
                    output_key="final_answer",
                    reason="explain 是最终用户回复步骤，缺失时必须 fallback。",
                    depends_on=["validate_recommendations"],
                    input_bindings=[_binding_dict("validated_recommendations", source_type="step_output", step_id="validate_recommendations")],
                ),
            ]
        )
        return self._make_draft(goal, "recommendation", steps)

    def _build_preference_action(
        self,
        goal: Goal,
        state: AgentState,
        context: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        resolve = self._require_tool(tools_by_name, "resolve_preference_target", required_tags={"preference", "retrieve"})
        update = self._require_tool(tools_by_name, "update_preference_store", required_tags={"memory_write", "preference"})
        verify = self._require_tool(tools_by_name, "verify_preference_update", required_tags={"validate", "preference"})
        synthesize = self._require_tool(tools_by_name, "synthesize_preference_response", required_tags={"answer", "preference"})
        if update.side_effect_level != "persistent_write":
            return self._fallback_draft(goal, "preference_action", f"update tool side_effect_level must be persistent_write, got {update.side_effect_level}")

        has_target = bool(_get_selected_paper_hint(context) or _has_candidate_paper_context(context) or _message_has_paper_hint(state.message))
        if not has_target:
            self._skip_step("update_preference_store", "目标论文不明确，拒绝生成 persistent_write 步骤。")
            return self._build_unclear(goal, state, context, tools_by_name)

        steps = [
            self._draft_step(
                "resolve_preference_target",
                resolve,
                action_type="retrieve",
                output_key="paper_reference",
                reason="写入偏好前必须先解析目标论文，避免无目标持久化写入。",
                input_bindings=[
                    _binding_dict("message", source_type="state", source_key="message"),
                    _binding_dict("selected_paper", source_type="context", source_key="selected_paper", required=False),
                    # 偏好动作也会用“喜欢第二篇”这种说法，必须把最近搜索列表传给解析器。
                    _binding_dict("context", source_type="state", source_key="context", required=False),
                ],
            ),
            self._draft_step(
                "update_preference_store",
                update,
                action_type="write_state",
                output_key="preference_action_result",
                reason="ToolSpec 标记为 persistent_write，因此在 draft/debug 中显式保留高风险信息。",
                depends_on=["resolve_preference_target"],
                input_bindings=[
                    _binding_dict("paper_reference", source_type="step_output", step_id="resolve_preference_target"),
                    _binding_dict("message", source_type="state", source_key="message"),
                ],
            ),
            self._draft_step(
                "verify_preference_update",
                verify,
                action_type="validate",
                output_key="verified_preference_update",
                reason="持久化写入后必须校验结果，再生成用户回复。",
                depends_on=["update_preference_store"],
                input_bindings=[_binding_dict("preference_action_result", source_type="step_output", step_id="update_preference_store")],
            ),
            self._draft_step(
                "synthesize_preference_response",
                synthesize,
                action_type="answer",
                output_key="final_answer",
                reason="把偏好写入结果转换为明确的用户可读反馈。",
                depends_on=["verify_preference_update"],
                input_bindings=[_binding_dict("verified_preference_update", source_type="step_output", step_id="verify_preference_update")],
            ),
        ]
        return self._make_draft(goal, "preference_action", steps)

    def _build_unclear(
        self,
        goal: Goal,
        state: AgentState,
        context: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        del state, context
        analyze = self._require_tool(tools_by_name, "analyze_ambiguity", required_tags={"clarify"})
        clarify = self._require_tool(tools_by_name, "generate_clarification", required_tags={"answer", "clarify"})
        steps = [
            self._draft_step(
                "analyze_ambiguity",
                analyze,
                action_type="clarify",
                output_key="missing_information",
                reason="不明确目标只能先分析缺失信息，不能暴露业务工具。",
                input_bindings=[_binding_dict("message", source_type="state", source_key="message")],
            ),
            self._draft_step(
                "generate_clarification",
                clarify,
                action_type="answer",
                output_key="final_answer",
                reason="生成澄清问题作为唯一用户回复。",
                depends_on=["analyze_ambiguity"],
                input_bindings=[_binding_dict("missing_information", source_type="step_output", step_id="analyze_ambiguity")],
            ),
        ]
        return self._make_draft(goal, "unclear", steps)

    def _build_unsupported(
        self,
        goal: Goal,
        state: AgentState,
        context: Mapping[str, Any],
        tools_by_name: Mapping[str, ToolSpec],
    ) -> PlanDraft:
        del state, context
        fallback = self._require_tool(tools_by_name, "generate_fallback_response", required_tags={"fallback", "answer"})
        steps = [
            self._draft_step(
                "generate_fallback_response",
                fallback,
                action_type="answer",
                output_key="final_answer",
                reason="unsupported 目标只能生成安全 fallback 回复，禁止调用业务工具。",
                input_bindings=[_binding_dict("message", source_type="state", source_key="message")],
            )
        ]
        return self._make_draft(goal, "unsupported", steps)

    def _make_draft(self, goal: Goal, plan_intent: str, steps: Sequence[PlanDraftStep]) -> PlanDraft:
        selected_tools = _dedupe_tool_names([step.tool_name for step in list(steps or [])])
        return PlanDraft(
            draft_id=f"{plan_intent}:rule-based:{_utcnow_iso()}",
            plan_intent=plan_intent,
            selected_tools=selected_tools,
            steps=list(steps or []),
            fallback_reason=self.last_debug.get("fallback_reason"),
            metadata={"source": "rule_based_tool_aware", "goal_id": goal.goal_id},
        )

    def _fallback_draft(self, goal: Goal, plan_intent: str, reason: str) -> PlanDraft:
        self.last_debug["fallback_reason"] = reason
        return PlanDraft(
            draft_id=f"{plan_intent}:fallback:{_utcnow_iso()}",
            plan_intent=plan_intent,
            selected_tools=[],
            steps=[],
            fallback_reason=reason,
            metadata={"source": "rule_based_tool_aware", "goal_id": goal.goal_id},
        )

    def _draft_step(
        self,
        step_id: str,
        tool: ToolSpec,
        *,
        action_type: str,
        output_key: str,
        reason: str,
        input_bindings: Optional[List[Dict[str, Any]]] = None,
        depends_on: Optional[List[str]] = None,
        retry_policy: Optional[StepPolicy] = None,
        fallback_reason: Optional[str] = None,
    ) -> PlanDraftStep:
        self.last_debug["selected_steps"].append(
            {
                "step_id": step_id,
                "tool_name": tool.tool_name,
                "reason": reason,
                "side_effect_level": tool.side_effect_level,
                "requires_confirmation": tool.requires_confirmation,
            }
        )
        return PlanDraftStep(
            step_id=step_id,
            action_type=action_type,
            tool_name=tool.tool_name,
            step_reason=reason,
            input_bindings=list(input_bindings or []),
            depends_on=list(depends_on or []),
            expected_output_key=output_key,
            retry_policy=retry_policy,
            risk_level=_risk_level_for_tool(tool),
            requires_confirmation=tool.requires_confirmation,
            fallback_reason=fallback_reason,
        )

    def _require_tool(self, tools_by_name: Mapping[str, ToolSpec], preferred_name: str, *, required_tags: Set[str]) -> ToolSpec:
        tool = self._find_tool(tools_by_name, preferred_name=preferred_name, required_tags=required_tags)
        if tool is None:
            raise PlanDraftPlanningError(
                f"missing required tool for {preferred_name}; required_tags={sorted(required_tags)}"
            )
        return tool

    def _optional_tool(self, tools_by_name: Mapping[str, ToolSpec], *, preferred_name: str, required_tags: Set[str]) -> Optional[ToolSpec]:
        tool = self._find_tool(tools_by_name, preferred_name=preferred_name, required_tags=required_tags)
        if tool is None:
            self.last_debug["skipped_tools"].append(
                {
                    "tool_name": preferred_name,
                    "reason": f"candidate with tags {sorted(required_tags)} not available",
                }
            )
        return tool

    def _find_tool(self, tools_by_name: Mapping[str, ToolSpec], *, preferred_name: str, required_tags: Set[str]) -> Optional[ToolSpec]:
        preferred = tools_by_name.get(preferred_name)
        if preferred is not None and required_tags.intersection(set(preferred.capability_tags or [])):
            return preferred
        # 当前执行器按 tool_name 绑定具体实现；必要步骤不能仅靠通用 tag 任意替换，否则会破坏输入输出契约。
        return None

    def _skip_step(self, step_id: str, reason: str) -> None:
        self.last_debug["skipped_steps"].append({"step_id": step_id, "reason": reason})


class PlanDraftConverter:
    """把不可信 PlanDraft 转换为当前执行器可消费的 ExecutablePlan。"""

    def __init__(self, tool_registry: ToolRegistry = PLANNER_TOOL_REGISTRY) -> None:
        self.tool_registry = tool_registry

    def convert(
        self,
        draft: PlanDraft,
        goal: Goal,
        *,
        allowed_tool_names: Optional[Iterable[str]] = None,
    ) -> ExecutablePlan:
        allowed_tools = {str(item).strip() for item in allowed_tool_names or [] if str(item).strip()}
        self._validate_selected_tools(draft, allowed_tools)

        steps: List[PlanStep] = []
        seen_step_ids: Set[str] = set()
        seen_output_keys: Dict[str, str] = {}

        for draft_step in list(draft.steps or []):
            step_id = str(draft_step.step_id or "").strip()
            if not step_id:
                raise PlanDraftConversionError("PlanDraft step_id cannot be empty")
            if step_id in seen_step_ids:
                raise PlanDraftConversionError(f"Duplicate draft step_id: {step_id}")
            seen_step_ids.add(step_id)

            tool_name = self._validate_tool_name(draft_step.tool_name, allowed_tools)
            output_key = str(draft_step.expected_output_key or "").strip() or None
            if output_key:
                if output_key in seen_output_keys:
                    raise PlanDraftConversionError(
                        f"Duplicate draft output_key {output_key}: used by {seen_output_keys[output_key]} and {step_id}"
                    )
                seen_output_keys[output_key] = step_id

        for draft_step in list(draft.steps or []):
            for dependency_step_id in list(draft_step.depends_on or []):
                if dependency_step_id not in seen_step_ids:
                    raise PlanDraftConversionError(
                        f"Draft step {draft_step.step_id} depends on missing step {dependency_step_id}"
                    )

        for draft_step in list(draft.steps or []):
            steps.append(self._convert_step(draft_step, known_step_ids=seen_step_ids, allowed_tool_names=allowed_tools))

        depended_ids = {dependency for step in steps for dependency in list(step.depends_on or [])}
        step_ids = [step.step_id for step in steps]
        return ExecutablePlan(
            plan_id=f"{goal.goal_type or 'unknown'}:tool-aware:{_utcnow_iso()}",
            goal=goal,
            steps=steps,
            entry_step_ids=[step.step_id for step in steps if not step.depends_on],
            final_step_ids=[step_id for step_id in step_ids if step_id not in depended_ids],
            metadata={
                "planner_version": "tool-aware-draft-v1",
                "draft_id": draft.draft_id,
                "plan_intent": draft.plan_intent,
                "selected_tools": list(draft.selected_tools or []),
                "built_at": _utcnow_iso(),
            },
        )

    def _validate_selected_tools(self, draft: PlanDraft, allowed_tool_names: Set[str]) -> None:
        for tool_name in list(draft.selected_tools or []):
            self._validate_tool_name(tool_name, allowed_tool_names)

    def _validate_tool_name(self, tool_name: str, allowed_tool_names: Set[str]) -> str:
        try:
            normalized_tool_name = self.tool_registry.validate_tool_name(tool_name)
        except ValueError as exc:
            raise PlanDraftConversionError(str(exc)) from exc
        if allowed_tool_names and normalized_tool_name not in allowed_tool_names:
            raise PlanDraftConversionError(f"Tool {normalized_tool_name} is not in selected candidate tools")
        return normalized_tool_name

    def _convert_step(
        self,
        draft_step: PlanDraftStep,
        *,
        known_step_ids: Set[str],
        allowed_tool_names: Set[str],
    ) -> PlanStep:
        tool_name = self._validate_tool_name(draft_step.tool_name, allowed_tool_names)
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            raise PlanDraftConversionError(f"Unknown planner tool: {tool_name}")
        input_bindings = self._validate_input_bindings(draft_step, tool=tool, known_step_ids=known_step_ids)

        confirmation_policy = None
        if tool.requires_confirmation or draft_step.requires_confirmation:
            # 草稿只声明“需要确认”，真正的确认策略在转换时统一补齐，避免未校验草稿绕过确认门。
            confirmation_policy = StepPolicy(
                policy_type="confirmation",
                mode="explicit_user_confirmation_required",
                requires_confirmation=True,
                note=f"{tool_name} requires confirmation before execution",
            )

        return PlanStep(
            step_id=str(draft_step.step_id).strip(),
            action_type=str(draft_step.action_type or "").strip(),
            tool_name=tool_name,
            tool=tool,
            input_bindings=input_bindings,
            output_key=str(draft_step.expected_output_key or "").strip() or None,
            depends_on=list(draft_step.depends_on or []),
            condition=StepCondition(condition_type="always"),
            retry_policy=draft_step.retry_policy,
            confirmation_policy=confirmation_policy,
            side_effect_level=tool.side_effect_level,
            status="pending",
        )

    def _validate_input_bindings(self, draft_step: PlanDraftStep, *, tool: ToolSpec, known_step_ids: Set[str]) -> List[StepInputBinding]:
        bindings: List[StepInputBinding] = []
        for raw_binding in list(draft_step.input_bindings or []):
            binding = StepInputBinding.model_validate(
                raw_binding.model_dump() if hasattr(raw_binding, "model_dump") else raw_binding
            )
            if not str(binding.input_key or "").strip():
                raise PlanDraftConversionError(f"Draft step {draft_step.step_id} has empty input_key")
            if binding.source_type == "step_output" and str(binding.step_id or "").strip() not in known_step_ids:
                raise PlanDraftConversionError(
                    f"Draft step {draft_step.step_id} input {binding.input_key} references missing step {binding.step_id}"
                )
            if binding.source_type == "literal" and binding.value is None and binding.required:
                raise PlanDraftConversionError(
                    f"Draft step {draft_step.step_id} input {binding.input_key} uses empty required literal"
                )
            bindings.append(binding)
        if tool.input_schema and not bindings:
            # 当前执行器允许部分输入键使用业务别名，因此这里只拦截“完全没有绑定”的草稿；
            # 具体键名是否可解析继续交给 Executor 的既有绑定解析逻辑处理。
            raise PlanDraftConversionError(
                f"Draft step {draft_step.step_id} missing required input bindings"
            )
        return bindings


def plan_to_draft(plan: ExecutablePlan, *, selected_tools: Sequence[str], fallback_reason: Optional[str] = None) -> PlanDraft:
    """把固定模板计划投影成 PlanDraft，作为当前阶段的规则型 draft 生成器。

    这样启用 Tool-Aware Planner 后仍先经过不可信草稿和转换边界，但不会改变现有模板行为。
    """
    used_tool_names = _dedupe_tool_names([step.tool_name for step in list(plan.steps or [])])
    return PlanDraft(
        draft_id=f"{plan.goal.goal_type or 'unknown'}:draft:{_utcnow_iso()}",
        plan_intent=plan.goal.goal_type,
        selected_tools=used_tool_names or _dedupe_tool_names(selected_tools),
        steps=[
            PlanDraftStep(
                step_id=step.step_id,
                action_type=step.action_type,
                tool_name=step.tool_name,
                step_reason=f"fixed template step for {plan.goal.goal_type}",
                input_bindings=list(step.input_bindings or []),
                depends_on=list(step.depends_on or []),
                expected_output_key=step.output_key,
                risk_level="high" if step.side_effect_level == "persistent_write" else ("medium" if step.side_effect_level == "external_call" else "low"),
                requires_confirmation=bool(step.confirmation_policy and step.confirmation_policy.requires_confirmation),
                fallback_reason=fallback_reason,
            )
            for step in list(plan.steps or [])
        ],
        fallback_reason=fallback_reason,
        metadata={"source": "fixed_template_projection"},
    )


def _dedupe_tool_names(tool_names: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for tool_name in tool_names or []:
        text = str(tool_name or "").strip()
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _binding_dict(
    input_key: str,
    *,
    source_type: str,
    source_key: Optional[str] = None,
    step_id: Optional[str] = None,
    value: Any = None,
    required: bool = True,
) -> Dict[str, Any]:
    return {
        "input_key": input_key,
        "source_type": source_type,
        "source_key": source_key,
        "step_id": step_id,
        "value": value,
        "required": required,
    }


def _has_any_context_key(state: AgentState, keys: Set[str]) -> bool:
    context = state.context if isinstance(state.context, Mapping) else {}
    for key in keys:
        value = context.get(key)
        if isinstance(value, Mapping) and value:
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def _get_selected_paper_hint(context: Mapping[str, Any]) -> Optional[str]:
    selected_paper = context.get("selected_paper")
    if isinstance(selected_paper, Mapping):
        for key in ("title", "arxiv_id", "paper_id"):
            text = str(selected_paper.get(key) or "").strip()
            if text:
                return text
    for key in ("selected_paper_title", "selected_paper_id"):
        text = str(context.get(key) or "").strip()
        if text:
            return text
    return None


def _has_candidate_paper_context(context: Mapping[str, Any]) -> bool:
    for key in ("papers", "last_papers", "candidate_papers"):
        value = context.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _preference_target_is_clear(state: AgentState) -> bool:
    context = state.context if isinstance(state.context, Mapping) else {}
    return bool(_get_selected_paper_hint(context) or _has_candidate_paper_context(context) or _message_has_paper_hint(state.message))


def _compact_mapping(mapping: Mapping[str, Any], *, limit: int = 12) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for index, (key, value) in enumerate(mapping.items()):
        if index >= limit:
            compact["_truncated"] = True
            break
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[str(key)] = value
        elif isinstance(value, Mapping):
            compact[str(key)] = {"type": "object", "keys": list(value.keys())[:8]}
        elif isinstance(value, list):
            compact[str(key)] = {"type": "list", "count": len(value)}
        else:
            compact[str(key)] = {"type": type(value).__name__}
    return compact


def _safe_debug_text(text: Any, *, limit: int = 4000) -> Optional[str]:
    if text is None:
        return None
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _message_has_paper_hint(message: Optional[str]) -> bool:
    text = str(message or "").strip()
    if not text:
        return False
    if "http" in text.lower() or "arxiv" in text.lower():
        return True
    return bool(re.search(r"\b\d{4}\.\d{4,5}\b", text))


def _risk_level_for_tool(tool: ToolSpec) -> str:
    if tool.side_effect_level == "persistent_write":
        return "high"
    if tool.side_effect_level == "external_call" or tool.requires_confirmation:
        return "medium"
    return "low"


def _side_effect_rank(side_effect_level: str) -> int:
    return {
        "none": 0,
        "session_write": 1,
        "external_call": 2,
        "persistent_write": 3,
    }.get(str(side_effect_level or ""), 4)


__all__ = [
    "PlanDraftConversionError",
    "PlanDraftPlanningError",
    "PlanDraftConverter",
    "LLMPlanDraftError",
    "LLMPlanDraftGenerator",
    "RuleBasedToolAwarePlanBuilder",
    "ToolCandidateSelector",
    "plan_to_draft",
]
