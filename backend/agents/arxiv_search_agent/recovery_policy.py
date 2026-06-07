from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

from .schemas import FailureCategory, ObservationResult, PlanRuntime, PlanStep, RecoveryCandidate
from .state import AgentState


class RecoveryPolicy(Protocol):
    """把 observation 分类映射成候选恢复动作，不在这里直接修改 plan。"""

    policy_name: str

    def applies_to(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> bool:
        ...

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        ...


class _BaseRecoveryPolicy:
    policy_name = "base_recovery_policy"
    failure_categories: Sequence[FailureCategory] = ()
    tool_names: Sequence[str] = ()

    def applies_to(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> bool:
        del runtime, state
        if observation.failure_category not in set(self.failure_categories):
            return False
        return not self.tool_names or step.tool_name in set(self.tool_names)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del step, observation, runtime, state
        return []


class SearchEmptyRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "search_empty_recovery_policy"
    failure_categories = ("search_empty",)
    tool_names = ("search_arxiv",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:rewrite_query_for_empty_search",
                action_type="patch_plan",
                failure_category=observation.failure_category or "search_empty",
                priority=90,
                confidence=0.9,
                reason="搜索没有返回可用论文，优先通过 query rewrite 放宽或改写检索条件。",
                target_step_id=step.step_id,
                required_tools=["rewrite_arxiv_query", "search_arxiv", "validate_arxiv_results"],
                patch_strategy="rewrite_search_chain",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=2,
                expected_effect="插入 rewrite -> search -> validate 链路，后续步骤改用重试搜索结果。",
                fallback_if_failed="fallback_answer",
            )
        ]


class SearchLowConfidenceRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "search_low_confidence_recovery_policy"
    failure_categories = ("search_low_confidence",)
    tool_names = ("validate_arxiv_results",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:rewrite_query_for_low_confidence_search",
                action_type="patch_plan",
                failure_category=observation.failure_category or "search_low_confidence",
                priority=80,
                confidence=0.85,
                reason="搜索结果质量不足，使用改写后的查询重新检索以提升相关性。",
                target_step_id=step.step_id,
                required_tools=["rewrite_arxiv_query", "search_arxiv", "validate_arxiv_results"],
                patch_strategy="rewrite_search_chain",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=2,
                expected_effect="插入新的检索链路，并把后续依赖切换到低质量结果的替代输出。",
                fallback_if_failed="fallback_answer",
            )
        ]


class PaperIndexMissingRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "paper_index_missing_recovery_policy"
    failure_categories = ("paper_index_missing",)
    tool_names = ("check_paper_index",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:confirm_and_build_paper_index",
                action_type="patch_plan",
                failure_category=observation.failure_category or "paper_index_missing",
                priority=90,
                confidence=0.95,
                reason="论文 QA 索引缺失，需要先征得用户确认，再执行解析和建索引这类外部副作用步骤。",
                target_step_id=step.step_id,
                required_tools=["request_confirmation", "parse_and_index_paper"],
                patch_strategy="inject_index_confirmation_chain",
                risk_level="high",
                requires_confirmation=True,
                max_attempts=1,
                expected_effect="插入确认步骤和索引构建步骤，确认通过后再继续论文问答。",
                fallback_if_failed="abort_with_error",
            )
        ]


class EmptyUserProfileRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "empty_user_profile_recovery_policy"
    failure_categories = ("empty_user_profile", "insufficient_context")
    tool_names = ("load_user_profile",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:downgrade_to_message_recommendation",
                action_type="patch_plan",
                failure_category=observation.failure_category or "empty_user_profile",
                priority=70,
                confidence=0.8,
                reason="用户画像为空时不能继续依赖个性化画像，降级为当前消息驱动的推荐链路。",
                target_step_id=step.step_id,
                required_tools=["load_candidate_papers", "generate_recommendations"],
                patch_strategy="downgrade_recommendation_chain",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="移除后续推荐步骤对空画像的强依赖，保留基于当前请求的推荐能力。",
                fallback_if_failed="fallback_answer",
            )
        ]


class QaNoAnswerRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "qa_no_answer_recovery_policy"
    failure_categories = ("qa_no_answer",)
    tool_names = ("answer_paper_question",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:retry_qa_with_more_top_k",
                action_type="retry_step",
                failure_category=observation.failure_category or "qa_no_answer",
                priority=85,
                confidence=0.82,
                reason="QA 返回空答案时，先用更宽的证据召回参数重试一次，避免过早 fallback。",
                target_step_id=step.step_id,
                required_tools=["answer_paper_question"],
                patch_strategy="retry_step_with_adjusted_arguments",
                strategy_payload={
                    "strategy_name": "retry_qa_with_more_top_k",
                    "retrieval_top_k": 30,
                    "evidence_query": "expand_question_terms",
                    "grounding_required": True,
                },
                risk_level="medium",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="用 qa_strategy 增加 top_k 并强化证据查询后重新执行问答。",
                fallback_if_failed="fallback_answer",
            ),
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:ask_clarification_for_qa",
                action_type="ask_clarification",
                failure_category=observation.failure_category or "qa_no_answer",
                priority=35,
                confidence=0.65,
                reason="多次 QA 无答案后应让用户缩小问题范围，而不是继续重复检索。",
                target_step_id=step.step_id,
                required_tools=["generate_clarification"],
                patch_strategy="clarification_chain",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="请求用户补充更明确的问题或章节范围。",
                fallback_if_failed="fallback_answer",
            ),
        ]


class QaNoSourcesRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "qa_no_sources_recovery_policy"
    failure_categories = ("qa_no_sources", "qa_low_grounding")
    tool_names = ("answer_paper_question",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:force_source_grounded_answer",
                action_type="retry_step",
                failure_category=observation.failure_category or "qa_no_sources",
                priority=88,
                confidence=0.86,
                reason="答案缺少 sources 时，必须优先尝试 source-grounded 策略，而不是把无证据答案当作完成。",
                target_step_id=step.step_id,
                required_tools=["answer_paper_question"],
                patch_strategy="retry_step_with_adjusted_arguments",
                strategy_payload={
                    "strategy_name": "force_source_grounded_answer",
                    "force_sources": True,
                    "retrieval_top_k": 30,
                    "route": "keyword_and_section",
                    "grounding_required": True,
                },
                risk_level="medium",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="强制要求带来源回答，并提高检索 top_k。",
                fallback_if_failed="fallback_answer",
            ),
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:fallback_with_grounding_warning",
                action_type="fallback_answer",
                failure_category=observation.failure_category or "qa_no_sources",
                priority=20,
                confidence=0.55,
                reason="证据补救失败后保留回答但明确标记 grounding 不足。",
                target_step_id=step.step_id,
                required_tools=[],
                patch_strategy="fallback_answer",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="返回带 warning 的回答，避免伪装成已充分引用。",
                fallback_if_failed="fallback_answer",
            ),
        ]


class PaperIndexStaleOrCorruptedRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "paper_index_stale_or_corrupted_recovery_policy"
    failure_categories = ("paper_index_stale", "paper_index_corrupted")
    tool_names = ("check_paper_index",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        has_last_good_index = bool((observation.evidence or {}).get("has_last_good_index"))
        candidates: List[RecoveryCandidate] = []
        if has_last_good_index:
            candidates.append(
                RecoveryCandidate(
                    candidate_id=f"{step.step_id}:use_last_good_index",
                    action_type="patch_plan",
                    failure_category=observation.failure_category or "paper_index_stale",
                    priority=92,
                    confidence=0.85,
                    reason="当前索引不可用但存在 last good index，应优先使用旧索引，不能自动删除或重建。",
                    target_step_id=step.step_id,
                    required_tools=["answer_paper_question"],
                    patch_strategy="use_last_good_index",
                    strategy_payload={"index_strategy": {"mode": "use_last_good_index"}},
                    risk_level="low",
                    requires_confirmation=False,
                    max_attempts=1,
                    expected_effect="后续 QA 使用 last good index 继续回答。",
                    fallback_if_failed="fallback_answer",
                )
            )
        candidates.append(
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:request_confirmation_to_rebuild_index",
                action_type="patch_plan",
                failure_category=observation.failure_category or "paper_index_corrupted",
                priority=75,
                confidence=0.8,
                reason="重建索引是高成本 external_call，必须通过确认链路触发。",
                target_step_id=step.step_id,
                required_tools=["request_confirmation", "parse_and_index_paper"],
                patch_strategy="inject_index_confirmation_chain",
                risk_level="high",
                requires_confirmation=True,
                max_attempts=1,
                expected_effect="请求用户确认后重建索引。",
                fallback_if_failed="abort_with_error",
            )
        )
        candidates.append(
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:fallback_to_metadata_answer",
                action_type="fallback_answer",
                failure_category=observation.failure_category or "paper_index_corrupted",
                priority=20,
                confidence=0.5,
                reason="没有可用索引且不能重建时，降级为基于元数据的说明。",
                target_step_id=step.step_id,
                required_tools=[],
                patch_strategy="fallback_answer",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="明确说明索引不可用，避免假装完成全文 QA。",
                fallback_if_failed="fallback_answer",
            )
        )
        return candidates


class PreferenceTargetMissingRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "preference_target_missing_recovery_policy"
    failure_categories = ("preference_target_missing", "ambiguous_user_request")
    tool_names = ("update_preference_store", "resolve_preference_target")

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:ask_user_to_specify_paper",
                action_type="ask_clarification",
                failure_category=observation.failure_category or "preference_target_missing",
                priority=90,
                confidence=0.9,
                reason="偏好写入目标不明确时不能猜测论文，必须先让用户指定目标。",
                target_step_id=step.step_id,
                required_tools=["generate_clarification"],
                patch_strategy="clarification_chain",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="返回澄清问题，并明确偏好尚未写入。",
                fallback_if_failed="fallback_answer",
            )
        ]


class ToolTimeoutRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "tool_timeout_recovery_policy"
    failure_categories = ("tool_timeout",)

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del observation, runtime, state
        if step.tool_name in {"search_arxiv", "answer_paper_question", "generate_recommendations"}:
            return [
                RecoveryCandidate(
                    candidate_id=f"{step.step_id}:retry_with_backoff",
                    action_type="retry_step",
                    failure_category="tool_timeout",
                    priority=80,
                    confidence=0.75,
                    reason="外部读型工具超时可以有限重试一次，但不能无限等待。",
                    target_step_id=step.step_id,
                    required_tools=[step.tool_name],
                    patch_strategy="retry_step_with_adjusted_arguments",
                    strategy_payload={"retry_mode": "backoff_once", "timeout_recovery": True},
                    risk_level="medium",
                    requires_confirmation=False,
                    max_attempts=1,
                    expected_effect="把目标步骤恢复为 pending，并附加 timeout retry 策略。",
                    fallback_if_failed="fallback_answer",
                )
            ]
        if step.tool_name == "parse_and_index_paper":
            return [
                RecoveryCandidate(
                    candidate_id=f"{step.step_id}:convert_to_async_job",
                    action_type="fallback_answer",
                    failure_category="tool_timeout",
                    priority=75,
                    confidence=0.7,
                    reason="索引构建超时不应阻塞当前对话，应转为可恢复的后台任务状态或提示稍后查询。",
                    target_step_id=step.step_id,
                    required_tools=[],
                    patch_strategy="fallback_answer",
                    risk_level="medium",
                    requires_confirmation=False,
                    max_attempts=1,
                    expected_effect="返回后台任务/稍后重试提示。",
                    fallback_if_failed="fallback_answer",
                )
            ]
        if step.tool_name == "update_preference_store":
            return [
                RecoveryCandidate(
                    candidate_id=f"{step.step_id}:preference_timeout_fallback",
                    action_type="fallback_answer",
                    failure_category="tool_timeout",
                    priority=90,
                    confidence=0.9,
                    reason="偏好写入是 persistent_write，超时后不能静默重复写入，必须明确尚未确认写入成功。",
                    target_step_id=step.step_id,
                    required_tools=[],
                    patch_strategy="fallback_answer",
                    risk_level="low",
                    requires_confirmation=False,
                    max_attempts=1,
                    expected_effect="提示用户稍后重试或检查状态，不重复写入。",
                    fallback_if_failed="fallback_answer",
                )
            ]
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:timeout_fallback_message",
                action_type="fallback_answer",
                failure_category="tool_timeout",
                priority=30,
                confidence=0.6,
                reason="未知工具超时，降级为明确的 fallback 提示。",
                target_step_id=step.step_id,
                required_tools=[],
                patch_strategy="fallback_answer",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="返回可恢复错误说明。",
                fallback_if_failed="fallback_answer",
            )
        ]


class AmbiguousRequestRecoveryPolicy(_BaseRecoveryPolicy):
    policy_name = "ambiguous_request_recovery_policy"
    failure_categories = ("ambiguous_user_request", "insufficient_context")

    def build_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        del runtime, state
        return [
            RecoveryCandidate(
                candidate_id=f"{step.step_id}:ask_clarification",
                action_type="ask_clarification",
                failure_category=observation.failure_category or "ambiguous_user_request",
                priority=70,
                confidence=0.75,
                reason="请求语义或上下文不足时应优先澄清，而不是猜测执行工具。",
                target_step_id=step.step_id,
                required_tools=["generate_clarification"],
                patch_strategy="clarification_chain",
                risk_level="low",
                requires_confirmation=False,
                max_attempts=1,
                expected_effect="向用户提出可回答的澄清问题。",
                fallback_if_failed="fallback_answer",
            )
        ]


class RecoveryPolicyRegistry:
    """集中注册规则型恢复策略，后续可替换 chooser 而不改 Observer 分类。"""

    def __init__(self, policies: Optional[Sequence[RecoveryPolicy]] = None) -> None:
        self._policies: List[RecoveryPolicy] = list(policies or [])

    def register(self, policy: RecoveryPolicy) -> None:
        self._policies.append(policy)

    def get_candidates(self, *, step: PlanStep, observation: ObservationResult, runtime: PlanRuntime, state: Optional[AgentState] = None) -> List[RecoveryCandidate]:
        candidates: List[RecoveryCandidate] = []
        for policy in self._policies:
            if policy.applies_to(step=step, observation=observation, runtime=runtime, state=state):
                candidates.extend(policy.build_candidates(step=step, observation=observation, runtime=runtime, state=state))
        return sorted(candidates, key=lambda item: (-item.priority, -item.confidence, item.candidate_id))


DEFAULT_RECOVERY_POLICY_REGISTRY = RecoveryPolicyRegistry(
    policies=[
        SearchEmptyRecoveryPolicy(),
        SearchLowConfidenceRecoveryPolicy(),
        PaperIndexMissingRecoveryPolicy(),
        EmptyUserProfileRecoveryPolicy(),
        QaNoAnswerRecoveryPolicy(),
        QaNoSourcesRecoveryPolicy(),
        PaperIndexStaleOrCorruptedRecoveryPolicy(),
        PreferenceTargetMissingRecoveryPolicy(),
        ToolTimeoutRecoveryPolicy(),
        AmbiguousRequestRecoveryPolicy(),
    ]
)


__all__ = [
    "DEFAULT_RECOVERY_POLICY_REGISTRY",
    "EmptyUserProfileRecoveryPolicy",
    "PaperIndexMissingRecoveryPolicy",
    "PaperIndexStaleOrCorruptedRecoveryPolicy",
    "PreferenceTargetMissingRecoveryPolicy",
    "QaNoAnswerRecoveryPolicy",
    "QaNoSourcesRecoveryPolicy",
    "RecoveryPolicy",
    "RecoveryPolicyRegistry",
    "SearchEmptyRecoveryPolicy",
    "SearchLowConfidenceRecoveryPolicy",
    "ToolTimeoutRecoveryPolicy",
    "AmbiguousRequestRecoveryPolicy",
]
