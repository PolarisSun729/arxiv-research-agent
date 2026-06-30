from __future__ import annotations

import json
import importlib

from tests.helpers.agent_runtime import load_agent_test_modules

_MODULES = load_agent_test_modules()
schemas = _MODULES["schemas"]
state_module = _MODULES["state_module"]
graph_module = _MODULES["graph_module"]

profile_module = importlib.import_module("backend.agents.arxiv_search_agent.research_task_profile")
planner_context_module = importlib.import_module("backend.agents.arxiv_search_agent.planner_context")
planner_module = importlib.import_module("backend.agents.arxiv_search_agent.planner")

AgentState = state_module.AgentState
ArxivSearchSpec = schemas.ArxivSearchSpec
Goal = schemas.Goal
ResearchTaskProfile = schemas.ResearchTaskProfile
build_research_task_profile = profile_module.build_research_task_profile
research_task_profile_debug = profile_module.research_task_profile_debug
build_goal_node = graph_module.build_goal_node
GoalBuilder = planner_module.GoalBuilder
build_planner_context = planner_context_module.build_planner_context
planner_context_debug = planner_context_module.planner_context_debug


def _goal_for(state: AgentState) -> Goal:
    return GoalBuilder.from_state(state)


# ---------------------------------------------------------------------------
# 同一个 intent 应能产出不同 research_task_type
# ---------------------------------------------------------------------------
def test_plain_search_maps_to_direction_exploration() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="找几篇 RAG 论文",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.research_task_type == "direction_exploration"
    assert profile.task_object.object_type == "topic"
    # Profile 必须携带中间产物与证据需求，而不仅是一个 label。
    assert "candidate_paper_set" in profile.intermediate_artifacts
    assert "abstract" in profile.evidence_requirements


def test_comparison_search_maps_to_multi_paper_comparison() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="找最近的 RAG 论文并比较它们的方法差异",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.research_task_type == "multi_paper_comparison"
    # 多论文比较需要方法卡片、实验信息和对比矩阵，以及更深的证据。
    assert "comparison_matrix" in profile.intermediate_artifacts
    assert "method_cards" in profile.intermediate_artifacts
    assert "method_chunk" in profile.evidence_requirements
    assert "table_evidence" in profile.evidence_requirements


def test_reading_plan_search_maps_to_reading_planning() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="我想入门 RAG，给我一个阅读顺序",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=8),
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.research_task_type == "reading_planning"
    assert "reading_order" in profile.intermediate_artifacts


def test_research_gap_search_maps_to_research_gap_analysis() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="RAG 方向目前有哪些研究空白和未解决的问题",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=8),
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.research_task_type == "research_gap_analysis"
    assert "comparison_matrix" in profile.intermediate_artifacts


# ---------------------------------------------------------------------------
# 其它 intent → 对应科研任务类型
# ---------------------------------------------------------------------------
def test_paper_qa_maps_to_single_paper_deep_read() -> None:
    state = AgentState(
        intent="paper_qa",
        message="这篇论文的核心方法是什么？",
        context={"selected_paper": {"title": "Attention Is All You Need", "arxiv_id": "1706.03762"}},
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.research_task_type == "single_paper_deep_read"
    assert profile.task_object.object_type == "paper"
    assert "1706.03762" in profile.task_object.paper_refs
    assert "method_chunk" in profile.evidence_requirements


def test_recommendation_maps_to_personalized_recommendation() -> None:
    state = AgentState(
        intent="recommendation",
        message="给我推荐几篇论文",
        context={"user_memory_summary": {"profile": {"topics": ["agents", "memory"]}}},
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.research_task_type == "personalized_recommendation"
    assert profile.task_object.object_type == "user_profile"
    assert profile.constraints.combine_with_interest is True
    # 结合个人兴趣时证据需求应包含 user_profile_evidence。
    assert "user_profile_evidence" in profile.evidence_requirements


# ---------------------------------------------------------------------------
# 非科研任务场景不产出 Profile，且不破坏既有流程
# ---------------------------------------------------------------------------
def test_unclear_and_unsupported_and_preference_have_no_profile() -> None:
    for intent in ("unclear", "unsupported", "preference_action"):
        state = AgentState(intent=intent, message="帮我做点什么")
        profile = build_research_task_profile(goal=_goal_for(state), state=state)
        assert profile is None, f"intent={intent} should not yield a research task profile"


# ---------------------------------------------------------------------------
# 显式约束应被捕获
# ---------------------------------------------------------------------------
def test_constraints_capture_time_count_and_fields() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="找最近 RAG 论文",
        search_spec=ArxivSearchSpec(
            intent="arxiv_search",
            query="RAG",
            categories=["cs.CL", "cs.IR"],
            submitted_days_ago=30,
            max_results=7,
        ),
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.constraints.time_range == "submitted_days_ago<=30"
    assert profile.constraints.max_count == 7
    assert profile.constraints.research_fields == ["cs.CL", "cs.IR"]


# ---------------------------------------------------------------------------
# build_goal 节点：Profile 进入 state 与 debug，且不破坏既有 goal 流程
# ---------------------------------------------------------------------------
def test_build_goal_node_attaches_profile_and_debug() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="找最近的 RAG 论文并比较它们的方法差异",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
    )
    result = build_goal_node(state)

    # 既有 goal 流程不受影响。
    assert result.goal is not None
    assert result.goal.goal_type == "arxiv_search"
    assert isinstance(result.debug.get("goal"), dict)

    # Profile 进入一等状态。
    assert result.research_task_profile is not None
    assert result.research_task_profile.research_task_type == "multi_paper_comparison"

    # debug 应能展示 Goal/intent/Profile 的对应关系。
    profile_debug = result.debug.get("research_task_profile")
    assert isinstance(profile_debug, dict)
    assert profile_debug["research_task_type"] == "multi_paper_comparison"
    assert profile_debug["intent"] == "arxiv_search"
    assert profile_debug["goal_type"] == "arxiv_search"
    assert "comparison_matrix" in profile_debug["intermediate_artifacts"]
    assert profile_debug["evidence_requirements"]

    # build_goal step trace 应记录 research_task_type。
    build_goal_step = next(step for step in result.steps if step.step == "build_goal")
    assert build_goal_step.outputs.get("research_task_type") == "multi_paper_comparison"


def test_build_goal_node_unsupported_has_no_profile_but_keeps_goal() -> None:
    state = AgentState(intent="unsupported", message="帮我画一张海报")
    result = build_goal_node(state)

    assert result.goal is not None
    assert result.goal.goal_type == "unsupported"
    assert result.research_task_profile is None
    # debug 中 profile 摘要为 None，但不应抛错或缺键。
    assert result.debug.get("research_task_profile") is None


# ---------------------------------------------------------------------------
# debug 摘要本身：携带完整对应关系而不仅是 label
# ---------------------------------------------------------------------------
def test_research_task_profile_debug_exposes_full_correspondence() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="比较这几篇 RAG 论文",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
    )
    profile = build_research_task_profile(goal=_goal_for(state), state=state)
    payload = research_task_profile_debug(profile)

    assert payload is not None
    for key in (
        "research_task_type",
        "intent",
        "goal_type",
        "task_object",
        "constraints",
        "intermediate_artifacts",
        "evidence_requirements",
        "confidence",
        "classification_basis",
        "needs_clarification",
    ):
        assert key in payload

    assert research_task_profile_debug(None) is None


# ---------------------------------------------------------------------------
# Profile 进入 PlannerContext，并出现在 planner debug 中
# ---------------------------------------------------------------------------
def test_planner_context_carries_research_task_profile() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="找最近的 RAG 论文并比较它们的方法差异",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
    )
    goal = _goal_for(state)
    state.research_task_profile = build_research_task_profile(goal=goal, state=state)

    planner_context = build_planner_context(goal=goal, state=state)
    assert planner_context.research_task_profile is not None

    debug = planner_context_debug(planner_context)
    profile_debug = debug.get("research_task_profile")
    assert isinstance(profile_debug, dict)
    assert profile_debug["research_task_type"] == "multi_paper_comparison"
    assert profile_debug["intermediate_artifacts"]


class _FakeGenerationService:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        return json.dumps(self.payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Step 2：混合式分类路线（规则高置信 + LLM 语义分类 + 本地仲裁）
# ---------------------------------------------------------------------------
def test_rule_high_confidence_does_not_call_llm_classifier() -> None:
    service = _FakeGenerationService({"primary_task_type": "direction_exploration"})
    state = AgentState(
        intent="arxiv_search",
        message="比较最近 RAG 论文的方法差异",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
    )

    profile = build_research_task_profile(goal=_goal_for(state), state=state, generation_service=service)

    assert profile is not None
    assert service.calls == 0
    assert profile.research_task_type == "multi_paper_comparison"
    assert profile.source in {"rule_high_confidence", "local_arbitration_adjusted"}
    assert profile.classification_trace["rule"]["matched"] is True
    assert profile.classification_trace["llm"]["attempted"] is False


def test_ambiguous_research_request_uses_llm_semantic_profile() -> None:
    service = _FakeGenerationService(
        {
            "primary_task_type": "direction_exploration",
            "secondary_task_types": ["reading_planning"],
            "task_object": {
                "object_type": "topic",
                "topic": "retrieval augmented generation",
                "paper_refs": [],
                "description": "explore recent notable work",
            },
            "intermediate_artifacts": ["candidate_paper_set", "representative_paper_set"],
            "evidence_requirements": ["metadata", "abstract"],
            "confidence": 0.71,
            "classification_basis": "用户询问这个方向最近值得关注的工作，属于方向探索，可附带阅读规划",
            "missing_context": [],
        }
    )
    state = AgentState(
        intent="arxiv_search",
        message="这个方向最近有什么值得关注的工作",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="retrieval augmented generation", categories=["cs.CL"], max_results=8),
    )

    profile = build_research_task_profile(goal=_goal_for(state), state=state, generation_service=service)

    assert profile is not None
    assert service.calls == 1
    assert profile.research_task_type == "direction_exploration"
    assert profile.secondary_task_types == ["reading_planning"]
    assert profile.source == "llm_semantic_classifier"
    assert profile.task_object.object_type == "topic"
    assert "candidate_paper_set" in profile.intermediate_artifacts
    assert profile.classification_trace["llm"]["attempted"] is True
    assert profile.classification_trace["final"]["source"] == "llm_semantic_classifier"


def test_local_arbitration_downgrades_llm_single_paper_without_target() -> None:
    service = _FakeGenerationService(
        {
            "primary_task_type": "single_paper_deep_read",
            "secondary_task_types": [],
            "task_object": {
                "object_type": "paper",
                "topic": "RAG",
                "paper_refs": [],
                "description": "LLM over-interpreted the request as a target-paper deep read",
            },
            "intermediate_artifacts": ["method_cards", "experiment_info"],
            "evidence_requirements": ["metadata", "abstract", "method_chunk"],
            "confidence": 0.66,
            "classification_basis": "模型误把模糊方向问题解释为单篇论文深读",
            "missing_context": ["selected_paper"],
        }
    )
    state = AgentState(
        intent="arxiv_search",
        message="这个方向最近有什么值得关注的工作",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=8),
    )

    profile = build_research_task_profile(goal=_goal_for(state), state=state, generation_service=service)

    assert profile is not None
    assert service.calls == 1
    assert profile.research_task_type == "direction_exploration"
    assert profile.source == "local_arbitration_adjusted"
    assert profile.execution_readiness == "ready"
    assert any("缺少 selected_paper" in note for note in profile.arbitration_notes)
    assert profile.classification_trace["llm"]["task_type"] == "single_paper_deep_read"
    assert profile.classification_trace["final"]["research_task_type"] == "direction_exploration"


def test_personalized_recommendation_without_profile_downgrades_to_exploration() -> None:
    state = AgentState(
        intent="arxiv_search",
        message="根据我的兴趣推荐一些 RAG 论文",
        search_spec=ArxivSearchSpec(intent="arxiv_search", query="RAG", categories=["cs.CL"], max_results=5),
    )

    profile = build_research_task_profile(goal=_goal_for(state), state=state)

    assert profile is not None
    assert profile.research_task_type == "direction_exploration"
    assert profile.source == "local_arbitration_adjusted"
    assert any("未加载到用户画像" in note for note in profile.arbitration_notes)
    assert "user_profile_evidence" not in profile.evidence_requirements
