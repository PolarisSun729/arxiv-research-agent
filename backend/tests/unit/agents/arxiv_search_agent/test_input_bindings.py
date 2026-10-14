from agents.arxiv_search_agent.execution.bindings import InputBindingResolver
from agents.arxiv_search_agent.schemas import ExecutablePlan, Goal, PlanRuntime, PlanStep, StepInputBinding, ToolSpec
from agents.arxiv_search_agent.state import AgentState


def _runtime(outputs: dict) -> tuple[PlanStep, PlanRuntime, AgentState]:
    personalize = PlanStep(
        step_id="step_5_personalize",
        action_type="rerank",
        tool_name="personalize_paper_results",
        tool=ToolSpec(tool_name="personalize_paper_results"),
        output_key="ranked_papers",
    )
    synthesize = PlanStep(
        step_id="step_6_synthesize",
        action_type="answer",
        tool_name="synthesize_arxiv_response",
        tool=ToolSpec(tool_name="synthesize_arxiv_response"),
        # LLM 草稿常把这类绑定标成 required，这里复现远端出错时的计划形态。
        input_bindings=[StepInputBinding(input_key="ranked_papers", source_type="step_output", step_id="step_5_personalize", required=True)],
        output_key="final_answer",
    )
    goal = Goal(goal_id="g1", goal_type="arxiv_search", user_request="search")
    plan = ExecutablePlan(
        plan_id="p1",
        goal=goal,
        steps=[personalize, synthesize],
        entry_step_ids=["step_5_personalize"],
        final_step_ids=["step_6_synthesize"],
    )
    runtime = PlanRuntime(goal=goal, plan=plan, outputs=outputs)
    state = AgentState(user_id="u1", session_id="s1", intent="arxiv_search", message="search")
    return synthesize, runtime, state


def test_required_step_output_accepts_recorded_empty_list() -> None:
    step, runtime, state = _runtime({"ranked_papers": []})

    resolved, missing = InputBindingResolver().resolve(step, runtime, state)

    assert missing is None
    assert resolved == {"ranked_papers": []}


def test_required_step_output_still_missing_when_upstream_never_ran() -> None:
    step, runtime, state = _runtime({})

    resolved, missing = InputBindingResolver().resolve(step, runtime, state)

    assert missing == "ranked_papers"
    assert resolved == {}
