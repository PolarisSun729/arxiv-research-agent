"""计划的数据流闭合检查（设计见 notes/agent-redesign.md D4-1）。

只看数据依赖，不规定工具序列：每一步的必需输入，要么来自前面某一步的产出，要么本轮开始时
state 里已经有（project_state_types）。同一类型有多个实例时（比如两份 PaperList），
绑定按 step_id 显式指定，不按类型自动匹配。

目前只提供检查函数，还没接入 planner。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .schemas import ExecutablePlan, PlanStep
from .semantic_types import FINAL_SEMANTIC_TYPES
from .tool_registry import UNIFIED_TOOL_REGISTRY, ToolRegistry


@dataclass
class ClosureReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    # step_id -> 该步成功时产出的语义类型，供调试和后续规划 prompt 使用。
    produced_by_step: Dict[str, List[str]] = field(default_factory=dict)
    final_types: List[str] = field(default_factory=list)


def check_plan_closure(
    plan: Union[ExecutablePlan, Sequence[PlanStep]],
    available: Iterable[str],
    *,
    registry: ToolRegistry = UNIFIED_TOOL_REGISTRY,
) -> ClosureReport:
    """按步骤顺序检查数据流是否闭合。

    available 是本轮开始时已有的语义类型名，通常传 project_state_types(state) 的结果。
    规则：
    1. 来自 step_output 的绑定：源步骤必须排在前面，且它的 produces 包含本输入要的类型；
    2. 来自 state / context / search_spec / goal / literal 的绑定：类型必须在 available 里；
    3. 每个 required 的 consumes 至少有一个输入键被绑定；
    4. 计划里至少有一步产出 AnswerDraft 或 CheckedQAResult。
    """
    steps = list(plan.steps or []) if isinstance(plan, ExecutablePlan) else list(plan or [])
    state_types = set(available)
    produced_by_step: Dict[str, List[str]] = {}
    errors: List[str] = []

    for step in steps:
        contract = registry.get_contract(step.tool_name)
        if contract is None:
            errors.append(f"{step.step_id}: 未注册的工具 {step.tool_name}")
            continue
        type_by_key: Dict[str, str] = {
            key: declaration.semantic_type for declaration in contract.consumes for key in declaration.input_keys
        }
        bound_types = set()
        for binding in list(step.input_bindings or []):
            expected = type_by_key.get(binding.input_key)
            if expected is None:
                # 环境输入（message、context、qa_mode 等）不参与闭合检查。
                continue
            problem = _binding_problem(binding.source_type, binding.step_id, expected, state_types, produced_by_step)
            if problem:
                errors.append(f"{step.step_id}.{binding.input_key}: {problem}")
            else:
                bound_types.add(expected)
        for declaration in contract.consumes:
            if declaration.required and declaration.semantic_type not in bound_types:
                errors.append(
                    f"{step.step_id}: 缺少必需输入 {declaration.semantic_type}（输入键 {'/'.join(declaration.input_keys)}）"
                )
        produced_by_step[step.step_id] = list(contract.produces)

    final_types = sorted({type_name for produced in produced_by_step.values() for type_name in produced if type_name in FINAL_SEMANTIC_TYPES})
    if not final_types:
        errors.append(f"计划没有产出最终回复类型（{' / '.join(FINAL_SEMANTIC_TYPES)}）")
    return ClosureReport(ok=not errors, errors=errors, produced_by_step=produced_by_step, final_types=final_types)


def _binding_problem(
    source_type: str,
    source_step_id: Optional[str],
    expected: str,
    state_types: set,
    produced_by_step: Mapping[str, List[str]],
) -> Optional[str]:
    if source_type == "step_output":
        step_id = str(source_step_id or "").strip()
        if step_id not in produced_by_step:
            return f"引用的步骤 {step_id or '<空>'} 不存在或排在后面"
        if expected not in produced_by_step[step_id]:
            produced = "、".join(produced_by_step[step_id]) or "无语义类型"
            return f"需要 {expected}，但步骤 {step_id} 产出的是 {produced}"
        return None
    if expected not in state_types:
        return f"需要 {expected}，但本轮 state 里没有（来源 {source_type}）"
    return None


__all__ = ["ClosureReport", "check_plan_closure"]
