import logging
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional

from pydantic import ValidationError

from utils.logging_utils import info_event

from .. import tool_registry as agent_tool_registry
from ..state import AgentState
from ..schemas import PlanStep
from ..tool_adapters.models import ToolError, ToolExecutionResult
from ..tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    return model_dump() if callable(model_dump) else value


def _run_id(state: AgentState) -> Optional[str]:
    context = state.context if isinstance(state.context, Mapping) else {}
    return str(context.get("run_id") or state.session_id or "").strip() or None


def _error(
    *,
    code: str,
    message: str,
    detail: Mapping[str, Any],
    recoverable: bool,
    stage: str,
    suggested_recovery: Optional[str] = None,
) -> ToolError:
    return ToolError(
        error_code=code,
        message=message,
        detail=dict(detail),
        recoverable=recoverable,
        retryable=False,
        failed_stage=stage,
        suggested_recovery=suggested_recovery,
    )


class ToolExecutionService:
    """拥有工具输入增强、契约校验、adapter 调用和输出校验，不修改 Agent 运行态。"""

    def __init__(self, *, registry: ToolRegistry, backend_invoker: Callable[..., Any]) -> None:
        self.registry = registry
        self.backend_invoker = backend_invoker

    def execute(self, *, step: PlanStep, arguments: Dict[str, Any], state: AgentState) -> ToolExecutionResult:
        contract = self.registry.get_contract(step.tool_name)
        if contract is None or contract.adapter is None:
            raise ValueError(f"Unsupported tool contract adapter: {step.tool_name}")
        # adapter 的 backend 入口在测试和部署时都可能被替换，因此每次调用前显式同步当前入口。
        agent_tool_registry.invoke_backend_tool = self.backend_invoker
        if hasattr(contract.adapter, "invoke_backend_tool"):
            contract.adapter.invoke_backend_tool = self.backend_invoker
        tool_input = self._validate_input(contract, arguments, state)
        if isinstance(tool_input, ToolExecutionResult):
            return tool_input

        started = perf_counter()
        info_event(
            logger,
            "arxiv_agent.tool_started",
            run_id=_run_id(state),
            session_id=state.session_id,
            step_id=step.step_id,
            tool_name=step.tool_name,
            backend_tool=getattr(contract, "backend_tool_name", None),
            adapter=contract.adapter.__class__.__name__,
            input=_plain(tool_input),
        )
        try:
            result = contract.adapter.execute(tool_input)
        except Exception:
            logger.exception(
                "arxiv_agent tool invocation raised: step_id=%s tool_name=%s elapsed_ms=%.1f",
                step.step_id,
                step.tool_name,
                (perf_counter() - started) * 1000,
            )
            raise
        if not isinstance(result, ToolExecutionResult):
            return ToolExecutionResult(
                ok=False,
                data=None,
                error=_error(
                    code="adapter_contract_violation",
                    message="ToolAdapter 未返回 ToolExecutionResult",
                    detail={"returned_type": type(result).__name__},
                    recoverable=False,
                    stage="adapter_return",
                ),
                adapter_name=contract.adapter.__class__.__name__,
                tool_name=step.tool_name,
            )
        validated = self._validate_output(contract, result) if result.ok else result
        info_event(
            logger,
            "arxiv_agent.tool_done",
            run_id=_run_id(state),
            session_id=state.session_id,
            step_id=step.step_id,
            tool_name=step.tool_name,
            backend_tool=getattr(contract, "backend_tool_name", None),
            ok=validated.ok,
            error_code=validated.error.error_code if validated.error else None,
            elapsed_ms=round((perf_counter() - started) * 1000, 1),
        )
        return validated

    def _validate_input(self, contract: Any, arguments: Dict[str, Any], state: AgentState) -> Any:
        raw_input = self._augment_input(contract.tool_name, arguments, state)
        if contract.input_model is None:
            return raw_input
        try:
            return contract.input_model.model_validate(raw_input)
        except ValidationError as exc:
            return ToolExecutionResult(
                ok=False,
                data=None,
                error=_error(
                    code="input_validation_error",
                    message="工具输入未通过 Pydantic 模型校验",
                    detail={
                        "errors": exc.errors(include_url=False),
                        "input_keys": sorted(str(key) for key in raw_input),
                        "input_types": {str(key): type(value).__name__ for key, value in raw_input.items()},
                    },
                    recoverable=True,
                    stage="input_validation",
                    suggested_recovery="ask_clarification",
                ),
                adapter_name=contract.adapter.__class__.__name__,
                tool_name=contract.tool_name,
            )

    def _validate_output(self, contract: Any, result: ToolExecutionResult) -> ToolExecutionResult:
        if contract.output_model is None or result.data is None:
            return result
        try:
            output = result.data if isinstance(result.data, contract.output_model) else contract.output_model.model_validate(_plain(result.data))
            return result.model_copy(update={"data": output})
        except ValidationError as exc:
            return result.model_copy(
                update={
                    "ok": False,
                    "data": None,
                    "error": _error(
                        code="output_validation_error",
                        message="工具输出未通过 Pydantic 模型校验",
                        detail={"errors": exc.errors(include_url=False)},
                        recoverable=False,
                        stage="output_validation",
                    ),
                }
            )

    def _augment_input(self, tool_name: str, arguments: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        payload = dict(arguments or {})
        context = dict(state.context or {}) if isinstance(state.context, Mapping) else {}
        if tool_name in {"normalize_request", "build_arxiv_search_spec"}:
            payload.setdefault("message", state.message)
            payload.setdefault("intent", state.intent)
            if state.search_spec is not None:
                payload.setdefault("search_spec", _plain(state.search_spec))
        if tool_name in {"resolve_paper", "resolve_preference_target", "load_user_profile", "load_candidate_papers", "analyze_ambiguity"}:
            payload.setdefault("context", context)
        if tool_name in {"load_user_profile", "generate_recommendations", "update_preference_store", "analyze_ambiguity"}:
            payload.setdefault("user_id", state.user_id)
        if tool_name in {"load_user_profile", "generate_recommendations", "answer_paper_question"}:
            payload.setdefault("message", state.message)
        if tool_name in {"answer_paper_question", "check_paper_index", "parse_and_index_paper"}:
            payload.setdefault("run_id", _run_id(state))
        if tool_name == "analyze_ambiguity":
            payload.setdefault("search_spec", _plain(state.search_spec) if state.search_spec is not None else {})
            payload.setdefault("goal", {"intent": state.intent or "unclear", "goal_type": state.intent or "unclear"})
        return payload
