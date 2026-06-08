from __future__ import annotations

from typing import Any, Mapping, Optional

from .schemas import ConfirmationRequest, PlanRuntime


def assemble_final_answer(runtime: PlanRuntime) -> Optional[str]:
    """从 runtime outputs 装配最终展示答案。

    PlanExecutor 只负责把各 step 的结构化输出写入 runtime；这里集中定义“哪个输出可作为最终
    用户答案”，避免调度器为了兼容不同业务工具继续散落回答拼接逻辑。
    """
    outputs = runtime.outputs if isinstance(runtime.outputs, Mapping) else {}
    explicit_answer = _clean_text(outputs.get("final_answer"))
    if explicit_answer:
        return explicit_answer

    paper_qa_result = outputs.get("paper_qa_result")
    if isinstance(paper_qa_result, Mapping):
        qa_answer = _clean_text(paper_qa_result.get("answer"))
        if qa_answer:
            return qa_answer

    return _clean_text(runtime.final_answer)


def record_confirmation_rejection(
    runtime: PlanRuntime,
    *,
    confirmation_request: ConfirmationRequest,
) -> None:
    """把用户拒绝确认后的展示结果写入 outputs。

    拒绝确认是调度状态流转，但面向用户的说明属于 response 装配职责；放在这里可以避免
    PlanExecutor 根据具体工具名手写业务文案。
    """
    label = _target_label(confirmation_request)
    if confirmation_request.tool_name == "parse_and_index_paper":
        final_answer = f"已取消解析 {label}，因此无法继续基于全文回答。"
    else:
        action_label = _clean_text(confirmation_request.title) or _clean_text(confirmation_request.tool_name) or "当前操作"
        final_answer = f"已取消{action_label}。"
    runtime.outputs["final_answer"] = final_answer
    runtime.final_answer = final_answer


def record_recovery_fallback(runtime: PlanRuntime, *, fallback_reason: str) -> None:
    """记录无法继续自动恢复时的兜底输出。

    这里不再触发额外工具调用；replanner 只决定 fallback，response assembler 负责把结构化原因转成
    可展示结果，避免 executor 在恢复分支里再次执行业务工具。
    """
    reason = _clean_text(fallback_reason) or "recovery_fallback"
    runtime.outputs["final_answer"] = f"当前步骤无法继续自动恢复：{reason}。请补充信息或稍后重试。"
    runtime.final_answer = runtime.outputs["final_answer"]


def _target_label(confirmation_request: ConfirmationRequest) -> str:
    target_paper = confirmation_request.target_paper if isinstance(confirmation_request.target_paper, Mapping) else {}
    return _clean_text(target_paper.get("title")) or _clean_text(target_paper.get("arxiv_id")) or "该论文"


def _clean_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None
