from __future__ import annotations

from typing import Any, Mapping, Optional

from .schemas import PlanRuntime


def assemble_final_answer(runtime: PlanRuntime) -> Optional[str]:
    """从 runtime outputs 装配最终展示答案。

    PlanExecutor 只负责把各 step 的结构化输出写入 runtime；这里集中定义“哪个输出可作为最终
    用户答案”，避免调度器为了兼容不同业务工具继续散落回答拼接逻辑。
    """
    outputs = runtime.outputs if isinstance(runtime.outputs, Mapping) else {}
    explicit_answer = _clean_text(outputs.get("final_answer"))
    if explicit_answer:
        return explicit_answer

    # LLM planner 允许为步骤输出自定义 key，但回答工具的稳定契约仍是 final_answer 字段；
    # 扫描结构化输出可以兼容 final_user_response 等命名，避免恢复后 runtime 找不到新答案。
    for value in reversed(list(outputs.values())):
        if not isinstance(value, Mapping):
            continue
        nested_answer = _clean_text(value.get("final_answer"))
        if nested_answer:
            return nested_answer

    paper_qa_result = outputs.get("paper_qa_result")
    if isinstance(paper_qa_result, Mapping):
        qa_answer = _clean_text(paper_qa_result.get("answer"))
        if qa_answer:
            return qa_answer

    return _clean_text(runtime.final_answer)


def record_recovery_fallback(
    runtime: PlanRuntime,
    *,
    fallback_reason: str,
    fallback_record: Optional[Mapping[str, Any]] = None,
) -> None:
    """记录无法继续自动恢复时的兜底输出。

    这里不再触发额外工具调用；replanner 只决定 fallback，response assembler 负责把结构化原因转成
    可展示结果，避免 executor 在恢复分支里再次执行业务工具。
    """
    reason = _clean_text(fallback_reason) or "recovery_fallback"
    if isinstance(fallback_record, Mapping):
        # fallback 原因需要同时写入结构化输出，方便前端和 trace 统一读取，而不是只剩一段自然语言。
        runtime.outputs["fallback_record"] = dict(fallback_record)
    runtime.outputs["final_answer"] = _build_recovery_fallback_answer(runtime, reason)
    runtime.final_answer = runtime.outputs["final_answer"]


def _clean_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _build_recovery_fallback_answer(runtime: PlanRuntime, reason: str) -> str:
    """为 Paper QA 质量失败生成保守答案，避免把低证据回答包装成确定结论。"""
    outputs = runtime.outputs if isinstance(runtime.outputs, Mapping) else {}
    paper_qa_result = outputs.get("paper_qa_result") if isinstance(outputs.get("paper_qa_result"), Mapping) else {}
    qa_observation = paper_qa_result.get("qa_observation") if isinstance(paper_qa_result.get("qa_observation"), Mapping) else {}
    lower_reason = reason.lower()
    if qa_observation or any(token in lower_reason for token in ("qa_", "paper_qa", "paper_index", "answer_paper_question")):
        retrieval_quality = _clean_text(qa_observation.get("retrieval_quality"))
        answer_insufficient = _clean_text(qa_observation.get("answer_insufficient_evidence"))
        observation_reason = _clean_text(
            qa_observation.get("answer_quality_reason")
            or qa_observation.get("retrieval_quality_reason")
            or qa_observation.get("observation_reason")
        )
        reason_bits = [item for item in [observation_reason, f"retrieval_quality={retrieval_quality}" if retrieval_quality else None, f"answer_insufficient_evidence={answer_insufficient}" if answer_insufficient else None] if item]
        reason_text = "；".join(reason_bits) or reason
        return f"当前无法给出足够可靠的论文回答：{reason_text}。我没有把低证据结果当作确定答案；请尝试重建索引、缩小问题范围或稍后重试。"
    return f"当前步骤无法继续自动恢复：{reason}。请补充信息或稍后重试。"
