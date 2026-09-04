"""LLM 优先的问题分析器：把用户问题拆解成证据需求账本，失败即降级到模板。

铁律：这里只做"提出需求"，不做任何预算或终态判断；账本里 ≥1 个 core need、需求
状态与证据归属等覆盖语义仍由图骨架的确定性初始化器清洗（见 graph._initialize_evidence_needs）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .degradation import DependencyDegradation, DegradationListener, REASON_LLM_CALL_FAILED, REASON_LLM_OUTPUT_INVALID
from .template_question_analyzer import TemplateQuestionAnalyzer

logger = logging.getLogger(__name__)

_ALLOWED_NEED_FIELDS = ("need_id", "description", "importance", "directly_required_by_question", "parent_need_id")
_VALID_IMPORTANCES = {"core", "supporting"}

_ANALYSIS_PROMPT = """You are the question analyzer of a bounded evidence-research engine for single-paper QA.
Decompose the user's question into evidence needs that must be satisfied by passages from THIS paper only.
Rules:
1. Return JSON only, no extra text before or after.
2. The JSON schema must be:
   {{"research_question": "...", "evidence_needs": [{{"need_id": "need-1", "description": "...", "importance": "core"}}]}}
3. Produce 1 to {max_needs} needs; at least one "core" need that must be satisfied before the question can be answered.
4. Each description is a question to verify against the paper, not an answer; never invent facts, names, or numbers.
5. need_id must be unique and short; write descriptions in the user's language.

User question: {question}"""


def _extract_json_payload(text: str) -> dict[str, Any]:
    """从 LLM 回复中提取第一个 JSON 对象；容忍 markdown 围栏与前后噪声。"""
    if not text or not text.strip():
        raise ValueError("llm response is empty")
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    if start >= 0:
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("llm response does not contain a json object")


def _normalize_need(raw: Any, known_ids: set[str]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    need_id = str(raw.get("need_id") or "").strip()
    description = str(raw.get("description") or "").strip()
    if not need_id or not description:
        return None
    importance = str(raw.get("importance") or "supporting").strip().lower()
    if importance not in _VALID_IMPORTANCES:
        importance = "supporting"
    normalized: dict[str, Any] = {
        "need_id": need_id,
        "description": description,
        "importance": importance,
        "directly_required_by_question": importance == "core",
    }
    parent = str(raw.get("parent_need_id") or "").strip()
    # 父需求引用必须指向已知需求，否则图的初始化器会直接抛错；这里软修正为无父需求。
    if parent and parent != need_id and parent in known_ids:
        normalized["parent_need_id"] = parent
    return normalized


def _validate_analysis(payload: dict[str, Any], original_question: str, max_needs: int) -> tuple[str, list[dict[str, Any]]]:
    raw_needs = payload.get("evidence_needs")
    if not isinstance(raw_needs, list) or not raw_needs:
        raise ValueError("evidence_needs must be a non-empty list")
    known_ids = {
        str(item.get("need_id") or "").strip()
        for item in raw_needs
        if isinstance(item, dict) and str(item.get("need_id") or "").strip()
    }
    needs: list[dict[str, Any]] = []
    for raw in raw_needs:
        normalized = _normalize_need(raw, known_ids)
        if normalized is not None:
            needs.append(normalized)
    if len({need["need_id"] for need in needs}) != len(needs):
        raise ValueError("duplicate evidence need ids")
    if not any(need["importance"] == "core" for need in needs):
        raise ValueError("at least one core evidence need is required")
    if len(needs) > max_needs:
        # 截断时核心需求优先保留，避免长尾清单把 core 挤出账本。
        needs.sort(key=lambda need: 0 if need["importance"] == "core" else 1)
        needs = needs[:max_needs]
    research_question = str(payload.get("research_question") or original_question).strip() or original_question
    return research_question, needs


class LlmQuestionAnalyzer:
    """LLM 分析器：结构化拆解证据需求；调用失败或输出不合法时降级到模板分析器。"""

    def __init__(
        self,
        generation_service: Any,
        *,
        fallback: Any = None,
        degradation_listener: DegradationListener | None = None,
        max_needs: int = 6,
    ) -> None:
        self._generation_service = generation_service
        self._fallback = fallback or TemplateQuestionAnalyzer()
        self._degradation_listener = degradation_listener
        self._max_needs = max(1, int(max_needs))

    def analyze(self, request: Any) -> dict[str, Any]:
        question = str(getattr(request, "original_question", "") or "").strip()
        prompt = _ANALYSIS_PROMPT.format(max_needs=self._max_needs, question=question)
        try:
            response = self._generation_service.complete_with_qwen(
                prompt,
                task_type="research_question_analysis",
            )
        except Exception as exc:
            return self._degrade(request, REASON_LLM_CALL_FAILED, {"error": str(exc)[:200]})
        try:
            payload = _extract_json_payload(response)
            research_question, needs = _validate_analysis(payload, question, self._max_needs)
        except Exception as exc:
            return self._degrade(request, REASON_LLM_OUTPUT_INVALID, {"error": str(exc)[:200]})
        return {
            "research_question": research_question,
            "evidence_needs": needs,
            "analyzer_source": "llm",
        }

    def _degrade(self, request: Any, reason: str, detail: dict[str, Any]) -> dict[str, Any]:
        logger.warning("research question analyzer degraded: reason=%s detail=%s", reason, detail)
        if self._degradation_listener is not None:
            try:
                self._degradation_listener(
                    DependencyDegradation(organ="question_analyzer", reason=reason, detail=detail)
                )
            except Exception:  # pragma: no cover - 监听器异常不能影响兜底路径
                logger.exception("degradation listener failed")
        return self._fallback.analyze(request)
