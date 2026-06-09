from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from services.llm.generation_service import GenerationService
from services.paper_qa.session_service import (
    MAX_CONVERSATION_TURNS,
    MAX_REFERENCED_SOURCE_IDS,
    PaperQASessionService,
)
from services.prompt_context import PromptContextBuilder

logger = logging.getLogger(__name__)

FOLLOW_UP_PATTERN = re.compile(
    r"(它|他|她|这个|这个方法|这个模块|这种方法|该方法|该模块|上述|上面|前面|这里|其中|其|这些|那些|第二步|第一步|第三步|这一步|上一步|下一步|baseline|上一轮|上一条|上一个)",
    re.IGNORECASE,
)


class QuestionContextualizer:
    """判断追问并把依赖历史的用户问题改写成可检索的自包含问题。"""

    def __init__(
        self,
        *,
        generation_service: GenerationService,
        session_service: PaperQASessionService,
    ) -> None:
        self.generation_service = generation_service
        self.session_service = session_service
        self.prompt_context_builder = PromptContextBuilder()

    def contextualize(
        self,
        question: str,
        paper_context: Dict[str, Any],
        conversation_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """结合短期记忆把追问改写成自包含问题，供检索阶段直接使用。"""
        if not bool(self.session_service.memory_flag("enable_short_term_memory", True)):
            return {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "memory_reason": "Short-term memory is disabled by runtime config.",
                "used_short_term_memory": False,
                "status": "disabled",
                "error": None,
            }
        if not conversation_context:
            return {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": [],
                "referenced_source_ids": [],
                "memory_reason": "No conversation context was provided, so retrieval used the original question.",
                "used_short_term_memory": False,
                "status": "not_provided",
                "error": None,
            }

        referenced_source_ids, referenced_turn_ids, prompt, prompt_context_debug = self._build_contextualization_prompt(
            question,
            paper_context,
            conversation_context,
        )
        heuristic_follow_up = self.looks_like_follow_up(question)
        try:
            response_text = self.generation_service.complete_with_qwen(
                prompt,
                task_type="paper_qa_question_contextualization",
            )
            parsed = self.extract_json_object(response_text)
            contextualized_question = self.session_service.truncate_text(parsed.get("contextualized_question", question), 800)
            candidate_turn_ids = [str(item).strip() for item in (parsed.get("referenced_turn_ids", []) or []) if str(item).strip()]
            candidate_source_ids = [
                str(item).strip() for item in (parsed.get("referenced_source_ids", []) or []) if str(item).strip()
            ]
            allowed_turn_ids = set(referenced_turn_ids)
            allowed_source_ids = set(referenced_source_ids)
            filtered_turn_ids = [item for item in candidate_turn_ids if item in allowed_turn_ids][:MAX_CONVERSATION_TURNS]
            filtered_source_ids = [item for item in candidate_source_ids if item in allowed_source_ids][:MAX_REFERENCED_SOURCE_IDS]
            is_follow_up = bool(parsed.get("is_follow_up", False) or heuristic_follow_up)
            if not contextualized_question:
                contextualized_question = question
            return {
                "original_question": question,
                "contextualized_question": contextualized_question,
                "is_follow_up": is_follow_up,
                "referenced_turn_ids": filtered_turn_ids,
                "referenced_source_ids": filtered_source_ids,
                "memory_reason": self.session_service.truncate_text(
                    parsed.get("memory_reason", "Short-term memory was used to resolve likely follow-up references."),
                    320,
                ),
                "used_short_term_memory": is_follow_up and contextualized_question != question,
                "status": "contextualized" if contextualized_question != question else "kept_original",
                "error": None,
                "prompt_context": prompt_context_debug,
            }
        except Exception as exc:
            logger.warning("Question contextualization failed, falling back: %s", exc)
            if heuristic_follow_up:
                fallback = self.build_contextualization_fallback(question, conversation_context, referenced_source_ids)
                fallback["error"] = str(exc)
                fallback["prompt_context"] = prompt_context_debug
                return fallback
            return {
                "original_question": question,
                "contextualized_question": question,
                "is_follow_up": False,
                "referenced_turn_ids": referenced_turn_ids[:MAX_CONVERSATION_TURNS],
                "referenced_source_ids": referenced_source_ids[:MAX_REFERENCED_SOURCE_IDS],
                "memory_reason": "Contextualization failed, so retrieval fell back to the original question.",
                "used_short_term_memory": False,
                "status": "fallback_original",
                "error": str(exc),
                "prompt_context": prompt_context_debug if "prompt_context_debug" in locals() else {},
            }

    @staticmethod
    def extract_json_object(text: str) -> Dict[str, Any]:
        """从模型输出中提取 JSON 对象，兼容 fenced code block 与裸 JSON 两种形式。"""
        normalized = str(text or "").strip()
        if not normalized:
            return {}
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", normalized, flags=re.DOTALL)
        if fenced_match:
            normalized = fenced_match.group(1)
        else:
            object_match = re.search(r"(\{.*\})", normalized, flags=re.DOTALL)
            if object_match:
                normalized = object_match.group(1)
        return json.loads(normalized)

    @staticmethod
    def looks_like_follow_up(question: str) -> bool:
        """基于代词、长度和语气特征，粗略判断问题是否像追问。"""
        normalized = re.sub(r"\s+", " ", str(question or "")).strip()
        if not normalized:
            return False
        if FOLLOW_UP_PATTERN.search(normalized):
            return True
        if len(normalized) <= 18:
            return True
        return normalized.endswith(("?", "？"))

    def build_contextualization_fallback(
        self,
        question: str,
        conversation_context: List[Dict[str, Any]],
        referenced_source_ids: List[str],
    ) -> Dict[str, Any]:
        """当模型改写失败时，使用最近一轮对话构造一个启发式追问改写结果。"""
        latest_turn = conversation_context[-1] if conversation_context else {}
        latest_question = str(latest_turn.get("question", "") or "").strip()
        latest_answer_summary = str(latest_turn.get("answer_summary", "") or "").strip()
        bridge_parts = []
        if latest_question:
            bridge_parts.append(f"Previous turn question: {latest_question}")
        if latest_answer_summary:
            bridge_parts.append(f"Previous turn answer summary: {latest_answer_summary}")
        if bridge_parts:
            # 启发式 fallback 只拼接最近一轮摘要，避免在模型失败时扩大提示规模或引入更远历史噪声。
            contextualized_question = "\n".join(bridge_parts + [f"Current follow-up question: {question}"])
            memory_reason = "Used the most recent QA turn as a fallback to disambiguate a likely follow-up question."
        else:
            contextualized_question = question
            memory_reason = "No usable prior turn summary was available, so the original question was kept."
        return {
            "original_question": question,
            "contextualized_question": contextualized_question,
            "is_follow_up": bool(conversation_context),
            "referenced_turn_ids": [latest_turn.get("turn_id")] if latest_turn.get("turn_id") else [],
            "referenced_source_ids": referenced_source_ids[:MAX_REFERENCED_SOURCE_IDS],
            "memory_reason": memory_reason,
            "used_short_term_memory": bool(conversation_context),
            "status": "heuristic_fallback",
            "error": None,
        }

    def _build_contextualization_prompt(
        self,
        question: str,
        paper_context: Dict[str, Any],
        conversation_context: List[Dict[str, Any]],
    ) -> tuple[List[str], List[str], str, Dict[str, Any]]:
        referenced_source_ids: List[str] = []
        referenced_turn_ids: List[str] = []
        source_lines: List[str] = []
        for index, turn in enumerate(conversation_context, start=1):
            # 把历史轮次摘要化展开，交给模型判断当前问题是否需要借助上下文改写。
            turn_id = str(turn.get("turn_id", "") or "").strip()
            is_summary = bool(turn.get("is_summary") or turn.get("context_type") == "session_summary")
            if turn_id and not is_summary:
                referenced_turn_ids.append(turn_id)
            for source in turn.get("sources", []):
                source_id = str(source.get("source_id", "") or "").strip()
                if source_id and source_id not in referenced_source_ids:
                    referenced_source_ids.append(source_id)
                source_lines.append(
                    "- "
                    f"turn_id={turn_id or f't{index}'}, source_id={source_id or 'unknown'}, "
                    f"section={source.get('section_path', '') or 'N/A'}, "
                    f"page={source.get('page_number', '') or 'N/A'}, "
                    f"chunk_type={source.get('chunk_type', '') or 'text'}, "
                    f"from_summary={bool(source.get('from_session_summary'))}, "
                    f"summary={source.get('content', '') or 'N/A'}"
                )

        session_summary = None
        recent_turns: List[Dict[str, Any]] = []
        for turn in conversation_context:
            if turn.get("is_summary") or turn.get("context_type") == "session_summary":
                session_summary = {"summary": turn.get("answer_summary", "")}
            else:
                recent_turns.append(turn)
        assembly = self.prompt_context_builder.build_question_contextualization_context(
            question=question,
            paper_context=paper_context,
            session_summary=session_summary,
            recent_turns=recent_turns,
        )
        prompt = (
            "You are contextualizing a follow-up question for retrieval over a single academic paper.\n"
            "Rewrite the current user question into a fully self-contained retrieval question when the history indicates a follow-up.\n"
            "Rules:\n"
            "1. Return JSON only.\n"
            "2. Do not invent paper facts, method names, datasets, steps, or results not stated in the conversation context.\n"
            "3. Use short-term memory only to resolve references like it/this method/second step/above.\n"
            "4. Session summary is compressed memory, not retrieval evidence; use it only to preserve earlier preferences and task focus.\n"
            "5. If the reference is unclear, keep the original question and explain the uncertainty in memory_reason.\n"
            "6. referenced_turn_ids must only contain ids from the provided non-summary turns.\n"
            "7. referenced_source_ids must only contain ids from the provided sources.\n"
            "8. The JSON schema is exactly: "
            '{"contextualized_question":"...","is_follow_up":true,"referenced_turn_ids":["..."],"referenced_source_ids":["..."],"memory_reason":"..."}.\n\n'
            "Prompt context sections:\n"
            f"{assembly.get('text') or 'N/A'}\n\n"
            # source id 白名单是输出约束，不作为事实证据；事实仍由后续 RAG 检索决定。
            "Allowed recent source references for referenced_source_ids:\n"
            f"{chr(10).join(source_lines[:12]) or 'N/A'}\n\n"
            "Return JSON only."
        )
        return referenced_source_ids, referenced_turn_ids, prompt, dict(assembly.get("debug") or {})
