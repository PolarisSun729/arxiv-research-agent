from __future__ import annotations

import re
import logging
from typing import Any, Dict, List

from services.paper_qa.citation_contract import (
    extract_cited_source_ids as extract_cited_source_ids_contract,
    strip_invalid_citations as strip_invalid_citations_contract,
    validate_citations as validate_citations_contract,
)
from services.paper_qa.answer_language import CHINESE_FINAL_ANSWER_INSTRUCTION
from services.prompt_context import PromptContextBuilder

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """封装论文 QA 的最终答案生成调用。"""

    def __init__(self, *, generation_service: Any) -> None:
        self.generation_service = generation_service
        # LLM prompt 压缩默认关闭；开启时复用生成服务，避免在 builder 内部隐式创建新的外部调用依赖。
        self.prompt_context_builder = PromptContextBuilder(llm_compaction_client=generation_service)

    def generate(
        self,
        *,
        generation_question: str,
        context_pack: Dict[str, Any],
        preferred_answer_style: str = "",
        style_already_applied: bool = False,
        original_question: str = "",
        session_summary: Dict[str, Any] | None = None,
        recent_turns: List[Dict[str, Any]] | None = None,
        user_memory_summary: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        prompt_assembly = self.prompt_context_builder.build_paper_qa_final_answer_context(
            question=original_question or generation_question,
            contextualized_question=generation_question,
            context_pack=context_pack,
            session_summary=session_summary,
            recent_turns=recent_turns,
            user_memory_summary=user_memory_summary,
            preferred_answer_style=preferred_answer_style,
        )
        generation_search_results = self.prompt_context_builder.generation_search_results_from_assembly(
            prompt_assembly,
            context_pack,
        )
        figure_metadata = [
            item for item in (context_pack.get("asset_metadata") or [])
            if str(item.get("chunk_type", "") or "").lower() == "figure"
        ]

        # 最终回答仍走原来的 Qwen 大模型任务路由；本类只收口生成输入和输出契约，不改模型选择策略。
        generation_result = self.generation_service.generate(
            provider="qwen",
            task_type="paper_qa_final_answer",
            query=(
                f"{generation_question}\n\n"
                f"{CHINESE_FINAL_ANSWER_INSTRUCTION}\n"
                "引用格式要求：每个证据引用必须单独写成 [source:{source_id}]；禁止把多个引用合并在一对方括号内，"
                "禁止裸 source:number/source-*、[Source 1]、[Image 1] 和其他数字引用。"
            ),
            search_results=generation_search_results,
            image_inputs=context_pack.get("image_inputs") or [],
            asset_metadata=figure_metadata,
        )
        answer = self.extract_answer(generation_result)
        source_ids = [
            str(source.get("source_id", "") or "").strip()
            for source in (context_pack.get("source_payload") or [])
            if str(source.get("source_id", "") or "").strip()
        ]
        citation_debug = self.validate_citations(answer, source_ids)
        citation_warning = None
        if citation_debug["invalid_citations"]:
            # 引用格式是前端定位证据的协议；只允许一次修复，避免坏模型输出触发无限重试。
            repair_query = (
                f"{generation_question}\n\n"
                f"{CHINESE_FINAL_ANSWER_INSTRUCTION}\n"
                "引用修复要求：重新回答同一个问题。所有证据引用必须严格写成 "
                "[source:{source_id}]，source_id 必须来自证据块；每个引用单独占一对方括号，禁止合并引用或裸 source:number/source-*；"
                "删除 [Source 1]、[Image 1] 及其他旧格式引用，不要解释修复过程。"
            )
            try:
                repaired_generation_result = self.generation_service.generate(
                    provider="qwen",
                    task_type="paper_qa_final_answer_citation_repair",
                    query=repair_query,
                    search_results=generation_search_results,
                    image_inputs=context_pack.get("image_inputs") or [],
                    asset_metadata=figure_metadata,
                )
                repaired_answer = self.extract_answer(repaired_generation_result)
                repaired_debug = self.validate_citations(repaired_answer, source_ids)
                citation_debug["repair_attempted"] = True
                citation_debug["repair_succeeded"] = not bool(repaired_debug["invalid_citations"])
                if citation_debug["repair_succeeded"]:
                    answer = repaired_answer
                    generation_result = repaired_generation_result
                    citation_debug = repaired_debug
                    citation_debug["repair_attempted"] = True
                    citation_debug["repair_succeeded"] = True
                else:
                    # 二次失败时只移除无法定位的标记，保留正文和证据面板，避免整段答案消失。
                    answer = self.strip_invalid_citations(repaired_answer, source_ids)
                    citation_warning = "部分引用未通过校验"
                    citation_debug = repaired_debug
                    citation_debug["repair_attempted"] = True
                    citation_debug["repair_succeeded"] = False
            except Exception as exc:
                # 修复调用本身失败时保留首轮答案，避免引用协议问题放大成整次 QA 失败。
                logger.warning("Citation repair generation failed; keep sanitized first answer: %s", exc)
                answer = self.strip_invalid_citations(answer, source_ids)
                citation_warning = "部分引用未通过校验"
                citation_debug["repair_attempted"] = True
                citation_debug["repair_succeeded"] = False
        cited_source_ids = self.extract_cited_source_ids(answer, source_ids)
        claims = self.extract_claims(answer)
        insufficient_evidence = self.detect_insufficient_evidence(answer)

        return {
            "answer": answer,
            "cited_source_ids": cited_source_ids,
            "claims": claims,
            "insufficient_evidence": insufficient_evidence,
            "citation_debug": citation_debug,
            "citation_warning": citation_warning,
            "raw_generation_result": generation_result,
            "generation_debug": {
                "provider": "qwen",
                "task_type": "paper_qa_final_answer",
                "preferred_answer_style": preferred_answer_style,
                "styled_question_applied": bool(style_already_applied),
                "search_result_count": len(generation_search_results),
                "image_input_count": len(context_pack.get("image_inputs") or []),
                "figure_asset_metadata_count": len(figure_metadata),
                "answer_chars": len(answer),
                "cited_source_ids": cited_source_ids,
                "citation_debug": citation_debug,
                "citation_warning": citation_warning,
                "claim_count": len(claims),
                "prompt_context": prompt_assembly.get("debug", {}),
                "model": generation_result.get("model") if isinstance(generation_result, dict) else None,
                "saved_filepath": generation_result.get("saved_filepath") if isinstance(generation_result, dict) else None,
                # Qwen 请求体预算由生成层统一处理，这里只透传诊断结果，便于区分“图片降级”和“检索证据不足”。
                "qwen_request_debug": generation_result.get("qwen_request_debug", {}) if isinstance(generation_result, dict) else {},
            },
        }

    @staticmethod
    def extract_answer(generation_result: Any) -> str:
        if isinstance(generation_result, dict):
            for key in ("response", "answer", "text", "result"):
                if key in generation_result and generation_result.get(key) is not None:
                    # 空字符串也是有效的生成结果，必须交给 verifier 判断，不能回退成 dict 字符串。
                    value = generation_result.get(key)
                    return str(value).strip()
        return str(generation_result or "").strip()

    def validate_and_repair_stream_answer(
        self,
        *,
        answer: str,
        generation_question: str,
        context_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        """为流式输出补齐与同步问答相同的一次引用修复闭环。"""
        source_ids = [
            str(source.get("source_id") or "").strip()
            for source in (context_pack.get("source_payload") or [])
            if str(source.get("source_id") or "").strip()
        ]
        citation_debug = self.validate_citations(answer, source_ids)
        citation_warning = None
        final_answer = answer or ""
        if citation_debug["invalid_citations"]:
            assembly = self.prompt_context_builder.build_paper_qa_final_answer_context(
                question=generation_question,
                contextualized_question=generation_question,
                context_pack=context_pack,
            )
            search_results = self.prompt_context_builder.generation_search_results_from_assembly(assembly, context_pack)
            figure_metadata = [
                item for item in (context_pack.get("asset_metadata") or [])
                if str(item.get("chunk_type") or "").lower() == "figure"
            ]
            try:
                repaired = self.generation_service.generate(
                    provider="qwen",
                    task_type="paper_qa_final_answer_citation_repair",
                    query=(
                        f"{generation_question}\n\n{CHINESE_FINAL_ANSWER_INSTRUCTION}\n"
                        "引用修复要求：所有证据引用必须严格写成 "
                        "[source:{source_id}]，source_id 必须来自证据块；每个引用单独占一对方括号，禁止合并引用或裸 source:number/source-*；"
                        "删除所有旧格式引用。"
                    ),
                    search_results=search_results,
                    image_inputs=context_pack.get("image_inputs") or [],
                    asset_metadata=figure_metadata,
                )
                repaired_answer = self.extract_answer(repaired)
                repaired_debug = self.validate_citations(repaired_answer, source_ids)
                repaired_debug["repair_attempted"] = True
                repaired_debug["repair_succeeded"] = not bool(repaired_debug["invalid_citations"])
                if repaired_debug["repair_succeeded"]:
                    final_answer = repaired_answer
                    citation_debug = repaired_debug
                else:
                    final_answer = self.strip_invalid_citations(repaired_answer, source_ids)
                    citation_debug = repaired_debug
                    citation_warning = "部分引用未通过校验"
            except Exception as exc:
                logger.warning("Streaming citation repair generation failed; keep sanitized answer: %s", exc)
                final_answer = self.strip_invalid_citations(final_answer, source_ids)
                citation_debug["repair_attempted"] = True
                citation_debug["repair_succeeded"] = False
                citation_warning = "部分引用未通过校验"
        return {
            "answer": final_answer,
            "cited_source_ids": self.extract_cited_source_ids(final_answer, source_ids),
            "citation_debug": citation_debug,
            "citation_warning": citation_warning,
        }

    @staticmethod
    def extract_cited_source_ids(answer: str, source_ids: List[str]) -> List[str]:
        return extract_cited_source_ids_contract(answer, source_ids)

    @staticmethod
    def validate_citations(answer: str, source_ids: List[str]) -> Dict[str, Any]:
        """校验新引用协议，并显式拒绝历史 [Source 1]/[Image 1] 格式。"""
        return validate_citations_contract(answer, source_ids)

    @staticmethod
    def strip_invalid_citations(answer: str, source_ids: List[str]) -> str:
        """仅清理无法定位的引用标记，保留用户仍可阅读的回答文本。"""
        return strip_invalid_citations_contract(answer, source_ids)

    @staticmethod
    def extract_claims(answer: str) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        for index, sentence in enumerate(re.split(r"(?<=[。！？.!?])\s+", answer or ""), start=1):
            text = sentence.strip()
            if not text:
                continue
            if AnswerGenerator.looks_like_specific_claim(text):
                claims.append({"claim_id": f"claim-{index}", "text": text})
        return claims[:12]

    @staticmethod
    def looks_like_specific_claim(text: str) -> bool:
        # 数值、实验结果、方法步骤和结论判断更容易产生无证据断言，先纳入轻量校验范围。
        patterns = [
            r"\d+(?:\.\d+)?\s*(?:%|percent|points?|epochs?|layers?|samples?|tokens?)",
            r"\b(?:outperform|improve|reduce|increase|achieve|conclude|show|demonstrate)s?\b",
            r"(?:优于|提升|降低|增加|达到|证明|表明|结论|实验|结果|步骤|方法|指标)",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def detect_insufficient_evidence(answer: str) -> bool:
        lowered = (answer or "").lower()
        return any(
            phrase in lowered
            for phrase in [
                "insufficient evidence",
                "not enough evidence",
                "无法从当前证据",
                "证据不足",
                "检索证据不足",
            ]
        )
