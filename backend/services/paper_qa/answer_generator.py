from __future__ import annotations

import re
from typing import Any, Dict, List

from services.prompt_context import PromptContextBuilder


class AnswerGenerator:
    """封装论文 QA 的最终答案生成调用。"""

    def __init__(self, *, generation_service: Any) -> None:
        self.generation_service = generation_service
        self.prompt_context_builder = PromptContextBuilder()

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
            query=generation_question,
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
        cited_source_ids = self.extract_cited_source_ids(answer, source_ids)
        claims = self.extract_claims(answer)
        insufficient_evidence = self.detect_insufficient_evidence(answer)

        return {
            "answer": answer,
            "cited_source_ids": cited_source_ids,
            "claims": claims,
            "insufficient_evidence": insufficient_evidence,
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

    @staticmethod
    def extract_cited_source_ids(answer: str, source_ids: List[str]) -> List[str]:
        if not answer or not source_ids:
            return []
        cited: List[str] = []
        for source_id in source_ids:
            if source_id and source_id in answer and source_id not in cited:
                cited.append(source_id)
        # 兼容模型按 Source 编号引用而没有直接写 source_id 的情况。
        for match in re.findall(r"(?:Source|来源|证据)\s*\[?#?(\d+)\]?", answer, flags=re.IGNORECASE):
            try:
                index = int(match) - 1
            except ValueError:
                continue
            if 0 <= index < len(source_ids) and source_ids[index] not in cited:
                cited.append(source_ids[index])
        return cited

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
