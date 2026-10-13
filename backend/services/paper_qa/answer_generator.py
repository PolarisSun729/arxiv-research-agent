"""论文 QA 最终答案生成器。

职责：把 ContextPackBuilder 打包好的证据交给生成服务（Qwen）作答，并保证输出满足引用协议
（每个引用单独写成 [source:{source_id}]）。引用不合法时最多做一次「引用修复」重新生成，
仍失败则只删除非法引用标记、保留正文，避免整段答案丢失。

当前生产调用方是 paper_evidence_research 的 ResearchDraftGenerator，它只消费返回值中的
answer / cited_source_ids / citation_debug；claims、insufficient_evidence 等字段是旧 QA 链路
遗留的轻量信号，主张抽取与校验已由研究图（extract_claims / verify 节点）独立完成。
"""

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
        """生成最终答案并完成引用校验/修复。

        Args:
            generation_question: 实际用于生成的问题（可能已被会话消歧改写）。
            context_pack: ContextPackBuilder.build() 的产物，包含证据块、图片输入和 source_payload。
            preferred_answer_style: 用户偏好的回答风格，交给 prompt 组装层处理。
            style_already_applied: 仅用于 debug 标记，表示风格提示是否已拼进问题。
            original_question: 用户原始问题；为空时回退到 generation_question。
            session_summary / recent_turns / user_memory_summary: 会话与用户记忆，只影响 prompt 上下文，不作为证据。

        Returns:
            包含 answer、cited_source_ids、citation_debug、citation_warning 以及 generation_debug 等字段的字典。
        """
        # 1. 组装 prompt 上下文（含预算裁剪），并得到真正送入模型的证据列表。
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
        # 只把图片类资产元数据传给生成层，表格证据已经以文本形式进入证据块。
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
        # 本轮可引用的 source_id 白名单，来自 context_pack 的 source_payload。
        source_ids = [
            str(source.get("source_id", "") or "").strip()
            for source in (context_pack.get("source_payload") or [])
            if str(source.get("source_id", "") or "").strip()
        ]
        citation_debug = self.validate_citations(answer, source_ids)
        citation_warning = None
        task_type = "paper_qa_final_answer"
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
                repair_succeeded = not repaired_debug["invalid_citations"]
                # 无论修复成功与否都采用修复后的答案：修复轮已经按更严格的引用要求重答，
                # 即使仍有少量非法引用，删除后的正文通常也比首轮更干净。
                answer = repaired_answer if repair_succeeded else self.strip_invalid_citations(repaired_answer, source_ids)
                generation_result = repaired_generation_result
                task_type = "paper_qa_final_answer_citation_repair"
                citation_debug = repaired_debug
                citation_debug["repair_attempted"] = True
                citation_debug["repair_succeeded"] = repair_succeeded
                if not repair_succeeded:
                    # 二次失败时只移除无法定位的标记，保留正文和证据面板，避免整段答案消失。
                    citation_warning = "部分引用未通过校验"
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
                # 发生引用修复时，最终答案来自修复调用，debug 中的 task_type/model 要与之对应。
                "task_type": task_type,
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
        """从生成服务的返回值中取出答案文本，兼容 dict（多种键名）和纯字符串两种返回形态。"""
        if isinstance(generation_result, dict):
            for key in ("response", "answer", "text", "result"):
                if key in generation_result and generation_result.get(key) is not None:
                    # 空字符串也是有效的生成结果，必须交给 verifier 判断，不能回退成 dict 字符串。
                    value = generation_result.get(key)
                    return str(value).strip()
        return str(generation_result or "").strip()

    @staticmethod
    def extract_cited_source_ids(answer: str, source_ids: List[str]) -> List[str]:
        """提取答案中合法且属于白名单的 source_id（按首次出现顺序去重）。"""
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
        """按句切分答案，粗筛出像「具体断言」的句子（最多 12 条），供 debug 与轻量校验使用。

        切分规则：中文句末标点（。！？）后直接断句，中文排版通常不带空格；
        英文句末标点（.!?）后必须跟空白才断句，避免把 "3.5"、"e.g." 之类拆开。
        """
        claims: List[Dict[str, Any]] = []
        sentences = re.split(r"(?<=[。！？])\s*|(?<=[.!?])\s+", answer or "")
        for index, sentence in enumerate(sentences, start=1):
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
        """通过关键短语判断模型是否在答案中声明了「证据不足」。"""
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
