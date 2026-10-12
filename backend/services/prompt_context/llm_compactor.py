from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.compactor import TABLE_BLOCK_TYPES
from services.prompt_context.token_counter import TokenCounter


@dataclass
class LLMCompactionConfig:
    """LLM 语义压缩参数。

    Attributes:
        enabled: 总开关，默认关闭。
        min_block_tokens: 只有 token 数不少于该值的 RAG block 才会尝试压缩。
        target_block_tokens: 写进压缩 prompt 的目标长度（软约束，模型不一定严格遵守）。
        max_blocks: 单次 prompt 构建最多发起的 LLM 压缩调用次数。
        model_name / task_type / enable_thinking: 透传给 complete_with_qwen 的调用参数。
    """

    enabled: bool = False
    min_block_tokens: int = 1200
    target_block_tokens: int = 700
    max_blocks: int = 8
    model_name: str = ""
    task_type: str = "prompt_context_compaction"
    enable_thinking: bool = False


class LLMCompactor:
    """在规则压缩之后做可选语义压缩；默认关闭，避免 prompt 构建链路隐式增加外部调用。

    按需压缩：PromptBudgetPlanner 会传入 required_savings（规则压缩后仍超出预算的 token 数）。
    预算够用时不发起任何 LLM 调用；超出时从最长的候选 block 开始压缩，
    累计节省量达到 required_savings 即停止，最多 max_blocks 次串行调用。

    client 支持三种形态（按优先级）：
    1. 可调用对象：``client(prompt) -> str | dict``；
    2. 提供 ``compact_prompt_block(prompt=..., target_tokens=...)`` 方法；
    3. 提供 ``complete_with_qwen(prompt, model_name=..., task_type=..., enable_thinking=...)`` 方法。
    """

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        config: LLMCompactionConfig | None = None,
        client: Any = None,
    ) -> None:
        self.token_counter = token_counter
        self.config = config or LLMCompactionConfig()
        self.client = client

    def compact_blocks(
        self,
        blocks: Iterable[PromptBlock],
        *,
        required_savings: Optional[int] = None,
    ) -> Tuple[List[PromptBlock], Dict[str, Any]]:
        """尝试压缩符合条件的 block，返回（处理后的 blocks, 诊断信息）。

        Args:
            blocks: 规则压缩后的 blocks。
            required_savings: 需要节省的 token 数。None 表示不考虑预算、按原顺序压缩所有符合条件的 block
                （直接调用压缩器时使用）；<= 0 表示预算已够用，不发起任何调用；> 0 时从最长的 block 开始压缩，
                累计节省量达到该值即停止。

        输出 blocks 与输入一一对应、顺序不变；任何失败都保留原 block，绝不丢块。
        """
        prepared = list(blocks)
        debug: Dict[str, Any] = {
            "enabled": bool(self.config.enabled),
            "applied": False,
            "reason": "",
            "required_savings_tokens": required_savings,
            "saved_tokens": 0,
            "attempted_blocks": 0,
            "compacted_blocks": [],
            "skipped_blocks": [],
            "failed_blocks": [],
        }
        if not self.config.enabled:
            debug["reason"] = "disabled"
            return prepared, debug
        if self.client is None:
            debug["reason"] = "client_not_configured"
            return prepared, debug
        if required_savings is not None and required_savings <= 0:
            debug["reason"] = "budget_sufficient"
            return prepared, debug

        candidate_indexes: List[int] = []
        for index, block in enumerate(prepared):
            eligible, reason = self._eligible(block)
            if eligible:
                candidate_indexes.append(index)
            else:
                debug["skipped_blocks"].append(self._debug_block(block, reason=reason))
        if required_savings is not None:
            # 按预算压缩时先压最长的 block：单次调用节省最多，能用最少的调用次数补上缺口。
            candidate_indexes.sort(key=lambda index: -int(prepared[index].token_count or 0))

        result = list(prepared)
        attempted = 0
        saved_tokens = 0
        max_blocks = max(0, int(self.config.max_blocks))
        for index in candidate_indexes:
            block = prepared[index]
            if required_savings is not None and saved_tokens >= required_savings:
                debug["skipped_blocks"].append(self._debug_block(block, reason="budget_satisfied"))
                continue
            if attempted >= max_blocks:
                debug["skipped_blocks"].append(self._debug_block(block, reason="max_blocks_reached"))
                continue
            attempted += 1
            compacted, failure_reason = self._compact_one(block)
            if failure_reason:
                # LLM 压缩只是优化层；失败时保留规则压缩结果，不能让证据链路因为 JSON 或服务波动丢块。
                debug["failed_blocks"].append(self._debug_block(block, reason=failure_reason))
                continue
            before_tokens = block.token_count or self.token_counter.count(block.text)
            saved_tokens += max(0, int(before_tokens) - int(compacted.token_count or 0))
            debug["applied"] = True
            debug["compacted_blocks"].append(
                {
                    **self._debug_block(compacted, reason="llm_compacted"),
                    "original_token_count": compacted.original_token_count,
                    "token_count": compacted.token_count,
                }
            )
            result[index] = compacted

        debug["attempted_blocks"] = attempted
        debug["saved_tokens"] = saved_tokens
        if not debug["applied"] and not debug["reason"]:
            debug["reason"] = "no_eligible_blocks" if not candidate_indexes else "all_attempts_failed"
        return result, debug

    def _eligible(self, block: PromptBlock) -> Tuple[bool, str]:
        """判断 block 是否允许 LLM 压缩：只压缩足够长的普通 RAG 文本，表格和数值证据一律跳过。"""
        if block.section != "rag_evidence":
            return False, "non_rag_section"
        if not block.compactable:
            return False, "block_not_compactable"
        if block.block_type in TABLE_BLOCK_TYPES:
            return False, "table_evidence_protected"
        if block.can_support_numeric_claim:
            return False, "numeric_evidence_protected"
        token_count = block.token_count or self.token_counter.count(block.text)
        if token_count < max(1, int(self.config.min_block_tokens)):
            return False, "below_min_block_tokens"
        return True, ""

    def _compact_one(self, block: PromptBlock) -> Tuple[PromptBlock, str]:
        """压缩单个 block。

        Returns:
            (新 block, "") 表示成功；(原 block, 失败原因) 表示失败。
            压缩结果为空或没有变短都视为失败。成功后的 block 被标记为 llm_compacted，
            并关闭数值结论与直接引用能力，因为文本已被模型改写。
        """
        try:
            raw_response = self._call_client(self._build_prompt(block))
            payload = self._parse_json_payload(raw_response)
            compressed_text = str(payload.get("compressed_text") or "").strip()
            if not compressed_text:
                return block, "empty_compressed_text"
            compressed_tokens = self.token_counter.count(compressed_text)
            current_tokens = block.token_count or self.token_counter.count(block.text)
            if compressed_tokens >= current_tokens:
                return block, "compressed_text_not_smaller"
            metadata = {
                **dict(block.metadata or {}),
                "compacted_by_llm": True,
                "llm_compaction_target_tokens": int(self.config.target_block_tokens),
                "llm_compaction_original_tokens": block.original_token_count or current_tokens,
                "llm_compaction_input_tokens": current_tokens,
                "llm_compaction_output_tokens": compressed_tokens,
            }
            for key in ("compression_notes", "loss_level"):
                if payload.get(key) not in (None, "", [], {}):
                    metadata[f"llm_{key}"] = payload.get(key)
            return block.with_updates(
                text=compressed_text,
                token_count=compressed_tokens,
                original_token_count=block.original_token_count or current_tokens,
                evidence_origin="llm_compacted",
                can_support_numeric_claim=False,
                can_support_direct_quote=False,
                metadata=metadata,
            ), ""
        except Exception as exc:
            return block, f"llm_compaction_error: {exc}"

    def _call_client(self, prompt: str) -> Any:
        """按类说明中的三种 client 形态依次尝试调用。"""
        if callable(self.client):
            return self.client(prompt)
        compact_prompt_block = getattr(self.client, "compact_prompt_block", None)
        if callable(compact_prompt_block):
            return compact_prompt_block(
                prompt=prompt,
                target_tokens=int(self.config.target_block_tokens),
            )
        complete_with_qwen = getattr(self.client, "complete_with_qwen", None)
        if callable(complete_with_qwen):
            # 复用生成服务的统一 Qwen 调用入口，但用独立 task_type，方便后续在配置层指定小模型。
            return complete_with_qwen(
                prompt,
                model_name=self.config.model_name or None,
                task_type=self.config.task_type,
                enable_thinking=bool(self.config.enable_thinking),
            )
        raise TypeError("llm compaction client must be callable or expose compact_prompt_block/complete_with_qwen")

    def _build_prompt(self, block: PromptBlock) -> str:
        """构造压缩 prompt：带上证据的定位 metadata，并要求模型只返回 JSON。"""
        target = max(1, int(self.config.target_block_tokens))
        metadata = {
            "source_id": block.source_id,
            "block_type": block.block_type,
            "context_role": block.context_role,
            "section_path": block.metadata.get("section_path", ""),
            "page_number": block.metadata.get("page_number", ""),
        }
        return (
            "你是 RAG prompt 压缩器。请只压缩下面的证据文本，不要引入外部知识。\n"
            f"目标：压缩到约 {target} tokens 内，保留可回答问题的实体、方法、条件、结论和 source_id。\n"
            "禁止改写表格单元格、数值证据或直接引用；如果需要这些能力，上游会跳过 LLM 压缩。\n"
            "只返回 JSON，不要返回 Markdown。格式：\n"
            '{"compressed_text":"...","compression_notes":["..."],"loss_level":"low|medium|high"}\n\n'
            f"metadata:\n{json.dumps(metadata, ensure_ascii=False, default=str)}\n\n"
            f"evidence:\n{block.text}"
        )

    @staticmethod
    def _parse_json_payload(raw_response: Any) -> Dict[str, Any]:
        """从模型输出中解析 JSON 对象。

        兼容三种情况：client 直接返回 dict；输出被 ```json 代码块包裹；
        JSON 前后夹带了说明文字（取第一个 "{" 到最后一个 "}" 之间的内容）。
        """
        if isinstance(raw_response, dict):
            return dict(raw_response)
        text = str(raw_response or "").strip()
        if not text:
            raise ValueError("empty_llm_response")
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        elif "{" in text and "}" in text:
            text = text[text.find("{") : text.rfind("}") + 1]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("llm_response_json_not_object")
        return payload

    @staticmethod
    def _debug_block(block: PromptBlock, *, reason: str) -> Dict[str, Any]:
        """生成单个 block 的压缩诊断记录（不含正文）。"""
        return {
            "block_id": block.block_id,
            "section": block.section,
            "block_type": block.block_type,
            "source_id": block.source_id,
            "reason": reason,
            "token_count": block.token_count,
            "evidence_origin": block.evidence_origin,
        }
