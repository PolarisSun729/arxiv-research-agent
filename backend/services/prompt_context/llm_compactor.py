from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple

from services.prompt_context.blocks import PromptBlock
from services.prompt_context.token_counter import TokenCounter


@dataclass
class LLMCompactionConfig:
    enabled: bool = False
    min_block_tokens: int = 1200
    target_block_tokens: int = 700
    max_blocks: int = 8
    model_name: str = ""
    task_type: str = "prompt_context_compaction"
    enable_thinking: bool = False


class LLMCompactor:
    """在规则压缩之后做可选语义压缩；默认关闭，避免 prompt 构建链路隐式增加外部调用。"""

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

    def compact_blocks(self, blocks: Iterable[PromptBlock]) -> Tuple[List[PromptBlock], Dict[str, Any]]:
        prepared = list(blocks)
        debug: Dict[str, Any] = {
            "enabled": bool(self.config.enabled),
            "applied": False,
            "reason": "",
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

        result: List[PromptBlock] = []
        attempted = 0
        for block in prepared:
            eligible, reason = self._eligible(block)
            if not eligible:
                debug["skipped_blocks"].append(self._debug_block(block, reason=reason))
                result.append(block)
                continue
            if attempted >= max(0, int(self.config.max_blocks)):
                debug["skipped_blocks"].append(self._debug_block(block, reason="max_blocks_reached"))
                result.append(block)
                continue
            attempted += 1
            debug["attempted_blocks"] = attempted
            compacted, failure_reason = self._compact_one(block)
            if failure_reason:
                # LLM 压缩只是优化层；失败时保留规则压缩结果，不能让证据链路因为 JSON 或服务波动丢块。
                debug["failed_blocks"].append(self._debug_block(block, reason=failure_reason))
                result.append(block)
                continue
            debug["applied"] = True
            debug["compacted_blocks"].append(
                {
                    **self._debug_block(compacted, reason="llm_compacted"),
                    "original_token_count": compacted.original_token_count,
                    "token_count": compacted.token_count,
                }
            )
            result.append(compacted)

        if not debug["applied"] and not debug["reason"]:
            debug["reason"] = "no_eligible_blocks"
        return result, debug

    def _eligible(self, block: PromptBlock) -> Tuple[bool, str]:
        if block.section != "rag_evidence":
            return False, "non_rag_section"
        if not block.compactable:
            return False, "block_not_compactable"
        if block.block_type in {"table_final_evidence", "table_candidate_evidence", "plain_table_context"}:
            return False, "table_evidence_protected"
        if block.can_support_numeric_claim:
            return False, "numeric_evidence_protected"
        token_count = block.token_count or self.token_counter.count(block.text)
        if token_count < max(1, int(self.config.min_block_tokens)):
            return False, "below_min_block_tokens"
        return True, ""

    def _compact_one(self, block: PromptBlock) -> Tuple[PromptBlock, str]:
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
        return {
            "block_id": block.block_id,
            "section": block.section,
            "block_type": block.block_type,
            "source_id": block.source_id,
            "reason": reason,
            "token_count": block.token_count,
            "evidence_origin": block.evidence_origin,
        }
