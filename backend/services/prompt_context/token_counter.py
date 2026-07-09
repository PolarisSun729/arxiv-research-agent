from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


class TokenCounter:
    """统一封装 token 计数；真实 tokenizer 不可用时用保守估算兜底。"""

    def __init__(
        self,
        *,
        tokenizer: Optional[Any] = None,
        tokenizer_name: str = "",
        fallback_chars_per_token: float = 3.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name or (type(tokenizer).__name__ if tokenizer is not None else "")
        self.fallback_chars_per_token = max(1.0, float(fallback_chars_per_token or 3.0))
        self._last_mode = "exact" if tokenizer is not None else "estimated"
        self._fallback_reason = "" if tokenizer is not None else "tokenizer_not_configured"

    def count(self, text: str) -> int:
        value = str(text or "")
        if not value:
            return 0
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer.encode(value)
                self._last_mode = "exact"
                self._fallback_reason = ""
                return max(1, len(encoded))
            except Exception as exc:  # pragma: no cover - 真实 tokenizer 故障只走兜底路径
                self._last_mode = "estimated"
                self._fallback_reason = f"tokenizer_error: {exc}"
        self._last_mode = "estimated"
        if not self._fallback_reason:
            self._fallback_reason = "tokenizer_not_configured"
        return max(1, int(math.ceil(len(value) / self.fallback_chars_per_token)))

    def debug(self) -> Dict[str, Any]:
        return {
            "token_counter_mode": self._last_mode,
            "tokenizer_name": self.tokenizer_name,
            "fallback_chars_per_token": self.fallback_chars_per_token,
            "fallback_reason": self._fallback_reason,
        }


def build_token_counter(config: Mapping[str, Any]) -> TokenCounter:
    provider = str(config.get("token_counter_provider", "auto") or "auto").strip().lower()
    fallback_chars_per_token = float(config.get("token_counter_fallback_chars_per_token", 3.0) or 3.0)
    tokenizer_name_or_path = str(config.get("tokenizer_name_or_path", "") or "").strip()
    tokenizer = None
    tokenizer_name = ""
    fallback_reason = "tokenizer_not_configured"

    if provider in {"auto", "qwen", "transformers"} and tokenizer_name_or_path:
        try:
            from transformers import AutoTokenizer  # type: ignore

            # tokenizer 只能读取本地已有文件，避免 prompt 构建链路因为网络下载而变慢或失败。
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
            tokenizer_name = tokenizer_name_or_path
        except Exception as exc:  # pragma: no cover - 是否安装 transformers 取决于本地环境
            fallback_reason = f"transformers_tokenizer_unavailable: {exc}"
    elif provider in {"auto", "qwen", "transformers"}:
        fallback_reason = "tokenizer_name_or_path_not_configured"
    elif provider in {"estimated", "fallback"}:
        fallback_reason = "estimated_provider_selected"
    else:
        fallback_reason = f"unsupported_token_counter_provider: {provider}"

    counter = TokenCounter(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        fallback_chars_per_token=fallback_chars_per_token,
    )
    if tokenizer is None:
        counter._last_mode = "estimated"
        counter._fallback_reason = fallback_reason
    return counter
