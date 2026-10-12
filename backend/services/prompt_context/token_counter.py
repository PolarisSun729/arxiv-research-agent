from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


class TokenCounter:
    """统一封装 token 计数；真实 tokenizer 不可用时用保守估算兜底。

    估算模式按“字符数 / fallback_chars_per_token”向上取整。默认 3 字符 ≈ 1 token，
    比英文常见的 4 字符更保守，目的是宁可高估 token 数，也不要让 prompt 超出模型上限。

    注意：``_last_mode`` 只反映“最近一次 count 调用”走的是精确还是估算路径，
    同一个实例被多处共享时它只是诊断参考，不代表全部调用的情况。
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[Any] = None,
        tokenizer_name: str = "",
        fallback_chars_per_token: float = 3.0,
        fallback_reason: str = "",
    ) -> None:
        """
        Args:
            tokenizer: 任意带 ``encode(text)`` 方法的 tokenizer；为 None 时只用估算。
            tokenizer_name: 诊断用名称，缺省时取 tokenizer 的类名。
            fallback_chars_per_token: 估算模式下每个 token 对应的字符数，最小为 1。
            fallback_reason: 没有 tokenizer 时的原因说明，会出现在 debug 输出里。
        """
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name or (type(tokenizer).__name__ if tokenizer is not None else "")
        self.fallback_chars_per_token = max(1.0, float(fallback_chars_per_token or 3.0))
        self._last_mode = "exact" if tokenizer is not None else "estimated"
        if tokenizer is not None:
            self._fallback_reason = ""
        else:
            self._fallback_reason = fallback_reason or "tokenizer_not_configured"

    def count(self, text: str) -> int:
        """返回文本的 token 数；空文本为 0，非空文本至少为 1。"""
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
        """输出计数器状态，合并进 prompt 预算的 debug 信息中。"""
        return {
            "token_counter_mode": self._last_mode,
            "tokenizer_name": self.tokenizer_name,
            "fallback_chars_per_token": self.fallback_chars_per_token,
            "fallback_reason": self._fallback_reason,
        }


def build_token_counter(config: Mapping[str, Any]) -> TokenCounter:
    """根据 PROMPT_CONTEXT_CONFIG 构造 TokenCounter。

    - provider 为 auto / qwen / transformers 且配置了 tokenizer 路径时，尝试加载本地 HF tokenizer；
    - provider 为 estimated / fallback 时直接使用估算；
    - 任何加载失败都不会抛异常，而是退回估算并在 fallback_reason 中记录原因。
    """
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

    return TokenCounter(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        fallback_chars_per_token=fallback_chars_per_token,
        fallback_reason=fallback_reason,
    )
