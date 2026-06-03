from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional


CHINESE_NUMBER_MAP = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _contains_chinese(text: str) -> bool:
    """判断文本中是否包含中文字符。"""
    return bool(re.search(r"[一-鿿]", text))


def _normalize_text(value: Optional[str]) -> str:
    """做最基础的文本标准化：转字符串、压缩连续空白、去首尾空格。"""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_optional_str(value: Any) -> Optional[str]:
    """把任意值归一化为可选字符串；空白值统一转成 None。"""
    text = _normalize_text(str(value or ""))
    return text or None


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    """判断文本是否命中任意一个正则模式。"""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _parse_small_chinese_number(value: Any) -> Optional[int]:
    """解析较小范围的中文数字表达，如“三”“十”“十二”。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text in CHINESE_NUMBER_MAP:
        return CHINESE_NUMBER_MAP[text]
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = CHINESE_NUMBER_MAP.get(left, 1 if left == "" else 0)
        ones = CHINESE_NUMBER_MAP.get(right, 0) if right else 0
        if tens == 0 and left:
            return None
        return tens * 10 + ones
    return None


def _extract_json_block(text: str) -> str:
    """从模型输出文本中提取 JSON 代码块或裸 JSON 对象。"""
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)
    raw_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if raw_match:
        return raw_match.group(1)
    return text


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = _normalize_text(text)
    if not candidate:
        return None

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s*```$", "", candidate).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None
