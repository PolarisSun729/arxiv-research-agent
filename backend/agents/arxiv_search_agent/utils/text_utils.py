"""文本清洗、正则匹配和 JSON 片段抽取工具。

这个模块承载的是 node / utils 多处都会用到的最基础文本辅助能力，
目标是把“零散、重复、容易写错”的字符串处理逻辑集中起来统一维护。

设计原则主要有两点：
1. 函数尽量保持轻量、纯函数化，方便被任意节点重复调用；
2. 只做通用文本处理，不掺杂具体业务语义判断，避免职责膨胀。
"""

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
    """判断文本中是否包含中文字符。

    这个判断主要服务于查询词放宽、分词和中英文混合策略选择等场景。
    当系统知道输入里包含中文时，可以采用与英文 token 不同的截断和清洗规则。
    """
    return bool(re.search(r"[一-鿿]", text))


def _normalize_text(value: Optional[str]) -> str:
    """做最基础的文本标准化：转字符串、压缩连续空白、去首尾空格。

    这是最底层的标准化入口，很多上层函数都会先经过这里，
    以减少因为空白符、None 或异常类型带来的边界问题。
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_optional_str(value: Any) -> Optional[str]:
    """把任意值归一化为可选字符串；空白值统一转成 None。

    当调用方需要明确区分“空字符串”和“没有值”时，优先使用这个函数，
    这样后续业务逻辑可以直接用 None 作为缺失值语义。
    """
    text = _normalize_text(str(value or ""))
    return text or None


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    """判断文本是否命中任意一个正则模式。

    该函数用来统一封装大小写不敏感的多模式匹配逻辑，
    让上层业务判断不必反复编写重复的 any(re.search(...)) 模板。
    """
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _parse_small_chinese_number(value: Any) -> Optional[int]:
    """解析较小范围的中文数字表达，如“三”“十”“十二”。

    这里主要用于搜索条数、序号引用等场景，只覆盖小范围自然语言数字，
    目标是满足 Agent 对话里的常见表达，而不是实现通用中文数字系统。
    """
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
    """从模型输出文本中提取 JSON 代码块或裸 JSON 对象。

    LLM 输出经常会混入解释性文本、Markdown 代码围栏或额外说明，
    这里先做一层宽松提取，给后续 json.loads 提高成功率。
    """
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1)
    raw_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if raw_match:
        return raw_match.group(1)
    return text


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """从任意文本中尽量恢复一个 JSON 对象。

    与 _extract_json_block 相比，这个函数更偏向“最终解析”：
    它会剥离代码块围栏、截取最外层花括号，并尝试直接 loads。
    适合处理要求严格 JSON 输出的模型返回值。
    """
    candidate = _normalize_text(text)
    if not candidate:
        return None

    if candidate.startswith("```"):
        # 先剥离 Markdown fenced code block，避免 ```json 这类前缀影响解析。
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s*```$", "", candidate).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        # 只保留最外层对象片段，尽量忽略前后残留解释文本。
        candidate = candidate[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None
