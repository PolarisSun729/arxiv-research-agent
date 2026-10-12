"""Paper QA 与 Agent 共用的证据引用契约。

引用是前后端之间的协议字段，不应由某个具体生成入口各自猜测。这个模块只负责
确定性识别、校验和清洗，不调用 LLM，也不负责决定引用是否真的支持某个事实。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Match


# 合法引用的唯一形态：[source:<source_id>]。
# 一个方括号只能承载一个完整 source_id；逗号和空白都会让它失去合法性。
_STRICT_MARKER_RE = re.compile(r"\[source:([^\]\s,]+)\]")

# 覆盖混合引用、未知引用以及旧的 Source/Image 展示格式，便于统一出站清洗。
_BRACKETED_CITATION_RE = re.compile(
    r"\[(?:(?:source|image|来源|证据)\s*[:：]\s*[^\]\n]+|(?:source|image|来源|证据)\s+[^\]\n]+)\]",
    flags=re.IGNORECASE,
)

# 只处理明显像引用的裸标记，避免误伤普通英文短语，例如“the data source is ...”。
_BARE_CITATION_RE = re.compile(
    r"(?<![\w\-\[])source:(?:\d+|source-[A-Za-z0-9_.:-]+)",
    flags=re.IGNORECASE,
)


def _known_source_ids(source_ids: Iterable[str]) -> set[str]:
    """把本轮可引用的 source_id 规范化成集合（去空白、去空值），便于 O(1) 判断合法性。"""
    return {str(source_id).strip() for source_id in source_ids if str(source_id).strip()}


def extract_cited_source_ids(answer: str, source_ids: Iterable[str]) -> List[str]:
    """只提取格式和来源都合法的单个引用，绝不从混合格式中猜测 ID。"""
    known = _known_source_ids(source_ids)
    cited: List[str] = []
    for match in _STRICT_MARKER_RE.finditer(answer or ""):
        source_id = match.group(1)
        if source_id in known and source_id not in cited:
            cited.append(source_id)
    return cited


def _is_inside_match(position: int, matches: List[Match[str]]) -> bool:
    """判断某个裸标记是否落在已识别的方括号引用内部，用于避免同一处错误被重复计数。"""
    return any(match.start() <= position < match.end() for match in matches)


def validate_citations(answer: str, source_ids: Iterable[str]) -> dict[str, Any]:
    """验证严格引用协议，并记录所有需要修复或清洗的引用形态。"""
    known = _known_source_ids(source_ids)
    text = answer or ""
    invalid: List[str] = []
    bracketed_matches = list(_BRACKETED_CITATION_RE.finditer(text))

    for match in bracketed_matches:
        token = match.group(0)
        strict_match = _STRICT_MARKER_RE.fullmatch(token)
        if strict_match and strict_match.group(1) in known:
            continue
        # 保留旧校验器对未知单 ID 的可诊断信息，同时对混合/旧格式保留原文。
        invalid_value = strict_match.group(1) if strict_match else token
        if invalid_value not in invalid:
            invalid.append(invalid_value)

    # 方括号内的第二个 source 标记已由整块判定，避免重复报告同一处错误。
    for match in _BARE_CITATION_RE.finditer(text):
        if _is_inside_match(match.start(), bracketed_matches):
            continue
        if match.group(0) not in invalid:
            invalid.append(match.group(0))

    recognized_count = sum(
        1 for match in _STRICT_MARKER_RE.finditer(text) if match.group(1) in known
    )
    return {
        "valid": not invalid,
        "invalid_citations": invalid,
        "recognized_citation_count": recognized_count,
        "repair_attempted": False,
        "repair_succeeded": None,
    }


def _strip_citation_whitespace(answer: str) -> str:
    """合并删除引用标记后残留的空白。

    清洗只移除协议噪声，不重写正文：
    - 只合并「非空白字符之后」的连续空格，保留行首缩进；否则 Markdown 嵌套列表、
      代码块的缩进会被压成单个空格，导致前端渲染错乱；
    - 删除标点前多余的空格，例如 "结论 [source:x]。" 删除非法引用后会残留 "结论 。"。
    """
    cleaned = re.sub(r"(?<=\S)[ \t]{2,}", " ", answer or "")
    cleaned = re.sub(r"[ \t]+([，。！？；：、,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


def strip_invalid_citations(answer: str, source_ids: Iterable[str]) -> str:
    """移除非法引用，保留合法引用和正文内容。"""
    known = _known_source_ids(source_ids)

    def replace_bracketed(match: Match[str]) -> str:
        # 方括号整体是合法单引用时原样保留，其余（混合/旧格式/未知 ID）整块删除。
        strict_match = _STRICT_MARKER_RE.fullmatch(match.group(0))
        if strict_match and strict_match.group(1) in known:
            return match.group(0)
        return ""

    # 先处理方括号引用，再删除方括号外的裸 source:xxx 标记；顺序不能反，
    # 否则合法的 [source:1] 内部会先被裸标记规则误删（裸规则的负向后顾已排除 "[" 紧邻的情况，这里是双保险）。
    cleaned = _BRACKETED_CITATION_RE.sub(replace_bracketed, answer or "")
    cleaned = _BARE_CITATION_RE.sub("", cleaned)
    return _strip_citation_whitespace(cleaned)
