"""部分回答的正文剪裁：在原草稿上删掉未通过校验的主张，保留排版与引用位置。

旧做法把受支持主张用"；"拼成一段，标题、列表和段落全部丢失，引用标记也挤在一起。
主张文本来自草稿原文（LLM 提取器逐字摘录、规则提取器按句切分），因此可以在草稿里
定位每条未受支持的主张并原位删除；任何一条定位失败都返回 None，由调用方退回拼接方案，
绝不让未校验的句子留在正文里。
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Iterable

_CITATION_MARKER_RE = re.compile(r"\[source:([^\]\s,]+)\]")
_EMPHASIS_MARKERS = ("**", "__")
# 删除主张后，紧随其后的句末标点、空白与引用标记一起删掉，避免留下孤立的 "；[source:3]"。
_TRAILING_PUNCTUATION = "。；;，,.!?！？：: \t"
# 只剩排版符号的行（空项目符号、孤立编号、空加粗）在剪裁后整行删除。
_EMPTY_MARKUP_LINE_RE = re.compile(r"^(?:\s|[-*+•]|\d{1,2}[.、)）]|#{1,6}|\*\*|__|[：:；;，,。])*$")
# 行首的项目符号、标题井号与 "1." 编号是排版，不算残留的数字内容。
_LEADING_MARKUP_RE = re.compile(r"^(?:[-*+•]\s+|#{1,6}\s+|\d{1,2}(?:\.(?!\d)|[、)）])\s*)+")
# 剪裁后残留的非主张文本只能是排版：短、无数字、不以句末标点收尾（像一句完整陈述就不算标题）。
_MAX_STRUCTURAL_CHARS = 40
_MAX_STRAY_LETTERS = 10
_SENTENCE_END = ("。", "！", "？", ".", "!", "?")
_LIST_MARKER_RE = re.compile(r"^(?:[-*+•]|\d{1,2}[.)、）])\s+")
# CommonMark：比所属列表项内容列多缩进 4 格及以上的行会被渲染成代码块。
_CODE_INDENT = 4


def _normalize_with_index(raw: str) -> tuple[str, list[int]]:
    """按提取器同样的规则归一化（去引用、去加粗、折叠空白），并记录每个字符在原文中的位置。"""
    marker_spans = {match.start(): match.end() for match in _CITATION_MARKER_RE.finditer(raw)}
    chars: list[str] = []
    positions: list[int] = []
    index = 0
    while index < len(raw):
        if index in marker_spans:
            index = marker_spans[index]
            if chars and chars[-1] != " ":
                chars.append(" ")
                positions.append(index - 1)
            continue
        if raw.startswith(_EMPHASIS_MARKERS, index):
            index += 2
            continue
        char = raw[index]
        if char.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                positions.append(index)
        else:
            chars.append(char)
            positions.append(index)
        index += 1
    return "".join(chars), positions


def _normalize(text: str) -> str:
    return _normalize_with_index(text)[0].strip()


def _extend_over_trailing(raw: str, end: int) -> int:
    while end < len(raw):
        marker = _CITATION_MARKER_RE.match(raw, end)
        if marker:
            end = marker.end()
        elif raw.startswith(_EMPHASIS_MARKERS, end):
            end += 2
        elif raw[end] in _TRAILING_PUNCTUATION:
            end += 1
        else:
            break
    return end


def prune_unsupported_claims(
    answer: str,
    *,
    removed_claim_texts: Iterable[str],
    supported_claim_texts: Iterable[str],
    allowed_citation_ids: Iterable[str],
) -> str | None:
    normalized, positions = _normalize_with_index(answer)
    spans: list[tuple[int, int]] = []
    search_from: dict[str, int] = {}
    for text in removed_claim_texts:
        needle = _normalize(text).rstrip(_TRAILING_PUNCTUATION)
        if not needle:
            continue
        found = normalized.find(needle, search_from.get(needle, 0))
        if found < 0:
            return None
        search_from[needle] = found + len(needle)
        start = positions[found]
        end = _extend_over_trailing(answer, positions[found + len(needle) - 1] + 1)
        # 主张从加粗段中间开始时（"**巩固：** 离线机制……"），把开头的 ** 一起删掉，避免留下不配对的加粗符号。
        if answer[start:end].count("**") % 2 == 1 and answer[max(0, start - 2):start] == "**":
            start -= 2
        spans.append((start, end))

    pruned = answer
    for start, end in sorted(spans, reverse=True):
        pruned = pruned[:start] + pruned[end:]

    allowed = set(allowed_citation_ids)
    # 被删主张的引用若残留在别处，或指向未进入最终引用列表的证据，都不能再对外展示。
    pruned = _CITATION_MARKER_RE.sub(lambda match: match.group(0) if match.group(1) in allowed else "", pruned)
    # 空行是 Markdown 的段落分隔，保留；剪裁后只剩排版符号的行整行删掉（不留空行，免得把紧凑列表
    # 拆成松散列表），最后把连续空行压成一个。
    lines = [
        line.rstrip()
        for line in pruned.splitlines()
        if not line.strip() or not _EMPTY_MARKUP_LINE_RE.fullmatch(line)
    ]
    lines = _reindent_orphaned_lines(lines)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if not result or not _only_structure_remains(result, supported_claim_texts):
        return None
    return result


def _reindent_orphaned_lines(lines: list[str]) -> list[str]:
    """父级列表项被删后，子项的缩进会让 Markdown 把它渲染成代码块；把这类行挪到最近保留的上级的内容列。

    只调整会被渲染成代码块的行（以及跟着它的子项），排版正常的行保持原样。
    """
    # 栈中每项：(原缩进, 新缩进, 原内容列)；子项按父项的位移一起平移。
    stack: list[tuple[int, int, int]] = []
    result: list[str] = []
    for line in lines:
        body = line.lstrip(" ")
        if not body:
            result.append(line)
            continue
        indent = len(line) - len(body)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            parent_indent, parent_new_indent, parent_content = stack[-1]
            content_column = parent_new_indent + parent_content - parent_indent
            new_indent = indent + parent_new_indent - parent_indent
            if new_indent >= content_column + _CODE_INDENT:
                new_indent = content_column
        else:
            new_indent = 0 if indent >= _CODE_INDENT else indent
        marker = _LIST_MARKER_RE.match(body)
        if marker:
            stack.append((indent, new_indent, indent + marker.end()))
        result.append(" " * new_indent + body)
    return result


def strip_leading_markup(text: str) -> str:
    """去掉行首的项目符号、标题井号与编号；它们是排版，不属于主张内容。"""
    return _LEADING_MARKUP_RE.sub("", text)


def _only_structure_remains(answer: str, supported_claim_texts: Iterable[str]) -> bool:
    """剪裁结果除受支持主张外只能剩排版结构（短、无数字、不像完整句子），否则说明提取漏了句子，不能原位保留。

    在整篇归一化文本上扣掉受支持主张，再按原文行检查残留：一条主张可以跨多个列表项，
    摘录带不带行首的 "* "、"1." 也都能扣干净（逐行比对时二者都会让主张对不上，整篇被误判为漏句）。
    """
    normalized, positions = _normalize_with_index(answer)
    covered = [False] * len(normalized)
    needles = {
        strip_leading_markup(_normalize(text)).rstrip(_TRAILING_PUNCTUATION) for text in supported_claim_texts
    }
    for needle in needles:
        if not needle:
            continue
        found = normalized.find(needle)
        while found >= 0:
            covered[found:found + len(needle)] = [True] * len(needle)
            found = normalized.find(needle, found + len(needle))
    line_starts = [0] + [index + 1 for index, char in enumerate(answer) if char == "\n"]
    leftovers: dict[int, list[str]] = {}
    for index, char in enumerate(normalized):
        if not covered[index]:
            leftovers.setdefault(bisect_right(line_starts, positions[index]) - 1, []).append(char)
    for chars in leftovers.values():
        rest = strip_leading_markup("".join(chars).strip())
        letters = "".join(char for char in rest if char.isalnum())
        if len(letters) > _MAX_STRUCTURAL_CHARS:
            return False
        if len(letters) > _MAX_STRAY_LETTERS and (
            any(char.isdigit() for char in letters) or rest.rstrip().endswith(_SENTENCE_END)
        ):
            return False
    return True
