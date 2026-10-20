"""把草稿切成原文区间片段：主张提取的最小单位。

切分点是句末标点（。！？!?；;）、后跟空白的英文句点和换行，另有三条约束：

1. 引号、括号里的标点不切：`分解为如“谁持有头衔？”和“配偶是谁？”等子问题` 是一句话，
   在问号处切开会得到三个各自无法校验的碎片；`[Figure 7: ... hit rate. Average ...]()` 这类图注链接同理。
2. 紧跟句末标点的 `[source:id]` 和收尾加粗归属前一句，引用不会落到下一句开头。
3. 行首编号里的点不切：`1. 方法`、`**2. 实验**` 不是句子边界（小数本来就不跟空白）。

片段是草稿的连续区间，按顺序拼回去就是原文；提取器据此保证主张逐字来自草稿、且没有句子被漏掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END_CHARS = "。！？!?；;"
_OPENING_TO_CLOSING = {"“": "”", "‘": "’", "「": "」", "『": "』", "《": "》", "（": "）", "(": ")", "[": "]", '"': '"'}
_CITATION_MARKER_RE = re.compile(r"\[source:[^\]\s,]+\]")
_EMPHASIS_MARKERS = ("**", "__")
# 编号前只允许出现排版符号：缩进、标题井号、项目符号、加粗与左括号。
_ORDINAL_PREFIX_RE = re.compile(r"[\s#*+\-•_(（]*\d{1,2}")
# 整行只是一个图注链接（可带项目符号与引用）：草稿照抄了证据里的图片标记，没有可校验的陈述。
_FIGURE_LINK_LINE_RE = re.compile(r"\s*(?:[-*+•]\s+)?!?\[[^\]]*\]\([^)]*\)\s*(?:\[source:[^\]\s,]+\]\s*)*")


@dataclass(frozen=True)
class DraftSegment:
    start: int
    end: int
    text: str


def is_figure_link_line(text: str) -> bool:
    return bool(_FIGURE_LINK_LINE_RE.fullmatch(text))


def split_draft_segments(answer: str) -> list[DraftSegment]:
    segments: list[DraftSegment] = []
    line_start = 0
    for line in answer.splitlines(keepends=True):
        segments.extend(_split_line(answer, line_start, line_start + len(line.rstrip("\r\n"))))
        line_start += len(line)
    return segments


def _split_line(answer: str, start: int, end: int) -> list[DraftSegment]:
    # 引号栈按行重置：一行里没闭合的引号最多让本行剩余部分不再切分，不会吞掉后面的段落。
    closers: list[str] = []
    pieces: list[tuple[int, int]] = []
    piece_start = index = start
    while index < end:
        char = answer[index]
        if closers and char == closers[-1]:
            closers.pop()
        elif char in _OPENING_TO_CLOSING:
            closers.append(_OPENING_TO_CLOSING[char])
        elif not closers and _is_sentence_end(answer, index, start, end):
            cut = _extend_over_trailing(answer, index + 1, end, piece_start)
            pieces.append((piece_start, cut))
            piece_start = index = cut
            continue
        index += 1
    pieces.append((piece_start, end))
    return [DraftSegment(begin, finish, answer[begin:finish]) for begin, finish in pieces if answer[begin:finish].strip()]


def _is_sentence_end(answer: str, index: int, line_start: int, end: int) -> bool:
    char = answer[index]
    if char in _SENTENCE_END_CHARS:
        # "![图注](路径)" 的感叹号是图片语法，不是句末。
        return not (
            char == "!" and answer.startswith("[", index + 1) and not _CITATION_MARKER_RE.match(answer, index + 1)
        )
    if char != "." or _ORDINAL_PREFIX_RE.fullmatch(answer, line_start, index):
        return False
    following = index + 1
    return following >= end or answer[following].isspace() or bool(_CITATION_MARKER_RE.match(answer, following))


def _extend_over_trailing(answer: str, index: int, end: int, piece_start: int) -> int:
    while index < end:
        marker = _CITATION_MARKER_RE.match(answer, index)
        if marker:
            index = marker.end()
        elif answer.startswith(_EMPHASIS_MARKERS, index) and answer[piece_start:index].count(answer[index:index + 2]) % 2 == 1:
            # 只吞闭合的加粗；偶数个说明后面的 ** 是下一句的开头。
            index += 2
        elif answer[index].isspace() or answer[index] in _SENTENCE_END_CHARS:
            index += 1
        else:
            break
    return index
