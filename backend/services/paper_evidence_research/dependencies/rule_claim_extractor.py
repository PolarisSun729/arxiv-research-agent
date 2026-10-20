"""规则版主张提取：切句、解析引用标记、绑定证据需求。

三条契约必须与图的完成门禁对齐：

1. 可见事实句都必须进入校验，包括没有引用的句子；否则生成器只需漏写引用即可绕过门禁。
   无引用主张以空 citation_ids 进入修复或最终删减，不能凭空继承其他句子的支持。
   句末标点后的 ``[source:id]`` 归属前一句，引用格式与 citation_contract.py 保持一致。
2. 主张绑定需求走两级信号：优先"引用继承"——草稿声明使用的候选带着 matched_need_ids
   （它就是为那个需求检索回来的），主张引用哪个候选就继承其匹配需求；词面重叠作为
   无候选信息时的兜底。两级都匹配不到时保持不绑定——宁可让需求保持 open 触发继续
   检索，也不能把无关主张硬绑到需求上造成虚假满足。
3. importance 按绑定到的需求定级：绑到 core 需求即 core，其余为 supporting。
4. Markdown 排版结构不是主张：加粗小标题、编号/项目符号、"具体步骤如下："这类引出句
   本身不陈述事实，若当作主张进入校验，必然因无引用判 unsupported，并经词面重叠绑到
   核心需求上，使整轮核心覆盖全部落空。只有短、无引用、无数字的结构行才会被跳过，
   带数字或引用的加粗行仍按事实校验，避免把结论藏进标题绕过门禁。
"""

from __future__ import annotations

import re
from typing import Any

from .draft_segments import is_figure_link_line, split_draft_segments

# 与 services/paper_qa/citation_contract.py 的 _STRICT_MARKER_RE 保持同构。
_CITATION_MARKER_RE = re.compile(r"\[source:([^\]\s,]+)\]")
_SECTION_LABEL_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:概述|方法|实验|结果|结论|总结|局限性|回答|Overview|Method|Results|Conclusion)[:：]?$",
    flags=re.IGNORECASE,
)
_LEADING_META_RE = re.compile(
    r"^(?:本文|该论文|这篇论文|作者们?|我们)\s*"
    r"(?:提出|认为|指出|发现|证明|介绍|给出|采用)\s*"
    r"(?:了|的)?[：：，,]?\s*"
    r"|^(?:this paper|the paper|the authors|we)\s+"
    r"(?:propose|proposes|present|presents|show|shows|find|finds|introduce|introduces)\s+that\s+",
    flags=re.IGNORECASE,
)
# 行首排版标记：Markdown 标题井号、项目符号、"1." "1、" "(1)" 这类编号；"3.5%" 的小数点不算编号。
_LEADING_MARKUP_RE = re.compile(
    r"^(?:#{1,6}\s+|[-*+•]\s+|\d{1,2}(?:\.(?!\d)|[、)）])\s*|[（(]\d{1,2}[)）]\s*)+"
)
_EMPHASIS_RE = re.compile(r"\*\*|__")
_MAX_STRUCTURAL_LINE_CHARS = 40
_MIN_ZH_SHARED_BIGRAMS = 2
_MIN_LATIN_TOKEN_LEN = 4


def _content_tokens(text: str) -> set[str]:
    """中文取相邻二字组、拉丁文取长度>=4 的词，作为词面重叠的最小单元。"""
    tokens: set[str] = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text):
        if len(word) >= _MIN_LATIN_TOKEN_LEN:
            tokens.add(word.lower())
    cleaned = re.sub(r"[^\u4e00-\u9fff]+", "", text)
    for index in range(len(cleaned) - 1):
        tokens.add(cleaned[index : index + 2])
    return tokens


def _need_field(need: Any, name: str) -> Any:
    if isinstance(need, dict):
        return need.get(name)
    return getattr(need, name, None)


def _candidate_field(candidate: Any, name: str) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name)
    return getattr(candidate, name, None)


def _match_need_ids(claim_text: str, needs: list[Any]) -> list[str]:
    claim_tokens = _content_tokens(claim_text)
    if not claim_tokens:
        return []
    matched: list[str] = []
    for need in needs:
        need_tokens = _content_tokens(str(_need_field(need, "description") or ""))
        shared = claim_tokens & need_tokens
        has_strong_latin_overlap = any(
            len(token) >= _MIN_LATIN_TOKEN_LEN and token.isascii() for token in shared
        )
        if len(shared) >= _MIN_ZH_SHARED_BIGRAMS or has_strong_latin_overlap:
            matched.append(str(_need_field(need, "need_id") or ""))
    return [need_id for need_id in matched if need_id]


def _strip_leading_markup(line: str) -> str:
    stripped = line.strip()
    # 编号可能包在加粗里（"**1. 方法**"），先剥项目符号，再剥加粗内的编号。
    stripped = _LEADING_MARKUP_RE.sub("", stripped)
    if stripped.startswith(("**", "__")):
        stripped = stripped[:2] + _LEADING_MARKUP_RE.sub("", stripped[2:])
    return stripped


def _is_structural_line(raw_line: str, body: str) -> bool:
    """判断一行是否只是排版结构（图注链接、标题、编号小标题、列表引出句），而非可校验事实。"""
    if is_figure_link_line(raw_line):
        return True
    if _CITATION_MARKER_RE.search(body):
        return False
    plain = _EMPHASIS_RE.sub("", body).strip()
    if not plain or len(plain) > _MAX_STRUCTURAL_LINE_CHARS or any(char.isdigit() for char in plain):
        return False
    if raw_line.lstrip().startswith("#"):
        return True
    # 整行加粗的小标题，如 "**记忆生成与演化层（顶层）**" 或 "**核心架构特点：**"。
    if re.fullmatch(r"(\*\*|__).+?\1\s*[:：]?", body.strip()):
        return True
    # 以冒号收尾的引出句，如 "具体步骤如下："，事实在后续列表项里。
    return plain.endswith(("：", ":"))


def _normalize_markdown(answer: str) -> str:
    lines: list[str] = []
    for raw_line in answer.splitlines():
        body = _strip_leading_markup(raw_line)
        if _is_structural_line(raw_line, body):
            continue
        lines.append(_EMPHASIS_RE.sub("", body))
    return "\n".join(lines)


def _clean_claim_text(segment: str) -> str:
    text = _CITATION_MARKER_RE.sub("", segment)
    text = _LEADING_META_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip(" \t，,。.；;：:")


class RuleClaimExtractor:
    """确定性地提取可见句子；倾向保留待核实文本，避免把漏提主张误当作全部受支持。"""

    def extract(self, request: Any) -> dict[str, Any]:
        answer = _normalize_markdown(str(getattr(request, "answer", "") or ""))
        draft_version = int(getattr(request, "draft_version", 1) or 1)
        needs = list(getattr(request, "evidence_needs", []) or [])
        candidates = list(getattr(request, "candidates", []) or [])
        need_ids_by_candidate: dict[str, list[str]] = {}
        for candidate in candidates:
            candidate_id = str(_candidate_field(candidate, "candidate_id") or "")
            matched = [str(item) for item in (_candidate_field(candidate, "matched_need_ids") or [])]
            if candidate_id:
                need_ids_by_candidate[candidate_id] = matched
        core_need_ids = {
            str(_need_field(need, "need_id") or "")
            for need in needs
            if str(_need_field(need, "importance") or "") == "core"
        }

        claims: list[dict[str, Any]] = []
        claim_sequence = 0
        # 切句与 LLM 提取器共用：句末引用归前一句，引号、括号里的标点不切。
        for segment in (item.text for item in split_draft_segments(answer)):
            citation_ids = _CITATION_MARKER_RE.findall(segment)
            if _SECTION_LABEL_RE.fullmatch(segment.strip()):
                continue
            text = _clean_claim_text(segment)
            # “下降5%”也是事实，不能因长度短而漏掉；只跳过空白、纯标记和常见章节标签。
            if not text or not any(char.isalnum() for char in text):
                continue
            # 引用继承优先：主张引用的候选曾为哪些需求检索回来，主张就先归属这些需求。
            bound_ids: list[str] = []
            for citation_id in dict.fromkeys(citation_ids):
                for need_id in need_ids_by_candidate.get(citation_id, []):
                    if need_id and need_id not in bound_ids:
                        bound_ids.append(need_id)
            for need_id in _match_need_ids(text, needs):
                if need_id not in bound_ids:
                    bound_ids.append(need_id)
            claims.append(
                {
                    "claim_id": f"claim-v{draft_version}-{claim_sequence}",
                    "text": text,
                    "importance": "core" if any(need_id in core_need_ids for need_id in bound_ids) else "supporting",
                    "addressed_need_ids": bound_ids,
                    "citation_ids": list(dict.fromkeys(citation_ids)),
                }
            )
            claim_sequence += 1
        return {"claims": claims}
