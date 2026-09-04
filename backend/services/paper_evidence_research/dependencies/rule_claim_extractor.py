"""规则版主张提取：切句、解析引用标记、绑定证据需求。

三条契约必须与图的完成门禁对齐：

1. 只有携带至少一个合法 ``[source:id]`` 引用标记的句子才构成可校验主张。门禁要求
   主张的 citation_ids 与校验器的支撑证据完全一致，未挂引用的句子永远无法被支撑，
   让它成为 claim 只会堵死 completed 终态。答案里的引用标记由草稿生成器负责写入，
   格式与 services/paper_qa/citation_contract.py 的严格协议一致。
2. 主张绑定需求走两级信号：优先"引用继承"——草稿声明使用的候选带着 matched_need_ids
   （它就是为那个需求检索回来的），主张引用哪个候选就继承其匹配需求；词面重叠作为
   无候选信息时的兜底。两级都匹配不到时保持不绑定——宁可让需求保持 open 触发继续
   检索，也不能把无关主张硬绑到需求上造成虚假满足。
3. importance 按绑定到的需求定级：绑到 core 需求即 core，其余为 supporting。
"""

from __future__ import annotations

import re
from typing import Any

# 与 services/paper_qa/citation_contract.py 的 _STRICT_MARKER_RE 保持同构。
_CITATION_MARKER_RE = re.compile(r"\[source:([^\]\s,]+)\]")
_SEGMENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")
_LEADING_META_RE = re.compile(
    r"^(?:本文|该论文|这篇论文|作者们?|我们)\s*"
    r"(?:提出|认为|指出|发现|证明|介绍|给出|采用)\s*"
    r"(?:了|的)?[：：，,]?\s*"
    r"|^(?:this paper|the paper|the authors|we)\s+"
    r"(?:propose|proposes|present|presents|show|shows|find|finds|introduce|introduces)\s+that\s+",
    flags=re.IGNORECASE,
)
_MIN_CLAIM_CHARS = 6
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


def _clean_claim_text(segment: str) -> str:
    text = _CITATION_MARKER_RE.sub("", segment)
    text = _LEADING_META_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip(" \t，,。.；;：:")


class RuleClaimExtractor:
    """确定性主张提取器；粗粒度是有意取舍——切粗只影响校验粒度，不引入误判。"""

    def extract(self, request: Any) -> dict[str, Any]:
        answer = str(getattr(request, "answer", "") or "")
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
        for segment in _SEGMENT_SPLIT_RE.split(answer):
            citation_ids = _CITATION_MARKER_RE.findall(segment)
            if not citation_ids:
                continue
            text = _clean_claim_text(segment)
            if len(text) < _MIN_CLAIM_CHARS:
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
