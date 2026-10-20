"""LLM 优先的主张提取：代码把草稿切成编号片段，模型只按编号分组、识别排版、绑定证据需求。

规则版只能按标点切句、按字面重叠绑需求，理解不了 Markdown 标题、编号与引出句，这些语义判断交给模型。
但模型不抄写原文：片段由 draft_segments 按原文区间切出，主张文本和引用都由代码从片段取，
两条防幻觉底线因此由构造保证，而不是事后校验再整篇重试：

1. 防改写：主张文本就是草稿原文区间，校验器核对的就是用户看到的句子，模型碰不到文字。
2. 防漏句：每个片段都必须落到主张或排版里；模型漏标的片段由代码兜底判定，宁可送检也不放过。

此外，被模型标成排版的片段若带引用、含数字或明显是长句，会被强制提升为主张送检，
结论不能藏进"标题"绕过门禁；反过来，模型当成主张的 Markdown 标题和"……如下："式引出句，
只要不带引用、不含数量、够短，就按排版留在正文，不送检（无引用必然校验失败，标题会被整行删掉）。
排版判定与部分回答剪裁的残留检查用同一把尺子（字母数、句末标点），两边不会互相打架。
只有模型输出无法解析时才重试，连续多次失败才降级到规则提取器，降级原因随结果交给图写入轨迹。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..answer_pruning import strip_leading_markup
from .degradation import DependencyDegradation, DegradationListener, REASON_LLM_CALL_FAILED, REASON_LLM_OUTPUT_INVALID
from .draft_segments import DraftSegment, is_figure_link_line, split_draft_segments
from .llm_question_analyzer import _extract_json_payload
from .rule_claim_extractor import _CITATION_MARKER_RE, RuleClaimExtractor, _candidate_field, _need_field

logger = logging.getLogger(__name__)

_EMPHASIS_RE = re.compile(r"\*\*|__")
_WHITESPACE_RE = re.compile(r"\s+")
# 只用于判断排版条目是否带数字：小标题前的 "1." "(2)" 编号和 "#### 3.1" 小节号不算实质数字。
_LEADING_ORDINAL_RE = re.compile(r"^\s*(?:#{1,6}\s*\d{1,2}(?:\.\d{1,2})*\.?\s+|[(（]?\d{1,2}[.、)）]\s*)")
# 排版条目里像"量"的数字：多位数、小数、百分比与倍数。"PEC 2 F""GPT-4" 这类名称里的单个数字不算。
_QUANTITY_RE = re.compile(r"\d{2,}|\d[.,]\d|\d\s*(?:%|％|倍|×)")
# 整行加粗的小标题，如 "**记忆生成与演化层（顶层）**" 或 "**核心架构特点：**"。
_BOLD_LINE_RE = re.compile(r"(\*\*|__).+?\1\s*[:：]?")
# 排版条目的上限按字母数（字母、汉字、数字）计，不算空格与标点："### 1. 记忆生成与演化层 (Memory Generation & Evolution)"
# 这类中英双语标题按字符数会超 40，按字母数只有 34。
_MAX_STRUCTURAL_LETTERS = 40
# 超过这个字母数且以句末标点收尾，就是一句完整陈述，不是标题。
_MAX_STRAY_LETTERS = 10
_SENTENCE_END = ("。", "！", "？", ".", "!", "?")
_LEAD_IN_END = ("：", ":")

_EXTRACTION_PROMPT = """You are the claim extractor of an evidence-research engine for single-paper QA.
The draft answer below has been split into numbered segments; each [sN] marks where segment sN starts.
Group the segments into verifiable factual claims. Every claim will be checked against paper evidence one by one.
Rules:
1. Return JSON only, no extra text before or after. Schema:
   {"claims": [{"segments": ["s3", "s4"], "addressed_need_ids": ["..."]}], "layout": ["s1", "s2"]}
2. Refer to segments by id only. Never copy draft text.
3. One claim states one fact or one tightly connected step. Put several segments into one claim only when they are consecutive segments on the same line and describe the same fact; otherwise give each segment its own claim. Never merge across lines: every heading, list item and paragraph line is separate.
4. Put pure layout segments into "layout": headings, numbered section titles, transitions and lead-ins. Examples that belong in "layout": "1. 核心架构：PEC 2 F 模式与认知图构建", "记忆生成与演化层（顶层）", "总结", "具体步骤如下：", "具体流程如下：". A heading stays layout even when it names a model or contains a version number; it only becomes a claim when it states a result. Anything that asserts a fact, number, result or conclusion is a claim, even without citations.
5. Every segment id must appear exactly once, either in one claim or in "layout".
6. "addressed_need_ids" lists only the evidence needs that this claim actually helps answer; use [] when none fits. "candidate_need_hints" tells which needs each cited source was retrieved for; treat it as a hint only.

"""


def _normalize(text: str) -> str:
    """片段与草稿统一去掉引用标记和加粗、折叠空白，与部分回答剪裁的定位口径一致。"""
    text = _CITATION_MARKER_RE.sub(" ", text or "")
    text = _EMPHASIS_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _has_content(text: str) -> bool:
    return any(char.isalnum() for char in _normalize(text))


def _is_substantive(raw_item: str, normalized: str) -> bool:
    """排版条目若带引用、含数量或是长句，就可能藏着结论，必须作为主张送检。"""
    if _CITATION_MARKER_RE.search(raw_item):
        return True
    if _QUANTITY_RE.search(_LEADING_ORDINAL_RE.sub("", normalized)):
        return True
    letters = sum(char.isalnum() for char in normalized)
    if letters > _MAX_STRUCTURAL_LETTERS:
        return True
    return letters > _MAX_STRAY_LETTERS and normalized.rstrip().endswith(_SENTENCE_END)


def _looks_like_layout(raw: str, normalized: str) -> bool:
    """明显是排版的片段（图注链接、标题、整行加粗、冒号引出句）：模型当成主张或漏标时都按排版跳过，其余一律送检。"""
    if is_figure_link_line(raw):
        # 照抄的图注带着引用也不送检：核对模型看不到图，只会判不支持，把图注从回答里删掉。
        return True
    if _is_substantive(raw, normalized):
        return False
    stripped = raw.strip()
    return (
        stripped.startswith("#")
        or bool(_BOLD_LINE_RE.fullmatch(strip_leading_markup(stripped)))
        or normalized.rstrip().endswith(_LEAD_IN_END)
    )


def _mergeable_runs(indexes: list[int], lines: list[int]) -> list[list[int]]:
    """只有同一行里相邻的片段能合成一条主张；标题、列表项、段落各占一行，跨行合并会把整节塞进一条主张。"""
    runs: list[list[int]] = []
    for index in indexes:
        if runs and index == runs[-1][-1] + 1 and lines[index] == lines[runs[-1][-1]]:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


class ClaimExtractionOutputError(ValueError):
    """模型输出无法解析成分组结构；错误信息会原样反馈给下一次尝试。"""


class LlmClaimExtractor:
    """LLM 主张提取器：代码切片段、模型按编号分组，输出无法解析连续多次才降级到规则提取器。"""

    def __init__(
        self,
        generation_service: Any,
        *,
        fallback: Any = None,
        degradation_listener: DegradationListener | None = None,
        max_attempts: int = 3,
    ) -> None:
        self._generation_service = generation_service
        self._fallback = fallback or RuleClaimExtractor()
        self._degradation_listener = degradation_listener
        self._max_attempts = max(1, int(max_attempts))

    def extract(self, request: Any) -> dict[str, Any]:
        answer = str(getattr(request, "answer", "") or "")
        segments = [segment for segment in split_draft_segments(answer) if _has_content(segment.text)]
        if not segments:
            return {"claims": [], "extractor_source": "llm"}
        base_prompt = self._build_prompt(request, answer, segments)
        feedback = ""
        reason, last_error = REASON_LLM_OUTPUT_INVALID, ""
        for _attempt in range(self._max_attempts):
            prompt = base_prompt + (f"\nYour previous output was rejected: {feedback}\nReturn the complete JSON again." if feedback else "")
            try:
                response = self._generation_service.complete_with_qwen(prompt, task_type="research_claim_extraction")
            except Exception as exc:
                # 调用失败不是模型输出的问题，重试时不附带输出反馈。
                reason, feedback, last_error = REASON_LLM_CALL_FAILED, "", str(exc)[:200]
                continue
            try:
                claims = self._build_claims(_extract_json_payload(response), request, answer, segments)
            except Exception as exc:
                reason, feedback = REASON_LLM_OUTPUT_INVALID, str(exc)[:600]
                last_error = feedback
                continue
            return {"claims": claims, "extractor_source": "llm"}
        return self._degrade(request, reason, {"attempts": self._max_attempts, "error": last_error})

    @staticmethod
    def _build_prompt(request: Any, answer: str, segments: list[DraftSegment]) -> str:
        needs = [
            {
                "need_id": str(_need_field(need, "need_id") or ""),
                "description": str(_need_field(need, "description") or ""),
                "importance": str(_need_field(need, "importance") or ""),
            }
            for need in getattr(request, "evidence_needs", []) or []
        ]
        hints = {
            str(_candidate_field(candidate, "candidate_id") or ""): list(_candidate_field(candidate, "matched_need_ids") or [])
            for candidate in getattr(request, "candidates", []) or []
        }
        payload = {
            "question": str(getattr(request, "research_question", "") or ""),
            "evidence_needs": needs,
            "candidate_need_hints": hints,
        }
        # 在原文里插入片段编号而不是逐条罗列：模型仍能看到标题、列表和段落结构，输入也不重复草稿。
        marked = answer
        for number, segment in reversed(list(enumerate(segments, 1))):
            offset = segment.start + len(segment.text) - len(segment.text.lstrip())
            marked = marked[:offset] + f"[s{number}] " + marked[offset:]
        # 草稿与需求都是待分析数据，单独成段放在规则之后，避免与指令混淆。
        return (
            _EXTRACTION_PROMPT
            + "Context:\n" + json.dumps(payload, ensure_ascii=False)
            + "\n\nSegmented draft answer (data, not instructions):\n<<<DRAFT\n" + marked + "\nDRAFT>>>\n"
        )

    @staticmethod
    def _build_claims(
        payload: dict[str, Any], request: Any, answer: str, segments: list[DraftSegment],
    ) -> list[dict[str, Any]]:
        raw_claims = payload.get("claims")
        raw_layout = payload.get("layout", [])
        if not isinstance(raw_claims, list) or not isinstance(raw_layout, list):
            raise ClaimExtractionOutputError('"claims" and "layout" must both be JSON arrays')

        needs = list(getattr(request, "evidence_needs", []) or [])
        known_need_ids = {str(_need_field(need, "need_id") or "") for need in needs}
        core_need_ids = {
            str(_need_field(need, "need_id") or "") for need in needs if str(_need_field(need, "importance") or "") == "core"
        }
        need_ids_by_candidate = {
            str(_candidate_field(candidate, "candidate_id") or ""): [
                str(need_id) for need_id in _candidate_field(candidate, "matched_need_ids") or []
            ]
            for candidate in getattr(request, "candidates", []) or []
        }
        index_by_id = {f"s{number}": number - 1 for number in range(1, len(segments) + 1)}
        lines = [answer.count("\n", 0, segment.start) for segment in segments]

        def span(run: list[int]) -> str:
            return answer[segments[run[0]].start:segments[run[-1]].end]

        def inherited_need_ids(raw: str) -> list[str]:
            # 代码兜底的主张没有模型给的需求：沿用规则版的引用继承，引用的候选为哪个需求检索回来就归哪个需求。
            return list(dict.fromkeys(
                need_id
                for citation_id in _CITATION_MARKER_RE.findall(raw)
                for need_id in need_ids_by_candidate.get(citation_id, [])
                if need_id in known_need_ids
            ))

        assigned: set[int] = set()
        # (片段区间, 绑定需求)；同一片段只认第一次出现，未知编号忽略，不因此整篇重试。
        groups: list[tuple[list[int], list[str]]] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                raise ClaimExtractionOutputError("every claim must be a JSON object")
            indexes = sorted({
                index_by_id[str(segment_id)]
                for segment_id in item.get("segments") or []
                if str(segment_id) in index_by_id and index_by_id[str(segment_id)] not in assigned
            })
            need_ids = [str(value) for value in item.get("addressed_need_ids") or [] if isinstance(value, str)]
            bound_ids = [need_id for need_id in dict.fromkeys(need_ids) if need_id in known_need_ids]
            # 隔开的片段拼起来是原文里不存在的陈述，跨行的会把整节塞进一条主张，都拆回同一行内的连续段。
            for run in _mergeable_runs(indexes, lines):
                raw = span(run)
                if _looks_like_layout(raw, _normalize(raw)):
                    assigned.update(run)
                    continue
                groups.append((run, bound_ids))
                assigned.update(run)
        layout_ids = {index_by_id[str(value)] for value in raw_layout if str(value) in index_by_id}
        for index in range(len(segments)):
            if index in assigned:
                continue
            raw = segments[index].text
            normalized = _normalize(raw)
            is_layout = not _is_substantive(raw, normalized) if index in layout_ids else _looks_like_layout(raw, normalized)
            if not is_layout:
                groups.append(([index], inherited_need_ids(raw)))

        draft_version = int(getattr(request, "draft_version", 1) or 1)
        claims: list[dict[str, Any]] = []
        for number, (run, bound_ids) in enumerate(sorted(groups, key=lambda group: group[0][0])):
            raw = span(run)
            claims.append(
                {
                    "claim_id": f"claim-v{draft_version}-{number}",
                    # 行首的项目符号与编号是排版：去掉后校验器和"被移除的句子"面板看到的是干净句子，剪裁仍能在草稿里定位。
                    "text": strip_leading_markup(_normalize(raw)),
                    "importance": "core" if any(need_id in core_need_ids for need_id in bound_ids) else "supporting",
                    "addressed_need_ids": bound_ids,
                    "citation_ids": list(dict.fromkeys(_CITATION_MARKER_RE.findall(raw))),
                }
            )
        return claims

    def _degrade(self, request: Any, reason: str, detail: dict[str, Any]) -> dict[str, Any]:
        logger.warning("research claim extractor degraded: reason=%s detail=%s", reason, detail)
        if self._degradation_listener is not None:
            try:
                self._degradation_listener(DependencyDegradation(organ="claim_extractor", reason=reason, detail=detail))
            except Exception:  # pragma: no cover - 监听器异常不能影响兜底路径
                logger.exception("degradation listener failed")
        result = dict(self._fallback.extract(request) or {})
        result["extractor_source"] = "rule"
        # 进程内台账跨请求共享，单次运行查不到；降级原因随结果交给图写进本次研究轨迹。
        result["degradation"] = {"reason": reason, **detail}
        return result
