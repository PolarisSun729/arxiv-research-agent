"""Agent 追问改写：把依赖上文的追问改写成能独立理解的请求。

"数量不够，扩大一下范围重新搜"这类话单看没有主题，意图识别只能拿到噪声。
这里在意图识别之前，借最近几轮对话让小模型补全被省略的主题和条件，后续 parse
仍然只面对一句完整的话，原话依据校验、规则兜底都不需要感知多轮。
改写只补全省略，不替用户决定新的条件；任何失败都退回原句，不阻断主链路。

同一次调用里还判断用户在指哪篇论文：代码把上下文里的论文排成带编号的候选清单，
模型只挑编号，落地和校验见 utils/paper_reference。指代判断失败时不退回正则，交给
resolve_paper 追问用户。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .utils.paper_reference import (
    build_reference_candidates,
    candidates_for_prompt,
    ground_paper_reference,
    unavailable_paper_reference,
)
from .utils.text_utils import _extract_json_block, _normalize_text

logger = logging.getLogger(__name__)

# 只取最近几条消息：追问基本只依赖上一两轮，更远的历史只会增加噪声和延迟。
MAX_RECENT_MESSAGES = 6
ASSISTANT_EXCERPT_CHARS = 300
MAX_STANDALONE_REQUEST_CHARS = 300


def build_recent_turns(messages: Sequence[Mapping[str, Any]], limit: int = MAX_RECENT_MESSAGES) -> List[Dict[str, Any]]:
    """把落库的对话气泡整理成改写用的精简轮次；助手回答附上当轮检索条件和论文数。"""
    turns: List[Dict[str, Any]] = []
    for message in list(messages or [])[-limit:]:
        role = str(message.get("role") or "").strip()
        content = _normalize_text(str(message.get("content") or ""))
        if role not in {"user", "assistant"} or not content:
            continue
        turn: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            turn["content"] = content[:ASSISTANT_EXCERPT_CHARS]
            response = message.get("response") if isinstance(message.get("response"), Mapping) else {}
            search = _search_summary(response.get("search_spec"))
            if search:
                turn["search"] = search
            papers = response.get("papers")
            if isinstance(papers, list):
                turn["paper_count"] = len(papers)
        turns.append(turn)
    return turns


def _search_summary(spec: Any) -> Dict[str, Any]:
    """只保留理解追问需要的检索条件；来源标注、排序细节等对改写没有帮助。"""
    if not isinstance(spec, Mapping):
        return {}
    concepts = [
        str(concept.get("term") or "").strip()
        for concept in spec.get("concepts") or []
        if isinstance(concept, Mapping) and str(concept.get("term") or "").strip()
    ]
    summary = {
        "concepts": concepts or None,
        "query": None if concepts else spec.get("query"),
        "optional_terms": spec.get("optional_terms") or None,
        "submitted_days_ago": spec.get("submitted_days_ago"),
        "max_results": spec.get("max_results"),
    }
    return {key: value for key, value in summary.items() if value not in (None, "", [])}


def contextualize_request(
    message: str,
    recent_turns: Sequence[Mapping[str, Any]],
    generation_service: Optional[Any],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """返回 {message, original_message, rewritten, status, error, paper_reference}。

    message 是交给意图识别的那句话；paper_reference 是落地后的论文指代，上下文里没有
    候选论文时为 None（没有可指的论文，resolve_paper 直接追问）。
    """
    candidates = build_reference_candidates(context or {})
    result: Dict[str, Any] = {
        "message": message,
        "original_message": message,
        "rewritten": False,
        "status": "skipped",
        "error": None,
        "paper_reference": None,
    }
    if not message or not (recent_turns or candidates):
        return result
    if generation_service is None or not hasattr(generation_service, "complete_with_qwen"):
        result["status"] = "llm_unavailable"
        if candidates:
            result["paper_reference"] = unavailable_paper_reference("llm_unavailable")
        return result

    try:
        response = generation_service.complete_with_qwen(
            _build_prompt(message, recent_turns, candidates),
            task_type="agent_request_contextualization",
        )
        payload = json.loads(_extract_json_block(str(response)))
        if not isinstance(payload, dict):
            raise ValueError("contextualization output is not a JSON object")
    except Exception as exc:
        logger.warning("Agent request contextualization failed, keeping original message: %s", exc)
        result.update(status="failed", error=str(exc))
        if candidates:
            result["paper_reference"] = unavailable_paper_reference("failed", str(exc))
        return result

    if candidates:
        result["paper_reference"] = ground_paper_reference(payload.get("paper_reference"), candidates)
        if result["paper_reference"]["status"] != "success":
            logger.warning("Agent paper reference output rejected: %s", result["paper_reference"]["error"])
    standalone = _normalize_text(str(payload.get("standalone_request") or ""))
    if not standalone or len(standalone) > MAX_STANDALONE_REQUEST_CHARS:
        result.update(status="invalid_output", error="standalone_request is empty or too long")
        return result
    result.update(message=standalone, rewritten=standalone != message, status="success")
    return result


def _build_prompt(
    message: str,
    recent_turns: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]] = (),
) -> str:
    history = "\n".join(json.dumps(dict(turn), ensure_ascii=False) for turn in recent_turns) or "（没有更早的对话）"
    prompt = (
        "你在一个 arXiv 论文检索助手里负责理解多轮对话中的追问。\n"
        "把用户最新一句改写成脱离对话也能看懂的一句请求，交给后续的意图识别。只输出 JSON。\n"
        '格式：{"standalone_request": string}\n'
        "要求：\n"
        "- 只补全这句话依赖上文才能理解的部分，通常是省略掉的研究主题；用户原本的意思和语气不要改。\n"
        "- 上一轮的时间范围、论文数量、分类等限制条件不要带进改写句，除非用户这句话明确说要沿用；"
        "改写句里出现的条件会被当成用户亲口提的硬性要求，带多了反而会把检索收窄。\n"
        "- 用户提出的调整（更多、扩大范围、只看最近一周等）原样写进改写句，具体数值不要替用户决定。\n"
        "  例：上一轮是“找最近 30 天图神经网络的 10 篇论文”，用户说“太少了，再多找点”，"
        "应改写为“再多找一些图神经网络的论文”，而不是“再多找一些最近 30 天图神经网络的 10 篇论文”。\n"
        "- 对具体论文的指代（第 2 篇、这篇、上面那篇）原样保留，不要替换成标题。\n"
        "- 不要加入对话里没有出现过的主题或条件。\n"
        "- 这句话本身已经完整，或者换了新的研究主题时，原样返回，不要带上一轮的任何内容。\n"
        "最近的对话（role=assistant 的 search 是那一轮实际使用的检索条件，paper_count 是返回的论文数）：\n"
        f"{history}\n"
        f"用户最新一句：{message}\n"
    )
    if not candidates:
        return prompt
    candidate_lines = "\n".join(json.dumps(row, ensure_ascii=False) for row in candidates_for_prompt(candidates))
    return prompt + (
        "\n另外判断用户最新一句是不是在指某篇或某几篇具体论文，在同一个 JSON 里多输出一个字段：\n"
        '{"standalone_request": string, "paper_reference": {"kind": "none|single|multiple|ambiguous|outside", "refs": [string]}}\n'
        "候选论文（key 是编号；sources 说明它出现在哪：前端当前选中的论文、上一轮问答的目标、上一轮搜索结果第几篇等）：\n"
        f"{candidate_lines}\n"
        "判断规则：\n"
        "- none：没有指向具体论文，例如新的检索、换了研究主题、泛泛的概念提问、设置偏好。refs 为空。\n"
        "- single：指的是候选里的一篇，refs 填它的 key。代词（他、它、这篇、这个方法）、方法名、"
        "序号（第二篇）、紧接着上一轮在聊的论文继续追问，都结合对话判断。\n"
        "- multiple：同时指候选里的几篇（例如比较两篇），refs 按用户提到的顺序填。\n"
        "- ambiguous：指的是候选里的论文，但对话里分不清是哪篇；refs 填可能的几篇，最可能的放第一个。\n"
        "- outside：指的是某篇具体论文，但它不在候选里（例如说了候选里没有的标题）。refs 为空。\n"
        "- refs 只能用上面出现过的 key，不要编造。\n"
    )


__all__ = ["build_recent_turns", "contextualize_request"]
