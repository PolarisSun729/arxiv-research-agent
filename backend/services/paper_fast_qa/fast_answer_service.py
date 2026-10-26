"""快速问答：把整篇论文全文（含图片）连同最近几轮对话直接送入模型，流式作答。

一次问答的执行顺序：
  等待全文就绪（缓存命中直接继续，否则以提问优先级触发解析并等待）
  → 加载最近几轮对话 → 组装输入（系统指令 → 论文全文 → 历史对话 → 问题）→ 流式生成
  → 校验答案里的 [source:块ID]、生成对外来源 → 持久化 turn → 发出唯一的 done 事件。

论文全文放在历史对话和问题之前，同一篇论文多次提问时输入前缀保持不变，能命中模型服务的上下文缓存。
引用只做存在性校验，不逐句核对；需要逐条核对证据时使用详细模式（研究引擎）。
"""

from __future__ import annotations

import io
import json
import logging
import re
from time import monotonic, perf_counter
from typing import Any, Dict, Generator, List, Optional, Tuple
from uuid import uuid4

from openai import BadRequestError

from core.errors import AppError, ErrorCode
from services.llm.call_metrics import LLMCallStats, use_call_stats
from services.llm.generation_service import QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES, GenerationService
from services.paper_fast_qa.fulltext_service import FulltextParseError, PaperFulltext, PaperFulltextService
from services.paper_qa.answer_language import CHINESE_FINAL_ANSWER_INSTRUCTION
from services.paper_qa.evidence_contract import build_public_source_payload
from services.paper_qa.session_service import PaperQASessionService
from utils.logging_utils import RequestTrace, info_event

logger = logging.getLogger(__name__)

_CITATION_MARKER = re.compile(r"\[source:([^\]\s]+)\]")
# 等待全文解析期间向客户端发送进度的间隔，同时充当 SSE 心跳，避免代理因长时间无数据断开连接。
_PARSE_PROGRESS_INTERVAL_SECONDS = 5.0
# 请求体超限时图片统一缩到原尺寸的这个比例再试一次。
_IMAGE_DOWNSCALE_RATIO = 0.5
# 服务端拒绝超长输入时的报错特征（输入长度、token 数超出模型上限）。
_INPUT_TOO_LONG_PATTERN = re.compile(r"input length|too long|maximum context|context length|exceed", re.IGNORECASE)

_SYSTEM_INSTRUCTION = (
    "你是论文阅读助手。用户消息里提供了一篇论文的全文，按阅读顺序切成块，每块前面方括号里是块 ID；"
    "图片、公式截图和无法转成文字的表格以图片形式放在原位置，前面同样标了块 ID 和页码。\n"
    "回答要求：\n"
    "1. 只依据这篇论文回答；论文没有提到的内容直接说明论文没有提到，不要用论文以外的知识补全。\n"
    "2. 每个依据论文内容的句子后面用 [source:块ID] 标出出处，例如 [source:p3-2]、[source:fig-1]；"
    "一句话有多个出处就连续标注。只能使用全文里出现过的块 ID。\n"
    "3. 提到图片时说明是哪张图、在第几页。\n"
    "4. 数字、公式和实验结果按原文给出，不要估算或改写数值。\n"
    f"{CHINESE_FINAL_ANSWER_INSTRUCTION}"
)

_PARSE_ERROR_MESSAGES = {
    "empty": "这篇论文的 PDF 没有可提取的文字（可能是扫描版），快速模式无法读取，请改用详细模式。",
    "timeout": "论文全文读取超时，请稍后重试或改用详细模式。",
}


class PaperFastQAService:
    """快速问答编排：全文准备、输入组装、流式生成、引用整理与持久化。"""

    def __init__(
        self,
        *,
        fulltext_service: PaperFulltextService,
        generation_service: GenerationService,
        session_service: PaperQASessionService,
        ask_wait_seconds: float,
        history_turns: int,
    ):
        self.fulltext_service = fulltext_service
        self.generation_service = generation_service
        self.session_service = session_service
        self.ask_wait_seconds = float(ask_wait_seconds)
        self.history_turns = max(0, int(history_turns))

    def answer_stream(self, arxiv_id: str, payload: Any) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """逐步 yield meta / progress / delta 及唯一的 done；异常统一转成 AppError 上抛，由 router 发出 error。"""
        payload_get = PaperQASessionService.payload_get
        question = str(payload_get(payload, "question", "") or "").strip()
        run_id = str(payload_get(payload, "run_id", "") or "").strip() or str(uuid4())
        user_id = PaperQASessionService.resolve_user_id(payload_get(payload, "user_id"))
        started = perf_counter()
        stats = LLMCallStats()
        trace = RequestTrace(
            run_id=run_id, route="paper_qa.fast", user_id=user_id, input=question,
            session_id=str(payload_get(payload, "session_id", "") or "") or None,
        )
        stage = "prepare_fulltext"
        trace.add_event("qa.request_start", arxiv_id=arxiv_id, user_id=user_id, engine="fast")
        info_event(logger, "qa.request_start", run_id=run_id, arxiv_id=arxiv_id, user_id=user_id, engine="fast")
        try:
            if not question:
                raise AppError(ErrorCode.REQUEST_VALIDATION_ERROR)
            yield {"event": "meta", "data": {"status": "started", "arxiv_id": arxiv_id, "question": question, "mode": "fast"}}

            wait_started = perf_counter()
            fulltext = self.fulltext_service.load_cached(arxiv_id)
            if fulltext is None:
                fulltext = yield from self._wait_for_fulltext(arxiv_id)
            wait_ms = (perf_counter() - wait_started) * 1000

            stage = "load_conversation"
            chat_session = self.session_service.resolve_chat_session(arxiv_id, payload, create_if_missing=False)
            state = self.session_service.load_conversation_state(
                arxiv_id=arxiv_id, payload=payload, chat_session=chat_session,
            )
            history = (state.get("recent_conversation_context") or [])[-self.history_turns:] if self.history_turns else []

            stage = "build_input"
            model_input, input_debug = self._build_model_input(fulltext, history, question)
            yield {"event": "progress", "data": {"stage": "generating"}}

            stage = "generation"
            usage: Dict[str, Any] = {}
            answer_parts: List[str] = []
            first_token_ms: Optional[float] = None
            generation_started = perf_counter()
            llm_stream = self.generation_service.stream_with_qwen(
                model_input, task_type="paper_fast_qa", enable_thinking=False, usage_sink=usage,
            )
            try:
                while True:
                    try:
                        # 与研究流一致：每次 next 单独绑定统计上下文，yield 前退出。
                        with use_call_stats(stats):
                            delta = next(llm_stream)
                    except StopIteration:
                        break
                    if first_token_ms is None:
                        first_token_ms = (perf_counter() - generation_started) * 1000
                    answer_parts.append(delta)
                    yield {"event": "delta", "data": {"delta": delta}}
            except BadRequestError as exc:
                if _INPUT_TOO_LONG_PATTERN.search(str(exc)):
                    raise AppError(ErrorCode.PAPER_TOO_LONG_FOR_FAST_QA, context={"stage": stage}) from exc
                raise
            finally:
                llm_stream.close()

            stage = "persist_completed_turn"
            answer, cited_ids, invalid_ids = _normalize_citations("".join(answer_parts), fulltext)
            sources = build_public_source_payload(_source_items(fulltext, cited_ids), arxiv_id=arxiv_id)
            retrieval_debug = {
                "engine": "fast_fulltext",
                "fulltext": {
                    key: fulltext.manifest.get(key)
                    for key in ("page_count", "char_count", "image_count", "parse_seconds", "pdf_sha256", "parser")
                },
                "input": input_debug,
                "history_turn_count": len(history),
                "usage": usage,
                "wait_fulltext_ms": round(wait_ms, 1),
                "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
                "generation_ms": round((perf_counter() - generation_started) * 1000, 1),
                "invalid_citation_ids": invalid_ids,
            }
            question_contextualization = {
                "original_question": question, "contextualized_question": question,
                "status": "not_applicable", "used_short_term_memory": bool(history),
            }
            # 生成成功后才创建新会话，失败时不留下空会话。
            chat_session = chat_session or self.session_service.resolve_chat_session(arxiv_id, payload, create_if_missing=True)
            persisted = self.session_service.persist_completed_turn(
                chat_session=chat_session, question=question, answer=answer,
                source_payload=sources, retrieval_debug=retrieval_debug,
                contextualized_question=question, question_contextualization=question_contextualization,
            )
            chat_session = persisted.get("chat_session") or chat_session
            result = {
                "status": "success", "mode": "fast", "arxiv_id": arxiv_id, "question": question,
                "session_id": chat_session.get("session_id"), "chat_session": chat_session,
                "turn_id": persisted.get("turn_id"), "original_question": question,
                "persistence_status": "saved" if persisted.get("turn_id") else "not_saved",
                "contextualized_question": question, "used_short_term_memory": bool(history),
                "question_contextualization": question_contextualization,
                "answer": answer, "sources": sources, "cited_source_ids": cited_ids,
                "citation_warning": None, "retrieval_debug": retrieval_debug,
                "usage": stats.to_dict(),
            }
            elapsed_ms = (perf_counter() - started) * 1000
            trace.set_output(result)
            trace.session_id = result.get("session_id") or trace.session_id
            trace.add_event("qa.request_done", status="success")
            trace_path = _write_trace(trace, reason="auto")
            info_event(
                logger, "qa.request_done", run_id=run_id, arxiv_id=arxiv_id, status="success", engine="fast",
                elapsed_ms=round(elapsed_ms, 1), first_token_ms=retrieval_debug["first_token_ms"], trace_path=trace_path,
            )
        except Exception as exc:
            logger.exception("Fast QA failed: arxiv_id=%s run_id=%s stage=%s", arxiv_id, run_id, stage)
            if isinstance(exc, AppError):
                error = exc
                error.context = {**error.context, "arxiv_id": arxiv_id, "run_id": run_id, "stage": error.context.get("stage") or stage}
            else:
                code = {
                    "generation": ErrorCode.LLM_GENERATION_FAILED,
                    "persist_completed_turn": ErrorCode.DATABASE_WRITE_FAILED,
                }.get(stage, ErrorCode.UNKNOWN_ERROR)
                # 公开错误仅使用稳定 code/stage；原异常只留在服务器日志。
                error = AppError(
                    code, detail={"stage": stage, "arxiv_id": arxiv_id},
                    context={"arxiv_id": arxiv_id, "run_id": run_id, "stage": stage},
                )
            trace.mark_failed()
            trace.set_output(error.to_payload())
            trace.add_event("qa.request_done", status="error", code=error.code)
            trace_path = _write_trace(trace, reason="fast_qa_error")
            info_event(
                logger, "qa.request_done", run_id=run_id, arxiv_id=arxiv_id, status="error", engine="fast",
                code=error.code, elapsed_ms=round((perf_counter() - started) * 1000, 1), trace_path=trace_path,
            )
            if error is exc:
                raise
            raise error from exc

        # done 必须在持久化完成后才发出，前端收到 done 即可认为本轮已写入历史。
        yield {"event": "done", "data": result}
        return result

    def _wait_for_fulltext(self, arxiv_id: str) -> Generator[Dict[str, Any], None, PaperFulltext]:
        """以提问优先级触发解析并等待；等待期间定时发送进度，超时或失败转成 AppError。"""
        deadline = monotonic() + self.ask_wait_seconds
        waited_started = monotonic()
        while True:
            status = self.fulltext_service.get_status(arxiv_id).get("status")
            yield {"event": "progress", "data": {
                "stage": "parsing", "status": status, "waited_seconds": round(monotonic() - waited_started),
            }}
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise AppError(
                    ErrorCode.PAPER_FULLTEXT_UNAVAILABLE,
                    message="论文全文还在读取中，请稍后再问，或改用详细模式。",
                    context={"stage": "prepare_fulltext"},
                )
            try:
                fulltext = self.fulltext_service.wait_for_fulltext(
                    arxiv_id, min(_PARSE_PROGRESS_INTERVAL_SECONDS, remaining),
                )
            except FulltextParseError as exc:
                raise AppError(
                    ErrorCode.PAPER_FULLTEXT_UNAVAILABLE,
                    message=_PARSE_ERROR_MESSAGES.get(exc.reason),
                    detail={"stage": "prepare_fulltext", "code": exc.reason},
                    context={"stage": "prepare_fulltext"},
                ) from exc
            if fulltext is not None:
                return fulltext

    def _build_model_input(
        self, fulltext: PaperFulltext, history: List[Dict[str, Any]], question: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """组装 Responses API 输入；请求体超过软上限时把图片缩小一次再试，仍超限则报论文过长。"""
        model_input = _assemble_input(fulltext, history, question, image_scale=1.0)
        request_bytes = _estimate_bytes(model_input)
        downscaled = False
        if request_bytes > QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES:
            model_input = _assemble_input(fulltext, history, question, image_scale=_IMAGE_DOWNSCALE_RATIO)
            request_bytes = _estimate_bytes(model_input)
            downscaled = True
            if request_bytes > QWEN_RESPONSES_REQUEST_BODY_SOFT_LIMIT_BYTES:
                raise AppError(
                    ErrorCode.PAPER_TOO_LONG_FOR_FAST_QA,
                    detail={"stage": "build_input"},
                    context={"stage": "build_input"},
                )
        image_count = sum(1 for block in fulltext.blocks if block.get("image"))
        return model_input, {
            "request_bytes": request_bytes, "image_count": image_count, "images_downscaled": downscaled,
        }


def _assemble_input(
    fulltext: PaperFulltext, history: List[Dict[str, Any]], question: str, *, image_scale: float,
) -> List[Dict[str, Any]]:
    """系统指令 → 论文全文（图片按原位置插入）→ 最近几轮对话 → 本轮问题。"""
    paper_parts: List[Dict[str, Any]] = []
    text_buffer: List[str] = ["以下是论文全文。"]

    def flush_text() -> None:
        if text_buffer:
            paper_parts.append({"type": "input_text", "text": "\n\n".join(text_buffer)})
            text_buffer.clear()

    for block in fulltext.blocks:
        block_id, page = block["id"], block.get("page")
        if not block.get("image"):
            text_buffer.append(f"[{block_id}] {block.get('text', '')}")
            continue
        image_path = fulltext.image_path(block_id)
        label = f"[{block_id}]（第 {page} 页，{_image_kind_label(block)}）"
        if block.get("text"):
            label += f"{'图注' if block.get('kind') == 'figure' else '可提取文字'}：{block['text']}"
        text_buffer.append(label)
        if image_path is None:
            continue
        flush_text()
        image_bytes = image_path.read_bytes()
        if image_scale < 1.0:
            image_bytes = _downscale_jpeg(image_bytes, image_scale)
        paper_parts.append({
            "type": "input_image",
            "image_url": GenerationService._image_bytes_to_data_url(image_bytes, "image/jpeg"),
        })
    flush_text()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_INSTRUCTION},
        {"role": "user", "content": paper_parts},
    ]
    for turn in history:
        if turn.get("question"):
            messages.append({"role": "user", "content": str(turn["question"])})
        if turn.get("answer_summary"):
            messages.append({"role": "assistant", "content": str(turn["answer_summary"])})
    messages.append({"role": "user", "content": question})
    return messages


def _image_kind_label(block: Dict[str, Any]) -> str:
    return {"figure": "图片", "formula": "公式", "table": "表格"}.get(str(block.get("kind") or ""), "图片")


def _downscale_jpeg(image_bytes: bytes, ratio: float) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        resized = image.convert("RGB").resize(
            (max(1, int(width * ratio)), max(1, int(height * ratio))), Image.Resampling.LANCZOS,
        )
    output = io.BytesIO()
    resized.save(output, format="JPEG", quality=85)
    return output.getvalue()


def _estimate_bytes(model_input: Any) -> int:
    return len(json.dumps(model_input, ensure_ascii=False).encode("utf-8"))


def _normalize_citations(answer: str, fulltext: PaperFulltext) -> Tuple[str, List[str], List[str]]:
    """去掉指向不存在块的引用标记，返回清理后的答案、按首次出现排序的有效引用和被去掉的 ID。"""
    known_ids = {block["id"] for block in fulltext.blocks}
    cited: List[str] = []
    invalid: List[str] = []

    def replace(match: "re.Match[str]") -> str:
        source_id = match.group(1)
        if source_id not in known_ids:
            if source_id not in invalid:
                invalid.append(source_id)
            return ""
        if source_id not in cited:
            cited.append(source_id)
        return match.group(0)

    cleaned = _CITATION_MARKER.sub(replace, answer).strip()
    return cleaned, cited, invalid


def _source_items(fulltext: PaperFulltext, cited_ids: List[str]) -> List[Dict[str, Any]]:
    """把被引用的块转成来源条目；带图片的块按 figure 输出，前端据此展示缩略图。"""
    by_id: Dict[str, Dict[str, Any]] = {}
    section_title = ""
    for block in fulltext.blocks:
        if block.get("kind") == "heading":
            section_title = block.get("text", "").lstrip("#").replace("**", "").strip()
        by_id[block["id"]] = {**block, "section_title": section_title}
    items: List[Dict[str, Any]] = []
    for source_id in cited_ids:
        block = by_id[source_id]
        has_image = bool(block.get("image"))
        chunk_type = "figure" if has_image else ("table" if block.get("kind") == "table" else "text")
        items.append({
            "source_id": source_id,
            "chunk_id": source_id,
            "content": block.get("text") or f"第 {block.get('page')} 页{_image_kind_label(block)}",
            "page_number": block.get("page"),
            "section_title": block.get("section_title", ""),
            "chunk_type": chunk_type,
            "has_asset": has_image,
        })
    return items


def _write_trace(trace: RequestTrace, *, reason: str) -> Optional[str]:
    """trace 只辅助排查；写入失败不能影响问答主链路。"""
    try:
        return trace.write(reason=reason)
    except Exception as exc:  # pragma: no cover
        logger.warning("fast QA trace write failed: run_id=%s reason=%s error=%s", trace.run_id, reason, exc)
        return None
