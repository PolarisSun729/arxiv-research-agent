"""检索索引层：PaperChunk → 多视角 RetrievalIndex，以及 collection 级 BM25 / lookup 索引。

- RetrievalIndexBuilder：建库期为每个 chunk 生成 body / section_anchor / table_or_figure / summary / question 索引；
- save_* / load_*：retrieval index 与 sparse（BM25）artifact 的落盘和读取；
- CollectionRetrievalIndexProvider：运行期优先加载 sparse artifact，失败时从 retrieval index artifact、
  chunk 文件或向量库现场构建，并按 collection 缓存。
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from services.retrieval.collection_profile import CollectionRetrievalProfile
from services.paper_qa.build_cache import (
    LLM_RETRIEVAL_QUESTIONS_PROMPT_VERSION,
    LLM_RETRIEVAL_SUMMARY_PROMPT_VERSION,
)
from utils.config import get_enhanced_retrieval_runtime_config

logger = logging.getLogger(__name__)

ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

# BM25 各字段的权重：章节/标题加权，图表 caption 与 OCR 辅助文本大幅降权以压制噪声。
KEYWORD_FIELD_WEIGHTS = {
    "body": 1.0,
    "title": 1.35,
    "section_title": 1.2,
    "section_path": 0.85,
    "summary": 0.95,
    "question": 0.9,
    "table_or_figure": 1.05,
    "asset_caption": 0.18,
    "asset_aux": 0.08,
}

RETRIEVAL_INDEX_TYPES = {"body", "section_anchor", "summary", "question", "table_or_figure"}
VECTOR_ROUTE_NAMES = {"vector", "vector_original", "vector_rewrite", "vector_hyde"}
KEYWORD_ROUTE_NAMES = {"keyword"}

# 各类索引默认参与的召回路线；section_anchor 只做关键词入口，不进向量库。
RETRIEVAL_INDEX_ROUTE_DEFAULTS = {
    "body": ["vector_original", "vector_rewrite", "vector_hyde", "keyword"],
    "section_anchor": ["keyword"],
    "summary": ["vector_original", "vector_rewrite", "vector_hyde", "keyword"],
    "question": ["vector_original", "vector_rewrite", "vector_hyde", "keyword"],
    "table_or_figure": ["vector_original", "vector_rewrite", "vector_hyde", "keyword", "table_structured"],
}

# 各类索引的默认权重：关键词后端用它缩放字段权重，index 级命中聚合回 chunk 时也会参考。
RETRIEVAL_INDEX_WEIGHTS = {
    "body": 1.0,
    "section_anchor": 0.45,
    "summary": 0.78,
    "question": 0.82,
    "table_or_figure": 1.12,
}

DEFAULT_RETRIEVAL_INDEX_MAX_PER_CHUNK = 6
DEFAULT_RETRIEVAL_INDEX_MAX_QUESTIONS_PER_CHUNK = 3
RETRIEVAL_INDEX_PREVIEW_CHARS = 360
RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION = "retrieval_index_artifact_v1"
DEFAULT_RETRIEVAL_INDEX_ARTIFACT_DIR = "02-retrieval-indexes"
SPARSE_INDEX_ARTIFACT_SCHEMA_VERSION = "sparse_index_artifact_v1"
BM25_SCHEMA_VERSION = "field_weighted_bm25_v1"
KEYWORD_TOKENIZER_VERSION = "retrieval_rules_keyword_tokenizer_v1"
KEYWORD_FIELD_WEIGHTS_VERSION = "keyword_field_weights_v1"
DEFAULT_SPARSE_INDEX_ARTIFACT_DIR = "02-sparse-indexes"
SPARSE_SOURCE_TYPES = {"chunk", "retrieval_index"}
SPARSE_SOURCE_TYPE_ALIASES = {
    "chunk_file": "chunk",
    "chunk-level": "chunk",
    "chunk_level": "chunk",
    "retrieval_index_artifact": "retrieval_index",
    "retrieval_index_file": "retrieval_index",
}
# 决定 token、字段权重与 BM25 结构含义的语义版本；任一不一致时持久化的 postings 都不能复用。
_SPARSE_SEMANTIC_VERSIONS = (
    ("tokenizer_version", KEYWORD_TOKENIZER_VERSION),
    ("field_weights_version", KEYWORD_FIELD_WEIGHTS_VERSION),
    ("bm25_schema_version", BM25_SCHEMA_VERSION),
)
# CollectionRetrievalIndex 上的 lookup 表（键 → doc_id 列表），构建、加载与复制时统一按这份清单处理。
_LOOKUP_TABLE_NAMES = (
    "by_chunk_id",
    "by_parent_chunk_id",
    "by_original_chunk_id",
    "by_page_section",
    "by_page_number",
    "by_section_path",
    "by_section_title",
    "by_chunk_type",
    "by_subchunk_index",
    "by_parent_subchunk",
)

# PaperChunk 保留的稳定元数据字段（其余 payload 字段只留在 PaperChunk.payload 中）。
PAPER_CHUNK_METADATA_KEYS = {
    "source",
    "document_name",
    "chunk_id",
    "chunk_index",
    "parent_chunk_id",
    "original_chunk_id",
    "total_chunks",
    "word_count",
    "rerank_text",
    "chunk_type",
    "page_number",
    "page_start",
    "page_end",
    "page_range",
    "section_path",
    "section_title",
    "section_level",
    "asset_kind",
    "asset_path",
    "asset_abs_path",
    "asset_summary",
    "asset_preview_text",
    "asset_caption",
    "asset_rows",
    "asset_columns",
    "table_structured_text",
    "table_id",
    "order_index",
    "subchunk_index",
    "subchunk_count",
    "subchunk_label",
    "content_part_index",
    "content_part_count",
    "content_part_label",
}


@dataclass
class PaperChunk:
    """最终证据单元的轻量模型；现有 chunk payload 仍是对外兼容的真实载体。"""

    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Dict[str, Any], fallback_index: int = 0) -> "PaperChunk":
        """从 chunk payload 构造；顶层字段优先于 metadata，空值字段不进入 metadata。"""
        metadata = dict(chunk.get("metadata", {}) or {})
        chunk_id = normalize_chunk_id(chunk, fallback_index=fallback_index)
        content = str(chunk.get("content") or chunk.get("text") or metadata.get("content") or metadata.get("text") or "")
        # 只补齐派生索引需要的稳定元数据，避免把 index 层字段混回 PaperChunk 的核心语义。
        normalized_metadata = {}
        for key in PAPER_CHUNK_METADATA_KEYS:
            value = _chunk_or_metadata_value(chunk, metadata, key)
            if value not in (None, ""):
                normalized_metadata[key] = value
        normalized_metadata.setdefault("chunk_id", chunk_id)
        return cls(chunk_id=chunk_id, content=content, metadata=normalized_metadata, payload=dict(chunk))


@dataclass
class RetrievalIndex:
    """检索入口单元；命中它以后必须通过 chunk_id 回填到 PaperChunk。"""

    index_id: str
    chunk_id: str
    index_type: str
    index_text: str
    index_weight: float = 1.0
    enabled_routes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成可落盘的字典；列表和 metadata 复制一份，避免与实例共享。"""
        return {
            "index_id": self.index_id,
            "chunk_id": self.chunk_id,
            "index_type": self.index_type,
            "index_text": self.index_text,
            "index_weight": self.index_weight,
            "enabled_routes": list(self.enabled_routes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievalIndex":
        """兼容短字段名（index_*）和向量库 metadata 中的长字段名（retrieval_index_*）。"""
        index_type = normalize_index_type(payload.get("index_type") or payload.get("retrieval_index_type"))
        return cls(
            index_id=str(payload.get("index_id") or payload.get("retrieval_index_id") or ""),
            chunk_id=str(payload.get("chunk_id") or payload.get("retrieval_index_chunk_id") or ""),
            index_type=index_type,
            index_text=str(payload.get("index_text") or payload.get("retrieval_index_text") or ""),
            index_weight=float(
                payload.get(
                    "index_weight",
                    payload.get("retrieval_index_weight", RETRIEVAL_INDEX_WEIGHTS.get(index_type, 1.0)),
                )
                or 0.0
            ),
            enabled_routes=normalize_enabled_routes(payload.get("enabled_routes") or payload.get("retrieval_index_enabled_routes")),
            metadata=dict(payload.get("metadata", payload.get("retrieval_index_metadata", {})) or {}),
        )


def _chunk_or_metadata_value(chunk: Dict[str, Any], metadata: Dict[str, Any], key: str) -> Any:
    """chunk 顶层有这个键就用顶层值（即使为空），否则读 metadata。"""
    return chunk.get(key) if key in chunk else metadata.get(key)


def normalize_index_type(value: Any) -> str:
    """规范化索引类型；未知类型一律视为 body。"""
    index_type = str(value or "body").strip().lower()
    return index_type if index_type in RETRIEVAL_INDEX_TYPES else "body"


def normalize_chunk_id(chunk: Dict[str, Any], fallback_index: int = 0) -> str:
    """按 chunk_id → chunk_index → parent_chunk_id → original_chunk_id 取第一个非空值，都没有时用 fallback_index。"""
    metadata = dict(chunk.get("metadata", {}) or {})
    for key in ("chunk_id", "chunk_index", "parent_chunk_id", "original_chunk_id"):
        normalized = str(_chunk_or_metadata_value(chunk, metadata, key) or "").strip()
        if normalized:
            return normalized
    return str(fallback_index or 0)


def normalize_enabled_routes(value: Any) -> List[str]:
    """把 enabled_routes 统一成字符串列表；支持列表、JSON 数组字符串以及逗号/竖线分隔的字符串。"""
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in re.split(r"[,|]", raw) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _routes_allow(raw_routes: Any, route_names: Set[str]) -> bool:
    """未声明 enabled_routes 视为对所有路线开放；否则只要与 route_names 有交集即可。"""
    routes = normalize_enabled_routes(raw_routes)
    if not routes:
        return True
    return bool(set(routes) & route_names)


def retrieval_index_enabled_for(index: RetrievalIndex | Dict[str, Any], route_names: Set[str]) -> bool:
    """判断索引是否参与给定路线（如向量路线集合 VECTOR_ROUTE_NAMES）。"""
    return _routes_allow(index.enabled_routes if isinstance(index, RetrievalIndex) else index.get("enabled_routes"), route_names)


class RetrievalIndexBuilder:
    """为 PaperChunk 生成多视角 RetrievalIndex，并把生成式失败限制在单 chunk 内。"""

    def __init__(
        self,
        *,
        generation_service: Any = None,
        enable_generative_indexes: bool = False,
        max_indexes_per_chunk: int = DEFAULT_RETRIEVAL_INDEX_MAX_PER_CHUNK,
        max_questions_per_chunk: int = DEFAULT_RETRIEVAL_INDEX_MAX_QUESTIONS_PER_CHUNK,
        max_workers: int = 1,
        generation_cache: Any = None,
        generation_model_name: str = "",
    ) -> None:
        self.generation_service = generation_service
        self.enable_generative_indexes = bool(enable_generative_indexes)
        self.max_indexes_per_chunk = max(2, int(max_indexes_per_chunk or DEFAULT_RETRIEVAL_INDEX_MAX_PER_CHUNK))
        # 0 是显式关闭 question index 的有效配置，不能被 Python 的 truthy fallback 恢复成默认值。
        question_limit = (
            DEFAULT_RETRIEVAL_INDEX_MAX_QUESTIONS_PER_CHUNK
            if max_questions_per_chunk is None
            else max_questions_per_chunk
        )
        self.max_questions_per_chunk = max(0, min(3, int(question_limit)))
        self.max_workers = max(1, int(max_workers or 1))
        self.generation_cache = generation_cache
        self.generation_model_name = str(generation_model_name or "retrieval_index_generation")

    def build(self, chunks: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """为全部 chunk 建索引，返回 (索引字典列表, 汇总 debug)；输出顺序与输入 chunk 顺序一致。"""
        all_indexes: List[Dict[str, Any]] = []
        chunk_debug: List[Dict[str, Any]] = []
        chunk_items = [(fallback_index, dict(chunk or {})) for fallback_index, chunk in enumerate(chunks or [], start=1)]
        if self.enable_generative_indexes and self.generation_service is not None and self.max_workers > 1 and len(chunk_items) > 1:
            # 生成式 summary/question 只依赖单个 chunk；有界并发能缩短建库等待，同时保留顺序稳定性便于 artifact 对比。
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="retrieval-index") as executor:
                built_items = list(executor.map(lambda item: self.build_for_chunk(item[1], fallback_index=item[0]), chunk_items))
        else:
            built_items = [self.build_for_chunk(chunk, fallback_index=fallback_index) for fallback_index, chunk in chunk_items]

        for indexes, debug in built_items:
            all_indexes.extend(index.to_dict() for index in indexes)
            chunk_debug.append(debug)

        summary = summarize_retrieval_indexes(all_indexes)
        generation_errors = [
            error
            for item in chunk_debug
            for error in item.get("generation_errors", [])
        ]
        summary.update(
            {
                "chunk_count": len(chunk_debug),
                "generative_enabled": self.enable_generative_indexes,
                "max_indexes_per_chunk": self.max_indexes_per_chunk,
                "max_questions_per_chunk": self.max_questions_per_chunk,
                "chunks_with_body_index": sum(1 for item in chunk_debug if "body" in item.get("index_types", [])),
                "chunks_with_section_anchor_index": sum(
                    1 for item in chunk_debug if "section_anchor" in item.get("index_types", [])
                ),
                "generated_summary_count": sum(int(item.get("generated_summary_count", 0) or 0) for item in chunk_debug),
                "generated_question_count": sum(int(item.get("generated_question_count", 0) or 0) for item in chunk_debug),
                "cached_summary_count": sum(int(item.get("cached_summary_count", 0) or 0) for item in chunk_debug),
                "cached_question_count": sum(int(item.get("cached_question_count", 0) or 0) for item in chunk_debug),
                "generation_error_count": len(generation_errors),
                "generation_errors": generation_errors[:20],
                "per_chunk": chunk_debug[:50],
            }
        )
        return all_indexes, summary

    def build_for_chunk(self, chunk: Dict[str, Any], fallback_index: int = 0) -> tuple[List[RetrievalIndex], Dict[str, Any]]:
        """为单个 chunk 生成索引，总数不超过 max_indexes_per_chunk。

        顺序即优先级：body → section_anchor →（图表）table_or_figure → summary → question，
        超出上限的索引直接丢弃，所以保底的规则索引总能保留下来。
        """
        paper_chunk = PaperChunk.from_chunk(chunk, fallback_index=fallback_index)
        metadata = dict(paper_chunk.metadata)
        chunk_type = str(metadata.get("chunk_type", chunk.get("chunk_type", "text")) or "text").strip().lower()
        metadata["chunk_type"] = chunk_type
        is_asset = chunk_type in {"figure", "table"}
        indexes: List[RetrievalIndex] = []
        generation_errors: List[Dict[str, Any]] = []
        generated_summary_count = 0
        generated_question_count = 0
        cached_summary_count = 0
        cached_question_count = 0

        def add_index(
            index_type: str,
            text: str,
            *,
            source_fields: List[str],
            generation_status: str = "rule",
        ) -> None:
            """追加一个索引；已达上限时静默丢弃。"""
            if len(indexes) >= self.max_indexes_per_chunk:
                return
            normalized_type = normalize_index_type(index_type)
            ordinal = 1 + sum(1 for item in indexes if item.index_type == normalized_type)
            index_metadata = {
                **metadata,
                "source_fields": source_fields,
                "generation_status": generation_status,
                "has_index_text": bool(str(text or "").strip()),
            }
            # index_id 只依赖 chunk_id、类型和序号，保证同一 chunk 重建时标识稳定，便于 trace 对比。
            indexes.append(
                RetrievalIndex(
                    index_id=f"{paper_chunk.chunk_id}:{normalized_type}:{ordinal}",
                    chunk_id=paper_chunk.chunk_id,
                    index_type=normalized_type,
                    index_text=str(text or "").strip(),
                    index_weight=float(RETRIEVAL_INDEX_WEIGHTS.get(normalized_type, 1.0)),
                    enabled_routes=list(RETRIEVAL_INDEX_ROUTE_DEFAULTS.get(normalized_type, ["keyword"])),
                    metadata=index_metadata,
                )
            )

        # body 和 section_anchor 是每个 chunk 的保底入口；后续生成式索引失败时仍能完成建库。
        body_text = self._asset_index_text(paper_chunk, metadata) if is_asset else paper_chunk.content
        add_index("body", body_text, source_fields=["asset_fields" if is_asset else "content"])

        section_anchor_text = self._section_anchor_text(paper_chunk, metadata)
        add_index("section_anchor", section_anchor_text, source_fields=["section_path", "section_title", "page_number", "summary"])

        # 图表/表格 chunk 的 body_text 就是 caption、summary、preview 和建库期结构化表格文本的合并；
        # 不混入问答运行时生成的 table evidence，避免索引语料随单次问题变化。
        if is_asset and body_text:
            add_index(
                "table_or_figure",
                body_text,
                source_fields=["asset_caption", "asset_summary", "asset_preview_text", "table_structured_text", "content"],
            )

        # summary/question 是召回增强层；数量受上限约束，避免一个 chunk 扩张出过多向量。
        summary_text, summary_status, summary_errors = self._summary_text(chunk, paper_chunk, metadata)
        generation_errors.extend(summary_errors)
        if summary_text and self._normalized_text(summary_text) != self._normalized_text(body_text):
            add_index("summary", summary_text, source_fields=["generated_summary"], generation_status=summary_status)
            if summary_status == "generated_llm":
                generated_summary_count += 1
            if summary_status == "cached_llm":
                cached_summary_count += 1

        question_texts, question_status, question_errors = self._question_texts(chunk, paper_chunk, metadata)
        generation_errors.extend(question_errors)
        for question_text in question_texts[: self.max_questions_per_chunk]:
            add_index("question", question_text, source_fields=["retrieval_questions", "generated_questions"], generation_status=question_status)
            if question_status == "generated_llm":
                generated_question_count += 1
            if question_status == "cached_llm":
                cached_question_count += 1

        debug = {
            "chunk_id": paper_chunk.chunk_id,
            "chunk_type": chunk_type,
            "index_count": len(indexes),
            "index_types": [index.index_type for index in indexes],
            "generated_summary_count": generated_summary_count,
            "generated_question_count": generated_question_count,
            "cached_summary_count": cached_summary_count,
            "cached_question_count": cached_question_count,
            "generation_errors": generation_errors,
        }
        return indexes, debug

    def _summary_text(
        self,
        chunk: Dict[str, Any],
        paper_chunk: PaperChunk,
        metadata: Dict[str, Any],
    ) -> tuple[str, str, List[Dict[str, Any]]]:
        """返回 (摘要文本, 来源状态, 生成错误)。

        开启生成式索引时查缓存或调用 LLM 生成检索用摘要。不复用 rerank 卡片（rerank_text）：
        它是给 rerank 看的结构化提示，带规则推断的「用途」字样，放进向量索引会让很多 chunk 都贴近同类问题。
        LLM 无结果或失败时不出 summary 索引，正文截断与 body 索引重复，没有额外召回价值。
        """
        if not self.enable_generative_indexes:
            return "", "disabled", []
        cache_kwargs = {
            "kind": "retrieval_summary",
            "model_name": self.generation_model_name,
            "prompt_version": LLM_RETRIEVAL_SUMMARY_PROMPT_VERSION,
            "chunk_text": self._index_source_text(paper_chunk, metadata),
            "metadata": metadata,
            "extra": {"preview_chars": RETRIEVAL_INDEX_PREVIEW_CHARS},
        }
        cached = self._get_cached_generation(cache_kwargs)
        if isinstance(cached, str) and cached.strip():
            return cached.strip(), "cached_llm", []
        try:
            generated = self._generate_summary_with_service(paper_chunk, metadata)
            if generated:
                self._set_cached_generation(cache_kwargs, generated)
                return generated, "generated_llm", []
        except Exception as exc:
            # 生成失败只少一个增强索引，body 等规则索引已经足够维持后续 embedding/BM25 流程。
            return "", "generation_failed", [
                {"chunk_id": paper_chunk.chunk_id, "index_type": "summary", "error": str(exc)[:300]}
            ]
        return "", "generation_failed", []

    def _question_texts(
        self,
        chunk: Dict[str, Any],
        paper_chunk: PaperChunk,
        metadata: Dict[str, Any],
    ) -> tuple[List[str], str, List[Dict[str, Any]]]:
        """返回 (问题列表, 来源状态, 生成错误)；chunk 自带问题时直接复用，否则同 _summary_text，失败时不出 question 索引。"""
        existing_questions = extract_question_index_texts(chunk)
        if existing_questions:
            return self._dedupe_texts(existing_questions), "reused_existing", []
        if not self.enable_generative_indexes or self.max_questions_per_chunk <= 0:
            return [], "disabled", []
        cache_kwargs = {
            "kind": "retrieval_questions",
            "model_name": self.generation_model_name,
            "prompt_version": LLM_RETRIEVAL_QUESTIONS_PROMPT_VERSION,
            "chunk_text": self._index_source_text(paper_chunk, metadata),
            "metadata": metadata,
            "extra": {"max_questions": self.max_questions_per_chunk},
        }
        cached = self._get_cached_generation(cache_kwargs)
        if isinstance(cached, list) and cached:
            return self._dedupe_texts([str(item) for item in cached])[: self.max_questions_per_chunk], "cached_llm", []
        try:
            generated = self._generate_questions_with_service(paper_chunk, metadata)
            if generated:
                normalized = self._dedupe_texts(generated)[: self.max_questions_per_chunk]
                self._set_cached_generation(cache_kwargs, normalized)
                return normalized, "generated_llm", []
        except Exception as exc:
            # 问题索引是可选召回视角；模板问题对所有 chunk 都差不多，兜底反而制造噪声，失败只记错误。
            return [], "generation_failed", [
                {"chunk_id": paper_chunk.chunk_id, "index_type": "question", "error": str(exc)[:300]}
            ]
        return [], "generation_failed", []

    def _get_cached_generation(self, cache_kwargs: Dict[str, Any]) -> Any:
        """读生成缓存；未配置缓存或缓存不支持读取时返回 None。"""
        getter = getattr(self.generation_cache, "get_llm_result", None)
        if not callable(getter):
            return None
        return getter(**cache_kwargs)

    def _set_cached_generation(self, cache_kwargs: Dict[str, Any], value: Any) -> None:
        """写生成缓存；未配置缓存时什么也不做。"""
        setter = getattr(self.generation_cache, "set_llm_result", None)
        if callable(setter):
            setter(value, **cache_kwargs)

    def _generate_summary_with_service(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> str:
        """优先调用生成服务的专用摘要接口，没有时用通用 complete_with_qwen + JSON prompt。"""
        if self.generation_service is None:
            return ""
        generator = getattr(self.generation_service, "generate_retrieval_index_summary", None)
        if callable(generator):
            result = generator(chunk=paper_chunk.payload, max_chars=RETRIEVAL_INDEX_PREVIEW_CHARS)
            return self._normalize_generated_text(result)
        completer = getattr(self.generation_service, "complete_with_qwen", None)
        if not callable(completer):
            return ""
        prompt = self._summary_prompt(paper_chunk, metadata)
        # 复用现有生成服务时只接受严格 JSON，防止自由文本被误当成可追踪的 index_text。
        response = completer(prompt, task_type="retrieval_index_generation", enable_thinking=False)
        parsed = self._parse_generated_payload(response)
        return self._normalize_generated_text(parsed.get("summary", ""))

    def _generate_questions_with_service(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> List[str]:
        """优先调用生成服务的专用问题接口，没有时用通用 complete_with_qwen + JSON prompt。"""
        if self.generation_service is None:
            return []
        generator = getattr(self.generation_service, "generate_retrieval_index_questions", None)
        if callable(generator):
            result = generator(
                chunk=paper_chunk.payload,
                max_questions=self.max_questions_per_chunk,
            )
            return self._normalize_generated_questions(result)
        completer = getattr(self.generation_service, "complete_with_qwen", None)
        if not callable(completer):
            return []
        prompt = self._question_prompt(paper_chunk, metadata)
        # question index 直接进入召回入口，因此解析失败必须显式降级而不是吞掉格式问题。
        response = completer(prompt, task_type="retrieval_index_generation", enable_thinking=False)
        parsed = self._parse_generated_payload(response)
        return self._normalize_generated_questions(parsed.get("questions", []))

    def _summary_prompt(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> str:
        """摘要生成 prompt：任务说明 + chunk 上下文。"""
        return (
            "请为论文 QA 检索索引生成一个短摘要，必须只基于给定 chunk，不要补充外部信息。\n"
            "摘要使用与 chunk content 相同的语言书写（content 是英文就写英文）。\n"
            "输出严格 JSON：{\"summary\":\"...\"}\n"
            + self._prompt_context(paper_chunk, metadata)
        )

    def _question_prompt(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> str:
        """问题生成 prompt：任务说明 + chunk 上下文。"""
        return (
            "请为论文 QA 检索索引生成 1-3 个这个 chunk 可以回答的问题。\n"
            "要求：问题必须具体、可由该 chunk 直接支持；不要生成无法从 chunk 证明的问题。\n"
            "问题使用与 chunk content 相同的语言书写（content 是英文就写英文）。\n"
            "输出严格 JSON：{\"questions\":[\"...\",\"...\"]}\n"
            + self._prompt_context(paper_chunk, metadata)
        )

    def _prompt_context(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> str:
        """两种生成 prompt 共用的 chunk 上下文：类型、章节、页码和截断到 900 字符的内容。"""
        content = self._short_text(self._index_source_text(paper_chunk, metadata), limit=900)
        return (
            f"chunk_type: {metadata.get('chunk_type', 'text')}\n"
            f"section: {metadata.get('section_path') or metadata.get('section_title') or ''}\n"
            f"page: {metadata.get('page_number') or metadata.get('page_range') or ''}\n"
            f"content:\n{content}"
        )

    @classmethod
    def _parse_generated_payload(cls, value: Any) -> Dict[str, Any]:
        """解析 LLM 输出的 JSON：兼容 ```json 代码块和前后夹杂文字；顶层是数组时视为 questions。

        JSON 不合法时直接抛出异常，由调用方记录为生成错误并降级。
        """
        if isinstance(value, dict):
            return value
        raw = str(value or "").strip()
        if not raw:
            return {}
        match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, flags=re.DOTALL)
        candidate = match.group(1).strip() if match else raw
        if not (candidate.startswith("{") or candidate.startswith("[")):
            braces = re.search(r"(\{.*\}|\[.*\])", candidate, flags=re.DOTALL)
            candidate = braces.group(1).strip() if braces else candidate
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return {"questions": parsed}
        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _normalize_generated_text(value: Any) -> str:
        """把生成结果（字符串 / 字典 / 列表）统一成单行文本。"""
        if isinstance(value, dict):
            value = value.get("summary") or value.get("text") or ""
        if isinstance(value, list):
            value = " ".join(str(item).strip() for item in value if str(item).strip())
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @classmethod
    def _normalize_generated_questions(cls, value: Any) -> List[str]:
        """把生成结果统一成去重后的问题列表，并去掉 "1." / "-" 之类的列表前缀。"""
        if isinstance(value, dict):
            value = value.get("questions") or value.get("question") or []
        if isinstance(value, str):
            value = re.split(r"[\n;]+", value)
        if not isinstance(value, (list, tuple, set)):
            return []
        normalized = []
        for item in value:
            text = re.sub(r"^\s*[-*\d.)]+\s*", "", str(item or "")).strip()
            if text:
                normalized.append(text)
        return cls._dedupe_texts(normalized)

    def _asset_index_text(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> str:
        """图表类文本：caption、摘要、预览、结构化表格文本和正文去重后按行拼接。"""
        parts = [
            str(metadata.get("asset_caption") or "").strip(),
            str(metadata.get("asset_summary") or "").strip(),
            str(metadata.get("asset_preview_text") or "").strip(),
            str(metadata.get("table_structured_text") or "").strip(),
            paper_chunk.content.strip(),
        ]
        return "\n".join(self._dedupe_texts([part for part in parts if part]))

    def _index_source_text(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> str:
        """生成摘要、问题和锚点时使用的源文本：图表字段与正文的合并，全部为空时退回原始正文。"""
        return self._asset_index_text(paper_chunk, metadata) or paper_chunk.content

    def _section_anchor_text(self, paper_chunk: PaperChunk, metadata: Dict[str, Any]) -> str:
        """章节锚点文本：章节、页码和一句摘要；三者都缺时退化为 "Chunk: <id>"。"""
        summary = str(metadata.get("rerank_text") or metadata.get("asset_summary") or "").strip()
        if not summary:
            summary = self._short_text(self._index_source_text(paper_chunk, metadata), limit=180)
        parts = [
            f"Section: {metadata.get('section_path') or metadata.get('section_title')}" if metadata.get("section_path") or metadata.get("section_title") else "",
            f"Page: {metadata.get('page_number') or metadata.get('page_range')}" if metadata.get("page_number") or metadata.get("page_range") else "",
            f"Summary: {summary}" if summary else "",
            f"Chunk: {paper_chunk.chunk_id}" if not summary and not metadata.get("section_path") and not metadata.get("section_title") else "",
        ]
        return "\n".join([part for part in parts if part]).strip() or f"Chunk: {paper_chunk.chunk_id}"

    @staticmethod
    def _short_text(text: str, limit: int = RETRIEVAL_INDEX_PREVIEW_CHARS) -> str:
        """压缩空白后截断到 limit 个字符，尽量在单词边界处截断（不加省略号）。"""
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rsplit(" ", 1)[0].strip() or normalized[:limit].strip()

    @staticmethod
    def _normalized_text(text: str) -> str:
        """去重比较用的归一化：小写并压缩空白。"""
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @classmethod
    def _dedupe_texts(cls, texts: Iterable[str]) -> List[str]:
        """按归一化结果保序去重，保留首次出现的原始写法（去首尾空白）。"""
        deduped: List[str] = []
        seen: Set[str] = set()
        for text in texts:
            normalized = cls._normalized_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(str(text).strip())
        return deduped


def build_retrieval_indexes(
    chunks: Iterable[Dict[str, Any]],
    *,
    generation_service: Any = None,
    enable_generative_indexes: bool = False,
    max_indexes_per_chunk: int = DEFAULT_RETRIEVAL_INDEX_MAX_PER_CHUNK,
    max_questions_per_chunk: int = DEFAULT_RETRIEVAL_INDEX_MAX_QUESTIONS_PER_CHUNK,
    max_workers: int = 1,
    generation_cache: Any = None,
    generation_model_name: str = "",
) -> List[Dict[str, Any]]:
    """只要索引列表、不关心 debug 时的便捷入口，参数同 build_retrieval_index_payload。"""
    indexes, _ = build_retrieval_index_payload(
        chunks,
        generation_service=generation_service,
        enable_generative_indexes=enable_generative_indexes,
        max_indexes_per_chunk=max_indexes_per_chunk,
        max_questions_per_chunk=max_questions_per_chunk,
        max_workers=max_workers,
        generation_cache=generation_cache,
        generation_model_name=generation_model_name,
    )
    return indexes


def build_retrieval_index_payload(
    chunks: Iterable[Dict[str, Any]],
    *,
    generation_service: Any = None,
    enable_generative_indexes: bool = False,
    max_indexes_per_chunk: int = DEFAULT_RETRIEVAL_INDEX_MAX_PER_CHUNK,
    max_questions_per_chunk: int = DEFAULT_RETRIEVAL_INDEX_MAX_QUESTIONS_PER_CHUNK,
    max_workers: int = 1,
    generation_cache: Any = None,
    generation_model_name: str = "",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """用给定参数构造 RetrievalIndexBuilder 建索引，返回 (索引字典列表, 汇总 debug)。"""
    builder = RetrievalIndexBuilder(
        generation_service=generation_service,
        enable_generative_indexes=enable_generative_indexes,
        max_indexes_per_chunk=max_indexes_per_chunk,
        max_questions_per_chunk=max_questions_per_chunk,
        max_workers=max_workers,
        generation_cache=generation_cache,
        generation_model_name=generation_model_name,
    )
    return builder.build(chunks)


def iter_retrieval_indexes_for_embedding(retrieval_indexes: Iterable[Dict[str, Any]]) -> List[RetrievalIndex]:
    """筛出需要写入向量库的索引：文本非空且参与至少一条向量路线。"""
    indexes: List[RetrievalIndex] = []
    for payload in retrieval_indexes or []:
        index = RetrievalIndex.from_dict(dict(payload or {}))
        # 空 index_text 只保留在调试/落盘模型里，不能进入 embedding 或 BM25 的真实候选池。
        if not index.index_text.strip():
            continue
        if not retrieval_index_enabled_for(index, VECTOR_ROUTE_NAMES):
            continue
        indexes.append(index)
    return indexes


def summarize_retrieval_indexes(retrieval_indexes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """统计索引总数、各类型数量和空文本索引数量。"""
    counts: Dict[str, int] = defaultdict(int)
    empty_count = 0
    for payload in retrieval_indexes or []:
        index = RetrievalIndex.from_dict(dict(payload or {}))
        counts[index.index_type] += 1
        if not index.index_text.strip():
            empty_count += 1
    return {
        "index_count": sum(counts.values()),
        "index_type_counts": dict(counts),
        "empty_index_text_count": empty_count,
    }


def _safe_artifact_slug(value: Any, fallback: str = "paper") -> str:
    """把任意值转成可用作文件名/目录名的片段，只保留字母数字和 ._-；结果为空时用 fallback。"""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip()).strip("._-")
    return slug or fallback


def _default_artifact_version(created_at: str) -> str:
    """未指定版本号时由创建时间生成，例如 2024-01-02T03:04:05 → 20240102T030405。"""
    return created_at.replace(":", "").replace("-", "")


def _write_json(path: Any, payload: Any) -> None:
    """以 UTF-8、缩进 2 写出 JSON；无法序列化的值（如 datetime）转成字符串。"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _read_json(path: Any) -> Any:
    """以 UTF-8 读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_nonblank(tokens: Iterable[Any]) -> int:
    """统计非空白 token 的数量。"""
    return sum(1 for token in tokens if str(token or "").strip())


def normalize_sparse_source_type(source_type: Any) -> str:
    """规范化 sparse artifact 的来源类型；为空时视为 chunk，并把历史别名映射到标准名。"""
    normalized = str(source_type or "").strip().lower()
    if not normalized:
        return "chunk"
    return SPARSE_SOURCE_TYPE_ALIASES.get(normalized, normalized)


def normalize_retrieval_index_artifact_records(
    retrieval_indexes: Iterable[Dict[str, Any]],
    *,
    paper_id: str,
    index_version: str,
    created_at: str,
) -> List[Dict[str, Any]]:
    """把索引转换成 artifact 中的记录格式，补上 paper_id、版本、创建时间和来源文档名。"""
    records: List[Dict[str, Any]] = []
    for payload in retrieval_indexes or []:
        index = RetrievalIndex.from_dict(dict(payload or {}))
        metadata = dict(index.metadata or {})
        source = str(
            (payload or {}).get("source")
            or metadata.get("source")
            or metadata.get("document_name")
            or ""
        ).strip()
        records.append(
            {
                "index_id": index.index_id,
                "paper_id": paper_id,
                "chunk_id": index.chunk_id,
                "index_type": index.index_type,
                "index_text": index.index_text,
                "index_weight": float(index.index_weight),
                "enabled_routes": list(index.enabled_routes),
                "source": source,
                "metadata": metadata,
                "index_version": index_version,
                "created_at": created_at,
            }
        )
    return records


def save_retrieval_index_artifact(
    *,
    paper_id: str,
    retrieval_indexes: Iterable[Dict[str, Any]],
    retrieval_index_debug: Optional[Dict[str, Any]] = None,
    chunks: Optional[Iterable[Dict[str, Any]]] = None,
    index_version: Optional[str] = None,
    output_dir: str = DEFAULT_RETRIEVAL_INDEX_ARTIFACT_DIR,
) -> str:
    """把 retrieval index 写成 JSON artifact，返回文件路径。

    传入 chunks 时会记录 chunk_ids，并列出引用了不存在 chunk 的索引（missing_chunk_ids）。
    """
    created_at = datetime.now().isoformat(timespec="seconds")
    version = str(index_version or _default_artifact_version(created_at)).strip()
    records = normalize_retrieval_index_artifact_records(
        retrieval_indexes,
        paper_id=str(paper_id or ""),
        index_version=version,
        created_at=created_at,
    )
    chunk_ids = {
        normalize_chunk_id(dict(chunk or {}), fallback_index=index)
        for index, chunk in enumerate(chunks or [], start=1)
    }
    missing_chunk_ids = sorted(
        {
            str(record.get("chunk_id") or "")
            for record in records
            if chunk_ids and str(record.get("chunk_id") or "") not in chunk_ids
        }
    )
    summary = dict(retrieval_index_debug or summarize_retrieval_indexes(records))
    payload = {
        "schema_version": RETRIEVAL_INDEX_ARTIFACT_SCHEMA_VERSION,
        "paper_id": str(paper_id or ""),
        "index_version": version,
        "created_at": created_at,
        "index_count": len(records),
        "index_type_counts": summary.get("index_type_counts", {}),
        "empty_index_text_count": summary.get("empty_index_text_count", 0),
        "chunk_count": len(chunk_ids),
        "chunk_ids": sorted(chunk_ids),
        "missing_chunk_ids": missing_chunk_ids,
        "debug": summary,
        # 完整 index_text 在这里持久化；Milvus metadata 只作为向量检索副本，不再是唯一来源。
        "retrieval_indexes": records,
    }

    os.makedirs(output_dir, exist_ok=True)
    filename = (
        f"{_safe_artifact_slug(paper_id)}_"
        f"{_safe_artifact_slug(version, fallback='version')}_retrieval_indexes.json"
    )
    filepath = os.path.join(output_dir, filename)
    _write_json(filepath, payload)
    return filepath


def _artifact_path_candidates(raw_text: str) -> Iterator[Path]:
    """依次给出路径的候选绝对位置：原样、相对当前工作目录、相对 backend 目录、相对仓库根目录。

    数据库里记录的 artifact 路径可能是相对路径，而服务可能从仓库根目录或 backend 目录启动，所以都要试。
    无法解析的候选会被跳过。
    """
    raw_path = Path(raw_text)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        backend_root = Path(__file__).resolve().parents[2]
        candidates.extend([Path(os.getcwd()) / raw_path, backend_root / raw_path, backend_root.parent / raw_path])
    for candidate in candidates:
        try:
            yield candidate.resolve()
        except OSError:
            continue


def resolve_artifact_file_path(file_path: Any) -> Optional[Path]:
    """在候选位置中找到第一个存在的文件；找不到时返回 None。"""
    raw_text = str(file_path or "").strip()
    if not raw_text:
        return None
    return next((path for path in _artifact_path_candidates(raw_text) if path.is_file()), None)


def load_retrieval_index_artifact(file_path: Any) -> Dict[str, Any]:
    """读取 retrieval index artifact；文件不存在时抛 FileNotFoundError，内容不是对象时抛 ValueError。"""
    resolved = resolve_artifact_file_path(file_path)
    if resolved is None:
        raise FileNotFoundError(str(file_path or ""))
    payload = _read_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid retrieval index artifact: {resolved}")
    return payload


def extract_question_index_texts(chunk: Dict[str, Any]) -> List[str]:
    """读取 chunk 自带的问题（retrieval_questions / questions，顶层优先）；字符串按换行或分号拆分。"""
    metadata = dict(chunk.get("metadata", {}) or {})
    raw_value = (
        chunk.get("retrieval_questions")
        or chunk.get("questions")
        or metadata.get("retrieval_questions")
        or metadata.get("questions")
        or []
    )
    if isinstance(raw_value, str):
        candidates = re.split(r"[\n;]+", raw_value)
    elif isinstance(raw_value, (list, tuple, set)):
        candidates = list(raw_value)
    else:
        candidates = []
    return [str(item).strip() for item in candidates if str(item).strip()]


@dataclass
class KeywordDocument:
    """BM25 文档项，保留 chunk payload 和预分词统计，查询时只遍历命中 postings 的候选。"""

    doc_id: int
    chunk: Dict[str, Any]
    token_counts: Counter
    doc_length: int
    content_prefix: str
    field_token_counts: Dict[str, Counter] = field(default_factory=dict)
    field_lengths: Dict[str, int] = field(default_factory=dict)
    token_sources: Dict[str, List[str]] = field(default_factory=dict)
    keyword_document_debug: Dict[str, Any] = field(default_factory=dict)
    retrieval_index_id: str = ""
    retrieval_index_type: str = "body"
    retrieval_index_text: str = ""
    retrieval_index_weight: float = 1.0
    retrieval_index_enabled_routes: List[str] = field(default_factory=list)


@dataclass
class CollectionRetrievalIndex:
    """collection 级检索索引，同时服务 keyword route 和 memory route 精确取回。"""

    arxiv_id: str
    collection_name: str
    index_version: str
    chunk_count: int
    build_id: str
    chunk_file_hash: str
    created_at: str
    documents: List[KeywordDocument] = field(default_factory=list)
    document_frequency: Dict[str, int] = field(default_factory=dict)
    postings: Dict[str, Set[int]] = field(default_factory=dict)
    avgdl: float = 0.0
    by_chunk_id: Dict[str, List[int]] = field(default_factory=dict)
    by_parent_chunk_id: Dict[str, List[int]] = field(default_factory=dict)
    by_original_chunk_id: Dict[str, List[int]] = field(default_factory=dict)
    by_page_section: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)
    by_page_number: Dict[str, List[int]] = field(default_factory=dict)
    by_section_path: Dict[str, List[int]] = field(default_factory=dict)
    by_section_title: Dict[str, List[int]] = field(default_factory=dict)
    by_chunk_type: Dict[str, List[int]] = field(default_factory=dict)
    by_subchunk_index: Dict[str, List[int]] = field(default_factory=dict)
    by_parent_subchunk: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)
    structured_tables: List[Dict[str, Any]] = field(default_factory=list)
    table_structure_debug: Dict[str, Any] = field(default_factory=dict)
    retrieval_index_artifact_debug: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    build_time: float = 0.0
    fallback_reason: str = ""
    build_source: str = "lazy_chunk_scan"

    def keyword_route_index_source(self) -> str:
        """关键词路线索引来源：persistent_sparse_artifact 或 runtime_build_fallback。"""
        # debug 对外只暴露迁移状态，不把内部 runtime build 的多种来源泄漏成新的前端契约。
        if self.build_source == "sparse_index_artifact":
            return "persistent_sparse_artifact"
        return "runtime_build_fallback"

    def sparse_index_debug_summary(self, candidate_count: int) -> Dict[str, Any]:
        """汇总 sparse 索引的加载情况：来源、版本、文档数、加载耗时，以及退回现场构建的原因。"""
        route_index_source = self.keyword_route_index_source()
        artifact_debug = dict(self.retrieval_index_artifact_debug or {})
        sparse_debug = artifact_debug if self.build_source == "sparse_index_artifact" else dict(artifact_debug.get("sparse_index_artifact") or {})
        stale_reason = str(sparse_debug.get("reason") or "")
        raw_source_type = str(sparse_debug.get("source_type") or artifact_debug.get("source_type") or "").strip()
        if not raw_source_type:
            source_hint = str(artifact_debug.get("source") or self.build_source or "").strip()
            # runtime fallback 仍要暴露稀疏索引源类型，方便后续 chunk-level 到 retrieval-index-level 的迁移评估。
            raw_source_type = "retrieval_index" if "retrieval_index" in source_hint else ("chunk" if source_hint else "")
        source_type = normalize_sparse_source_type(raw_source_type) if raw_source_type else ""
        return {
            "load_source": route_index_source,
            "build_id": str(sparse_debug.get("build_id") or self.build_id or ""),
            "index_version": str(sparse_debug.get("index_version") or self.index_version or ""),
            "source_type": source_type if source_type in SPARSE_SOURCE_TYPES else "",
            "backend": str(sparse_debug.get("backend") or ""),
            "schema_version": str(sparse_debug.get("schema_version") or ""),
            "manifest_file": str(sparse_debug.get("sparse_index_manifest_file") or ""),
            "document_count": len(self.documents),
            "keyword_route_hit_count": int(candidate_count or 0),
            "load_time_ms": round(float(self.build_time or 0.0) * 1000.0, 3) if route_index_source == "persistent_sparse_artifact" else 0.0,
            "fallback_count": 1 if route_index_source == "runtime_build_fallback" and stale_reason else 0,
            "artifact_stale_reason": stale_reason,
        }

    def to_keyword_debug(self, candidate_count: int, *, full_scan_used: bool = False) -> Dict[str, Any]:
        """关键词路线 debug 中与索引相关的字段。"""
        route_index_source = self.keyword_route_index_source()
        sparse_summary = self.sparse_index_debug_summary(candidate_count)
        return {
            "keyword_index_hit": self.cache_hit,
            "keyword_index_version": self.index_version,
            "keyword_candidate_count": candidate_count,
            "keyword_index_chunk_count": self.chunk_count,
            "keyword_index_document_count": len(self.documents),
            "keyword_index_build_id": self.build_id,
            "keyword_index_created_at": self.created_at,
            "keyword_index_build_time": self.build_time,
            "keyword_full_scan_used": full_scan_used,
            "keyword_index_build_source": self.build_source,
            "keyword_route_index_source": route_index_source,
            "keyword_index_fallback_used": route_index_source == "runtime_build_fallback",
            "keyword_index_fallback_reason": self.fallback_reason,
            "keyword_index_model": "retrieval_index_v1",
            "sparse_index": sparse_summary,
            "retrieval_index_artifact": dict(self.retrieval_index_artifact_debug),
        }

    def to_memory_debug(
        self,
        *,
        lookup_mode: str,
        exact_hit_count: int,
        fallback_used: bool,
        candidate_count: int,
        full_scan_used: bool = False,
    ) -> Dict[str, Any]:
        """记忆路线 debug 中与索引相关的字段。"""
        return {
            "memory_lookup_mode": lookup_mode,
            "memory_exact_hit_count": exact_hit_count,
            "memory_fallback_used": fallback_used,
            "memory_candidate_count": candidate_count,
            "memory_index_hit": self.cache_hit,
            "memory_index_version": self.index_version,
            "memory_index_chunk_count": self.chunk_count,
            "memory_full_scan_used": full_scan_used,
            "memory_index_fallback_reason": self.fallback_reason,
            "retrieval_index_artifact": dict(self.retrieval_index_artifact_debug),
        }


def compute_artifact_file_hash(file_path: Any) -> str:
    """计算 artifact 源文件 hash，用于判断 sparse index 是否仍对应当前 active build。"""
    resolved = resolve_artifact_file_path(file_path)
    if resolved is None:
        raise FileNotFoundError(str(file_path or ""))
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counter_to_json(counter: Counter) -> Dict[str, float]:
    """Counter → JSON 对象；计数可能是加权后的小数，统一存成 float。"""
    return {str(token): float(count) for token, count in dict(counter or {}).items()}


def _counter_from_json(payload: Any) -> Counter:
    """JSON 对象 → Counter；无法转成数字的项直接跳过。"""
    if not isinstance(payload, dict):
        return Counter()
    values = Counter()
    for token, count in payload.items():
        try:
            values[str(token)] = float(count)
        except (TypeError, ValueError):
            continue
    return values


def _keyword_document_to_sparse_record(document: KeywordDocument) -> Dict[str, Any]:
    """KeywordDocument → documents.jsonl 中的一行记录。"""
    return {
        "doc_id": int(document.doc_id),
        "chunk": dict(document.chunk or {}),
        "token_counts": _counter_to_json(document.token_counts),
        "doc_length": int(document.doc_length or 0),
        "content_prefix": str(document.content_prefix or ""),
        "field_token_counts": {
            str(field_name): _counter_to_json(field_counts)
            for field_name, field_counts in (document.field_token_counts or {}).items()
        },
        "field_lengths": {str(field_name): int(length or 0) for field_name, length in (document.field_lengths or {}).items()},
        "token_sources": {
            str(token): [str(source) for source in (sources or [])]
            for token, sources in (document.token_sources or {}).items()
        },
        "keyword_document_debug": dict(document.keyword_document_debug or {}),
        "retrieval_index_id": str(document.retrieval_index_id or ""),
        "retrieval_index_type": str(document.retrieval_index_type or "body"),
        "retrieval_index_text": str(document.retrieval_index_text or ""),
        "retrieval_index_weight": float(document.retrieval_index_weight or 1.0),
        "retrieval_index_enabled_routes": list(document.retrieval_index_enabled_routes or []),
    }


def _keyword_document_from_sparse_record(record: Dict[str, Any], fallback_doc_id: int) -> KeywordDocument:
    """documents.jsonl 中的一行记录 → KeywordDocument；缺失字段按构建时的默认值补齐。"""
    field_token_counts = {
        str(field_name): _counter_from_json(field_counts)
        for field_name, field_counts in (record.get("field_token_counts") or {}).items()
        if isinstance(field_counts, dict)
    }
    return KeywordDocument(
        doc_id=int(record.get("doc_id", fallback_doc_id) or fallback_doc_id),
        chunk=dict(record.get("chunk") or {}),
        token_counts=_counter_from_json(record.get("token_counts") or {}),
        doc_length=int(record.get("doc_length", 0) or 0),
        content_prefix=str(record.get("content_prefix") or ""),
        field_token_counts=field_token_counts,
        field_lengths={str(key): int(value or 0) for key, value in (record.get("field_lengths") or {}).items()},
        token_sources={
            str(token): [str(source) for source in (sources or [])]
            for token, sources in (record.get("token_sources") or {}).items()
        },
        keyword_document_debug=dict(record.get("keyword_document_debug") or {}),
        retrieval_index_id=str(record.get("retrieval_index_id") or ""),
        retrieval_index_type=str(record.get("retrieval_index_type") or "body"),
        retrieval_index_text=str(record.get("retrieval_index_text") or ""),
        retrieval_index_weight=float(record.get("retrieval_index_weight", 1.0) or 1.0),
        retrieval_index_enabled_routes=normalize_enabled_routes(record.get("retrieval_index_enabled_routes")),
    )


def _sparse_token_stats(index: CollectionRetrievalIndex) -> Dict[str, Any]:
    """汇总 sparse artifact 的 token 统计，供 manifest/debug 直接解释 document_count 和 avgdl 来源。"""
    field_stats: Dict[str, Dict[str, Any]] = {}
    total_document_length = 0
    for document in index.documents:
        total_document_length += int(document.doc_length or 0)
        for field_name, length in (document.field_lengths or {}).items():
            stats = field_stats.setdefault(
                str(field_name),
                {"document_count": 0, "total_token_count": 0, "unique_token_count": 0},
            )
            stats["document_count"] += 1
            stats["total_token_count"] += int(length or 0)
            field_counts = (document.field_token_counts or {}).get(field_name) or Counter()
            stats["unique_token_count"] += _count_nonblank(field_counts)
    return {
        "document_count": len(index.documents),
        "avgdl": float(index.avgdl or 0.0),
        "total_document_length": total_document_length,
        "unique_token_count": _count_nonblank(index.document_frequency or {}),
        "posting_token_count": _count_nonblank(index.postings or {}),
        "field_stats": field_stats,
        "field_weights": dict(KEYWORD_FIELD_WEIGHTS),
        "field_weights_version": KEYWORD_FIELD_WEIGHTS_VERSION,
        "bm25_schema_version": BM25_SCHEMA_VERSION,
    }


def save_sparse_index_artifact(
    *,
    index: CollectionRetrievalIndex,
    paper_id: str,
    build_id: str,
    index_version: str,
    source_type: str,
    source_file: str,
    source_hash: Optional[str] = None,
    backend: str = "",
    output_dir: str = DEFAULT_SPARSE_INDEX_ARTIFACT_DIR,
) -> Dict[str, Any]:
    """把 BM25/keyword route 需要的稀疏结构落成正式 artifact，避免重启后再临时扫描重建。"""
    created_at = datetime.now().isoformat(timespec="seconds")
    version = str(index_version or index.index_version or _default_artifact_version(created_at)).strip()
    source_hash_value = str(source_hash or compute_artifact_file_hash(source_file)).strip()
    normalized_source_type = normalize_sparse_source_type(source_type)
    if normalized_source_type not in SPARSE_SOURCE_TYPES:
        raise ValueError(f"Unsupported sparse source_type: {source_type}")
    artifact_dir = Path(output_dir) / _safe_artifact_slug(paper_id) / _safe_artifact_slug(version, fallback="version")
    os.makedirs(artifact_dir, exist_ok=True)

    documents_file = artifact_dir / "documents.jsonl"
    postings_file = artifact_dir / "postings.json"
    document_frequency_file = artifact_dir / "document_frequency.json"
    token_stats_file = artifact_dir / "token_stats.json"
    manifest_file = artifact_dir / "manifest.json"

    with open(documents_file, "w", encoding="utf-8") as handle:
        for document in index.documents:
            handle.write(json.dumps(_keyword_document_to_sparse_record(document), ensure_ascii=False, default=str))
            handle.write("\n")

    # postings 的 doc_id 集合排序后落盘，保证同一索引多次导出的文件内容一致。
    _write_json(
        postings_file,
        {
            str(token): sorted(int(doc_id) for doc_id in doc_ids)
            for token, doc_ids in (index.postings or {}).items()
            if str(token or "").strip()
        },
    )
    _write_json(
        document_frequency_file,
        {
            str(token): int(freq or 0)
            for token, freq in (index.document_frequency or {}).items()
            if str(token or "").strip()
        },
    )

    token_stats = _sparse_token_stats(index)
    _write_json(token_stats_file, token_stats)
    token_count = int(token_stats.get("total_document_length", 0) or 0)

    manifest = {
        "schema_version": SPARSE_INDEX_ARTIFACT_SCHEMA_VERSION,
        "paper_id": str(paper_id or ""),
        "build_id": str(build_id or index.build_id or ""),
        "index_version": version,
        "source_type": normalized_source_type,
        "source_file": str(source_file or ""),
        "source_hash": source_hash_value,
        "backend": str(backend or "internal_bm25"),
        "tokenizer_version": KEYWORD_TOKENIZER_VERSION,
        "field_weights_version": KEYWORD_FIELD_WEIGHTS_VERSION,
        "bm25_schema_version": BM25_SCHEMA_VERSION,
        "document_count": len(index.documents),
        "token_count": token_count,
        "avgdl": float(index.avgdl or 0.0),
        "created_at": created_at,
        "files": {
            "documents": documents_file.name,
            "postings": postings_file.name,
            "document_frequency": document_frequency_file.name,
            "token_stats": token_stats_file.name,
        },
    }
    # manifest 最后写入；读取方通过它定位其余文件，并用其中的 document_count 校验完整性。
    _write_json(manifest_file, manifest)

    return {
        "artifact_dir": str(artifact_dir),
        "manifest_file": str(manifest_file),
        "schema_version": SPARSE_INDEX_ARTIFACT_SCHEMA_VERSION,
        "document_count": len(index.documents),
        "token_count": token_count,
        "backend": manifest["backend"],
        "source_file": manifest["source_file"],
        "source_hash": source_hash_value,
        "avgdl": float(index.avgdl or 0.0),
        "manifest": manifest,
    }


def _resolve_sparse_manifest_path(file_or_dir: Any) -> Optional[Path]:
    """定位 sparse manifest：参数可以是 manifest 文件本身，也可以是包含 manifest.json 的目录。"""
    raw_text = str(file_or_dir or "").strip()
    if not raw_text:
        return None
    for resolved in _artifact_path_candidates(raw_text):
        if resolved.is_dir():
            manifest = resolved / "manifest.json"
            if manifest.is_file():
                return manifest
        if resolved.is_file():
            return resolved
    return None


def load_sparse_index_artifact_files(file_or_dir: Any) -> Dict[str, Any]:
    """读取 sparse artifact 的全部结构文件；调用方负责按 active build 做一致性判断。"""
    manifest_path = _resolve_sparse_manifest_path(file_or_dir)
    if manifest_path is None:
        raise FileNotFoundError(str(file_or_dir or ""))
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid sparse index manifest: {manifest_path}")
    if manifest.get("schema_version") != SPARSE_INDEX_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported sparse index schema: {manifest.get('schema_version')}")

    artifact_dir = manifest_path.parent
    files = dict(manifest.get("files") or {})
    documents_path = artifact_dir / str(files.get("documents") or "documents.jsonl")
    postings_path = artifact_dir / str(files.get("postings") or "postings.json")
    document_frequency_path = artifact_dir / str(files.get("document_frequency") or "document_frequency.json")
    token_stats_path = artifact_dir / str(files.get("token_stats") or "token_stats.json")

    documents: List[KeywordDocument] = []
    with open(documents_path, "r", encoding="utf-8") as handle:
        for fallback_doc_id, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                documents.append(_keyword_document_from_sparse_record(record, fallback_doc_id))

    raw_postings = _read_json(postings_path)
    raw_document_frequency = _read_json(document_frequency_path)
    token_stats = _read_json(token_stats_path)

    postings = {
        str(token): {int(doc_id) for doc_id in (doc_ids or [])}
        for token, doc_ids in (raw_postings or {}).items()
        if isinstance(doc_ids, list)
    }
    document_frequency = {
        str(token): int(freq or 0)
        for token, freq in (raw_document_frequency or {}).items()
    }
    return {
        "manifest": manifest,
        "manifest_file": str(manifest_path),
        "artifact_dir": str(artifact_dir),
        "documents": documents,
        "postings": postings,
        "document_frequency": document_frequency,
        "token_stats": dict(token_stats or {}),
    }


def _record_index_version(record: Optional[Dict[str, Any]]) -> str:
    """论文索引记录中的当前索引版本：active_index_version 优先，其次 index_version，都没有时为空串。"""
    record = record or {}
    return str(record.get("active_index_version") or record.get("index_version") or "")


def _record_build_id(record: Optional[Dict[str, Any]]) -> str:
    """论文索引记录中的当前构建 ID：active_build_id 优先，其次 build_id，都没有时为空串。"""
    record = record or {}
    return str(record.get("active_build_id") or record.get("build_id") or "")


def _new_lookup_tables() -> Dict[str, Dict[Any, List[int]]]:
    """为 _LOOKUP_TABLE_NAMES 中的每张表创建一个空的 defaultdict(list)。"""
    return {name: defaultdict(list) for name in _LOOKUP_TABLE_NAMES}


def _copy_lookup_tables(tables: Dict[str, Dict[Any, List[int]]]) -> Dict[str, Dict[Any, List[int]]]:
    """复制 lookup 表：转成普通 dict（不再自动建键），每个 doc_id 列表也复制一份。"""
    return {name: {key: list(value) for key, value in tables[name].items()} for name in _LOOKUP_TABLE_NAMES}


class CollectionRetrievalIndexProvider:
    """缓存 collection 的 keyword postings 和 chunk lookup，避免 route 查询时全量扫 chunk。

    缓存按最近最少使用淘汰，最多保留 max_entries 篇论文；被淘汰的论文下次查询时重建。
    """

    def __init__(
        self,
        *,
        vector_store_service: Any,
        chunk_normalizer: Callable[[Dict[str, Any]], Dict[str, Any]],
        tokenizer: Callable[[str], List[str]],
        max_entries: Optional[int] = None,
    ) -> None:
        self.vector_store_service = vector_store_service
        self.chunk_normalizer = chunk_normalizer
        self.tokenizer = tokenizer
        if max_entries is None:
            max_entries = int(ENHANCED_RETRIEVAL_CONFIG.get("collection_index_cache_max_entries", 16))
        self._max_entries = max(1, int(max_entries))
        self._cache: "OrderedDict[str, CollectionRetrievalIndex]" = OrderedDict()
        self._lock = threading.RLock()

    def get_index(
        self,
        collection_name: str,
        *,
        collection_profile: Optional[CollectionRetrievalProfile] = None,
        index_record: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> CollectionRetrievalIndex:
        """取 collection 检索索引：缓存有效时返回副本，否则重建并写回缓存。

        index_record 是数据库中的论文索引记录，用于定位 artifact 并判断缓存是否过期
        （chunk 数、索引版本、构建 ID、sparse 源 hash 等任一变化都会触发重建）。
        """
        resolved_name = self._resolve_collection_name(collection_name)
        with self._lock:
            cached = self._cache.get(resolved_name)
            stale_reason = self._stale_reason(cached, collection_profile, index_record)
            if cached and not force_refresh and not stale_reason:
                self._cache.move_to_end(resolved_name)
                index = self._clone_index(cached)
                index.cache_hit = True
                index.build_time = 0.0
                return index

            if cached and stale_reason:
                logger.info("Collection retrieval index stale: collection=%s reason=%s", resolved_name, stale_reason)

            index = self._build_index(
                resolved_name,
                collection_profile=collection_profile,
                index_record=index_record,
                stale_reason=stale_reason,
            )
            self._cache[resolved_name] = self._clone_index(index)
            self._cache.move_to_end(resolved_name)
            while len(self._cache) > self._max_entries:
                evicted_name, _ = self._cache.popitem(last=False)
                logger.debug("Collection retrieval index evicted: collection=%s", evicted_name)
            return index

    def _build_index(
        self,
        collection_name: str,
        *,
        collection_profile: Optional[CollectionRetrievalProfile],
        index_record: Optional[Dict[str, Any]],
        stale_reason: str,
    ) -> CollectionRetrievalIndex:
        """构建索引，按优先级尝试各数据源，每次降级都把原因记入 fallback_reason。

        1. 构建期落盘的 sparse artifact（直接恢复 postings，最快）；
        2. 开启 index 级 BM25 时：retrieval index artifact + chunk 文件，失败再全量读向量库；
        3. 关闭 index 级 BM25 时：chunk 文件或向量库中折叠回 chunk 粒度的旧数据。
        2、3 两种情况都在内存中现场分词并构建 postings 与 lookup 表。
        """
        started = perf_counter()
        fallback_reasons: List[str] = []
        if stale_reason:
            fallback_reasons.append(f"rebuilt_after_stale: {stale_reason}")

        sparse_index, sparse_debug = self._load_sparse_index_artifact(
            index_record,
            collection_name=collection_name,
            collection_profile=collection_profile,
        )
        if sparse_index is not None:
            structured_tables, table_structure_debug = self._load_structured_tables(index_record)
            sparse_index.structured_tables = [dict(item) for item in structured_tables]
            sparse_index.table_structure_debug = dict(table_structure_debug)
            sparse_index.build_time = perf_counter() - started
            sparse_index.fallback_reason = "; ".join(
                [reason for reason in fallback_reasons if reason]
            )
            return sparse_index
        if sparse_debug.get("enabled") and sparse_debug.get("reason"):
            fallback_reasons.append(f"sparse_index_artifact_unavailable:{sparse_debug.get('reason')}")

        use_index_level_bm25 = bool(ENHANCED_RETRIEVAL_CONFIG.get("enable_index_level_bm25", True))
        if use_index_level_bm25:
            raw_chunks, retrieval_index_artifact_debug = self._load_retrieval_index_chunks(index_record)
            build_source = "retrieval_index_artifact" if raw_chunks else "milvus_chunk_scan"
            if raw_chunks:
                fallback_reasons.append(f"retrieval_index_artifact_count:{len(raw_chunks)}")
            else:
                reason = str(retrieval_index_artifact_debug.get("reason") or "").strip()
                if retrieval_index_artifact_debug.get("enabled") and reason:
                    fallback_reasons.append(f"retrieval_index_artifact_unavailable:{reason}")
                try:
                    raw_chunks = self.vector_store_service.get_all_chunks(collection_name) or []
                except Exception as exc:
                    # route 查询不能继续回到无上限扫描；构建失败时返回空索引并把原因暴露给 debug。
                    raw_chunks = []
                    fallback_reasons.append(f"chunk_index_build_failed: {exc}")
                    logger.warning("Failed to build collection retrieval index: collection=%s error=%s", collection_name, exc)
        else:
            # 灰度开关关闭时强制回到 chunk-level BM25，避免未迁移 collection 因多 index 行产生重复候选。
            raw_chunks, retrieval_index_artifact_debug = self._load_legacy_chunk_rows(index_record)
            build_source = "legacy_chunk_file" if raw_chunks else "legacy_milvus_chunk_scan"
            fallback_reasons.append("index_level_bm25_disabled")
            if raw_chunks:
                fallback_reasons.append(f"legacy_chunk_file_count:{len(raw_chunks)}")
            else:
                try:
                    raw_chunks = self._collapse_to_legacy_chunk_rows(
                        self.vector_store_service.get_all_chunks(collection_name) or []
                    )
                    retrieval_index_artifact_debug["source"] = "legacy_milvus_chunk_scan"
                    retrieval_index_artifact_debug["row_count"] = len(raw_chunks)
                    if raw_chunks:
                        retrieval_index_artifact_debug["reason"] = "index_level_bm25_disabled"
                except Exception as exc:
                    # 兼容路径同样要显式失败原因，便于确认是灰度开关还是底层读库异常。
                    raw_chunks = []
                    fallback_reasons.append(f"legacy_chunk_index_build_failed: {exc}")
                    logger.warning("Failed to build legacy collection retrieval index: collection=%s error=%s", collection_name, exc)

        retrieval_index_artifact_debug["index_level_bm25_enabled"] = use_index_level_bm25
        retrieval_index_artifact_debug["keyword_route_index_source"] = "runtime_build_fallback"
        if sparse_debug.get("enabled"):
            # artifact 优先加载失败时，旧 runtime build 仍可继续，但必须把拒绝原因留给 trace 排查。
            retrieval_index_artifact_debug["sparse_index_artifact"] = dict(sparse_debug)

        structured_tables, table_structure_debug = self._load_structured_tables(index_record)
        if structured_tables:
            fallback_reasons.append(f"structured_table_count:{len(structured_tables)}")
        elif table_structure_debug.get("enabled", False) and table_structure_debug.get("reason"):
            fallback_reasons.append(f"structured_table_unavailable:{table_structure_debug.get('reason')}")

        normalized_chunks = [self.chunk_normalizer(chunk) for chunk in raw_chunks]
        documents: List[KeywordDocument] = []
        document_frequency: Dict[str, int] = defaultdict(int)
        postings: Dict[str, Set[int]] = defaultdict(set)
        lookup_tables = _new_lookup_tables()

        for raw_chunk in normalized_chunks:
            chunk = self._ensure_retrieval_index_fields(raw_chunk)
            if not self._retrieval_index_enabled_for_keyword(chunk):
                continue
            if not str(chunk.get("retrieval_index_text", "") or "").strip():
                continue

            doc_id = len(documents)
            chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
            keyword_document = self._build_keyword_document_fields(chunk, chunk_type)
            field_token_counts = keyword_document["field_token_counts"]
            if not field_token_counts:
                continue
            token_sources = keyword_document["token_sources"]
            field_lengths = keyword_document["field_lengths"]
            # token_counts 保留兼容字段，但计数已按字段权重合成，避免图表 OCR 与正文等权进入 BM25。
            token_counts = Counter()
            for field_name, field_counts in field_token_counts.items():
                weight = KEYWORD_FIELD_WEIGHTS.get(field_name, 1.0)
                for token, count in field_counts.items():
                    token_counts[token] += count * weight
            for token in token_counts:
                document_frequency[token] += 1
                postings[token].add(doc_id)

            doc_length = max(int(round(sum(token_counts.values()))), 1)
            documents.append(
                KeywordDocument(
                    doc_id=doc_id,
                    chunk=chunk,
                    token_counts=token_counts,
                    doc_length=doc_length,
                    content_prefix=str(chunk.get("retrieval_index_text") or chunk.get("content", "") or "").lower()[:300],
                    field_token_counts=field_token_counts,
                    field_lengths=field_lengths,
                    token_sources=token_sources,
                    keyword_document_debug=keyword_document["debug"],
                    retrieval_index_id=str(chunk.get("retrieval_index_id") or ""),
                    retrieval_index_type=str(chunk.get("retrieval_index_type") or "body"),
                    retrieval_index_text=str(chunk.get("retrieval_index_text") or ""),
                    retrieval_index_weight=float(chunk.get("retrieval_index_weight", 1.0) or 1.0),
                    retrieval_index_enabled_routes=normalize_enabled_routes(chunk.get("retrieval_index_enabled_routes")),
                )
            )
            self._add_lookup_entries(doc_id, chunk, lookup_tables)

        chunk_count = self._resolve_chunk_count(collection_profile, index_record, len(normalized_chunks))
        record = index_record or {}
        return CollectionRetrievalIndex(
            arxiv_id=str(record.get("arxiv_id", "") or ""),
            collection_name=collection_name,
            index_version=_record_index_version(record) or collection_name,
            chunk_count=chunk_count,
            build_id=_record_build_id(record),
            chunk_file_hash=str(record.get("chunk_file_hash") or record.get("chunk_file") or ""),
            created_at=datetime.now().isoformat(timespec="seconds"),
            documents=documents,
            document_frequency=dict(document_frequency),
            postings={token: set(doc_ids) for token, doc_ids in postings.items()},
            avgdl=sum(doc.doc_length for doc in documents) / max(len(documents), 1),
            **_copy_lookup_tables(lookup_tables),
            structured_tables=[dict(item) for item in structured_tables],
            table_structure_debug=dict(table_structure_debug),
            retrieval_index_artifact_debug=dict(retrieval_index_artifact_debug),
            cache_hit=False,
            build_time=perf_counter() - started,
            fallback_reason="; ".join(fallback_reasons),
            build_source=build_source,
        )

    def _stale_reason(
        self,
        index: Optional[CollectionRetrievalIndex],
        collection_profile: Optional[CollectionRetrievalProfile],
        index_record: Optional[Dict[str, Any]],
    ) -> str:
        """返回缓存过期原因；没有缓存或缓存仍有效时返回空串。"""
        if index is None:
            return ""
        expected_index_level = bool(ENHANCED_RETRIEVAL_CONFIG.get("enable_index_level_bm25", True))
        cached_index_level = index.retrieval_index_artifact_debug.get("index_level_bm25_enabled")
        if cached_index_level is not None and bool(cached_index_level) != expected_index_level:
            # 灰度开关切换会改变 BM25 文档粒度，缓存必须重建，否则 route 输出会混用新旧语义。
            return f"index_level_bm25_changed:{bool(cached_index_level)}->{expected_index_level}"
        expected_count = self._resolve_expected_count(collection_profile, index_record)
        if expected_count is not None and expected_count != index.chunk_count:
            return f"chunk_count_changed:{index.chunk_count}->{expected_count}"

        expected_version = _record_index_version(index_record).strip()
        if expected_version and index.index_version and expected_version != index.index_version:
            return f"index_version_changed:{index.index_version}->{expected_version}"

        expected_build_id = _record_build_id(index_record).strip()
        if expected_build_id and index.build_id and expected_build_id != index.build_id:
            return f"build_id_changed:{index.build_id}->{expected_build_id}"
        expected_sparse_hash = str((index_record or {}).get("sparse_index_source_hash") or "").strip()
        cached_sparse_hash = str((index.retrieval_index_artifact_debug or {}).get("source_hash") or "").strip()
        if expected_sparse_hash and cached_sparse_hash and expected_sparse_hash != cached_sparse_hash:
            # sparse artifact 的源 hash 是 active build 一致性边界；hash 变化时必须重读磁盘 artifact。
            return f"sparse_source_hash_changed:{cached_sparse_hash}->{expected_sparse_hash}"
        expected_sparse_manifest = str((index_record or {}).get("sparse_index_manifest_file") or "").strip()
        cached_sparse_manifest = str((index.retrieval_index_artifact_debug or {}).get("sparse_index_manifest_file") or "").strip()
        if expected_sparse_manifest and cached_sparse_manifest and expected_sparse_manifest != cached_sparse_manifest:
            return "sparse_manifest_changed"
        if index.build_source == "sparse_index_artifact":
            sparse_debug = index.retrieval_index_artifact_debug or {}
            for field_name, expected_value in _SPARSE_SEMANTIC_VERSIONS:
                cached_value = str(sparse_debug.get(field_name) or "").strip()
                if cached_value and cached_value != expected_value:
                    # sparse artifact 的语义版本变化会改变 postings 解释方式，缓存必须失效后重新校验 manifest。
                    return f"sparse_{field_name}_changed:{cached_value}->{expected_value}"
        return ""

    @staticmethod
    def _validate_sparse_manifest_versions(manifest: Dict[str, Any]) -> str:
        """逐项比较 manifest 与当前代码的语义版本，返回第一处不一致的原因；全部一致时返回空串。"""
        # 与 _stale_reason 不同，这里 manifest 缺版本字段也视为不一致，因为无法确认 postings 的含义。
        for field_name, expected_value in _SPARSE_SEMANTIC_VERSIONS:
            actual_value = str(manifest.get(field_name) or "").strip()
            if actual_value != expected_value:
                return f"{field_name}_mismatch:{actual_value or 'missing'}->{expected_value}"
        return ""

    def _resolve_collection_name(self, collection_name: str) -> str:
        """向量库支持别名解析时用它规范化名称，保证缓存键与实际 collection 一致。"""
        resolver = getattr(self.vector_store_service, "resolve_collection_name", None)
        if callable(resolver):
            return resolver(collection_name)
        return collection_name

    def _ensure_retrieval_index_fields(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """兼容新旧 collection：新数据使用 retrieval index，旧数据自动合成 body index。"""
        normalized = dict(chunk)
        metadata = dict(normalized.get("metadata", {}) or {})
        chunk_id = normalize_chunk_id(normalized)

        def index_field(name: str) -> Any:
            # 先查顶层的 retrieval_index_<name> / index_<name>，再查 metadata 中的同名字段。
            return (
                normalized.get(f"retrieval_index_{name}")
                or normalized.get(f"index_{name}")
                or metadata.get(f"retrieval_index_{name}")
                or metadata.get(f"index_{name}")
            )

        index_id = str(index_field("id") or "").strip()
        index_type = normalize_index_type(index_field("type"))
        index_text = str(index_field("text") or "").strip()
        has_explicit_index = bool(index_id)

        if not index_id:
            # 旧 collection 没有 index 层时只合成 body index，保证现有问答链路仍可读取旧 chunk。
            index_id = f"{chunk_id}:body:1"
            index_type = "body"
            index_text = str(normalized.get("content") or normalized.get("text") or metadata.get("content") or metadata.get("text") or "").strip()

        normalized["retrieval_index_id"] = index_id
        normalized["retrieval_index_type"] = index_type
        normalized["retrieval_index_text"] = index_text
        normalized["retrieval_index_weight"] = float(index_field("weight") or RETRIEVAL_INDEX_WEIGHTS.get(index_type, 1.0))
        normalized["retrieval_index_enabled_routes"] = normalize_enabled_routes(
            normalized.get("retrieval_index_enabled_routes")
            or metadata.get("retrieval_index_enabled_routes")
            or RETRIEVAL_INDEX_ROUTE_DEFAULTS.get(index_type, [])
        )
        # index_* 是新向量 schema 的短字段名；BM25 内部继续同步旧字段，兼容历史调用方。
        normalized["index_id"] = normalized["retrieval_index_id"]
        normalized["index_type"] = normalized["retrieval_index_type"]
        normalized["index_text"] = normalized["retrieval_index_text"]
        normalized["index_weight"] = normalized["retrieval_index_weight"]
        normalized["retrieval_index_synthetic"] = not has_explicit_index
        return normalized

    @staticmethod
    def _retrieval_index_enabled_for_keyword(chunk: Dict[str, Any]) -> bool:
        """该索引行是否参与关键词路线（只看 retrieval_index_enabled_routes）。"""
        return _routes_allow(chunk.get("retrieval_index_enabled_routes"), KEYWORD_ROUTE_NAMES)

    def _build_keyword_document_fields(self, chunk: Dict[str, Any], chunk_type: str) -> Dict[str, Any]:
        """构建字段化 keyword document，避免正文、结构字段和图表 OCR 噪声无差别混入索引。"""
        index_type = normalize_index_type(chunk.get("retrieval_index_type"))
        index_text = str(chunk.get("retrieval_index_text") or "").strip()
        # 真正的 retrieval-index 文档只索引 index_text；合成 body 文档仍按 chunk 字段建索引，保留章节和图表字段权重。
        if index_text and not bool(chunk.get("retrieval_index_synthetic", False)):
            return self._build_keyword_document_fields_from_retrieval_index(chunk, chunk_type, index_type, index_text)

        field_texts: Dict[str, List[str]] = {
            "body": [str(chunk.get("content", "") or "")],
            "title": [],
            "section_title": [],
            "section_path": [],
            "asset_caption": [],
            "asset_aux": [],
        }
        skipped_fields: Dict[str, str] = {}

        body_text = self._normalize_keyword_field_text(field_texts["body"][0])
        title = str(chunk.get("title", "") or "")
        if self._keyword_field_looks_noisy(title, field_name="title"):
            skipped_fields["title"] = "noisy_or_too_short"
        else:
            field_texts["title"].append(title)

        allow_section_terms = self._asset_section_anchor_allowed_for_search_text(chunk, chunk_type)
        if allow_section_terms:
            for field_name in ("section_title", "section_path"):
                value = str(chunk.get(field_name, "") or "")
                if self._keyword_field_looks_noisy(value, field_name=field_name):
                    skipped_fields[field_name] = "noisy_or_too_short"
                else:
                    field_texts[field_name].append(value)
        else:
            # 图表 chunk 的弱章节锚点来自启发式匹配时不进入 keyword 索引，避免错误 section header 放大召回噪声。
            skipped_fields["section_title"] = "asset_section_anchor_not_allowed"
            skipped_fields["section_path"] = "asset_section_anchor_not_allowed"

        for field_name, target_field in (("asset_caption", "asset_caption"), ("asset_summary", "asset_aux"), ("asset_preview_text", "asset_aux")):
            value = str(chunk.get(field_name, "") or "")
            if not value:
                continue
            normalized_value = self._normalize_keyword_field_text(value)
            if normalized_value and normalized_value in body_text:
                skipped_fields[field_name] = "duplicated_in_body"
                continue
            if self._keyword_field_looks_noisy(value, field_name=field_name):
                skipped_fields[field_name] = "noisy_or_low_information"
                continue
            field_texts[target_field].append(value)

        document = self._tokenize_fields(field_texts)
        document["debug"].update(
            {
                "skipped_fields": skipped_fields,
                "has_asset_field": any(name in document["field_token_counts"] for name in ("asset_caption", "asset_aux")),
            }
        )
        return document

    def _tokenize_fields(self, field_texts: Dict[str, List[str]]) -> Dict[str, Any]:
        """对各字段文本分词，返回字段词频、字段长度、token 来源字段，以及 debug 的公共部分。

        没有任何 token 的字段不出现在结果中。
        """
        field_token_counts: Dict[str, Counter] = {}
        field_lengths: Dict[str, int] = {}
        token_sources: Dict[str, List[str]] = defaultdict(list)
        for field_name, texts in field_texts.items():
            tokens: List[str] = []
            for text in texts:
                tokens.extend(self.tokenizer(text))
            if not tokens:
                continue
            counts = Counter(tokens)
            field_token_counts[field_name] = counts
            field_lengths[field_name] = len(tokens)
            for token in counts:
                token_sources[token].append(field_name)
        return {
            "field_token_counts": field_token_counts,
            "field_lengths": field_lengths,
            "token_sources": {token: list(sources) for token, sources in token_sources.items()},
            "debug": {
                "field_lengths": dict(field_lengths),
                "field_weights": dict(KEYWORD_FIELD_WEIGHTS),
            },
        }

    def _build_keyword_document_fields_from_retrieval_index(
        self,
        chunk: Dict[str, Any],
        chunk_type: str,
        index_type: str,
        index_text: str,
    ) -> Dict[str, Any]:
        """把 RetrievalIndex 文本映射到 BM25 字段，命中后仍返回原 chunk payload。"""
        field_name = self._keyword_field_for_retrieval_index(index_type, chunk_type)
        document = self._tokenize_fields({field_name: [index_text]})
        document["debug"].update(
            {
                "skipped_fields": {},
                "has_asset_field": field_name in {"asset_caption", "asset_aux", "table_or_figure"},
                "retrieval_index_id": str(chunk.get("retrieval_index_id") or ""),
                "retrieval_index_type": index_type,
                "retrieval_index_weight": float(chunk.get("retrieval_index_weight", 1.0) or 1.0),
            }
        )
        return document

    @staticmethod
    def _keyword_field_for_retrieval_index(index_type: str, chunk_type: str) -> str:
        """保持 index_type 可见，同时复用既有字段加权和图表噪声控制规则。"""
        if index_type == "section_anchor":
            return "section_title"
        if index_type == "summary":
            return "summary"
        if index_type == "question":
            return "question"
        if index_type == "table_or_figure" or chunk_type in {"figure", "table"}:
            return "asset_caption"
        return "body"

    @staticmethod
    def _normalize_keyword_field_text(text: str) -> str:
        """小写并压缩空白，用于字段去重和噪声判断。"""
        return " ".join(str(text or "").strip().lower().split())

    @classmethod
    def _keyword_field_looks_noisy(cls, text: str, *, field_name: str) -> bool:
        """识别短标题、乱码和数值密集 preview，降低 OCR/版面解析污染进入 keyword 索引的概率。"""
        normalized = cls._normalize_keyword_field_text(text)
        if not normalized:
            return True
        alpha_cjk_count = len([char for char in normalized if char.isalnum() or "\u4e00" <= char <= "\u9fff"])
        if alpha_cjk_count < 3:
            return True
        # 出现 Unicode 替换字符（U+FFFD）说明文本在解码时已经损坏。
        if "\ufffd" in normalized:
            return True
        symbol_count = len(re.findall(r"[^a-z0-9\u4e00-\u9fff\s_\-./:]", normalized))
        if symbol_count / max(len(normalized), 1) > 0.28:
            return True
        if field_name in {"asset_preview_text", "asset_summary"}:
            digit_count = len(re.findall(r"\d", normalized))
            # preview 中纯表格数值和 OCR 残片信息量低，只让结构化表格 route 负责精确数值问题。
            if digit_count / max(len(normalized), 1) > 0.45:
                return True
        return False

    @staticmethod
    def _asset_section_anchor_allowed_for_search_text(chunk: Dict[str, Any], chunk_type: str) -> bool:
        """控制弱章节锚点是否进入 keyword 文本，和 embedding/rerank 的门控保持一致。"""
        if chunk_type not in {"figure", "table"}:
            return True
        match_type = str(chunk.get("asset_section_match_type", "") or "").strip()
        if not match_type:
            return True
        return bool(chunk.get("asset_section_match_allow_embedding", False))

    def _load_sparse_index_artifact(
        self,
        index_record: Optional[Dict[str, Any]],
        *,
        collection_name: str,
        collection_profile: Optional[CollectionRetrievalProfile],
    ) -> tuple[Optional[CollectionRetrievalIndex], Dict[str, Any]]:
        """优先从构建期 sparse artifact 恢复 BM25 结构；校验失败时只返回原因，由旧路径兜底。"""
        fallback_record = self._fallback_index_record(index_record)
        debug: Dict[str, Any] = {
            "enabled": bool(fallback_record),
            "source": "sparse_index_artifact",
            "sparse_index_dir": "",
            "sparse_index_manifest_file": "",
            "document_count": 0,
            "reason": "",
        }
        manifest_file = str((fallback_record or {}).get("sparse_index_manifest_file") or "").strip()
        sparse_index_dir = str((fallback_record or {}).get("sparse_index_dir") or "").strip()
        manifest_source = manifest_file or sparse_index_dir
        if not manifest_source:
            debug["reason"] = "missing_sparse_index_manifest"
            return None, debug

        try:
            payload = load_sparse_index_artifact_files(manifest_source)
        except Exception as exc:
            logger.warning("Failed to load sparse index artifact %s: %s", manifest_source, exc)
            debug["reason"] = f"sparse_index_read_failed:{exc}"
            return None, debug

        manifest = dict(payload.get("manifest") or {})
        normalized_source_type = normalize_sparse_source_type(manifest.get("source_type"))
        debug["sparse_index_dir"] = str(payload.get("artifact_dir") or sparse_index_dir)
        debug["sparse_index_manifest_file"] = str(payload.get("manifest_file") or manifest_file)
        for key in (
            "build_id",
            "index_version",
            "source_type",
            "source_file",
            "source_hash",
            "backend",
            "schema_version",
            "tokenizer_version",
            "field_weights_version",
            "bm25_schema_version",
        ):
            debug[key] = str(manifest.get(key) or "")
        debug["source_type"] = normalized_source_type
        debug["keyword_route_index_source"] = "persistent_sparse_artifact"
        if normalized_source_type not in SPARSE_SOURCE_TYPES:
            debug["reason"] = f"source_type_unsupported:{manifest.get('source_type')}"
            return None, debug

        # 以下任一校验失败都说明 artifact 不属于当前 active build，交给旧链路现场构建。
        expected_build_id = _record_build_id(fallback_record).strip()
        if expected_build_id and expected_build_id != str(manifest.get("build_id") or "").strip():
            debug["reason"] = "build_id_mismatch"
            return None, debug
        expected_index_version = _record_index_version(fallback_record).strip()
        if expected_index_version and expected_index_version != str(manifest.get("index_version") or "").strip():
            debug["reason"] = "index_version_mismatch"
            return None, debug
        expected_source_hash = str((fallback_record or {}).get("sparse_index_source_hash") or "").strip()
        if expected_source_hash and expected_source_hash != str(manifest.get("source_hash") or "").strip():
            debug["reason"] = "source_hash_mismatch"
            return None, debug
        version_mismatch = self._validate_sparse_manifest_versions(manifest)
        if version_mismatch:
            debug["reason"] = version_mismatch
            return None, debug

        documents = list(payload.get("documents") or [])
        expected_document_count = int(manifest.get("document_count", 0) or 0)
        debug["document_count"] = len(documents)
        if expected_document_count != len(documents):
            # manifest 是后续一致性判断的入口；数量不一致说明 artifact 不完整，必须退回旧构建链路。
            debug["reason"] = "document_count_mismatch"
            return None, debug

        lookup_tables = _new_lookup_tables()
        for doc_id, document in enumerate(documents):
            # lookup 表不单独落盘，加载时按 document.chunk 复建，避免源 chunk 字段和二级索引字段出现两套真相。
            document.doc_id = doc_id
            self._add_lookup_entries(doc_id, document.chunk, lookup_tables)

        token_stats = dict(payload.get("token_stats") or {})
        avgdl = float(token_stats.get("avgdl", manifest.get("avgdl", 0.0)) or 0.0)
        chunk_count = self._resolve_chunk_count(collection_profile, fallback_record, len(documents))
        index = CollectionRetrievalIndex(
            arxiv_id=str((fallback_record or {}).get("arxiv_id", "") or manifest.get("paper_id") or ""),
            collection_name=collection_name,
            index_version=str(manifest.get("index_version") or (fallback_record or {}).get("active_index_version") or ""),
            chunk_count=chunk_count,
            build_id=str(manifest.get("build_id") or (fallback_record or {}).get("active_build_id") or ""),
            chunk_file_hash=str(manifest.get("source_hash") or ""),
            created_at=str(manifest.get("created_at") or datetime.now().isoformat(timespec="seconds")),
            documents=documents,
            document_frequency=dict(payload.get("document_frequency") or {}),
            postings={token: set(doc_ids) for token, doc_ids in (payload.get("postings") or {}).items()},
            avgdl=avgdl,
            **_copy_lookup_tables(lookup_tables),
            retrieval_index_artifact_debug=debug,
            cache_hit=False,
            build_time=0.0,
            fallback_reason="",
            build_source="sparse_index_artifact",
        )
        return index, debug

    def _load_structured_tables(self, index_record: Optional[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """从 chunk 调试产物中恢复结构化表格索引。"""
        fallback_record = self._fallback_index_record(index_record)
        debug: Dict[str, Any] = {"enabled": bool(index_record or fallback_record), "structured_table_count": 0, "reason": ""}
        chunk_file = str((index_record or {}).get("chunk_file") or (fallback_record or {}).get("chunk_file") or "").strip()
        if not chunk_file:
            debug["reason"] = "missing_chunk_file"
            return [], debug

        payload, read_error = self._read_chunk_file(chunk_file, purpose="structured tables")
        if read_error:
            debug["reason"] = read_error
            return [], debug

        structured_tables = [item for item in (payload.get("structured_tables") or []) if isinstance(item, dict)]
        table_debug = payload.get("table_structure_debug") if isinstance(payload.get("table_structure_debug"), dict) else {}
        debug.update(table_debug)
        debug["structured_table_count"] = len(structured_tables)
        if not structured_tables:
            debug["reason"] = debug.get("reason") or "structured_tables_empty"
        return structured_tables, debug

    def _load_retrieval_index_chunks(self, index_record: Optional[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """从 retrieval index artifact 和 chunk file 恢复 BM25 输入，避免重启后依赖 Milvus metadata。"""
        fallback_record = self._fallback_index_record(index_record)
        debug: Dict[str, Any] = {
            "enabled": bool(fallback_record),
            "source": "retrieval_index_artifact",
            "retrieval_index_file": "",
            "chunk_file": "",
            "index_count": 0,
            "row_count": 0,
            "missing_chunk_ids": [],
            "reason": "",
        }
        retrieval_index_file = str((fallback_record or {}).get("retrieval_index_file") or "").strip()
        if not retrieval_index_file:
            debug["reason"] = "missing_retrieval_index_file"
            return [], debug
        debug["retrieval_index_file"] = retrieval_index_file

        try:
            artifact_payload = load_retrieval_index_artifact(retrieval_index_file)
        except Exception as exc:
            logger.warning("Failed to load retrieval index artifact %s: %s", retrieval_index_file, exc)
            debug["reason"] = f"retrieval_index_file_read_failed:{exc}"
            return [], debug

        raw_indexes = artifact_payload.get("retrieval_indexes") or artifact_payload.get("indexes") or []
        if not isinstance(raw_indexes, list):
            debug["reason"] = "retrieval_indexes_invalid"
            return [], debug
        debug["index_count"] = len(raw_indexes)

        chunk_file = str((fallback_record or {}).get("chunk_file") or "").strip()
        if not chunk_file:
            debug["reason"] = "missing_chunk_file"
            return [], debug
        debug["chunk_file"] = chunk_file
        chunk_payload, read_error = self._read_chunk_file(chunk_file, purpose="retrieval index rows")
        if read_error:
            debug["reason"] = read_error
            return [], debug

        chunks = chunk_payload.get("chunks")
        if not isinstance(chunks, list):
            debug["reason"] = "chunk_file_missing_chunks"
            return [], debug
        chunk_lookup = {
            normalize_chunk_id(dict(chunk or {}), fallback_index=index): dict(chunk or {})
            for index, chunk in enumerate(chunks, start=1)
            if isinstance(chunk, dict)
        }

        rows: List[Dict[str, Any]] = []
        missing_chunk_ids: Set[str] = set()
        for index_payload in raw_indexes:
            if not isinstance(index_payload, dict):
                continue
            retrieval_index = RetrievalIndex.from_dict(index_payload)
            chunk = chunk_lookup.get(str(retrieval_index.chunk_id))
            if chunk is None:
                missing_chunk_ids.add(str(retrieval_index.chunk_id))
                continue
            metadata = dict(chunk.get("metadata", {}) or {})
            retrieval_metadata = {
                "index_id": retrieval_index.index_id,
                "index_type": retrieval_index.index_type,
                "index_text": retrieval_index.index_text,
                "index_weight": float(retrieval_index.index_weight),
                "retrieval_index_id": retrieval_index.index_id,
                "retrieval_index_chunk_id": retrieval_index.chunk_id,
                "retrieval_index_type": retrieval_index.index_type,
                "retrieval_index_text": retrieval_index.index_text,
                "retrieval_index_weight": float(retrieval_index.index_weight),
                "retrieval_index_enabled_routes": list(retrieval_index.enabled_routes),
                "retrieval_index_metadata": dict(retrieval_index.metadata),
            }
            # BM25 需要 index_text，但最终 evidence 仍读取原 chunk；这里把两层字段并存后交给 normalize_chunk。
            row = {
                **chunk,
                **retrieval_metadata,
                "metadata": {
                    **metadata,
                    **retrieval_metadata,
                },
            }
            rows.append(row)

        debug["row_count"] = len(rows)
        debug["missing_chunk_ids"] = sorted(missing_chunk_ids)[:50]
        if not rows:
            debug["reason"] = debug["reason"] or "no_retrieval_index_rows"
        elif missing_chunk_ids:
            debug["reason"] = f"missing_chunk_ids:{len(missing_chunk_ids)}"
        return rows, debug

    def _load_legacy_chunk_rows(self, index_record: Optional[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """读取原始 chunk 文件作为旧 BM25 输入；只在灰度关闭 index-level BM25 时使用。"""
        fallback_record = self._fallback_index_record(index_record)
        debug: Dict[str, Any] = {
            "enabled": bool(fallback_record),
            "source": "legacy_chunk_file",
            "index_level_bm25_enabled": False,
            "chunk_file": "",
            "row_count": 0,
            "reason": "index_level_bm25_disabled",
        }
        chunk_file = str((fallback_record or {}).get("chunk_file") or "").strip()
        if not chunk_file:
            debug["reason"] = "missing_chunk_file"
            return [], debug
        debug["chunk_file"] = chunk_file

        payload, read_error = self._read_chunk_file(chunk_file, purpose="legacy chunk rows")
        if read_error:
            debug["reason"] = read_error
            return [], debug

        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            debug["reason"] = "chunk_file_missing_chunks"
            return [], debug

        rows = self._collapse_to_legacy_chunk_rows([chunk for chunk in chunks if isinstance(chunk, dict)])
        debug["row_count"] = len(rows)
        if not rows:
            debug["reason"] = "no_legacy_chunk_rows"
        return rows, debug

    @staticmethod
    def _collapse_to_legacy_chunk_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把 index-level 行折叠回 chunk 行，并清除 index 字段以触发旧 body index 合成。"""
        legacy_index_keys = {
            "index_id",
            "index_type",
            "index_text",
            "index_weight",
            "retrieval_index_id",
            "retrieval_index_type",
            "retrieval_index_text",
            "retrieval_index_weight",
            "retrieval_index_enabled_routes",
            "retrieval_index_metadata",
            "matched_index_id",
            "matched_index_type",
            "matched_index_text",
            "matched_index_score",
            "matched_indexes",
        }
        collapsed: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str, str]] = set()
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            metadata = dict(row.get("metadata", {}) or {})
            chunk_key = (
                normalize_chunk_id(row, fallback_index=index),
                str(row.get("content_part_label") or metadata.get("content_part_label") or ""),
                str(row.get("subchunk_index") or metadata.get("subchunk_index") or ""),
                str(row.get("order_index") or metadata.get("order_index") or ""),
                str(row.get("content") or metadata.get("content") or "")[:120],
            )
            if chunk_key in seen:
                continue
            seen.add(chunk_key)

            legacy_row = {key: value for key, value in row.items() if key not in legacy_index_keys}
            legacy_metadata = {key: value for key, value in metadata.items() if key not in legacy_index_keys}
            if legacy_metadata:
                legacy_row["metadata"] = legacy_metadata
            elif "metadata" in legacy_row:
                legacy_row["metadata"] = {}
            collapsed.append(legacy_row)
        return collapsed

    def _fallback_index_record(self, index_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """兼容测试夹具：未显式传入 paper_context 时，从 fake vector store 读取 chunk_file。"""
        if index_record:
            return index_record
        index_records = getattr(self.vector_store_service, "index_records", None)
        if isinstance(index_records, dict):
            for record in index_records.values():
                if isinstance(record, dict) and (record.get("chunk_file") or record.get("retrieval_index_file")):
                    return record
        return None

    @staticmethod
    def _read_chunk_file(chunk_file: str, *, purpose: str) -> Tuple[Any, str]:
        """定位并读取 chunk 文件，返回 (JSON 对象, 失败原因)；成功时失败原因为空串。

        顶层不是 JSON 对象时视为读取失败（chunk_file_not_object），调用方因此可以放心地对结果调用 .get。
        purpose 只用于日志，说明是哪个加载步骤读取失败。
        """
        chunk_path = resolve_artifact_file_path(chunk_file)
        if chunk_path is None:
            return None, "chunk_file_not_found"
        try:
            payload = _read_json(chunk_path)
        except Exception as exc:
            logger.warning("Failed to read chunk file for %s: path=%s error=%s", purpose, chunk_path, exc)
            return None, f"chunk_file_read_failed:{exc}"
        if not isinstance(payload, dict):
            logger.warning("Chunk file for %s is not a JSON object: path=%s", purpose, chunk_path)
            return None, "chunk_file_not_object"
        return payload, ""

    @staticmethod
    def _resolve_expected_count(
        collection_profile: Optional[CollectionRetrievalProfile],
        index_record: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        """期望的 chunk 数：数据库记录优先，其次 collection 画像；都没有时返回 None（不参与过期判断）。"""
        for key in ("chunk_count", "active_chunk_count"):
            if index_record and index_record.get(key) not in (None, ""):
                try:
                    return int(index_record.get(key) or 0)
                except (TypeError, ValueError):
                    return None
        if collection_profile is not None:
            return int(collection_profile.chunk_count or 0)
        return None

    @classmethod
    def _resolve_chunk_count(
        cls,
        collection_profile: Optional[CollectionRetrievalProfile],
        index_record: Optional[Dict[str, Any]],
        fallback_count: int,
    ) -> int:
        """记录的 chunk 数；没有可用记录时用实际读到的数量。"""
        expected = cls._resolve_expected_count(collection_profile, index_record)
        if expected is not None:
            return expected
        return fallback_count

    @staticmethod
    def _add_lookup_entries(doc_id: int, chunk: Dict[str, Any], tables: Dict[str, Dict[Any, List[int]]]) -> None:
        """把文档登记到各 lookup 表；空值字段不建键，chunk_type 缺失时按 text 处理。"""

        def add(table_name: str, key: Any) -> None:
            normalized = str(key or "").strip()
            if normalized:
                tables[table_name][normalized].append(doc_id)

        add("by_chunk_id", chunk.get("chunk_id"))
        add("by_parent_chunk_id", chunk.get("parent_chunk_id"))
        add("by_original_chunk_id", chunk.get("original_chunk_id"))

        page_number = str(chunk.get("page_number", "") or "").strip()
        section_path = str(chunk.get("section_path", "") or "").strip()
        section_title = str(chunk.get("section_title", "") or "").strip()
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        subchunk_index = str(chunk.get("subchunk_index", "") or "").strip()
        add("by_page_number", page_number)
        add("by_section_path", section_path)
        add("by_section_title", section_title)
        add("by_chunk_type", chunk_type)
        add("by_subchunk_index", subchunk_index)
        if page_number and section_path:
            tables["by_page_section"][(page_number, section_path.lower())].append(doc_id)
        if chunk.get("parent_chunk_id") not in (None, "") and subchunk_index:
            # parent+subchunk 组合索引用于上下文扩展阶段快速定位 sibling，避免每个 anchor 再全量扫描。
            tables["by_parent_subchunk"][(str(chunk.get("parent_chunk_id")).strip(), subchunk_index)].append(doc_id)

    @staticmethod
    def _clone_index(index: CollectionRetrievalIndex) -> CollectionRetrievalIndex:
        """复制索引供调用方使用：各张表都复制一份；documents 体积较大，按引用共享（调用方不要修改其中的文档）。"""
        return replace(
            index,
            document_frequency=dict(index.document_frequency),
            postings={token: set(doc_ids) for token, doc_ids in index.postings.items()},
            **_copy_lookup_tables({name: getattr(index, name) for name in _LOOKUP_TABLE_NAMES}),
            structured_tables=[dict(item) for item in index.structured_tables],
            table_structure_debug=dict(index.table_structure_debug),
            retrieval_index_artifact_debug=dict(index.retrieval_index_artifact_debug),
        )
