from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from services.retrieval.collection_profile import CollectionRetrievalProfile

logger = logging.getLogger(__name__)


@dataclass
class KeywordDocument:
    """BM25 文档项，保留 chunk payload 和预分词统计，查询时只遍历命中 postings 的候选。"""

    doc_id: int
    chunk: Dict[str, Any]
    token_counts: Counter
    doc_length: int
    content_prefix: str


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
    cache_hit: bool = False
    build_time: float = 0.0
    fallback_reason: str = ""
    build_source: str = "lazy_chunk_scan"

    def to_keyword_debug(self, candidate_count: int, *, full_scan_used: bool = False) -> Dict[str, Any]:
        return {
            "keyword_index_hit": self.cache_hit,
            "keyword_index_version": self.index_version,
            "keyword_candidate_count": candidate_count,
            "keyword_index_chunk_count": self.chunk_count,
            "keyword_index_build_id": self.build_id,
            "keyword_index_created_at": self.created_at,
            "keyword_index_build_time": self.build_time,
            "keyword_full_scan_used": full_scan_used,
            "keyword_index_build_source": self.build_source,
            "keyword_index_fallback_reason": self.fallback_reason,
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
        }


class CollectionRetrievalIndexProvider:
    """缓存 collection 的 keyword postings 和 chunk lookup，避免 route 查询时全量扫 chunk。"""

    def __init__(
        self,
        *,
        vector_store_service: Any,
        chunk_normalizer: Callable[[Dict[str, Any]], Dict[str, Any]],
        tokenizer: Callable[[str], List[str]],
    ) -> None:
        self.vector_store_service = vector_store_service
        self.chunk_normalizer = chunk_normalizer
        self.tokenizer = tokenizer
        self._cache: Dict[str, CollectionRetrievalIndex] = {}
        self._lock = threading.RLock()

    def get_index(
        self,
        collection_name: str,
        *,
        collection_profile: Optional[CollectionRetrievalProfile] = None,
        index_record: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> CollectionRetrievalIndex:
        resolved_name = self._resolve_collection_name(collection_name)
        with self._lock:
            cached = self._cache.get(resolved_name)
            stale_reason = self._stale_reason(cached, collection_profile, index_record)
            if cached and not force_refresh and not stale_reason:
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
            return index

    def invalidate_index(self, collection_name: str) -> None:
        """索引重建或 collection 删除后可主动清理本地 retrieval index。"""
        with self._lock:
            self._cache.pop(self._resolve_collection_name(collection_name), None)

    def _build_index(
        self,
        collection_name: str,
        *,
        collection_profile: Optional[CollectionRetrievalProfile],
        index_record: Optional[Dict[str, Any]],
        stale_reason: str,
    ) -> CollectionRetrievalIndex:
        started = perf_counter()
        fallback_reasons: List[str] = []
        if stale_reason:
            fallback_reasons.append(f"rebuilt_after_stale: {stale_reason}")

        try:
            raw_chunks = self.vector_store_service.get_all_chunks(collection_name) or []
        except Exception as exc:
            # route 查询不能继续回到无上限扫描；构建失败时返回空索引并把原因暴露给 debug。
            raw_chunks = []
            fallback_reasons.append(f"chunk_index_build_failed: {exc}")
            logger.warning("Failed to build collection retrieval index: collection=%s error=%s", collection_name, exc)

        structured_tables, table_structure_debug = self._load_structured_tables(index_record)
        if structured_tables:
            fallback_reasons.append(f"structured_table_count:{len(structured_tables)}")
        elif table_structure_debug.get("enabled", False) and table_structure_debug.get("reason"):
            fallback_reasons.append(f"structured_table_unavailable:{table_structure_debug.get('reason')}")

        normalized_chunks = [self.chunk_normalizer(chunk) for chunk in raw_chunks]
        documents: List[KeywordDocument] = []
        document_frequency: Dict[str, int] = defaultdict(int)
        postings: Dict[str, Set[int]] = defaultdict(set)
        by_chunk_id: Dict[str, List[int]] = defaultdict(list)
        by_parent_chunk_id: Dict[str, List[int]] = defaultdict(list)
        by_original_chunk_id: Dict[str, List[int]] = defaultdict(list)
        by_page_section: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        by_page_number: Dict[str, List[int]] = defaultdict(list)
        by_section_path: Dict[str, List[int]] = defaultdict(list)
        by_section_title: Dict[str, List[int]] = defaultdict(list)
        by_chunk_type: Dict[str, List[int]] = defaultdict(list)
        by_subchunk_index: Dict[str, List[int]] = defaultdict(list)
        by_parent_subchunk: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        for doc_id, chunk in enumerate(normalized_chunks):
            chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
            allow_section_terms = self._asset_section_anchor_allowed_for_search_text(chunk, chunk_type)
            text_parts = [
                str(chunk.get("content", "") or ""),
                str(chunk.get("title", "") or ""),
                str(chunk.get("section_path", "") or "") if allow_section_terms else "",
                str(chunk.get("section_title", "") or "") if allow_section_terms else "",
                str(chunk.get("asset_summary", "") or ""),
                str(chunk.get("asset_caption", "") or ""),
                str(chunk.get("asset_preview_text", "") or ""),
            ]
            tokens = self.tokenizer(" ".join(text_parts))
            token_counts = Counter(tokens)
            for token in token_counts:
                document_frequency[token] += 1
                postings[token].add(doc_id)

            documents.append(
                KeywordDocument(
                    doc_id=doc_id,
                    chunk=chunk,
                    token_counts=token_counts,
                    doc_length=max(len(tokens), 1),
                    content_prefix=str(chunk.get("content", "") or "").lower()[:300],
                )
            )
            self._add_lookup_entries(
                doc_id=doc_id,
                chunk=chunk,
                by_chunk_id=by_chunk_id,
                by_parent_chunk_id=by_parent_chunk_id,
                by_original_chunk_id=by_original_chunk_id,
                by_page_section=by_page_section,
                by_page_number=by_page_number,
                by_section_path=by_section_path,
                by_section_title=by_section_title,
                by_chunk_type=by_chunk_type,
                by_subchunk_index=by_subchunk_index,
                by_parent_subchunk=by_parent_subchunk,
            )

        chunk_count = self._resolve_chunk_count(collection_profile, index_record, len(normalized_chunks))
        index = CollectionRetrievalIndex(
            arxiv_id=str((index_record or {}).get("arxiv_id", "") or ""),
            collection_name=collection_name,
            index_version=str((index_record or {}).get("active_index_version") or (index_record or {}).get("index_version") or collection_name),
            chunk_count=chunk_count,
            build_id=str((index_record or {}).get("active_build_id") or (index_record or {}).get("build_id") or ""),
            chunk_file_hash=str((index_record or {}).get("chunk_file_hash") or (index_record or {}).get("chunk_file") or ""),
            created_at=datetime.now().isoformat(timespec="seconds"),
            documents=documents,
            document_frequency=dict(document_frequency),
            postings={token: set(doc_ids) for token, doc_ids in postings.items()},
            avgdl=sum(doc.doc_length for doc in documents) / max(len(documents), 1),
            by_chunk_id={key: list(value) for key, value in by_chunk_id.items()},
            by_parent_chunk_id={key: list(value) for key, value in by_parent_chunk_id.items()},
            by_original_chunk_id={key: list(value) for key, value in by_original_chunk_id.items()},
            by_page_section={key: list(value) for key, value in by_page_section.items()},
            by_page_number={key: list(value) for key, value in by_page_number.items()},
            by_section_path={key: list(value) for key, value in by_section_path.items()},
            by_section_title={key: list(value) for key, value in by_section_title.items()},
            by_chunk_type={key: list(value) for key, value in by_chunk_type.items()},
            by_subchunk_index={key: list(value) for key, value in by_subchunk_index.items()},
            by_parent_subchunk={key: list(value) for key, value in by_parent_subchunk.items()},
            structured_tables=[dict(item) for item in structured_tables],
            table_structure_debug=dict(table_structure_debug),
            cache_hit=False,
            build_time=perf_counter() - started,
            fallback_reason="; ".join(fallback_reasons),
        )
        return index

    def _stale_reason(
        self,
        index: Optional[CollectionRetrievalIndex],
        collection_profile: Optional[CollectionRetrievalProfile],
        index_record: Optional[Dict[str, Any]],
    ) -> str:
        if index is None:
            return ""
        expected_count = self._resolve_expected_count(collection_profile, index_record)
        if expected_count is not None and expected_count != index.chunk_count:
            return f"chunk_count_changed:{index.chunk_count}->{expected_count}"

        expected_version = str((index_record or {}).get("active_index_version") or (index_record or {}).get("index_version") or "").strip()
        if expected_version and index.index_version and expected_version != index.index_version:
            return f"index_version_changed:{index.index_version}->{expected_version}"

        expected_build_id = str((index_record or {}).get("active_build_id") or (index_record or {}).get("build_id") or "").strip()
        if expected_build_id and index.build_id and expected_build_id != index.build_id:
            return f"build_id_changed:{index.build_id}->{expected_build_id}"
        return ""

    def _resolve_collection_name(self, collection_name: str) -> str:
        resolver = getattr(self.vector_store_service, "resolve_collection_name", None)
        if callable(resolver):
            return resolver(collection_name)
        return collection_name

    @staticmethod
    def _asset_section_anchor_allowed_for_search_text(chunk: Dict[str, Any], chunk_type: str) -> bool:
        """控制弱章节锚点是否进入 keyword 文本，和 embedding/rerank 的门控保持一致。"""
        if chunk_type not in {"figure", "table"}:
            return True
        match_type = str(chunk.get("asset_section_match_type", "") or "").strip()
        if not match_type:
            return True
        return bool(chunk.get("asset_section_match_allow_embedding", False))

    def _load_structured_tables(self, index_record: Optional[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """从 chunk 调试产物中恢复结构化表格索引。"""
        fallback_record = self._fallback_index_record(index_record)
        debug: Dict[str, Any] = {"enabled": bool(index_record or fallback_record), "structured_table_count": 0, "reason": ""}
        chunk_file = str((index_record or {}).get("chunk_file") or (fallback_record or {}).get("chunk_file") or "").strip()
        if not chunk_file:
            debug["reason"] = "missing_chunk_file"
            return [], debug

        chunk_path = self._resolve_chunk_file_path(chunk_file)
        if chunk_path is None:
            debug["reason"] = "chunk_file_not_found"
            return [], debug

        try:
            with open(chunk_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load structured tables from chunk file %s: %s", chunk_path, exc)
            debug["reason"] = f"chunk_file_read_failed:{exc}"
            return [], debug

        structured_tables = [item for item in (payload.get("structured_tables") or []) if isinstance(item, dict)]
        table_debug = payload.get("table_structure_debug") if isinstance(payload.get("table_structure_debug"), dict) else {}
        debug.update(table_debug)
        debug["structured_table_count"] = len(structured_tables)
        if not structured_tables:
            debug["reason"] = debug.get("reason") or "structured_tables_empty"
        return structured_tables, debug

    def _fallback_index_record(self, index_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """兼容测试夹具：未显式传入 paper_context 时，从 fake vector store 读取 chunk_file。"""
        if index_record:
            return index_record
        index_records = getattr(self.vector_store_service, "index_records", None)
        if isinstance(index_records, dict):
            for record in index_records.values():
                if isinstance(record, dict) and record.get("chunk_file"):
                    return record
        return None

    def _resolve_chunk_file_path(self, chunk_file: str) -> Optional[Path]:
        raw_path = Path(str(chunk_file or "").strip())
        candidates = [raw_path]
        backend_root = Path(__file__).resolve().parents[2]
        repo_root = backend_root.parent
        if not raw_path.is_absolute():
            candidates.extend(
                [
                    Path(os.getcwd()) / raw_path,
                    backend_root / raw_path,
                    repo_root / raw_path,
                ]
            )
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_file():
                return resolved
        return None

    @staticmethod
    def _resolve_expected_count(
        collection_profile: Optional[CollectionRetrievalProfile],
        index_record: Optional[Dict[str, Any]],
    ) -> Optional[int]:
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
        expected = cls._resolve_expected_count(collection_profile, index_record)
        if expected is not None:
            return expected
        return fallback_count

    @staticmethod
    def _add_lookup_entries(
        *,
        doc_id: int,
        chunk: Dict[str, Any],
        by_chunk_id: Dict[str, List[int]],
        by_parent_chunk_id: Dict[str, List[int]],
        by_original_chunk_id: Dict[str, List[int]],
        by_page_section: Dict[Tuple[str, str], List[int]],
        by_page_number: Dict[str, List[int]],
        by_section_path: Dict[str, List[int]],
        by_section_title: Dict[str, List[int]],
        by_chunk_type: Dict[str, List[int]],
        by_subchunk_index: Dict[str, List[int]],
        by_parent_subchunk: Dict[Tuple[str, str], List[int]],
    ) -> None:
        def add(mapping: Dict[Any, List[int]], key: Any) -> None:
            normalized = str(key or "").strip()
            if normalized:
                mapping[normalized].append(doc_id)

        add(by_chunk_id, chunk.get("chunk_id"))
        add(by_parent_chunk_id, chunk.get("parent_chunk_id"))
        add(by_original_chunk_id, chunk.get("original_chunk_id"))

        page_number = str(chunk.get("page_number", "") or "").strip()
        section_path = str(chunk.get("section_path", "") or "").strip()
        section_title = str(chunk.get("section_title", "") or "").strip()
        chunk_type = str(chunk.get("chunk_type", "text") or "text").strip().lower()
        subchunk_index = str(chunk.get("subchunk_index", "") or "").strip()
        add(by_page_number, page_number)
        add(by_section_path, section_path)
        add(by_section_title, section_title)
        add(by_chunk_type, chunk_type)
        add(by_subchunk_index, subchunk_index)
        if page_number and section_path:
            by_page_section[(page_number, section_path.lower())].append(doc_id)
        if chunk.get("parent_chunk_id") not in (None, "") and subchunk_index:
            # parent+subchunk 组合索引用于上下文扩展阶段快速定位 sibling，避免每个 anchor 再全量扫描。
            by_parent_subchunk[(str(chunk.get("parent_chunk_id")).strip(), subchunk_index)].append(doc_id)

    @staticmethod
    def _clone_index(index: CollectionRetrievalIndex) -> CollectionRetrievalIndex:
        return CollectionRetrievalIndex(
            arxiv_id=index.arxiv_id,
            collection_name=index.collection_name,
            index_version=index.index_version,
            chunk_count=index.chunk_count,
            build_id=index.build_id,
            chunk_file_hash=index.chunk_file_hash,
            created_at=index.created_at,
            documents=index.documents,
            document_frequency=dict(index.document_frequency),
            postings={token: set(doc_ids) for token, doc_ids in index.postings.items()},
            avgdl=index.avgdl,
            by_chunk_id={key: list(value) for key, value in index.by_chunk_id.items()},
            by_parent_chunk_id={key: list(value) for key, value in index.by_parent_chunk_id.items()},
            by_original_chunk_id={key: list(value) for key, value in index.by_original_chunk_id.items()},
            by_page_section={key: list(value) for key, value in index.by_page_section.items()},
            by_page_number={key: list(value) for key, value in index.by_page_number.items()},
            by_section_path={key: list(value) for key, value in index.by_section_path.items()},
            by_section_title={key: list(value) for key, value in index.by_section_title.items()},
            by_chunk_type={key: list(value) for key, value in index.by_chunk_type.items()},
            by_subchunk_index={key: list(value) for key, value in index.by_subchunk_index.items()},
            by_parent_subchunk={key: list(value) for key, value in index.by_parent_subchunk.items()},
            structured_tables=[dict(item) for item in index.structured_tables],
            table_structure_debug=dict(index.table_structure_debug),
            cache_hit=index.cache_hit,
            build_time=index.build_time,
            fallback_reason=index.fallback_reason,
            build_source=index.build_source,
        )
