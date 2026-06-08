from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CollectionRetrievalProfile:
    """collection 级检索画像，集中保存向量检索和后续 route 可复用的稳定元信息。"""

    collection_name: str
    vector_dimension: Optional[int] = None
    embedding_provider: str = ""
    embedding_model: str = ""
    chunk_count: int = 0
    chunk_type_distribution: Dict[str, int] = field(default_factory=dict)
    section_distribution: Dict[str, int] = field(default_factory=dict)
    has_figure_or_table_chunk: bool = False
    metadata_availability: Dict[str, bool] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    is_stale: bool = False
    build_sources: List[str] = field(default_factory=list)
    cache_hit: bool = False
    build_time: float = 0.0
    fallback_reason: str = ""

    def has_required_fields(self) -> bool:
        return bool(self.collection_name and self.embedding_provider and self.embedding_model and self.vector_dimension)

    def to_debug(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "profile_cache_hit": self.cache_hit,
            "profile_build_time": self.build_time,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "vector_dimension": self.vector_dimension,
            "chunk_count": self.chunk_count,
            "chunk_type_distribution": self.chunk_type_distribution,
            "section_distribution": self.section_distribution,
            "has_figure_or_table_chunk": self.has_figure_or_table_chunk,
            "metadata_availability": self.metadata_availability,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_stale": self.is_stale,
            "build_sources": self.build_sources,
            "fallback_reason": self.fallback_reason,
        }


class CollectionRetrievalProfileProvider:
    """构建和缓存 collection 检索画像，避免每条 vector route 重复探测 schema/sample。"""

    METADATA_FIELDS = [
        "chunk_type",
        "section_title",
        "section_path",
        "asset_kind",
        "asset_path",
        "asset_summary",
        "embedding_provider",
        "embedding_model",
        "page_number",
        "source",
    ]

    def __init__(self, *, vector_store_service: Any, embedding_service: Any) -> None:
        self.vector_store_service = vector_store_service
        self.embedding_service = embedding_service
        self._cache: Dict[str, CollectionRetrievalProfile] = {}
        self._lock = threading.RLock()

    def get_profile(
        self,
        collection_name: str,
        *,
        index_record: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> CollectionRetrievalProfile:
        resolved_name = self._resolve_collection_name(collection_name)
        with self._lock:
            cached = self._cache.get(resolved_name)
            stale_reason = self._stale_reason(cached, index_record)
            if cached and not force_refresh and not stale_reason:
                profile = self._clone_profile(cached)
                profile.cache_hit = True
                profile.build_time = 0.0
                return profile

            if cached and stale_reason:
                cached.is_stale = True
                logger.info("Collection retrieval profile stale: collection=%s reason=%s", resolved_name, stale_reason)

            profile = self._build_profile(resolved_name, index_record=index_record, stale_reason=stale_reason)
            self._cache[resolved_name] = self._clone_profile(profile)
            return profile

    def refresh_profile(self, collection_name: str, *, index_record: Optional[Dict[str, Any]] = None) -> CollectionRetrievalProfile:
        return self.get_profile(collection_name, index_record=index_record, force_refresh=True)

    def invalidate_profile(self, collection_name: str) -> None:
        """索引重建或外部删除 collection 后可主动失效，避免继续复用旧画像。"""
        with self._lock:
            self._cache.pop(self._resolve_collection_name(collection_name), None)

    def _build_profile(
        self,
        collection_name: str,
        *,
        index_record: Optional[Dict[str, Any]],
        stale_reason: str = "",
    ) -> CollectionRetrievalProfile:
        started = perf_counter()
        now = datetime.now().isoformat(timespec="seconds")
        default_config = self.embedding_service.get_default_embedding_config()
        sources: List[str] = []
        fallback_reasons: List[str] = []
        collection_info: Dict[str, Any] = {}
        chunks: List[Dict[str, Any]] = []

        try:
            collection_info = self.vector_store_service.get_collection_info("milvus", collection_name) or {}
            if collection_info:
                sources.append("vector_store_schema")
        except Exception as exc:
            # schema 探测失败不能直接中断 QA；后面会用默认 embedding 配置兜底并把原因放进 debug。
            fallback_reasons.append(f"collection_info_failed: {exc}")
            logger.warning("Failed to read collection info for profile: collection=%s error=%s", collection_name, exc)

        try:
            chunks = self.vector_store_service.get_all_chunks(collection_name) or []
            if chunks:
                sources.append("chunk_metadata")
        except Exception as exc:
            fallback_reasons.append(f"chunk_metadata_failed: {exc}")
            logger.warning("Failed to read chunk metadata for profile: collection=%s error=%s", collection_name, exc)

        if index_record:
            sources.append("db_index_record")

        sample_metadata = self._first_metadata(chunks)
        expected_model = str((index_record or {}).get("embedding_model") or "").strip()
        vector_dimension = self._extract_vector_dimension(collection_info)
        chunk_count = self._resolve_chunk_count(collection_info, chunks, index_record)
        embedding_provider = str(sample_metadata.get("embedding_provider") or getattr(default_config, "provider", "") or "")
        embedding_model = str(sample_metadata.get("embedding_model") or expected_model or getattr(default_config, "model_name", "") or "")

        if not chunks:
            fallback_reasons.append("no_chunk_metadata")
        if stale_reason:
            fallback_reasons.append(f"rebuilt_after_stale: {stale_reason}")
        if not vector_dimension:
            # schema 里缺 vector dim 时保留默认维度，让检索尽量继续，同时在 debug 暴露降级原因。
            vector_dimension = getattr(default_config, "dimension", None)
            fallback_reasons.append("missing_vector_dimension_fallback_to_default")
        if not sample_metadata.get("embedding_provider"):
            fallback_reasons.append("missing_embedding_provider_fallback_to_default")
        if not sample_metadata.get("embedding_model") and not expected_model:
            fallback_reasons.append("missing_embedding_model_fallback_to_default")

        profile = CollectionRetrievalProfile(
            collection_name=collection_name,
            vector_dimension=int(vector_dimension) if vector_dimension else None,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chunk_count=chunk_count,
            chunk_type_distribution=self._chunk_type_distribution(chunks),
            section_distribution=self._section_distribution(chunks),
            has_figure_or_table_chunk=self._has_figure_or_table_chunk(chunks),
            metadata_availability=self._metadata_availability(chunks),
            created_at=now,
            updated_at=now,
            is_stale=False,
            build_sources=self._dedupe_sources(sources),
            cache_hit=False,
            build_time=perf_counter() - started,
            fallback_reason="; ".join(fallback_reasons),
        )
        return profile

    def _stale_reason(self, profile: Optional[CollectionRetrievalProfile], index_record: Optional[Dict[str, Any]]) -> str:
        if profile is None:
            return ""
        if not profile.has_required_fields():
            return "missing_required_profile_fields"

        expected_count = self._index_int(index_record, "chunk_count")
        if expected_count is not None and expected_count != profile.chunk_count:
            return f"chunk_count_changed:{profile.chunk_count}->{expected_count}"

        expected_model = str((index_record or {}).get("embedding_model") or "").strip()
        if expected_model and profile.embedding_model and expected_model != profile.embedding_model:
            return f"embedding_model_changed:{profile.embedding_model}->{expected_model}"
        return ""

    def _resolve_collection_name(self, collection_name: str) -> str:
        resolver = getattr(self.vector_store_service, "resolve_collection_name", None)
        if callable(resolver):
            return resolver(collection_name)
        return collection_name

    @staticmethod
    def _clone_profile(profile: CollectionRetrievalProfile) -> CollectionRetrievalProfile:
        return CollectionRetrievalProfile(
            collection_name=profile.collection_name,
            vector_dimension=profile.vector_dimension,
            embedding_provider=profile.embedding_provider,
            embedding_model=profile.embedding_model,
            chunk_count=profile.chunk_count,
            chunk_type_distribution=dict(profile.chunk_type_distribution),
            section_distribution=dict(profile.section_distribution),
            has_figure_or_table_chunk=profile.has_figure_or_table_chunk,
            metadata_availability=dict(profile.metadata_availability),
            created_at=profile.created_at,
            updated_at=profile.updated_at,
            is_stale=profile.is_stale,
            build_sources=list(profile.build_sources),
            cache_hit=profile.cache_hit,
            build_time=profile.build_time,
            fallback_reason=profile.fallback_reason,
        )

    @staticmethod
    def _extract_vector_dimension(collection_info: Dict[str, Any]) -> Optional[int]:
        schema = collection_info.get("schema", {}) if isinstance(collection_info, dict) else {}
        if not isinstance(schema, dict):
            return None
        for field in schema.get("fields", []) if isinstance(schema.get("fields", []), list) else []:
            if field.get("name") == "vector":
                value = field.get("dim") or (field.get("params", {}) or {}).get("dim")
                return int(value) if value else None
        return None

    @classmethod
    def _metadata_availability(cls, chunks: List[Dict[str, Any]]) -> Dict[str, bool]:
        availability = {field: False for field in cls.METADATA_FIELDS}
        for chunk in chunks:
            metadata = cls._metadata(chunk)
            for field_name in availability:
                if chunk.get(field_name) not in (None, "") or metadata.get(field_name) not in (None, ""):
                    availability[field_name] = True
        return availability

    @classmethod
    def _chunk_type_distribution(cls, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for chunk in chunks:
            metadata = cls._metadata(chunk)
            chunk_type = str(chunk.get("chunk_type") or metadata.get("chunk_type") or "text").strip() or "text"
            counter[chunk_type] += 1
        return dict(counter)

    @classmethod
    def _section_distribution(cls, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for chunk in chunks:
            metadata = cls._metadata(chunk)
            section = str(chunk.get("section_title") or metadata.get("section_title") or "").strip()
            if not section:
                section = str(chunk.get("section_path") or metadata.get("section_path") or "").strip()
            if section:
                counter[section] += 1
        return dict(counter)

    @classmethod
    def _has_figure_or_table_chunk(cls, chunks: List[Dict[str, Any]]) -> bool:
        for chunk in chunks:
            metadata = cls._metadata(chunk)
            chunk_type = str(chunk.get("chunk_type") or metadata.get("chunk_type") or "").lower()
            asset_kind = str(chunk.get("asset_kind") or metadata.get("asset_kind") or "").lower()
            if chunk_type in {"figure", "table"} or asset_kind in {"image", "table"}:
                return True
        return False

    @classmethod
    def _first_metadata(cls, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        for chunk in chunks:
            metadata = cls._metadata(chunk)
            if metadata:
                return metadata
        return {}

    @staticmethod
    def _metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _resolve_chunk_count(
        collection_info: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        index_record: Optional[Dict[str, Any]],
    ) -> int:
        for key in ("chunk_count", "active_chunk_count"):
            value = CollectionRetrievalProfileProvider._index_int(index_record, key)
            if value is not None:
                return value
        if isinstance(collection_info, dict):
            value = collection_info.get("num_entities", collection_info.get("row_count"))
            if value is not None:
                return int(value or 0)
        return len(chunks)

    @staticmethod
    def _index_int(index_record: Optional[Dict[str, Any]], key: str) -> Optional[int]:
        if not index_record or index_record.get(key) in (None, ""):
            return None
        try:
            return int(index_record.get(key) or 0)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dedupe_sources(sources: List[str]) -> List[str]:
        deduped: List[str] = []
        for source in sources:
            if source not in deduped:
                deduped.append(source)
        return deduped
