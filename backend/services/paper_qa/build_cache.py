from __future__ import annotations

import hashlib
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from utils.config import get_paper_qa_build_cache_config
from utils.storage_paths import BACKEND_DATA_ROOT, resolve_storage_path

logger = logging.getLogger(__name__)

LLM_RERANK_TEXT_PROMPT_VERSION = "paper_qa_rerank_text_v1"
LLM_RETRIEVAL_SUMMARY_PROMPT_VERSION = "paper_qa_retrieval_summary_v1"
LLM_RETRIEVAL_QUESTIONS_PROMPT_VERSION = "paper_qa_retrieval_questions_v1"
EMBEDDING_CACHE_SCHEMA_VERSION = "paper_qa_embedding_v1"


class PaperQABuildCache:
    """论文 QA 建库阶段的持久缓存封装。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or get_paper_qa_build_cache_config())
        self.enabled = bool(self.config.get("enabled", True))
        # 自定义缓存目录和默认目录共用同一规则，避免昂贵的建库缓存散落到仓库根目录。
        self.root_dir = Path(
            resolve_storage_path(
                self.config.get("root_dir"),
                default_path=BACKEND_DATA_ROOT,
                option_name="PAPER_QA_BUILD_CACHE_DIR",
            )
        )
        self.llm_cache_name = str(self.config.get("llm_cache_name") or "paper_qa_llm_cache")
        self.embedding_cache_name = str(self.config.get("embedding_cache_name") or "paper_qa_embedding_cache")
        self.llm_size_limit = int(self.config.get("llm_size_limit") or 512 * 1024 * 1024)
        self.embedding_size_limit = int(self.config.get("embedding_size_limit") or 2 * 1024 * 1024 * 1024)
        self._cache_lock = threading.Lock()
        self._llm_cache: Any = None
        self._embedding_cache: Any = None
        self._disabled_reason = ""

    @staticmethod
    def _stable_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _hash_payload(cls, value: Any) -> str:
        return hashlib.sha256(cls._stable_json(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata_fingerprint(metadata: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "chunk_type",
            "section_title",
            "section_path",
            "page_number",
            "page_range",
            "word_count",
            "asset_kind",
            "asset_caption",
            "asset_summary",
            "asset_preview_text",
        )
        return {key: metadata.get(key) for key in keys if metadata.get(key) not in (None, "", [], {})}

    def _open_cache(self, cache_attr: str, cache_name: str, size_limit: int) -> Any:
        if not self.enabled:
            return None
        with self._cache_lock:
            cached = getattr(self, cache_attr)
            if cached is not None:
                return cached
            try:
                from diskcache import Cache

                cache_dir = self.root_dir / cache_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                opened = Cache(str(cache_dir), size_limit=max(1, int(size_limit or 0)))
                setattr(self, cache_attr, opened)
                return opened
            except Exception as exc:
                # 缓存只能优化成本，不能影响建库正确性；初始化失败时直接降级为无缓存路径。
                self.enabled = False
                self._disabled_reason = str(exc)
                logger.warning("Paper QA build cache disabled: %s", exc)
                return None

    def _llm(self) -> Any:
        return self._open_cache("_llm_cache", self.llm_cache_name, self.llm_size_limit)

    def _embedding(self) -> Any:
        return self._open_cache("_embedding_cache", self.embedding_cache_name, self.embedding_size_limit)

    def make_llm_key(
        self,
        *,
        kind: str,
        model_name: str,
        prompt_version: str,
        chunk_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = {
            "kind": str(kind or ""),
            "model_name": str(model_name or ""),
            "prompt_version": str(prompt_version or ""),
            "chunk_text": str(chunk_text or ""),
            "metadata": self._metadata_fingerprint(dict(metadata or {})),
            "extra": dict(extra or {}),
        }
        return f"llm:{kind}:{self._hash_payload(payload)}"

    def get_llm_result(self, **key_kwargs: Any) -> Optional[Any]:
        cache = self._llm()
        if cache is None:
            return None
        try:
            return cache.get(self.make_llm_key(**key_kwargs))
        except Exception as exc:
            logger.warning("Read paper QA LLM cache failed: %s", exc)
            return None

    def set_llm_result(self, value: Any, **key_kwargs: Any) -> None:
        cache = self._llm()
        if cache is None:
            return
        try:
            cache.set(self.make_llm_key(**key_kwargs), value)
        except Exception as exc:
            # 写缓存失败不应让已完成的 LLM 结果丢失，主流程继续使用本次结果。
            logger.warning("Write paper QA LLM cache failed: %s", exc)

    def make_embedding_key(
        self,
        *,
        provider: str,
        model_name: str,
        dimension: Any,
        embedding_input: Dict[str, Any],
    ) -> str:
        payload = {
            "schema": EMBEDDING_CACHE_SCHEMA_VERSION,
            "provider": str(provider or "").strip().lower(),
            "model_name": str(model_name or "").strip(),
            "dimension": str(dimension or ""),
            "input": self._embedding_input_fingerprint(dict(embedding_input or {})),
        }
        return f"embedding:{self._hash_payload(payload)}"

    def _embedding_input_fingerprint(self, embedding_input: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(embedding_input.get("mode") or "text").strip().lower()
        payload: Dict[str, Any] = {
            "mode": mode,
            "text": str(embedding_input.get("text") or ""),
            "chunk_type": str(embedding_input.get("chunk_type") or ""),
        }
        if mode == "multimodal":
            image_path = str(embedding_input.get("image_path") or "").strip()
            payload["image_path"] = image_path
            payload["image_hash"] = self._image_file_hash(image_path)
        return payload

    @staticmethod
    def _image_file_hash(image_path: str) -> str:
        if not image_path:
            return ""
        path = Path(image_path)
        if not path.is_file():
            return ""
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except Exception as exc:
            logger.warning("Hash paper QA embedding image failed: path=%s error=%s", image_path, exc)
            return ""

    def get_embedding(
        self,
        *,
        provider: str,
        model_name: str,
        dimension: Any,
        embedding_input: Dict[str, Any],
    ) -> Optional[list]:
        cache = self._embedding()
        if cache is None:
            return None
        try:
            vector = cache.get(
                self.make_embedding_key(
                    provider=provider,
                    model_name=model_name,
                    dimension=dimension,
                    embedding_input=embedding_input,
                )
            )
            if isinstance(vector, list):
                return [float(value) for value in vector]
        except Exception as exc:
            logger.warning("Read paper QA embedding cache failed: %s", exc)
        return None

    def set_embedding(
        self,
        vector: Any,
        *,
        provider: str,
        model_name: str,
        dimension: Any,
        embedding_input: Dict[str, Any],
    ) -> None:
        cache = self._embedding()
        if cache is None:
            return
        try:
            cache.set(
                self.make_embedding_key(
                    provider=provider,
                    model_name=model_name,
                    dimension=dimension,
                    embedding_input=embedding_input,
                ),
                [float(value) for value in vector],
            )
        except Exception as exc:
            logger.warning("Write paper QA embedding cache failed: %s", exc)


@lru_cache(maxsize=1)
def get_paper_qa_build_cache() -> PaperQABuildCache:
    return PaperQABuildCache()
