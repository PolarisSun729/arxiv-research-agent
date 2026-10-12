"""论文 QA 建库阶段的持久缓存（基于 diskcache）。

建库时最贵的两类调用是：
- LLM：为 chunk 生成 rerank 压缩文本、检索摘要、生成式问题；
- Embedding：为每个 retrieval index 生成向量。
同一篇论文重复建库（换加载方式、修 bug 后重建）时，输入大多不变，用内容哈希作 key 即可复用结果。

缓存只是成本优化：打开、读、写任何一步失败都只记录日志并退化为「无缓存」，绝不影响建库正确性。
key 中包含 prompt 版本 / 模型名 / schema 版本，修改 prompt 或模型时应同步提升版本号让旧缓存自然失效。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from utils.config import get_paper_qa_build_cache_config
from utils.storage_paths import BACKEND_DATA_ROOT, resolve_storage_path

logger = logging.getLogger(__name__)

# ---- 缓存 key 的版本号：对应 prompt / 数据结构变更时必须提升，否则会命中旧结果 ----
LLM_RERANK_TEXT_PROMPT_VERSION = "paper_qa_rerank_text_v1"
LLM_RETRIEVAL_SUMMARY_PROMPT_VERSION = "paper_qa_retrieval_summary_v1"
LLM_RETRIEVAL_QUESTIONS_PROMPT_VERSION = "paper_qa_retrieval_questions_v1"
EMBEDDING_CACHE_SCHEMA_VERSION = "paper_qa_embedding_v1"

# 图片内容哈希的进程内记忆容量：一次建库的 figure 数量通常远小于该值。
IMAGE_HASH_MEMO_SIZE = 1024


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
            )
        )
        self.llm_cache_name = str(self.config.get("llm_cache_name") or "paper_qa_llm_cache")
        self.embedding_cache_name = str(self.config.get("embedding_cache_name") or "paper_qa_embedding_cache")
        self.llm_size_limit = int(self.config.get("llm_size_limit") or 512 * 1024 * 1024)
        self.embedding_size_limit = int(self.config.get("embedding_size_limit") or 2 * 1024 * 1024 * 1024)
        self._cache_lock = threading.Lock()
        self._llm_cache: Any = None
        self._embedding_cache: Any = None
        # 按缓存实例分别记录打开失败原因：LLM 缓存损坏不应连带禁用 embedding 缓存，反之亦然。
        self._disabled_reasons: Dict[str, str] = {}
        # 图片哈希记忆：(真实路径, mtime_ns, size) -> sha256。get/set 同一张图只需完整读取一次，
        # 文件被替换时 mtime/size 变化会自然得到新 key。
        self._image_hash_memo: "OrderedDict[Tuple[str, int, int], str]" = OrderedDict()
        self._image_hash_lock = threading.Lock()

    @staticmethod
    def _stable_json(value: Any) -> str:
        """稳定序列化：键排序、紧凑分隔符，保证同一内容总是得到同一字符串（从而同一哈希）。"""
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _hash_payload(cls, value: Any) -> str:
        """对稳定序列化结果做 SHA-256，作为缓存 key 的主体。"""
        return hashlib.sha256(cls._stable_json(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata_fingerprint(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """只挑选会影响 LLM 输出的元数据字段参与 key 计算，忽略时间戳等易变字段，提高命中率。"""
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
        """懒加载打开 diskcache 实例；加锁保证多线程建库时只打开一次。

        self.enabled 是配置层面的总开关；单个缓存打开失败只禁用它自己（记录在 _disabled_reasons），
        另一个缓存仍可正常使用，失败的缓存也不会在每次调用时反复重试打开。
        """
        if not self.enabled or cache_attr in self._disabled_reasons:
            return None
        with self._cache_lock:
            cached = getattr(self, cache_attr)
            if cached is not None:
                return cached
            if cache_attr in self._disabled_reasons:
                return None
            try:
                from diskcache import Cache

                cache_dir = self.root_dir / cache_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                opened = Cache(str(cache_dir), size_limit=max(1, int(size_limit or 0)))
                setattr(self, cache_attr, opened)
                return opened
            except Exception as exc:
                # 缓存只能优化成本，不能影响建库正确性；初始化失败时直接降级为无缓存路径。
                self._disabled_reasons[cache_attr] = str(exc)
                logger.warning("Paper QA build cache disabled: cache=%s error=%s", cache_name, exc)
                return None

    def _llm(self) -> Any:
        """获取 LLM 结果缓存实例（可能为 None）。"""
        return self._open_cache("_llm_cache", self.llm_cache_name, self.llm_size_limit)

    def _embedding(self) -> Any:
        """获取 embedding 向量缓存实例（可能为 None）。"""
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
        """生成 LLM 缓存 key：任务类型 + 模型 + prompt 版本 + chunk 原文 + 关键元数据 + 额外参数。"""
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
        """读取 LLM 缓存；未命中、缓存不可用或读取异常都返回 None。"""
        cache = self._llm()
        if cache is None:
            return None
        try:
            return cache.get(self.make_llm_key(**key_kwargs))
        except Exception as exc:
            logger.warning("Read paper QA LLM cache failed: %s", exc)
            return None

    def set_llm_result(self, value: Any, **key_kwargs: Any) -> None:
        """写入 LLM 缓存；失败只记日志。"""
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
        """生成 embedding 缓存 key：provider + 模型 + 维度 + 输入内容指纹（多模态时包含图片内容哈希）。"""
        payload = {
            "schema": EMBEDDING_CACHE_SCHEMA_VERSION,
            "provider": str(provider or "").strip().lower(),
            "model_name": str(model_name or "").strip(),
            "dimension": str(dimension or ""),
            "input": self._embedding_input_fingerprint(dict(embedding_input or {})),
        }
        return f"embedding:{self._hash_payload(payload)}"

    def _embedding_input_fingerprint(self, embedding_input: Dict[str, Any]) -> Dict[str, Any]:
        """计算 embedding 输入指纹；多模态输入额外纳入图片文件内容哈希，图片被替换时缓存会自动失效。"""
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

    def _image_file_hash(self, image_path: str) -> str:
        """按 1MB 分块计算图片文件的 SHA-256；文件不存在或读取失败返回空串。

        get_embedding 与 set_embedding 会各算一次 key：这里按 (真实路径, mtime_ns, size) 做有界 LRU 记忆，
        同一张未变化的图片只完整读取一次；stat 失败时按「文件不存在」处理。
        """
        if not image_path:
            return ""
        path = Path(image_path)
        try:
            if not path.is_file():
                return ""
            stat = path.stat()
            memo_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            return ""
        with self._image_hash_lock:
            memoized = self._image_hash_memo.get(memo_key)
            if memoized is not None:
                self._image_hash_memo.move_to_end(memo_key)
                return memoized
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except Exception as exc:
            logger.warning("Hash paper QA embedding image failed: path=%s error=%s", image_path, exc)
            return ""
        image_hash = digest.hexdigest()
        with self._image_hash_lock:
            self._image_hash_memo[memo_key] = image_hash
            self._image_hash_memo.move_to_end(memo_key)
            while len(self._image_hash_memo) > IMAGE_HASH_MEMO_SIZE:
                self._image_hash_memo.popitem(last=False)
        return image_hash

    def get_embedding(
        self,
        *,
        provider: str,
        model_name: str,
        dimension: Any,
        embedding_input: Dict[str, Any],
    ) -> Optional[list]:
        """读取缓存向量；只有命中且类型为 list 时才返回 float 列表，否则返回 None。"""
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
        """写入向量缓存（统一转为 float 列表存储）；失败只记日志。"""
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
    """进程级单例：同一进程内所有建库任务共享同一组 diskcache 句柄。"""
    return PaperQABuildCache()
