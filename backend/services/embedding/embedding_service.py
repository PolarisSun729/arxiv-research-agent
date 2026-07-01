"""向量生成服务模块。

该模块统一管理 embedding 的配置构造、provider 分发、结果缓存以及元数据
透传逻辑。这里不仅负责生成向量，也负责保证每个向量仍然可以追溯到原始
chunk、来源页码以及图片/表格等多模态资产，方便后续入库、检索与调试。
"""

import hashlib
import json
from datetime import datetime
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
import os
import base64
import mimetypes
import threading
import torch
from utils.model_utils import get_huggingface_model_path
import numpy as np
import sys
import requests
from services.retrieval.retrieval_index import (
    RetrievalIndex,
    build_retrieval_indexes,
    iter_retrieval_indexes_for_embedding,
    normalize_chunk_id,
)
from utils.config import EMBEDDING_CONFIG, get_enhanced_retrieval_runtime_config

logger = logging.getLogger(__name__)
ENHANCED_RETRIEVAL_CONFIG = get_enhanced_retrieval_runtime_config()

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    DASHSCOPE = "dashscope"
    LOCAL = "local"


class EmbeddingConfig:
    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_fusion: bool = False,
    ):
        """保存一次 embedding 调用所需的配置参数。

        参数:
            provider (str): 向量服务提供方名称。
            model_name (str): 具体模型名称。
            api_key (Optional[str]): 远程 provider 的访问密钥。
            base_url (Optional[str]): 自定义服务地址。
            dimension (Optional[int]): 目标向量维度。
            batch_size (Optional[int]): 批量调用大小。
            enable_fusion (bool): 是否开启多模态融合能力。
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.dimension = dimension
        self.batch_size = batch_size
        self.enable_fusion = enable_fusion
        self.aws_region = "ap-southeast-1"

    @classmethod
    def from_env(
        cls,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "EmbeddingConfig":
        """从运行时配置中构造默认 embedding 配置。

        参数:
            provider (Optional[str]): 可选 provider 覆盖值。
            model_name (Optional[str]): 可选模型名覆盖值。

        返回:
            EmbeddingConfig: 已补齐默认值的配置对象。

        这里会根据 provider 的不同补齐对应的 API key、默认模型、维度和
        批大小，避免业务层散落 provider-specific 的配置拼装逻辑。
        """
        normalized_provider = str(provider or EMBEDDING_CONFIG["provider"]).strip().lower()
        normalized_model = str(model_name or EMBEDDING_CONFIG["model_name"]).strip()

        if normalized_provider == EmbeddingProvider.DASHSCOPE.value:
            return cls(
                provider=normalized_provider,
                model_name=normalized_model or EMBEDDING_CONFIG["model_name"],
                api_key=EMBEDDING_CONFIG["dashscope_api_key"] or EMBEDDING_CONFIG["api_key"],
                base_url=EMBEDDING_CONFIG["base_url"],
                dimension=int(EMBEDDING_CONFIG["dimension"]),
                batch_size=int(EMBEDDING_CONFIG["batch_size"]) or 10,
            )

        if normalized_provider == EmbeddingProvider.LOCAL.value:
            return cls(
                provider=normalized_provider,
                model_name=normalized_model or EMBEDDING_CONFIG["model_name"],
                batch_size=int(EMBEDDING_CONFIG["batch_size"]) or 20,
            )

        if normalized_provider == EmbeddingProvider.OPENAI.value:
            return cls(
                provider=normalized_provider,
                model_name=normalized_model or EMBEDDING_CONFIG["openai_model"],
                api_key=EMBEDDING_CONFIG["openai_api_key"],
                base_url=EMBEDDING_CONFIG["openai_base_url"] or None,
                dimension=int(EMBEDDING_CONFIG["dimension"]) or None,
                batch_size=int(EMBEDDING_CONFIG["batch_size"]) or 20,
            )

        return cls(provider=normalized_provider, model_name=normalized_model)


class EmbeddingService:
    LOCAL_EMBEDDING_MODEL_PATH = EMBEDDING_CONFIG["local_model_path"]
    LOCAL_EMBEDDING_MODEL_SCRIPTS_PATH = EMBEDDING_CONFIG["local_model_scripts_path"]
    DASHSCOPE_EMBEDDING_URL = EMBEDDING_CONFIG["base_url"]
    DEFAULT_LOCAL_MODEL_NAME = EMBEDDING_CONFIG["model_name"]
    DEFAULT_DASHSCOPE_MODEL_NAME = EMBEDDING_CONFIG["model_name"]
    DEFAULT_DASHSCOPE_DIMENSION = int(EMBEDDING_CONFIG["dimension"])

    def __init__(self):
        """初始化 provider 工厂、本地模型句柄和进程内 embedding 缓存。

        返回:
            None
        """
        self.embedding_factory = EmbeddingFactory()
        self._local_embedder = None
        self._embedding_cache: dict[str, list] = {}
        self._embedding_cache_lock = threading.Lock()

    def get_default_embedding_config(self) -> EmbeddingConfig:
        """返回当前环境下的默认 embedding 配置。

        返回:
            EmbeddingConfig: 当前运行环境对应的默认配置。
        """
        return EmbeddingConfig.from_env()

    @staticmethod
    def _normalize_vector_output(embedding) -> list:
        """把不同 provider 返回的向量统一转换为 Python float list。

        参数:
            embedding: 任意 provider 返回的向量对象。

        返回:
            list: 标准化后的浮点数组。
        """
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        if isinstance(embedding, torch.Tensor):
            return embedding.to(dtype=torch.float32).cpu().tolist()
        if isinstance(embedding, list):
            return [float(x) for x in embedding]
        return [float(x) for x in np.asarray(embedding, dtype=np.float32).tolist()]

    @staticmethod
    def build_paper_embedding_text(title: str, abstract: str) -> str:
        """把论文标题和摘要拼装成单段 embedding 输入文本。

        参数:
            title (str): 论文标题。
            abstract (str): 论文摘要。

        返回:
            str: 适合直接送入向量模型的拼接文本。
        """
        title_value = str(title or "").strip()
        abstract_value = str(abstract or "").strip()
        return f"{title_value}\n\nAbstract: {abstract_value}".strip()

    def _build_embedding_cache_key(
        self,
        text: str,
        provider: str,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        dimension: Optional[int],
    ) -> str:
        """根据文本内容与调用配置生成稳定的缓存键。

        参数:
            text (str): 原始文本。
            provider (str): 向量 provider。
            model (str): 模型名。
            api_key (Optional[str]): 访问密钥。
            base_url (Optional[str]): 服务地址。
            dimension (Optional[int]): 目标维度。

        返回:
            str: 可用于进程内缓存的稳定键值。
        """
        text_hash = hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()
        return "|".join(
            [
                str(provider or "").strip().lower(),
                str(model or "").strip(),
                str(api_key or "").strip(),
                str(base_url or "").strip(),
                str(dimension or ""),
                text_hash,
            ]
        )

    def _get_cached_embedding(self, cache_key: str) -> Optional[list]:
        """线程安全地读取 embedding 缓存。

        参数:
            cache_key (str): 缓存键。

        返回:
            Optional[list]: 命中的向量；未命中时返回 None。
        """
        with self._embedding_cache_lock:
            cached = self._embedding_cache.get(cache_key)
            if cached is None:
                return None
            return [float(value) for value in cached]

    def _set_cached_embedding(self, cache_key: str, embedding: list) -> None:
        """线程安全地写入 embedding 缓存。

        参数:
            cache_key (str): 缓存键。
            embedding (list): 待缓存向量。

        返回:
            None
        """
        with self._embedding_cache_lock:
            self._embedding_cache[cache_key] = [float(value) for value in embedding]

    def _extract_dashscope_embeddings(self, payload: dict) -> list:
        """从 DashScope 的多种响应结构中提取并排序向量结果。"""
        output = payload.get("output", {}) if isinstance(payload, dict) else {}
        candidates = []
        for source in (
            output.get("embeddings") if isinstance(output, dict) else None,
            output.get("data") if isinstance(output, dict) else None,
            payload.get("embeddings") if isinstance(payload, dict) else None,
            payload.get("data") if isinstance(payload, dict) else None,
        ):
            if source:
                candidates = source
                break

        if not candidates and isinstance(output, dict) and "embedding" in output:
            candidates = [output]

        vectors = []
        for item in candidates or []:
            if isinstance(item, dict):
                vector = (
                    item.get("embedding")
                    or item.get("vector")
                    or item.get("text_embedding")
                    or item.get("embedding_vector")
                )
                if vector is None and "output" in item:
                    vector = item.get("output")
                if vector is not None:
                    vectors.append((int(item.get("index", item.get("text_index", len(vectors)))), self._normalize_vector_output(vector)))
            elif item is not None:
                vectors.append((len(vectors), self._normalize_vector_output(item)))

        # DashScope 可能返回 index/text_index，这里统一按原始顺序重排。
        vectors.sort(key=lambda item: item[0])
        return [vector for _, vector in vectors]

    @staticmethod
    def _extract_dashscope_usage(payload: dict) -> dict:
        """提取 DashScope 返回的 token 使用量信息。"""
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        return usage if isinstance(usage, dict) else {}

    def _create_dashscope_embeddings_from_inputs(self, embedding_inputs: list, config: EmbeddingConfig) -> tuple[list, dict]:
        """调用 DashScope 批量生成文本或多模态向量。"""
        api_key = config.api_key or EMBEDDING_CONFIG["dashscope_api_key"] or EMBEDDING_CONFIG["api_key"]
        if not api_key:
            raise ValueError("DashScope API key not provided. Set DASHSCOPE_API_KEY.")

        url = config.base_url or self.DASHSCOPE_EMBEDDING_URL
        contents = []
        for item in embedding_inputs:
            if item.get("mode") == "text":
                contents.append({"text": item["text"]})
                continue
            if item.get("mode") == "multimodal":
                if not item.get("image"):
                    raise ValueError("DashScope multimodal embedding input requires image data")
                # 多模态路径要求文本和图像一起提交，便于模型生成联合表征。
                contents.append(
                    {
                        "text": item["text"],
                        "image": item["image"],
                    }
                )
                continue
            raise ValueError(f"Unsupported DashScope embedding input mode: {item.get('mode')}")

        payload = {
            "model": config.model_name,
            "input": {
                "contents": contents,
            },
            "parameters": {
                "dimension": int(config.dimension or self.DEFAULT_DASHSCOPE_DIMENSION),
            },
        }
        if config.enable_fusion:
            # enable_fusion 交给后端做文本/图像融合，避免调用侧硬编码融合策略。
            payload["parameters"]["enable_fusion"] = True

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()

        vectors = self._extract_dashscope_embeddings(data)
        usage = self._extract_dashscope_usage(data)
        if not vectors:
            raise ValueError(f"DashScope embedding response did not contain vectors: {data}")
        if len(vectors) == len(embedding_inputs):
            return vectors, usage
        if len(embedding_inputs) == 1:
            return [vectors[0]], usage
        logger.warning(
            "DashScope returned %s vectors for %s inputs; falling back to single-item requests",
            len(vectors),
            len(embedding_inputs),
        )
        # 个别情况下批量结果数可能异常，这里退化到逐条请求以保证结果完整性。
        fallback_vectors = [self._create_dashscope_embedding_from_input(item, config) for item in embedding_inputs]
        return fallback_vectors, usage

    def _create_dashscope_embeddings(self, texts: list, config: EmbeddingConfig) -> list:
        """DashScope 文本批量 embedding 的轻量封装。"""
        embedding_inputs = [{"mode": "text", "text": text} for text in texts]
        vectors, _ = self._create_dashscope_embeddings_from_inputs(embedding_inputs, config)
        return vectors

    def _create_dashscope_embedding(self, text: str, config: EmbeddingConfig) -> list:
        """生成单条 DashScope 文本 embedding。"""
        return self._create_dashscope_embeddings([text], config)[0]

    def _create_dashscope_embedding_from_input(self, embedding_input: dict, config: EmbeddingConfig) -> list:
        """生成单条 DashScope 输入的 embedding，支持文本或多模态载荷。"""
        vectors, _ = self._create_dashscope_embeddings_from_inputs([embedding_input], config)
        return vectors[0]

    def _create_dashscope_embeddings_with_usage(self, texts: list, config: EmbeddingConfig) -> tuple[list, dict]:
        """生成 DashScope 文本 embedding，并返回 usage 信息。"""
        embedding_inputs = [{"mode": "text", "text": text} for text in texts]
        return self._create_dashscope_embeddings_from_inputs(embedding_inputs, config)

    def create_single_embedding_with_usage(
        self,
        text: str,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> tuple[list, dict]:
        """生成单条向量并在支持时返回 usage 信息。"""
        config = EmbeddingConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
        )
        cache_key = self._build_embedding_cache_key(
            text=text,
            provider=config.provider,
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            dimension=config.dimension,
        )
        cached_embedding = self._get_cached_embedding(cache_key)
        if cached_embedding is not None:
            return cached_embedding, {}

        normalized_provider = str(provider).strip().lower()
        if normalized_provider == EmbeddingProvider.DASHSCOPE.value:
            # DashScope 能返回调用计量信息，因此优先走专门分支。
            embedding, usage = self._create_dashscope_embeddings_with_usage([text], config)
            normalized_embedding = embedding[0]
            self._set_cached_embedding(cache_key, normalized_embedding)
            return normalized_embedding, usage

        embedding = self.create_single_embedding(text, provider, model, api_key, base_url, dimension)
        return embedding, {}

    @property
    def local_embedder(self):
        """按需懒加载本地多模态 embedding 模型。"""
        if self._local_embedder is None:
            self._local_embedder = self._load_local_qwen3_embedding_model()
        return self._local_embedder

    def _load_local_qwen3_embedding_model(self):
        """加载本地 Qwen3-VL embedding 模型，失败时返回 None。"""
        try:
            if os.path.exists(self.LOCAL_EMBEDDING_MODEL_PATH):
                sys.path.append(self.LOCAL_EMBEDDING_MODEL_SCRIPTS_PATH)
                from qwen3_vl_embedding import Qwen3VLEmbedder
                print(f"Loading local Qwen3-VL-Embedding-2B model from {self.LOCAL_EMBEDDING_MODEL_PATH}...")
                embedder = Qwen3VLEmbedder(
                    model_name_or_path=self.LOCAL_EMBEDDING_MODEL_PATH,
                    device_map="auto",
                )
                print("Local Qwen3-VL-Embedding-2B model loaded successfully!")
                return embedder
            print(f"Local model path not found: {self.LOCAL_EMBEDDING_MODEL_PATH}")
            return None
        except Exception as e:
            print(f"Error loading local Qwen3-VL-Embedding-2B model: {str(e)}")
            return None

    def create_embeddings(self, input_data: dict, config: EmbeddingConfig) -> tuple:
        """
        生成 embedding，同时保留页级元数据。

        这里不要只把 content 送进向量模型；chunk 的 source、页码、序号都要原样带下去，后面入库和问答才好溯源。
        """
        chunks = input_data.get("chunks", [])
        input_metadata = input_data.get("metadata", {})
        filename = input_metadata.get("filename", "")
        retrieval_indexes = input_data.get("retrieval_indexes")
        multi_index_embedding_enabled = self._metadata_bool(
            input_metadata.get("enable_multi_index_embedding"),
            bool(ENHANCED_RETRIEVAL_CONFIG.get("enable_multi_index_embedding", True)),
        )
        chunk_level_fallback_enabled = self._metadata_bool(
            input_metadata.get("enable_chunk_level_retrieval_fallback"),
            bool(ENHANCED_RETRIEVAL_CONFIG.get("enable_chunk_level_retrieval_fallback", True)),
        )
        default_embedding_mode = "retrieval_index" if multi_index_embedding_enabled else "legacy_chunk_level"
        embedding_mode = str(
            input_data.get("embedding_mode") or input_metadata.get("embedding_mode") or default_embedding_mode
        ).strip().lower()
        use_legacy_chunk_embeddings = (
            embedding_mode in {"chunk", "chunk_level", "legacy_chunk", "legacy_chunk_level"}
            or not multi_index_embedding_enabled
        )
        if retrieval_indexes is None and not use_legacy_chunk_embeddings:
            retrieval_indexes = build_retrieval_indexes(chunks)

        provider_key = str(config.provider).strip().lower()
        batch_size = int(config.batch_size or (10 if provider_key == EmbeddingProvider.DASHSCOPE.value else 20))
        results = []
        if use_legacy_chunk_embeddings:
            prepared_inputs = self._prepare_legacy_chunk_inputs(chunks, provider_key)
        else:
            prepared_inputs = self._prepare_retrieval_index_inputs(chunks, retrieval_indexes or [], provider_key)
            if not prepared_inputs and chunks:
                if chunk_level_fallback_enabled:
                    logger.warning(
                        "No valid retrieval index embedding inputs; falling back to legacy chunk-level embeddings"
                    )
                    # 兼容旧灰度路径：index 层为空或全部被路由过滤时，仍按 chunk.content 建向量，避免整篇论文无法建库。
                    prepared_inputs = self._prepare_legacy_chunk_inputs(chunks, provider_key)
                else:
                    logger.warning(
                        "No valid retrieval index embedding inputs and chunk-level fallback is disabled"
                    )

        if provider_key == EmbeddingProvider.LOCAL.value:
            if self.local_embedder is None:
                raise ValueError("Local Qwen3-VL-Embedding-2B model not loaded")

            for prepared in prepared_inputs:
                chunk = prepared["chunk"]
                # 本地模型逐条处理，便于同时支持文本和多模态资产输入。
                embedding_vector = self.create_single_embedding_local_input(prepared["embedding_input"])
                results.append(
                    {
                        "embedding": embedding_vector,
                        "metadata": self._build_embedding_metadata(
                            chunk=chunk,
                            chunk_count=len(chunks),
                            embedding_vector=embedding_vector,
                            provider=provider_key,
                            model=config.model_name,
                            filename=filename,
                            retrieval_index=prepared["retrieval_index"],
                        ),
                    }
                )
            return results, {}

        if provider_key == EmbeddingProvider.DASHSCOPE.value:
            if any(item["embedding_input"].get("mode") == "multimodal" for item in prepared_inputs):
                # DashScope 多模态输入目前按单条调用，避免图像载荷批量拼接复杂化。
                for prepared in prepared_inputs:
                    chunk = prepared["chunk"]
                    embedding_vector = self._create_dashscope_embedding_from_input(prepared["embedding_input"], config)
                    results.append(
                        {
                            "embedding": embedding_vector,
                            "metadata": self._build_embedding_metadata(
                                chunk=chunk,
                                chunk_count=len(chunks),
                                embedding_vector=embedding_vector,
                                provider=provider_key,
                                model=config.model_name,
                                filename=filename,
                                retrieval_index=prepared["retrieval_index"],
                            ),
                        }
                    )
            else:
                for i in range(0, len(prepared_inputs), batch_size):
                    batch = prepared_inputs[i : i + batch_size]
                    texts = [item["embedding_input"]["text"] for item in batch]
                    # 纯文本场景优先走批量调用，降低远程 provider 的请求成本。
                    embedding_vectors = self._create_dashscope_embeddings(texts, config)

                    for prepared, embedding_vector in zip(batch, embedding_vectors):
                        chunk = prepared["chunk"]
                        results.append(
                            {
                                "embedding": embedding_vector,
                                "metadata": self._build_embedding_metadata(
                                    chunk=chunk,
                                    chunk_count=len(chunks),
                                    embedding_vector=embedding_vector,
                                    provider=provider_key,
                                    model=config.model_name,
                                    filename=filename,
                                    retrieval_index=prepared["retrieval_index"],
                                ),
                            }
                        )
            return results, {}

        embedding_function = self.embedding_factory.create_embedding_function(config)
        if any(item["embedding_input"].get("mode") == "multimodal" for item in prepared_inputs):
            unsupported = provider_key not in {EmbeddingProvider.MODELSCOPE.value}
            if unsupported:
                raise ValueError(
                    f"Embedding provider/model does not support multimodal figure embedding: provider={config.provider}, model={config.model_name}"
                )

        if provider_key == EmbeddingProvider.OPENAI.value:
            for i in range(0, len(prepared_inputs), batch_size):
                batch = prepared_inputs[i : i + batch_size]
                texts = [item["embedding_input"]["text"] for item in batch]
                embedding_vectors = embedding_function.embed_documents(texts)

                for prepared, embedding_vector in zip(batch, embedding_vectors):
                    chunk = prepared["chunk"]
                    results.append(
                        {
                            "embedding": embedding_vector,
                            "metadata": self._build_embedding_metadata(
                                chunk=chunk,
                                chunk_count=len(chunks),
                                embedding_vector=embedding_vector,
                                provider=provider_key,
                                model=config.model_name,
                                filename=filename,
                                retrieval_index=prepared["retrieval_index"],
                            ),
                        }
                    )
        else:
            for prepared in prepared_inputs:
                chunk = prepared["chunk"]
                embedding_input = prepared["embedding_input"]
                if embedding_input.get("mode") == "multimodal":
                    # ModelScope 等 provider 在多模态场景直接接收结构化 payload。
                    embedding_vector = embedding_function.embed_query(embedding_input)
                else:
                    embedding_vector = embedding_function.embed_query(embedding_input["text"])
                results.append(
                    {
                        "embedding": embedding_vector,
                            "metadata": self._build_embedding_metadata(
                                chunk=chunk,
                                chunk_count=len(chunks),
                                embedding_vector=embedding_vector,
                                provider=provider_key,
                                model=config.model_name,
                                filename=filename,
                                retrieval_index=prepared["retrieval_index"],
                            ),
                        }
                )

        return results, {}

    @staticmethod
    def _metadata_bool(value: Any, default: bool) -> bool:
        """解析单次建库 metadata 中的布尔开关；缺失时回到全局配置。"""
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value or "").strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _prepare_retrieval_index_inputs(
        self,
        chunks: List[Dict[str, Any]],
        retrieval_indexes: List[Dict[str, Any]],
        provider_key: str,
    ) -> List[Dict[str, Any]]:
        chunk_lookup = {
            normalize_chunk_id(chunk, fallback_index=index): chunk
            for index, chunk in enumerate(chunks or [], start=1)
        }
        prepared_inputs: List[Dict[str, Any]] = []
        for retrieval_index in iter_retrieval_indexes_for_embedding(retrieval_indexes):
            chunk = chunk_lookup.get(str(retrieval_index.chunk_id))
            if chunk is None:
                logger.warning(
                    "Skip retrieval index without source chunk: index_id=%s chunk_id=%s",
                    retrieval_index.index_id,
                    retrieval_index.chunk_id,
                )
                continue
            # 向量输入来自 index_text，元数据来自 PaperChunk；两者分开才能让召回入口更细、最终证据仍完整。
            prepared_inputs.append(
                {
                    "chunk": chunk,
                    "retrieval_index": retrieval_index,
                    "embedding_input": self.build_embedding_input(
                        chunk,
                        provider_key,
                        retrieval_index=retrieval_index,
                    ),
                }
            )
        return prepared_inputs

    def _prepare_legacy_chunk_inputs(
        self,
        chunks: List[Dict[str, Any]],
        provider_key: str,
    ) -> List[Dict[str, Any]]:
        prepared_inputs: List[Dict[str, Any]] = []
        for chunk in chunks or []:
            embedding_input = self.build_embedding_input(chunk, provider_key, retrieval_index=None)
            if embedding_input.get("mode") == "text" and not str(embedding_input.get("text") or "").strip():
                continue
            # 旧 chunk-level 路径只作为兼容兜底；metadata 会合成 body index 标识，方便下游统一按 index 字段消费。
            prepared_inputs.append(
                {
                    "chunk": chunk,
                    "retrieval_index": None,
                    "embedding_input": embedding_input,
                    "embedding_source_level": "chunk_level_fallback",
                }
            )
        return prepared_inputs

    def build_embedding_input(self, chunk: dict, provider_key: str, retrieval_index: Optional[RetrievalIndex] = None) -> dict:
        """把 chunk 或 retrieval index 转成 provider 可消费的 embedding 输入载荷。"""
        metadata = chunk.get("metadata", {}) or {}
        chunk_type = str(chunk.get("chunk_type") or metadata.get("chunk_type") or "text").strip().lower()
        text = str(
            (retrieval_index.index_text if retrieval_index is not None else "")
            or chunk.get("content")
            or metadata.get("content")
            or chunk.get("text")
            or metadata.get("text")
            or ""
        ).strip()

        if chunk_type == "figure":
            if provider_key not in {
                EmbeddingProvider.DASHSCOPE.value,
                EmbeddingProvider.LOCAL.value,
                EmbeddingProvider.MODELSCOPE.value,
            }:
                raise ValueError(
                    f"Embedding provider/model does not support multimodal figure embedding: provider={provider_key}"
                )
            image_path = str(chunk.get("asset_abs_path") or metadata.get("asset_abs_path") or "").strip()
            if not image_path:
                raise ValueError("Figure chunk is missing asset_abs_path for multimodal embedding")
            if not os.path.exists(image_path):
                raise ValueError(f"Figure asset image path does not exist: {image_path}")
            image_input = image_path
            if provider_key == EmbeddingProvider.DASHSCOPE.value:
                # DashScope 接口需要 data URL，而不是本地磁盘路径。
                image_input = self._image_path_to_data_url(image_path)
            return {
                "mode": "multimodal",
                "text": text,
                "image": image_input,
                "image_path": image_path,
                "chunk_type": chunk_type,
            }

        return {
            "mode": "text",
            "text": text,
            "chunk_type": chunk_type,
        }

    def _build_embedding_metadata(
        self,
        chunk: dict,
        chunk_count: int,
        embedding_vector: list,
        provider: str,
        model: str,
        filename: str,
        retrieval_index: Optional[RetrievalIndex] = None,
    ) -> dict:
        """构造随向量一起保存的溯源元数据。"""
        chunk_metadata = chunk.get("metadata", {})
        # 这里把 chunk 层的页码和来源字段继续向下传，避免 embedding 阶段把结构压扁。
        page_start = int(chunk_metadata.get("page_start", chunk_metadata.get("page_number", 1)))
        page_end = int(chunk_metadata.get("page_end", page_start))
        source = chunk_metadata.get("source", filename)
        chunk_index = int(chunk_metadata.get("chunk_index", chunk_metadata.get("chunk_id", 0)))

        retrieval_index_payload = {}
        if retrieval_index is not None:
            retrieval_index_payload = {
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
                "matched_index_id": retrieval_index.index_id,
                "matched_index_type": retrieval_index.index_type,
                "matched_index_text": retrieval_index.index_text,
                "embedding_source_level": "retrieval_index",
            }
        else:
            legacy_chunk_id = normalize_chunk_id(chunk)
            legacy_index_text = str(
                chunk.get("content")
                or chunk_metadata.get("content")
                or chunk.get("text")
                or chunk_metadata.get("text")
                or ""
            ).strip()
            # chunk-level 兜底也写成 body index 形态，保证 Milvus/检索输出不需要再区分两套 metadata 合约。
            retrieval_index_payload = {
                "index_id": f"{legacy_chunk_id}:body:legacy",
                "index_type": "body",
                "index_text": legacy_index_text,
                "index_weight": 1.0,
                "retrieval_index_id": f"{legacy_chunk_id}:body:legacy",
                "retrieval_index_chunk_id": legacy_chunk_id,
                "retrieval_index_type": "body",
                "retrieval_index_text": legacy_index_text,
                "retrieval_index_weight": 1.0,
                "retrieval_index_enabled_routes": ["vector_original", "vector_rewrite", "vector_hyde", "keyword"],
                "retrieval_index_metadata": {"fallback": "chunk_level"},
                "matched_index_id": f"{legacy_chunk_id}:body:legacy",
                "matched_index_type": "body",
                "matched_index_text": legacy_index_text,
                "embedding_source_level": "chunk_level_fallback",
            }

        return {
            **chunk_metadata,
            "source": source,
            "document_name": chunk_metadata.get("document_name", source),
            "chunk_index": chunk_index,
            "chunk_id": int(chunk_metadata.get("chunk_id", chunk_index or 0)),
            "page_start": page_start,
            "page_end": page_end,
            "page_number": str(page_start),
            "page_range": chunk_metadata.get("page_range", f"{page_start}-{page_end}"),
            "content": chunk["content"],
            "rerank_text": str(chunk.get("rerank_text", chunk_metadata.get("rerank_text", "")) or ""),
            "chunk_type": str(chunk.get("chunk_type", chunk_metadata.get("chunk_type", "text")) or "text"),
            "asset_kind": str(chunk.get("asset_kind", chunk_metadata.get("asset_kind", "")) or ""),
            "asset_path": str(chunk.get("asset_path", chunk_metadata.get("asset_path", "")) or ""),
            "asset_abs_path": str(chunk.get("asset_abs_path", chunk_metadata.get("asset_abs_path", "")) or ""),
            "asset_summary": str(chunk.get("asset_summary", chunk_metadata.get("asset_summary", "")) or ""),
            "asset_preview_text": str(chunk.get("asset_preview_text", chunk_metadata.get("asset_preview_text", "")) or ""),
            "asset_rows": int(chunk.get("asset_rows", chunk_metadata.get("asset_rows", 0)) or 0),
            "asset_columns": int(chunk.get("asset_columns", chunk_metadata.get("asset_columns", 0)) or 0),
            "asset_caption": str(chunk.get("asset_caption", chunk_metadata.get("asset_caption", "")) or ""),
            # asset-section 匹配结果是弱章节锚点语义，下游需要保留依据和门控结果来解释召回来源。
            "asset_section_match_type": str(chunk.get("asset_section_match_type", chunk_metadata.get("asset_section_match_type", "")) or ""),
            "asset_section_match_confidence": float(
                chunk.get("asset_section_match_confidence", chunk_metadata.get("asset_section_match_confidence", 0.0)) or 0.0
            ),
            "asset_section_match_reason": str(chunk.get("asset_section_match_reason", chunk_metadata.get("asset_section_match_reason", "")) or ""),
            "asset_section_match_is_heuristic": bool(
                chunk.get("asset_section_match_is_heuristic", chunk_metadata.get("asset_section_match_is_heuristic", False))
            ),
            "asset_section_match_allow_embedding": bool(
                chunk.get("asset_section_match_allow_embedding", chunk_metadata.get("asset_section_match_allow_embedding", False))
            ),
            "order_index": int(chunk.get("order_index", chunk_metadata.get("order_index", 0)) or 0),
            "word_count": int(chunk_metadata.get("word_count", len(chunk["content"].split()))),
            "total_chunks": int(chunk_count),
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_timestamp": datetime.now().isoformat(),
            "vector_dimension": len(embedding_vector),
            "filename": filename,
            **retrieval_index_payload,
        }

    def save_embeddings(self, doc_name: str, embeddings: list) -> str:
        """把 embedding 结果保存到磁盘，便于后续入库或调试复用。"""
        os.makedirs("02-embedded-docs", exist_ok=True)

        first_embedding = embeddings[0]
        provider = first_embedding["metadata"]["embedding_provider"]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        retrieval_index_ids = [
            str(item.get("metadata", {}).get("retrieval_index_id") or "")
            for item in embeddings
            if str(item.get("metadata", {}).get("retrieval_index_id") or "")
        ]
        retrieval_index_types = sorted(
            {
                str(item.get("metadata", {}).get("retrieval_index_type") or "")
                for item in embeddings
                if str(item.get("metadata", {}).get("retrieval_index_type") or "")
            }
        )

        base_name = doc_name.split("_")[0]
        if not base_name.endswith(".pdf"):
            base_name += ".pdf"

        filename = f"{base_name.replace('.pdf', '')}_{provider}_{timestamp}.json"
        filepath = os.path.join("02-embedded-docs", filename)

        config_info = {
            # 顶层信息保留文档来源，便于之后从 embedding 文件追到原 PDF。
            "filename": base_name,
            "chunked_doc_name": doc_name,
            "created_at": datetime.now().isoformat(),
            "embedding_provider": provider,
            "embedding_model": first_embedding["metadata"]["embedding_model"],
            "vector_dimension": first_embedding["metadata"]["vector_dimension"],
            "source_file": first_embedding["metadata"].get("source", base_name),
            # embedding 文件的条数现在对应 RetrievalIndex，单独记录数量避免和原始 chunk_count 混淆。
            "retrieval_index_count": len(retrieval_index_ids),
            "retrieval_index_types": retrieval_index_types,
        }

        class CompactJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)

            def encode(self, obj):
                def format_list(value):
                    if isinstance(value, list):
                        if value and isinstance(value[0], (int, float)):
                            return "[" + ",".join(map(str, value)) + "]"
                        return [format_list(item) for item in value]
                    if isinstance(value, dict):
                        return {k: format_list(v) for k, v in value.items()}
                    return value

                return super().encode(format_list(obj))

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **config_info,
                    "embeddings": embeddings,
                },
                f,
                ensure_ascii=False,
                indent=2,
                cls=CompactJSONEncoder,
            )

        return filepath

    def create_single_embedding_local_input(self, embedding_input: dict) -> list:
        """使用本地模型处理单条文本或多模态输入。"""
        if self.local_embedder is None:
            raise ValueError("Local Qwen3-VL-Embedding-2B model not loaded")

        try:
            payload = {"text": embedding_input.get("text", "")}
            if embedding_input.get("mode") == "multimodal":
                payload["image"] = embedding_input.get("image_path") or embedding_input.get("image")
            embeddings = self.local_embedder.process([payload])
            embedding_tensor = embeddings[0]
            if isinstance(embedding_tensor, torch.Tensor):
                return embedding_tensor.to(dtype=torch.float32).cpu().tolist()
            return np.asarray(embedding_tensor, dtype=np.float32).tolist()
        except Exception as e:
            print(f"Error creating embedding with local model: {str(e)}")
            raise

    def create_single_embedding(
        self,
        text: str,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> list:
        """生成单条文本 embedding，并优先复用缓存。"""
        config = EmbeddingConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
        )
        cache_key = self._build_embedding_cache_key(
            text=text,
            provider=config.provider,
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            dimension=config.dimension,
        )
        cached_embedding = self._get_cached_embedding(cache_key)
        if cached_embedding is not None:
            return cached_embedding

        normalized_provider = str(provider).strip().lower()
        if normalized_provider == EmbeddingProvider.LOCAL.value:
            embedding = self.create_single_embedding_local(text)
            self._set_cached_embedding(cache_key, embedding)
            return embedding
        if normalized_provider == EmbeddingProvider.DASHSCOPE.value:
            embedding = self._create_dashscope_embedding(text, config)
            self._set_cached_embedding(cache_key, embedding)
            return embedding

        # 其余 provider 通过统一工厂创建具体 embedding function。
        embedding_function = self.embedding_factory.create_embedding_function(config)
        embedding = embedding_function.embed_query(text)
        normalized_embedding = self._normalize_vector_output(embedding)
        self._set_cached_embedding(cache_key, normalized_embedding)
        return normalized_embedding

    def create_text_embeddings(
        self,
        texts: List[str],
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[list]:
        """批量生成文本 embedding，不关心 usage 统计时使用该封装。"""
        embeddings, _ = self.create_text_embeddings_with_usage(
            texts=texts,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
            batch_size=batch_size,
        )
        return embeddings

    def create_text_embeddings_with_usage(
        self,
        texts: List[str],
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> tuple[List[list], dict]:
        """批量生成文本 embedding，并在支持时汇总 usage 信息。"""
        normalized_texts = [str(text or "") for text in texts]
        if not normalized_texts:
            return [], {}

        config = EmbeddingConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            dimension=dimension,
            batch_size=batch_size,
        )

        normalized_provider = str(config.provider).strip().lower()
        results: List[Optional[list]] = [None] * len(normalized_texts)
        pending_indexes: List[int] = []
        pending_texts: List[str] = []
        usage: dict = {}

        for index, text in enumerate(normalized_texts):
            cache_key = self._build_embedding_cache_key(
                text=text,
                provider=config.provider,
                model=config.model_name,
                api_key=config.api_key,
                base_url=config.base_url,
                dimension=config.dimension,
            )
            cached_embedding = self._get_cached_embedding(cache_key)
            if cached_embedding is not None:
                results[index] = cached_embedding
                continue
            pending_indexes.append(index)
            pending_texts.append(text)

        if pending_texts:
            if normalized_provider == EmbeddingProvider.DASHSCOPE.value:
                effective_batch_size = max(1, int(batch_size or config.batch_size or 20))
                for start in range(0, len(pending_texts), effective_batch_size):
                    batch_indexes = pending_indexes[start : start + effective_batch_size]
                    batch_texts = pending_texts[start : start + effective_batch_size]
                    batch_embeddings, batch_usage = self._create_dashscope_embeddings_with_usage(batch_texts, config)
                    if len(batch_embeddings) != len(batch_texts):
                        raise ValueError(
                            f"DashScope returned {len(batch_embeddings)} embeddings for {len(batch_texts)} texts"
                        )
                    if batch_usage:
                        # 多批次时把 usage 累加起来，方便上层做总量统计。
                        usage = {
                            "input_tokens": int(usage.get("input_tokens", 0) or 0) + int(batch_usage.get("input_tokens", 0) or 0),
                            "output_tokens": int(usage.get("output_tokens", 0) or 0) + int(batch_usage.get("output_tokens", 0) or 0),
                            "total_tokens": int(usage.get("total_tokens", 0) or 0) + int(batch_usage.get("total_tokens", 0) or 0),
                        }
                    for index, text, embedding in zip(batch_indexes, batch_texts, batch_embeddings):
                        cache_key = self._build_embedding_cache_key(
                            text=text,
                            provider=config.provider,
                            model=config.model_name,
                            api_key=config.api_key,
                            base_url=config.base_url,
                            dimension=config.dimension,
                        )
                        normalized_embedding = self._normalize_vector_output(embedding)
                        self._set_cached_embedding(cache_key, normalized_embedding)
                        results[index] = normalized_embedding
            elif normalized_provider == EmbeddingProvider.OPENAI.value:
                embedding_function = self.embedding_factory.create_embedding_function(config)
                effective_batch_size = max(1, int(batch_size or config.batch_size or 20))
                for start in range(0, len(pending_texts), effective_batch_size):
                    batch_indexes = pending_indexes[start : start + effective_batch_size]
                    batch_texts = pending_texts[start : start + effective_batch_size]
                    batch_embeddings = embedding_function.embed_documents(batch_texts)
                    if len(batch_embeddings) != len(batch_texts):
                        raise ValueError(
                            f"OpenAI-compatible embedder returned {len(batch_embeddings)} embeddings for {len(batch_texts)} texts"
                        )
                    for index, text, embedding in zip(batch_indexes, batch_texts, batch_embeddings):
                        cache_key = self._build_embedding_cache_key(
                            text=text,
                            provider=config.provider,
                            model=config.model_name,
                            api_key=config.api_key,
                            base_url=config.base_url,
                            dimension=config.dimension,
                        )
                        normalized_embedding = self._normalize_vector_output(embedding)
                        self._set_cached_embedding(cache_key, normalized_embedding)
                        results[index] = normalized_embedding
            else:
                for index, text in zip(pending_indexes, pending_texts):
                    # 不支持稳定批量接口的 provider 退化到逐条生成，保证兼容性。
                    results[index] = self.create_single_embedding(
                        text=text,
                        provider=provider,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        dimension=dimension,
                    )

        return [embedding if embedding is not None else [] for embedding in results], usage

    def create_single_embedding_dashscope(
        self,
        text: str,
        model: str = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list:
        """使用 DashScope 配置生成单条文本 embedding。"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.DASHSCOPE.value,
            model_name=model or self.DEFAULT_DASHSCOPE_MODEL_NAME,
            api_key=api_key or EMBEDDING_CONFIG["dashscope_api_key"] or EMBEDDING_CONFIG["api_key"],
            base_url=base_url,
            dimension=dimension or self.DEFAULT_DASHSCOPE_DIMENSION,
        )
        return self._create_dashscope_embedding(text, config)

    def create_single_embedding_local(self, text: str) -> list:
        """使用本地模型生成单条纯文本 embedding。"""
        return self.create_single_embedding_local_input({"mode": "text", "text": text})

    def create_single_embedding_modelscope(self, text: str, model: str = "Qwen/Qwen3-VL-Embedding-2B") -> list:
        """直接调用 ModelScope pipeline 生成 embedding。"""
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            pipe = pipeline(Tasks.multi_modal_embedding, model=model)
            payload = text if isinstance(text, dict) else {"text": text}
            result = pipe(payload)
            if isinstance(result, dict) and "text_embedding" in result:
                embedding = result["text_embedding"]
            elif isinstance(result, list) and len(result) > 0:
                embedding = result[0]
            else:
                embedding = result

            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = [float(x) for x in embedding]

            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding with ModelScope: {str(e)}")
            raise

    def get_document_embedding_config(self, collection_name: str) -> EmbeddingConfig:
        """根据已保存的 embedding 文件反查某个集合对应的向量配置。"""
        try:
            doc_name = collection_name.split("_")[0]
            embedded_docs_dir = "02-embedded-docs"
            for filename in os.listdir(embedded_docs_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(embedded_docs_dir, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("filename") == doc_name:
                            return EmbeddingConfig(
                                provider=data.get("embedding_provider"),
                                model_name=data.get("embedding_model"),
                                dimension=int(data.get("vector_dimension")) if data.get("vector_dimension") else None,
                            )
            raise ValueError(f"No matching embedding configuration found for collection: {collection_name}")
        except Exception as e:
            raise ValueError(f"Error getting embedding config: {str(e)}")

    def _image_path_to_data_url(self, image_path: str) -> str:
        """把本地图片文件转换为 provider 可直接提交的 data URL。"""
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


class EmbeddingFactory:
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        """根据 provider 创建对应的 embedding function 适配器。"""
        if config.provider == EmbeddingProvider.BEDROCK:
            import boto3
            from langchain_community.embeddings import BedrockEmbeddings
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=config.aws_region,
                aws_access_key_id=EMBEDDING_CONFIG["aws_access_key_id"] or None,
                aws_secret_access_key=EMBEDDING_CONFIG["aws_secret_access_key"] or None,
            )
            return BedrockEmbeddings(client=bedrock_client, model_id=config.model_name)

        if config.provider == EmbeddingProvider.OPENAI:
            from langchain_community.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(model=config.model_name, openai_api_key=EMBEDDING_CONFIG["openai_api_key"] or None)

        if config.provider == EmbeddingProvider.HUGGINGFACE:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model_name = get_huggingface_model_path(config.model_name)
            return HuggingFaceEmbeddings(model_name=model_name)

        if config.provider == EmbeddingProvider.MODELSCOPE:
            class ModelScopeEmbedding:
                def __init__(self, model_name):
                    """初始化 ModelScope 多模态 embedding pipeline。"""
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks
                    self.model_name = model_name
                    self.pipe = pipeline(Tasks.multi_modal_embedding, model=model_name)

                def embed_query(self, text):
                    """生成单条查询向量，兼容文本和结构化多模态载荷。"""
                    payload = text if isinstance(text, dict) else {"text": text}
                    result = self.pipe(payload)
                    if isinstance(result, dict) and "text_embedding" in result:
                        embedding = result["text_embedding"]
                    elif isinstance(result, list) and len(result) > 0:
                        embedding = result[0]
                    else:
                        embedding = result

                    if isinstance(embedding, np.ndarray):
                        return embedding.tolist()
                    if not isinstance(embedding, list):
                        return [float(x) for x in embedding]
                    return embedding

                def embed_documents(self, texts):
                    """逐条生成文档向量，保持与 LangChain 风格接口兼容。"""
                    return [self.embed_query(text) for text in texts]

            return ModelScopeEmbedding(config.model_name)

        raise ValueError(f"Unsupported embedding provider: {config.provider}")
