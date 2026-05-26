import hashlib
import json
from datetime import datetime
import logging
from enum import Enum
from typing import Optional
import os
import base64
import mimetypes
import threading
import torch
from utils.model_utils import get_huggingface_model_path
import numpy as np
import sys
import requests
from utils.config import EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

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
        self.embedding_factory = EmbeddingFactory()
        self._local_embedder = None
        self._embedding_cache: dict[str, list] = {}
        self._embedding_cache_lock = threading.Lock()

    def get_default_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig.from_env()

    @staticmethod
    def _normalize_vector_output(embedding) -> list:
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        if isinstance(embedding, torch.Tensor):
            return embedding.to(dtype=torch.float32).cpu().tolist()
        if isinstance(embedding, list):
            return [float(x) for x in embedding]
        return [float(x) for x in np.asarray(embedding, dtype=np.float32).tolist()]

    @staticmethod
    def build_paper_embedding_text(title: str, abstract: str) -> str:
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
        with self._embedding_cache_lock:
            cached = self._embedding_cache.get(cache_key)
            if cached is None:
                return None
            return [float(value) for value in cached]

    def _set_cached_embedding(self, cache_key: str, embedding: list) -> None:
        with self._embedding_cache_lock:
            self._embedding_cache[cache_key] = [float(value) for value in embedding]

    def _extract_dashscope_embeddings(self, payload: dict) -> list:
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

        vectors.sort(key=lambda item: item[0])
        return [vector for _, vector in vectors]

    def _create_dashscope_embeddings_from_inputs(self, embedding_inputs: list, config: EmbeddingConfig) -> list:
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
        if not vectors:
            raise ValueError(f"DashScope embedding response did not contain vectors: {data}")
        if len(vectors) == len(embedding_inputs):
            return vectors
        if len(embedding_inputs) == 1:
            return [vectors[0]]
        logger.warning(
            "DashScope returned %s vectors for %s inputs; falling back to single-item requests",
            len(vectors),
            len(embedding_inputs),
        )
        return [self._create_dashscope_embedding_from_input(item, config) for item in embedding_inputs]

    def _create_dashscope_embeddings(self, texts: list, config: EmbeddingConfig) -> list:
        embedding_inputs = [{"mode": "text", "text": text} for text in texts]
        return self._create_dashscope_embeddings_from_inputs(embedding_inputs, config)

    def _create_dashscope_embedding(self, text: str, config: EmbeddingConfig) -> list:
        return self._create_dashscope_embeddings([text], config)[0]

    def _create_dashscope_embedding_from_input(self, embedding_input: dict, config: EmbeddingConfig) -> list:
        return self._create_dashscope_embeddings_from_inputs([embedding_input], config)[0]

    @property
    def local_embedder(self):
        if self._local_embedder is None:
            self._local_embedder = self._load_local_qwen3_embedding_model()
        return self._local_embedder

    def _load_local_qwen3_embedding_model(self):
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

        provider_key = str(config.provider).strip().lower()
        batch_size = int(config.batch_size or (10 if provider_key == EmbeddingProvider.DASHSCOPE.value else 20))
        results = []
        prepared_inputs = [
            {
                "chunk": chunk,
                "embedding_input": self.build_embedding_input(chunk, provider_key),
            }
            for chunk in chunks
        ]

        if provider_key == EmbeddingProvider.LOCAL.value:
            if self.local_embedder is None:
                raise ValueError("Local Qwen3-VL-Embedding-2B model not loaded")

            for prepared in prepared_inputs:
                chunk = prepared["chunk"]
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
                        ),
                    }
                )
            return results, {}

        if provider_key == EmbeddingProvider.DASHSCOPE.value:
            if any(item["embedding_input"].get("mode") == "multimodal" for item in prepared_inputs):
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
                            ),
                        }
                    )
            else:
                for i in range(0, len(prepared_inputs), batch_size):
                    batch = prepared_inputs[i : i + batch_size]
                    texts = [item["embedding_input"]["text"] for item in batch]
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
                            ),
                        }
                    )
        else:
            for prepared in prepared_inputs:
                chunk = prepared["chunk"]
                embedding_input = prepared["embedding_input"]
                if embedding_input.get("mode") == "multimodal":
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
                            ),
                        }
                )

        return results, {}

    def build_embedding_input(self, chunk: dict, provider_key: str) -> dict:
        metadata = chunk.get("metadata", {}) or {}
        chunk_type = str(chunk.get("chunk_type") or metadata.get("chunk_type") or "text").strip().lower()
        text = str(
            chunk.get("content")
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
    ) -> dict:
        chunk_metadata = chunk.get("metadata", {})
        # 这里把 chunk 层的页码和来源字段继续向下传，避免 embedding 阶段把结构压扁。
        page_start = int(chunk_metadata.get("page_start", chunk_metadata.get("page_number", 1)))
        page_end = int(chunk_metadata.get("page_end", page_start))
        source = chunk_metadata.get("source", filename)
        chunk_index = int(chunk_metadata.get("chunk_index", chunk_metadata.get("chunk_id", 0)))

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
            "order_index": int(chunk.get("order_index", chunk_metadata.get("order_index", 0)) or 0),
            "word_count": int(chunk_metadata.get("word_count", len(chunk["content"].split()))),
            "total_chunks": int(chunk_count),
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_timestamp": datetime.now().isoformat(),
            "vector_dimension": len(embedding_vector),
            "filename": filename,
        }

    def save_embeddings(self, doc_name: str, embeddings: list) -> str:
        os.makedirs("02-embedded-docs", exist_ok=True)

        first_embedding = embeddings[0]
        provider = first_embedding["metadata"]["embedding_provider"]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

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

        embedding_function = self.embedding_factory.create_embedding_function(config)
        embedding = embedding_function.embed_query(text)
        normalized_embedding = self._normalize_vector_output(embedding)
        self._set_cached_embedding(cache_key, normalized_embedding)
        return normalized_embedding

    def create_single_embedding_dashscope(
        self,
        text: str,
        model: str = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> list:
        config = EmbeddingConfig(
            provider=EmbeddingProvider.DASHSCOPE.value,
            model_name=model or self.DEFAULT_DASHSCOPE_MODEL_NAME,
            api_key=api_key or EMBEDDING_CONFIG["dashscope_api_key"] or EMBEDDING_CONFIG["api_key"],
            base_url=base_url,
            dimension=dimension or self.DEFAULT_DASHSCOPE_DIMENSION,
        )
        return self._create_dashscope_embedding(text, config)

    def create_single_embedding_local(self, text: str) -> list:
        return self.create_single_embedding_local_input({"mode": "text", "text": text})

    def create_single_embedding_modelscope(self, text: str, model: str = "Qwen/Qwen3-VL-Embedding-2B") -> list:
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
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"


class EmbeddingFactory:
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
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
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks
                    self.model_name = model_name
                    self.pipe = pipeline(Tasks.multi_modal_embedding, model=model_name)

                def embed_query(self, text):
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
                    return [self.embed_query(text) for text in texts]

            return ModelScopeEmbedding(config.model_name)

        raise ValueError(f"Unsupported embedding provider: {config.provider}")
