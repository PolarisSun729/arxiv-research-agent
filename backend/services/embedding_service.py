import os
import dotenv

dotenv.load_dotenv()

import json
from datetime import datetime
from enum import Enum
import torch
from utils.model_utils import get_huggingface_model_path
import numpy as np
import sys

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    LOCAL = "local"


class EmbeddingConfig:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.aws_region = "ap-southeast-1"


class EmbeddingService:
    LOCAL_EMBEDDING_MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "00-models",
        "Qwen3-VL-Embedding-2B",
    )

    def __init__(self):
        self.embedding_factory = EmbeddingFactory()
        self._local_embedder = None

    @property
    def local_embedder(self):
        if self._local_embedder is None:
            self._local_embedder = self._load_local_qwen3_embedding_model()
        return self._local_embedder

    def _load_local_qwen3_embedding_model(self):
        try:
            if os.path.exists(self.LOCAL_EMBEDDING_MODEL_PATH):
                sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "00-models", "Qwen3-VL-Embedding-2B", "scripts"))
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

        batch_size = 20
        results = []

        if config.provider == EmbeddingProvider.LOCAL:
            if self.local_embedder is None:
                raise ValueError("Local Qwen3-VL-Embedding-2B model not loaded")

            for chunk in chunks:
                embedding_vector = self.create_single_embedding_local(chunk["content"])
                results.append(
                    {
                        "embedding": embedding_vector,
                        "metadata": self._build_embedding_metadata(
                            chunk=chunk,
                            chunk_count=len(chunks),
                            embedding_vector=embedding_vector,
                            provider=config.provider,
                            model="Qwen3-VL-Embedding-2B",
                            filename=filename,
                        ),
                    }
                )
            return results, {}

        embedding_function = self.embedding_factory.create_embedding_function(config)

        if config.provider == EmbeddingProvider.OPENAI:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [chunk.get("content", "") for chunk in batch]
                embedding_vectors = embedding_function.embed_documents(texts)

                for chunk, embedding_vector in zip(batch, embedding_vectors):
                    results.append(
                        {
                            "embedding": embedding_vector,
                            "metadata": self._build_embedding_metadata(
                                chunk=chunk,
                                chunk_count=len(chunks),
                                embedding_vector=embedding_vector,
                                provider=config.provider,
                                model=config.model_name,
                                filename=filename,
                            ),
                        }
                    )
        else:
            for chunk in chunks:
                embedding_vector = embedding_function.embed_query(chunk["content"])
                results.append(
                    {
                        "embedding": embedding_vector,
                        "metadata": self._build_embedding_metadata(
                            chunk=chunk,
                            chunk_count=len(chunks),
                            embedding_vector=embedding_vector,
                            provider=config.provider,
                            model=config.model_name,
                            filename=filename,
                        ),
                    }
                )

        return results, {}

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

    def create_single_embedding(self, text: str, provider: str, model: str) -> list:
        config = EmbeddingConfig(provider=provider, model_name=model)
        embedding_function = self.embedding_factory.create_embedding_function(config)
        return embedding_function.embed_query(text)

    def create_single_embedding_local(self, text: str) -> list:
        if self.local_embedder is None:
            raise ValueError("Local Qwen3-VL-Embedding-2B model not loaded")

        try:
            inputs = [{"text": text}]
            embeddings = self.local_embedder.process(inputs)
            embedding_tensor = embeddings[0]
            if isinstance(embedding_tensor, torch.Tensor):
                embedding = embedding_tensor.to(dtype=torch.float32).cpu().tolist()
            else:
                embedding = np.asarray(embedding_tensor, dtype=np.float32).tolist()
            return embedding
        except Exception as e:
            print(f"Error creating embedding with local model: {str(e)}")
            raise

    def create_single_embedding_modelscope(self, text: str, model: str = "Qwen/Qwen3-VL-Embedding-2B") -> list:
        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            pipe = pipeline(Tasks.multi_modal_embedding, model=model)
            result = pipe({"text": text})
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
                            )
            raise ValueError(f"No matching embedding configuration found for collection: {collection_name}")
        except Exception as e:
            raise ValueError(f"Error getting embedding config: {str(e)}")


class EmbeddingFactory:
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        if config.provider == EmbeddingProvider.BEDROCK:
            import boto3
            from langchain_community.embeddings import BedrockEmbeddings
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=config.aws_region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            return BedrockEmbeddings(client=bedrock_client, model_id=config.model_name)

        if config.provider == EmbeddingProvider.OPENAI:
            from langchain_community.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(model=config.model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))

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
                    result = self.pipe({"text": text})
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
