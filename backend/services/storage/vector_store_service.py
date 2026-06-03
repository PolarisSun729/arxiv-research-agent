import os
import warnings
import ast
from datetime import datetime
import json
import re
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)
from utils.config import VectorDBProvider, MILVUS_CONFIG, get_vector_store_runtime_config
from pypinyin import lazy_pinyin, Style

logger = logging.getLogger(__name__)

VECTOR_STORE_CONFIG = get_vector_store_runtime_config()

warnings.filterwarnings(
    "ignore",
    message=r".*ORM-style PyMilvus API.*",
    category=DeprecationWarning,
)

CONTENT_MAX_LENGTH = VECTOR_STORE_CONFIG["content_max_length"]
RERANK_TEXT_MAX_LENGTH = VECTOR_STORE_CONFIG["rerank_text_max_length"]
ASSET_PATH_MAX_LENGTH = VECTOR_STORE_CONFIG["asset_path_max_length"]
ASSET_SUMMARY_MAX_LENGTH = VECTOR_STORE_CONFIG["asset_summary_max_length"]
ASSET_PREVIEW_MAX_LENGTH = VECTOR_STORE_CONFIG["asset_preview_max_length"]

COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_collection_name(collection_name: str) -> str:
    name = (collection_name or "").strip()
    if not name:
        return "collection"
    name = name.replace("-", "_").replace(".", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "collection"
    if not name[0].isalpha() and name[0] != "_":
        name = f"_{name}"
    return name


def is_valid_collection_name(collection_name: str) -> bool:
    return bool(COLLECTION_NAME_PATTERN.match(collection_name or ""))

class VectorDBConfig:
    """
    向量数据库配置类，用于存储和管理向量数据库的配置信息。
    """
    def __init__(self, provider: str, index_mode: str):
        """
        初始化向量数据库配置。

        参数:
            provider: 向量数据库提供商名称
            index_mode: 索引模式
        """
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = MILVUS_CONFIG["uri"]

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        根据索引模式获取 Milvus 索引类型。

        参数:
            index_mode: 索引模式

        返回:
            对应的 Milvus 索引类型
        """
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        根据索引模式获取 Milvus 索引参数。

        参数:
            index_mode: 索引模式

        返回:
            对应的 Milvus 索引参数字典
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})

class VectorStoreService:
    """
    向量存储服务类，提供向量数据的索引、查询和管理功能。
    """
    def __init__(self):
        """
        初始化向量存储服务。
        """
        self.initialized_dbs = {}
        # 确保存储目录存在
        os.makedirs("03-vector-store", exist_ok=True)

    def _get_client(self) -> MilvusClient:
        return MilvusClient(uri=MILVUS_CONFIG["uri"])

    def resolve_collection_name(self, collection_name: str) -> str:
        """
        将任意 collection 名称规范化为 Milvus 可接受的形式。
        """
        return normalize_collection_name(collection_name)

    def collection_exists(self, provider: str, collection_name: str) -> bool:
        if provider == VectorDBProvider.MILVUS:
            client = None
            try:
                client = self._get_client()
                resolved = self.resolve_collection_name(collection_name)
                return client.has_collection(collection_name=resolved)
            finally:
                if client is not None:
                    del client
        return False
    
    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        从配置对象获取 Milvus 索引类型。

        参数:
            config: 向量数据库配置对象

        返回:
            Milvus 索引类型
        """
        return config._get_milvus_index_type(config.index_mode)
    
    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        从配置对象获取 Milvus 索引参数。

        参数:
            config: 向量数据库配置对象

        返回:
            Milvus 索引参数字典
        """
        return config._get_milvus_index_params(config.index_mode)
    
    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        将 embedding 文件索引到向量数据库。

        参数:
            embedding_file: embedding 文件路径
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        start_time = datetime.now()
        
                # 读取 embedding 文件
        embeddings_data = self._load_embeddings(embedding_file)
        
        # 根据提供商选择索引方法
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)
        else:
            raise ValueError(f"Unsupported vector database provider: {config.provider}")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings_data["embeddings"]),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A")
        }
    
    def _load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """
        加载 embedding 文件，返回配置和 embeddings。

        参数:
            file_path: embedding 文件路径

        返回:
            包含 embedding 数据和顶层配置的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")
                
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                # 返回完整数据，包括顶层配置
                logger.info(f"Found {len(data['embeddings'])} embeddings")
                return data
                
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise
    
    def _index_to_milvus(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        将 embedding 数据索引到 Milvus。

        参数:
            embeddings_data: embedding 数据
            config: 向量数据库配置对象

        返回:
            索引结果信息字典
        """
        try:
            # 使用 filename 作为 collection 名称前缀
            filename = embeddings_data.get("filename", "")
            # 如果有 .pdf 后缀则去掉
            base_name = filename.replace('.pdf', '') if filename else "doc"
            
            # 将中文文件名转换为拼音，避免 Milvus collection 名称非法
            base_name = ''.join(lazy_pinyin(base_name, style=Style.NORMAL))
            
            # 把连字符和点号替换为下划线
            base_name = base_name.replace('-', '_').replace('.', '_')
            
            # 确保 collection 名称以字母或下划线开头
            if not base_name[0].isalpha() and base_name[0] != '_':
                base_name = f"_{base_name}"
            
            # 生成 collection 名称
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = normalize_collection_name(f"{base_name}_{embedding_provider}_{timestamp}")
            
            # 连接 Milvus
            connections.connect(
                alias="default", 
                uri=config.milvus_uri
            )
            
            # 从配置中读取向量维度
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")
            
            logger.info(f"Creating collection with dimension: {vector_dim}")
            
            # 定义 collection 字段
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": CONTENT_MAX_LENGTH},
                {"name": "rerank_text", "dtype": "VARCHAR", "max_length": RERANK_TEXT_MAX_LENGTH},
                {"name": "chunk_type", "dtype": "VARCHAR", "max_length": 24},
                {"name": "asset_kind", "dtype": "VARCHAR", "max_length": 24},
                {"name": "asset_path", "dtype": "VARCHAR", "max_length": ASSET_PATH_MAX_LENGTH},
                {"name": "asset_abs_path", "dtype": "VARCHAR", "max_length": ASSET_PATH_MAX_LENGTH},
                {"name": "asset_summary", "dtype": "VARCHAR", "max_length": ASSET_SUMMARY_MAX_LENGTH},
                {"name": "asset_preview_text", "dtype": "VARCHAR", "max_length": ASSET_PREVIEW_MAX_LENGTH},
                {"name": "asset_caption", "dtype": "VARCHAR", "max_length": ASSET_SUMMARY_MAX_LENGTH},
                {"name": "asset_rows", "dtype": "INT64"},
                {"name": "asset_columns", "dtype": "INT64"},
                {"name": "order_index", "dtype": "INT64"},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                # 单独保留 source，方便 QA / 检索时直接回溯到原始 PDF 文件名
                {"name": "source", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "chunk_index", "dtype": "INT64"},
                {"name": "parent_chunk_id", "dtype": "INT64"},
                {"name": "original_chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                # page_start / page_end 预留给未来跨页 chunk 使用，目前页内切分时两者相同
                {"name": "page_start", "dtype": "INT64"},
                {"name": "page_end", "dtype": "INT64"},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                {"name": "subchunk_index", "dtype": "INT64"},
                {"name": "subchunk_count", "dtype": "INT64"},
                {"name": "subchunk_label", "dtype": "VARCHAR", "max_length": 64},
                {"name": "content_part_index", "dtype": "INT64"},
                {"name": "content_part_count", "dtype": "INT64"},
                {"name": "content_part_label", "dtype": "VARCHAR", "max_length": 32},
                # {"name": "chunking_method", "dtype": "VARCHAR", "max_length": 50},
                {"name": "section_path", "dtype": "VARCHAR", "max_length": 1024},
                {"name": "embedding_provider", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_model", "dtype": "VARCHAR", "max_length": 50},
                {"name": "embedding_timestamp", "dtype": "VARCHAR", "max_length": 50},
                {
                    "name": "vector",
                    "dtype": "FLOAT_VECTOR",
                    "dim": vector_dim,
                    "params": self._get_milvus_index_params(config)
                }
            ]
            
            # 准备写入 collection 的实体数据
            entities = []
            for emb in embeddings_data["embeddings"]:
                metadata = emb["metadata"]
                content = str(metadata.get("content", ""))
                rerank_text = str(metadata.get("rerank_text", ""))
                parent_chunk_id = int(metadata.get("parent_chunk_id", metadata.get("chunk_id", metadata.get("chunk_index", 0))))
                original_chunk_id = int(metadata.get("original_chunk_id", parent_chunk_id))
                subchunk_index = int(metadata.get("subchunk_index", 1))
                subchunk_count = int(metadata.get("subchunk_count", 1))
                subchunk_label = str(
                    metadata.get("subchunk_label", f"chunk {parent_chunk_id} part {subchunk_index}/{subchunk_count}")
                )

                # 入库前把页码、来源和 chunk 序号统一落到向量库字段里，方便后续检索回溯
                page_start = int(metadata.get("page_start", metadata.get("page_number", 0)))
                page_end = int(metadata.get("page_end", page_start))
                source = str(metadata.get("source", embeddings_data.get("filename", "")))
                chunk_index = int(metadata.get("chunk_index", metadata.get("chunk_id", 0)))
                total_chunks = int(metadata.get("total_chunks", 0))
                base_chunk_id = int(metadata.get("chunk_id", chunk_index))
                entity = {
                    "content": content,
                    "rerank_text": rerank_text,
                    "chunk_type": str(metadata.get("chunk_type", "text") or "text"),
                    "asset_kind": str(metadata.get("asset_kind", "") or ""),
                    "asset_path": str(metadata.get("asset_path", "") or ""),
                    "asset_abs_path": str(metadata.get("asset_abs_path", "") or ""),
                    "asset_summary": str(metadata.get("asset_summary", "") or ""),
                    "asset_preview_text": str(metadata.get("asset_preview_text", "") or ""),
                    "asset_caption": str(metadata.get("asset_caption", "") or ""),
                    "asset_rows": int(metadata.get("asset_rows", 0) or 0),
                    "asset_columns": int(metadata.get("asset_columns", 0) or 0),
                    "order_index": int(metadata.get("order_index", 0) or 0),
                    "document_name": embeddings_data.get("filename", ""),
                    "source": source,
                    "chunk_id": base_chunk_id,
                    "chunk_index": chunk_index,
                    "parent_chunk_id": parent_chunk_id,
                    "original_chunk_id": original_chunk_id,
                    "total_chunks": total_chunks,
                    "word_count": int(metadata.get("word_count", 0)),
                    "page_number": str(page_start),
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_range": str(metadata.get("page_range", f"{page_start}-{page_end}")),
                    "subchunk_index": subchunk_index,
                    "subchunk_count": subchunk_count,
                    "subchunk_label": subchunk_label,
                    "content_part_index": int(metadata.get("content_part_index", subchunk_index)),
                    "content_part_count": int(metadata.get("content_part_count", subchunk_count)),
                    "content_part_label": str(
                        metadata.get("content_part_label", subchunk_label)
                    ),
                    "section_path": str(metadata.get("section_path", "")),
                    # "chunking_method": str(metadata.get("chunking_method", "")),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),
                    "embedding_model": embeddings_data.get("embedding_model", ""),
                    "embedding_timestamp": str(metadata.get("embedding_timestamp", "")),
                    "vector": [float(x) for x in emb.get("embedding", [])]
                }
                entities.append(entity)
            
            logger.info(f"Creating Milvus collection: {collection_name}")
            
            # 创建 collection
            # field_schemas = [
            #     FieldSchema(name=field["name"], 
            #                dtype=getattr(DataType, field["dtype"]),
            #                is_primary="is_primary" in field and field["is_primary"],
            #                auto_id="auto_id" in field and field["auto_id"],
            #                max_length=field.get("max_length"),
            #                dim=field.get("dim"),
            #                params=field.get("params"))
            #     for field in fields
            # ]

            field_schemas = []
            for field in fields:
                extra_params = {}
                if field.get('max_length') is not None:
                    extra_params['max_length'] = field['max_length']
                if field.get('dim') is not None:
                    extra_params['dim'] = field['dim']
                if field.get('params') is not None:
                    extra_params['params'] = field['params']
                field_schema = FieldSchema(
                    name=field["name"], 
                    dtype=getattr(DataType, field["dtype"]),
                    is_primary=field.get("is_primary", False),
                    auto_id=field.get("auto_id", False),
                    **extra_params
                )
                field_schemas.append(field_schema)

            schema = CollectionSchema(fields=field_schemas, description=f"Collection for {collection_name}")
            collection = Collection(name=collection_name, schema=schema)
            
            # 插入数据
            logger.info(f"Inserting {len(entities)} vectors")
            insertable_fields = [field.name for field in collection.schema.fields if not getattr(field, "auto_id", False)]
            normalized_entities = [
                {key: value for key, value in entity.items() if key in insertable_fields}
                for entity in entities
            ]
            self._validate_varchar_lengths(normalized_entities, collection.schema.fields)
            insert_columns = [
                [entity.get(field_name) for entity in normalized_entities]
                for field_name in insertable_fields
            ]
            insert_result = collection.insert(insert_columns)
            collection.flush()
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": self._get_milvus_index_type(config),
                "params": self._get_milvus_index_params(config)
            }
            collection.create_index(field_name="vector", index_params=index_params)
            collection.load()
            
            return {
                "index_size": len(insert_result.primary_keys),
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            raise
        
        finally:
            connections.disconnect("default")

    def _validate_varchar_lengths(self, entities: List[Dict[str, Any]], fields: List[Any]) -> None:
        varchar_type = getattr(DataType.VARCHAR, "value", DataType.VARCHAR)
        varchar_limits = {
            (field.get("name") if isinstance(field, dict) else getattr(field, "name", "")): (
                (field.get("params", {}) or {}).get("max_length")
                if isinstance(field, dict)
                else getattr(field, "max_length", None)
            )
            for field in fields
            if (
                (field.get("type") if isinstance(field, dict) else getattr(field, "dtype", None)) == varchar_type
            )
        }

        violations = []
        for entity in entities:
            chunk_ref = entity.get(
                "parent_chunk_id",
                entity.get("original_chunk_id", entity.get("chunk_id", entity.get("chunk_index", 0))),
            )
            for field_name, max_length in varchar_limits.items():
                if not max_length:
                    continue
                value = entity.get(field_name)
                if isinstance(value, str) and len(value.encode("utf-8")) > max_length:
                    violations.append(
                        f"{field_name} bytes={len(value.encode('utf-8'))} > {max_length} for chunk {chunk_ref}"
                    )

        if violations:
            raise ValueError(
                "Milvus varchar validation failed: " + "; ".join(violations[:5])
            )

    def _truncate_utf8_text(self, value: str, max_bytes: Optional[int]) -> str:
        if not isinstance(value, str) or not max_bytes or max_bytes <= 0:
            return value

        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value

        truncated = encoded[:max_bytes]
        while truncated:
            try:
                return truncated.decode("utf-8")
            except UnicodeDecodeError:
                truncated = truncated[:-1]
        return ""

    def _truncate_entities_to_varchar_limits(self, entities: List[Dict[str, Any]], fields: List[Any]) -> List[Dict[str, Any]]:
        varchar_type = getattr(DataType.VARCHAR, "value", DataType.VARCHAR)
        varchar_limits = {
            (field.get("name") if isinstance(field, dict) else getattr(field, "name", "")): (
                (field.get("params", {}) or {}).get("max_length")
                if isinstance(field, dict)
                else getattr(field, "max_length", None)
            )
            for field in fields
            if (
                (field.get("type") if isinstance(field, dict) else getattr(field, "dtype", None)) == varchar_type
            )
        }

        normalized_entities: List[Dict[str, Any]] = []
        for entity in entities:
            normalized_entity = dict(entity)
            chunk_ref = entity.get(
                "parent_chunk_id",
                entity.get("original_chunk_id", entity.get("chunk_id", entity.get("chunk_index", 0))),
            )
            for field_name, max_length in varchar_limits.items():
                value = normalized_entity.get(field_name)
                if not max_length or not isinstance(value, str):
                    continue

                truncated_value = self._truncate_utf8_text(value, int(max_length))
                if truncated_value != value:
                    logger.warning(
                        "Truncated Milvus varchar field '%s' for chunk %s from %s to %s bytes",
                        field_name,
                        chunk_ref,
                        len(value.encode("utf-8")),
                        len(truncated_value.encode("utf-8")),
                    )
                    normalized_entity[field_name] = truncated_value
            normalized_entities.append(normalized_entity)

        return normalized_entities

    def list_collections(self, provider: str) -> List[str]:
        """
        列出指定提供商的所有 collection。
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                client = self._get_client()
                return client.list_collections()
            except Exception:
                logger.exception("Error listing Milvus collections")
                raise
        return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        删除指定 collection。
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                client = self._get_client()
                resolved_name = self.resolve_collection_name(collection_name)
                if client.has_collection(collection_name=resolved_name):
                    client.drop_collection(collection_name=resolved_name)
                    return True
                return False
            except Exception:
                logger.exception("Error deleting Milvus collection: %s", collection_name)
                raise
        return False

    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """
        获取指定 collection 的信息。
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                client = self._get_client()
                resolved_name = self.resolve_collection_name(collection_name)
                if not client.has_collection(collection_name=resolved_name):
                    return {}
                collection_desc = client.describe_collection(collection_name=resolved_name)
                collection_stats = client.get_collection_stats(collection_name=resolved_name)
                return {
                    "name": resolved_name,
                    "num_entities": int(collection_stats.get("row_count", 0) or 0),
                    "schema": collection_desc,
                }
            except Exception:
                logger.exception("Error getting Milvus collection info: %s", collection_name)
                raise
        return {}

    def insert_single_embedding(self, collection_name: str, embedding: List[float], metadata: Dict[str, Any]) -> int:
        """
        将单个 embedding 插入到指定 collection。
        """
        try:
            return self.insert_embeddings(collection_name, [{"embedding": embedding, "metadata": metadata}])
        except Exception as e:
            logger.error(f"Error inserting single embedding: {str(e)}")
            raise

    def insert_embeddings(self, collection_name: str, items: List[Dict[str, Any]]) -> int:
        """
        批量插入多个 embedding 到指定 collection。
        """
        if not items:
            return 0

        try:
            client = self._get_client()
            resolved_name = normalize_collection_name(collection_name)

            embeddings = [[float(value) for value in item.get("embedding", [])] for item in items]
            vector_dim = len(embeddings[0]) if embeddings and embeddings[0] else 0
            if not vector_dim:
                raise ValueError("Cannot insert empty embeddings")

            if client.has_collection(collection_name=resolved_name):
                collection_desc = client.describe_collection(collection_name=resolved_name)
                vector_field = next((field for field in collection_desc.get("fields", []) if field.get("name") == "vector"), None)
                existing_dim = None
                if vector_field is not None:
                    params = vector_field.get("params", {}) or {}
                    existing_dim = params.get("dim")
                if existing_dim and int(existing_dim) != vector_dim:
                    raise ValueError(
                        f"Collection '{resolved_name}' already uses vector dimension {existing_dim}, "
                        f"but the new embeddings have dimension {vector_dim}. "
                        "Rebuild the collection or keep the embedding model/dimension consistent."
                    )
            else:
                schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=CONTENT_MAX_LENGTH)
                schema.add_field(field_name="arxiv_id", datatype=DataType.VARCHAR, max_length=100)
                schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000)
                schema.add_field(field_name="authors", datatype=DataType.VARCHAR, max_length=2000)
                schema.add_field(field_name="categories", datatype=DataType.VARCHAR, max_length=500)
                schema.add_field(field_name="published_date", datatype=DataType.VARCHAR, max_length=50)
                schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=500)
                schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=100)
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)
                client.create_collection(collection_name=resolved_name, schema=schema)

                index_params = MilvusClient.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type="FLAT",
                    metric_type="COSINE",
                )
                client.create_index(collection_name=resolved_name, index_params=index_params)

            entities = []
            for embedding, metadata in zip(embeddings, [item.get("metadata", {}) for item in items]):
                entity = {
                    "content": str(metadata.get("content", "")),
                    "arxiv_id": str(metadata.get("arxiv_id", "")),
                    "title": str(metadata.get("title", "")),
                    "authors": str(metadata.get("authors", "")),
                    "categories": str(metadata.get("categories", "")),
                    "published_date": str(metadata.get("published_date", "")),
                    "url": str(metadata.get("url", "")),
                    "embedding_model": str(metadata.get("embedding_model", "")),
                    "vector": [float(x) for x in embedding],
                }
                entities.append(entity)

            collection_desc = client.describe_collection(collection_name=resolved_name)
            normalized_entities = self._truncate_entities_to_varchar_limits(entities, collection_desc.get("fields", []))
            self._validate_varchar_lengths(normalized_entities, collection_desc.get("fields", []))
            insert_result = client.insert(collection_name=resolved_name, data=normalized_entities)
            client.flush(collection_name=resolved_name)
            client.load_collection(collection_name=resolved_name)

            if isinstance(insert_result, dict):
                return int(insert_result.get("insert_count", len(entities)))
            if hasattr(insert_result, "insert_count"):
                return int(getattr(insert_result, "insert_count", len(entities)))
            if isinstance(insert_result, list):
                return len(insert_result)
            return len(entities)
        except Exception as e:
            logger.error(f"Error inserting embeddings batch: {str(e)}")
            raise

    def search_similar_vectors(self, collection_name: str, query_vector: List[float], top_k: int = 10, filter_arxiv_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        检索 chunk 级向量，并把页码等溯源信息一起返回。
        这样生成答案时就能直接带出来源页码，而不是只给一段不知道出处的文本。
        """
        try:
            client = self._get_client()
            resolved_name = self.resolve_collection_name(collection_name)

            if not client.has_collection(collection_name=resolved_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []

            collection_desc = client.describe_collection(collection_name=resolved_name)
            field_names = {field.get("name") for field in collection_desc.get("fields", [])}
            candidate_fields = [
                "content",
                "rerank_text",
                "chunk_type",
                "asset_kind",
                "asset_path",
                "asset_abs_path",
                "asset_summary",
                "asset_preview_text",
                "asset_caption",
                "asset_rows",
                "asset_columns",
                "order_index",
                "document_name",
                "source",
                "chunk_id",
                "chunk_index",
                "parent_chunk_id",
                "original_chunk_id",
                "total_chunks",
                "word_count",
                "page_number",
                "page_start",
                "page_end",
                "page_range",
                "subchunk_index",
                "subchunk_count",
                "subchunk_label",
                "content_part_index",
                "content_part_count",
                "content_part_label",
                "section_path",
                "section_title",
                "section_level",
                "section_part_index",
                "section_part_count",
                "embedding_provider",
                "embedding_model",
                "embedding_timestamp",
                "arxiv_id",
                "title",
                "authors",
                "categories",
                "published_date",
                "url",
            ]
            output_fields = [field for field in candidate_fields if field in field_names]

            expr = None
            if filter_arxiv_ids and "arxiv_id" in field_names:
                quoted_ids = ", ".join(f'"{arxiv_id}"' for arxiv_id in filter_arxiv_ids if arxiv_id)
                if quoted_ids:
                    expr = f"arxiv_id not in [{quoted_ids}]"

            results = client.search(
                collection_name=resolved_name,
                data=[query_vector],
                filter=expr or "",
                limit=top_k,
                anns_field="vector",
                output_fields=output_fields,
                search_params={"metric_type": "COSINE", "params": {}},
            )

            similar_vectors = [
                self._build_chunk_payload(
                    entity=hit.get("entity", hit),
                    score=float(hit.get("score")) if hit.get("score") is not None else None,
                    distance=float(hit.get("distance")) if hit.get("distance") is not None else None,
                )
                for hit in results[0]
            ]

            return similar_vectors

        except Exception as e:
            logger.error(f"Error searching similar vectors: {str(e)}")
            raise

    def search_similar_papers(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_arxiv_ids: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索论文级向量，并返回论文展示所需的顶层字段。
        这个方法专门用于 arxiv_paper_embeddings 这类论文集合，不沿用 chunk 级 payload。
        """
        try:
            client = self._get_client()
            resolved_name = self.resolve_collection_name(collection_name)

            if not client.has_collection(collection_name=resolved_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []

            collection_desc = client.describe_collection(collection_name=resolved_name)
            field_names = {field.get("name") for field in collection_desc.get("fields", [])}
            candidate_fields = [
                "arxiv_id",
                "title",
                "authors",
                "abstract",
                "categories",
                "published_date",
                "url",
                "embedding_model",
                "content",
            ]
            output_fields = [field for field in candidate_fields if field in field_names]

            expr = None
            if filter_arxiv_ids and "arxiv_id" in field_names:
                quoted_ids = ", ".join(f'"{arxiv_id}"' for arxiv_id in filter_arxiv_ids if arxiv_id)
                if quoted_ids:
                    expr = f"arxiv_id not in [{quoted_ids}]"

            results = client.search(
                collection_name=resolved_name,
                data=[query_vector],
                filter=expr or "",
                limit=top_k,
                anns_field="vector",
                output_fields=output_fields,
                search_params={"metric_type": "COSINE", "params": {}},
            )

            return [
                self._build_paper_payload(
                    entity=hit.get("entity", hit),
                    score=float(hit.get("score")) if hit.get("score") is not None else None,
                    distance=float(hit.get("distance")) if hit.get("distance") is not None else None,
                )
                for hit in results[0]
            ]

        except Exception as e:
            logger.error(f"Error searching similar papers: {str(e)}")
            raise

    def get_paper_embeddings_by_arxiv_ids(
        self,
        collection_name: str,
        arxiv_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        按 arxiv_id 批量读取论文级向量和元数据。
        主要用于复用已经写入 Milvus 的论文 embedding，避免重复重算。
        """
        try:
            normalized_ids = [str(arxiv_id).strip() for arxiv_id in arxiv_ids if str(arxiv_id).strip()]
            if not normalized_ids:
                return []

            client = self._get_client()
            resolved_name = self.resolve_collection_name(collection_name)

            if not client.has_collection(collection_name=resolved_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []

            collection_desc = client.describe_collection(collection_name=resolved_name)
            field_names = {field.get("name") for field in collection_desc.get("fields", [])}
            if "arxiv_id" not in field_names or "vector" not in field_names:
                logger.warning(
                    "Collection %s does not expose required fields arxiv_id/vector",
                    resolved_name,
                )
                return []

            quoted_ids = ", ".join(f'"{arxiv_id}"' for arxiv_id in normalized_ids)
            expr = f"arxiv_id in [{quoted_ids}]"

            candidate_fields = [
                "id",
                "arxiv_id",
                "title",
                "authors",
                "categories",
                "published_date",
                "url",
                "embedding_model",
                "vector",
            ]
            output_fields = [field for field in candidate_fields if field in field_names]

            results = client.query(
                collection_name=resolved_name,
                filter=expr,
                output_fields=output_fields,
                limit=len(normalized_ids),
            )

            return [self._build_paper_embedding_payload(entity=row) for row in results]
        except Exception as e:
            logger.error(f"Error reading paper embeddings by arxiv ids: {str(e)}")
            raise

    def get_all_chunks(self, collection_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            client = self._get_client()
            resolved_name = self.resolve_collection_name(collection_name)

            if not client.has_collection(collection_name=resolved_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []

            collection_desc = client.describe_collection(collection_name=resolved_name)
            field_names = {field.get("name") for field in collection_desc.get("fields", [])}
            candidate_fields = [
                "id",
                "content",
                "rerank_text",
                "chunk_type",
                "asset_kind",
                "asset_path",
                "asset_abs_path",
                "asset_summary",
                "asset_preview_text",
                "asset_caption",
                "asset_rows",
                "asset_columns",
                "order_index",
                "document_name",
                "source",
                "chunk_id",
                "chunk_index",
                "parent_chunk_id",
                "original_chunk_id",
                "total_chunks",
                "word_count",
                "page_number",
                "page_start",
                "page_end",
                "page_range",
                "subchunk_index",
                "subchunk_count",
                "subchunk_label",
                "content_part_index",
                "content_part_count",
                "content_part_label",
                "section_path",
                "section_title",
                "section_level",
                "section_part_index",
                "section_part_count",
                "embedding_provider",
                "embedding_model",
                "embedding_timestamp",
                "arxiv_id",
                "title",
                "authors",
                "categories",
                "published_date",
                "url",
            ]
            output_fields = [field for field in candidate_fields if field in field_names]
            query_limit = limit if limit is not None else 10000
            if query_limit <= 0:
                logger.warning(
                    f"Collection {collection_name} has no entities or requested limit is non-positive: {query_limit}"
                )
                return []
            entities = client.query(
                collection_name=resolved_name,
                filter="id >= 0",
                output_fields=output_fields,
                limit=query_limit,
            )
            return [self._build_chunk_payload(entity=entity, score=None, distance=None) for entity in entities]
        except Exception as e:
            logger.error(f"Error reading collection chunks: {str(e)}")
            raise

    def _build_chunk_payload(self, entity: Any, score: Optional[float], distance: Optional[float]) -> Dict[str, Any]:
        if entity is None:
            return {
                "text": "",
                "content": "",
                "score": score,
                "distance": distance,
                "metadata": {},
            }

        reader = entity if isinstance(entity, dict) else None
        getter = (lambda key, default=None: reader.get(key, default)) if reader is not None else (lambda key, default=None: getattr(entity, key, default))

        content = getter("content", "") or ""
        rerank_text = getter("rerank_text", "") or ""
        chunk_type = getter("chunk_type", "text") or "text"
        asset_kind = getter("asset_kind", "") or ""
        asset_path = getter("asset_path", "") or ""
        asset_abs_path = getter("asset_abs_path", "") or ""
        asset_summary = getter("asset_summary", "") or ""
        asset_preview_text = getter("asset_preview_text", "") or ""
        asset_caption = getter("asset_caption", "") or ""
        asset_rows = getter("asset_rows", None)
        asset_columns = getter("asset_columns", None)
        order_index = getter("order_index", None)
        source = getter("source", "") or ""
        document_name = getter("document_name", "") or ""
        page_start = getter("page_start", None)
        page_end = getter("page_end", None)
        page_number = getter("page_number", "") or ""
        page_range = getter("page_range", "") or ""
        chunk_index = getter("chunk_index", None)
        chunk_id = getter("chunk_id", None)
        parent_chunk_id = getter("parent_chunk_id", None)
        original_chunk_id = getter("original_chunk_id", None)
        subchunk_index = getter("subchunk_index", None)
        subchunk_count = getter("subchunk_count", None)
        subchunk_label = getter("subchunk_label", "") or ""
        content_part_index = getter("content_part_index", None)
        content_part_count = getter("content_part_count", None)
        content_part_label = getter("content_part_label", "") or ""
        section_path = getter("section_path", "") or ""
        section_title = getter("section_title", "") or ""
        section_level = getter("section_level", None)
        section_part_index = getter("section_part_index", None)
        section_part_count = getter("section_part_count", None)

        if not source:
            source = document_name or getter("arxiv_id", "") or ""
        if not page_range and page_start is not None and page_end is not None:
            page_range = f"{page_start}-{page_end}"
        if not page_number and page_start is not None:
            page_number = str(page_start)
        if chunk_index is None and chunk_id is not None:
            chunk_index = chunk_id

        metadata = {
            "source": source,
            "document_name": document_name or source,
            "chunk_id": int(chunk_id or 0),
            "chunk_index": int(chunk_index or 0),
            "parent_chunk_id": int(parent_chunk_id or 0),
            "original_chunk_id": int(original_chunk_id or 0),
            "total_chunks": int(getter("total_chunks", 0) or 0),
            "word_count": int(getter("word_count", 0) or 0),
            "rerank_text": str(rerank_text or ""),
            "chunk_type": str(chunk_type or "text"),
            "asset_kind": str(asset_kind or ""),
            "asset_path": str(asset_path or ""),
            "asset_abs_path": str(asset_abs_path or ""),
            "asset_summary": str(asset_summary or ""),
            "asset_preview_text": str(asset_preview_text or ""),
            "asset_caption": str(asset_caption or ""),
            "asset_rows": int(asset_rows or 0),
            "asset_columns": int(asset_columns or 0),
            "order_index": int(order_index or 0),
            "page_number": str(page_number or ""),
            "page_start": int(page_start or 0) if page_start is not None else None,
            "page_end": int(page_end or 0) if page_end is not None else None,
            "page_range": str(page_range or ""),
            "subchunk_index": int(subchunk_index or 0),
            "subchunk_count": int(subchunk_count or 0),
            "subchunk_label": str(subchunk_label or ""),
            "content_part_index": int(content_part_index or 0),
            "content_part_count": int(content_part_count or 0),
            "content_part_label": str(content_part_label or ""),
            "section_path": str(section_path or ""),
            "section_title": str(section_title or ""),
            "section_level": int(section_level or 0) if section_level is not None else None,
            "section_part_index": int(section_part_index or 0) if section_part_index is not None else None,
            "section_part_count": int(section_part_count or 0) if section_part_count is not None else None,
            "embedding_provider": getter("embedding_provider", "") or "",
            "embedding_model": getter("embedding_model", "") or "",
            "embedding_timestamp": getter("embedding_timestamp", "") or "",
        }

        payload = {
            "text": content,
            "content": content,
            "rerank_text": metadata["rerank_text"],
            "chunk_type": metadata["chunk_type"],
            "asset_kind": metadata["asset_kind"],
            "asset_path": metadata["asset_path"],
            "asset_abs_path": metadata["asset_abs_path"],
            "asset_summary": metadata["asset_summary"],
            "asset_preview_text": metadata["asset_preview_text"],
            "asset_caption": metadata["asset_caption"],
            "asset_rows": metadata["asset_rows"],
            "asset_columns": metadata["asset_columns"],
            "order_index": metadata["order_index"],
            "score": score,
            "distance": distance,
            "source": source,
            "document_name": document_name or source,
            "chunk_id": metadata["chunk_id"],
            "chunk_index": metadata["chunk_index"],
            "parent_chunk_id": metadata["parent_chunk_id"],
            "original_chunk_id": metadata["original_chunk_id"],
            "page_number": metadata["page_number"],
            "page_start": metadata["page_start"],
            "page_end": metadata["page_end"],
            "page_range": metadata["page_range"],
            "subchunk_index": metadata["subchunk_index"],
            "subchunk_count": metadata["subchunk_count"],
            "subchunk_label": metadata["subchunk_label"],
            "content_part_index": metadata["content_part_index"],
            "content_part_count": metadata["content_part_count"],
            "content_part_label": metadata["content_part_label"],
            "section_path": metadata["section_path"],
            "section_title": metadata["section_title"],
            "section_level": metadata["section_level"],
            "section_part_index": metadata["section_part_index"],
            "section_part_count": metadata["section_part_count"],
            "metadata": metadata,
        }
        return payload

    def _build_paper_payload(self, entity: Any, score: Optional[float], distance: Optional[float]) -> Dict[str, Any]:
        if entity is None:
            return {
                "arxiv_id": "",
                "title": "",
                "authors": [],
                "abstract": "",
                "categories": [],
                "published_date": "",
                "url": "",
                "similarity_score": score,
                "score": score,
                "distance": distance,
                "metadata": {},
            }

        reader = entity if isinstance(entity, dict) else None
        getter = (lambda key, default=None: reader.get(key, default)) if reader is not None else (lambda key, default=None: getattr(entity, key, default))

        authors_raw = getter("authors", "") or ""
        categories_raw = getter("categories", "") or ""
        abstract = getter("abstract", "") or getter("content", "") or ""
        published_date = getter("published_date", "") or ""

        def _split_values(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if not value:
                return []
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("[") or stripped.startswith("("):
                    try:
                        parsed = ast.literal_eval(stripped)
                        if isinstance(parsed, (list, tuple)):
                            return [str(item).strip() for item in parsed if str(item).strip()]
                    except (ValueError, SyntaxError):
                        pass
            normalized = str(value).replace(";", ",").replace("|", ",")
            parts = []
            for token in normalized.replace("\n", " ").split():
                if "," in token:
                    parts.extend(token.split(","))
                else:
                    parts.append(token)
            parts = [part.strip() for part in parts]
            return [part for part in parts if part]

        payload = {
            "arxiv_id": str(getter("arxiv_id", "") or ""),
            "title": str(getter("title", "") or ""),
            "authors": _split_values(authors_raw),
            "abstract": str(abstract),
            "categories": _split_values(categories_raw),
            "published_date": str(published_date),
            "url": str(getter("url", "") or ""),
            "similarity_score": score,
            "score": score,
            "distance": distance,
            "metadata": {
                "embedding_model": str(getter("embedding_model", "") or ""),
            },
        }
        return payload

    def _build_paper_embedding_payload(self, entity: Any) -> Dict[str, Any]:
        reader = entity if isinstance(entity, dict) else None
        getter = (lambda key, default=None: reader.get(key, default)) if reader is not None else (lambda key, default=None: getattr(entity, key, default))

        vector_value = getter("vector", []) or []
        if vector_value and isinstance(vector_value[0], (list, tuple)):
            vector_value = vector_value[0]

        return {
            "embedding_id": int(getter("id", 0) or 0),
            "arxiv_id": str(getter("arxiv_id", "") or ""),
            "title": str(getter("title", "") or ""),
            "authors": getter("authors", []) or [],
            "categories": getter("categories", []) or [],
            "published_date": str(getter("published_date", "") or ""),
            "url": str(getter("url", "") or ""),
            "embedding_model": str(getter("embedding_model", "") or ""),
            "vector": [float(value) for value in vector_value] if vector_value else [],
        }
