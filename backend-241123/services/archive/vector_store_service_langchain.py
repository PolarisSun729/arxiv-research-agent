import os
from datetime import datetime
from enum import Enum
import json
from typing import List, Dict, Any
import logging
from pathlib import Path

from langchain_milvus import Milvus
from pymilvus import connections, utility

logger = logging.getLogger(__name__)

class VectorDBProvider(str, Enum):
    MILVUS = "milvus"
    # ... 其他数据库待添加

class VectorDBConfig:
    def __init__(self, provider: str, index_mode: str):
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = "03-vector-store/langchain_milvus.db"

class VectorStoreService:
    def __init__(self):
        self.initialized_dbs = {}
        # 确保存储目录存在
        os.makedirs("03-vector-store", exist_ok=True)
    
    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        start_time = datetime.now()
        
        # 读取embedding文件
        embeddings = self._load_embeddings(embedding_file)
        
        # 根据不同的数据库进行索引
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings, config)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "database": config.provider,
            "index_mode": config.index_mode,
            "total_vectors": len(embeddings),
            "index_size": result.get("index_size", "N/A"),
            "processing_time": processing_time,
            "collection_name": result.get("collection_name", "N/A")
        }
    
    def _load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")
                
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                embeddings = data["embeddings"]
                logger.info(f"Found {len(embeddings)} embeddings")
                return embeddings
                
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise
    
    def _index_to_milvus(self, embeddings: List[Dict], config: VectorDBConfig) -> Dict[str, Any]:
        try:
            # 准备collection名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            collection_name = f"doc_{timestamp}"
            
            # 连接到Milvus
            connections.connect(
                alias="default", 
                uri=config.milvus_uri
            )
            
            # 获取第一个embedding的维度
            first_embedding = embeddings[0].get("metadata", {}).get("vector_dimension")
            if not first_embedding:
                raise ValueError("Missing vector dimension in embedding metadata")
            
            # 准备schema
            vector_schema = {
                "dim": first_embedding,
                "index_type": self._get_milvus_index_type(config.index_mode),
                "metric_type": "COSINE",
                "params": self._get_milvus_index_params(config.index_mode)
            }
            
            # 准备数据
            texts = []
            vectors = []
            metadatas = []
            
            for emb in embeddings:
                metadata = emb.get("metadata", {})
                texts.append(metadata.get("content", ""))
                vectors.append(emb.get("embedding", []))
                metadatas.append({
                    "document_name": metadata.get("document_name", ""),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "total_chunks": metadata.get("total_chunks", 0),
                    "page_number": metadata.get("page_number", 0),
                    "page_range": metadata.get("page_range", ""),
                    "chunking_method": metadata.get("chunking_method", ""),
                    "content": metadata.get("content", ""),
                    "embedding_provider": metadata.get("embedding_provider", ""),
                    "embedding_model": metadata.get("embedding_model", ""),
                    "embedding_timestamp": metadata.get("embedding_timestamp", "")
                })
            
            logger.info(f"Indexing {len(vectors)} vectors to Milvus")
            
            # 创建Milvus实例并添加embeddings
            vector_store = Milvus(
                collection_name=collection_name,
                connection_args={"uri": config.milvus_uri},
                embedding_function=None,  # 我们已经有了embeddings，不需要embedding function
                index_params=vector_schema,
                search_params={"metric_type": "COSINE"},
                consistency_level="Strong"
            )
            
            # 添加embeddings
            ids = vector_store.add_embeddings(
                texts=texts,
                embeddings=vectors,
                metadatas=metadatas,
                batch_size=100
            )
            
            logger.info(f"Successfully indexed {len(ids)} vectors")
            
            return {
                "index_size": len(ids),
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            raise
        
        finally:
            connections.disconnect("default")
    
    def _get_milvus_index_type(self, index_mode: str) -> str:
        index_types = {
            "flat": "FLAT",
            "ivf_flat": "IVF_FLAT",
            "ivf_sq8": "IVF_SQ8",
            "hnsw": "HNSW"
        }
        return index_types.get(index_mode, "FLAT")
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        params = {
            "flat": {},
            "ivf_flat": {"nlist": 1024},
            "ivf_sq8": {"nlist": 1024},
            "hnsw": {
                "M": 16,
                "efConstruction": 500
            }
        }
        return params.get(index_mode, {})