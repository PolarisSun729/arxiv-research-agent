import os
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from pymilvus import connections, utility
from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
from utils.config import VectorDBProvider, MILVUS_CONFIG  # Updated import
from pypinyin import lazy_pinyin, Style

logger = logging.getLogger(__name__)

CONTENT_MAX_LENGTH = 12000

class VectorDBConfig:
    """
    鍚戦噺鏁版嵁搴撻厤缃被锛岀敤浜庡瓨鍌ㄥ拰绠＄悊鍚戦噺鏁版嵁搴撶殑閰嶇疆淇℃伅
    """
    def __init__(self, provider: str, index_mode: str):
        """
        鍒濆鍖栧悜閲忔暟鎹簱閰嶇疆
        
        鍙傛暟:
            provider: 鍚戦噺鏁版嵁搴撴彁渚涘晢鍚嶇О
            index_mode: 绱㈠紩妯″紡
        """
        self.provider = provider
        self.index_mode = index_mode
        self.milvus_uri = MILVUS_CONFIG["uri"]

    def _get_milvus_index_type(self, index_mode: str) -> str:
        """
        鏍规嵁绱㈠紩妯″紡鑾峰彇Milvus绱㈠紩绫诲瀷
        
        鍙傛暟:
            index_mode: 绱㈠紩妯″紡
            
        杩斿洖:
            瀵瑰簲鐨凪ilvus绱㈠紩绫诲瀷
        """
        return MILVUS_CONFIG["index_types"].get(index_mode, "FLAT")
    
    def _get_milvus_index_params(self, index_mode: str) -> Dict[str, Any]:
        """
        鏍规嵁绱㈠紩妯″紡鑾峰彇Milvus绱㈠紩鍙傛暟
        
        鍙傛暟:
            index_mode: 绱㈠紩妯″紡
            
        杩斿洖:
            瀵瑰簲鐨凪ilvus绱㈠紩鍙傛暟瀛楀吀
        """
        return MILVUS_CONFIG["index_params"].get(index_mode, {})

class VectorStoreService:
    """
    鍚戦噺瀛樺偍鏈嶅姟绫伙紝鎻愪緵鍚戦噺鏁版嵁鐨勭储寮曘€佹煡璇㈠拰绠＄悊鍔熻兘
    """
    def __init__(self):
        """
        鍒濆鍖栧悜閲忓瓨鍌ㄦ湇鍔?
        """
        self.initialized_dbs = {}
        # 纭繚瀛樺偍鐩綍瀛樺湪
        os.makedirs("03-vector-store", exist_ok=True)
    
    def _get_milvus_index_type(self, config: VectorDBConfig) -> str:
        """
        浠庨厤缃璞¤幏鍙朚ilvus绱㈠紩绫诲瀷
        
        鍙傛暟:
            config: 鍚戦噺鏁版嵁搴撻厤缃璞?
            
        杩斿洖:
            Milvus绱㈠紩绫诲瀷
        """
        return config._get_milvus_index_type(config.index_mode)
    
    def _get_milvus_index_params(self, config: VectorDBConfig) -> Dict[str, Any]:
        """
        浠庨厤缃璞¤幏鍙朚ilvus绱㈠紩鍙傛暟
        
        鍙傛暟:
            config: 鍚戦噺鏁版嵁搴撻厤缃璞?
            
        杩斿洖:
            Milvus绱㈠紩鍙傛暟瀛楀吀
        """
        return config._get_milvus_index_params(config.index_mode)
    
    def index_embeddings(self, embedding_file: str, config: VectorDBConfig) -> Dict[str, Any]:
        """
        灏嗗祵鍏ュ悜閲忕储寮曞埌鍚戦噺鏁版嵁搴?
        
        鍙傛暟:
            embedding_file: 宓屽叆鍚戦噺鏂囦欢璺緞
            config: 鍚戦噺鏁版嵁搴撻厤缃璞?
            
        杩斿洖:
            绱㈠紩缁撴灉淇℃伅瀛楀吀
        """
        start_time = datetime.now()
        
        # 璇诲彇embedding鏂囦欢
        embeddings_data = self._load_embeddings(embedding_file)
        
        # 鏍规嵁涓嶅悓鐨勬暟鎹簱杩涜绱㈠紩
        if config.provider == VectorDBProvider.MILVUS:
            result = self._index_to_milvus(embeddings_data, config)
        
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
        鍔犺浇embedding鏂囦欢锛岃繑鍥為厤缃俊鎭拰embeddings
        
        鍙傛暟:
            file_path: 宓屽叆鍚戦噺鏂囦欢璺緞
            
        杩斿洖:
            鍖呭惈宓屽叆鍚戦噺鍜屽厓鏁版嵁鐨勫瓧鍏?
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loading embeddings from {file_path}")
                
                if not isinstance(data, dict) or "embeddings" not in data:
                    raise ValueError("Invalid embedding file format: missing 'embeddings' key")
                    
                # 杩斿洖瀹屾暣鐨勬暟鎹紝鍖呮嫭椤跺眰閰嶇疆
                logger.info(f"Found {len(data['embeddings'])} embeddings")
                return data
                
        except Exception as e:
            logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
            raise
    
    def _index_to_milvus(self, embeddings_data: Dict[str, Any], config: VectorDBConfig) -> Dict[str, Any]:
        """
        灏嗗祵鍏ュ悜閲忕储寮曞埌Milvus鏁版嵁搴?
        
        鍙傛暟:
            embeddings_data: 宓屽叆鍚戦噺鏁版嵁
            config: 鍚戦噺鏁版嵁搴撻厤缃璞?
            
        杩斿洖:
            绱㈠紩缁撴灉淇℃伅瀛楀吀
        """
        try:
            # 浣跨敤 filename 浣滀负 collection 鍚嶇О鍓嶇紑
            filename = embeddings_data.get("filename", "")
            # 濡傛灉鏈?.pdf 鍚庣紑锛岀Щ闄ゅ畠
            base_name = filename.replace('.pdf', '') if filename else "doc"
            
            # Convert Chinese characters to pinyin
            base_name = ''.join(lazy_pinyin(base_name, style=Style.NORMAL))
            
            # Replace hyphens and dots with underscores in the base name
            base_name = base_name.replace('-', '_').replace('.', '_')
            
            # Ensure the collection name starts with a letter or underscore
            if not base_name[0].isalpha() and base_name[0] != '_':
                base_name = f"_{base_name}"
            
            # Get embedding provider
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = f"{base_name}_{embedding_provider}_{timestamp}"
            
            # 杩炴帴鍒癕ilvus
            connections.connect(
                alias="default", 
                uri=config.milvus_uri
            )
            
            # 浠庨《灞傞厤缃幏鍙栧悜閲忕淮搴?
            vector_dim = int(embeddings_data.get("vector_dimension"))
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")
            
            logger.info(f"Creating collection with dimension: {vector_dim}")
            
            # 瀹氫箟瀛楁
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": CONTENT_MAX_LENGTH},
                {"name": "document_name", "dtype": "VARCHAR", "max_length": 255},
                # 杩欓噷鍗曠嫭瀛?source锛屽悗闈?QA / 妫€绱㈠睍绀烘椂鍙互鐩存帴鍥炲埌鍘熷 PDF 鏂囦欢鍚嶃€?                {"name": "source", "dtype": "VARCHAR", "max_length": 255},
                {"name": "chunk_id", "dtype": "INT64"},
                {"name": "chunk_index", "dtype": "INT64"},
                {"name": "parent_chunk_id", "dtype": "INT64"},
                {"name": "original_chunk_id", "dtype": "INT64"},
                {"name": "total_chunks", "dtype": "INT64"},
                {"name": "word_count", "dtype": "INT64"},
                {"name": "page_number", "dtype": "VARCHAR", "max_length": 10},
                # page_start / page_end 棰勭暀缁欐湭鏉ヨ法椤?chunk锛岀洰鍓嶉〉鍐呭垏鍒嗘椂涓よ€呯浉鍚屻€?                {"name": "page_start", "dtype": "INT64"},
                {"name": "page_end", "dtype": "INT64"},
                {"name": "page_range", "dtype": "VARCHAR", "max_length": 10},
                {"name": "subchunk_index", "dtype": "INT64"},
                {"name": "subchunk_count", "dtype": "INT64"},
                {"name": "subchunk_label", "dtype": "VARCHAR", "max_length": 64},
                {"name": "content_part_index", "dtype": "INT64"},
                {"name": "content_part_count", "dtype": "INT64"},
                {"name": "content_part_label", "dtype": "VARCHAR", "max_length": 32},
                # {"name": "chunking_method", "dtype": "VARCHAR", "max_length": 50},
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
            
            # 鍑嗗鏁版嵁涓哄垪琛ㄦ牸寮?
            entities = []
            for emb in embeddings_data["embeddings"]:
                metadata = emb["metadata"]
                content = str(metadata.get("content", ""))
                parent_chunk_id = int(metadata.get("parent_chunk_id", metadata.get("chunk_id", metadata.get("chunk_index", 0))))
                original_chunk_id = int(metadata.get("original_chunk_id", parent_chunk_id))
                subchunk_index = int(metadata.get("subchunk_index", 1))
                subchunk_count = int(metadata.get("subchunk_count", 1))
                subchunk_label = str(
                    metadata.get("subchunk_label", f"chunk {parent_chunk_id} part {subchunk_index}/{subchunk_count}")
                )

                # 鍏ュ簱鍓嶆妸椤电爜銆佹潵婧愬拰 chunk 搴忓彿缁熶竴钀藉埌鍚戦噺搴撳瓧娈甸噷锛岄伩鍏嶅悗缁绱涪淇℃伅銆?
                page_start = int(metadata.get("page_start", metadata.get("page_number", 0)))
                page_end = int(metadata.get("page_end", page_start))
                source = str(metadata.get("source", embeddings_data.get("filename", "")))
                chunk_index = int(metadata.get("chunk_index", metadata.get("chunk_id", 0)))
                total_chunks = int(metadata.get("total_chunks", 0))
                base_chunk_id = int(metadata.get("chunk_id", chunk_index))
                entity = {
                    "content": content,
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
                    # "chunking_method": str(metadata.get("chunking_method", "")),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),
                    "embedding_model": embeddings_data.get("embedding_model", ""),
                    "embedding_timestamp": str(metadata.get("embedding_timestamp", "")),
                    "vector": [float(x) for x in emb.get("embedding", [])]
                }
                entities.append(entity)
            
            logger.info(f"Creating Milvus collection: {collection_name}")
            
            # 鍒涘缓collection
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
            
            # 鎻掑叆鏁版嵁
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
            
            # 鍒涘缓绱㈠紩
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

    def _validate_varchar_lengths(self, entities: List[Dict[str, Any]], fields: List[FieldSchema]) -> None:
        varchar_limits = {
            field.name: getattr(field, "max_length", None)
            for field in fields
            if getattr(field, "dtype", None) == DataType.VARCHAR
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

    def list_collections(self, provider: str) -> List[str]:
        """
        鍒楀嚭鎸囧畾鎻愪緵鍟嗙殑鎵€鏈夐泦鍚?
        
        鍙傛暟:
            provider: 鍚戦噺鏁版嵁搴撴彁渚涘晢
            
        杩斿洖:
            闆嗗悎鍚嶇О鍒楄〃
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collections = utility.list_collections()
                return collections
            finally:
                connections.disconnect("default")
        return []

    def delete_collection(self, provider: str, collection_name: str) -> bool:
        """
        鍒犻櫎鎸囧畾鐨勯泦鍚?
        
        鍙傛暟:
            provider: 鍚戦噺鏁版嵁搴撴彁渚涘晢
            collection_name: 闆嗗悎鍚嶇О
            
        杩斿洖:
            鏄惁鍒犻櫎鎴愬姛
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                utility.drop_collection(collection_name)
                return True
            finally:
                connections.disconnect("default")
        return False

    def get_collection_info(self, provider: str, collection_name: str) -> Dict[str, Any]:
        """
        鑾峰彇鎸囧畾闆嗗悎鐨勪俊鎭?
        
        鍙傛暟:
            provider: 鍚戦噺鏁版嵁搴撴彁渚涘晢
            collection_name: 闆嗗悎鍚嶇О
            
        杩斿洖:
            闆嗗悎淇℃伅瀛楀吀
        """
        if provider == VectorDBProvider.MILVUS:
            try:
                connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
                collection = Collection(collection_name)
                return {
                    "name": collection_name,
                    "num_entities": collection.num_entities,
                    "schema": collection.schema.to_dict()
                }
            finally:
                connections.disconnect("default")
        return {}

    def insert_single_embedding(self, collection_name: str, embedding: List[float], metadata: Dict[str, Any]) -> int:
        """
        鎻掑叆鍗曚釜宓屽叆鍚戦噺鍒版寚瀹氶泦鍚?
        
        鍙傛暟:
            collection_name: 闆嗗悎鍚嶇О
            embedding: 宓屽叆鍚戦噺
            metadata: 鍏冩暟鎹瓧鍏革紝鍖呭惈content, arxiv_id, title绛変俊鎭?
            
        杩斿洖:
            鎻掑叆鐨勫悜閲廔D锛坧rimary key锛?
        """
        try:
            connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])
            
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
            else:
                vector_dim = len(embedding)
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=CONTENT_MAX_LENGTH),
                    FieldSchema(name="arxiv_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="authors", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="categories", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="published_date", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="embedding_model", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
                ]
                schema = CollectionSchema(fields=fields, description=f"arXiv paper abstract embeddings collection")
                collection = Collection(name=collection_name, schema=schema)
                
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "FLAT",
                    "params": {}
                }
                collection.create_index(field_name="vector", index_params=index_params)
            
            entity = {
                "content": str(metadata.get("content", "")),
                "arxiv_id": str(metadata.get("arxiv_id", "")),
                "title": str(metadata.get("title", "")),
                "authors": str(metadata.get("authors", "")),
                "categories": str(metadata.get("categories", "")),
                "published_date": str(metadata.get("published_date", "")),
                "url": str(metadata.get("url", "")),
                "embedding_model": str(metadata.get("embedding_model", "")),
                "vector": [float(x) for x in embedding]
            }
            self._validate_varchar_lengths([entity], collection.schema.fields)
            insertable_fields = [field.name for field in collection.schema.fields if not getattr(field, "auto_id", False)]
            insert_columns = [[entity.get(field_name)] for field_name in insertable_fields]
            insert_result = collection.insert(insert_columns)
            collection.load()
            
            return insert_result.primary_keys[0]
            
        except Exception as e:
            logger.error(f"Error inserting single embedding: {str(e)}")
            raise
        finally:
            connections.disconnect("default")

    def search_similar_vectors(self, collection_name: str, query_vector: List[float], top_k: int = 10, filter_arxiv_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        鎼滅储 chunk 绾у悜閲忥紝骞舵妸椤电爜淇℃伅涓€骞惰繑鍥炪€?
        杩欐牱 QA 鐢熸垚绛旀鏃跺氨鑳界洿鎺ュ甫鍑烘潵婧愰〉鐮侊紝鑰屼笉鏄彧缁欎竴娈典笉鐭ュ嚭澶勭殑鏂囨湰銆?        """
        try:
            connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])

            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []

            collection = Collection(collection_name)
            collection.load()

            field_names = {field.name for field in collection.schema.fields}
            candidate_fields = [
                "content",
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
                expr = f"arxiv_id not in {filter_arxiv_ids}"

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=top_k,
                expr=expr,
                output_fields=output_fields,
            )

            similar_vectors = [
                self._build_chunk_payload(
                    entity=hit.entity,
                    score=float(hit.score),
                    distance=float(hit.distance) if hasattr(hit, "distance") else None,
                )
                for hit in results[0]
            ]

            return similar_vectors

        except Exception as e:
            logger.error(f"Error searching similar vectors: {str(e)}")
            raise
        finally:
            connections.disconnect("default")

    def get_all_chunks(self, collection_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            connections.connect(alias="default", uri=MILVUS_CONFIG["uri"])

            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []

            collection = Collection(collection_name)
            collection.load()

            field_names = {field.name for field in collection.schema.fields}
            candidate_fields = [
                "id",
                "content",
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
            query_limit = limit or collection.num_entities
            entities = collection.query(
                expr="id >= 0",
                output_fields=output_fields,
                limit=query_limit,
            )
            return [self._build_chunk_payload(entity=entity, score=None, distance=None) for entity in entities]
        except Exception as e:
            logger.error(f"Error reading collection chunks: {str(e)}")
            raise
        finally:
            connections.disconnect("default")

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
            "embedding_provider": getter("embedding_provider", "") or "",
            "embedding_model": getter("embedding_model", "") or "",
            "embedding_timestamp": getter("embedding_timestamp", "") or "",
        }

        payload = {
            "text": content,
            "content": content,
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
            "metadata": metadata,
        }
        return payload

