import os
import ast
from datetime import datetime
import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging
from pymilvus import (
    DataType,
    MilvusClient,
)
from pymilvus.client.types import LoadState
from utils.config import VectorDBProvider, MILVUS_CONFIG, get_vector_store_runtime_config
from utils.storage_paths import resolve_backend_artifact_path
from pypinyin import lazy_pinyin, Style

logger = logging.getLogger(__name__)

VECTOR_STORE_CONFIG = get_vector_store_runtime_config()

CONTENT_MAX_LENGTH = VECTOR_STORE_CONFIG["content_max_length"]
RERANK_TEXT_MAX_LENGTH = VECTOR_STORE_CONFIG["rerank_text_max_length"]
ASSET_PATH_MAX_LENGTH = VECTOR_STORE_CONFIG["asset_path_max_length"]
ASSET_SUMMARY_MAX_LENGTH = VECTOR_STORE_CONFIG["asset_summary_max_length"]
ASSET_PREVIEW_MAX_LENGTH = VECTOR_STORE_CONFIG["asset_preview_max_length"]


# chunk 级 collection 检索/导出时希望带回的字段（按需与实际 schema 求交集，旧 collection 缺字段也能读）。
# 顺序即 Milvus output_fields 顺序；get_all_chunks 会在最前面额外加上主键 id。
CHUNK_OUTPUT_FIELDS = (
    "content",
    "rerank_text",
    "index_id",
    "index_type",
    "index_text",
    "index_weight",
    "retrieval_index_id",
    "retrieval_index_type",
    "retrieval_index_text",
    "retrieval_index_weight",
    "retrieval_index_enabled_routes",
    "chunk_type",
    "asset_kind",
    "asset_path",
    "asset_abs_path",
    "asset_summary",
    "asset_preview_text",
    "asset_caption",
    "asset_section_match_type",
    "asset_section_match_confidence",
    "asset_section_match_reason",
    "asset_section_match_is_heuristic",
    "asset_section_match_allow_embedding",
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
)

# 论文级 collection（如 arxiv_paper_embeddings）检索时带回的展示字段。
PAPER_SEARCH_OUTPUT_FIELDS = (
    "arxiv_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published_date",
    "url",
    "embedding_model",
    "content",
)

# 按 arxiv_id 复用论文向量时读取的字段（包含主键 id 与向量本身）。
PAPER_EMBEDDING_OUTPUT_FIELDS = (
    "id",
    "arxiv_id",
    "title",
    "authors",
    "categories",
    "published_date",
    "url",
    "embedding_model",
    "vector",
)

# _build_chunk_payload 返回值中，从 metadata 平铺到顶层的字段，分两段是为了保持原有的键顺序：
# 第一段在 score/distance 之前，第二段在其之后。
_CHUNK_PAYLOAD_KEYS_BEFORE_SCORE = (
    "rerank_text",
    "index_id",
    "index_type",
    "index_text",
    "index_weight",
    "retrieval_index_id",
    "retrieval_index_type",
    "retrieval_index_text",
    "retrieval_index_weight",
    "retrieval_index_enabled_routes",
    "matched_index_id",
    "matched_index_type",
    "matched_index_text",
    "matched_index_score",
    "chunk_type",
    "asset_kind",
    "asset_path",
    "asset_abs_path",
    "asset_summary",
    "asset_preview_text",
    "asset_caption",
    "asset_section_match_type",
    "asset_section_match_confidence",
    "asset_section_match_reason",
    "asset_section_match_is_heuristic",
    "asset_section_match_allow_embedding",
    "asset_rows",
    "asset_columns",
    "order_index",
)
_CHUNK_PAYLOAD_KEYS_AFTER_SCORE = (
    "source",
    "document_name",
    "chunk_id",
    "chunk_index",
    "parent_chunk_id",
    "original_chunk_id",
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
)

# 建索引与查询共用同一个度量类型，两处必须保持一致。
_METRIC_TYPE = "COSINE"
_SEARCH_PARAMS = {"metric_type": _METRIC_TYPE, "params": {}}

# Milvus 命中里只有 "distance" 字段，其含义随度量类型变化；这里统一换算成“越大越相似”的 score。
# Hit.score 只是 pymilvus 中即将废弃的属性别名，并不是 dict 键，不能用 hit.get("score") 读取。
_DISTANCE_TO_SCORE: Dict[str, Callable[[float], float]] = {
    "COSINE": lambda distance: distance,  # 余弦相似度，范围 [-1, 1]
    "IP": lambda distance: distance,  # 内积，越大越相似
    "L2": lambda distance: 1.0 / (1.0 + distance),  # 平方欧氏距离，越小越相似，映射到 (0, 1]
}
if _METRIC_TYPE not in _DISTANCE_TO_SCORE:
    raise ValueError(f"Unsupported Milvus metric type: {_METRIC_TYPE}")


def _entity_getter(entity: Any) -> Callable[..., Any]:
    """为 Milvus 返回的实体生成统一取值函数：dict 用 .get，其它对象用 getattr。"""
    if isinstance(entity, dict):
        return lambda key, default=None: entity.get(key, default)
    return lambda key, default=None: getattr(entity, key, default)


def _hit_score_and_distance(hit: Dict[str, Any], metric_type: str = _METRIC_TYPE) -> Tuple[float, float]:
    """从检索命中读取 distance，并按度量类型换算出 score；缺少 distance 时直接报错，不伪造分数。"""
    # 只看顶层键：Hit.get 找不到时会退到 entity 里查同名字段，可能误读到业务字段。
    if "distance" not in hit or hit["distance"] is None:
        raise ValueError(f"Milvus search hit has no distance field; hit keys={sorted(hit.keys())}")
    distance = float(hit["distance"])
    return _DISTANCE_TO_SCORE[metric_type](distance), distance


def _split_display_values(value: Any) -> List[str]:
    """把作者、分类等多值字段统一拆成字符串列表。

    兼容三种存储形态：真实列表、Python 列表字面量字符串（如 "['a', 'b']"）、
    以逗号/分号/竖线/空白分隔的纯文本。
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("("):
            try:
                # literal_eval 只解析字面量，不会执行代码，可以安全处理历史数据中的 repr 字符串。
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


def _build_arxiv_id_exclude_filter(filter_arxiv_ids: Optional[List[str]], field_names: Set[str]) -> str:
    """生成排除指定 arxiv_id 的 Milvus 过滤表达式；collection 没有 arxiv_id 字段或列表为空时返回空串。"""
    if not filter_arxiv_ids or "arxiv_id" not in field_names:
        return ""
    quoted_ids = ", ".join(f'"{arxiv_id}"' for arxiv_id in filter_arxiv_ids if arxiv_id)
    return f"arxiv_id not in [{quoted_ids}]" if quoted_ids else ""


def normalize_collection_name(collection_name: str) -> str:
    """把任意字符串规范化为合法的 Milvus collection 名称。

    规则：非字母数字下划线的字符替换为下划线，合并连续下划线并去掉首尾下划线；
    首字符不是字母时补一个前导下划线；结果为空时回落为 "collection"。
    """
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
        # 向量库本地产物归属后端目录，避免从仓库根目录启动时重新生成根目录资产。
        self.vector_store_dir = resolve_backend_artifact_path(
            "03-vector-store",
        )
        os.makedirs(self.vector_store_dir, exist_ok=True)

    def _get_client(self) -> MilvusClient:
        """每次调用创建一个新的 Milvus 客户端；客户端不跨请求复用，避免连接状态在线程间共享。"""
        return MilvusClient(uri=MILVUS_CONFIG["uri"])

    def resolve_collection_name(self, collection_name: str) -> str:
        """
        将任意 collection 名称规范化为 Milvus 可接受的形式。
        """
        return normalize_collection_name(collection_name)

    def collection_exists(self, provider: str, collection_name: str) -> bool:
        """判断 collection 是否存在；非 Milvus 提供商一律返回 False。"""
        if provider == VectorDBProvider.MILVUS:
            client = self._get_client()
            return client.has_collection(collection_name=self.resolve_collection_name(collection_name))
        return False

    def _describe_existing_collection(self, client: MilvusClient, collection_name: str) -> Optional[Tuple[str, Set[str]]]:
        """返回 (规范化后的 collection 名, schema 字段名集合)；collection 不存在时记录告警并返回 None。

        读取前先查 schema，是为了让 output_fields 只请求实际存在的字段，兼容不同时期建的 collection。
        Milvus Lite 进程重启后 collection 会回到 released 状态，未 load 时 search/query 直接报错，
        所以读取前顺带确认已加载。
        """
        resolved_name = self.resolve_collection_name(collection_name)
        if not client.has_collection(collection_name=resolved_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return None
        if client.get_load_state(collection_name=resolved_name).get("state") != LoadState.Loaded:
            logger.info("Collection %s is not loaded; loading before read", resolved_name)
            client.load_collection(collection_name=resolved_name)
        collection_desc = client.describe_collection(collection_name=resolved_name)
        field_names = {field.get("name") for field in collection_desc.get("fields", [])}
        return resolved_name, field_names
    
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

    @staticmethod
    def _serialize_json_list(value: Any) -> str:
        """把列表或分隔字符串序列化为 JSON 数组字符串，用于在 VARCHAR 字段里保存列表。

        已经是 JSON 数组的字符串原样返回；"a,b|c" 这类分隔文本会被拆分后再序列化。
        """
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return "[]"
            if raw.startswith("["):
                return raw
            return json.dumps([item.strip() for item in re.split(r"[,|]", raw) if item.strip()], ensure_ascii=False)
        if isinstance(value, (list, tuple, set)):
            return json.dumps([str(item).strip() for item in value if str(item).strip()], ensure_ascii=False)
        return "[]"

    @staticmethod
    def _parse_json_list(value: Any) -> List[str]:
        """_serialize_json_list 的逆操作：解析 JSON 数组，失败时按逗号/竖线分隔兜底。"""
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raw = str(value or "").strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in re.split(r"[,|]", raw) if item.strip()]
    
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

            # 生成 collection 名称；非法字符替换、首字符补下划线等规则统一交给 normalize_collection_name。
            # （此前这里手工处理了一遍首字符，base_name 为空时会触发 IndexError，且结果与统一规则等价。）
            embedding_provider = embeddings_data.get("embedding_provider", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            collection_name = normalize_collection_name(f"{base_name}_{embedding_provider}_{timestamp}")

            client = self._get_client()

            # 从配置中读取向量维度；缺失时走下面的明确报错，而不是 int(None) 的 TypeError。
            vector_dim = int(embeddings_data.get("vector_dimension") or 0)
            if not vector_dim:
                raise ValueError("Missing vector_dimension in embedding file")
            
            logger.info(f"Creating collection with dimension: {vector_dim}")
            
            # 定义 collection 字段
            fields = [
                {"name": "id", "dtype": "INT64", "is_primary": True, "auto_id": True},
                {"name": "content", "dtype": "VARCHAR", "max_length": CONTENT_MAX_LENGTH},
                {"name": "rerank_text", "dtype": "VARCHAR", "max_length": RERANK_TEXT_MAX_LENGTH},
                {"name": "index_id", "dtype": "VARCHAR", "max_length": 192},
                {"name": "index_type", "dtype": "VARCHAR", "max_length": 32},
                {"name": "index_text", "dtype": "VARCHAR", "max_length": CONTENT_MAX_LENGTH},
                {"name": "index_weight", "dtype": "DOUBLE"},
                {"name": "retrieval_index_id", "dtype": "VARCHAR", "max_length": 192},
                {"name": "retrieval_index_type", "dtype": "VARCHAR", "max_length": 32},
                {"name": "retrieval_index_text", "dtype": "VARCHAR", "max_length": CONTENT_MAX_LENGTH},
                {"name": "retrieval_index_weight", "dtype": "DOUBLE"},
                {"name": "retrieval_index_enabled_routes", "dtype": "VARCHAR", "max_length": 512},
                {"name": "chunk_type", "dtype": "VARCHAR", "max_length": 24},
                {"name": "asset_kind", "dtype": "VARCHAR", "max_length": 24},
                {"name": "asset_path", "dtype": "VARCHAR", "max_length": ASSET_PATH_MAX_LENGTH},
                {"name": "asset_abs_path", "dtype": "VARCHAR", "max_length": ASSET_PATH_MAX_LENGTH},
                {"name": "asset_summary", "dtype": "VARCHAR", "max_length": ASSET_SUMMARY_MAX_LENGTH},
                {"name": "asset_preview_text", "dtype": "VARCHAR", "max_length": ASSET_PREVIEW_MAX_LENGTH},
                {"name": "asset_caption", "dtype": "VARCHAR", "max_length": ASSET_SUMMARY_MAX_LENGTH},
                {"name": "asset_section_match_type", "dtype": "VARCHAR", "max_length": 32},
                {"name": "asset_section_match_confidence", "dtype": "DOUBLE"},
                {"name": "asset_section_match_reason", "dtype": "VARCHAR", "max_length": 512},
                {"name": "asset_section_match_is_heuristic", "dtype": "BOOL"},
                {"name": "asset_section_match_allow_embedding", "dtype": "BOOL"},
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
                {"name": "section_path", "dtype": "VARCHAR", "max_length": 1024},
                {"name": "section_title", "dtype": "VARCHAR", "max_length": 512},
                {"name": "section_level", "dtype": "INT64"},
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
                index_id = str(metadata.get("index_id") or metadata.get("retrieval_index_id") or "")
                index_type = str(metadata.get("index_type") or metadata.get("retrieval_index_type") or "body")
                index_text = str(metadata.get("index_text") or metadata.get("retrieval_index_text") or "")
                index_weight = float(metadata.get("index_weight", metadata.get("retrieval_index_weight", 1.0)) or 1.0)
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
                    # index_* 是新的 index-level 合约字段；retrieval_index_* 暂时保留，避免旧调用方立刻失效。
                    "index_id": index_id,
                    "index_type": index_type,
                    "index_text": index_text,
                    "index_weight": index_weight,
                    "retrieval_index_id": index_id,
                    "retrieval_index_type": index_type,
                    "retrieval_index_text": index_text,
                    "retrieval_index_weight": index_weight,
                    "retrieval_index_enabled_routes": self._serialize_json_list(
                        metadata.get("retrieval_index_enabled_routes", [])
                    ),
                    "chunk_type": str(metadata.get("chunk_type", "text") or "text"),
                    "asset_kind": str(metadata.get("asset_kind", "") or ""),
                    "asset_path": str(metadata.get("asset_path", "") or ""),
                    "asset_abs_path": str(metadata.get("asset_abs_path", "") or ""),
                    "asset_summary": str(metadata.get("asset_summary", "") or ""),
                    "asset_preview_text": str(metadata.get("asset_preview_text", "") or ""),
                    "asset_caption": str(metadata.get("asset_caption", "") or ""),
                    # 这些字段说明 section 是弱锚点还是可信归属，真实检索结果需要原样带回 debug/source。
                    "asset_section_match_type": str(metadata.get("asset_section_match_type", "") or ""),
                    "asset_section_match_confidence": float(metadata.get("asset_section_match_confidence", 0.0) or 0.0),
                    "asset_section_match_reason": str(metadata.get("asset_section_match_reason", "") or ""),
                    "asset_section_match_is_heuristic": bool(metadata.get("asset_section_match_is_heuristic", False)),
                    "asset_section_match_allow_embedding": bool(metadata.get("asset_section_match_allow_embedding", False)),
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
                    "section_title": str(metadata.get("section_title", "")),
                    "section_level": int(metadata.get("section_level", 0) or 0),
                    "embedding_provider": embeddings_data.get("embedding_provider", ""),
                    "embedding_model": embeddings_data.get("embedding_model", ""),
                    "embedding_timestamp": str(metadata.get("embedding_timestamp", "")),
                    "vector": [float(x) for x in emb.get("embedding", [])]
                }
                entities.append(entity)
            
            logger.info(f"Creating Milvus collection: {collection_name}")
            
            schema = MilvusClient.create_schema(
                auto_id=True,
                enable_dynamic_field=False,
            )
            for field in fields:
                field_kwargs = {}
                if field.get("is_primary"):
                    field_kwargs["is_primary"] = True
                if field.get("auto_id"):
                    field_kwargs["auto_id"] = True
                if field.get("max_length") is not None:
                    field_kwargs["max_length"] = field["max_length"]
                if field.get("dim") is not None:
                    field_kwargs["dim"] = field["dim"]

                schema.add_field(
                    field_name=field["name"],
                    datatype=getattr(DataType, field["dtype"]),
                    **field_kwargs,
                )

            index_params = None
            if hasattr(client, "prepare_index_params"):
                index_params = client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    metric_type=_METRIC_TYPE,
                    index_type=self._get_milvus_index_type(config),
                    params=self._get_milvus_index_params(config),
                )

            client.create_collection(
                collection_name=collection_name,
                dimension=vector_dim,
                schema=schema,
                index_params=index_params,
            )
            
            # 插入数据
            logger.info(f"Inserting {len(entities)} vectors")
            insertable_fields = [field["name"] for field in fields if not field.get("auto_id", False)]
            normalized_entities = [
                {key: value for key, value in entity.items() if key in insertable_fields}
                for entity in entities
            ]
            # Milvus 只保存可检索 preview 字段，完整 index_text 已落在 retrieval index artifact / embedding 文件中。
            normalized_entities = self._truncate_entities_to_varchar_limits(normalized_entities, fields)
            self._validate_varchar_lengths(normalized_entities, fields)
            insert_result = client.insert(
                collection_name=collection_name,
                data=normalized_entities,
            )
            
            return {
                "index_size": int(insert_result.get("insert_count", len(normalized_entities))),
                "collection_name": collection_name
            }
            
        except Exception as e:
            logger.error(f"Error indexing to Milvus: {str(e)}")
            raise

    @staticmethod
    def _is_varchar_field(field: Any) -> bool:
        """判断字段是否为 VARCHAR。

        字段既可能是建表时的 dict 定义（dtype 为字符串），也可能是 describe_collection 返回的
        dict/对象（type 为 DataType 枚举或其整数值），这里统一兼容这些形态。
        """
        dtype = (
            (field.get("dtype") or field.get("type"))
            if isinstance(field, dict)
            else (getattr(field, "dtype", None) or getattr(field, "type", None))
        )
        varchar_type = getattr(DataType.VARCHAR, "value", DataType.VARCHAR)
        if dtype == DataType.VARCHAR or dtype == varchar_type:
            return True
        dtype_text = str(getattr(dtype, "name", dtype) or "").upper()
        return "VARCHAR" in dtype_text or "VAR_CHAR" in dtype_text

    @staticmethod
    def _varchar_max_length(field: Any) -> Optional[int]:
        """读取 VARCHAR 字段的 max_length（字节数）；可能在顶层或 params 中，取不到时返回 None。"""
        if isinstance(field, dict):
            raw_value = field.get("max_length") or (field.get("params", {}) or {}).get("max_length")
        else:
            raw_value = getattr(field, "max_length", None)
            if raw_value is None:
                raw_value = (getattr(field, "params", {}) or {}).get("max_length")
        if raw_value in (None, ""):
            return None
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return None

    def _varchar_limits(self, fields: List[Any]) -> Dict[str, Optional[int]]:
        """从字段定义中提取 {VARCHAR 字段名: 最大字节数}，供截断与校验共用。"""
        return {
            (field.get("name") if isinstance(field, dict) else getattr(field, "name", "")): (
                self._varchar_max_length(field)
            )
            for field in fields
            if self._is_varchar_field(field)
        }

    @staticmethod
    def _entity_chunk_ref(entity: Dict[str, Any]) -> Any:
        """日志/报错中用于定位实体的 chunk 编号，按 parent → original → chunk_id → chunk_index 依次回退。"""
        return entity.get(
            "parent_chunk_id",
            entity.get("original_chunk_id", entity.get("chunk_id", entity.get("chunk_index", 0))),
        )

    def _validate_varchar_lengths(self, entities: List[Dict[str, Any]], fields: List[Any]) -> None:
        """插入前校验 VARCHAR 字节长度，超限时抛出 ValueError（最多列出 5 条违规）。

        Milvus 的 max_length 按 UTF-8 字节计算，中文一个字占 3 字节，所以这里按字节而不是字符数比较。
        """
        varchar_limits = self._varchar_limits(fields)

        violations = []
        for entity in entities:
            chunk_ref = self._entity_chunk_ref(entity)
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
        """按 UTF-8 字节数截断字符串，并保证不会切断多字节字符（逐字节回退直到可以解码）。"""
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
        """把超长 VARCHAR 字段截断到 schema 允许的字节数，返回新的实体列表（不修改入参）。

        被截断的字段会记 warning，便于发现 preview 字段长度配置过小的问题。
        """
        varchar_limits = self._varchar_limits(fields)

        normalized_entities: List[Dict[str, Any]] = []
        for entity in entities:
            normalized_entity = dict(entity)
            chunk_ref = self._entity_chunk_ref(entity)
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
                    metric_type=_METRIC_TYPE,
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

            # 以实际 schema 为准做截断/校验：collection 可能是旧版本建的，max_length 与上面的建表定义不一定相同。
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
            described = self._describe_existing_collection(client, collection_name)
            if described is None:
                return []
            resolved_name, field_names = described
            output_fields = [field for field in CHUNK_OUTPUT_FIELDS if field in field_names]

            results = client.search(
                collection_name=resolved_name,
                data=[query_vector],
                filter=_build_arxiv_id_exclude_filter(filter_arxiv_ids, field_names),
                limit=top_k,
                anns_field="vector",
                output_fields=output_fields,
                search_params=_SEARCH_PARAMS,
            )

            # 不同 pymilvus 版本的命中结构不同：有的把字段放在 hit["entity"]，有的直接平铺在 hit 上。
            similar_vectors = []
            for hit in results[0]:
                score, distance = _hit_score_and_distance(hit)
                similar_vectors.append(self._build_chunk_payload(entity=hit.get("entity", hit), score=score, distance=distance))

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
            described = self._describe_existing_collection(client, collection_name)
            if described is None:
                return []
            resolved_name, field_names = described
            output_fields = [field for field in PAPER_SEARCH_OUTPUT_FIELDS if field in field_names]

            results = client.search(
                collection_name=resolved_name,
                data=[query_vector],
                filter=_build_arxiv_id_exclude_filter(filter_arxiv_ids, field_names),
                limit=top_k,
                anns_field="vector",
                output_fields=output_fields,
                search_params=_SEARCH_PARAMS,
            )

            similar_papers = []
            for hit in results[0]:
                score, distance = _hit_score_and_distance(hit)
                similar_papers.append(self._build_paper_payload(entity=hit.get("entity", hit), score=score, distance=distance))
            return similar_papers

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
            described = self._describe_existing_collection(client, collection_name)
            if described is None:
                return []
            resolved_name, field_names = described
            if "arxiv_id" not in field_names or "vector" not in field_names:
                logger.warning(
                    "Collection %s does not expose required fields arxiv_id/vector",
                    resolved_name,
                )
                return []

            quoted_ids = ", ".join(f'"{arxiv_id}"' for arxiv_id in normalized_ids)
            expr = f"arxiv_id in [{quoted_ids}]"

            output_fields = [field for field in PAPER_EMBEDDING_OUTPUT_FIELDS if field in field_names]

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

    def delete_paper_embeddings_published_before(
        self,
        collection_name: str,
        cutoff_date: str,
        keep_arxiv_ids: Set[str],
        dry_run: bool = False,
    ) -> int:
        """删除论文级 collection 中 published_date 早于 cutoff_date 的向量，返回删除（或 dry_run 时将删除）的条数。

        published_date 是 YYYY-MM-DD 或 ISO 时间字符串，按字符串比较即可；为空的行无法判断新旧，保留不动。
        keep_arxiv_ids 里的论文无论多旧都保留。失败直接抛出。
        """
        client = self._get_client()
        described = self._describe_existing_collection(client, collection_name)
        if described is None:
            return 0
        resolved_name, _field_names = described
        cutoff = str(cutoff_date).replace('"', "")
        iterator = client.query_iterator(
            collection_name=resolved_name,
            batch_size=5000,
            filter=f'published_date < "{cutoff}" and published_date != ""',
            output_fields=["arxiv_id"],
        )
        ids_to_delete: List[int] = []
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                ids_to_delete.extend(row["id"] for row in rows if str(row.get("arxiv_id") or "").strip() not in keep_arxiv_ids)
        finally:
            iterator.close()
        if dry_run or not ids_to_delete:
            return len(ids_to_delete)
        for start in range(0, len(ids_to_delete), 1000):
            client.delete(collection_name=resolved_name, ids=ids_to_delete[start : start + 1000])
        client.flush(collection_name=resolved_name)
        return len(ids_to_delete)

    def get_all_chunks(self, collection_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """按主键扫描读取 collection 中的 chunk（默认最多 10000 条），返回结构与向量检索结果一致。

        主要供关键词检索、上下文扩展等需要全量 chunk 的场景使用；score/distance 恒为 None。
        """
        try:
            client = self._get_client()
            described = self._describe_existing_collection(client, collection_name)
            if described is None:
                return []
            resolved_name, field_names = described
            output_fields = [field for field in ("id", *CHUNK_OUTPUT_FIELDS) if field in field_names]
            query_limit = limit if limit is not None else 10000
            if query_limit <= 0:
                logger.warning(
                    f"Collection {collection_name} has no entities or requested limit is non-positive: {query_limit}"
                )
                return []
            # Milvus query 必须带过滤条件，用恒真的主键条件表示“全部”。
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
        """把 Milvus chunk 实体转换为检索层统一的 chunk 结构。

        - 同一份字段既放在 metadata 中，也平铺到顶层，兼容按两种方式读取的调用方；
        - index_* 与 retrieval_index_* 是新旧两套命名，值始终保持一致；
        - 旧 collection 缺少 index 字段时，合成一个 body 类型的 legacy index，保证下游合约统一。
        """
        if entity is None:
            return {
                "text": "",
                "content": "",
                "score": score,
                "distance": distance,
                "metadata": {},
            }

        getter = _entity_getter(entity)

        content = getter("content", "") or ""
        rerank_text = getter("rerank_text", "") or ""
        retrieval_index_id = getter("retrieval_index_id", "") or getter("index_id", "") or ""
        retrieval_index_type = getter("retrieval_index_type", "") or getter("index_type", "") or ""
        retrieval_index_text = getter("retrieval_index_text", "") or getter("index_text", "") or ""
        retrieval_index_weight = getter("retrieval_index_weight", getter("index_weight", 1.0))
        retrieval_index_enabled_routes = self._parse_json_list(getter("retrieval_index_enabled_routes", ""))
        chunk_type = getter("chunk_type", "text") or "text"
        asset_kind = getter("asset_kind", "") or ""
        asset_path = getter("asset_path", "") or ""
        asset_abs_path = getter("asset_abs_path", "") or ""
        asset_summary = getter("asset_summary", "") or ""
        asset_preview_text = getter("asset_preview_text", "") or ""
        asset_caption = getter("asset_caption", "") or ""
        asset_section_match_type = getter("asset_section_match_type", "") or ""
        asset_section_match_confidence = getter("asset_section_match_confidence", 0.0)
        asset_section_match_reason = getter("asset_section_match_reason", "") or ""
        asset_section_match_is_heuristic = getter("asset_section_match_is_heuristic", False)
        asset_section_match_allow_embedding = getter("asset_section_match_allow_embedding", False)
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
        if not retrieval_index_id:
            chunk_ref = parent_chunk_id or original_chunk_id or chunk_id or chunk_index or 0
            # 旧 chunk-level collection 没有 index 字段；这里合成 body index，保证检索输出仍满足统一 index 合约。
            retrieval_index_id = f"{chunk_ref}:body:legacy"
            retrieval_index_type = "body"
            retrieval_index_text = str(content or "")
            retrieval_index_weight = 1.0
        matched_index_score = score

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
            "index_id": str(retrieval_index_id or ""),
            "index_type": str(retrieval_index_type or ""),
            "index_text": str(retrieval_index_text or ""),
            "index_weight": float(retrieval_index_weight or 1.0),
            "retrieval_index_id": str(retrieval_index_id or ""),
            "retrieval_index_type": str(retrieval_index_type or ""),
            "retrieval_index_text": str(retrieval_index_text or ""),
            "retrieval_index_weight": float(retrieval_index_weight or 1.0),
            "retrieval_index_enabled_routes": list(retrieval_index_enabled_routes),
            "matched_index_id": str(retrieval_index_id or ""),
            "matched_index_type": str(retrieval_index_type or ""),
            "matched_index_text": str(retrieval_index_text or ""),
            "matched_index_score": matched_index_score,
            "chunk_type": str(chunk_type or "text"),
            "asset_kind": str(asset_kind or ""),
            "asset_path": str(asset_path or ""),
            "asset_abs_path": str(asset_abs_path or ""),
            "asset_summary": str(asset_summary or ""),
            "asset_preview_text": str(asset_preview_text or ""),
            "asset_caption": str(asset_caption or ""),
            "asset_section_match_type": str(asset_section_match_type or ""),
            "asset_section_match_confidence": float(asset_section_match_confidence or 0.0),
            "asset_section_match_reason": str(asset_section_match_reason or ""),
            "asset_section_match_is_heuristic": bool(asset_section_match_is_heuristic),
            "asset_section_match_allow_embedding": bool(asset_section_match_allow_embedding),
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

        # 顶层字段直接取自 metadata，保证两处取值永远一致；键顺序与历史响应保持不变。
        payload = {
            "text": content,
            "content": content,
            **{key: metadata[key] for key in _CHUNK_PAYLOAD_KEYS_BEFORE_SCORE},
            "score": score,
            "distance": distance,
            **{key: metadata[key] for key in _CHUNK_PAYLOAD_KEYS_AFTER_SCORE},
            "metadata": metadata,
        }
        return payload

    def _build_paper_payload(self, entity: Any, score: Optional[float], distance: Optional[float]) -> Dict[str, Any]:
        """把论文级实体转换为论文卡片结构；摘要缺失时用 content 兜底，作者/分类拆成列表。"""
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

        getter = _entity_getter(entity)

        authors_raw = getter("authors", "") or ""
        categories_raw = getter("categories", "") or ""
        abstract = getter("abstract", "") or getter("content", "") or ""
        published_date = getter("published_date", "") or ""

        payload = {
            "arxiv_id": str(getter("arxiv_id", "") or ""),
            "title": str(getter("title", "") or ""),
            "authors": _split_display_values(authors_raw),
            "abstract": str(abstract),
            "categories": _split_display_values(categories_raw),
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
        """把论文向量实体转换为可复用的 embedding 记录（含原始向量）。"""
        getter = _entity_getter(entity)

        vector_value = getter("vector", []) or []
        # 部分 pymilvus 版本会把向量包成二维 [[...]]，这里取第一行还原为一维向量。
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
