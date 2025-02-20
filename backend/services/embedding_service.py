import os
import dotenv
dotenv.load_dotenv()
import json
from datetime import datetime
from enum import Enum
import boto3
from langchain_community.embeddings import BedrockEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"

class EmbeddingConfig:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        self.aws_region = "ap-southeast-1"  # 可配置

class EmbeddingService:
    def __init__(self):
        self.embedding_factory = EmbeddingFactory()

    def create_embeddings(self, input_data: dict, config: EmbeddingConfig) -> tuple:
        """创建embeddings并返回必要的信息"""
        embedding_function = self.embedding_factory.create_embedding_function(config)
        
        chunks = input_data.get('chunks', [])
        filename = input_data.get('metadata', {}).get('filename', '')  # 获取文件名
        
        # 批处理大小
        BATCH_SIZE = 20
        results = []
        
        # 如果是OpenAI，使用批处理
        if config.provider == EmbeddingProvider.OPENAI:
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                # 提取当前批次的文本内容
                texts = [chunk.get("content", "") for chunk in batch]
                
                # 批量获取embeddings
                embedding_vectors = embedding_function.embed_documents(texts)
                
                # 将结果与原始chunk数据组合
                for chunk, embedding_vector in zip(batch, embedding_vectors):
                    metadata = {
                        "chunk_id": chunk["metadata"]["chunk_id"],
                        "page_number": chunk["metadata"]["page_number"],
                        "page_range": chunk["metadata"]["page_range"],
                        "content": chunk["content"],
                        "word_count": chunk["metadata"]["word_count"],
                        # "chunking_method": input_data.get("chunking_method", "loaded"),
                        "total_chunks": len(chunks),
                        "embedding_provider": config.provider,
                        "embedding_model": config.model_name,
                        "embedding_timestamp": datetime.now().isoformat(),
                        "vector_dimension": len(embedding_vector),
                        "filename": filename  # 添加文件名到metadata
                    }
                    
                    embedding_result = {
                        "embedding": embedding_vector,
                        "metadata": metadata
                    }
                    results.append(embedding_result)
        else:
            # 对其他提供商保持原有的逐个处理逻辑
            for chunk in chunks:
                embedding_vector = embedding_function.embed_query(chunk["content"])
                metadata = {
                    "chunk_id": chunk["metadata"]["chunk_id"],
                    "page_number": chunk["metadata"]["page_number"],
                    "page_range": chunk["metadata"]["page_range"],
                    "content": chunk["content"],
                    "word_count": chunk["metadata"]["word_count"],
                    # "chunking_method": input_data.get("chunking_method", "loaded"),
                    "total_chunks": len(chunks),
                    "embedding_provider": config.provider,
                    "embedding_model": config.model_name,
                    "embedding_timestamp": datetime.now().isoformat(),
                    "vector_dimension": len(embedding_vector),
                    "filename": filename  # 添加文件名到metadata
                }
                
                embedding_result = {
                    "embedding": embedding_vector,
                    "metadata": metadata
                }
                results.append(embedding_result)
        
        # 返回结果和空的metadata（因为metadata已经包含在每个embedding中）
        return results, {}

    def save_embeddings(self, doc_name: str, embeddings: list) -> str:
        """保存embeddings并在顶层保存配置信息"""
        os.makedirs("02-embedded-docs", exist_ok=True)
        
        # 获取第一个embedding的元数据
        first_embedding = embeddings[0]
        provider = first_embedding["metadata"]["embedding_provider"]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 保持原始文件名（包括扩展名）
        base_name = doc_name.split('_')[0]
        if not base_name.endswith('.pdf'):
            base_name += '.pdf'
        
        # 构建新的文件名：基础名称_provider_时间戳
        filename = f"{base_name.replace('.pdf', '')}_{provider}_{timestamp}.json"
        filepath = os.path.join("02-embedded-docs", filename)
        
        # 从第一个embedding中获取配置信息
        config_info = {
            "filename": base_name,  # 使用完整的文件名（包括.pdf）
            "chunked_doc_name": doc_name,  # Add chunked_doc_name
            "created_at": datetime.now().isoformat(),
            "embedding_provider": provider,
            "embedding_model": first_embedding["metadata"]["embedding_model"],
            "vector_dimension": first_embedding["metadata"]["vector_dimension"]
        }
        
        class CompactJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
            
            def encode(self, obj):
                # 将 embedding 数组转换为单行，其他保持格式化
                def format_list(lst):
                    if isinstance(lst, list):
                        # 检查是否为 embedding 数组（通过检查第一个元素是否为数字）
                        if lst and isinstance(lst[0], (int, float)):
                            return '[' + ','.join(map(str, lst)) + ']'
                        return [format_list(item) for item in lst]
                    elif isinstance(lst, dict):
                        return {k: format_list(v) for k, v in lst.items()}
                    return lst
                
                return super().encode(format_list(obj))
        
        # 保存数据，配置信息放在顶层
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                **config_info,  # 配置信息放在顶层
                "embeddings": embeddings
            }, f, ensure_ascii=False, indent=2, cls=CompactJSONEncoder)
            
        return filepath

    def create_single_embedding(self, text: str, provider: str, model: str) -> list:
        """创建单个文本的embedding"""
        config = EmbeddingConfig(provider=provider, model_name=model)
        embedding_function = self.embedding_factory.create_embedding_function(config)
        return embedding_function.embed_query(text)

    def get_document_embedding_config(self, collection_name: str) -> EmbeddingConfig:
        """从已存在的文档中获取embedding配置"""
        try:
            # 只取第一个下划线之前的部分
            doc_name = collection_name.split('_')[0]
            
            # 查找对应的embedding文件
            embedded_docs_dir = "02-embedded-docs"
            for filename in os.listdir(embedded_docs_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(embedded_docs_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 使用 filename 而不是 document_name
                        if data.get("filename") == doc_name:
                            return EmbeddingConfig(
                                provider=data.get("embedding_provider"),
                                model_name=data.get("embedding_model")
                            )
                            
            raise ValueError(f"No matching embedding configuration found for collection: {collection_name}")
        except Exception as e:
            raise ValueError(f"Error getting embedding config: {str(e)}")

class EmbeddingFactory:
    @staticmethod
    def create_embedding_function(config: EmbeddingConfig):
        if config.provider == EmbeddingProvider.BEDROCK:
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=config.aws_region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            return BedrockEmbeddings(
                client=bedrock_client,
                model_id=config.model_name
            )
            
        elif config.provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbeddings(
                model=config.model_name,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
        elif config.provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbeddings(
                model_name=config.model_name
            )
            
        raise ValueError(f"Unsupported embedding provider: {config.provider}")