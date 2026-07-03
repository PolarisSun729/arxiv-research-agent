#!/usr/bin/env python3
"""
重构 arxiv_oai_service.py 的完整脚本
"""
import re

def refactor_arxiv_oai_service():
    """执行完整的重构"""

    with open('services/arxiv/arxiv_oai_service.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 更新 imports
    old_imports = """import requests

from services.arxiv.contracts import ArxivSearchError"""

    new_imports = """import requests

from services.arxiv.arxiv_embedding_handler import ArxivEmbeddingHandler, EmbeddingStats
from services.arxiv.arxiv_oai_xml_parser import ArxivOaiXmlParser
from services.arxiv.arxiv_query_compiler import ArxivQueryCompiler
from services.arxiv.arxiv_query_parser import ArxivQueryParser, LOCAL_OAI_SUPPORTED_QUERY_SUBSET
from services.arxiv.contracts import ArxivSearchError"""

    content = content.replace(old_imports, new_imports)

    # 2. 删除 LOCAL_OAI_SUPPORTED_QUERY_SUBSET 定义
    content = re.sub(
        r'\nLOCAL_OAI_SUPPORTED_QUERY_SUBSET = \[\n(?:    .*\n)*\]\n',
        '\n',
        content
    )

    # 3. 在 ArxivOaiDatabaseService.__init__ 中添加 parser 和 compiler
    content = re.sub(
        r'(        self\.check_same_thread = \(\n            OAI_SQLITE_CONFIG\["check_same_thread"\] if check_same_thread is None else bool\(check_same_thread\)\n        \))\n(        # 目录和表结构)',
        r'\1\n        # 初始化辅助组件\n        self.query_parser = ArxivQueryParser()\n        self.query_compiler = ArxivQueryCompiler()\n\2',
        content
    )

    # 4. 替换 search 方法中的调用
    content = re.sub(
        r'query_node = self\._parse_local_oai_query\(normalized_query\)',
        r'query_node = self.query_parser.parse(normalized_query)',
        content
    )

    content = re.sub(
        r'query_sql, query_params, uses_fts = self\._compile_local_oai_query\(',
        r'query_sql, query_params, uses_fts = self.query_compiler.compile(',
        content
    )

    # 5. 删除查询解析和编译方法
    methods_to_remove = [
        r'    def _strip_query_outer_parentheses\(.*?\n(?:        .*\n)*?        return text\n\n',
        r'    def _split_query_top_level\(.*?\n(?:        .*\n)*?        return \[part for part in parts if part\]\n\n',
        r'    def _parse_submitted_date_range\(.*?\n(?:        .*\n)*?        return \{"type": "date", "start": start_dt, "end": end_dt\}\n\n',
        r'    def _parse_local_oai_atomic_query\(.*?\n(?:        .*\n)*?        return \{"type": "text", "field": field, "value": value, "phrase": phrase\}\n\n',
        r'    def _parse_local_oai_query\(.*?\n(?:        .*\n)*?        return self\._parse_local_oai_atomic_query\(text, original_query=query\)\n\n',
        r'    def _fts_query_for_text\(.*?\n(?:        .*\n)*?        return " AND "\.join\(tokens\)\n\n',
        r'    def _compile_local_oai_query\(.*?\n(?:        .*\n)*?            reason=f"unsupported_node:\{node_type\}",\n        \)\n\n',
    ]

    for pattern in methods_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # 6. 更新 ArxivOaiSyncService.__init__
    content = re.sub(
        r'(        self\.database_service = database_service)\n        self\.embedding_service = embedding_service or EmbeddingService\(\)\n        self\.vector_store_service = vector_store_service or VectorStoreService\(\)\n        self\.embedding_collection_name = embedding_collection_name\n        self\.embedding_config = self\.embedding_service\.get_default_embedding_config\(\)',
        r'\1\n        # 初始化 XML 解析器\n        self.xml_parser = ArxivOaiXmlParser()\n        # 初始化 embedding 处理器\n        self.embedding_handler = None\n        if embedding_service or vector_store_service:\n            self.embedding_handler = ArxivEmbeddingHandler(\n                embedding_service=embedding_service or EmbeddingService(),\n                vector_store_service=vector_store_service or VectorStoreService(),\n                embedding_collection_name=embedding_collection_name,\n                embedding_batch_size=OAI_EMBEDDING_BATCH_SIZE,\n                vector_query_batch_size=OAI_VECTOR_QUERY_BATCH_SIZE,\n                token_price_per_1k=OAI_DASHSCOPE_TEXT_TOKEN_PRICE_PER_1K,\n            )',
        content
    )

    # 7. 更新 _persist_matched_papers
    content = re.sub(
        r'        # 只对真正落库成功的论文生成 embedding，避免向量库和主库出现悬空记录。\n        self\._store_paper_embeddings_batch\(persisted_papers, stats\)',
        r'''        # 只对真正落库成功的论文生成 embedding，避免向量库和主库出现悬空记录。
        if persisted_papers and self.embedding_handler:
            embedding_stats = EmbeddingStats(
                embeddings_attempted=stats.embeddings_attempted,
                embeddings_written=stats.embeddings_written,
                embeddings_skipped_existing=stats.embeddings_skipped_existing,
                embedding_errors=stats.embedding_errors,
                embedding_input_tokens=stats.embedding_input_tokens,
                embedding_output_tokens=stats.embedding_output_tokens,
                embedding_total_tokens=stats.embedding_total_tokens,
                embedding_cost_yuan=stats.embedding_cost_yuan,
            )
            embedding_stats = self.embedding_handler.store_paper_embeddings_batch(persisted_papers, embedding_stats)
            # 同步回主统计对象
            stats.embeddings_attempted = embedding_stats.embeddings_attempted
            stats.embeddings_written = embedding_stats.embeddings_written
            stats.embeddings_skipped_existing = embedding_stats.embeddings_skipped_existing
            stats.embedding_errors = embedding_stats.embedding_errors
            stats.embedding_input_tokens = embedding_stats.embedding_input_tokens
            stats.embedding_output_tokens = embedding_stats.embedding_output_tokens
            stats.embedding_total_tokens = embedding_stats.embedding_total_tokens
            stats.embedding_cost_yuan = embedding_stats.embedding_cost_yuan''',
        content
    )

    # 8. 删除 embedding 相关方法
    embedding_methods = [
        r'    def _build_oai_embedding_metadata\(.*?\n(?:        .*\n)*?        \}\n\n',
        r'    def _store_paper_embeddings_batch\(.*?\n(?:        .*\n)*?                self\._maybe_store_paper_embedding\(paper, stats\)\n\n',
        r'    def _accumulate_embedding_usage\(.*?\n(?:        .*\n)*?        \)\n\n',
        r'    def _maybe_store_paper_embedding\(.*?\n(?:        .*\n)*?            logger\.warning\("Failed to embed OAI paper %s into vector store: %s", arxiv_id, exc\)\n\n',
    ]

    for pattern in embedding_methods:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # 9. 替换 XML 解析方法调用
    xml_replacements = [
        (r'self\._parse_paper_metadata\(', r'self.xml_parser.parse_paper_metadata('),
        (r'self\._extract_resumption_token\(', r'self.xml_parser.extract_resumption_token('),
        (r'self\._extract_oai_error\(', r'self.xml_parser.extract_oai_error('),
        (r'self\._find_first_child\(', r'self.xml_parser.find_first_child('),
        (r'self\._find_first_element_child\(', r'self.xml_parser.find_first_element_child('),
    ]

    for old, new in xml_replacements:
        content = re.sub(old, new, content)

    # 10. 删除 XML 解析相关方法
    xml_methods = [
        r'    def _parse_paper_metadata\(.*?\n(?:        .*\n)*?            return None\n\n',
        r'    def _extract_authors\(.*?\n(?:        .*\n)*?        return \[author for author in authors if author\]\n\n',
        r'    def _extract_primary_category\(.*?\n(?:        .*\n)*?        return self\._normalize_whitespace\(primary_category\.text or ""\)\n\n',
        r'    def _extract_resumption_token\(.*?\n(?:        .*\n)*?        return token_elem\.text\.strip\(\)\n\n',
        r'    def _extract_oai_error\(.*?\n(?:        .*\n)*?        return code or text\n\n',
        r'    def _split_categories\(.*?\n(?:        .*\n)*?        return \[item for item in \(part\.strip\(\) for part in raw_items\) if item\]\n\n',
        r'    def _normalize_arxiv_id\(.*?\n(?:        .*\n)*?        return value\n\n',
        r'    def _find_first_child\(.*?\n(?:        .*\n)*?        return None\n\n',
        r'    def _find_first_element_child\(.*?\n(?:        .*\n)*?        return None\n\n',
        r'    def _find_first_descendant\(.*?\n(?:        .*\n)*?        return None\n\n',
        r'    def _extract_text\(.*?\n(?:        .*\n)*?        return element\.text\.strip\(\)\n\n',
        r'    def _local_name\(.*?\n(?:        .*\n)*?        return tag\n\n',
        r'    def _normalize_whitespace\(.*?\n(?:        .*\n)*?        return " "\.join\(value\.split\(\)\)\.strip\(\)\n',
    ]

    for pattern in xml_methods:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    with open('services/arxiv/arxiv_oai_service.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("重构完成！")
    print(f"原文件约 2241 行")
    print(f"新文件约 {len(content.splitlines())} 行")

if __name__ == '__main__':
    refactor_arxiv_oai_service()
