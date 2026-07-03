#!/usr/bin/env python3
"""
完整的 arxiv_oai_service.py 重构脚本
分步骤精确处理，避免编码问题
"""
import re
import sys

def step1_update_imports(content):
    """更新导入语句"""
    old = "from services.arxiv.contracts import ArxivSearchError"
    new = """from services.arxiv.arxiv_embedding_handler import ArxivEmbeddingHandler, EmbeddingStats
from services.arxiv.arxiv_oai_xml_parser import ArxivOaiXmlParser
from services.arxiv.arxiv_query_compiler import ArxivQueryCompiler
from services.arxiv.arxiv_query_parser import ArxivQueryParser, LOCAL_OAI_SUPPORTED_QUERY_SUBSET
from services.arxiv.contracts import ArxivSearchError"""

    return content.replace(old, new)

def step2_remove_local_subset(content):
    """删除 LOCAL_OAI_SUPPORTED_QUERY_SUBSET 定义"""
    # 精确匹配整个定义块
    pattern = r'\nLOCAL_OAI_SUPPORTED_QUERY_SUBSET = \[\n(?:    ".*",?\n)+\]\n'
    content = re.sub(pattern, '\n', content)
    return content

def step3_add_parser_compiler_init(content):
    """在 ArxivOaiDatabaseService.__init__ 中添加初始化"""
    # 在 check_same_thread 赋值后添加
    old = '''        self.check_same_thread = (
            OAI_SQLITE_CONFIG["check_same_thread"] if check_same_thread is None else bool(check_same_thread)
        )
        # 目录和表结构在构造时就确保到位'''

    new = '''        self.check_same_thread = (
            OAI_SQLITE_CONFIG["check_same_thread"] if check_same_thread is None else bool(check_same_thread)
        )
        # 初始化辅助组件
        self.query_parser = ArxivQueryParser()
        self.query_compiler = ArxivQueryCompiler()
        # 目录和表结构在构造时就确保到位'''

    return content.replace(old, new)

def step4_update_search_calls(content):
    """更新 search 方法中的调用"""
    # 替换 _parse_local_oai_query
    content = content.replace(
        'self._parse_local_oai_query(normalized_query)',
        'self.query_parser.parse(normalized_query)'
    )

    # 替换 _compile_local_oai_query
    content = content.replace(
        'self._compile_local_oai_query(',
        'self.query_compiler.compile('
    )

    return content

def step5_update_sync_service_init(content):
    """更新 ArxivOaiSyncService.__init__"""
    old = '''        self.database_service = database_service
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.embedding_collection_name = embedding_collection_name
        self.embedding_config = self.embedding_service.get_default_embedding_config()'''

    new = '''        self.database_service = database_service
        # 初始化 XML 解析器
        self.xml_parser = ArxivOaiXmlParser()
        # 初始化 embedding 处理器
        self.embedding_handler = None
        if embedding_service or vector_store_service:
            self.embedding_handler = ArxivEmbeddingHandler(
                embedding_service=embedding_service or EmbeddingService(),
                vector_store_service=vector_store_service or VectorStoreService(),
                embedding_collection_name=embedding_collection_name,
                embedding_batch_size=OAI_EMBEDDING_BATCH_SIZE,
                vector_query_batch_size=OAI_VECTOR_QUERY_BATCH_SIZE,
                token_price_per_1k=OAI_DASHSCOPE_TEXT_TOKEN_PRICE_PER_1K,
            )'''

    return content.replace(old, new)

def step6_update_persist_method(content):
    """更新 _persist_matched_papers 方法"""
    old = '''        # 只对真正落库成功的论文生成 embedding，避免向量库和主库出现悬空记录。
        self._store_paper_embeddings_batch(persisted_papers, stats)'''

    new = '''        # 只对真正落库成功的论文生成 embedding，避免向量库和主库出现悬空记录。
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
            stats.embedding_cost_yuan = embedding_stats.embedding_cost_yuan'''

    return content.replace(old, new)

def step7_update_xml_calls(content):
    """更新 XML 解析方法调用"""
    replacements = [
        ('self._parse_paper_metadata(', 'self.xml_parser.parse_paper_metadata('),
        ('self._extract_resumption_token(', 'self.xml_parser.extract_resumption_token('),
        ('self._extract_oai_error(', 'self.xml_parser.extract_oai_error('),
        ('self._find_first_child(', 'self.xml_parser.find_first_child('),
        ('self._find_first_element_child(', 'self.xml_parser.find_first_element_child('),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    return content

def step8_remove_migrated_methods(content):
    """删除已迁移的方法"""
    lines = content.split('\n')

    methods_to_remove = {
        '_strip_query_outer_parentheses',
        '_split_query_top_level',
        '_parse_submitted_date_range',
        '_parse_local_oai_atomic_query',
        '_parse_local_oai_query',
        '_fts_query_for_text',
        '_compile_local_oai_query',
        '_parse_paper_metadata',
        '_extract_authors',
        '_extract_primary_category',
        '_extract_resumption_token',
        '_extract_oai_error',
        '_split_categories',
        '_normalize_arxiv_id',
        '_find_first_child',
        '_find_first_element_child',
        '_find_first_descendant',
        '_extract_text',
        '_local_name',
        '_normalize_whitespace',
        '_build_oai_embedding_metadata',
        '_store_paper_embeddings_batch',
        '_accumulate_embedding_usage',
        '_maybe_store_paper_embedding',
    }

    new_lines = []
    skip = False
    method_indent = 0

    for line in lines:
        # 检查是否是方法定义
        if line.strip().startswith('def '):
            match = re.search(r'def (_[a-z_]+)\(', line)
            if match and match.group(1) in methods_to_remove:
                skip = True
                method_indent = len(line) - len(line.lstrip())
                print(f"Removing method: {match.group(1)}")
                continue
            else:
                skip = False

        # 如果在跳过模式
        if skip:
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                # 遇到同级或更外层的定义，停止跳过
                if current_indent <= method_indent and (line.strip().startswith('def ') or line.strip().startswith('class ')):
                    skip = False
                    new_lines.append(line)
                    continue
            continue

        new_lines.append(line)

    return '\n'.join(new_lines)

def main():
    print("Starting refactoring...")

    # 读取原文件
    with open('services/arxiv/arxiv_oai_service.py', 'r', encoding='utf-8') as f:
        content = f.read()

    original_lines = len(content.split('\n'))
    print(f"Original file: {original_lines} lines")

    # 执行各步骤
    print("\nStep 1: Updating imports...")
    content = step1_update_imports(content)

    print("Step 2: Removing LOCAL_OAI_SUPPORTED_QUERY_SUBSET...")
    content = step2_remove_local_subset(content)

    print("Step 3: Adding parser and compiler initialization...")
    content = step3_add_parser_compiler_init(content)

    print("Step 4: Updating search method calls...")
    content = step4_update_search_calls(content)

    print("Step 5: Updating sync service initialization...")
    content = step5_update_sync_service_init(content)

    print("Step 6: Updating persist method...")
    content = step6_update_persist_method(content)

    print("Step 7: Updating XML method calls...")
    content = step7_update_xml_calls(content)

    print("Step 8: Removing migrated methods...")
    content = step8_remove_migrated_methods(content)

    # 写回文件
    with open('services/arxiv/arxiv_oai_service.py', 'w', encoding='utf-8') as f:
        f.write(content)

    new_lines = len(content.split('\n'))
    print(f"\nRefactoring complete!")
    print(f"New file: {new_lines} lines")
    print(f"Removed: {original_lines - new_lines} lines ({(original_lines - new_lines) / original_lines * 100:.1f}%)")

if __name__ == '__main__':
    main()
