"""arXiv OAI 数据同步与本地检索服务模块。

该模块负责从 arXiv OAI-PMH 接口同步论文元数据到本地 SQLite 数据库，并
提供围绕这份本地镜像数据的检索、过滤、解析和向量化辅助能力。它是本地
 arXiv 数据能力的核心实现。
"""

import json
import logging
import os
import re
import time
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from services.arxiv.contracts import ArxivSearchError
from services.embedding.embedding_service import EmbeddingService
from services.storage.vector_store_service import VectorStoreService
from utils.config import OAI_SQLITE_CONFIG, get_arxiv_oai_runtime_config

logger = logging.getLogger(__name__)

ARXIV_OAI_CONFIG = get_arxiv_oai_runtime_config()
OAI_ENDPOINT = ARXIV_OAI_CONFIG["endpoint"]
TARGET_CATEGORIES = set(ARXIV_OAI_CONFIG["target_categories"])
OAI_EMBEDDING_BATCH_SIZE = ARXIV_OAI_CONFIG["embedding_batch_size"]
OAI_VECTOR_QUERY_BATCH_SIZE = ARXIV_OAI_CONFIG["vector_query_batch_size"]
OAI_DASHSCOPE_TEXT_TOKEN_PRICE_PER_1K = ARXIV_OAI_CONFIG["dashscope_text_token_price_per_1k"]

LOCAL_OAI_SUPPORTED_QUERY_SUBSET = [
    "id",
    "cat",
    "submittedDate",
    "ti",
    "abs",
    "au",
    "all",
    "AND",
    "OR",
    "ANDNOT",
    "phrase",
]


class LocalArxivSearchError(ArxivSearchError):
    """本地 OAI 检索的稳定错误基类，用于向工具层和前端暴露可解释失败。"""

    code = "local_arxiv_search_error"
    status_code = 400

    def __init__(self, message: str, *, query: str = "", reason: str = ""):
        """构造带查询上下文的稳定错误对象。

        这里额外缓存 query / reason / query_capability，是为了让上层在不解析
        日志文本的前提下，也能把“为什么失败、接下来怎么改写查询”直接返回给前端。
        """
        super().__init__(message)
        self.query = query
        self.reason = reason or message
        # 错误对象内直接挂能力描述，保证任意调用方都能拿到一致的兜底说明。
        self.query_capability = build_local_oai_query_capability(
            mode="unsupported",
            unsupported_reason=self.reason,
            suggested_action="切换远程 arXiv API 后重试，或改写为本地 OAI 镜像支持的高精度查询子集。",
        )

    def to_error_detail(self) -> Dict[str, Any]:
        """把异常转换成稳定的结构化错误详情。"""
        # 统一错误载荷格式，避免路由层和前端分别猜测字段命名。
        return {
            "code": self.code,
            "message": str(self),
            "details": {
                "query": self.query,
                "source": "local_oai",
                "reason": self.reason,
                "query_capability": self.query_capability,
                "suggested_action": self.query_capability.get("suggested_action"),
            },
        }


class UnsupportedLocalArxivQuery(LocalArxivSearchError):
    """查询语法超出本地 OAI 可数据库下推子集时抛出，避免静默误召回。"""

    code = "unsupported_local_arxiv_query"


class LocalArxivSearchIndexUnavailable(LocalArxivSearchError):
    """FTS5 不可用或索引未就绪时抛出，禁止退回低精度 LIKE 兜底。"""

    code = "local_search_index_unavailable"


def build_local_oai_query_capability(
    *,
    mode: str = "local_oai_sqlite_fts",
    unsupported_reason: Optional[str] = None,
    suggested_action: Optional[str] = None,
    fts5_available: Optional[bool] = None,
    search_index_status: Optional[str] = None,
) -> Dict[str, Any]:
    """构造后端和前端共享的本地检索能力契约。

    这个结构会同时出现在成功响应和失败响应里，用来明确说明当前本地 OAI
    搜索支持哪些语法、依赖了哪些索引能力，以及用户下一步应该如何改写查询。
    """

    # supported_subset 是对外契约的一部分，前端可以直接据此决定提示文案和交互限制。
    capability = {
        "source": "local_oai",
        "mode": mode,
        "supported_subset": list(LOCAL_OAI_SUPPORTED_QUERY_SUBSET),
        "precision_policy": "strict_token_or_phrase",
        "full_arxiv_syntax_supported": False,
        "unsupported_reason": unsupported_reason,
        "suggested_action": suggested_action,
    }
    if fts5_available is not None:
        capability["fts5_available"] = bool(fts5_available)
    if search_index_status:
        capability["search_index_status"] = search_index_status
    return capability


@dataclass
class ArxivOaiSyncStats:
    """记录一次 OAI 同步任务的统计信息。"""
    requests_made: int = 0
    pages_processed: int = 0
    records_seen: int = 0
    records_with_metadata: int = 0
    records_matched: int = 0
    records_written: int = 0
    records_skipped: int = 0
    skipped_no_metadata: int = 0
    skipped_no_arxiv_id: int = 0
    skipped_no_categories: int = 0
    skipped_category_filter: int = 0
    skipped_parse_errors: int = 0
    embeddings_attempted: int = 0
    embeddings_written: int = 0
    embeddings_skipped_existing: int = 0
    embedding_errors: int = 0
    embedding_input_tokens: int = 0
    embedding_output_tokens: int = 0
    embedding_total_tokens: int = 0
    embedding_cost_yuan: float = 0.0
    sync_duration_seconds: float = 0.0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """把同步统计对象转换成普通字典，便于日志和接口直接序列化。"""
        # 统一通过 dataclass 导出，避免手写字段列表和统计结构漂移。
        return asdict(self)


class ArxivOaiDatabaseService:
    def __init__(self, db_path: Optional[str] = None, check_same_thread: Optional[bool] = None):
        """初始化 OAI SQLite 数据库服务。

        参数:
            db_path (Optional[str]): 数据库文件路径；为空时使用运行时配置。
            check_same_thread (Optional[bool]): SQLite 线程检查配置。

        返回:
            None
        """
        # 优先复用运行时配置，避免数据库路径和线程策略在调用方各自散落。
        self.db_path = db_path or OAI_SQLITE_CONFIG["database_path"]
        self.check_same_thread = (
            OAI_SQLITE_CONFIG["check_same_thread"] if check_same_thread is None else bool(check_same_thread)
        )
        # 目录和表结构在构造时就确保到位，调用方无需感知“第一次使用”的初始化细节。
        self._ensure_database_directory()
        self._initialize_database()

    def _ensure_database_directory(self) -> None:
        """确保数据库目录存在，避免首次写入时路径缺失。"""
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            # SQLite 不会自动创建父目录；这里前置处理能把失败点收敛到初始化阶段。
            os.makedirs(directory, exist_ok=True)
            logger.info("Created OAI database directory: %s", directory)

    def _get_connection(self) -> sqlite3.Connection:
        """创建一个 SQLite 连接。

        这里不做长连接缓存，是为了减少跨线程复用连接带来的状态污染和锁竞争。
        """
        # 每次按需创建连接，让上层通过上下文管理器控制事务边界。
        return sqlite3.connect(self.db_path, check_same_thread=self.check_same_thread)

    def _is_fts5_available(self, cursor: sqlite3.Cursor) -> bool:
        """检测当前 SQLite 是否支持 FTS5；本地文本检索依赖它来保证精确且可下推。"""
        try:
            # 用临时虚表探测真实运行时能力，比只看 SQLite 版本更可靠。
            cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.__oai_fts5_probe USING fts5(value)")
            cursor.execute("DROP TABLE IF EXISTS temp.__oai_fts5_probe")
            return True
        except sqlite3.Error as exc:
            logger.warning("SQLite FTS5 is unavailable for local OAI search: %s", exc)
            return False

    def _normalize_categories_for_index(self, paper: Dict[str, Any]) -> List[str]:
        """提取规范化分类列表，用独立表做精确过滤，避免 JSON 文本 LIKE 误命中。"""
        raw_categories = paper.get("categories_list")
        if not raw_categories:
            # 兼容历史数据：老数据可能只存了 JSON 字符串形式的 categories。
            raw_categories = self._parse_list_field(paper.get("categories", ""))

        if isinstance(raw_categories, (list, tuple)):
            categories = [str(item).strip() for item in raw_categories if str(item).strip()]
        else:
            categories = [
                item.strip()
                for item in re.split(r"[\s,]+", str(raw_categories or "").strip())
                if item.strip()
            ]

        primary_category = str(paper.get("primary_category") or "").strip()
        if primary_category and primary_category not in categories:
            # 主分类是很多业务过滤和展示的关键字段，缺失时需要补回索引数据。
            categories.append(primary_category)
        return categories

    def _normalize_authors_for_index(self, paper: Dict[str, Any]) -> str:
        """把作者字段整理成 FTS 可索引文本，兼容历史 JSON 字符串和列表两种形态。"""
        authors = self._parse_list_field(paper.get("authors", ""))
        if isinstance(authors, (list, tuple)):
            # FTS 侧只需要可检索的纯文本，不保留原始 JSON 结构。
            return " ".join(str(item).strip() for item in authors if str(item).strip())
        return str(authors or "").strip()

    def _sync_search_index_for_papers(
        self,
        cursor: sqlite3.Cursor,
        papers: Sequence[Dict[str, Any]],
        *,
        fts5_available: Optional[bool] = None,
    ) -> None:
        """同步分类映射表和 FTS 表，保证新增/更新论文不再依赖内存过滤。"""
        if not papers:
            return
        if fts5_available is None:
            fts5_available = self._is_fts5_available(cursor)

        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id") or "").strip()
            if not arxiv_id:
                continue

            categories = self._normalize_categories_for_index(paper)
            primary_category = str(paper.get("primary_category") or "").strip()
            # 采用“先删后插”的方式刷新从表，避免分类变更后残留脏索引。
            cursor.execute("DELETE FROM arxiv_oai_paper_categories WHERE arxiv_id = ?", (arxiv_id,))
            cursor.executemany(
                """
                INSERT OR REPLACE INTO arxiv_oai_paper_categories (arxiv_id, category, is_primary)
                VALUES (?, ?, ?)
                """,
                [(arxiv_id, category, 1 if category == primary_category else 0) for category in categories],
            )

            if not fts5_available:
                # 没有 FTS5 时仍要保留分类索引，让 id/category/date 查询继续可用。
                continue

            authors_text = self._normalize_authors_for_index(paper)
            categories_text = " ".join(categories)
            title = str(paper.get("title", "") or "").strip()
            abstract = str(paper.get("abstract", "") or "").strip()
            # all_text 聚合多个文本字段，给 all: 查询提供统一检索入口。
            all_text = " ".join(
                item
                for item in [title, abstract, authors_text, categories_text, primary_category]
                if item
            )
            # FTS 表同样走全量重建，确保字段更新时不会保留旧文本。
            cursor.execute("DELETE FROM arxiv_oai_papers_fts WHERE arxiv_id = ?", (arxiv_id,))
            cursor.execute(
                """
                INSERT INTO arxiv_oai_papers_fts (arxiv_id, title, abstract, authors, categories, all_text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (arxiv_id, title, abstract, authors_text, categories_text, all_text),
            )

    def _backfill_search_indexes(self, cursor: sqlite3.Cursor, *, fts5_available: bool) -> None:
        """启动时从旧主表回填新索引；该逻辑幂等，避免用户必须重新同步 OAI。"""
        cursor.execute("SELECT COUNT(*) FROM arxiv_oai_papers")
        total_papers = int((cursor.fetchone() or [0])[0] or 0)
        if total_papers <= 0:
            return

        cursor.execute("SELECT COUNT(*) FROM arxiv_oai_paper_categories")
        category_count = int((cursor.fetchone() or [0])[0] or 0)
        fts_count = 0
        if fts5_available:
            cursor.execute("SELECT COUNT(*) FROM arxiv_oai_papers_fts")
            fts_count = int((cursor.fetchone() or [0])[0] or 0)

        if category_count > 0 and (not fts5_available or fts_count > 0):
            # 分类索引和 FTS 索引都已存在时直接跳过，避免每次启动都重刷大表。
            return

        logger.info(
            "Backfilling local OAI search indexes: papers=%s categories=%s fts=%s",
            total_papers,
            category_count,
            fts_count,
        )
        cursor.execute(
            """
            SELECT
                arxiv_id,
                title,
                abstract,
                authors,
                categories,
                primary_category,
                created,
                updated,
                abs_url,
                pdf_url,
                oai_datestamp,
                fetched_at,
                created_at,
                updated_at
            FROM arxiv_oai_papers
            """
        )
        rows = cursor.fetchall()
        # 回填逻辑统一复用在线同步时的索引构建流程，减少两套实现漂移。
        self._sync_search_index_for_papers(
            cursor,
            [self._parse_row(row) for row in rows],
            fts5_available=fts5_available,
        )

    def rebuild_oai_search_index(self) -> Dict[str, Any]:
        """手动重建本地搜索索引；用于 FTS schema 变化或历史索引损坏后的修复。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            fts5_available = self._is_fts5_available(cursor)
            # 手动重建的语义是“按主表真相重刷所有从索引”，因此先清空再回填。
            cursor.execute("DELETE FROM arxiv_oai_paper_categories")
            if fts5_available:
                cursor.execute("DELETE FROM arxiv_oai_papers_fts")
            self._backfill_search_indexes(cursor, fts5_available=fts5_available)
            cursor.execute("SELECT COUNT(*) FROM arxiv_oai_paper_categories")
            category_rows = int((cursor.fetchone() or [0])[0] or 0)
            fts_rows = 0
            if fts5_available:
                cursor.execute("SELECT COUNT(*) FROM arxiv_oai_papers_fts")
                fts_rows = int((cursor.fetchone() or [0])[0] or 0)
            conn.commit()
        return {
            "source": "local_oai",
            "fts5_available": fts5_available,
            "category_rows": category_rows,
            "fts_rows": fts_rows,
            # 把重建后的能力状态一并返回，方便调用方立即更新 UI 提示。
            "query_capability": build_local_oai_query_capability(
                mode="local_oai_sqlite_fts" if fts5_available else "local_oai_sqlite_index",
                fts5_available=fts5_available,
                search_index_status="ready" if fts5_available else "text_index_unavailable",
            ),
        }

    def _parse_list_field(self, value: Any) -> Any:
        """把数据库中的列表字段还原成更自然的 Python 结构。"""
        if isinstance(value, (list, tuple)):
            # 调用方可能已经给了结构化列表，这里只做轻量规整。
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            if text.startswith("[") or text.startswith("("):
                try:
                    # 优先按 JSON 解析，兼容数据库里保存的 authors/categories 历史格式。
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    # 无法解析时保留原文，避免因为脏数据直接让读取流程失败。
                    return text
            return text
        return value

    def get_total_paper_count(self) -> int:
        """返回 OAI 本地镜像库中的论文总数。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 单独提供轻量计数接口，给状态页或诊断逻辑使用。
                cursor.execute('SELECT COUNT(*) FROM arxiv_oai_papers')
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as exc:
            logger.error("Error getting total OAI paper count: %s", exc)
            return 0

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """按 arXiv ID 获取单篇论文元数据。"""
        normalized_arxiv_id = str(arxiv_id or "").strip()
        if not normalized_arxiv_id:
            return None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT
                        arxiv_id,
                        title,
                        abstract,
                        authors,
                        categories,
                        primary_category,
                        created,
                        updated,
                        abs_url,
                        pdf_url,
                        oai_datestamp,
                        fetched_at,
                        created_at,
                        updated_at
                    FROM arxiv_oai_papers
                    WHERE arxiv_id = ?
                    ''',
                    (normalized_arxiv_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                # 读取时把 authors/categories 还原成更自然的结构，减少上层分支判断。
                return {
                    "arxiv_id": row[0],
                    "title": row[1],
                    "abstract": row[2],
                    "authors": self._parse_list_field(row[3]),
                    "categories": self._parse_list_field(row[4]),
                    "primary_category": row[5],
                    "created": row[6],
                    "updated": row[7],
                    "abs_url": row[8],
                    "pdf_url": row[9],
                    "oai_datestamp": row[10],
                    "fetched_at": row[11],
                    "created_at": row[12],
                    "updated_at": row[13],
                }
        except Exception as exc:
            logger.error("Error getting OAI paper %s: %s", normalized_arxiv_id, exc)
            return None

    def get_recent_papers(
        self,
        categories: Optional[Sequence[str]] = None,
        max_age_months: int = 6,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取最近若干个月内的论文列表，并可按分类过滤。"""
        try:
            query_parts = [
                '''
                SELECT
                    arxiv_id,
                    title,
                    abstract,
                    authors,
                    categories,
                    primary_category,
                    created,
                    updated,
                    abs_url,
                    pdf_url,
                    oai_datestamp,
                    fetched_at,
                    created_at,
                    updated_at
                FROM arxiv_oai_papers
                '''
            ]
            params: List[Any] = []
            where_clauses: List[str] = []

            if max_age_months > 0:
                # 最近论文视图更看重“可读的新鲜度”，因此按近似月窗口截断。
                cutoff_date = (datetime.utcnow() - timedelta(days=max_age_months * 30)).date().isoformat()
                where_clauses.append(
                    "COALESCE(created, updated, oai_datestamp, fetched_at) >= ?"
                )
                params.append(cutoff_date)

            normalized_categories = [
                str(category).strip()
                for category in (categories or [])
                if str(category).strip()
            ]
            if normalized_categories:
                category_filters = []
                for category in normalized_categories:
                    # 同时检查主分类和分类映射表，兼容 cross-list 以及主分类缺省场景。
                    category_filters.append(
                        "primary_category = ? OR EXISTS ("
                        "SELECT 1 FROM arxiv_oai_paper_categories c "
                        "WHERE c.arxiv_id = arxiv_oai_papers.arxiv_id AND c.category = ?"
                        ")"
                    )
                    params.extend([category, category])
                where_clauses.append(f"({' OR '.join(f'({item})' for item in category_filters)})")

            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))

            query_parts.append(
                '''
                ORDER BY
                    COALESCE(created, updated, oai_datestamp, fetched_at) DESC,
                    arxiv_id DESC
                LIMIT ?
                '''
            )
            # LIMIT 至少为 1，避免上层传入 0 时拼出没有意义的空页查询。
            params.append(max(1, int(max_results)))

            query = "\n".join(part.strip() for part in query_parts if part.strip())

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                # 这里直接返回标准论文结构，供推荐、诊断和工具层复用。
                return [
                    {
                        "arxiv_id": row[0],
                        "title": row[1],
                        "abstract": row[2],
                        "authors": self._parse_list_field(row[3]),
                        "categories": self._parse_list_field(row[4]),
                        "primary_category": row[5],
                        "created": row[6],
                        "updated": row[7],
                        "abs_url": row[8],
                        "pdf_url": row[9],
                        "oai_datestamp": row[10],
                        "fetched_at": row[11],
                        "created_at": row[12],
                        "updated_at": row[13],
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Error getting recent OAI papers: %s", exc)
            return []

    def _normalize_text_value(self, value: Optional[str]) -> str:
        """规范化文本值，折叠多余空白。"""
        # 旧版内存过滤逻辑依赖稳定文本形态，先在这里消除换行和重复空格差异。
        return " ".join(str(value or "").strip().split())

    def _strip_outer_parentheses(self, query: str) -> str:
        """移除查询字符串最外层成对括号。

        这是旧版内存查询解释器使用的轻量括号剥离逻辑，不处理引号语义，
        只服务于简单 AND/OR/ANDNOT 场景。
        """
        text = query.strip()
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            balanced = True
            for index, char in enumerate(text):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(text) - 1:
                        balanced = False
                        break
            if balanced and depth == 0:
                text = text[1:-1].strip()
            else:
                break
        return text

    def _split_top_level(self, query: str, token: str) -> List[str]:
        """在不破坏括号层级的前提下按顶层逻辑符切分查询字符串。"""
        text = query.strip()
        parts: List[str] = []
        depth = 0
        start = 0
        index = 0
        token_length = len(token)
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
                index += 1
                continue
            if char == ")":
                depth = max(depth - 1, 0)
                index += 1
                continue
            if depth == 0 and text.startswith(token, index):
                # 只允许顶层切分，避免把括号内子表达式提前拆散。
                parts.append(text[start:index].strip())
                index += token_length
                start = index
                continue
            index += 1
        parts.append(text[start:].strip())
        return [part for part in parts if part]

    def _parse_row(self, row: Sequence[Any]) -> Dict[str, Any]:
        """把数据库查询返回的一行记录转换为标准论文字典。"""
        # 统一行转字典逻辑，避免不同查询路径各自手写字段映射。
        return {
            "arxiv_id": row[0],
            "title": row[1],
            "abstract": row[2],
            "authors": self._parse_list_field(row[3]),
            "categories": self._parse_list_field(row[4]),
            "primary_category": row[5],
            "created": row[6],
            "updated": row[7],
            "abs_url": row[8],
            "pdf_url": row[9],
            "oai_datestamp": row[10],
            "fetched_at": row[11],
            "created_at": row[12],
            "updated_at": row[13],
        }

    def _paper_to_search_fields(self, paper: Dict[str, Any]) -> Dict[str, str]:
        """把论文对象展开为便于搜索匹配的字段视图。

        旧版内存过滤和评分逻辑依赖这个字段视图，因此这里同时保留简写字段
        和别名字段，避免不同调用点重复展开 authors/categories。
        """
        authors = paper.get("authors", "")
        categories = paper.get("categories", "")
        if isinstance(authors, (list, tuple)):
            authors_text = ", ".join([str(item).strip() for item in authors if str(item).strip()])
        else:
            authors_text = str(authors or "")
        if isinstance(categories, (list, tuple)):
            categories_text = ", ".join([str(item).strip() for item in categories if str(item).strip()])
        else:
            categories_text = str(categories or "")
        return {
            "ti": str(paper.get("title", "") or "").lower(),
            "title": str(paper.get("title", "") or "").lower(),
            "au": authors_text.lower(),
            "authors": authors_text.lower(),
            "abs": str(paper.get("abstract", "") or "").lower(),
            "abstract": str(paper.get("abstract", "") or "").lower(),
            "cat": categories_text.lower(),
            "category": categories_text.lower(),
            "all": " ".join(
                [
                    str(paper.get("title", "") or ""),
                    authors_text,
                    str(paper.get("abstract", "") or ""),
                    categories_text,
                    str(paper.get("primary_category", "") or ""),
                ]
            ).lower(),
        }

    def _matches_submitted_date(self, paper: Dict[str, Any], query: str) -> bool:
        """判断论文时间是否命中 submittedDate 范围。

        这是旧版内存过滤的日期匹配实现。解析失败时默认返回 True，是为了避免
        因历史脏数据或格式差异把本可返回的论文静默过滤掉。
        """
        match = re.match(r"submittedDate:\[(\d{12})\s+TO\s+(\d{12})\]", query.strip())
        if not match:
            return True
        start_raw, end_raw = match.groups()
        # 时间字段优先选 updated，其次 created / OAI datestamp，保持和 SQL 路径一致。
        candidate = paper.get("updated") or paper.get("created") or paper.get("oai_datestamp") or ""
        if not candidate:
            return True
        candidate_text = str(candidate).strip()
        try:
            if len(candidate_text) >= 16 and "T" in candidate_text:
                candidate_dt = datetime.fromisoformat(candidate_text.replace("Z", "+00:00"))
            elif len(candidate_text) >= 16 and " " in candidate_text:
                candidate_dt = datetime.strptime(candidate_text[:16], "%Y-%m-%d %H:%M")
            else:
                candidate_dt = datetime.strptime(candidate_text[:10], "%Y-%m-%d")
            start_dt = datetime.strptime(start_raw, "%Y%m%d%H%M")
            end_dt = datetime.strptime(end_raw, "%Y%m%d%H%M")
            return start_dt <= candidate_dt.replace(tzinfo=None) <= end_dt
        except Exception:
            # 旧版回退匹配宁可放宽，也不在格式异常时制造“假阴性”。
            return True

    def _matches_text_query(self, haystack: str, query_text: str) -> bool:
        """判断字段文本是否命中查询词，避免短英文主题被当成任意子串。

        本地 OAI 搜索没有 arXiv API 的分词检索能力，因此需要在这里补一层
        轻量匹配语义：像 RAG 这类短英文主题词必须按完整 token 命中，防止
        误匹配到 Dragoi、Dragomir、granularity 这类作者名或普通单词片段。
        """
        normalized_query = str(query_text or "").strip().lower()
        if not normalized_query:
            return True
        normalized_haystack = str(haystack or "").lower()
        if not normalized_haystack:
            return False

        if re.fullmatch(r"[a-z0-9]+", normalized_query):
            # 纯英文/数字短词按 token 边界匹配，避免 RAG 命中 Dragoi 这类子串。
            return re.search(rf"(?<![a-z0-9]){re.escape(normalized_query)}(?![a-z0-9])", normalized_haystack) is not None
        return normalized_query in normalized_haystack

    def _score_text_query(self, paper: Dict[str, Any], query_text: str) -> int:
        """计算本地检索的轻量相关性分数，用于 relevance 排序。

        这里不做复杂 IR，只区分标题、摘要、分类、作者等字段的命中位置：
        标题和摘要更能代表论文主题，权重高于作者名，避免“作者名偶然包含关键词”
        的论文排在真正讨论该主题的论文前面。
        """
        normalized_query = str(query_text or "").strip().lower()
        if not normalized_query:
            return 0

        fields = self._paper_to_search_fields(paper)
        score = 0
        # 权重只表达“字段重要性”而非严格 IR 分值，保持可解释和低维护成本。
        weighted_fields = (
            ("title", 8),
            ("abstract", 5),
            ("category", 3),
            ("author", 1),
        )
        for field, weight in weighted_fields:
            haystack = fields.get(field, "")
            if self._matches_text_query(haystack, normalized_query):
                score += weight
        return score

    def _score_relevance_query(self, paper: Dict[str, Any], query: str) -> int:
        """按当前支持的 arXiv 查询子集计算相关性分数。

        日期和分类约束主要承担过滤职责，不应把分数抬高；AND/OR 查询则递归
        汇总文本子句分数，保证本地 relevance 排序和过滤逻辑使用同一套解析路径。
        """
        text = self._strip_outer_parentheses(query.strip())
        if not text or text.startswith("submittedDate:["):
            return 0
        if " ANDNOT " in text:
            parts = self._split_top_level(text, " ANDNOT ")
            if len(parts) > 1:
                # 排除子句不增加正向相关性，只保留左侧正条件的得分。
                return self._score_relevance_query(paper, parts[0])
            if not parts:
                return 0
        if " AND " in text:
            parts = self._split_top_level(text, " AND ")
            # relevance 排序和过滤共用同一套轻量 query 解释，只有顶层切分成功才递归，
            # 避免括号内操作符让评分阶段也重复处理原查询。
            if len(parts) > 1:
                return sum(self._score_relevance_query(paper, part) for part in parts)
        if " OR " in text:
            parts = self._split_top_level(text, " OR ")
            # 和匹配阶段保持一致：括号内部 OR 不应触发对原字符串的递归评分。
            if len(parts) > 1:
                return max((self._score_relevance_query(paper, part) for part in parts), default=0)
        if ":" in text:
            field, raw_query = text.split(":", 1)
            field = field.strip().lower()
            if field in {"cat", "category"}:
                # 分类约束承担过滤语义，不参与抬高相关性。
                return 0
            query_text = self._strip_outer_parentheses(raw_query.strip())
            if query_text.startswith('"') and query_text.endswith('"'):
                query_text = query_text[1:-1]
            query_text = query_text.replace('\\"', '"').replace("\\\\", "\\")
            fields = self._paper_to_search_fields(paper)
            if field == "all":
                return self._score_text_query(paper, query_text)
            # 指定字段命中时给固定高分，让“精确字段命中”优先于 all 的宽泛匹配。
            return 10 if self._matches_text_query(fields.get(field, fields["all"]), query_text) else 0
        return self._score_text_query(paper, text.replace('"', ""))

    def _matches_atomic_clause(self, paper: Dict[str, Any], clause: str) -> bool:
        """匹配不再可继续拆分的最小查询子句。"""
        text = self._strip_outer_parentheses(clause.strip())
        if not text:
            return True
        if text.startswith("submittedDate:["):
            return self._matches_submitted_date(paper, text)
        if ":" in text:
            field, raw_query = text.split(":", 1)
            field = field.strip().lower()
            query_text = self._strip_outer_parentheses(raw_query.strip())
            if query_text.startswith('"') and query_text.endswith('"'):
                query_text = query_text[1:-1]
            query_text = query_text.replace('\\"', '"').replace("\\\\", "\\").lower()
            fields = self._paper_to_search_fields(paper)
            # 未知字段在旧版内存匹配里退回 all 字段，尽量保持向后兼容。
            haystack = fields.get(field, fields["all"])
            return self._matches_text_query(haystack, query_text)
        query_text = text.replace('"', "").lower()
        return self._matches_text_query(self._paper_to_search_fields(paper)["all"], query_text)

    def _matches_query(self, paper: Dict[str, Any], query: str) -> bool:
        """递归判断论文是否命中旧版查询表达式。"""
        text = self._strip_outer_parentheses(query.strip())
        if not text:
            return True
        if " ANDNOT " in text:
            parts = self._split_top_level(text, " ANDNOT ")
            if len(parts) > 1:
                # ANDNOT 的语义是“左侧命中且后续子句全部不命中”。
                return self._matches_query(paper, parts[0]) and all(
                    not self._matches_query(paper, part) for part in parts[1:]
                )
            if not parts:
                return True
        if " AND " in text:
            parts = self._split_top_level(text, " AND ")
            # 只有顶层逻辑符真正拆出多个子句时才递归；括号内部的 AND 不能让原串原样递归，
            # 否则类似 `A OR (B AND C)` 会反复处理同一段文本直到栈溢出。
            if len(parts) > 1:
                return all(self._matches_query(paper, part) for part in parts)
        if " OR " in text:
            parts = self._split_top_level(text, " OR ")
            # 同上，OR 也必须确认发生了顶层切分，避免操作符只出现在括号内时递归不收敛。
            if len(parts) > 1:
                return any(self._matches_query(paper, part) for part in parts)
        return self._matches_atomic_clause(paper, text)

    def _build_paper_response(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """把数据库论文对象转换成对外搜索响应格式。

        这里刻意保留接近 arXiv API 的字段形状，减少上层已有消费代码的改动范围。
        """
        arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
        # 部分下游仍按 arXiv API 风格读取 id/title/summary 等字段，因此统一在此适配。
        return {
            "id": f"http://arxiv.org/abs/{arxiv_id}",
            "title": str(paper.get("title", "") or "").replace("\n", " ").strip(),
            "summary": str(paper.get("abstract", "") or "").replace("\n", " ").strip(),
            "published": str(paper.get("created", "") or paper.get("updated", "") or ""),
            "updated": str(paper.get("updated", "") or paper.get("created", "") or ""),
            "authors": self._parse_list_field(paper.get("authors", "")),
            "categories": self._parse_list_field(paper.get("categories", "")),
            "pdf_url": str(paper.get("pdf_url", "") or ""),
            "abs_url": str(paper.get("abs_url", "") or f"https://arxiv.org/abs/{arxiv_id}"),
            "journal_reference": "",
            "comment": "",
            "doi": "",
            "arxiv_id": arxiv_id,
            "submitter": "",
            "versions": [],
            "authors_parsed": [],
        }

    def _fetch_all_searchable_papers(self, id_list: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """拉取可供旧版内存检索使用的论文集合。"""
        columns = """
            arxiv_id,
            title,
            abstract,
            authors,
            categories,
            primary_category,
            created,
            updated,
            abs_url,
            pdf_url,
            oai_datestamp,
            fetched_at,
            created_at,
            updated_at
        """
        query = f"SELECT {columns} FROM arxiv_oai_papers"
        params: List[Any] = []
        normalized_ids = [str(item).strip() for item in (id_list or []) if str(item).strip()]
        if normalized_ids:
            # 只在显式给出 id_list 时收窄范围，避免旧逻辑误扫全表。
            placeholders = ",".join("?" for _ in normalized_ids)
            query += f" WHERE arxiv_id IN ({placeholders})"
            params.extend(normalized_ids)
        query += " ORDER BY COALESCE(created, updated, oai_datestamp, fetched_at) DESC, arxiv_id DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [self._parse_row(row) for row in rows]

    def _strip_query_outer_parentheses(self, query: str) -> str:
        """移除查询最外层成对括号；解析阶段需要尊重引号，避免 phrase 被误拆。"""
        text = query.strip()
        while text.startswith("(") and text.endswith(")"):
            depth = 0
            in_quote = False
            escaped = False
            balanced = True
            for index, char in enumerate(text):
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    # 解析本地子集时需要保留 phrase 内部原样内容，不能把引号里的括号当结构符。
                    in_quote = not in_quote
                    continue
                if in_quote:
                    continue
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(text) - 1:
                        balanced = False
                        break
            if balanced and depth == 0 and not in_quote:
                text = text[1:-1].strip()
            else:
                break
        return text

    def _split_query_top_level(self, query: str, token: str) -> List[str]:
        """按顶层布尔操作符切分查询；引号和括号内的操作符只作为普通文本处理。"""
        text = query.strip()
        parts: List[str] = []
        depth = 0
        in_quote = False
        escaped = False
        start = 0
        index = 0
        token_length = len(token)
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\":
                escaped = True
                index += 1
                continue
            if char == '"':
                in_quote = not in_quote
                index += 1
                continue
            if not in_quote:
                if char == "(":
                    depth += 1
                    index += 1
                    continue
                if char == ")":
                    depth = max(depth - 1, 0)
                    index += 1
                    continue
                if depth == 0 and text.startswith(token, index):
                    # 只有真正位于顶层的布尔操作符，才允许成为 AST 的拆分边界。
                    parts.append(text[start:index].strip())
                    index += token_length
                    start = index
                    continue
            index += 1
        parts.append(text[start:].strip())
        return [part for part in parts if part]

    def _parse_submitted_date_range(self, text: str, *, original_query: str) -> Dict[str, Any]:
        """把 submittedDate 范围子句解析成标准日期节点。"""
        match = re.fullmatch(r"submittedDate:\[(\d{12})\s+TO\s+(\d{12})\]", text.strip())
        if not match:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像只支持 submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM] 日期范围。",
                query=original_query,
                reason="unsupported_submitted_date_syntax",
            )
        start_raw, end_raw = match.groups()
        # 统一编译成 SQLite 可直接比较的 datetime 字符串，减少后续节点类型分支复杂度。
        start_dt = datetime.strptime(start_raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
        return {"type": "date", "start": start_dt, "end": end_dt}

    def _parse_local_oai_atomic_query(self, text: str, *, original_query: str) -> Dict[str, Any]:
        """解析单个不可再拆分的本地 OAI 查询子句。"""
        text = self._strip_query_outer_parentheses(text.strip())
        if not text:
            return {"type": "empty"}
        if text.startswith("submittedDate:"):
            return self._parse_submitted_date_range(text, original_query=original_query)

        field = "all"
        raw_value = text
        if ":" in text:
            field, raw_value = text.split(":", 1)
            field = field.strip().lower()
            raw_value = raw_value.strip()

        aliases = {
            "title": "ti",
            "abstract": "abs",
            "authors": "au",
            "author": "au",
            "category": "cat",
        }
        # 兼容常见字段别名，但最终仍收敛到本地 OAI 明确支持的最小字段集合。
        field = aliases.get(field, field)
        supported_fields = {"id", "cat", "ti", "abs", "au", "all"}
        if field not in supported_fields:
            raise UnsupportedLocalArxivQuery(
                f"本地 OAI 镜像不支持字段 `{field}`，请改用支持的查询子集。",
                query=original_query,
                reason=f"unsupported_field:{field}",
            )

        if not raw_value:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像不支持空字段查询。",
                query=original_query,
                reason="empty_field_query",
            )
        if "*" in raw_value or "?" in raw_value:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像不支持通配符查询，避免低精度误召回。",
                query=original_query,
                reason="unsupported_wildcard_query",
            )

        # phrase 和普通 token 查询在 FTS 编译阶段有不同语义，这里先把标记保留下来。
        phrase = raw_value.startswith('"') and raw_value.endswith('"') and len(raw_value) >= 2
        value = raw_value[1:-1] if phrase else raw_value
        value = value.replace('\\"', '"').replace("\\\\", "\\").strip()

        if field == "id":
            return {"type": "id", "value": value}
        if field == "cat":
            return {"type": "category", "value": value}
        return {"type": "text", "field": field, "value": value, "phrase": phrase}

    def _parse_local_oai_query(self, query: str) -> Optional[Dict[str, Any]]:
        """解析本地支持的 arXiv 查询子集；超出子集时显式失败而不是 Python 兜底。"""
        text = self._strip_query_outer_parentheses(str(query or "").strip())
        if not text:
            return None
        for operator, node_type in ((" ANDNOT ", "andnot"), (" AND ", "and"), (" OR ", "or")):
            if operator in text:
                parts = self._split_query_top_level(text, operator)
                if len(parts) > 1:
                    # 只有顶层真正发生拆分时才递归构树，避免括号内操作符造成死递归。
                    return {
                        "type": node_type,
                        "children": [self._parse_local_oai_query(part) for part in parts],
                    }
        return self._parse_local_oai_atomic_query(text, original_query=query)

    def _fts_query_for_text(self, value: str, *, phrase: bool, original_query: str) -> str:
        """把文本查询编译成 FTS5 语法；只使用 token/phrase，禁止子串模糊匹配。"""
        # 只抽取英文和数字 token，主动拒绝“空 token / 纯符号”输入，避免 MATCH 语法漂移。
        tokens = re.findall(r"[A-Za-z0-9]+", str(value or "").lower())
        if not tokens:
            raise UnsupportedLocalArxivQuery(
                "本地 OAI 镜像文本检索需要至少一个英文/数字 token。",
                query=original_query,
                reason="empty_fts_tokens",
            )
        if phrase:
            # phrase 查询保持原始顺序，适合 title/abstract 中的连续短语检索。
            return '"' + " ".join(tokens) + '"'
        # 非短语查询按 AND 拼接，显式要求所有 token 命中，保证高精度。
        return " AND ".join(tokens)

    def _compile_local_oai_query(
        self,
        node: Optional[Dict[str, Any]],
        *,
        fts5_available: bool,
        original_query: str,
    ) -> Tuple[str, List[Any], bool]:
        """把查询 AST 编译成 SQL WHERE 片段，返回是否消费了 FTS 文本索引。"""
        if not node or node.get("type") == "empty":
            return "", [], False

        node_type = node.get("type")
        if node_type in {"and", "or"}:
            # 复合节点先递归编译子节点，再在 SQL 层按原布尔关系拼接。
            compiled = [
                self._compile_local_oai_query(child, fts5_available=fts5_available, original_query=original_query)
                for child in node.get("children") or []
                if child
            ]
            clauses = [item[0] for item in compiled if item[0]]
            params = [param for item in compiled for param in item[1]]
            uses_fts = any(item[2] for item in compiled)
            if not clauses:
                return "", [], uses_fts
            joiner = " AND " if node_type == "and" else " OR "
            return "(" + joiner.join(clauses) + ")", params, uses_fts

        if node_type == "andnot":
            children = [child for child in node.get("children") or [] if child]
            if not children:
                return "", [], False
            # ANDNOT 只把第一个子句当正条件，后续子句全部编译成 NOT (...)。
            left_sql, left_params, left_fts = self._compile_local_oai_query(
                children[0],
                fts5_available=fts5_available,
                original_query=original_query,
            )
            negative_parts = [
                self._compile_local_oai_query(child, fts5_available=fts5_available, original_query=original_query)
                for child in children[1:]
            ]
            clauses = [left_sql] if left_sql else []
            params = list(left_params)
            uses_fts = left_fts
            for sql, item_params, item_fts in negative_parts:
                if sql:
                    clauses.append(f"NOT ({sql})")
                    params.extend(item_params)
                uses_fts = uses_fts or item_fts
            return "(" + " AND ".join(clauses) + ")", params, uses_fts

        if node_type == "id":
            # arXiv ID 本身就是主键级过滤条件，直接等值命中最精确。
            return "p.arxiv_id = ?", [node.get("value")], False
        if node_type == "category":
            return (
                "EXISTS (SELECT 1 FROM arxiv_oai_paper_categories c "
                "WHERE c.arxiv_id = p.arxiv_id AND c.category = ?)"
            ), [node.get("value")], False
        if node_type == "date":
            # 日期查询在多种时间字段之间做一致兜底，尽量贴近实际可用时间语义。
            return (
                "datetime(COALESCE(p.updated, p.created, p.oai_datestamp, p.fetched_at)) "
                "BETWEEN datetime(?) AND datetime(?)"
            ), [node.get("start"), node.get("end")], False
        if node_type == "text":
            if not fts5_available:
                raise LocalArxivSearchIndexUnavailable(
                    "本地 OAI 镜像文本索引不可用，无法执行 ti/abs/au/all 文本检索。",
                    query=original_query,
                    reason="fts5_unavailable",
                )
            field_map = {"ti": "title", "abs": "abstract", "au": "authors", "all": "all_text"}
            fts_field = field_map.get(str(node.get("field") or "all"), "all_text")
            # MATCH 语句始终只打到 FTS 表，避免退回 LIKE 造成语义和性能不可控。
            fts_query = self._fts_query_for_text(
                str(node.get("value") or ""),
                phrase=bool(node.get("phrase")),
                original_query=original_query,
            )
            return (
                "p.arxiv_id IN (SELECT arxiv_id FROM arxiv_oai_papers_fts "
                "WHERE arxiv_oai_papers_fts MATCH ?)"
            ), [f"{fts_field}:({fts_query})"], True

        raise UnsupportedLocalArxivQuery(
            "本地 OAI 镜像无法识别该查询节点。",
            query=original_query,
            reason=f"unsupported_node:{node_type}",
        )

    def search(
        self,
        search_query: str = "",
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> Dict[str, Any]:
        """执行本地 OAI 搜索，并返回兼容上层消费的分页结果。"""
        normalized_query = str(search_query or "").strip()
        normalized_id_list = [str(item).strip() for item in (id_list or []) if str(item).strip()]
        # 查询字符串先解析成 AST，后续过滤、错误提示和能力说明都围绕这份结构展开。
        query_node = self._parse_local_oai_query(normalized_query) if normalized_query else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            fts5_available = self._is_fts5_available(cursor)
            where_clauses: List[str] = []
            params: List[Any] = []
            uses_fts = False

            if normalized_id_list:
                # id_list 作为额外硬过滤条件，常用于上游已知候选集合内的二次筛选。
                placeholders = ",".join("?" for _ in normalized_id_list)
                where_clauses.append(f"p.arxiv_id IN ({placeholders})")
                params.extend(normalized_id_list)

            query_sql, query_params, uses_fts = self._compile_local_oai_query(
                query_node,
                fts5_available=fts5_available,
                original_query=normalized_query,
            )
            if query_sql:
                where_clauses.append(query_sql)
                params.extend(query_params)

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            # 先算总数，再分页取数据，保持和远端 arXiv API 分页接口一致的返回结构。
            count_query = f"SELECT COUNT(*) FROM arxiv_oai_papers p {where_sql}"
            cursor.execute(count_query, params)
            total_results = int((cursor.fetchone() or [0])[0] or 0)

            direction = "DESC" if sort_order == "descending" else "ASC"
            order_sql = (
                "ORDER BY datetime(COALESCE(p.updated, p.created, p.oai_datestamp, p.fetched_at)) "
                f"{direction}, p.arxiv_id {direction}"
            )
            if sort_by not in {"relevance", "submittedDate", "lastUpdatedDate"}:
                # 未知排序字段时回退到稳定主键顺序，避免拼出不可控 SQL。
                order_sql = "ORDER BY p.arxiv_id ASC"

            columns = """
                p.arxiv_id,
                p.title,
                p.abstract,
                p.authors,
                p.categories,
                p.primary_category,
                p.created,
                p.updated,
                p.abs_url,
                p.pdf_url,
                p.oai_datestamp,
                p.fetched_at,
                p.created_at,
                p.updated_at
            """
            select_query = f"""
                SELECT {columns}
                FROM arxiv_oai_papers p
                {where_sql}
                {order_sql}
                LIMIT ? OFFSET ?
            """
            cursor.execute(select_query, params + [max(1, int(max_results)), max(0, int(start))])
            papers = [self._build_paper_response(self._parse_row(row)) for row in cursor.fetchall()]

        # 把本次查询到底使用了哪一层索引能力显式带回，方便前端解释“为什么和 arXiv API 不完全一样”。
        query_capability = build_local_oai_query_capability(
            mode="local_oai_sqlite_fts" if uses_fts else "local_oai_sqlite_index",
            fts5_available=fts5_available,
            search_index_status="ready" if fts5_available or not uses_fts else "text_index_unavailable",
        )
        warnings = [
            "当前使用本地 OAI 镜像库，仅支持高精度可下推查询子集，不等价完整 arXiv API 语法。"
        ]
        if sort_by == "relevance" and uses_fts:
            warnings.append("本地 relevance 当前使用 FTS 命中过滤和日期排序，不等价 arXiv API relevance。")
        return {
            "query": normalized_query,
            "id_list": normalized_id_list,
            "source": "local_oai",
            "query_capability": query_capability,
            "warnings": warnings,
            "total_results": total_results,
            "start_index": start,
            "items_per_page": len(papers),
            "papers": papers,
            "timestamp": datetime.now().isoformat(),
        }

    def get_available_fields(self) -> List[Dict[str, str]]:
        """返回本地 OAI 搜索后端支持展示的字段列表。"""
        # 本地 OAI 只承诺可下推的高精度字段，避免前端展示实际无法执行的远程 API 字段。
        return [
            {"prefix": "ti", "field": "Title", "description": "搜索论文标题"},
            {"prefix": "au", "field": "Author", "description": "搜索作者姓名"},
            {"prefix": "abs", "field": "Abstract", "description": "搜索摘要"},
            {"prefix": "cat", "field": "Subject Category", "description": "搜索学科分类"},
            {"prefix": "id", "field": "ID", "description": "搜索论文 ID"},
            {"prefix": "all", "field": "All Fields", "description": "搜索所有字段"},
        ]

    def get_subject_categories(self) -> List[Dict[str, str]]:
        """返回本地 OAI 搜索后端可展示的常见 arXiv 分类列表。"""
        # 分类列表是 UI 辅助数据，不代表本地镜像一定已经同步了全部分类论文。
        return [
            {"code": "cs.AI", "name": "Artificial Intelligence"},
            {"code": "cs.AR", "name": "Hardware Architecture"},
            {"code": "cs.CC", "name": "Computational Complexity"},
            {"code": "cs.CE", "name": "Computational Engineering, Finance, and Science"},
            {"code": "cs.CG", "name": "Computational Geometry"},
            {"code": "cs.CL", "name": "Computation and Language"},
            {"code": "cs.CR", "name": "Cryptography and Security"},
            {"code": "cs.CV", "name": "Computer Vision and Pattern Recognition"},
            {"code": "cs.CY", "name": "Computers and Society"},
            {"code": "cs.DB", "name": "Databases"},
            {"code": "cs.DC", "name": "Distributed, Parallel, and Cluster Computing"},
            {"code": "cs.DL", "name": "Digital Libraries"},
            {"code": "cs.DM", "name": "Discrete Mathematics"},
            {"code": "cs.DS", "name": "Data Structures and Algorithms"},
            {"code": "cs.ET", "name": "Emerging Technologies"},
            {"code": "cs.FL", "name": "Formal Languages and Automata Theory"},
            {"code": "cs.GL", "name": "General Literature"},
            {"code": "cs.GR", "name": "Graphics"},
            {"code": "cs.GT", "name": "Computer Science and Game Theory"},
            {"code": "cs.HC", "name": "Human-Computer Interaction"},
            {"code": "cs.IR", "name": "Information Retrieval"},
            {"code": "cs.IT", "name": "Information Theory"},
            {"code": "cs.LG", "name": "Machine Learning"},
            {"code": "cs.LO", "name": "Logic in Computer Science"},
            {"code": "cs.MA", "name": "Multiagent Systems"},
            {"code": "cs.MM", "name": "Multimedia"},
            {"code": "cs.MS", "name": "Mathematical Software"},
            {"code": "cs.NA", "name": "Numerical Analysis"},
            {"code": "cs.NE", "name": "Neural and Evolutionary Computing"},
            {"code": "cs.NI", "name": "Networking"},
            {"code": "cs.OH", "name": "Other Computer Science"},
            {"code": "cs.OS", "name": "Operating Systems"},
            {"code": "cs.PF", "name": "Performance"},
            {"code": "cs.PL", "name": "Programming Languages"},
            {"code": "cs.RO", "name": "Robotics"},
            {"code": "cs.SC", "name": "Symbolic Computation"},
            {"code": "cs.SD", "name": "Sound"},
            {"code": "cs.SE", "name": "Software Engineering"},
            {"code": "cs.SI", "name": "Social and Information Networks"},
            {"code": "cs.SY", "name": "Systems and Control"},
            {"code": "stat.ML", "name": "Machine Learning"},
            {"code": "physics.quant-ph", "name": "Quantum Physics"},
            {"code": "hep-th", "name": "High Energy Physics - Theory"},
            {"code": "hep-ph", "name": "High Energy Physics - Phenomenology"},
            {"code": "astro-ph", "name": "Astrophysics"},
        ]

    def _initialize_database(self) -> None:
        """初始化 OAI SQLite 主表、从表和全文索引。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            fts5_available = self._is_fts5_available(cursor)
            # 主表负责保存 OAI 元数据真相，从表和 FTS 都视为可重建索引。
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS arxiv_oai_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT NOT NULL UNIQUE,
                    title TEXT,
                    abstract TEXT,
                    authors TEXT,
                    categories TEXT,
                    primary_category TEXT,
                    created TEXT,
                    updated TEXT,
                    abs_url TEXT,
                    pdf_url TEXT,
                    oai_datestamp TEXT,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                '''
            )
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS arxiv_oai_paper_categories (
                    arxiv_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (arxiv_id, category),
                    FOREIGN KEY (arxiv_id) REFERENCES arxiv_oai_papers(arxiv_id) ON DELETE CASCADE
                )
                '''
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_arxiv_oai_papers_arxiv_id ON arxiv_oai_papers(arxiv_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_arxiv_oai_papers_created ON arxiv_oai_papers(created)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_arxiv_oai_papers_updated ON arxiv_oai_papers(updated)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_arxiv_oai_categories_category ON arxiv_oai_paper_categories(category)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_arxiv_oai_categories_arxiv_id ON arxiv_oai_paper_categories(arxiv_id)"
            )
            if fts5_available:
                # FTS5 是本地文本检索的硬依赖；不可用时只保留 id/category/date 等精确过滤能力。
                cursor.execute(
                    '''
                    CREATE VIRTUAL TABLE IF NOT EXISTS arxiv_oai_papers_fts USING fts5(
                        arxiv_id UNINDEXED,
                        title,
                        abstract,
                        authors,
                        categories,
                        all_text,
                        tokenize = 'unicode61'
                    )
                    '''
                )
            # 旧库升级后无需重新全量同步，初始化阶段自动把缺失索引补齐。
            self._backfill_search_indexes(cursor, fts5_available=fts5_available)
            conn.commit()
            logger.info("OAI database tables initialized successfully: %s", self.db_path)

    def upsert_arxiv_oai_paper(self, paper: Dict[str, Any]) -> bool:
        """写入或更新单篇 OAI 论文，并同步相关检索索引。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 主表 upsert 和索引刷新放在同一事务里，避免主表/从表状态不一致。
                cursor.execute(
                    '''
                    INSERT INTO arxiv_oai_papers (
                        arxiv_id,
                        title,
                        abstract,
                        authors,
                        categories,
                        primary_category,
                        created,
                        updated,
                        abs_url,
                        pdf_url,
                        oai_datestamp,
                        fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        abstract = excluded.abstract,
                        authors = excluded.authors,
                        categories = excluded.categories,
                        primary_category = excluded.primary_category,
                        created = excluded.created,
                        updated = excluded.updated,
                        abs_url = excluded.abs_url,
                        pdf_url = excluded.pdf_url,
                        oai_datestamp = excluded.oai_datestamp,
                        fetched_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    (
                        paper.get("arxiv_id"),
                        paper.get("title"),
                        paper.get("abstract"),
                        paper.get("authors"),
                        paper.get("categories"),
                        paper.get("primary_category"),
                        paper.get("created"),
                        paper.get("updated"),
                        paper.get("abs_url"),
                        paper.get("pdf_url"),
                        paper.get("oai_datestamp"),
                    ),
                )
                # 单篇写入后立即刷新分类索引和 FTS 索引，保证后续搜索可见性一致。
                self._sync_search_index_for_papers(cursor, [paper])
                conn.commit()
                logger.info("OAI paper upserted: %s", paper.get("arxiv_id"))
                return True
        except Exception as exc:
            logger.error("Error upserting OAI paper: %s", exc)
            return False

    def upsert_arxiv_oai_papers(self, papers: Sequence[Dict[str, Any]]) -> int:
        """批量写入或更新 OAI 论文，并在同一事务中维护索引。"""
        normalized_papers = [paper for paper in papers if paper and paper.get("arxiv_id")]
        if not normalized_papers:
            return 0

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 批量 upsert 优先保证吞吐；只有整批失败时上层才会回退到单篇模式。
                cursor.executemany(
                    '''
                    INSERT INTO arxiv_oai_papers (
                        arxiv_id,
                        title,
                        abstract,
                        authors,
                        categories,
                        primary_category,
                        created,
                        updated,
                        abs_url,
                        pdf_url,
                        oai_datestamp,
                        fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        abstract = excluded.abstract,
                        authors = excluded.authors,
                        categories = excluded.categories,
                        primary_category = excluded.primary_category,
                        created = excluded.created,
                        updated = excluded.updated,
                        abs_url = excluded.abs_url,
                        pdf_url = excluded.pdf_url,
                        oai_datestamp = excluded.oai_datestamp,
                        fetched_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    ''',
                    [
                        (
                            paper.get("arxiv_id"),
                            paper.get("title"),
                            paper.get("abstract"),
                            paper.get("authors"),
                            paper.get("categories"),
                            paper.get("primary_category"),
                            paper.get("created"),
                            paper.get("updated"),
                            paper.get("abs_url"),
                            paper.get("pdf_url"),
                            paper.get("oai_datestamp"),
                        )
                        for paper in normalized_papers
                    ],
                )
                # 批量刷新从索引，确保分类和全文检索状态与主表提交点一致。
                self._sync_search_index_for_papers(cursor, normalized_papers)
                conn.commit()
                logger.info("OAI paper batch upserted: %s", len(normalized_papers))
                return len(normalized_papers)
        except Exception as exc:
            logger.error("Error upserting OAI papers batch: %s", exc)
            return 0


class ArxivOaiSyncService:
    def __init__(
        self,
        endpoint: str = OAI_ENDPOINT,
        request_interval_seconds: float = 5.0,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 5,
        database_service: Optional[ArxivOaiDatabaseService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        embedding_collection_name: str = "arxiv_paper_embeddings",
        user_agent: str = "rag-project01-framework-oai-sync/1.0",
    ):
        """初始化 OAI-PMH 同步服务及其依赖。

        该服务负责远端拉取、分页续传、分类过滤、落库和 embedding 写入，因此会在
        构造期统一准备网络会话、限流参数和向量化依赖。
        """
        self.endpoint = endpoint
        # arXiv OAI-PMH 对抓取频率较敏感，这里强制下限避免调用方误设过小间隔。
        self.request_interval_seconds = max(3.5, float(request_interval_seconds))
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        # database_service 支持惰性注入，便于测试或只做 dry-run 时减少副作用。
        self.database_service = database_service
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.embedding_collection_name = embedding_collection_name
        self.embedding_config = self.embedding_service.get_default_embedding_config()
        # 复用 Session，统一携带 User-Agent 并减少多页抓取的连接开销。
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def sync(self, from_date: str, until_date: str, dry_run: bool = False, count_only: bool = False) -> ArxivOaiSyncStats:
        """执行一次 OAI-PMH 区间同步。

        dry_run 仅记录“本会写入什么”，count_only 只做命中统计但不记为写入；
        两者都会跳过真正的数据库和向量库落地。
        """
        sync_started_at = time.perf_counter()
        stats = ArxivOaiSyncStats()
        request_params: Dict[str, str] = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "from": from_date,
            "until": until_date,
        }
        resumption_token: Optional[str] = None

        logger.info(
            "Starting arXiv OAI-PMH sync: from=%s until=%s dry_run=%s count_only=%s endpoint=%s",
            from_date,
            until_date,
            dry_run,
            count_only,
            self.endpoint,
        )

        while True:
            if resumption_token:
                # OAI-PMH 翻页必须改用 resumptionToken，原始时间窗口参数不能继续携带。
                request_params = {"verb": "ListRecords", "resumptionToken": resumption_token}

            xml_text = self._fetch_page(request_params, stats)
            if xml_text is None:
                break

            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as exc:
                stats.errors += 1
                logger.error("Failed to parse OAI-PMH XML response: %s", exc)
                break

            page_error = self._extract_oai_error(root)
            if page_error:
                stats.errors += 1
                logger.error("OAI-PMH error response: %s", page_error)
                break

            records = root.findall(".//{*}record")
            stats.pages_processed += 1
            logger.info("Processing page %s with %s record nodes", stats.pages_processed, len(records))

            # 记录当前页进入持久化前的命中情况，避免“有翻页但没有写库日志”时难以判断是无命中还是写库异常。
            page_matched_before = stats.records_matched
            page_filtered_before = stats.skipped_category_filter
            matched_papers: List[Dict[str, Any]] = []
            for record in records:
                paper = self._process_record(record, stats, dry_run=dry_run, count_only=count_only)
                if paper is not None:
                    matched_papers.append(paper)

            page_matched = stats.records_matched - page_matched_before
            page_filtered = stats.skipped_category_filter - page_filtered_before
            logger.info(
                "Page %s category summary: matched=%s filtered_by_category=%s",
                stats.pages_processed,
                page_matched,
                page_filtered,
            )

            if matched_papers and not (dry_run or count_only):
                # 只对真正需要落库的命中结果做批量持久化，保持 dry-run/count-only 无副作用。
                self._persist_matched_papers(matched_papers, stats)

            resumption_token = self._extract_resumption_token(root)
            if not resumption_token:
                logger.info("OAI-PMH resumptionToken exhausted; sync complete.")
                break

            logger.info("Received resumptionToken; continuing with next page.")
            time.sleep(self.request_interval_seconds)

        stats.sync_duration_seconds = time.perf_counter() - sync_started_at
        logger.info(
            (
                "OAI-PMH sync summary: requests=%s pages=%s seen=%s matched=%s "
                "written=%s embeddings_written=%s embeddings_skipped_existing=%s embedding_errors=%s "
                "embedding_input_tokens=%s embedding_output_tokens=%s embedding_total_tokens=%s embedding_cost_yuan=%.6f "
                "duration_seconds=%.2f "
                "skipped=%s filtered=%s no_metadata=%s no_categories=%s errors=%s dry_run=%s count_only=%s"
            ),
            stats.requests_made,
            stats.pages_processed,
            stats.records_seen,
            stats.records_matched,
            stats.records_written,
            stats.embeddings_written,
            stats.embeddings_skipped_existing,
            stats.embedding_errors,
            stats.embedding_input_tokens,
            stats.embedding_output_tokens,
            stats.embedding_total_tokens,
            stats.embedding_cost_yuan,
            stats.sync_duration_seconds,
            stats.records_skipped,
            stats.skipped_category_filter,
            stats.skipped_no_metadata + stats.skipped_no_arxiv_id + stats.skipped_parse_errors,
            stats.skipped_no_categories,
            stats.errors,
            dry_run,
            count_only,
        )
        logger.info(
            "OAI sync cost summary: duration=%.2fs input_tokens=%s output_tokens=%s total_tokens=%s estimated_cost=%.6f yuan",
            stats.sync_duration_seconds,
            stats.embedding_input_tokens,
            stats.embedding_output_tokens,
            stats.embedding_total_tokens,
            stats.embedding_cost_yuan,
        )
        return stats

    def _fetch_page(self, params: Dict[str, str], stats: ArxivOaiSyncStats) -> Optional[str]:
        """抓取单页 OAI-PMH 响应，并带指数退避重试。"""
        retry_delay = self.request_interval_seconds

        for attempt in range(1, self.max_retries + 1):
            stats.requests_made += 1
            try:
                response = self.session.get(self.endpoint, params=params, timeout=self.request_timeout_seconds)
            except requests.Timeout as exc:
                logger.warning(
                    "OAI-PMH request timeout on attempt %s/%s for params=%s: %s",
                    attempt,
                    self.max_retries,
                    params,
                    exc,
                )
                if attempt < self.max_retries:
                    # 超时通常意味着服务端繁忙或网络抖动，按指数退避继续尝试。
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None
            except requests.RequestException as exc:
                logger.error("OAI-PMH request error for params=%s: %s", params, exc)
                if attempt < self.max_retries:
                    # 普通请求异常同样保留重试机会，减少短时网络故障带来的整批失败。
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None

            if response.status_code in {429, 503}:
                retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                # 优先尊重服务端 Retry-After，同时不低于本地限流间隔和当前退避延迟。
                sleep_seconds = max(retry_delay, retry_after or 0.0, self.request_interval_seconds)
                logger.warning(
                    "OAI-PMH returned %s on attempt %s/%s; sleeping %.1fs before retry",
                    response.status_code,
                    attempt,
                    self.max_retries,
                    sleep_seconds,
                )
                if attempt < self.max_retries:
                    time.sleep(sleep_seconds)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None

            if response.status_code >= 400:
                stats.errors += 1
                logger.error(
                    "OAI-PMH request failed with HTTP %s for params=%s: %s",
                    response.status_code,
                    params,
                    response.text[:1000],
                )
                return None

            # 显式指定 UTF-8 编码，避免 requests 自动推断编码错误导致字符乱码
            response.encoding = 'utf-8'
            return response.text

        return None

    def _process_record(self, record: ET.Element, stats: ArxivOaiSyncStats, dry_run: bool, count_only: bool) -> Optional[Dict[str, Any]]:
        """解析并过滤单条 OAI record，返回通过筛选的论文字典。"""
        stats.records_seen += 1
        metadata = self._find_first_child(record, "metadata")
        if metadata is None:
            stats.skipped_no_metadata += 1
            stats.records_skipped += 1
            logger.debug("Skipping record without metadata")
            return None

        paper_elem = self._find_first_element_child(metadata)
        if paper_elem is None:
            stats.skipped_no_metadata += 1
            stats.records_skipped += 1
            logger.debug("Skipping record with empty metadata")
            return None

        stats.records_with_metadata += 1
        # 元数据解析失败与业务过滤是两类问题，统计上需要分开记录。
        paper = self._parse_paper_metadata(record, paper_elem)
        if paper is None:
            stats.skipped_parse_errors += 1
            stats.records_skipped += 1
            return None

        arxiv_id = paper.get("arxiv_id", "")
        if not arxiv_id:
            stats.skipped_no_arxiv_id += 1
            stats.records_skipped += 1
            logger.warning("Skipping record without arxiv_id")
            return None

        categories = paper.get("categories_list", [])
        if not categories:
            # 没有分类的记录无法参与目标类别同步，也很难支撑后续检索和推荐。
            stats.skipped_no_categories += 1
            stats.records_skipped += 1
            logger.info("Skipping %s because categories are empty", arxiv_id)
            return None

        if not self._is_allowed_categories(categories):
            stats.skipped_category_filter += 1
            stats.records_skipped += 1
            logger.debug("Skipping %s because categories=%s do not match target set", arxiv_id, categories)
            return None

        stats.records_matched += 1
        if dry_run or count_only:
            if dry_run:
                # dry-run 需要让“预期会写入多少条”可见，因此模拟累加 records_written。
                stats.records_written += 1
            logger.info(
                "[%s] Would upsert %s | title=%s | categories=%s",
                "dry-run" if dry_run else "count-only",
                arxiv_id,
                paper.get("title", ""),
                categories,
            )
        return paper

    def _persist_matched_papers(self, papers: Sequence[Dict[str, Any]], stats: ArxivOaiSyncStats) -> None:
        """批量落库存活论文，并在需要时回退到单条写入。"""
        normalized_papers = [paper for paper in papers if paper and paper.get("arxiv_id")]
        if not normalized_papers:
            return

        if self.database_service is None:
            self.database_service = ArxivOaiDatabaseService()

        persisted_papers: List[Dict[str, Any]] = []
        persisted_count = self.database_service.upsert_arxiv_oai_papers(normalized_papers)
        if persisted_count == 0:
            # 整批失败时回退到单条写入，尽量保住能成功持久化的论文，同时暴露坏数据。
            logger.warning("Batch OAI upsert failed; falling back to individual upserts for %s papers", len(normalized_papers))
            for paper in normalized_papers:
                if self.database_service.upsert_arxiv_oai_paper(paper):
                    persisted_papers.append(paper)
                    persisted_count += 1
                else:
                    stats.errors += 1
                    logger.error("Failed to upsert %s", paper.get("arxiv_id", ""))
        elif persisted_count == len(normalized_papers):
            persisted_papers = list(normalized_papers)

        stats.records_written += persisted_count
        if persisted_papers:
            for paper in persisted_papers:
                logger.info("OAI paper upserted: %s", paper.get("arxiv_id"))

        # 只对真正落库成功的论文生成 embedding，避免向量库和主库出现悬空记录。
        self._store_paper_embeddings_batch(persisted_papers, stats)

    def _build_oai_embedding_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """构造写入向量库的 OAI embedding 元数据。"""
        # metadata 保持与检索侧常用字段一致，方便后续按 arxiv_id / title / date 回查。
        return {
            "content": str(paper.get("abstract", "") or "").strip(),
            "arxiv_id": str(paper.get("arxiv_id", "") or "").strip(),
            "title": str(paper.get("title", "") or "").strip(),
            "authors": paper.get("authors", ""),
            "categories": paper.get("categories", ""),
            "published_date": str(
                paper.get("created")
                or paper.get("updated")
                or paper.get("oai_datestamp")
                or ""
            ).strip(),
            "url": str(paper.get("abs_url") or paper.get("pdf_url") or "").strip(),
            "embedding_model": self.embedding_config.model_name,
        }

    def _store_paper_embeddings_batch(self, papers: Sequence[Dict[str, Any]], stats: ArxivOaiSyncStats) -> None:
        """批量为已落库论文生成 embedding，并写入向量库。"""
        eligible_papers = []
        eligible_ids = []
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
            title = str(paper.get("title", "") or "").strip()
            abstract = str(paper.get("abstract", "") or "").strip()
            if not arxiv_id or not title or not abstract:
                # 没有标题或摘要时 embedding 质量不可控，直接跳过比写入噪声向量更安全。
                logger.debug(
                    "Skipping embedding for OAI paper %s because title or abstract is missing",
                    arxiv_id or "<unknown>",
                )
                continue
            eligible_papers.append(paper)
            eligible_ids.append(arxiv_id)

        if not eligible_papers:
            return

        existing_ids = set()
        try:
            for start in range(0, len(eligible_ids), OAI_VECTOR_QUERY_BATCH_SIZE):
                batch_ids = eligible_ids[start : start + OAI_VECTOR_QUERY_BATCH_SIZE]
                # 先批量探测已存在向量，避免重复生成 embedding 造成额外成本。
                existing_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
                    collection_name=self.embedding_collection_name,
                    arxiv_ids=batch_ids,
                )
                for item in existing_embeddings:
                    arxiv_id = str(item.get("arxiv_id", "") or "").strip()
                    if arxiv_id:
                        existing_ids.add(arxiv_id)
        except Exception as exc:  # pragma: no cover - vector store runtime dependent
            logger.warning("Failed to inspect existing embeddings for OAI papers: %s", exc)
        if existing_ids:
            stats.embeddings_skipped_existing += len(existing_ids)

        missing_papers = [paper for paper in eligible_papers if str(paper.get("arxiv_id", "") or "").strip() not in existing_ids]
        if not missing_papers:
            return

        stats.embeddings_attempted += len(missing_papers)
        texts = [
            self.embedding_service.build_paper_embedding_text(
                str(paper.get("title", "") or "").strip(),
                str(paper.get("abstract", "") or "").strip(),
            )
            for paper in missing_papers
        ]

        try:
            # 批量 embedding 优先追求成本和吞吐；失败时再回退到单篇粒度定位问题。
            embeddings, usage = self.embedding_service.create_text_embeddings_with_usage(
                texts,
                provider=self.embedding_config.provider,
                model=self.embedding_config.model_name,
                api_key=self.embedding_config.api_key,
                base_url=self.embedding_config.base_url,
                dimension=self.embedding_config.dimension,
                batch_size=OAI_EMBEDDING_BATCH_SIZE,
            )
            self._accumulate_embedding_usage(stats, usage, len(missing_papers))
            if len(embeddings) != len(missing_papers):
                raise ValueError(
                    f"Embedding batch returned {len(embeddings)} vectors for {len(missing_papers)} papers"
                )

            items = [
                {
                    "embedding": embedding,
                    "metadata": self._build_oai_embedding_metadata(paper),
                }
                for paper, embedding in zip(missing_papers, embeddings)
                if embedding
            ]
            if not items:
                return

            inserted_count = self.vector_store_service.insert_embeddings(self.embedding_collection_name, items)
            stats.embeddings_written += int(inserted_count)
            # 批量写入后逐条记日志，便于后续按 arxiv_id 追踪 embedding 生命周期。
            for paper in missing_papers:
                logger.info("Embedded OAI paper into vector store: %s", paper.get("arxiv_id"))
        except Exception as exc:  # pragma: no cover - vector store/runtime dependent
            logger.warning("Batch embedding for OAI papers failed, falling back to single-item processing: %s", exc)
            for paper in missing_papers:
                self._maybe_store_paper_embedding(paper, stats)

    def _accumulate_embedding_usage(self, stats: ArxivOaiSyncStats, usage: Dict[str, Any], item_count: int) -> None:
        """把 embedding 调用的 token 用量和估算成本累计到同步统计中。"""
        if not usage:
            return

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))

        # 统计统一在这里收口，避免批量和单篇写入两条路径各自维护计费逻辑。
        stats.embedding_input_tokens += input_tokens
        stats.embedding_output_tokens += output_tokens
        stats.embedding_total_tokens += total_tokens

        if input_tokens > 0:
            # 当前成本估算只按输入 token 计价，和现有 DashScope 文本 embedding 计费规则保持一致。
            stats.embedding_cost_yuan += (input_tokens / 1000.0) * OAI_DASHSCOPE_TEXT_TOKEN_PRICE_PER_1K

        logger.info(
            "OAI embedding batch usage: items=%s input_tokens=%s output_tokens=%s total_tokens=%s estimated_cost=%.6f yuan",
            item_count,
            input_tokens,
            output_tokens,
            total_tokens,
            (input_tokens / 1000.0) * OAI_DASHSCOPE_TEXT_TOKEN_PRICE_PER_1K if input_tokens > 0 else 0.0,
        )

    def _maybe_store_paper_embedding(self, paper: Dict[str, Any], stats: ArxivOaiSyncStats) -> None:
        """为单篇论文生成并写入 embedding，作为批量失败时的兜底路径。"""
        arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
        title = str(paper.get("title", "") or "").strip()
        abstract = str(paper.get("abstract", "") or "").strip()
        if not arxiv_id or not title or not abstract:
            logger.debug("Skipping embedding for OAI paper %s because title or abstract is missing", arxiv_id or "<unknown>")
            return

        try:
            # 单篇兜底前先检查是否已有向量，避免批量失败后重复写入。
            existing_embeddings = self.vector_store_service.get_paper_embeddings_by_arxiv_ids(
                collection_name=self.embedding_collection_name,
                arxiv_ids=[arxiv_id],
            )
        except Exception as exc:  # pragma: no cover - vector store runtime dependent
            logger.warning("Failed to inspect existing embedding for OAI paper %s: %s", arxiv_id, exc)
            existing_embeddings = []

        if existing_embeddings:
            stats.embeddings_skipped_existing += 1
            return

        stats.embeddings_attempted += 1
        text_to_embed = self.embedding_service.build_paper_embedding_text(title, abstract)
        if not text_to_embed:
            stats.embedding_errors += 1
            logger.warning("Skipping OAI embedding for %s because embedding text is empty", arxiv_id)
            return

        try:
            embedding, usage = self.embedding_service.create_single_embedding_with_usage(
                text_to_embed,
                provider=self.embedding_config.provider,
                model=self.embedding_config.model_name,
                api_key=self.embedding_config.api_key,
                base_url=self.embedding_config.base_url,
                dimension=self.embedding_config.dimension,
            )
            self._accumulate_embedding_usage(stats, usage, 1)
            # 单篇兜底沿用和批量路径一致的 metadata 结构，方便后续统一检索与排查。
            metadata = {
                "content": abstract,
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": paper.get("authors", ""),
                "categories": paper.get("categories", ""),
                "published_date": str(
                    paper.get("created")
                    or paper.get("updated")
                    or paper.get("oai_datestamp")
                    or ""
                ).strip(),
                "url": str(paper.get("abs_url") or paper.get("pdf_url") or "").strip(),
                "embedding_model": self.embedding_config.model_name,
            }
            self.vector_store_service.insert_single_embedding(
                self.embedding_collection_name,
                embedding,
                metadata,
            )
            stats.embeddings_written += 1
            logger.info("Embedded OAI paper into vector store: %s", arxiv_id)
        except Exception as exc:  # pragma: no cover - embedding/vector store runtime dependent
            stats.embedding_errors += 1
            logger.warning("Failed to embed OAI paper %s into vector store: %s", arxiv_id, exc)

    def _parse_paper_metadata(self, record: ET.Element, paper_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """把单条 OAI-PMH record 解析成内部统一论文结构。"""
        try:
            raw_arxiv_id = self._extract_text(paper_elem, "id")
            arxiv_id = self._normalize_arxiv_id(raw_arxiv_id)
            title = self._normalize_whitespace(self._extract_text(paper_elem, "title"))
            abstract = self._normalize_whitespace(self._extract_text(paper_elem, "abstract"))
            created = self._normalize_whitespace(self._extract_text(paper_elem, "created"))
            updated = self._normalize_whitespace(self._extract_text(paper_elem, "updated") or self._extract_text(paper_elem, "updateDate"))
            categories_text = self._normalize_whitespace(self._extract_text(paper_elem, "categories"))
            categories = self._split_categories(categories_text)
            primary_category = self._extract_primary_category(paper_elem)
            authors = self._extract_authors(paper_elem)
            oai_datestamp = self._normalize_whitespace(self._extract_text(record, "datestamp"))

            # authors/categories 继续存成 JSON 字符串，兼容现有数据库结构和旧调用方读取习惯。
            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": json.dumps(authors, ensure_ascii=False),
                "categories": json.dumps(categories, ensure_ascii=False),
                "categories_list": categories,
                "primary_category": primary_category,
                "created": created,
                "updated": updated,
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "oai_datestamp": oai_datestamp,
            }
        except Exception as exc:
            logger.error("Failed to parse OAI-PMH record: %s", exc)
            return None

    def _extract_authors(self, paper_elem: ET.Element) -> List[str]:
        """从 OAI 元数据中提取作者列表，兼容多种 author 结构。"""
        authors_parent = self._find_first_child(paper_elem, "authors")
        # 优先限定在 authors 节点下遍历；缺失时再退回整棵 paper 子树，兼容历史 XML 变体。
        search_root = authors_parent if authors_parent is not None else paper_elem
        authors: List[str] = []
        for node in search_root.iter():
            if self._local_name(node.tag) != "author":
                continue
            name_parts: List[str] = []
            forenames = self._normalize_whitespace(self._extract_text(node, "forenames"))
            keyname = self._normalize_whitespace(self._extract_text(node, "keyname"))
            full_name = self._normalize_whitespace(self._extract_text(node, "name"))
            if full_name:
                # 某些 XML 直接提供完整姓名；优先使用它，避免再拼接出重复空格或错序。
                authors.append(full_name)
                continue
            if forenames:
                name_parts.append(forenames)
            if keyname:
                name_parts.append(keyname)
            if name_parts:
                authors.append(" ".join(name_parts))
        return [author for author in authors if author]

    def _extract_primary_category(self, paper_elem: ET.Element) -> str:
        """提取主分类，兼容不同 OAI 扩展字段命名。"""
        primary_category = self._find_first_descendant(paper_elem, "primary_category")
        if primary_category is None:
            # 不同 arXiv/OAI 元数据版本里字段名可能不同，这里同时兼容 snake/camel 两种写法。
            primary_category = self._find_first_descendant(paper_elem, "primaryCategory")
        if primary_category is None:
            return ""
        term = primary_category.attrib.get("term", "").strip()
        if term:
            # term 属性通常比节点文本更规范，优先使用它作为主分类真值。
            return term
        return self._normalize_whitespace(primary_category.text or "")

    def _extract_resumption_token(self, root: ET.Element) -> str:
        """从 OAI-PMH 响应中提取翻页续传 token。"""
        token_elem = self._find_first_descendant(root, "resumptionToken")
        if token_elem is None or token_elem.text is None:
            return ""
        # 空串由上层统一解释为“没有下一页”，避免把 XML 细节暴露到同步主循环。
        return token_elem.text.strip()

    def _extract_oai_error(self, root: ET.Element) -> str:
        """提取 OAI-PMH 错误节点，拼成可直接记录的错误文本。"""
        error_elem = self._find_first_descendant(root, "error")
        if error_elem is None:
            return ""
        code = error_elem.attrib.get("code", "").strip()
        text = self._normalize_whitespace(error_elem.text or "")
        if code and text:
            # 把 code 和文本合并成单行字符串，方便日志与告警直接展示。
            return f"{code}: {text}"
        return code or text

    def _is_allowed_categories(self, categories: Sequence[str]) -> bool:
        """判断论文分类是否命中允许同步的目标分类集合。"""
        if not categories:
            return False
        normalized = [category.strip() for category in categories if str(category).strip()]
        if not normalized:
            return False
        # OAI 记录经常带 cross-list 分类；这里只要命中任一目标分类就保留，
        # 避免把主业务相关论文仅因额外挂了非白名单分类而整体过滤掉。
        return any(category in TARGET_CATEGORIES for category in normalized)

    def _split_categories(self, categories_text: str) -> List[str]:
        """把分类字符串拆成分类列表。"""
        if not categories_text:
            return []
        # OAI categories 常见为空格分隔，但也兼容逗号分隔的历史数据。
        raw_items = re.split(r"[\s,]+", categories_text.strip())
        return [item for item in (part.strip() for part in raw_items) if item]

    def _normalize_arxiv_id(self, raw_value: str) -> str:
        """把 OAI / URL / 带版本号的原始 ID 规整成统一 arXiv ID。"""
        value = self._normalize_whitespace(raw_value)
        if not value:
            return ""
        # 去掉常见前缀和 URL 包装，保证后续主键、链接和查询都基于同一 ID 形态。
        value = re.sub(r"^oai:arXiv\.org:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
        if "arxiv.org/abs/" in value.lower():
            value = value.rsplit("/", 1)[-1]
        value = value.split("?", 1)[0].strip()
        # 版本号不参与主键归一化，避免同一论文不同版本重复入库。
        value = re.sub(r"v\d+$", "", value)
        return value

    def _parse_retry_after(self, retry_after: Optional[str]) -> float:
        """把 Retry-After 头解析成秒数；解析失败时返回 0。"""
        if not retry_after:
            return 0.0
        try:
            return float(retry_after)
        except ValueError:
            # 这里只支持秒级数值；无法解析时交给上层退避策略自行兜底。
            return 0.0

    def _find_first_child(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        """在直接子节点里查找第一个指定本地名的元素。"""
        for child in list(parent):
            if self._local_name(child.tag) == local_name:
                return child
        # 返回 None 让上层自行决定是“字段可缺省”还是“解析失败”。
        return None

    def _find_first_element_child(self, parent: ET.Element) -> Optional[ET.Element]:
        """返回第一个真正的元素子节点，跳过注释等非元素节点。"""
        for child in list(parent):
            if isinstance(child.tag, str):
                return child
        # metadata 节点可能为空；这里显式返回 None 供上层统计缺失原因。
        return None

    def _find_first_descendant(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        """在整棵子树里查找第一个指定本地名的元素。"""
        for node in parent.iter():
            if self._local_name(node.tag) == local_name:
                return node
        # 统一使用 None 表示未命中，减少辅助函数之间的异常分支。
        return None

    def _extract_text(self, parent: ET.Element, local_name: str) -> str:
        """提取第一个命中节点的文本内容；缺失时返回空串。"""
        element = self._find_first_descendant(parent, local_name)
        if element is None or element.text is None:
            return ""
        # 解析阶段直接做 strip，避免调用方反复处理首尾空白。
        return element.text.strip()

    def _local_name(self, tag: Any) -> str:
        """剥离 XML 命名空间前缀，返回纯本地标签名。"""
        if not isinstance(tag, str):
            return ""
        if "}" in tag:
            # ElementTree 会把命名空间编码进 `{namespace}tag` 形式，这里统一去掉前缀。
            return tag.rsplit("}", 1)[-1]
        return tag

    def _normalize_whitespace(self, value: str) -> str:
        """折叠连续空白字符，得到稳定的展示和入库文本。"""
        if not value:
            return ""
        # 标题、摘要、作者名都可能带换行或多空格，统一规整后更利于检索和比较。
        return " ".join(value.split()).strip()
