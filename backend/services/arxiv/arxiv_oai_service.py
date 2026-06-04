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
        """把同步统计对象转换成普通字典。"""
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
        self.db_path = db_path or OAI_SQLITE_CONFIG["database_path"]
        self.check_same_thread = (
            OAI_SQLITE_CONFIG["check_same_thread"] if check_same_thread is None else bool(check_same_thread)
        )
        self._ensure_database_directory()
        self._initialize_database()

    def _ensure_database_directory(self) -> None:
        """确保数据库目录存在，避免首次写入时路径缺失。"""
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info("Created OAI database directory: %s", directory)

    def _get_connection(self) -> sqlite3.Connection:
        """创建一个 SQLite 连接。"""
        return sqlite3.connect(self.db_path, check_same_thread=self.check_same_thread)

    def _parse_list_field(self, value: Any) -> Any:
        """把数据库中的列表字段还原成更自然的 Python 结构。"""
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            if text.startswith("[") or text.startswith("("):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    return text
            return text
        return value

    def get_total_paper_count(self) -> int:
        """返回 OAI 本地镜像库中的论文总数。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
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
                    category_filters.append("categories LIKE ? OR primary_category = ?")
                    params.extend([f'%"{category}"%', category])
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
            params.append(max(1, int(max_results)))

            query = "\n".join(part.strip() for part in query_parts if part.strip())

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
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
        return " ".join(str(value or "").strip().split())

    def _strip_outer_parentheses(self, query: str) -> str:
        """移除查询字符串最外层成对括号。"""
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
                parts.append(text[start:index].strip())
                index += token_length
                start = index
                continue
            index += 1
        parts.append(text[start:].strip())
        return [part for part in parts if part]

    def _parse_row(self, row: Sequence[Any]) -> Dict[str, Any]:
        """把数据库查询返回的一行记录转换为标准论文字典。"""
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
        """把论文对象展开为便于搜索匹配的字段视图。"""
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
        match = re.match(r"submittedDate:\[(\d{12})\s+TO\s+(\d{12})\]", query.strip())
        if not match:
            return True
        start_raw, end_raw = match.groups()
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
            return True

    def _matches_atomic_clause(self, paper: Dict[str, Any], clause: str) -> bool:
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
            haystack = self._paper_to_search_fields(paper).get(field, self._paper_to_search_fields(paper)["all"])
            return query_text in haystack
        query_text = text.replace('"', "").lower()
        return query_text in self._paper_to_search_fields(paper)["all"]

    def _matches_query(self, paper: Dict[str, Any], query: str) -> bool:
        text = self._strip_outer_parentheses(query.strip())
        if not text:
            return True
        if " ANDNOT " in text:
            parts = self._split_top_level(text, " ANDNOT ")
            if not parts:
                return True
            return self._matches_query(paper, parts[0]) and all(
                not self._matches_query(paper, part) for part in parts[1:]
            )
        if " AND " in text:
            parts = self._split_top_level(text, " AND ")
            return all(self._matches_query(paper, part) for part in parts)
        if " OR " in text:
            parts = self._split_top_level(text, " OR ")
            return any(self._matches_query(paper, part) for part in parts)
        return self._matches_atomic_clause(paper, text)

    def _build_paper_response(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
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
            placeholders = ",".join("?" for _ in normalized_ids)
            query += f" WHERE arxiv_id IN ({placeholders})"
            params.extend(normalized_ids)
        query += " ORDER BY COALESCE(created, updated, oai_datestamp, fetched_at) DESC, arxiv_id DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [self._parse_row(row) for row in rows]

    def search(
        self,
        search_query: str = "",
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> Dict[str, Any]:
        normalized_query = str(search_query or "").strip()
        normalized_id_list = [str(item).strip() for item in (id_list or []) if str(item).strip()]
        rows = self._fetch_all_searchable_papers(normalized_id_list or None)

        if normalized_query:
            rows = [paper for paper in rows if self._matches_query(paper, normalized_query)]

        reverse = sort_order == "descending"
        if sort_by in {"submittedDate", "lastUpdatedDate"}:
            rows.sort(key=lambda p: p.get("updated") or p.get("created") or "", reverse=reverse)

        paginated_rows = rows[start:start + max_results]
        papers = [self._build_paper_response(paper) for paper in paginated_rows]
        return {
            "query": normalized_query,
            "id_list": normalized_id_list,
            "total_results": len(rows),
            "start_index": start,
            "items_per_page": len(papers),
            "papers": papers,
            "timestamp": datetime.now().isoformat(),
        }

    def search_papers(
        self,
        search_query: str = "",
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        submitted_days_ago: Optional[int] = None,
    ) -> Dict[str, Any]:
        if submitted_days_ago is not None:
            if submitted_days_ago < 0:
                raise ValueError("submitted_days_ago must be greater than or equal to 0")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(submitted_days_ago))
            submitted_date_query = f"submittedDate:[{start_date.strftime('%Y%m%d%H%M')} TO {end_date.strftime('%Y%m%d%H%M')}]"
            search_query = f"({search_query}) AND {submitted_date_query}" if search_query else submitted_date_query
        return self.search(
            search_query=search_query,
            id_list=id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def search_advanced(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        abstract: Optional[str] = None,
        category: Optional[str] = None,
        comment: Optional[str] = None,
        journal_ref: Optional[str] = None,
        report_number: Optional[str] = None,
        operator: str = "AND",
        id_list: Optional[List[str]] = None,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        submitted_days_ago: Optional[int] = None,
    ) -> Dict[str, Any]:
        query_parts: List[str] = []
        if title:
            query_parts.append(f'ti:"{title}"' if " " in title else f"ti:{title}")
        if author:
            query_parts.append(f'au:"{author}"' if " " in author else f"au:{author}")
        if abstract:
            query_parts.append(f'abs:"{abstract}"' if " " in abstract else f"abs:{abstract}")
        if category:
            query_parts.append(f"cat:{category}")
        if comment:
            query_parts.append(f'co:"{comment}"' if " " in comment else f"co:{comment}")
        if journal_ref:
            query_parts.append(f'jr:"{journal_ref}"' if " " in journal_ref else f"jr:{journal_ref}")
        if report_number:
            query_parts.append(f'rn:"{report_number}"' if " " in report_number else f"rn:{report_number}")
        combined_query = f" {operator} ".join(query_parts)
        return self.search_papers(
            search_query=combined_query,
            id_list=id_list,
            max_results=max_results,
            start=start,
            sort_by=sort_by,
            sort_order=sort_order,
            submitted_days_ago=submitted_days_ago,
        )

    def _initialize_database(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
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
            conn.commit()
            logger.info("OAI database tables initialized successfully: %s", self.db_path)

    def upsert_arxiv_oai_paper(self, paper: Dict[str, Any]) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
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
                conn.commit()
                logger.info("OAI paper upserted: %s", paper.get("arxiv_id"))
                return True
        except Exception as exc:
            logger.error("Error upserting OAI paper: %s", exc)
            return False

    def upsert_arxiv_oai_papers(self, papers: Sequence[Dict[str, Any]]) -> int:
        normalized_papers = [paper for paper in papers if paper and paper.get("arxiv_id")]
        if not normalized_papers:
            return 0

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
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
        self.endpoint = endpoint
        self.request_interval_seconds = max(3.5, float(request_interval_seconds))
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.database_service = database_service
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = vector_store_service or VectorStoreService()
        self.embedding_collection_name = embedding_collection_name
        self.embedding_config = self.embedding_service.get_default_embedding_config()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def sync(self, from_date: str, until_date: str, dry_run: bool = False, count_only: bool = False) -> ArxivOaiSyncStats:
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

            matched_papers: List[Dict[str, Any]] = []
            for record in records:
                paper = self._process_record(record, stats, dry_run=dry_run, count_only=count_only)
                if paper is not None:
                    matched_papers.append(paper)

            if matched_papers and not (dry_run or count_only):
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
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None
            except requests.RequestException as exc:
                logger.error("OAI-PMH request error for params=%s: %s", params, exc)
                if attempt < self.max_retries:
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                stats.errors += 1
                return None

            if response.status_code in {429, 503}:
                retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
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

            return response.text

        return None

    def _process_record(self, record: ET.Element, stats: ArxivOaiSyncStats, dry_run: bool, count_only: bool) -> Optional[Dict[str, Any]]:
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
        normalized_papers = [paper for paper in papers if paper and paper.get("arxiv_id")]
        if not normalized_papers:
            return

        if self.database_service is None:
            self.database_service = ArxivOaiDatabaseService()

        persisted_papers: List[Dict[str, Any]] = []
        persisted_count = self.database_service.upsert_arxiv_oai_papers(normalized_papers)
        if persisted_count == 0:
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

        self._store_paper_embeddings_batch(persisted_papers, stats)

    def _build_oai_embedding_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
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
        eligible_papers = []
        eligible_ids = []
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
            title = str(paper.get("title", "") or "").strip()
            abstract = str(paper.get("abstract", "") or "").strip()
            if not arxiv_id or not title or not abstract:
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
            for paper in missing_papers:
                logger.info("Embedded OAI paper into vector store: %s", paper.get("arxiv_id"))
        except Exception as exc:  # pragma: no cover - vector store/runtime dependent
            logger.warning("Batch embedding for OAI papers failed, falling back to single-item processing: %s", exc)
            for paper in missing_papers:
                self._maybe_store_paper_embedding(paper, stats)

    def _accumulate_embedding_usage(self, stats: ArxivOaiSyncStats, usage: Dict[str, Any], item_count: int) -> None:
        if not usage:
            return

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))

        stats.embedding_input_tokens += input_tokens
        stats.embedding_output_tokens += output_tokens
        stats.embedding_total_tokens += total_tokens

        if input_tokens > 0:
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
        arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
        title = str(paper.get("title", "") or "").strip()
        abstract = str(paper.get("abstract", "") or "").strip()
        if not arxiv_id or not title or not abstract:
            logger.debug("Skipping embedding for OAI paper %s because title or abstract is missing", arxiv_id or "<unknown>")
            return

        try:
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
        authors_parent = self._find_first_child(paper_elem, "authors")
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
        primary_category = self._find_first_descendant(paper_elem, "primary_category")
        if primary_category is None:
            primary_category = self._find_first_descendant(paper_elem, "primaryCategory")
        if primary_category is None:
            return ""
        term = primary_category.attrib.get("term", "").strip()
        if term:
            return term
        return self._normalize_whitespace(primary_category.text or "")

    def _extract_resumption_token(self, root: ET.Element) -> str:
        token_elem = self._find_first_descendant(root, "resumptionToken")
        if token_elem is None or token_elem.text is None:
            return ""
        return token_elem.text.strip()

    def _extract_oai_error(self, root: ET.Element) -> str:
        error_elem = self._find_first_descendant(root, "error")
        if error_elem is None:
            return ""
        code = error_elem.attrib.get("code", "").strip()
        text = self._normalize_whitespace(error_elem.text or "")
        if code and text:
            return f"{code}: {text}"
        return code or text

    def _is_allowed_categories(self, categories: Sequence[str]) -> bool:
        if not categories:
            return False
        normalized = [category.strip() for category in categories if str(category).strip()]
        if not normalized:
            return False
        return all(category in TARGET_CATEGORIES for category in normalized)

    def _split_categories(self, categories_text: str) -> List[str]:
        if not categories_text:
            return []
        raw_items = re.split(r"[\s,]+", categories_text.strip())
        return [item for item in (part.strip() for part in raw_items) if item]

    def _normalize_arxiv_id(self, raw_value: str) -> str:
        value = self._normalize_whitespace(raw_value)
        if not value:
            return ""
        value = re.sub(r"^oai:arXiv\.org:", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^arxiv:", "", value, flags=re.IGNORECASE)
        if "arxiv.org/abs/" in value.lower():
            value = value.rsplit("/", 1)[-1]
        value = value.split("?", 1)[0].strip()
        value = re.sub(r"v\d+$", "", value)
        return value

    def _parse_retry_after(self, retry_after: Optional[str]) -> float:
        if not retry_after:
            return 0.0
        try:
            return float(retry_after)
        except ValueError:
            return 0.0

    def _find_first_child(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        for child in list(parent):
            if self._local_name(child.tag) == local_name:
                return child
        return None

    def _find_first_element_child(self, parent: ET.Element) -> Optional[ET.Element]:
        for child in list(parent):
            if isinstance(child.tag, str):
                return child
        return None

    def _find_first_descendant(self, parent: ET.Element, local_name: str) -> Optional[ET.Element]:
        for node in parent.iter():
            if self._local_name(node.tag) == local_name:
                return node
        return None

    def _extract_text(self, parent: ET.Element, local_name: str) -> str:
        element = self._find_first_descendant(parent, local_name)
        if element is None or element.text is None:
            return ""
        return element.text.strip()

    def _local_name(self, tag: Any) -> str:
        if not isinstance(tag, str):
            return ""
        if "}" in tag:
            return tag.rsplit("}", 1)[-1]
        return tag

    def _normalize_whitespace(self, value: str) -> str:
        if not value:
            return ""
        return " ".join(value.split()).strip()
